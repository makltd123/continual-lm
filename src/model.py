"""Base model, LoRA adapter, precision, adapter merging and rollback snapshots (FR-1)."""

from contextlib import nullcontext

import psutil
import torch
import transformers
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device(setting="auto"):
    if setting != "auto":
        return torch.device(setting)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Precision:
    """Weights stay in fp32 (so repeated adapter merges lose nothing to rounding);
    on GPU the forward pass runs under autocast: bf16 where supported, else fp16 with a grad scaler."""

    def __init__(self, device, setting="auto"):
        self.amp = None
        if device.type == "cuda" and setting != "fp32":
            # is_bf16_supported() is also true when bf16 is only emulated (e.g. T4, compute 7.5), which is
            # several times slower than fp16 on tensor cores; auto picks bf16 only where it is native (8.0+).
            native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
            if setting == "bf16" or (setting == "auto" and native_bf16):
                self.amp = torch.bfloat16
            else:
                self.amp = torch.float16
        self.scaler = torch.amp.GradScaler("cuda") if self.amp == torch.float16 else None
        self.name = {None: "fp32", torch.bfloat16: "bf16-autocast", torch.float16: "fp16-autocast"}[self.amp]

    def autocast(self):
        return torch.autocast("cuda", dtype=self.amp) if self.amp else nullcontext()


def load_model(cfg, device):
    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # Transformers 5 renamed torch_dtype to dtype and defaults to the checkpoint dtype.
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    # fp32 by default; bfloat16 halves memory (for machines with little RAM), at some cost to exact merges.
    dtype = getattr(torch, m.get("dtype") or "float32")
    model = AutoModelForCausalLM.from_pretrained(m["name"], revision=m["revision"], **{dtype_key: dtype})
    model.to(device)
    model.config.use_cache = False
    if m["method"] == "lora":
        model = add_lora(model, cfg)
    elif m["method"] != "full":
        raise ValueError(f"model.method must be 'lora' or 'full', got {m['method']!r}")
    return model, tok


def add_lora(model, cfg):
    lora = cfg["model"]["lora"]
    config = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                        target_modules=lora["target_modules"], task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def merge_and_renew(model, cfg):
    """Fold the adapter into the base weights and start a fresh adapter (sleep phase)."""
    return add_lora(model.merge_and_unload(), cfg)


def is_lora(model):
    return hasattr(model, "peft_config")


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def snapshot(model):
    """CPU copy of everything a sleep phase can change, for rollback."""
    if is_lora(model):
        base = model.get_base_model().state_dict()
        return {
            "base": {k: v.detach().cpu().clone() for k, v in base.items() if ".base_layer." in k},
            "adapter": {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()},
        }
    return {"full": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def restore(model, snap):
    """Load a snapshot back. After a merge the module layout is the same (same target modules)."""
    if "full" in snap:
        model.load_state_dict(snap["full"])
        return
    missing = set(snap["base"]) - set(model.get_base_model().state_dict())
    if missing:
        raise RuntimeError(f"snapshot does not match the model: {sorted(missing)[:3]} ...")
    model.get_base_model().load_state_dict(snap["base"], strict=False)
    set_peft_model_state_dict(model, snap["adapter"])


def memory_gb(device):
    """Peak GPU memory on CUDA; on CPU the process's peak (Windows) or current resident memory."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1e9
