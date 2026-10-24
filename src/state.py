"""Saving and restoring the learner's whole state (FR-14), checkpoints with rollback (FR-17), export (FR-15).

A state directory looks like this:

    state/
      config.yaml               the learner's config
      merged/merge-001.safetensors ...   adapters folded into the weights by sleep phases, in order
      ckpt-000012/              one checkpoint per portion (the newest `keep` are kept)
        adapter.safetensors     current LoRA adapter (or model.safetensors for full fine-tuning)
        buffer.json             replay buffer: examples, priorities, counters, random state
        learner.json            counters, seen tasks, merged adapters in use, random states, pending examples
        torch_rng.pt
      latest                    name of the newest checkpoint

The base weights are not copied: the base model is downloaded at its pinned revision and the merged
adapters are folded in again in the same order, which gives the same weights (in fp32).
"""

import json
import shutil
from pathlib import Path

import torch
import yaml
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from src.model import is_lora, load_model, merge_and_renew


def _cpu(sd):
    return {k: v.detach().to("cpu").contiguous() for k, v in sd.items()}


def adapter_weights(model):
    return _cpu(get_peft_model_state_dict(model))


def save_merged_adapter(state_dir, weights, index):
    path = Path(state_dir) / "merged" / f"merge-{index:03d}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(path))
    return path.relative_to(state_dir).as_posix()


def _rng_to_json(state):
    version, internal, gauss = state
    return [version, list(internal), gauss]


def _rng_from_json(state):
    version, internal, gauss = state
    return version, tuple(internal), gauss


def save_checkpoint(run, state_dir, keep=3):
    state_dir = Path(state_dir)
    name = f"ckpt-{run.portions_done:06d}"
    tmp = state_dir / (name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if is_lora(run.model):
        save_file(adapter_weights(run.model), str(tmp / "adapter.safetensors"))
    else:
        save_file(_cpu(run.model.state_dict()), str(tmp / "model.safetensors"))
    if run.buffer is not None:
        b = run.buffer
        (tmp / "buffer.json").write_text(json.dumps({
            "items": b.items, "priority": b.priority, "n_seen": b.n_seen, "rng": b.rng.bit_generator.state,
        }, ensure_ascii=False), encoding="utf-8")
    (tmp / "learner.json").write_text(json.dumps({
        "name": run.cfg["name"], "seed": run.seed, "portions_done": run.portions_done,
        "sleeps_done": run.sleeps_done, "steps": run.steps, "examples_new": run.examples_new,
        "examples_total": run.examples_total, "examples_new_task": run.examples_new_task,
        "current_task": run.current_task, "seen_tasks": run.seen_tasks, "task_start": run.task_start,
        "merged": run.merged, "pending": run.pending, "elapsed_s": run.elapsed(),
        "python_rng": _rng_to_json(run.rng.getstate()), "numpy_rng": run.np_rng.bit_generator.state,
    }, ensure_ascii=False), encoding="utf-8")
    torch.save({"cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, tmp / "torch_rng.pt")
    final = state_dir / name
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    if not (state_dir / "config.yaml").exists():  # a state keeps the config it was created with;
        # session-only settings (low-memory mode, --set) are not written into it
        (state_dir / "config.yaml").write_text(yaml.safe_dump(run.cfg, sort_keys=False, allow_unicode=True),
                                               encoding="utf-8")
    (state_dir / "latest").write_text(name, encoding="utf-8")
    for old in list_checkpoints(state_dir)[:-keep]:
        shutil.rmtree(state_dir / old, ignore_errors=True)
    return name


def list_checkpoints(state_dir):
    state_dir = Path(state_dir)
    return sorted(p.name for p in state_dir.glob("ckpt-*") if p.is_dir() and not p.name.endswith(".tmp"))


def latest_checkpoint(state_dir):
    marker = Path(state_dir) / "latest"
    if marker.exists() and (Path(state_dir) / marker.read_text(encoding="utf-8").strip()).is_dir():
        return marker.read_text(encoding="utf-8").strip()
    found = list_checkpoints(state_dir)
    return found[-1] if found else None


def rebuild_weights(model, cfg, state_dir, ckpt_dir, merged, device):
    """Fold the merged adapters into the fresh base model, then load the checkpoint's own weights."""
    ckpt_dir = Path(ckpt_dir)
    if not is_lora(model):
        model.load_state_dict(load_file(str(ckpt_dir / "model.safetensors"), device=str(device)))
        return model
    for rel in merged:
        set_peft_model_state_dict(model, load_file(str(Path(state_dir) / rel), device=str(device)))
        model = merge_and_renew(model, cfg)
    set_peft_model_state_dict(model, load_file(str(ckpt_dir / "adapter.safetensors"), device=str(device)))
    return model


def load_checkpoint(run, state_dir, name=None):
    """Restore the run from a checkpoint (the newest one by default). The run must be freshly built."""
    state_dir = Path(state_dir)
    name = name or latest_checkpoint(state_dir)
    ckpt = state_dir / name
    meta = json.loads((ckpt / "learner.json").read_text(encoding="utf-8"))
    run.model = rebuild_weights(run.model, run.cfg, state_dir, ckpt, meta["merged"], run.device)
    if run.buffer is not None and (ckpt / "buffer.json").exists():
        b = json.loads((ckpt / "buffer.json").read_text(encoding="utf-8"))
        run.buffer.items, run.buffer.priority, run.buffer.n_seen = b["items"], b["priority"], b["n_seen"]
        run.buffer.members = {ex["id"] for items in b["items"].values() for ex in items}
        run.buffer.rng.bit_generator.state = b["rng"]
    for key in ("portions_done", "sleeps_done", "steps", "examples_new", "examples_total", "examples_new_task",
                "current_task", "seen_tasks", "task_start", "merged", "pending"):
        setattr(run, key, meta[key])
    run.elapsed_before = meta.get("elapsed_s", 0.0)
    run.rng.setstate(_rng_from_json(meta["python_rng"]))
    run.np_rng.bit_generator.state = meta["numpy_rng"]
    rng = torch.load(ckpt / "torch_rng.pt", weights_only=True)  # tensors only: safe for downloaded states
    torch.set_rng_state(rng["cpu"])
    if rng["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return name


def rollback(state_dir, name):
    """Make an older checkpoint the newest one (newer checkpoints are deleted)."""
    state_dir = Path(state_dir)
    if not (state_dir / name).is_dir():
        raise SystemExit(f"no checkpoint {name}; available: {', '.join(list_checkpoints(state_dir))}")
    for newer in [c for c in list_checkpoints(state_dir) if c > name]:
        shutil.rmtree(state_dir / newer)
    (state_dir / "latest").write_text(name, encoding="utf-8")


def export_model(state_dir, out_dir, dtype="bfloat16", name=None):
    """Write the learner as a plain Hugging Face model: base weights with every adapter folded in (FR-15)."""
    from transformers import GenerationConfig

    state_dir, out_dir = Path(state_dir), Path(out_dir)
    cfg = yaml.safe_load((state_dir / "config.yaml").read_text(encoding="utf-8"))
    name = name or latest_checkpoint(state_dir)
    meta = json.loads((state_dir / name / "learner.json").read_text(encoding="utf-8"))
    device = torch.device("cpu")
    model, tok = load_model(cfg, device)
    model = rebuild_weights(model, cfg, state_dir, state_dir / name, meta["merged"], device)
    if is_lora(model):
        model = model.merge_and_unload()
    model.to(getattr(torch, dtype))
    model.config.use_cache = True
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    try:
        GenerationConfig.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"]).save_pretrained(out_dir)
    except OSError:
        pass
    (out_dir / "README.md").write_text(MODEL_CARD.format(base=cfg["model"]["name"], ckpt=name, cfg=cfg["name"],
                                                         portions=meta["portions_done"], sleeps=meta["sleeps_done"],
                                                         path=out_dir.as_posix()), encoding="utf-8")
    return out_dir


MODEL_CARD = """# Continually trained {base}

Exported from checkpoint `{ckpt}` of configuration `{cfg}`: {portions} portions learnt, {sleeps} sleep phases.
All adapters are merged into the weights, so this is an ordinary Hugging Face model; the project's code is not needed.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{path}")
model = AutoModelForCausalLM.from_pretrained("{path}")
messages = [{{"role": "user", "content": "What is 17 + 25? Reply with the answer only."}}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

To keep training this model, use the project's `live.py` / `chat.py` with the saved state, not this export.
"""
