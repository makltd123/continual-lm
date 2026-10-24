"""Model-level checks on a tiny random Qwen2 (downloads ~10 MB on first run)."""

import pytest
import torch

from src.model import Precision, load_model, merge_and_renew, restore, snapshot
from src.scoring import Tokens, collate, forward_losses

CFG = {"model": {"name": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                 "revision": "ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda", "method": "lora", "precision": "fp32",
                 "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": "all-linear"}}}
DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model, tok = load_model(CFG, DEVICE)
    for name, p in model.named_parameters():  # a non-trivial adapter (lora_B starts at zero)
        if "lora_B" in name:
            p.data.normal_(0, 0.5)
    model.eval()
    tokens = Tokens(tok, 64)
    examples = [{"id": f"t{i}", "kind": "qa", "prompt": f"Question: what is {i}?\nAnswer:", "target": f" {i}"}
                for i in range(4)]
    tokens.add(examples)
    batch = collate([tokens[e] for e in examples], tok.pad_token_id, DEVICE)
    return model, tok, tokens, examples, batch


def logits(model, batch):
    with torch.no_grad():
        return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits


def test_merge_keeps_outputs_and_snapshot_restores(setup):
    model, _, _, _, batch = setup
    before = logits(model, batch)
    snap = snapshot(model)
    merged = merge_and_renew(model, CFG)
    merged.eval()
    assert torch.allclose(logits(merged, batch), before, atol=1e-4)  # merging changes nothing in fp32
    for name, p in merged.named_parameters():  # now break the weights and roll back
        if "base_layer" in name:
            p.data.add_(1.0)
    assert not torch.allclose(logits(merged, batch), before, atol=1e-2)
    restore(merged, snap)
    assert torch.allclose(logits(merged, batch), before, atol=1e-5)


def test_teacher_forced_em_equals_greedy_decoding(setup):
    model, tok, tokens, examples, _ = setup
    model.eval()
    pairs = []
    for e in examples:
        prompt = tokens[e][0]
        out = model.generate(input_ids=torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
                             max_new_tokens=3, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=None)
        pairs.append((prompt, out[0, len(prompt):].tolist()))
    res = forward_losses(model, collate(pairs, tok.pad_token_id, DEVICE), need_em=True)
    assert res["em"].all()  # the greedy continuation is an exact match by construction
    vocab = model.config.vocab_size
    wrong = [(p, t[:-1] + [(t[-1] + 1) % vocab]) for p, t in pairs]
    res = forward_losses(model, collate(wrong, tok.pad_token_id, DEVICE), need_em=True)
    assert not res["em"].any()


@pytest.mark.parametrize("chunk", [2048, 5])
def test_target_only_head_matches_full_logits(setup, chunk, monkeypatch):
    """The fast path (output layer on target tokens only, in pieces of `chunk` tokens) must equal the
    plain full-logits loss, exact match and gradients."""
    from src import scoring
    monkeypatch.setattr(scoring, "HEAD_CHUNK", chunk)
    model, tok, tokens, examples, batch = setup
    lm = [{"id": f"lm{i}", "kind": "lm", "prompt": "", "target": "Once upon a time " * (i + 1)} for i in range(3)]
    tokens.add(lm)
    mixed = collate([tokens[e] for e in examples + lm], tok.pad_token_id, DEVICE)
    model.train()
    res = forward_losses(model, mixed, need_em=True)
    res["mean"].sum().backward()
    grad_fast = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()

    full = model(input_ids=mixed["input_ids"], attention_mask=mixed["attention_mask"]).logits[:, :-1]
    labels = mixed["labels"][:, 1:]
    mask = labels != -100
    nll = torch.nn.functional.cross_entropy(full.transpose(1, 2).float(), labels.clamp_min(0), reduction="none")
    mean = (nll * mask).sum(1) / mask.sum(1)
    em = ((full.argmax(-1) == labels) | ~mask).all(1)
    mean.sum().backward()
    model.eval()

    assert torch.allclose(res["mean"], mean, atol=1e-5)
    assert torch.equal(res["em"], em)
    assert grad_fast and all(torch.allclose(grad_fast[n], p.grad, atol=1e-5)
                             for n, p in model.named_parameters() if p.grad is not None)
    model.zero_grad()


def test_precision_on_cpu_is_fp32():
    p = Precision(DEVICE, "auto")
    assert p.amp is None and p.scaler is None and p.name == "fp32"
