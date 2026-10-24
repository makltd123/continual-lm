"""State saving (FR-14), rollback (FR-17) and export (FR-15) on the tiny model with the smoke preset."""

import json

import pytest
import torch
import yaml
from transformers import AutoModelForCausalLM

from src.config import deep_update, load_config
from src.state import export_model, list_checkpoints, rollback
from src.train_loop import Run


def smoke_cfg():
    cfg = load_config("configs/E5.yaml")  # replay + surprise + curriculum + sleep with merging
    cfg = deep_update(cfg, yaml.safe_load(open("configs/presets/smoke.yaml", encoding="utf-8")))
    cfg["checkpoint"]["enabled"] = True
    cfg["eval"]["curve_every_steps"] = 0
    return cfg


class Crash(Exception):
    pass


class CrashAt5(Run):
    def step_portion(self, k, *args, **kwargs):
        if k == 5:
            raise Crash
        return super().step_portion(k, *args, **kwargs)


def journal(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_resume_after_crash_matches_uninterrupted_run(tmp_path):
    cfg = smoke_cfg()
    full_run = Run(cfg, 0, tmp_path / "a", state_dir=tmp_path / "a" / "state")
    full = full_run.run()

    with pytest.raises(Crash):  # stops after portion 4; the sleep after portion 3 has merged an adapter
        CrashAt5(cfg, 0, tmp_path / "b", state_dir=tmp_path / "b" / "state").run()
    assert len(json.loads((tmp_path / "b" / "state" / "ckpt-000004" / "learner.json").read_text())["merged"]) == 1
    resumed = Run(cfg, 0, tmp_path / "b", state_dir=tmp_path / "b" / "state").run()

    for task, m in full["test"].items():
        assert resumed["test"][task]["score"] == pytest.approx(m["score"], abs=1e-6), task
    steps = lambda d: [(r["portion"], r["phase"]) for r in journal(d / "metrics.jsonl")]
    losses = lambda d: [r.get("train_loss") for r in journal(d / "metrics.jsonl")]
    assert steps(tmp_path / "b") == steps(tmp_path / "a")
    for a, b in zip(losses(tmp_path / "a"), losses(tmp_path / "b")):
        assert (a is None and b is None) or a == pytest.approx(b, abs=1e-6)

    # Export: a plain Hugging Face model with the same outputs as the learner.
    out = export_model(tmp_path / "a" / "state", tmp_path / "export", dtype="float32")
    exported = AutoModelForCausalLM.from_pretrained(out)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    full_run.model.eval()
    with torch.no_grad():
        assert torch.allclose(exported(ids).logits, full_run.model(input_ids=ids).logits, atol=1e-4)

    # Rollback: an older checkpoint becomes the newest one and a new run starts from it.
    older = list_checkpoints(tmp_path / "a" / "state")[0]
    rollback(tmp_path / "a" / "state", older)
    assert list_checkpoints(tmp_path / "a" / "state")[-1] == older
