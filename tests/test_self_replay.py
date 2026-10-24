"""Self-generated replay: the original model's answers to general requests enter the buffer as task R."""

from pathlib import Path

import yaml

from src.config import deep_update, load_config
from src.train_loop import Run


def test_self_replay_fills_buffer_and_is_cached(tmp_path):
    cfg = load_config("configs/extra/E5_selfreplay.yaml")
    cfg = deep_update(cfg, yaml.safe_load(open("configs/presets/smoke.yaml", encoding="utf-8")))
    cfg["self_replay"].update(n=6, max_new_tokens=6)
    cfg["eval"]["curve_every_steps"] = 0
    run = Run(cfg, 0, tmp_path / "run")
    final = run.run()

    assert run.buffer.sizes().get("R", 0) > 0
    assert all(ex["task"] == "R" and ex["target"] for ex in run.buffer.items["R"])
    assert '"phase": "self_replay"' in (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8")
    assert "T0" in final["test"]
    assert list(Path(cfg["data"]["cache_dir"]).glob("R_answers_*.json"))  # shared by later seeds and runs
