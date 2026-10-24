"""Experiment journal (FR-6): config, seed, code version, environment, data versions, metrics.

Everything goes to plain files in the run folder; Weights & Biases is optional.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["torch", "transformers", "peft", "datasets", "trl", "numpy", "pyyaml"]


def code_version():
    """Git commit (if the project is a git repo) plus a hash of all source files."""
    h = hashlib.sha256()
    for path in sorted([*ROOT.glob("src/*.py"), *ROOT.glob("data/*.py"), ROOT / "run.py"]):
        if path.exists():
            h.update(path.relative_to(ROOT).as_posix().encode())
            h.update(path.read_bytes())
    out = {"source_sha256": h.hexdigest()}
    try:
        git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        out["git_commit"] = git("rev-parse", "HEAD")
        out["git_dirty"] = bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        out["git_commit"] = None
    return out


def environment(device):
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}


class RunLogger:
    def __init__(self, run_dir, cfg, seed, fresh=True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if fresh:  # a run started from scratch starts its journal over; a resumed one appends
            for name in ("metrics.jsonl", "curve.jsonl", "surprise_samples.jsonl"):
                (self.dir / name).unlink(missing_ok=True)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                              encoding="utf-8")
        self.wandb = None
        if cfg["logging"].get("wandb"):
            import wandb
            self.wandb = wandb.init(project=cfg["logging"]["wandb_project"], name=f"{cfg['name']}-s{seed}",
                                    config=cfg, reinit=True)

    def write_json(self, name, obj):
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.dir / name)

    def append(self, name, obj):
        with open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def rewrite(self, name, records):
        tmp = self.dir / (name + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        tmp.replace(self.dir / name)

    def metrics(self, rec):
        self.append("metrics.jsonl", rec)
        if self.wandb and "eval" in rec:
            flat = {f"{t}/{k}": v for t, m in rec["eval"].items() for k, v in m.items()
                    if k != "n" and isinstance(v, (int, float))}
            self.wandb.log({**flat, "steps": rec.get("steps"), "portion": rec.get("portion")})

    def close(self):
        if self.wandb:
            self.wandb.finish()


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
