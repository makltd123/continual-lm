"""Run experiments from YAML configs.

    python run.py configs/E1.yaml                 # all seeds listed in the config
    python run.py --all                           # every configs/E*.yaml (E0-E6)
    python run.py configs/E2.yaml --seeds 0       # selected seeds
    python run.py --all --preset configs/presets/smoke.yaml   # quick check on a tiny model
    python run.py configs/E1.yaml --set replay.ratio=0.1      # override any key

Runs that already have final.json are skipped (useful when a Colab session drops); --force reruns.
"""

import argparse
import glob
import sys
from pathlib import Path

import yaml

from src.config import deep_update, load_config, set_dotted
from src.train_loop import run_experiment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("configs", nargs="*", help="YAML experiment configs")
    p.add_argument("--all", action="store_true", help="run every configs/E*.yaml")
    p.add_argument("--seeds", nargs="+", type=int, help="override the seeds in the config")
    p.add_argument("--preset", help="YAML merged on top of every config (e.g. smaller sizes)")
    p.add_argument("--set", nargs="+", default=[], dest="overrides", metavar="KEY=VALUE")
    p.add_argument("--force", action="store_true", help="rerun runs that already finished")
    p.add_argument("--prepare", action="store_true",
                   help="only build the data caches and download the model, then exit "
                        "(do this once before starting parallel runs)")
    args = p.parse_args(argv)

    paths = sorted(glob.glob("configs/E*.yaml")) if args.all else args.configs
    if not paths:
        p.error("give config files or --all")
    preset = yaml.safe_load(Path(args.preset).read_text(encoding="utf-8")) if args.preset else {}

    for path in paths:
        cfg = load_config(path)
        cfg = deep_update(cfg, {**preset, "name": cfg["name"]})
        for item in args.overrides:  # command-line overrides win over the preset
            key, _, value = item.partition("=")
            set_dotted(cfg, key.strip(), yaml.safe_load(value))
        if args.prepare:
            prepare(cfg)
            continue
        for seed in args.seeds or cfg["seeds"]:
            run_dir = Path(cfg["output_dir"]) / cfg["name"] / f"seed{seed}"
            if (run_dir / "final.json").exists() and not args.force:
                print(f"skip {cfg['name']} seed {seed}: already finished ({run_dir})")
                continue
            print(f"=== {cfg['name']} seed {seed} -> {run_dir}", flush=True)
            run_experiment(cfg, seed, run_dir)


def prepare(cfg):
    from huggingface_hub import snapshot_download

    from data.tasks import load_all

    load_all(cfg["data"], cfg["stream"]["order"])
    snapshot_download(cfg["model"]["name"], revision=cfg["model"]["revision"],
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    print(f"prepared {cfg['name']}: data in {cfg['data']['cache_dir']}, model {cfg['model']['name']}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
