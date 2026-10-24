"""Continuous mode (FR-13): one model that keeps learning and keeps its state between launches.

    # Train the final model on the task stream from its config; after a stop it continues where it was.
    python live.py --config configs/final/E5_final.yaml --state states/ewai-e5 --stream builtin \
                   --export exported/ewai-e5

    # Keep teaching it: drop .jsonl files into the inbox folder, one example per line:
    #   {"prompt": "user request", "response": "the answer the model should give"}
    # Every `live.portion_size` examples become a portion (surprise, replay, update, sleep when due).
    python live.py --state states/ewai-e5 --inbox inbox

    python live.py --state states/ewai-e5 --list                     # checkpoints
    python live.py --state states/ewai-e5 --rollback ckpt-000012     # go back to one of them (FR-17)
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import yaml

from src.config import deep_update, load_config, set_dotted
from src.state import export_model, list_checkpoints, rollback
from src.train_loop import Run


# For a computer without a GPU and with ~8 GB of RAM: half-precision weights, one example at a time, the loss
# in small pieces, and a light sleep (16 replayed examples, no re-scoring of the whole buffer): ~3.5 min on a CPU.
LOW_MEMORY = {"model.dtype": "bfloat16", "train.batch_size": 8, "train.micro_batch_size": 1, "train.head_chunk": 128,
              "eval.batch_size": 4, "sleep.replay_examples": 16, "sleep.cleanup": False}


def learner_config(state, config=None, preset=None, overrides=(), low_memory=False):
    """An existing state keeps the config it was created with; a new one is built from --config."""
    saved = Path(state) / "config.yaml"
    if saved.exists():
        cfg = yaml.safe_load(saved.read_text(encoding="utf-8"))
        # A state trained elsewhere (e.g. downloaded from Kaggle) carries that machine's paths.
        if not Path(cfg["data"]["cache_dir"]).exists():
            cfg["data"]["cache_dir"] = "data/cache"
        cfg["checkpoint"]["state_dir"] = cfg["checkpoint"]["export_dir"] = None
    elif config:
        cfg = load_config(config)
        if preset:
            cfg = deep_update(cfg, {**yaml.safe_load(Path(preset).read_text(encoding="utf-8")), "name": cfg["name"]})
    else:
        raise SystemExit(f"{state} has no saved state yet: give --config for a new model")
    if low_memory:
        for key, value in LOW_MEMORY.items():
            set_dotted(cfg, key, value)
    for item in overrides:
        key, _, value = item.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(value))
    cfg["checkpoint"]["enabled"] = True
    return cfg


def read_examples(path, default_task="U"):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" in row:  # chat format: the last user message and the assistant's reply
            prompt = next(m["content"] for m in reversed(row["messages"]) if m["role"] == "user")
            response = row["messages"][-1]["content"]
        else:
            prompt, response = row["prompt"], row["response"]
        digest = hashlib.sha1(f"{prompt}\x00{response}".encode()).hexdigest()[:16]
        out.append({"id": f"U-{digest}", "task": row.get("task", default_task), "kind": "lm", "chat": True,
                    "prompt": prompt, "target": response, "meta": {"source": Path(path).name}})
    return out


def serve_inbox(run, inbox, once=False):
    inbox, cfg = Path(inbox), run.cfg["live"]
    done = inbox / "done"
    done.mkdir(parents=True, exist_ok=True)
    size, every = cfg["portion_size"], cfg["eval_every"]
    print(f"watching {inbox.resolve()} (portion = {size} examples, {len(run.pending)} waiting); Ctrl+C to stop",
          flush=True)
    try:
        while True:
            for path in sorted(inbox.glob("*.jsonl")):
                new = read_examples(path)
                run.pending.extend(new)
                path.replace(done / path.name)
                print(f"+{len(new)} examples from {path.name}", flush=True)
            while len(run.pending) >= size or (once and run.pending):
                portion, run.pending = run.pending[:size], run.pending[size:]
                task = Counter(ex["task"] for ex in portion).most_common(1)[0][0]
                evaluate = bool(every) and (run.portions_done + 1) % every == 0
                for rec in run.learn(portion, task=task, evaluate=evaluate):
                    run._print(rec, "-")
            if once:
                break
            time.sleep(cfg["poll_seconds"])
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        run.checkpoint()  # also keeps examples that did not fill a portion yet
        print(f"state saved: {run.state_dir} ({run.portions_done} portions, {len(run.pending)} waiting)", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True, help="state directory of the model")
    p.add_argument("--config", help="config for a new model (an existing state keeps its own)")
    p.add_argument("--preset", help="YAML merged on top of a new model's config")
    p.add_argument("--low-memory", "--bf16", action="store_true", dest="low_memory",
                   help="economy mode for a computer without a GPU and with ~8 GB of RAM")
    p.add_argument("--stream", choices=["builtin", "inbox"], default="inbox")
    p.add_argument("--inbox", default="inbox")
    p.add_argument("--journal", help="where to write the journal (default: the state directory)")
    p.add_argument("--export", help="after the builtin stream, export a Hugging Face model here")
    p.add_argument("--once", action="store_true", help="learn what is in the inbox now and exit")
    p.add_argument("--list", action="store_true")
    p.add_argument("--rollback", metavar="CHECKPOINT")
    p.add_argument("--set", nargs="+", default=[], dest="overrides", metavar="KEY=VALUE")
    args = p.parse_args(argv)
    state = Path(args.state)

    if args.list:
        print("\n".join(list_checkpoints(state)) or "no checkpoints")
        return
    if args.rollback:
        rollback(state, args.rollback)
        print(f"{args.rollback} is now the newest checkpoint")
        return

    cfg = learner_config(state, args.config, args.preset, args.overrides, args.low_memory)
    run = Run(cfg, cfg["seeds"][0], args.journal or state, state_dir=state)
    if args.stream == "builtin":
        run.run()  # resumes from the newest checkpoint by itself
        if args.export:
            print("exported to", export_model(state, args.export))
        return
    if not run.restore():
        run.seed_self_replay()  # a brand-new model: its own answers to general requests go to the buffer
    serve_inbox(run, args.inbox, once=args.once)


if __name__ == "__main__":
    main()
