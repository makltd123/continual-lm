"""Chat with the model and teach it (FR-16).

    python chat.py --state states/ewai-e5

Type a message to get an answer. Then you can give feedback on that answer:
    /good            the answer is right: learn it as an example
    /fix <answer>    the answer is wrong, this is the right one: learn it, plus a DPO pair (bad -> good)
    /learn           learn the collected feedback now (otherwise every `chat.learn_every` examples)
    /sleep           run a sleep phase now
    /save            save the state;  /export <dir>  export a Hugging Face model
    /quit            save and exit
Each message is answered on its own (no conversation history), like the training examples.
"""

import argparse
import hashlib
from pathlib import Path

from live import learner_config
from src.dpo import preference_pair
from src.state import export_model
from src.train_loop import Run


def example(prompt, answer):
    digest = hashlib.sha1(f"{prompt}\x00{answer}".encode()).hexdigest()[:16]
    return {"id": f"U-{digest}", "task": "U", "kind": "lm", "chat": True, "prompt": prompt, "target": answer,
            "meta": {"source": "chat"}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--config", help="config for a new model (an existing state keeps its own)")
    p.add_argument("--preset", help="YAML merged on top of a new model's config")
    p.add_argument("--low-memory", "--bf16", action="store_true", dest="low_memory",
                   help="economy mode for a computer without a GPU and with ~8 GB of RAM")
    p.add_argument("--set", nargs="+", default=[], dest="overrides", metavar="KEY=VALUE")
    args = p.parse_args(argv)
    state = Path(args.state)
    cfg = learner_config(state, args.config, args.preset, args.overrides, args.low_memory)

    run = Run(cfg, cfg["seeds"][0], state, state_dir=state)
    restored = run.restore()
    if not restored:
        run.seed_self_replay()
    print(f"loaded {restored or 'the base model'}; /help for commands", flush=True)
    feedback, pairs, last = list(run.pending), [], None

    def learn():
        nonlocal feedback, pairs
        if not feedback:
            print("(nothing to learn yet)")
            return
        recs = run.learn(feedback, task="U", pairs=pairs or None)
        run.pending = []
        print(f"learnt {len(feedback)} examples"
              f"{f' and {len(pairs)} preference pairs' if pairs else ''}"
              f"{', slept' if len(recs) > 1 else ''}; state saved", flush=True)
        feedback, pairs = [], []

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            text = "/quit"
        if not text:
            continue
        cmd, _, rest = text.partition(" ")
        if cmd == "/help":
            print(__doc__)
        elif cmd == "/good" and last:
            feedback.append(example(*last))
            print(f"ok ({len(feedback)} to learn)")
        elif cmd == "/fix" and last and rest:
            feedback.append(example(last[0], rest))
            pair = preference_pair(run, feedback[-1], last[1])
            if pair:
                pairs.append(pair)
            print(f"ok ({len(feedback)} to learn)")
        elif cmd == "/learn":
            learn()
        elif cmd == "/sleep" and not cfg["sleep"]["enabled"]:
            print("sleep is off in this config")
        elif cmd == "/sleep":
            print("sleeping (a few minutes on a CPU)...", flush=True)
            run.sleep(evaluate=False)
            run.checkpoint()
            print("slept; state saved")
        elif cmd == "/save":
            run.pending = feedback
            run.checkpoint()
            print("state saved")
        elif cmd == "/export" and rest:
            run.pending = feedback
            run.checkpoint()
            print("exported to", export_model(state, rest))
        elif cmd == "/quit":
            run.pending = feedback  # unlearnt feedback waits in the state for the next session
            run.checkpoint()
            print("state saved, bye")
            return
        elif cmd.startswith("/"):
            print("unknown command or missing argument; /help")
        else:
            last = (text, run.reply(text))
            print("model>", last[1], flush=True)
            if len(feedback) >= cfg["chat"]["learn_every"]:
                learn()
            continue
        if cmd in ("/good", "/fix") and len(feedback) >= cfg["chat"]["learn_every"]:
            learn()


if __name__ == "__main__":
    main()
