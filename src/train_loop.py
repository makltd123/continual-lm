"""Main continual-learning loop (FR-3, FR-5, FR-10).

For every portion of the stream:
  1. take the portion (already ordered easy-to-hard if the curriculum is on);
  2. score it and keep the "surprising" examples (if enabled);
  3. add replay examples from the buffer (if enabled);
  4. train the adapter (plus a DPO round for tasks with a script verifier, if enabled);
  5. evaluate every task and write the journal.
Every `sleep.every` portions a sleep phase runs (if enabled).

`Run` is one learner: a model with its buffer, counters and random state. With a state directory it
saves a checkpoint after every portion and continues from the newest one after a restart (FR-14);
the same class serves the experiments (run.py), the continuous mode (live.py) and the chat (chat.py).
"""

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from data.tasks import VERIFIERS, load_all
from src import curriculum, self_replay, surprise
from src.buffer import ReplayBuffer
from src.dpo import dpo_round, dpo_train
from src.eval import evaluate, summarize
from src.logger import RunLogger, code_version, environment, stamp
from src.model import Precision, load_model, memory_gb, pick_device, trainable_parameters
from src import scoring
from src.scoring import Tokens, collate, forward_losses, greedy_generate, score_examples
from src.sleep import sleep_phase
from src.state import export_model, latest_checkpoint, load_checkpoint, save_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Run:
    def __init__(self, cfg, seed, run_dir, state_dir=None, resume=True):
        self.cfg, self.seed = cfg, seed
        set_seed(seed)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.device = pick_device(cfg.get("device", "auto"))
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.precision = Precision(self.device, cfg["model"]["precision"])
        scoring.HEAD_CHUNK = cfg["train"].get("head_chunk", scoring.HEAD_CHUNK)
        self.state_dir = Path(state_dir) if state_dir else None
        self.resume_from = latest_checkpoint(self.state_dir) if self.state_dir and resume else None
        self.log = RunLogger(run_dir, cfg, seed, fresh=self.resume_from is None)
        self.order = cfg["stream"]["order"]
        self.tasks, manifests = load_all(cfg["data"], self.order)
        self.model, self.tok = load_model(cfg, self.device)
        self.tokens = Tokens(self.tok, cfg["data"]["max_len"])

        ev = cfg["eval"]
        # Trained tasks are checked on validation data; eval-only tasks (G, T0) have a test split only.
        self.eval_sets = {t: s.get("val", s["test"])[: ev["quick_n"]] for t, s in self.tasks.items()}
        self.curve_sets = {t: s[: ev["curve_n"]] for t, s in self.eval_sets.items()}
        self.test_sets = {t: s["test"] for t, s in self.tasks.items()}

        rcfg = cfg["replay"]
        self.buffer = None
        if rcfg["enabled"] or cfg["sleep"]["enabled"]:
            self.buffer = ReplayBuffer(rcfg["capacity"], rcfg["sample"], rcfg["alpha"], rcfg["noise_quantile"], seed)

        self.steps = {"train": 0, "sleep": 0, "dpo": 0}
        self.examples_new = self.examples_total = self.examples_new_task = 0
        self.current_task, self.seen_tasks, self.task_start = None, [], 0
        self.portions_done, self.sleeps_done = 0, 0
        self.merged = []    # adapters folded into the weights by sleep phases (paths inside state_dir)
        self.pending = []   # examples received but not learnt yet (continuous mode)
        self.t0, self.elapsed_before = time.time(), 0.0
        n_params = sum(p.numel() for p in self.model.parameters())
        meta = {"name": cfg["name"], "seed": seed, "started": stamp(), "code": code_version(),
                "env": environment(self.device), "precision": self.precision.name, "data": manifests,
                "model": {"name": cfg["model"]["name"], "revision": cfg["model"]["revision"], "params": n_params,
                          "trainable": sum(p.numel() for p in trainable_parameters(self.model))}}
        if self.resume_from:
            self.log.append("resumes.jsonl", {**meta, "checkpoint": self.resume_from})
        else:
            self.log.write_json("meta.json", meta)

    def elapsed(self):
        return self.elapsed_before + time.time() - self.t0

    def checkpoint(self):
        if self.state_dir:
            save_checkpoint(self, self.state_dir, self.cfg["checkpoint"]["keep"])

    def restore(self):
        """Load the newest checkpoint if there is one; returns its name."""
        if self.resume_from:
            load_checkpoint(self, self.state_dir, self.resume_from)
        return self.resume_from

    def seed_self_replay(self):
        """Put the original model's answers to general requests into the buffer (before any training)."""
        if not self.cfg["self_replay"]["enabled"]:
            return
        if self.buffer is None:
            raise ValueError("self_replay needs a replay buffer: enable replay or sleep")
        examples = self_replay.build(self)
        self.buffer.add(examples)
        rec = {"portion": 0, "phase": "self_replay", "task": "R", "n": len(examples), **self._counters()}
        self.log.metrics(rec)
        print(f"[{self.cfg['name']} s{self.seed}] self-replay: {len(examples)} answers of the original model "
              f"in the buffer", flush=True)

    # --- low-level helpers -------------------------------------------------

    def evaluate(self, sets):
        ev = self.cfg["eval"]
        gen = {"max_new_tokens": ev["gen_max_new_tokens"], "repetition_penalty": ev["repetition_penalty"]}
        return evaluate(self.model, self.tokens, sets, self.precision, self.device, ev["batch_size"], gen)

    @property
    def protected_tasks(self):
        """Tasks a sleep phase must not damage: the ones learnt so far plus instruction following."""
        return [t for t in self.seen_tasks + ["T0"] if t in self.eval_sets]

    def _backward(self, loss):
        scaler = self.precision.scaler
        (scaler.scale(loss) if scaler else loss).backward()

    def _step(self, opt):
        scaler = self.precision.scaler
        params = [p for group in opt.param_groups for p in group["params"]]
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, self.cfg["train"]["max_grad_norm"])
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

    def optimizer_step(self, opt, loss, kind="dpo"):
        self._backward(loss)
        self._step(opt)
        self.steps[kind] += 1

    def train(self, examples, lr, ordered, kind="train", on_step=None):
        """Train on the examples; return steps, mean loss and the per-example loss seen in training.

        A fresh AdamW per call: the adapter can be replaced by a sleep phase between calls.
        """
        tcfg = self.cfg["train"]
        bs, micro = tcfg["batch_size"], tcfg["micro_batch_size"]
        self.model.train()
        self.tokens.add(examples)
        opt = torch.optim.AdamW(trainable_parameters(self.model), lr=lr, weight_decay=tcfg["weight_decay"])
        per_example, batch_losses = {}, []
        epochs = tcfg["epochs_per_portion"] if kind == "train" else 1
        for _ in range(epochs):
            items = list(examples)
            if not ordered:
                self.rng.shuffle(items)
            for start in range(0, len(items), bs):
                batch = items[start: start + bs]
                n_tok = sum(self.tokens.n_target(ex) for ex in batch)
                total = 0.0
                for m in range(0, len(batch), micro):
                    mb = batch[m: m + micro]
                    data = collate([self.tokens[ex] for ex in mb], self.tok.pad_token_id, self.device)
                    with self.precision.autocast():
                        res = forward_losses(self.model, data)
                    if tcfg["loss_reduction"] == "token":
                        loss = res["sum"].sum() / n_tok
                    else:  # every example weighs the same, whatever its length
                        loss = res["mean"].sum() / len(batch)
                    self._backward(loss)
                    total += loss.item()
                    for ex, v in zip(mb, res["mean"].detach().float().cpu().tolist()):
                        per_example[ex["id"]] = v
                self._step(opt)
                batch_losses.append(total)
                self.steps[kind] += 1
                if on_step:
                    on_step()
        return {"steps": len(batch_losses), "loss": float(np.mean(batch_losses)) if batch_losses else None,
                "losses": per_example}

    def _curve_hook(self, force=False):
        """Learning curve of the current task on a small set, every `curve_every_steps` steps."""
        every = self.cfg["eval"]["curve_every_steps"]
        task_step = self.steps["train"] - self.task_start
        if self.current_task in self.curve_sets and (force or task_step % every == 0):
            task = self.current_task
            ev = self.evaluate({task: self.curve_sets[task]})[task]
            self.log.append("curve.jsonl", {"task": task, "task_step": task_step, "steps": self.global_step,
                                            "score": ev["score"], "loss": ev["loss"]})

    @property
    def global_step(self):
        return sum(self.steps.values())

    def _counters(self):
        return {"steps": self.global_step, "steps_train": self.steps["train"], "steps_sleep": self.steps["sleep"],
                "steps_dpo": self.steps["dpo"], "task_step": self.steps["train"] - self.task_start,
                "examples_new": self.examples_new, "examples_total": self.examples_total,
                "time_s": self.elapsed(), "mem_gb": memory_gb(self.device)}

    # --- the stream ----------------------------------------------------------

    def build_stream(self, ref_loss):
        s, cur = self.cfg["stream"], self.cfg["curriculum"]
        size, per_task = s["portion_size"], s["portions_per_task"]
        if s["mode"] == "joint":  # E6: all tasks mixed, same number of examples and portions
            pool = [ex for t in self.order for ex in self.tasks[t]["train"][: per_task * size]]
            self.rng.shuffle(pool)
            return [("joint", i, pool[i * size: (i + 1) * size]) for i in range(len(pool) // size)]
        if s["mode"] != "sequential":
            raise ValueError(f"stream.mode must be 'sequential' or 'joint', got {s['mode']!r}")
        stream = []
        for task in self.order:
            train = list(self.tasks[task]["train"][: per_task * size])
            self.tokens.add(train)
            if cur["enabled"] and cur["scope"] == "task":
                scores = curriculum.difficulty(train, cur["measure"], self.tokens, ref_loss)
                train = curriculum.sort_easy_first(train, scores, self.np_rng)
            else:
                self.rng.shuffle(train)
            for i in range(per_task):
                portion = train[i * size: (i + 1) * size]
                if not portion:
                    break
                if cur["enabled"] and cur["scope"] == "portion":
                    scores = curriculum.difficulty(portion, cur["measure"], self.tokens, ref_loss)
                    portion = curriculum.sort_easy_first(portion, scores, self.np_rng)
                stream.append((task, i, portion))
        return stream

    def start_task(self, task):
        self.current_task = task
        self.task_start = self.steps["train"]
        for t in (self.order if task == "joint" else [task]):
            if t not in self.seen_tasks:
                self.seen_tasks.append(t)
        if self.cfg["eval"]["curve_every_steps"] and task != "joint":
            self._curve_hook(force=True)  # step 0 of the curve: quality before training on the task

    def step_portion(self, k, task, task_portion, portion, evaluate=True):
        cfg = self.cfg
        started = time.time()
        if task != self.current_task:
            self.start_task(task)
        rec = {"portion": k, "phase": "train", "task": task, "task_portion": task_portion, "n_offered": len(portion)}

        new = portion
        if cfg["surprise"]["enabled"]:
            res = score_examples(self.model, self.tokens, portion, self.precision, self.device,
                                 cfg["eval"]["batch_size"])
            new, keep, rec["surprise"] = surprise.select(portion, res["mean"], cfg["surprise"])
            for row in surprise.samples_for_review(portion, res["mean"], keep, cfg["surprise"]["log_samples"]):
                self.log.append("surprise_samples.jsonl", {"portion": k, "task": task, **row})

        replay = []
        if cfg["replay"]["enabled"]:
            ratio = cfg["replay"]["ratio"]
            exclude = task if cfg["replay"]["exclude_current_task"] else None
            replay = self.buffer.sample(round(len(new) * ratio / (1 - ratio)), exclude_task=exclude)

        ordered = cfg["curriculum"]["enabled"]
        batch = curriculum.mix(new, replay, ordered, self.rng)
        hook = self._curve_hook if cfg["eval"]["curve_every_steps"] and task != "joint" else None
        stats = self.train(batch, cfg["train"]["lr"], ordered, on_step=hook)
        self.examples_new += len(new)
        self.examples_total += len(batch)

        if self.buffer is not None:
            self.buffer.update_priorities(stats["losses"])
            self.buffer.add(new, [stats["losses"][ex["id"]] for ex in new])
            rec["buffer"] = self.buffer.sizes()

        if cfg["dpo"]["enabled"] and task in cfg["dpo"]["tasks"]:
            verifier = VERIFIERS[cfg["data"]["tasks"][task]["source"]]
            rec["dpo"] = dpo_round(self, new, verifier)

        rec.update(n_selected=len(new), n_replay=len(replay), train_steps=stats["steps"], train_loss=stats["loss"])
        if evaluate:
            rec["eval"] = self.evaluate(self.eval_sets)
        rec.update(self._counters(), portion_time_s=time.time() - started)
        return rec

    def sleep(self, before=None, evaluate=True):
        rec = sleep_phase(self, before=before, evaluate=evaluate)
        self.sleeps_done += 1
        rec.update(portion=self.portions_done, **self._counters())
        return rec

    def journal(self):
        path = self.log.dir / "metrics.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def run(self):
        """The task stream from the config, portion by portion; resumes from the newest checkpoint."""
        cfg = self.cfg
        if not self.resume_from:
            history = [{"portion": 0, "phase": "base", "task": None, "eval": self.evaluate(self.eval_sets),
                        **self._counters()}]
            self.log.metrics(history[0])
            self.seed_self_replay()

        ref_loss = None
        if cfg["curriculum"]["enabled"] and cfg["curriculum"]["measure"] == "ref_loss":
            used = cfg["stream"]["portions_per_task"] * cfg["stream"]["portion_size"]
            train = [ex for t in self.order for ex in self.tasks[t]["train"][:used]]  # only what the stream uses
            ref_loss = curriculum.reference_losses(
                self.model, self.tokens, train, self.precision, self.device, cfg["eval"]["batch_size"],
                cfg["data"]["cache_dir"], f"{cfg['model']['name']}@{cfg['model']['revision']}")

        stream = self.build_stream(ref_loss)  # built before restoring, so it is the same stream as before
        if self.restore():
            # Records after the checkpoint belong to a portion that will be redone.
            history = [r for r in self.journal() if r["portion"] <= self.portions_done]
            self.log.rewrite("metrics.jsonl", history)
            print(f"[{cfg['name']} s{self.seed}] resumed from {self.resume_from}: "
                  f"{self.portions_done}/{len(stream)} portions done", flush=True)
        every = cfg["checkpoint"]["every"]
        for k, (task, task_portion, portion) in enumerate(stream, 1):
            if k <= self.portions_done:
                continue
            rec = self.step_portion(k, task, task_portion, portion)
            history.append(rec)
            self.log.metrics(rec)
            self._print(rec, len(stream))
            self.portions_done = k
            if cfg["sleep"]["enabled"] and k % cfg["sleep"]["every"] == 0:
                srec = self.sleep(before=rec["eval"])
                history.append(srec)
                self.log.metrics(srec)
                self._print(srec, len(stream))
            if k % every == 0 or k == len(stream):
                self.checkpoint()

        test = self.evaluate(self.test_sets)
        final = {"name": cfg["name"], "seed": self.seed, "finished": stamp(), "test": test,
                 "summary": summarize(history, self.order, cfg["stream"]["mode"]), "cost": self._counters()}
        self.log.write_json("final.json", final)
        if self.state_dir and cfg["checkpoint"].get("export_dir"):
            export_model(self.state_dir, cfg["checkpoint"]["export_dir"])
        self.log.close()
        return final

    def reply(self, message, max_new_tokens=None):
        """The model's answer to one user message (chat mode)."""
        ex = {"id": f"chat-{time.time_ns()}", "kind": "inst", "chat": True, "prompt": message, "target": ""}
        answer = greedy_generate(self.model, self.tokens, [ex], self.device, self.precision,
                                 max_new_tokens or self.cfg["chat"]["max_new_tokens"], 1,
                                 self.cfg["eval"]["repetition_penalty"])[0]
        del self.tokens.cache[ex["id"]]
        return answer

    def learn(self, examples, task="U", evaluate=False, pairs=None):
        """One portion of new examples outside the task stream (continuous mode, chat):
        surprise/replay/update as configured, a DPO round on given preference pairs, sleep when due,
        then a checkpoint. Returns the journal records."""
        k = self.portions_done + 1
        rec = self.step_portion(k, task, None, examples, evaluate=evaluate)
        if pairs:
            rec["dpo"] = dpo_train(self, pairs)
        self.portions_done = k
        self.log.metrics(rec)
        records = [rec]
        if self.cfg["sleep"]["enabled"] and k % self.cfg["sleep"]["every"] == 0:
            records.append(self.sleep(evaluate=evaluate))
            self.log.metrics(records[-1])
        self.checkpoint()
        return records

    def _print(self, rec, total):
        scores = " ".join(f"{t}={m['score']:.3f}" for t, m in rec.get("eval", {}).items())
        extra = " rolled_back" if rec.get("rolled_back") else ""
        print(f"[{self.cfg['name']} s{self.seed}] {rec['phase']:5s} {rec['portion']:>3}/{total} "
              f"task={rec['task']} steps={rec['steps']} {scores} t={rec['time_s']:.0f}s{extra}", flush=True)


def run_experiment(cfg, seed, run_dir):
    # With checkpoints on, the state lives next to the journal and a restarted run continues.
    state_dir = None
    if cfg["checkpoint"]["enabled"]:
        state_dir = Path(cfg["checkpoint"].get("state_dir") or Path(run_dir) / "state")
    run = Run(cfg, seed, run_dir, state_dir=state_dir)
    try:
        return run.run()
    finally:
        del run
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
