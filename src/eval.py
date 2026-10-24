"""Evaluation and forgetting metrics (FR-5).

Each task gets a `score` where higher is better:
- QA tasks (B, C): exact match (teacher-forced, equal to greedy-decoding exact match);
- answer/LM tasks (A, D, general test G): minus the mean token NLL in nats (perplexity = exp(-score));
- instruction following (T0): share of instructions whose every condition holds in the greedy answer.

Metrics follow reports/research.md and the spec:
- control base: the untrained model's scores (row 0); `delta_vs_base` = final - base for every task;
- forgetting (Chaudhry et al. 2018): best score at any earlier checkpoint minus final score;
- normalised forgetting: forgetting divided by the gain over the base model (share of the gain lost);
- retention of T0: final / base, the acceptance criterion for the final model (E5);
- BWT, FWT (Lopez-Paz & Ranzato 2017), with an evaluation after every portion.
"""

import math
from collections import defaultdict

import numpy as np

from data.instructions import check_one

MIN_GAIN = 0.01


def evaluate(model, tokens, sets, precision, device, batch_size, gen=None):
    """gen: {'max_new_tokens', 'repetition_penalty'} for generation-based tasks (instructions)."""
    from src.scoring import score_examples  # torch only where the model is evaluated, so report.py stays light

    out = {}
    for name, examples in sets.items():
        kind = examples[0]["kind"]
        if kind == "inst":
            out[name] = evaluate_instructions(model, tokens, examples, precision, device, batch_size, gen or {})
            continue
        res = score_examples(model, tokens, examples, precision, device, batch_size, need_em=kind == "qa")
        loss = sum(res["sum"]) / max(1, sum(res["n"]))
        metrics = {"n": len(examples), "loss": loss}
        if kind == "qa":
            metrics["em"] = float(np.mean(res["em"]))
            metrics["score"] = metrics["em"]
        else:
            metrics["ppl"] = math.exp(min(loss, 50.0))
            metrics["score"] = -loss
        out[name] = metrics
    return out


def evaluate_instructions(model, tokens, examples, precision, device, batch_size, gen):
    from src.scoring import greedy_generate

    answers = greedy_generate(model, tokens, examples, device, precision, gen.get("max_new_tokens", 160),
                              batch_size, gen.get("repetition_penalty", 1.0))
    by_type = defaultdict(list)
    for ex, answer in zip(examples, answers):
        by_type[ex["meta"]["type"]].append(all(check_one(c, answer) for c in ex["meta"]["checks"]))
    passed = [ok for oks in by_type.values() for ok in oks]
    acc = float(np.mean(passed))
    return {"n": len(examples), "acc": acc, "score": acc,
            "by_type": {t: float(np.mean(oks)) for t, oks in sorted(by_type.items())},
            "samples": [{"id": ex["id"], "answer": a[:300]} for ex, a in zip(examples[:3], answers[:3])]}


def relative_drop(before, after, tasks):
    """Mean drop over tasks: absolute for accuracies, relative to the loss for LM tasks."""
    drops = []
    for t in tasks:
        b, a = before[t], after[t]
        scale = 1.0 if ("em" in b or "acc" in b) else max(abs(b["score"]), 1e-6)
        drops.append((b["score"] - a["score"]) / scale)
    return float(np.mean(drops)) if drops else 0.0


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(records, order, mode, retention_task="T0"):
    """Per-task metrics from the evaluation history (row 0 = control base, then every portion / sleep)."""
    rows = [r for r in records if "eval" in r]
    tasks = order + [t for t in rows[0]["eval"] if t not in order]
    per_task = {}
    for task in tasks:
        s = [r["eval"][task]["score"] for r in rows]
        base, final = s[0], s[-1]
        best = max(s[1:-1]) if len(s) > 2 else base
        d = {"base": base, "final": final, "best": best, "forgetting": best - final,
             "delta_vs_base": final - base}
        gain = best - base
        # Below MIN_GAIN (1 point of accuracy / 0.01 nats) the ratio is noise divided by noise.
        d["forgetting_norm"] = (best - final) / gain if gain >= MIN_GAIN else None
        if mode == "sequential" and task in order:
            idx = [i for i, r in enumerate(rows) if r.get("task") == task and r.get("phase") == "train"]
            first, last = idx[0], idx[-1]
            d.update(R_jj=s[last], bwt=final - s[last], fwt=s[first - 1] - base,
                     first_portion=s[first], aulc=float(np.mean(s[first:last + 1])))
        per_task[task] = d
    out = {"tasks": per_task,
           # Raw scores mix units, so only the normalised value is averaged. Tasks before the last one,
           # in every mode, so joint training (E6) is comparable.
           "avg_forgetting_norm": _mean([per_task[t]["forgetting_norm"] for t in order[:-1]])}
    if retention_task in per_task and per_task[retention_task]["base"] > 0:
        d = per_task[retention_task]
        out["retention"] = d["final"] / d["base"]
        out["retention_min"] = min(r["eval"][retention_task]["score"] for r in rows[1:]) / d["base"] \
            if len(rows) > 1 else 1.0
    return out
