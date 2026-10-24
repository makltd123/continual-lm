"""Curriculum (FR-8): order examples from easy to hard.

Difficulty measures:
- length:   number of tokens;
- ref_loss: loss of the frozen reference (base) model, computed once and cached;
- native:   a generator's own level (task C), falling back to length.
Scope `task` sorts the whole task, so early portions are easier; scope `portion` sorts inside
each portion only.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.scoring import score_examples


def difficulty(examples, measure, tokens, ref_loss=None):
    if measure == "length":
        return np.array([tokens.length(ex) for ex in examples], dtype=float)
    if measure == "ref_loss":
        return np.array([ref_loss[ex["id"]] for ex in examples], dtype=float)
    if measure == "native":
        return np.array([ex["meta"].get("level", tokens.length(ex)) for ex in examples], dtype=float)
    raise ValueError(f"curriculum.measure must be length, ref_loss or native, got {measure!r}")


def sort_easy_first(examples, scores, rng):
    """Ascending difficulty; ties broken randomly so equal-difficulty runs still differ by seed."""
    tiebreak = rng.random(len(examples))
    order = np.lexsort((tiebreak, scores))
    return [examples[i] for i in order]


def mix(new, replay, ordered, rng):
    """Combine new and replay examples.

    ordered=True keeps the curriculum order of the new examples and spreads replay evenly
    between them; otherwise everything is shuffled together.
    """
    if not ordered:
        batch = list(new) + list(replay)
        rng.shuffle(batch)
        return batch
    keyed = [((i + 0.5) / max(1, len(new)), 0, ex) for i, ex in enumerate(new)]
    keyed += [((j + 0.5) / max(1, len(replay)), 1, ex) for j, ex in enumerate(replay)]
    keyed.sort(key=lambda x: (x[0], x[1]))
    return [ex for _, _, ex in keyed]


def reference_losses(model, tokens, examples, precision, device, batch_size, cache_dir, model_id):
    """Loss of the untrained model on every example, cached on disk (shared by all seeds)."""
    content = sorted((ex["id"], ex["prompt"], ex["target"]) for ex in examples)
    digest = hashlib.sha256(json.dumps([model_id, tokens.max_len, content]).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"refloss_{digest}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    res = score_examples(model, tokens, examples, precision, device, batch_size)
    losses = {ex["id"]: loss for ex, loss in zip(examples, res["mean"])}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(losses), encoding="utf-8")
    return losses
