"""Fast tests of the pure-Python parts: buffer, surprise selection, curriculum mixing, metrics, task C."""

import numpy as np
import torch

from data import arith
from src import curriculum, surprise
from src.buffer import ReplayBuffer
from src.config import deep_update
from src.eval import relative_drop, summarize


def ex(task, i, kind="qa"):
    return {"id": f"{task}-{i}", "task": task, "kind": kind, "prompt": "q", "target": " a", "meta": {}}


def test_buffer_quotas_and_reservoir():
    buf = ReplayBuffer(capacity=100, sample="uniform", seed=0)
    buf.add([ex("A", i) for i in range(500)])
    assert buf.sizes() == {"A": 100}
    buf.add([ex("B", i) for i in range(500)])
    assert buf.sizes() == {"A": 50, "B": 50}
    buf.add([ex("C", i) for i in range(10)])
    assert buf.sizes() == {"A": 33, "B": 33, "C": 10}
    assert len(buf) == len({e["id"] for e in buf.all_examples()}) == 76
    # Reservoir sampling keeps a spread over the whole stream, not just the first examples.
    ids = [int(e["id"].split("-")[1]) for e in buf.items["A"]]
    assert max(ids) > 250


def test_buffer_sampling_excludes_task_and_prefers_high_loss():
    buf = ReplayBuffer(capacity=40, sample="loss", alpha=1.0, noise_quantile=1.0, seed=0)
    buf.add([ex("A", i) for i in range(20)], losses=[0.01] * 19 + [10.0])
    buf.add([ex("B", i) for i in range(20)], losses=[1.0] * 20)
    sample = buf.sample(1000, exclude_task="B")
    assert {e["task"] for e in sample} == {"A"}
    share_hard = np.mean([e["id"] == "A-19" for e in sample])
    assert share_hard > 0.5
    both = buf.sample(10)
    assert sorted(e["task"] for e in both).count("A") == 5


def test_buffer_cleanup_drops_noise_and_keeps_quota():
    buf = ReplayBuffer(capacity=20, sample="loss", noise_quantile=0.9, seed=0)
    items = [ex("A", i) for i in range(20)]
    buf.add(items)
    losses = {e["id"]: float(i) for i, e in enumerate(items)}
    emb = {e["id"]: torch.nn.functional.normalize(torch.randn(8), dim=0) for e in items}
    report = buf.cleanup(keep_frac=0.5, losses=losses, embeddings=emb, diversity_weight=0.0)
    assert report["A"]["after"] == 10 and report["A"]["noise_dropped"] >= 1
    kept = {e["id"] for e in buf.items["A"]}
    # Losses above the 0.9 quantile (17.1) are noise; with diversity_weight=0 the next-highest are kept.
    assert kept == {f"A-{i}" for i in range(8, 18)}


def test_surprise_two_sided_threshold():
    examples = [ex("A", i) for i in range(100)]
    losses = np.arange(100, dtype=float)
    kept, keep, info = surprise.select(examples, losses, {"mode": "quantile", "low": 0.3, "high": 0.9,
                                                          "min_keep_frac": 0.1})
    kept_losses = losses[keep]
    assert kept_losses.min() >= 29 and kept_losses.max() <= 90
    assert [e["id"] for e in kept] == [examples[i]["id"] for i in np.flatnonzero(keep)]  # order preserved
    rows = surprise.samples_for_review(examples, losses, keep, 2)
    assert {r["why"] for r in rows} == {"dropped_easy", "dropped_noise", "kept"}


def test_curriculum_mix_spreads_replay():
    new = [ex("B", i) for i in range(10)]
    replay = [ex("A", i) for i in range(5)]
    mixed = curriculum.mix(new, replay, ordered=True, rng=None)
    assert [e for e in mixed if e["task"] == "B"] == new  # curriculum order kept
    positions = [i for i, e in enumerate(mixed) if e["task"] == "A"]
    assert max(np.diff(positions)) <= 4


def test_summarize_forgetting_bwt_fwt():
    def row(phase, task, a, b):
        return {"phase": phase, "task": task, "eval": {"A": {"score": a, "em": a}, "B": {"score": b, "em": b}}}
    records = [row("base", None, 0.1, 0.0),
               row("train", "A", 0.6, 0.1),
               row("train", "A", 0.8, 0.2),
               row("train", "B", 0.5, 0.6),
               row("train", "B", 0.4, 0.7)]
    s = summarize(records, ["A", "B"], "sequential")
    a = s["tasks"]["A"]
    assert np.isclose(a["forgetting"], 0.4)                 # best 0.8 - final 0.4
    assert np.isclose(a["forgetting_norm"], 0.4 / 0.7)      # share of the gain over the base model
    assert np.isclose(a["bwt"], 0.4 - 0.8)
    assert np.isclose(s["tasks"]["B"]["fwt"], 0.2 - 0.0)    # before training on B vs base
    assert np.isclose(s["avg_forgetting_norm"], 0.4 / 0.7)


def test_relative_drop_units():
    before = {"A": {"score": -2.0, "loss": 2.0}, "B": {"score": 0.5, "em": 0.5}}
    after = {"A": {"score": -2.1, "loss": 2.1}, "B": {"score": 0.4, "em": 0.4}}
    assert np.isclose(relative_drop(before, after, ["A", "B"]), (0.05 + 0.1) / 2)


def test_arith_generator_is_deterministic_and_verifiable():
    a, b = arith.generate(50, seed=7), arith.generate(50, seed=7)
    assert a == b
    assert {e["meta"]["level"] for e in a} == {1, 2, 3, 4, 5}
    for e in a:
        assert arith.verify(e, e["target"] + " and some trailing text")
        assert not arith.verify(e, " 123456789")


def test_deep_update_keeps_siblings():
    base = {"replay": {"enabled": False, "ratio": 0.3}}
    assert deep_update(base, {"replay": {"enabled": True}}) == {"replay": {"enabled": True, "ratio": 0.3}}
