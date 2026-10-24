"""Selection by "surprise" (FR-7): train only on examples the current model finds hard.

The threshold is two-sided. Too-easy examples (low loss) are dropped as in the spec; the top
tail is dropped too, because the highest losses are often noise rather than useful
difficulty (Mindermann et al. 2022; Hu et al. 2021, see reports/research.md).
"""

import numpy as np


def thresholds(losses, cfg):
    losses = np.asarray(losses, dtype=float)
    if cfg["mode"] == "quantile":
        return float(np.quantile(losses, cfg["low"])), float(np.quantile(losses, cfg["high"]))
    if cfg["mode"] == "absolute":
        return float(cfg["low"]), float(cfg["high"])
    raise ValueError(f"surprise.mode must be 'quantile' or 'absolute', got {cfg['mode']!r}")


def select(examples, losses, cfg):
    """Return (kept examples in their original order, info for the journal)."""
    losses = np.asarray(losses, dtype=float)
    lo, hi = thresholds(losses, cfg)
    keep = (losses >= lo) & (losses <= hi)
    min_keep = int(np.ceil(cfg["min_keep_frac"] * len(examples)))
    if keep.sum() < min_keep:
        # Fall back to the hardest examples that are still under the noise threshold.
        order = [i for i in np.argsort(-losses) if losses[i] <= hi][:min_keep]
        keep = np.zeros(len(examples), dtype=bool)
        keep[order] = True
    kept = [ex for ex, k in zip(examples, keep) if k]
    info = {"low": lo, "high": hi, "n_offered": len(examples), "n_kept": int(keep.sum()),
            "loss_mean": float(losses.mean()), "loss_kept_mean": float(losses[keep].mean())}
    return kept, keep, info


def samples_for_review(examples, losses, keep, n):
    """A few dropped-as-easy, dropped-as-noise and kept examples, for manual inspection."""
    losses = np.asarray(losses, dtype=float)
    order = np.argsort(losses)
    dropped = [i for i in order if not keep[i]]
    median = float(np.median(losses))
    easy = [i for i in dropped if losses[i] < median][:n]
    noisy = [i for i in reversed(dropped) if losses[i] >= median][:n]
    kept = [i for i in order if keep[i]]
    kept = kept[:: max(1, len(kept) // n)][:n] if kept else []

    def row(i, why):
        ex = examples[i]
        return {"why": why, "loss": float(losses[i]), "id": ex["id"],
                "prompt": ex["prompt"][-300:], "target": ex["target"][:300]}

    return ([row(i, "dropped_easy") for i in easy] + [row(i, "dropped_noise") for i in noisy]
            + [row(i, "kept") for i in kept])
