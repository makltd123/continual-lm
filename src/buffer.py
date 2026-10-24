"""Replay buffer with per-task quotas (FR-4).

- Capacity is split equally between the tasks seen so far (m = M/T, as in GEM),
  so the buffer cannot drift towards one task.
- Insertion: reservoir sampling within each task (a uniform sample of the task's stream).
- Sampling: tasks get equal shares; within a task either uniform or with probability
  proportional to loss**alpha ("examples where the model erred come first").
  Priorities are refreshed every time an example is replayed.
- Cleanup (sleep phase): keep the most useful examples by loss and diversity.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, sample="loss", alpha=1.0, noise_quantile=0.98, seed=0):
        if sample not in ("uniform", "loss"):
            raise ValueError(f"replay.sample must be 'uniform' or 'loss', got {sample!r}")
        self.capacity, self.sample_policy, self.alpha = capacity, sample, alpha
        self.noise_quantile = noise_quantile
        self.rng = np.random.default_rng(seed)
        self.items = {}     # task -> list of examples
        self.members = set()
        self.priority = {}  # example id -> latest loss
        self.n_seen = {}    # task -> number of examples offered (for reservoir sampling)

    def __len__(self):
        return len(self.members)

    def sizes(self):
        return {t: len(v) for t, v in self.items.items()}

    def quota(self):
        return self.capacity // max(1, len(self.items))

    def _drop(self, ex):
        self.members.discard(ex["id"])
        self.priority.pop(ex["id"], None)

    def add(self, examples, losses=None):
        for i, ex in enumerate(examples):
            if ex["id"] in self.members:
                continue
            task = ex["task"]
            if task not in self.items:
                self.items[task], self.n_seen[task] = [], 0
                self._shrink_to_quota()
            items, quota = self.items[task], self.quota()
            self.n_seen[task] += 1
            if len(items) < quota:
                items.append(ex)
            else:
                j = int(self.rng.integers(self.n_seen[task]))
                if j >= quota:
                    continue
                self._drop(items[j])
                items[j] = ex
            self.members.add(ex["id"])
            if losses is not None and np.isfinite(losses[i]):  # an fp16 overflow must not poison sampling
                self.priority[ex["id"]] = float(losses[i])

    def _shrink_to_quota(self):
        quota = self.quota()
        for task, items in self.items.items():
            if len(items) > quota:
                keep = set(self.rng.choice(len(items), size=quota, replace=False).tolist())
                for k, ex in enumerate(items):
                    if k not in keep:
                        self._drop(ex)
                self.items[task] = [ex for k, ex in enumerate(items) if k in keep]

    def update_priorities(self, losses_by_id):
        for ex_id, loss in losses_by_id.items():
            if ex_id in self.members and np.isfinite(loss):
                self.priority[ex_id] = float(loss)

    def _weights(self, items):
        if self.sample_policy == "uniform":
            return None
        known = [self.priority[ex["id"]] for ex in items if ex["id"] in self.priority]
        fill = float(np.mean(known)) if known else 1.0
        loss = np.array([self.priority.get(ex["id"], fill) for ex in items], dtype=float)
        # Cap the top tail: the highest losses are often noise rather than useful difficulty.
        loss = np.minimum(loss, np.quantile(loss, self.noise_quantile))
        w = np.maximum(loss, 1e-6) ** self.alpha
        return w / w.sum()

    def sample(self, n, exclude_task=None):
        """n examples, split evenly between eligible tasks."""
        tasks = [t for t, v in self.items.items() if v and t != exclude_task]
        if n <= 0 or not tasks:
            return []
        shares = np.full(len(tasks), n // len(tasks))
        shares[: n % len(tasks)] += 1
        out = []
        for task, k in zip(tasks, self.rng.permutation(shares)):
            items, k = self.items[task], int(k)
            idx = self.rng.choice(len(items), size=k, replace=k > len(items), p=self._weights(items))
            out.extend(items[i] for i in idx)
        return out

    def all_examples(self):
        return [ex for items in self.items.values() for ex in items]

    def cleanup(self, keep_frac, losses, embeddings, diversity_weight):
        """Keep keep_frac of each task's quota, chosen greedily by loss and diversity (MMR).

        losses / embeddings: dicts id -> value for every buffered example (fresh scores).
        Examples above the noise quantile of loss are dropped first.
        """
        report = {}
        keep_n = max(1, int(self.quota() * keep_frac))
        for task, items in self.items.items():
            loss = np.array([losses[ex["id"]] for ex in items], dtype=float)
            loss[~np.isfinite(loss)] = np.inf  # overflowed scores count as noise
            ok = loss <= np.quantile(loss[np.isfinite(loss)], self.noise_quantile) if np.isfinite(loss).any() \
                else np.zeros(len(loss), dtype=bool)
            cand = [i for i in range(len(items)) if ok[i]]
            if len(cand) > keep_n:
                emb = np.stack([embeddings[items[i]["id"]].numpy() for i in cand])
                cand = _mmr(cand, loss, emb, keep_n, diversity_weight)
            kept = [items[i] for i in cand]
            report[task] = {"before": len(items), "after": len(kept), "noise_dropped": int((~ok).sum())}
            for ex in items:
                self._drop(ex)
            for ex in kept:
                self.members.add(ex["id"])
                self.priority[ex["id"]] = float(losses[ex["id"]])
            self.items[task] = kept
        return report


def _mmr(cand, loss, emb, k, diversity_weight):
    """Greedy selection: score = (1-w) * normalised loss + w * distance to the nearest kept example."""
    lc = loss[cand]
    norm = (lc - lc.min()) / (np.ptp(lc) + 1e-9)
    chosen = [int(np.argmax(norm))]
    nearest = 1.0 - emb @ emb[chosen[0]]  # cosine distance (embeddings are unit vectors)
    for _ in range(k - 1):
        score = (1 - diversity_weight) * norm + diversity_weight * nearest
        score[chosen] = -np.inf
        j = int(np.argmax(score))
        chosen.append(j)
        nearest = np.minimum(nearest, 1.0 - emb @ emb[j])
    return [cand[j] for j in sorted(chosen)]
