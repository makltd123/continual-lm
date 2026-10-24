"""Sleep phase (FR-9): consolidation without new data.

1. Replay: a few steps on examples drawn from the buffer (all tasks, current one included).
2. Merge: fold the LoRA adapter into the base weights and start a fresh adapter.
3. Check: evaluate before/after; if the learnt tasks or instruction following got worse by more
   than the tolerance, roll back to the pre-sleep weights.
4. Buffer cleanup: re-score the buffer and keep the most useful examples (loss + diversity).
The merged adapter is written to the state directory, so the weights can be rebuilt after a restart.
"""

from src.eval import relative_drop
from src.model import is_lora, merge_and_renew, restore, snapshot
from src.scoring import score_examples
from src.state import adapter_weights, save_merged_adapter


def sleep_phase(run, before=None, evaluate=True):
    """`before`: the evaluation made right after the last portion (same weights), if there is one.
    evaluate=False skips the before/after check and the rollback (used in chat, where it is too slow)."""
    cfg = run.cfg["sleep"]
    rec = {"phase": "sleep", "task": run.current_task}
    if evaluate and before is None:
        before = run.evaluate(run.eval_sets)
    snap = snapshot(run.model) if cfg["rollback"] and evaluate else None

    replay = run.buffer.sample(cfg["replay_examples"])
    stats = run.train(replay, lr=cfg["lr"], ordered=False, kind="sleep")
    run.buffer.update_priorities(stats["losses"])
    rec["replay"] = {"n": len(replay), "steps": stats["steps"], "loss": stats["loss"]}

    merged = cfg["merge"] and is_lora(run.model)
    if merged:
        weights = adapter_weights(run.model) if run.state_dir else None
        run.model = merge_and_renew(run.model, run.cfg)
    rec["merged"] = merged

    rec["rolled_back"] = False
    if evaluate:
        after = run.evaluate(run.eval_sets)
        drop = relative_drop(before, after, run.protected_tasks)
        rec["drop_vs_before"] = drop
        rec["rolled_back"] = bool(snap is not None and drop > cfg["rollback_tolerance"])
        if rec["rolled_back"]:
            restore(run.model, snap)
            after = before  # identical weights to the moment `before` was measured
        rec["eval"] = after
    if merged and not rec["rolled_back"] and run.state_dir:
        run.merged.append(save_merged_adapter(run.state_dir, weights, len(run.merged) + 1))

    if cfg.get("cleanup", True):  # re-scores the whole buffer: minutes on a CPU, so low-memory mode skips it
        examples = run.buffer.all_examples()
        res = score_examples(run.model, run.tokens, examples, run.precision, run.device,
                             run.cfg["eval"]["batch_size"], need_emb=True)
        losses = {ex["id"]: v for ex, v in zip(examples, res["mean"])}
        embeddings = {ex["id"]: v for ex, v in zip(examples, res["emb"])}
        rec["buffer_cleanup"] = run.buffer.cleanup(cfg["buffer_keep_frac"], losses, embeddings,
                                                   cfg["diversity_weight"])
    return rec
