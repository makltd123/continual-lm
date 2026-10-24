"""DPO step from a script verifier's feedback (FR-11).

The model answers a sample of prompts greedily; a script checks each answer. For every wrong
answer we form a pair (chosen = reference answer, rejected = model answer) and take a few
steps on the DPO loss (Rafailov et al. 2023; same formula as TRL's DPOTrainer):

    L = -log sigmoid(beta * [(log pi(y+|x) - log ref(y+|x)) - (log pi(y-|x) - log ref(y-|x))])

The reference is the model at the start of the round, as in TRL when no ref_model is given.
TRL's trainer is built for a standalone run, so the loss is computed here inside our loop.
"""

import torch
import torch.nn.functional as F

from src.model import trainable_parameters
from src.scoring import collate, forward_losses, greedy_generate


def _logps(model, pairs, run):
    batch = collate(pairs, run.tok.pad_token_id, run.device)
    with run.precision.autocast():
        return -forward_losses(model, batch)["sum"]


def preference_pair(run, example, rejected_text):
    """((prompt, chosen), (prompt, rejected)) token pairs; chosen is the example's own target."""
    run.tokens.add([example])
    prompt_ids, chosen = run.tokens[example]
    rejected = run.tok(rejected_text, add_special_tokens=False)["input_ids"] + [run.tok.eos_token_id]
    return None if rejected == chosen else ((prompt_ids, chosen), (prompt_ids, rejected))


def dpo_round(run, examples, verifier):
    """Answer a sample of prompts, check the answers by script, train on the wrong ones."""
    cfg = run.cfg["dpo"]
    prompts = run.rng.sample(examples, min(cfg["n_prompts"], len(examples)))
    answers = greedy_generate(run.model, run.tokens, prompts, run.device, run.precision,
                              cfg["max_new_tokens"], run.cfg["eval"]["batch_size"],
                              run.cfg["eval"]["repetition_penalty"])
    pairs = [preference_pair(run, ex, a) for ex, a in zip(prompts, answers) if not verifier(ex, a)]
    pairs = [p for p in pairs if p]
    stats = {"n_prompts": len(prompts), "accuracy_before": 1 - len(pairs) / max(1, len(prompts))}
    return {**stats, **dpo_train(run, pairs)}


def dpo_train(run, pairs):
    """A few DPO steps on preference pairs; the reference is the model before these steps."""
    cfg = run.cfg["dpo"]
    stats = {"n_pairs": len(pairs), "steps": 0}
    if not pairs:
        return stats
    model = run.model
    half = max(1, run.cfg["train"]["micro_batch_size"] // 2)
    with torch.no_grad():
        model.eval()
        ref = [(_logps(model, [c for c, _ in pairs[i:i + half]], run),
                _logps(model, [r for _, r in pairs[i:i + half]], run)) for i in range(0, len(pairs), half)]
    model.train()
    opt = torch.optim.AdamW(trainable_parameters(model), lr=cfg["lr"])
    losses = []
    for _ in range(cfg["epochs"]):
        for b, i in enumerate(range(0, len(pairs), half)):
            chunk = pairs[i:i + half]
            logp = _logps(model, [c for c, _ in chunk] + [r for _, r in chunk], run)
            pc, pr = logp[: len(chunk)], logp[len(chunk):]
            rc, rr = ref[b]
            loss = -F.logsigmoid(cfg["beta"] * ((pc - rc) - (pr - rr))).mean()
            run.optimizer_step(opt, loss)
            losses.append(loss.item())
            stats["steps"] += 1
    stats.update(loss_first=losses[0], loss_last=losses[-1])
    return stats
