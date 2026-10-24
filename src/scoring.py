"""Tokenization, batching and per-example losses shared by training, selection and evaluation.

Examples are dicts with `prompt` and `target`; the loss is taken on the target only.
Chat examples (`chat: true`) put the prompt as the user's message into the model's chat template,
and the target is the assistant's answer ending with the end-of-turn token. Plain-text examples
(the general language test) have an empty prompt and the text as the target.
"""

import torch
import torch.utils.checkpoint
import torch.nn.functional as F


class Tokens:
    """Cache of (prompt_ids, target_ids) per example id."""

    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len
        self.sep = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        self.cache = {}

    def prompt_text(self, ex):
        if ex.get("chat"):
            return self.tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}],
                                                add_generation_prompt=True, tokenize=False)
        return ex["prompt"] or ""

    def add(self, examples):
        todo = [ex for ex in examples if ex["id"] not in self.cache]
        if not todo:
            return
        enc = lambda texts: self.tok(texts, add_special_tokens=False)["input_ids"]
        prompts = enc([self.prompt_text(ex) for ex in todo])
        targets = enc([ex["target"] for ex in todo])
        eos = self.tok.eos_token_id
        for ex, p, t in zip(todo, prompts, targets):
            if not p:
                p = [self.sep]  # plain text: the separator lets the model predict the first word too
            t = t + [eos]
            if ex["kind"] != "lm":
                t = t[: self.max_len // 2]
            over = len(p) + len(t) - self.max_len
            if over > 0:
                if ex["kind"] == "lm":  # long text: cut its end
                    t = t[: len(t) - over]
                else:  # long request: cut its middle (the text passage), keep the template and the question
                    mid = len(p) // 2
                    p = p[:mid] + p[mid + over:]
            self.cache[ex["id"]] = (p, t)

    def __getitem__(self, ex):
        return self.cache[ex["id"]]

    def length(self, ex):
        p, t = self.cache[ex["id"]]
        return len(p) + len(t)

    def n_target(self, ex):
        return len(self.cache[ex["id"]][1])


def collate(pairs, pad_id, device):
    width = max(len(p) + len(t) for p, t in pairs)
    input_ids = torch.full((len(pairs), width), pad_id, dtype=torch.long)
    labels = torch.full((len(pairs), width), -100, dtype=torch.long)
    attention = torch.zeros((len(pairs), width), dtype=torch.long)
    for i, (p, t) in enumerate(pairs):
        seq = p + t
        input_ids[i, : len(seq)] = torch.tensor(seq)
        labels[i, len(p): len(seq)] = torch.tensor(t)
        attention[i, : len(seq)] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": attention.to(device), "labels": labels.to(device)}


def _decoder_and_head(model):
    """(transformer body, output layer) when the model is body + plain linear head, else None.

    Then the head can run on target positions only: with a 152k-token vocabulary it is the most
    expensive part, and a QA prompt has ~150 tokens for a ~5-token answer.
    """
    hf = model.get_base_model() if hasattr(model, "get_base_model") else model
    if getattr(hf.config, "final_logit_softcapping", None) or not hasattr(hf, "get_decoder"):
        return None
    head = hf.get_output_embeddings()
    return (hf.get_decoder(), head) if isinstance(head, torch.nn.Linear) else None


# Target tokens per piece when the output layer and the loss are computed. With a 152k-token vocabulary the
# logits of one long batch take gigabytes; in pieces (recomputed in the backward pass) the peak stays small.
# The result is the same. Run sets it from train.head_chunk.
HEAD_CHUNK = 2048


def _target_nll(head, h, target, need_em):
    """Per-token NLL of the target tokens (and whether the argmax is wrong), computed in pieces."""
    def piece(hc, tc):
        return F.cross_entropy(head(hc).float(), tc, reduction="none")

    nll, wrong = [], []
    for i in range(0, h.shape[0], HEAD_CHUNK):
        hc, tc = h[i:i + HEAD_CHUNK], target[i:i + HEAD_CHUNK]
        if torch.is_grad_enabled() and h.shape[0] > HEAD_CHUNK:
            nll.append(torch.utils.checkpoint.checkpoint(piece, hc, tc, use_reentrant=False))
        else:
            nll.append(piece(hc, tc))
        if need_em:
            with torch.no_grad():
                wrong.append((head(hc).argmax(-1) != tc).long())
    return torch.cat(nll), (torch.cat(wrong) if need_em else None)


def forward_losses(model, batch, need_em=False, need_emb=False):
    """Per-example target NLL (mean and sum), token count, and optionally exact match / embeddings.

    `em` is teacher-forced exact match: every target token (including EOS) is the argmax given
    the gold prefix. That is exactly the condition for greedy decoding to reproduce the target.
    """
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    parts = _decoder_and_head(model)
    target = labels[mask]
    if parts:
        decoder, head = parts
        hidden = decoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        nll, wrong = _target_nll(head, hidden[:, :-1][mask], target, need_em)
    else:
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    output_hidden_states=need_emb)
        hidden = out.hidden_states[-1] if need_emb else None
        logits = out.logits[:, :-1][mask]
        nll = F.cross_entropy(logits.float(), target, reduction="none")
        wrong = (logits.argmax(-1) != target).long() if need_em else None
    row = mask.nonzero()[:, 0]  # example index of every target token
    n_batch = labels.shape[0]
    total = torch.zeros(n_batch, device=nll.device, dtype=nll.dtype).index_add(0, row, nll)
    n_tok = mask.sum(1)
    res = {"mean": total / n_tok.clamp_min(1), "sum": total, "n": n_tok}
    if need_em:
        res["em"] = torch.zeros(n_batch, device=wrong.device, dtype=torch.long).index_add(0, row, wrong) == 0
    if need_emb:
        att = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        res["emb"] = F.normalize(((hidden * att).sum(1) / att.sum(1)).float(), dim=-1)
    return res


@torch.no_grad()
def score_examples(model, tokens, examples, precision, device, batch_size, need_em=False, need_emb=False):
    """Run the model over examples (sorted by length for speed) and return per-example results."""
    was_training = model.training
    model.eval()
    tokens.add(examples)
    order = sorted(range(len(examples)), key=lambda i: tokens.length(examples[i]))
    keys = ["mean", "sum", "n"] + (["em"] if need_em else []) + (["emb"] if need_emb else [])
    results = {k: [None] * len(examples) for k in keys}
    for start in range(0, len(order), batch_size):
        idx = order[start: start + batch_size]
        batch = collate([tokens[examples[i]] for i in idx], tokens.tok.pad_token_id, device)
        with precision.autocast():
            res = forward_losses(model, batch, need_em, need_emb)
        for k in keys:
            values = res[k].cpu()
            for j, i in enumerate(idx):
                results[k][i] = values[j] if k == "emb" else values[j].item()
    if was_training:
        model.train()
    return results


def stop_ids(tok):
    """End-of-turn and end-of-text tokens: either one ends a generated answer."""
    return sorted({i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None})


@torch.no_grad()
def greedy_generate(model, tokens, examples, device, precision, max_new_tokens, batch_size,
                    repetition_penalty=1.0, temperature=None, top_p=None):
    """Answers to the examples' prompts: greedy by default (instruction test, DPO step); sampled when a
    temperature is given (self-generated replay, so it keeps the model's variety)."""
    tok = tokens.tok
    stops = stop_ids(tok)
    was_training = model.training
    model.eval()
    tokens.add(examples)
    answers = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start: start + batch_size]
        prompts = [tokens[ex][0] for ex in chunk]
        width = max(len(p) for p in prompts)
        ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, p in enumerate(prompts):  # left padding for batched generation
            ids[i, width - len(p):] = torch.tensor(p)
            att[i, width - len(p):] = 1
        with precision.autocast():
            out = model.generate(input_ids=ids.to(device), attention_mask=att.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=bool(temperature), use_cache=True,
                                 repetition_penalty=repetition_penalty, pad_token_id=tok.pad_token_id,
                                 eos_token_id=stops, temperature=temperature or None,
                                 top_p=top_p if temperature else None, top_k=None)
        for row in out[:, width:].tolist():
            cut = [row.index(s) for s in stops if s in row]
            answers.append(tok.decode(row[: min(cut)] if cut else row, skip_special_tokens=True))
    if was_training:
        model.train()
    return answers
