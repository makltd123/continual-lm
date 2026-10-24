"""Self-generated replay of general instructions (reports/research.md, source 3: Marek et al. 2026).

Before any training, the original model answers a set of general user requests (Dolly, human-written,
not the T0 test). Its own answers join the replay buffer as task "R", so replay also protects the
model's general instruction following, which narrow tasks wear away first. The answers depend only on
the model, the requests and the generation settings, so they are cached and shared by every seed and run.

Greedy answers are the model's single most likely reply; training on a few hundred of them over and over
narrows the model (perplexity on ordinary text rises). Sampling with the model's own temperature and more
requests keeps its variety (`temperature`, `top_p`, `n` in the config).
"""

import hashlib
import json
from pathlib import Path

import torch

from data.tasks import general_prompts
from src.scoring import greedy_generate


def build(run):
    cfg, mcfg = run.cfg["self_replay"], run.cfg["model"]
    prompts = general_prompts(run.cfg["data"], cfg)
    temperature, top_p = cfg.get("temperature"), cfg.get("top_p")
    key = json.dumps([mcfg["name"], mcfg["revision"], prompts, cfg["max_new_tokens"],
                      run.cfg["eval"]["repetition_penalty"], temperature, top_p, run.cfg["data"]["seed"]])
    path = Path(run.cfg["data"]["cache_dir"]) / f"R_answers_{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"
    if path.exists():
        answers = json.loads(path.read_text(encoding="utf-8"))
    else:
        asks = [{"id": f"R-ask-{i:05d}", "kind": "inst", "chat": True, "prompt": p, "target": ""}
                for i, p in enumerate(prompts)]
        devices = [run.device] if run.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):  # sampling must not shift the run's own random stream
            torch.manual_seed(run.cfg["data"]["seed"])
            answers = greedy_generate(run.model, run.tokens, asks, run.device, run.precision, cfg["max_new_tokens"],
                                      run.cfg["eval"]["batch_size"], run.cfg["eval"]["repetition_penalty"],
                                      temperature, top_p)
        for ex in asks:
            run.tokens.cache.pop(ex["id"], None)
        if path.exists():  # a parallel run finished first: use its answers, so both runs share one set
            answers = json.loads(path.read_text(encoding="utf-8"))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{run.seed}.tmp")
            tmp.write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
    return [{"id": f"R-{i:05d}", "task": "R", "kind": "lm", "chat": True, "prompt": p, "target": a.strip(),
             "meta": {"source": "self_replay"}}
            for i, (p, a) in enumerate(zip(prompts, answers)) if a.strip()]
