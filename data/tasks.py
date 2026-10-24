"""Loaders for tasks A-D and the eval-only tasks: general language test G and instructions T0 (FR-2).

Tasks A-D are dialogues: the prompt is the user's request, the target is the assistant's answer.

Every source is pinned to a dataset revision in the YAML config. A task's
pool is drawn with `data.seed`, split 80/10/10 and cached as JSON, and the
SHA-256 of the cached file goes into the run journal, so every run can be
traced back to the exact examples it used.
"""

import hashlib
import json
import random
import re
import string
from itertools import islice
from pathlib import Path

from data import arith, instructions


def _read_rows(src, limit=None):
    """Stream rows of a single file of a Hugging Face dataset at a fixed revision."""
    from datasets import load_dataset

    url = f"hf://datasets/{src['repo']}@{src['revision']}/{src['file']}"
    ds = load_dataset(src["format"], data_files={"train": url}, split="train", streaming=True)
    return list(islice(ds, limit)) if limit else list(ds)


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _sample(rows, n, rng, what):
    if len(rows) < n:
        raise ValueError(f"{what}: only {len(rows)} usable rows, need {n}; raise scan_rows or lower pool_size")
    return rng.sample(rows, n)


def _examples(pairs, task):
    """(user request, assistant answer) pairs -> examples."""
    return [{"id": f"{task}-{i:06d}", "prompt": p, "target": t, "meta": {}} for i, (p, t) in enumerate(pairs)]


_STOP = set("""about after again also always another around away back because been before being below between
both came come could didn does doing down each even every from gave give going gone good great have having here
into just know like little look made make many more most much must never next once only other over said same
saying says should some something soon still such take than that their them then there these they thing this
those through time told took under until very want wanted went were what when where which while will with would
your""".split())


def _story_words(text, rng):
    """Three content words of a story, like the word lists TinyStories itself was generated from."""
    words = sorted({w for w in re.findall(r"\b[a-z]{4,}\b", text) if w not in _STOP})
    return rng.sample(words, 3) if len(words) >= 3 else ["happy", "friend", "play"]


def build_tinystories(task, tcfg, src, n, rng):
    rows = _read_rows(src, tcfg.get("scan_rows"))
    texts = [_clean(r["text"]) for r in rows]
    texts = _sample(sorted({t for t in texts if tcfg["min_chars"] <= len(t) <= tcfg["max_chars"]}), n, rng, task)
    pairs = []
    for text in texts:
        a, b, c = _story_words(text, rng)
        pairs.append((f'Write a short story for young children that uses the words "{a}", "{b}" and "{c}".', text))
    return _examples(pairs, task)


def build_arxiv(task, tcfg, src, n, rng):
    rows = _read_rows(src, tcfg.get("scan_rows"))
    by_abstract = {}
    for r in rows:
        abstract, title = _clean(str(r.get("abstract") or "")), _clean(str(r.get("title") or ""))
        if title and tcfg["min_chars"] <= len(abstract) <= tcfg["max_chars"]:
            by_abstract.setdefault(abstract, title)
    abstracts = _sample(sorted(by_abstract), n, rng, task)
    return _examples([(f'Write the abstract of a machine learning research paper titled "{by_abstract[a]}".', a)
                      for a in abstracts], task)


def build_squad(task, tcfg, src, n, rng):
    # One question per paragraph, so no paragraph is shared between train and test.
    by_context = {}
    for r in _read_rows(src, tcfg.get("scan_rows")):
        context = _clean(r["context"])
        if len(context) <= tcfg["max_context_chars"] and context not in by_context:
            by_context[context] = r
    contexts = _sample(sorted(by_context), n, rng, task)
    out = []
    for i, context in enumerate(contexts):
        r = by_context[context]
        out.append({
            "id": f"{task}-{i:06d}",
            "prompt": (f"Read the text and answer the question with a short phrase taken from the text.\n\n"
                       f"Text: {context}\n\nQuestion: {_clean(r['question'])}"),
            "target": _clean(r["answers"]["text"][0]),
            "meta": {"squad_id": r["id"]},
        })
    return out


def build_arith(task, tcfg, src, n, rng):
    return arith.generate(n, seed=rng.randrange(2**31), prefix=task)


def build_wikitext(task, tcfg, src, n, rng):
    """Plain text without a request: the general language test is perplexity on raw text."""
    rows = _read_rows(src)
    paras = [_clean(r["text"]) for r in rows]
    paras = [p for p in paras if len(p) >= tcfg["min_chars"] and not p.startswith("=")]
    return _examples([("", p) for p in paras[:n]], task)


def build_instructions(task, tcfg, src, n, rng):
    return instructions.generate(n, seed=rng.randrange(2**31), prefix=task)


BUILDERS = {
    "tinystories": build_tinystories,
    "squad": build_squad,
    "arith": build_arith,
    "arxiv": build_arxiv,
    "wikitext": build_wikitext,
    "instructions": build_instructions,
}


def _split(examples, fractions, rng):
    examples = examples[:]
    rng.shuffle(examples)
    n = len(examples)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return {
        "train": examples[:n_train],
        "val": examples[n_train:n_train + n_val],
        "test": examples[n_train + n_val:],
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_task(task, data_cfg, eval_only=False):
    """Return ({'train','val','test'} -> examples, manifest) for one task.

    Eval-only tasks (general language test G, instruction following T0) have a test split only.
    """
    tcfg = data_cfg["eval_tasks" if eval_only else "tasks"][task]
    src = data_cfg["sources"].get(tcfg["source"], {})
    n = tcfg.get("n", data_cfg["pool_size"])
    # The loader code is part of the key, so editing a builder never reuses a stale cache.
    code = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))))
    key = json.dumps({"task": task, "tcfg": tcfg, "src": src, "n": n, "seed": data_cfg["seed"],
                      "split": data_cfg["split"], "eval_only": eval_only, "code": code.hexdigest()},
                     sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:10]
    cache = Path(data_cfg["cache_dir"]) / f"{task}_{tcfg['source']}_{digest}.json"

    if not cache.exists():
        # The task name is part of the seed, so tasks do not share random draws.
        rng = random.Random(f"{data_cfg['seed']}-{task}")
        examples = BUILDERS[tcfg["source"]](task, tcfg, src, n, rng)
        for ex in examples:
            # chat: the prompt is the user's message, wrapped in the model's chat template.
            ex["task"], ex["kind"], ex["chat"] = task, tcfg["kind"], tcfg.get("chat", False)
        splits = {"test": examples} if eval_only else _split(examples, data_cfg["split"], rng)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(splits, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache)

    splits = json.loads(cache.read_text(encoding="utf-8"))
    manifest = {"task": task, "source": tcfg["source"], "revision": src.get("revision"),
                "file": cache.name, "sha256": _sha256(cache),
                "sizes": {k: len(v) for k, v in splits.items()}}
    return splits, manifest


def load_all(data_cfg, order):
    tasks, manifests = {}, []
    for task in order:
        tasks[task], m = load_task(task, data_cfg)
        manifests.append(m)
    for task in data_cfg.get("eval_tasks") or {}:
        tasks[task], m = load_task(task, data_cfg, eval_only=True)
        manifests.append(m)
    return tasks, manifests


def general_prompts(data_cfg, scfg):
    """General user requests for self-generated replay: Dolly instructions without context, of the chosen
    categories and length, sampled with the data seed. Human-written, so they do not repeat the T0 test."""
    src = data_cfg["sources"][scfg["source"]]
    key = json.dumps({"src": src, "scfg": {k: scfg[k] for k in ("n", "categories", "max_chars")},
                      "seed": data_cfg["seed"]}, sort_keys=True)
    cache = Path(data_cfg["cache_dir"]) / f"R_prompts_{hashlib.sha256(key.encode()).hexdigest()[:10]}.json"
    if not cache.exists():
        rows = _read_rows(src)
        prompts = sorted({_clean(r["instruction"]) for r in rows
                          if not r["context"] and r["category"] in scfg["categories"]
                          and len(r["instruction"]) <= scfg["max_chars"]})
        prompts = _sample(prompts, scfg["n"], random.Random(f"{data_cfg['seed']}-R"), "self_replay")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
    return json.loads(cache.read_text(encoding="utf-8"))


# --- answer checking for extractive QA (task B), SQuAD-style normalisation ---

def _normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def verify_qa(example, generated):
    return _normalize_answer(generated) == _normalize_answer(example["target"])


VERIFIERS = {"arith": arith.verify, "squad": verify_qa, "instructions": instructions.verify}
