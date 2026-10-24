"""Task C: synthetic arithmetic and simple logic with difficulty levels 1-5.

Every problem has a single short answer, so a script can check the model's
output (used as the feedback signal for the DPO step, FR-11).
"""

import random
import re

LEVELS = (1, 2, 3, 4, 5)


def _level1(rng):
    # Adding or subtracting a single digit: ~1700 distinct problems.
    a, b = rng.randint(10, 99), rng.randint(1, 9)
    if rng.random() < 0.5:
        return f"What is {a} + {b}?", a + b
    return f"What is {a} - {b}?", a - b


def _level2(rng):
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    if rng.random() < 0.5:
        return f"What is {a} + {b}?", a + b
    a, b = max(a, b), min(a, b)
    return f"What is {a} - {b}?", a - b


def _level3(rng):
    if rng.random() < 0.5:
        a, b = rng.randint(10, 99), rng.randint(2, 9)
        return f"What is {a} * {b}?", a * b
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    if rng.random() < 0.5:
        return f"What is {a} + {b}?", a + b
    a, b = max(a, b), min(a, b)
    return f"What is {a} - {b}?", a - b


def _level4(rng):
    kind = rng.randrange(3)
    if kind == 0:
        a, b, c = rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 9)
        return f"What is ({a} + {b}) * {c}?", (a + b) * c
    if kind == 1:
        a, b, c = rng.randint(2, 12), rng.randint(2, 12), rng.randint(1, 50)
        return f"What is {a} * {b} - {c}?", a * b - c
    a, b = rng.sample(range(10, 1000), 2)
    return f"Which number is larger, {a} or {b}?", max(a, b)


def _level5(rng):
    kind = rng.randrange(3)
    if kind == 0:
        x, m, k = rng.randint(2, 15), rng.randint(2, 6), rng.randint(1, 20)
        return f"Let x = {x}. Let y = x * {m} - {k}. What is y?", x * m - k
    if kind == 1:
        x, d, k = rng.randint(2, 30), rng.randint(2, 9), rng.randint(2, 5)
        return f"Let x = {x}. Let y = x + {d}. Let z = y * {k}. What is z?", (x + d) * k
    a, b, c = rng.randint(1, 50), rng.randint(1, 50), rng.randint(1, 50)
    truth = (a > b) and (b > c)
    return f"Is it true that {a} > {b} and {b} > {c}? Answer yes or no.", "yes" if truth else "no"


_MAKERS = {1: _level1, 2: _level2, 3: _level3, 4: _level4, 5: _level5}


def make_problem(rng, level):
    question, answer = _MAKERS[level](rng)
    return question, str(answer)


def generate(n, seed, prefix="C"):
    """n unique problems, levels spread evenly; deterministic for a given seed."""
    rng = random.Random(seed)
    seen, out, attempts = set(), [], 0
    while len(out) < n:
        attempts += 1
        if attempts > 50 * n:
            raise ValueError(f"cannot generate {n} distinct problems; lower pool_size")
        level = LEVELS[len(out) % len(LEVELS)]
        question, answer = make_problem(rng, level)
        if question in seen:
            continue
        seen.add(question)
        out.append({
            "id": f"{prefix}-{len(out):06d}",
            "prompt": f"{question}\nReply with the answer only.",  # the user's message in the chat
            "target": answer,
            "meta": {"level": level},
        })
    return out


def normalize(text):
    text = text.strip().lower()
    match = re.match(r"-?\d+|yes|no", text)
    return match.group(0) if match else text


def verify(example, generated):
    """True if the generated answer matches the reference answer."""
    return normalize(generated) == normalize(example["target"])
