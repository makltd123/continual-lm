"""Task 0: instruction following with automatically verifiable conditions.

In the spirit of IFEval (Zhou et al. 2023) but shorter, so it can run after every portion:
~150 instructions of 17 types (word limits, lists, JSON, translation, letter case, keywords,
forbidden words, start/end phrases, paragraphs, ...). An answer passes only if every check
of its instruction passes (prompt-level strict accuracy). Pure Python, no extra dependencies.
"""

import json
import random
import re
import unicodedata

CONCEPTS = [("photosynthesis", "photosynthesis"), ("gravity", "gravity"), ("a rainbow", "rainbow"),
            ("democracy", "democracy"), ("the internet", "internet"), ("an earthquake", "earthquake"),
            ("a vaccine", "vaccine"), ("inflation", "inflation"), ("a black hole", "black hole"),
            ("evaporation", "evaporation"), ("electricity", "electricity"), ("a volcano", "volcano")]
CATEGORIES = ["fruits", "colors", "animals that live in the sea", "musical instruments", "countries in Europe",
              "kitchen tools", "sports", "planets", "vegetables", "programming languages", "birds", "flowers"]
TOPICS = ["the ocean", "autumn", "a busy city", "a small village", "the moon", "coffee", "trains",
          "a library", "mountains", "friendship", "rain", "a garden"]
AVOID = [("the ocean", "water"), ("autumn", "leaves"), ("the moon", "night"), ("coffee", "drink"),
         ("trains", "travel"), ("a library", "books"), ("rain", "wet"), ("a garden", "flowers")]
KEYWORD_PAIRS = [("the ocean", "blue", "wave"), ("autumn", "golden", "wind"), ("a busy city", "noise", "lights"),
                 ("the moon", "silver", "quiet"), ("coffee", "morning", "warm"), ("trains", "station", "fast"),
                 ("a library", "silence", "story"), ("mountains", "snow", "climb"), ("friendship", "trust", "laugh"),
                 ("a garden", "seeds", "sun")]
SENTENCES = ["The cat sat quietly by the warm window.", "We will meet at the old bridge tomorrow morning.",
             "My brother bought a red bicycle last summer.", "The library closes early on Sundays.",
             "Fresh bread smells wonderful in the morning.", "She painted the fence green and white.",
             "Our team won the final game in the rain.", "The museum has a new exhibition about whales.",
             "He forgot his umbrella at the station.", "Children love building castles in the sand."]
ONE_WORD = ["What is the opposite of hot?", "What color is fresh grass?", "Which animal says moo?",
            "What is the capital of France?", "What do bees make?", "What is frozen water called?",
            "Which planet do we live on?", "What is the opposite of up?", "What season comes after winter?",
            "What do you call a baby dog?"]
NUMBER = ["How many legs does a spider have?", "How many days are in a week?", "What is 12 times 3?",
          "How many minutes are in an hour?", "How many sides does a triangle have?", "What is 100 minus 37?",
          "How many months are in a year?", "What is 9 plus 16?", "How many hours are in a day?",
          "What is half of 50?"]
QUESTIONS = ["Why do we need sleep?", "What makes a good friend?", "Why is the sky blue?",
             "How do plants get water?", "Why should people exercise?", "What is a good way to learn a language?",
             "Why do leaves change color in autumn?", "How can we save energy at home?"]
JSON_THINGS = [("a cat", "color"), ("the planet Mars", "size"), ("a bicycle", "wheels"), ("an apple", "taste"),
               ("the city of Paris", "country"), ("a piano", "keys"), ("a penguin", "habitat"),
               ("a laptop", "price"), ("the Moon", "distance"), ("a tulip", "season")]
TRANSLATIONS = {
    "French": [("Good morning", ["bonjour"]), ("Thank you very much", ["merci"]), ("Good night", ["bonne nuit"]),
               ("The water is cold", ["eau"])],
    "Spanish": [("Thank you", ["gracias"]), ("Good morning", ["buenos dias"]), ("The house is big", ["casa"]),
                ("I like cats", ["gato"])],
    "German": [("Thank you", ["danke"]), ("Good morning", ["guten morgen"]), ("The dog is small", ["hund"]),
               ("I like apples", ["apfel", "äpfel"])],
    "Russian": [("Good morning", None), ("Thank you very much", None), ("The weather is nice today", None),
                ("I am reading a book", None)],
}
END_PHRASES = ["Hope this helps.", "That is all.", "Thank you for asking."]
START_WORDS = ["Yes", "Sure", "Honestly", "Well"]


def generate(n=None, seed=0, prefix="T0"):
    rng = random.Random(seed)
    items = []

    def add(kind, prompt, checks):
        items.append({"prompt": prompt, "target": "", "meta": {"type": kind, "checks": checks}})

    for concept, key in CONCEPTS:
        limit = rng.choice([20, 25, 30, 40])
        add("max_words", f"Explain what {concept} is in no more than {limit} words.",
            [{"type": "max_words", "n": limit}, {"type": "contains", "words": [key]}])
    for cat in CATEGORIES:
        k = rng.choice([3, 4, 5])
        add("bullets", f'List exactly {k} {cat}. Put each one on its own line starting with "- " '
                       f"and write nothing else.", [{"type": "bullets", "n": k}])
    for q in ONE_WORD:
        add("one_word", f"Answer with exactly one word. {q}", [{"type": "exact_words", "n": 1}])
    for s in SENTENCES:
        add("upper", f'Rewrite this sentence using only capital letters: "{s}"',
            [{"type": "all_upper"}, {"type": "overlap", "text": s}])
    for s in SENTENCES[:8]:
        add("lower", f'Rewrite this sentence using only lowercase letters: "{s.upper()}"',
            [{"type": "all_lower"}, {"type": "overlap", "text": s}])
    for thing, key in JSON_THINGS:
        add("json", f'Describe {thing} as a JSON object with the keys "name" and "{key}". Output only the JSON.',
            [{"type": "json_keys", "keys": ["name", key]}])
    for lang, pairs in TRANSLATIONS.items():
        for phrase, expected in pairs:
            check = {"type": "cyrillic"} if expected is None else {"type": "contains_any", "words": expected}
            add("translate", f'Translate into {lang}: "{phrase}". Reply with the translation only.', [check])
    for topic, w1, w2 in KEYWORD_PAIRS:
        add("keywords", f'Write one sentence about {topic} that includes the words "{w1}" and "{w2}".',
            [{"type": "contains", "words": [w1, w2]}, {"type": "max_sentences", "n": 2}])
    for topic, word in AVOID:
        add("avoid", f'Describe {topic} in two or three sentences without using the word "{word}".',
            [{"type": "absent", "words": [word]}, {"type": "max_sentences", "n": 3}])
    for q in QUESTIONS:
        phrase = rng.choice(END_PHRASES)
        add("ends_with", f'{q} End your answer with the exact phrase "{phrase}"',
            [{"type": "ends_with", "text": phrase}])
    for q in QUESTIONS:
        word = rng.choice(START_WORDS)
        add("starts_with", f'Start your answer with the word "{word}". {q}', [{"type": "starts_with", "word": word}])
    for q in QUESTIONS[:6]:
        add("quoted", f"Wrap your whole answer in double quotation marks. {q}", [{"type": "quoted"}])
    for q in NUMBER:
        add("number_only", f"Reply with a number only, without any words. {q}", [{"type": "number_only"}])
    for topic in TOPICS[:6]:
        k = rng.choice([2, 3])
        add("paragraphs", f"Write exactly {k} short paragraphs about {topic}. Separate the paragraphs "
                          f"with a blank line.", [{"type": "paragraphs", "n": k}])
    for topic in TOPICS[6:12]:
        add("no_commas", f"Describe {topic} in a few sentences without using any commas.", [{"type": "no_commas"}])
    for topic in TOPICS[:6]:
        add("title", f"Write a four-line poem about {topic}. Give it a title wrapped in double angular "
                     f"brackets, like <<my title>>.", [{"type": "title"}])
    for topic in TOPICS[6:12]:
        k = rng.choice([40, 50, 60])
        add("min_words", f"Write at least {k} words about {topic}.", [{"type": "min_words", "n": k}])

    rng.shuffle(items)
    items = items[:n] if n else items
    for i, item in enumerate(items):
        item["id"] = f"{prefix}-{i:03d}"
    return items


# --- checks -------------------------------------------------------------------

def _plain(text):
    """Lowercase without accents, so 'buenos días' matches 'buenos dias'."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _words(text):
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", text)


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if re.search(r"\w", s)]


def _letters(text):
    return [ch for ch in text if ch.isalpha()]


def _json_object(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    match = re.search(r"\{.*\}", text, flags=re.S)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


def _strip_quotes(text):
    return text.strip().strip('"\'“”«»').strip()


def check_one(check, response):
    r = response.strip()
    t = check["type"]
    if t == "max_words":
        return 0 < len(_words(r)) <= check["n"]
    if t == "min_words":
        return len(_words(r)) >= check["n"]
    if t == "exact_words":
        return len(_words(r)) == check["n"]
    if t == "contains":
        return all(_plain(w) in _plain(r) for w in check["words"])
    if t == "contains_any":
        return any(_plain(w) in _plain(r) for w in check["words"])
    if t == "absent":
        return not any(re.search(rf"\b{re.escape(w)}\b", r, flags=re.I) for w in check["words"])
    if t == "bullets":
        lines = [line for line in r.splitlines() if line.strip()]
        bullets = [line for line in lines if re.match(r"\s*[-*•]\s+\S", line)]
        return len(bullets) == check["n"] == len(lines)
    if t == "all_upper":
        letters = _letters(r)
        return bool(letters) and all(not ch.islower() for ch in letters)
    if t == "all_lower":
        letters = _letters(r)
        return bool(letters) and all(not ch.isupper() for ch in letters)
    if t == "overlap":
        want = {w.lower() for w in _words(check["text"])}
        got = {w.lower() for w in _words(r)}
        return len(want & got) >= 0.6 * len(want)
    if t == "json_keys":
        obj = _json_object(r)
        return isinstance(obj, dict) and all(k in obj for k in check["keys"])
    if t == "cyrillic":
        letters = _letters(r)
        return bool(letters) and sum("Ѐ" <= ch <= "ӿ" for ch in letters) >= 0.6 * len(letters)
    if t == "max_sentences":
        return 0 < len(_sentences(r)) <= check["n"]
    if t == "ends_with":
        return _strip_quotes(r).lower().endswith(_strip_quotes(check["text"]).lower())
    if t == "starts_with":
        return _plain(_strip_quotes(r)).startswith(_plain(check["word"]))
    if t == "quoted":
        return len(r) > 2 and r[0] in '"“' and r[-1] in '"”'
    if t == "number_only":
        return re.fullmatch(r"-?\d+(?:[.,]\d+)?", r.rstrip(".")) is not None
    if t == "paragraphs":
        return len([p for p in re.split(r"\n\s*\n", r) if p.strip()]) == check["n"]
    if t == "no_commas":
        return bool(r) and "," not in r
    if t == "title":
        return re.search(r"<<[^<>\n]+>>", r) is not None
    raise ValueError(f"unknown check type {t!r}")


def verify(example, response):
    """True if the response satisfies every condition of the instruction."""
    return all(check_one(c, response) for c in example["meta"]["checks"])
