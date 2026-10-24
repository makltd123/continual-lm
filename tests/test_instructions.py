"""Task 0 checks: every check type accepts a good answer and rejects a bad one."""

from data import instructions
from data.instructions import check_one, verify

CASES = [
    ({"type": "max_words", "n": 5}, "Gravity pulls things down.", "Gravity is the force that pulls every object down."),
    ({"type": "min_words", "n": 5}, "One two three four five six.", "Too short."),
    ({"type": "exact_words", "n": 1}, "Cold.", "It is cold."),
    ({"type": "contains", "words": ["blue", "wave"]}, "A blue wave rolled in.", "A blue sea."),
    ({"type": "contains_any", "words": ["buenos dias"]}, "¡Buenos días!", "Good morning"),
    ({"type": "absent", "words": ["water"]}, "The sea is deep.", "The water is deep."),
    ({"type": "bullets", "n": 3}, "- apple\n- pear\n- plum", "Here they are:\n- apple\n- pear\n- plum"),
    ({"type": "all_upper"}, "THE CAT SAT.", "The cat sat."),
    ({"type": "all_lower"}, "the cat sat.", "The cat sat."),
    ({"type": "overlap", "text": "The cat sat quietly by the warm window."},
     "THE CAT SAT QUIETLY BY THE WARM WINDOW.", "A dog ran."),
    ({"type": "json_keys", "keys": ["name", "color"]}, '```json\n{"name": "Tom", "color": "grey"}\n```',
     '{"name": "Tom"}'),
    ({"type": "cyrillic"}, "Доброе утро!", "Good morning!"),
    ({"type": "max_sentences", "n": 2}, "One. Two.", "One. Two. Three."),
    ({"type": "ends_with", "text": "That is all."}, "Sleep restores the body. That is all.", "That is all. Sleep helps."),
    ({"type": "starts_with", "word": "Sure"}, "Sure, here it is.", "Here it is, sure."),
    ({"type": "quoted"}, '"Because light scatters."', "Because light scatters."),
    ({"type": "number_only"}, "8", "Eight legs."),
    ({"type": "paragraphs", "n": 2}, "First part.\n\nSecond part.", "Only one part."),
    ({"type": "no_commas"}, "Rain falls softly.", "Rain falls, softly."),
    ({"type": "title"}, "<<Autumn>>\nLeaves fall down.", "Autumn\nLeaves fall down."),
]


def test_each_check_accepts_good_and_rejects_bad():
    for check, good, bad in CASES:
        assert check_one(check, good), (check, good)
        assert not check_one(check, bad), (check, bad)


def test_generator_covers_all_types_deterministically():
    a, b = instructions.generate(seed=3), instructions.generate(seed=3)
    assert a == b and 100 <= len(a) <= 200
    used = {c["type"] for item in a for c in item["meta"]["checks"]}
    assert used == {check["type"] for check, _, _ in CASES}
    assert len(instructions.generate(n=12, seed=3)) == 12


def test_verify_needs_every_check():
    item = {"meta": {"checks": [{"type": "contains", "words": ["blue"]}, {"type": "max_sentences", "n": 1}]}}
    assert verify(item, "A blue sky.")
    assert not verify(item, "A blue sky. A grey sea.")
