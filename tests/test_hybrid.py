import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hybrid  # noqa: E402


def cats(text, spans):
    return [(text[s:e], c) for s, e, c in spans]


def test_snap_grows_to_words_and_strips_punctuation():
    text = "позивач Петренко Олег Іванович, звернувся"
    s = text.index("тренко")
    out = hybrid.snap(text, [(s, text.index(",") + 1, "ОСОБА")])
    assert cats(text, out) == [("Петренко Олег Іванович", "ОСОБА")]


def test_snap_keeps_dot_of_initials_and_joins_pieces():
    text = "суддя Петренко О.О. та адреса м. Київ, вул. Шевченка, 5."
    a = text.index("Петренко")
    addr1 = text.index("м. Київ")
    addr2 = text.index("вул.")
    out = hybrid.snap(text, [(a, a + 8, "ОСОБА"), (a + 9, a + 13, "ОСОБА"),
                             (addr1, addr1 + 7, "АДРЕСА"), (addr2, len(text), "АДРЕСА")])
    assert cats(text, out) == [("Петренко О.О.", "ОСОБА"), ("м. Київ, вул. Шевченка, 5", "АДРЕСА")]


def test_rules_find_structured_pii_and_skip_public_accounts():
    text = ("ОСОБА: РНОКПП 3012345678, тел. +380671234567, e-mail ivan.petrenko@gmail.com, "
            "ДНЗ АА1234ВВ. Кошти на рахунок UA908999980313111256000026001 ГУК у м. Києві.")
    found = {(text[s:e], c) for s, e, c, _ in hybrid.rule_spans(text)}
    assert ("3012345678", "НОМЕР") in found
    assert ("+380671234567", "НОМЕР") in found
    assert ("ivan.petrenko@gmail.com", "ІНФОРМАЦІЯ") in found
    assert ("АА1234ВВ", "НОМЕР") in found
    assert not any(t.startswith("UA90") for t, _ in found)


def test_birth_date_after_person_and_rule_overrides_model():
    text = "Романишин Мар'яна, 27.10.1989 р. нар., пише"
    person = (0, text.index(","), "ОСОБА")
    rules = hybrid.rule_spans(text, [person])
    assert ("27.10.1989 р.", "ІНФОРМАЦІЯ") in {(text[s:e], c) for s, e, c, _ in rules}
    d = text.index("27")
    merged = hybrid.merge_with_rules([person, (d, d + 5, "НОМЕР")], rules)
    assert cats(text, merged) == [("Романишин Мар'яна", "ОСОБА"), ("27.10.1989 р.", "ІНФОРМАЦІЯ")]
