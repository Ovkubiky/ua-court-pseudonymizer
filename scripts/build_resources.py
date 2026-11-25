"""Build the generator resources used by notebooks/synth_utils.py (synthetic data v2).

Outputs (all in notebooks/resources/):
- name_lexicon.csv  - first names, patronymics and surnames x 7 cases x gender, inflected automatically with
                      pymorphy3 from lang-uk/name_freq_dict_uk (GPL-3.0, https://github.com/lang-uk/name_freq_dict_uk);
- streets.csv       - street names with their street type, mined from clear-text (organisation / court) addresses in
                      the training corpus;
- settlements.csv   - villages / towns with district and oblast, mined from the training corpus.

Mining uses only documents that are NOT in our dev / test_a / test_b splits, and never touches the reference test.
Only street and settlement *names* are kept, never full addresses.

Run:  python scripts/build_resources.py
Needs data/external/name_freq_dict_uk (git clone https://github.com/lang-uk/name_freq_dict_uk data/external/name_freq_dict_uk)
and data/train/2024-court-decisions.parquet.
"""
from __future__ import annotations

import collections
import csv
import re
from pathlib import Path

import pandas as pd
import pymorphy3

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "notebooks" / "resources"
NAMES_DIR = ROOT / "data" / "external" / "name_freq_dict_uk" / "generated"
CASES = ["nomn", "gent", "datv", "accs", "ablt", "loct", "voct"]
CASE_UA = ["називний", "родовий", "давальний", "знахідний", "орудний", "місцевий", "кличний"]
M = pymorphy3.MorphAnalyzer(lang="uk")


def cap(word: str) -> str:
    """Capitalise every part of a hyphenated name, keep apostrophes: кам'янська -> Кам'янська."""
    return "-".join(p[:1].upper() + p[1:] for p in word.split("-"))


def _forms_from(parse, gender: str) -> list[str] | None:
    out = []
    for case in CASES:
        cands = [x.word for x in parse.lexeme
                 if case in x.tag and "plur" not in x.tag and (gender in x.tag or "Fixd" in x.tag)]
        if not cands:
            return None
        out.append(cands[0])
    return out


def inflect(lemma: str, kind: str, gender: str) -> list[str] | None:
    """7 case forms of `lemma` as a `kind` (Name / Patr / Surn) of `gender` (masc / femn), or None."""
    low = lemma.lower()
    parses = M.parse(low)
    exact = [p for p in parses if p.word == low and "nomn" in p.tag and "plur" not in p.tag]
    for p in exact:  # proper-name parse of the right gender first
        if kind in p.tag and gender in p.tag:
            f = _forms_from(p, gender)
            if f:
                return [cap(w) for w in f]
    if kind == "Surn":
        # adjectival surnames (-ський, -цький, -ий): the masculine lemma gives both genders
        for p in parses:
            if p.tag.POS == "ADJF" and p.normal_form == low:
                f = _forms_from(p, gender)
                if f:
                    return [cap(w) for w in f]
        if gender == "femn" and re.search(r"(енко|ко|о|ук|юк|ак|як|ик|ич|ець|ар|ій|ун|ин|ів|іш)$", low) and not low.endswith(("ський", "цький", "ий")):
            return [cap(low)] * 7  # consonant / -енко surnames do not change for women
    return None


def build_name_lexicon() -> None:
    rows = []
    sources = [("ім'я", "ч", "male_fname_freq_dict.csv", "Name", ["masc"]),
               ("ім'я", "ж", "female_fname_freq_dict.csv", "Name", ["femn"]),
               ("по батькові", "ч", "male_pname_freq_dict.csv", "Patr", ["masc"]),
               ("по батькові", "ж", "female_pname_freq_dict.csv", "Patr", ["femn"]),
               ("прізвище", None, "lname_freq_dict.csv", "Surn", ["masc", "femn"])]
    stats = collections.Counter()
    for kind_ua, gender_ua, fname, kind, genders in sources:
        for r in csv.DictReader(open(NAMES_DIR / fname, encoding="utf-8")):
            lemma, freq = r["name"].strip(), float(r["freq_in_corpus"])
            if not re.fullmatch(r"[А-ЯІЇЄҐ][а-яіїєґ'’ʼ-]+", lemma) or len(lemma) < 3:
                stats[f"{kind}:bad"] += 1
                continue
            forms = {g: inflect(lemma, kind, g) for g in genders}
            if any(f is None for f in forms.values()):
                stats[f"{kind}:dropped"] += 1
                continue
            stats[f"{kind}:ok"] += 1
            for g, f in forms.items():
                rows.append({"тип": kind_ua, "стать": gender_ua or ("ч" if g == "masc" else "ж"), "лема": lemma,
                             "частота": f"{freq:.6g}", **dict(zip(CASE_UA, f))})
    with open(RES / "name_lexicon.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["тип", "стать", "лема", "частота", *CASE_UA])
        w.writeheader()
        w.writerows(rows)
    print("name lexicon:", dict(stats), "->", len(rows), "rows")


STREET_RE = re.compile(
    r"(?<![А-ЯІЇЄҐа-яіїєґ])(вул\.|вулиця|вулиці|просп\.|проспект|пров\.|провулок|бульв\.|бульвар|пл\.|площа|майдан|узвіз|шосе|набережна)"
    r"\s*([А-ЯІЇЄҐ0-9][А-ЯІЇЄҐа-яіїєґ'’ʼ`\.\- ]{2,35}?)(?=\s*,\s*(?:буд\.|будинок|\d)|\s+буд\.|\s+\d{1,3}[А-ЯІЇЄҐа-я]?\b)")
STREET_KIND = {"вул.": "вулиця", "вулиця": "вулиця", "вулиці": "вулиця", "просп.": "проспект", "проспект": "проспект",
               "пров.": "провулок", "провулок": "провулок", "бульв.": "бульвар", "бульвар": "бульвар", "пл.": "площа",
               "площа": "площа", "майдан": "майдан", "узвіз": "узвіз", "шосе": "шосе", "набережна": "набережна"}
SETTLEMENT_RE = re.compile(
    r"(?<![А-ЯІЇЄҐа-яіїєґ])(с\.|смт|селище|село|м\.)\s?([А-ЯІЇЄҐ][а-яіїєґ'’ʼ`-]+(?: [А-ЯІЇЄҐ][а-яіїєґ'’ʼ`-]+)?)"
    r",?\s+([А-ЯІЇЄҐ][а-яіїєґ'’ʼ`-]+(?:ського|цького|зького)) (?:району|р-ну)(?:,?\s+([А-ЯІЇЄҐ][а-яіїєґ'’ʼ`-]+ської) (?:області|обл\.))?")
BAD_STREET = re.compile(r"\b(від|до|та|і|на|у|в|за|по|з|що|року|р\.)\b|\d{4}|[.]$")


def build_address_resources() -> None:
    splits = pd.read_csv(ROOT / "data" / "synthetic" / "splits.csv", dtype={"id": str})
    held_out = set(splits.loc[splits["split"] != "train", "id"])
    df = pd.read_parquet(ROOT / "data" / "train" / "2024-court-decisions.parquet", columns=["id", "text"])
    df = df[~df["id"].astype(str).isin(held_out)]
    streets, settlements = collections.Counter(), collections.Counter()
    for text in df["text"]:
        for m in STREET_RE.finditer(text):
            name = re.sub(r"\s+", " ", m.group(2)).strip(" .,-").replace("`", "'")
            if 3 <= len(name) <= 35 and not BAD_STREET.search(name) and not re.search(r"(ОСОБА|АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)", name):
                streets[(STREET_KIND[m.group(1)], name)] += 1
        for m in SETTLEMENT_RE.finditer(text):
            kind = {"с.": "с.", "село": "с.", "смт": "смт", "селище": "смт", "м.": "м."}[m.group(1)]
            settlements[(kind, m.group(2).replace("`", "'"), m.group(3).replace("`", "'"), (m.group(4) or "").replace("`", "'"))] += 1
    with open(RES / "streets.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["тип", "назва", "частота"])
        for (kind, name), n in streets.most_common():
            if n >= 2:
                w.writerow([kind, name, n])
    # one row per settlement name + district; keep the most frequent oblast spelling
    best = {}
    for (kind, name, district, oblast), n in settlements.most_common():
        key = (kind, name, district)
        if key not in best:
            best[key] = [oblast, n]
        else:
            best[key][1] += n
            best[key][0] = best[key][0] or oblast
    with open(RES / "settlements.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["тип", "назва", "район_род", "область_род", "частота"])
        for (kind, name, district), (oblast, n) in sorted(best.items(), key=lambda x: -x[1][1]):
            if n >= 2 and oblast:
                w.writerow([kind, name, district, oblast, n])
    print("streets:", sum(1 for v in streets.values() if v >= 2), "settlements:", sum(1 for v in best.values() if v[1] >= 2 and v[0]))


if __name__ == "__main__":
    build_name_lexicon()
    build_address_resources()
