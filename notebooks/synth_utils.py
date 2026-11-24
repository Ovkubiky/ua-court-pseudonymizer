"""Helpers for generating labelled training data from pseudonymized court decisions (used by generate_data.ipynb).

Two steps per document, both code only (no LLM):
1. annotate_rules: for every ОСОБА mention -> grammatical case (pymorphy3 + context rules); for every person -> gender.
2. assemble_doc: replaces placeholders with fictional values (names inflected with resources/name_declensions.csv),
   marks clear-text names (judges, clerks, lawyers) as ОСОБА, and returns a record in the test format.
"""
from __future__ import annotations

import csv
import random
import re
from pathlib import Path

TYPES = ["ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ"]
PH = re.compile(r"(ОСОБА|АДРЕСА|НОМЕР|ІНФОРМАЦІЯ)_(\d+)")
CASES = ["називний", "родовий", "давальний", "знахідний", "орудний", "місцевий", "кличний"]

# ---------------------------------------------------------------- proceeding type × instance (same as data_stats)
PROV = re.compile(r"[Пп]ровадження\s*№\s*:?\s*([^\s,;|]+)")
KW = {"criminal": r"\bКПК\b|Кримінальн\w+ процесуальн", "admin offence": r"КУпАП|Кодекс\w* України про адміністративні правопорушення",
      "administrative": r"\bКАС\b|Кодекс\w* адміністративного судочинства", "civil": r"\bЦПК\b|Цивільн\w+ процесуальн",
      "commercial": r"\bГПК\b|Господарськ\w+ процесуальн"}
NUM_RULES = [
    (r"^61-\d+с[кцвз]|^1[1-4]-\d+цс", ("civil", "cassation")), (r"^51-\d+", ("criminal", "cassation")),
    (r"^К/\d+|^1[1-4]-\d+[аk]с|^11-\d+апп", ("administrative", "cassation")), (r"^1[1-4]-\d+гс", ("commercial", "cassation")),
    (r"^11-?(кп|сс|п|в|кс)", ("criminal", "appeal")), (r"^22-?ц|^22-з", ("civil", "appeal")),
    (r"^22-?а", ("administrative", "appeal")), (r"^33/|^23-з", ("admin offence", "appeal")),
    (r"^1-?(кп|кс|в|во|кд|р|м|о|і|вп)\b|^1-?КП", ("criminal", "first")), (r"^2-?а|^П/", ("administrative", "first")),
    (r"^3/|^3-(в|зв)", ("admin offence", "first")), (r"^2/|^2-?[а-яіїє]+/|^4-с|^6/", ("civil", "first")),
]
COURT = [("cassation", r"Верховн\w+ Суд|Касаційн\w+ \w+ суд"), ("appeal", r"[Аа]пеляційн\w+ суд"),
         ("first", r"районн\w+ суд|міськрайонн\w+ суд|міськ\w+ суд|окружн\w+ \w*\s?суд|[Гг]осподарськ\w+ суд\w* (міста|\w+ області)")]


def classify(text: str) -> tuple[str, str]:
    m = PROV.search(text)
    if m:
        for pat, res in NUM_RULES:
            if re.search(pat, m.group(1)):
                return res
    c = {k: len(re.findall(p, text)) for k, p in KW.items()}
    ptype = max(c, key=lambda k: c[k]) if max(c.values()) else "unknown"
    if re.search(r"касаційн\w+ скарг", text) and re.search(r"Верховн\w+ Суд", text):
        return ptype, "cassation"
    if re.search(r"(розглянув\w*|переглянув\w*)[^.]{0,150}апеляційн\w+ скарг|апеляційн\w+ скарг\w*[^.]{0,150}(задовольн|залиш\w+ без задоволення)", text):
        return ptype, "appeal"
    hits = [(mm.start(), lvl) for lvl, p in COURT if (mm := re.search(p, text[:1500]))]
    return ptype, (min(hits)[1] if hits else "unknown")


# ---------------------------------------------------------------- subtypes by left context (same as data_stats §8)
SUB_RULES = {
    "ІНФОРМАЦІЯ": [("death_date", r"(помер\w*|смерт\w*|загину\w*|загибел\w*)[^.]{0,40}$"),
                   ("birth_date", r"ОСОБА_\d+\s*[,(]\s*$|народж|народи\w+|р\.\s?н\."),
                   ("email", r"[eе]-?mail|пошт"), ("url", r"http|www|сайт|посиланн"),
                   ("nickname", r"\bнік\b|нікнейм|акаунт|логін\w*|псевдонім\w*|телеграм|telegram|instagram|facebook|viber"),
                   ("unit_or_institution", r"(начальник|командир|військов|частин|в/ч|ТЦК|центр|управлін|відділ|госпітал|КНП|комісі)\w*\W{0,4}$"),
                   # v2: organisation names in quotes after a legal form ("ПрАТ « __ »") and titles ("під назвою « __ »")
                   ("company", r"(?:\b(?:ТОВ|ПрАТ|ПАТ|АТ|КБ|АБ|ПП|ФГ|ДП|КП|СТОВ|ТзОВ)|магазин\w*|компані\w*|підприємств\w*|кафе|ресторан\w*)\s*[«\"“„]\s*$"),
                   ("title", r"назв\w*:?\s*[«\"“„]\s*$")],
    "НОМЕР": [("ipn", r"ІПН|РНОКПП|ідентифікаційн|облікової картки"), ("passport", r"паспорт"), ("phone", r"тел\.|телефон|моб"),
              ("iban", r"IBAN|рахун"), ("plate", r"номерн\w* знак|д\.?н\.?з|реєстраційн\w* номер|автомобіл|транспортн"),
              ("military_unit", r"в/ч|військов\w* частин|ВЧ"), ("certificate", r"посвідчен|свідоцтв|сері")],
}
# labels the LLM chooses from (Ukrainian) -> internal codes
NUM_LABELS = {"ІПН": "ipn", "паспорт": "passport", "ID-картка": "id_card", "телефон": "phone", "IBAN": "iban",
              "номерний знак": "plate", "номер військової частини": "military_unit", "посвідчення": "certificate", "інше": "other"}
INFO_LABELS = {"дата народження": "birth_date", "дата смерті": "death_date", "email": "email", "URL": "url", "нік": "nickname",
               "назва установи або військової частини": "unit_or_institution", "інше": "other"}


def subtype_by_context(text: str, start: int, typ: str) -> str:
    """The keyword closest to the placeholder wins (e.g. "РНОКПП …, тел. НОМЕР_2" -> phone)."""
    ctx = text[max(0, start - 60):start]
    best, best_end = "other", -1
    for name, pat in SUB_RULES.get(typ, []):
        for mm in re.finditer(pat, ctx, re.I):
            if mm.end() > best_end:
                best, best_end = name, mm.end()
    return best


def doc_subtypes(text: str) -> set[str]:
    return {f"{m.group(1)}:{subtype_by_context(text, m.start(), m.group(1))}"
            for m in re.finditer(r"(ІНФОРМАЦІЯ|НОМЕР)_\d+", text)}


# ================================================================ STEP 1: rule-based annotation
# Evaluated on a gold set of 359 real clear-text names from train (generate_data.ipynb, appendix).
_MORPH = None


def morph():
    global _MORPH
    if _MORPH is None:
        import pymorphy3
        _MORPH = pymorphy3.MorphAnalyzer(lang="uk")
    return _MORPH


CASE_CODE = {"nomn": "називний", "gent": "родовий", "datv": "давальний", "accs": "знахідний", "ablt": "орудний", "loct": "місцевий", "voct": "кличний"}
CASE_PREF = ["родовий", "називний", "знахідний", "давальний", "орудний", "місцевий", "кличний"]  # tie-break for ambiguous role words
PREP = {"родовий": "до від для без біля щодо відносно стосовно у в коло проти після замість крім окрім серед посеред уздовж навпроти".split(),
        "орудний": "з із зі між перед над під".split(), "знахідний": "про на через".split(), "місцевий": "при".split()}
GEN_NOUNS = set("участю користь особі позовом позов позову позовної заявою заяву заяви заява скаргою скаргу скарги скарга "
                "клопотанням клопотання поданням подання ім'я імені".split())
_TOK = re.compile(r"[А-ЯІЇЄҐа-яіїєґ'’ʼ-]+|[^\sА-ЯІЇЄҐа-яіїєґ'’ʼ-]")
_WORD = re.compile(r"[А-ЯІЇЄҐа-яіїєґ'’ʼ-]+")


def _sides(text: str, s: int, e: int, width: int = 200) -> tuple[str, str]:
    left, right = text[max(0, s - width):s], text[e:e + width]
    cut = max(left.rfind(". "), left.rfind("\n"))
    left = left[cut + 1:] if cut >= 0 else left
    ends = [i for i in (right.find(". "), right.find("\n")) if i >= 0]
    return left, (right[:min(ends)] if ends else right)


def _gender(tag) -> str | None:
    return "ч" if "masc" in tag else "ж" if "femn" in tag else None


def rule_case(left: str, right: str) -> tuple[set, bool, str | None]:
    """Case(s) of a name gap from its context. Walks back over modifiers ("нотаріус Київського міського нотаріального
    округу ___") to the nearest animate role noun and copies its case; prepositions, verbs, possessive nouns otherwise.
    Returns (cases, confident, gender hint from the role noun)."""
    m = morph()
    lw, rw = _TOK.findall(left), _WORD.findall(right)[:1]
    nxt = m.parse(rw[0].lower())[0] if rw else None
    adjacent = True
    for j in range(len(lw) - 1, max(-1, len(lw) - 14), -1):
        w = lw[j]
        if w in (",", ";", ":", "(", ")", "–", "-", "—"):
            break
        if not re.match(r"[а-яіїєґ]", w):  # quotes, placeholders, abbreviations, org / proper names: skip
            # ...but a normal word capitalised at the start of a sentence ("Обвинуваченого ___") is kept
            if not (re.fullmatch(r"[А-ЯІЇЄҐ][а-яіїєґ'’ʼ-]+", w) and j == 0
                    and all(t not in str(q.tag) for q in m.parse(w.lower())[:1] for t in ("Name", "Surn", "Patr", "Geox", "Orgn"))
                    and m.parse(w.lower())[0].is_known):
                continue
            w = w.lower()
        if adjacent:
            if j >= 1 and lw[j - 1] in ("з", "із") and w == "боку":
                return {"родовий"}, True, None
            for case, ws in PREP.items():
                if w in ws:
                    if case == "орудний" and re.search(r"стягн", left[-40:]):
                        return {"родовий"}, True, None
                    return {case}, True, None
            if w in GEN_NOUNS:
                return {"родовий"}, True, None
        ps = [p for p in m.parse(w) if p.tag.case in CASE_CODE]
        anim = [p for p in ps if p.tag.POS == "NOUN" and "anim" in p.tag]
        if anim:  # incl. plural ("у складі суддів ___" -> genitive) and words that also parse as adjectives ("суддів")
            sing = [p for p in anim if "plur" not in p.tag] or anim
            return {CASE_CODE[p.tag.case] for p in sing}, True, _gender(sing[0].tag)
        if adjacent and ps and ps[0].tag.POS in ("ADJF", "PRTF") and not (nxt and nxt.tag.POS == "NOUN"):
            return {CASE_CODE[p.tag.case] for p in ps}, True, _gender(ps[0].tag)  # substantivised: керуючого, засудженої
        first = m.parse(w)[0]
        if adjacent and first.tag.POS in ("VERB", "INFN"):
            return {"знахідний"}, True, None
        if first.tag.POS in ("PREP", "VERB", "INFN"):
            break
        adjacent = False
    if lw and re.match(r"[а-яіїєґ]", lw[-1]) and m.parse(lw[-1])[0].tag.POS == "NOUN":
        return {"родовий"}, True, None  # inanimate noun right before: possessive genitive (позов ___)
    if nxt is not None and nxt.tag.POS == "INFN":
        return {"знахідний"}, True, None
    if nxt is not None and nxt.tag.POS == "VERB":
        return {"називний"}, True, None
    return {"називний"}, False, None


def gender_from_right(right: str) -> str | None:
    """'___ звернулася', '___, яка …', 'визнати ___ винною' -> gender."""
    m = morph()
    for w in _WORD.findall(right.lower())[:4]:
        if w in ("який", "якого", "якому", "яким", "якому"):
            return "ч"
        if w in ("яка", "якої", "якій", "яку", "якою"):
            return "ж"
        p = m.parse(w)[0]
        if (p.tag.POS == "VERB" and "past" in p.tag) or (p.tag.POS in ("ADJF", "ADJS", "PRTF") and p.tag.case in ("ablt", "nomn")):
            g = _gender(p.tag)
            if g:
                return g
    return None


def annotate_rules(text: str) -> dict:
    """Gender per person (where the context tells it), case per name mention keyed by its position, confidence flags."""
    ann = {"gender": {}, "case": {}, "subtype": {}, "confident": {}, "source": "rules"}
    votes: dict[str, dict[str, int]] = {}
    prev_case, prev_conf = None, False
    for mm in PH.finditer(text):
        if mm.group(1) != "ОСОБА":
            continue
        left, right = _sides(text, mm.start(), mm.end())
        cases, conf, g = rule_case(left, right)
        case = next(c for c in CASE_PREF if c in cases)
        # enumerations "ОСОБА_1 , ОСОБА_2 та ___": same case as the previous name in the list
        if prev_case and re.search(r"ОСОБА_\d+\s*(,|та|і|й|або|чи)\s*$", left):
            case, conf = prev_case, prev_conf
        ann["case"][str(mm.start())] = case
        ann["confident"][str(mm.start())] = conf
        prev_case, prev_conf = case, conf
        v = votes.setdefault(mm.group(0), {})
        if g:  # feminine role nouns (позивачка) are informative; masculine ones (нотаріус, суддя) are used for women too
            v[g] = v.get(g, 0) + (2 if g == "ж" else 0.5)
        g2 = gender_from_right(right)
        if g2:
            v[g2] = v.get(g2, 0) + 1.5
    ann["gender"] = {lab: max(v, key=lambda k: v[k]) for lab, v in votes.items() if v}
    return ann


# ================================================================ STEP 2: assembling documents
def load_declensions(path: Path) -> dict:
    """Reads the reviewed table into lists of names / patronymic pairs / surname pairs with all 7 case forms."""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    forms = lambda r: [r[c].strip() for c in CASES]
    tab = {"name": {"ч": [], "ж": []}, "patronymic": {}, "surname": {}}
    kind = {"ім'я": "name", "по батькові": "patronymic", "прізвище": "surname"}
    for r in rows:
        k, g = kind[r["тип"]], r["стать"]
        if k == "name":
            tab["name"][g].append(forms(r))
        else:
            tab[k].setdefault(r["пара"], {})[g] = forms(r)
    for k in ("patronymic", "surname"):
        tab[k] = [v for _, v in sorted(tab[k].items(), key=lambda x: int(x[0])) if "ч" in v and "ж" in v]
    return tab


def _half(lst, pool):
    """train and eval use different halves of the lists, so our test sets do not "recognise" names seen in train."""
    return lst[0::2] if pool == "train" else lst[1::2]


def _pool_of(lemma: str) -> str:
    """Stable train / eval assignment of a lexicon entry (v2): by a hash of its lemma, not by list position."""
    import hashlib
    return "train" if int(hashlib.md5(lemma.lower().encode()).hexdigest(), 16) % 2 == 0 else "eval"


def load_name_tables(res_dir: Path) -> dict:
    """v2 name tables: the reviewed table (name_declensions.csv) + the automatic lexicon (name_lexicon.csv, built by
    scripts/build_resources.py from lang-uk/name_freq_dict_uk). Every entry carries a weight (sqrt of corpus frequency,
    so frequent names dominate but the long tail is still sampled) and a pool ("train" / "eval")."""
    base = load_declensions(res_dir / "name_declensions.csv")
    tab = {"version": 2, "name": {"ч": [], "ж": []}, "patronymic": {"ч": [], "ж": []}, "surname": []}
    seen = set()
    for g in ("ч", "ж"):
        for forms in base["name"][g]:
            tab["name"][g].append((forms, 1e-3, _pool_of(forms[0]))); seen.add(("name", g, forms[0]))
    for pair in base["patronymic"]:
        for g in ("ч", "ж"):
            tab["patronymic"][g].append((pair[g], 1e-3, "both")); seen.add(("patronymic", g, pair[g][0]))
    for pair in base["surname"]:
        tab["surname"].append((pair, 1e-3, _pool_of(pair["ч"][0]))); seen.add(("surname", pair["ч"][0]))
    lex = res_dir / "name_lexicon.csv"
    if lex.exists():
        surn: dict[str, dict] = {}
        for r in csv.DictReader(open(lex, encoding="utf-8")):
            forms, w = [r[c].strip() for c in CASES], float(r["частота"]) ** .5
            if r["тип"] == "ім'я" and ("name", r["стать"], forms[0]) not in seen:
                tab["name"][r["стать"]].append((forms, w, _pool_of(r["лема"])))
            elif r["тип"] == "по батькові" and ("patronymic", r["стать"], forms[0]) not in seen:
                tab["patronymic"][r["стать"]].append((forms, w, "both"))
            elif r["тип"] == "прізвище":
                surn.setdefault(r["лема"], {"w": w})[r["стать"]] = forms
        for lemma, d in surn.items():
            if "ч" in d and "ж" in d and ("surname", d["ч"][0]) not in seen:
                tab["surname"].append(({"ч": d["ч"], "ж": d["ж"]}, d["w"], _pool_of(lemma)))
    # sampling lists per pool: (items, weights). Weights are capped at the 90th percentile so that the head of the
    # frequency list (politicians: Тимошенко, Порошенко, ...) does not dominate the synthetic persons.
    def lists(entries):
        ws = sorted(w for _, w, _ in entries)
        cap = ws[int(.9 * (len(ws) - 1))] if ws else 1
        out = {}
        for pool in ("train", "eval"):
            sel = [(f, min(w, cap)) for f, w, p in entries if p in (pool, "both")]
            out[pool] = ([f for f, _ in sel], [w for _, w in sel])
        return out
    tab["_pick"] = {"surname": lists(tab["surname"]),
                    **{f"name_{g}": lists(tab["name"][g]) for g in ("ч", "ж")},
                    **{f"patronymic_{g}": lists(tab["patronymic"][g]) for g in ("ч", "ж")}}
    return tab


def _pick(tab, key, pool, rng):
    items, weights = tab["_pick"][key][pool]
    return rng.choices(items, weights=weights)[0]


def make_person(tab: dict, rng: random.Random, pool: str = "train") -> dict:
    if tab.get("version") == 2:
        s = _pick(tab, "surname", pool, rng)
        return {g: {"surname": s[g], "name": _pick(tab, f"name_{g}", pool, rng),
                    "patronymic": _pick(tab, f"patronymic_{g}", pool, rng)} for g in ("ч", "ж")}
    s, p = rng.choice(_half(tab["surname"], pool)), rng.choice(tab["patronymic"])
    return {g: {"surname": s[g], "name": rng.choice(_half(tab["name"][g], pool)), "patronymic": p[g]} for g in ("ч", "ж")}


def render_name(p: dict, case: str, rng: random.Random) -> str:
    i = CASES.index(case)
    s, n, pt = p["surname"][i], p["name"][i], p["patronymic"][i]
    a, b = p["name"][0][0], p["patronymic"][0][0]
    initials = rng.choice([f"{a}. {b}.", f"{a}.{b}."])
    # v2 adds "Прізвище Ім'я" and "Ім'я Прізвище" (seen in court texts for persons without a known patronymic)
    return rng.choices([f"{s} {n} {pt}", f"{s} {initials}", f"{initials} {s}", f"{s} {n}", f"{n} {s}"],
                       weights=[6, 3, 1, .4, .2])[0]


L_PLATE = "АВЕІКМНОРСТХ"
UA_MONTHS = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня", "листопада", "грудня"]
# Regions: (oblast, oblast-adjective stem and city stems found in court texts, [(settlement, postal code), ...]).
# The first settlement is the regional centre. Addresses in a document are generated mostly in the region the case is from.
REGIONS = [
    (None, r"Київськ\w+ міськ|м\. ?Києв|м\. ?Київ|Києва\b", [("м. Київ", "01001"), ("м. Київ", "02000"), ("м. Київ", "03000"), ("м. Київ", "04000")]),
    ("Київська область", r"Київськ\w+ област", [("м. Біла Церква", "09100"), ("м. Бориспіль", "08300"), ("м. Бровари", "07400"), ("смт Бородянка", "07801")]),
    ("Вінницька область", r"Вінницьк|Вінниц", [("м. Вінниця", "21000"), ("м. Жмеринка", "23100"), ("с. Агрономічне", "23227")]),
    ("Волинська область", r"Волинськ|Луцьк", [("м. Луцьк", "43000"), ("м. Ковель", "45000"), ("с. Липини", "45285")]),
    ("Дніпропетровська область", r"Дніпропетровськ|Дніпр[оа]\b|Дніпровськ", [("м. Дніпро", "49000"), ("м. Кривий Ріг", "50000"), ("м. Кам'янське", "51900")]),
    ("Донецька область", r"Донецьк|Краматорськ|Слов.янськ", [("м. Краматорськ", "84300"), ("м. Слов'янськ", "84100"), ("м. Дружківка", "84200")]),
    ("Житомирська область", r"Житомирськ|Житомир", [("м. Житомир", "10000"), ("м. Бердичів", "13300"), ("с. Глибочиця", "12430")]),
    ("Закарпатська область", r"Закарпатськ|Ужгород", [("м. Ужгород", "88000"), ("м. Мукачево", "89600"), ("с. Онок", "90350")]),
    ("Запорізька область", r"Запорізьк|Запоріжж", [("м. Запоріжжя", "69000"), ("м. Мелітополь", "72300"), ("м. Бердянськ", "71100")]),
    ("Івано-Франківська область", r"Івано-Франківськ", [("м. Івано-Франківськ", "76000"), ("м. Коломия", "78200"), ("с. Крихівці", "76493")]),
    ("Кіровоградська область", r"Кіровоградськ|Кропивницьк", [("м. Кропивницький", "25000"), ("м. Олександрія", "28000"), ("с. Обознівка", "27632")]),
    ("Луганська область", r"Луганськ|Сєвєродонецьк", [("м. Сєвєродонецьк", "93400"), ("м. Лисичанськ", "93100"), ("м. Старобільськ", "92700")]),
    ("Львівська область", r"Львівськ|Львов|Львів", [("м. Львів", "79000"), ("м. Дрогобич", "82100"), ("м. Стрий", "82400"), ("с. Сокільники", "81130")]),
    ("Миколаївська область", r"Миколаївськ|Миколаєв|Миколаїв", [("м. Миколаїв", "54000"), ("м. Первомайськ", "55200"), ("м. Вознесенськ", "56500")]),
    ("Одеська область", r"Одеськ|Одес", [("м. Одеса", "65000"), ("м. Ізмаїл", "68600"), ("м. Чорноморськ", "68000")]),
    ("Полтавська область", r"Полтавськ|Полтав", [("м. Полтава", "36000"), ("м. Кременчук", "39600"), ("м. Лубни", "37500")]),
    ("Рівненська область", r"Рівненськ|Рівного|м\. ?Рівне", [("м. Рівне", "33000"), ("м. Дубно", "35600"), ("м. Сарни", "34500")]),
    ("Сумська область", r"Сумськ|м\. ?Сум", [("м. Суми", "40000"), ("м. Шостка", "41100"), ("м. Конотоп", "41600")]),
    ("Тернопільська область", r"Тернопільськ|Тернопол|Тернопіл", [("м. Тернопіль", "46000"), ("м. Чортків", "48500"), ("с. Великі Гаї", "47722")]),
    ("Харківська область", r"Харківськ|Харков|Харків", [("м. Харків", "61000"), ("м. Лозова", "64600"), ("м. Чугуїв", "63500")]),
    ("Херсонська область", r"Херсонськ|Херсон", [("м. Херсон", "73000"), ("м. Нова Каховка", "74900"), ("с. Велика Олександрівка", "74100")]),
    ("Хмельницька область", r"Хмельницьк", [("м. Хмельницький", "29000"), ("м. Кам'янець-Подільський", "32300"), ("м. Шепетівка", "30400")]),
    ("Черкаська область", r"Черкаськ|Черкас", [("м. Черкаси", "18000"), ("м. Умань", "20300"), ("м. Сміла", "20700")]),
    ("Чернівецька область", r"Чернівецьк|Чернівц", [("м. Чернівці", "58000"), ("м. Хотин", "60000"), ("с. Мамаївці", "59342")]),
    ("Чернігівська область", r"Чернігівськ|Чернігов|Чернігів", [("м. Чернігів", "14000"), ("м. Ніжин", "16600"), ("м. Прилуки", "17500")]),
]
_REGION_RE = [re.compile(pat) for _, pat, _ in REGIONS]


def doc_region(text: str) -> int | None:
    """Index in REGIONS of the region the document is mostly about (courts, cities, oblasts named in it), or None."""
    counts = [len(r.findall(text)) for r in _REGION_RE]
    best = max(range(len(REGIONS)), key=counts.__getitem__)
    return best if counts[best] else None


STREETS = ["Шевченка", "Незалежності", "Лесі Українки", "Грушевського", "Садова", "Миру", "Центральна", "Івана Франка",
           "Соборна", "Героїв Майдану", "Зелена", "Козацька", "Шкільна", "Степана Бандери", "Січових Стрільців"]
UNIT_NAMES = ['Військовий навчальний центр "Захисник"', "Центр комплектування та соціальної підтримки", "Третій прикордонний загін",
              "Військово-лікувальна комісія Центрального госпіталю", "Окремий батальйон територіальної оборони", "Мобільний госпіталь",
              "Об'єднаний територіальний центр комплектування", "Окремий полк зв'язку"]
TRANSLIT = str.maketrans({"а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e", "є": "ie", "ж": "zh", "з": "z", "и": "y",
                          "і": "i", "ї": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s",
                          "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ь": "", "ю": "iu",
                          "я": "ia", "'": ""})


def _date(rng, y0, y1, right: str = ""):
    y, m, d = rng.randint(y0, y1), rng.randint(1, 12), rng.randint(1, 28)
    if re.match(r"\s*(року|р\.|рік)", right):  # "ІНФОРМАЦІЯ_1 року народження": the year word stays outside the span
        return rng.choices([f"{d:02d}.{m:02d}.{y}", f"{d} {UA_MONTHS[m-1]} {y}", f"{y}"], weights=[5, 3, 1])[0]
    return rng.choices([f"{d:02d}.{m:02d}.{y}", f"{y}-{m:02d}-{d:02d}", f"{d} {UA_MONTHS[m-1]} {y} року", f"{d:02d}.{m:02d}.{y} р.",
                        f"{d:02d}.{m:02d}.{y} року", f"{d:02d}.{m:02d}.{y}р.", f"{d} {UA_MONTHS[m-1]} {y} р.", f"{d}.{m}.{y}"],
                       weights=[6, 1, 3, 2, 1, .5, 1, .5])[0]


def ipn(rng):
    d = [rng.randint(0, 9) for _ in range(9)]
    return "".join(map(str, d)) + str(sum(a * b for a, b in zip(d, [-1, 5, 7, 9, 4, 6, 10, 5, 7])) % 11 % 10)


def iban(rng):
    bban = rng.choice(["305299", "320649", "322001", "300465", "380805", "351005"]) + "".join(str(rng.randint(0, 9)) for _ in range(19))
    return f"UA{98 - int(bban + '301000') % 97:02d}{bban}"  # U=30, A=10, "00"


def gen_number(sub: str | None, rng: random.Random, version: int = 1) -> str:
    L = lambda n: "".join(rng.choice(L_PLATE) for _ in range(n))
    D = lambda n: "".join(str(rng.randint(0, 9)) for _ in range(n))
    if version == 2 and sub in ("phone", "passport", "plate", "iban", "id_card"):
        op = rng.choice(["67", "50", "63", "93", "97", "66", "68", "73", "95", "99", "98"])
        return {
            "phone": rng.choice([f"+380{op}{D(7)}", f"0{op}{D(7)}", f"(0{op}) {D(3)}-{D(2)}-{D(2)}", f"+38 (0{op}) {D(3)}-{D(2)}-{D(2)}",
                                 f"+38(0{op}){D(7)}", f"0{op}-{D(3)}-{D(2)}-{D(2)}", f"+380 {op} {D(3)} {D(2)} {D(2)}", f"0{op} {D(3)} {D(4)}"]),
            "passport": rng.choice([f"{L(2)} {D(6)}", f"{L(2)}{D(6)}", f"{L(2)} № {D(6)}", f"{L(2)} №{D(6)}", D(9)]),
            "plate": rng.choice([f"{L(2)}{D(4)}{L(2)}", f"{L(2)} {D(4)} {L(2)}", f"{L(2)}{D(4)}{L(2)}"]),
            "iban": rng.choice([iban(rng)] * 3 + [" ".join([(s := iban(rng))[:4]] + [s[i:i + 4] for i in range(4, len(s), 4)])]),
            "id_card": D(9),
        }[sub]
    return {
        "ipn": ipn(rng), "iban": iban(rng), "id_card": D(9), "military_unit": "А" + D(4),
        "passport": rng.choice([f"{L(2)} {D(6)}", f"{L(2)}{D(6)}"]),
        "phone": rng.choice([f"+380{rng.choice(['67','50','63','93','97'])}{D(7)}", f"0{rng.choice(['67','50','63','93'])}{D(7)}",
                             f"({rng.choice(['067','050','093'])}) {D(3)}-{D(2)}-{D(2)}"]),
        "plate": rng.choice([f"{L(2)}{D(4)}{L(2)}", f"{L(2)} {D(4)} {L(2)}"]),
        "certificate": f"{L(2)} № {D(6)}",
    }.get(sub or "other", rng.choice([D(rng.randint(6, 10)), f"{L(2)}{D(6)}"]))


def gen_info(sub: str | None, rng: random.Random, surname: str | None = None, left: str = "", right: str = "",
             version: int = 1) -> str:
    nick = (surname or "user").lower().translate(TRANSLIT) + str(rng.randint(1, 99))
    if version == 1:
        return {
            "birth_date": _date(rng, 1940, 2006), "death_date": _date(rng, 2000, 2024),
            "email": f"{nick}@{rng.choice(['ukr.net', 'gmail.com', 'i.ua'])}",
            "url": f"https://{rng.choice(['facebook.com', 'instagram.com', 't.me'])}/{nick}", "nickname": "@" + nick,
            "unit_or_institution": rng.choice(UNIT_NAMES),
        }.get(sub or "other", _date(rng, 1940, 2006))
    quoted = bool(re.search(r"[«\"“„]\s*$", left))
    if sub == "other" or not sub:
        sub = "title" if quoted else "birth_date"
    nick2 = rng.choice([nick, nick.replace("1", "_"), (surname or "user")[:3].lower().translate(TRANSLIT) + rng.choice(["_ua", "777", "_pro", "x", "2000"])])
    gen = {
        "birth_date": lambda: _date(rng, 1940, 2006, right), "death_date": lambda: _date(rng, 1995, 2024, right),
        "email": lambda: f"{rng.choice([nick, nick2, nick.replace('1', '.')])}@{rng.choice(['ukr.net', 'gmail.com', 'i.ua', 'meta.ua', 'outlook.com', 'yahoo.com', 'email.ua'])}",
        "url": lambda: rng.choice([f"https://{rng.choice(['facebook.com', 'www.facebook.com', 'instagram.com', 'www.instagram.com', 't.me', 'tiktok.com/@', 'x.com'])}/{nick2}".replace("@/", "@"),
                                   f"https://{nick2}.{rng.choice(['com.ua', 'in.ua', 'blogspot.com'])}",
                                   f"www.{rng.choice(['facebook.com', 'instagram.com'])}/{nick2}"]),
        "nickname": lambda: (nick2 if quoted else rng.choice(["@" + nick2, nick2, nick2.capitalize()])),
        "unit_or_institution": lambda: gen_unit(rng),
        "company": lambda: gen_company(rng), "title": lambda: rng.choice([gen_company(rng), gen_title(rng)]),
    }
    return gen.get(sub, gen["birth_date"])()


# v2: institutions, military units, companies and titles are built combinatorially instead of an 8-item list
_ORD = ["1", "3", "10", "14", "24", "28", "30", "47", "53", "58", "59", "72", "80", "92", "93", "95", "110", "112", "115", "128", "155", "211"]
_BRIG = ["окрема механізована бригада", "окрема штурмова бригада", "окрема десантно-штурмова бригада", "окрема танкова бригада",
         "окрема артилерійська бригада", "окрема бригада територіальної оборони", "окрема гірсько-штурмова бригада",
         "окрема бригада морської піхоти", "окремий батальйон територіальної оборони", "окремий полк зв'язку", "прикордонний загін"]
_OBL_ADJ = ["Київський", "Львівський", "Харківський", "Одеський", "Дніпропетровський", "Полтавський", "Вінницький", "Житомирський",
            "Черкаський", "Сумський", "Волинський", "Рівненський", "Хмельницький", "Чернігівський", "Закарпатський"]
_DIST_ADJ = ["Шевченківський", "Солом'янський", "Подільський", "Галицький", "Франківський", "Центральний", "Індустріальний",
             "Бориспільський", "Броварський", "Білоцерківський", "Самбірський", "Стрийський", "Ковельський", "Уманський", "Лубенський"]


def gen_unit(rng: random.Random) -> str:
    return rng.choice([
        lambda: f"{rng.choice(_ORD)} {rng.choice(_BRIG)}",
        lambda: f"{rng.choice(_DIST_ADJ)} районний територіальний центр комплектування та соціальної підтримки",
        lambda: f"{rng.choice(_OBL_ADJ)} обласний ТЦК та СП",
        lambda: f"{rng.choice(_DIST_ADJ)} РТЦК та СП",
        lambda: rng.choice(UNIT_NAMES),
        lambda: f"{rng.choice(['Військово-медичний клінічний центр', 'Військовий госпіталь', 'Госпіталь ветеранів війни'])} {rng.choice(['Західного', 'Південного', 'Центрального', 'Північного', 'Східного'])} регіону",
    ])()


_CO_A = ["Агро", "Укр", "Буд", "Енерго", "Тех", "Транс", "Мега", "Нова", "Світ", "Лайф", "Еко", "Альфа", "Дельта", "Прайм", "Сіті", "Голд"]
_CO_B = ["інвест", "сервіс", "трейд", "буд", "постач", "профі", "мережа", "груп", "лайн", "капітал", "холдинг", "маркет", "плюс", "телеком"]


def gen_company(rng: random.Random) -> str:
    a, b = rng.choice(_CO_A), rng.choice(_CO_B)
    return rng.choice([f"{a}{b}", f"{a}-{b.capitalize()}", f"{a}{b.capitalize()}", f"{a} {b.capitalize()}",
                       f"{rng.choice(['Західний', 'Східний', 'Перший', 'Український', 'Київський'])} {rng.choice(['Альянс', 'Дім', 'Центр', 'Стандарт', 'Капітал'])}",
                       f"{a}{b} {rng.choice(['Україна', 'ЛТД', 'Груп'])}"])


def gen_title(rng: random.Random) -> str:
    return rng.choice([f"{rng.choice(['Новини', 'Події', 'Правда', 'Голос', 'Вісті'])} {rng.choice(['Полісся', 'Поділля', 'регіону', 'громади', 'міста'])}",
                       f"{rng.choice(['Чорна', 'Біла', 'Стара', 'Тиха'])} {rng.choice(['гавань', 'зоря', 'хвиля', 'долина'])}",
                       f"{rng.choice(_CO_A)}{rng.choice(_CO_B)}"])


def gen_address(rng: random.Random, region: int | None = None, near: dict | None = None) -> tuple[str, dict]:
    """Fictional address. Mostly in the document's region (85%, of which 60% the regional centre); `near` = an address
    mentioned together with this one ("квартир АДРЕСА_2 та АДРЕСА_3"): same building, another flat."""
    if near:
        parts = {**near, "flat": rng.randint(1, 200)}
    else:
        if region is None or rng.random() > .85:
            region = rng.randrange(len(REGIONS))
        obl, _, places = REGIONS[region]
        city, idx = places[0] if rng.random() < .6 else rng.choice(places)
        parts = {"city": city, "obl": obl, "idx": idx, "street": rng.choice(STREETS), "house": rng.randint(1, 150),
                 "flat": rng.randint(1, 200) if rng.random() < .75 else None, "style": rng.randrange(3), "order": rng.random()}
    st, b, f = parts["street"], parts["house"], parts["flat"]
    street = [f"вул. {st}, буд. {b}", f"вулиця {st}, будинок {b}", f"вул. {st}, {b}"][parts["style"]]
    if f:
        street += [", кв. ", ", квартира ", ", кв. "][parts["style"]] + str(f)
    if parts["order"] < .5:
        out = [p for p in [parts["idx"] if parts["order"] < .25 else None, parts["city"], street, parts["obl"]] if p]
    else:
        out = [street, parts["city"]] + ([parts["obl"]] if parts["obl"] else [])
    return ", ".join(out), parts


# ---------------------------------------------------------------- v2 addresses
_ADDR_RES: dict | None = None
_STREET_ABBR = {"вулиця": ["вул.", "вул.", "вул.", "вулиця"], "проспект": ["просп.", "проспект", "пр-т", "пр."],
                "провулок": ["пров.", "провулок"], "бульвар": ["бульв.", "бульвар", "б-р"], "площа": ["пл.", "площа"],
                "майдан": ["майдан"], "узвіз": ["узвіз"], "шосе": ["шосе"], "набережна": ["набережна"]}


def _gen2nom(adj_gen: str) -> str:
    """Броварського -> Броварський, Київської -> Київська."""
    return re.sub(r"ого$", "ий", re.sub(r"ої$", "а", adj_gen))


def load_address_resources(res_dir: Path) -> dict:
    """streets.csv / settlements.csv from scripts/build_resources.py, settlements grouped by REGIONS index."""
    global _ADDR_RES
    streets = [(r["тип"], r["назва"], float(r["частота"]) ** .5) for r in csv.DictReader(open(res_dir / "streets.csv", encoding="utf-8"))]
    by_region: dict[int, list] = {}
    for r in csv.DictReader(open(res_dir / "settlements.csv", encoding="utf-8")):
        obl = _gen2nom(r["область_род"]) + " область"
        idx = next((i for i, (o, _, _) in enumerate(REGIONS) if o == obl), None)
        if idx is not None:
            by_region.setdefault(idx, []).append((r["тип"], r["назва"], r["район_род"], r["область_род"]))
    _ADDR_RES = {"streets": ([s[:2] for s in streets], [s[2] for s in streets]), "settlements": by_region}
    return _ADDR_RES


SETTLEMENT_CTX = re.compile(r"(уродж\w*|жител\w*|мешкан\w*|населен\w* пункт\w*|поблизу|біля|околиц\w*|території|\bв|\bу|\bселі|\bмісті)\s*$", re.I)


def _house(rng):
    n = rng.choices([rng.randint(1, 40), rng.randint(1, 200)], weights=[3, 2])[0]
    return rng.choices([f"{n}", f"{n}-{rng.choice('АБВГ')}", f"{n}{rng.choice('абв')}", f"{n}/{rng.randint(1, 30)}", f"{n} {rng.choice('АБ')}"],
                       weights=[10, 2, 1, 1.5, .5])[0]


def gen_address_v2(rng: random.Random, region: int | None = None, near: dict | None = None, left: str = "") -> tuple[str, dict]:
    """Address with mined street / settlement names and ~12 surface formats. `left` = text before the placeholder:
    after "уродженця", "жителя", "в", "у" ... only a settlement is generated (that is how the register uses АДРЕСА there)."""
    res = _ADDR_RES
    if near:
        parts = {**near, "flat": rng.randint(1, 200)}
    else:
        if region is None or rng.random() > .85:
            region = rng.randrange(len(REGIONS))
        obl, _, places = REGIONS[region]
        rural = res["settlements"].get(region) if res else None
        only_settlement = bool(SETTLEMENT_CTX.search(left[-40:])) and "адрес" not in left[-25:].lower()
        if rural and rng.random() < (.7 if only_settlement else .3):
            kind, name, dist, obl_gen = rng.choice(rural)
            city, idx, district = f"{kind} {name}", None, dist
        else:
            city, idx = places[0] if rng.random() < .6 else rng.choice(places)
            district, obl_gen = None, None
        (skind, sname) = rng.choices(*res["streets"])[0] if res else ("вулиця", rng.choice(STREETS))
        parts = {"city": city, "obl": obl, "obl_gen": obl_gen, "district": district, "idx": idx,
                 "street": f"{rng.choice(_STREET_ABBR.get(skind, [skind]))} {sname}", "house": _house(rng),
                 "flat": rng.randint(1, 250) if rng.random() < .6 else None, "only_settlement": only_settlement,
                 "style": rng.randrange(4), "order": rng.randrange(6), "ukraine": rng.random() < .05}
    p = parts
    if p["only_settlement"]:
        if p["district"]:
            return rng.choice([f"{p['city']} {p['district']} району {p['obl_gen']} області",
                               f"{p['city']}, {_gen2nom(p['district'])} район, {_gen2nom(p['obl_gen'])} область",
                               f"{p['city']} {p['district']} р-ну {p['obl_gen']} обл."]), parts
        return (p["city"] if not p["obl"] or rng.random() < .6 else f"{p['city']}, {p['obl']}"), parts
    house = [f"буд. {p['house']}", f"будинок {p['house']}", f"{p['house']}", f"б. {p['house']}"][p["style"]]
    street = f"{p['street']}, {house}"
    if p["flat"]:
        street += [", кв. ", ", квартира ", ", кв. ", ", кв."][p["style"]] + str(p["flat"])
    region_tail = ([f"{p['district']} району", f"{p['obl_gen']} області"] if p["district"] else [p["obl"]] if p["obl"] else [])
    region_head = ([f"{_gen2nom(p['obl_gen'])} область", f"{_gen2nom(p['district'])} район"] if p["district"] else [p["obl"]] if p["obl"] else [])
    o = p["order"]
    if o == 0:
        out = [street, p["city"]] + region_tail
    elif o == 1:
        out = [p["city"], street]
    elif o == 2:
        out = ([p["idx"]] if p["idx"] else []) + [p["city"], street]
    elif o == 3:
        out = region_head + [p["city"], street]
    elif o == 4:
        out = ([p["idx"]] if p["idx"] else []) + region_head + [p["city"], street]
    else:
        out = [p["city"], street] + region_tail
    if p["ukraine"]:
        out = ["Україна"] + out
    return ", ".join(out), parts


ROLE_NAME = re.compile(r"(?:[Сс]удд\w*|[Гг]оловуюч\w*|[Сс]екретар\w*|[Аа]двокат\w*|[Зз]ахисник\w*|[Нн]отаріус\w*)[^\n]{0,60}?"
                       r"([А-ЯІЇЄҐ][а-яіїєґ'’ʼ-]+\s+[А-ЯІЇЄҐ]\.\s?[А-ЯІЇЄҐ]\.|[А-ЯІЇЄҐ]\.\s?[А-ЯІЇЄҐ]\.\s?[А-ЯІЇЄҐ][а-яіїєґ'’ʼ-]+)")


FULL_NAME = re.compile(r"(?<![А-ЯІЇЄҐа-яіїєґ'’])([А-ЯІЇЄҐ][а-яіїєґ'’-]{2,}) ([А-ЯІЇЄҐ][а-яіїєґ'’-]{2,}) "
                       r"([А-ЯІЇЄҐ][а-яіїєґ'’-]+(?:ович|овича|овичу|овичем|овичеві|івна|івни|івні|івну|івною|ївна|ївни|ївні|ївну|ївною))(?![а-яіїєґ])")


def clear_names(text: str) -> list[tuple[str, tuple]]:
    """Names written in clear text: role + 'Прізвище І.П.' / 'І.П. Прізвище', and full 'Прізвище Ім'я По-батькові'
    (validated with pymorphy3). Returns (surface, person key); one key = one person within the document."""
    out = []
    for mm in ROLE_NAME.finditer(text):
        name = mm.group(1)
        parts = re.findall(r"[А-ЯІЇЄҐ][а-яіїєґ'’ʼ-]+|[А-ЯІЇЄҐ](?=\.)", name)
        sur = next((p for p in parts if len(p) > 1), "")
        ini = [p for p in parts if len(p) == 1]
        if sur and len(ini) == 2:
            out.append((name, (sur[:4].lower(), ini[0], ini[1])))
    m = morph()
    for mm in FULL_NAME.finditer(text):
        sur, nm, pt = mm.groups()
        if any("Name" in p.tag for p in m.parse(nm.lower())) and any("Patr" in p.tag for p in m.parse(pt.lower())):
            out.append((mm.group(0), (sur[:4].lower(), nm[0], pt[0])))
    return out


def assemble_doc(doc_id, text: str, tab: dict, ann: dict | None, rng: random.Random, pool: str = "train") -> dict:
    """Replaces placeholders using the annotation `ann` from annotate_rules (None -> random gender, nominative case)."""
    ann = ann or {}
    mentions = list(PH.finditer(text))
    persons, genders = {}, {}
    for m in mentions:
        if m.group(1) == "ОСОБА" and m.group(0) not in persons:
            p = make_person(tab, rng, pool)
            g = ann.get("gender", {}).get(m.group(0)) or rng.choice("чж")
            persons[m.group(0)] = p[g]
            genders[m.group(0)] = (g, m.group(0) in ann.get("gender", {}))
    edits, values, n_default_case = [], {}, 0
    region, addr_parts = doc_region(text), {}
    for m in mentions:
        typ, label = m.group(1), m.group(0)
        if typ == "ОСОБА":
            case = ann.get("case", {}).get(str(m.start()))
            n_default_case += case is None
            g, g_ctx = genders[label]
            info = {"case": case or "називний", "case_confident": bool(ann.get("confident", {}).get(str(m.start()))),
                    "gender": g, "gender_from_context": g_ctx}  # for review/analysis; evaluate.py ignores extra keys
            edits.append((m.start(), m.end(), render_name(persons[label], case or "називний", rng), label, info))
            continue
        if label not in values:  # one placeholder = one value within the document
            sub = "" if typ == "АДРЕСА" else (ann.get("subtype", {}).get(label) or subtype_by_context(text, m.start(), typ))
            first = next(iter(persons.values()), None)
            left, right = text[max(0, m.start() - 80):m.start()], text[m.end():m.end() + 20]
            if typ == "АДРЕСА":
                prev = re.search(r"(АДРЕСА_\d+)\s*(?:,|та|і|й)\s*$", text[max(0, m.start() - 30):m.start()])
                near = addr_parts.get(prev.group(1)) if prev else None
                value, addr_parts[label] = (gen_address_v2(rng, region, near, left) if tab.get("version") == 2
                                            else gen_address(rng, region, near))
            else:
                value = (gen_number(sub, rng, tab.get("version", 1)) if typ == "НОМЕР" else
                         gen_info(sub, rng, first["surname"][0] if first else None, left, right, tab.get("version", 1)))
            values[label] = (value, sub)
        value, sub = values[label]
        edits.append((m.start(), m.end(), value, label, {"subtype": sub} if sub else {}))

    # names written in clear text (judges, clerks, lawyers, ...) -> ОСОБА with new numbers
    taken = [(s, e) for s, e, *_ in edits]
    next_n = max([int(n) for t, n in (m.groups() for m in mentions) if t == "ОСОБА"], default=0) + 1
    groups, spans = {}, []
    for name, key in clear_names(text):
        if key not in groups:
            groups[key] = f"ОСОБА_{next_n}"; next_n += 1
        for mm in re.finditer(re.escape(name), text):
            if all(mm.end() <= s or mm.start() >= e for s, e in taken):
                edits.append((mm.start(), mm.end(), name, groups[key], {"subtype": "clear_name"})); taken.append(mm.span())

    out, anns, last = [], [], 0
    for s, e, value, label, info in sorted(edits, key=lambda x: x[0]):
        out.append(text[last:s])
        start = sum(map(len, out))
        out.append(value)
        anns.append({"start": start, "end": start + len(value), "entity": value, "pseudonym": label, **info})
        last = e
    out.append(text[last:])
    original = "".join(out)
    assert all(original[a["start"]:a["end"]] == a["entity"] for a in anns)
    return {"id": doc_id, "original": original, "annotations": anns,
            "meta": {"name_gaps": len(ann.get("case", {})), "confident_cases": sum(ann.get("confident", {}).values()),
                     "persons_gender_from_context": len(ann.get("gender", {})), "default_case": n_default_case,
                     "clear_names": len(groups)}}
