"""Hybrid layer on top of the neural NER: high-precision rules for structured PII + span-boundary repair.

    spans = model spans [(start, end, category), ...]
    final = combine(text, spans)            # snap boundaries, add rule spans, resolve overlaps

Three independent steps (each can be switched off for ablations):

1. ``snap``   - boundary repair of model spans: extend a span that starts / ends inside a word to the whole word
                (sub-word tokens), strip surrounding whitespace / punctuation / quotes, keep the final dot of
                initials and abbreviations ("Петренко О.О.", "буд.", "р."), join АДРЕСА pieces separated only by
                ", " and ОСОБА pieces separated only by a space.
2. ``rules``  - deterministic detectors for PII whose shape is fixed: IBAN (mod-97 check), РНОКПП (checksum or
                keyword), phones, passports / ID cards, licence plates, military-unit numbers, e-mails, social URLs,
                @nicknames, birth / death dates (by keyword or right after a person span).
3. ``merge``  - a rule span replaces overlapping model spans of the same category (its boundaries are exact);
                for a different category the rule wins unless the model span is ОСОБА.

Every rule was designed from the task description and the synthetic train / dev data only; the official test is
never used to choose or tune them.
"""
from __future__ import annotations

import re

CATEGORIES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")
WORD_CH = re.compile(r"[0-9A-Za-zА-ЯІЇЄҐа-яіїєґ'’ʼ`]")
STRIP_CH = " \t\n\r,;:()[]{}«»\"“”„'’`|—–"
ABBR_DOT = re.compile(r"(?:(?<![А-ЯІЇЄҐа-яіїєґA-Za-z])[А-ЯІЇЄҐA-Z]|\b(?:буд|кв|вул|просп|пров|бульв|пл|обл|р|м|с|смт|корп|прим|сел|р-ну|н))\.$")
DATE_CORE = (r"(?:\d{1,2}[./]\d{1,2}[./](?:19|20)\d{2}|(?:19|20)\d{2}-\d{2}-\d{2}|"
             r"\d{1,2}\s+(?:січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)\s+(?:19|20)\d{2})")
DATE_RE = re.compile(rf"(?<![\d.]){DATE_CORE}(?:\s?(?:року|р\.))?(?![\d])")


# ------------------------------------------------------------------ 1. boundary repair
def _is_word(ch: str) -> bool:
    return bool(WORD_CH.match(ch))


def snap_span(text: str, start: int, end: int, category: str) -> tuple[int, int] | None:
    # grow to whole words (a span that cuts a word in half is always a tokenisation artefact)
    while start > 0 and _is_word(text[start - 1]) and _is_word(text[start]):
        start -= 1
    while end < len(text) and _is_word(text[end - 1]) and _is_word(text[end]):
        end += 1
    # strip surrounding junk
    while start < end and text[start] in STRIP_CH:
        start += 1
    while end > start and text[end - 1] in STRIP_CH:
        end -= 1
    # a final dot belongs to the span only after an initial or an abbreviation
    while end > start and text[end - 1] == "." and not ABBR_DOT.search(text[max(start, end - 8):end]):
        end -= 1
        while end > start and text[end - 1] in STRIP_CH:
            end -= 1
    # an initial without its dot ("Петренко О.О" + ".") -> take the dot
    if category == "ОСОБА" and end < len(text) and text[end] == "." and re.search(r"(?<![А-ЯІЇЄҐа-яіїєґ])[А-ЯІЇЄҐ]$", text[start:end]):
        end += 1
    if end - start < 2 and category != "НОМЕР":
        return None
    return (start, end) if end > start else None


def join_pieces(text: str, spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for s, e, c in sorted(spans):
        if out and out[-1][2] == c:
            gap = text[out[-1][1]:s]
            if (c == "АДРЕСА" and re.fullmatch(r"\s*,?\s*", gap)) or (c == "ОСОБА" and re.fullmatch(r" ?", gap)) \
                    or (c in ("НОМЕР", "ІНФОРМАЦІЯ") and gap == "") or s < out[-1][1]:
                out[-1] = (out[-1][0], max(e, out[-1][1]), c)
                continue
        out.append((s, e, c))
    return out


def snap(text: str, spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    fixed = []
    for s, e, c in spans:
        r = snap_span(text, s, e, c)
        if r:
            fixed.append((*r, c))
    return join_pieces(text, fixed)


# ------------------------------------------------------------------ 2. rules
def _iban_ok(s: str) -> bool:
    s = s.replace(" ", "")
    if not re.fullmatch(r"UA\d{27}", s):
        return False
    num = "".join(str(int(ch, 36)) for ch in s[4:] + s[:4])
    return int(num) % 97 == 1


def _rnokpp_ok(s: str) -> bool:
    d = [int(x) for x in s]
    return sum(a * b for a, b in zip(d[:9], [-1, 5, 7, 9, 4, 6, 10, 5, 7])) % 11 % 10 == d[9]


PUBLIC_ACCOUNT = re.compile(r"казначейств|ГУК\b|УДКСУ|ДКСУ|бюджет|ЄДРПОУ|ДПС|прокуратур|суду|депозитн|отримувач\w*\s*:?\s*(?:ГУ|Головн|Державн)", re.I)
TREASURY_BANKS = ("820172", "899998")  # State Treasury bank codes: accounts of public bodies, not personal data
ROLE_MAILBOX = re.compile(r"^(?:office|info|inbox|support|abuse|admin|mail|post|sales|contact|kanc|kancelaria|court|zvern\w*|press|hr|secretary|reception|noreply|no-reply|help|service|ukrposhta|bank|legal|law|kyiv|lviv)[._-]?\d*@", re.I)
LEGAL_ENTITY = re.compile(r"\b(?:ТОВ|ТзОВ|ПрАТ|ПАТ|АТ|ЗАТ|ПП|ФГ|ДП|КП|СТОВ|ОСББ)\b|компані\w*|центр\w* зайнятост|банк\w*|рад[аиі]\b|управлінн\w*|департамент\w*|фонд\w*")
ORG_CTX = re.compile(r"факс|гаряч\w* лін\w*|ЄДРПОУ|ідентифікаційний код юридичн|код\s+\d{8}|офіс|канцеляр|приймальн", re.I)
NOT_ID_CTX = re.compile(r"(справ\w*|провадженн\w*|ЄДРПОУ|кадастров\w*|договор\w*|рахун\w*|ЄУН|№)\s*:?\s*$", re.I)
ID_CTX = re.compile(r"(РНОКПП|ІПН|ідентифікаційн\w*|облікової картки|податков\w* номер\w*)\D{0,25}$", re.I)
PASSPORT_CTX = re.compile(r"паспорт\w*|ID[- ]?карт\w*|посвідченн\w*|документ\w*, що посвідчує", re.I)
PERSON_PHONE_CTX = re.compile(r"тел\w*|моб\w*|номер\w*|сім[- ]?карт\w*|абонент\w*|viber|telegram|whatsapp", re.I)
OFFICIAL_DOMAIN = re.compile(r"(?:gov\.ua|court\.gov|rada\.gov|reyestr|zakon\.|minjust|ccu\.gov|npu\.gov|e-court)", re.I)
SOCIAL_URL = re.compile(r"(?:https?://)?(?:www\.)?(?:facebook\.com|fb\.com|instagram\.com|t\.me|tiktok\.com|youtube\.com|youtu\.be|x\.com|twitter\.com|"
                        r"vk\.com|ok\.ru|olx\.ua|linkedin\.com|threads\.net)/[^\s<>()«»\"“”,;]+", re.I)
PLATE_L = "ABCEHIKMOPTXАВЕІКМНОРСТХ"


def rule_spans(text: str, person_spans: list[tuple[int, int, str]] | None = None) -> list[tuple[int, int, str, str]]:
    """(start, end, category, rule name) for structured PII."""
    out: list[tuple[int, int, str, str]] = []
    add = lambda m, c, name, g=0: out.append((m.start(g), m.end(g), c, name))
    left = lambda i, n=60: text[max(0, i - n):i]

    for m in re.finditer(r"(?<![A-Z0-9])UA\d{2}(?: ?[0-9]){25}(?![0-9])", text):
        bank = m.group(0).replace(" ", "")[4:10]
        if _iban_ok(m.group(0)) and bank not in TREASURY_BANKS and not PUBLIC_ACCOUNT.search(left(m.start(), 200) + text[m.end():m.end() + 40]):
            add(m, "НОМЕР", "iban")
    for m in re.finditer(r"(?<![\d/.-])\d{10}(?![\d/.:-])", text):
        ctx = left(m.start())
        if ID_CTX.search(ctx):
            add(m, "НОМЕР", "rnokpp")
            continue
        if NOT_ID_CTX.search(ctx[-25:]) or re.search(r"(?<![А-ЯІЇЄҐA-Z])[A-ZА-ЯІЇЄҐ]{1,3}\s?$", ctx) or re.search(r"кадастров", ctx[-40:], re.I):
            continue
        if _rnokpp_ok(m.group(0)) and not m.group(0).startswith(("0", "380")):
            add(m, "НОМЕР", "rnokpp")
    for m in re.finditer(r"(?<![\d+])(?:\+?38[\s-]?)?(?:\(0[1-9]\d\)|0[1-9]\d)[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?![\d:-])|"
                         r"(?<![\d+])\+?380\s?\d{2}\s?\d{3}\s?\d{2}\s?\d{2}(?!\d)|(?<!\d)0\d{2} \d{3} \d{4}(?!\d)", text):
        g = m.group(0)
        formatted = g.startswith(("+", "(")) or bool(re.search(r"[\s-]", g))
        if PERSON_PHONE_CTX.search(left(m.start(), 40)) or formatted:
            if not re.search(r"\bсуд\w*|\bСУД\b|канцеляр\w*|приймальн\w*|прокуратур\w*", left(m.start(), 150), re.I) \
                    and not re.search(r"inbox@|fax|факс", text[m.end():m.end() + 40], re.I) \
                    and not ORG_CTX.search(left(m.start(), 100)) and not re.search(r"[A-ZА-ЯІЇЄҐ]{1,3}\s?$", left(m.start(), 5)):
                add(m, "НОМЕР", "phone")
    for m in re.finditer(rf"(?<![А-ЯІЇЄҐA-Z0-9])[{PLATE_L}]{{2}} ?\d{{4}} ?[{PLATE_L}]{{2}}(?![А-ЯІЇЄҐа-яіїєґA-Za-z0-9])", text):
        add(m, "НОМЕР", "plate")
    for m in re.finditer(r"(?<![А-ЯІЇЄҐA-Z0-9])[А-ЯІЇЄҐ]{2} ?(?:№ ?)?\d{6}(?!\d)", text):
        if PASSPORT_CTX.search(left(m.start(), 60)):
            add(m, "НОМЕР", "passport")
    for m in re.finditer(r"(?<!\d)\d{9}(?!\d)", text):
        if re.search(r"(ID[- ]?карт\w*|паспорт\w* громадянина України у формі картки|паспорт\w*)[^.\n]{0,40}$", left(m.start(), 70), re.I):
            add(m, "НОМЕР", "id_card")
    for m in re.finditer(r"(?:в/ч|військов\w+ частин\w+|ВЧ)\s*(?:№\s*)?([АВ]\s?\d{4})(?!\d)", text):
        add(m, "НОМЕР", "military_unit", 1)

    for m in re.finditer(r"(?<![\w.@-])[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])", text):
        if not OFFICIAL_DOMAIN.search(m.group(0)) and not ROLE_MAILBOX.match(m.group(0)) and not ORG_CTX.search(left(m.start(), 120)) \
                and not LEGAL_ENTITY.search(left(m.start(), 70)):
            add(m, "ІНФОРМАЦІЯ", "email")
    for m in SOCIAL_URL.finditer(text):
        s, e = m.span()
        while e > s and text[e - 1] in ".)":
            e -= 1
        out.append((s, e, "ІНФОРМАЦІЯ", "url"))
    for m in re.finditer(r"(?<![\w@./|])@[A-Za-z][A-Za-z0-9_.]{2,30}(?![\w@])", text):
        if not re.search(r"\.(?:net|com|ua|org|ru)$", m.group(0)):
            add(m, "ІНФОРМАЦІЯ", "nickname")

    person_ends = sorted(e for s, e, c in (person_spans or []) if c == "ОСОБА")
    for m in DATE_RE.finditer(text):
        ctx, right = left(m.start(), 70), text[m.end():m.end() + 30]
        birth = re.search(r"(народж\w*|народи\w+|уродж\w*|р\.\s?н\.|дата народження)\W{0,12}$", ctx, re.I) \
            or re.match(r"\s*(?:р\.|року)?\s*(народження|н\.|нар\.)", right, re.I)
        death = re.search(r"(помер\w*|смерт\w*|загину\w*|загибел\w*)\W{0,15}$", ctx, re.I)
        after_person = any(0 <= m.start() - pe <= 3 and re.fullmatch(r"\s*[,(]\s*", text[pe:m.start()]) for pe in person_ends[-200:]) \
            if person_ends else False
        if birth or death or after_person:
            add(m, "ІНФОРМАЦІЯ", "birth_date" if not death else "death_date")
    return out


# ------------------------------------------------------------------ 3. merge
def merge_with_rules(model: list[tuple[int, int, str]], rules: list[tuple[int, int, str, str]]) -> list[tuple[int, int, str]]:
    # rule spans among themselves: keep the longest of overlapping ones
    chosen: list[tuple[int, int, str]] = []
    for s, e, c, _ in sorted(rules, key=lambda r: (-(r[1] - r[0]), r[0])):
        if all(e <= cs or s >= ce for cs, ce, _ in chosen):
            chosen.append((s, e, c))
    result = []
    for s, e, c in model:
        clash = [(rs, re_, rc) for rs, re_, rc in chosen if not (e <= rs or s >= re_)]
        if not clash:
            result.append((s, e, c))
            continue
        if c == "ОСОБА" and all(rc != "ОСОБА" for *_, rc in clash):
            # keep the person, drop the clashing rule spans
            chosen = [r for r in chosen if r not in clash]
            result.append((s, e, c))
            continue
        if c == "АДРЕСА" and all(rc == "НОМЕР" for *_, rc in clash) and all(rs >= s and re_ <= e for rs, re_, _ in clash):
            chosen = [r for r in chosen if r not in clash]  # digits inside an address are the address
            result.append((s, e, c))
            continue
        # otherwise the rule span replaces the model span; keep any model remainder outside the rule spans
    return sorted(set(result) | set(chosen))


def combine(text: str, model_spans: list[tuple[int, int, str]], use_snap: bool = True, use_rules: bool = True) -> list[tuple[int, int, str]]:
    spans = snap(text, model_spans) if use_snap else sorted(model_spans)
    if use_rules:
        spans = merge_with_rules(spans, rule_spans(text, spans))
    return sorted(spans)
