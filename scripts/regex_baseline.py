"""Deterministic regex/dictionary baseline for Ukrainian court PII pseudonymization."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import tracemalloc
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECLENSIONS = ROOT / "notebooks" / "resources" / "name_declensions.csv"
OUTPUT_DIR = ROOT / "baselines"

CYRILLIC_WORD = r"[А-ЯІЇЄҐ][А-ЯІЇЄҐа-яіїєґ'’ʼ\-]*"
PATRONYMIC_END = (
    r"(?:ович[а-яіїєґ]*|івн[а-яіїєґ]*|ївн[а-яіїєґ]*"
)


def load_name_inventory() -> tuple[dict[str, str], set[str], set[str], set[str]]:
    surnames: dict[str, str] = {}
    names: set[str] = set()
    patronymics: set[str] = set()

    with DECLENSIONS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            kind = row["тип"]
            if kind == "прізвище":
                nominative = row["називний"].strip()
                if not nominative:
                    continue
                for column in ("називний", "родовий", "давальний", "знахідний", "орудний", "місцевий", "кличний"):
                    form = row[column].strip()
                    if form:
                        surnames[form.lower()] = nominative.lower()
            elif kind == "ім'я":
                for column in ("називний", "родовий", "давальний", "знахідний", "орудний", "місцевий", "кличний"):
                    form = row[column].strip()
                    if form:
                        names.add(form.lower())
            elif kind == "по батькові":
                for column in ("називний", "родовий", "давальний", "знахідний", "орудний", "місцевий", "кличний"):
                    form = row[column].strip()
                    if form:
                        patronymics.add(form.lower())

    return surnames, names, patronymics, set()


SURNAMES, NAMES, PATRONYMICS, _ = load_name_inventory()

PATRONYMIC_RE = re.compile(
    r"[А-ЯІЇЄҐа-яіїєґ'’ʼ\-]*(?:ович|овича|овичу|овичем|овичеві|"
    r"івна|івни|івні|івну|івною|ївна|ївни|ївні|ївну|ївною)$",
    re.I,
)

ROLE_WORD_RE = re.compile(
    r"(?:судд[яіюєюаів]|головуюч\w*|секретар\w*|адвокат\w*|"
    r"захисник\w*|нотариус\w*|нотаріус\w*|прокурор\w*|"
    r"державн[а-яіїєґ]* виконав[а-яіїєґ]*|підсудн[а-яіїєґ]*|"
    r"позивач[а-яіїєґ]*|відповідач[а-яіїєґ]*)",
    re.I,
)


def merge_spans(spans: list[tuple[int, int, str, str]]) -> list[tuple[int, int, str, str]]:
    by_category: dict[str, list[tuple[int, int, str, str]]] = {}
    for start, end, kind, key in spans:
        by_category.setdefault(kind, []).append((start, end, kind, key))

    merged: list[tuple[int, int, str, str]] = []
    for category in ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ"):
        category_spans = sorted(by_category.get(category, []))
        for start, end, kind, key in category_spans:
            if merged and merged[-1][2] == kind and start <= merged[-1][1]:
                previous_start, previous_end, previous_kind, previous_key = merged[-1]
                if end > previous_end:
                    merged[-1] = (previous_start, end, previous_kind, previous_key)
            else:
                merged.append((start, end, kind, key))
    return sorted(merged)


def person_key(surface: str) -> str:
    cleaned = re.sub(r"\s+", " ", surface.strip().lower())
    cleaned = cleaned.replace(" ", "")
    words = re.findall(r"[а-яіїєґ'’ʼ-]+", surface.lower())
    for word in words:
        if word in SURNAMES:
            return "surname:" + SURNAMES[word]

    if words:
        return "surface:" + cleaned
    return "surface:" + cleaned


def add_person_full_names(text: str, spans: list[tuple[int, int, str, str]]) -> None:
    token_re = re.compile(
        rf"(?<![А-ЯІЇЄҐа-яіїєґ])({CYRILLIC_WORD})\s+"
        rf"({CYRILLIC_WORD})(?:\s+({CYRILLIC_WORD}))?(?![А-ЯІЇЄҐа-яіїєґ])"
    )
    for match in token_re.finditer(text):
        surface = match.group(0)
        words = re.findall(r"[А-ЯІЇЄҐа-яіїєґ'’ʼ-]+", surface)
        if not words:
            continue

        is_full_name = len(words) == 3 and PATRONYMIC_RE.fullmatch(words[2])
        is_known_name = (
            len(words) >= 2
            and words[0].lower() in SURNAMES
            and words[1].lower() in NAMES
        )
        if not (is_full_name or is_known_name):
            continue

        spans.append((match.start(), match.end(), "ОСОБА", person_key(surface)))


def add_initial_names(text: str, spans: list[tuple[int, int, str, str]]) -> None:
    patterns = (
        re.compile(
            rf"(?<![А-ЯІЇЄҐа-яіїєґ])({CYRILLIC_WORD})\s+"
            rf"[А-ЯІЇЄҐ]\.\s*[А-ЯІЇЄҐ]\.(?![А-ЯІЇЄҐа-яіїєґ])"
        ),
        re.compile(
            rf"(?<![А-ЯІЇЄҐа-яіїєґ])[А-ЯІЇЄҐ]\.\s*"
            rf"[А-ЯІЇЄҐ]\.\s*({CYRILLIC_WORD})(?![А-ЯІЇЄҐа-яіїєґ])"
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            surface = match.group(0)
            if patterns[0].pattern == pattern.pattern:
                surname = match.group(1).lower()
            else:
                surname = match.group(1).lower()
            if patterns[0].pattern == pattern.pattern and surname not in SURNAMES:
                continue
            if patterns[1].pattern == pattern.pattern and surname not in SURNAMES:
                continue
            spans.append((match.start(), match.end(), "ОСОБА", person_key(surface)))


def add_role_names(text: str, spans: list[tuple[int, int, str, str]]) -> None:
    role_initial = re.compile(
        rf"(?:{ROLE_WORD_RE.pattern})[^.\n]{{0,80}}?"
        rf"(?<![А-ЯІЇЄҐа-яіїєґ])"
        rf"((?:{CYRILLIC_WORD}\s+)?(?:{CYRILLIC_WORD})"
        rf"(?:\s+[А-ЯІЇЄҐ]\.\s*[А-ЯІЇЄҐ]\.)"
        rf"|(?:[А-ЯІЇЄҐ]\.\s*[А-ЯІЇЄҐ]\.\s*{CYRILLIC_WORD}))",
        re.I,
    )
    for match in role_initial.finditer(text):
        name = match.group(1)
        start = match.start(1)
        end = match.end(1)
        spans.append((start, end, "ОСОБА", person_key(name)))


def add_addresses(text: str, spans: list[tuple[int, int, str, str]]) -> None:
    street_marker = re.compile(
        rf"(?<![А-ЯІЇЄҐа-яіїєґ])(?:по\s+|на\s+)?"
        rf"(?:вул(?:иц[ія])?\.?|просп(?:ект)?\.?|пров(?:улок)?\.?|"
        rf"бул(?:ьвар)?\.?|площа)\s+",
        re.I,
    )
    stop_words = re.compile(
        r"(?i)(?:площа\s+|приміщення\s+|будинок\s+|буд\.\s*|"
        r"квартира\s+|кв\.\s*|м\.\s*|місто\s+|село\s+|смт\s+|"
        r"област|район|[.,]\s*(?:про|за|від|до|у|в|на)\b)"
    )

    for marker in street_marker.finditer(text):
        start = marker.start()
        pos = marker.end()
        limit = min(len(text), pos + 180)
        stop = limit

        stop_match = re.search(r"\n|[;.!?]", text[pos:limit])
        if stop_match:
            stop = pos + stop_match.start()
        stop_match = stop_words.search(text[pos:limit])
        if stop_match:
            stop_match_end = pos + stop_match.end()
            stop = min(stop, stop_match_end)

        segment = text[start:stop].strip().rstrip(", ")
        segment = re.sub(r"\s+", " ", segment)
        if len(segment) >= 10:
            spans.append((start, start + len(segment), "АДРЕСА", "address:" + segment.lower()))


def add_numbers(text: str, spans: list[tuple[int, int, str, str]]) -> None:
    patterns = (
        (re.compile(r"\b(?:UA\d{2}[A-Z0-9]{27,35}|UA\d{2}\s?[0-9A-Z ]{20,32})\b", re.I),
         "ibAN"),
        (re.compile(r"(?<!\d)(?:\+380|0)(?:\d[\d ()-]?){9,12}\d(?!\d)", re.I),
         "phone"),
        (re.compile(r"\b\d{10}\b"),
         "id"),
        (re.compile(r"\b\d{9,10}\b"),
         "identifier"),
        (re.compile(r"(?<![А-ЯІЇЄҐа-яіїєґ])(?:[А-ЯІЇЄҐ]{2}\s*\d{6}|[А-ЯІЇЄҐ]{2}\d{6})(?!\d)", re.I),
         "passport"),
        (re.compile(r"(?<![А-ЯІЇЄҐа-яіїєґ])[А-ЯІЇЄҐ]{2}\s*\d{4}\s*[А-ЯІЇЄҐ]{2}(?![А-ЯІЇЄҐа-яіїєґ])", re.I),
         "plate"),
        (re.compile(r"(?<!\w)(?:в/?ч|ВЧ)\s*[А-ЯІЇЄҐ]?\d{3,6}(?!\w)", re.I),
         "military-unit"),
        (re.compile(r"(?<![A-ZА-ЯІЇЄҐa-zа-яіїєґ0-9])[A-HJ-NPR-Z0-9]{17}(?![A-ZА-ЯІЇЄҐa-zа-яіїєґ0-9])", re.I),
         "vehicle-vin"),
    )

    for pattern, subtype in patterns:
        for match in pattern.finditer(text):
            surface = re.sub(r"\s+", " ", match.group(0).strip())
            spans.append((match.start(), match.end(), "НОМЕР", subtype + ":" + surface.lower()))


def add_information(text: str, spans: list[tuple[int, int, str, str]]) -> None:
    patterns = (
        re.compile(r"\b[^\s@]+@[A-Za-zА-ЯІЇЄҐа-яіїєґ0-9.-]+\.[A-Za-z]{2,}\b", re.I),
        re.compile(r"https?://[^\s<>()]+", re.I),
        re.compile(r"(?:https?://)?(?:www\.)?(?:facebook\.com|instagram\.com|t\.me)/[^\s<>()]+", re.I),
        re.compile(r"(?<![\w@])@[A-Za-zА-ЯІЇЄҐа-яіїєґ0-9_.\-]+", re.I),
        re.compile(r"\b(?:19|20)\d{2}[./-](?:0?[1-9]|1[0-2])[./-](?:0?[1-9]|[12]\d|3[01])\b"),
        re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[./](?:0?[1-9]|1[0-2])[./](?:19|20)\d{2}\b"),
        re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])\s+(?:січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)\s+(?:19|20)\d{2}(?:\s+року(?:\s+народження)?)?\b", re.I),
    )
    subtype_names = ("email", "url", "url", "nickname", "date", "date", "date")
    for pattern, subtype in zip(patterns, subtype_names):
        for match in pattern.finditer(text):
            if subtype == "date":
                left = text[max(0, match.start() - 100):match.start()].lower()
                right = text[match.end():min(len(text), match.end() + 100)].lower()
                context = left + " " + right
                date_context = re.search(
                    r"(?:народжен|народивс|уродж|помер|смерт|день народжен|"
                    r"дат[аі].{0,20}(?:народжен|смерт)|р\.н\.)",
                    context,
                    re.I,
                )
                if not date_context:
                    continue
            surface = re.sub(r"\s+", " ", match.group(0).strip())
            spans.append((match.start(), match.end(), "ІНФОРМАЦІЯ", subtype + ":" + surface.lower()))


def predict(text: str) -> list[dict]:
    spans: list[tuple[int, int, str, str]] = []
    add_person_full_names(text, spans)
    add_initial_names(text, spans)
    add_role_names(text, spans)
    add_addresses(text, spans)
    add_numbers(text, spans)
    add_information(text, spans)
    spans = merge_spans(spans)

    entity_numbers: dict[tuple[str, str], int] = {}
    annotations = []
    for start, end, kind, key in spans:
        category_key = (kind, key)
        if category_key not in entity_numbers:
            category_count = sum(1 for ckey in entity_numbers if ckey[0] == kind) + 1
            entity_numbers[category_key] = category_count
        annotations.append({
            "start": start,
            "end": end,
            "pseudonym": f"{kind}_{entity_numbers[category_key]}",
        })
    return annotations


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_split(name: str, path: Path) -> dict:
    gold = load_jsonl(path)
    predictions = []
    started = time.perf_counter()
    tracemalloc.start()
    for row in gold:
        text = row["original"]
        predictions.append({"id": row["id"], "annotations": predict(text)})
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    sys.path.insert(0, str(ROOT / "data"))
    import evaluate as official_evaluate

    result = official_evaluate.evaluate(gold, predictions)
    result["runtime_seconds"] = round(elapsed, 3)
    result["peak_traced_memory_mib"] = round(peak_bytes / 1024 / 1024, 2)

    output_path = OUTPUT_DIR / f"regex_{name}_submission.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    return result


def compact_summary(name: str, result: dict) -> dict:
    return {
        "split": name,
        "documents": result["n_docs"],
        "final_score": result["final_score"],
        "relaxed_macro_precision": result["ner_relaxed"]["macro"]["precision"],
        "relaxed_macro_recall": result["ner_relaxed"]["macro"]["recall"],
        "relaxed_macro_f1": result["ner_relaxed"]["macro"]["f1"],
        "strict_macro_f1": result["ner_strict"]["macro"]["f1"],
        "deanonymization_risk": result["deanonymization_risk_by_name"],
        "consistency_b3": result["consistency_b3"]["f1"],
        "runtime_seconds": result["runtime_seconds"],
        "peak_traced_memory_mib": result["peak_traced_memory_mib"],
    }


def markdown_report(summaries: list[dict]) -> str:
    lines = [
        "# Regex Baseline Evaluation",
        "",
        "## Scope",
        "",
        "This baseline uses only deterministic regexes and the reviewed Ukrainian name",
        "declension table. It does not use an LLM, pretrained NER model, GPU, or paid API.",
        "",
        "## Summary",
        "",
        "| Split | Documents | Relaxed macro F1 | Strict macro F1 | Deanonymization risk | Consistency B³ | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['split']} | {row['documents']} | {row['relaxed_macro_f1']:.4f} | "
            f"{row['strict_macro_f1']:.4f} | {row['deanonymization_risk']:.4f} | "
            f"{row['consistency_b3']:.4f} | {row['runtime_seconds']:.3f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Low strict F1 may be caused by boundary differences, even if relaxed F1 is higher.",
        "- Low person recall likely comes from name forms absent from the declension table.",
        "- Low address/number recall is not evidence that synthetic gold labels are wrong;",
        "  it may reflect limited or incorrect baseline rules.",
        "- High deanonymization risk on an official domain indicates missed repeated person names.",
        "- Consistency is meaningful only where person mentions are found; no consistency",
        "  improvement is inferred from strict/relaxed F1 alone.",
        "",
        "## Data-regeneration decision",
        "",
        "Do not regenerate generated datasets solely because this baseline scores low.",
        "First compare results with the gold labels:",
        "",
        "1. If many synthetic missed spans are real PII with correct gold boundaries, improve",
        "   baseline patterns or the model baseline.",
        "2. If gold spans or synthetic case/gender metadata are grammatically wrong, fix",
        "   `synth_utils.py` and regenerate.",
        "3. If the official test also exposes repeated-name or boundary weaknesses, that is",
        "   primarily a model-rule issue rather than a training-data regeneration issue.",
        "",
        "## Runtime and resource note",
        "",
        "Reported runtime is Python process inference time for the selected split.",
        "Peak traced-memory is an allocator estimate, not OS peak RSS.",
        "A final CPU/RAM measurement must use the complete official submission and",
        "`/usr/bin/time` on Linux or an equivalent Windows process-memory measurement.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Evaluate only data/test/court_cases.jsonl",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = (
        ("dev", ROOT / "data" / "synthetic" / "dev.jsonl"),
        ("test_a", ROOT / "data" / "synthetic" / "test_a.jsonl"),
        ("test_b", ROOT / "data" / "synthetic" / "test_b.jsonl"),
        ("official", ROOT / "data" / "test" / "court_cases.jsonl"),
    )
    if args.official_only:
        datasets = (datasets[-1],)

    summaries = []
    full_results = {}
    for name, path in datasets:
        if not path.exists():
            raise FileNotFoundError(path)
        result = evaluate_split(name, path)
        full_results[name] = result
        summaries.append(compact_summary(name, result))

    (OUTPUT_DIR / "regex_baseline_results.json").write_text(
        json.dumps(full_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "regex_baseline_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "REGEX_BASELINE.md").write_text(
        markdown_report(summaries),
        encoding="utf-8",
    )

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
