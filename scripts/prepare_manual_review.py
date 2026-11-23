"""Prepare a deterministic, risk-ranked manual-review sample for 20 dev documents."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEV_PATH = ROOT / "data" / "synthetic" / "dev.jsonl"
OUTPUT_CSV = ROOT / "data" / "synthetic" / "manual_review_sample.csv"
OUTPUT_MD = ROOT / "MANUAL_REVIEW.md"


def load_rows() -> list[dict]:
    with DEV_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def risk_flags(row: dict) -> tuple[list[str], int]:
    flags = []
    score = 0
    annotations = row["annotations"]

    uncertain = [
        ann for ann in annotations
        if ann["pseudonym"].startswith("ОСОБА_")
        and ann.get("case_confident") is False
    ]
    random_gender = [
        ann for ann in annotations
        if ann["pseudonym"].startswith("ОСОБА_")
        and ann.get("gender_from_context") is False
    ]
    clear_names = [
        ann for ann in annotations if ann.get("subtype") == "clear_name"
    ]
    rare_subtypes = [
        ann for ann in annotations
        if ann.get("subtype") in {
            "email", "url", "nickname", "death_date",
            "unit_or_institution", "iban", "passport", "plate"
        }
    ]
    addresses = [
        ann for ann in annotations if ann["pseudonym"].startswith("АДРЕСА_")
    ]

    if uncertain:
        flags.append(f"uncertain_case={len(uncertain)}")
        score += len(uncertain) * 3
    if random_gender:
        flags.append(f"random_gender={len(random_gender)}")
        score += len(random_gender) * 2
    if clear_names:
        flags.append(f"clear_text_name={len(clear_names)}")
        score += len(clear_names)
    if rare_subtypes:
        flags.append(f"rare_subtype={len(rare_subtypes)}")
        score += len(rare_subtypes) * 2
    if addresses:
        flags.append(f"address={len(addresses)}")
        score += len(addresses)
    return flags, score


def make_snippet(row: dict) -> str:
    text = row["original"]
    priorities = []
    for ann in row["annotations"]:
        flags = []
        if ann.get("case_confident") is False:
            flags.append("uncertain case")
        if ann.get("gender_from_context") is False:
            flags.append("random gender")
        if ann.get("subtype") == "clear_name":
            flags.append("clear name")
        if ann.get("subtype") in {"email", "url", "nickname", "death_date", "unit_or_institution", "iban", "passport", "plate"}:
            flags.append("rare subtype")
        if flags:
            priorities.append((ann["start"], ann["end"], ann["pseudonym"], flags))

    if priorities:
        start, end, _, flags = min(priorities, key=lambda item: item[0])
        left = max(0, start - 120)
        right = min(len(text), end + 160)
        snippet = text[left:right].replace("\n", " ").replace("\u0000", "")
        return f"{snippet} [Flags: {'; '.join(flags)}]"

    start = 0
    end = min(len(text), 300)
    snippet = text[start:end].replace("\n", " ").replace("\u0000", "")
    return snippet


def main() -> None:
    rows = load_rows()
    ranked = []
    for row in rows:
        flags, score = risk_flags(row)
        ranked.append((score, row, flags))

    ranked.sort(key=lambda item: (-item[0], str(item[1]["id"])))
    selected = ranked[:20]

    fieldnames = [
        "document_id",
        "rank",
        "risk_score",
        "risk_flags",
        "annotation_count",
        "snippet",
        "human_verdict",
        "comment",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (score, row, flags) in enumerate(selected, 1):
            writer.writerow({
                "document_id": row["id"],
                "rank": rank,
                "risk_score": score,
                "risk_flags": "; ".join(flags),
                "annotation_count": len(row["annotations"]),
                "snippet": make_snippet(row),
                "human_verdict": "pending",
                "comment": "",
            })

    lines = [
        "# Manual Review",
        "",
        "## Scope",
        "",
        "This file records the preparation and results of manual review for 20 `dev` documents.",
        "The sample is risk-ranked automatically: uncertain cases, random gender, clear-text",
        "names, rare subtypes, and addresses receive a higher review priority.",
        "",
        "## Important Status",
        "",
        "A generated review sample is not the same as a completed human verdict.",
        "Reviewers must open each document in `notebooks/review_app.py` or inspect",
        "`data/synthetic/manual_review_sample.csv`, then record `ok` or `problem`.",
        "",
        "## Review Procedure",
        "",
        "For every selected document, verify:",
        "",
        "1. Every synthetic PII span is complete.",
        "2. Span boundaries exclude surrounding punctuation and whitespace.",
        "3. The pseudonym is consistent across repeated mentions.",
        "4. Person case and gender agree with surrounding Ukrainian grammar.",
        "5. Clear-text names are complete and correctly classified.",
        "6. Address, number, and information subtypes match the context.",
        "7. No unrelated text was replaced.",
        "",
        "## Selected Documents",
        "",
        "| Rank | Document ID | Risk score | Risk flags | Annotations | Verdict |",
        "|---:|---:|---:|---|---:|---|",
    ]

    for rank, (score, row, flags) in enumerate(selected, 1):
        lines.append(
            f"| {rank} | `{row['id']}` | {score} | "
            f"{'; '.join(flags) or 'representative'} | "
            f"{len(row['annotations'])} | pending human review |"
        )

    lines += [
        "",
        "## Automated Structural Pre-Review",
        "",
        "Before human review, the sample is checked against:",
        "",
        "- valid JSON and character offsets;",
        "- non-empty spans;",
        "- known PII-prefix types;",
        "- no overlapping annotation spans;",
        "- the existing official evaluator format.",
        "",
        "## Machine-Readable Checklist",
        "",
        "`data/synthetic/manual_review_sample.csv` contains document IDs, risk scores,",
        "risk flags, snippets, and empty `human_verdict` / `comment` fields.",
        "",
        "## Human Review Results",
        "",
        "No human verdict has been fabricated. This file must be updated after a reviewer",
        "uses the local review UI and records `ok` or `problem` for each document.",
    ]

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Selected {len(selected)} dev documents")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
