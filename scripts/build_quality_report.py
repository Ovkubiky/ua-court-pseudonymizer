"""Build an English quality report from the synthetic and official datasets."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
OFFICIAL_TEST = ROOT / "data" / "test" / "court_cases.jsonl"
PII_TYPES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")
SPLITS = ("train", "dev", "test_a", "test_b")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dataset_stats(split: str) -> dict:
    rows = load_jsonl(SYNTHETIC_DIR / f"{split}.jsonl")
    annotations = [ann for row in rows for ann in row["annotations"]]
    persons = [
        ann for ann in annotations if ann["pseudonym"].startswith("ОСОБА_")
    ]

    return {
        "split": split,
        "documents": len(rows),
        "annotations": len(annotations),
        "annotations_per_document": round(len(annotations) / len(rows), 2),
        "median_document_length": int(statistics.median(len(row["original"]) for row in rows)),
        "persons": len(persons),
        "uncertain_cases": sum(
            ann.get("case_confident") is False for ann in persons
        ),
        "random_gender": sum(
            ann.get("gender_from_context") is False for ann in persons
        ),
        "clear_text_names": sum(
            ann.get("subtype") == "clear_name" for ann in annotations
        ),
        "person_type": sum(
            ann["pseudonym"].startswith("ОСОБА_") for ann in annotations
        ),
        "address_type": sum(
            ann["pseudonym"].startswith("АДРЕСА_") for ann in annotations
        ),
        "number_type": sum(
            ann["pseudonym"].startswith("НОМЕР_") for ann in annotations
        ),
        "information_type": sum(
            ann["pseudonym"].startswith("ІНФОРМАЦІЯ_") for ann in annotations
        ),
    }


def subtype_counts(split: str) -> Counter:
    rows = load_jsonl(SYNTHETIC_DIR / f"{split}.jsonl")
    return Counter(
        ann.get("subtype", "missing")
        for row in rows
        for ann in row["annotations"]
        if ann.get("subtype")
    )


def main() -> None:
    stats = [dataset_stats(split) for split in SPLITS]
    stats_df = pd.DataFrame(stats)
    stats_csv = ROOT / "data" / "synthetic" / "quality_report.csv"
    stats_df.to_csv(stats_csv, index=False)

    official_rows = load_jsonl(OFFICIAL_TEST)
    official_annotations = [
        ann for row in official_rows for ann in row["annotations"]
    ]
    official_types = Counter(
        ann["pseudonym"].split("_", 1)[0].strip()
        for ann in official_annotations
    )

    issue_rows = [
        {
            "issue": "Uncertain grammatical cases",
            "frequency": f"{sum(row['uncertain_cases'] for row in stats) / sum(row['persons'] for row in stats):.1%}",
            "impact": "Person mentions may receive an incorrect grammatical form.",
            "evidence": "Case-confidence flags in generated annotations.",
            "current_resolution": "Fall back to nominative case.",
            "next_action": "Improve the rule priority and manually resolve gold-set errors.",
            "status": "Known limitation",
        },
        {
            "issue": "Random gender assignment",
            "frequency": f"{sum(row['random_gender'] for row in stats) / sum(row['persons'] for row in stats):.1%}",
            "impact": "A name or surrounding agreement may have a mismatched gender.",
            "evidence": "gender_from_context=false in person annotations.",
            "current_resolution": "Assign gender only from confidence-bearing context.",
            "next_action": "Expand context inference and flag remaining mismatches.",
            "status": "Known limitation",
        },
        {
            "issue": "Clear-text names",
            "frequency": f"{sum(row['clear_text_names'] for row in stats)} annotations",
            "impact": "A model can learn to ignore public names that remain in training text.",
            "evidence": "Subtype clear_name in synthetic annotations.",
            "current_resolution": "Detected initialism/full-name patterns are labelled.",
            "next_action": "Add surname-only and first-name-only detection.",
            "status": "Partial",
        },
        {
            "issue": "Address-region mismatch",
            "frequency": "Not yet measured from saved annotations",
            "impact": "Generated addresses may be geographically inconsistent.",
            "evidence": "Generator uses document region for 85% of addresses.",
            "current_resolution": "Region detection is used during generation.",
            "next_action": "Measure detected-region/address consistency on all splits.",
            "status": "Partially implemented",
        },
        {
            "issue": "Limited name inventory",
            "frequency": "24 given names and 32 surnames per gender, approximately",
            "impact": "Synthetic documents may encourage memorization of a small name set.",
            "evidence": "docs/DECISION_LOG.md and name-declension table.",
            "current_resolution": "Train/test names come from separate pools.",
            "next_action": "Integrate lang-uk/name_freq_dict_uk.",
            "status": "Planned",
        },
        {
            "issue": "Small official-test type coverage",
            "frequency": "168 ІНФОРМАЦІЯ entities",
            "impact": "Subtype-level conclusions have wide confidence intervals.",
            "evidence": "Official test statistics.",
            "current_resolution": "Rare-subtype test_b is used for internal measurement.",
            "next_action": "Report confidence intervals and per-subtype errors.",
            "status": "Mitigated",
        },
        {
            "issue": "Official test domain shift",
            "frequency": "240 decisions; 97.5% Supreme Court cassation",
            "impact": "A court-model may not generalize to the broad training domain.",
            "evidence": "data_stats.ipynb and docs/DECISION_LOG.md.",
            "current_resolution": "Stratify synthetic sets across proceeding type and instance.",
            "next_action": "Compare in-domain and cross-domain results.",
            "status": "Measured",
        },
    ]
    issues_df = pd.DataFrame(issue_rows)
    issues_csv = ROOT / "data" / "synthetic" / "quality_issues.csv"
    issues_df.to_csv(issues_csv, index=False)

    subtype_tables = []
    for split in SPLITS:
        counts = subtype_counts(split)
        subtype_tables.append(
            f"### {split}\n\n"
            + "\n".join(
                f"- `{subtype}`: {count}"
                for subtype, count in counts.most_common()
            )
        )

    lines = [
        "# Data Quality Report",
        "",
        "## Scope",
        "",
        "This report describes synthetic-data quality and known limitations. It does not claim",
        "model quality: no baseline or fine-tuned NER model has been evaluated yet.",
        "",
        "## Dataset Counts",
        "",
        "| Split | Documents | Annotations | Annotations/document | Persons | Uncertain cases | Random gender | Clear-text names |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stats:
        lines.append(
            f"| {row['split']} | {row['documents']} | {row['annotations']} | "
            f"{row['annotations_per_document']} | {row['persons']} | "
            f"{row['uncertain_cases']} | {row['random_gender']} | "
            f"{row['clear_text_names']} |"
        )

    lines += [
        "",
        "## Official Test Coverage",
        "",
        f"- Documents: {len(official_rows)}.",
        f"- Total annotations: {len(official_annotations)}.",
        "",
        "| PII type | Official-test mentions |",
        "|---|---:|",
    ]
    for pii_type in PII_TYPES:
        lines.append(f"| {pii_type} | {official_types.get(pii_type, 0)} |")

    lines += [
        "",
        "The official test contains a malformed leading-space address string in one raw count;",
        "normalized generated counts use exact PII prefixes.",
        "",
        "## PII Subtypes",
        "",
        *subtype_tables,
        "",
        "## Quality Issues",
        "",
        "| Issue | Frequency / evidence | Impact | Current resolution | Next action | Status |",
        "|---|---|---|---|---|---|",
    ]
    for row in issue_rows:
        lines.append(
            f"| {row['issue']} | {row['frequency']}. {row['evidence']} | "
            f"{row['impact']} | {row['current_resolution']} | "
            f"{row['next_action']} | {row['status']} |"
        )

    lines += [
        "",
        "## Case-Rule Findings",
        "",
        "- Rules with `pymorphy3` achieved 72% accuracy on the 359-name gold set.",
        "- Accuracy reached 83% on confident cases.",
        "- Lapa one-gap label accuracy was 16%; Lapa choice accuracy was 31%.",
        "- The current uncertain-case policy defaults to nominative.",
        "- Feminine accusative, dative, and masculine accusative/genitive ambiguity remain weak areas.",
        "",
        "## Address and Identifier Findings",
        "",
        "- Address generation is designed to use the document region for 85% of addresses.",
        "- Addresses in the same sentence share a building where the surrounding text indicates co-location.",
        "- РНОКПП/IPН values use valid checksum logic.",
        "- IBAN values use valid mod-97 logic.",
        "- Dates are generated in multiple formats.",
        "",
        "## Required Improvements",
        "",
        "1. Measure address-region consistency from generated addresses.",
        "2. Complete 15–20 dev-document manual reviews.",
        "3. Expand name, patronymic, and surname coverage.",
        "4. Improve clear-text surname-only and first-name-only name detection.",
        "5. Add per-subtype precision and recall after baseline training.",
        "6. Keep official-test annotations locked from training and tuning.",
        "",
        "## Machine-Readable Outputs",
        "",
        "- `data/synthetic/quality_report.csv`: aggregate split statistics.",
        "- `data/synthetic/quality_issues.csv`: issue, frequency, evidence, and remediation.",
    ]

    report_path = ROOT / "docs" / "data-quality.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {stats_csv}")
    print(f"Wrote {issues_csv}")


if __name__ == "__main__":
    main()
