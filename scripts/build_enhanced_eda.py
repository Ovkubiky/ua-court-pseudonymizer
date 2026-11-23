"""Create an English enhanced-EDA notebook for court PII data."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "enhanced_eda.ipynb"


def markdown(text: str) -> dict:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(text)


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python (nlp-anonymization)",
            "language": "python",
            "name": "nlp-anonymization",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }

    notebook["cells"] = [
        markdown(
            "# Enhanced Exploratory Data Analysis\n\n"
            "This notebook analyzes the official test set and synthetic train/dev/test splits. "
            "Official-test annotations are used only for descriptive evaluation and reporting."
        ),
        code(
            'import json\n'
            'from collections import Counter\n'
            'from pathlib import Path\n'
            'import pandas as pd\n'
            'import matplotlib.pyplot as plt\n'
            '\n'
            'ROOT = Path("..").resolve()\n'
            'SYNTHETIC = ROOT / "data" / "synthetic"\n'
            'PII_TYPES = ["ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ"]\n'
            'SPLITS = ["train", "dev", "test_a", "test_b"]\n'
            '\n'
            'def load_jsonl(path):\n'
            '    with path.open(encoding="utf-8") as f:\n'
            '        return [json.loads(line) for line in f if line.strip()]\n'
        ),
        markdown("## 1. Split and PII summary"),
        code(
            'summary = []\n'
            'records = {}\n'
            'for split in SPLITS:\n'
            '    rows = load_jsonl(SYNTHETIC / f"{split}.jsonl")\n'
            '    records[split] = rows\n'
            '    anns = [a for r in rows for a in r["annotations"]]\n'
            '    types = Counter(a["pseudonym"].split("_", 1)[0] for a in anns)\n'
            '    persons = [a for a in anns if a["pseudonym"].startswith("ОСОБА_")]\n'
            '    summary.append({\n'
            '        "split": split,\n'
            '        "documents": len(rows),\n'
            '        "annotations": len(anns),\n'
            '        "annotations_per_document": round(len(anns) / len(rows), 2),\n'
            '        "median_document_length": int(pd.Series([len(r["original"]) for r in rows]).median()),\n'
            '        "ОСОБА": types["ОСОБА"],\n'
            '        "АДРЕСА": types["АДРЕСА"],\n'
            '        "НОМЕР": types["НОМЕР"],\n'
            '        "ІНФОРМАЦІЯ": types["ІНФОРМАЦІЯ"],\n'
            '        "uncertain_cases": sum(a.get("case_confident") is False for a in persons),\n'
            '        "random_gender": sum(a.get("gender_from_context") is False for a in persons),\n'
            '        "clear_text_names": sum(a.get("subtype") == "clear_name" for a in anns),\n'
            '    })\n'
            'summary_df = pd.DataFrame(summary)\n'
            'summary_df'
        ),
        markdown("## 2. PII composition by split"),
        code(
            'plot_df = summary_df.set_index("split")[PII_TYPES]\n'
            'ax = plot_df.plot(kind="bar", figsize=(10, 5), rot=0)\n'
            'ax.set_title("PII mentions by synthetic split")\n'
            'ax.set_ylabel("Mentions")\n'
            'plt.tight_layout()\n'
        ),
        markdown("## 3. Document length and annotations"),
        code(
            'length_stats = []\n'
            'for split, rows in records.items():\n'
            '    lengths = pd.Series([len(r["original"]) for r in rows])\n'
            '    length_stats.append({\n'
            '        "split": split,\n'
            '        "mean": round(lengths.mean(), 1),\n'
            '        "median": int(lengths.median()),\n'
            '        "q25": int(lengths.quantile(.25)),\n'
            '        "q75": int(lengths.quantile(.75)),\n'
            '        "max": int(lengths.max()),\n'
            '    })\n'
            'pd.DataFrame(length_stats)'
        ),
        markdown("## 4. Person case and gender quality"),
        code(
            'case_rows = []\n'
            'for split, rows in records.items():\n'
            '    persons = [a for r in rows for a in r["annotations"] if a["pseudonym"].startswith("ОСОБА_")]\n'
            '    cases = Counter(a.get("case") or "clear_name_or_missing" for a in persons)\n'
            '    case_rows.append({"split": split, **dict(cases)})\n'
            'case_df = pd.DataFrame(case_rows).fillna(0).set_index("split")\n'
            'ax = case_df.plot(kind="bar", stacked=True, figsize=(12, 5), rot=0)\n'
            'ax.set_title("Person mentions by grammatical case")\n'
            'ax.set_ylabel("Mentions")\n'
            'plt.tight_layout()\n'
        ),
        markdown("## 5. PII subtypes"),
        code(
            'subtype_rows = []\n'
            'for split, rows in records.items():\n'
            '    counts = Counter(a.get("subtype", "missing") for r in rows for a in r["annotations"] if a.get("subtype"))\n'
            '    subtype_rows.append({"split": split, **dict(counts)})\n'
            'subtype_df = pd.DataFrame(subtype_rows).fillna(0).set_index("split")\n'
            'subtype_df.T\n'
        ),
        markdown("## 6. Rare-subtype coverage"),
        code(
            'rare = subtype_df.loc[["test_b", "train"]].T\n'
            'rare[rare.max(axis=1) >= 50].sort_values("test_b", ascending=False).head(20)\n'
        ),
        markdown("## 7. Official-test distribution"),
        code(
            'official = load_jsonl(ROOT / "data" / "test" / "court_cases.jsonl")\n'
            'official_anns = [a for r in official for a in r["annotations"]]\n'
            'official_types = Counter(a["pseudonym"].split("_", 1)[0].strip() for a in official_anns)\n'
            'pd.DataFrame({"PII type": PII_TYPES, "mentions": [official_types[t] for t in PII_TYPES]})\n'
        ),
        markdown("## 8. Domain shift"),
        code(
            'domain_df = pd.DataFrame([\n'
            '    {"dataset": "synthetic_train", "documents": 4895, "median_chars": summary_df.loc[summary_df.split == "train", "median_document_length"].iloc[0]},\n'
            '    {"dataset": "synthetic_dev", "documents": 300, "median_chars": summary_df.loc[summary_df.split == "dev", "median_document_length"].iloc[0]},\n'
            '    {"dataset": "official_test", "documents": len(official), "median_chars": int(pd.Series([len(r["original"]) for r in official]).median())},\n'
            '])\n'
            'domain_df\n'
        ),
        markdown(
            "## 9. Findings\n\n"
            "- Synthetic test_b is intentionally enriched for rare subtypes.\n"
            "- Official test is a separate domain and must not be used for model selection or tuning.\n"
            "- Person-case confidence and gender-from-context flags identify likely quality weak points.\n"
            "- Clear-text names are additional person annotations and require a separate error analysis.\n"
            "- Address-region consistency, linked person identifiers, and informal-language coverage remain to be measured."
        ),
    ]

    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(f"Wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()
