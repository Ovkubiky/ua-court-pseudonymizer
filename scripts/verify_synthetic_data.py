"""Verify synthetic split structure, offsets, and basic anti-leakage constraints."""

from __future__ import annotations

import json
import sys
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
SPLITS = ("train", "dev", "test_a", "test_b")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise AssertionError(f"Missing required file: {path}")

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise AssertionError(
                        f"Invalid JSON in {path} at line {line_number}: {error}"
                    ) from error
    return rows


def verify_annotation_offsets(row: dict) -> None:
    text = row["original"]
    for annotation in row["annotations"]:
        start = annotation["start"]
        end = annotation["end"]
        pseudonym = annotation["pseudonym"]

        if not isinstance(start, int) or not isinstance(end, int):
            raise AssertionError(f"Non-integer offsets for document {row['id']}")
        if not (0 <= start < end <= len(text)):
            raise AssertionError(
                f"Invalid offset [{start}, {end}) in document {row['id']}"
            )
        if not any(
            pseudonym.startswith(prefix + "_")
            for prefix in ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")
        ):
            raise AssertionError(
                f"Invalid pseudonym {pseudonym!r} in document {row['id']}"
            )

        span = text[start:end]
        if not span.strip():
            raise AssertionError(
                f"Empty or whitespace-only span in document {row['id']}"
            )


def verify_name_pool_separation() -> None:
    declension_path = ROOT / "notebooks" / "resources" / "name_declensions.csv"
    if not declension_path.exists():
        raise AssertionError(f"Missing name-declension table: {declension_path}")

    # Read the actual name inventory once. The generator uses even/odd row
    # positions per gender, so reconstruct that rule and prove the pools differ.
    first_names = {"ч": [], "ж": []}
    patronymics = {"ч": [], "ж": []}
    surnames = {"ч": [], "ж": []}
    with declension_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["тип"] == "ім'я":
                first_names[row["стать"]].append(row["називний"].strip())
            elif row["тип"] == "по батькові":
                patronymics[row["стать"]].append(row["називний"].strip())
            elif row["тип"] == "прізвище":
                surnames[row["стать"]].append(row["називний"].strip())

    train_names = {
        name
        for gender_names in first_names.values()
        for name in gender_names[0::2]
    }
    eval_names = {
        name
        for gender_names in first_names.values()
        for name in gender_names[1::2]
    }
    overlap = train_names & eval_names
    if overlap:
        raise AssertionError(
            f"Train/evaluation first-name pools overlap: {sorted(overlap)}"
        )

    train_surnames = {
        name
        for gender_names in surnames.values()
        for name in gender_names[0::2]
    }
    eval_surnames = {
        name
        for gender_names in surnames.values()
        for name in gender_names[1::2]
    }
    overlap = train_surnames & eval_surnames
    if overlap:
        raise AssertionError(
            f"Train/evaluation surname pools overlap: {sorted(overlap)}"
        )

    # Patronymics are not currently split in make_person(), so they are allowed
    # to overlap; this is reported rather than silently misrepresenting it.
    patronymic_overlap = {
        name
        for gender_names in patronymics.values()
        for name in gender_names
    }

    print(
        f"name pools: train={len(train_names)}, eval={len(eval_names)}, "
        "no first-name overlap"
    )
    print(
        f"surname pools: train={len(train_surnames)}, eval={len(eval_surnames)}, "
        "no surname overlap"
    )
    print(
        f"patronymics: {len(patronymic_overlap)} unique nominative forms; "
        "the current generator does not separate patronymic pools"
    )


def verify_pseudonym_consistency(row: dict) -> None:
    mentions: dict[str, list[dict]] = {}
    for annotation in row["annotations"]:
        if annotation["pseudonym"].startswith(("ОСОБА_", "АДРЕСА_", "НОМЕР_", "ІНФОРМАЦІЯ_")):
            mentions.setdefault(annotation["pseudonym"], []).append(annotation)

    # Same logical pseudonym should keep the same category inside one document.
    for pseudonym, annotations in mentions.items():
        categories = {annotation["pseudonym"].split("_", 1)[0] for annotation in annotations}
        if len(categories) != 1:
            raise AssertionError(
                f"Inconsistent categories for pseudonym {pseudonym} "
                f"in document {row['id']}"
            )

    ordered = sorted(
        (annotation["start"], annotation["end"], annotation["pseudonym"])
        for annotation in row["annotations"]
    )
    for previous, current in zip(ordered, ordered[1:]):
        if previous[1] > current[0]:
            raise AssertionError(
                f"Overlapping annotation spans in document {row['id']}: "
                f"{previous} and {current}"
            )


def main() -> None:
    split_ids: dict[str, set[str]] = {}

    for split in SPLITS:
        rows = load_jsonl(SYNTHETIC_DIR / f"{split}.jsonl")
        ids = {str(row["id"]) for row in rows}
        if len(ids) != len(rows):
            raise AssertionError(f"Duplicate document IDs in {split}")
        split_ids[split] = ids

        for row in rows:
            if not isinstance(row["original"], str):
                raise AssertionError(f"Missing original text in document {row['id']}")
            if not isinstance(row["annotations"], list):
                raise AssertionError(f"Missing annotations in document {row['id']}")
            verify_annotation_offsets(row)
            verify_pseudonym_consistency(row)

        print(f"{split}: {len(rows)} documents")

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise AssertionError(
                    f"Document overlap between {left} and {right}: "
                    f"{len(overlap)} IDs"
                )

    splits_path = SYNTHETIC_DIR / "splits.csv"
    if splits_path.exists():
        import pandas as pd

        splits = pd.read_csv(splits_path, dtype={"id": str})
        csv_ids = set(splits["id"].astype(str))
        jsonl_ids = set().union(*split_ids.values())
        if csv_ids != jsonl_ids:
            raise AssertionError(
                "splits.csv IDs differ from generated JSONL document IDs"
            )

        print(f"splits.csv: {len(splits)} split assignments")

    sys.path.insert(0, str(ROOT / "data"))
    import evaluate

    for split in SPLITS:
        rows = load_jsonl(SYNTHETIC_DIR / f"{split}.jsonl")
        score = evaluate.evaluate(rows, rows)["final_score"]
        if score != 1.0:
            raise AssertionError(f"Self-check failed for {split}: {score}")
        print(f"{split}: self-check {score:.1f}")

    official_test = ROOT / "data" / "test" / "court_cases.jsonl"
    if official_test.exists():
        official_rows = load_jsonl(official_test)
        official_ids = {str(row["id"]) for row in official_rows}
        for split, ids in split_ids.items():
            overlap = ids & official_ids
            if overlap:
                raise AssertionError(
                    f"{split} overlaps the official test on {len(overlap)} IDs"
                )
        print("official test: no document-ID overlap")

    verify_name_pool_separation()

    print("Synthetic-data verification passed.")


if __name__ == "__main__":
    main()
