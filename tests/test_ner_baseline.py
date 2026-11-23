import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from transformers import AutoTokenizer

import ner_baseline as baseline


def test_alignment_special_tokens_and_offsets():
    tokenizer = AutoTokenizer.from_pretrained(
        baseline.MODEL_ID, use_fast=True
    )
    text = "Іван живе на вул. Хрещатик."
    encoded = tokenizer(text, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    assert len(offsets) == len(encoded["input_ids"])
    assert offsets[0] == (0, 0)
    assert offsets[-1] == (0, 0)


def test_bio_decoding():
    offsets = [(0, 4), (5, 9), (10, 13), (13, 14)]
    tags = ["B-ОСОБА", "I-ОСОБА", "B-НОМЕР", "O"]
    spans = baseline.decode_bio(tags, offsets)
    assert spans == [(0, 9, "ОСОБА"), (10, 13, "НОМЕР")]


def test_invalid_and_overlapping_annotations():
    text = "Іван живе в Києві."
    rows = [{"id": "1", "original": text, "annotations": [
        {"start": 0, "end": 4, "pseudonym": "ОСОБА_1"},
        {"start": 3, "end": 9, "pseudonym": "АДРЕСА_1"},
    ]}]
    try:
        baseline.validate_rows(rows, Path("test.jsonl"))
    except AssertionError as error:
        assert "overlapping" in str(error)
    else:
        raise AssertionError("overlapping annotations were not rejected")

    invalid = [{"id": "2", "original": text, "annotations": [
        {"start": 0, "end": 100, "pseudonym": "ОСОБА_1"}
    ]}]
    try:
        baseline.validate_rows(invalid, Path("test.jsonl"))
    except AssertionError as error:
        assert "invalid span" in str(error)
    else:
        raise AssertionError("invalid span was not rejected")
