"""First pretrained neural NER baseline for Ukrainian court PII."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hybrid  # noqa: E402  (rules + boundary repair on top of the model)


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "FacebookAI/xlm-roberta-base"
CATEGORIES = ("ОСОБА", "АДРЕСА", "НОМЕР", "ІНФОРМАЦІЯ")
LABELS = ["O"]
for category in CATEGORIES:
    LABELS.extend((f"B-{category}", f"I-{category}"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise AssertionError(f"Missing required file: {path}")
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AssertionError(f"Invalid JSON in {path}:{line_number}: {error}") from error
            rows.append(row)
    return rows


def validate_rows(rows: Iterable[dict], source: Path, require_annotations: bool = True) -> None:
    seen_ids = set()
    for row in rows:
        doc_id = row.get("id")
        text = row.get("original")
        if doc_id is None:
            raise AssertionError(f"{source}: missing document id")
        if str(doc_id) in seen_ids:
            raise AssertionError(f"{source}: duplicate document id {doc_id}")
        seen_ids.add(str(doc_id))
        if not isinstance(text, str):
            raise AssertionError(f"{source}: document {doc_id} has no string original text")
        anns = row.get("annotations") if require_annotations else None
        if require_annotations:
            if not isinstance(anns, list):
                raise AssertionError(f"{source}: document {doc_id} annotations must be a list")
            ordered = []
            for annotation in anns:
                if not isinstance(annotation, dict):
                    raise AssertionError(f"{source}: document {doc_id} has malformed annotation")
                start = annotation.get("start")
                end = annotation.get("end")
                pseudonym = annotation.get("pseudonym")
                if not isinstance(start, int) or not isinstance(end, int):
                    raise AssertionError(f"{source}: document {doc_id} has non-integer offsets")
                if not (0 <= start < end <= len(text)):
                    raise AssertionError(
                        f"{source}: document {doc_id} has invalid span [{start}, {end}) "
                        f"for text length {len(text)}"
                    )
                if text[start:end].strip() == "":
                    raise AssertionError(f"{source}: document {doc_id} has whitespace-only span")
                if not isinstance(pseudonym, str):
                    raise AssertionError(f"{source}: document {doc_id} has no pseudonym")
                category = pseudonym.strip().replace("I", "І").split("_", 1)[0].strip()
                if category not in CATEGORIES:
                    raise AssertionError(
                        f"{source}: document {doc_id} has unsupported label {pseudonym!r}"
                    )
                ordered.append((start, end, category))
            ordered.sort()
            for previous, current in zip(ordered, ordered[1:]):
                if previous[1] > current[0]:
                    raise AssertionError(
                        f"{source}: overlapping annotations in document {doc_id}: "
                        f"{previous[:2]} and {current[:2]}"
                    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


class WindowDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int, stride: int, label2id: dict[str, int]):
        self.samples = []
        for row in rows:
            text = row["original"]
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt",
            )
            for window_index in range(encoded["input_ids"].shape[0]):
                offsets = encoded["offset_mapping"][window_index].tolist()
                labels = [label2id["O"]] * len(offsets)
                for annotation in row["annotations"]:
                    span_start = annotation["start"]
                    span_end = annotation["end"]
                    category = annotation["pseudonym"].split("_", 1)[0].strip()
                    started = False
                    for token_index, (token_start, token_end) in enumerate(offsets):
                        if token_start == token_end:
                            continue
                        if token_end <= span_start or token_start >= span_end:
                            continue
                        prefix = "B" if not started else "I"
                        labels[token_index] = label2id[f"{prefix}-{category}"]
                        started = True
                attention = encoded["attention_mask"][window_index].tolist()
                sample = {
                    "input_ids": encoded["input_ids"][window_index],
                    "attention_mask": torch.tensor(attention),
                    "labels": torch.tensor(labels),
                }
                if "token_type_ids" in encoded:
                    sample["token_type_ids"] = encoded["token_type_ids"][window_index]
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def decode_bio(tags: list[str], offsets: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    spans = []
    current_category = None
    current_start = None
    current_end = None
    for tag, (token_start, token_end) in zip(tags, offsets):
        if token_start == token_end:
            continue
        if tag.startswith("B-"):
            if current_category is not None:
                spans.append((current_start, current_end, current_category))
            current_category = tag[2:]
            current_start = token_start
            current_end = token_end
        elif tag.startswith("I-") and current_category == tag[2:] and current_start is not None:
            current_end = max(current_end, token_end)
        elif tag.startswith("I-"):  # I- without a B- (window start, model slip): start a span instead of dropping it
            if current_category is not None:
                spans.append((current_start, current_end, current_category))
            current_category, current_start, current_end = tag[2:], token_start, token_end
        else:
            if current_category is not None:
                spans.append((current_start, current_end, current_category))
            current_category = None
            current_start = None
            current_end = None
    if current_category is not None:
        spans.append((current_start, current_end, current_category))
    return spans


def merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    merged = []
    for start, end, category in sorted(spans):
        if merged and merged[-1][2] == category and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]), category)
        else:
            merged.append((start, end, category))
    return sorted(merged)


def predict_rows(model, tokenizer, rows: list[dict], max_length: int, stride: int,
                 batch_size: int, device: torch.device, use_hybrid: bool = False,
                 raw_sink: list | None = None) -> list[dict]:
    model.eval()
    id2label = {index: label for index, label in enumerate(LABELS)}
    predictions = []
    with torch.inference_mode():
        for row in rows:
            text = row["original"]
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt",
            )
            window_spans = []
            for start in range(0, encoded["input_ids"].shape[0], batch_size):
                stop = min(start + batch_size, encoded["input_ids"].shape[0])
                inputs = {
                    "input_ids": encoded["input_ids"][start:stop].to(device),
                    "attention_mask": encoded["attention_mask"][start:stop].to(device),
                }
                if "token_type_ids" in encoded:
                    inputs["token_type_ids"] = encoded["token_type_ids"][start:stop].to(device)
                logits = model(**inputs).logits
                predicted = logits.argmax(dim=-1).cpu().tolist()
                for relative, tags in enumerate(predicted):
                    offsets = encoded["offset_mapping"][start + relative].tolist()
                    window_spans.extend(
                        decode_bio([id2label[tag] for tag in tags], offsets)
                    )
            spans = merge_spans(window_spans)
            if raw_sink is not None:
                raw_sink.append({"id": row["id"], "spans": [list(span) for span in spans]})
            if use_hybrid:
                spans = hybrid.combine(text, spans)
            annotations = assign_pseudonyms(text, spans)
            predictions.append({"id": row["id"], "annotations": annotations})
    return predictions


def load_surname_map() -> dict[str, str]:
    result = {}
    path = ROOT / "notebooks" / "resources" / "name_declensions.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["тип"] != "прізвище":
                continue
            nominative = row["називний"].strip().lower()
            for column in ("називний", "родовий", "давальний", "знахідний", "орудний", "місцевий", "кличний"):
                form = row[column].strip().lower()
                if form:
                    result[form] = nominative
    return result


SURNAME_MAP = load_surname_map()


def normalized_surface(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def person_key(surface: str) -> str:
    words = re.findall(r"[А-ЯІЇЄҐа-яіїєґ'’ʼ\-]+", surface.lower())
    for word in words:
        if word in SURNAME_MAP:
            return "surname:" + SURNAME_MAP[word]
    return "surface:" + normalized_surface(surface).replace(" ", "")


def assign_pseudonyms(text: str, spans: list[tuple[int, int, str]]) -> list[dict]:
    counters = {category: 0 for category in CATEGORIES}
    keys = {category: {} for category in CATEGORIES}
    annotations = []
    for start, end, category in spans:
        if start < 0 or end > len(text) or start >= end:
            raise AssertionError(
                f"Decoded invalid span [{start}, {end}) for text length {len(text)}"
            )
        surface = text[start:end]
        if not surface.strip():
            raise AssertionError(f"Decoded whitespace-only span [{start}, {end})")
        key = person_key(surface) if category == "ОСОБА" else "surface:" + normalized_surface(surface)
        if key not in keys[category]:
            counters[category] += 1
            keys[category][key] = counters[category]
        annotations.append({
            "start": start,
            "end": end,
            "pseudonym": f"{category}_{keys[category][key]}",
        })
    return sorted(annotations, key=lambda item: (item["start"], item["end"], item["pseudonym"]))


def evaluate_predictions(truth_path: Path, prediction_rows: list[dict], cosine: bool = False) -> dict:
    spec = importlib.util.spec_from_file_location("court_evaluator", ROOT / "data" / "evaluate.py")
    evaluator = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["court_evaluator"] = evaluator
    spec.loader.exec_module(evaluator)
    truth = load_jsonl(truth_path)
    return evaluator.evaluate(truth, prediction_rows, evaluator.cosine_model if cosine else None)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def hardware_info() -> dict:
    info = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_vram_bytes": properties.total_memory,
        })
    return info


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"
    train_rows = load_jsonl(train_path)
    dev_rows = load_jsonl(dev_path)
    if args.max_train_docs:
        train_rows = random.Random(args.seed).sample(train_rows, min(args.max_train_docs, len(train_rows)))
    if args.max_dev_docs:
        dev_rows = dev_rows[:args.max_dev_docs]
        dev_path = Path(str(dev_path) + f".first{args.max_dev_docs}")
        write_jsonl(dev_path, dev_rows)
    validate_rows(train_rows, train_path)
    validate_rows(dev_rows, dev_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if not tokenizer.is_fast:
        raise AssertionError("A fast tokenizer is required for offset mappings")
    label2id = {label: index for index, label in enumerate(LABELS)}
    id2label = {index: label for label, index in label2id.items()}
    train_dataset = WindowDataset(train_rows, tokenizer, args.max_length, args.stride, label2id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )
    device = pick_device(args.device)
    model.to(device)

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    started = time.perf_counter()
    peak_allocated = 0
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            inputs = {key: value.to(device) for key, value in batch.items()}
            loss = model(**inputs).loss
            (loss / args.gradient_accumulation).backward()
            epoch_loss += float(loss.detach().cpu())
            if step % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            if args.log_every and step % args.log_every == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={epoch_loss / step:.4f} "
                      f"elapsed={time.perf_counter() - started:.0f}s", flush=True)
        if len(train_loader) % args.gradient_accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        predictions = predict_rows(
            model, tokenizer, dev_rows, args.max_length, args.stride,
            args.inference_batch_size, device,
        )
        metrics = evaluate_predictions(dev_path, predictions)
        epoch_dir = output_dir / f"epoch-{epoch}"
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        write_jsonl(output_dir / f"dev_epoch_{epoch}.jsonl", predictions)
        record = {
            "epoch": epoch,
            "training_loss": epoch_loss / len(train_loader),
            "dev_metrics": metrics,
        }
        history.append(record)
        if best is None or (
            metrics["final_score"] > best["metrics"]["final_score"]
            or (
                metrics["final_score"] == best["metrics"]["final_score"]
                and metrics["ner_strict"]["macro"]["f1"] > best["metrics"]["ner_strict"]["macro"]["f1"]
            )
        ):
            best = {"epoch": epoch, "dir": epoch_dir, "metrics": metrics}
            model.save_pretrained(output_dir / "selected")
            tokenizer.save_pretrained(output_dir / "selected")
            write_jsonl(output_dir / "dev_predictions.jsonl", predictions)
        (output_dir / "training_log.json").write_text(
            json.dumps({"history": history, "best_epoch": best["epoch"]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"epoch={epoch} loss={record['training_loss']:.6f} "
              f"dev_relaxed_f1={metrics['final_score']:.4f}", flush=True)

    duration = time.perf_counter() - started
    config = {key: value for key, value in vars(args).items() if key != "func"}
    config.update({
        "model_id": MODEL_ID,
        "label_mapping": label2id,
        "seed": args.seed,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "hardware": hardware_info(),
        "gpu_peak_allocated_bytes": peak_allocated,
        "training_duration_seconds": duration,
        "best_epoch_selection": "maximum dev relaxed macro F1, then dev strict macro F1",
        "selected_epoch": best["epoch"],
        "selected_metrics": best["metrics"],
    })
    (output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "label_mapping.json").write_text(
        json.dumps({"labels": LABELS, "label2id": label2id, "id2label": id2label}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"selected_epoch": best["epoch"], "selected_metrics": best["metrics"]},
                     ensure_ascii=False))


def infer(args: argparse.Namespace) -> None:
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    split_paths = {
        "train": data_dir / "train.jsonl",
        "dev": data_dir / "dev.jsonl",
        "test_a": data_dir / "test_a.jsonl",
        "test_b": data_dir / "test_b.jsonl",
        "official": ROOT / "data" / "test" / "court_cases.jsonl",
    }
    if args.split not in split_paths:
        raise AssertionError(f"Unknown split: {args.split}")
    truth_path = split_paths[args.split]
    rows = load_jsonl(truth_path)
    validate_rows(rows, truth_path, require_annotations=args.split != "official")
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, local_files_only=args.offline)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint, local_files_only=args.offline)
    device = torch.device("cpu") if args.cpu else pick_device(args.device)
    model.to(device)
    tracemalloc.start()
    started = time.perf_counter()
    raw: list = []
    predictions = predict_rows(
        model, tokenizer, rows, args.max_length, args.stride, args.inference_batch_size, device,
        use_hybrid=args.hybrid, raw_sink=raw,
    )
    duration = time.perf_counter() - started
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    write_jsonl(output, predictions)
    if args.raw_output:
        write_jsonl(args.raw_output if args.raw_output.is_absolute() else ROOT / args.raw_output, raw)
    print(json.dumps({
        "hybrid": args.hybrid,
        "split": args.split,
        "document_count": len(predictions),
        "output": str(output),
        "device": str(device),
        "inference_duration_seconds": duration,
        "traced_peak_bytes": traced_peak,
    }, ensure_ascii=False))


def evaluate(args: argparse.Namespace) -> None:
    prediction_path = args.predictions
    if not prediction_path.is_absolute():
        prediction_path = ROOT / prediction_path
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    truth_paths = {
        "dev": data_dir / "dev.jsonl",
        "test_a": data_dir / "test_a.jsonl",
        "test_b": data_dir / "test_b.jsonl",
        "official": ROOT / "data" / "test" / "court_cases.jsonl",
    }
    truth = truth_paths[args.split]
    predictions = load_jsonl(prediction_path)
    validate_rows(
        [{"id": row["id"], "original": row_text["original"], "annotations": []} for row_text in []],
        truth,
        require_annotations=False,
    ) if False else None
    truth_rows = load_jsonl(truth)
    if args.split != "official":
        validate_rows(truth_rows, truth)
    original_by_id = {str(row["id"]): row["original"] for row in truth_rows}
    for prediction in predictions:
        doc_id = str(prediction["id"])
        if doc_id not in original_by_id:
            raise AssertionError(f"Prediction contains unknown document id {doc_id}")
        text = original_by_id[doc_id]
        for annotation in prediction["annotations"]:
            start, end = annotation["start"], annotation["end"]
            if not (0 <= start < end <= len(text)):
                raise AssertionError(f"Prediction id={doc_id} has invalid span [{start}, {end})")
            category = annotation["pseudonym"].split("_", 1)[0].strip()
            if category not in CATEGORIES:
                raise AssertionError(f"Prediction id={doc_id} has invalid category")

    metrics = evaluate_predictions(truth, predictions)
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and select the best dev checkpoint")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--gradient-accumulation", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--max-length", type=int, default=256)
    train_parser.add_argument("--stride", type=int, default=128)
    train_parser.add_argument("--inference-batch-size", type=int, default=16)
    train_parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"), help="synthetic data version to train on")
    train_parser.add_argument("--output-dir", type=Path, default=Path("baselines/ner_xlm_roberta"))
    train_parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    train_parser.add_argument("--max-train-docs", type=int, default=0, help="random subset of train (smoke runs)")
    train_parser.add_argument("--max-dev-docs", type=int, default=0, help="first N dev documents for epoch selection")
    train_parser.add_argument("--log-every", type=int, default=0)
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser("infer", help="Write JSONL predictions for a split")
    infer_parser.add_argument("--split", choices=("train", "dev", "test_a", "test_b", "official"), required=True)
    infer_parser.add_argument("--checkpoint", type=Path, default=Path("baselines/ner_xlm_roberta/selected"))
    infer_parser.add_argument("--output", type=Path, required=True)
    infer_parser.add_argument("--max-length", type=int, default=256)
    infer_parser.add_argument("--stride", type=int, default=128)
    infer_parser.add_argument("--inference-batch-size", type=int, default=16)
    infer_parser.add_argument("--cpu", action="store_true")
    infer_parser.add_argument("--offline", action="store_true")
    infer_parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu (--cpu overrides)")
    infer_parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    infer_parser.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=True,
                              help="apply scripts/hybrid.py (boundary repair + rules); --no-hybrid = model only")
    infer_parser.add_argument("--raw-output", type=Path, default=None,
                              help="also write the raw model spans (for scripts/hybrid_eval.py ablation)")
    infer_parser.set_defaults(func=infer)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate JSONL predictions with data/evaluate.py")
    evaluate_parser.add_argument("--split", choices=("dev", "test_a", "test_b", "official"), required=True)
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
