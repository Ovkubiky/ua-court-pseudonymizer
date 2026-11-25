# Neural NER Baseline

## Scope and model

This baseline uses `FacebookAI/xlm-roberta-base` with a token-classification head
and the BIO labels `O`, `B/I-ОСОБА`, `B/I-АДРЕСА`, `B/I-НОМЕР`, and
`B/I-ІНФОРМАЦІЯ`. It predicts character spans from the `original` field and assigns
deterministic document-level pseudonyms.

Implementation: `scripts/ner_baseline.py`; focused tests:
`tests/test_ner_baseline.py`.

## Data and leakage controls

The actual synthetic JSONL schema contains `id`, `original`, `annotations`, and
`meta`. Each annotation has character offsets, `entity`, `pseudonym`, and subtype
metadata. All 0-based annotation spans were validated as in range, non-empty,
non-overlapping, and category-compatible.

Document counts:

| Split | Documents |
|---|---:|
| train | 4,895 |
| dev | 300 |
| test_a | 526 |
| test_b | 377 |
| reference | 240 |

Only `train.jsonl` supplied gradient updates. Only `dev.jsonl` was used for
checkpoint selection. `test_a` and `test_b` were evaluated only after training
and selection were fixed. The reference file was used for one final fixed-model
evaluation after all internal evaluations. No reference `pseudonymized`,
`annotations`, or `annotations_by_pseudonym` field was used as model input.

The synthetic verifier passed for all splits, including duplicate-ID checks,
offset checks, overlap checks, split separation, and gold self-checks.

## Training setup and hardware

Hardware: NVIDIA GeForce RTX 4090, 49,866,487,296 bytes of VRAM.

Software:

| Component | Version |
|---|---|
| Python | 3.12.16 |
| PyTorch | 2.15.0+cu132 |
| Transformers | 4.57.6 |
| Accelerate | 1.16.0 |
| Tokenizers | 0.22.2 |

Training configuration:

| Parameter | Value |
|---|---:|
| Seed | 42 |
| Epochs | 3 |
| Batch size | 8 |
| Gradient accumulation | 2 |
| Effective batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Maximum sequence length | 256 |
| Stride | 128 |
| Inference batch size | 16 |
| Gradient clipping | 1.0 |

Selection rule: maximum dev relaxed macro F1, with strict dev macro F1 as a
tie-breaker. Epoch 2 was selected. Training took 1,585.38 seconds
(26.42 minutes). PyTorch peak allocated GPU memory was 5,760,739,328 bytes
(5.36 GiB), as reported by `torch.cuda.max_memory_allocated()`; this is a
device allocator peak, not an independent process-wide measurement.

The model revision/hash was not captured in machine-readable configuration.
The exact pretrained identifier was captured as
`FacebookAI/xlm-roberta-base`.

## Results by split

| Split | Relaxed macro F1 | Strict macro F1 | Deanonymization risk | Consistency B³ |
|---|---:|---:|---:|---:|
| dev | 0.9860 | 0.9777 | 0.0136 | 0.8706 |
| test_a | 0.9665 | 0.9598 | 0.1754 | 0.9291 |
| test_b | 0.9681 | 0.9627 | 0.1883 | 0.9119 |
| reference | 0.7479 | 0.6632 | 0.1543 | 0.8185 |

Raw evaluator JSON files are under `baselines/ner_xlm_roberta/metrics/`.

## Per-category results

Relaxed NER results:

| Split | Category | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| dev | ОСОБА | 0.9730 | 0.9921 | 0.9824 |
| dev | АДРЕСА | 0.9880 | 1.0000 | 0.9940 |
| dev | НОМЕР | 0.9804 | 0.9890 | 0.9847 |
| dev | ІНФОРМАЦІЯ | 0.9757 | 0.9901 | 0.9829 |
| test_a | ОСОБА | 0.9438 | 0.9056 | 0.9243 |
| test_a | АДРЕСА | 0.9853 | 0.9958 | 0.9905 |
| test_a | НОМЕР | 0.9371 | 0.9864 | 0.9611 |
| test_a | ІНФОРМАЦІЯ | 0.9804 | 1.0000 | 0.9901 |
| test_b | ОСОБА | 0.9558 | 0.8908 | 0.9222 |
| test_b | АДРЕСА | 0.9929 | 1.0000 | 0.9965 |
| test_b | НОМЕР | 0.9749 | 0.9592 | 0.9670 |
| test_b | ІНФОРМАЦІЯ | 0.9882 | 0.9852 | 0.9867 |
| reference | ОСОБА | 0.9906 | 0.9313 | 0.9600 |
| reference | АДРЕСА | 0.8790 | 0.4678 | 0.6106 |
| reference | НОМЕР | 0.9274 | 0.7718 | 0.8425 |
| reference | ІНФОРМАЦІЯ | 0.9459 | 0.4167 | 0.5785 |

## Comparison with regex baseline

The same reference evaluator was used. Cosine similarity was not measured.

| Split | Neural relaxed F1 | Regex relaxed F1 | Neural strict F1 | Regex strict F1 |
|---|---:|---:|---:|---:|
| dev | 0.9860 | 0.4519 | 0.9777 | 0.4006 |
| test_a | 0.9665 | 0.4562 | 0.9598 | 0.3991 |
| test_b | 0.9681 | 0.5226 | 0.9627 | 0.4570 |
| reference | 0.7479 | 0.4845 | 0.6632 | 0.3916 |

The neural model is substantially higher on all evaluated synthetic splits and
also on this reference run. This comparison is not a claim that the synthetic
gain predicts future real-world performance. The regex reference score is
exploratory because regex changes used the reference set. The neural reference
score is a single final result.

## Runtime and memory measurements

Internal inference was run on GPU:

| Split | Documents | Wall time (s) |
|---|---:|---:|
| dev | 300 | 20.25 |
| test_a | 526 | 34.97 |
| test_b | 377 | 37.27 |

Reference inference was run on CPU with `--cpu --offline` and completed in
1,243.19 seconds by the script's inference timer; an outer wall measurement
reported 1,248.47 seconds (20.81 minutes). Linux process child maximum RSS was
1,692,544 KB (approximately 1.69 GB). The script's traced allocator peak was
2,202,469 bytes, which is not process RSS and is not comparable to it.

These measurements cover the 240-document reference set and demonstrate that
this implementation completes within the stated 2-hour limit in this environment.
RSS was measured with `resource.getrusage(RUSAGE_CHILDREN)`,
`ru_maxrss`, on Linux; this is a process-wide high-water mark but may not include
every non-Python resource precisely.

## Error analysis

The largest reference-domain weakness is address and information recall. On the
reference set, address relaxed recall is 0.4678 and information relaxed recall is
0.4167. Person recall remains high at 0.9313.

One reference example was document `116258650`. The model missed several repeated
office/address spans such as:
`по вулиці Великій Перспективній, буд. 56, кв. 14, м. Кропивницький, Кіровоградська область`
and the date `2020-05-13`. This indicates boundary and rare-domain coverage
issues despite strong synthetic performance.

The model also had 133 leaked persons among 862 in the reference deanonymization
metric. The exact linking behavior should therefore be treated cautiously.

## Pseudonym and linking behavior

Identical normalized surface forms receive a stable pseudonym within a document.
For persons, a reviewed surname-declension map links inflected surname forms to
their nominative form; otherwise the model uses a whitespace-normalized surface
key. This is conservative mention linking, not full cross-document entity
resolution. Pseudonyms are numbered deterministically and spans are sorted by
character offset.

The strict reference result remains lower than relaxed F1, especially for addresses
and information, showing that exact character boundaries are a material weakness.

## Limitations

- Synthetic train/dev/test figures are not an untouched real-domain estimate.
- The reference set has a court-domain and writing-style shift from synthetic data.
- Address and institution/date recognition has lower reference recall.
- Person linking uses surname normalization and exact normalized surfaces only.
- Inflected names without an available surname table entry may receive separate
  pseudonyms.
- The reference result is a single final evaluation and must not be used to tune
  the neural model.
- Neural training is not guaranteed to be perfectly deterministic across GPUs.

## Reproduction

```bash
python scripts/verify_synthetic_data.py
python -m pytest -q tests/test_ner_baseline.py
python scripts/ner_baseline.py train --seed 42 --epochs 3 --batch-size 8 \
  --gradient-accumulation 2 --learning-rate 2e-5 --weight-decay 0.01 \
  --max-length 256 --stride 128 --inference-batch-size 16
python scripts/ner_baseline.py infer --split dev \
  --output baselines/ner_xlm_roberta/predictions/dev.jsonl
python scripts/ner_baseline.py infer --split test_a \
  --output baselines/ner_xlm_roberta/predictions/test_a.jsonl
python scripts/ner_baseline.py infer --split test_b \
  --output baselines/ner_xlm_roberta/predictions/test_b.jsonl
python scripts/ner_baseline.py infer --split reference --cpu --offline \
  --output baselines/ner_xlm_roberta/predictions/reference.jsonl
python scripts/ner_baseline.py evaluate --split dev \
  --predictions baselines/ner_xlm_roberta/predictions/dev.jsonl \
  --output baselines/ner_xlm_roberta/metrics/dev.json
python scripts/ner_baseline.py evaluate --split test_a \
  --predictions baselines/ner_xlm_roberta/predictions/test_a.jsonl \
  --output baselines/ner_xlm_roberta/metrics/test_a.json
python scripts/ner_baseline.py evaluate --split test_b \
  --predictions baselines/ner_xlm_roberta/predictions/test_b.jsonl \
  --output baselines/ner_xlm_roberta/metrics/test_b.json
python scripts/ner_baseline.py evaluate --split reference \
  --predictions baselines/ner_xlm_roberta/predictions/reference.jsonl \
  --output baselines/ner_xlm_roberta/metrics/reference.json
```
