# Interim Presentation Evidence

## Purpose

This file collects the factual evidence, slide structure, figures, limitations, and
team handoff notes needed for the interim PDF presentation.

## Presentation Metadata

- Project title: Ukrainian Court-Decision Pseudonymizer
- Submission format: one PDF per team
- Required sections:
  1. Problem understanding
  2. Data understanding
  3. Model training or equivalent process
  4. Automatic evaluation
  5. Team contributions

## 1. Problem Understanding

### Problem

Court decisions contain natural-person PII that must be removed before publication,
without removing more information than necessary.

### Proposed solution

Identify four PII categories and deterministically replace recurring entities with
codes:

- `ОСОБА_*`
- `АДРЕСА_*`
- `НОМЕР_*`
- `ІНФОРМАЦІЯ_*`

### Input

The original text field of a court decision.

### Output

One JSONL record per document with character offsets and pseudonyms:

```json
{
  "id": 116075957,
  "annotations": [
    {
      "start": 19,
      "end": 42,
      "pseudonym": "ОСОБА_1"
    }
  ]
}
```

### Constraints

- Open models and libraries.
- Local execution.
- No paid external inference API.
- CPU only.
- 8 GB RAM maximum.
- Complete official test in no more than two hours.
- Offline deployment without runtime data downloads.

### Scope boundaries

In scope:

- Ukrainian court decisions.
- Four PII categories.
- Deterministic entity consistency within a document.
- Court, synthetic, and internal evaluation data.

Out of scope:

- Court IDs and cadastral numbers unless tied to a natural person.
- Public office names that are deliberately not pseudonymized in the register.
- Paid cloud inference.
- Enterprise-scale scrapers or live register crawling.

### Evidence status

| Item | Status | Evidence |
|---|---|---|
| Problem and motivation | Complete | README and task description |
| Input and output format | Complete | `data/evaluate.py` and README |
| Technical constraints | Complete | README and task description |
| Scope boundaries | Complete | README |

## 2. Data Understanding

### Official test data

Source: `data/test/court_cases.jsonl`

| Item | Count |
|---|---:|
| Documents | 240 |
| PII entities | 3,685 |
| `ОСОБА_*` | 2,837 |
| `НОМЕР_*` | 447 |
| `АДРЕСА_*` | 233 |
| `ІНФОРМАЦІЯ_*` | 168 |
| Unique persons | 864 |

The official test is mostly a 97.5% Supreme Court cassation/court domain, while
training covers broader court types and instances.

### Training source

Source: `data/train/2024-court-decisions.parquet`

| Item | Count |
|---|---:|
| Court decisions | 44,758 |
| Documents in the filtered pool | 37,577 |
| Fields | `id`, `text` |

The source is pseudonymized and contains placeholder values rather than existing
NER annotations.

### Synthetic split sizes

| Split | Documents | Annotations | Annotations/document |
|---|---:|---:|---:|
| train | 4,895 | 103,482 | 21.14 |
| dev | 300 | 6,258 | 20.86 |
| test_a | 526 | 9,649 | 18.34 |
| test_b | 377 | 14,505 | 38.47 |

### Synthetic split purpose

- `train`: model fitting and large-context coverage.
- `dev`: development experiments and manual review.
- `test_a`: stratified broad internal test.
- `test_b`: rare-subtype stress test.

### Representative findings

- 48% of official-test persons appear in 2–5 surface forms in one document.
- `ІНФОРМАЦІЯ` is dominated by birth dates and unit/institution names.
- `НОМЕР` includes IBAN, phone, IPN, passport, plate, certificate, and unit numbers.
- Real public names remain clear in 64% of training documents.
- The official test is small for `ІНФОРМАЦІЯ`; subtype confidence intervals are wide.

### Evidence status

| Item | Status | Evidence |
|---|---|---|
| Data sizes | Complete | README, quality report, notebook |
| Entity distributions | Complete | quality report, enhanced EDA |
| Domain shift | Complete | data statistics and quality report |
| Synthetic methodology | Complete | README and DECISION_LOG |
| Representative examples | Prepared | review sample and dev JSONL |
| Manual linguistic review | In progress | 20-document review sample |

## 3. Model Training or Equivalent Process

### Current stage

No final NER model or fine-tuning run has been completed yet.

Completed equivalent process:

- Deterministic name-case analysis.
- Lapa vs rules experiment.
- Synthetic data generation.
- Clear-text name annotation.
- Train/dev/internal-test construction.

### Case-generation experiment

| Method | Accuracy |
|---|---:|
| Lapa: name the case | 16% |
| Lapa: choose the form | 31% |
| Rules + pymorphy3 | 72% |
| Rules on confident gaps | 83% |

The project selected rules plus `pymorphy3`, not Lapa, because:

- rules are offline;
- responses are deterministic;
- dependency on an external API is avoided;
- the API variant had unacceptable failure and 429/timeout behavior.

### Known current quality limitations

| Issue | Frequency | Status |
|---|---:|---|
| Uncertain person cases | 19.7% | Known limitation |
| Random gender assignment | 17.8% | Known limitation |
| Clear-text name annotations | 16,010 mentions | Partial |
| Address-region mismatch | Not yet measured | Planned |
| Limited name inventory | Approximately 24 names / 32 surnames per gender | Planned |

### Planned baselines

Priority order:

1. Rule/dictionary baseline.
2. `uk_ner_web_trf_13class` or `uk_ner_web_trf_base`.
3. `xlm-roberta-base` token classification.
4. Optional spaCy/Stanza comparison.

### Required experiments

- Train/dev split only.
- Ablation: nominative names versus rule-inflected names.
- Ablation: with/without clear-text names.
- Full official test evaluation only after model selection.
- CPU, RAM, and runtime benchmarking.

### Evidence status

| Item | Status | Next step |
|---|---|---|
| Baseline choice | In progress | Implement rule baseline |
| Training | Not started | Prepare training format and baseline |
| Fine-tuning | Not started | Run bounded CPU experiment |
| Ablations | Planned | Execute after baseline |
| Resource validation | Planned | Measure runtime/RAM |

## 4. Automatic Evaluation

### Metrics

Primary metric:

- Relaxed NER macro F1 over four PII types.

Additional metrics:

- Strict NER macro F1.
- Per-type precision, recall, and F1.
- Deanonymization risk by name.
- Consistency B3 F1.
- Cosine similarity with
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

### Match rules

- Entity must have the correct PII type.
- Relaxed match: IoU at least 0.5.
- Strict match: exact character boundaries.
- Each gold entity is matched at most once.

### Evaluation data

- Official test: 240 documents.
- Internal dev: 300 documents.
- Internal test_a: 526 documents.
- Internal test_b: 377 documents.

### Current result

All synthetic splits score 1.0 against themselves.

This result proves only:

- JSON format;
- offsets;
- annotation/entity consistency;
- evaluator compatibility.

It does **not** prove model quality.

### Evaluation artifacts

- `data/evaluate.py`
- `data/sample_submission.jsonl`
- `scripts/verify_synthetic_data.py`
- `QUALITY_REPORT.md`

### Evidence status

| Item | Status |
|---|---|
| Evaluator installed | Complete |
| Submission format implemented | Complete |
| Format self-check | Complete |
| Model baseline evaluation | Not started |
| Strict and relaxed model F1 | Not started |
| Deanonymization risk | Not started |
| Consistency | Not started |
| Cosine similarity | Not started |

## 5. Manual Review Preparation

### Prepared review sample

The 20-document dev sample is in:

- `MANUAL_REVIEW.md`
- `data/synthetic/manual_review_sample.csv`

The sample is automatically risk-ranked using:

- uncertain grammatical cases;
- random gender assignments;
- clear-text names;
- rare subtypes;
- addresses.

### Honest review status

No human verdicts have been fabricated. The sample is prepared, but final document
review must be completed by a human reviewer.

### Structural pre-review

The sample is covered by automated verification for:

- valid JSON;
- valid offsets;
- non-empty spans;
- valid PII prefixes;
- no overlap;
- no split overlap;
- no official-test overlap;
- no train/eval first-name overlap.

## 6. Anti-Leakage Evidence

### Document isolation

Verified:

- train versus dev;
- train versus test_a;
- train versus test_b;
- dev versus test_a;
- dev versus test_b;
- test_a versus test_b;
- all synthetic splits versus official test.

### Name isolation

Verified:

- 24 train first names;
- 24 evaluation first names;
- no first-name overlap.

### Test-only fields

The official test fields `pseudonymized` and `annotations` are excluded from synthetic
training generation and model tuning.

### Verification command

```powershell
.venv\Scripts\python scripts\verify_synthetic_data.py
```

## 7. Slide Deck Structure

Suggested slides:

1. Title and team.
2. Problem, motivation, and scope.
3. Input/output example.
4. Official test statistics.
5. Training corpus and synthetic generation process.
6. Train/dev/test_a/test_b split explanation.
7. Representative synthetic document example.
8. Name case/gender generation and gold-set result.
9. Lapa vs rules experiment.
10. Manual review and quality limitations.
11. Model baseline plan.
12. Evaluation methodology.
13. Completed/in-progress/planned table.
14. Team contributions.

## 8. Completed / In Progress / Planned

| Work area | Completed | In progress | Planned |
|---|---|---|---|
| Problem | Scope, output, constraints | — | — |
| Data | Source analysis, synthetic pipeline | Manual review | Name expansion |
| EDA | Basic and enhanced EDA | Address consistency | Informal-text set |
| Training | Case/gender experiment | Baseline selection | Fine-tuning |
| Evaluation | Evaluator and format checks | Metric plan | Actual model results |
| Deployment | — | — | Docker/offline app |

## 9. Team-Contribution Table Template

| Team member | Interim responsibilities | Evidence |
|---|---|---|
| Member 1 | Problem, README, synthetic data, EDA, quality report | This repository |
| Member 2 | NER baseline and training experiments | To be added |
| Member 3 | Evaluation, Docker, presentation, team handoff | To be added |

Add actual names and concrete deliverables before submission.

## 10. Presentation Claims to Avoid

Do not claim:

- "our model achieved 1.0";
- "we completed automatic evaluation";
- "the synthetic dataset is fully quality-verified";
- "our 20-document sample was manually reviewed";
- "addresses are region-consistent" unless measured;
- "the system is complete and deployable".

Use claims such as:

- "format and offsets are verified";
- "a manual-review sample has been prepared";
- "a deterministic rule baseline was chosen";
- "actual model evaluation remains to be run";
- "the system is designed for offline CPU use";
- "the project has identified and documented known limitations".
