# Ukrainian Court-Decision Pseudonymizer

## Project Goal

This project builds a local pseudonymizer for Ukrainian court decisions. It identifies
personal data and replaces every occurrence of the same real entity with a deterministic
code in the form `ОСОБА_1`, `АДРЕСА_1`, `НОМЕР_1`, or `ІНФОРМАЦІЯ_1`.

The project is motivated by the conflict between public access to court decisions and
the need to remove personal data before publication. The solution must preserve useful
court information while masking every identifying detail of natural persons.

## Scope

### In scope

- Ukrainian court-decision text.
- Four PII categories:
  - `ОСОБА_*`: a natural person's name.
  - `АДРЕСА_*`: a residential or registration address.
  - `НОМЕР_*`: identifiers tied to a natural person, such as an РНОКПП/IPН, passport,
    ID-card number, phone number, IBAN, license plate, or military-unit identifier.
  - `ІНФОРМАЦІЯ_*`: birth or death dates, email addresses, URLs, social-network
    nicknames, and military-unit or institution names.
- Deterministic replacement within one decision.
- Open models and libraries only.
- Fully local CPU inference.
- Offline execution after setup.
- Deployment and execution within 8 GB RAM and a 2-hour limit for the 240-document
  official test set.

### Out of scope

- Cloud or paid inference APIs.
- Scraping or publishing data from the Unified State Register of Court Decisions.
- Automatic anonymization of information that is not personal data.
- Court numbers, cadastral numbers, government-account numbers, contracts, and
  institution identifiers when they do not identify a natural person.
- The public names of judges or public officials that the project must not remove.

## Problem Formulation

```text
Court decision text
        |
        v
Personal-data recognizer
        |
        v
Character spans + PII category + entity pseudonym
        |
        v
Pseudonymized text
```

The recognizer receives the `original` field of a court decision and returns character
offsets in JSON Lines format:

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

## Data

### Official test data

`data/test/court_cases.jsonl` contains 240 annotated decisions. It has:

- 3,685 annotated entities;
- 2,837 `ОСОБА_*` entities;
- 447 `НОМЕР_*` entities;
- 233 `АДРЕСА_*` entities;
- 168 `ІНФОРМАЦІЯ_*` entities;
- 864 unique persons.

The official test fields are:

- `id`: the Unified State Register case identifier;
- `pseudonymized`: a version from the register;
- `original`: a synthetic imitation of the original text;
- `annotations`: character-level PII spans;
- `annotations_by_pseudonym`: the same annotations grouped by pseudonym.

The file `data/evaluate.py` implements the task's official scoring:

- NER relaxed macro F1;
- NER strict macro F1;
- deanonymization risk by name;
- consistency B3 F1;
- optional cosine similarity.

The `pseudonymized` and `annotations` fields must remain test-only. They must never be
used for model training or parameter selection.

### Training data

The organizer-provided court corpus is
`data/train/2024-court-decisions.parquet`. It contains 44,758 deduplicated Ukrainian
court decisions from 2024. The texts contain placeholders instead of PII, for example
`ОСОБА_1`, `АДРЕСА_2`, and `НОМЕР_1`.

The corpus is not yet the final model-training set. It is converted into labelled
synthetic data by `notebooks/generate_data.ipynb`.

The following public NER datasets are proposed in the task description but are not yet
used in the current implementation:

- NER-UK 2.0;
- Ukr-Synth;
- Polyglot-NER;
- WikiAnn.

## Synthetic Data Generation

`notebooks/generate_data.ipynb` creates:

| Dataset | Documents | Purpose |
|---|---:|---|
| `train` | 4,895 | Model training |
| `dev` | 300 | Development and manual inspection |
| `test_a` | 526 | Internal stratified test set |
| `test_b` | 377 | Rare-subtype test set |

Each document is assigned to exactly one split. The split is keyed by document ID,
and each generated document is reproducible using a seed derived from its ID.

The synthetic pipeline:

1. fills placeholders with fictional names and values;
2. applies case and gender rules to Ukrainian person names;
3. inflects names using `pymorphy3`;
4. labels clear-text names that remain in the original training data;
5. generates valid dates, IBANs, РНОКПП values, addresses, and IDs;
6. writes character-level annotations;
7. checks all JSONL rows with the official evaluator format.

The test-side name lists are separate from the train-side name lists, so a test name
does not appear in the synthetic training set.

## Known Data Findings

- 48% of official-test persons appear in two to five surface forms in one document.
- `ІНФОРМАЦІЯ` is mostly birth dates and institution/unit names.
- `НОМЕР` includes phones, РНОКПП values, license plates, IBANs, and passports.
- Real judge, clerk, lawyer, and notary names remain in some training documents.
- The official test is mostly Supreme Court cassation decisions, while training spans
  multiple courts and proceeding types.
- About 25% of synthetic name-case decisions are not confident and fall back to the
  nominative case.
- Feminine accusative names are the weakest case for the current rules.
- Clear-text surname-only or first-name-only names are not fully detected.

These findings must be reported as limitations, not as if they were solved.

Enhanced exploratory analysis is available in
[`notebooks/enhanced_eda.ipynb`](/D:/УКУ/4%20курс/NLP/project/ua-court-pseudonymizer/notebooks/enhanced_eda.ipynb).
The generated quality assessment is documented in
[`QUALITY_REPORT.md`](/D:/УКУ/4%20курс/NLP/project/ua-court-pseudonymizer/QUALITY_REPORT.md),
with machine-readable summary and issue tables in `data/synthetic/quality_report.csv`
and `data/synthetic/quality_issues.csv`.

## Regex Baseline

A deterministic rule-only baseline is implemented in
[`scripts/regex_baseline.py`](/D:/УКУ/4%20курс/NLP/project/ua-court-pseudonymizer/scripts/regex_baseline.py).
It uses regular expressions, the reviewed Ukrainian name table, and no model,
GPU, API, or network access.

Run all internal splits and the official test:

```powershell
.venv\Scripts\python scripts\regex_baseline.py
```

Run only the official test:

```powershell
.venv\Scripts\python scripts\regex_baseline.py --official-only
```

The results and submission files are written to `baselines/`:

- `REGEX_BASELINE.md`: summary and interpretation;
- `regex_baseline_summary.json`: compact metrics;
- `regex_baseline_results.json`: full per-type metrics;
- `regex_<split>_submission.jsonl`: predicted annotations.

## Reproducing Synthetic Data

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name nlp-anonymization --display-name "Python (nlp-anonymization)"
```

Place the organizer files in:

```text
data/train/2024-court-decisions.parquet
data/test/court_cases.jsonl
data/evaluate.py
```

Open `notebooks/generate_data.ipynb`, select the `nlp-anonymization` kernel, and run
all cells. The notebook writes:

```text
data/synthetic/splits.csv
data/synthetic/train.jsonl
data/synthetic/dev.jsonl
data/synthetic/test_a.jsonl
data/synthetic/test_b.jsonl
```

Review notes are optional and are created only when a reviewer saves a decision
through `notebooks/review_app.py`.

Run the structural and anti-leakage verification:

```powershell
.venv\Scripts\python scripts\verify_synthetic_data.py
```

To launch the manual review tool:

```powershell
cd notebooks
..\.venv\Scripts\python review_app.py
```

Open `http://localhost:8765`.

To rebuild the enhanced EDA notebook and quality report:

```powershell
.venv\Scripts\python scripts\build_enhanced_eda.py
.venv\Scripts\python scripts\build_quality_report.py

cd notebooks
..\.venv\Scripts\python -m jupyter nbconvert --execute enhanced_eda.ipynb --to notebook --inplace --ExecutePreprocessor.timeout=600
```

## Current Implementation Status

Completed:

- data-source analysis;
- train/test EDA;
- synthetic split construction;
- placeholder replacement;
- name case and gender rules;
- clear-text name labelling;
- address-region handling;
- self-check against `data/evaluate.py`;
- a local manual-review UI.

In progress:

- manual verification of synthetic documents;
- review of uncertain declensions;
- quality analysis of rare subtypes.

Remaining:

- baseline NER implementation;
- model fine-tuning;
- actual strict and relaxed F1 measurements;
- deanonymization-risk and consistency measurements;
- cosine-similarity evaluation;
- CPU/RAM/time validation;
- Docker or another offline deployment method;
- final submission and interim PDF presentation.

## Evaluation

The self-score of every synthetic set against itself must be 1.0. This proves offsets
and output format, not model quality.

The primary ranking metric is relaxed NER macro F1 over the four PII categories.
Strict F1, deanonymization risk, consistency, and cosine similarity are also available
from `data/evaluate.py`.

Example:

```powershell
python data/evaluate.py data/test/court_cases.jsonl submission.jsonl
```

## Anti-Leakage Rules

- Never use official-test `annotations` for training or tuning.
- Never use official-test `pseudonymized` text as training labels.
- Never tune hyperparameters on the official test.
- Keep all splits disjoint by document ID.
- Keep synthetic test names outside the synthetic training name pool.
- Reuse `data/synthetic/splits.csv` unless an explicit, documented split rebuild is needed.
- Report the official-test result only after model selection and training are fixed.

## Repository Layout

| Path | Purpose |
|---|---|
| `DECISION_LOG.md` | Decisions, experiments, and synthetic-data methodology |
| `notebooks/data_stats.ipynb` | Exploratory data analysis |
| `notebooks/generate_data.ipynb` | Synthetic training-data pipeline |
| `notebooks/synth_utils.py` | Generation and name-inflection logic |
| `notebooks/resources/name_declensions.csv` | Reviewed Ukrainian name forms |
| `notebooks/case_eval/` | Real-name gold set and Lapa comparison |
| `notebooks/review_app.py` | Local manual-review application |
| `data/evaluate.py` | Official task evaluator |
| `PRESENTATION_EVIDENCE.md` | Interim presentation facts and slide outline |
| `MANUAL_REVIEW.md` | Manual-review scope and selected dev sample |
| `data/synthetic/manual_review_sample.csv` | Machine-readable manual-review checklist |

## License

The project code must remain open and licenced for reuse. Dataset licences and personal
data-handling restrictions must be included before final release.
