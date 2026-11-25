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
  reference test set.

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

### Reference test data

`data/test/court_cases.jsonl` contains 240 annotated decisions. It has:

- 3,685 annotated entities;
- 2,837 `ОСОБА_*` entities;
- 447 `НОМЕР_*` entities;
- 233 `АДРЕСА_*` entities;
- 168 `ІНФОРМАЦІЯ_*` entities;
- 864 unique persons.

The reference test fields are:

- `id`: the Unified State Register case identifier;
- `pseudonymized`: a version from the register;
- `original`: a synthetic imitation of the original text;
- `annotations`: character-level PII spans;
- `annotations_by_pseudonym`: the same annotations grouped by pseudonym.

The file `data/evaluate.py` implements the reference scoring:

- NER relaxed macro F1;
- NER strict macro F1;
- deanonymization risk by name;
- consistency B3 F1;
- optional cosine similarity.

The `pseudonymized` and `annotations` fields must remain test-only. They must never be
used for model training or parameter selection.


Two earlier versions of the test exist (`6.10.2026_court_cases.jsonl`, `8.10.2026_court_cases.jsonl`).
`court_cases.jsonl` is the latest one (a few annotation fixes); always evaluate on it.

### Training data

The court corpus is `data/train/2024-court-decisions.parquet`. It
contains 44,758 deduplicated Ukrainian court decisions from 2024. The texts contain
placeholders instead of PII, for example `ОСОБА_1`, `АДРЕСА_2`, and `НОМЕР_1`. It is
converted into labelled synthetic data by `notebooks/generate_data.ipynb`.

The public NER datasets (NER-UK 2.0, Ukr-Synth,
Polyglot-NER, WikiAnn) are not used yet.

### External resources

| Resource | Used for | Licence |
|---|---|---|
| [`lang-uk/name_freq_dict_uk`](https://github.com/lang-uk/name_freq_dict_uk) | first names, patronymics and surnames of the v2 generator | GPL-3.0 |
| clear-text organisation / court addresses in the training corpus | street and settlement names of the v2 generator | — |

Only street and settlement *names* are mined, never full addresses, and only from documents
that are not in our dev / test splits.

## Getting the Data

Git holds code, hand-made resources and documentation only. Everything that a command can regenerate
(synthetic data, generator resources, predictions, model checkpoints) lives in the private HF dataset repo
[`ua-court-pseudonymizer/ua-court-pseudonymizer-artifacts`](https://huggingface.co/datasets/ua-court-pseudonymizer/ua-court-pseudonymizer-artifacts):

```bash
hf auth login
hf download ua-court-pseudonymizer/ua-court-pseudonymizer-artifacts --repo-type dataset --local-dir artifacts
mkdir -p data/synthetic data/synthetic_v1
cp artifacts/synthetic/v2/* data/synthetic/ && cp artifacts/synthetic/v1/* data/synthetic_v1/
cp artifacts/resources/*.csv notebooks/resources/
```

Upload new checkpoints / predictions to `models/<name>/` and `preds/<name>/` of the same repo.
The reference test and the court-decision parquet are not stored there; download them from the source dataset folder:

```bash
# test set, evaluator and training corpus (Google Drive folder)
uvx gdown --folder "https://drive.google.com/drive/folders/1T7Ub_6xuS5zwP5LioX745dnfQ5JrGkEg" -O data
# name dictionary for the v2 generator
git clone --depth 1 https://github.com/lang-uk/name_freq_dict_uk data/external/name_freq_dict_uk
```

Expected layout:

```text
data/train/2024-court-decisions.parquet
data/test/court_cases.jsonl
data/evaluate.py
data/sample_submission.jsonl
data/external/name_freq_dict_uk/
```

## Synthetic Data Generation

`notebooks/generate_data.ipynb` creates (document IDs fixed by `data/synthetic/splits.csv`):

| Dataset | Documents | Purpose |
|---|---:|---|
| `train` | 4,895 | Model training |
| `dev` | 300 | Checkpoint selection, rule design, manual inspection |
| `test_a` | 526 | Internal stratified test set |
| `test_b` | 377 | Rare-subtype test set |

Generation is deterministic: each document is seeded by its ID, so the same code produces
the same files byte for byte. The pipeline:

1. fills placeholders with fictional names and values;
2. applies case and gender rules to Ukrainian person names (`pymorphy3` + context rules);
3. labels clear-text names that remain in the original training data;
4. generates valid dates, IBANs (mod-97), РНОКПП (checksum), addresses, and IDs;
5. writes character-level annotations and checks them with the reference evaluator.

### Data versions

The notebook setting `DATA_VERSION` selects the generator.

| | v1 (`data/synthetic_v1/`) | v2 (`data/synthetic/`, default) |
|---|---|---|
| Surnames / first names per pool | 20 / 24 | 1,255 / 274 (train), 1,223 / 242 (eval) |
| Patronymics | 32 | 868 |
| Name surface forms | full, `Прізвище І.П.`, `І.П. Прізвище` | + `Прізвище Ім'я`, `Ім'я Прізвище` |
| Streets | 15 | 3,527 mined, with street type (вул., просп., пров., бульв., пл., …) |
| Address shapes | 3 street styles × 3 orders | ~12 shapes, house numbers `12-А`, `12/3`, village + district + oblast, settlement-only addresses after `уродженця`, `жителя`, `в`, `у` |
| `ІНФОРМАЦІЯ` | dates, 8 unit names, e-mail, URL, nick | + company names after `ТОВ / ПрАТ / АТ «…»`, titles after `під назвою «…»`, generated units / ТЦК names, logins; unknown subtype is no longer always a birth date |
| Dates / phones / passports | 4 / 3 / 2 formats | 8 / 8 / 5 formats; `року` stays outside the span when the text already has it |

Train and eval name pools are split by a hash of the lemma; `scripts/verify_synthetic_data.py`
proves that they do not overlap. Rebuild the v2 resources with:

```bash
python scripts/build_resources.py   # writes notebooks/resources/{name_lexicon,streets,settlements}.csv
```

The inflected lexicon is automatic (pymorphy3), not manually reviewed like
`name_declensions.csv`; surnames that pymorphy3 cannot inflect for both genders are dropped.

## Known Data Findings

- 48% of reference-test persons appear in two to five surface forms in one document.
- `ІНФОРМАЦІЯ` is mostly birth dates and institution/unit names.
- `НОМЕР` includes phones, РНОКПП values, license plates, IBANs, and passports.
- Real judge, clerk, lawyer, and notary names remain in some training documents.
- The register also leaves some personal e-mails and РНОКПП unmasked; the synthetic gold
  misses them, so a few "false positives" on synthetic data are real PII.
- The reference test is mostly Supreme Court cassation decisions, while training spans
  multiple courts and proceeding types.
- About 25% of synthetic name-case decisions are not confident and fall back to the
  nominative case. Feminine accusative names are the weakest case.
- Clear-text surname-only or first-name-only names are not fully detected.

Enhanced exploratory analysis: [`notebooks/enhanced_eda.ipynb`](notebooks/enhanced_eda.ipynb).
Data quality report: [`docs/data-quality.md`](docs/data-quality.md) (written for v1).

## Models

### Regex baseline

[`scripts/regex_baseline.py`](scripts/regex_baseline.py): regular expressions plus the
reviewed name table; no model. Results in [`docs/baselines/regex.md`](docs/baselines/regex.md).
Reference relaxed macro F1 0.4845. **Caveat:** the "improved" regex rules were compared on the
reference test, so that number is exploratory, not a clean held-out score.

### Neural NER baseline

[`scripts/ner_baseline.py`](scripts/ner_baseline.py): `FacebookAI/xlm-roberta-base` with a
BIO token-classification head, trained on v1 synthetic data (RTX 4090, 26 min).
Results in [`docs/baselines/ner.md`](docs/baselines/ner.md).

| Reference test (single final run) | P | R | F1 |
|---|---:|---:|---:|
| ОСОБА | 0.991 | 0.931 | 0.960 |
| АДРЕСА | 0.879 | 0.468 | 0.611 |
| НОМЕР | 0.927 | 0.772 | 0.843 |
| ІНФОРМАЦІЯ | 0.946 | 0.417 | 0.579 |
| **Macro** | | | **0.748** (strict 0.663) |

CPU inference on the reference test: 21 min, 1.7 GB RSS.

The trained weights are **not** in git. Checkpoints go to `models/` of the HF artifacts repo;
place the selected one in `baselines/ner_xlm_roberta/selected/` (or pass `--checkpoint`).

### Hybrid layer

[`scripts/hybrid.py`](scripts/hybrid.py) post-processes the model spans:

1. **boundary repair**: spans are grown to whole words, stripped of punctuation and
   quotes, the final dot is kept only for initials / abbreviations, address pieces split by
   `, ` and name pieces split by a space are joined;
2. **rules** for fixed-shape PII: IBAN (mod-97; State Treasury and court / prosecutor
   accounts excluded), РНОКПП (keyword or checksum), phones (court / fax / office numbers
   excluded), passports and ID cards (keyword), plates, military-unit numbers, personal
   e-mails (role mailboxes and legal-entity addresses excluded), social-network URLs,
   `@nicknames`, birth / death dates (keyword or directly after a person);
3. **merge**: a rule span replaces overlapping model spans, except that persons and
   addresses keep digits inside them.

Rules were designed on train / dev data only. Rule precision (IoU ≥ 0.5) on the internal
sets: РНОКПП 0.97–0.99, IBAN 1.00, plates 0.98–1.00 (0.72 on dev, where fleet lists of
company vehicles are unlabelled), phones 0.90–1.00, passports 1.00.

`ner_baseline.py infer` applies the hybrid layer by default (`--no-hybrid` switches it off)
and can save the raw model spans (`--raw-output`) for ablations:

```bash
python scripts/hybrid_eval.py rules --truth data/synthetic/dev.jsonl
python scripts/hybrid_eval.py ablation --truth data/synthetic/dev.jsonl --raw dev.raw.jsonl
```

`hybrid_eval.py` refuses the reference test unless `--final` is passed.

## Setup

macOS / Linux:

```bash
python -m venv .venv            # or: uv venv --python 3.12 .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name nlp-anonymization --display-name "Python (nlp-anonymization)"
```

Windows PowerShell: the same with `.venv\Scripts\python`.

## Reproducing Everything

```bash
python scripts/build_resources.py                       # v2 resources (needs data/external/name_freq_dict_uk)
cd notebooks && python -m jupyter nbconvert --to notebook --execute generate_data.ipynb --inplace && cd ..
python scripts/verify_synthetic_data.py                 # offsets, self-check 1.0, split / name-pool separation
python -m pytest -q tests

# training (GPU; --device mps works on Apple Silicon but is slow)
python scripts/ner_baseline.py train --data-dir data/synthetic --output-dir baselines/ner_xlm_roberta_v2 \
  --seed 42 --epochs 3 --batch-size 8 --gradient-accumulation 2 --learning-rate 2e-5 \
  --max-length 256 --stride 128

# internal evaluation (raw spans saved for the hybrid ablation)
python scripts/ner_baseline.py infer --split dev --checkpoint baselines/ner_xlm_roberta_v2/selected \
  --output preds/dev.jsonl --raw-output preds/dev.raw.jsonl
python scripts/hybrid_eval.py ablation --truth data/synthetic/dev.jsonl --raw preds/dev.raw.jsonl

# reference test: once, after all choices are fixed
python scripts/ner_baseline.py infer --split reference --cpu --offline \
  --checkpoint baselines/ner_xlm_roberta_v2/selected --output preds/reference.jsonl
python data/evaluate.py data/test/court_cases.jsonl preds/reference.jsonl
```

Manual review UI: `cd notebooks && python review_app.py`, then open `http://localhost:8765`.

Rebuild the EDA notebook and the quality report:

```bash
python scripts/build_enhanced_eda.py
python scripts/build_quality_report.py
```

## Current Implementation Status

Done:

- data-source analysis and train / test EDA;
- synthetic data v1 and v2 (larger name lexicon, mined addresses, more formats);
- case and gender rules, clear-text name labelling, address-region handling;
- self-check against `data/evaluate.py`, split and name-pool separation checks;
- manual review of 20 dev documents ([`docs/manual-review.md`](docs/manual-review.md));
- regex baseline;
- neural NER baseline (XLM-R base, v1 data): reference relaxed F1 0.748;
- hybrid layer: rules for structured PII and span-boundary repair;
- CPU / RAM / time check of the neural baseline (21 min, 1.7 GB).

Next:

- retrain XLM-R on v2 data; compare v1-trained vs v2-trained on both internal test versions;
- hybrid ablation on dev / test_a / test_b with the new model;
- person-mention linking (`Петренко О.О.` ↔ `Петренка Олега Олеговича`) to cut
  deanonymization risk and raise consistency;
- cosine-similarity evaluation (`data/evaluate.py --cosine`);
- a single CLI + Docker image with the model inside (offline);
- public NER datasets / other models (`uk_ner_web_trf_13class`, spaCy, Stanza) as comparisons;
- choose an open licence and add a `LICENSE` file;
- final submission and presentation.

## Evaluation

The self-score of every synthetic set against itself must be 1.0. This proves offsets
and output format, not model quality.

The primary ranking metric is relaxed NER macro F1 over the four PII categories.
Strict F1, deanonymization risk, consistency, and cosine similarity are also available
from `data/evaluate.py`:

```bash
python data/evaluate.py data/test/court_cases.jsonl submission.jsonl [--cosine]
```

## Anti-Leakage Rules

- Never use reference-test `annotations` or `pseudonymized` text for training, rule design or tuning.
- Never tune hyperparameters, thresholds or rules on the reference test.
- Keep all splits disjoint by document ID; reuse `data/synthetic/splits.csv`.
- Keep synthetic test names outside the synthetic training name pool.
- Report the reference-test result only after model selection and training are fixed.

## Repository Layout

| Path | Purpose |
|---|---|
| `docs/DECISION_LOG.md` | Decisions, experiments, and synthetic-data methodology |
| `docs/data-quality.md`, `docs/manual-review.md` | Synthetic-data quality report and manual review |
| `docs/baselines/` | Regex and NER baseline reports |
| `docs/presentation-notes.md` | Interim presentation facts and slide outline |
| `notebooks/data_stats.ipynb` | Exploratory data analysis |
| `notebooks/generate_data.ipynb` | Synthetic training-data pipeline (`DATA_VERSION` 1 / 2) |
| `notebooks/synth_utils.py` | Generation and name-inflection logic |
| `notebooks/resources/name_declensions.csv` | Reviewed Ukrainian name forms (v1 + v2) |
| `notebooks/resources/name_lexicon.csv`, `streets.csv`, `settlements.csv` | v2 resources, built by `scripts/build_resources.py` (not in git) |
| `notebooks/case_eval/` | Real-name gold set and Lapa comparison |
| `notebooks/review_app.py` | Local manual-review application |
| `scripts/build_resources.py` | Builds the v2 resources |
| `scripts/regex_baseline.py` | Regex baseline |
| `scripts/ner_baseline.py` | XLM-R training / inference / evaluation |
| `scripts/hybrid.py`, `scripts/hybrid_eval.py` | Hybrid layer and its evaluation |
| `scripts/verify_synthetic_data.py` | Structural and anti-leakage checks |
| `scripts/vast_train.sh` | One-command GPU training and artifact upload (vast.ai) |
| `tests/` | Unit tests for the hybrid layer and NER baseline (`python -m pytest -q tests`) |
| `data/evaluate.py` | Reference evaluator (not in git) |

## License

No licence has been chosen yet; the solution is to be published under an open
licence, so a `LICENSE` file must be added before the final release. Note that
`lang-uk/name_freq_dict_uk` is GPL-3.0 and dataset licences must be listed in the release.
