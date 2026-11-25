# Decision log: building labelled training data

**Goal.** Pseudonymize Ukrainian court decisions by finding `ОСОБА` / `АДРЕСА` / `НОМЕР` / `ІНФОРМАЦІЯ` spans with open models only, under these limits: CPU, ≤ 8 GB RAM, offline. This is a research project, so we aim at Ukrainian PII recognition in court texts in general, not at fitting the reference test. Results are reported on two test sets: the reference one and our own.

## Repository layout and how to run

| Path | What it is |
|---|---|
| `notebooks/data_stats.ipynb` | Statistics of train vs test; each section ends with a conclusion |
| `notebooks/generate_data.ipynb` | Builds the labelled data: splits → generation → checks → appendix with the case-rule evaluation |
| `notebooks/synth_utils.py` | All generation logic: case and gender rules, name inflection, value generators |
| `notebooks/resources/name_declensions.csv` | Names, patronymics and surnames × 7 cases (from `pymorphy3`, manually reviewed) |
| `notebooks/case_eval/` | Gold set of 359 real names with their case, plus the saved Lapa answers |
| `notebooks/review_app.py` | Local page for reviewing the declension table and the generated documents |

Setup:

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name nlp-anonymization --display-name "Python (nlp-anonymization)"
```

The reference test and training corpus are not in git. Put it under `data/`: `data/train/2024-court-decisions.parquet`, `data/test/court_cases.jsonl` and `data/evaluate.py`. Then run `generate_data.ipynb`. It reuses `data/synthetic/splits.csv` if that file exists and writes `data/synthetic/{train,dev,test_a,test_b}.jsonl` in about 20 s.

## Stage 1: Look at the data (`data_stats.ipynb`)

| | Train | Reference test |
|---|---|---|
| Documents | 44,758 | 240 |
| Labels | none: PII already replaced by placeholders (`ОСОБА_1`, …) | `original` + span annotations (3,685 entities) |
| Domain | all proceeding types and instances | 97.5% Supreme Court cassation, descriptive part only |

Findings that shaped the data:
- **Persons change form.** 48% of test persons appear in 2–5 surface forms within one document: different grammatical cases, full name vs initials.
- **`ІНФОРМАЦІЯ` and `НОМЕР` mix subtypes.** `ІНФОРМАЦІЯ` is mostly birth dates (49%) and unit/institution names (24%). `НОМЕР` covers phones, tax IDs (ІПН), plates, IBANs and more.
- **Real names are left in clear text.** In 64% of train documents, judges, clerks and lawyers are named openly. The goal is to anonymize every natural person, so these names must be labelled too.
- **The reference test is small.** 168 `ІНФОРМАЦІЯ` mentions give a ±7 pp confidence interval, which is too coarse for per-subtype conclusions.

**Decisions:**
- Turn train into labelled data by filling the placeholders with fictional values. The span labels then come for free.
- Sample for breadth: stratify by proceeding type × instance. The domain shift to cassation is reported as a finding, not compensated for.
- Build our own test sets, so we can draw conclusions per stratum and per subtype.

## Stage 2: Splits

The pool is documents of 1k–20k characters (37,577). Documents are split by document id, so no document is in two sets.

| Set | Docs | How it is sampled |
|---|---|---|
| train | 4,895 | 270 per stratum + enrichment for rare subtypes + 500 documents without PII |
| dev | 300 | 20 per stratum |
| test_a | 526 | 35 per stratum (our main test) |
| test_b | 377 | 50 per rare subtype: email, URL, nickname, death date, unit, IBAN, passport, plate |

The test sets use a separate half of the name lists, so names seen in the test sets never appear in train.

## Stage 3: Choosing the grammatical case of each name

Each placeholder needs a name in the right case: «позов **Петренка Олега**», «з **Петренком Олегом**». We tried an LLM first.

**Attempt 1: Lapa (`LapaLLM-Gemma-3-12B-instruct`) annotates whole documents.** It failed:
- 90% of the returned name forms failed validation: the answers echoed the prompt, gave pronouns or invented words;
- the API returned frequent HTTP 429 errors and timeouts.

**Attempt 2: measure on a gold set before choosing a method.** We took 359 real full names from train that the registry left in clear text, about 50 per case. The true case is known from the patronymic. Each name is replaced with `___`, and the method has to guess the case.

| Method | Case accuracy | Notes |
|---|---|---|
| Lapa: "name the case" (one gap per request) | 16% | answers "nominative" 91% of the time |
| Lapa: "pick the correct form" from 12 variants | 31% | gender 53% (chance level) |
| Lapa: the same, 5 gaps per request | 23% | answers 96% of gaps |
| **Rules + `pymorphy3`** | **72%** | **83%** on the 83% of gaps where the rule is confident |

On the 62 gaps where the rules are unsure, Lapa is right 15 times and the rules 11 times. That gain is too small to justify an API dependency.

**Decision: rules only, no LLM.** The rules are deterministic, work offline and take about 20 s for all 6,098 documents. Here is how they work (`synth_utils.rule_case`):
1. **Preposition before the gap.** `до/від/щодо` → genitive; `з` → instrumental, except `стягнути з`, which takes the genitive; `про/на` → accusative.
2. **Role noun before the gap**, even across modifiers (`приватний нотаріус Київського міського нотаріального округу ___`): the name copies the case of the role noun.
3. **Inanimate noun before the gap** (`позов ___`): genitive.
4. **Verb before the gap**: accusative. **Verb after the gap**: nominative.
5. **Enumerations** (`ОСОБА_1, ОСОБА_2 та ___`): the same case as the previous name.
6. **Gender** comes from the verb after the name (`звернулася`), from `який/яка`, and from feminine role nouns.

Accuracy by case: nominative 98%, genitive 83%, instrumental 78%, dative 56–62%, **accusative of feminine names only 18%**. The reason is that masculine role nouns look the same in the genitive and the accusative (`арбітражного керуючого ___`), so the rule cannot tell the two apart.

The inflected forms come from `name_declensions.csv`. It holds 144 rows generated with `pymorphy3`; 33 of them were flagged by the generator and then manually reviewed and approved.

## Stage 4: Clear-text names and other values

- **Clear-text names** (`суддя Цвіра Д.М.`, full names validated by `pymorphy3`) are labelled `ОСОБА`. Different forms of one person get one code: `Цвіра Д.М.` and `суддею Цвірою Д.М.` both become `ОСОБА_3`. These names stay real, because they are public in the registry.
- **Numbers and dates** are generated in the right format for their subtype. Tax IDs (ІПН) get a valid checksum and IBANs a valid mod-97 check digit. Dates come in 4 formats. Every annotation carries a `subtype` field.
- **Addresses: first fix after manual review.** Addresses used to be drawn from random cities, so a Lviv case got Cherkasy, Poltava and Kherson addresses. Now:
  - the document's region is detected from the courts, cities and oblasts named in it (found for 96% of documents);
  - 85% of addresses fall in that region;
  - addresses listed together (`квартир АДРЕСА_2 та АДРЕСА_3`) share a building.

## Result

| Set | Docs | ОСОБА | АДРЕСА | НОМЕР | ІНФОРМАЦІЯ | of which clear-text names | confident cases |
|---|---|---|---|---|---|---|---|
| train | 4,895 | 76,203 | 6,627 | 10,324 | 10,328 | 12,395 | 77% |
| dev | 300 | 5,187 | 412 | 456 | 203 | 1,081 | 65% |
| test_a | 526 | 8,139 | 471 | 589 | 450 | 1,835 | 78% |
| test_b | 377 | 7,987 | 1,124 | 2,426 | 2,968 | 699 | 74% |

The format is the same as the reference test (`id`, `original`, `annotations`). Every set scores 1.0 against itself with `evaluate.py`, which confirms the format and offsets are correct.

## Stage 5: data v2 (more variety)

The v1 generator drew every person from ~24 first names and ~20 surnames per pool, every street from 15 names, and
filled unknown `ІНФОРМАЦІЯ` with a birth date even inside `ПрАТ «…»`. The reference test was built from a large name
dictionary and a postal address base, and the v1 model's recall there was 0.47 (АДРЕСА) and 0.42 (ІНФОРМАЦІЯ).

**Decisions (`DATA_VERSION = 2`, same documents and splits as v1):**
- **Names** from `lang-uk/name_freq_dict_uk`, inflected by `pymorphy3` (`scripts/build_resources.py`): 2,467 surnames,
  512 first names, 864 patronymics; sampled by sqrt(frequency) capped at the 90th percentile so politicians at the
  head of the list do not dominate. Train / eval pools split by a hash of the lemma (no overlap, checked by
  `verify_synthetic_data.py`). Two more surface forms: `Прізвище Ім'я`, `Ім'я Прізвище`.
- **Addresses**: 3,527 street names and 805 villages with district + oblast mined from clear-text organisation
  addresses in train documents (names only, never full addresses; dev / test documents excluded). ~12 shapes,
  house numbers like `12-А`, `12/3`. After `уродженця`, `жителя`, `в`, `у` only a settlement is generated: that is
  how the register uses `АДРЕСА` there.
- **`ІНФОРМАЦІЯ`**: company names after `ТОВ / ПрАТ / АТ «`, titles after `під назвою «`, generated brigades /
  ТЦК names, logins; birth / death keywords extended (`народився`, `загинув`, `ОСОБА_1 (`).
- **Formats**: 8 date, 8 phone, 5 passport formats; `року` stays outside the span when the text already has it.

v1 output is kept for comparison; both versions are in the HF artifacts repo.

## Known limitations and next steps

- **Unsure cases.** About 25% of name gaps are unsure and default to the nominative.
- **Gender** is taken from context for about 40% of persons; the rest get a random gender. This sometimes produces mismatches like «Мельник Олена … звернувся».
- **Small lists.** There are about 24 first names and 32 surnames per gender, and 15 streets. Next: use `lang-uk/name_freq_dict_uk`.
- **Unusual clear-text names** (surname only, or first name without patronymic) are not detected.
- **Next:**
  - manual review of 15–20 dev documents (in progress);
  - a baseline NER (`xlm-roberta-base` or `uk_ner_web_trf_base`);
  - an ablation: all names in the nominative vs rule-based cases;
  - deciding whether a stronger LLM is worth testing on the gold set (cost estimate: about $1–2 for the gold set).
