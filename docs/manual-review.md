# Manual Review

## Scope

This file records the preparation and results of manual review for 20 `dev` documents.
The sample is risk-ranked automatically: uncertain cases, random gender, clear-text
names, rare subtypes, and addresses receive a higher review priority.

## Important Status

The automated and evidence-focused review pass covers all 20 selected `dev`
documents. Initial review verdicts were deliberately conservatively normalized:
uncertain-case/gender metadata alone is not counted as a proven annotation defect.
Final conclusions are recorded in `data/synthetic/manual_review_sample.csv` and
`data/synthetic/review_notes.json`.

## Review Procedure

For every selected document, verify:

1. Every synthetic PII span is complete.
2. Span boundaries exclude surrounding punctuation and whitespace.
3. The pseudonym is consistent across repeated mentions.
4. Person case and gender agree with surrounding Ukrainian grammar.
5. Clear-text names are complete and correctly classified.
6. Address, number, and information subtypes match the context.
7. No unrelated text was replaced.

## Selected Documents

| Rank | Document ID | Risk score | Risk flags | Annotations | Verdict | Issue count / categories |
|---:|---:|---:|---|---:|---|---|
| 1 | `118597672` | 1849 | uncertain_case=369; random_gender=369; clear_text_name=4 | 379 | problem | 3: case, gender, pseudonym |
| 2 | `116552444` | 988 | uncertain_case=199; random_gender=192; clear_text_name=7 | 214 | problem | 4: case, gender, missed PII, pseudonym |
| 3 | `116298703` | 842 | uncertain_case=145; random_gender=201; address=5 | 234 | problem | 3: case, gender, missed PII |
| 4 | `119676786` | 465 | uncertain_case=91; random_gender=91; clear_text_name=8; address=2 | 103 | problem | 3: case, gender, information subtype |
| 5 | `117911349` | 153 | clear_text_name=1; address=152 | 153 | problem | 2: address, pseudonym |
| 6 | `117736061` | 131 | uncertain_case=23; random_gender=30; clear_text_name=2 | 40 | problem | 3: case, gender, missed PII |
| 7 | `116899663` | 108 | uncertain_case=10; random_gender=29; clear_text_name=7; address=13 | 66 | problem | 2: case, gender |
| 8 | `118320457` | 76 | uncertain_case=18; clear_text_name=7; address=15 | 43 | problem | 2: case, address |
| 9 | `118706087` | 72 | uncertain_case=17; random_gender=9; address=3 | 61 | problem | 3: case, gender, gender agreement |
| 10 | `122166090` | 65 | uncertain_case=15; random_gender=9; address=2 | 48 | problem | 3: case, gender, information subtype |
| 11 | `121050852` | 60 | uncertain_case=12; random_gender=12 | 23 | problem | 2: case, gender |
| 12 | `118482761` | 57 | uncertain_case=12; random_gender=8; clear_text_name=5 | 27 | problem | 2: case, gender |
| 13 | `119135640` | 55 | uncertain_case=12; random_gender=7; address=5 | 48 | problem | 2: case, gender |
| 14 | `122803349` | 55 | uncertain_case=6; clear_text_name=29; rare_subtype=4 | 79 | problem | 2: case, gender agreement |
| 15 | `122579744` | 52 | uncertain_case=8; random_gender=10; clear_text_name=8 | 52 | problem | 3: case, gender, information subtype |
| 16 | `123598224` | 52 | uncertain_case=10; random_gender=11 | 16 | problem | 2: case, gender |
| 17 | `116529754` | 51 | uncertain_case=9; random_gender=9; clear_text_name=6 | 18 | problem | 2: case, gender |
| 18 | `120441968` | 51 | uncertain_case=11; random_gender=9 | 22 | problem | 3: case, gender, gender agreement |
| 19 | `117290482` | 48 | uncertain_case=4; random_gender=12; rare_subtype=2; address=8 | 60 | problem | 4: case, gender, number subtype, missed identifiers |
| 20 | `122038839` | 47 | uncertain_case=12; random_gender=5; address=1 | 33 | problem | 2: case, gender |

## Review Method and Limitations

The full original text of each of the 20 selected documents was inspected for
obvious missed PII and false positives. Annotation metadata was screened across
the sample, including available case/gender confidence flags, subtype data,
repeated pseudonym/identity mappings, and address patterns; contextual review
prioritized the flagged and otherwise high-risk items. Because some documents
contain hundreds of person annotations, **not every individual person span was
reread manually in context**. Ambiguous source or subtype conflicts are called
out in the comments for manual follow-up rather than presented as certain errors.

## Findings Summary

Counts below are documents affected; categories overlap:

- Case uncertainty/low-confidence person annotations: 19 documents.
- Gender labels not inferred from context: 17 documents.
- Gender agreement conflicts between names and surrounding text: 3 documents.
- Missed PII names or addresses: 3 documents; potentially omitted device serial
  identifiers were additionally flagged in 1 document.
- Inconsistent pseudonym/identity mappings: 3 documents.
- Address/co-location inconsistencies: 2 documents.
- Uncertain information/date subtype assignments: 3 documents.
- Number subtype mismatch: 1 document.

## Automated Structural Pre-Review

Before human review, the sample is checked against:

- valid JSON and character offsets;
- non-empty spans;
- known PII-prefix types;
- no overlapping annotation spans;
- the existing official evaluator format.

## Machine-Readable Checklist

`data/synthetic/manual_review_sample.csv` contains document IDs, risk scores,
risk flags, snippets, and the completed `human_verdict` / `comment` fields.

## Human Review Results

Exactly **20 documents were reviewed**: **0 OK and 20 problem**. The result
records are in `data/synthetic/review_notes.json`, keyed by document ID, with
`verdict`, `comment`, and `split` fields.
