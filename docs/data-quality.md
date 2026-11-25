# Data Quality Report

## Scope

This report describes synthetic-data quality and known limitations. It does not claim
model quality: no baseline or fine-tuned NER model has been evaluated yet.

## Dataset Counts

| Split | Documents | Annotations | Annotations/document | Persons | Uncertain cases | Random gender | Clear-text names |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 4895 | 103482 | 21.14 | 76203 | 14486 | 13058 | 12395 |
| dev | 300 | 6258 | 20.86 | 5187 | 1423 | 1479 | 1081 |
| test_a | 526 | 9649 | 18.34 | 8139 | 1396 | 1452 | 1835 |
| test_b | 377 | 14505 | 38.47 | 7987 | 1865 | 1371 | 699 |

## Reference Test Coverage

- Documents: 240.
- Total annotations: 3685.

| PII type | Reference-test mentions |
|---|---:|
| ОСОБА | 2837 |
| АДРЕСА | 233 |
| НОМЕР | 447 |
| ІНФОРМАЦІЯ | 168 |

The reference test contains a malformed leading-space address string in one raw count;
normalized generated counts use exact PII prefixes.

## PII Subtypes

### train

- `clear_name`: 12395
- `other`: 6118
- `birth_date`: 3755
- `unit_or_institution`: 2827
- `ipn`: 1793
- `plate`: 1696
- `military_unit`: 950
- `death_date`: 898
- `certificate`: 771
- `phone`: 632
- `iban`: 630
- `passport`: 290
- `email`: 267
- `url`: 25
### dev

- `clear_name`: 1081
- `other`: 193
- `military_unit`: 120
- `birth_date`: 107
- `plate`: 102
- `ipn`: 70
- `phone`: 19
- `certificate`: 19
- `death_date`: 8
- `email`: 8
- `unit_or_institution`: 7
- `iban`: 4
- `passport`: 2
### test_a

- `clear_name`: 1835
- `other`: 285
- `birth_date`: 262
- `military_unit`: 153
- `ipn`: 148
- `plate`: 107
- `certificate`: 30
- `phone`: 22
- `iban`: 12
- `death_date`: 9
- `unit_or_institution`: 6
- `email`: 4
- `passport`: 1
### test_b

- `other`: 1958
- `unit_or_institution`: 1186
- `clear_name`: 699
- `birth_date`: 546
- `iban`: 292
- `plate`: 256
- `ipn`: 251
- `phone`: 205
- `death_date`: 147
- `url`: 126
- `email`: 109
- `passport`: 98
- `certificate`: 95
- `nickname`: 78
- `military_unit`: 47

## Quality Issues

| Issue | Frequency / evidence | Impact | Current resolution | Next action | Status |
|---|---|---|---|---|---|
| Uncertain grammatical cases | 19.7%. Case-confidence flags in generated annotations. | Person mentions may receive an incorrect grammatical form. | Fall back to nominative case. | Improve the rule priority and manually resolve gold-set errors. | Known limitation |
| Random gender assignment | 17.8%. gender_from_context=false in person annotations. | A name or surrounding agreement may have a mismatched gender. | Assign gender only from confidence-bearing context. | Expand context inference and flag remaining mismatches. | Known limitation |
| Clear-text names | 16010 annotations. Subtype clear_name in synthetic annotations. | A model can learn to ignore public names that remain in training text. | Detected initialism/full-name patterns are labelled. | Add surname-only and first-name-only detection. | Partial |
| Address-region mismatch | Not yet measured from saved annotations. Generator uses document region for 85% of addresses. | Generated addresses may be geographically inconsistent. | Region detection is used during generation. | Measure detected-region/address consistency on all splits. | Partially implemented |
| Limited name inventory | 24 given names and 32 surnames per gender, approximately. docs/DECISION_LOG.md and name-declension table. | Synthetic documents may encourage memorization of a small name set. | Train/test names come from separate pools. | Integrate lang-uk/name_freq_dict_uk. | Planned |
| Small reference-test type coverage | 168 ІНФОРМАЦІЯ entities. Reference test statistics. | Subtype-level conclusions have wide confidence intervals. | Rare-subtype test_b is used for internal measurement. | Report confidence intervals and per-subtype errors. | Mitigated |
| Reference test domain shift | 240 decisions; 97.5% Supreme Court cassation. data_stats.ipynb and docs/DECISION_LOG.md. | A court-model may not generalize to the broad training domain. | Stratify synthetic sets across proceeding type and instance. | Compare in-domain and cross-domain results. | Measured |

## Case-Rule Findings

- Rules with `pymorphy3` achieved 72% accuracy on the 359-name gold set.
- Accuracy reached 83% on confident cases.
- Lapa one-gap label accuracy was 16%; Lapa choice accuracy was 31%.
- The current uncertain-case policy defaults to nominative.
- Feminine accusative, dative, and masculine accusative/genitive ambiguity remain weak areas.

## Address and Identifier Findings

- Address generation is designed to use the document region for 85% of addresses.
- Addresses in the same sentence share a building where the surrounding text indicates co-location.
- РНОКПП/IPН values use valid checksum logic.
- IBAN values use valid mod-97 logic.
- Dates are generated in multiple formats.

## Required Improvements

1. Measure address-region consistency from generated addresses.
2. Complete 15–20 dev-document manual reviews.
3. Expand name, patronymic, and surname coverage.
4. Improve clear-text surname-only and first-name-only name detection.
5. Add per-subtype precision and recall after baseline training.
6. Keep reference-test annotations locked from training and tuning.

## Machine-Readable Outputs

- `data/synthetic/quality_report.csv`: aggregate split statistics.
- `data/synthetic/quality_issues.csv`: issue, frequency, evidence, and remediation.
