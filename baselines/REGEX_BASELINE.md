# Regex Baseline Evaluation

## Scope

This baseline uses only deterministic regexes and the reviewed Ukrainian name
declension table. It does not use an LLM, pretrained NER model, GPU, or paid API.

## Summary

| Split | Documents | Relaxed macro F1 | Strict macro F1 | Deanonymization risk | Consistency B³ | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| dev | 300 | 0.4519 | 0.4006 | 0.0765 | 0.8671 | 4.033 |
| test_a | 526 | 0.4562 | 0.3991 | 0.0947 | 0.9261 | 6.751 |
| test_b | 377 | 0.5226 | 0.4570 | 0.1004 | 0.9102 | 6.994 |
| official | 240 | 0.4845 | 0.3916 | 0.2378 | 0.8241 | 3.041 |

## Improved Run

The first run used broad date matching and a longer address parser. The second run:

- added address starts based on street/prospect/provulok/boulevard markers;
- added district, settlement, postal-code, building, flat, and garage context;
- replaced broad digit patterns with narrower identifier patterns;
- added VIN and military-unit patterns;
- filtered date matches to birth/death context.

Compared with the first run:

| Split | Initial relaxed F1 | Improved relaxed F1 | Initial strict F1 | Improved strict F1 |
|---|---:|---:|---:|---:|
| dev | 0.4849 | 0.4519 | 0.3929 | 0.4006 |
| test_a | 0.5192 | 0.4562 | 0.3947 | 0.3991 |
| test_b | 0.6278 | 0.5226 | 0.4658 | 0.4570 |
| official | 0.4713 | 0.4845 | 0.3146 | 0.3916 |

Official per-type relaxed/strict F1:

| PII type | Initial relaxed F1 | Improved relaxed F1 | Initial strict F1 | Improved strict F1 |
|---|---:|---:|---:|---:|
| ОСОБА | 0.9340 | 0.9340 | 0.9082 | 0.9082 |
| АДРЕСА | 0.6044 | 0.3042 | 0.0444 | 0.0444 |
| НОМЕР | 0.3018 | 0.3567 | 0.2687 | 0.2879 |
| ІНФОРМАЦІЯ | 0.0449 | 0.3433 | 0.0372 | 0.0398 |

The improved address parser became more conservative but did not improve exact
boundaries and reduced relaxed address F1. Number F1 improved. Information precision
improved and false ordinary-date matches decreased, although recall remained limited.

## Data Quality Conclusion

The improvements do not show evidence of systematic synthetic gold-label corruption.
The major person result is unchanged, while address/information changes reflect weak
rule coverage and difficult boundaries.

Do not wholesale-regenerate the generated data on the basis of these F1 changes.
Targeted regeneration or rule fixing is warranted only after confirmed gold/annotation
defects. Priority areas are address span boundaries, birth/death date context,
institution/unit names, and number subtype/context matching.

## Interpretation

- Low strict F1 may be caused by boundary differences, even if relaxed F1 is higher.
- Low person recall likely comes from name forms absent from the declension table.
- Low address/number recall is not evidence that synthetic gold labels are wrong;
  it may reflect limited or incorrect baseline rules.
- High deanonymization risk on an official domain indicates missed repeated person names.
- Consistency is meaningful only where person mentions are found; no consistency
  improvement is inferred from strict/relaxed F1 alone.

## Data-regeneration decision

Do not regenerate generated datasets solely because this baseline scores low.
First compare results with the gold labels:

1. If many synthetic missed spans are real PII with correct gold boundaries, improve
   baseline patterns or the model baseline.
2. If gold spans or synthetic case/gender metadata are grammatically wrong, fix
   `synth_utils.py` and regenerate.
3. If the official test also exposes repeated-name or boundary weaknesses, that is
   primarily a model-rule issue rather than a training-data regeneration issue.

## Runtime and resource note

Reported runtime is Python process inference time for the selected split.
Peak traced-memory is an allocator estimate, not OS peak RSS.
A final CPU/RAM measurement must use the complete official submission and
`/usr/bin/time` on Linux or an equivalent Windows process-memory measurement.
