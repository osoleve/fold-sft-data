# core/base/span SFT Dataset

Synthetic SFT data for `core/base/span.ss`.

## Scope
- Source module: `core/base/span.ss`
- Source tests: `core/base/test-error.ss`
- Families: `spec_to_code`, `translation`, `bugfix`, `composition`
- Total samples: 54

## Split
- Train: 43
- Eval: 11

## Family counts
- `spec_to_code`: 18
- `translation`: 9
- `bugfix`: 9
- `composition`: 18

## Files
- `generate_core_span_sft.py`: deterministic generator
- `validate_core_span_sft.py`: schema + split + executable verification
- `all.jsonl`: full dataset
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: generation summary
- `validation-report.json`: latest validation report

## Regenerate
```bash
python3 data/core-base-span/generate_core_span_sft.py
python3 data/core-base-span/validate_core_span_sft.py
```
