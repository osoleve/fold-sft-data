# core/base/prelude (Extended) SFT Dataset

Synthetic SFT data for additional pure functions in `core/base/prelude.ss`.

## Scope
- Source module: `core/base/prelude.ss`
- Source tests: `core/base/test-prelude.ss`
- Covered functions: 13 (`unique-simple`, `unique-fast`, `cons*`, `assoc-ref`, `assq-ref`, `alist-update`, `ok?`, `error?`, `unwrap-ok`, `unwrap-error`, `result-map`, `result-bind`, `result-sequence`)
- Families: `spec_to_code`, `translation`, `bugfix`, `composition`
- Total samples: 78

## Split
- Train: 62
- Eval: 16

## Family counts
- `spec_to_code`: 26
- `translation`: 13
- `bugfix`: 13
- `composition`: 26

## Files
- `generate_core_prelude_extended_sft.py`: deterministic generator
- `validate_core_prelude_extended_sft.py`: schema + split + executable verification
- `all.jsonl`: full dataset
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: generation summary
- `validation-report.json`: latest validation report

## Regenerate
```bash
python3 data/core-base-prelude-extended/generate_core_prelude_extended_sft.py
python3 data/core-base-prelude-extended/validate_core_prelude_extended_sft.py
```
