# tier1-info-entropy

SFT dataset for `lattice/info/entropy.ss` and `lattice/info/test-entropy.ss`.

## Contents

- `generate_entropy_sft.py` - deterministic sample generator
- `validate_entropy_sft.py` - schema + executable verification
- `all.jsonl` - all samples
- `train.jsonl` - train split
- `eval.jsonl` - eval split
- `summary.json` - counts and distributions
- `validation-report.json` - validator status/report

## Dataset shape

- Total: 80
- Families:
  - `spec_to_code`: 16
  - `translation`: 16
  - `bugfix`: 16
  - `composition`: 32
- Difficulty: 22 easy / 16 medium / 42 hard
- Split: 66 train / 14 eval
- Source functions: 8 (`entropy`, `entropy-normalized`, `binary-entropy`, `mutual-information`, `cross-entropy`, `kl-divergence`, `jensen-shannon-divergence`, `renyi-entropy`)

## Run

```bash
python3 data/tier1-info-entropy/generate_entropy_sft.py
python3 data/tier1-info-entropy/validate_entropy_sft.py
```

## Quality notes

- Deterministic prompt diversification is applied via `data/sft_prompt_diversity.py`.
- Verify dependency closure includes explicit roots plus symbol-scanned references from each verify expression (`verify_refs`).
- Numeric verification uses tolerance checks (`approx=?`) plus exact infinity checks (`+inf.0`) for support-mismatch cases.
- Bugfix tasks target realistic entropy/divergence defects (sign errors, normalization mistakes, missing boundary handling, incorrect formulas).
