# tier0-number-theory-primality

SFT dataset for `lattice/number-theory/primality.ss` and `lattice/number-theory/test-primality.ss`.

## Contents

- `generate_primality_sft.py` - deterministic sample generator
- `validate_primality_sft.py` - schema + executable verification
- `all.jsonl` - all samples
- `train.jsonl` - train split
- `eval.jsonl` - eval split
- `summary.json` - counts and distributions

## Dataset shape

- Total: 80
- Families:
  - `spec_to_code`: 16
  - `translation`: 16
  - `bugfix`: 16
  - `composition`: 32
- Difficulty: 14 easy / 38 medium / 28 hard
- Split: 66 train / 14 eval
- Source functions: 8 (`prime?`, `miller-rabin?`, `trial-division`, `factorize`, `prime-factorization`, `divisors`, `euler-totient`, `jacobi-symbol`)

## Run

```bash
python3 data/tier0-number-theory-primality/generate_primality_sft.py
python3 data/tier0-number-theory-primality/validate_primality_sft.py
```

## Quality notes

- Verify dependency closure includes implementation dependencies and function symbols referenced by each `verify_expr`.
- Miller-Rabin samples include deterministic witness-range branches and Carmichael counterexamples.
- Bugfix coverage includes primality edge ordering, repeated-factor extraction, factor-list compression, and Jacobi reciprocity/power-of-two sign handling.
- Miller-Rabin bugfix cases retain the full deterministic threshold table and isolate single-point defects.
