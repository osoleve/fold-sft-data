# Tier1 Query Aho-Corasick SFT Dataset

- Source module: `lattice/query/aho-corasick.ss`
- Source tests: `lattice/query/test-aho-corasick.ss`
- Total samples (current): `96`
- Split (current): `79 train / 17 eval` (deterministic, ~18% eval with source-function coverage)
- Families (current): `24 spec_to_code / 24 translation / 16 bugfix / 32 composition`

## Covered Functions

1. `build-trie`
2. `insert-pattern-mut!`
3. `compute-failures`
4. `bfs-mut!`
5. `find-fail`
6. `get-next`
7. `make-automaton`
8. `search`

## Notes

- Focuses on multi-pattern matching build/step/scan workflow: trie construction, failure-link propagation, transition fallback, and streaming match extraction.
- Composition prompts include overlap-heavy matches, fallback-bridge behavior (`ab` + `bc`), and DNA-style codon scans.
- Verify expressions run directly against `(require 'aho-corasick)` during validation.
- Detailed maintainer instructions: `DATASET_MAINTENANCE_GUIDE.md`

## Generate / Validate

```bash
python3 data/tier1-query-aho-corasick/generate_query_aho_corasick_sft.py
python3 data/tier1-query-aho-corasick/validate_query_aho_corasick_sft.py
```

Validation should end with `VALIDATION_OK`.
