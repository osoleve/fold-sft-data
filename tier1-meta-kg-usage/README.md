# Tier 1 SFT: Meta / KG Usage

This folder contains a curated Tier-1 SFT dataset focused on **using** the enhanced knowledge graph APIs (query/search/filter/composition), not KG construction.

- Module: `lattice/meta/search.ss`
- Tests referenced: `lattice/meta/test-meta.ss`

## Files

- `generate_meta_kg_usage_sft.py`: deterministic dataset generator
- `validate_meta_kg_usage_sft.py`: schema + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split/difficulty counts
- `validation-report.json`: validator output

## Covered functions (8)

- `concept-boost-for-query`
- `result->skill-name`
- `apply-concept-boosts`
- `lattice-export-source`
- `lattice-find-prefix`
- `lattice-find-substring`
- `lattice-find-by-tier`
- `lattice-find-by-purity`

## Dataset shape

- Total: 80 samples
- Split: 66 train / 14 eval
- Families:
  - `spec_to_code`: 16
  - `translation`: 16
  - `bugfix`: 16
  - `composition`: 32
- Difficulty: 27 easy / 36 medium / 17 hard

## Design notes

- Verification is self-contained: each `verify_expr` hydrates a mock in-memory KG/query state.
- Composition prompts emphasize API usage patterns (search filtering, source attribution, concept-based re-ranking).
- Bugfix tasks are single-point defects in realistic query logic (ranking, field selection, mapping, score math).

## Generation

```bash
python3 data/tier1-meta-kg-usage/generate_meta_kg_usage_sft.py
```

## Validation

```bash
python3 data/tier1-meta-kg-usage/validate_meta_kg_usage_sft.py
```

Validation executes all `verify_expr` checks in Scheme and enforces eval coverage for every source function in this dataset.
