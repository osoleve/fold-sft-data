# Tier 1 SFT: Meta / KG Ingest

This folder contains a curated Tier-1 SFT dataset focused on **adding new KG facts**:
- extracting triples from natural language
- canonicalizing entities/relations
- validating triples
- upserting triples into a nested KG store

- Target module (new): `lattice/meta/kg-ingest.ss`
- Target tests (new): `lattice/meta/test-kg-ingest.ss`

## Files

- `generate_meta_kg_ingest_sft.py`: deterministic dataset generator
- `validate_meta_kg_ingest_sft.py`: schema + executable validator
- `all.jsonl`: full dataset (train + eval)
- `train.jsonl`: train split
- `eval.jsonl`: eval split
- `summary.json`: family/split/difficulty counts
- `validation-report.json`: validator output

## Covered functions (8)

- `kg-normalize-entity`
- `kg-normalize-relation`
- `kg-make-triple`
- `kg-valid-triple?`
- `kg-parse-simple-fact`
- `kg-extract-triples`
- `kg-upsert-triple`
- `kg-upsert-triples`

## Dataset shape

- Total: 80 samples
- Split: 66 train / 14 eval
- Families:
  - `spec_to_code`: 16
  - `translation`: 16
  - `bugfix`: 16
  - `composition`: 32
- Difficulty: 21 easy / 38 medium / 21 hard

## Design notes

- Verification is self-contained: each `verify_expr` includes all helper/runtime definitions.
- NL extraction tasks use controlled sentence patterns (works at / employed by / lives in / part of / founded).
- Store shape is explicit and testable:
  - `((subject (predicate obj1 obj2 ...) ...) ...)`
- Bugfix tasks are single-point issues (alias coverage, unknown handling, dedupe, accumulator threading).

## Generation

```bash
python3 data/tier1-meta-kg-ingest/generate_meta_kg_ingest_sft.py
```

## Validation

```bash
python3 data/tier1-meta-kg-ingest/validate_meta_kg_ingest_sft.py
```

Validation executes every `verify_expr` in Scheme and enforces eval coverage over all source functions.
