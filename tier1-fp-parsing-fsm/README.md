# Tier1 FP Parsing FSM SFT Dataset

- Source module: `lattice/fp/parsing/fsm.ss`
- Source tests: `lattice/fp/parsing/test-fsm.ss`
- Total samples (current): `96`
- Split (current): `79 train / 17 eval` (deterministic, ~18% eval with source-function coverage)
- Families (current): `24 spec_to_code / 24 translation / 16 bugfix / 32 composition`

## Covered Functions

1. `fsm-delta`
2. `epsilon-closure`
3. `fsm-move`
4. `fsm-accepts?`
5. `fsm-char`
6. `fsm-literal`
7. `nfa->dfa`
8. `fsm-complement`

## Notes

- Focuses on parser-FSM prerequisites: transition lookup, epsilon-closure semantics, move/acceptance flow, machine builders, determinization, and language complement.
- Composition prompts emphasize acceptance/rejection behavior and language-equivalence probes across NFA/DFA transforms.
- Verify expressions run directly against `(require 'fsm)` during validation.

## Generate / Validate

```bash
python3 data/tier1-fp-parsing-fsm/generate_fp_parsing_fsm_sft.py
python3 data/tier1-fp-parsing-fsm/validate_fp_parsing_fsm_sft.py
```

Validation should end with `VALIDATION_OK`.
