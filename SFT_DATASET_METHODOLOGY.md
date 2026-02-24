# Fold SFT Dataset Methodology (General)

This document describes the standard methodology used to build, maintain, and expand deterministic SFT datasets in `data/`.

It is intentionally module-agnostic and should be used as the default playbook for new dataset generators.

## 1. Purpose and Scope

These datasets train models on Fold-native reasoning and implementation patterns by grounding tasks in real lattice modules and executable tests.

Core design goals:

- Behavior-first supervision, not style imitation
- Deterministic generation and splitting
- Executable acceptance checks (`verify_expr`)
- High-signal bugfix and composition tasks
- Repeatable maintenance when source modules evolve

## 2. Standard Dataset Shape

Each dataset directory should include:

- `generate_<name>_sft.py`
- `validate_<name>_sft.py`
- `all.jsonl`
- `train.jsonl`
- `eval.jsonl`
- `summary.json`
- `validation-report.json`
- `README.md`

Expected sample fields:

- `id`
- `family`
- `category`
- `difficulty`
- `source_module`
- `source_test`
- `source_function`
- `prompt`
- `ground_truth`
- `verify_expr`
- `tags`
- `split` (`train` or `eval`)

## 3. Module Selection Criteria

Pick modules that satisfy most of the following:

- Strong and stable unit tests
- Clear API boundaries (8-14 core functions usually works well)
- Behavior with meaningful edge cases
- Value as a model target (algorithmic, compositional, or high-use primitives)
- Manageable dependency footprint for validator execution

Avoid modules that are:

- Largely unstable or under active semantic churn
- Mostly wrappers with weak behavior signals
- Difficult to verify without heavy integration harnesses

## 4. Function Coverage Strategy

For each selected module:

1. Identify a focused function set (`FUNCTION_ORDER`).
2. Include enough helper/internal functions if needed for behavior coverage.
3. Prefer balanced coverage across:
   - construction/setup functions
   - core transition/compute functions
   - public “user-facing” entry points

Guideline:

- Keep per-function sample counts comparable unless a function is intrinsically higher value.

## 5. Family Design

Use four families unless there is a strong reason not to:

- `spec_to_code`
- `translation`
- `bugfix`
- `composition`

Default balance used in many Tier-1 datasets:

- `16 / 16 / 16 / 32` (total `80`)

Important:

- `80` is a practical template, not a hard requirement.
- If module complexity demands different coverage, adjust counts and document why.

## 6. Ground Truth Authoring

`ground_truth` must be:

- Behaviorally canonical with source module semantics
- Minimal and direct
- Valid Fold Scheme

For non-composition families:

- Ground truth is typically a full function definition `(define (...))`.

For composition:

- Ground truth is usually an expression that exercises module APIs.

## 7. Verify Expression Design

`verify_expr` is the core quality gate. It should:

- Return `#t` only for correct behavior
- Reject plausible incorrect implementations
- Cover at least one edge case per function where practical
- Stay self-contained and executable under validator imports

Strong verify patterns:

- Multi-assert checks in a single expression
- Positive and negative cases together
- Checks that target known failure modes

Weak verify patterns to avoid:

- Type-only checks
- Single happy-path input
- Conditions that buggy variants can still satisfy

## 8. Translation Prompt Method

Use two translation styles per function:

- Python-like pseudocode to Fold Scheme
- Chez-style snippet to canonical Fold style

Rules:

- Snippets must preserve exact semantics
- Avoid placeholders like `...`
- Keep target function name explicit in prompt

## 9. Bugfix Sample Method

Each bugfix sample should include one realistic defect with minimal repair scope.

Preferred bug classes:

- Off-by-one / boundary
- Incorrect branch/fallback behavior
- Wrong helper choice
- Missing state/metadata update
- Semantics-preserving shape but wrong behavior

Avoid:

- Pure syntax breakage unless syntax repair is explicitly the intended skill
- Multi-defect snippets that make attribution unclear

## 10. Composition Sample Method

Composition should test usage fluency and behavior integration, not re-implementation.

Good composition prompts:

- Short API pipelines
- Behavioral property checks
- Multi-step expressions with deterministic output

Each composition expression should still be easy to execute in validator context.

## 11. Deterministic Split Policy

Split generation must be deterministic.

Common policy:

- `66 train / 14 eval` for `80` total
- Quotas per family (for example 3/3/3/5)
- Post-pass to ensure eval contains every covered source function

Never allow:

- Train/eval ID overlap
- Eval missing function coverage

## 12. Build and Validation Workflow

From repo root:

```bash
python3 data/<dataset>/generate_<name>_sft.py
python3 data/<dataset>/validate_<name>_sft.py
```

Run source module tests as an independent oracle:

```bash
scheme --quiet --script <source-test-file>
```

Both dataset validator and module tests must pass.

## 13. Mandatory QA Passes

After generation and basic validation, run deeper QA:

1. Bugfix semantic sweep:
- Replace target function with each buggy snippet
- Run that function’s `verify_expr`
- Ensure each buggy case fails

2. Prompt/answer integrity:
- No unresolved placeholders (`<TODO>`, incomplete snippets)
- Prompt intent matches ground truth family

3. Composition utility:
- Confirm expressions are behaviorally meaningful, not trivial wrappers

4. Diversity sanity:
- Ensure examples are not near-duplicates across families

## 14. Maintenance Triggers

Update dataset when:

- Source module behavior changes
- Source tests change expected behavior
- Supporting dependencies affect semantics

Maintenance flow:

1. Diff source module + test files.
2. Update impacted `DEFS`, snippets, and `verify_expr`.
3. Regenerate artifacts.
4. Re-run validator + module tests.
5. Re-run bugfix semantic QA.
6. Update dataset README and top-level `data/README.md` if coverage/count changed.

## 15. Expansion Playbook

You can expand by:

- Adding new function coverage
- Increasing sample count per family
- Adding richer composition/bugfix templates

When adding functions:

1. Add function to canonical maps:
- `DEFS`
- `FUNCTION_ORDER`
- `FUNCTION_SPECS`
- `SKELETONS`
- `VERIFY_BY_FUNCTION`
- translation snippet maps
2. Add bugfix cases (at least 2).
3. Add composition cases (at least 4).
4. Regenerate + validate + QA.

When increasing total samples:

- Update all hard-coded count assertions.
- Revisit split quotas.
- Document why this dataset diverges from template count.

## 16. Count Policy and Coverage Tradeoffs

Use this policy when deciding dataset size:

- Prefer behavior coverage quality over hitting a fixed number.
- If `80` forces omission of high-value behaviors, increase count.
- If module is genuinely small, smaller totals are acceptable.

Document tradeoffs in dataset `README.md`:

- What was included
- What was deferred
- Why

## 17. Issue Tracking and Review Discipline

For ongoing programs:

- Track each dataset in BBS issue tracker.
- Record completion notes with:
  - generated counts
  - validator result
  - module test result
  - bugfix QA result

Also create follow-up items for:

- any compromised coverage decisions
- any known weak samples to revisit

## 18. Definition of Done

A dataset update is done only when all are true:

1. Generator runs and writes all expected artifacts.
2. Validator reports success (`VALIDATION_OK`).
3. Source module tests pass.
4. Buggy variants fail their corresponding verifies.
5. Dataset README is accurate.
6. `data/README.md` index entry is accurate.
7. Tracking issue status/comments are updated.

## 19. Common Failure Modes

- Verify checks too weak (buggy code passes)
- Composition examples too trivial to train useful behavior
- Split logic fails eval function coverage
- Drift between source semantics and ground truths
- Hidden syntax issues in embedded snippet strings

When in doubt, prefer explicit behavior assertions and small deterministic examples.

