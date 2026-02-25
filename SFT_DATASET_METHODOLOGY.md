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
- `prompt_body` (pre-diversification task text)
- `prompt` (DSL-diversified final prompt)
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

Preferred balance for new Tier-1 datasets:

- `24 / 24 / 16 / 32` (total `96`)

Legacy balance (still valid for historical sets):

- `16 / 16 / 16 / 32` (total `80`)

Important:

- `80` is a practical template, not a hard requirement.
- If module complexity demands different coverage, adjust counts and document why.

Difficulty policy:

- Calibrate difficulty per task, not only per source function.
- The same function may appear as easy (predicate check), medium (translation), or hard (multi-step composition) depending on task shape.
- Record explicit per-sample difficulty for composition/bugfix rows when they diverge from function baseline.

## 6. Prompt Generation Pipeline (DSL-First)

New and refreshed generators should follow a two-stage prompt pipeline:

1. Build deterministic pre-diversification rows in Python with:
- `prompt_body`
- all task metadata (`id`, `family`, `category`, `source_function`, etc.)
- deterministic `split` assigned by dataset split policy

2. Run `user/sft/generate.ss` to produce:
- final diversified `prompt`
- preserved `prompt_body`
- stable split output (`all/train/eval`)

Rules:

- Do not use `data/sft_prompt_diversity.py` for new datasets.
- Keep `prompt_body` as the canonical task intent (for re-diversification/replay).
- Treat DSL output (`prompt`) as presentation layer, not supervision source.
- Preserve split if already present in pre-diversification rows.
- Require `prompt_body` on every generated row; `prompt` fallback is legacy/migration-only.
- For `composition`, keep `prompt_body` intent-only. Do not embed verifier snippets/code fences.

## 7. Ground Truth Authoring

`ground_truth` must be:

- Behaviorally canonical with source module semantics
- Minimal and direct
- Valid Fold Scheme

For non-composition families:

- Ground truth is typically a full function definition `(define (...))`.

For composition:

- Ground truth is usually an expression that exercises module APIs.

## 8. Verify Expression Design

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
- Re-defining the target function inside `verify_expr` for non-composition tasks (oracle leakage)

When augmenting verifies with extra checks:

- Avoid nested `(and (and ...))` forms.
- If base verify already has top-level `and`, flatten before appending.
- Keep final `verify_expr` easy to decompose/analyze.

## 9. Translation Prompt Method

Use two translation styles per function:

- Python-like pseudocode to Fold Scheme
- Chez-style snippet to canonical Fold style
- Legacy/noisy fragment extraction to Fold (optional but recommended)

Rules:

- Snippets must preserve exact semantics
- Avoid placeholders like `...`
- Keep target function name explicit in prompt
- Ensure translation snippets are feature-complete relative to target function
- Include at least one non-trivial structural rewrite per function set (not just “add doc forms”)
- Include some doc-free targets so models learn executable core forms without metadata wrappers
- Prefer realistic source-style excerpts over synthetic wrapper boilerplate when building fragment-translation prompts

Required QA for translation snippets:

- Python and Chez snippets must cover the same behavior surface as `ground_truth`.
- Missing control-flow arms (for example missing `(pair? ...)` branch) are not acceptable in translation tasks.
- If partial snippets are intentional, classify as completion/spec tasks, not translation tasks.

## 10. Bugfix Sample Method

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

## 11. Composition Sample Method

Composition should test usage fluency and behavior integration, not re-implementation.

Good composition prompts:

- Short API pipelines
- Behavioral property checks
- Multi-step expressions with deterministic output
- Novel compositions that are not copied directly from source tests
- Prompts that state intent without embedding exact verifier code

Each composition expression should still be easy to execute in validator context.
Recommended floor: at least 50% of composition samples should invoke 2+ module APIs together.
Avoid directly pasting full `verify_expr` snippets into composition prompts; keep the executable oracle in dataset fields, not prompt text.
Never include prompt sections like `Target properties` or `Behavior check to satisfy` that inline verifier code.

## 12. Deterministic Split Policy

Split generation must be deterministic.

Recommended policy:

- Use a deterministic source-function-disjoint split when multiple variants share `ground_truth`/`verify_expr` structure.
- Assign each `source_function` wholly to `train` or `eval`.
- Keep eval size near target ratio by selecting function buckets deterministically (optionally stratified by difficulty).

Legacy policy (only when rows are demonstrably independent across variants):

- Family-quotas over rows (for example `66/14` with `3/3/3/5` family floors).

Never allow:

- Train/eval ID overlap
- `source_function` overlap across splits when tasks share targets/verifier templates

## 13. Build and Validation Workflow

From repo root:

```bash
python3 data/<dataset>/generate_<name>_sft.py
python3 data/<dataset>/validate_<name>_sft.py
```

Generator responsibilities:

- Emit/retain `prompt_body`.
- Run DSL prompt generation (`user/sft/generate.ss`) as part of build.
- Write deterministic `all/train/eval`.

Run source module tests as an independent oracle:

```bash
scheme --quiet --script <source-test-file>
```

Both dataset validator and module tests must pass.

## 14. Mandatory QA Passes

After generation and basic validation, run deeper QA:

1. Bugfix semantic sweep:
- Replace target function with each buggy snippet
- Run that function’s `verify_expr`
- Ensure each buggy case fails

2. Prompt/answer integrity:
- No unresolved placeholders (`<TODO>`, incomplete snippets)
- Prompt intent matches ground truth family
- `prompt_body` present and meaningful for every row
- `prompt` present for every row after DSL generation
- Composition prompts do not leak exact answers via pasted verifier expressions

3. Composition utility:
- Confirm expressions are behaviorally meaningful, not trivial wrappers
- Confirm composition rows are not near-clones of module test assertions

4. Diversity sanity:
- Ensure examples are not near-duplicates across families
- Run near-duplicate prompt checks within `(family, source_function)` groups

5. Translation fidelity:
- Translation snippets are semantically complete (no missing branches/arms)
- Translation samples are not covert completion tasks
- Chez→Fold pairs are not trivial near-identical transforms after doc-form stripping

6. Verify structure sanity:
- No nested top-level `(and (and ...))` in augmented verifies
- Verifies remain decomposable for downstream grammar tooling
- Non-composition verifies do not redefine their own target function

7. Documentation sanity:
- README explicitly states train/eval split counts and family counts

## 15. Maintenance Triggers

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

## 16. Expansion Playbook

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

- Update all hard-coded count assertions and expected schema keys.
- Revisit split quotas.
- Document why this dataset diverges from template count.

## 17. Count Policy and Coverage Tradeoffs

Use this policy when deciding dataset size:

- Prefer behavior coverage quality over hitting a fixed number.
- If `80` forces omission of high-value behaviors, increase count.
- If module is genuinely small, smaller totals are acceptable.

Document tradeoffs in dataset `README.md`:

- What was included
- What was deferred
- Why

## 18. Issue Tracking and Review Discipline

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

## 19. Definition of Done

A dataset update is done only when all are true:

1. Generator runs and writes all expected artifacts.
2. Validator reports success (`VALIDATION_OK`).
3. Source module tests pass.
4. Buggy variants fail their corresponding verifies.
5. Prompt pipeline is DSL-first (`prompt_body` + DSL-generated `prompt`).
6. Dataset README is accurate.
7. `data/README.md` index entry is accurate.
8. Tracking issue status/comments are updated.

## 20. Common Failure Modes

- Verify checks too weak (buggy code passes)
- Composition examples too trivial to train useful behavior
- Split logic fails eval function coverage
- Drift between source semantics and ground truths
- Hidden syntax issues in embedded snippet strings
- Translation snippets missing semantic branches (becoming implicit completion tasks)
- Legacy prompt diversification path used instead of DSL-first pipeline
- Missing `prompt_body` prevents clean regeneration/re-diversification
- Verify augmentation introduces nested `(and (and ...))` structures

When in doubt, prefer explicit behavior assertions and small deterministic examples.
