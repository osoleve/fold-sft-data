#!/usr/bin/env python3
"""Generate Tier-1 random probability SFT samples for lattice/random/probability.ss."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parents[1]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from sft_prompt_diversity import diversify_prompt

ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

SOURCE_MODULE = "lattice/random/probability.ss"
SOURCE_TEST = "lattice/random/test-probability.ss"

SOURCE_DEFS: Dict[str, str] = {
    "make-prob": """(define (make-prob state-comp)
  (cons 'prob state-comp))""",
    "prob?": """(define (prob? x)
  (and (pair? x) (eq? (car x) 'prob)))""",
    "run-prob": """(define (run-prob p prng)
  (let ([result (run-state (prob-state p) (cons prng 0.0))])
       (let ([value (car result)]
             [final-state (cdr result)])
            (cons (cons value (cdr final-state))
                  (car final-state)))))""",
    "sample-prob": """(define (sample-prob p prng)
  (car (car (run-prob p prng))))""",
    "weight-prob": """(define (weight-prob p prng)
  (cdr (car (run-prob p prng))))""",
    "prob-bind": """(define (prob-bind p f)
  (make-prob
   (state-bind (prob-state p)
               (lambda (a)
                       (prob-state (f a))))))""",
    "log-sum-exp": """(define (log-sum-exp xs)
  (if (null? xs)
      -inf.0
      (let ([max-x (apply max xs)])
           (if (= max-x -inf.0)
               -inf.0
               (+ max-x (log-num (fold-left + 0 (map (lambda (x) (exp-num (- x max-x))) xs))))))))""",
    "normalize-log-weights": """(define (normalize-log-weights log-ws)
  (let ([total (log-sum-exp log-ws)])
       (if (= total -inf.0)
           (map (lambda (_) (/ 1.0 (length log-ws))) log-ws)
           (map (lambda (lw) (- lw total)) log-ws))))""",
}

SUPPORT_DEFS: Dict[str, str] = {
    "make-state": """(define (make-state run-fn)
  (cons 'state run-fn))""",
    "state-fn": """(define (state-fn st)
  (cdr st))""",
    "run-state": """(define (run-state st initial-state)
  ((state-fn st) initial-state))""",
    "state-pure": """(define (state-pure x)
  (make-state (lambda (s) (cons x s))))""",
    "state-bind": """(define (state-bind st f)
  (make-state
   (lambda (s)
           (let* ([result (run-state st s)]
                  [a (car result)]
                  [s2 (cdr result)])
                 (run-state (f a) s2)))))""",
    "prob-state": """(define (prob-state p)
  (cdr p))""",
    "prob-pure": """(define (prob-pure x)
  (make-prob (state-pure x)))""",
    "log-num": """(define (log-num x)
  (log x))""",
    "exp-num": """(define (exp-num x)
  (exp x))""",
    "approx=?": """(define (approx=? a b tol)
  (< (abs (- a b)) tol))""",
}

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **SOURCE_DEFS}

SOURCE_FUNCTION_ORDER = [
    "make-prob",
    "prob?",
    "run-prob",
    "sample-prob",
    "weight-prob",
    "prob-bind",
    "log-sum-exp",
    "normalize-log-weights",
]

SUPPORT_ORDER = [
    "make-state",
    "state-fn",
    "run-state",
    "state-pure",
    "state-bind",
    "prob-state",
    "prob-pure",
    "log-num",
    "exp-num",
    "approx=?",
]

DEPENDS: Dict[str, List[str]] = {
    "make-state": [],
    "state-fn": [],
    "run-state": ["state-fn"],
    "state-pure": ["make-state"],
    "state-bind": ["make-state", "run-state"],
    "prob-state": [],
    "prob-pure": ["make-prob", "state-pure"],
    "log-num": [],
    "exp-num": [],
    "approx=?": [],
    "make-prob": [],
    "prob?": [],
    "run-prob": ["run-state", "prob-state"],
    "sample-prob": ["run-prob"],
    "weight-prob": ["run-prob"],
    "prob-bind": ["make-prob", "state-bind", "prob-state"],
    "log-sum-exp": ["log-num", "exp-num"],
    "normalize-log-weights": ["log-sum-exp"],
}

FUNCTION_SPECS = {
    "make-prob": "Wrap a state computation as a probability monad value tagged with 'prob.",
    "prob?": "Return #t exactly when a value is a tagged probability computation with leading symbol 'prob.",
    "run-prob": "Execute a probability computation from an initial PRNG and zero log-weight, returning ((value . log-weight) . new-prng).",
    "sample-prob": "Project the sampled value from run-prob output.",
    "weight-prob": "Project the final log-weight from run-prob output.",
    "prob-bind": "Monad bind: sequence probability computations while threading internal state.",
    "log-sum-exp": "Compute numerically stable log(sum(exp(xs))) with empty/all -inf edge handling.",
    "normalize-log-weights": "Normalize log-weights by subtracting log-sum-exp; return uniform probabilities when all entries are -inf.",
}

SKELETONS = {
    "make-prob": """(define (make-prob state-comp)
  ;; TODO: construct probability wrapper tagged with 'prob
  <TODO>)""",
    "prob?": """(define (prob? x)
  ;; TODO: recognize probability wrappers tagged with 'prob
  <TODO>)""",
    "run-prob": """(define (run-prob p prng)
  ;; TODO: run state with (prng . 0.0) and return ((value . log-weight) . new-prng)
  <TODO>)""",
    "sample-prob": """(define (sample-prob p prng)
  ;; TODO: extract sample value from run-prob result
  <TODO>)""",
    "weight-prob": """(define (weight-prob p prng)
  ;; TODO: extract log-weight from run-prob result
  <TODO>)""",
    "prob-bind": """(define (prob-bind p f)
  ;; TODO: sequence p then f, threading probability state
  <TODO>)""",
    "log-sum-exp": """(define (log-sum-exp xs)
  ;; TODO: stable log-sum-exp with empty and all -inf handling
  <TODO>)""",
    "normalize-log-weights": """(define (normalize-log-weights log-ws)
  ;; TODO: normalize by total log-mass; fallback uniform when all impossible
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "make-prob": """(and
  (prob? (make-prob (state-pure 1)))
  (eq? (car (make-prob (state-pure 1))) 'prob)
  (equal? (run-state (prob-state (make-prob (state-pure 'ok))) (cons 'g0 0.0))
          (cons 'ok (cons 'g0 0.0)))
  (not (prob? '(not-prob . 1))))""",
    "prob?": """(and
  (prob? (make-prob (state-pure 3)))
  (prob? '(prob . fake))
  (not (prob? '(state . 2)))
  (not (prob? 17)))""",
    "run-prob": """(let* ([p1 (prob-pure 42)]
       [r1 (run-prob p1 'seed-a)]
       [p2 (make-prob
            (make-state (lambda (ps)
                          (cons 'done
                                (cons 'seed-b (+ (cdr ps) 1.25))))))]
       [r2 (run-prob p2 'seed-a)])
  (and (equal? r1 (cons (cons 42 0.0) 'seed-a))
       (equal? (car (car r2)) 'done)
       (equal? (cdr r2) 'seed-b)
       (approx=? (cdr (car r2)) 1.25 1e-12)))""",
    "sample-prob": """(let* ([p1 (prob-pure 7)]
       [p2 (make-prob
            (make-state (lambda (ps)
                          (cons (+ 1 (cdr ps))
                                (cons (car ps) (+ (cdr ps) 0.5))))))])
  (and (= (sample-prob p1 'g0) 7)
       (approx=? (sample-prob p2 'g0) 1.0 1e-12)
       (approx=? (weight-prob p2 'g0) 0.5 1e-12)))""",
    "weight-prob": """(let* ([p1 (prob-pure 'x)]
       [p2 (make-prob
            (make-state (lambda (ps)
                          (cons 'v (cons 'next (+ (cdr ps) -0.75))))))])
  (and (approx=? (weight-prob p1 'g0) 0.0 1e-12)
       (approx=? (weight-prob p2 'g0) -0.75 1e-12)
       (equal? (sample-prob p2 'g0) 'v)))""",
    "prob-bind": """(let* ([left (run-prob (prob-bind (prob-pure 9)
                                  (lambda (x) (prob-pure (+ x 2))))
                       'seed-x)]
       [right (run-prob (prob-pure 11) 'seed-x)]
       [p (make-prob
           (make-state
            (lambda (ps)
              (cons 3 (cons 'inner (+ (cdr ps) 0.5))))))]
       [f (lambda (x)
            (make-prob
             (make-state
              (lambda (ps)
                (cons (+ x 4) (cons (car ps) (+ (cdr ps) 1.0)))))))]
       [r (run-prob (prob-bind p f) 'seed0)])
  (and (equal? left right)
       (= (car (car r)) 7)
       (equal? (cdr r) 'inner)
       (approx=? (cdr (car r)) 1.5 1e-12)))""",
    "log-sum-exp": """(let* ([a (log-sum-exp '(0.0 0.0))]
       [b (log-sum-exp (list (log-num 2.0) 0.0))]
       [shifted (- (log-sum-exp '(2.0 1.0 -1.0)) 2.0)]
       [base (log-sum-exp '(0.0 -1.0 -3.0))])
  (and (= (log-sum-exp '()) -inf.0)
       (= (log-sum-exp (list -inf.0 -inf.0)) -inf.0)
       (approx=? a (log-num 2.0) 1e-9)
       (approx=? b (log-num 3.0) 1e-9)
       (approx=? shifted base 1e-9)))""",
    "normalize-log-weights": """(let* ([norm (normalize-log-weights (list 0.0 (log-num 2.0) (log-num 3.0)))]
       [probs (map exp-num norm)]
       [sum-p (fold-left + 0.0 probs)]
       [all-neg? (= (length (filter (lambda (x) (<= x 0.0)) norm)) (length norm))]
       [fallback (normalize-log-weights (list -inf.0 -inf.0 -inf.0))]
       [fallback-sum (fold-left + 0.0 fallback)]
       [uniform? (= (length (filter (lambda (x) (approx=? x (/ 1.0 3.0) 1e-9)) fallback)) 3)])
  (and (approx=? sum-p 1.0 1e-9)
       all-neg?
       (approx=? fallback-sum 1.0 1e-9)
       uniform?))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = SOURCE_FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "make-prob": """def make_prob(state_comp):
    return ('prob', state_comp)""",
    "prob?": """def is_prob(x):
    return isinstance(x, tuple) and len(x) > 0 and x[0] == 'prob'""",
    "run-prob": """def run_prob(p, prng):
    value, final_state = run_state(prob_state(p), (prng, 0.0))
    final_prng, log_w = final_state
    return (value, log_w), final_prng""",
    "sample-prob": """def sample_prob(p, prng):
    return run_prob(p, prng)[0][0]""",
    "weight-prob": """def weight_prob(p, prng):
    return run_prob(p, prng)[0][1]""",
    "prob-bind": """def prob_bind(p, f):
    return make_prob(
        state_bind(prob_state(p), lambda a: prob_state(f(a)))
    )""",
    "log-sum-exp": """def log_sum_exp(xs):
    if len(xs) == 0:
        return float('-inf')
    m = max(xs)
    if m == float('-inf'):
        return float('-inf')
    return m + log(sum(exp(x - m) for x in xs))""",
    "normalize-log-weights": """def normalize_log_weights(log_ws):
    total = log_sum_exp(log_ws)
    if total == float('-inf'):
        return [1.0 / len(log_ws) for _ in log_ws]
    return [lw - total for lw in log_ws]""",
}

CHEZ_SNIPPETS = {
    "make-prob": """(define (wrap-prob state-comp)
  (cons 'prob state-comp))""",
    "prob?": """(define (probability? x)
  (and (pair? x)
       (eq? (car x) 'prob)))""",
    "run-prob": """(define (execute-prob p prng)
  (let ([result (run-state (prob-state p) (cons prng 0.0))])
    (let ([value (car result)]
          [final-state (cdr result)])
      (cons (cons value (cdr final-state))
            (car final-state)))))""",
    "sample-prob": """(define (prob-sample p prng)
  (car (car (run-prob p prng))))""",
    "weight-prob": """(define (prob-weight p prng)
  (cdr (car (run-prob p prng))))""",
    "prob-bind": """(define (bind-prob p f)
  (make-prob
   (state-bind (prob-state p)
               (lambda (a)
                 (prob-state (f a))))))""",
    "log-sum-exp": """(define (lse xs)
  (if (null? xs)
      -inf.0
      (let ([mx (apply max xs)])
        (if (= mx -inf.0)
            -inf.0
            (+ mx (log-num (fold-left + 0 (map (lambda (x) (exp-num (- x mx))) xs))))))))""",
    "normalize-log-weights": """(define (normalize-lw log-ws)
  (let ([total (log-sum-exp log-ws)])
    (if (= total -inf.0)
        (map (lambda (_) (/ 1.0 (length log-ws))) log-ws)
        (map (lambda (lw) (- lw total)) log-ws))))""",
}

BUGGY_CASES = [
    {
        "fn": "make-prob",
        "buggy": """(define (make-prob state-comp)
  (list 'prob state-comp))""",
        "note": "The representation must be a pair/tagged cons, not a list with an extra nesting layer.",
    },
    {
        "fn": "make-prob",
        "buggy": """(define (make-prob state-comp)
  (cons 'probability state-comp))""",
        "note": "The tag symbol must be exactly 'prob for predicate compatibility.",
    },
    {
        "fn": "prob?",
        "buggy": """(define (prob? x)
  (and (pair? x) (eq? (car x) 'state)))""",
        "note": "The predicate should recognize 'prob values, not state wrappers.",
    },
    {
        "fn": "prob?",
        "buggy": """(define (prob? x)
  (and (pair? x)
       (or (eq? (car x) 'prob)
           (eq? (car x) 'state))))""",
        "note": "The predicate is too permissive: only the 'prob tag is valid.",
    },
    {
        "fn": "run-prob",
        "buggy": """(define (run-prob p prng)
  (let ([result (run-state (prob-state p) (cons prng 1.0))])
       (let ([value (car result)]
             [final-state (cdr result)])
            (cons (cons value (cdr final-state))
                  (car final-state)))))""",
        "note": "Probability computations must start with zero log-weight, not 1.0.",
    },
    {
        "fn": "run-prob",
        "buggy": """(define (run-prob p prng)
  (let ([result (run-state (prob-state p) (cons prng 0.0))])
       (let ([value (car result)]
             [final-state (cdr result)])
            (cons (cons value (car final-state))
                  (cdr final-state)))))""",
        "note": "The output pair structure is swapped; preserve ((value . log-weight) . new-prng).",
    },
    {
        "fn": "sample-prob",
        "buggy": """(define (sample-prob p prng)
  (cdr (car (run-prob p prng))))""",
        "note": "This extracts weight instead of sampled value.",
    },
    {
        "fn": "sample-prob",
        "buggy": """(define (sample-prob p prng)
  (car (run-prob p prng)))""",
        "note": "This returns the (value . weight) pair, not the sample value itself.",
    },
    {
        "fn": "weight-prob",
        "buggy": """(define (weight-prob p prng)
  (car (car (run-prob p prng))))""",
        "note": "This returns value; weight-prob must project the log-weight field.",
    },
    {
        "fn": "weight-prob",
        "buggy": """(define (weight-prob p prng)
  (cdr (run-prob p prng)))""",
        "note": "This returns the new PRNG state instead of log-weight.",
    },
    {
        "fn": "prob-bind",
        "buggy": """(define (prob-bind p f)
  (make-prob
   (state-bind (prob-state p)
               (lambda (a)
                       (f a)))))""",
        "note": "state-bind expects the continuation to return state computations; extract prob-state from f(a).",
    },
    {
        "fn": "prob-bind",
        "buggy": """(define (prob-bind p f)
  (make-prob
   (state-bind (prob-state p)
               (lambda (a)
                       (prob-state (f p))))))""",
        "note": "The continuation must apply f to the bound value a, not to the original computation p.",
    },
    {
        "fn": "log-sum-exp",
        "buggy": """(define (log-sum-exp xs)
  (if (null? xs)
      -inf.0
      (log-num (fold-left + 0 (map exp-num xs)))))""",
        "note": "The implementation must use max-subtraction for numerical stability.",
    },
    {
        "fn": "log-sum-exp",
        "buggy": """(define (log-sum-exp xs)
  (if (null? xs)
      -inf.0
      (let ([max-x (apply max xs)])
           (if (= max-x -inf.0)
               0.0
               (+ max-x (log-num (fold-left + 0 (map (lambda (x) (exp-num (- x max-x))) xs))))))))""",
        "note": "All -inf inputs should return -inf.0, not 0.0.",
    },
    {
        "fn": "normalize-log-weights",
        "buggy": """(define (normalize-log-weights log-ws)
  (let ([total (log-sum-exp log-ws)])
       (if (= total -inf.0)
           (map (lambda (_) (/ 1.0 (length log-ws))) log-ws)
           (map (lambda (lw) (+ lw total)) log-ws))))""",
        "note": "Normalization should subtract total log mass, not add it.",
    },
    {
        "fn": "normalize-log-weights",
        "buggy": """(define (normalize-log-weights log-ws)
  (let ([total (log-sum-exp log-ws)])
       (if (= total -inf.0)
           log-ws
           (map (lambda (lw) (- lw total)) log-ws))))""",
        "note": "When all entries are impossible (-inf), return uniform fallback probabilities.",
    },
]

DIFFICULTY = {
    "make-prob": "easy",
    "prob?": "easy",
    "run-prob": "medium",
    "sample-prob": "medium",
    "weight-prob": "medium",
    "prob-bind": "hard",
    "log-sum-exp": "hard",
    "normalize-log-weights": "hard",
}

REQUIRED_KEYS = {
    "id",
    "family",
    "category",
    "difficulty",
    "source_module",
    "source_test",
    "source_function",
    "prompt",
    "ground_truth",
    "verify_expr",
    "tags",
}

samples: List[Dict[str, object]] = []
family_counter: Dict[str, int] = defaultdict(int)


def add_sample(
    family: str,
    category: str,
    difficulty: str,
    source_function: str,
    prompt: str,
    ground_truth: str,
    verify_expr: str,
    tags: List[str],
) -> None:
    family_counter[family] += 1
    sid = f"random_probability_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
        "prompt_body": prompt.strip(),
        "prompt": diversify_prompt(
            prompt.strip(),
            family,
            source_function,
            family_counter[family],
            category,
            verify_expr,
        ),
        "ground_truth": ground_truth.strip(),
        "verify_expr": verify_expr.strip(),
        "tags": tags,
    }
    for key in REQUIRED_KEYS:
        if key not in sample:
            raise ValueError(f"missing key {key}")
    samples.append(sample)


def verify_refs(verify_expr: str) -> List[str]:
    tokens = set(TOKEN_RE.findall(verify_expr))
    names = SOURCE_FUNCTION_ORDER + SUPPORT_ORDER
    return [name for name in names if name in tokens]


def dependency_closure(roots: List[str]) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        for dep in DEPENDS.get(name, []):
            visit(dep)
        if name in ALL_DEFS:
            ordered.append(name)

    for root in roots:
        visit(root)

    return ordered


def build_verify(verify_check: str, roots: List[str] | None = None) -> str:
    wanted: List[str] = []
    for root in roots or []:
        if root not in wanted:
            wanted.append(root)
    for ref in verify_refs(verify_check):
        if ref not in wanted:
            wanted.append(ref)

    defs_needed = dependency_closure(wanted)
    parts = [ALL_DEFS[name] for name in defs_needed] + [verify_check.strip()]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


def def_verify(fn: str) -> str:
    return build_verify(VERIFY_BY_FUNCTION[fn], [fn])


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in SOURCE_FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement this function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "random", "probability", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Complete this Fold Scheme skeleton.

Module: {SOURCE_MODULE}
Function target: `{fn}`
Behavior contract: {FUNCTION_SPECS[fn]}

```scheme
{SKELETONS[fn]}
```

Output only the completed function definition.""",
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "random", "probability", "skeleton", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (16)
# -----------------------------------------------------------------------------
for fn in TRANSLATION_FUNCTIONS:
    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Translate the following Python function into Fold-native Scheme.
Preserve behavior exactly.

Target function name: `{fn}`

```python
{PYTHON_SNIPPETS[fn]}
```

Return only the Scheme definition.""",
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "random", "probability", "translation", "python", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Convert this Chez-style snippet to canonical Fold style.
Keep semantics identical.

Target function: `{fn}`

```scheme
{CHEZ_SNIPPETS[fn]}
```

Return only Fold code.""",
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "random", "probability", "translation", "chez", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
# -----------------------------------------------------------------------------
if len(BUGGY_CASES) != 16:
    raise ValueError(f"expected 16 bugfix cases, got {len(BUGGY_CASES)}")

for case in BUGGY_CASES:
    fn = str(case["fn"])
    add_sample(
        family="bugfix",
        category="debugging",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Fix the bug in this Fold Scheme function with minimal semantic changes.
Target: `{fn}` in `{SOURCE_MODULE}`.
Known issue: {case['note']}

```scheme
{case['buggy']}
```

Return only the corrected definition.""",
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "random", "probability", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition/use (32)
# -----------------------------------------------------------------------------
def add_composition(
    source_function: str,
    prompt: str,
    ground_truth: str,
    verify_check: str,
    difficulty: str,
    extra_tags: List[str],
) -> None:
    add_sample(
        family="composition",
        category="usage",
        difficulty=difficulty,
        source_function=source_function,
        prompt=prompt,
        ground_truth=ground_truth,
        verify_expr=build_verify(verify_check, [source_function]),
        tags=["tier1", "random", "probability", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # make-prob
    {
        "fn": "make-prob",
        "prompt": "Wrap `(state-pure 12)` with `make-prob`, run it, and return the sampled value.",
        "gt": "(car (car (run-prob (make-prob (state-pure 12)) 's0)))",
        "verify": "(equal? (car (car (run-prob (make-prob (state-pure 12)) 's0))) 12)",
        "difficulty": "easy",
        "tags": ["state-wrap"],
    },
    {
        "fn": "make-prob",
        "prompt": "Return whether `make-prob (state-pure 5)` and `prob-pure 5` produce identical `run-prob` outputs.",
        "gt": "(equal? (run-prob (make-prob (state-pure 5)) 'seed-a) (run-prob (prob-pure 5) 'seed-a))",
        "verify": "(equal? (equal? (run-prob (make-prob (state-pure 5)) 'seed-a) (run-prob (prob-pure 5) 'seed-a)) #t)",
        "difficulty": "medium",
        "tags": ["equivalence"],
    },
    {
        "fn": "make-prob",
        "prompt": "Build a custom weighted computation with `make-prob` and return its resulting weight.",
        "gt": "(cdr (car (run-prob (make-prob (make-state (lambda (ps) (cons 'ok (cons (car ps) (+ (cdr ps) 2.0)))))) 'seed-b)))",
        "verify": "(approx=? (cdr (car (run-prob (make-prob (make-state (lambda (ps) (cons 'ok (cons (car ps) (+ (cdr ps) 2.0)))))) 'seed-b))) 2.0 1e-12)",
        "difficulty": "medium",
        "tags": ["weighted-state"],
    },
    {
        "fn": "make-prob",
        "prompt": "Use `make-prob` plus `state-bind` to square 3 and sample the resulting value.",
        "gt": "(sample-prob (make-prob (state-bind (state-pure 3) (lambda (x) (state-pure (* x x))))) 'seed-c)",
        "verify": "(equal? (sample-prob (make-prob (state-bind (state-pure 3) (lambda (x) (state-pure (* x x))))) 'seed-c) 9)",
        "difficulty": "hard",
        "tags": ["state-bind"],
    },

    # prob?
    {
        "fn": "prob?",
        "prompt": "Count how many values in a mixed list are recognized by `prob?`.",
        "gt": "(length (filter prob? (list (make-prob (state-pure 1)) '(prob . bogus) '(state . 2) 99)))",
        "verify": "(equal? (length (filter prob? (list (make-prob (state-pure 1)) '(prob . bogus) '(state . 2) 99))) 2)",
        "difficulty": "easy",
        "tags": ["filter"],
    },
    {
        "fn": "prob?",
        "prompt": "Return whether `prob-bind` preserves the probability wrapper tag.",
        "gt": "(prob? (prob-bind (prob-pure 1) (lambda (x) (prob-pure (+ x 1)))))",
        "verify": "(equal? (prob? (prob-bind (prob-pure 1) (lambda (x) (prob-pure (+ x 1))))) #t)",
        "difficulty": "medium",
        "tags": ["bind-tag"],
    },
    {
        "fn": "prob?",
        "prompt": "Map `prob?` across mixed values and return the boolean result vector.",
        "gt": "(map prob? (list (make-prob (state-pure 'a)) '(prob . 1) 'prob (cons 'x 1)))",
        "verify": "(equal? (map prob? (list (make-prob (state-pure 'a)) '(prob . 1) 'prob (cons 'x 1))) '(#t #t #f #f))",
        "difficulty": "medium",
        "tags": ["map"],
    },
    {
        "fn": "prob?",
        "prompt": "Check that `run-prob` output is not itself a tagged probability value.",
        "gt": "(not (prob? (run-prob (prob-pure 7) 'seed-d)))",
        "verify": "(equal? (not (prob? (run-prob (prob-pure 7) 'seed-d))) #t)",
        "difficulty": "medium",
        "tags": ["projection"],
    },

    # run-prob
    {
        "fn": "run-prob",
        "prompt": "Run `(prob-pure 42)` with seed marker `'seed-r1` and return the full result pair.",
        "gt": "(run-prob (prob-pure 42) 'seed-r1)",
        "verify": "(equal? (run-prob (prob-pure 42) 'seed-r1) (cons (cons 42 0.0) 'seed-r1))",
        "difficulty": "easy",
        "tags": ["shape"],
    },
    {
        "fn": "run-prob",
        "prompt": "Run a custom transition and return the final PRNG marker from `run-prob`.",
        "gt": "(cdr (run-prob (make-prob (make-state (lambda (ps) (cons 'v (cons 'next-r (+ (cdr ps) 0.25)))))) 'seed-r2))",
        "verify": "(equal? (cdr (run-prob (make-prob (make-state (lambda (ps) (cons 'v (cons 'next-r (+ (cdr ps) 0.25)))))) 'seed-r2)) 'next-r)",
        "difficulty": "medium",
        "tags": ["state-thread"],
    },
    {
        "fn": "run-prob",
        "prompt": "Return whether `run-prob` is deterministic for identical computation and seed.",
        "gt": "(equal? (run-prob (prob-bind (prob-pure 2) (lambda (x) (prob-pure (+ x 8)))) 'seed-r3) (run-prob (prob-bind (prob-pure 2) (lambda (x) (prob-pure (+ x 8)))) 'seed-r3))",
        "verify": "(equal? (equal? (run-prob (prob-bind (prob-pure 2) (lambda (x) (prob-pure (+ x 8)))) 'seed-r3) (run-prob (prob-bind (prob-pure 2) (lambda (x) (prob-pure (+ x 8)))) 'seed-r3)) #t)",
        "difficulty": "medium",
        "tags": ["determinism"],
    },
    {
        "fn": "run-prob",
        "prompt": "Use `run-prob` and return `(value . weight)` for a custom weighted computation.",
        "gt": "(let ([r (run-prob (make-prob (make-state (lambda (ps) (cons 11 (cons 'seed-r4 (+ (cdr ps) -0.5)))))) 'seed-r4)]) (cons (car (car r)) (cdr (car r))))",
        "verify": "(equal? (let ([r (run-prob (make-prob (make-state (lambda (ps) (cons 11 (cons 'seed-r4 (+ (cdr ps) -0.5)))))) 'seed-r4)]) (cons (car (car r)) (cdr (car r)))) (cons 11 -0.5))",
        "difficulty": "hard",
        "tags": ["value-weight"],
    },

    # sample-prob
    {
        "fn": "sample-prob",
        "prompt": "Sample from a bound deterministic computation that doubles and increments.",
        "gt": "(sample-prob (prob-bind (prob-pure 10) (lambda (x) (prob-pure (+ (* 2 x) 1)))) 'seed-s1)",
        "verify": "(equal? (sample-prob (prob-bind (prob-pure 10) (lambda (x) (prob-pure (+ (* 2 x) 1)))) 'seed-s1) 21)",
        "difficulty": "easy",
        "tags": ["bind"],
    },
    {
        "fn": "sample-prob",
        "prompt": "Use `sample-prob` to return two sampled values from two independent pure computations.",
        "gt": "(list (sample-prob (prob-pure 'a) 'seed-s2) (sample-prob (prob-pure 'b) 'seed-s2))",
        "verify": "(equal? (list (sample-prob (prob-pure 'a) 'seed-s2) (sample-prob (prob-pure 'b) 'seed-s2)) '(a b))",
        "difficulty": "easy",
        "tags": ["multi"],
    },
    {
        "fn": "sample-prob",
        "prompt": "Check that `sample-prob` matches projecting the value field out of `run-prob`.",
        "gt": "(let ([p (make-prob (make-state (lambda (ps) (cons 3.5 (cons (car ps) (+ (cdr ps) 1.0))))))]) (= (sample-prob p 'seed-s3) (car (car (run-prob p 'seed-s3)))))",
        "verify": "(equal? (let ([p (make-prob (make-state (lambda (ps) (cons 3.5 (cons (car ps) (+ (cdr ps) 1.0))))))]) (= (sample-prob p 'seed-s3) (car (car (run-prob p 'seed-s3))))) #t)",
        "difficulty": "medium",
        "tags": ["projection"],
    },
    {
        "fn": "sample-prob",
        "prompt": "Map `sample-prob` over a list of deterministic probability computations.",
        "gt": "(map (lambda (p) (sample-prob p 'seed-s4)) (list (prob-pure 1) (prob-pure 2) (prob-pure 3)))",
        "verify": "(equal? (map (lambda (p) (sample-prob p 'seed-s4)) (list (prob-pure 1) (prob-pure 2) (prob-pure 3))) '(1 2 3))",
        "difficulty": "medium",
        "tags": ["map"],
    },

    # weight-prob
    {
        "fn": "weight-prob",
        "prompt": "Return weights for a pure computation and a weighted custom computation as a two-element list.",
        "gt": "(let ([p1 (prob-pure 'x)] [p2 (make-prob (make-state (lambda (ps) (cons 'x (cons (car ps) (+ (cdr ps) -2.0))))))]) (list (weight-prob p1 'seed-w1) (weight-prob p2 'seed-w1)))",
        "verify": "(equal? (let ([p1 (prob-pure 'x)] [p2 (make-prob (make-state (lambda (ps) (cons 'x (cons (car ps) (+ (cdr ps) -2.0))))))]) (list (weight-prob p1 'seed-w1) (weight-prob p2 'seed-w1))) '(0.0 -2.0))",
        "difficulty": "medium",
        "tags": ["comparison"],
    },
    {
        "fn": "weight-prob",
        "prompt": "Verify `weight-prob` matches the weight projection from `run-prob` for a custom computation.",
        "gt": "(let ([p (make-prob (make-state (lambda (ps) (cons 99 (cons 'seed-w2 (+ (cdr ps) 0.75))))))]) (approx=? (weight-prob p 'seed-w2) (cdr (car (run-prob p 'seed-w2))) 1e-12))",
        "verify": "(equal? (let ([p (make-prob (make-state (lambda (ps) (cons 99 (cons 'seed-w2 (+ (cdr ps) 0.75))))))]) (approx=? (weight-prob p 'seed-w2) (cdr (car (run-prob p 'seed-w2))) 1e-12)) #t)",
        "difficulty": "medium",
        "tags": ["projection"],
    },
    {
        "fn": "weight-prob",
        "prompt": "Bind into a weighted inner computation and return the resulting log-weight.",
        "gt": "(weight-prob (prob-bind (prob-pure 5) (lambda (x) (make-prob (make-state (lambda (ps) (cons (+ x 1) (cons (car ps) (+ (cdr ps) -0.25)))))))) 'seed-w3)",
        "verify": "(approx=? (weight-prob (prob-bind (prob-pure 5) (lambda (x) (make-prob (make-state (lambda (ps) (cons (+ x 1) (cons (car ps) (+ (cdr ps) -0.25)))))))) 'seed-w3) -0.25 1e-12)",
        "difficulty": "hard",
        "tags": ["bind-weight"],
    },
    {
        "fn": "weight-prob",
        "prompt": "Return whether chaining two weighted steps accumulates to total log-weight 1.0.",
        "gt": "(let ([p (prob-bind (make-prob (make-state (lambda (ps) (cons 4 (cons (car ps) (+ (cdr ps) 1.5)))))) (lambda (x) (make-prob (make-state (lambda (ps) (cons x (cons (car ps) (+ (cdr ps) -0.5))))))))]) (approx=? (weight-prob p 'seed-w4) 1.0 1e-12))",
        "verify": "(equal? (let ([p (prob-bind (make-prob (make-state (lambda (ps) (cons 4 (cons (car ps) (+ (cdr ps) 1.5)))))) (lambda (x) (make-prob (make-state (lambda (ps) (cons x (cons (car ps) (+ (cdr ps) -0.5))))))))]) (approx=? (weight-prob p 'seed-w4) 1.0 1e-12)) #t)",
        "difficulty": "hard",
        "tags": ["accumulation"],
    },

    # prob-bind
    {
        "fn": "prob-bind",
        "prompt": "Chain two pure transforms with `prob-bind` and sample the final value.",
        "gt": "(sample-prob (prob-bind (prob-pure 3) (lambda (x) (prob-bind (prob-pure (+ x 2)) (lambda (y) (prob-pure (* y 4)))))) 'seed-b1)",
        "verify": "(equal? (sample-prob (prob-bind (prob-pure 3) (lambda (x) (prob-bind (prob-pure (+ x 2)) (lambda (y) (prob-pure (* y 4)))))) 'seed-b1) 20)",
        "difficulty": "medium",
        "tags": ["nested-bind"],
    },
    {
        "fn": "prob-bind",
        "prompt": "Check the left-identity law for `prob-bind` with `x -> x+1` at x=7.",
        "gt": "(equal? (run-prob (prob-bind (prob-pure 7) (lambda (x) (prob-pure (+ x 1)))) 'seed-b2) (run-prob ((lambda (x) (prob-pure (+ x 1))) 7) 'seed-b2))",
        "verify": "(equal? (equal? (run-prob (prob-bind (prob-pure 7) (lambda (x) (prob-pure (+ x 1)))) 'seed-b2) (run-prob ((lambda (x) (prob-pure (+ x 1))) 7) 'seed-b2)) #t)",
        "difficulty": "hard",
        "tags": ["law-left-identity"],
    },
    {
        "fn": "prob-bind",
        "prompt": "Check associativity of `prob-bind` with stateful first and second stages.",
        "gt": "(let* ([m (make-prob (make-state (lambda (ps) (cons 2 (cons (car ps) (+ (cdr ps) 0.1))))))] [f (lambda (x) (make-prob (make-state (lambda (ps) (cons (+ x 3) (cons (car ps) (+ (cdr ps) 0.2)))))))] [g (lambda (y) (prob-pure (* y 5)))]) (equal? (run-prob (prob-bind (prob-bind m f) g) 'seed-b3) (run-prob (prob-bind m (lambda (x) (prob-bind (f x) g))) 'seed-b3)))",
        "verify": "(equal? (let* ([m (make-prob (make-state (lambda (ps) (cons 2 (cons (car ps) (+ (cdr ps) 0.1))))))] [f (lambda (x) (make-prob (make-state (lambda (ps) (cons (+ x 3) (cons (car ps) (+ (cdr ps) 0.2)))))))] [g (lambda (y) (prob-pure (* y 5)))]) (equal? (run-prob (prob-bind (prob-bind m f) g) 'seed-b3) (run-prob (prob-bind m (lambda (x) (prob-bind (f x) g))) 'seed-b3))) #t)",
        "difficulty": "hard",
        "tags": ["law-associativity"],
    },
    {
        "fn": "prob-bind",
        "prompt": "Use `prob-bind` to transform a value while preserving PRNG threading from the first stage; return `(value . new-prng)`.",
        "gt": "(let* ([m (make-prob (make-state (lambda (ps) (cons 4 (cons 'seed-b4-next (+ (cdr ps) 1.0))))))] [r (run-prob (prob-bind m (lambda (x) (prob-pure (+ x 6)))) 'seed-b4)]) (cons (car (car r)) (cdr r)))",
        "verify": "(equal? (let* ([m (make-prob (make-state (lambda (ps) (cons 4 (cons 'seed-b4-next (+ (cdr ps) 1.0))))))] [r (run-prob (prob-bind m (lambda (x) (prob-pure (+ x 6)))) 'seed-b4)]) (cons (car (car r)) (cdr r))) (cons 10 'seed-b4-next))",
        "difficulty": "hard",
        "tags": ["threading"],
    },

    # log-sum-exp
    {
        "fn": "log-sum-exp",
        "prompt": "Compute a softmax denominator in log-space from logits `(2, 1, 0)` after subtracting 2 from each logit.",
        "gt": "(log-sum-exp (map (lambda (x) (- x 2.0)) '(2.0 1.0 0.0)))",
        "verify": "(approx=? (log-sum-exp (map (lambda (x) (- x 2.0)) '(2.0 1.0 0.0))) (log-num (+ 1.0 (exp-num -1.0) (exp-num -2.0))) 1e-9)",
        "difficulty": "medium",
        "tags": ["softmax"],
    },
    {
        "fn": "log-sum-exp",
        "prompt": "Return whether `log-sum-exp` is shift-invariant for vectors `(0,-1,-2)` and `(5,4,3)`.",
        "gt": "(approx=? (- (log-sum-exp '(5.0 4.0 3.0)) 5.0) (log-sum-exp '(0.0 -1.0 -2.0)) 1e-9)",
        "verify": "(equal? (approx=? (- (log-sum-exp '(5.0 4.0 3.0)) 5.0) (log-sum-exp '(0.0 -1.0 -2.0)) 1e-9) #t)",
        "difficulty": "hard",
        "tags": ["invariance"],
    },
    {
        "fn": "log-sum-exp",
        "prompt": "Exponentiate `log-sum-exp` of normalized log-probabilities and return the recovered mass.",
        "gt": "(exp-num (log-sum-exp (list (log-num 0.2) (log-num 0.3) (log-num 0.5))))",
        "verify": "(approx=? (exp-num (log-sum-exp (list (log-num 0.2) (log-num 0.3) (log-num 0.5)))) 1.0 1e-9)",
        "difficulty": "medium",
        "tags": ["recovery"],
    },
    {
        "fn": "log-sum-exp",
        "prompt": "Normalize log-weights, exponentiate, and return the resulting probability sum.",
        "gt": "(let* ([logs '(0.0 -2.0 -4.0)] [norm (normalize-log-weights logs)]) (log-sum-exp norm))",
        "verify": "(approx=? (let* ([logs '(0.0 -2.0 -4.0)] [norm (normalize-log-weights logs)]) (log-sum-exp norm)) 0.0 1e-9)",
        "difficulty": "hard",
        "tags": ["integration"],
    },

    # normalize-log-weights
    {
        "fn": "normalize-log-weights",
        "prompt": "Use `normalize-log-weights` on ratios 1:2:3 and return linear-space probabilities.",
        "gt": "(map exp-num (normalize-log-weights (list 0.0 (log-num 2.0) (log-num 3.0))))",
        "verify": "(let ([p (map exp-num (normalize-log-weights (list 0.0 (log-num 2.0) (log-num 3.0))))]) (and (approx=? (car p) (/ 1.0 6.0) 1e-9) (approx=? (cadr p) (/ 2.0 6.0) 1e-9) (approx=? (caddr p) (/ 3.0 6.0) 1e-9)))",
        "difficulty": "medium",
        "tags": ["ratios"],
    },
    {
        "fn": "normalize-log-weights",
        "prompt": "Return whether `normalize-log-weights` is invariant to adding a constant shift to every log-weight.",
        "gt": "(let ([a (normalize-log-weights '(0.0 -1.0 -2.0))] [b (normalize-log-weights '(3.0 2.0 1.0))]) (and (approx=? (car a) (car b) 1e-9) (approx=? (cadr a) (cadr b) 1e-9) (approx=? (caddr a) (caddr b) 1e-9)))",
        "verify": "(equal? (let ([a (normalize-log-weights '(0.0 -1.0 -2.0))] [b (normalize-log-weights '(3.0 2.0 1.0))]) (and (approx=? (car a) (car b) 1e-9) (approx=? (cadr a) (cadr b) 1e-9) (approx=? (caddr a) (caddr b) 1e-9))) #t)",
        "difficulty": "hard",
        "tags": ["shift-invariance"],
    },
    {
        "fn": "normalize-log-weights",
        "prompt": "Use `normalize-log-weights` on all `-inf` log-weights and return the sum of fallback values.",
        "gt": "(fold-left + 0.0 (normalize-log-weights (list -inf.0 -inf.0 -inf.0 -inf.0)))",
        "verify": "(approx=? (fold-left + 0.0 (normalize-log-weights (list -inf.0 -inf.0 -inf.0 -inf.0))) 1.0 1e-9)",
        "difficulty": "medium",
        "tags": ["fallback"],
    },
    {
        "fn": "normalize-log-weights",
        "prompt": "Apply `normalize-log-weights` to finite log-weights and return whether their log-sum-exp equals zero.",
        "gt": "(approx=? (log-sum-exp (normalize-log-weights '(2.0 0.0 -1.0))) 0.0 1e-9)",
        "verify": "(equal? (approx=? (log-sum-exp (normalize-log-weights '(2.0 0.0 -1.0))) 0.0 1e-9) #t)",
        "difficulty": "hard",
        "tags": ["consistency"],
    },
]

for case in composition_cases:
    add_composition(
        source_function=str(case["fn"]),
        prompt=str(case["prompt"]),
        ground_truth=str(case["gt"]),
        verify_check=str(case["verify"]),
        difficulty=str(case["difficulty"]),
        extra_tags=list(case["tags"]),
    )

if sum(1 for s in samples if s["family"] == "composition") != 32:
    raise ValueError("composition family must contain exactly 32 samples")

# -----------------------------------------------------------------------------
# Split train/eval
# -----------------------------------------------------------------------------
if len(samples) != 80:
    raise ValueError(f"expected 80 samples, got {len(samples)}")

by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

EVAL_QUOTA = {
    "spec_to_code": 3,
    "translation": 3,
    "bugfix": 3,
    "composition": 5,
}


def spread_indices(n: int, k: int) -> Set[int]:
    if k <= 0:
        return set()
    if k >= n:
        return set(range(n))
    if k == 1:
        return {n // 2}
    idxs = {round(i * (n - 1) / (k - 1)) for i in range(k)}
    cursor = 0
    while len(idxs) < k:
        if cursor not in idxs:
            idxs.add(cursor)
        cursor += 1
    return idxs


eval_ids: Set[str] = set()
for family, family_samples in by_family.items():
    picked = spread_indices(len(family_samples), EVAL_QUOTA[family])
    for i, s in enumerate(family_samples):
        if i in picked:
            eval_ids.add(str(s["id"]))

id_to_sample: Dict[str, Dict[str, object]] = {str(s["id"]): s for s in samples}
all_source_functions = sorted({str(s["source_function"]) for s in samples})


def eval_source_fn_counts(ids: Set[str]) -> Counter:
    return Counter(str(id_to_sample[sid]["source_function"]) for sid in ids)


changed = True
while changed:
    changed = False
    fn_counts = eval_source_fn_counts(eval_ids)
    missing_functions = [fn for fn in all_source_functions if fn_counts[fn] == 0]
    if not missing_functions:
        break

    for fn in missing_functions:
        candidates = [s for s in samples if str(s["source_function"]) == fn and str(s["id"]) not in eval_ids]
        swapped = False
        for cand in candidates:
            fam = str(cand["family"])
            fam_eval = [id_to_sample[sid] for sid in eval_ids if str(id_to_sample[sid]["family"]) == fam]
            removable = [r for r in fam_eval if fn_counts[str(r["source_function"])] > 1]
            if not removable:
                continue
            removable.sort(
                key=lambda r: (fn_counts[str(r["source_function"])], str(r["id"])),
                reverse=True,
            )
            out = removable[0]
            eval_ids.remove(str(out["id"]))
            eval_ids.add(str(cand["id"]))
            changed = True
            swapped = True
            break
        if swapped:
            break

missing_after = [fn for fn in all_source_functions if eval_source_fn_counts(eval_ids)[fn] == 0]
if missing_after:
    raise ValueError(f"eval split is missing source functions: {missing_after}")

train_rows: List[Dict[str, object]] = []
eval_rows: List[Dict[str, object]] = []
all_rows: List[Dict[str, object]] = []
for s in samples:
    row = dict(s)
    row["split"] = "eval" if s["id"] in eval_ids else "train"
    all_rows.append(row)
    if row["split"] == "eval":
        eval_rows.append(row)
    else:
        train_rows.append(row)

if len(train_rows) != 66 or len(eval_rows) != 14:
    raise ValueError(f"split mismatch: train={len(train_rows)}, eval={len(eval_rows)}")


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


write_jsonl(ALL_PATH, all_rows)
write_jsonl(TRAIN_PATH, train_rows)
write_jsonl(EVAL_PATH, eval_rows)

summary = {
    "total": len(samples),
    "train": len(train_rows),
    "eval": len(eval_rows),
    "families": {
        fam: {
            "total": len(group),
            "eval": sum(1 for x in group if x["id"] in eval_ids),
            "train": sum(1 for x in group if x["id"] not in eval_ids),
        }
        for fam, group in sorted(by_family.items())
    },
    "difficulty": dict(sorted(Counter(str(s["difficulty"]) for s in samples).items())),
    "source_functions": dict(sorted(Counter(str(s["source_function"]) for s in samples).items())),
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
