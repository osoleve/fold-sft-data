#!/usr/bin/env python3
"""Generate Tier-1 FP meta combinators SFT samples for lattice/fp/meta/combinators.ss."""

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

SOURCE_MODULE = "lattice/fp/meta/combinators.ss"
SOURCE_TEST = "lattice/fp/meta/test-combinators.ss"

DEFS: Dict[str, str] = {
    "compose": """(define (compose . fns)
  (if (null? fns)
      id
      (fold-right compose2 id fns)))""",
    "pipe": """(define (pipe . fns)
  (if (null? fns)
      id
      (fold-left pipe2 id fns)))""",
    "curry2": """(define (curry2 f)
  (lambda (x)
          (lambda (y)
                  (f x y))))""",
    "partial": """(define (partial f x)
  (lambda args (apply f (cons x args))))""",
    "maybe-bind": """(define (maybe-bind m f)
  (if (just? m)
      (f (from-just m))
      nothing))""",
    "sequence-maybe": """(define (sequence-maybe ms)
  (if (null? ms)
      (just '())
      (maybe-bind (car ms)
                  (lambda (x)
                          (maybe-bind (sequence-maybe (cdr ms))
                                      (lambda (xs)
                                              (just (cons x xs))))))))""",
    "either-bind": """(define (either-bind e f)
  (if (right? e)
      (f (from-right e))
      e))""",
    "group-by": """(define (group-by key xs)
  (if (null? xs)
      '()
      (let loop ([xs (cdr xs)]
                 [current-key (key (car xs))]
                 [current-group (list (car xs))]
                 [groups '()])
           (if (null? xs)
               (reverse (cons (reverse current-group) groups))
               (let ([k (key (car xs))])
                    (if (equal? k current-key)
                        (loop (cdr xs) current-key (cons (car xs) current-group) groups)
                        (loop (cdr xs) k (list (car xs)) (cons (reverse current-group) groups))))))))""",
}

FUNCTION_ORDER = [
    "compose",
    "pipe",
    "curry2",
    "partial",
    "maybe-bind",
    "sequence-maybe",
    "either-bind",
    "group-by",
]

FUNCTION_SPECS = {
    "compose": "Compose functions right-to-left. Empty input must produce identity.",
    "pipe": "Compose functions left-to-right. Empty input must produce identity.",
    "curry2": "Transform a binary function into a curried unary-unary chain.",
    "partial": "Partially apply the first argument while preserving variadic tail arguments.",
    "maybe-bind": "Monadic bind for Maybe: apply continuation on Just, propagate Nothing unchanged.",
    "sequence-maybe": "Convert a list of Maybe values into Maybe of list with short-circuit on Nothing.",
    "either-bind": "Monadic bind for Either on the Right branch; Left passes through unchanged.",
    "group-by": "Group consecutive elements by key equality while preserving encounter order within groups.",
}

SKELETONS = {
    "compose": """(define (compose . fns)
  ;; TODO: compose functions right-to-left, defaulting to identity
  <TODO>)""",
    "pipe": """(define (pipe . fns)
  ;; TODO: compose functions left-to-right, defaulting to identity
  <TODO>)""",
    "curry2": """(define (curry2 f)
  ;; TODO: return curried form of a binary function
  <TODO>)""",
    "partial": """(define (partial f x)
  ;; TODO: pre-apply x and forward remaining arguments
  <TODO>)""",
    "maybe-bind": """(define (maybe-bind m f)
  ;; TODO: apply f only for Just values; propagate Nothing
  <TODO>)""",
    "sequence-maybe": """(define (sequence-maybe ms)
  ;; TODO: collect list of Maybes into Maybe of list
  <TODO>)""",
    "either-bind": """(define (either-bind e f)
  ;; TODO: bind over Right and preserve Left
  <TODO>)""",
    "group-by": """(define (group-by key xs)
  ;; TODO: group consecutive elements with equal keys
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "compose": "(and (= ((compose (lambda (x) (* x 2)) (lambda (x) (+ x 1))) 5) 12) (= ((compose) 42) 42) (= ((compose (lambda (x) (- x 3)) (lambda (x) (* x 2)) (lambda (x) (+ x 1))) 10) 19))",
    "pipe": "(and (= ((pipe (lambda (x) (+ x 1)) (lambda (x) (* x 2))) 5) 12) (= ((pipe) 42) 42) (= ((pipe (lambda (x) (+ x 1)) (lambda (x) (* x 2)) (lambda (x) (- x 3))) 10) 19))",
    "curry2": "(let ([sub (curry2 (lambda (x y) (- x y)))] [pair (curry2 cons)]) (and (= ((sub 7) 2) 5) (= ((sub 2) 7) -5) (equal? ((pair 'a) 'b) '(a . b))))",
    "partial": "(let ([minus10 (partial - 10)] [prepend (partial cons 'x)]) (and (= (minus10 3) 7) (= (minus10 3 2) 5) (equal? (prepend '(a b)) '(x a b))))",
    "maybe-bind": "(let ([safe-div (lambda (n) (if (= n 0) nothing (just (/ 10 n))))]) (and (equal? (maybe-bind (just 2) safe-div) (just 5)) (equal? (maybe-bind (just 0) safe-div) nothing) (equal? (maybe-bind nothing safe-div) nothing)))",
    "sequence-maybe": "(and (equal? (sequence-maybe (list (just 1) (just 2) (just 3))) (just '(1 2 3))) (equal? (sequence-maybe (list (just 1) nothing (just 3))) nothing) (equal? (sequence-maybe '()) (just '())))",
    "either-bind": "(let ([validate (lambda (x) (if (positive? x) (right (* x 2)) (left 'negative)))]) (and (equal? (either-bind (right 5) validate) (right 10)) (equal? (either-bind (right -5) validate) (left 'negative)) (equal? (either-bind (left 'already-error) validate) (left 'already-error))))",
    "group-by": """(and (equal? (group-by even? '(2 4 1 3 6 8 5)) '((2 4) (1 3) (6 8) (5))) (equal? (group-by even? '()) '()) (equal? (group-by string-length '("a" "b" "cc" "dd" "e")) '(("a" "b") ("cc" "dd") ("e"))))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "compose": """def compose(*fns):
    if not fns:
        return lambda x: x
    def composed(x):
        out = x
        for f in reversed(fns):
            out = f(out)
        return out
    return composed""",
    "pipe": """def pipe(*fns):
    if not fns:
        return lambda x: x
    def piped(x):
        out = x
        for f in fns:
            out = f(out)
        return out
    return piped""",
    "curry2": """def curry2(f):
    return lambda x: (lambda y: f(x, y))""",
    "partial": """def partial(f, x):
    return lambda *args: f(x, *args)""",
    "maybe-bind": """def maybe_bind(m, f):
    if just_q(m):
        return f(from_just(m))
    return nothing""",
    "sequence-maybe": """def sequence_maybe(ms):
    if not ms:
        return just([])
    head, *tail = ms
    if not just_q(head):
        return nothing
    rest = sequence_maybe(tail)
    if not just_q(rest):
        return nothing
    return just([from_just(head)] + from_just(rest))""",
    "either-bind": """def either_bind(e, f):
    if right_q(e):
        return f(from_right(e))
    return e""",
    "group-by": """def group_by(key, xs):
    if not xs:
        return []
    groups = []
    current = [xs[0]]
    current_key = key(xs[0])
    for x in xs[1:]:
        k = key(x)
        if k == current_key:
            current.append(x)
        else:
            groups.append(current)
            current = [x]
            current_key = k
    groups.append(current)
    return groups""",
}

CHEZ_SNIPPETS = {
    "compose": """(define (compose* . fs)
  (if (null? fs)
      identity
      (fold-right compose2 identity fs)))""",
    "pipe": """(define (pipe* . fs)
  (if (null? fs)
      identity
      (fold-left pipe2 identity fs)))""",
    "curry2": """(define (make-curried2 f)
  (lambda (x)
    (lambda (y)
      (f x y))))""",
    "partial": """(define (partial-left f x)
  (lambda args
    (apply f (cons x args))))""",
    "maybe-bind": """(define (bind-maybe m f)
  (if (just? m)
      (f (from-just m))
      nothing))""",
    "sequence-maybe": """(define (sequence-ms ms)
  (if (null? ms)
      (just '())
      (maybe-bind (car ms)
                  (lambda (x)
                    (maybe-bind (sequence-ms (cdr ms))
                                (lambda (xs)
                                  (just (cons x xs))))))))""",
    "either-bind": """(define (bind-either e f)
  (if (right? e)
      (f (from-right e))
      e))""",
    "group-by": """(define (group-by* key xs)
  (if (null? xs)
      '()
      (let loop ([rest (cdr xs)]
                 [current-key (key (car xs))]
                 [current-group (list (car xs))]
                 [groups '()])
        (if (null? rest)
            (reverse (cons (reverse current-group) groups))
            (let ([k (key (car rest))])
              (if (equal? k current-key)
                  (loop (cdr rest) current-key (cons (car rest) current-group) groups)
                  (loop (cdr rest) k (list (car rest)) (cons (reverse current-group) groups))))))))""",
}

BUGGY_CASES = [
    {
        "fn": "compose",
        "buggy": """(define (compose . fns)
  (if (null? fns)
      id
      (fold-left pipe2 id fns)))""",
        "note": "compose must apply functions right-to-left, not left-to-right.",
    },
    {
        "fn": "compose",
        "buggy": """(define (compose . fns)
  (if (null? fns)
      id
      (fold-right pipe2 id fns)))""",
        "note": "compose should use compose2 semantics; this incorrectly threads using pipe2.",
    },
    {
        "fn": "pipe",
        "buggy": """(define (pipe . fns)
  (if (null? fns)
      id
      (fold-right compose2 id fns)))""",
        "note": "pipe must run left-to-right, not right-to-left composition.",
    },
    {
        "fn": "pipe",
        "buggy": """(define (pipe . fns)
  (if (null? fns)
      id
      (fold-left compose2 id fns)))""",
        "note": "pipe should use pipe2 with fold-left to preserve pipeline order.",
    },
    {
        "fn": "curry2",
        "buggy": """(define (curry2 f)
  (lambda (x y)
    (f x y)))""",
        "note": "curry2 must return nested single-argument lambdas, not an uncurried binary lambda.",
    },
    {
        "fn": "curry2",
        "buggy": """(define (curry2 f)
  (lambda (x)
    (lambda (y)
      (f y x))))""",
        "note": "curry2 must preserve argument order (x then y).",
    },
    {
        "fn": "partial",
        "buggy": """(define (partial f x)
  (lambda args (apply f (append args (list x)))))""",
        "note": "partial should fix the first argument, not append it at the end.",
    },
    {
        "fn": "partial",
        "buggy": """(define (partial f x)
  (lambda (y) (f x y)))""",
        "note": "partial must preserve variadic tail arguments, not force a single extra argument.",
    },
    {
        "fn": "maybe-bind",
        "buggy": """(define (maybe-bind m f)
  (if (just? m)
      (just (f (from-just m)))
      nothing))""",
        "note": "maybe-bind must not wrap continuation result in an extra Just layer.",
    },
    {
        "fn": "maybe-bind",
        "buggy": """(define (maybe-bind m f)
  (if (just? m)
      (f (from-just m))
      (f nothing)))""",
        "note": "Nothing branch should propagate nothing directly, not invoke continuation.",
    },
    {
        "fn": "sequence-maybe",
        "buggy": """(define (sequence-maybe ms)
  (if (null? ms)
      nothing
      (maybe-bind (car ms)
                  (lambda (x)
                    (maybe-bind (sequence-maybe (cdr ms))
                                (lambda (xs)
                                  (just (cons x xs))))))))""",
        "note": "Base case for empty list must be (just '()), not nothing.",
    },
    {
        "fn": "sequence-maybe",
        "buggy": """(define (sequence-maybe ms)
  (if (null? ms)
      (just '())
      (just (map from-just ms))))""",
        "note": "sequence-maybe must short-circuit on nothing and avoid unsafe from-just on all elements.",
    },
    {
        "fn": "either-bind",
        "buggy": """(define (either-bind e f)
  (if (left? e)
      (f (from-left e))
      e))""",
        "note": "either-bind must bind on Right values, not Left values.",
    },
    {
        "fn": "either-bind",
        "buggy": """(define (either-bind e f)
  (if (right? e)
      (right (f (from-right e)))
      e))""",
        "note": "either-bind must return continuation result directly; do not wrap it again in Right.",
    },
    {
        "fn": "group-by",
        "buggy": """(define (group-by key xs)
  (if (null? xs)
      '()
      (let loop ([xs (cdr xs)]
                 [current-key (key (car xs))]
                 [current-group (list (car xs))]
                 [groups '()])
           (if (null? xs)
               (reverse (cons (reverse current-group) groups))
               (let ([k (key (car xs))])
                    (if (equal? k current-key)
                        (loop (cdr xs) current-key (cons (car xs) current-group) groups)
                        (loop (cdr xs) current-key (list (car xs)) (cons (reverse current-group) groups))))))))""",
        "note": "On key change, current-key must update to the new key.",
    },
    {
        "fn": "group-by",
        "buggy": """(define (group-by key xs)
  (if (null? xs)
      '()
      (let loop ([xs (cdr xs)]
                 [current-key (key (car xs))]
                 [current-group (list (car xs))]
                 [groups '()])
           (if (null? xs)
               (reverse (cons current-group groups))
               (let ([k (key (car xs))])
                    (if (equal? k current-key)
                        (loop (cdr xs) current-key (cons (car xs) current-group) groups)
                        (loop (cdr xs) k (list (car xs)) (cons current-group groups))))))))""",
        "note": "Each group must preserve encounter order; this version leaves groups reversed.",
    },
]

DIFFICULTY = {
    "compose": "medium",
    "pipe": "medium",
    "curry2": "easy",
    "partial": "medium",
    "maybe-bind": "medium",
    "sequence-maybe": "hard",
    "either-bind": "medium",
    "group-by": "hard",
}

REQUIRED_KEYS = [
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
]

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
    sid = f"fp_meta_combinators_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
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


def build_verify(verify_check: str) -> str:
    return verify_check.strip()


def def_verify(fn: str) -> str:
    return build_verify(VERIFY_BY_FUNCTION[fn])


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement this combinator utility in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "meta", "combinators", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Complete this Fold Scheme skeleton.

```scheme
{SKELETONS[fn]}
```

Replace `<TODO>` and return only the completed definition for `{fn}`.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "meta", "combinators", "skeleton-completion", fn],
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
Preserve behavior exactly and use target function name `{fn}`.
Return only the Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "meta", "combinators", "python-to-scheme", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Convert this Chez-style snippet to canonical Fold style.
Target function name must be `{fn}`.
Return only the corrected Fold definition.

```scheme
{CHEZ_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "meta", "combinators", "chez-to-fold", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = case["fn"]
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
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "fp", "meta", "combinators", "bugfix", fn],
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
        verify_expr=build_verify(verify_check),
        tags=["tier1", "fp", "meta", "combinators", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # compose
    {
        "fn": "compose",
        "prompt": "Compose add1 then double, and evaluate at 5.",
        "gt": "((compose (lambda (x) (* x 2)) (lambda (x) (+ x 1))) 5)",
        "verify": "(= ((compose (lambda (x) (* x 2)) (lambda (x) (+ x 1))) 5) 12)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "compose",
        "prompt": "Evaluate a 3-stage right-to-left composition at input 10.",
        "gt": "((compose (lambda (x) (- x 3)) (lambda (x) (* x 2)) (lambda (x) (+ x 1))) 10)",
        "verify": "(= ((compose (lambda (x) (- x 3)) (lambda (x) (* x 2)) (lambda (x) (+ x 1))) 10) 19)",
        "difficulty": "medium",
        "tags": ["order"],
    },
    {
        "fn": "compose",
        "prompt": "Evaluate empty compose on 42.",
        "gt": "((compose) 42)",
        "verify": "(= ((compose) 42) 42)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "compose",
        "prompt": "Return whether compose(f3,f2,f1) and pipe(f1,f2,f3) agree on input 10.",
        "gt": "(let ([f1 (lambda (x) (+ x 1))] [f2 (lambda (x) (* x 2))] [f3 (lambda (x) (- x 3))]) (= ((compose f3 f2 f1) 10) ((pipe f1 f2 f3) 10)))",
        "verify": "(equal? (let ([f1 (lambda (x) (+ x 1))] [f2 (lambda (x) (* x 2))] [f3 (lambda (x) (- x 3))]) (= ((compose f3 f2 f1) 10) ((pipe f1 f2 f3) 10))) #t)",
        "difficulty": "medium",
        "tags": ["consistency"],
    },

    # pipe
    {
        "fn": "pipe",
        "prompt": "Pipe add1 then double, and evaluate at 5.",
        "gt": "((pipe (lambda (x) (+ x 1)) (lambda (x) (* x 2))) 5)",
        "verify": "(= ((pipe (lambda (x) (+ x 1)) (lambda (x) (* x 2))) 5) 12)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "pipe",
        "prompt": "Evaluate a 3-stage left-to-right pipeline at input 10.",
        "gt": "((pipe (lambda (x) (+ x 1)) (lambda (x) (* x 2)) (lambda (x) (- x 3))) 10)",
        "verify": "(= ((pipe (lambda (x) (+ x 1)) (lambda (x) (* x 2)) (lambda (x) (- x 3))) 10) 19)",
        "difficulty": "medium",
        "tags": ["order"],
    },
    {
        "fn": "pipe",
        "prompt": "Evaluate empty pipe on 99.",
        "gt": "((pipe) 99)",
        "verify": "(= ((pipe) 99) 99)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "pipe",
        "prompt": "Return whether pipe(f1,f2,f3) equals compose(f3,f2,f1) on input 7.",
        "gt": "(let ([f1 (lambda (x) (+ x 4))] [f2 (lambda (x) (* x 3))] [f3 (lambda (x) (- x 5))]) (= ((pipe f1 f2 f3) 7) ((compose f3 f2 f1) 7)))",
        "verify": "(equal? (let ([f1 (lambda (x) (+ x 4))] [f2 (lambda (x) (* x 3))] [f3 (lambda (x) (- x 5))]) (= ((pipe f1 f2 f3) 7) ((compose f3 f2 f1) 7))) #t)",
        "difficulty": "medium",
        "tags": ["consistency"],
    },

    # curry2
    {
        "fn": "curry2",
        "prompt": "Use curry2 to build curried addition and evaluate ((add 3) 4).",
        "gt": "(let ([add (curry2 (lambda (x y) (+ x y)))]) ((add 3) 4))",
        "verify": "(= (let ([add (curry2 (lambda (x y) (+ x y)))]) ((add 3) 4)) 7)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "curry2",
        "prompt": "Use curry2 and partial application to add 5 to 10.",
        "gt": "(let* ([add (curry2 (lambda (x y) (+ x y)))] [add5 (add 5)]) (add5 10))",
        "verify": "(= (let* ([add (curry2 (lambda (x y) (+ x y)))] [add5 (add 5)]) (add5 10)) 15)",
        "difficulty": "easy",
        "tags": ["partial-application"],
    },
    {
        "fn": "curry2",
        "prompt": "Use curry2 with cons to produce a pair '(a . b).",
        "gt": "(let ([pair-maker (curry2 cons)]) ((pair-maker 'a) 'b))",
        "verify": "(equal? (let ([pair-maker (curry2 cons)]) ((pair-maker 'a) 'b)) '(a . b))",
        "difficulty": "medium",
        "tags": ["pair"],
    },
    {
        "fn": "curry2",
        "prompt": "Return whether curried less-than with left argument 3 accepts 4 and rejects 2.",
        "gt": "(let ([lt3 ((curry2 <) 3)]) (and (lt3 4) (not (lt3 2))))",
        "verify": "(equal? (let ([lt3 ((curry2 <) 3)]) (and (lt3 4) (not (lt3 2)))) #t)",
        "difficulty": "medium",
        "tags": ["predicate"],
    },

    # partial
    {
        "fn": "partial",
        "prompt": "Partially apply + with 3, then apply to 4.",
        "gt": "(let ([add3 (partial + 3)]) (add3 4))",
        "verify": "(= (let ([add3 (partial + 3)]) (add3 4)) 7)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "partial",
        "prompt": "Partially apply + with 3 and evaluate with two additional arguments 4 and 5.",
        "gt": "(let ([add3 (partial + 3)]) (add3 4 5))",
        "verify": "(= (let ([add3 (partial + 3)]) (add3 4 5)) 12)",
        "difficulty": "medium",
        "tags": ["variadic"],
    },
    {
        "fn": "partial",
        "prompt": "Use partial to prepend a fixed prefix in string-append.",
        "gt": "(let ([with-hello (partial string-append \"hello-\")]) (with-hello \"world\"))",
        "verify": "(equal? (let ([with-hello (partial string-append \"hello-\")]) (with-hello \"world\")) \"hello-world\")",
        "difficulty": "medium",
        "tags": ["string"],
    },
    {
        "fn": "partial",
        "prompt": "Use partial to fix the first argument of list and build a 3-item list.",
        "gt": "(let ([list-with-a (partial list 'a)]) (list-with-a 'b 'c))",
        "verify": "(equal? (let ([list-with-a (partial list 'a)]) (list-with-a 'b 'c)) '(a b c))",
        "difficulty": "medium",
        "tags": ["list"],
    },

    # maybe-bind
    {
        "fn": "maybe-bind",
        "prompt": "Use maybe-bind with safe division and input (just 2).",
        "gt": "(let ([safe-div (lambda (n) (if (= n 0) nothing (just (/ 10 n))))]) (maybe-bind (just 2) safe-div))",
        "verify": "(equal? (let ([safe-div (lambda (n) (if (= n 0) nothing (just (/ 10 n))))]) (maybe-bind (just 2) safe-div)) (just 5))",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "maybe-bind",
        "prompt": "Use maybe-bind with safe division and input (just 0).",
        "gt": "(let ([safe-div (lambda (n) (if (= n 0) nothing (just (/ 10 n))))]) (maybe-bind (just 0) safe-div))",
        "verify": "(equal? (let ([safe-div (lambda (n) (if (= n 0) nothing (just (/ 10 n))))]) (maybe-bind (just 0) safe-div)) nothing)",
        "difficulty": "medium",
        "tags": ["edge-case"],
    },
    {
        "fn": "maybe-bind",
        "prompt": "Return maybe-bind on nothing with any continuation.",
        "gt": "(maybe-bind nothing (lambda (x) (just (+ x 1))))",
        "verify": "(equal? (maybe-bind nothing (lambda (x) (just (+ x 1)))) nothing)",
        "difficulty": "easy",
        "tags": ["propagation"],
    },
    {
        "fn": "maybe-bind",
        "prompt": "Chain two maybe-bind steps to add 3 and 4 in Maybe context.",
        "gt": "(maybe-bind (just 3) (lambda (x) (maybe-bind (just 4) (lambda (y) (just (+ x y))))))",
        "verify": "(equal? (maybe-bind (just 3) (lambda (x) (maybe-bind (just 4) (lambda (y) (just (+ x y)))))) (just 7))",
        "difficulty": "medium",
        "tags": ["chaining"],
    },

    # sequence-maybe
    {
        "fn": "sequence-maybe",
        "prompt": "Sequence a list of all-Just values '(just 1, just 2, just 3).",
        "gt": "(sequence-maybe (list (just 1) (just 2) (just 3)))",
        "verify": "(equal? (sequence-maybe (list (just 1) (just 2) (just 3))) (just '(1 2 3)))",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "sequence-maybe",
        "prompt": "Sequence a list containing a Nothing in the middle.",
        "gt": "(sequence-maybe (list (just 1) nothing (just 3)))",
        "verify": "(equal? (sequence-maybe (list (just 1) nothing (just 3))) nothing)",
        "difficulty": "medium",
        "tags": ["short-circuit"],
    },
    {
        "fn": "sequence-maybe",
        "prompt": "Sequence an empty list of Maybe values.",
        "gt": "(sequence-maybe '())",
        "verify": "(equal? (sequence-maybe '()) (just '()))",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "sequence-maybe",
        "prompt": "Sequence a single-element Just list.",
        "gt": "(sequence-maybe (list (just 9)))",
        "verify": "(equal? (sequence-maybe (list (just 9))) (just '(9)))",
        "difficulty": "easy",
        "tags": ["singleton"],
    },

    # either-bind
    {
        "fn": "either-bind",
        "prompt": "Bind Right 5 through validator that doubles positives and rejects negatives.",
        "gt": "(let ([validate (lambda (x) (if (positive? x) (right (* x 2)) (left 'negative)))]) (either-bind (right 5) validate))",
        "verify": "(equal? (let ([validate (lambda (x) (if (positive? x) (right (* x 2)) (left 'negative)))]) (either-bind (right 5) validate)) (right 10))",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "either-bind",
        "prompt": "Bind Right -5 through validator that rejects negatives.",
        "gt": "(let ([validate (lambda (x) (if (positive? x) (right (* x 2)) (left 'negative)))]) (either-bind (right -5) validate))",
        "verify": "(equal? (let ([validate (lambda (x) (if (positive? x) (right (* x 2)) (left 'negative)))]) (either-bind (right -5) validate)) (left 'negative))",
        "difficulty": "medium",
        "tags": ["error-path"],
    },
    {
        "fn": "either-bind",
        "prompt": "Bind a pre-existing Left value through any continuation.",
        "gt": "(either-bind (left 'already-error) (lambda (x) (right (* x 2))))",
        "verify": "(equal? (either-bind (left 'already-error) (lambda (x) (right (* x 2)))) (left 'already-error))",
        "difficulty": "easy",
        "tags": ["propagation"],
    },
    {
        "fn": "either-bind",
        "prompt": "Chain two either-bind stages and sum transformed Right values.",
        "gt": "(either-bind (right 3) (lambda (x) (either-bind (right 4) (lambda (y) (right (+ x y))))))",
        "verify": "(equal? (either-bind (right 3) (lambda (x) (either-bind (right 4) (lambda (y) (right (+ x y)))))) (right 7))",
        "difficulty": "medium",
        "tags": ["chaining"],
    },

    # group-by
    {
        "fn": "group-by",
        "prompt": "Group consecutive numbers by parity for '(2 4 1 3 6 8 5).",
        "gt": "(group-by even? '(2 4 1 3 6 8 5))",
        "verify": "(equal? (group-by even? '(2 4 1 3 6 8 5)) '((2 4) (1 3) (6 8) (5)))",
        "difficulty": "hard",
        "tags": ["direct"],
    },
    {
        "fn": "group-by",
        "prompt": "Group an empty input list by parity.",
        "gt": "(group-by even? '())",
        "verify": "(equal? (group-by even? '()) '())",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "group-by",
        "prompt": "Group consecutive strings by string-length for '(\"a\" \"b\" \"cc\" \"dd\" \"e\").",
        "gt": "(group-by string-length '(\"a\" \"b\" \"cc\" \"dd\" \"e\"))",
        "verify": "(equal? (group-by string-length '(\"a\" \"b\" \"cc\" \"dd\" \"e\")) '((\"a\" \"b\") (\"cc\" \"dd\") (\"e\")))",
        "difficulty": "hard",
        "tags": ["string"],
    },
    {
        "fn": "group-by",
        "prompt": "Group '(1 3 2 4 5 7) by parity and return the grouped result.",
        "gt": "(group-by even? '(1 3 2 4 5 7))",
        "verify": "(equal? (group-by even? '(1 3 2 4 5 7)) '((1 3) (2 4) (5 7)))",
        "difficulty": "medium",
        "tags": ["consecutive-runs"],
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

if len(samples) != 80:
    raise ValueError(f"expected 80 samples, got {len(samples)}")

by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for sample in samples:
    by_family[str(sample["family"])].append(sample)

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
    for idx, sample in enumerate(family_samples):
        if idx in picked:
            eval_ids.add(str(sample["id"]))

id_to_sample: Dict[str, Dict[str, object]] = {str(sample["id"]): sample for sample in samples}
all_source_functions = sorted({str(sample["source_function"]) for sample in samples})


def eval_source_fn_counts(ids: Set[str]) -> Counter:
    return Counter(str(id_to_sample[sid]["source_function"]) for sid in ids)


changed = True
while changed:
    changed = False
    fn_counts = eval_source_fn_counts(eval_ids)
    missing_fns = [fn for fn in all_source_functions if fn_counts[fn] == 0]
    if not missing_fns:
        break

    for fn in missing_fns:
        candidates = [
            sample
            for sample in samples
            if str(sample["source_function"]) == fn and str(sample["id"]) not in eval_ids
        ]
        swapped = False
        for cand in candidates:
            family = str(cand["family"])
            fam_eval = [id_to_sample[sid] for sid in eval_ids if str(id_to_sample[sid]["family"]) == family]
            removable = [row for row in fam_eval if fn_counts[str(row["source_function"])] > 1]
            if not removable:
                continue
            removable.sort(key=lambda row: (fn_counts[str(row["source_function"])], str(row["id"])), reverse=True)
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
for sample in samples:
    row = dict(sample)
    if sample["id"] in eval_ids:
        row["split"] = "eval"
        eval_rows.append(row)
    else:
        row["split"] = "train"
        train_rows.append(row)

if len(train_rows) != 66 or len(eval_rows) != 14:
    raise ValueError(f"split mismatch: train={len(train_rows)}, eval={len(eval_rows)}")


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


write_jsonl(ALL_PATH, [dict(sample, split=("eval" if sample["id"] in eval_ids else "train")) for sample in samples])
write_jsonl(TRAIN_PATH, train_rows)
write_jsonl(EVAL_PATH, eval_rows)

summary = {
    "total": len(samples),
    "train": len(train_rows),
    "eval": len(eval_rows),
    "families": {
        family: {
            "total": len(family_samples),
            "eval": sum(1 for sample in family_samples if sample["id"] in eval_ids),
            "train": sum(1 for sample in family_samples if sample["id"] not in eval_ids),
        }
        for family, family_samples in sorted(by_family.items())
    },
    "difficulty": dict(sorted(Counter(str(sample["difficulty"]) for sample in samples).items())),
    "source_functions": dict(sorted(Counter(str(sample["source_function"]) for sample in samples).items())),
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
