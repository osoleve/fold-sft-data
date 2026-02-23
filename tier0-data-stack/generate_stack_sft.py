#!/usr/bin/env python3
"""Generate Tier-0 SFT samples for lattice/data/stack.ss."""

from __future__ import annotations

import json
import sys
import re
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

SOURCE_MODULE = "lattice/data/stack.ss"
SOURCE_TEST = "lattice/data/test-data-structures.ss"

DEFS: Dict[str, str] = {
    "stack-empty?": """(define (stack-empty? stack)
  (null? stack))""",
    "stack-push": """(define (stack-push elem stack)
  (cons elem stack))""",
    "stack-pop": """(define (stack-pop stack)
  (if (null? stack)
      (error 'stack-pop "Cannot pop from empty stack")
      (values (cdr stack) (car stack))))""",
    "stack-peek": """(define (stack-peek stack)
  (if (null? stack)
      (error 'stack-peek "Cannot peek empty stack")
      (car stack)))""",
    "stack-size": """(define (stack-size stack)
  (length stack))""",
    "stack->list": """(define (stack->list stack)
  stack)""",
    "list->stack": """(define (list->stack lst)
  lst)""",
}

FUNCTION_ORDER = [
    "stack-empty?",
    "stack-push",
    "stack-pop",
    "stack-peek",
    "stack-size",
    "stack->list",
    "list->stack",
]

FUNCTION_SPECS = {
    "stack-empty?": "Return #t iff the stack has no elements.",
    "stack-push": "Push elem onto the top of stack and return the new stack.",
    "stack-pop": "Pop top element; return two values: (new-stack, popped-elem). Raise an error on empty stack.",
    "stack-peek": "Return top element without removing it. Raise an error on empty stack.",
    "stack-size": "Return the number of elements in stack.",
    "stack->list": "Convert stack to list in top-to-bottom order.",
    "list->stack": "Convert list to stack where the first list element becomes the stack top.",
}

SKELETONS = {
    "stack-empty?": """(define (stack-empty? stack)
  ;; TODO: return whether stack is empty
  <TODO>)""",
    "stack-push": """(define (stack-push elem stack)
  ;; TODO: push elem onto stack top
  <TODO>)""",
    "stack-pop": """(define (stack-pop stack)
  ;; TODO: return (values new-stack popped-elem), error on empty
  <TODO>)""",
    "stack-peek": """(define (stack-peek stack)
  ;; TODO: return top element, error on empty
  <TODO>)""",
    "stack-size": """(define (stack-size stack)
  ;; TODO: return number of elements
  <TODO>)""",
    "stack->list": """(define (stack->list stack)
  ;; TODO: convert stack to list
  <TODO>)""",
    "list->stack": """(define (list->stack lst)
  ;; TODO: convert list to stack
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "stack-empty?": "(and (stack-empty? '()) (not (stack-empty? '(x))))",
    "stack-push": "(and (equal? (stack-push 'a '()) '(a)) (equal? (stack-push 'a '(b c)) '(a b c)))",
    "stack-pop": "(and (call-with-values (lambda () (stack-pop '(a b c))) (lambda (s x) (and (equal? s '(b c)) (equal? x 'a)))) (guard (ex [else #t]) (begin (stack-pop '()) #f)))",
    "stack-peek": "(and (equal? (stack-peek '(z y)) 'z) (guard (ex [else #t]) (begin (stack-peek '()) #f)))",
    "stack-size": "(and (= (stack-size '()) 0) (= (stack-size '(a b c d)) 4))",
    "stack->list": "(and (equal? (stack->list '(c b a)) '(c b a)) (equal? (stack->list '()) '()))",
    "list->stack": "(and (equal? (list->stack '(1 2 3)) '(1 2 3)) (equal? (list->stack '()) '()))",
}

DEPENDS: Dict[str, List[str]] = {
    "stack-empty?": [],
    "stack-push": [],
    "stack-pop": [],
    "stack-peek": [],
    "stack-size": [],
    "stack->list": [],
    "list->stack": [],
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "stack-empty?": "def stack_empty(stack):\n    return len(stack) == 0",
    "stack-push": "def stack_push(elem, stack):\n    return [elem] + stack",
    "stack-pop": "def stack_pop(stack):\n    if len(stack) == 0:\n        raise ValueError('empty stack')\n    return stack[1:], stack[0]",
    "stack-peek": "def stack_peek(stack):\n    if len(stack) == 0:\n        raise ValueError('empty stack')\n    return stack[0]",
    "stack-size": "def stack_size(stack):\n    return len(stack)",
    "stack->list": "def stack_to_list(stack):\n    return stack",
    "list->stack": "def list_to_stack(lst):\n    return lst",
}

CHEZ_SNIPPETS = {
    "stack-empty?": "(define (empty? s)\n  (null? s))",
    "stack-push": "(define (push x s)\n  (cons x s))",
    "stack-pop": "(define (pop s)\n  (if (null? s)\n      (error 'pop \"empty\")\n      (values (cdr s) (car s))))",
    "stack-peek": "(define (peek s)\n  (if (null? s)\n      (error 'peek \"empty\")\n      (car s)))",
    "stack-size": "(define (size s)\n  (length s))",
    "stack->list": "(define (to-list s)\n  s)",
    "list->stack": "(define (from-list xs)\n  xs)",
}

BUGGY_CASES = [
    {
        "fn": "stack-empty?",
        "buggy": "(define (stack-empty? stack)\n  (null? (cdr stack)))",
        "note": "This crashes on empty stacks and also misclassifies single-element stacks as empty.",
    },
    {
        "fn": "stack-empty?",
        "buggy": "(define (stack-empty? stack)\n  #f)",
        "note": "Empty stacks must return #t.",
    },
    {
        "fn": "stack-push",
        "buggy": "(define (stack-push elem stack)\n  (append stack (list elem)))",
        "note": "Push must add at the top, not the bottom.",
    },
    {
        "fn": "stack-push",
        "buggy": "(define (stack-push elem stack)\n  stack)",
        "note": "The pushed element is discarded.",
    },
    {
        "fn": "stack-pop",
        "buggy": "(define (stack-pop stack)\n  (if (null? stack)\n      (values '() #f)\n      (values (cdr stack) (car stack))))",
        "note": "Empty stack must raise an error, not return sentinel values.",
    },
    {
        "fn": "stack-pop",
        "buggy": "(define (stack-pop stack)\n  (if (null? stack)\n      (error 'stack-pop \"Cannot pop from empty stack\")\n      (values (cdr stack) (cdr stack))))",
        "note": "Second returned value must be the popped element.",
    },
    {
        "fn": "stack-peek",
        "buggy": "(define (stack-peek stack)\n  (if (null? stack)\n      #f\n      (car stack)))",
        "note": "Peeking an empty stack must raise an error.",
    },
    {
        "fn": "stack-peek",
        "buggy": "(define (stack-peek stack)\n  (if (null? stack)\n      (error 'stack-peek \"Cannot peek empty stack\")\n      (last stack)))",
        "note": "Peek should return the top (head), not the bottom.",
    },
    {
        "fn": "stack-size",
        "buggy": "(define (stack-size stack)\n  (if (null? stack) 1 (length stack)))",
        "note": "Empty stack size must be 0.",
    },
    {
        "fn": "stack-size",
        "buggy": "(define (stack-size stack)\n  (- (length stack) 1))",
        "note": "Size should equal list length exactly.",
    },
    {
        "fn": "stack->list",
        "buggy": "(define (stack->list stack)\n  (reverse stack))",
        "note": "Order must remain top-to-bottom.",
    },
    {
        "fn": "stack->list",
        "buggy": "(define (stack->list stack)\n  '())",
        "note": "Conversion should preserve contents.",
    },
    {
        "fn": "list->stack",
        "buggy": "(define (list->stack lst)\n  (reverse lst))",
        "note": "First list element must become the stack top.",
    },
    {
        "fn": "list->stack",
        "buggy": "(define (list->stack lst)\n  '())",
        "note": "Conversion should preserve elements.",
    },
]

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
    sid = f"stack_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
        "prompt": diversify_prompt(prompt.strip(), family, source_function, family_counter[family], category, verify_expr),
        "ground_truth": ground_truth.strip(),
        "verify_expr": verify_expr.strip(),
        "tags": tags,
    }
    for k in REQUIRED_KEYS:
        if k not in sample:
            raise ValueError(f"missing key {k}")
    samples.append(sample)


def verify_refs(fn: str) -> List[str]:
    tokens = set(TOKEN_RE.findall(VERIFY_BY_FUNCTION[fn]))
    return [name for name in FUNCTION_ORDER if name != fn and name in tokens]


def dependency_closure(fn: str) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    def visit(name: str) -> None:
        for dep in DEPENDS.get(name, []):
            if dep not in seen:
                seen.add(dep)
                visit(dep)
                ordered.append(dep)

    for dep in DEPENDS.get(fn, []):
        if dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    for dep in verify_refs(fn):
        if dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    return ordered


def def_verify(fn: str) -> str:
    parts = [DEFS[dep] for dep in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (14)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    diff = "easy" if fn in {"stack-empty?", "stack-push", "stack-size", "stack->list", "list->stack"} else "medium"

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=diff,
        source_function=fn,
        prompt=f"""You are implementing Tier-0 data-structure code in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "stack", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=diff,
        source_function=fn,
        prompt=f"""Complete this Fold Scheme skeleton.

```scheme
{SKELETONS[fn]}
```

Replace `<TODO>` and return only the completed definition for `{fn}`.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "stack", "skeleton-completion", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (14)
# -----------------------------------------------------------------------------
for fn in TRANSLATION_FUNCTIONS:
    diff = "easy" if fn in {"stack-empty?", "stack-push", "stack-size", "stack->list", "list->stack"} else "medium"

    add_sample(
        family="translation",
        category="translation",
        difficulty=diff,
        source_function=fn,
        prompt=f"""Translate the following Python function into Fold-native Scheme.
Preserve behavior exactly and use the target function name `{fn}`.
Return only the Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "stack", "python-to-scheme", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=diff,
        source_function=fn,
        prompt=f"""Convert this Chez-style snippet to canonical Fold style.
Target function name must be `{fn}`.
Return only the corrected Fold definition.

```scheme
{CHEZ_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "stack", "chez-to-fold", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (14)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = case["fn"]
    diff = "easy" if fn in {"stack-empty?", "stack-push", "stack-size", "stack->list", "list->stack"} else "medium"

    add_sample(
        family="bugfix",
        category="debugging",
        difficulty=diff,
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
        tags=["tier0", "data", "stack", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition/use (28)
# -----------------------------------------------------------------------------


def add_composition(
    source_function: str,
    prompt: str,
    ground_truth: str,
    verify_expr: str,
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
        verify_expr=verify_expr,
        tags=["tier0", "data", "stack", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # Direct operations
    ("stack-empty?", "Return whether `stack-empty` is empty.", "(stack-empty? stack-empty)", "(equal? (stack-empty? stack-empty) #t)", "easy", ["direct"]),
    ("stack-push", "Push `42` onto `stack-empty`.", "(stack-push 42 stack-empty)", "(equal? (stack-push 42 stack-empty) '(42))", "easy", ["direct"]),
    ("stack-push", "Push `'a`, then `'b`, then `'c` onto `stack-empty` and return the final stack.", "(stack-push 'c (stack-push 'b (stack-push 'a stack-empty)))", "(equal? (stack-push 'c (stack-push 'b (stack-push 'a stack-empty))) '(c b a))", "easy", ["direct"]),
    ("stack-peek", "Return the top element of stack `'(top mid low)`.", "(stack-peek '(top mid low))", "(equal? (stack-peek '(top mid low)) 'top)", "easy", ["direct"]),
    ("stack-size", "Return stack size for `'(x y z w)`.", "(stack-size '(x y z w))", "(equal? (stack-size '(x y z w)) 4)", "easy", ["direct"]),
    ("stack->list", "Convert stack `'(s3 s2 s1)` to list.", "(stack->list '(s3 s2 s1))", "(equal? (stack->list '(s3 s2 s1)) '(s3 s2 s1))", "easy", ["direct"]),
    ("list->stack", "Convert list `'(1 2 3 4)` to stack.", "(list->stack '(1 2 3 4))", "(equal? (list->stack '(1 2 3 4)) '(1 2 3 4))", "easy", ["direct"]),
    ("stack-pop", "Pop once from stack `'(k j i)` and return `(list popped rest)`.", "(call-with-values (lambda () (stack-pop '(k j i))) (lambda (rest popped) (list popped rest)))", "(equal? (call-with-values (lambda () (stack-pop '(k j i))) (lambda (rest popped) (list popped rest))) '(k (j i)))", "medium", ["direct"]),
    ("stack->list", "Round-trip list->stack->list for `'(m n o)`.", "(stack->list (list->stack '(m n o)))", "(equal? (stack->list (list->stack '(m n o))) '(m n o))", "easy", ["direct"]),
    ("stack-empty?", "Return whether stack `'(a)` is empty.", "(stack-empty? '(a))", "(equal? (stack-empty? '(a)) #f)", "easy", ["direct"]),

    # Properties
    ("stack-pop", "Return #t iff pushing then popping restores the original stack for `'(b a)` with element `'x`.", "(call-with-values (lambda () (stack-pop (stack-push 'x '(b a)))) (lambda (rest popped) (and (equal? rest '(b a)) (equal? popped 'x))))", "(equal? (call-with-values (lambda () (stack-pop (stack-push 'x '(b a)))) (lambda (rest popped) (and (equal? rest '(b a)) (equal? popped 'x)))) #t)", "medium", ["property"]),
    ("stack-pop", "Return #t iff two pushes then two pops return elements in LIFO order.", "(call-with-values (lambda () (stack-pop (stack-push 'y (stack-push 'x stack-empty)))) (lambda (s1 p1) (call-with-values (lambda () (stack-pop s1)) (lambda (s2 p2) (and (equal? p1 'y) (equal? p2 'x) (stack-empty? s2))))))", "(equal? (call-with-values (lambda () (stack-pop (stack-push 'y (stack-push 'x stack-empty)))) (lambda (s1 p1) (call-with-values (lambda () (stack-pop s1)) (lambda (s2 p2) (and (equal? p1 'y) (equal? p2 'x) (stack-empty? s2)))))) #t)", "hard", ["property"]),
    ("stack-size", "Check that stack size increases by one after `stack-push`.", "(= (stack-size (stack-push 'n '(c b a))) (+ 1 (stack-size '(c b a))))", "(equal? (= (stack-size (stack-push 'n '(c b a))) (+ 1 (stack-size '(c b a)))) #t)", "medium", ["property"]),
    ("stack-size", "Check that popping a non-empty stack decreases size by one.", "(call-with-values (lambda () (stack-pop '(q p o n))) (lambda (rest popped) (= (stack-size rest) 3)))", "(equal? (call-with-values (lambda () (stack-pop '(q p o n))) (lambda (rest popped) (= (stack-size rest) 3))) #t)", "medium", ["property"]),
    ("stack-peek", "Return #t iff `stack-peek` after push returns the pushed value `'new`.", "(equal? (stack-peek (stack-push 'new '(old))) 'new)", "(equal? (equal? (stack-peek (stack-push 'new '(old))) 'new) #t)", "medium", ["property"]),
    ("list->stack", "Return #t iff `list->stack` keeps `'a` at the top for `'(a b c)`.", "(equal? (car (list->stack '(a b c))) 'a)", "(equal? (equal? (car (list->stack '(a b c))) 'a) #t)", "medium", ["property"]),
    ("stack-pop", "Return #t iff popping the only element yields empty stack and that element.", "(call-with-values (lambda () (stack-pop '(solo))) (lambda (rest popped) (and (stack-empty? rest) (equal? popped 'solo))))", "(equal? (call-with-values (lambda () (stack-pop '(solo))) (lambda (rest popped) (and (stack-empty? rest) (equal? popped 'solo)))) #t)", "medium", ["property"]),
    ("stack-pop", "Return #t iff popping empty stack raises an exception.", "(guard (ex [else #t]) (begin (stack-pop stack-empty) #f))", "(equal? (guard (ex [else #t]) (begin (stack-pop stack-empty) #f)) #t)", "medium", ["edge-case"]),
    ("stack-peek", "Return #t iff peeking empty stack raises an exception.", "(guard (ex [else #t]) (begin (stack-peek stack-empty) #f))", "(equal? (guard (ex [else #t]) (begin (stack-peek stack-empty) #f)) #t)", "medium", ["edge-case"]),

    # Fold/loop usage
    ("stack-push", "Build a stack from `'(1 2 3 4)` using `fold-left` and `stack-push`.", "(fold-left (lambda (s x) (stack-push x s)) stack-empty '(1 2 3 4))", "(equal? (fold-left (lambda (s x) (stack-push x s)) stack-empty '(1 2 3 4)) '(4 3 2 1))", "medium", ["fold"]),
    ("stack-push", "Push all symbols in `'(a b c d)` with a named-let loop and return the stack.", "(let loop ([xs '(a b c d)] [s stack-empty]) (if (null? xs) s (loop (cdr xs) (stack-push (car xs) s))))", "(equal? (let loop ([xs '(a b c d)] [s stack-empty]) (if (null? xs) s (loop (cdr xs) (stack-push (car xs) s)))) '(d c b a))", "medium", ["loop"]),
    ("stack-pop", "Pop every element from `'(4 3 2 1)` and accumulate their sum.", "(let loop ([s '(4 3 2 1)] [acc 0]) (if (stack-empty? s) acc (call-with-values (lambda () (stack-pop s)) (lambda (rest popped) (loop rest (+ acc popped))))))", "(equal? (let loop ([s '(4 3 2 1)] [acc 0]) (if (stack-empty? s) acc (call-with-values (lambda () (stack-pop s)) (lambda (rest popped) (loop rest (+ acc popped)))))) 10)", "hard", ["loop"]),
    ("stack-pop", "Count how many pops are required to empty stack `'(d c b a)`.", "(let loop ([s '(d c b a)] [n 0]) (if (stack-empty? s) n (call-with-values (lambda () (stack-pop s)) (lambda (rest popped) (loop rest (+ n 1))))))", "(equal? (let loop ([s '(d c b a)] [n 0]) (if (stack-empty? s) n (call-with-values (lambda () (stack-pop s)) (lambda (rest popped) (loop rest (+ n 1)))))) 4)", "hard", ["loop"]),
    ("stack-size", "Return a list of stack sizes for stacks `'((a) (b c) (d e f))`.", "(map stack-size '((a) (b c) (d e f)))", "(equal? (map stack-size '((a) (b c) (d e f))) '(1 2 3))", "medium", ["list"]),
    ("stack->list", "Given stacks `'((x y) () (z))`, map `stack->list` over them.", "(map stack->list '((x y) () (z)))", "(equal? (map stack->list '((x y) () (z))) '((x y) () (z)))", "easy", ["list"]),

    # Integration tasks
    ("stack-pop", "Pop twice from `'(c b a)` and return `(list first second final-stack)`.", "(call-with-values (lambda () (stack-pop '(c b a))) (lambda (s1 p1) (call-with-values (lambda () (stack-pop s1)) (lambda (s2 p2) (list p1 p2 s2)))))", "(equal? (call-with-values (lambda () (stack-pop '(c b a))) (lambda (s1 p1) (call-with-values (lambda () (stack-pop s1)) (lambda (s2 p2) (list p1 p2 s2))))) '(c b (a)))", "hard", ["integration"]),
    ("stack-peek", "Duplicate the top of stack `'(k j i)` by peeking then pushing.", "(let ([s '(k j i)]) (stack-push (stack-peek s) s))", "(equal? (let ([s '(k j i)]) (stack-push (stack-peek s) s)) '(k k j i))", "medium", ["integration"]),
    ("stack-pop", "From stack `'(q p o)`, pop once, push `'z`, and return resulting stack.", "(call-with-values (lambda () (stack-pop '(q p o))) (lambda (rest popped) (stack-push 'z rest)))", "(equal? (call-with-values (lambda () (stack-pop '(q p o))) (lambda (rest popped) (stack-push 'z rest))) '(z p o))", "medium", ["integration"]),
    ("list->stack", "Check whether converting `'( )` with `list->stack` yields an empty stack.", "(stack-empty? (list->stack '()))", "(equal? (stack-empty? (list->stack '())) #t)", "easy", ["integration"]),
    ("stack-pop", "Pop three times from `'(3 2 1)` and return popped values as a list.", "(call-with-values (lambda () (stack-pop '(3 2 1))) (lambda (s1 p1) (call-with-values (lambda () (stack-pop s1)) (lambda (s2 p2) (call-with-values (lambda () (stack-pop s2)) (lambda (s3 p3) (list p1 p2 p3)))))))", "(equal? (call-with-values (lambda () (stack-pop '(3 2 1))) (lambda (s1 p1) (call-with-values (lambda () (stack-pop s1)) (lambda (s2 p2) (call-with-values (lambda () (stack-pop s2)) (lambda (s3 p3) (list p1 p2 p3))))))) '(3 2 1))", "hard", ["integration"]),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

if sum(1 for s in samples if s["family"] == "composition") != 30:
    raise ValueError("composition family must contain exactly 30 samples")

# -----------------------------------------------------------------------------
# Split train/eval
# -----------------------------------------------------------------------------
if len(samples) != 72:
    raise ValueError(f"expected 72 samples, got {len(samples)}")

by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

EVAL_QUOTA = {
    "spec_to_code": 3,
    "translation": 2,
    "bugfix": 2,
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
for fam, fam_samples in by_family.items():
    picked = spread_indices(len(fam_samples), EVAL_QUOTA[fam])
    for i, s in enumerate(fam_samples):
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
    missing_fns = [fn for fn in all_source_functions if fn_counts[fn] == 0]
    if not missing_fns:
        break

    for fn in missing_fns:
        candidates = [s for s in samples if str(s["source_function"]) == fn and str(s["id"]) not in eval_ids]
        swapped = False
        for cand in candidates:
            fam = str(cand["family"])
            fam_eval = [id_to_sample[sid] for sid in eval_ids if str(id_to_sample[sid]["family"]) == fam]
            removable = [r for r in fam_eval if fn_counts[str(r["source_function"])] > 1]
            if not removable:
                continue
            removable.sort(key=lambda r: (fn_counts[str(r["source_function"])], str(r["id"])), reverse=True)
            out = removable[0]
            eval_ids.remove(str(out["id"]))
            eval_ids.add(str(cand["id"]))
            changed = True
            swapped = True
            break
        if not swapped:
            continue

missing_after = [fn for fn in all_source_functions if eval_source_fn_counts(eval_ids)[fn] == 0]
if missing_after:
    raise ValueError(f"eval split is missing source functions: {missing_after}")

train_rows: List[Dict[str, object]] = []
eval_rows: List[Dict[str, object]] = []
for s in samples:
    row = dict(s)
    if s["id"] in eval_ids:
        row["split"] = "eval"
        eval_rows.append(row)
    else:
        row["split"] = "train"
        train_rows.append(row)

if len(train_rows) != 60 or len(eval_rows) != 12:
    raise ValueError(f"split mismatch: train={len(train_rows)}, eval={len(eval_rows)}")


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


write_jsonl(ALL_PATH, [dict(s, split=("eval" if s["id"] in eval_ids else "train")) for s in samples])
write_jsonl(TRAIN_PATH, train_rows)
write_jsonl(EVAL_PATH, eval_rows)

summary = {
    "total": len(samples),
    "train": len(train_rows),
    "eval": len(eval_rows),
    "families": {
        fam: {
            "total": len(fam_samples),
            "eval": sum(1 for s in fam_samples if s["id"] in eval_ids),
            "train": sum(1 for s in fam_samples if s["id"] not in eval_ids),
        }
        for fam, fam_samples in sorted(by_family.items())
    },
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
