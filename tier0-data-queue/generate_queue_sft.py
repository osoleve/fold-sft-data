#!/usr/bin/env python3
"""Generate Tier-0 SFT samples for lattice/data/queue.ss."""

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

SOURCE_MODULE = "lattice/data/queue.ss"
SOURCE_TEST = "lattice/data/test-data-structures.ss"

DEFS: Dict[str, str] = {
    "make-queue": """(define (make-queue front back)
  (if (null? front)
      (cons (reverse back) '())
      (cons front back)))""",
    "queue-empty?": """(define (queue-empty? queue)
  (and (null? (car queue))
       (null? (cdr queue))))""",
    "queue-enqueue": """(define (queue-enqueue elem queue)
  (let ([front (car queue)]
        [back (cdr queue)])
    (make-queue front (cons elem back))))""",
    "queue-dequeue": """(define (queue-dequeue queue)
  (let ([front (car queue)]
        [back (cdr queue)])
    (if (null? front)
        (error 'queue-dequeue "Cannot dequeue from empty queue")
        (values (make-queue (cdr front) back)
                (car front)))))""",
    "queue-peek": """(define (queue-peek queue)
  (let ([front (car queue)])
    (if (null? front)
        (error 'queue-peek "Cannot peek empty queue")
        (car front))))""",
    "queue-size": """(define (queue-size queue)
  (+ (length (car queue))
     (length (cdr queue))))""",
    "queue->list": """(define (queue->list queue)
  (append (car queue) (reverse (cdr queue))))""",
    "list->queue": """(define (list->queue lst)
  (cons lst '()))""",
}

FUNCTION_ORDER = [
    "make-queue",
    "queue-empty?",
    "queue-enqueue",
    "queue-dequeue",
    "queue-peek",
    "queue-size",
    "queue->list",
    "list->queue",
]

FUNCTION_SPECS = {
    "make-queue": "Internal constructor: if front is empty, reverse back into front and clear back.",
    "queue-empty?": "Return #t iff both internal lists (front/back) are empty.",
    "queue-enqueue": "Add an element at queue back while preserving queue invariants.",
    "queue-dequeue": "Remove front element and return two values: (new-queue, dequeued-elem). Raise error on empty queue.",
    "queue-peek": "Return front element without removing it. Raise error on empty queue.",
    "queue-size": "Return the number of elements in queue.",
    "queue->list": "Convert queue to front-to-back list order.",
    "list->queue": "Convert list to queue with first list element at the queue front.",
}

SKELETONS = {
    "make-queue": """(define (make-queue front back)
  ;; TODO: maintain two-list queue invariant
  <TODO>)""",
    "queue-empty?": """(define (queue-empty? queue)
  ;; TODO: return whether both internal lists are empty
  <TODO>)""",
    "queue-enqueue": """(define (queue-enqueue elem queue)
  ;; TODO: enqueue element and preserve invariant
  <TODO>)""",
    "queue-dequeue": """(define (queue-dequeue queue)
  ;; TODO: dequeue front, return (values new-queue elem), error if empty
  <TODO>)""",
    "queue-peek": """(define (queue-peek queue)
  ;; TODO: return front element, error if empty
  <TODO>)""",
    "queue-size": """(define (queue-size queue)
  ;; TODO: compute element count
  <TODO>)""",
    "queue->list": """(define (queue->list queue)
  ;; TODO: convert queue to front-to-back list
  <TODO>)""",
    "list->queue": """(define (list->queue lst)
  ;; TODO: convert list to queue representation
  <TODO>)""",
}

DEPENDS: Dict[str, List[str]] = {
    "make-queue": [],
    "queue-empty?": [],
    "queue-enqueue": ["make-queue"],
    "queue-dequeue": ["make-queue"],
    "queue-peek": [],
    "queue-size": [],
    "queue->list": [],
    "list->queue": [],
}

VERIFY_BY_FUNCTION = {
    "make-queue": "(and (equal? (make-queue '(a b) '(c d)) '((a b) c d)) (equal? (make-queue '() '(x y z)) '((z y x))))",
    "queue-empty?": "(and (queue-empty? (cons '() '())) (not (queue-empty? (cons '(a) '()))) (not (queue-empty? (cons '() '(a)))))",
    "queue-enqueue": "(and (equal? (queue-enqueue 'x (cons '() '())) '((x))) (equal? (queue-enqueue 'z (cons '(x y) '())) '((x y) z)))",
    "queue-dequeue": "(and (call-with-values (lambda () (queue-dequeue '((a b c)))) (lambda (q x) (and (equal? q '((b c))) (equal? x 'a)))) (call-with-values (lambda () (queue-dequeue (cons '(a) '(c b)))) (lambda (q x) (and (equal? q '((b c))) (equal? x 'a)))) (guard (ex [else #t]) (begin (queue-dequeue (cons '() '())) #f)))",
    "queue-peek": "(and (equal? (queue-peek '((x y z))) 'x) (guard (ex [else #t]) (begin (queue-peek (cons '() '())) #f)))",
    "queue-size": "(and (= (queue-size (cons '() '())) 0) (= (queue-size (cons '(a b) '(d c))) 4))",
    "queue->list": "(and (equal? (queue->list (cons '(a b) '(d c))) '(a b c d)) (equal? (queue->list (cons '() '())) '()))",
    "list->queue": "(and (equal? (list->queue '(1 2 3)) '((1 2 3))) (equal? (list->queue '()) '(())))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "make-queue": "def make_queue(front, back):\n    if len(front) == 0:\n        return list(reversed(back)), []\n    return front, back",
    "queue-empty?": "def queue_empty(queue):\n    front, back = queue\n    return len(front) == 0 and len(back) == 0",
    "queue-enqueue": "def queue_enqueue(elem, queue):\n    front, back = queue\n    return make_queue(front, [elem] + back)",
    "queue-dequeue": "def queue_dequeue(queue):\n    front, back = queue\n    if len(front) == 0:\n        raise ValueError('empty queue')\n    return make_queue(front[1:], back), front[0]",
    "queue-peek": "def queue_peek(queue):\n    front, back = queue\n    if len(front) == 0:\n        raise ValueError('empty queue')\n    return front[0]",
    "queue-size": "def queue_size(queue):\n    front, back = queue\n    return len(front) + len(back)",
    "queue->list": "def queue_to_list(queue):\n    front, back = queue\n    return front + list(reversed(back))",
    "list->queue": "def list_to_queue(lst):\n    return lst, []",
}

CHEZ_SNIPPETS = {
    "make-queue": "(define (mkq f b)\n  (if (null? f)\n      (cons (reverse b) '())\n      (cons f b)))",
    "queue-empty?": "(define (empty? q)\n  (and (null? (car q))\n       (null? (cdr q))))",
    "queue-enqueue": "(define (enqueue x q)\n  (let ([f (car q)] [b (cdr q)])\n    (make-queue f (cons x b))))",
    "queue-dequeue": "(define (dequeue q)\n  (let ([f (car q)] [b (cdr q)])\n    (if (null? f)\n        (error 'dequeue \"empty\")\n        (values (make-queue (cdr f) b) (car f)))))",
    "queue-peek": "(define (peek q)\n  (if (null? (car q))\n      (error 'peek \"empty\")\n      (car (car q))))",
    "queue-size": "(define (size q)\n  (+ (length (car q)) (length (cdr q))))",
    "queue->list": "(define (to-list q)\n  (append (car q) (reverse (cdr q))))",
    "list->queue": "(define (from-list xs)\n  (cons xs '()))",
}

BUGGY_CASES = [
    {
        "fn": "make-queue",
        "buggy": "(define (make-queue front back)\n  (cons front back))",
        "note": "When front is empty, back must be reversed into front.",
    },
    {
        "fn": "make-queue",
        "buggy": "(define (make-queue front back)\n  (if (null? front)\n      (cons back '())\n      (cons front back)))",
        "note": "Back must be reversed before becoming new front.",
    },
    {
        "fn": "queue-empty?",
        "buggy": "(define (queue-empty? queue)\n  (null? (car queue)))",
        "note": "Queue is empty only when both front and back are empty.",
    },
    {
        "fn": "queue-empty?",
        "buggy": "(define (queue-empty? queue)\n  (null? (cdr queue)))",
        "note": "Front list must also be checked.",
    },
    {
        "fn": "queue-enqueue",
        "buggy": "(define (queue-enqueue elem queue)\n  (let ([front (car queue)] [back (cdr queue)])\n    (cons (cons elem front) back)))",
        "note": "Enqueue must add at back, not front.",
    },
    {
        "fn": "queue-enqueue",
        "buggy": "(define (queue-enqueue elem queue)\n  queue)",
        "note": "The enqueued element is ignored.",
    },
    {
        "fn": "queue-dequeue",
        "buggy": "(define (queue-dequeue queue)\n  (let ([front (car queue)] [back (cdr queue)])\n    (if (null? front)\n        (error 'queue-dequeue \"Cannot dequeue from empty queue\")\n        (values queue (car front)))))",
        "note": "Queue must advance after dequeue.",
    },
    {
        "fn": "queue-dequeue",
        "buggy": "(define (queue-dequeue queue)\n  (let ([front (car queue)] [back (cdr queue)])\n    (if (null? front)\n        (error 'queue-dequeue \"Cannot dequeue from empty queue\")\n        (values (cons (cdr front) back) (car front)))))",
        "note": "Dequeuing must call make-queue to preserve invariant when front empties.",
    },
    {
        "fn": "queue-peek",
        "buggy": "(define (queue-peek queue)\n  (let ([front (car queue)])\n    (if (null? front)\n        #f\n        (car front))))",
        "note": "Peeking empty queue must raise an error.",
    },
    {
        "fn": "queue-peek",
        "buggy": "(define (queue-peek queue)\n  (let ([front (car queue)])\n    (if (null? front)\n        (error 'queue-peek \"Cannot peek empty queue\")\n        (last front))))",
        "note": "Peek should return the front element, not the last front element.",
    },
    {
        "fn": "queue-size",
        "buggy": "(define (queue-size queue)\n  (length (car queue)))",
        "note": "Size must include both front and back lengths.",
    },
    {
        "fn": "queue-size",
        "buggy": "(define (queue-size queue)\n  (+ 1 (length (car queue)) (length (cdr queue))))",
        "note": "Do not add an extra offset.",
    },
    {
        "fn": "queue->list",
        "buggy": "(define (queue->list queue)\n  (append (car queue) (cdr queue)))",
        "note": "Back list must be reversed to recover queue order.",
    },
    {
        "fn": "queue->list",
        "buggy": "(define (queue->list queue)\n  (reverse (car queue)))",
        "note": "Conversion must include both front and back elements in FIFO order.",
    },
    {
        "fn": "list->queue",
        "buggy": "(define (list->queue lst)\n  (cons (reverse lst) '()))",
        "note": "First list element must stay at queue front.",
    },
    {
        "fn": "list->queue",
        "buggy": "(define (list->queue lst)\n  (cons '() lst))",
        "note": "Queue representation is (front . back), with list in front and empty back.",
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
    sid = f"queue_{family}_{family_counter[family]:03d}"
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
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    diff = "easy" if fn in {"queue-empty?", "queue-size", "queue->list", "list->queue"} else "medium"

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=diff,
        source_function=fn,
        prompt=f"""You are implementing Tier-0 queue code in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "queue", "spec-to-code", fn],
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
        tags=["tier0", "data", "queue", "skeleton-completion", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (16)
# -----------------------------------------------------------------------------
for fn in TRANSLATION_FUNCTIONS:
    diff = "easy" if fn in {"queue-empty?", "queue-size", "queue->list", "list->queue"} else "medium"

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
        tags=["tier0", "data", "queue", "python-to-scheme", fn],
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
        tags=["tier0", "data", "queue", "chez-to-fold", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = case["fn"]
    diff = "easy" if fn in {"queue-empty?", "queue-size", "queue->list", "list->queue"} else "medium"

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
        tags=["tier0", "data", "queue", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition/use (32)
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
        tags=["tier0", "data", "queue", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # Direct
    ("queue-empty?", "Return whether `queue-empty` is empty.", "(queue-empty? queue-empty)", "(equal? (queue-empty? queue-empty) #t)", "easy", ["direct"]),
    ("queue-enqueue", "Enqueue `'x` into `queue-empty`.", "(queue-enqueue 'x queue-empty)", "(equal? (queue-enqueue 'x queue-empty) '((x)))", "easy", ["direct"]),
    ("queue->list", "Enqueue `'x`, `'y`, `'z` into empty queue and return queue order as list.", "(queue->list (queue-enqueue 'z (queue-enqueue 'y (queue-enqueue 'x queue-empty))))", "(equal? (queue->list (queue-enqueue 'z (queue-enqueue 'y (queue-enqueue 'x queue-empty)))) '(x y z))", "easy", ["direct"]),
    ("queue-peek", "Return front element of queue built from list `'(a b c)`.", "(queue-peek (list->queue '(a b c)))", "(equal? (queue-peek (list->queue '(a b c))) 'a)", "easy", ["direct"]),
    ("queue-size", "Return queue size for `(list->queue '(p q r s))`.", "(queue-size (list->queue '(p q r s)))", "(equal? (queue-size (list->queue '(p q r s))) 4)", "easy", ["direct"]),
    ("queue-dequeue", "Dequeue once from `(list->queue '(m n o))` and return `(list popped rest-list)`.", "(call-with-values (lambda () (queue-dequeue (list->queue '(m n o)))) (lambda (q x) (list x (queue->list q))))", "(equal? (call-with-values (lambda () (queue-dequeue (list->queue '(m n o)))) (lambda (q x) (list x (queue->list q)))) '(m (n o)))", "medium", ["direct"]),
    ("list->queue", "Convert list `'(1 2 3)` to queue.", "(list->queue '(1 2 3))", "(equal? (list->queue '(1 2 3)) '((1 2 3)))", "easy", ["direct"]),
    ("queue->list", "Convert queue representation `(cons '(a b) '(d c))` to list.", "(queue->list (cons '(a b) '(d c)))", "(equal? (queue->list (cons '(a b) '(d c))) '(a b c d))", "easy", ["direct"]),
    ("queue-dequeue", "Dequeue from queue `(cons '(a) '(c b))` and return `(list popped rest-list)`.", "(call-with-values (lambda () (queue-dequeue (cons '(a) '(c b)))) (lambda (q x) (list x (queue->list q))))", "(equal? (call-with-values (lambda () (queue-dequeue (cons '(a) '(c b)))) (lambda (q x) (list x (queue->list q)))) '(a (b c)))", "medium", ["direct"]),
    ("queue-empty?", "Check if queue `(list->queue '(only))` is empty.", "(queue-empty? (list->queue '(only)))", "(equal? (queue-empty? (list->queue '(only))) #f)", "easy", ["direct"]),
    ("queue->list", "Round-trip `list->queue->queue->list` for `'(u v w)`.", "(queue->list (list->queue '(u v w)))", "(equal? (queue->list (list->queue '(u v w))) '(u v w))", "easy", ["direct"]),
    ("queue-size", "Return the size of `queue-empty`.", "(queue-size queue-empty)", "(equal? (queue-size queue-empty) 0)", "easy", ["direct"]),

    # Properties
    ("queue-dequeue", "Return #t iff two dequeues from `(list->queue '(1 2 3))` produce `1` then `2`.", "(call-with-values (lambda () (queue-dequeue (list->queue '(1 2 3)))) (lambda (q1 x1) (call-with-values (lambda () (queue-dequeue q1)) (lambda (q2 x2) (and (= x1 1) (= x2 2) (equal? (queue->list q2) '(3)))))))", "(equal? (call-with-values (lambda () (queue-dequeue (list->queue '(1 2 3)))) (lambda (q1 x1) (call-with-values (lambda () (queue-dequeue q1)) (lambda (q2 x2) (and (= x1 1) (= x2 2) (equal? (queue->list q2) '(3))))))) #t)", "hard", ["property"]),
    ("queue-size", "Check that enqueue increases queue size by one.", "(= (queue-size (queue-enqueue 'z (list->queue '(a b)))) (+ 1 (queue-size (list->queue '(a b)))))", "(equal? (= (queue-size (queue-enqueue 'z (list->queue '(a b)))) (+ 1 (queue-size (list->queue '(a b))))) #t)", "medium", ["property"]),
    ("queue-size", "Check that dequeue decreases size by one for non-empty queue.", "(call-with-values (lambda () (queue-dequeue (list->queue '(a b c d)))) (lambda (q x) (= (queue-size q) 3)))", "(equal? (call-with-values (lambda () (queue-dequeue (list->queue '(a b c d)))) (lambda (q x) (= (queue-size q) 3))) #t)", "medium", ["property"]),
    ("queue-peek", "Return #t iff peeking after enqueue on empty queue returns the new element.", "(equal? (queue-peek (queue-enqueue 'n queue-empty)) 'n)", "(equal? (queue-peek (queue-enqueue 'n queue-empty)) 'n)", "medium", ["property"]),
    ("list->queue", "Return #t iff first list element stays at queue front after conversion.", "(equal? (queue-peek (list->queue '(h i j))) 'h)", "(equal? (queue-peek (list->queue '(h i j))) 'h)", "medium", ["property"]),
    ("queue-dequeue", "Return #t iff dequeuing single-element queue yields empty queue and that element.", "(call-with-values (lambda () (queue-dequeue (list->queue '(solo)))) (lambda (q x) (and (queue-empty? q) (equal? x 'solo))))", "(equal? (call-with-values (lambda () (queue-dequeue (list->queue '(solo)))) (lambda (q x) (and (queue-empty? q) (equal? x 'solo)))) #t)", "medium", ["property"]),
    ("queue-dequeue", "Return #t iff dequeuing empty queue raises an exception.", "(guard (ex [else #t]) (begin (queue-dequeue queue-empty) #f))", "(equal? (guard (ex [else #t]) (begin (queue-dequeue queue-empty) #f)) #t)", "medium", ["edge-case"]),
    ("queue-peek", "Return #t iff peeking empty queue raises an exception.", "(guard (ex [else #t]) (begin (queue-peek queue-empty) #f))", "(equal? (guard (ex [else #t]) (begin (queue-peek queue-empty) #f)) #t)", "medium", ["edge-case"]),
    ("queue->list", "Check that queue->list after enqueue appends at end.", "(equal? (queue->list (queue-enqueue 'd (list->queue '(a b c)))) '(a b c d))", "(equal? (queue->list (queue-enqueue 'd (list->queue '(a b c)))) '(a b c d))", "medium", ["property"]),
    ("queue-size", "Check that `queue-size` equals `(length (queue->list q))` for q = `(cons '(a b) '(d c))`.", "(let ([q (cons '(a b) '(d c))]) (= (queue-size q) (length (queue->list q))))", "(equal? (let ([q (cons '(a b) '(d c))]) (= (queue-size q) (length (queue->list q)))) #t)", "medium", ["property"]),

    # Fold/loop
    ("queue-enqueue", "Build queue from `'(1 2 3 4)` using `fold-left` and `queue-enqueue`, then return list form.", "(queue->list (fold-left (lambda (q x) (queue-enqueue x q)) queue-empty '(1 2 3 4)))", "(equal? (queue->list (fold-left (lambda (q x) (queue-enqueue x q)) queue-empty '(1 2 3 4))) '(1 2 3 4))", "medium", ["fold"]),
    ("queue-dequeue", "Consume `(list->queue '(x y z))` with repeated dequeue and collect popped elements.", "(let loop ([q (list->queue '(x y z))] [acc '()]) (if (queue-empty? q) (reverse acc) (call-with-values (lambda () (queue-dequeue q)) (lambda (q2 x) (loop q2 (cons x acc))))))", "(equal? (let loop ([q (list->queue '(x y z))] [acc '()]) (if (queue-empty? q) (reverse acc) (call-with-values (lambda () (queue-dequeue q)) (lambda (q2 x) (loop q2 (cons x acc)))))) '(x y z))", "hard", ["loop"]),
    ("queue-dequeue", "Count how many dequeues empty `(list->queue '(a b c d))`.", "(let loop ([q (list->queue '(a b c d))] [n 0]) (if (queue-empty? q) n (call-with-values (lambda () (queue-dequeue q)) (lambda (q2 x) (loop q2 (+ n 1))))))", "(equal? (let loop ([q (list->queue '(a b c d))] [n 0]) (if (queue-empty? q) n (call-with-values (lambda () (queue-dequeue q)) (lambda (q2 x) (loop q2 (+ n 1)))))) 4)", "hard", ["loop"]),
    ("queue-dequeue", "Sum all elements by repeatedly dequeuing `(list->queue '(2 4 6))`.", "(let loop ([q (list->queue '(2 4 6))] [acc 0]) (if (queue-empty? q) acc (call-with-values (lambda () (queue-dequeue q)) (lambda (q2 x) (loop q2 (+ acc x))))))", "(equal? (let loop ([q (list->queue '(2 4 6))] [acc 0]) (if (queue-empty? q) acc (call-with-values (lambda () (queue-dequeue q)) (lambda (q2 x) (loop q2 (+ acc x)))))) 12)", "hard", ["loop"]),
    ("queue-size", "Map queue sizes over queues `'((() . ()) ((a b) . ()) ((x) . (z y)))`.", "(map queue-size (list (cons '() '()) (cons '(a b) '()) (cons '(x) '(z y))))", "(equal? (map queue-size (list (cons '() '()) (cons '(a b) '()) (cons '(x) '(z y)))) '(0 2 3))", "medium", ["list"]),
    ("queue-enqueue", "Use a loop to enqueue all numbers in `'(5 6 7)` and return queue->list.", "(let loop ([xs '(5 6 7)] [q queue-empty]) (if (null? xs) (queue->list q) (loop (cdr xs) (queue-enqueue (car xs) q))))", "(equal? (let loop ([xs '(5 6 7)] [q queue-empty]) (if (null? xs) (queue->list q) (loop (cdr xs) (queue-enqueue (car xs) q)))) '(5 6 7))", "medium", ["loop"]),

    # Integration
    ("queue-dequeue", "From `(list->queue '(a b c))`, dequeue once then enqueue `'d`; return queue order.", "(call-with-values (lambda () (queue-dequeue (list->queue '(a b c)))) (lambda (q x) (queue->list (queue-enqueue 'd q))))", "(equal? (call-with-values (lambda () (queue-dequeue (list->queue '(a b c)))) (lambda (q x) (queue->list (queue-enqueue 'd q)))) '(b c d))", "hard", ["integration"]),
    ("queue-dequeue", "Dequeue twice from `(list->queue '(q r s t))` and return `(list first second remaining-list)`.", "(call-with-values (lambda () (queue-dequeue (list->queue '(q r s t)))) (lambda (q1 x1) (call-with-values (lambda () (queue-dequeue q1)) (lambda (q2 x2) (list x1 x2 (queue->list q2))))))", "(equal? (call-with-values (lambda () (queue-dequeue (list->queue '(q r s t)))) (lambda (q1 x1) (call-with-values (lambda () (queue-dequeue q1)) (lambda (q2 x2) (list x1 x2 (queue->list q2)))))) '(q r (s t)))", "hard", ["integration"]),
    ("queue-enqueue", "Simulate operations: enqueue `1`, enqueue `2`, dequeue once, enqueue `3`; return queue order.", "(call-with-values (lambda () (queue-dequeue (queue-enqueue 2 (queue-enqueue 1 queue-empty)))) (lambda (q x) (queue->list (queue-enqueue 3 q))))", "(equal? (call-with-values (lambda () (queue-dequeue (queue-enqueue 2 (queue-enqueue 1 queue-empty)))) (lambda (q x) (queue->list (queue-enqueue 3 q)))) '(2 3))", "hard", ["integration"]),
    ("queue-dequeue", "Return #t iff dequeuing all elements from `(list->queue '(1 2 3 4))` yields them in insertion order.", "(let loop ([q (list->queue '(1 2 3 4))] [expected '(1 2 3 4)]) (if (null? expected) (queue-empty? q) (call-with-values (lambda () (queue-dequeue q)) (lambda (q2 x) (and (= x (car expected)) (loop q2 (cdr expected)))))))", "(equal? (let loop ([q (list->queue '(1 2 3 4))] [expected '(1 2 3 4)]) (if (null? expected) (queue-empty? q) (call-with-values (lambda () (queue-dequeue q)) (lambda (q2 x) (and (= x (car expected)) (loop q2 (cdr expected))))))) #t)", "hard", ["integration"]),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

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

if len(train_rows) != 66 or len(eval_rows) != 14:
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
