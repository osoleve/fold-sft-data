#!/usr/bin/env python3
"""Generate Tier-0 SFT samples for lattice/data/heap.ss."""

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
from sft_split_utils import compute_leakage_aware_eval_ids
ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

SOURCE_MODULE = "lattice/data/heap.ss"
SOURCE_TEST = "lattice/data/test-heap.ss"

DEFS: Dict[str, str] = {
    "heap-empty?": """(define (heap-empty? heap)
  (eq? heap 'heap-empty))""",
    "make-heap-node": """(define (make-heap-node value left right)
  (let ([rank-l (heap-rank left)]
        [rank-r (heap-rank right)])
    (if (>= rank-l rank-r)
        (heap-node (+ 1 rank-r) value left right)
        (heap-node (+ 1 rank-l) value right left))))""",
    "heap-merge": """(define (heap-merge h1 h2)
  (cond
    [(heap-empty? h1) h2]
    [(heap-empty? h2) h1]
    [else
     (let ([v1 (heap-value h1)]
           [v2 (heap-value h2)])
       (if (<= v1 v2)
           (make-heap-node v1 (heap-left h1) (heap-merge (heap-right h1) h2))
           (make-heap-node v2 (heap-left h2) (heap-merge h1 (heap-right h2)))))]))""",
    "heap-insert": """(define (heap-insert elem heap)
  (heap-merge (heap-node 1 elem heap-empty heap-empty) heap))""",
    "heap-min": """(define (heap-min heap)
  (if (heap-empty? heap)
      (error 'heap-min "Cannot get min of empty heap")
      (heap-value heap)))""",
    "heap-delete-min": """(define (heap-delete-min heap)
  (if (heap-empty? heap)
      (error 'heap-delete-min "Cannot delete from empty heap")
      (heap-merge (heap-left heap) (heap-right heap))))""",
    "heap-pop": """(define (heap-pop heap)
  (if (heap-empty? heap)
      (error 'heap-pop "Cannot pop from empty heap")
      (values (heap-merge (heap-left heap) (heap-right heap))
              (heap-value heap))))""",
    "heap-size": """(define (heap-size heap)
  (if (heap-empty? heap)
      0
      (+ 1 (heap-size (heap-left heap)) (heap-size (heap-right heap)))))""",
    "heap-merge-by": """(define (heap-merge-by cmp h1 h2)
  (cond
    [(heap-empty? h1) h2]
    [(heap-empty? h2) h1]
    [else
     (let ([v1 (heap-value h1)]
           [v2 (heap-value h2)])
       (if (cmp v1 v2)
           (make-heap-node v1 (heap-left h1) (heap-merge-by cmp (heap-right h1) h2))
           (make-heap-node v2 (heap-left h2) (heap-merge-by cmp h1 (heap-right h2)))))]))""",
    "heap-insert-by": """(define (heap-insert-by cmp elem heap)
  (heap-merge-by cmp (heap-node 1 elem heap-empty heap-empty) heap))""",
    "heap-delete-top-by": """(define (heap-delete-top-by cmp heap)
  (if (heap-empty? heap)
      (error 'heap-delete-top-by "Cannot delete from empty heap")
      (heap-merge-by cmp (heap-left heap) (heap-right heap))))""",
    "heap-fold": """(define (heap-fold fn init heap)
  (if (heap-empty? heap)
      init
      (let* ([acc (fn init (heap-value heap))]
             [acc-right (heap-fold fn acc (heap-right heap))])
        (heap-fold fn acc-right (heap-left heap)))))""",
    "list->heap-by": """(define (list->heap-by cmp lst)
  (fold-left (lambda (h x) (heap-insert-by cmp x h)) heap-empty lst))""",
    "heap->list": """(define (heap->list heap)
  (if (heap-empty? heap)
      '()
      (cons (heap-min heap)
            (heap->list (heap-delete-min heap)))))""",
    "heapsort": """(define (heapsort lst)
  (heap->list (list->heap lst)))""",
    "heapsort-by": """(define (heapsort-by cmp lst)
  (let loop ([h (list->heap-by cmp lst)] [acc '()])
    (if (heap-empty? h)
        (reverse acc)
        (loop (heap-delete-top-by cmp h) (cons (heap-value h) acc)))))""",
}

SUPPORT_DEFS: Dict[str, str] = {
    "heap-empty": "(define heap-empty 'heap-empty)",
    "heap-node": """(define (heap-node rank value left right)
  (list 'heap-node rank value left right))""",
    "heap-node?": """(define (heap-node? x)
  (and (pair? x)
       (eq? (car x) 'heap-node)))""",
    "heap-rank": """(define (heap-rank h)
  (if (heap-empty? h) 0 (cadr h)))""",
    "heap-value": """(define (heap-value h)
  (caddr h))""",
    "heap-left": """(define (heap-left h)
  (cadddr h))""",
    "heap-right": """(define (heap-right h)
  (car (cddddr h)))""",
    "heap-min": """(define (heap-min heap)
  (if (heap-empty? heap)
      (error 'heap-min "Cannot get min of empty heap")
      (heap-value heap)))""",
    "list->heap": """(define (list->heap lst)
  (fold-left (lambda (h x) (heap-insert x h)) heap-empty lst))""",
}

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **DEFS}

DEPENDS: Dict[str, List[str]] = {
    "heap-empty": [],
    "heap-node": [],
    "heap-node?": [],
    "heap-empty?": [],
    "heap-rank": ["heap-empty?"],
    "heap-value": [],
    "heap-left": [],
    "heap-right": [],
    "heap-min": ["heap-empty?", "heap-value"],
    "make-heap-node": ["heap-rank", "heap-node", "heap-empty?"],
    "heap-merge": ["heap-empty?", "heap-value", "make-heap-node", "heap-left", "heap-right"],
    "heap-insert": ["heap-merge", "heap-node", "heap-empty"],
    "heap-delete-min": ["heap-empty?", "heap-merge", "heap-left", "heap-right"],
    "heap-pop": ["heap-empty?", "heap-merge", "heap-left", "heap-right", "heap-value"],
    "heap-size": ["heap-empty?", "heap-left", "heap-right"],
    "heap-merge-by": ["heap-empty?", "heap-value", "make-heap-node", "heap-left", "heap-right"],
    "heap-insert-by": ["heap-merge-by", "heap-node", "heap-empty"],
    "heap-delete-top-by": ["heap-empty?", "heap-merge-by", "heap-left", "heap-right"],
    "heap-fold": ["heap-empty?", "heap-value", "heap-right", "heap-left"],
    "list->heap-by": ["heap-insert-by", "heap-empty"],
    "heap->list": ["heap-empty?", "heap-min", "heap-delete-min"],
    "heapsort": ["heap->list", "list->heap"],
    "heapsort-by": ["list->heap-by", "heap-empty?", "heap-delete-top-by", "heap-value"],
    "list->heap": ["heap-insert", "heap-empty"],
}

CORE_FUNCTIONS = [
    "heap-empty?",
    "make-heap-node",
    "heap-merge",
    "heap-insert",
    "heap-delete-min",
    "heap-pop",
    "heap-size",
    "heap->list",
]

FUNCTION_ORDER = [
    "heap-empty?",
    "make-heap-node",
    "heap-merge",
    "heap-insert",
    "heap-min",
    "heap-delete-min",
    "heap-pop",
    "heap-size",
    "heap-merge-by",
    "heap-insert-by",
    "heap-delete-top-by",
    "heap-fold",
    "list->heap-by",
    "heap->list",
    "heapsort",
    "heapsort-by",
]

SUPPORT_ORDER = [
    "heap-empty",
    "heap-node",
    "heap-node?",
    "heap-rank",
    "heap-value",
    "heap-left",
    "heap-right",
    "heap-min",
    "list->heap",
]

FUNCTION_SPECS = {
    "heap-empty?": "Return #t iff heap is the empty-heap sentinel.",
    "make-heap-node": "Construct a node while enforcing leftist rank ordering (larger rank subtree on the left).",
    "heap-merge": "Merge two min-heaps preserving heap ordering and leftist property.",
    "heap-insert": "Insert elem by merging heap with singleton node.",
    "heap-min": "Return minimum element at root; raise an error for empty heap.",
    "heap-delete-min": "Remove minimum element from non-empty heap; raise an error on empty heap.",
    "heap-pop": "Return two values (new-heap, min-value); raise an error on empty heap.",
    "heap-size": "Return the number of nodes in the heap tree.",
    "heap-merge-by": "Merge two heaps using comparator `(cmp a b)` deciding which root wins.",
    "heap-insert-by": "Insert with custom comparator by merging a singleton heap.",
    "heap-delete-top-by": "Delete comparator-defined top element; raise an error on empty heap.",
    "heap-fold": "Fold all heap values with an accumulator function.",
    "list->heap-by": "Build a heap from list using a custom comparator.",
    "heap->list": "Extract all heap elements in ascending order.",
    "heapsort": "Sort a list in ascending order via heap extraction.",
    "heapsort-by": "Sort a list using comparator ordering via custom heap operations.",
}

SKELETONS = {
    "heap-empty?": """(define (heap-empty? heap)
  ;; TODO: check empty-heap sentinel
  <TODO>)""",
    "make-heap-node": """(define (make-heap-node value left right)
  ;; TODO: enforce leftist rank invariant when building a node
  <TODO>)""",
    "heap-merge": """(define (heap-merge h1 h2)
  ;; TODO: merge two min-heaps recursively
  <TODO>)""",
    "heap-insert": """(define (heap-insert elem heap)
  ;; TODO: insert by merging a singleton node with heap
  <TODO>)""",
    "heap-min": """(define (heap-min heap)
  ;; TODO: return root value and error on empty heap
  <TODO>)""",
    "heap-delete-min": """(define (heap-delete-min heap)
  ;; TODO: remove root and merge children; error on empty
  <TODO>)""",
    "heap-pop": """(define (heap-pop heap)
  ;; TODO: return (values new-heap min-value)
  <TODO>)""",
    "heap-size": """(define (heap-size heap)
  ;; TODO: count nodes recursively
  <TODO>)""",
    "heap-merge-by": """(define (heap-merge-by cmp h1 h2)
  ;; TODO: merge with custom comparator
  <TODO>)""",
    "heap-insert-by": """(define (heap-insert-by cmp elem heap)
  ;; TODO: insert using heap-merge-by
  <TODO>)""",
    "heap-delete-top-by": """(define (heap-delete-top-by cmp heap)
  ;; TODO: delete comparator-defined top element
  <TODO>)""",
    "heap-fold": """(define (heap-fold fn init heap)
  ;; TODO: fold all heap elements
  <TODO>)""",
    "list->heap-by": """(define (list->heap-by cmp lst)
  ;; TODO: build heap with comparator-aware inserts
  <TODO>)""",
    "heap->list": """(define (heap->list heap)
  ;; TODO: repeatedly extract min to produce sorted list
  <TODO>)""",
    "heapsort": """(define (heapsort lst)
  ;; TODO: sort ascending using heap->list and list->heap
  <TODO>)""",
    "heapsort-by": """(define (heapsort-by cmp lst)
  ;; TODO: sort using custom comparator heap pipeline
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "heap-empty?": "(and (heap-empty? heap-empty) (not (heap-empty? (heap-node 1 5 heap-empty heap-empty))))",
    "make-heap-node": "(let* ([l (heap-node 2 4 (heap-node 1 6 heap-empty heap-empty) heap-empty)] [r (heap-node 1 5 heap-empty heap-empty)] [n (make-heap-node 3 l r)]) (and (heap-node? n) (>= (heap-rank (heap-left n)) (heap-rank (heap-right n)))))",
    "heap-merge": "(equal? (heap->list (heap-merge (list->heap '(5 1 9)) (list->heap '(4 2 8)))) '(1 2 4 5 8 9))",
    "heap-insert": "(let ([h (heap-insert 3 (list->heap '(7 2 9)))]) (and (= (heap-min h) 2) (= (heap-size h) 4) (equal? (heap->list h) '(2 3 7 9))))",
    "heap-min": "(and (= (heap-min (list->heap '(5 1 4))) 1) (guard (ex [else #t]) (begin (heap-min heap-empty) #f)))",
    "heap-delete-min": "(and (equal? (heap->list (heap-delete-min (list->heap '(5 1 4 3)))) '(3 4 5)) (guard (ex [else #t]) (begin (heap-delete-min heap-empty) #f)))",
    "heap-pop": "(and (call-with-values (lambda () (heap-pop (list->heap '(5 1 4 3)))) (lambda (h x) (and (= x 1) (equal? (heap->list h) '(3 4 5))))) (guard (ex [else #t]) (begin (heap-pop heap-empty) #f)))",
    "heap-size": "(and (= (heap-size heap-empty) 0) (= (heap-size (list->heap '(9 2 7 1))) 4))",
    "heap-merge-by": "(let loop ([h (heap-merge-by > (list->heap-by > '(3 7)) (list->heap-by > '(4 6)))] [acc '()]) (if (heap-empty? h) (equal? (reverse acc) '(7 6 4 3)) (loop (heap-delete-top-by > h) (cons (heap-value h) acc))))",
    "heap-insert-by": "(let ([h (heap-insert-by > 9 (list->heap-by > '(3 7 4)))]) (and (= (heap-value h) 9) (= (heap-size h) 4)))",
    "heap-delete-top-by": "(let* ([h (list->heap-by > '(4 1 7 3))] [h2 (heap-delete-top-by > h)]) (and (= (heap-value h) 7) (= (heap-value h2) 4) (= (heap-size h2) 3)))",
    "heap-fold": "(let ([h (list->heap '(5 1 4 3))]) (and (= (heap-fold (lambda (acc x) (+ acc x)) 0 h) 13) (= (heap-fold (lambda (acc x) (+ acc 1)) 0 h) 4)))",
    "list->heap-by": "(let loop ([h (list->heap-by > '(3 7 4 6))] [acc '()]) (if (heap-empty? h) (equal? (reverse acc) '(7 6 4 3)) (loop (heap-delete-top-by > h) (cons (heap-value h) acc))))",
    "heap->list": "(and (equal? (heap->list (list->heap '(5 2 8 1 9 3))) '(1 2 3 5 8 9)) (equal? (heap->list heap-empty) '()))",
    "heapsort": "(equal? (heapsort '(5 2 8 1 2)) '(1 2 2 5 8))",
    "heapsort-by": "(equal? (heapsort-by > '(5 2 8 1 2)) '(8 5 2 2 1))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = CORE_FUNCTIONS

PYTHON_SNIPPETS = {
    "heap-empty?": "def heap_empty(heap):\n    return heap == 'heap-empty'",
    "make-heap-node": "def make_heap_node(value, left, right):\n    rank_l = heap_rank(left)\n    rank_r = heap_rank(right)\n    if rank_l >= rank_r:\n        return heap_node(1 + rank_r, value, left, right)\n    return heap_node(1 + rank_l, value, right, left)",
    "heap-merge": "def heap_merge(h1, h2):\n    if heap_empty(h1):\n        return h2\n    if heap_empty(h2):\n        return h1\n    v1 = heap_value(h1)\n    v2 = heap_value(h2)\n    if v1 <= v2:\n        return make_heap_node(v1, heap_left(h1), heap_merge(heap_right(h1), h2))\n    return make_heap_node(v2, heap_left(h2), heap_merge(h1, heap_right(h2)))",
    "heap-insert": "def heap_insert(elem, heap):\n    return heap_merge(heap_node(1, elem, heap_empty_const, heap_empty_const), heap)",
    "heap-delete-min": "def heap_delete_min(heap):\n    if heap_empty(heap):\n        raise ValueError('Cannot delete from empty heap')\n    return heap_merge(heap_left(heap), heap_right(heap))",
    "heap-pop": "def heap_pop(heap):\n    if heap_empty(heap):\n        raise ValueError('Cannot pop from empty heap')\n    return (heap_merge(heap_left(heap), heap_right(heap)), heap_value(heap))",
    "heap-size": "def heap_size(heap):\n    if heap_empty(heap):\n        return 0\n    return 1 + heap_size(heap_left(heap)) + heap_size(heap_right(heap))",
    "heap->list": "def heap_to_list(heap):\n    if heap_empty(heap):\n        return []\n    return [heap_min(heap)] + heap_to_list(heap_delete_min(heap))",
}

CHEZ_SNIPPETS = {
    "heap-empty?": "(define (empty? h)\n  (eq? h 'heap-empty))",
    "make-heap-node": "(define (mk-node v l r)\n  (let ((rl (heap-rank l))\n        (rr (heap-rank r)))\n    (if (>= rl rr)\n        (heap-node (+ rr 1) v l r)\n        (heap-node (+ rl 1) v r l))))",
    "heap-merge": "(define (merge0 a b)\n  (cond\n    ((heap-empty? a) b)\n    ((heap-empty? b) a)\n    (else\n      (let ((va (heap-value a))\n            (vb (heap-value b)))\n        (if (<= va vb)\n            (make-heap-node va (heap-left a) (merge0 (heap-right a) b))\n            (make-heap-node vb (heap-left b) (merge0 a (heap-right b))))))))",
    "heap-insert": "(define (insert0 x h)\n  (heap-merge (heap-node 1 x heap-empty heap-empty) h))",
    "heap-delete-min": "(define (delete-min0 h)\n  (if (heap-empty? h)\n      (error 'heap-delete-min \"Cannot delete from empty heap\")\n      (heap-merge (heap-left h) (heap-right h))))",
    "heap-pop": "(define (pop0 h)\n  (if (heap-empty? h)\n      (error 'heap-pop \"Cannot pop from empty heap\")\n      (values (heap-merge (heap-left h) (heap-right h))\n              (heap-value h))))",
    "heap-size": "(define (size0 h)\n  (if (heap-empty? h)\n      0\n      (+ 1 (size0 (heap-left h)) (size0 (heap-right h)))))",
    "heap->list": "(define (to-list h)\n  (if (heap-empty? h)\n      '()\n      (cons (heap-min h) (to-list (heap-delete-min h)))))",
}

BUGGY_CASES = [
    {
        "fn": "heap-empty?",
        "buggy": "(define (heap-empty? heap)\n  (null? heap))",
        "note": "Empty heap is represented by the sentinel symbol, not the empty list.",
    },
    {
        "fn": "heap-empty?",
        "buggy": "(define (heap-empty? heap)\n  #f)",
        "note": "Must return #t for heap-empty.",
    },
    {
        "fn": "make-heap-node",
        "buggy": "(define (make-heap-node value left right)\n  (heap-node (+ 1 (heap-rank right)) value left right))",
        "note": "Must swap children when right rank exceeds left rank.",
    },
    {
        "fn": "make-heap-node",
        "buggy": "(define (make-heap-node value left right)\n  (let ([rank-l (heap-rank left)]\n        [rank-r (heap-rank right)])\n    (if (>= rank-l rank-r)\n        (heap-node (+ 1 rank-l) value left right)\n        (heap-node (+ 1 rank-r) value right left))))",
        "note": "Node rank should be based on the right subtree rank after ordering.",
    },
    {
        "fn": "heap-merge",
        "buggy": "(define (heap-merge h1 h2)\n  (if (heap-empty? h1) h2 h1))",
        "note": "Merge must handle both non-empty heaps recursively.",
    },
    {
        "fn": "heap-merge",
        "buggy": "(define (heap-merge h1 h2)\n  (cond\n    [(heap-empty? h1) h2]\n    [(heap-empty? h2) h1]\n    [else\n      (let ([v1 (heap-value h1)]\n            [v2 (heap-value h2)])\n        (if (>= v1 v2)\n            (make-heap-node v1 (heap-left h1) (heap-merge (heap-right h1) h2))\n            (make-heap-node v2 (heap-left h2) (heap-merge h1 (heap-right h2)))))]))",
        "note": "Min-heap merge must keep the smaller root, not the larger one.",
    },
    {
        "fn": "heap-insert",
        "buggy": "(define (heap-insert elem heap)\n  heap)",
        "note": "Insert must add the new element.",
    },
    {
        "fn": "heap-insert",
        "buggy": "(define (heap-insert elem heap)\n  (if (heap-empty? heap)\n      (heap-node 1 elem heap-empty heap-empty)\n      heap))",
        "note": "Insert must add elem for non-empty heaps too.",
    },
    {
        "fn": "heap-delete-min",
        "buggy": "(define (heap-delete-min heap)\n  (heap-right heap))",
        "note": "Deleting min must merge left and right sub-heaps.",
    },
    {
        "fn": "heap-delete-min",
        "buggy": "(define (heap-delete-min heap)\n  (if (heap-empty? heap)\n      heap-empty\n      (heap-merge (heap-left heap) (heap-right heap))))",
        "note": "Empty input should raise an error, not silently return heap-empty.",
    },
    {
        "fn": "heap-pop",
        "buggy": "(define (heap-pop heap)\n  (if (heap-empty? heap)\n      (error 'heap-pop \"Cannot pop from empty heap\")\n      (values heap (heap-value heap))))",
        "note": "Returned heap must remove the top element.",
    },
    {
        "fn": "heap-pop",
        "buggy": "(define (heap-pop heap)\n  (if (heap-empty? heap)\n      heap-empty\n      (values (heap-merge (heap-left heap) (heap-right heap))\n              (heap-value heap))))",
        "note": "On empty heap this function must raise an error and preserve return arity.",
    },
    {
        "fn": "heap-size",
        "buggy": "(define (heap-size heap)\n  (if (heap-empty? heap)\n      1\n      (+ (heap-size (heap-left heap)) (heap-size (heap-right heap)))))",
        "note": "Base case for empty heap must be 0 and node count must include current root.",
    },
    {
        "fn": "heap-size",
        "buggy": "(define (heap-size heap)\n  (if (heap-empty? heap)\n      0\n      (+ 1 (heap-size (heap-left heap)))))",
        "note": "Size must include both subtrees.",
    },
    {
        "fn": "heap->list",
        "buggy": "(define (heap->list heap)\n  (if (heap-empty? heap)\n      '()\n      (cons (heap-min heap)\n            (heap->list (heap-right heap)))))",
        "note": "Extraction must repeatedly delete min; traversing only right subtree drops elements.",
    },
    {
        "fn": "heap->list",
        "buggy": "(define (heap->list heap)\n  '())",
        "note": "Non-empty heaps must produce all elements in ascending order.",
    },
]

DIFFICULTY = {
    "heap-empty?": "easy",
    "make-heap-node": "medium",
    "heap-merge": "hard",
    "heap-insert": "medium",
    "heap-min": "easy",
    "heap-delete-min": "medium",
    "heap-pop": "hard",
    "heap-size": "easy",
    "heap-merge-by": "hard",
    "heap-insert-by": "medium",
    "heap-delete-top-by": "hard",
    "heap-fold": "medium",
    "list->heap-by": "medium",
    "heap->list": "hard",
    "heapsort": "easy",
    "heapsort-by": "hard",
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
    sid = f"heap_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
        "prompt_body": prompt.strip(),
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
    names = FUNCTION_ORDER + SUPPORT_ORDER
    return [name for name in names if name != fn and name in tokens]


def dependency_closure(fn: str) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    def visit(name: str) -> None:
        for dep in DEPENDS.get(name, []):
            if dep == fn:
                continue
            if dep not in seen:
                seen.add(dep)
                visit(dep)
                ordered.append(dep)

    for dep in DEPENDS.get(fn, []):
        if dep == fn:
            continue
        if dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    for dep in verify_refs(fn):
        if dep == fn:
            continue
        if dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    return ordered


def def_verify(fn: str) -> str:
    parts = [ALL_DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


# -----------------------------------------------------------------------------
# Family 1: spec_to_code
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""You are implementing Tier-0 heap code in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "heap", "spec-to-code", fn],
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
        tags=["tier0", "data", "heap", "skeleton-completion", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation
# -----------------------------------------------------------------------------
for fn in TRANSLATION_FUNCTIONS:
    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Translate the following Python function into Fold-native Scheme.
Preserve behavior exactly and use the target function name `{fn}`.
Return only the Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "heap", "python-to-scheme", fn],
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
        tags=["tier0", "data", "heap", "chez-to-fold", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix
# -----------------------------------------------------------------------------
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
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "heap", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition/use
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
        tags=["tier0", "data", "heap", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # Direct
    ("heap-empty?", "Return whether heap-empty is empty.", "(heap-empty? heap-empty)", "(equal? (heap-empty? heap-empty) #t)", "easy", ["direct"]),
    ("heap-empty?", "Return whether a singleton heap is empty.", "(heap-empty? (heap-node 1 9 heap-empty heap-empty))", "(equal? (heap-empty? (heap-node 1 9 heap-empty heap-empty)) #f)", "easy", ["direct"]),
    ("make-heap-node", "Create a heap node with children ranked 2 and 1, then return whether left rank is >= right rank.", "(let* ([l (heap-node 2 10 (heap-node 1 11 heap-empty heap-empty) heap-empty)] [r (heap-node 1 12 heap-empty heap-empty)] [n (make-heap-node 5 l r)]) (>= (heap-rank (heap-left n)) (heap-rank (heap-right n))))", "(equal? (let* ([l (heap-node 2 10 (heap-node 1 11 heap-empty heap-empty) heap-empty)] [r (heap-node 1 12 heap-empty heap-empty)] [n (make-heap-node 5 l r)]) (>= (heap-rank (heap-left n)) (heap-rank (heap-right n)))) #t)", "medium", ["direct"]),
    ("make-heap-node", "Build make-heap-node with swapped-rank inputs and return root value.", "(heap-value (make-heap-node 4 (heap-node 1 7 heap-empty heap-empty) (heap-node 2 8 (heap-node 1 9 heap-empty heap-empty) heap-empty)))", "(equal? (heap-value (make-heap-node 4 (heap-node 1 7 heap-empty heap-empty) (heap-node 2 8 (heap-node 1 9 heap-empty heap-empty) heap-empty))) 4)", "medium", ["direct"]),
    ("heap-merge", "Merge heaps built from '(1 5 9) and '(2 4 8), then return sorted extraction.", "(heap->list (heap-merge (list->heap '(1 5 9)) (list->heap '(2 4 8))))", "(equal? (heap->list (heap-merge (list->heap '(1 5 9)) (list->heap '(2 4 8)))) '(1 2 4 5 8 9))", "medium", ["direct"]),
    ("heap-merge", "Merge heap-empty with a non-empty heap and return heap->list.", "(heap->list (heap-merge heap-empty (list->heap '(3 1 2))))", "(equal? (heap->list (heap-merge heap-empty (list->heap '(3 1 2)))) '(1 2 3))", "easy", ["edge-case"]),
    ("heap-insert", "Insert 3 into heap from '(7 2 9) and return heap minimum.", "(heap-min (heap-insert 3 (list->heap '(7 2 9))))", "(equal? (heap-min (heap-insert 3 (list->heap '(7 2 9)))) 2)", "easy", ["direct"]),
    ("heap-insert", "Insert 6 into heap from '(7 2 9) and return resulting sorted list.", "(heap->list (heap-insert 6 (list->heap '(7 2 9))))", "(equal? (heap->list (heap-insert 6 (list->heap '(7 2 9)))) '(2 6 7 9))", "medium", ["direct"]),
    ("heap-delete-min", "Delete minimum from heap built from '(5 1 4 3) and return heap->list.", "(heap->list (heap-delete-min (list->heap '(5 1 4 3))))", "(equal? (heap->list (heap-delete-min (list->heap '(5 1 4 3)))) '(3 4 5))", "medium", ["direct"]),
    ("heap-delete-min", "Return #t iff heap-delete-min on empty heap raises an error.", "(guard (ex [else #t]) (begin (heap-delete-min heap-empty) #f))", "(equal? (guard (ex [else #t]) (begin (heap-delete-min heap-empty) #f)) #t)", "medium", ["edge-case"]),
    ("heap-pop", "Pop heap from '(10 2 7 5) and return (list popped remaining-list).", "(call-with-values (lambda () (heap-pop (list->heap '(10 2 7 5)))) (lambda (h x) (list x (heap->list h))))", "(equal? (call-with-values (lambda () (heap-pop (list->heap '(10 2 7 5)))) (lambda (h x) (list x (heap->list h)))) '(2 (5 7 10)))", "hard", ["direct"]),
    ("heap-pop", "Return #t iff heap-pop on empty heap raises an error.", "(guard (ex [else #t]) (begin (heap-pop heap-empty) #f))", "(equal? (guard (ex [else #t]) (begin (heap-pop heap-empty) #f)) #t)", "hard", ["edge-case"]),
    ("heap-size", "Return size of heap-empty.", "(heap-size heap-empty)", "(equal? (heap-size heap-empty) 0)", "easy", ["direct"]),
    ("heap-size", "Return size of heap built from '(9 2 7 1).", "(heap-size (list->heap '(9 2 7 1)))", "(equal? (heap-size (list->heap '(9 2 7 1))) 4)", "easy", ["direct"]),
    ("heap->list", "Convert heap from '(5 2 8 1 9 3) to sorted list.", "(heap->list (list->heap '(5 2 8 1 9 3)))", "(equal? (heap->list (list->heap '(5 2 8 1 9 3))) '(1 2 3 5 8 9))", "medium", ["direct"]),
    ("heap->list", "Convert heap-empty to list.", "(heap->list heap-empty)", "(equal? (heap->list heap-empty) '())", "easy", ["direct"]),

    # Properties
    ("heap-merge", "Check that merge size equals the sum of input sizes.", "(= (heap-size (heap-merge (list->heap '(1 5 9)) (list->heap '(2 4 8 10)))) (+ (heap-size (list->heap '(1 5 9))) (heap-size (list->heap '(2 4 8 10)))))", "(equal? (= (heap-size (heap-merge (list->heap '(1 5 9)) (list->heap '(2 4 8 10)))) (+ (heap-size (list->heap '(1 5 9))) (heap-size (list->heap '(2 4 8 10))))) #t)", "medium", ["property"]),
    ("heap-merge", "Check that merge minimum equals min of both input minima.", "(= (heap-min (heap-merge (list->heap '(4 7 9)) (list->heap '(2 5 8)))) (min (heap-min (list->heap '(4 7 9))) (heap-min (list->heap '(2 5 8)))))", "(equal? (= (heap-min (heap-merge (list->heap '(4 7 9)) (list->heap '(2 5 8)))) (min (heap-min (list->heap '(4 7 9))) (heap-min (list->heap '(2 5 8))))) #t)", "medium", ["property"]),
    ("heap-insert", "Check that heap-insert increases size by one.", "(= (heap-size (heap-insert 6 (list->heap '(7 2 9)))) (+ 1 (heap-size (list->heap '(7 2 9)))))", "(equal? (= (heap-size (heap-insert 6 (list->heap '(7 2 9)))) (+ 1 (heap-size (list->heap '(7 2 9))))) #t)", "medium", ["property"]),
    ("heap-insert", "Check that inserting a smaller element updates heap-min.", "(= (heap-min (heap-insert 1 (list->heap '(7 2 9)))) 1)", "(equal? (= (heap-min (heap-insert 1 (list->heap '(7 2 9)))) 1) #t)", "medium", ["property"]),
    ("heap-delete-min", "Check that deleting min decreases size by one for non-empty heap.", "(let ([h (list->heap '(6 1 4 2))]) (= (heap-size (heap-delete-min h)) (- (heap-size h) 1)))", "(equal? (let ([h (list->heap '(6 1 4 2))]) (= (heap-size (heap-delete-min h)) (- (heap-size h) 1))) #t)", "medium", ["property"]),
    ("heap-pop", "Check that heap-pop returns heap-min as the popped value.", "(let ([h (list->heap '(6 1 4 2))]) (call-with-values (lambda () (heap-pop h)) (lambda (h2 x) (= x (heap-min h)))))", "(equal? (let ([h (list->heap '(6 1 4 2))]) (call-with-values (lambda () (heap-pop h)) (lambda (h2 x) (= x (heap-min h))))) #t)", "hard", ["property"]),
    ("heap-size", "Check that heap-size equals length of heap->list.", "(let ([h (list->heap '(8 3 5 1 9))]) (= (heap-size h) (length (heap->list h))))", "(equal? (let ([h (list->heap '(8 3 5 1 9))]) (= (heap-size h) (length (heap->list h)))) #t)", "medium", ["property"]),
    ("heap->list", "Check that heap->list output is sorted for '(8 3 5 1 9).", "(let ([xs (heap->list (list->heap '(8 3 5 1 9)))]) (or (null? xs) (null? (cdr xs)) (let loop ([rest (cdr xs)] [prev (car xs)]) (if (null? rest) #t (and (<= prev (car rest)) (loop (cdr rest) (car rest)))))))", "(equal? (let ([xs (heap->list (list->heap '(8 3 5 1 9)))]) (or (null? xs) (null? (cdr xs)) (let loop ([rest (cdr xs)] [prev (car xs)]) (if (null? rest) #t (and (<= prev (car rest)) (loop (cdr rest) (car rest))))))) #t)", "hard", ["property"]),

    # Fold/loop/integration
    ("heap-insert", "Build a heap with fold-left inserts from '(5 2 8 1 9 3), then return heap->list.", "(let ([h (fold-left (lambda (acc x) (heap-insert x acc)) heap-empty '(5 2 8 1 9 3))]) (heap->list h))", "(equal? (let ([h (fold-left (lambda (acc x) (heap-insert x acc)) heap-empty '(5 2 8 1 9 3))]) (heap->list h)) '(1 2 3 5 8 9))", "hard", ["fold"]),
    ("heap-pop", "Pop repeatedly from heap '(5 2 8 1 9 3) and collect values.", "(let loop ([h (list->heap '(5 2 8 1 9 3))] [acc '()]) (if (heap-empty? h) (reverse acc) (call-with-values (lambda () (heap-pop h)) (lambda (h2 x) (loop h2 (cons x acc))))))", "(equal? (let loop ([h (list->heap '(5 2 8 1 9 3))] [acc '()]) (if (heap-empty? h) (reverse acc) (call-with-values (lambda () (heap-pop h)) (lambda (h2 x) (loop h2 (cons x acc)))))) '(1 2 3 5 8 9))", "hard", ["loop"]),
    ("heap-merge", "Merge two heaps then pop two values and return them as a list.", "(let ([m (heap-merge (list->heap '(7 3 9)) (list->heap '(6 2 8)))]) (call-with-values (lambda () (heap-pop m)) (lambda (h1 a) (call-with-values (lambda () (heap-pop h1)) (lambda (h2 b) (list a b))))))", "(equal? (let ([m (heap-merge (list->heap '(7 3 9)) (list->heap '(6 2 8)))]) (call-with-values (lambda () (heap-pop m)) (lambda (h1 a) (call-with-values (lambda () (heap-pop h1)) (lambda (h2 b) (list a b)))))) '(2 3))", "hard", ["integration"]),
    ("heap-delete-min", "Insert 0 then delete min; result should match sorted original values.", "(heap->list (heap-delete-min (heap-insert 0 (list->heap '(4 2 6 8)))))", "(equal? (heap->list (heap-delete-min (heap-insert 0 (list->heap '(4 2 6 8))))) '(2 4 6 8))", "medium", ["integration"]),
    ("heap-size", "Count how many pops are needed to empty heap '(4 1 3 2).", "(let loop ([h (list->heap '(4 1 3 2))] [n 0]) (if (heap-empty? h) n (call-with-values (lambda () (heap-pop h)) (lambda (h2 x) (loop h2 (+ n 1))))))", "(equal? (let loop ([h (list->heap '(4 1 3 2))] [n 0]) (if (heap-empty? h) n (call-with-values (lambda () (heap-pop h)) (lambda (h2 x) (loop h2 (+ n 1)))))) 4)", "hard", ["loop"]),
    ("heap-size", "Check that merged heap size matches length of its extracted list.", "(let ([m (heap-merge (list->heap '(1 4 7)) (list->heap '(2 5 8 9)))]) (= (heap-size m) (length (heap->list m))))", "(equal? (let ([m (heap-merge (list->heap '(1 4 7)) (list->heap '(2 5 8 9)))]) (= (heap-size m) (length (heap->list m)))) #t)", "medium", ["integration"]),
    ("make-heap-node", "Return #t iff make-heap-node output satisfies leftist rank ordering.", "(let* ([a (heap-node 1 8 heap-empty heap-empty)] [b (heap-node 2 9 (heap-node 1 11 heap-empty heap-empty) heap-empty)] [n (make-heap-node 7 a b)]) (>= (heap-rank (heap-left n)) (heap-rank (heap-right n))))", "(equal? (let* ([a (heap-node 1 8 heap-empty heap-empty)] [b (heap-node 2 9 (heap-node 1 11 heap-empty heap-empty) heap-empty)] [n (make-heap-node 7 a b)]) (>= (heap-rank (heap-left n)) (heap-rank (heap-right n)))) #t)", "medium", ["property"]),
    ("heap->list", "Sort '(9 1 5 3 7) by going through list->heap and heap->list.", "(heap->list (list->heap '(9 1 5 3 7)))", "(equal? (heap->list (list->heap '(9 1 5 3 7))) '(1 3 5 7 9))", "medium", ["integration"]),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

if sum(1 for s in samples if s["family"] == "composition") != 32:
    raise ValueError("composition family must contain exactly 32 samples")

# -----------------------------------------------------------------------------
# Split train/eval
# -----------------------------------------------------------------------------
eval_ids = compute_leakage_aware_eval_ids(
    samples,
    eval_ratio=0.20,
    eval_min_by_family={
        "spec_to_code": max(3, len(FUNCTION_ORDER) // 6),
        "translation": 3,
        "bugfix": 3,
        "composition": 5,
    },
    enforce_source_function_coverage=True,
)

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

if not train_rows or not eval_rows:
    raise ValueError("split generation failed: train/eval must both be non-empty")


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


write_jsonl(ALL_PATH, [dict(s, split=("eval" if s["id"] in eval_ids else "train")) for s in samples])
write_jsonl(TRAIN_PATH, train_rows)
write_jsonl(EVAL_PATH, eval_rows)

by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

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
    "difficulty": dict(sorted(Counter(str(s["difficulty"]) for s in samples).items())),
    "source_functions": dict(sorted(Counter(str(s["source_function"]) for s in samples).items())),
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
