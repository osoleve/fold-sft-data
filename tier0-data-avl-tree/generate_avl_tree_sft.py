#!/usr/bin/env python3
"""Generate Tier-0 SFT samples for lattice/data/avl-tree.ss."""

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
from sft_split_utils import compute_leakage_aware_eval_ids

ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

SOURCE_MODULE = "lattice/data/avl-tree.ss"
SOURCE_TEST = "lattice/data/test-avl-tree.ss"

DEFS: Dict[str, str] = {
    "avl-empty?": """(define (avl-empty? tree)
  (eq? tree 'avl-empty))""",
    "make-avl-node": """(define (make-avl-node key value left right)
  (avl-node (+ 1 (max (avl-height left) (avl-height right)))
            key value left right))""",
    "rebalance": """(define (rebalance tree)
  (let ([bf (avl-balance-factor tree)])
    (cond
      [(> bf 1)
       (if (< (avl-balance-factor (avl-left tree)) 0)
           (rotate-right (make-avl-node (avl-key tree) (avl-value tree)
                                        (rotate-left (avl-left tree))
                                        (avl-right tree)))
           (rotate-right tree))]
      [(< bf -1)
       (if (> (avl-balance-factor (avl-right tree)) 0)
           (rotate-left (make-avl-node (avl-key tree) (avl-value tree)
                                       (avl-left tree)
                                       (rotate-right (avl-right tree))))
           (rotate-left tree))]
      [else tree])))""",
    "avl-lookup": """(define (avl-lookup key tree)
  (avl-lookup-by < key tree))""",
    "avl-lookup-by": """(define (avl-lookup-by cmp key tree)
  (if (avl-empty? tree)
      #f
      (let ([k (avl-key tree)])
        (cond
          [(cmp key k) (avl-lookup-by cmp key (avl-left tree))]
          [(cmp k key) (avl-lookup-by cmp key (avl-right tree))]
          [else (avl-value tree)]))))""",
    "avl-contains?": """(define (avl-contains? key tree)
  (avl-contains-by? < key tree))""",
    "avl-insert": """(define (avl-insert key value tree)
  (avl-insert-by < key value tree))""",
    "avl-insert-by": """(define (avl-insert-by cmp key value tree)
  (if (avl-empty? tree)
      (make-avl-node key value avl-empty avl-empty)
      (let ([k (avl-key tree)]
            [v (avl-value tree)]
            [left (avl-left tree)]
            [right (avl-right tree)])
        (cond
          [(cmp key k)
           (rebalance (make-avl-node k v (avl-insert-by cmp key value left) right))]
          [(cmp k key)
           (rebalance (make-avl-node k v left (avl-insert-by cmp key value right)))]
          [else
           (make-avl-node key value left right)]))))""",
    "avl-delete-min-by": """(define (avl-delete-min-by cmp tree)
  (if (avl-empty? tree)
      (error 'avl-delete-min "Cannot delete from empty tree")
          (if (avl-empty? (avl-left tree))
              (avl-right tree)
              (rebalance (make-avl-node (avl-key tree) (avl-value tree)
                                    (avl-delete-min-by cmp (avl-left tree))
                                    (avl-right tree))))))""",
    "avl-delete": """(define (avl-delete key tree)
  (avl-delete-by < key tree))""",
    "avl-delete-by": """(define (avl-delete-by cmp key tree)
  (if (avl-empty? tree)
      tree
      (let ([k (avl-key tree)]
            [v (avl-value tree)]
            [left (avl-left tree)]
            [right (avl-right tree)])
        (cond
          [(cmp key k)
           (rebalance (make-avl-node k v (avl-delete-by cmp key left) right))]
          [(cmp k key)
           (rebalance (make-avl-node k v left (avl-delete-by cmp key right)))]
          [else
           (cond
             [(avl-empty? left) right]
             [(avl-empty? right) left]
             [else
              (let ([succ (avl-min-node right)])
                (rebalance (make-avl-node (car succ) (cdr succ)
                                          left
                                          (avl-delete-min-by cmp right))))])]))))""",
    "avl-range": """(define (avl-range lo hi tree)
  (avl-range-by < lo hi tree))""",
    "avl-range-by": """(define (avl-range-by cmp lo hi tree)
  (if (avl-empty? tree)
      '()
      (let ([k (avl-key tree)]
            [v (avl-value tree)])
        (append
         (if (cmp lo k)
             (avl-range-by cmp lo hi (avl-left tree))
             '())
         (if (and (not (cmp k lo)) (not (cmp hi k)))
             (list (cons k v))
             '())
         (if (cmp k hi)
             (avl-range-by cmp lo hi (avl-right tree))
             '())))))""",
    "avl-keys-between": """(define (avl-keys-between lo hi tree)
  (map car (avl-range lo hi tree)))""",
    "avl-less-than": """(define (avl-less-than bound tree)
  (if (avl-empty? tree)
      '()
      (let ([k (avl-key tree)])
        (if (< k bound)
            (append (avl-less-than bound (avl-left tree))
                    (list (cons k (avl-value tree)))
                    (avl-less-than bound (avl-right tree)))
            (avl-less-than bound (avl-left tree))))))""",
    "avl-greater-than": """(define (avl-greater-than bound tree)
  (if (avl-empty? tree)
      '()
      (let ([k (avl-key tree)])
        (if (> k bound)
            (append (avl-greater-than bound (avl-left tree))
                    (list (cons k (avl-value tree)))
                    (avl-greater-than bound (avl-right tree)))
            (avl-greater-than bound (avl-right tree))))))""",
}

SUPPORT_DEFS: Dict[str, str] = {
    "avl-empty": "(define avl-empty 'avl-empty)",
    "avl-node": """(define (avl-node height key value left right)
  (list 'avl-node height key value left right))""",
    "avl-height": """(define (avl-height tree)
  (if (avl-empty? tree) 0 (cadr tree)))""",
    "avl-key": """(define (avl-key tree)
  (caddr tree))""",
    "avl-value": """(define (avl-value tree)
  (cadddr tree))""",
    "avl-left": """(define (avl-left tree)
  (car (cddddr tree)))""",
    "avl-right": """(define (avl-right tree)
  (cadr (cddddr tree)))""",
    "avl-balance-factor": """(define (avl-balance-factor tree)
  (if (avl-empty? tree)
      0
      (- (avl-height (avl-left tree))
         (avl-height (avl-right tree)))))""",
    "rotate-left": """(define (rotate-left tree)
  (let ([x-key (avl-key tree)]
        [x-val (avl-value tree)]
        [a (avl-left tree)]
        [y (avl-right tree)])
    (let ([y-key (avl-key y)]
          [y-val (avl-value y)]
          [b (avl-left y)]
          [c (avl-right y)])
      (make-avl-node y-key y-val
                     (make-avl-node x-key x-val a b)
                     c))))""",
    "rotate-right": """(define (rotate-right tree)
  (let ([y-key (avl-key tree)]
        [y-val (avl-value tree)]
        [x (avl-left tree)]
        [c (avl-right tree)])
    (let ([x-key (avl-key x)]
          [x-val (avl-value x)]
          [a (avl-left x)]
          [b (avl-right x)])
      (make-avl-node x-key x-val
                     a
                     (make-avl-node y-key y-val b c)))))""",
    "avl-insert": """(define (avl-insert key value tree)
  (avl-insert-by < key value tree))""",
    "list->avl": """(define (list->avl lst)
  (fold-left (lambda (tree pair)
               (avl-insert (car pair) (cdr pair) tree))
             avl-empty
             lst))""",
    "avl->list": """(define (avl->list tree)
  (if (avl-empty? tree)
      '()
      (append (avl->list (avl-left tree))
              (list (cons (avl-key tree) (avl-value tree)))
              (avl->list (avl-right tree)))))""",
    "avl-keys": """(define (avl-keys tree)
  (map car (avl->list tree)))""",
    "avl-size": """(define (avl-size tree)
  (if (avl-empty? tree)
      0
      (+ 1 (avl-size (avl-left tree)) (avl-size (avl-right tree)))))""",
    "avl-valid?": """(define (avl-valid? tree)
  (if (avl-empty? tree)
      #t
      (let ([bf (avl-balance-factor tree)])
        (and (>= bf -1)
             (<= bf 1)
             (avl-valid? (avl-left tree))
             (avl-valid? (avl-right tree))))))""",
    "avl-bst-valid?": """(define (avl-bst-valid? tree)
  (avl-bst-valid-range? tree #f #f))""",
    "avl-bst-valid-range?": """(define (avl-bst-valid-range? tree lo hi)
  (if (avl-empty? tree)
      #t
      (let ([k (avl-key tree)])
        (and (or (not lo) (< lo k))
             (or (not hi) (< k hi))
             (avl-bst-valid-range? (avl-left tree) lo k)
             (avl-bst-valid-range? (avl-right tree) k hi)))))""",
    "avl-min-node": """(define (avl-min-node tree)
  (if (avl-empty? tree)
      (error 'avl-min-node "Cannot find min in empty tree")
      (if (avl-empty? (avl-left tree))
          (cons (avl-key tree) (avl-value tree))
          (avl-min-node (avl-left tree)))))""",
    "avl-contains-by?": """(define (avl-contains-by? cmp key tree)
  (if (avl-empty? tree)
      #f
      (let ([k (avl-key tree)])
        (cond
          [(cmp key k) (avl-contains-by? cmp key (avl-left tree))]
          [(cmp k key) (avl-contains-by? cmp key (avl-right tree))]
          [else #t]))))""",
}

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **DEFS}

CORE_FUNCTIONS = [
    "avl-empty?",
    "make-avl-node",
    "rebalance",
    "avl-lookup-by",
    "avl-insert-by",
    "avl-delete-min-by",
    "avl-delete-by",
    "avl-range-by",
]

FUNCTION_ORDER = [
    "avl-empty?",
    "make-avl-node",
    "rebalance",
    "avl-lookup",
    "avl-lookup-by",
    "avl-contains?",
    "avl-insert",
    "avl-insert-by",
    "avl-delete-min-by",
    "avl-delete",
    "avl-delete-by",
    "avl-range",
    "avl-range-by",
    "avl-keys-between",
    "avl-less-than",
    "avl-greater-than",
]

SUPPORT_ORDER = list(SUPPORT_DEFS.keys())
ALL_NAMES = FUNCTION_ORDER + SUPPORT_ORDER

FUNCTION_SPECS = {
    "avl-empty?": "Return #t iff tree is the avl-empty sentinel.",
    "make-avl-node": "Build an AVL node and recompute its cached height as 1 + max(height(left), height(right)).",
    "rebalance": "Restore AVL balance for a node by applying single/double rotations when balance factor is outside [-1, 1].",
    "avl-lookup": "Lookup key with default `<` comparator; return value or #f.",
    "avl-lookup-by": "Lookup key with comparator cmp; return stored value or #f when key is absent.",
    "avl-contains?": "Return whether key is present using default `<` comparator.",
    "avl-insert": "Insert/update with default `<` comparator and preserve AVL invariants.",
    "avl-insert-by": "Insert/update (key, value) under cmp and rebalance on the way back up.",
    "avl-delete-min-by": "Delete the minimum element under cmp; raise an error on empty tree.",
    "avl-delete": "Delete key using default `<` comparator while preserving AVL invariants.",
    "avl-delete-by": "Delete key under cmp, using in-order successor replacement for two-child nodes and preserving AVL invariants.",
    "avl-range": "Return inclusive range pairs using default `<` comparator.",
    "avl-range-by": "Return in-order key/value pairs with lo <= key <= hi under comparator cmp.",
    "avl-keys-between": "Return all keys in inclusive [lo, hi].",
    "avl-less-than": "Return all key/value pairs with key < bound, in key order.",
    "avl-greater-than": "Return all key/value pairs with key > bound, in key order.",
}

SKELETONS = {
    "avl-empty?": """(define (avl-empty? tree)
  ;; TODO: recognize the AVL empty sentinel
  <TODO>)""",
    "make-avl-node": """(define (make-avl-node key value left right)
  ;; TODO: rebuild node with recomputed cached height
  <TODO>)""",
    "rebalance": """(define (rebalance tree)
  ;; TODO: rotate for LL/LR/RR/RL imbalance cases
  <TODO>)""",
    "avl-lookup": """(define (avl-lookup key tree)
  ;; TODO: delegate to comparator-based lookup with <
  <TODO>)""",
    "avl-lookup-by": """(define (avl-lookup-by cmp key tree)
  ;; TODO: comparator-driven BST lookup; return #f when missing
  <TODO>)""",
    "avl-contains?": """(define (avl-contains? key tree)
  ;; TODO: delegate to comparator-based contains? with <
  <TODO>)""",
    "avl-insert": """(define (avl-insert key value tree)
  ;; TODO: delegate to comparator-based insert with <
  <TODO>)""",
    "avl-insert-by": """(define (avl-insert-by cmp key value tree)
  ;; TODO: insert/update and rebalance recursively
  <TODO>)""",
    "avl-delete-min-by": """(define (avl-delete-min-by cmp tree)
  ;; TODO: delete smallest key; raise error on empty tree
  <TODO>)""",
    "avl-delete": """(define (avl-delete key tree)
  ;; TODO: delegate to comparator-based delete with <
  <TODO>)""",
    "avl-delete-by": """(define (avl-delete-by cmp key tree)
  ;; TODO: delete key with successor replacement + rebalancing
  <TODO>)""",
    "avl-range": """(define (avl-range lo hi tree)
  ;; TODO: delegate to comparator-based range with <
  <TODO>)""",
    "avl-range-by": """(define (avl-range-by cmp lo hi tree)
  ;; TODO: return sorted pairs in inclusive [lo, hi] range
  <TODO>)""",
    "avl-keys-between": """(define (avl-keys-between lo hi tree)
  ;; TODO: map car over avl-range result
  <TODO>)""",
    "avl-less-than": """(define (avl-less-than bound tree)
  ;; TODO: return in-order pairs where key < bound
  <TODO>)""",
    "avl-greater-than": """(define (avl-greater-than bound tree)
  ;; TODO: return in-order pairs where key > bound
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "avl-empty?": "(and (avl-empty? avl-empty) (not (avl-empty? (avl-node 1 5 \"x\" avl-empty avl-empty))))",
    "make-avl-node": "(let* ([left (avl-node 2 2 \"b\" (avl-node 1 1 \"a\" avl-empty avl-empty) avl-empty)] [right (avl-node 1 4 \"d\" avl-empty avl-empty)] [n (make-avl-node 3 \"c\" left right)]) (and (= (avl-height n) 3) (= (avl-key n) 3) (equal? (avl-value n) \"c\") (equal? (avl-left n) left) (equal? (avl-right n) right)))",
    "rebalance": "(let* ([n3 (make-avl-node 3 \"c\" avl-empty avl-empty)] [n2 (make-avl-node 2 \"b\" avl-empty n3)] [n1 (make-avl-node 1 \"a\" avl-empty n2)] [r (rebalance n1)]) (and (avl-valid? r) (avl-bst-valid? r) (equal? (avl-keys r) '(1 2 3))))",
    "avl-lookup": "(let ([t (list->avl '((5 . \"e\") (3 . \"c\") (7 . \"g\")))]) (and (equal? (avl-lookup 7 t) \"g\") (equal? (avl-lookup 8 t) #f)))",
    "avl-lookup-by": "(let* ([pairs '((5 . \"five\") (2 . \"two\") (9 . \"nine\"))] [asc (fold-left (lambda (acc kv) (avl-insert-by < (car kv) (cdr kv) acc)) avl-empty pairs)] [desc (fold-left (lambda (acc kv) (avl-insert-by > (car kv) (cdr kv) acc)) avl-empty pairs)]) (and (equal? (avl-lookup-by < 9 asc) \"nine\") (equal? (avl-lookup-by < 8 asc) #f) (equal? (avl-lookup-by > 2 desc) \"two\")))",
    "avl-contains?": "(let ([t (list->avl '((5 . \"e\") (3 . \"c\") (7 . \"g\")))]) (and (avl-contains? 3 t) (not (avl-contains? 42 t))))",
    "avl-insert": "(let* ([t0 (list->avl '((2 . \"two\") (1 . \"one\") (3 . \"three\")))] [t1 (avl-insert 2 \"TWO\" t0)] [t2 (avl-insert 4 \"four\" t1)]) (and (= (avl-size t2) 4) (equal? (avl-lookup 2 t2) \"TWO\") (equal? (avl-keys t2) '(1 2 3 4))))",
    "avl-insert-by": "(let* ([t0 (fold-left (lambda (acc k) (avl-insert-by < k (* k 10) acc)) avl-empty '(5 3 7))] [t1 (avl-insert-by < 6 60 t0)] [t2 (avl-insert-by < 7 700 t1)]) (and (avl-valid? t2) (avl-bst-valid? t2) (= (avl-size t2) 4) (= (avl-lookup-by < 7 t2) 700) (equal? (avl-keys t2) '(3 5 6 7))))",
    "avl-delete-min-by": "(and (equal? (avl-keys (avl-delete-min-by < (list->avl '((4 . \"d\") (2 . \"b\") (6 . \"f\") (1 . \"a\"))))) '(2 4 6)) (guard (ex [else #t]) (begin (avl-delete-min-by < avl-empty) #f)))",
    "avl-delete": "(let* ([t (list->avl '((5 . \"e\") (3 . \"c\") (7 . \"g\") (2 . \"b\")))] [d (avl-delete 3 t)] [d2 (avl-delete 99 d)]) (and (equal? (avl-keys d) '(2 5 7)) (equal? (avl-keys d2) '(2 5 7)) (avl-valid? d2) (avl-bst-valid? d2)))",
    "avl-delete-by": "(let* ([t (list->avl '((5 . \"e\") (3 . \"c\") (7 . \"g\") (2 . \"b\") (4 . \"d\") (6 . \"f\") (8 . \"h\")))] [d1 (avl-delete-by < 7 t)] [d2 (avl-delete-by < 5 d1)] [d3 (avl-delete-by < 42 d2)]) (and (avl-valid? d2) (avl-bst-valid? d2) (not (avl-contains-by? < 7 d1)) (not (avl-contains-by? < 5 d2)) (= (avl-size d2) 5) (= (avl-size d3) 5) (equal? (avl-keys d2) '(2 3 4 6 8))))",
    "avl-range": "(let ([t (list->avl '((1 . \"a\") (3 . \"c\") (5 . \"e\") (7 . \"g\") (9 . \"i\")))]) (equal? (avl-range 3 7 t) '((3 . \"c\") (5 . \"e\") (7 . \"g\"))))",
    "avl-range-by": "(let* ([t (list->avl '((1 . \"a\") (3 . \"c\") (5 . \"e\") (7 . \"g\") (9 . \"i\")))] [r1 (avl-range-by < 3 7 t)] [r2 (avl-range-by < 8 10 t)] [r3 (avl-range-by < 20 30 t)]) (and (equal? r1 '((3 . \"c\") (5 . \"e\") (7 . \"g\"))) (equal? r2 '((9 . \"i\"))) (equal? r3 '())))",
    "avl-keys-between": "(let ([t (list->avl '((1 . \"a\") (3 . \"c\") (5 . \"e\") (7 . \"g\") (9 . \"i\")))]) (equal? (avl-keys-between 3 7 t) '(3 5 7)))",
    "avl-less-than": "(let ([t (list->avl '((1 . \"a\") (3 . \"c\") (5 . \"e\") (7 . \"g\")))]) (equal? (avl-less-than 5 t) '((1 . \"a\") (3 . \"c\"))))",
    "avl-greater-than": "(let ([t (list->avl '((1 . \"a\") (3 . \"c\") (5 . \"e\") (7 . \"g\")))]) (equal? (avl-greater-than 3 t) '((5 . \"e\") (7 . \"g\"))))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "avl-empty?": "def avl_empty(tree):\n    return tree == 'avl-empty'",
    "make-avl-node": "def make_avl_node(key, value, left, right):\n    return avl_node(1 + max(avl_height(left), avl_height(right)), key, value, left, right)",
    "rebalance": "def rebalance(tree):\n    bf = avl_balance_factor(tree)\n    if bf > 1:\n        if avl_balance_factor(avl_left(tree)) < 0:\n            return rotate_right(make_avl_node(avl_key(tree), avl_value(tree), rotate_left(avl_left(tree)), avl_right(tree)))\n        return rotate_right(tree)\n    if bf < -1:\n        if avl_balance_factor(avl_right(tree)) > 0:\n            return rotate_left(make_avl_node(avl_key(tree), avl_value(tree), avl_left(tree), rotate_right(avl_right(tree))))\n        return rotate_left(tree)\n    return tree",
    "avl-lookup": "def avl_lookup(key, tree):\n    return avl_lookup_by(lt, key, tree)",
    "avl-lookup-by": "def avl_lookup_by(cmp, key, tree):\n    if avl_empty(tree):\n        return False\n    k = avl_key(tree)\n    if cmp(key, k):\n        return avl_lookup_by(cmp, key, avl_left(tree))\n    if cmp(k, key):\n        return avl_lookup_by(cmp, key, avl_right(tree))\n    return avl_value(tree)",
    "avl-contains?": "def avl_contains(key, tree):\n    return avl_contains_by(lt, key, tree)",
    "avl-insert": "def avl_insert(key, value, tree):\n    return avl_insert_by(lt, key, value, tree)",
    "avl-insert-by": "def avl_insert_by(cmp, key, value, tree):\n    if avl_empty(tree):\n        return make_avl_node(key, value, avl_empty_const, avl_empty_const)\n    k = avl_key(tree)\n    v = avl_value(tree)\n    left = avl_left(tree)\n    right = avl_right(tree)\n    if cmp(key, k):\n        return rebalance(make_avl_node(k, v, avl_insert_by(cmp, key, value, left), right))\n    if cmp(k, key):\n        return rebalance(make_avl_node(k, v, left, avl_insert_by(cmp, key, value, right)))\n    return make_avl_node(key, value, left, right)",
    "avl-delete-min-by": "def avl_delete_min_by(cmp, tree):\n    if avl_empty(tree):\n        raise ValueError('Cannot delete from empty tree')\n    if avl_empty(avl_left(tree)):\n        return avl_right(tree)\n    return rebalance(make_avl_node(avl_key(tree), avl_value(tree), avl_delete_min_by(cmp, avl_left(tree)), avl_right(tree)))",
    "avl-delete": "def avl_delete(key, tree):\n    return avl_delete_by(lt, key, tree)",
    "avl-delete-by": "def avl_delete_by(cmp, key, tree):\n    if avl_empty(tree):\n        return tree\n    k = avl_key(tree)\n    v = avl_value(tree)\n    left = avl_left(tree)\n    right = avl_right(tree)\n    if cmp(key, k):\n        return rebalance(make_avl_node(k, v, avl_delete_by(cmp, key, left), right))\n    if cmp(k, key):\n        return rebalance(make_avl_node(k, v, left, avl_delete_by(cmp, key, right)))\n    if avl_empty(left):\n        return right\n    if avl_empty(right):\n        return left\n    succ_key, succ_val = avl_min_node(right)\n    return rebalance(make_avl_node(succ_key, succ_val, left, avl_delete_min_by(cmp, right)))",
    "avl-range": "def avl_range(lo, hi, tree):\n    return avl_range_by(lt, lo, hi, tree)",
    "avl-range-by": "def avl_range_by(cmp, lo, hi, tree):\n    if avl_empty(tree):\n        return []\n    k = avl_key(tree)\n    v = avl_value(tree)\n    left = avl_range_by(cmp, lo, hi, avl_left(tree)) if cmp(lo, k) else []\n    mid = [(k, v)] if (not cmp(k, lo) and not cmp(hi, k)) else []\n    right = avl_range_by(cmp, lo, hi, avl_right(tree)) if cmp(k, hi) else []\n    return left + mid + right",
    "avl-keys-between": "def avl_keys_between(lo, hi, tree):\n    return [k for (k, _) in avl_range(lo, hi, tree)]",
    "avl-less-than": "def avl_less_than(bound, tree):\n    if avl_empty(tree):\n        return []\n    k = avl_key(tree)\n    if k < bound:\n        return avl_less_than(bound, avl_left(tree)) + [(k, avl_value(tree))] + avl_less_than(bound, avl_right(tree))\n    return avl_less_than(bound, avl_left(tree))",
    "avl-greater-than": "def avl_greater_than(bound, tree):\n    if avl_empty(tree):\n        return []\n    k = avl_key(tree)\n    if k > bound:\n        return avl_greater_than(bound, avl_left(tree)) + [(k, avl_value(tree))] + avl_greater_than(bound, avl_right(tree))\n    return avl_greater_than(bound, avl_right(tree))",
}

CHEZ_SNIPPETS = {
    "avl-empty?": "(define (empty-tree? t)\n  (eq? t 'avl-empty))",
    "make-avl-node": "(define (mk-node k v l r)\n  (avl-node (+ 1 (max (avl-height l) (avl-height r)))\n            k v l r))",
    "rebalance": "(define (rebalance0 t)\n  (let ((bf (avl-balance-factor t)))\n    (cond\n      ((> bf 1)\n       (if (< (avl-balance-factor (avl-left t)) 0)\n           (rotate-right (make-avl-node (avl-key t) (avl-value t)\n                                        (rotate-left (avl-left t))\n                                        (avl-right t)))\n           (rotate-right t)))\n      ((< bf -1)\n       (if (> (avl-balance-factor (avl-right t)) 0)\n           (rotate-left (make-avl-node (avl-key t) (avl-value t)\n                                       (avl-left t)\n                                       (rotate-right (avl-right t))))\n           (rotate-left t)))\n      (else t))))",
    "avl-lookup": "(define (lookup-default k t)\n  (avl-lookup-by < k t))",
    "avl-lookup-by": "(define (lookup0 cmp key t)\n  (if (avl-empty? t)\n      #f\n      (let ((k (avl-key t)))\n        (cond\n          ((cmp key k) (lookup0 cmp key (avl-left t)))\n          ((cmp k key) (lookup0 cmp key (avl-right t)))\n          (else (avl-value t))))))",
    "avl-contains?": "(define (contains0 key t)\n  (avl-contains-by? < key t))",
    "avl-insert": "(define (insert-default key value t)\n  (avl-insert-by < key value t))",
    "avl-insert-by": "(define (insert0 cmp key value t)\n  (if (avl-empty? t)\n      (make-avl-node key value avl-empty avl-empty)\n      (let ((k (avl-key t))\n            (v (avl-value t))\n            (l (avl-left t))\n            (r (avl-right t)))\n        (cond\n          ((cmp key k) (rebalance (make-avl-node k v (insert0 cmp key value l) r)))\n          ((cmp k key) (rebalance (make-avl-node k v l (insert0 cmp key value r))))\n          (else (make-avl-node key value l r))))))",
    "avl-delete-min-by": "(define (delete-min0 cmp t)\n  (if (avl-empty? t)\n      (error 'avl-delete-min \"Cannot delete from empty tree\")\n      (if (avl-empty? (avl-left t))\n          (avl-right t)\n          (rebalance (make-avl-node (avl-key t)\n                                    (avl-value t)\n                                    (delete-min0 cmp (avl-left t))\n                                    (avl-right t))))))",
    "avl-delete": "(define (delete-default key t)\n  (avl-delete-by < key t))",
    "avl-delete-by": "(define (delete0 cmp key t)\n  (if (avl-empty? t)\n      t\n      (let ((k (avl-key t))\n            (v (avl-value t))\n            (l (avl-left t))\n            (r (avl-right t)))\n        (cond\n          ((cmp key k) (rebalance (make-avl-node k v (delete0 cmp key l) r)))\n          ((cmp k key) (rebalance (make-avl-node k v l (delete0 cmp key r))))\n          (else\n            (cond\n              ((avl-empty? l) r)\n              ((avl-empty? r) l)\n              (else\n                (let ((succ (avl-min-node r)))\n                  (rebalance (make-avl-node (car succ)\n                                            (cdr succ)\n                                            l\n                                            (avl-delete-min-by cmp r)))))))))))",
    "avl-range": "(define (range-default lo hi t)\n  (avl-range-by < lo hi t))",
    "avl-range-by": "(define (range0 cmp lo hi t)\n  (if (avl-empty? t)\n      '()\n      (let ((k (avl-key t))\n            (v (avl-value t)))\n        (append\n          (if (cmp lo k)\n              (range0 cmp lo hi (avl-left t))\n              '())\n          (if (and (not (cmp k lo)) (not (cmp hi k)))\n              (list (cons k v))\n              '())\n          (if (cmp k hi)\n              (range0 cmp lo hi (avl-right t))\n              '())))))",
    "avl-keys-between": "(define (keys-between0 lo hi t)\n  (map car (avl-range lo hi t)))",
    "avl-less-than": "(define (lt0 bound t)\n  (if (avl-empty? t)\n      '()\n      (let ((k (avl-key t)))\n        (if (< k bound)\n            (append (lt0 bound (avl-left t))\n                    (list (cons k (avl-value t)))\n                    (lt0 bound (avl-right t)))\n            (lt0 bound (avl-left t))))))",
    "avl-greater-than": "(define (gt0 bound t)\n  (if (avl-empty? t)\n      '()\n      (let ((k (avl-key t)))\n        (if (> k bound)\n            (append (gt0 bound (avl-left t))\n                    (list (cons k (avl-value t)))\n                    (gt0 bound (avl-right t)))\n            (gt0 bound (avl-right t))))))",
}

BUGGY_CASES = [
    {
        "fn": "avl-empty?",
        "buggy": "(define (avl-empty? tree)\n  (null? tree))",
        "note": "AVL emptiness uses the avl-empty sentinel, not the null list.",
    },
    {
        "fn": "avl-empty?",
        "buggy": "(define (avl-empty? tree)\n  #f)",
        "note": "The empty tree must return #t.",
    },
    {
        "fn": "make-avl-node",
        "buggy": "(define (make-avl-node key value left right)\n  (avl-node (+ 1 (min (avl-height left) (avl-height right)))\n            key value left right))",
        "note": "Height must use max child height, not min.",
    },
    {
        "fn": "make-avl-node",
        "buggy": "(define (make-avl-node key value left right)\n  (avl-node (max (avl-height left) (avl-height right))\n            key value left right))",
        "note": "Node height must include the current node (+1).",
    },
    {
        "fn": "rebalance",
        "buggy": "(define (rebalance tree)\n  (let ([bf (avl-balance-factor tree)])\n    (if (> bf 1)\n        (rotate-right tree)\n        tree)))",
        "note": "Right-heavy and double-rotation cases are missing.",
    },
    {
        "fn": "rebalance",
        "buggy": "(define (rebalance tree)\n  (let ([bf (avl-balance-factor tree)])\n    (cond\n      [(> bf 1)\n       (rotate-left tree)]\n      [(< bf -1)\n       (rotate-right tree)]\n      [else tree])))",
        "note": "Rotation direction is inverted for both imbalance directions.",
    },
    {
        "fn": "avl-lookup",
        "buggy": "(define (avl-lookup key tree)\n  (avl-lookup-by > key tree))",
        "note": "Default lookup must use `<` ordering to match the tree construction convention.",
    },
    {
        "fn": "avl-lookup-by",
        "buggy": "(define (avl-lookup-by cmp key tree)\n  (if (avl-empty? tree)\n      #f\n      (let ([k (avl-key tree)])\n        (cond\n          [(cmp key k) (avl-lookup-by cmp key (avl-right tree))]\n          [(cmp k key) (avl-lookup-by cmp key (avl-left tree))]\n          [else (avl-value tree)]))))",
        "note": "Comparator branches should follow BST direction: key<k goes left, k<key goes right.",
    },
    {
        "fn": "avl-lookup-by",
        "buggy": "(define (avl-lookup-by cmp key tree)\n  (if (avl-empty? tree)\n      #f\n      (let ([k (avl-key tree)])\n        (cond\n          [(cmp key k) (avl-lookup-by cmp key (avl-left tree))]\n          [(cmp k key) (avl-lookup-by cmp key (avl-right tree))]\n          [else #t]))))",
        "note": "Lookup must return the stored value, not a boolean marker.",
    },
    {
        "fn": "avl-contains?",
        "buggy": "(define (avl-contains? key tree)\n  (not (avl-empty? tree)))",
        "note": "Containment must depend on the queried key, not just whether tree is non-empty.",
    },
    {
        "fn": "avl-insert",
        "buggy": "(define (avl-insert key value tree)\n  (avl-insert-by > key value tree))",
        "note": "Default insert must delegate with `<`, not `>`.",
    },
    {
        "fn": "avl-insert-by",
        "buggy": "(define (avl-insert-by cmp key value tree)\n  (if (avl-empty? tree)\n      (make-avl-node key value avl-empty avl-empty)\n      (let ([k (avl-key tree)]\n            [v (avl-value tree)]\n            [left (avl-left tree)]\n            [right (avl-right tree)])\n        (cond\n          [(cmp key k)\n           (make-avl-node k v (avl-insert-by cmp key value left) right)]\n          [(cmp k key)\n           (make-avl-node k v left (avl-insert-by cmp key value right))]\n          [else\n           (make-avl-node key value left right)]))))",
        "note": "Recursive insert paths must rebalance before returning.",
    },
    {
        "fn": "avl-insert-by",
        "buggy": "(define (avl-insert-by cmp key value tree)\n  (if (avl-empty? tree)\n      (make-avl-node key value avl-empty avl-empty)\n      (let ([k (avl-key tree)]\n            [v (avl-value tree)]\n            [left (avl-left tree)]\n            [right (avl-right tree)])\n        (cond\n          [(cmp key k)\n           (rebalance (make-avl-node k v (avl-insert-by cmp key value left) right))]\n          [(cmp k key)\n           (rebalance (make-avl-node k v left (avl-insert-by cmp key value right)))]\n          [else\n           (make-avl-node k v left right)]))))",
        "note": "Updating an existing key must store the new value, not keep the old one.",
    },
    {
        "fn": "avl-delete-min-by",
        "buggy": "(define (avl-delete-min-by cmp tree)\n  (if (avl-empty? tree)\n      avl-empty\n      (if (avl-empty? (avl-left tree))\n          (avl-right tree)\n          (rebalance (make-avl-node (avl-key tree) (avl-value tree)\n                                    (avl-delete-min-by cmp (avl-left tree))\n                                    (avl-right tree))))))",
        "note": "Deleting min from an empty tree should raise an error.",
    },
    {
        "fn": "avl-delete-min-by",
        "buggy": "(define (avl-delete-min-by cmp tree)\n  (if (avl-empty? tree)\n      (error 'avl-delete-min \"Cannot delete from empty tree\")\n      (if (avl-empty? (avl-left tree))\n          (avl-left tree)\n          (rebalance (make-avl-node (avl-key tree) (avl-value tree)\n                                    (avl-delete-min-by cmp (avl-left tree))\n                                    (avl-right tree))))))",
        "note": "When left child is empty, the node should be replaced by its right subtree.",
    },
    {
        "fn": "avl-delete",
        "buggy": "(define (avl-delete key tree)\n  (avl-delete-by > key tree))",
        "note": "Default delete must delegate with `<` comparator.",
    },
    {
        "fn": "avl-delete-by",
        "buggy": "(define (avl-delete-by cmp key tree)\n  (if (avl-empty? tree)\n      tree\n      (let ([k (avl-key tree)]\n            [v (avl-value tree)]\n            [left (avl-left tree)]\n            [right (avl-right tree)])\n        (cond\n          [(cmp key k)\n           (rebalance (make-avl-node k v (avl-delete-by cmp key left) right))]\n          [(cmp k key)\n           (rebalance (make-avl-node k v left (avl-delete-by cmp key right)))]\n          [else right]))))",
        "note": "Deleting a two-child node cannot discard the left subtree.",
    },
    {
        "fn": "avl-delete-by",
        "buggy": "(define (avl-delete-by cmp key tree)\n  (if (avl-empty? tree)\n      tree\n      (let ([k (avl-key tree)]\n            [v (avl-value tree)]\n            [left (avl-left tree)]\n            [right (avl-right tree)])\n        (cond\n          [(cmp key k)\n           (rebalance (make-avl-node k v (avl-delete-by cmp key left) right))]\n          [(cmp k key)\n           (rebalance (make-avl-node k v left (avl-delete-by cmp key right)))]\n          [else\n           (cond\n             [(avl-empty? left) right]\n             [(avl-empty? right) left]\n             [else\n              (let ([succ (avl-min-node right)])\n                (make-avl-node (car succ) (cdr succ)\n                               left\n                               (avl-delete-min-by cmp right)))])]))))",
        "note": "Missing rebalance after successor replacement in two-child deletion case.",
    },
    {
        "fn": "avl-range-by",
        "buggy": "(define (avl-range-by cmp lo hi tree)\n  (if (avl-empty? tree)\n      '()\n      (let ([k (avl-key tree)]\n            [v (avl-value tree)])\n        (append\n         (if (cmp lo k)\n             (avl-range-by cmp lo hi (avl-left tree))\n             '())\n         (if (and (cmp lo k) (cmp k hi))\n             (list (cons k v))\n             '())\n         (if (cmp k hi)\n             (avl-range-by cmp lo hi (avl-right tree))\n             '())))))",
        "note": "Range is inclusive; endpoint checks must not exclude lo/hi matches.",
    },
    {
        "fn": "avl-range-by",
        "buggy": "(define (avl-range-by cmp lo hi tree)\n  (if (avl-empty? tree)\n      '()\n      (let ([k (avl-key tree)]\n            [v (avl-value tree)])\n        (append\n         (if (cmp lo k)\n             (avl-range-by cmp lo hi (avl-left tree))\n             '())\n         (if (and (not (cmp k lo)) (not (cmp hi k)))\n             (list (cons k v))\n             '())\n         '()))))",
        "note": "The right-subtree traversal is required when k is below the high bound.",
    },
    {
        "fn": "avl-range",
        "buggy": "(define (avl-range lo hi tree)\n  (avl-range-by > lo hi tree))",
        "note": "Default range query must delegate with `<` comparator.",
    },
    {
        "fn": "avl-keys-between",
        "buggy": "(define (avl-keys-between lo hi tree)\n  (map cdr (avl-range lo hi tree)))",
        "note": "Keys-between must return keys, so it should map `car`, not `cdr`.",
    },
    {
        "fn": "avl-less-than",
        "buggy": "(define (avl-less-than bound tree)\n  (if (avl-empty? tree)\n      '()\n      (let ([k (avl-key tree)])\n        (if (<= k bound)\n            (append (avl-less-than bound (avl-left tree))\n                    (list (cons k (avl-value tree)))\n                    (avl-less-than bound (avl-right tree)))\n            (avl-less-than bound (avl-left tree))))))",
        "note": "This operation is strict (`key < bound`), so keys equal to bound must be excluded.",
    },
    {
        "fn": "avl-greater-than",
        "buggy": "(define (avl-greater-than bound tree)\n  (if (avl-empty? tree)\n      '()\n      (let ([k (avl-key tree)])\n        (if (>= k bound)\n            (append (avl-greater-than bound (avl-left tree))\n                    (list (cons k (avl-value tree)))\n                    (avl-greater-than bound (avl-right tree)))\n            (avl-greater-than bound (avl-right tree))))))",
        "note": "This operation is strict (`key > bound`), so keys equal to bound must be excluded.",
    },
]

DIFFICULTY = {
    "avl-empty?": "easy",
    "make-avl-node": "medium",
    "rebalance": "hard",
    "avl-lookup": "easy",
    "avl-lookup-by": "medium",
    "avl-contains?": "easy",
    "avl-insert": "medium",
    "avl-insert-by": "hard",
    "avl-delete-min-by": "medium",
    "avl-delete": "medium",
    "avl-delete-by": "hard",
    "avl-range": "medium",
    "avl-range-by": "hard",
    "avl-keys-between": "easy",
    "avl-less-than": "medium",
    "avl-greater-than": "medium",
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
    sid = f"avl_tree_{family}_{family_counter[family]:03d}"
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


def refs_in_text(text: str, exclude: str | None = None) -> List[str]:
    tokens = set(TOKEN_RE.findall(text))
    return [name for name in ALL_NAMES if name != exclude and name in tokens]


def verify_refs(fn: str) -> List[str]:
    return refs_in_text(VERIFY_BY_FUNCTION[fn], exclude=fn)


DEPENDS: Dict[str, List[str]] = {name: refs_in_text(defn, exclude=name) for name, defn in ALL_DEFS.items()}


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

    # Verify expressions can reference helpers not present in direct impl deps.
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
        prompt=f"""You are implementing Tier-0 AVL tree code in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "avl-tree", "spec-to-code", fn],
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
        tags=["tier0", "data", "avl-tree", "skeleton-completion", fn],
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
        tags=["tier0", "data", "avl-tree", "python-to-scheme", fn],
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
        tags=["tier0", "data", "avl-tree", "chez-to-fold", fn],
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
        tags=["tier0", "data", "avl-tree", "bugfix", fn],
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
        tags=["tier0", "data", "avl-tree", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # avl-empty?
    (
        "avl-empty?",
        "Return whether avl-empty is empty.",
        "(avl-empty? avl-empty)",
        "(equal? (avl-empty? avl-empty) #t)",
        "easy",
        ["direct"],
    ),
    (
        "avl-empty?",
        "Insert one key into avl-empty and test avl-empty?.",
        "(avl-empty? (avl-insert-by < 5 \"x\" avl-empty))",
        "(equal? (avl-empty? (avl-insert-by < 5 \"x\" avl-empty)) #f)",
        "easy",
        ["direct"],
    ),
    (
        "avl-empty?",
        "Delete the only key from a singleton AVL tree and report whether it is empty.",
        "(avl-empty? (avl-delete-by < 5 (avl-insert-by < 5 \"x\" avl-empty)))",
        "(equal? (avl-empty? (avl-delete-by < 5 (avl-insert-by < 5 \"x\" avl-empty))) #t)",
        "medium",
        ["edge-case"],
    ),
    (
        "avl-empty?",
        "Build a tree from two pairs and report whether it is empty.",
        "(avl-empty? (list->avl '((2 . \"b\") (1 . \"a\"))))",
        "(equal? (avl-empty? (list->avl '((2 . \"b\") (1 . \"a\")))) #f)",
        "easy",
        ["integration"],
    ),

    # make-avl-node
    (
        "make-avl-node",
        "Construct a singleton node with make-avl-node and return its height.",
        "(avl-height (make-avl-node 5 \"v\" avl-empty avl-empty))",
        "(equal? (avl-height (make-avl-node 5 \"v\" avl-empty avl-empty)) 1)",
        "easy",
        ["direct"],
    ),
    (
        "make-avl-node",
        "Construct a node with child heights 2 and 1, then return the cached height.",
        "(let* ([l (avl-node 2 2 \"b\" (avl-node 1 1 \"a\" avl-empty avl-empty) avl-empty)] [r (avl-node 1 4 \"d\" avl-empty avl-empty)] [n (make-avl-node 3 \"c\" l r)]) (avl-height n))",
        "(equal? (let* ([l (avl-node 2 2 \"b\" (avl-node 1 1 \"a\" avl-empty avl-empty) avl-empty)] [r (avl-node 1 4 \"d\" avl-empty avl-empty)] [n (make-avl-node 3 \"c\" l r)]) (avl-height n)) 3)",
        "medium",
        ["direct"],
    ),
    (
        "make-avl-node",
        "Return #t iff make-avl-node preserves the provided left and right subtrees.",
        "(let* ([l (avl-node 1 1 \"a\" avl-empty avl-empty)] [r (avl-node 1 3 \"c\" avl-empty avl-empty)] [n (make-avl-node 2 \"b\" l r)]) (and (equal? (avl-left n) l) (equal? (avl-right n) r)))",
        "(equal? (let* ([l (avl-node 1 1 \"a\" avl-empty avl-empty)] [r (avl-node 1 3 \"c\" avl-empty avl-empty)] [n (make-avl-node 2 \"b\" l r)]) (and (equal? (avl-left n) l) (equal? (avl-right n) r))) #t)",
        "medium",
        ["property"],
    ),
    (
        "make-avl-node",
        "Build an uneven node with make-avl-node and return its balance factor.",
        "(let* ([l (avl-node 1 1 \"a\" avl-empty avl-empty)] [r (avl-node 2 4 \"d\" (avl-node 1 3 \"c\" avl-empty avl-empty) avl-empty)] [n (make-avl-node 2 \"b\" l r)]) (avl-balance-factor n))",
        "(equal? (let* ([l (avl-node 1 1 \"a\" avl-empty avl-empty)] [r (avl-node 2 4 \"d\" (avl-node 1 3 \"c\" avl-empty avl-empty) avl-empty)] [n (make-avl-node 2 \"b\" l r)]) (avl-balance-factor n)) -1)",
        "medium",
        ["property"],
    ),

    # rebalance
    (
        "rebalance",
        "Rebalance a right-heavy chain built from keys 1,2,3 and return avl-keys.",
        "(let* ([n3 (make-avl-node 3 \"c\" avl-empty avl-empty)] [n2 (make-avl-node 2 \"b\" avl-empty n3)] [n1 (make-avl-node 1 \"a\" avl-empty n2)]) (avl-keys (rebalance n1)))",
        "(equal? (let* ([n3 (make-avl-node 3 \"c\" avl-empty avl-empty)] [n2 (make-avl-node 2 \"b\" avl-empty n3)] [n1 (make-avl-node 1 \"a\" avl-empty n2)]) (avl-keys (rebalance n1))) '(1 2 3))",
        "hard",
        ["rotation"],
    ),
    (
        "rebalance",
        "Rebalance a left-heavy chain built from keys 3,2,1 and return avl-keys.",
        "(let* ([n1 (make-avl-node 1 \"a\" avl-empty avl-empty)] [n2 (make-avl-node 2 \"b\" n1 avl-empty)] [n3 (make-avl-node 3 \"c\" n2 avl-empty)]) (avl-keys (rebalance n3)))",
        "(equal? (let* ([n1 (make-avl-node 1 \"a\" avl-empty avl-empty)] [n2 (make-avl-node 2 \"b\" n1 avl-empty)] [n3 (make-avl-node 3 \"c\" n2 avl-empty)]) (avl-keys (rebalance n3))) '(1 2 3))",
        "hard",
        ["rotation"],
    ),
    (
        "rebalance",
        "Apply rebalance to a left-right imbalance and return whether AVL/BST invariants hold.",
        "(let* ([n2 (make-avl-node 2 \"b\" avl-empty avl-empty)] [n1 (make-avl-node 1 \"a\" avl-empty n2)] [n3 (make-avl-node 3 \"c\" n1 avl-empty)] [r (rebalance n3)]) (and (avl-valid? r) (avl-bst-valid? r) (equal? (avl-keys r) '(1 2 3))))",
        "(equal? (let* ([n2 (make-avl-node 2 \"b\" avl-empty avl-empty)] [n1 (make-avl-node 1 \"a\" avl-empty n2)] [n3 (make-avl-node 3 \"c\" n1 avl-empty)] [r (rebalance n3)]) (and (avl-valid? r) (avl-bst-valid? r) (equal? (avl-keys r) '(1 2 3)))) #t)",
        "hard",
        ["rotation", "invariant"],
    ),
    (
        "rebalance",
        "Call rebalance on an already balanced tree and confirm invariants still hold.",
        "(let* ([t (list->avl '((2 . \"b\") (1 . \"a\") (3 . \"c\")))] [r (rebalance t)]) (and (avl-valid? r) (avl-bst-valid? r) (equal? (avl-keys r) '(1 2 3))))",
        "(equal? (let* ([t (list->avl '((2 . \"b\") (1 . \"a\") (3 . \"c\")))] [r (rebalance t)]) (and (avl-valid? r) (avl-bst-valid? r) (equal? (avl-keys r) '(1 2 3)))) #t)",
        "medium",
        ["invariant"],
    ),

    # avl-lookup-by
    (
        "avl-lookup-by",
        "Look up key 7 in a small AVL tree built from association pairs.",
        "(avl-lookup-by < 7 (list->avl '((5 . \"e\") (3 . \"c\") (7 . \"g\"))))",
        "(equal? (avl-lookup-by < 7 (list->avl '((5 . \"e\") (3 . \"c\") (7 . \"g\")))) \"g\")",
        "easy",
        ["direct"],
    ),
    (
        "avl-lookup-by",
        "Look up a missing key and return the lookup result.",
        "(avl-lookup-by < 8 (list->avl '((5 . \"e\") (3 . \"c\") (7 . \"g\"))))",
        "(equal? (avl-lookup-by < 8 (list->avl '((5 . \"e\") (3 . \"c\") (7 . \"g\")))) #f)",
        "easy",
        ["edge-case"],
    ),
    (
        "avl-lookup-by",
        "Build a tree with descending comparator > and look up key 2 with the same comparator.",
        "(let ([t (fold-left (lambda (acc kv) (avl-insert-by > (car kv) (cdr kv) acc)) avl-empty '((5 . \"five\") (2 . \"two\") (9 . \"nine\")))]) (avl-lookup-by > 2 t))",
        "(equal? (let ([t (fold-left (lambda (acc kv) (avl-insert-by > (car kv) (cdr kv) acc)) avl-empty '((5 . \"five\") (2 . \"two\") (9 . \"nine\")))]) (avl-lookup-by > 2 t)) \"two\")",
        "medium",
        ["comparator"],
    ),
    (
        "avl-lookup-by",
        "Delete key 2 from a tree and then look it up.",
        "(let* ([t (list->avl '((1 . \"a\") (2 . \"b\") (3 . \"c\")))] [d (avl-delete-by < 2 t)]) (avl-lookup-by < 2 d))",
        "(equal? (let* ([t (list->avl '((1 . \"a\") (2 . \"b\") (3 . \"c\")))] [d (avl-delete-by < 2 t)]) (avl-lookup-by < 2 d)) #f)",
        "medium",
        ["integration"],
    ),

    # avl-insert-by
    (
        "avl-insert-by",
        "Insert keys 5,3,7,1,9,4 with avl-insert-by and return sorted keys.",
        "(avl-keys (fold-left (lambda (acc k) (avl-insert-by < k k acc)) avl-empty '(5 3 7 1 9 4)))",
        "(equal? (avl-keys (fold-left (lambda (acc k) (avl-insert-by < k k acc)) avl-empty '(5 3 7 1 9 4))) '(1 3 4 5 7 9))",
        "medium",
        ["direct"],
    ),
    (
        "avl-insert-by",
        "Update an existing key with avl-insert-by and return (size lookup-value).",
        "(let* ([t (list->avl '((2 . \"two\") (1 . \"one\") (3 . \"three\")))] [u (avl-insert-by < 2 \"TWO\" t)]) (list (avl-size u) (avl-lookup-by < 2 u)))",
        "(equal? (let* ([t (list->avl '((2 . \"two\") (1 . \"one\") (3 . \"three\")))] [u (avl-insert-by < 2 \"TWO\" t)]) (list (avl-size u) (avl-lookup-by < 2 u))) '(3 \"TWO\"))",
        "medium",
        ["update"],
    ),
    (
        "avl-insert-by",
        "Insert keys under comparator > and return key order from avl-keys.",
        "(avl-keys (fold-left (lambda (acc k) (avl-insert-by > k k acc)) avl-empty '(1 4 2 5 3)))",
        "(equal? (avl-keys (fold-left (lambda (acc k) (avl-insert-by > k k acc)) avl-empty '(1 4 2 5 3))) '(5 4 3 2 1))",
        "hard",
        ["comparator"],
    ),
    (
        "avl-insert-by",
        "Insert keys 0..19 and return whether AVL and BST invariants plus a height bound hold.",
        "(let ([t (fold-left (lambda (acc k) (avl-insert-by < k k acc)) avl-empty (iota 20))]) (and (avl-valid? t) (avl-bst-valid? t) (<= (avl-height t) 7)))",
        "(equal? (let ([t (fold-left (lambda (acc k) (avl-insert-by < k k acc)) avl-empty (iota 20))]) (and (avl-valid? t) (avl-bst-valid? t) (<= (avl-height t) 7))) #t)",
        "hard",
        ["invariant", "stress"],
    ),

    # avl-delete-min-by
    (
        "avl-delete-min-by",
        "Delete the minimum key from a tree and return remaining keys.",
        "(avl-keys (avl-delete-min-by < (list->avl '((4 . \"d\") (2 . \"b\") (6 . \"f\") (1 . \"a\")))))",
        "(equal? (avl-keys (avl-delete-min-by < (list->avl '((4 . \"d\") (2 . \"b\") (6 . \"f\") (1 . \"a\"))))) '(2 4 6))",
        "medium",
        ["direct"],
    ),
    (
        "avl-delete-min-by",
        "Return #t iff deleting min from avl-empty raises an error.",
        "(guard (ex [else #t]) (begin (avl-delete-min-by < avl-empty) #f))",
        "(equal? (guard (ex [else #t]) (begin (avl-delete-min-by < avl-empty) #f)) #t)",
        "medium",
        ["edge-case"],
    ),
    (
        "avl-delete-min-by",
        "Delete the minimum key twice and return resulting keys.",
        "(avl-keys (avl-delete-min-by < (avl-delete-min-by < (list->avl '((5 . \"e\") (3 . \"c\") (7 . \"g\") (1 . \"a\") (4 . \"d\"))))))",
        "(equal? (avl-keys (avl-delete-min-by < (avl-delete-min-by < (list->avl '((5 . \"e\") (3 . \"c\") (7 . \"g\") (1 . \"a\") (4 . \"d\")))))) '(4 5 7))",
        "hard",
        ["integration"],
    ),
    (
        "avl-delete-min-by",
        "Delete min once and return whether size drops by one while invariants hold.",
        "(let* ([t0 (list->avl '((8 . \"h\") (4 . \"d\") (10 . \"j\") (2 . \"b\") (6 . \"f\")))] [t1 (avl-delete-min-by < t0)]) (and (= (avl-size t1) (- (avl-size t0) 1)) (avl-valid? t1) (avl-bst-valid? t1)))",
        "(equal? (let* ([t0 (list->avl '((8 . \"h\") (4 . \"d\") (10 . \"j\") (2 . \"b\") (6 . \"f\")))] [t1 (avl-delete-min-by < t0)]) (and (= (avl-size t1) (- (avl-size t0) 1)) (avl-valid? t1) (avl-bst-valid? t1))) #t)",
        "hard",
        ["property", "invariant"],
    ),

    # avl-delete-by
    (
        "avl-delete-by",
        "Delete leaf key 1 and return remaining keys.",
        "(avl-keys (avl-delete-by < 1 (list->avl '((3 . \"c\") (1 . \"a\") (4 . \"d\") (2 . \"b\")))))",
        "(equal? (avl-keys (avl-delete-by < 1 (list->avl '((3 . \"c\") (1 . \"a\") (4 . \"d\") (2 . \"b\"))))) '(2 3 4))",
        "medium",
        ["direct"],
    ),
    (
        "avl-delete-by",
        "Delete root key 5 from a full tree and return remaining keys.",
        "(let ([t (list->avl '((5 . \"e\") (3 . \"c\") (7 . \"g\") (2 . \"b\") (4 . \"d\") (6 . \"f\") (8 . \"h\")))]) (avl-keys (avl-delete-by < 5 t)))",
        "(equal? (let ([t (list->avl '((5 . \"e\") (3 . \"c\") (7 . \"g\") (2 . \"b\") (4 . \"d\") (6 . \"f\") (8 . \"h\")))]) (avl-keys (avl-delete-by < 5 t))) '(2 3 4 6 7 8))",
        "hard",
        ["direct", "two-children"],
    ),
    (
        "avl-delete-by",
        "Delete a missing key and return (size valid? bst-valid?).",
        "(let* ([t (list->avl '((2 . \"b\") (1 . \"a\") (3 . \"c\")))] [d (avl-delete-by < 99 t)]) (list (avl-size d) (avl-valid? d) (avl-bst-valid? d)))",
        "(equal? (let* ([t (list->avl '((2 . \"b\") (1 . \"a\") (3 . \"c\")))] [d (avl-delete-by < 99 t)]) (list (avl-size d) (avl-valid? d) (avl-bst-valid? d))) '(3 #t #t))",
        "medium",
        ["edge-case", "invariant"],
    ),
    (
        "avl-delete-by",
        "Delete keys 2 then 4 and return (keys size).",
        "(let* ([t0 (list->avl '((1 . \"a\") (2 . \"b\") (3 . \"c\") (4 . \"d\") (5 . \"e\")))] [t1 (avl-delete-by < 2 t0)] [t2 (avl-delete-by < 4 t1)]) (list (avl-keys t2) (avl-size t2)))",
        "(equal? (let* ([t0 (list->avl '((1 . \"a\") (2 . \"b\") (3 . \"c\") (4 . \"d\") (5 . \"e\")))] [t1 (avl-delete-by < 2 t0)] [t2 (avl-delete-by < 4 t1)]) (list (avl-keys t2) (avl-size t2))) '((1 3 5) 3))",
        "hard",
        ["integration"],
    ),

    # avl-range-by
    (
        "avl-range-by",
        "Return pairs for inclusive range [3,7].",
        "(avl-range-by < 3 7 (list->avl '((1 . \"a\") (3 . \"c\") (5 . \"e\") (7 . \"g\") (9 . \"i\"))))",
        "(equal? (avl-range-by < 3 7 (list->avl '((1 . \"a\") (3 . \"c\") (5 . \"e\") (7 . \"g\") (9 . \"i\")))) '((3 . \"c\") (5 . \"e\") (7 . \"g\")))",
        "medium",
        ["direct"],
    ),
    (
        "avl-range-by",
        "Return pairs for range [20,30] over a small tree.",
        "(avl-range-by < 20 30 (list->avl '((1 . \"a\") (3 . \"c\") (5 . \"e\"))))",
        "(equal? (avl-range-by < 20 30 (list->avl '((1 . \"a\") (3 . \"c\") (5 . \"e\")))) '())",
        "easy",
        ["edge-case"],
    ),
    (
        "avl-range-by",
        "Return the exact-boundary range [7,7].",
        "(avl-range-by < 7 7 (list->avl '((5 . \"e\") (7 . \"g\") (9 . \"i\"))))",
        "(equal? (avl-range-by < 7 7 (list->avl '((5 . \"e\") (7 . \"g\") (9 . \"i\")))) '((7 . \"g\")))",
        "easy",
        ["edge-case"],
    ),
    (
        "avl-range-by",
        "Return keys extracted from range [4,10] over a five-element tree.",
        "(map car (avl-range-by < 4 10 (list->avl '((1 . \"a\") (4 . \"d\") (6 . \"f\") (10 . \"j\") (12 . \"l\")))))",
        "(equal? (map car (avl-range-by < 4 10 (list->avl '((1 . \"a\") (4 . \"d\") (6 . \"f\") (10 . \"j\") (12 . \"l\"))))) '(4 6 10))",
        "medium",
        ["integration"],
    ),

    # expanded APIs
    (
        "avl-lookup",
        "Lookup key 6 using default comparator wrapper.",
        "(avl-lookup 6 (list->avl '((4 . \"d\") (6 . \"f\") (8 . \"h\"))))",
        "(equal? (avl-lookup 6 (list->avl '((4 . \"d\") (6 . \"f\") (8 . \"h\")))) \"f\")",
        "easy",
        ["direct"],
    ),
    (
        "avl-contains?",
        "Check present and missing keys with avl-contains?.",
        "(let ([t (list->avl '((2 . \"b\") (1 . \"a\") (3 . \"c\")))]) (and (avl-contains? 2 t) (not (avl-contains? 9 t))))",
        "(equal? (let ([t (list->avl '((2 . \"b\") (1 . \"a\") (3 . \"c\")))]) (and (avl-contains? 2 t) (not (avl-contains? 9 t)))) #t)",
        "easy",
        ["property"],
    ),
    (
        "avl-insert",
        "Insert via default wrapper and return sorted keys.",
        "(avl-keys (avl-insert 4 \"d\" (list->avl '((2 . \"b\") (1 . \"a\") (3 . \"c\")))))",
        "(equal? (avl-keys (avl-insert 4 \"d\" (list->avl '((2 . \"b\") (1 . \"a\") (3 . \"c\"))))) '(1 2 3 4))",
        "medium",
        ["direct"],
    ),
    (
        "avl-delete",
        "Delete key 3 using default wrapper and return keys.",
        "(avl-keys (avl-delete 3 (list->avl '((1 . \"a\") (2 . \"b\") (3 . \"c\") (4 . \"d\")))))",
        "(equal? (avl-keys (avl-delete 3 (list->avl '((1 . \"a\") (2 . \"b\") (3 . \"c\") (4 . \"d\"))))) '(1 2 4))",
        "medium",
        ["direct"],
    ),
    (
        "avl-range",
        "Run default range wrapper for [2,4].",
        "(avl-range 2 4 (list->avl '((1 . \"a\") (2 . \"b\") (3 . \"c\") (4 . \"d\") (5 . \"e\"))))",
        "(equal? (avl-range 2 4 (list->avl '((1 . \"a\") (2 . \"b\") (3 . \"c\") (4 . \"d\") (5 . \"e\")))) '((2 . \"b\") (3 . \"c\") (4 . \"d\")))",
        "medium",
        ["direct"],
    ),
    (
        "avl-keys-between",
        "Return keys between 2 and 6 inclusive.",
        "(avl-keys-between 2 6 (list->avl '((1 . \"a\") (2 . \"b\") (4 . \"d\") (6 . \"f\") (8 . \"h\"))))",
        "(equal? (avl-keys-between 2 6 (list->avl '((1 . \"a\") (2 . \"b\") (4 . \"d\") (6 . \"f\") (8 . \"h\")))) '(2 4 6))",
        "easy",
        ["direct"],
    ),
    (
        "avl-less-than",
        "Return pairs with keys strictly less than 5.",
        "(avl-less-than 5 (list->avl '((2 . \"b\") (5 . \"e\") (1 . \"a\") (7 . \"g\") (4 . \"d\"))))",
        "(equal? (avl-less-than 5 (list->avl '((2 . \"b\") (5 . \"e\") (1 . \"a\") (7 . \"g\") (4 . \"d\")))) '((1 . \"a\") (2 . \"b\") (4 . \"d\")))",
        "medium",
        ["direct"],
    ),
    (
        "avl-greater-than",
        "Return pairs with keys strictly greater than 5.",
        "(avl-greater-than 5 (list->avl '((2 . \"b\") (5 . \"e\") (1 . \"a\") (7 . \"g\") (6 . \"f\"))))",
        "(equal? (avl-greater-than 5 (list->avl '((2 . \"b\") (5 . \"e\") (1 . \"a\") (7 . \"g\") (6 . \"f\")))) '((6 . \"f\") (7 . \"g\")))",
        "medium",
        ["direct"],
    ),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

if sum(1 for s in samples if s["family"] == "composition") < 32:
    raise ValueError("composition family must contain at least 32 samples")

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
