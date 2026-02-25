#!/usr/bin/env python3
"""Generate Tier-1 SFT samples for lattice/egraph/union-find.ss."""

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

SOURCE_MODULE = "lattice/egraph/union-find.ss"
SOURCE_TEST = "lattice/egraph/test-union-find.ss"

FUNCTION_ORDER = [
    "make-uf",
    "uf?",
    "uf-count",
    "uf-size",
    "uf-make-set!",
    "uf-find",
    "uf-union!",
    "uf-same-set?",
]

HELPER_ORDER = [
    "uf-tag",
    "uf-parent",
    "uf-rank",
    "uf-count-box",
    "uf-next-id-box",
    "ensure-capacity!",
]

DEF_ORDER = HELPER_ORDER + FUNCTION_ORDER

DEFS: Dict[str, str] = {
    "uf-tag": """(define uf-tag 'union-find)""",
    "make-uf": """(define (make-uf)
  (vector uf-tag
          (make-vector 64 #f)
          (make-vector 64 0)
          (box 0)
          (box 0)))""",
    "uf?": """(define (uf? x)
  (and (vector? x)
       (>= (vector-length x) 5)
       (eq? (vector-ref x 0) uf-tag)))""",
    "uf-parent": """(define (uf-parent uf)
  (vector-ref uf 1))""",
    "uf-rank": """(define (uf-rank uf)
  (vector-ref uf 2))""",
    "uf-count-box": """(define (uf-count-box uf)
  (vector-ref uf 3))""",
    "uf-next-id-box": """(define (uf-next-id-box uf)
  (vector-ref uf 4))""",
    "uf-count": """(define (uf-count uf)
  (unbox (uf-count-box uf)))""",
    "uf-size": """(define (uf-size uf)
  (unbox (uf-next-id-box uf)))""",
    "ensure-capacity!": """(define (ensure-capacity! uf id)
  (let ([parent (uf-parent uf)]
        [rank (uf-rank uf)])
    (when (>= id (vector-length parent))
      (let* ([old-len (vector-length parent)]
             [new-len (max (* 2 old-len) (+ id 1))]
             [new-parent (make-vector new-len #f)]
             [new-rank (make-vector new-len 0)])
        (do ([i 0 (+ i 1)])
            ((>= i old-len))
          (vector-set! new-parent i (vector-ref parent i))
          (vector-set! new-rank i (vector-ref rank i)))
        (vector-set! uf 1 new-parent)
        (vector-set! uf 2 new-rank)))))""",
    "uf-make-set!": """(define (uf-make-set! uf)
  (let* ([id-box (uf-next-id-box uf)]
         [id (unbox id-box)])
    (ensure-capacity! uf id)
    (vector-set! (uf-parent uf) id id)
    (vector-set! (uf-rank uf) id 0)
    (set-box! id-box (+ id 1))
    (set-box! (uf-count-box uf) (+ (unbox (uf-count-box uf)) 1))
    id))""",
    "uf-find": """(define (uf-find uf id)
  (let ([parent (uf-parent uf)])
    (let loop ([i id])
      (let ([p (vector-ref parent i)])
        (if (= p i)
            i
            (let ([root (loop p)])
              (vector-set! parent i root)
              root))))))""",
    "uf-union!": """(define (uf-union! uf id1 id2)
  (let ([root1 (uf-find uf id1)]
        [root2 (uf-find uf id2)])
    (if (= root1 root2)
        root1
        (let ([parent (uf-parent uf)]
              [rank (uf-rank uf)]
              [rank1 (vector-ref (uf-rank uf) root1)]
              [rank2 (vector-ref (uf-rank uf) root2)])
          (set-box! (uf-count-box uf) (- (unbox (uf-count-box uf)) 1))
          (cond
            [(< rank1 rank2)
             (vector-set! parent root1 root2)
             root2]
            [(> rank1 rank2)
             (vector-set! parent root2 root1)
             root1]
            [else
             (vector-set! parent root2 root1)
             (vector-set! rank root1 (+ rank1 1))
             root1])))))""",
    "uf-same-set?": """(define (uf-same-set? uf id1 id2)
  (= (uf-find uf id1) (uf-find uf id2)))""",
}

DEPENDS: Dict[str, List[str]] = {
    "uf-tag": [],
    "make-uf": ["uf-tag"],
    "uf?": ["uf-tag"],
    "uf-parent": [],
    "uf-rank": [],
    "uf-count-box": [],
    "uf-next-id-box": [],
    "uf-count": ["uf-count-box"],
    "uf-size": ["uf-next-id-box"],
    "ensure-capacity!": ["uf-parent", "uf-rank"],
    "uf-make-set!": ["uf-next-id-box", "ensure-capacity!", "uf-parent", "uf-rank", "uf-count-box"],
    "uf-find": ["uf-parent"],
    "uf-union!": ["uf-find", "uf-parent", "uf-rank", "uf-count-box"],
    "uf-same-set?": ["uf-find"],
}

FUNCTION_SPECS = {
    "make-uf": "Create an empty union-find structure with growable parent/rank vectors and zeroed size/count.",
    "uf?": "Recognize a union-find value by vector shape and tag.",
    "uf-count": "Return the number of distinct disjoint sets currently tracked.",
    "uf-size": "Return total allocated element count (next ID).",
    "uf-make-set!": "Allocate a new singleton set, initialize parent/rank, and update size/count.",
    "uf-find": "Return the set representative for an element with path compression.",
    "uf-union!": "Merge two sets by rank, decrement set count, and return merged representative.",
    "uf-same-set?": "Check whether two IDs belong to the same set representative.",
}

SKELETONS = {
    "make-uf": """(define (make-uf)
  ;; TODO: create an empty union-find with initial capacity and counters
  <TODO>)""",
    "uf?": """(define (uf? x)
  ;; TODO: verify union-find representation tag and shape
  <TODO>)""",
    "uf-count": """(define (uf-count uf)
  ;; TODO: read number of disjoint sets
  <TODO>)""",
    "uf-size": """(define (uf-size uf)
  ;; TODO: read number of allocated IDs
  <TODO>)""",
    "uf-make-set!": """(define (uf-make-set! uf)
  ;; TODO: allocate singleton element and update metadata
  <TODO>)""",
    "uf-find": """(define (uf-find uf id)
  ;; TODO: find root with path compression
  <TODO>)""",
    "uf-union!": """(define (uf-union! uf id1 id2)
  ;; TODO: merge sets by rank and return merged representative
  <TODO>)""",
    "uf-same-set?": """(define (uf-same-set? uf id1 id2)
  ;; TODO: test set-equivalence via representatives
  <TODO>)""",
}

DIFFICULTY = {
    "make-uf": "medium",
    "uf?": "easy",
    "uf-count": "easy",
    "uf-size": "easy",
    "uf-make-set!": "medium",
    "uf-find": "hard",
    "uf-union!": "hard",
    "uf-same-set?": "easy",
}

VERIFY_BY_FUNCTION = {
    "make-uf": """(and
  (let ([uf (make-uf)])
    (and (uf? uf)
         (= (uf-size uf) 0)
         (= (uf-count uf) 0)))
  (let ([uf (make-uf)])
    (and (= (vector-length (uf-parent uf)) 64)
         (= (vector-length (uf-rank uf)) 64)
         (eq? (vector-ref uf 0) 'union-find))))""",
    "uf?": """(and
  (uf? (make-uf))
  (not (uf? '(union-find)))
  (not (uf? (vector 'union-find
                    (make-vector 1 #f)
                    (make-vector 1 0)
                    (box 0))))
  (not (uf? (vector 'wrong-tag
                    (make-vector 1 #f)
                    (make-vector 1 0)
                    (box 0)
                    (box 0)))))""",
    "uf-count": """(and
  (let ([uf (make-uf)])
    (= (uf-count uf) 0))
  (let ([uf (make-uf)])
    (let ([a (uf-make-set! uf)]
          [b (uf-make-set! uf)]
          [c (uf-make-set! uf)])
      (uf-union! uf a b)
      (uf-union! uf b c)
      (and (= (uf-count uf) 1)
           (= (uf-size uf) 3)
           (= (uf-union! uf a c) (uf-find uf a))
           (= (uf-count uf) 1)))))""",
    "uf-size": """(and
  (let ([uf (make-uf)])
    (= (uf-size uf) 0))
  (let ([uf (make-uf)])
    (do ([i 0 (+ i 1)])
        ((>= i 70))
      (uf-make-set! uf))
    (uf-union! uf 0 69)
    (and (= (uf-size uf) 70)
         (= (uf-count uf) 69))))""",
    "uf-make-set!": """(and
  (let ([uf (make-uf)])
    (let* ([a (uf-make-set! uf)]
           [b (uf-make-set! uf)]
           [c (uf-make-set! uf)])
      (and (= a 0)
           (= b 1)
           (= c 2)
           (= (vector-ref (uf-parent uf) a) a)
           (= (vector-ref (uf-parent uf) c) c)
           (= (vector-ref (uf-rank uf) b) 0)
           (= (uf-size uf) 3)
           (= (uf-count uf) 3))))
  (let ([uf (make-uf)])
    (do ([i 0 (+ i 1)])
        ((>= i 100))
      (uf-make-set! uf))
    (and (= (uf-size uf) 100)
         (= (vector-ref (uf-parent uf) 99) 99))))""",
    "uf-find": """(and
  (let ([uf (make-uf)])
    (let ([a (uf-make-set! uf)]
          [b (uf-make-set! uf)]
          [c (uf-make-set! uf)]
          [d (uf-make-set! uf)])
      (uf-union! uf a b)
      (uf-union! uf c d)
      (uf-union! uf b c)
      (let* ([root (uf-find uf d)]
             [parent (uf-parent uf)])
        (and (= root (uf-find uf a))
             (= root (uf-find uf b))
             (= root (uf-find uf c))
             (= root (uf-find uf d))
             (= (vector-ref parent d) root)))))
  (let ([uf (make-uf)])
    (let ([x (uf-make-set! uf)])
      (= (uf-find uf x) x))))""",
    "uf-union!": """(and
  (let ([uf (make-uf)])
    (let ([a (uf-make-set! uf)]
          [b (uf-make-set! uf)])
      (let ([root (uf-union! uf a b)])
        (and (= root (uf-find uf a))
             (= root (uf-find uf b))
             (= (uf-count uf) 1)
             (= (vector-ref (uf-rank uf) root) 1)))))
  (let ([uf (make-uf)])
    (let ([a (uf-make-set! uf)]
          [b (uf-make-set! uf)]
          [c (uf-make-set! uf)])
      (uf-union! uf a b)
      (let ([root (uf-union! uf b c)])
        (and (= root (uf-find uf a))
             (= root (uf-find uf c))
             (= (uf-count uf) 1)
             (= (uf-union! uf a c) root)
             (= (uf-count uf) 1))))))""",
    "uf-same-set?": """(and
  (let ([uf (make-uf)])
    (let ([a (uf-make-set! uf)]
          [b (uf-make-set! uf)]
          [c (uf-make-set! uf)])
      (and (uf-same-set? uf a a)
           (not (uf-same-set? uf a b))
           (begin (uf-union! uf a b)
                  (uf-same-set? uf a b))
           (not (uf-same-set? uf a c))
           (begin (uf-union! uf b c)
                  (uf-same-set? uf a c)))))
  (let ([uf (make-uf)])
    (let ([x (uf-make-set! uf)]
          [y (uf-make-set! uf)])
      (uf-union! uf x y)
      (and (uf-same-set? uf x y)
           (uf-same-set? uf y x)))))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "make-uf": """def make_uf():
    return [\"union-find\", [None] * 64, [0] * 64, box(0), box(0)]""",
    "uf?": """def uf_pred(x):
    return isinstance(x, list) and len(x) >= 5 and x[0] == \"union-find\"""",
    "uf-count": """def uf_count(uf):
    return unbox(uf_count_box(uf))""",
    "uf-size": """def uf_size(uf):
    return unbox(uf_next_id_box(uf))""",
    "uf-make-set!": """def uf_make_set(uf):
    id_box = uf_next_id_box(uf)
    element_id = unbox(id_box)
    ensure_capacity(uf, element_id)
    uf_parent(uf)[element_id] = element_id
    uf_rank(uf)[element_id] = 0
    set_box(id_box, element_id + 1)
    set_box(uf_count_box(uf), unbox(uf_count_box(uf)) + 1)
    return element_id""",
    "uf-find": """def uf_find(uf, element_id):
    parent = uf_parent(uf)
    p = parent[element_id]
    if p == element_id:
        return element_id
    root = uf_find(uf, p)
    parent[element_id] = root
    return root""",
    "uf-union!": """def uf_union(uf, id1, id2):
    root1 = uf_find(uf, id1)
    root2 = uf_find(uf, id2)
    if root1 == root2:
        return root1
    parent = uf_parent(uf)
    rank = uf_rank(uf)
    rank1 = rank[root1]
    rank2 = rank[root2]
    set_box(uf_count_box(uf), unbox(uf_count_box(uf)) - 1)
    if rank1 < rank2:
        parent[root1] = root2
        return root2
    if rank1 > rank2:
        parent[root2] = root1
        return root1
    parent[root2] = root1
    rank[root1] = rank1 + 1
    return root1""",
    "uf-same-set?": """def uf_same_set(uf, id1, id2):
    return uf_find(uf, id1) == uf_find(uf, id2)""",
}

CHEZ_SNIPPETS = {
    "make-uf": """(define (mk-uf)
  (vector uf-tag
          (make-vector 64 #f)
          (make-vector 64 0)
          (box 0)
          (box 0)))""",
    "uf?": """(define (is-uf? x)
  (and (vector? x)
       (>= (vector-length x) 5)
       (eq? (vector-ref x 0) uf-tag)))""",
    "uf-count": """(define (count-sets uf)
  (unbox (uf-count-box uf)))""",
    "uf-size": """(define (total-elems uf)
  (unbox (uf-next-id-box uf)))""",
    "uf-make-set!": """(define (add-set! uf)
  (let* ((id-box (uf-next-id-box uf))
         (id (unbox id-box)))
    (ensure-capacity! uf id)
    (vector-set! (uf-parent uf) id id)
    (vector-set! (uf-rank uf) id 0)
    (set-box! id-box (+ id 1))
    (set-box! (uf-count-box uf) (+ (unbox (uf-count-box uf)) 1))
    id))""",
    "uf-find": """(define (find-root uf id)
  (let ((parent (uf-parent uf)))
    (let loop ((i id))
      (let ((p (vector-ref parent i)))
        (if (= p i)
            i
            (let ((root (loop p)))
              (vector-set! parent i root)
              root))))))""",
    "uf-union!": """(define (merge! uf id1 id2)
  (let ((root1 (uf-find uf id1))
        (root2 (uf-find uf id2)))
    (if (= root1 root2)
        root1
        (let ((parent (uf-parent uf))
              (rank (uf-rank uf))
              (rank1 (vector-ref (uf-rank uf) root1))
              (rank2 (vector-ref (uf-rank uf) root2)))
          (set-box! (uf-count-box uf) (- (unbox (uf-count-box uf)) 1))
          (cond
            ((< rank1 rank2)
             (vector-set! parent root1 root2)
             root2)
            ((> rank1 rank2)
             (vector-set! parent root2 root1)
             root1)
            (else
             (vector-set! parent root2 root1)
             (vector-set! rank root1 (+ rank1 1))
             root1))))))""",
    "uf-same-set?": """(define (same-class? uf a b)
  (= (uf-find uf a) (uf-find uf b)))""",
}

BUGGY_CASES = [
    {
        "fn": "make-uf",
        "buggy": """(define (make-uf)
  (vector uf-tag
          (make-vector 64 #f)
          (make-vector 64 #f)
          (box 0)
          (box 0)))""",
        "note": "Rank entries must start at 0, not #f.",
    },
    {
        "fn": "make-uf",
        "buggy": """(define (make-uf)
  (vector uf-tag
          (make-vector 64 #f)
          (make-vector 64 0)
          (box 1)
          (box 0)))""",
        "note": "Fresh union-find must start with zero set count.",
    },
    {
        "fn": "uf?",
        "buggy": """(define (uf? x)
  (and (vector? x)
       (> (vector-length x) 5)
       (eq? (vector-ref x 0) uf-tag)))""",
        "note": "Valid union-find values have length exactly 5 or more; strict > 5 rejects valid instances.",
    },
    {
        "fn": "uf?",
        "buggy": """(define (uf? x)
  (and (vector? x)
       (>= (vector-length x) 5)
       (eq? (vector-ref x 1) uf-tag)))""",
        "note": "Tag is stored in slot 0, not slot 1.",
    },
    {
        "fn": "uf-count",
        "buggy": """(define (uf-count uf)
  (unbox (uf-next-id-box uf)))""",
        "note": "Count must read the count box, not next-id box.",
    },
    {
        "fn": "uf-count",
        "buggy": """(define (uf-count uf)
  (uf-count-box uf))""",
        "note": "The boxed count must be unboxed before returning.",
    },
    {
        "fn": "uf-size",
        "buggy": """(define (uf-size uf)
  (unbox (uf-count-box uf)))""",
        "note": "Size tracks allocated IDs via next-id, not number of sets.",
    },
    {
        "fn": "uf-size",
        "buggy": """(define (uf-size uf)
  (+ (unbox (uf-next-id-box uf)) 1))""",
        "note": "Size should be exact next-id value with no offset.",
    },
    {
        "fn": "uf-make-set!",
        "buggy": """(define (uf-make-set! uf)
  (let* ([id-box (uf-next-id-box uf)]
         [id (unbox id-box)])
    (ensure-capacity! uf id)
    (vector-set! (uf-parent uf) id id)
    (vector-set! (uf-rank uf) id 0)
    (set-box! id-box (+ id 1))
    id))""",
        "note": "Adding a singleton set must also increment set-count.",
    },
    {
        "fn": "uf-make-set!",
        "buggy": """(define (uf-make-set! uf)
  (let* ([id-box (uf-next-id-box uf)]
         [id (unbox id-box)])
    (ensure-capacity! uf id)
    (vector-set! (uf-parent uf) id (+ id 1))
    (vector-set! (uf-rank uf) id 0)
    (set-box! id-box (+ id 1))
    (set-box! (uf-count-box uf) (+ (unbox (uf-count-box uf)) 1))
    id))""",
        "note": "New singleton must point to itself as parent.",
    },
    {
        "fn": "uf-find",
        "buggy": """(define (uf-find uf id)
  (let* ([parent (uf-parent uf)]
         [p (vector-ref parent id)])
    (if (= p id)
        id
        p)))""",
        "note": "Find must recurse to the true root, not return only one parent hop.",
    },
    {
        "fn": "uf-find",
        "buggy": """(define (uf-find uf id)
  (let ([parent (uf-parent uf)])
    (let loop ([i id])
      (let ([p (vector-ref parent i)])
        (if (= p i)
            i
            (let ([root (loop p)])
              (vector-set! parent i root)
              p))))))""",
        "note": "After recursion, find must return the root, not the immediate parent.",
    },
    {
        "fn": "uf-union!",
        "buggy": """(define (uf-union! uf id1 id2)
  (let ([root1 (uf-find uf id1)]
        [root2 (uf-find uf id2)])
    (if (= root1 root2)
        root1
        (let ([parent (uf-parent uf)]
              [rank (uf-rank uf)]
              [rank1 (vector-ref (uf-rank uf) root1)]
              [rank2 (vector-ref (uf-rank uf) root2)])
          (cond
            [(< rank1 rank2)
             (vector-set! parent root1 root2)
             root2]
            [(> rank1 rank2)
             (vector-set! parent root2 root1)
             root1]
            [else
             (vector-set! parent root2 root1)
             (vector-set! rank root1 (+ rank1 1))
             root1])))))""",
        "note": "Merging two distinct sets must decrement the set count.",
    },
    {
        "fn": "uf-union!",
        "buggy": """(define (uf-union! uf id1 id2)
  (let ([root1 (uf-find uf id1)]
        [root2 (uf-find uf id2)])
    (if (= root1 root2)
        root1
        (let ([parent (uf-parent uf)]
              [rank (uf-rank uf)]
              [rank1 (vector-ref (uf-rank uf) root1)]
              [rank2 (vector-ref (uf-rank uf) root2)])
          (set-box! (uf-count-box uf) (- (unbox (uf-count-box uf)) 1))
          (cond
            [(< rank1 rank2)
             (vector-set! parent root1 root2)
             root2]
            [(> rank1 rank2)
             (vector-set! parent root2 root1)
             root1]
            [else
             (vector-set! parent root2 root1)
             (vector-set! rank root2 (+ rank2 1))
             root1])))))""",
        "note": "When ranks are equal and root2 attaches under root1, increment root1 rank (the new root).",
    },
    {
        "fn": "uf-same-set?",
        "buggy": """(define (uf-same-set? uf id1 id2)
  (= id1 id2))""",
        "note": "Set equivalence must compare representatives, not raw IDs.",
    },
    {
        "fn": "uf-same-set?",
        "buggy": """(define (uf-same-set? uf id1 id2)
  (not (= (uf-find uf id1) (uf-find uf id2))))""",
        "note": "The predicate is inverted; it should return true when roots are equal.",
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
    sid = f"egraph_union_find_{family}_{family_counter[family]:03d}"
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
    return [name for name in DEF_ORDER if name != fn and name in tokens]


def dependency_closure(fn: str) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    def add_name(name: str) -> None:
        if name == fn or name in seen:
            return
        if name not in DEFS:
            return
        seen.add(name)
        for dep in DEPENDS.get(name, []):
            add_name(dep)
        ordered.append(name)

    for dep in DEPENDS.get(fn, []):
        add_name(dep)

    for dep in verify_refs(fn):
        add_name(dep)

    return ordered


def def_verify(fn: str) -> str:
    parts = [DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


def wrap_verify_expr(expr: str) -> str:
    parts = [DEFS[name] for name in DEF_ORDER] + [expr]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
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
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "union-find", "spec-to-code", fn],
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
        tags=["tier1", "egraph", "union-find", "skeleton-completion", fn],
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
Preserve behavior exactly and use the target function name `{fn}`.
Return only the Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "union-find", "python-to-scheme", fn],
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
        tags=["tier1", "egraph", "union-find", "chez-to-fold", fn],
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
        tags=["tier1", "egraph", "union-find", "bugfix", fn],
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
        verify_expr=wrap_verify_expr(verify_expr),
        tags=["tier1", "egraph", "union-find", "composition", source_function] + extra_tags,
    )


composition_cases = [
    (
        "make-uf",
        "Create an empty union-find and return `(list size count)`.",
        "(let ([uf (make-uf)]) (list (uf-size uf) (uf-count uf)))",
        "(equal? (let ([uf (make-uf)]) (list (uf-size uf) (uf-count uf))) '(0 0))",
        "easy",
        ["direct"],
    ),
    (
        "make-uf",
        "Create a union-find and return whether it is tagged correctly.",
        "(let ([uf (make-uf)]) (and (uf? uf) (eq? (vector-ref uf 0) 'union-find)))",
        "(equal? (let ([uf (make-uf)]) (and (uf? uf) (eq? (vector-ref uf 0) 'union-find))) #t)",
        "easy",
        ["direct"],
    ),
    (
        "make-uf",
        "Create a union-find, add two singleton sets, and return current count.",
        "(let ([uf (make-uf)]) (uf-make-set! uf) (uf-make-set! uf) (uf-count uf))",
        "(equal? (let ([uf (make-uf)]) (uf-make-set! uf) (uf-make-set! uf) (uf-count uf)) 2)",
        "medium",
        ["integration"],
    ),
    (
        "make-uf",
        "Create a union-find and return `(list parent-capacity rank-capacity)`.",
        "(let ([uf (make-uf)]) (list (vector-length (uf-parent uf)) (vector-length (uf-rank uf))))",
        "(equal? (let ([uf (make-uf)]) (list (vector-length (uf-parent uf)) (vector-length (uf-rank uf)))) '(64 64))",
        "easy",
        ["direct"],
    ),
    (
        "uf?",
        "Return `(list (uf? valid) (uf? invalid-list))`.",
        "(let ([valid (make-uf)] [invalid '(union-find)]) (list (uf? valid) (uf? invalid)))",
        "(equal? (let ([valid (make-uf)] [invalid '(union-find)]) (list (uf? valid) (uf? invalid))) '(#t #f))",
        "easy",
        ["direct"],
    ),
    (
        "uf?",
        "Return whether a wrong-tag vector is rejected by `uf?`.",
        "(uf? (vector 'not-uf (make-vector 1 #f) (make-vector 1 0) (box 0) (box 0)))",
        "(equal? (uf? (vector 'not-uf (make-vector 1 #f) (make-vector 1 0) (box 0) (box 0))) #f)",
        "easy",
        ["edge-case"],
    ),
    (
        "uf?",
        "Create a union-find, perform one union, and test that `uf?` still holds.",
        "(let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)]) (uf-union! uf a b) (uf? uf)))",
        "(equal? (let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)]) (uf-union! uf a b) (uf? uf))) #t)",
        "medium",
        ["integration"],
    ),
    (
        "uf?",
        "Map `uf?` over `(list (make-uf) 42 #t)`.",
        "(map uf? (list (make-uf) 42 #t))",
        "(equal? (map uf? (list (make-uf) 42 #t)) '(#t #f #f))",
        "easy",
        ["list"],
    ),
    (
        "uf-count",
        "Create three sets, union one pair, and return count.",
        "(let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)] [c (uf-make-set! uf)]) (uf-union! uf a b) (uf-count uf)))",
        "(equal? (let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)] [c (uf-make-set! uf)]) (uf-union! uf a b) (uf-count uf))) 2)",
        "medium",
        ["direct"],
    ),
    (
        "uf-count",
        "Union the same pair twice and return final count.",
        "(let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)]) (uf-union! uf a b) (uf-union! uf a b) (uf-count uf)))",
        "(equal? (let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)]) (uf-union! uf a b) (uf-union! uf a b) (uf-count uf))) 1)",
        "medium",
        ["property"],
    ),
    (
        "uf-count",
        "Create five sets and union all into one; return count.",
        "(let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 5)) (uf-make-set! uf)) (do ([i 1 (+ i 1)]) ((>= i 5)) (uf-union! uf 0 i)) (uf-count uf))",
        "(equal? (let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 5)) (uf-make-set! uf)) (do ([i 1 (+ i 1)]) ((>= i 5)) (uf-union! uf 0 i)) (uf-count uf)) 1)",
        "hard",
        ["loop"],
    ),
    (
        "uf-count",
        "Return count of a fresh union-find.",
        "(let ([uf (make-uf)]) (uf-count uf))",
        "(equal? (let ([uf (make-uf)]) (uf-count uf)) 0)",
        "easy",
        ["direct"],
    ),
    (
        "uf-size",
        "Create three sets, merge one pair, and return size.",
        "(let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)] [c (uf-make-set! uf)]) (uf-union! uf a b) (uf-size uf)))",
        "(equal? (let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)] [c (uf-make-set! uf)]) (uf-union! uf a b) (uf-size uf))) 3)",
        "medium",
        ["property"],
    ),
    (
        "uf-size",
        "Create seventy sets and return final size.",
        "(let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 70)) (uf-make-set! uf)) (uf-size uf))",
        "(equal? (let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 70)) (uf-make-set! uf)) (uf-size uf)) 70)",
        "medium",
        ["loop"],
    ),
    (
        "uf-size",
        "Return size of an empty union-find.",
        "(uf-size (make-uf))",
        "(equal? (uf-size (make-uf)) 0)",
        "easy",
        ["direct"],
    ),
    (
        "uf-size",
        "Return whether repeated unions keep size unchanged.",
        "(let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 4)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 1 2) (uf-union! uf 2 3) (= (uf-size uf) 4))",
        "(equal? (let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 4)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 1 2) (uf-union! uf 2 3) (= (uf-size uf) 4)) #t)",
        "medium",
        ["property"],
    ),
    (
        "uf-make-set!",
        "Create three sets and return their allocated IDs as a list.",
        "(let ([uf (make-uf)]) (list (uf-make-set! uf) (uf-make-set! uf) (uf-make-set! uf)))",
        "(equal? (let ([uf (make-uf)]) (list (uf-make-set! uf) (uf-make-set! uf) (uf-make-set! uf))) '(0 1 2))",
        "easy",
        ["direct"],
    ),
    (
        "uf-make-set!",
        "Create 65 sets and return the last allocated ID.",
        "(let ([uf (make-uf)]) (do ([i 0 (+ i 1)] [last -1 (uf-make-set! uf)]) ((>= i 65) last)))",
        "(equal? (let ([uf (make-uf)]) (do ([i 0 (+ i 1)] [last -1 (uf-make-set! uf)]) ((>= i 65) last))) 64)",
        "hard",
        ["loop"],
    ),
    (
        "uf-make-set!",
        "Create one set and return whether its parent points to itself.",
        "(let ([uf (make-uf)]) (let ([id (uf-make-set! uf)]) (= (vector-ref (uf-parent uf) id) id)))",
        "(equal? (let ([uf (make-uf)]) (let ([id (uf-make-set! uf)]) (= (vector-ref (uf-parent uf) id) id))) #t)",
        "medium",
        ["property"],
    ),
    (
        "uf-make-set!",
        "Create one set and return `(list rank count size)` for that ID.",
        "(let ([uf (make-uf)]) (let ([id (uf-make-set! uf)]) (list (vector-ref (uf-rank uf) id) (uf-count uf) (uf-size uf))))",
        "(equal? (let ([uf (make-uf)]) (let ([id (uf-make-set! uf)]) (list (vector-ref (uf-rank uf) id) (uf-count uf) (uf-size uf)))) '(0 1 1))",
        "medium",
        ["direct"],
    ),
    (
        "uf-find",
        "Build three connected IDs via unions and return `(list (find 0) (find 2))`.",
        "(let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 3)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 1 2) (list (uf-find uf 0) (uf-find uf 2)))",
        "(let ([out (let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 3)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 1 2) (list (uf-find uf 0) (uf-find uf 2)))]) (= (car out) (cadr out)))",
        "hard",
        ["integration"],
    ),
    (
        "uf-find",
        "Create one singleton and return its representative.",
        "(let ([uf (make-uf)]) (let ([x (uf-make-set! uf)]) (uf-find uf x)))",
        "(equal? (let ([uf (make-uf)]) (let ([x (uf-make-set! uf)]) (uf-find uf x))) 0)",
        "easy",
        ["direct"],
    ),
    (
        "uf-find",
        "Create four IDs, merge into one class, call `(uf-find uf 3)`, then return parent of 3.",
        "(let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 4)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 2 3) (uf-union! uf 1 2) (let ([r (uf-find uf 3)]) (vector-ref (uf-parent uf) 3)))",
        "(let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 4)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 2 3) (uf-union! uf 1 2) (let ([r (uf-find uf 3)]) (= (vector-ref (uf-parent uf) 3) r)))",
        "hard",
        ["path-compression"],
    ),
    (
        "uf-find",
        "Create four sets, union 0-1 and 2-3, and return whether roots differ.",
        "(let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 4)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 2 3) (not (= (uf-find uf 0) (uf-find uf 2))))",
        "(equal? (let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 4)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 2 3) (not (= (uf-find uf 0) (uf-find uf 2)))) #t)",
        "medium",
        ["property"],
    ),
    (
        "uf-union!",
        "Create two sets, union them, and return whether both IDs now share the returned root.",
        "(let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)]) (let ([r (uf-union! uf a b)]) (and (= r (uf-find uf a)) (= r (uf-find uf b))))))",
        "(equal? (let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)]) (let ([r (uf-union! uf a b)]) (and (= r (uf-find uf a)) (= r (uf-find uf b)))))) #t)",
        "medium",
        ["direct"],
    ),
    (
        "uf-union!",
        "Create two singleton sets, union once, and return root rank.",
        "(let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)]) (let ([r (uf-union! uf a b)]) (vector-ref (uf-rank uf) r))))",
        "(equal? (let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)]) (let ([r (uf-union! uf a b)]) (vector-ref (uf-rank uf) r)))) 1)",
        "hard",
        ["rank"],
    ),
    (
        "uf-union!",
        "Union two IDs twice and return count.",
        "(let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)]) (uf-union! uf a b) (uf-union! uf a b) (uf-count uf)))",
        "(equal? (let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)]) (uf-union! uf a b) (uf-union! uf a b) (uf-count uf))) 1)",
        "medium",
        ["idempotence"],
    ),
    (
        "uf-union!",
        "Create four sets, merge 0-1 and 2-3, then merge those groups and return count.",
        "(let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 4)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 2 3) (uf-union! uf 1 2) (uf-count uf))",
        "(equal? (let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 4)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 2 3) (uf-union! uf 1 2) (uf-count uf)) 1)",
        "hard",
        ["integration"],
    ),
    (
        "uf-same-set?",
        "Create two sets, union them, and return `(list (same? a b) (same? b a))`.",
        "(let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)]) (uf-union! uf a b) (list (uf-same-set? uf a b) (uf-same-set? uf b a))))",
        "(equal? (let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)]) (uf-union! uf a b) (list (uf-same-set? uf a b) (uf-same-set? uf b a)))) '(#t #t))",
        "easy",
        ["direct"],
    ),
    (
        "uf-same-set?",
        "Create three fresh sets and return whether first and third are disjoint.",
        "(let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)] [c (uf-make-set! uf)]) (not (uf-same-set? uf a c))))",
        "(equal? (let ([uf (make-uf)]) (let ([a (uf-make-set! uf)] [b (uf-make-set! uf)] [c (uf-make-set! uf)]) (not (uf-same-set? uf a c)))) #t)",
        "easy",
        ["direct"],
    ),
    (
        "uf-same-set?",
        "Create chain unions 0-1 and 1-2, then test if 0 and 2 are equivalent.",
        "(let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 3)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 1 2) (uf-same-set? uf 0 2))",
        "(equal? (let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 3)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 1 2) (uf-same-set? uf 0 2)) #t)",
        "medium",
        ["transitivity"],
    ),
    (
        "uf-same-set?",
        "Create four sets, merge pairs (0,1) and (2,3), and return `(list same01 same02)`.",
        "(let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 4)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 2 3) (list (uf-same-set? uf 0 1) (uf-same-set? uf 0 2)))",
        "(equal? (let ([uf (make-uf)]) (do ([i 0 (+ i 1)]) ((>= i 4)) (uf-make-set! uf)) (uf-union! uf 0 1) (uf-union! uf 2 3) (list (uf-same-set? uf 0 1) (uf-same-set? uf 0 2))) '(#t #f))",
        "medium",
        ["integration"],
    ),
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
