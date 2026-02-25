#!/usr/bin/env python3
"""Generate Tier-1 SFT samples for lattice/egraph/egraph.ss."""

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

SOURCE_MODULE = "lattice/egraph/egraph.ss"
SOURCE_TEST = "lattice/egraph/test-egraph.ss"

VERIFY_LOADS = [
    '(load "lattice/egraph/union-find.ss")',
    '(load "lattice/egraph/eclass.ss")',
]

DEFS: Dict[str, str] = {
    "make-egraph": """(define (make-egraph)
  (vector egraph-tag
          (make-uf)
          (make-eclass-store)
          (make-hashtable enode-hash enode-equal?)
          hamt-empty
          (vector 0 0 0 0)))""",
    "egraph?": """(define (egraph? x)
  (and (vector? x)
       (>= (vector-length x) 6)
       (eq? (vector-ref x 0) egraph-tag)))""",
    "egraph-find": """(define (egraph-find eg id)
  (uf-find (egraph-uf eg) id))""",
    "egraph-lookup": """(define (egraph-lookup eg enode)
  (let* ([uf (egraph-uf eg)]
         [canonical (enode-canonicalize enode uf)]
         [hashcons (egraph-hashcons eg)]
         [found (hashtable-ref hashcons canonical #f)])
    (and found (uf-find uf found))))""",
    "egraph-add-enode!": """(define (egraph-add-enode! eg enode)
  (let* ([uf (egraph-uf eg)]
         [canonical (enode-canonicalize enode uf)]
         [hashcons (egraph-hashcons eg)]
         [existing (hashtable-ref hashcons canonical #f)])
    (if existing
        (begin
          (egraph-inc-stat! eg 3)
          (uf-find uf existing))
        (let* ([classes (egraph-classes eg)]
               [new-id (uf-make-set! uf)])
          (egraph-inc-stat! eg 0)
          (hashtable-set! hashcons canonical new-id)
          (eclass-add-node! classes new-id canonical)
          (let ([children (enode-children canonical)])
            (do ([i 0 (+ i 1)])
                ((>= i (vector-length children)))
              (let ([child-id (egraph-find eg (vector-ref children i))])
                (eclass-add-parent! classes child-id new-id))))
          new-id))))""",
    "egraph-merge!": """(define (egraph-merge! eg id1 id2)
  (let ([uf (egraph-uf eg)]
        [classes (egraph-classes eg)])
    (let ([root1 (uf-find uf id1)]
          [root2 (uf-find uf id2)])
      (if (= root1 root2)
          root1
          (begin
            (egraph-inc-stat! eg 1)
            (let ([new-root (uf-union! uf id1 id2)])
              (let ([affected (eclass-merge! classes uf id1 id2)])
                (for-each (lambda (p) (egraph-mark-dirty! eg p))
                          affected))
              new-root))))))""",
    "egraph-rebuild!": """(define (egraph-rebuild! eg)
  (let ([uf (egraph-uf eg)]
        [classes (egraph-classes eg)]
        [hashcons (egraph-hashcons eg)]
        [visited (make-eqv-hashtable)]
        [count 0])
    (let loop ()
      (let ([dirty-id (egraph-pop-dirty! eg)])
        (if (not dirty-id)
            count
            (let ([root (uf-find uf dirty-id)])
              (unless (hashtable-ref visited root #f)
                (hashtable-set! visited root #t)
                (egraph-inc-stat! eg 2)
                (set! count (+ count 1))
                (let ([nodes (eclass-get-nodes classes root)])
                  (for-each
                   (lambda (enode)
                     (let ([canonical (enode-canonicalize enode uf)])
                       (unless (enode-equal? enode canonical)
                         (hashtable-delete! hashcons enode))
                       (let ([existing (hashtable-ref hashcons canonical #f)])
                         (cond
                           [(not existing)
                            (hashtable-set! hashcons canonical root)]
                           [(not (= (uf-find uf existing) root))
                            (let ([new-root (egraph-merge! eg root existing)])
                              (hashtable-set! hashcons canonical new-root))]))))
                   nodes)))
              (loop)))))))""",
    "egraph-add-term!": """(define (egraph-add-term! eg term)
  (cond
    [(pair? term)
     (let* ([op (car term)]
            [args (cdr term)]
            [child-ids (map (lambda (arg) (egraph-add-term! eg arg)) args)]
            [enode (make-enode op (list->vector child-ids))])
       (egraph-add-enode! eg enode))]
    [else
     (egraph-add-enode! eg (make-enode term (vector)))]))""",
}

SUPPORT_DEFS: Dict[str, str] = {
    "egraph-tag": """(define egraph-tag 'egraph)""",
    "egraph-uf": """(define (egraph-uf eg) (vector-ref eg 1))""",
    "egraph-classes": """(define (egraph-classes eg) (vector-ref eg 2))""",
    "egraph-hashcons": """(define (egraph-hashcons eg) (vector-ref eg 3))""",
    "egraph-dirty-set": """(define (egraph-dirty-set eg) (vector-ref eg 4))""",
    "egraph-set-dirty-set!": """(define (egraph-set-dirty-set! eg ds) (vector-set! eg 4 ds))""",
    "egraph-stats": """(define (egraph-stats eg) (vector-ref eg 5))""",
    "egraph-stat-adds": """(define (egraph-stat-adds eg) (vector-ref (egraph-stats eg) 0))""",
    "egraph-stat-merges": """(define (egraph-stat-merges eg) (vector-ref (egraph-stats eg) 1))""",
    "egraph-stat-rebuilds": """(define (egraph-stat-rebuilds eg) (vector-ref (egraph-stats eg) 2))""",
    "egraph-stat-hits": """(define (egraph-stat-hits eg) (vector-ref (egraph-stats eg) 3))""",
    "egraph-inc-stat!": """(define (egraph-inc-stat! eg idx)
  (let ([stats (egraph-stats eg)])
    (vector-set! stats idx (+ (vector-ref stats idx) 1))))""",
    "egraph-dirty": """(define (egraph-dirty eg)
  (hamt-keys (egraph-dirty-set eg)))""",
    "egraph-mark-dirty!": """(define (egraph-mark-dirty! eg class-id)
  (egraph-set-dirty-set! eg (hamt-assoc class-id #t (egraph-dirty-set eg))))""",
    "egraph-clear-dirty!": """(define (egraph-clear-dirty! eg)
  (egraph-set-dirty-set! eg hamt-empty))""",
    "egraph-pop-dirty!": """(define (egraph-pop-dirty! eg)
  (let ([ds (egraph-dirty-set eg)])
    (if (hamt-empty? ds)
        #f
        (let ([id (hamt-first-key ds)])
          (egraph-set-dirty-set! eg (hamt-dissoc id ds))
          id))))""",
    "egraph-saturate-rebuild!": """(define (egraph-saturate-rebuild! eg)
  (let loop ([total 0])
    (let ([processed (egraph-rebuild! eg)])
      (if (zero? processed)
          total
          (loop (+ total processed))))))""",
    "egraph-class-count": """(define (egraph-class-count eg)
  (uf-count (egraph-uf eg)))""",
    "egraph-node-count": """(define (egraph-node-count eg)
  (eclass-node-count (egraph-classes eg) (egraph-uf eg)))""",
    "egraph-size": """(define (egraph-size eg)
  (uf-size (egraph-uf eg)))""",
    "egraph-stats-report": """(define (egraph-stats-report eg)
  `((classes . ,(egraph-class-count eg))
    (nodes . ,(egraph-node-count eg))
    (adds . ,(egraph-stat-adds eg))
    (merges . ,(egraph-stat-merges eg))
    (rebuilds . ,(egraph-stat-rebuilds eg))
    (hashcons-hits . ,(egraph-stat-hits eg))
    (dirty . ,(length (egraph-dirty eg)))))""",
    "egraph-class-nodes": """(define (egraph-class-nodes eg id)
  (let ([root (egraph-find eg id)])
    (eclass-get-nodes (egraph-classes eg) root)))""",
}

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **DEFS}

FUNCTION_ORDER = [
    "make-egraph",
    "egraph?",
    "egraph-find",
    "egraph-lookup",
    "egraph-add-enode!",
    "egraph-merge!",
    "egraph-rebuild!",
    "egraph-add-term!",
]

SUPPORT_ORDER = [
    "egraph-tag",
    "egraph-uf",
    "egraph-classes",
    "egraph-hashcons",
    "egraph-dirty-set",
    "egraph-set-dirty-set!",
    "egraph-stats",
    "egraph-stat-adds",
    "egraph-stat-merges",
    "egraph-stat-rebuilds",
    "egraph-stat-hits",
    "egraph-inc-stat!",
    "egraph-dirty",
    "egraph-mark-dirty!",
    "egraph-clear-dirty!",
    "egraph-pop-dirty!",
    "egraph-saturate-rebuild!",
    "egraph-class-count",
    "egraph-node-count",
    "egraph-size",
    "egraph-stats-report",
    "egraph-class-nodes",
]

DEF_ORDER = SUPPORT_ORDER + FUNCTION_ORDER

DEPENDS: Dict[str, List[str]] = {
    "egraph-tag": [],
    "egraph-uf": [],
    "egraph-classes": [],
    "egraph-hashcons": [],
    "egraph-dirty-set": [],
    "egraph-set-dirty-set!": [],
    "egraph-stats": [],
    "egraph-stat-adds": ["egraph-stats"],
    "egraph-stat-merges": ["egraph-stats"],
    "egraph-stat-rebuilds": ["egraph-stats"],
    "egraph-stat-hits": ["egraph-stats"],
    "egraph-inc-stat!": ["egraph-stats"],
    "egraph-dirty": ["egraph-dirty-set"],
    "egraph-mark-dirty!": ["egraph-dirty-set", "egraph-set-dirty-set!"],
    "egraph-clear-dirty!": ["egraph-set-dirty-set!"],
    "egraph-pop-dirty!": ["egraph-dirty-set", "egraph-set-dirty-set!"],
    "egraph-saturate-rebuild!": ["egraph-rebuild!"],
    "egraph-class-count": ["egraph-uf"],
    "egraph-node-count": ["egraph-classes", "egraph-uf"],
    "egraph-size": ["egraph-uf"],
    "egraph-stats-report": [
        "egraph-class-count",
        "egraph-node-count",
        "egraph-stat-adds",
        "egraph-stat-merges",
        "egraph-stat-rebuilds",
        "egraph-stat-hits",
        "egraph-dirty",
    ],
    "egraph-class-nodes": ["egraph-find", "egraph-classes"],
    "make-egraph": ["egraph-tag"],
    "egraph?": ["egraph-tag"],
    "egraph-find": ["egraph-uf"],
    "egraph-lookup": ["egraph-uf", "egraph-hashcons"],
    "egraph-add-enode!": [
        "egraph-uf",
        "egraph-hashcons",
        "egraph-classes",
        "egraph-inc-stat!",
        "egraph-find",
    ],
    "egraph-merge!": [
        "egraph-uf",
        "egraph-classes",
        "egraph-inc-stat!",
        "egraph-mark-dirty!",
    ],
    "egraph-rebuild!": [
        "egraph-uf",
        "egraph-classes",
        "egraph-hashcons",
        "egraph-pop-dirty!",
        "egraph-inc-stat!",
        "egraph-merge!",
    ],
    "egraph-add-term!": ["egraph-add-enode!"],
}

FUNCTION_SPECS = {
    "make-egraph": "Create an empty e-graph with union-find, e-class store, hashcons table, empty dirty set, and zeroed stats.",
    "egraph?": "Recognize e-graph values by vector shape and tag.",
    "egraph-find": "Return canonical e-class ID for a class by delegating to union-find.",
    "egraph-lookup": "Canonicalize an e-node, probe hashcons, and return canonical class ID or #f.",
    "egraph-add-enode!": "Insert canonical e-node with hashcons deduplication, stats updates, and parent registration.",
    "egraph-merge!": "Merge two classes, merge e-class data, and mark affected parent classes dirty.",
    "egraph-rebuild!": "Process dirty classes, canonicalize nodes, maintain hashcons, and trigger congruence merges.",
    "egraph-add-term!": "Recursively convert S-expression terms to e-nodes and insert them into the e-graph.",
}

SKELETONS = {
    "make-egraph": """(define (make-egraph)
  ;; TODO: initialize all e-graph components and counters
  <TODO>)""",
    "egraph?": """(define (egraph? x)
  ;; TODO: validate e-graph tag and structural shape
  <TODO>)""",
    "egraph-find": """(define (egraph-find eg id)
  ;; TODO: return canonical class ID via union-find
  <TODO>)""",
    "egraph-lookup": """(define (egraph-lookup eg enode)
  ;; TODO: canonicalize enode, query hashcons, return canonical class or #f
  <TODO>)""",
    "egraph-add-enode!": """(define (egraph-add-enode! eg enode)
  ;; TODO: deduplicate by hashcons or create a new class with parent links
  <TODO>)""",
    "egraph-merge!": """(define (egraph-merge! eg id1 id2)
  ;; TODO: merge classes and mark affected parents dirty
  <TODO>)""",
    "egraph-rebuild!": """(define (egraph-rebuild! eg)
  ;; TODO: rebuild dirty classes and propagate congruence merges
  <TODO>)""",
    "egraph-add-term!": """(define (egraph-add-term! eg term)
  ;; TODO: recursively lower term to e-nodes and insert
  <TODO>)""",
}

DIFFICULTY = {
    "make-egraph": "medium",
    "egraph?": "easy",
    "egraph-find": "easy",
    "egraph-lookup": "medium",
    "egraph-add-enode!": "hard",
    "egraph-merge!": "hard",
    "egraph-rebuild!": "hard",
    "egraph-add-term!": "medium",
}

VERIFY_BY_FUNCTION = {
    "make-egraph": """(and
  (let ([eg (make-egraph)])
    (and (egraph? eg)
         (= (egraph-class-count eg) 0)
         (= (egraph-node-count eg) 0)
         (= (egraph-stat-adds eg) 0)
         (= (egraph-stat-hits eg) 0)
         (hamt-empty? (egraph-dirty-set eg))))
  (let ([eg (make-egraph)])
    (let ([x (egraph-add-term! eg 'x)])
      (and (= x 0)
           (= (egraph-size eg) 1)
           (= (egraph-class-count eg) 1)))))""",
    "egraph?": """(and
  (egraph? (make-egraph))
  (not (egraph? '(egraph)))
  (not (egraph? (vector 'egraph
                       (make-uf)
                       (make-eclass-store)
                       (make-eqv-hashtable)
                       hamt-empty)))
  (not (egraph? (vector 'not-egraph
                       (make-uf)
                       (make-eclass-store)
                       (make-eqv-hashtable)
                       hamt-empty
                       (vector 0 0 0 0)))))""",
    "egraph-find": """(and
  (let ([eg (make-egraph)])
    (let ([a (egraph-add-term! eg 'a)]
          [b (egraph-add-term! eg 'b)]
          [c (egraph-add-term! eg 'c)])
      (egraph-merge! eg a b)
      (egraph-merge! eg b c)
      (and (= (egraph-find eg a) (egraph-find eg b))
           (= (egraph-find eg b) (egraph-find eg c)))))
  (let ([eg (make-egraph)])
    (let ([x (egraph-add-term! eg 'x)])
      (= (egraph-find eg x) x))))""",
    "egraph-lookup": """(and
  (let ([eg (make-egraph)])
    (let ([n (make-enode 'q (vector))])
      (and (not (egraph-lookup eg n))
           (let ([id (egraph-add-enode! eg n)])
             (= (egraph-lookup eg n) id)))))
  (let ([eg (make-egraph)])
    (let ([fx (egraph-add-term! eg '(f x))]
          [fy (egraph-add-term! eg '(f y))]
          [x (egraph-add-term! eg 'x)]
          [y (egraph-add-term! eg 'y)])
      (egraph-merge! eg x y)
      (egraph-saturate-rebuild! eg)
      (= (egraph-find eg fx)
         (egraph-lookup eg (make-enode 'f (vector y)))))))""",
    "egraph-add-enode!": """(and
  (let ([eg (make-egraph)]
        [n (make-enode 'x (vector))])
    (let ([id1 (egraph-add-enode! eg n)]
          [id2 (egraph-add-enode! eg n)])
      (and (= id1 id2)
           (= (egraph-class-count eg) 1)
           (= (egraph-stat-adds eg) 1)
           (= (egraph-stat-hits eg) 1))))
  (let ([eg (make-egraph)])
    (let ([x (egraph-add-term! eg 'x)])
      (let* ([parent-id (egraph-add-enode! eg (make-enode 'f (vector x)))]
             [parents (eclass-get-parents (egraph-classes eg) x)])
        (and (not (null? parents))
             (not (not (memv parent-id parents))))))))""",
    "egraph-merge!": """(and
  (let ([eg (make-egraph)])
    (let ([a (egraph-add-term! eg 'a)]
          [b (egraph-add-term! eg 'b)])
      (egraph-merge! eg a b)
      (egraph-merge! eg a b)
      (and (= (egraph-find eg a) (egraph-find eg b))
           (= (egraph-class-count eg) 1)
           (= (egraph-stat-merges eg) 1))))
  (let ([eg (make-egraph)])
    (let* ([x (egraph-add-term! eg 'x)]
           [y (egraph-add-term! eg 'y)]
           [fx (egraph-add-term! eg '(f x))]
           [fy (egraph-add-term! eg '(f y))])
      (egraph-merge! eg x y)
      (and (> (length (egraph-dirty eg)) 0)
           (begin
             (egraph-saturate-rebuild! eg)
             (= (egraph-find eg fx) (egraph-find eg fy)))))))""",
    "egraph-rebuild!": """(and
  (let ([eg (make-egraph)])
    (let ([fx (egraph-add-term! eg '(f x))]
          [fy (egraph-add-term! eg '(f y))]
          [x (egraph-add-term! eg 'x)]
          [y (egraph-add-term! eg 'y)])
      (egraph-merge! eg x y)
      (let ([processed (egraph-rebuild! eg)])
        (and (> processed 0)
             (= (egraph-find eg fx) (egraph-find eg fy))
             (= (egraph-rebuild! eg) 0)))))
  (let ([eg (make-egraph)])
    (let ([fa (egraph-add-term! eg '(f a))]
          [fb (egraph-add-term! eg '(f b))]
          [fc (egraph-add-term! eg '(f c))]
          [a (egraph-add-term! eg 'a)]
          [b (egraph-add-term! eg 'b)]
          [c (egraph-add-term! eg 'c)])
      (egraph-merge! eg a b)
      (egraph-merge! eg b c)
      (let ([total (egraph-saturate-rebuild! eg)])
        (and (> total 0)
             (= (egraph-find eg fa) (egraph-find eg fb))
             (= (egraph-find eg fb) (egraph-find eg fc)))))))""",
    "egraph-add-term!": """(and
  (let ([eg (make-egraph)])
    (egraph-add-term! eg '(+ x y))
    (= (egraph-class-count eg) 3))
  (let ([eg (make-egraph)])
    (egraph-add-term! eg '(+ x x))
    (= (egraph-class-count eg) 2))
  (let ([eg (make-egraph)])
    (egraph-add-term! eg '(+ (* a b) c))
    (= (egraph-class-count eg) 5))
  (let ([eg (make-egraph)])
    (let ([id1 (egraph-add-term! eg "hello")]
          [id2 (egraph-add-term! eg "hello")])
      (= (egraph-find eg id1) (egraph-find eg id2)))))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "make-egraph": """def make_egraph():
    return {
        "tag": "egraph",
        "uf": make_uf(),
        "classes": make_eclass_store(),
        "hashcons": {},
        "dirty": persistent_empty_map(),
        "stats": [0, 0, 0, 0],
    }""",
    "egraph?": """def egraph_pred(x):
    return isinstance(x, list) and len(x) >= 6 and x[0] == "egraph""" ,
    "egraph-find": """def egraph_find(eg, class_id):
    return uf_find(egraph_uf(eg), class_id)""",
    "egraph-lookup": """def egraph_lookup(eg, enode):
    uf = egraph_uf(eg)
    canonical = enode_canonicalize(enode, uf)
    found = egraph_hashcons(eg).get(canonical)
    return uf_find(uf, found) if found is not None else False""",
    "egraph-add-enode!": """def egraph_add_enode(eg, enode):
    uf = egraph_uf(eg)
    canonical = enode_canonicalize(enode, uf)
    table = egraph_hashcons(eg)
    existing = table.get(canonical)
    if existing is not None:
        egraph_inc_stat(eg, 3)
        return uf_find(uf, existing)

    classes = egraph_classes(eg)
    new_id = uf_make_set(uf)
    egraph_inc_stat(eg, 0)
    table[canonical] = new_id
    eclass_add_node(classes, new_id, canonical)

    for child in enode_children(canonical):
        child_id = egraph_find(eg, child)
        eclass_add_parent(classes, child_id, new_id)
    return new_id""",
    "egraph-merge!": """def egraph_merge(eg, id1, id2):
    uf = egraph_uf(eg)
    classes = egraph_classes(eg)
    root1 = uf_find(uf, id1)
    root2 = uf_find(uf, id2)
    if root1 == root2:
        return root1

    egraph_inc_stat(eg, 1)
    new_root = uf_union(uf, id1, id2)
    affected = eclass_merge(classes, uf, id1, id2)
    for parent in affected:
        egraph_mark_dirty(eg, parent)
    return new_root""",
    "egraph-rebuild!": """def egraph_rebuild(eg):
    uf = egraph_uf(eg)
    classes = egraph_classes(eg)
    table = egraph_hashcons(eg)
    visited = set()
    processed = 0

    while True:
        dirty = egraph_pop_dirty(eg)
        if dirty is False:
            return processed

        root = uf_find(uf, dirty)
        if root in visited:
            continue

        visited.add(root)
        egraph_inc_stat(eg, 2)
        processed += 1

        for enode in eclass_get_nodes(classes, root):
            canonical = enode_canonicalize(enode, uf)
            if enode != canonical:
                table.pop(enode, None)

            existing = table.get(canonical)
            if existing is None:
                table[canonical] = root
            elif uf_find(uf, existing) != root:
                new_root = egraph_merge(eg, root, existing)
                table[canonical] = new_root""",
    "egraph-add-term!": """def egraph_add_term(eg, term):
    if isinstance(term, (list, tuple)) and len(term) > 0:
        op = term[0]
        args = term[1:]
        child_ids = [egraph_add_term(eg, arg) for arg in args]
        return egraph_add_enode(eg, make_enode(op, child_ids))
    return egraph_add_enode(eg, make_enode(term, []))""",
}

CHEZ_SNIPPETS = {
    "make-egraph": """(define (mk-egraph)
  (vector egraph-tag
          (make-uf)
          (make-eclass-store)
          (make-hashtable enode-hash enode-equal?)
          hamt-empty
          (vector 0 0 0 0)))""",
    "egraph?": """(define (eg? x)
  (and (vector? x)
       (>= (vector-length x) 6)
       (eq? (vector-ref x 0) egraph-tag)))""",
    "egraph-find": """(define (eg-find eg id)
  (uf-find (egraph-uf eg) id))""",
    "egraph-lookup": """(define (eg-lookup eg enode)
  (let* ([uf (egraph-uf eg)]
         [canon (enode-canonicalize enode uf)]
         [tbl (egraph-hashcons eg)]
         [found (hashtable-ref tbl canon #f)])
    (and found (uf-find uf found))))""",
    "egraph-add-enode!": """(define (eg-add-enode! eg enode)
  (let* ([uf (egraph-uf eg)]
         [canon (enode-canonicalize enode uf)]
         [tbl (egraph-hashcons eg)]
         [existing (hashtable-ref tbl canon #f)])
    (if existing
        (begin
          (egraph-inc-stat! eg 3)
          (uf-find uf existing))
        (let* ([classes (egraph-classes eg)]
               [cid (uf-make-set! uf)])
          (egraph-inc-stat! eg 0)
          (hashtable-set! tbl canon cid)
          (eclass-add-node! classes cid canon)
          (let ([kids (enode-children canon)])
            (do ([i 0 (+ i 1)])
                ((>= i (vector-length kids)))
              (eclass-add-parent! classes (egraph-find eg (vector-ref kids i)) cid)))
          cid))))""",
    "egraph-merge!": """(define (eg-merge! eg id1 id2)
  (let ([uf (egraph-uf eg)]
        [classes (egraph-classes eg)])
    (let ([r1 (uf-find uf id1)]
          [r2 (uf-find uf id2)])
      (if (= r1 r2)
          r1
          (begin
            (egraph-inc-stat! eg 1)
            (let ([root (uf-union! uf id1 id2)]
                  [affected (eclass-merge! classes uf id1 id2)])
              (for-each (lambda (p) (egraph-mark-dirty! eg p)) affected)
              root))))))""",
    "egraph-rebuild!": """(define (eg-rebuild! eg)
  (let ([uf (egraph-uf eg)]
        [classes (egraph-classes eg)]
        [tbl (egraph-hashcons eg)]
        [visited (make-eqv-hashtable)]
        [count 0])
    (let loop ()
      (let ([dirty (egraph-pop-dirty! eg)])
        (if (not dirty)
            count
            (let ([root (uf-find uf dirty)])
              (unless (hashtable-ref visited root #f)
                (hashtable-set! visited root #t)
                (egraph-inc-stat! eg 2)
                (set! count (+ count 1))
                (for-each
                 (lambda (enode)
                   (let ([canon (enode-canonicalize enode uf)])
                     (unless (enode-equal? enode canon)
                       (hashtable-delete! tbl enode))
                     (let ([existing (hashtable-ref tbl canon #f)])
                       (cond
                         [(not existing)
                          (hashtable-set! tbl canon root)]
                         [(not (= (uf-find uf existing) root))
                          (hashtable-set! tbl canon (egraph-merge! eg root existing))]))))
                 (eclass-get-nodes classes root)))
              (loop)))))))""",
    "egraph-add-term!": """(define (eg-add-term! eg term)
  (if (pair? term)
      (let* ([op (car term)]
             [args (cdr term)]
             [kids (map (lambda (arg) (eg-add-term! eg arg)) args)])
        (egraph-add-enode! eg (make-enode op (list->vector kids))))
      (egraph-add-enode! eg (make-enode term (vector)))))""",
}

BUGGY_CASES = [
    {
        "fn": "make-egraph",
        "buggy": """(define (make-egraph)
  (vector egraph-tag
          (make-uf)
          (make-eclass-store)
          (make-hashtable enode-hash enode-equal?)
          hamt-empty
          (vector 0 0 0)))""",
        "note": "Stats vector must track adds, merges, rebuilds, and hashcons hits (4 slots).",
    },
    {
        "fn": "make-egraph",
        "buggy": """(define (make-egraph)
  (vector egraph-tag
          (make-uf)
          (make-eclass-store)
          (make-hashtable enode-hash enode-equal?)
          '()
          (vector 0 0 0 0)))""",
        "note": "Dirty set must start as `hamt-empty` for O(1) membership operations.",
    },
    {
        "fn": "egraph?",
        "buggy": """(define (egraph? x)
  (and (vector? x)
       (> (vector-length x) 6)
       (eq? (vector-ref x 0) egraph-tag)))""",
        "note": "Valid e-graphs have length 6; strict `> 6` rejects canonical values.",
    },
    {
        "fn": "egraph?",
        "buggy": """(define (egraph? x)
  (and (vector? x)
       (>= (vector-length x) 6)
       (eq? (vector-ref x 1) egraph-tag)))""",
        "note": "Tag is stored at index 0, not index 1.",
    },
    {
        "fn": "egraph-find",
        "buggy": """(define (egraph-find eg id)
  id)""",
        "note": "Class IDs must be canonicalized through union-find.",
    },
    {
        "fn": "egraph-find",
        "buggy": """(define (egraph-find eg id)
  (uf-find (egraph-uf eg) (+ id 1)))""",
        "note": "The function must look up the provided ID directly; no offset is valid.",
    },
    {
        "fn": "egraph-lookup",
        "buggy": """(define (egraph-lookup eg enode)
  (let* ([uf (egraph-uf eg)]
         [hashcons (egraph-hashcons eg)]
         [found (hashtable-ref hashcons enode #f)])
    (and found (uf-find uf found))))""",
        "note": "Lookups must canonicalize e-node children before probing hashcons.",
    },
    {
        "fn": "egraph-lookup",
        "buggy": """(define (egraph-lookup eg enode)
  (let* ([uf (egraph-uf eg)]
         [canonical (enode-canonicalize enode uf)]
         [hashcons (egraph-hashcons eg)]
         [found (hashtable-ref hashcons canonical #f)])
    found))""",
        "note": "Returned IDs must be canonicalized with `uf-find` to avoid stale representatives.",
    },
    {
        "fn": "egraph-add-enode!",
        "buggy": """(define (egraph-add-enode! eg enode)
  (let* ([uf (egraph-uf eg)]
         [canonical (enode-canonicalize enode uf)]
         [hashcons (egraph-hashcons eg)]
         [existing (hashtable-ref hashcons canonical #f)])
    (if existing
        (begin
          (egraph-inc-stat! eg 3)
          existing)
        (let* ([classes (egraph-classes eg)]
               [new-id (uf-make-set! uf)])
          (egraph-inc-stat! eg 0)
          (hashtable-set! hashcons canonical new-id)
          (eclass-add-node! classes new-id canonical)
          (let ([children (enode-children canonical)])
            (do ([i 0 (+ i 1)])
                ((>= i (vector-length children)))
              (let ([child-id (egraph-find eg (vector-ref children i))])
                (eclass-add-parent! classes child-id new-id))))
          new-id))))""",
        "note": "Hashcons hit path must return canonical representative, not possibly stale ID.",
    },
    {
        "fn": "egraph-add-enode!",
        "buggy": """(define (egraph-add-enode! eg enode)
  (let* ([uf (egraph-uf eg)]
         [canonical (enode-canonicalize enode uf)]
         [hashcons (egraph-hashcons eg)]
         [existing (hashtable-ref hashcons canonical #f)])
    (if existing
        (begin
          (egraph-inc-stat! eg 3)
          (uf-find uf existing))
        (let* ([classes (egraph-classes eg)]
               [new-id (uf-make-set! uf)])
          (egraph-inc-stat! eg 0)
          (hashtable-set! hashcons canonical new-id)
          (eclass-add-node! classes new-id canonical)
          new-id))))""",
        "note": "New nodes must register parent links for each child to support rebuild propagation.",
    },
    {
        "fn": "egraph-merge!",
        "buggy": """(define (egraph-merge! eg id1 id2)
  (let ([uf (egraph-uf eg)]
        [classes (egraph-classes eg)])
    (let ([root1 (uf-find uf id1)]
          [root2 (uf-find uf id2)])
      (if (= root1 root2)
          root1
          (begin
            (egraph-inc-stat! eg 1)
            (let ([new-root (uf-union! uf id1 id2)])
              (eclass-merge! classes uf id1 id2)
              new-root))))))""",
        "note": "Affected parent classes from `eclass-merge!` must be marked dirty for rebuild.",
    },
    {
        "fn": "egraph-merge!",
        "buggy": """(define (egraph-merge! eg id1 id2)
  (let ([uf (egraph-uf eg)]
        [classes (egraph-classes eg)])
    (egraph-inc-stat! eg 1)
    (let ([new-root (uf-union! uf id1 id2)])
      (let ([affected (eclass-merge! classes uf id1 id2)])
        (for-each (lambda (p) (egraph-mark-dirty! eg p)) affected))
      new-root)))""",
        "note": "Already-equivalent classes must short-circuit before incrementing merge stats or mutating structures.",
    },
    {
        "fn": "egraph-rebuild!",
        "buggy": """(define (egraph-rebuild! eg)
  (let ([uf (egraph-uf eg)]
        [classes (egraph-classes eg)]
        [hashcons (egraph-hashcons eg)]
        [visited (make-eqv-hashtable)]
        [count 0])
    (let loop ()
      (let ([dirty-id (egraph-pop-dirty! eg)])
        (if (not dirty-id)
            count
            (let ([root dirty-id])
              (unless (hashtable-ref visited root #f)
                (hashtable-set! visited root #t)
                (egraph-inc-stat! eg 2)
                (set! count (+ count 1))
                (let ([nodes (eclass-get-nodes classes root)])
                  (for-each
                   (lambda (enode)
                     (let ([canonical (enode-canonicalize enode uf)])
                       (unless (enode-equal? enode canonical)
                         (hashtable-delete! hashcons enode))
                       (let ([existing (hashtable-ref hashcons canonical #f)])
                         (cond
                           [(not existing)
                            (hashtable-set! hashcons canonical root)]
                           [(not (= (uf-find uf existing) root))
                            (let ([new-root (egraph-merge! eg root existing)])
                              (hashtable-set! hashcons canonical new-root))]))))
                   nodes)))
              (loop)))))))""",
        "note": "Dirty IDs must be canonicalized to roots before processing to avoid skipped rebuild work.",
    },
    {
        "fn": "egraph-rebuild!",
        "buggy": """(define (egraph-rebuild! eg)
  (let ([uf (egraph-uf eg)]
        [classes (egraph-classes eg)]
        [hashcons (egraph-hashcons eg)]
        [visited (make-eqv-hashtable)]
        [count 0])
    (let loop ()
      (let ([dirty-id (egraph-pop-dirty! eg)])
        (if (not dirty-id)
            count
            (let ([root (uf-find uf dirty-id)])
              (unless (hashtable-ref visited root #f)
                (hashtable-set! visited root #t)
                (egraph-inc-stat! eg 2)
                (set! count (+ count 1))
                (let ([nodes (eclass-get-nodes classes root)])
                  (for-each
                   (lambda (enode)
                     (let ([canonical (enode-canonicalize enode uf)])
                       (unless (enode-equal? enode canonical)
                         (hashtable-delete! hashcons enode))
                       (let ([existing (hashtable-ref hashcons canonical #f)])
                         (cond
                           [(not existing)
                            (hashtable-set! hashcons canonical root)]
                           [(not (= (uf-find uf existing) root))
                            (hashtable-set! hashcons canonical root)]))))
                   nodes)))
              (loop)))))))""",
        "note": "When canonical enodes already exist in another class, rebuild must merge classes (not overwrite hashcons only).",
    },
    {
        "fn": "egraph-add-term!",
        "buggy": """(define (egraph-add-term! eg term)
  (cond
    [(pair? term)
     (let* ([op (car term)]
            [args (cdr term)]
            [enode (make-enode op (list->vector args))])
       (egraph-add-enode! eg enode))]
    [else
     (egraph-add-enode! eg (make-enode term (vector)))]))""",
        "note": "Application arguments must be recursively inserted and replaced by child e-class IDs.",
    },
    {
        "fn": "egraph-add-term!",
        "buggy": """(define (egraph-add-term! eg term)
  (cond
    [(pair? term)
     (let* ([op (car term)]
            [args (cdr term)]
            [child-ids (map (lambda (arg) (egraph-add-term! eg arg)) args)]
            [enode (make-enode op (list->vector child-ids))])
       (egraph-add-enode! eg enode))]
    [else
     (egraph-add-enode! eg (make-enode 'term (vector)))]))""",
        "note": "Leaf terms must preserve the literal value; they cannot all map to symbol `'term`.",
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
    sid = f"egraph_egraph_{family}_{family_counter[family]:03d}"
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


def verify_refs(verify_expr: str) -> List[str]:
    tokens = set(TOKEN_RE.findall(verify_expr))
    return [name for name in DEF_ORDER if name in tokens]


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
    parts = [ALL_DEFS[name] for name in defs_needed] + VERIFY_LOADS + [verify_check.strip()]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


def def_verify(fn: str) -> str:
    return build_verify(VERIFY_BY_FUNCTION[fn], [fn])


def normalize_ws(s: str) -> str:
    return " ".join(s.split())


def defined_function_name(code: str) -> str | None:
    m = re.search(r"\(define\s+\(([^\s)]+)", code)
    return m.group(1) if m else None


def enforce_quality(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        family = str(row["family"])
        source_fn = str(row["source_function"])
        prompt = str(row["prompt"])
        gt = str(row["ground_truth"])
        verify = str(row["verify_expr"])

        if family in {"spec_to_code", "translation", "bugfix"}:
            fn_name = defined_function_name(gt)
            if fn_name != source_fn:
                raise ValueError(
                    f"source_function mismatch for {row['id']}: expected {source_fn}, ground_truth defines {fn_name}"
                )
            if source_fn not in prompt:
                raise ValueError(f"prompt/function mismatch for {row['id']}: prompt does not mention {source_fn}")

        if family == "composition":
            gt_tokens = set(TOKEN_RE.findall(gt))
            if source_fn not in gt_tokens:
                raise ValueError(
                    f"composition source_function mismatch for {row['id']}: {source_fn} not used in ground_truth"
                )

        if family == "bugfix":
            if "Known issue:" not in prompt:
                raise ValueError(f"bugfix prompt too weak for {row['id']}: missing explicit issue")
            if "<TODO>" in prompt or "<TODO>" in gt:
                raise ValueError(f"bugfix stub detected for {row['id']}")

        gt_norm = normalize_ws(gt)
        verify_norm = normalize_ws(verify)
        if gt_norm == verify_norm:
            raise ValueError(f"tautological verify_expr for {row['id']}: identical to ground_truth")
        if f"(equal? {gt_norm} {gt_norm})" in verify_norm:
            raise ValueError(f"tautological equality verify_expr for {row['id']}")
        if f"(eq? {gt_norm} {gt_norm})" in verify_norm:
            raise ValueError(f"tautological eq verify_expr for {row['id']}")


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement this e-graph function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Return exactly one definition for `{fn}` and no extra text.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "egraph-core", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Complete this Fold Scheme skeleton for `{fn}`.

```scheme
{SKELETONS[fn]}
```

Replace `<TODO>` and return only the completed function definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "egraph-core", "skeleton-completion", fn],
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
        prompt=f"""Translate this Python implementation into Fold-native Scheme.
Preserve behavior and use the exact target name `{fn}`.
Return only the Scheme function definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "egraph-core", "python-to-scheme", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Convert this Chez-style snippet into canonical Fold style.
Target function name: `{fn}`.
Return only the final Fold definition.

```scheme
{CHEZ_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "egraph", "egraph-core", "chez-to-fold", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = case["fn"]
    if case["buggy"].strip() == DEFS[fn].strip():
        raise ValueError(f"bugfix case for {fn} is identical to ground_truth")

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
        tags=["tier1", "egraph", "egraph-core", "bugfix", fn],
    )

if sum(1 for s in samples if s["family"] == "bugfix") != 16:
    raise ValueError("bugfix family must contain exactly 16 samples")


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
        verify_expr=build_verify(verify_expr, [source_function]),
        tags=["tier1", "egraph", "egraph-core", "composition", source_function] + extra_tags,
    )


composition_cases = [
    (
        "make-egraph",
        "Create an e-graph, add `x` and `y`, then return `(list class-count node-count adds)`.",
        "(let ([eg (make-egraph)]) (egraph-add-term! eg 'x) (egraph-add-term! eg 'y) (list (egraph-class-count eg) (egraph-node-count eg) (egraph-stat-adds eg)))",
        "(equal? (let ([eg (make-egraph)]) (egraph-add-term! eg 'x) (egraph-add-term! eg 'y) (list (egraph-class-count eg) (egraph-node-count eg) (egraph-stat-adds eg))) '(2 2 2))",
        "medium",
        ["integration"],
    ),
    (
        "make-egraph",
        "Create an e-graph, add the same symbol twice, and report `(same-class? hashcons-hits class-count)`.",
        "(let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg 'x)] [b (egraph-add-term! eg 'x)]) (list (= (egraph-find eg a) (egraph-find eg b)) (egraph-stat-hits eg) (egraph-class-count eg))))",
        "(equal? (let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg 'x)] [b (egraph-add-term! eg 'x)]) (list (= (egraph-find eg a) (egraph-find eg b)) (egraph-stat-hits eg) (egraph-class-count eg)))) '(#t 1 1))",
        "medium",
        ["dedup"],
    ),
    (
        "make-egraph",
        "Initialize an e-graph, add `(+ x x)`, and return `(list class-count lookup-matches?)`.",
        "(let ([eg (make-egraph)]) (let ([term-id (egraph-add-term! eg '(+ x x))] [lookup-id (egraph-lookup eg (make-enode '+ (vector (egraph-add-term! eg 'x) (egraph-add-term! eg 'x))))]) (list (egraph-class-count eg) (= (egraph-find eg term-id) (egraph-find eg lookup-id)))))",
        "(equal? (let ([eg (make-egraph)]) (let ([term-id (egraph-add-term! eg '(+ x x))] [lookup-id (egraph-lookup eg (make-enode '+ (vector (egraph-add-term! eg 'x) (egraph-add-term! eg 'x))))]) (list (egraph-class-count eg) (= (egraph-find eg term-id) (egraph-find eg lookup-id))))) '(2 #t))",
        "hard",
        ["lookup"],
    ),
    (
        "make-egraph",
        "Create an e-graph, merge `x` and `y`, saturate rebuild, and return `(list classes dirty merges)`.",
        "(let ([eg (make-egraph)]) (let ([x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-saturate-rebuild! eg) (list (egraph-class-count eg) (length (egraph-dirty eg)) (egraph-stat-merges eg))))",
        "(equal? (let ([eg (make-egraph)]) (let ([x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-saturate-rebuild! eg) (list (egraph-class-count eg) (length (egraph-dirty eg)) (egraph-stat-merges eg)))) '(1 0 1))",
        "hard",
        ["merge"],
    ),
    (
        "egraph?",
        "After inserting one term, return `(list (egraph? eg) (egraph? '(egraph)))`.",
        "(let ([eg (make-egraph)]) (egraph-add-term! eg 'x) (list (egraph? eg) (egraph? '(egraph))))",
        "(equal? (let ([eg (make-egraph)]) (egraph-add-term! eg 'x) (list (egraph? eg) (egraph? '(egraph)))) '(#t #f))",
        "easy",
        ["predicate"],
    ),
    (
        "egraph?",
        "Map `egraph?` over `(list (make-egraph) 7 #t)` and return the boolean list.",
        "(map egraph? (list (make-egraph) 7 #t))",
        "(equal? (map egraph? (list (make-egraph) 7 #t)) '(#t #f #f))",
        "easy",
        ["predicate"],
    ),
    (
        "egraph?",
        "Return whether a wrong-tag vector is rejected while a normal e-graph is accepted.",
        "(let ([bad (vector 'wrong (make-uf) (make-eclass-store) (make-eqv-hashtable) hamt-empty (vector 0 0 0 0))] [good (make-egraph)]) (and (egraph? good) (not (egraph? bad))))",
        "(equal? (let ([bad (vector 'wrong (make-uf) (make-eclass-store) (make-eqv-hashtable) hamt-empty (vector 0 0 0 0))] [good (make-egraph)]) (and (egraph? good) (not (egraph? bad)))) #t)",
        "medium",
        ["edge-case"],
    ),
    (
        "egraph?",
        "Build and mutate an e-graph, then return `(and (egraph? eg) (pair? (egraph-stats-report eg)))`.",
        "(let ([eg (make-egraph)]) (egraph-add-term! eg '(+ x y)) (egraph-merge! eg (egraph-add-term! eg 'x) (egraph-add-term! eg 'y)) (and (egraph? eg) (pair? (egraph-stats-report eg))))",
        "(equal? (let ([eg (make-egraph)]) (egraph-add-term! eg '(+ x y)) (egraph-merge! eg (egraph-add-term! eg 'x) (egraph-add-term! eg 'y)) (and (egraph? eg) (pair? (egraph-stats-report eg)))) #t)",
        "medium",
        ["integration"],
    ),
    (
        "egraph-find",
        "Add `a` and `b`, merge them, and return whether their roots are equal.",
        "(let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg 'a)] [b (egraph-add-term! eg 'b)]) (egraph-merge! eg a b) (= (egraph-find eg a) (egraph-find eg b))))",
        "(equal? (let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg 'a)] [b (egraph-add-term! eg 'b)]) (egraph-merge! eg a b) (= (egraph-find eg a) (egraph-find eg b)))) #t)",
        "medium",
        ["merge"],
    ),
    (
        "egraph-find",
        "Create `a`, `b`, `c`, merge through `b`, and return `(list eq-ab eq-ac)`.",
        "(let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg 'a)] [b (egraph-add-term! eg 'b)] [c (egraph-add-term! eg 'c)]) (egraph-merge! eg a b) (egraph-merge! eg b c) (list (= (egraph-find eg a) (egraph-find eg b)) (= (egraph-find eg a) (egraph-find eg c)))))",
        "(equal? (let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg 'a)] [b (egraph-add-term! eg 'b)] [c (egraph-add-term! eg 'c)]) (egraph-merge! eg a b) (egraph-merge! eg b c) (list (= (egraph-find eg a) (egraph-find eg b)) (= (egraph-find eg a) (egraph-find eg c))))) '(#t #t))",
        "hard",
        ["transitivity"],
    ),
    (
        "egraph-find",
        "Check congruence propagation by comparing roots of `(f x)` and `(f y)` after merging `x` and `y`.",
        "(let ([eg (make-egraph)]) (let ([fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))] [x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-saturate-rebuild! eg) (= (egraph-find eg fx) (egraph-find eg fy))))",
        "(equal? (let ([eg (make-egraph)]) (let ([fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))] [x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-saturate-rebuild! eg) (= (egraph-find eg fx) (egraph-find eg fy)))) #t)",
        "hard",
        ["congruence"],
    ),
    (
        "egraph-find",
        "Insert `x` twice and return whether both IDs have the same canonical root.",
        "(let ([eg (make-egraph)]) (let ([id1 (egraph-add-term! eg 'x)] [id2 (egraph-add-term! eg 'x)]) (= (egraph-find eg id1) (egraph-find eg id2))))",
        "(equal? (let ([eg (make-egraph)]) (let ([id1 (egraph-add-term! eg 'x)] [id2 (egraph-add-term! eg 'x)]) (= (egraph-find eg id1) (egraph-find eg id2)))) #t)",
        "easy",
        ["dedup"],
    ),
    (
        "egraph-lookup",
        "Lookup a nullary node before and after insertion, returning `(list missing-before found-after)`.",
        "(let ([eg (make-egraph)] [n (make-enode 'q (vector))]) (list (not (egraph-lookup eg n)) (begin (egraph-add-enode! eg n) (not (not (egraph-lookup eg n))))))",
        "(equal? (let ([eg (make-egraph)] [n (make-enode 'q (vector))]) (list (not (egraph-lookup eg n)) (begin (egraph-add-enode! eg n) (not (not (egraph-lookup eg n)))))) '(#t #t))",
        "medium",
        ["hashcons"],
    ),
    (
        "egraph-lookup",
        "Add `(f x)`, then lookup the same canonical enode and return whether roots match.",
        "(let ([eg (make-egraph)]) (let ([fx (egraph-add-term! eg '(f x))] [x (egraph-add-term! eg 'x)]) (= (egraph-find eg fx) (egraph-lookup eg (make-enode 'f (vector x))))))",
        "(equal? (let ([eg (make-egraph)]) (let ([fx (egraph-add-term! eg '(f x))] [x (egraph-add-term! eg 'x)]) (= (egraph-find eg fx) (egraph-lookup eg (make-enode 'f (vector x)))))) #t)",
        "medium",
        ["canonical"],
    ),
    (
        "egraph-lookup",
        "Merge `x` and `y`, rebuild, then test whether lookup of `(f y)` resolves to class of `(f x)`.",
        "(let ([eg (make-egraph)]) (let ([fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))] [x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-saturate-rebuild! eg) (= (egraph-find eg fx) (egraph-lookup eg (make-enode 'f (vector y))))))",
        "(equal? (let ([eg (make-egraph)]) (let ([fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))] [x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-saturate-rebuild! eg) (= (egraph-find eg fx) (egraph-lookup eg (make-enode 'f (vector y)))))) #t)",
        "hard",
        ["congruence"],
    ),
    (
        "egraph-lookup",
        "After adding `(+ x y)`, verify lookup of unrelated `(* x y)` stays absent.",
        "(let ([eg (make-egraph)]) (egraph-add-term! eg '(+ x y)) (not (egraph-lookup eg (make-enode '* (vector 0 1)))))",
        "(equal? (let ([eg (make-egraph)]) (egraph-add-term! eg '(+ x y)) (not (egraph-lookup eg (make-enode '* (vector 0 1))))) #t)",
        "medium",
        ["negative"],
    ),
    (
        "egraph-add-enode!",
        "Insert the same enode twice with `egraph-add-enode!` and return `(same-id? hits classes)`.",
        "(let ([eg (make-egraph)] [n (make-enode 'x (vector))]) (let ([id1 (egraph-add-enode! eg n)] [id2 (egraph-add-enode! eg n)]) (list (= id1 id2) (egraph-stat-hits eg) (egraph-class-count eg))))",
        "(equal? (let ([eg (make-egraph)] [n (make-enode 'x (vector))]) (let ([id1 (egraph-add-enode! eg n)] [id2 (egraph-add-enode! eg n)]) (list (= id1 id2) (egraph-stat-hits eg) (egraph-class-count eg)))) '(#t 1 1))",
        "medium",
        ["dedup"],
    ),
    (
        "egraph-add-enode!",
        "Add child `x`, then add parent `(f x)` manually; return whether parent registration includes the new class ID.",
        "(let ([eg (make-egraph)]) (let ([x (egraph-add-term! eg 'x)]) (let* ([parent-id (egraph-add-enode! eg (make-enode 'f (vector x)))] [parents (eclass-get-parents (egraph-classes eg) x)]) (not (not (memv parent-id parents))))))",
        "(equal? (let ([eg (make-egraph)]) (let ([x (egraph-add-term! eg 'x)]) (let* ([parent-id (egraph-add-enode! eg (make-enode 'f (vector x)))] [parents (eclass-get-parents (egraph-classes eg) x)]) (not (not (memv parent-id parents)))))) #t)",
        "hard",
        ["parents"],
    ),
    (
        "egraph-add-enode!",
        "Insert two distinct nullary enodes and return class count.",
        "(let ([eg (make-egraph)]) (egraph-add-enode! eg (make-enode 'x (vector))) (egraph-add-enode! eg (make-enode 'y (vector))) (egraph-class-count eg))",
        "(equal? (let ([eg (make-egraph)]) (egraph-add-enode! eg (make-enode 'x (vector))) (egraph-add-enode! eg (make-enode 'y (vector))) (egraph-class-count eg)) 2)",
        "easy",
        ["basic"],
    ),
    (
        "egraph-add-enode!",
        "After merging `x` and `y`, add `(f x)` and `(f y)` manually; return whether both IDs are equivalent.",
        "(let ([eg (make-egraph)]) (let ([x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-saturate-rebuild! eg) (let ([id1 (egraph-add-enode! eg (make-enode 'f (vector x)))] [id2 (egraph-add-enode! eg (make-enode 'f (vector y)))]) (= (egraph-find eg id1) (egraph-find eg id2)))))",
        "(equal? (let ([eg (make-egraph)]) (let ([x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-saturate-rebuild! eg) (let ([id1 (egraph-add-enode! eg (make-enode 'f (vector x)))] [id2 (egraph-add-enode! eg (make-enode 'f (vector y)))]) (= (egraph-find eg id1) (egraph-find eg id2))))) #t)",
        "hard",
        ["canonical"],
    ),
    (
        "egraph-merge!",
        "Build `(f x)` and `(f y)`, merge `x` with `y`, and return `(list classes merges dirty>0)`.",
        "(let ([eg (make-egraph)]) (let* ([x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)] [fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))]) (egraph-merge! eg x y) (list (egraph-class-count eg) (egraph-stat-merges eg) (> (length (egraph-dirty eg)) 0))))",
        "(equal? (let ([eg (make-egraph)]) (let* ([x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)] [fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))]) (egraph-merge! eg x y) (list (egraph-class-count eg) (egraph-stat-merges eg) (> (length (egraph-dirty eg)) 0)))) '(3 1 #t))",
        "medium",
        ["stats"],
    ),
    (
        "egraph-merge!",
        "Run the same merge twice and return `(list class-count merge-count)`.",
        "(let ([eg (make-egraph)]) (let ([x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-merge! eg x y) (list (egraph-class-count eg) (egraph-stat-merges eg))))",
        "(equal? (let ([eg (make-egraph)]) (let ([x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-merge! eg x y) (list (egraph-class-count eg) (egraph-stat-merges eg)))) '(1 1))",
        "medium",
        ["idempotence"],
    ),
    (
        "egraph-merge!",
        "Create `a`, `b`, `c`, merge through `b`, and return final class count.",
        "(let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg 'a)] [b (egraph-add-term! eg 'b)] [c (egraph-add-term! eg 'c)]) (egraph-merge! eg a b) (egraph-merge! eg b c) (egraph-class-count eg)))",
        "(equal? (let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg 'a)] [b (egraph-add-term! eg 'b)] [c (egraph-add-term! eg 'c)]) (egraph-merge! eg a b) (egraph-merge! eg b c) (egraph-class-count eg))) 1)",
        "hard",
        ["transitivity"],
    ),
    (
        "egraph-merge!",
        "Show congruence closure by merging `x`/`y` and testing roots of `(f x)` and `(f y)` after rebuild.",
        "(let ([eg (make-egraph)]) (let ([fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))] [x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-saturate-rebuild! eg) (= (egraph-find eg fx) (egraph-find eg fy))))",
        "(equal? (let ([eg (make-egraph)]) (let ([fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))] [x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-saturate-rebuild! eg) (= (egraph-find eg fx) (egraph-find eg fy)))) #t)",
        "hard",
        ["congruence"],
    ),
    (
        "egraph-rebuild!",
        "After merging `x` and `y`, call `egraph-rebuild!` once and return `(list processed>0 fx=fy)`.",
        "(let ([eg (make-egraph)]) (let ([fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))] [x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (let ([processed (egraph-rebuild! eg)]) (list (> processed 0) (= (egraph-find eg fx) (egraph-find eg fy))))))",
        "(equal? (let ([eg (make-egraph)]) (let ([fx (egraph-add-term! eg '(f x))] [fy (egraph-add-term! eg '(f y))] [x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (let ([processed (egraph-rebuild! eg)]) (list (> processed 0) (= (egraph-find eg fx) (egraph-find eg fy)))))) '(#t #t))",
        "hard",
        ["single-pass"],
    ),
    (
        "egraph-rebuild!",
        "Exercise non-root dirty IDs (`a=b`, then `b=c`) and return whether all `(f _)` terms become equivalent.",
        "(let ([eg (make-egraph)]) (let ([fa (egraph-add-term! eg '(f a))] [fb (egraph-add-term! eg '(f b))] [fc (egraph-add-term! eg '(f c))] [a (egraph-add-term! eg 'a)] [b (egraph-add-term! eg 'b)] [c (egraph-add-term! eg 'c)]) (egraph-merge! eg a b) (egraph-merge! eg b c) (egraph-saturate-rebuild! eg) (egraph-rebuild! eg) (and (= (egraph-find eg fa) (egraph-find eg fb)) (= (egraph-find eg fb) (egraph-find eg fc)))))",
        "(equal? (let ([eg (make-egraph)]) (let ([fa (egraph-add-term! eg '(f a))] [fb (egraph-add-term! eg '(f b))] [fc (egraph-add-term! eg '(f c))] [a (egraph-add-term! eg 'a)] [b (egraph-add-term! eg 'b)] [c (egraph-add-term! eg 'c)]) (egraph-merge! eg a b) (egraph-merge! eg b c) (egraph-saturate-rebuild! eg) (egraph-rebuild! eg) (and (= (egraph-find eg fa) (egraph-find eg fb)) (= (egraph-find eg fb) (egraph-find eg fc))))) #t)",
        "hard",
        ["non-root-dirty"],
    ),
    (
        "egraph-rebuild!",
        "Build `(g (f x))` and `(g (f y))`, merge `x`/`y`, saturate rebuild, and return whether roots match.",
        "(let ([eg (make-egraph)]) (let ([gfx (egraph-add-term! eg '(g (f x)))] [gfy (egraph-add-term! eg '(g (f y)))] [x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-saturate-rebuild! eg) (egraph-rebuild! eg) (= (egraph-find eg gfx) (egraph-find eg gfy))))",
        "(equal? (let ([eg (make-egraph)]) (let ([gfx (egraph-add-term! eg '(g (f x)))] [gfy (egraph-add-term! eg '(g (f y)))] [x (egraph-add-term! eg 'x)] [y (egraph-add-term! eg 'y)]) (egraph-merge! eg x y) (egraph-saturate-rebuild! eg) (egraph-rebuild! eg) (= (egraph-find eg gfx) (egraph-find eg gfy)))) #t)",
        "hard",
        ["nested"],
    ),
    (
        "egraph-rebuild!",
        "On a clean graph after saturation, call `egraph-rebuild!` and return the processed count.",
        "(let ([eg (make-egraph)]) (egraph-add-term! eg 'x) (egraph-saturate-rebuild! eg) (egraph-rebuild! eg))",
        "(equal? (let ([eg (make-egraph)]) (egraph-add-term! eg 'x) (egraph-saturate-rebuild! eg) (egraph-rebuild! eg)) 0)",
        "medium",
        ["fixpoint"],
    ),
    (
        "egraph-add-term!",
        "Add `(+ x y)` with `egraph-add-term!` and return `(list class-count node-count)`.",
        "(let ([eg (make-egraph)]) (egraph-add-term! eg '(+ x y)) (list (egraph-class-count eg) (egraph-node-count eg)))",
        "(equal? (let ([eg (make-egraph)]) (egraph-add-term! eg '(+ x y)) (list (egraph-class-count eg) (egraph-node-count eg))) '(3 3))",
        "medium",
        ["structure"],
    ),
    (
        "egraph-add-term!",
        "Insert `(+ x x)` and return class count to confirm subterm sharing.",
        "(let ([eg (make-egraph)]) (egraph-add-term! eg '(+ x x)) (egraph-class-count eg))",
        "(equal? (let ([eg (make-egraph)]) (egraph-add-term! eg '(+ x x)) (egraph-class-count eg)) 2)",
        "medium",
        ["sharing"],
    ),
    (
        "egraph-add-term!",
        "Add nested term `(+ (* a b) c)` and return resulting class count.",
        "(let ([eg (make-egraph)]) (egraph-add-term! eg '(+ (* a b) c)) (egraph-class-count eg))",
        "(equal? (let ([eg (make-egraph)]) (egraph-add-term! eg '(+ (* a b) c)) (egraph-class-count eg)) 5)",
        "hard",
        ["nested"],
    ),
    (
        "egraph-add-term!",
        "Insert `(concat \"hello\" \" \" \"world\")` twice and return `(list class-count same-root?)`.",
        "(let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg '(concat \"hello\" \" \" \"world\"))] [b (egraph-add-term! eg '(concat \"hello\" \" \" \"world\"))]) (list (egraph-class-count eg) (= (egraph-find eg a) (egraph-find eg b)))))",
        "(equal? (let ([eg (make-egraph)]) (let ([a (egraph-add-term! eg '(concat \"hello\" \" \" \"world\"))] [b (egraph-add-term! eg '(concat \"hello\" \" \" \"world\"))]) (list (egraph-class-count eg) (= (egraph-find eg a) (egraph-find eg b))))) '(4 #t))",
        "hard",
        ["literals"],
    ),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

if sum(1 for s in samples if s["family"] == "composition") != 32:
    raise ValueError("composition family must contain exactly 32 samples")


# -----------------------------------------------------------------------------
# Dataset-level QA checks
# -----------------------------------------------------------------------------
enforce_quality(samples)

if len(samples) != 80:
    raise ValueError(f"expected 80 samples, got {len(samples)}")

family_sizes = Counter(str(s["family"]) for s in samples)
expected_family_sizes = {
    "spec_to_code": 16,
    "translation": 16,
    "bugfix": 16,
    "composition": 32,
}
if dict(family_sizes) != expected_family_sizes:
    raise ValueError(f"family split mismatch: {dict(family_sizes)}")


# -----------------------------------------------------------------------------
# Split train/eval
# -----------------------------------------------------------------------------
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


# Re-run QA on split rows
for row in train_rows + eval_rows:
    if str(row["split"]) not in {"train", "eval"}:
        raise ValueError(f"invalid split in row {row['id']}")

enforce_quality(train_rows + eval_rows)


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
    "qa": {
        "composition_source_function_check": "passed",
        "non_tautological_verify_check": "passed",
        "bugfix_strength_check": "passed",
        "prompt_ground_truth_alignment_check": "passed",
    },
}

SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
