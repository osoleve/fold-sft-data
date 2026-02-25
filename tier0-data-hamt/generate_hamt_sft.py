#!/usr/bin/env python3
"""Generate SFT samples for lattice/data/hamt.ss."""

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

SOURCE_MODULE = "lattice/data/hamt.ss"
SOURCE_TEST = "lattice/data/test-hamt.ss"

GLOBAL_DEFS = [
    "(define *hamt-bits* 5)",
    "(define *hamt-mask* #b11111)",
    "(define *hamt-hash-mask* #x7FFFFFFF)",
    "(define hamt-empty 'hamt-empty)",
]

DEFS: Dict[str, str] = {
    "hamt-empty?": """(define (hamt-empty? x)
  (eq? x 'hamt-empty))""",
    "hamt-leaf?": """(define (hamt-leaf? x)
  (and (pair? x) (eq? (car x) 'hamt-leaf)))""",
    "hamt-collision?": """(define (hamt-collision? x)
  (and (pair? x) (eq? (car x) 'hamt-collision)))""",
    "hamt-node?": """(define (hamt-node? x)
  (and (pair? x) (eq? (car x) 'hamt-node)))""",
    "make-hamt-leaf": """(define (make-hamt-leaf hash key value)
  (list 'hamt-leaf hash key value))""",
    "make-hamt-collision": """(define (make-hamt-collision hash entries)
  (list 'hamt-collision hash entries))""",
    "make-hamt-node": """(define (make-hamt-node bitmap children)
  (cons 'hamt-node (cons bitmap children)))""",
    "hamt-leaf-hash": """(define (hamt-leaf-hash n)
  (cadr n))""",
    "hamt-leaf-key": """(define (hamt-leaf-key n)
  (caddr n))""",
    "hamt-leaf-value": """(define (hamt-leaf-value n)
  (cadddr n))""",
    "hamt-collision-hash": """(define (hamt-collision-hash n)
  (cadr n))""",
    "hamt-collision-entries": """(define (hamt-collision-entries n)
  (caddr n))""",
    "hamt-node-bitmap": """(define (hamt-node-bitmap n)
  (cadr n))""",
    "hamt-node-children": """(define (hamt-node-children n)
  (cddr n))""",
    "hash-fragment": """(define (hash-fragment shift hash)
  (fxlogand (fxarithmetic-shift-right hash shift) *hamt-mask*))""",
    "bit-pos": """(define (bit-pos fragment)
  (fxarithmetic-shift-left 1 fragment))""",
    "bitmap-index": """(define (bitmap-index bitmap bit)
  (fxbit-count (fxlogand bitmap (fx- bit 1))))""",
    "bitmap-has?": """(define (bitmap-has? bitmap bit)
  (not (fxzero? (fxlogand bitmap bit))))""",
    "list-set": """(define (list-set lst idx val)
  (let loop ([l lst] [i 0] [acc '()])
    (cond
      [(null? l) (reverse acc)]
      [(= i idx) (loop (cdr l) (+ i 1) (cons val acc))]
      [else (loop (cdr l) (+ i 1) (cons (car l) acc))])))""",
    "list-insert": """(define (list-insert lst idx val)
  (let loop ([l lst] [i 0] [acc '()])
    (cond
      [(= i idx) (append (reverse (cons val acc)) l)]
      [(null? l) (reverse (cons val acc))]
      [else (loop (cdr l) (+ i 1) (cons (car l) acc))])))""",
    "list-remove": """(define (list-remove lst idx)
  (let loop ([l lst] [i 0] [acc '()])
    (cond
      [(null? l) (reverse acc)]
      [(= i idx) (append (reverse acc) (cdr l))]
      [else (loop (cdr l) (+ i 1) (cons (car l) acc))])))""",
    "hamt-hash-key": """(define (hamt-hash-key key)
  (fxlogand (equal-hash key) *hamt-hash-mask*))""",
    "hamt-lookup": """(define (hamt-lookup key hamt)
  (let ([hash (hamt-hash-key key)])
    (hamt-lookup-hash hash key hamt 0)))""",
    "hamt-lookup-hash": """(define (hamt-lookup-hash hash key node shift)
  (cond
    [(hamt-empty? node) #f]
    [(hamt-leaf? node)
     (if (equal? key (hamt-leaf-key node))
         (hamt-leaf-value node)
         #f)]
    [(hamt-collision? node)
     (let ([pair (assoc key (hamt-collision-entries node))])
       (if pair (cdr pair) #f))]
    [(hamt-node? node)
     (let* ([frag (hash-fragment shift hash)]
            [bit (bit-pos frag)]
            [bitmap (hamt-node-bitmap node)])
       (if (bitmap-has? bitmap bit)
           (let ([idx (bitmap-index bitmap bit)])
             (hamt-lookup-hash hash key
                               (list-ref (hamt-node-children node) idx)
                               (fx+ shift *hamt-bits*)))
           #f))]
    [else #f]))""",
    "hamt-has-key?": """(define (hamt-has-key? key hamt)
  (let ([sentinel (list 'not-found)])
    (not (eq? sentinel (hamt-lookup-or key hamt sentinel)))))""",
    "hamt-lookup-or": """(define (hamt-lookup-or key hamt default)
  (let ([hash (hamt-hash-key key)])
    (hamt-lookup-or-hash hash key hamt 0 default)))""",
    "hamt-lookup-or-hash": """(define (hamt-lookup-or-hash hash key node shift default)
  (cond
    [(hamt-empty? node) default]
    [(hamt-leaf? node)
     (if (equal? key (hamt-leaf-key node))
         (hamt-leaf-value node)
         default)]
    [(hamt-collision? node)
     (let ([pair (assoc key (hamt-collision-entries node))])
       (if pair (cdr pair) default))]
    [(hamt-node? node)
     (let* ([frag (hash-fragment shift hash)]
            [bit (bit-pos frag)]
            [bitmap (hamt-node-bitmap node)])
       (if (bitmap-has? bitmap bit)
           (let ([idx (bitmap-index bitmap bit)])
             (hamt-lookup-or-hash hash key
                                  (list-ref (hamt-node-children node) idx)
                                  (fx+ shift *hamt-bits*) default))
           default))]
    [else default]))""",
    "hamt-assoc": """(define (hamt-assoc key value hamt)
  (let ([hash (hamt-hash-key key)])
    (hamt-assoc-hash hash key value hamt 0)))""",
    "hamt-assoc-hash": """(define (hamt-assoc-hash hash key value node shift)
  (cond
    [(hamt-empty? node)
     (make-hamt-leaf hash key value)]

    [(hamt-leaf? node)
     (let ([existing-hash (hamt-leaf-hash node)]
           [existing-key (hamt-leaf-key node)])
       (cond
         [(equal? key existing-key)
          (make-hamt-leaf hash key value)]
         [(= hash existing-hash)
          (make-hamt-collision hash
            (list (cons key value)
                  (cons existing-key (hamt-leaf-value node))))]
         [else
          (create-branch existing-hash node hash
                         (make-hamt-leaf hash key value)
                         shift)]))]

    [(hamt-collision? node)
     (if (= hash (hamt-collision-hash node))
         (make-hamt-collision hash
           (collision-assoc key value (hamt-collision-entries node)))
         (create-branch (hamt-collision-hash node) node
                        hash (make-hamt-leaf hash key value)
                        shift))]

    [(hamt-node? node)
     (let* ([frag (hash-fragment shift hash)]
            [bit (bit-pos frag)]
            [bitmap (hamt-node-bitmap node)]
            [children (hamt-node-children node)])
       (if (bitmap-has? bitmap bit)
           (let* ([idx (bitmap-index bitmap bit)]
                  [child (list-ref children idx)]
                  [new-child (hamt-assoc-hash hash key value child
                                              (fx+ shift *hamt-bits*))])
             (make-hamt-node bitmap (list-set children idx new-child)))
           (let* ([idx (bitmap-index bitmap bit)]
                  [new-bitmap (fxlogor bitmap bit)])
             (make-hamt-node new-bitmap
                             (list-insert children idx
                                          (make-hamt-leaf hash key value))))))]

    [else (make-hamt-leaf hash key value)]))""",
    "create-branch": """(define (create-branch hash1 node1 hash2 node2 shift)
  (let* ([frag1 (hash-fragment shift hash1)]
         [frag2 (hash-fragment shift hash2)]
         [bit1 (bit-pos frag1)]
         [bit2 (bit-pos frag2)])
    (if (= frag1 frag2)
        (let ([child (create-branch hash1 node1 hash2 node2
                                    (fx+ shift *hamt-bits*))])
          (make-hamt-node bit1 (list child)))
        (if (< frag1 frag2)
            (make-hamt-node (fxlogor bit1 bit2) (list node1 node2))
            (make-hamt-node (fxlogor bit1 bit2) (list node2 node1))))))""",
    "collision-assoc": """(define (collision-assoc key value entries)
  (let loop ([es entries] [acc '()] [found #f])
    (cond
      [(null? es)
       (if found
           (reverse acc)
           (reverse (cons (cons key value) acc)))]
      [(equal? key (caar es))
       (loop (cdr es) (cons (cons key value) acc) #t)]
      [else
       (loop (cdr es) (cons (car es) acc) found)])))""",
    "hamt-dissoc": """(define (hamt-dissoc key hamt)
  (let ([hash (hamt-hash-key key)])
    (hamt-dissoc-hash hash key hamt 0)))""",
    "hamt-dissoc-hash": """(define (hamt-dissoc-hash hash key node shift)
  (cond
    [(hamt-empty? node) hamt-empty]

    [(hamt-leaf? node)
     (if (equal? key (hamt-leaf-key node))
         hamt-empty
         node)]

    [(hamt-collision? node)
     (if (= hash (hamt-collision-hash node))
         (let ([new-entries (collision-dissoc key (hamt-collision-entries node))])
           (cond
             [(null? new-entries) hamt-empty]
             [(null? (cdr new-entries))
              (make-hamt-leaf hash (caar new-entries) (cdar new-entries))]
             [else
              (make-hamt-collision hash new-entries)]))
         node)]

    [(hamt-node? node)
     (let* ([frag (hash-fragment shift hash)]
            [bit (bit-pos frag)]
            [bitmap (hamt-node-bitmap node)]
            [children (hamt-node-children node)])
       (if (bitmap-has? bitmap bit)
           (let* ([idx (bitmap-index bitmap bit)]
                  [child (list-ref children idx)]
                  [new-child (hamt-dissoc-hash hash key child
                                               (fx+ shift *hamt-bits*))])
             (cond
               [(hamt-empty? new-child)
                (let ([new-bitmap (fxlogand bitmap (fxlognot bit))])
                  (if (fxzero? new-bitmap)
                      hamt-empty
                      (let ([new-children (list-remove children idx)])
                        (if (and (null? (cdr new-children))
                                 (not (hamt-node? (car new-children))))
                            (car new-children)
                            (make-hamt-node new-bitmap new-children)))))]
               [(and (not (hamt-node? new-child))
                     (= 1 (length children)))
                new-child]
               [else
                (make-hamt-node bitmap (list-set children idx new-child))]))
           node))]

    [else node]))""",
    "collision-dissoc": """(define (collision-dissoc key entries)
  (let loop ([es entries] [acc '()])
    (cond
      [(null? es) (reverse acc)]
      [(equal? key (caar es)) (loop (cdr es) acc)]
      [else (loop (cdr es) (cons (car es) acc))])))""",
    "hamt-fold": """(define (hamt-fold f init hamt)
  (cond
    [(hamt-empty? hamt) init]
    [(hamt-leaf? hamt)
     (f init (hamt-leaf-key hamt) (hamt-leaf-value hamt))]
    [(hamt-collision? hamt)
     (fold-left (lambda (acc pair) (f acc (car pair) (cdr pair)))
                init (hamt-collision-entries hamt))]
    [(hamt-node? hamt)
     (fold-left (lambda (acc child) (hamt-fold f acc child))
                init (hamt-node-children hamt))]
    [else init]))""",
    "hamt-size": """(define (hamt-size hamt)
  (hamt-fold (lambda (acc k v) (+ acc 1)) 0 hamt))""",
    "hamt-keys": """(define (hamt-keys hamt)
  (hamt-fold (lambda (acc k v) (cons k acc)) '() hamt))""",
    "hamt-values": """(define (hamt-values hamt)
  (hamt-fold (lambda (acc k v) (cons v acc)) '() hamt))""",
    "hamt-entries": """(define (hamt-entries hamt)
  (hamt-fold (lambda (acc k v) (cons (cons k v) acc)) '() hamt))""",
    "hamt-map-values": """(define (hamt-map-values f hamt)
  (cond
    [(hamt-empty? hamt) hamt-empty]
    [(hamt-leaf? hamt)
     (make-hamt-leaf (hamt-leaf-hash hamt)
                     (hamt-leaf-key hamt)
                     (f (hamt-leaf-value hamt)))]
    [(hamt-collision? hamt)
     (make-hamt-collision
       (hamt-collision-hash hamt)
       (map (lambda (pair) (cons (car pair) (f (cdr pair))))
            (hamt-collision-entries hamt)))]
    [(hamt-node? hamt)
     (make-hamt-node
       (hamt-node-bitmap hamt)
       (map (lambda (child) (hamt-map-values f child))
            (hamt-node-children hamt)))]
    [else hamt]))""",
    "hamt-filter": """(define (hamt-filter pred hamt)
  (hamt-fold (lambda (acc k v)
               (if (pred k v) (hamt-assoc k v acc) acc))
             hamt-empty hamt))""",
    "hamt-merge": """(define (hamt-merge h1 h2)
  (hamt-fold (lambda (acc k v) (hamt-assoc k v acc))
             h1 h2))""",
    "hamt-merge-with": """(define (hamt-merge-with f h1 h2)
  (hamt-fold (lambda (acc k v)
               (if (hamt-has-key? k acc)
                   (hamt-assoc k (f (hamt-lookup k acc) v) acc)
                   (hamt-assoc k v acc)))
             h1 h2))""",
    "dict->hamt": """(define (dict->hamt dict)
  (fold-left (lambda (acc pair)
               (hamt-assoc (car pair) (cdr pair) acc))
             hamt-empty dict))""",
    "alist->hamt": """(define (alist->hamt alist)
  (dict->hamt alist))""",
}

DEPENDS: Dict[str, List[str]] = {
    "hamt-empty?": [],
    "hamt-leaf?": [],
    "hamt-collision?": [],
    "hamt-node?": [],
    "make-hamt-leaf": [],
    "make-hamt-collision": [],
    "make-hamt-node": [],
    "hamt-leaf-hash": [],
    "hamt-leaf-key": [],
    "hamt-leaf-value": [],
    "hamt-collision-hash": [],
    "hamt-collision-entries": [],
    "hamt-node-bitmap": [],
    "hamt-node-children": [],
    "hash-fragment": [],
    "bit-pos": [],
    "bitmap-index": [],
    "bitmap-has?": [],
    "list-set": [],
    "list-insert": [],
    "list-remove": [],
    "hamt-hash-key": [],
    "hamt-lookup": ["hamt-hash-key", "hamt-lookup-hash"],
    "hamt-lookup-hash": [
        "hamt-empty?",
        "hamt-leaf?",
        "hamt-collision?",
        "hamt-node?",
        "hamt-leaf-key",
        "hamt-leaf-value",
        "hamt-collision-entries",
        "hash-fragment",
        "bit-pos",
        "hamt-node-bitmap",
        "bitmap-has?",
        "bitmap-index",
        "hamt-node-children",
    ],
    "hamt-has-key?": ["hamt-lookup-or"],
    "hamt-lookup-or": ["hamt-hash-key", "hamt-lookup-or-hash"],
    "hamt-lookup-or-hash": [
        "hamt-empty?",
        "hamt-leaf?",
        "hamt-collision?",
        "hamt-node?",
        "hamt-leaf-key",
        "hamt-leaf-value",
        "hamt-collision-entries",
        "hash-fragment",
        "bit-pos",
        "hamt-node-bitmap",
        "bitmap-has?",
        "bitmap-index",
        "hamt-node-children",
    ],
    "hamt-assoc": ["hamt-hash-key", "hamt-assoc-hash"],
    "hamt-assoc-hash": [
        "hamt-empty?",
        "hamt-leaf?",
        "hamt-collision?",
        "hamt-node?",
        "make-hamt-leaf",
        "hamt-leaf-hash",
        "hamt-leaf-key",
        "hamt-leaf-value",
        "make-hamt-collision",
        "create-branch",
        "hamt-collision-hash",
        "collision-assoc",
        "hamt-collision-entries",
        "hash-fragment",
        "bit-pos",
        "hamt-node-bitmap",
        "hamt-node-children",
        "bitmap-has?",
        "bitmap-index",
        "list-set",
        "list-insert",
        "make-hamt-node",
    ],
    "create-branch": ["hash-fragment", "bit-pos", "make-hamt-node"],
    "collision-assoc": [],
    "hamt-dissoc": ["hamt-hash-key", "hamt-dissoc-hash"],
    "hamt-dissoc-hash": [
        "hamt-empty?",
        "hamt-leaf?",
        "hamt-collision?",
        "hamt-node?",
        "hamt-leaf-key",
        "hamt-collision-hash",
        "hamt-collision-entries",
        "collision-dissoc",
        "make-hamt-leaf",
        "make-hamt-collision",
        "hash-fragment",
        "bit-pos",
        "hamt-node-bitmap",
        "hamt-node-children",
        "bitmap-has?",
        "bitmap-index",
        "list-remove",
        "make-hamt-node",
        "list-set",
    ],
    "collision-dissoc": [],
    "hamt-fold": [
        "hamt-empty?",
        "hamt-leaf?",
        "hamt-collision?",
        "hamt-node?",
        "hamt-leaf-key",
        "hamt-leaf-value",
        "hamt-collision-entries",
        "hamt-node-children",
    ],
    "hamt-size": ["hamt-fold"],
    "hamt-keys": ["hamt-fold"],
    "hamt-values": ["hamt-fold"],
    "hamt-entries": ["hamt-fold"],
    "hamt-map-values": [
        "hamt-empty?",
        "hamt-leaf?",
        "hamt-collision?",
        "hamt-node?",
        "make-hamt-leaf",
        "hamt-leaf-hash",
        "hamt-leaf-key",
        "hamt-leaf-value",
        "make-hamt-collision",
        "hamt-collision-hash",
        "hamt-collision-entries",
        "make-hamt-node",
        "hamt-node-bitmap",
        "hamt-node-children",
    ],
    "hamt-filter": ["hamt-fold", "hamt-assoc"],
    "hamt-merge": ["hamt-fold", "hamt-assoc"],
    "hamt-merge-with": ["hamt-fold", "hamt-has-key?", "hamt-assoc", "hamt-lookup"],
    "dict->hamt": ["hamt-assoc"],
    "alist->hamt": ["dict->hamt"],
}

CORE_FUNCTIONS = [
    "hamt-empty?",
    "hamt-lookup",
    "hamt-has-key?",
    "hamt-assoc",
    "hamt-dissoc",
    "hamt-size",
    "hamt-merge-with",
    "alist->hamt",
]

FUNCTION_ORDER = [
    "hamt-empty?",
    "hamt-lookup",
    "hamt-lookup-or",
    "hamt-has-key?",
    "hamt-assoc",
    "hamt-dissoc",
    "hamt-fold",
    "hamt-size",
    "hamt-keys",
    "hamt-values",
    "hamt-entries",
    "hamt-map-values",
    "hamt-filter",
    "hamt-merge",
    "hamt-merge-with",
    "dict->hamt",
    "alist->hamt",
]

FUNCTION_SPECS = {
    "hamt-empty?": "Return #t only for the singleton empty HAMT marker `'hamt-empty`.",
    "hamt-lookup": "Lookup key in HAMT and return its value, or #f when absent.",
    "hamt-lookup-or": "Lookup key and return default when key is absent.",
    "hamt-has-key?": "Return key existence even when value is #f (must not use lookup truthiness).",
    "hamt-assoc": "Insert/update key -> value in persistent HAMT and return new structure.",
    "hamt-dissoc": "Remove key from HAMT and return updated HAMT without mutating input.",
    "hamt-fold": "Fold all key/value pairs with accumulator function `(f acc key value)`.",
    "hamt-size": "Count the number of key-value pairs in HAMT.",
    "hamt-keys": "Collect all keys contained in the HAMT.",
    "hamt-values": "Collect all values contained in the HAMT.",
    "hamt-entries": "Collect key/value pairs as an association list.",
    "hamt-map-values": "Map a value transform over HAMT values while preserving keys/shape.",
    "hamt-filter": "Keep only entries where predicate `(pred key value)` is true.",
    "hamt-merge": "Merge two HAMTs with right-biased conflict resolution (h2 wins).",
    "hamt-merge-with": "Merge two HAMTs and resolve conflicts with function `(f old new)`.",
    "dict->hamt": "Convert an association-list dictionary into a HAMT.",
    "alist->hamt": "Convert association list to HAMT; duplicate keys keep the last value.",
}

SKELETONS = {
    "hamt-empty?": """(define (hamt-empty? x)
  ;; TODO: detect the canonical empty HAMT marker
  <TODO>)""",
    "hamt-lookup": """(define (hamt-lookup key hamt)
  ;; TODO: hash key and delegate to recursive lookup
  <TODO>)""",
    "hamt-lookup-or": """(define (hamt-lookup-or key hamt default)
  ;; TODO: hash key and delegate to recursive lookup-or
  <TODO>)""",
    "hamt-has-key?": """(define (hamt-has-key? key hamt)
  ;; TODO: distinguish missing key from key mapped to #f
  <TODO>)""",
    "hamt-assoc": """(define (hamt-assoc key value hamt)
  ;; TODO: hash key and insert/update persistently
  <TODO>)""",
    "hamt-dissoc": """(define (hamt-dissoc key hamt)
  ;; TODO: hash key and remove persistently
  <TODO>)""",
    "hamt-fold": """(define (hamt-fold f init hamt)
  ;; TODO: fold over leaves/collisions/nodes
  <TODO>)""",
    "hamt-size": """(define (hamt-size hamt)
  ;; TODO: fold through HAMT and count entries
  <TODO>)""",
    "hamt-keys": """(define (hamt-keys hamt)
  ;; TODO: collect all keys with hamt-fold
  <TODO>)""",
    "hamt-values": """(define (hamt-values hamt)
  ;; TODO: collect all values with hamt-fold
  <TODO>)""",
    "hamt-entries": """(define (hamt-entries hamt)
  ;; TODO: collect key/value pairs with hamt-fold
  <TODO>)""",
    "hamt-map-values": """(define (hamt-map-values f hamt)
  ;; TODO: recursively map value transformer across HAMT nodes
  <TODO>)""",
    "hamt-filter": """(define (hamt-filter pred hamt)
  ;; TODO: filter entries by predicate on key and value
  <TODO>)""",
    "hamt-merge": """(define (hamt-merge h1 h2)
  ;; TODO: right-biased merge using hamt-fold and hamt-assoc
  <TODO>)""",
    "hamt-merge-with": """(define (hamt-merge-with f h1 h2)
  ;; TODO: merge h2 into h1 using conflict resolver (old,new)
  <TODO>)""",
    "dict->hamt": """(define (dict->hamt dict)
  ;; TODO: fold association pairs into a HAMT
  <TODO>)""",
    "alist->hamt": """(define (alist->hamt alist)
  ;; TODO: build HAMT from association list
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "hamt-empty?": "(and (hamt-empty? hamt-empty) (not (hamt-empty? (hamt-assoc 'a 1 hamt-empty))) (hamt-empty? (hamt-dissoc 'a (hamt-assoc 'a 1 hamt-empty))))",
    "hamt-lookup": "(let ([h (hamt-assoc 'a 1 (hamt-assoc 'b 2 hamt-empty))]) (and (= (hamt-lookup 'a h) 1) (= (hamt-lookup 'b h) 2) (equal? (hamt-lookup 'missing h) #f)))",
    "hamt-lookup-or": "(let ([h (hamt-assoc 'k #f (hamt-assoc 'a 1 hamt-empty))]) (and (equal? (hamt-lookup-or 'a h 'missing) 1) (equal? (hamt-lookup-or 'k h 'missing) #f) (equal? (hamt-lookup-or 'z h 'missing) 'missing)))",
    "hamt-has-key?": "(let ([h (hamt-assoc 'k #f hamt-empty)]) (and (hamt-has-key? 'k h) (not (hamt-has-key? 'x h))))",
    "hamt-assoc": "(let* ([h0 hamt-empty] [h1 (hamt-assoc 'a 1 h0)] [h2 (hamt-assoc 'b 2 h1)] [h3 (hamt-assoc 'a 9 h2)]) (and (= (hamt-size h2) 2) (= (hamt-size h3) 2) (= (hamt-lookup 'a h3) 9) (= (hamt-lookup 'b h3) 2)))",
    "hamt-dissoc": "(let* ([h (alist->hamt '((a . 1) (b . 2) (c . 3)))] [h2 (hamt-dissoc 'b h)] [h3 (hamt-dissoc 'z h2)]) (and (= (hamt-size h2) 2) (not (hamt-has-key? 'b h2)) (= (hamt-size h3) 2) (= (hamt-lookup 'a h3) 1)))",
    "hamt-fold": "(let* ([h (alist->hamt '((a . 1) (b . 2) (c . 3)))] [sum (hamt-fold (lambda (acc k v) (+ acc v)) 0 h)] [count (hamt-fold (lambda (acc k v) (+ acc 1)) 0 h)]) (and (= sum 6) (= count 3)))",
    "hamt-size": "(let* ([h1 (alist->hamt '((a . 1) (b . 2) (c . 3)))] [h2 (hamt-assoc 'b 9 h1)]) (and (= (hamt-size hamt-empty) 0) (= (hamt-size h1) 3) (= (hamt-size h2) 3)))",
    "hamt-keys": "(let* ([h (alist->hamt '((a . 1) (b . 2) (c . 3)))] [ks (hamt-keys h)]) (and (= (length ks) 3) (not (not (member 'a ks))) (not (not (member 'b ks))) (not (not (member 'c ks)))))",
    "hamt-values": "(let* ([h (alist->hamt '((a . 1) (b . 2) (c . 3)))] [vs (hamt-values h)]) (and (= (length vs) 3) (not (not (member 1 vs))) (not (not (member 2 vs))) (not (not (member 3 vs)))))",
    "hamt-entries": "(let* ([h (alist->hamt '((a . 1) (b . 2)))] [es (hamt-entries h)] [h2 (alist->hamt es)]) (and (= (hamt-size h2) 2) (= (hamt-lookup 'a h2) 1) (= (hamt-lookup 'b h2) 2)))",
    "hamt-map-values": "(let* ([h (alist->hamt '((a . 1) (b . 2)))] [m (hamt-map-values (lambda (v) (+ v 10)) h)]) (and (= (hamt-lookup 'a m) 11) (= (hamt-lookup 'b m) 12) (= (hamt-size m) 2)))",
    "hamt-filter": "(let* ([h (alist->hamt '((a . 1) (b . 2) (c . 3)))] [f (hamt-filter (lambda (k v) (> v 1)) h)]) (and (= (hamt-size f) 2) (not (hamt-has-key? 'a f)) (= (hamt-lookup 'b f) 2) (= (hamt-lookup 'c f) 3)))",
    "hamt-merge": "(let* ([h1 (alist->hamt '((a . 1) (b . 2)))] [h2 (alist->hamt '((b . 9) (c . 3)))] [m (hamt-merge h1 h2)]) (and (= (hamt-size m) 3) (= (hamt-lookup 'a m) 1) (= (hamt-lookup 'b m) 9) (= (hamt-lookup 'c m) 3)))",
    "hamt-merge-with": "(let* ([h1 (alist->hamt '((a . 1) (b . #f)))] [h2 (alist->hamt '((b . 10) (c . 3)))] [m (hamt-merge-with (lambda (old new) (if (equal? old #f) 'had-false (+ old new))) h1 h2)]) (and (= (hamt-lookup 'a m) 1) (equal? (hamt-lookup 'b m) 'had-false) (= (hamt-lookup 'c m) 3) (= (hamt-size m) 3)))",
    "dict->hamt": "(let ([h (dict->hamt '((x . 1) (y . 2) (x . 7)))]) (and (= (hamt-size h) 2) (= (hamt-lookup 'x h) 7) (= (hamt-lookup 'y h) 2)))",
    "alist->hamt": "(let ([h (alist->hamt '((x . 1) (y . 2) (x . 7)))]) (and (= (hamt-size h) 2) (= (hamt-lookup 'x h) 7) (= (hamt-lookup 'y h) 2)))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = CORE_FUNCTIONS

PYTHON_SNIPPETS = {
    "hamt-empty?": "def hamt_empty_p(x):\n    return x == 'hamt-empty'",
    "hamt-lookup": "def hamt_lookup(key, hamt):\n    h = hamt_hash_key(key)\n    return hamt_lookup_hash(h, key, hamt, 0)",
    "hamt-has-key?": "def hamt_has_key(key, hamt):\n    sentinel = ['not-found']\n    return sentinel is not hamt_lookup_or(key, hamt, sentinel)",
    "hamt-assoc": "def hamt_assoc(key, value, hamt):\n    h = hamt_hash_key(key)\n    return hamt_assoc_hash(h, key, value, hamt, 0)",
    "hamt-dissoc": "def hamt_dissoc(key, hamt):\n    h = hamt_hash_key(key)\n    return hamt_dissoc_hash(h, key, hamt, 0)",
    "hamt-size": "def hamt_size(hamt):\n    return hamt_fold(lambda acc, k, v: acc + 1, 0, hamt)",
    "hamt-merge-with": "def hamt_merge_with(f, h1, h2):\n    def step(acc, k, v):\n        if hamt_has_key(k, acc):\n            return hamt_assoc(k, f(hamt_lookup(k, acc), v), acc)\n        return hamt_assoc(k, v, acc)\n    return hamt_fold(step, h1, h2)",
    "alist->hamt": "def alist_to_hamt(alist):\n    return dict_to_hamt(alist)",
}

CHEZ_SNIPPETS = {
    "hamt-empty?": "(define (hamt-empty0? x)\n  (eq? x 'hamt-empty))",
    "hamt-lookup": "(define (hamt-get key h)\n  (let ((hash (hamt-hash-key key)))\n    (hamt-lookup-hash hash key h 0)))",
    "hamt-has-key?": "(define (hamt-has0? k h)\n  (let ((sentinel (list 'not-found)))\n    (not (eq? sentinel (hamt-lookup-or k h sentinel)))))",
    "hamt-assoc": "(define (hamt-put k v h)\n  (let ((hash (hamt-hash-key k)))\n    (hamt-assoc-hash hash k v h 0)))",
    "hamt-dissoc": "(define (hamt-del k h)\n  (let ((hash (hamt-hash-key k)))\n    (hamt-dissoc-hash hash k h 0)))",
    "hamt-size": "(define (hamt-count h)\n  (hamt-fold (lambda (acc k v) (+ acc 1)) 0 h))",
    "hamt-merge-with": "(define (hamt-merge0 f h1 h2)\n  (hamt-fold (lambda (acc k v)\n               (if (hamt-has-key? k acc)\n                   (hamt-assoc k (f (hamt-lookup k acc) v) acc)\n                   (hamt-assoc k v acc)))\n             h1 h2))",
    "alist->hamt": "(define (alist->hamt0 al)\n  (dict->hamt al))",
}

BUGGY_CASES = [
    {
        "fn": "hamt-empty?",
        "buggy": "(define (hamt-empty? x)\n  (eq? x '()))",
        "note": "Empty HAMT marker is `'hamt-empty`, not the empty list.",
    },
    {
        "fn": "hamt-empty?",
        "buggy": "(define (hamt-empty? x)\n  (not (eq? x 'hamt-empty)))",
        "note": "Predicate polarity is reversed.",
    },
    {
        "fn": "hamt-lookup",
        "buggy": "(define (hamt-lookup key hamt)\n  (let ([hash (hamt-hash-key hamt)])\n    (hamt-lookup-hash hash key hamt 0)))",
        "note": "Lookup hash must be derived from key, not from hamt object.",
    },
    {
        "fn": "hamt-lookup",
        "buggy": "(define (hamt-lookup key hamt)\n  (let ([hash (hamt-hash-key key)])\n    (hamt-lookup-hash hash key hamt *hamt-bits*)))",
        "note": "Recursive lookup must start at shift 0.",
    },
    {
        "fn": "hamt-has-key?",
        "buggy": "(define (hamt-has-key? key hamt)\n  (if (hamt-lookup key hamt) #t #f))",
        "note": "Keys mapped to #f must still count as present.",
    },
    {
        "fn": "hamt-has-key?",
        "buggy": "(define (hamt-has-key? key hamt)\n  (not (eq? #f (hamt-lookup-or key hamt #f))))",
        "note": "Using #f as sentinel breaks presence checks for keys with #f values.",
    },
    {
        "fn": "hamt-assoc",
        "buggy": "(define (hamt-assoc key value hamt)\n  (let ([hash (hamt-hash-key value)])\n    (hamt-assoc-hash hash key value hamt 0)))",
        "note": "Assoc hash must be computed from key, not value.",
    },
    {
        "fn": "hamt-assoc",
        "buggy": "(define (hamt-assoc key value hamt)\n  (let ([hash (hamt-hash-key key)])\n    (hamt-assoc-hash hash key value hamt *hamt-bits*)))",
        "note": "Insertion recursion must start at shift 0.",
    },
    {
        "fn": "hamt-dissoc",
        "buggy": "(define (hamt-dissoc key hamt)\n  hamt)",
        "note": "Function must actually remove key when present.",
    },
    {
        "fn": "hamt-dissoc",
        "buggy": "(define (hamt-dissoc key hamt)\n  (let ([hash (hamt-hash-key key)])\n    (hamt-dissoc-hash hash key hamt *hamt-bits*)))",
        "note": "Deletion recursion must start at shift 0.",
    },
    {
        "fn": "hamt-size",
        "buggy": "(define (hamt-size hamt)\n  (hamt-fold (lambda (acc k v) (+ acc v)) 0 hamt))",
        "note": "Size should count entries, not sum values.",
    },
    {
        "fn": "hamt-size",
        "buggy": "(define (hamt-size hamt)\n  (hamt-fold (lambda (acc k v) (+ acc 1)) 1 hamt))",
        "note": "Initial count must be 0.",
    },
    {
        "fn": "hamt-merge-with",
        "buggy": "(define (hamt-merge-with f h1 h2)\n  (hamt-fold (lambda (acc k v)\n               (if (hamt-has-key? k acc)\n                   (hamt-assoc k (f v (hamt-lookup k acc)) acc)\n                   (hamt-assoc k v acc)))\n             h1 h2))",
        "note": "Conflict resolver argument order must be (old-value, new-value).",
    },
    {
        "fn": "hamt-merge-with",
        "buggy": "(define (hamt-merge-with f h1 h2)\n  (hamt-fold (lambda (acc k v)\n               (if (hamt-lookup k acc)\n                   (hamt-assoc k (f (hamt-lookup k acc) v) acc)\n                   (hamt-assoc k v acc)))\n             h1 h2))",
        "note": "Presence test must use hamt-has-key? so #f values are treated as existing keys.",
    },
    {
        "fn": "alist->hamt",
        "buggy": "(define (alist->hamt alist)\n  hamt-empty)",
        "note": "Must convert all association-list entries into HAMT bindings.",
    },
    {
        "fn": "alist->hamt",
        "buggy": "(define (alist->hamt alist)\n  (dict->hamt (map (lambda (pair) (cons (cdr pair) (car pair))) alist)))",
        "note": "Conversion should preserve key/value orientation.",
    },
]

DIFFICULTY = {
    "hamt-empty?": "easy",
    "hamt-lookup": "medium",
    "hamt-lookup-or": "medium",
    "hamt-has-key?": "hard",
    "hamt-assoc": "hard",
    "hamt-dissoc": "hard",
    "hamt-fold": "hard",
    "hamt-size": "easy",
    "hamt-keys": "easy",
    "hamt-values": "easy",
    "hamt-entries": "medium",
    "hamt-map-values": "hard",
    "hamt-filter": "hard",
    "hamt-merge": "medium",
    "hamt-merge-with": "hard",
    "dict->hamt": "medium",
    "alist->hamt": "medium",
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
    sid = f"hamt_{family}_{family_counter[family]:03d}"
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
    return [name for name in DEFS.keys() if name != fn and name in tokens]


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
    parts = GLOBAL_DEFS + [DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
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
        prompt=f"""Implement this function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "hamt", "spec-to-code", fn],
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
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "hamt", "skeleton-completion", fn],
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
Preserve behavior exactly, including key-absence semantics.

Target function name: `{fn}`

```python
{PYTHON_SNIPPETS[fn]}
```

Return only the Scheme definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "hamt", "python-to-scheme", fn],
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
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "hamt", "chez-to-fold", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix
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
        tags=["tier0", "data", "hamt", "bugfix", fn],
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
        tags=["tier0", "data", "hamt", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # hamt-empty?
    ("hamt-empty?", "Check whether canonical hamt-empty is empty.", "(hamt-empty? hamt-empty)", "(equal? (hamt-empty? hamt-empty) #t)", "easy", ["direct"]),
    ("hamt-empty?", "Associate key 'a then check emptiness.", "(hamt-empty? (hamt-assoc 'a 1 hamt-empty))", "(equal? (hamt-empty? (hamt-assoc 'a 1 hamt-empty)) #f)", "easy", ["edge-case"]),
    ("hamt-empty?", "Associate then dissoc same key and check emptiness.", "(hamt-empty? (hamt-dissoc 'a (hamt-assoc 'a 1 hamt-empty)))", "(equal? (hamt-empty? (hamt-dissoc 'a (hamt-assoc 'a 1 hamt-empty))) #t)", "medium", ["integration"]),
    ("hamt-empty?", "Convert empty alist and check hamt-empty?.", "(hamt-empty? (alist->hamt '()))", "(equal? (hamt-empty? (alist->hamt '())) #t)", "easy", ["integration"]),

    # hamt-lookup
    ("hamt-lookup", "Lookup key 'b after inserting a and b.", "(hamt-lookup 'b (hamt-assoc 'b 2 (hamt-assoc 'a 1 hamt-empty)))", "(equal? (hamt-lookup 'b (hamt-assoc 'b 2 (hamt-assoc 'a 1 hamt-empty))) 2)", "medium", ["direct"]),
    ("hamt-lookup", "Lookup missing key and return #f.", "(hamt-lookup 'missing (hamt-assoc 'a 1 hamt-empty))", "(equal? (hamt-lookup 'missing (hamt-assoc 'a 1 hamt-empty)) #f)", "easy", ["edge-case"]),
    ("hamt-lookup", "Lookup numeric key 1 and string key \"1\" in same HAMT.", "(list (hamt-lookup 1 (alist->hamt '((1 . int) (\"1\" . str)))) (hamt-lookup \"1\" (alist->hamt '((1 . int) (\"1\" . str)))))", "(equal? (list (hamt-lookup 1 (alist->hamt '((1 . int) (\"1\" . str)))) (hamt-lookup \"1\" (alist->hamt '((1 . int) (\"1\" . str))))) '(int str))", "medium", ["property"]),
    ("hamt-lookup", "Merge two HAMTs with + and lookup merged 'k.", "(hamt-lookup 'k (hamt-merge-with + (alist->hamt '((k . 2))) (alist->hamt '((k . 3)))))", "(equal? (hamt-lookup 'k (hamt-merge-with + (alist->hamt '((k . 2))) (alist->hamt '((k . 3))))) 5)", "hard", ["integration"]),

    # hamt-has-key?
    ("hamt-has-key?", "Check that key 'a exists after insertion.", "(hamt-has-key? 'a (hamt-assoc 'a 7 hamt-empty))", "(equal? (hamt-has-key? 'a (hamt-assoc 'a 7 hamt-empty)) #t)", "easy", ["direct"]),
    ("hamt-has-key?", "Check missing key returns #f.", "(hamt-has-key? 'x hamt-empty)", "(equal? (hamt-has-key? 'x hamt-empty) #f)", "easy", ["edge-case"]),
    ("hamt-has-key?", "Key mapped to #f should still be reported present.", "(hamt-has-key? 'k (hamt-assoc 'k #f hamt-empty))", "(equal? (hamt-has-key? 'k (hamt-assoc 'k #f hamt-empty)) #t)", "hard", ["property"]),
    ("hamt-has-key?", "Insert then dissoc key 'a and test presence.", "(hamt-has-key? 'a (hamt-dissoc 'a (hamt-assoc 'a 1 hamt-empty)))", "(equal? (hamt-has-key? 'a (hamt-dissoc 'a (hamt-assoc 'a 1 hamt-empty))) #f)", "medium", ["integration"]),

    # hamt-assoc
    ("hamt-assoc", "Associate key 'a -> 1 then lookup 'a.", "(hamt-lookup 'a (hamt-assoc 'a 1 hamt-empty))", "(equal? (hamt-lookup 'a (hamt-assoc 'a 1 hamt-empty)) 1)", "medium", ["direct"]),
    ("hamt-assoc", "Update existing key and keep size at 1.", "(hamt-size (hamt-assoc 'a 9 (hamt-assoc 'a 1 hamt-empty)))", "(equal? (hamt-size (hamt-assoc 'a 9 (hamt-assoc 'a 1 hamt-empty))) 1)", "medium", ["property"]),
    ("hamt-assoc", "Insert three distinct keys and return size.", "(hamt-size (hamt-assoc 'c 3 (hamt-assoc 'b 2 (hamt-assoc 'a 1 hamt-empty))))", "(equal? (hamt-size (hamt-assoc 'c 3 (hamt-assoc 'b 2 (hamt-assoc 'a 1 hamt-empty)))) 3)", "medium", ["direct"]),
    ("hamt-assoc", "Associate key with #f then verify hamt-has-key? is true.", "(hamt-has-key? 'z (hamt-assoc 'z #f hamt-empty))", "(equal? (hamt-has-key? 'z (hamt-assoc 'z #f hamt-empty)) #t)", "hard", ["integration"]),

    # hamt-dissoc
    ("hamt-dissoc", "Remove key 'b from ((a . 1) (b . 2)) and lookup 'b.", "(hamt-lookup 'b (hamt-dissoc 'b (alist->hamt '((a . 1) (b . 2)))))", "(equal? (hamt-lookup 'b (hamt-dissoc 'b (alist->hamt '((a . 1) (b . 2))))) #f)", "hard", ["direct"]),
    ("hamt-dissoc", "Dissoc missing key should preserve size.", "(hamt-size (hamt-dissoc 'x (alist->hamt '((a . 1) (b . 2)))))", "(equal? (hamt-size (hamt-dissoc 'x (alist->hamt '((a . 1) (b . 2))))) 2)", "medium", ["edge-case"]),
    ("hamt-dissoc", "Dissoc from empty should stay empty.", "(hamt-empty? (hamt-dissoc 'x hamt-empty))", "(equal? (hamt-empty? (hamt-dissoc 'x hamt-empty)) #t)", "easy", ["edge-case"]),
    ("hamt-dissoc", "Dissoc one key and ensure another key remains reachable.", "(hamt-lookup 'a (hamt-dissoc 'b (alist->hamt '((a . 1) (b . 2) (c . 3)))))", "(equal? (hamt-lookup 'a (hamt-dissoc 'b (alist->hamt '((a . 1) (b . 2) (c . 3))))) 1)", "hard", ["integration"]),

    # hamt-size
    ("hamt-size", "Return size of hamt-empty.", "(hamt-size hamt-empty)", "(equal? (hamt-size hamt-empty) 0)", "easy", ["direct"]),
    ("hamt-size", "Return size of alist-converted HAMT with three keys.", "(hamt-size (alist->hamt '((a . 1) (b . 2) (c . 3))))", "(equal? (hamt-size (alist->hamt '((a . 1) (b . 2) (c . 3)))) 3)", "easy", ["direct"]),
    ("hamt-size", "Updating existing key should not increase size.", "(hamt-size (hamt-assoc 'a 2 (hamt-assoc 'a 1 hamt-empty)))", "(equal? (hamt-size (hamt-assoc 'a 2 (hamt-assoc 'a 1 hamt-empty))) 1)", "medium", ["property"]),
    ("hamt-size", "Merge two HAMTs and return resulting size.", "(hamt-size (hamt-merge-with + (alist->hamt '((a . 1) (b . 2))) (alist->hamt '((b . 4) (c . 8)))))", "(equal? (hamt-size (hamt-merge-with + (alist->hamt '((a . 1) (b . 2))) (alist->hamt '((b . 4) (c . 8))))) 3)", "hard", ["integration"]),

    # hamt-merge-with
    ("hamt-merge-with", "Merge disjoint HAMTs and lookup new key 'c.", "(hamt-lookup 'c (hamt-merge-with + (alist->hamt '((a . 1))) (alist->hamt '((c . 3)))))", "(equal? (hamt-lookup 'c (hamt-merge-with + (alist->hamt '((a . 1))) (alist->hamt '((c . 3))))) 3)", "medium", ["direct"]),
    ("hamt-merge-with", "Merge with conflict on 'b using +.", "(hamt-lookup 'b (hamt-merge-with + (alist->hamt '((a . 1) (b . 2))) (alist->hamt '((b . 5)))))", "(equal? (hamt-lookup 'b (hamt-merge-with + (alist->hamt '((a . 1) (b . 2))) (alist->hamt '((b . 5))))) 7)", "hard", ["direct"]),
    ("hamt-merge-with", "Merge with resolver returning (old new) pair order for key 'k.", "(hamt-lookup 'k (hamt-merge-with (lambda (old new) (list old new)) (alist->hamt '((k . 1))) (alist->hamt '((k . 10)))))", "(equal? (hamt-lookup 'k (hamt-merge-with (lambda (old new) (list old new)) (alist->hamt '((k . 1))) (alist->hamt '((k . 10))))) '(1 10))", "hard", ["property"]),
    ("hamt-merge-with", "Merge where old value is #f and resolver should still run.", "(hamt-lookup 'k (hamt-merge-with (lambda (old new) 'hit) (alist->hamt '((k . #f))) (alist->hamt '((k . 99)))))", "(equal? (hamt-lookup 'k (hamt-merge-with (lambda (old new) 'hit) (alist->hamt '((k . #f))) (alist->hamt '((k . 99))))) 'hit)", "hard", ["property"]),

    # alist->hamt
    ("alist->hamt", "Convert alist ((a . 1) (b . 2)) and lookup 'a.", "(hamt-lookup 'a (alist->hamt '((a . 1) (b . 2))))", "(equal? (hamt-lookup 'a (alist->hamt '((a . 1) (b . 2)))) 1)", "medium", ["direct"]),
    ("alist->hamt", "Duplicate key in alist should keep last value.", "(hamt-lookup 'a (alist->hamt '((a . 1) (a . 7))))", "(equal? (hamt-lookup 'a (alist->hamt '((a . 1) (a . 7)))) 7)", "medium", ["property"]),
    ("alist->hamt", "Empty alist conversion should be empty HAMT.", "(hamt-empty? (alist->hamt '()))", "(equal? (hamt-empty? (alist->hamt '())) #t)", "easy", ["edge-case"]),
    ("alist->hamt", "Convert mixed-key alist and return size.", "(hamt-size (alist->hamt '((1 . one) (\"1\" . str) ((1 2) . pair))))", "(equal? (hamt-size (alist->hamt '((1 . one) (\"1\" . str) ((1 2) . pair)))) 3)", "medium", ["integration"]),
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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


write_jsonl(ALL_PATH, train_rows + eval_rows)
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
