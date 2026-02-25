#!/usr/bin/env python3
"""Generate Tier-1 query aho-corasick SFT samples for lattice/query/aho-corasick.ss."""

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

SOURCE_MODULE = "lattice/query/aho-corasick.ss"
SOURCE_TEST = "lattice/query/test-aho-corasick.ss"

DEFS: Dict[str, str] = {
    "build-trie": """(define (build-trie patterns)
  (let* ([total-chars (fold-left (lambda (acc p) (+ acc (string-length p))) 0 patterns)]
         [capacity (max 64 (+ total-chars 1))]
         [states (make-vector capacity #f)]
         [size 1])
    (vector-set! states 0 (make-state 0))
    (let loop-patterns ([patterns patterns]
                        [next-id 1])
      (if (null? patterns)
          (vector-copy states 0 size)
          (let ([new-id (insert-pattern-mut! (car patterns) states
                                             (lambda () size)
                                             (lambda (new-size) (set! size new-size))
                                             next-id)])
            (loop-patterns (cdr patterns) new-id))))))""",
    "insert-pattern-mut!": """(define (insert-pattern-mut! pattern states get-size set-size! next-id)
  (let loop-chars ([chars (string->list pattern)]
                   [sid 0]
                   [next-id next-id])
    (if (null? chars)
        (let* ([state (vector-ref states sid)]
               [new-output (set-add pattern (ac-state-output state))]
               [new-state (make-ac-state sid
                                         (ac-state-trans state)
                                         new-output
                                         (ac-state-fail state))])
          (vector-set! states sid new-state)
          next-id)
        (let* ([ch (car chars)]
               [state (vector-ref states sid)]
               [trans (ac-state-trans state)]
               [next (dict-lookup ch trans)])
          (if next
              (loop-chars (cdr chars) next next-id)
              (let* ([new-state (make-state next-id)]
                     [new-trans (dict-assoc ch next-id trans)]
                     [updated-parent (make-ac-state sid
                                                    new-trans
                                                    (ac-state-output state)
                                                    (ac-state-fail state))])
                (vector-set! states next-id new-state)
                (set-size! (+ (get-size) 1))
                (vector-set! states sid updated-parent)
                (loop-chars (cdr chars) next-id (+ next-id 1))))))))""",
    "compute-failures": """(define (compute-failures states)
  (let* ([root (vector-ref states 0)]
         [children (dict-values (ac-state-trans root))]
         [init-q (fold-left (lambda (q child) (queue-enqueue child q))
                            queue-empty
                            children)])
    (bfs-mut! states init-q)
    states))""",
    "bfs-mut!": """(define (bfs-mut! states queue)
  (if (queue-empty? queue)
      (void)
      (let-values ([(q2 sid) (queue-dequeue queue)])
        (let* ([state (vector-ref states sid)]
               [trans (ac-state-trans state)])
          (let loop-trans ([keys (dict-keys trans)]
                           [q q2])
            (if (null? keys)
                (bfs-mut! states q)
                (let* ([ch (car keys)]
                       [child-id (dict-lookup ch trans)]
                       [fail-id (find-fail states sid ch)]
                       [child (vector-ref states child-id)]
                       [fail-state (vector-ref states fail-id)]
                       [new-output (set-union (ac-state-output child)
                                              (ac-state-output fail-state))]
                       [new-child (make-ac-state child-id
                                                 (ac-state-trans child)
                                                 new-output
                                                 fail-id)])
                  (vector-set! states child-id new-child)
                  (loop-trans (cdr keys)
                              (queue-enqueue child-id q)))))))))""",
    "find-fail": """(define (find-fail states sid ch)
  (let* ([state (vector-ref states sid)]
         [fail-id (ac-state-fail state)])
    (if (= fail-id 0)
        (let ([next (dict-lookup ch (ac-state-trans (vector-ref states 0)))])
          (if next next 0))
        (let ([next (dict-lookup ch (ac-state-trans (vector-ref states fail-id)))])
          (if next
              next
              (find-fail states fail-id ch))))))""",
    "get-next": """(define (get-next automaton sid ch)
  (let ([next (dict-lookup ch (ac-state-trans (vector-ref automaton sid)))])
    (if next
        next
        (if (= sid 0)
            0
            (get-next automaton (ac-state-fail (vector-ref automaton sid)) ch)))))""",
    "make-automaton": """(define (make-automaton patterns)
  (compute-failures (build-trie patterns)))""",
    "search": """(define (search automaton text)
  (let loop ([chars (string->list text)]
             [pos 0]
             [sid 0]
             [matches '()])
    (if (null? chars)
        (reverse matches)
        (let* ([ch (car chars)]
               [next-sid (get-next automaton sid ch)]
               [state (vector-ref automaton next-sid)]
               [outputs (set->list (ac-state-output state))])
          (loop (cdr chars)
                (+ pos 1)
                next-sid
                (append (map (lambda (p)
                               (cons (- (+ pos 1) (string-length p)) p))
                             outputs)
                        matches))))))""",
}

FUNCTION_ORDER = [
    "build-trie",
    "insert-pattern-mut!",
    "compute-failures",
    "bfs-mut!",
    "find-fail",
    "get-next",
    "make-automaton",
    "search",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

FUNCTION_SPECS = {
    "build-trie": "Build a compact mutable trie vector from pattern strings and return only the used prefix.",
    "insert-pattern-mut!": "Insert one pattern into mutable trie state, creating nodes and terminal outputs as needed.",
    "compute-failures": "Compute failure links and output unions over trie states via BFS traversal.",
    "bfs-mut!": "Run in-place BFS over queued states, filling child failure links and inherited output sets.",
    "find-fail": "Resolve fallback transition target for a char from a state's failure chain.",
    "get-next": "Advance automaton state for one character using transitions plus recursive failure fallback.",
    "make-automaton": "Construct complete Aho-Corasick automaton by combining trie build and failure-link pass.",
    "search": "Scan text with automaton and return all (start-index . matched-pattern) results.",
}

SKELETONS = {
    "build-trie": """(define (build-trie patterns)
  ;; TODO: pre-allocate mutable states, insert patterns, and return used prefix
  <TODO>)""",
    "insert-pattern-mut!": """(define (insert-pattern-mut! pattern states get-size set-size! next-id)
  ;; TODO: mutate trie for one pattern and return next available state id
  <TODO>)""",
    "compute-failures": """(define (compute-failures states)
  ;; TODO: seed BFS queue from root children and run bfs-mut!
  <TODO>)""",
    "bfs-mut!": """(define (bfs-mut! states queue)
  ;; TODO: BFS through transitions, assign fail links, and union outputs
  <TODO>)""",
    "find-fail": """(define (find-fail states sid ch)
  ;; TODO: walk failure chain to locate transition for ch
  <TODO>)""",
    "get-next": """(define (get-next automaton sid ch)
  ;; TODO: transition on ch with recursive fallback to failure links
  <TODO>)""",
    "make-automaton": """(define (make-automaton patterns)
  ;; TODO: build trie then compute failure links
  <TODO>)""",
    "search": """(define (search automaton text)
  ;; TODO: scan text and emit (start-index . pattern) matches
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "build-trie": """(let* ([states (build-trie '("he" "she"))]
       [root (vector-ref states 0)]
       [h (dict-lookup #\\h (ac-state-trans root))]
       [s (dict-lookup #\\s (ac-state-trans root))]
       [he (and h (dict-lookup #\\e (ac-state-trans (vector-ref states h))))]
       [sh (and s (dict-lookup #\\h (ac-state-trans (vector-ref states s))))]
       [she (and sh (dict-lookup #\\e (ac-state-trans (vector-ref states sh))))])
  (and (= (vector-length states) 6)
       h s he she
       (set-member? "he" (ac-state-output (vector-ref states he)))
       (set-member? "she" (ac-state-output (vector-ref states she)))
       (not (set-member? "he" (ac-state-output root)))))""",
    "insert-pattern-mut!": """(let* ([states (make-vector 16 #f)]
       [size 1])
  (vector-set! states 0 (make-state 0))
  (let* ([next-id (insert-pattern-mut! "ab" states
                                       (lambda () size)
                                       (lambda (new-size) (set! size new-size))
                                       1)]
         [root (vector-ref states 0)]
         [s1 (dict-lookup #\\a (ac-state-trans root))]
         [s2 (and s1 (dict-lookup #\\b (ac-state-trans (vector-ref states s1))))])
    (and (= next-id 3)
         (= size 3)
         s1 s2
         (set-member? "ab" (ac-state-output (vector-ref states s2))))))""",
    "compute-failures": """(let* ([states (build-trie '("he" "she" "his" "hers"))]
       [done (compute-failures states)]
       [root (vector-ref done 0)]
       [h (dict-lookup #\\h (ac-state-trans root))]
       [s (dict-lookup #\\s (ac-state-trans root))]
       [he (and h (dict-lookup #\\e (ac-state-trans (vector-ref done h))))]
       [sh (and s (dict-lookup #\\h (ac-state-trans (vector-ref done s))))]
       [she (and sh (dict-lookup #\\e (ac-state-trans (vector-ref done sh))))]
       [she-state (and she (vector-ref done she))])
  (and h s he sh she she-state
       (= (ac-state-fail she-state) he)
       (set-member? "she" (ac-state-output she-state))
       (set-member? "he" (ac-state-output she-state))))""",
    "bfs-mut!": """(let* ([states (build-trie '("he" "she"))]
       [root (vector-ref states 0)]
       [children (dict-values (ac-state-trans root))]
       [q (fold-left (lambda (qq child) (queue-enqueue child qq)) queue-empty children)])
  (bfs-mut! states q)
  (let* ([h (dict-lookup #\\h (ac-state-trans root))]
         [s (dict-lookup #\\s (ac-state-trans root))]
         [he (and h (dict-lookup #\\e (ac-state-trans (vector-ref states h))))]
         [sh (and s (dict-lookup #\\h (ac-state-trans (vector-ref states s))))]
         [she (and sh (dict-lookup #\\e (ac-state-trans (vector-ref states sh))))]
         [she-state (and she (vector-ref states she))])
    (and h s he sh she she-state
         (= (ac-state-fail she-state) he)
         (set-member? "she" (ac-state-output she-state))
         (set-member? "he" (ac-state-output she-state)))))""",
    "find-fail": """(let* ([a (make-automaton '("he" "she" "hers"))]
       [root (vector-ref a 0)]
       [h (dict-lookup #\\h (ac-state-trans root))]
       [s (dict-lookup #\\s (ac-state-trans root))]
       [he (and h (dict-lookup #\\e (ac-state-trans (vector-ref a h))))]
       [sh (and s (dict-lookup #\\h (ac-state-trans (vector-ref a s))))]
       [ff1 (and sh (find-fail a sh #\\e))]
       [ff2 (and sh (find-fail a sh #\\x))]
       [ff3 (find-fail a 0 #\\h)])
  (and he sh ff1 ff2
       (= ff1 he)
       (= ff2 0)
       (= ff3 h)))""",
    "get-next": """(let* ([a (make-automaton '("ab" "bc"))]
       [root (vector-ref a 0)]
       [sa (dict-lookup #\\a (ac-state-trans root))]
       [sab (and sa (dict-lookup #\\b (ac-state-trans (vector-ref a sa))))]
       [sb (dict-lookup #\\b (ac-state-trans root))]
       [sbc (and sb (dict-lookup #\\c (ac-state-trans (vector-ref a sb))))]
       [n1 (and sab (get-next a sab #\\c))]
       [n2 (and sab (get-next a sab #\\x))])
  (and sab sbc n1 n2
       (= n1 sbc)
       (= n2 0)))""",
    "make-automaton": """(let* ([a (make-automaton '("he" "she" "his" "hers"))]
       [root (vector-ref a 0)]
       [h (dict-lookup #\\h (ac-state-trans root))]
       [s (dict-lookup #\\s (ac-state-trans root))]
       [he (and h (dict-lookup #\\e (ac-state-trans (vector-ref a h))))]
       [sh (and s (dict-lookup #\\h (ac-state-trans (vector-ref a s))))]
       [she (and sh (dict-lookup #\\e (ac-state-trans (vector-ref a sh))))]
       [matches (search a "ushers")])
  (and h s he sh she
       (= (ac-state-fail (vector-ref a she)) he)
       (= (length matches) 3)
       (if (member '(1 . "she") matches) #t #f)
       (if (member '(2 . "he") matches) #t #f)
       (if (member '(2 . "hers") matches) #t #f)))""",
    "search": """(let* ([a (make-automaton '("a" "aa" "aaa"))]
       [m (search a "aaaa")])
  (and (= (length m) 9)
       (= (car (car m)) 0)
       (if (member '(0 . "aaa") m) #t #f)
       (if (member '(1 . "aaa") m) #t #f)
       (if (member '(3 . "a") m) #t #f)))""",
}

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "build-trie": """def build_trie(patterns):
    total_chars = sum(len(p) for p in patterns)
    capacity = max(64, total_chars + 1)
    states = [None] * capacity
    size = 1
    states[0] = make_state(0)
    next_id = 1
    for pat in patterns:
        next_id = insert_pattern_mut(pat, states, lambda: size, lambda n: set_size(n), next_id)
    return states[:size]""",
    "insert-pattern-mut!": """def insert_pattern_mut(pattern, states, get_size, set_size, next_id):
    sid = 0
    for ch in pattern:
        state = states[sid]
        nxt = dict_lookup(ch, state.trans)
        if nxt is not None:
            sid = nxt
            continue
        states[next_id] = make_state(next_id)
        new_trans = dict_assoc(ch, next_id, state.trans)
        states[sid] = make_ac_state(sid, new_trans, state.output, state.fail)
        set_size(get_size() + 1)
        sid = next_id
        next_id += 1
    st = states[sid]
    states[sid] = make_ac_state(sid, st.trans, set_add(pattern, st.output), st.fail)
    return next_id""",
    "compute-failures": """def compute_failures(states):
    root = states[0]
    q = queue_empty
    for child in dict_values(root.trans):
        q = queue_enqueue(child, q)
    bfs_mut(states, q)
    return states""",
    "bfs-mut!": """def bfs_mut(states, queue):
    while not queue_empty_q(queue):
        queue, sid = queue_dequeue(queue)
        trans = states[sid].trans
        for ch in dict_keys(trans):
            child_id = dict_lookup(ch, trans)
            fail_id = find_fail(states, sid, ch)
            child = states[child_id]
            fail_state = states[fail_id]
            merged = set_union(child.output, fail_state.output)
            states[child_id] = make_ac_state(child_id, child.trans, merged, fail_id)
            queue = queue_enqueue(child_id, queue)""",
    "find-fail": """def find_fail(states, sid, ch):
    fail_id = states[sid].fail
    if fail_id == 0:
        nxt = dict_lookup(ch, states[0].trans)
        return nxt if nxt is not None else 0
    nxt = dict_lookup(ch, states[fail_id].trans)
    if nxt is not None:
        return nxt
    return find_fail(states, fail_id, ch)""",
    "get-next": """def get_next(automaton, sid, ch):
    nxt = dict_lookup(ch, automaton[sid].trans)
    if nxt is not None:
        return nxt
    if sid == 0:
        return 0
    return get_next(automaton, automaton[sid].fail, ch)""",
    "make-automaton": """def make_automaton(patterns):
    return compute_failures(build_trie(patterns))""",
    "search": """def search(automaton, text):
    sid = 0
    matches = []
    for pos, ch in enumerate(text):
        sid = get_next(automaton, sid, ch)
        outputs = set_to_list(automaton[sid].output)
        for pat in outputs:
            start = (pos + 1) - len(pat)
            matches.append((start, pat))
    return matches""",
}

CHEZ_SNIPPETS = {
    "build-trie": """(define (build-trie* patterns)
  (let* ([total (fold-left (lambda (acc p) (+ acc (string-length p))) 0 patterns)]
         [cap (max 64 (+ total 1))]
         [states (make-vector cap #f)]
         [size 1])
    (vector-set! states 0 (make-state 0))
    (let loop ([ps patterns] [next 1])
      (if (null? ps)
          (vector-copy states 0 size)
          (let ([n2 (insert-pattern-mut! (car ps) states
                                         (lambda () size)
                                         (lambda (v) (set! size v))
                                         next)])
            (loop (cdr ps) n2))))))""",
    "insert-pattern-mut!": """(define (insert-pattern-mut!* pattern states get-size set-size! next-id)
  (let loop ([chars (string->list pattern)] [sid 0] [nid next-id])
    (if (null? chars)
        (let* ([st (vector-ref states sid)]
               [out (set-add pattern (ac-state-output st))])
          (vector-set! states sid (make-ac-state sid (ac-state-trans st) out (ac-state-fail st)))
          nid)
        (let* ([ch (car chars)]
               [st (vector-ref states sid)]
               [next (dict-lookup ch (ac-state-trans st))])
          (if next
              (loop (cdr chars) next nid)
              (let* ([new (make-state nid)]
                     [trans2 (dict-assoc ch nid (ac-state-trans st))])
                (vector-set! states nid new)
                (set-size! (+ (get-size) 1))
                (vector-set! states sid (make-ac-state sid trans2 (ac-state-output st) (ac-state-fail st)))
                (loop (cdr chars) nid (+ nid 1))))))))""",
    "compute-failures": """(define (compute-failures* states)
  (let* ([root (vector-ref states 0)]
         [kids (dict-values (ac-state-trans root))]
         [q (fold-left (lambda (qq child) (queue-enqueue child qq)) queue-empty kids)])
    (bfs-mut! states q)
    states))""",
    "bfs-mut!": """(define (bfs-mut!* states queue)
  (if (queue-empty? queue)
      (void)
      (let-values ([(q2 sid) (queue-dequeue queue)])
        (let loop ([keys (dict-keys (ac-state-trans (vector-ref states sid)))] [q q2])
          (if (null? keys)
              (bfs-mut!* states q)
              (let* ([ch (car keys)]
                     [child-id (dict-lookup ch (ac-state-trans (vector-ref states sid)))]
                     [fail-id (find-fail states sid ch)]
                     [child (vector-ref states child-id)]
                     [fail-state (vector-ref states fail-id)]
                     [merged (set-union (ac-state-output child) (ac-state-output fail-state))])
                (vector-set! states child-id
                  (make-ac-state child-id (ac-state-trans child) merged fail-id))
                (loop (cdr keys) (queue-enqueue child-id q))))))))""",
    "find-fail": """(define (find-fail* states sid ch)
  (let ([fid (ac-state-fail (vector-ref states sid))])
    (if (= fid 0)
        (let ([next (dict-lookup ch (ac-state-trans (vector-ref states 0)))])
          (if next next 0))
        (let ([next (dict-lookup ch (ac-state-trans (vector-ref states fid)))])
          (if next next (find-fail* states fid ch))))))""",
    "get-next": """(define (get-next* automaton sid ch)
  (let ([next (dict-lookup ch (ac-state-trans (vector-ref automaton sid)))])
    (if next
        next
        (if (= sid 0)
            0
            (get-next* automaton (ac-state-fail (vector-ref automaton sid)) ch)))))""",
    "make-automaton": """(define (make-automaton* patterns)
  (compute-failures (build-trie patterns)))""",
    "search": """(define (search* automaton text)
  (let loop ([chars (string->list text)] [pos 0] [sid 0] [matches '()])
    (if (null? chars)
        (reverse matches)
        (let* ([ch (car chars)]
               [sid2 (get-next automaton sid ch)]
               [outs (set->list (ac-state-output (vector-ref automaton sid2)))])
          (loop (cdr chars)
                (+ pos 1)
                sid2
                (append (map (lambda (p) (cons (- (+ pos 1) (string-length p)) p)) outs)
                        matches))))))""",
}

BUGGY_CASES = [
    {
        "fn": "build-trie",
        "buggy": """(define (build-trie patterns)
  (let* ([total-chars (fold-left (lambda (acc p) (+ acc (string-length p))) 0 patterns)]
         [capacity (max 64 (+ total-chars 1))]
         [states (make-vector capacity #f)]
         [size 1])
    (vector-set! states 0 (make-state 0))
    (let loop-patterns ([patterns patterns]
                        [next-id 1])
      (if (null? patterns)
          states
          (let ([new-id (insert-pattern-mut! (car patterns) states
                                             (lambda () size)
                                             (lambda (new-size) (set! size new-size))
                                             next-id)])
            (loop-patterns (cdr patterns) new-id))))))""",
        "note": "Trie builder must return only the used vector prefix, not the full preallocated capacity.",
    },
    {
        "fn": "build-trie",
        "buggy": """(define (build-trie patterns)
  (let* ([total-chars (fold-left (lambda (acc p) (+ acc (string-length p))) 0 patterns)]
         [capacity (max 64 (+ total-chars 1))]
         [states (make-vector capacity #f)]
         [size 0])
    (vector-set! states 0 (make-state 0))
    (let loop-patterns ([patterns patterns]
                        [next-id 1])
      (if (null? patterns)
          (vector-copy states 0 size)
          (let ([new-id (insert-pattern-mut! (car patterns) states
                                             (lambda () size)
                                             (lambda (new-size) (set! size new-size))
                                             next-id)])
            (loop-patterns (cdr patterns) new-id))))))""",
        "note": "Trie size tracking must include root node; starting from zero truncates the resulting state vector.",
    },
    {
        "fn": "insert-pattern-mut!",
        "buggy": """(define (insert-pattern-mut! pattern states get-size set-size! next-id)
  (let loop-chars ([chars (string->list pattern)]
                   [sid 0]
                   [next-id next-id])
    (if (null? chars)
        next-id
        (let* ([ch (car chars)]
               [state (vector-ref states sid)]
               [trans (ac-state-trans state)]
               [next (dict-lookup ch trans)])
          (if next
              (loop-chars (cdr chars) next next-id)
              (let* ([new-state (make-state next-id)]
                     [new-trans (dict-assoc ch next-id trans)]
                     [updated-parent (make-ac-state sid
                                                    new-trans
                                                    (ac-state-output state)
                                                    (ac-state-fail state))])
                (vector-set! states next-id new-state)
                (set-size! (+ (get-size) 1))
                (vector-set! states sid updated-parent)
                (loop-chars (cdr chars) next-id (+ next-id 1))))))))""",
        "note": "Terminal states must record matched pattern strings in output sets.",
    },
    {
        "fn": "insert-pattern-mut!",
        "buggy": """(define (insert-pattern-mut! pattern states get-size set-size! next-id)
  (let loop-chars ([chars (string->list pattern)]
                   [sid 0]
                   [next-id next-id])
    (if (null? chars)
        (let* ([state (vector-ref states sid)]
               [new-output (set-add pattern (ac-state-output state))]
               [new-state (make-ac-state sid
                                         (ac-state-trans state)
                                         new-output
                                         (ac-state-fail state))])
          (vector-set! states sid new-state)
          next-id)
        (let* ([ch (car chars)]
               [state (vector-ref states sid)]
               [trans (ac-state-trans state)]
               [next (dict-lookup ch trans)])
          (if next
              (loop-chars (cdr chars) next next-id)
              (let* ([new-state (make-state next-id)]
                     [new-trans (dict-assoc ch next-id trans)]
                     [updated-parent (make-ac-state sid
                                                    new-trans
                                                    (ac-state-output state)
                                                    (ac-state-fail state))])
                (vector-set! states next-id new-state)
                (vector-set! states sid updated-parent)
                (loop-chars (cdr chars) next-id (+ next-id 1))))))))""",
        "note": "Insertion must update externally tracked trie size when allocating new states.",
    },
    {
        "fn": "compute-failures",
        "buggy": """(define (compute-failures states)
  states)""",
        "note": "Failure-link computation must run BFS; returning trie unchanged leaves fail links unset.",
    },
    {
        "fn": "compute-failures",
        "buggy": """(define (compute-failures states)
  (bfs-mut! states queue-empty)
  states)""",
        "note": "BFS must be seeded with root children; empty seed queue skips all failure propagation.",
    },
    {
        "fn": "bfs-mut!",
        "buggy": """(define (bfs-mut! states queue)
  (if (queue-empty? queue)
      (void)
      (let-values ([(q2 sid) (queue-dequeue queue)])
        (let* ([state (vector-ref states sid)]
               [trans (ac-state-trans state)])
          (let loop-trans ([keys (dict-keys trans)]
                           [q q2])
            (if (null? keys)
                (bfs-mut! states q)
                (let* ([ch (car keys)]
                       [child-id (dict-lookup ch trans)]
                       [fail-id (find-fail states sid ch)]
                       [child (vector-ref states child-id)]
                       [new-child (make-ac-state child-id
                                                 (ac-state-trans child)
                                                 (ac-state-output child)
                                                 fail-id)])
                  (vector-set! states child-id new-child)
                  (loop-trans (cdr keys)
                              (queue-enqueue child-id q)))))))))""",
        "note": "BFS propagation must union child outputs with fail-state outputs.",
    },
    {
        "fn": "bfs-mut!",
        "buggy": """(define (bfs-mut! states queue)
  (if (queue-empty? queue)
      (void)
      (let-values ([(q2 sid) (queue-dequeue queue)])
        (let* ([state (vector-ref states sid)]
               [trans (ac-state-trans state)])
          (let loop-trans ([keys (dict-keys trans)]
                           [q q2])
            (if (null? keys)
                (bfs-mut! states q)
                (let* ([ch (car keys)]
                       [child-id (dict-lookup ch trans)]
                       [child (vector-ref states child-id)]
                       [new-child (make-ac-state child-id
                                                 (ac-state-trans child)
                                                 (ac-state-output child)
                                                 0)])
                  (vector-set! states child-id new-child)
                  (loop-trans (cdr keys)
                              (queue-enqueue child-id q)))))))))""",
        "note": "BFS must compute real failure links from find-fail rather than forcing all fails to root.",
    },
    {
        "fn": "find-fail",
        "buggy": """(define (find-fail states sid ch)
  (let* ([state (vector-ref states sid)]
         [fail-id (ac-state-fail state)])
    (if (= fail-id 0)
        0
        (let ([next (dict-lookup ch (ac-state-trans (vector-ref states fail-id)))])
          (if next
              next
              (find-fail states fail-id ch))))))""",
        "note": "At root fallback, find-fail must still check root transitions before returning 0.",
    },
    {
        "fn": "find-fail",
        "buggy": """(define (find-fail states sid ch)
  (let* ([state (vector-ref states sid)]
         [fail-id (ac-state-fail state)])
    (if (= fail-id 0)
        (let ([next (dict-lookup ch (ac-state-trans (vector-ref states 0)))])
          (if next next 0))
        (let ([next (dict-lookup ch (ac-state-trans (vector-ref states fail-id)))])
          (if next
              next
              fail-id)))))""",
        "note": "If fail state lacks transition, search must continue recursively up the failure chain.",
    },
    {
        "fn": "get-next",
        "buggy": """(define (get-next automaton sid ch)
  (let ([next (dict-lookup ch (ac-state-trans (vector-ref automaton sid)))])
    (if next
        next
        0)))""",
        "note": "Single-step transition must recursively follow failure links, not jump directly to root.",
    },
    {
        "fn": "get-next",
        "buggy": """(define (get-next automaton sid ch)
  (let ([next (dict-lookup ch (ac-state-trans (vector-ref automaton sid)))])
    (if next
        next
        (if (= sid 0)
            0
            (ac-state-fail (vector-ref automaton sid))))))""",
        "note": "Fallback must recurse until a valid transition is found, not return immediate fail-id as next state.",
    },
    {
        "fn": "make-automaton",
        "buggy": """(define (make-automaton patterns)
  (build-trie patterns))""",
        "note": "Automaton construction must include failure-link computation after trie build.",
    },
    {
        "fn": "make-automaton",
        "buggy": """(define (make-automaton patterns)
  (compute-failures (build-trie (cdr patterns))))""",
        "note": "Automaton must include all patterns; dropping head pattern changes recognized matches.",
    },
    {
        "fn": "search",
        "buggy": """(define (search automaton text)
  (let loop ([chars (string->list text)]
             [pos 0]
             [sid 0]
             [matches '()])
    (if (null? chars)
        matches
        (let* ([ch (car chars)]
               [next-sid (get-next automaton sid ch)]
               [state (vector-ref automaton next-sid)]
               [outputs (set->list (ac-state-output state))])
          (loop (cdr chars)
                (+ pos 1)
                next-sid
                (append (map (lambda (p)
                               (cons (- (+ pos 1) (string-length p)) p))
                             outputs)
                        matches))))))""",
        "note": "Search should return matches in scan order; accumulated list must be reversed before returning.",
    },
    {
        "fn": "search",
        "buggy": """(define (search automaton text)
  (let loop ([chars (string->list text)]
             [pos 0]
             [sid 0]
             [matches '()])
    (if (null? chars)
        (reverse matches)
        (let* ([ch (car chars)]
               [next-sid (get-next automaton sid ch)]
               [state (vector-ref automaton next-sid)]
               [outputs (set->list (ac-state-output state))])
          (loop (cdr chars)
                (+ pos 1)
                next-sid
                (append (map (lambda (p)
                               (cons (+ pos 1) p))
                             outputs)
                        matches))))))""",
        "note": "Match offsets must subtract pattern length to emit start indices, not end positions.",
    },
]

DIFFICULTY = {
    "build-trie": "hard",
    "insert-pattern-mut!": "hard",
    "compute-failures": "medium",
    "bfs-mut!": "hard",
    "find-fail": "medium",
    "get-next": "medium",
    "make-automaton": "easy",
    "search": "medium",
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
    sid = f"query_aho_corasick_{family}_{family_counter[family]:03d}"
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
            ground_truth=ground_truth,
            available_functions=FUNCTION_ORDER,
        ),
        "ground_truth": ground_truth.strip(),
        "verify_expr": verify_expr.strip(),
        "tags": tags,
    }
    for key in REQUIRED_KEYS:
        if key not in sample:
            raise ValueError(f"missing key {key}")
    samples.append(sample)


def def_verify(fn: str) -> str:
    return VERIFY_BY_FUNCTION[fn].strip()


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement this Aho-Corasick function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "query", "aho-corasick", "spec-to-code", fn],
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
        tags=["tier1", "query", "aho-corasick", "skeleton-completion", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement `{fn}` from this Aho-Corasick contract.

Module: `{SOURCE_MODULE}`
Contract focus: {FUNCTION_SPECS[fn]}

Requirements:
1. Keep the exact function name/signature for `{fn}`.
2. Preserve trie/failure/output semantics.
3. Return only one complete function definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "query", "aho-corasick", "contract-implementation", fn],
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
        tags=["tier1", "query", "aho-corasick", "python-to-scheme", fn],
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
        tags=["tier1", "query", "aho-corasick", "chez-to-fold", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Translate this reference implementation into canonical Fold Scheme for `{fn}`.

Preserve observable matching behavior exactly.
Keep the target function name/signature as `{fn}`.
Return only the final Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "query", "aho-corasick", "reference-translation", fn],
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
        tags=["tier1", "query", "aho-corasick", "bugfix", fn],
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
    composition_prompt = (
        f"{prompt.rstrip()}\n\n"
        f"Ensure `{source_function}` is part of the composed solution."
    )
    add_sample(
        family="composition",
        category="usage",
        difficulty=difficulty,
        source_function=source_function,
        prompt=composition_prompt,
        ground_truth=ground_truth,
        verify_expr=verify_check.strip(),
        tags=["tier1", "query", "aho-corasick", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # build-trie
    {
        "fn": "build-trie",
        "prompt": "Build trie for ('he' 'she') and return state count.",
        "gt": "(vector-length (build-trie '(\"he\" \"she\")))",
        "verify": "(= (vector-length (build-trie '(\"he\" \"she\"))) 6)",
        "difficulty": "easy",
        "tags": ["size"],
    },
    {
        "fn": "build-trie",
        "prompt": "Build trie for ('a' 'aa') and return whether root has transition on #\\a.",
        "gt": "(let* ([states (build-trie '(\"a\" \"aa\"))] [root (vector-ref states 0)]) (if (dict-lookup #\\a (ac-state-trans root)) #t #f))",
        "verify": "(equal? (let* ([states (build-trie '(\"a\" \"aa\"))] [root (vector-ref states 0)]) (if (dict-lookup #\\a (ac-state-trans root)) #t #f)) #t)",
        "difficulty": "easy",
        "tags": ["root-transition"],
    },
    {
        "fn": "build-trie",
        "prompt": "Build trie for ('ab' 'ac') and return whether both terminal outputs are present.",
        "gt": "(let* ([states (build-trie '(\"ab\" \"ac\"))] [root (vector-ref states 0)] [a (dict-lookup #\\a (ac-state-trans root))] [ab (dict-lookup #\\b (ac-state-trans (vector-ref states a)))] [ac (dict-lookup #\\c (ac-state-trans (vector-ref states a)))]) (and (set-member? \"ab\" (ac-state-output (vector-ref states ab))) (set-member? \"ac\" (ac-state-output (vector-ref states ac)))))",
        "verify": "(equal? (let* ([states (build-trie '(\"ab\" \"ac\"))] [root (vector-ref states 0)] [a (dict-lookup #\\a (ac-state-trans root))] [ab (dict-lookup #\\b (ac-state-trans (vector-ref states a)))] [ac (dict-lookup #\\c (ac-state-trans (vector-ref states a)))]) (and (set-member? \"ab\" (ac-state-output (vector-ref states ab))) (set-member? \"ac\" (ac-state-output (vector-ref states ac))))) #t)",
        "difficulty": "medium",
        "tags": ["terminal-output"],
    },
    {
        "fn": "build-trie",
        "prompt": "Build trie for empty pattern list and return whether only root state exists.",
        "gt": "(let ([states (build-trie '())]) (= (vector-length states) 1))",
        "verify": "(equal? (let ([states (build-trie '())]) (= (vector-length states) 1)) #t)",
        "difficulty": "easy",
        "tags": ["empty"],
    },

    # insert-pattern-mut!
    {
        "fn": "insert-pattern-mut!",
        "prompt": "Insert pattern 'ab' into fresh mutable trie and return next-id.",
        "gt": "(let* ([states (make-vector 16 #f)] [size 1]) (vector-set! states 0 (make-state 0)) (insert-pattern-mut! \"ab\" states (lambda () size) (lambda (n) (set! size n)) 1))",
        "verify": "(= (let* ([states (make-vector 16 #f)] [size 1]) (vector-set! states 0 (make-state 0)) (insert-pattern-mut! \"ab\" states (lambda () size) (lambda (n) (set! size n)) 1)) 3)",
        "difficulty": "medium",
        "tags": ["next-id"],
    },
    {
        "fn": "insert-pattern-mut!",
        "prompt": "Insert 'ab' then 'ac' and return resulting mutable trie size.",
        "gt": "(let* ([states (make-vector 16 #f)] [size 1]) (vector-set! states 0 (make-state 0)) (let ([id2 (insert-pattern-mut! \"ab\" states (lambda () size) (lambda (n) (set! size n)) 1)]) (insert-pattern-mut! \"ac\" states (lambda () size) (lambda (n) (set! size n)) id2) size))",
        "verify": "(= (let* ([states (make-vector 16 #f)] [size 1]) (vector-set! states 0 (make-state 0)) (let ([id2 (insert-pattern-mut! \"ab\" states (lambda () size) (lambda (n) (set! size n)) 1)]) (insert-pattern-mut! \"ac\" states (lambda () size) (lambda (n) (set! size n)) id2) size)) 4)",
        "difficulty": "medium",
        "tags": ["size-update"],
    },
    {
        "fn": "insert-pattern-mut!",
        "prompt": "Insert pattern 'she' into fresh trie and return whether terminal state output contains 'she'.",
        "gt": "(let* ([states (make-vector 16 #f)] [size 1]) (vector-set! states 0 (make-state 0)) (insert-pattern-mut! \"she\" states (lambda () size) (lambda (n) (set! size n)) 1) (let* ([root (vector-ref states 0)] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref states s)))] [she (dict-lookup #\\e (ac-state-trans (vector-ref states sh)))]) (set-member? \"she\" (ac-state-output (vector-ref states she)))))",
        "verify": "(equal? (let* ([states (make-vector 16 #f)] [size 1]) (vector-set! states 0 (make-state 0)) (insert-pattern-mut! \"she\" states (lambda () size) (lambda (n) (set! size n)) 1) (let* ([root (vector-ref states 0)] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref states s)))] [she (dict-lookup #\\e (ac-state-trans (vector-ref states sh)))]) (set-member? \"she\" (ac-state-output (vector-ref states she))))) #t)",
        "difficulty": "medium",
        "tags": ["terminal"],
    },
    {
        "fn": "insert-pattern-mut!",
        "prompt": "Insert 'ab' twice and return whether second insert keeps next-id unchanged.",
        "gt": "(let* ([states (make-vector 16 #f)] [size 1]) (vector-set! states 0 (make-state 0)) (let* ([id2 (insert-pattern-mut! \"ab\" states (lambda () size) (lambda (n) (set! size n)) 1)] [id3 (insert-pattern-mut! \"ab\" states (lambda () size) (lambda (n) (set! size n)) id2)]) (= id2 id3)))",
        "verify": "(equal? (let* ([states (make-vector 16 #f)] [size 1]) (vector-set! states 0 (make-state 0)) (let* ([id2 (insert-pattern-mut! \"ab\" states (lambda () size) (lambda (n) (set! size n)) 1)] [id3 (insert-pattern-mut! \"ab\" states (lambda () size) (lambda (n) (set! size n)) id2)]) (= id2 id3))) #t)",
        "difficulty": "hard",
        "tags": ["idempotent-path"],
    },

    # compute-failures
    {
        "fn": "compute-failures",
        "prompt": "Compute failures for trie ('he' 'she') and return fail id of state 'she'.",
        "gt": "(let* ([states (build-trie '(\"he\" \"she\"))] [done (compute-failures states)] [root (vector-ref done 0)] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref done s)))] [she (dict-lookup #\\e (ac-state-trans (vector-ref done sh)))]) (ac-state-fail (vector-ref done she)))",
        "verify": "(let* ([states (build-trie '(\"he\" \"she\"))] [done (compute-failures states)] [root (vector-ref done 0)] [h (dict-lookup #\\h (ac-state-trans root))] [he (dict-lookup #\\e (ac-state-trans (vector-ref done h)))] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref done s)))] [she (dict-lookup #\\e (ac-state-trans (vector-ref done sh)))]) (= (ac-state-fail (vector-ref done she)) he))",
        "difficulty": "medium",
        "tags": ["fail-link"],
    },
    {
        "fn": "compute-failures",
        "prompt": "After compute-failures on ('he' 'she'), return whether she-state output includes inherited 'he'.",
        "gt": "(let* ([states (build-trie '(\"he\" \"she\"))] [done (compute-failures states)] [root (vector-ref done 0)] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref done s)))] [she (dict-lookup #\\e (ac-state-trans (vector-ref done sh)))]) (set-member? \"he\" (ac-state-output (vector-ref done she))))",
        "verify": "(equal? (let* ([states (build-trie '(\"he\" \"she\"))] [done (compute-failures states)] [root (vector-ref done 0)] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref done s)))] [she (dict-lookup #\\e (ac-state-trans (vector-ref done sh)))]) (set-member? \"he\" (ac-state-output (vector-ref done she)))) #t)",
        "difficulty": "medium",
        "tags": ["output-union"],
    },
    {
        "fn": "compute-failures",
        "prompt": "Compute failures for empty trie and return whether result remains a single root state.",
        "gt": "(let ([done (compute-failures (build-trie '()))]) (= (vector-length done) 1))",
        "verify": "(equal? (let ([done (compute-failures (build-trie '()))]) (= (vector-length done) 1)) #t)",
        "difficulty": "easy",
        "tags": ["empty"],
    },
    {
        "fn": "compute-failures",
        "prompt": "Run compute-failures for ('ab' 'bc') and return whether fail of state 'ab' points to state 'b'.",
        "gt": "(let* ([done (compute-failures (build-trie '(\"ab\" \"bc\")))] [root (vector-ref done 0)] [a (dict-lookup #\\a (ac-state-trans root))] [ab (dict-lookup #\\b (ac-state-trans (vector-ref done a)))] [b (dict-lookup #\\b (ac-state-trans root))]) (= (ac-state-fail (vector-ref done ab)) b))",
        "verify": "(equal? (let* ([done (compute-failures (build-trie '(\"ab\" \"bc\")))] [root (vector-ref done 0)] [a (dict-lookup #\\a (ac-state-trans root))] [ab (dict-lookup #\\b (ac-state-trans (vector-ref done a)))] [b (dict-lookup #\\b (ac-state-trans root))]) (= (ac-state-fail (vector-ref done ab)) b)) #t)",
        "difficulty": "hard",
        "tags": ["cross-prefix"],
    },

    # bfs-mut!
    {
        "fn": "bfs-mut!",
        "prompt": "Run bfs-mut! over trie ('he' 'she') and return whether she fail-link points to he.",
        "gt": "(let* ([states (build-trie '(\"he\" \"she\"))] [root (vector-ref states 0)] [children (dict-values (ac-state-trans root))] [q (fold-left (lambda (qq child) (queue-enqueue child qq)) queue-empty children)]) (bfs-mut! states q) (let* ([h (dict-lookup #\\h (ac-state-trans root))] [he (dict-lookup #\\e (ac-state-trans (vector-ref states h)))] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref states s)))] [she (dict-lookup #\\e (ac-state-trans (vector-ref states sh)))]) (= (ac-state-fail (vector-ref states she)) he)))",
        "verify": "(equal? (let* ([states (build-trie '(\"he\" \"she\"))] [root (vector-ref states 0)] [children (dict-values (ac-state-trans root))] [q (fold-left (lambda (qq child) (queue-enqueue child qq)) queue-empty children)]) (bfs-mut! states q) (let* ([h (dict-lookup #\\h (ac-state-trans root))] [he (dict-lookup #\\e (ac-state-trans (vector-ref states h)))] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref states s)))] [she (dict-lookup #\\e (ac-state-trans (vector-ref states sh)))]) (= (ac-state-fail (vector-ref states she)) he))) #t)",
        "difficulty": "hard",
        "tags": ["fail-link"],
    },
    {
        "fn": "bfs-mut!",
        "prompt": "Run bfs-mut! and return whether she output set contains inherited he after propagation.",
        "gt": "(let* ([states (build-trie '(\"he\" \"she\"))] [root (vector-ref states 0)] [children (dict-values (ac-state-trans root))] [q (fold-left (lambda (qq child) (queue-enqueue child qq)) queue-empty children)]) (bfs-mut! states q) (let* ([s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref states s)))] [she (dict-lookup #\\e (ac-state-trans (vector-ref states sh)))]) (set-member? \"he\" (ac-state-output (vector-ref states she)))))",
        "verify": "(equal? (let* ([states (build-trie '(\"he\" \"she\"))] [root (vector-ref states 0)] [children (dict-values (ac-state-trans root))] [q (fold-left (lambda (qq child) (queue-enqueue child qq)) queue-empty children)]) (bfs-mut! states q) (let* ([s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref states s)))] [she (dict-lookup #\\e (ac-state-trans (vector-ref states sh)))]) (set-member? \"he\" (ac-state-output (vector-ref states she))))) #t)",
        "difficulty": "hard",
        "tags": ["output-propagation"],
    },
    {
        "fn": "bfs-mut!",
        "prompt": "Run bfs-mut! on empty queue and return whether automaton states remain unchanged in count.",
        "gt": "(let* ([states (build-trie '(\"a\"))] [n (vector-length states)]) (bfs-mut! states queue-empty) (= (vector-length states) n))",
        "verify": "(equal? (let* ([states (build-trie '(\"a\"))] [n (vector-length states)]) (bfs-mut! states queue-empty) (= (vector-length states) n)) #t)",
        "difficulty": "easy",
        "tags": ["empty-queue"],
    },
    {
        "fn": "bfs-mut!",
        "prompt": "Build trie ('ab' 'bc'), run bfs-mut!, and return whether get-next can bridge from state 'ab' with char c.",
        "gt": "(let* ([states (build-trie '(\"ab\" \"bc\"))] [root (vector-ref states 0)] [kids (dict-values (ac-state-trans root))] [q (fold-left (lambda (qq child) (queue-enqueue child qq)) queue-empty kids)]) (bfs-mut! states q) (let* ([a (dict-lookup #\\a (ac-state-trans root))] [ab (dict-lookup #\\b (ac-state-trans (vector-ref states a)))] [b (dict-lookup #\\b (ac-state-trans root))] [bc (dict-lookup #\\c (ac-state-trans (vector-ref states b)))]) (= (get-next states ab #\\c) bc)))",
        "verify": "(equal? (let* ([states (build-trie '(\"ab\" \"bc\"))] [root (vector-ref states 0)] [kids (dict-values (ac-state-trans root))] [q (fold-left (lambda (qq child) (queue-enqueue child qq)) queue-empty kids)]) (bfs-mut! states q) (let* ([a (dict-lookup #\\a (ac-state-trans root))] [ab (dict-lookup #\\b (ac-state-trans (vector-ref states a)))] [b (dict-lookup #\\b (ac-state-trans root))] [bc (dict-lookup #\\c (ac-state-trans (vector-ref states b)))]) (= (get-next states ab #\\c) bc))) #t)",
        "difficulty": "hard",
        "tags": ["fallback-bridge"],
    },

    # find-fail
    {
        "fn": "find-fail",
        "prompt": "Return find-fail result for state 'sh' with char e in automaton ('he' 'she').",
        "gt": "(let* ([a (make-automaton '(\"he\" \"she\"))] [root (vector-ref a 0)] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref a s)))]) (find-fail a sh #\\e))",
        "verify": "(let* ([a (make-automaton '(\"he\" \"she\"))] [root (vector-ref a 0)] [h (dict-lookup #\\h (ac-state-trans root))] [he (dict-lookup #\\e (ac-state-trans (vector-ref a h)))] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref a s)))]) (= (find-fail a sh #\\e) he))",
        "difficulty": "medium",
        "tags": ["match-fallback"],
    },
    {
        "fn": "find-fail",
        "prompt": "Return whether find-fail from state 'sh' on char x returns root fallback 0.",
        "gt": "(let* ([a (make-automaton '(\"he\" \"she\"))] [root (vector-ref a 0)] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref a s)))]) (= (find-fail a sh #\\x) 0))",
        "verify": "(equal? (let* ([a (make-automaton '(\"he\" \"she\"))] [root (vector-ref a 0)] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref a s)))]) (= (find-fail a sh #\\x) 0)) #t)",
        "difficulty": "easy",
        "tags": ["root-fallback"],
    },
    {
        "fn": "find-fail",
        "prompt": "On automaton ('ab' 'bc'), return whether find-fail from state 'ab' on c lands at state 'bc'.",
        "gt": "(let* ([a (make-automaton '(\"ab\" \"bc\"))] [root (vector-ref a 0)] [a1 (dict-lookup #\\a (ac-state-trans root))] [ab (dict-lookup #\\b (ac-state-trans (vector-ref a a1)))] [b (dict-lookup #\\b (ac-state-trans root))] [bc (dict-lookup #\\c (ac-state-trans (vector-ref a b)))]) (= (find-fail a ab #\\c) bc))",
        "verify": "(equal? (let* ([a (make-automaton '(\"ab\" \"bc\"))] [root (vector-ref a 0)] [a1 (dict-lookup #\\a (ac-state-trans root))] [ab (dict-lookup #\\b (ac-state-trans (vector-ref a a1)))] [b (dict-lookup #\\b (ac-state-trans root))] [bc (dict-lookup #\\c (ac-state-trans (vector-ref a b)))]) (= (find-fail a ab #\\c) bc)) #t)",
        "difficulty": "hard",
        "tags": ["chain-fallback"],
    },
    {
        "fn": "find-fail",
        "prompt": "Return whether root-state find-fail on unknown char z is 0.",
        "gt": "(let* ([a (make-automaton '(\"he\"))]) (= (find-fail a 0 #\\z) 0))",
        "verify": "(equal? (let* ([a (make-automaton '(\"he\"))]) (= (find-fail a 0 #\\z) 0)) #t)",
        "difficulty": "easy",
        "tags": ["root-case"],
    },

    # get-next
    {
        "fn": "get-next",
        "prompt": "Return get-next from root on char h in automaton ('he' 'she').",
        "gt": "(let* ([a (make-automaton '(\"he\" \"she\"))]) (get-next a 0 #\\h))",
        "verify": "(let* ([a (make-automaton '(\"he\" \"she\"))] [root (vector-ref a 0)] [h (dict-lookup #\\h (ac-state-trans root))]) (= (get-next a 0 #\\h) h))",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "get-next",
        "prompt": "Return whether get-next from root on unknown char x returns 0.",
        "gt": "(let* ([a (make-automaton '(\"he\" \"she\"))]) (= (get-next a 0 #\\x) 0))",
        "verify": "(equal? (let* ([a (make-automaton '(\"he\" \"she\"))]) (= (get-next a 0 #\\x) 0)) #t)",
        "difficulty": "easy",
        "tags": ["missing-root"],
    },
    {
        "fn": "get-next",
        "prompt": "For automaton ('ab' 'bc'), return whether get-next from state 'ab' with c reaches state 'bc'.",
        "gt": "(let* ([a (make-automaton '(\"ab\" \"bc\"))] [root (vector-ref a 0)] [a1 (dict-lookup #\\a (ac-state-trans root))] [ab (dict-lookup #\\b (ac-state-trans (vector-ref a a1)))] [b (dict-lookup #\\b (ac-state-trans root))] [bc (dict-lookup #\\c (ac-state-trans (vector-ref a b)))]) (= (get-next a ab #\\c) bc))",
        "verify": "(equal? (let* ([a (make-automaton '(\"ab\" \"bc\"))] [root (vector-ref a 0)] [a1 (dict-lookup #\\a (ac-state-trans root))] [ab (dict-lookup #\\b (ac-state-trans (vector-ref a a1)))] [b (dict-lookup #\\b (ac-state-trans root))] [bc (dict-lookup #\\c (ac-state-trans (vector-ref a b)))]) (= (get-next a ab #\\c) bc)) #t)",
        "difficulty": "hard",
        "tags": ["recursive-fallback"],
    },
    {
        "fn": "get-next",
        "prompt": "For automaton ('ab' 'bc'), return whether get-next from state 'ab' with x falls back to 0.",
        "gt": "(let* ([a (make-automaton '(\"ab\" \"bc\"))] [root (vector-ref a 0)] [a1 (dict-lookup #\\a (ac-state-trans root))] [ab (dict-lookup #\\b (ac-state-trans (vector-ref a a1)))]) (= (get-next a ab #\\x) 0))",
        "verify": "(equal? (let* ([a (make-automaton '(\"ab\" \"bc\"))] [root (vector-ref a 0)] [a1 (dict-lookup #\\a (ac-state-trans root))] [ab (dict-lookup #\\b (ac-state-trans (vector-ref a a1)))]) (= (get-next a ab #\\x) 0)) #t)",
        "difficulty": "medium",
        "tags": ["dead-transition"],
    },

    # make-automaton
    {
        "fn": "make-automaton",
        "prompt": "Build automaton for ('he' 'she') and return state count.",
        "gt": "(vector-length (make-automaton '(\"he\" \"she\")))",
        "verify": "(= (vector-length (make-automaton '(\"he\" \"she\"))) 6)",
        "difficulty": "easy",
        "tags": ["size"],
    },
    {
        "fn": "make-automaton",
        "prompt": "Build automaton for ('he' 'she') and return whether she state fail-link points to he.",
        "gt": "(let* ([a (make-automaton '(\"he\" \"she\"))] [root (vector-ref a 0)] [h (dict-lookup #\\h (ac-state-trans root))] [he (dict-lookup #\\e (ac-state-trans (vector-ref a h)))] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref a s)))] [she (dict-lookup #\\e (ac-state-trans (vector-ref a sh)))]) (= (ac-state-fail (vector-ref a she)) he))",
        "verify": "(equal? (let* ([a (make-automaton '(\"he\" \"she\"))] [root (vector-ref a 0)] [h (dict-lookup #\\h (ac-state-trans root))] [he (dict-lookup #\\e (ac-state-trans (vector-ref a h)))] [s (dict-lookup #\\s (ac-state-trans root))] [sh (dict-lookup #\\h (ac-state-trans (vector-ref a s)))] [she (dict-lookup #\\e (ac-state-trans (vector-ref a sh)))]) (= (ac-state-fail (vector-ref a she)) he)) #t)",
        "difficulty": "medium",
        "tags": ["fail-link"],
    },
    {
        "fn": "make-automaton",
        "prompt": "Build automaton for ('he' 'she' 'his' 'hers') and return whether searching 'ushers' yields three expected matches.",
        "gt": "(let* ([a (make-automaton '(\"he\" \"she\" \"his\" \"hers\"))] [m (search a \"ushers\")]) (and (= (length m) 3) (if (member '(1 . \"she\") m) #t #f) (if (member '(2 . \"he\") m) #t #f) (if (member '(2 . \"hers\") m) #t #f)))",
        "verify": "(equal? (let* ([a (make-automaton '(\"he\" \"she\" \"his\" \"hers\"))] [m (search a \"ushers\")]) (and (= (length m) 3) (if (member '(1 . \"she\") m) #t #f) (if (member '(2 . \"he\") m) #t #f) (if (member '(2 . \"hers\") m) #t #f))) #t)",
        "difficulty": "medium",
        "tags": ["search-integrated"],
    },
    {
        "fn": "make-automaton",
        "prompt": "Build automaton for empty pattern list and return whether searching any text returns empty list.",
        "gt": "(let* ([a (make-automaton '())] [m (search a \"anything\")]) (null? m))",
        "verify": "(equal? (let* ([a (make-automaton '())] [m (search a \"anything\")]) (null? m)) #t)",
        "difficulty": "easy",
        "tags": ["empty-patterns"],
    },

    # search
    {
        "fn": "search",
        "prompt": "Search automaton ('he' 'she' 'his' 'hers') on 'ushers' and return whether key matches exist.",
        "gt": "(let* ([a (make-automaton '(\"he\" \"she\" \"his\" \"hers\"))] [m (search a \"ushers\")]) (and (if (member '(1 . \"she\") m) #t #f) (if (member '(2 . \"he\") m) #t #f) (if (member '(2 . \"hers\") m) #t #f)))",
        "verify": "(equal? (let* ([a (make-automaton '(\"he\" \"she\" \"his\" \"hers\"))] [m (search a \"ushers\")]) (and (if (member '(1 . \"she\") m) #t #f) (if (member '(2 . \"he\") m) #t #f) (if (member '(2 . \"hers\") m) #t #f))) #t)",
        "difficulty": "medium",
        "tags": ["classic-example"],
    },
    {
        "fn": "search",
        "prompt": "Search automaton ('a' 'aa' 'aaa') on 'aaaa' and return match count.",
        "gt": "(let* ([a (make-automaton '(\"a\" \"aa\" \"aaa\"))] [m (search a \"aaaa\")]) (length m))",
        "verify": "(= (let* ([a (make-automaton '(\"a\" \"aa\" \"aaa\"))] [m (search a \"aaaa\")]) (length m)) 9)",
        "difficulty": "medium",
        "tags": ["overlap-count"],
    },
    {
        "fn": "search",
        "prompt": "Search automaton ('a' 'aa' 'aaa') on 'aaaa' and return whether earliest match position is 0.",
        "gt": "(let* ([a (make-automaton '(\"a\" \"aa\" \"aaa\"))] [m (search a \"aaaa\")]) (= (car (car m)) 0))",
        "verify": "(equal? (let* ([a (make-automaton '(\"a\" \"aa\" \"aaa\"))] [m (search a \"aaaa\")]) (= (car (car m)) 0)) #t)",
        "difficulty": "medium",
        "tags": ["ordering"],
    },
    {
        "fn": "search",
        "prompt": "Search automaton ('ATG' 'TGA' 'TAG' 'TAA') on 'AAATGATAG' and return whether start/stop codon matches are present.",
        "gt": "(let* ([a (make-automaton '(\"ATG\" \"TGA\" \"TAG\" \"TAA\"))] [m (search a \"AAATGATAG\")]) (and (if (member '(2 . \"ATG\") m) #t #f) (if (member '(3 . \"TGA\") m) #t #f) (if (member '(6 . \"TAG\") m) #t #f)))",
        "verify": "(equal? (let* ([a (make-automaton '(\"ATG\" \"TGA\" \"TAG\" \"TAA\"))] [m (search a \"AAATGATAG\")]) (and (if (member '(2 . \"ATG\") m) #t #f) (if (member '(3 . \"TGA\") m) #t #f) (if (member '(6 . \"TAG\") m) #t #f))) #t)",
        "difficulty": "medium",
        "tags": ["dna"],
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

# Strengthen non-composition rows with an independent behavior check selected
# from composition verifies of the same source function.
composition_verify_by_fn: Dict[str, List[str]] = defaultdict(list)
for case in composition_cases:
    fn = str(case["fn"])
    check = str(case["verify"]).strip()
    if check not in composition_verify_by_fn[fn]:
        composition_verify_by_fn[fn].append(check)

for sample in samples:
    family = str(sample["family"])
    if family == "composition":
        continue

    fn = str(sample["source_function"])
    checks = composition_verify_by_fn.get(fn, [])
    if not checks:
        continue

    sid = str(sample["id"])
    base = sum(ord(ch) for ch in sid)
    pick1 = base % len(checks)
    selected_checks = [checks[pick1]]
    if len(checks) > 1:
        pick2 = (base * 7 + len(sid)) % len(checks)
        if pick2 == pick1:
            pick2 = (pick1 + 1) % len(checks)
        selected_checks.append(checks[pick2])

    sample["verify_expr"] = (
        f"(and {str(sample['verify_expr']).strip()} {' '.join(selected_checks)})"
    )
    checks_block = "\n\n".join(
        f"Check {idx + 1}:\n```scheme\n{check}\n```"
        for idx, check in enumerate(selected_checks)
    )
    sample["prompt"] = (
        f"{str(sample['prompt']).rstrip()}\n\n"
        "Independent behavior checks to satisfy:\n"
        f"{checks_block}"
    )

if sum(1 for s in samples if s["family"] == "composition") != 32:
    raise ValueError("composition family must contain exactly 32 samples")

by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for sample in samples:
    by_family[str(sample["family"])].append(sample)

EVAL_RATIO = 0.18
EVAL_MIN_BY_FAMILY = {
    "spec_to_code": 3,
    "translation": 3,
    "bugfix": 2,
    "composition": 5,
}

eval_ids = compute_leakage_aware_eval_ids(
    samples,
    eval_ratio=EVAL_RATIO,
    eval_min_by_family=EVAL_MIN_BY_FAMILY,
    enforce_source_function_coverage=True,
)

id_to_sample: Dict[str, Dict[str, object]] = {str(sample["id"]): sample for sample in samples}
all_source_functions = sorted({str(sample["source_function"]) for sample in samples})
missing_after = [
    fn
    for fn in all_source_functions
    if not any(str(id_to_sample[sid]["source_function"]) == fn for sid in eval_ids)
]
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

if not train_rows or not eval_rows:
    raise ValueError(f"split mismatch: train={len(train_rows)}, eval={len(eval_rows)}")


def quality_gate(rows: List[Dict[str, object]]) -> None:
    local_symbols = set(FUNCTION_ORDER)

    for i, row in enumerate(rows, start=1):
        family = str(row["family"])
        source_function = str(row["source_function"])
        prompt = str(row["prompt"])
        gt = str(row["ground_truth"]).strip()
        verify = str(row["verify_expr"]).strip()

        if gt == verify:
            raise ValueError(f"row {i}: tautological verify_expr equals ground_truth")

        if family in {"spec_to_code", "translation", "bugfix"}:
            expected_prefix = f"(define ({source_function}"
            if not gt.lstrip().startswith(expected_prefix):
                raise ValueError(
                    f"row {i}: source function mismatch; expected ground_truth to start with {expected_prefix}"
                )
            if source_function not in prompt:
                raise ValueError(f"row {i}: prompt does not mention source_function {source_function}")

        if family == "composition":
            tokens = set(TOKEN_RE.findall(gt + "\n" + verify))
            if source_function not in tokens:
                raise ValueError(f"row {i}: composition source_function not used in solution/verify")
            used_local = {name for name in local_symbols if name in tokens}
            if len(used_local) < 2:
                # Warn but don't fail — aho-corasick compositions naturally compose
                # one API function with internal data structure accessors
                import sys as _sys
                print(f"  warn: row {i}: composition uses only {used_local} from FUNCTION_ORDER", file=_sys.stderr)

        if family == "bugfix":
            if "Known issue:" not in prompt:
                raise ValueError(f"row {i}: bugfix prompt missing issue description")
            if "```scheme" not in prompt:
                raise ValueError(f"row {i}: bugfix prompt missing code block")
            if "<TODO>" in prompt:
                raise ValueError(f"row {i}: weak bugfix prompt contains TODO placeholder")


quality_gate(train_rows + eval_rows)


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
