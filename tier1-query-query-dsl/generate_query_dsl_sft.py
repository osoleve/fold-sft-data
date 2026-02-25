#!/usr/bin/env python3
"""Generate Tier-1 SFT samples for lattice/query/query-dsl.ss."""

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

SOURCE_MODULE = "lattice/query/query-dsl.ss"
SOURCE_TEST = "lattice/query/test-query-dsl.ss"

DEFS: Dict[str, str] = {
    "build-tag-predicate": """(define (build-tag-predicate tag)
  (lambda (block)
          (eq? (block-tag block) tag)))""",
    "build-has-refs-predicate": """(define (build-has-refs-predicate)
  (lambda (block)
          (> (vector-length (block-refs block)) 0)))""",
    "build-refs-count-predicate": """(define (build-refs-count-predicate n)
  (lambda (block)
          (= (vector-length (block-refs block)) n)))""",
    "build-refs-to-predicate": """(define (build-refs-to-predicate target-hash)
  (lambda (block)
          (let ([refs (block-refs block)])
               (let check-refs ([i 0])
                    (cond
                     [(>= i (vector-length refs)) #f]
                     [(bytevector=? (vector-ref refs i) target-hash) #t]
                     [else (check-refs (+ i 1))])))))""",
    "build-payload-size-predicate": """(define (build-payload-size-predicate comparator size)
  (lambda (block)
          (comparator (bytevector-length (block-payload block)) size)))""",
    "interpret-match": """(define (interpret-match pattern)
  (let ([key (car pattern)]
        [value (cdr pattern)])
       (case key
             [(tag)
              (build-tag-predicate value)]
             [(payload-contains)
              (build-payload-contains-predicate value)]
             [(payload-matches)
              (build-payload-matches-predicate value)]
             [(has-refs)
              (build-has-refs-predicate)]
             [(refs-count)
              (build-refs-count-predicate value)]
             [(payload-size-gt)
              (build-payload-size-predicate > value)]
             [(payload-size-lt)
              (build-payload-size-predicate < value)]
             [(payload-size-eq)
              (build-payload-size-predicate = value)]
             [else
              (error 'interpret-match "Unknown match pattern" key)])))""",
    "and-all": """(define (and-all predicates block)
  (let loop ([preds predicates])
       (or (null? preds)
           (and ((car preds) block)
                (loop (cdr preds))))))""",
    "or-any": """(define (or-any predicates block)
  (let loop ([preds predicates])
       (and (pair? preds)
            (or ((car preds) block)
                (loop (cdr preds))))))""",
}

FUNCTION_ORDER = [
    "build-tag-predicate",
    "build-has-refs-predicate",
    "build-refs-count-predicate",
    "build-refs-to-predicate",
    "build-payload-size-predicate",
    "interpret-match",
    "and-all",
    "or-any",
]

SUPPORT_DEFS: Dict[str, str] = {
    "make-test-block": """(define (make-test-block tag payload refs)
  (vector tag payload refs))""",
    "block-tag": """(define (block-tag block)
  (vector-ref block 0))""",
    "block-payload": """(define (block-payload block)
  (vector-ref block 1))""",
    "block-refs": """(define (block-refs block)
  (vector-ref block 2))""",
    "safe-utf8->string": """(define (safe-utf8->string bv)
  (guard (ex [else ""])
         (utf8->string bv)))""",
    "query-string-contains?": """(define (query-string-contains? haystack needle)
  (let ([h-len (string-length haystack)]
        [n-len (string-length needle)])
       (cond
        [(= n-len 0) #t]
        [(> n-len h-len) #f]
        [else
         (let ([first-char (string-ref needle 0)])
              (let outer ([i 0])
                   (cond
                    [(> (+ i n-len) h-len) #f]
                    [(char=? (string-ref haystack i) first-char)
                     (if (let inner ([j 1])
                              (cond
                               [(= j n-len) #t]
                               [(char=? (string-ref haystack (+ i j))
                                        (string-ref needle j))
                                (inner (+ j 1))]
                               [else #f]))
                         #t
                         (outer (+ i 1)))]
                    [else (outer (+ i 1))])))])))""",
    "build-payload-contains-predicate": """(define (build-payload-contains-predicate substring)
  (lambda (block)
          (let ([payload-str (safe-utf8->string (block-payload block))])
               (query-string-contains? payload-str substring))))""",
    "build-payload-matches-predicate": """(define (build-payload-matches-predicate matcher)
  (lambda (block)
          (let ([payload-str (safe-utf8->string (block-payload block))])
               (matcher payload-str))))""",
}

SUPPORT_ORDER = [
    "make-test-block",
    "block-tag",
    "block-payload",
    "block-refs",
    "safe-utf8->string",
    "query-string-contains?",
    "build-payload-contains-predicate",
    "build-payload-matches-predicate",
]

ALL_DEFS: Dict[str, str] = dict(SUPPORT_DEFS)
ALL_DEFS.update(DEFS)
ALL_ORDER = list(dict.fromkeys(FUNCTION_ORDER + SUPPORT_ORDER))

FUNCTION_SPECS = {
    "build-tag-predicate": "Return a predicate that matches blocks by exact tag symbol.",
    "build-has-refs-predicate": "Return a predicate that is true only for blocks with at least one reference.",
    "build-refs-count-predicate": "Return a predicate that matches blocks with exactly n references.",
    "build-refs-to-predicate": "Return a predicate that matches blocks containing the given target hash in refs.",
    "build-payload-size-predicate": "Return a predicate comparing payload byte length against a threshold via comparator.",
    "interpret-match": "Interpret one match pattern and return the corresponding block predicate.",
    "and-all": "Return #t only when every predicate in the list matches the block.",
    "or-any": "Return #t when at least one predicate in the list matches the block.",
}

SKELETONS = {
    "build-tag-predicate": """(define (build-tag-predicate tag)
  ;; TODO: return block predicate for exact tag match
  <TODO>)""",
    "build-has-refs-predicate": """(define (build-has-refs-predicate)
  ;; TODO: return predicate true when block has at least one ref
  <TODO>)""",
    "build-refs-count-predicate": """(define (build-refs-count-predicate n)
  ;; TODO: return predicate true when refs count equals n
  <TODO>)""",
    "build-refs-to-predicate": """(define (build-refs-to-predicate target-hash)
  ;; TODO: return predicate true when target-hash appears in block refs
  <TODO>)""",
    "build-payload-size-predicate": """(define (build-payload-size-predicate comparator size)
  ;; TODO: return predicate that compares payload size via comparator
  <TODO>)""",
    "interpret-match": """(define (interpret-match pattern)
  ;; TODO: dispatch on match key and return corresponding predicate
  <TODO>)""",
    "and-all": """(define (and-all predicates block)
  ;; TODO: true only if every predicate matches block
  <TODO>)""",
    "or-any": """(define (or-any predicates block)
  ;; TODO: true if any predicate matches block
  <TODO>)""",
}

DIFFICULTY = {
    "build-tag-predicate": "easy",
    "build-has-refs-predicate": "easy",
    "build-refs-count-predicate": "easy",
    "build-refs-to-predicate": "medium",
    "build-payload-size-predicate": "easy",
    "interpret-match": "hard",
    "and-all": "medium",
    "or-any": "medium",
}

VERIFY_BY_FUNCTION = {
    "build-tag-predicate": """(let ([pred (build-tag-predicate 'entity)])
  (and (pred (make-test-block 'entity (string->utf8 "alice") (vector)))
       (not (pred (make-test-block 'relation (string->utf8 "knows") (vector))))))""",
    "build-has-refs-predicate": """(let ([pred (build-has-refs-predicate)])
  (and (pred (make-test-block 'entity (string->utf8 "alice") (vector (string->utf8 "h1"))))
       (not (pred (make-test-block 'entity (string->utf8 "bob") (vector))))))""",
    "build-refs-count-predicate": """(let ([pred2 (build-refs-count-predicate 2)])
  (and (pred2 (make-test-block 'relation (string->utf8 "r") (vector (string->utf8 "a") (string->utf8 "b"))))
       (not (pred2 (make-test-block 'relation (string->utf8 "r") (vector (string->utf8 "a")))))))""",
    "build-refs-to-predicate": """(let* ([target (string->utf8 "target")]
       [pred (build-refs-to-predicate target)])
  (and (pred (make-test-block 'relation (string->utf8 "r") (vector target (string->utf8 "other"))))
       (not (pred (make-test-block 'relation (string->utf8 "r") (vector (string->utf8 "x") (string->utf8 "y")))))))""",
    "build-payload-size-predicate": """(let* ([gt3 (build-payload-size-predicate > 3)]
       [eq5 (build-payload-size-predicate = 5)]
       [blk4 (make-test-block 'doc (string->utf8 "abcd") (vector))]
       [blk2 (make-test-block 'doc (string->utf8 "ab") (vector))]
       [blk5 (make-test-block 'doc (string->utf8 "hello") (vector))])
  (and (gt3 blk4)
       (not (gt3 blk2))
       (eq5 blk5)))""",
    "interpret-match": """(let* ([b-entity (make-test-block 'entity (string->utf8 "abc") (vector))]
       [b-two-refs (make-test-block 'relation (string->utf8 "xy") (vector (string->utf8 "r1") (string->utf8 "r2")))]
       [b-one-ref (make-test-block 'relation (string->utf8 "xy") (vector (string->utf8 "r1")))]
       [p-tag (interpret-match '(tag . entity))]
       [p-has (interpret-match '(has-refs . #t))]
       [p-count (interpret-match '(refs-count . 2))]
       [p-size (interpret-match '(payload-size-gt . 2))])
  (and (p-tag b-entity)
       (p-has b-two-refs)
       (not (p-has b-entity))
       (p-count b-two-refs)
       (not (p-count b-one-ref))
       (p-size (make-test-block 'doc (string->utf8 "abc") (vector)))
       (guard (ex [else #t])
              (begin (interpret-match '(unknown . 1)) #f))))""",
    "and-all": """(let* ([p1 (lambda (n) (> n 0))]
       [p2 (lambda (n) (even? n))]
       [p3 (lambda (n) (< n 10))])
  (and (and-all (list p1 p2 p3) 4)
       (not (and-all (list p1 p2 p3) 11))
       (and-all '() 42)))""",
    "or-any": """(let* ([p1 (lambda (n) (= (modulo n 2) 0))]
       [p2 (lambda (n) (> n 10))])
  (and (or-any (list p1 p2) 8)
       (not (or-any (list p1 p2) 7))
       (not (or-any '() 1))))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "build-tag-predicate": "def build_tag_predicate(tag):\n    return lambda block: block['tag'] == tag",
    "build-has-refs-predicate": "def build_has_refs_predicate():\n    return lambda block: len(block['refs']) > 0",
    "build-refs-count-predicate": "def build_refs_count_predicate(n):\n    return lambda block: len(block['refs']) == n",
    "build-refs-to-predicate": "def build_refs_to_predicate(target_hash):\n    def pred(block):\n        return any(ref == target_hash for ref in block['refs'])\n    return pred",
    "build-payload-size-predicate": "def build_payload_size_predicate(comparator, size):\n    return lambda block: comparator(len(block['payload']), size)",
    "interpret-match": "def interpret_match(pattern):\n    key, value = pattern\n    if key == 'tag':\n        return build_tag_predicate(value)\n    if key == 'payload-contains':\n        return build_payload_contains_predicate(value)\n    if key == 'payload-matches':\n        return build_payload_matches_predicate(value)\n    if key == 'has-refs':\n        return build_has_refs_predicate()\n    if key == 'refs-count':\n        return build_refs_count_predicate(value)\n    if key == 'payload-size-gt':\n        return build_payload_size_predicate(lambda a, b: a > b, value)\n    if key == 'payload-size-lt':\n        return build_payload_size_predicate(lambda a, b: a < b, value)\n    if key == 'payload-size-eq':\n        return build_payload_size_predicate(lambda a, b: a == b, value)\n    raise ValueError('unknown pattern')",
    "and-all": "def and_all(predicates, block):\n    for pred in predicates:\n        if not pred(block):\n            return False\n    return True",
    "or-any": "def or_any(predicates, block):\n    for pred in predicates:\n        if pred(block):\n            return True\n    return False",
}

CHEZ_SNIPPETS = {
    "build-tag-predicate": "(define (mk-tag-pred t)\n  (lambda (b)\n    (eq? (block-tag b) t)))",
    "build-has-refs-predicate": "(define (mk-has-refs)\n  (lambda (b)\n    (> (vector-length (block-refs b)) 0)))",
    "build-refs-count-predicate": "(define (mk-refs-count n)\n  (lambda (b)\n    (= (vector-length (block-refs b)) n)))",
    "build-refs-to-predicate": "(define (mk-refs-to target)\n  (lambda (b)\n    (let ([refs (block-refs b)])\n      (let loop ([i 0])\n        (cond\n          [(>= i (vector-length refs)) #f]\n          [(bytevector=? (vector-ref refs i) target) #t]\n          [else (loop (+ i 1))])))))",
    "build-payload-size-predicate": "(define (mk-payload-size-pred cmp n)\n  (lambda (b)\n    (cmp (bytevector-length (block-payload b)) n)))",
    "interpret-match": "(define (interp-match pat)\n  (let ([k (car pat)] [v (cdr pat)])\n    (case k\n      [(tag) (build-tag-predicate v)]\n      [(payload-contains) (build-payload-contains-predicate v)]\n      [(payload-matches) (build-payload-matches-predicate v)]\n      [(has-refs) (build-has-refs-predicate)]\n      [(refs-count) (build-refs-count-predicate v)]\n      [(payload-size-gt) (build-payload-size-predicate > v)]\n      [(payload-size-lt) (build-payload-size-predicate < v)]\n      [(payload-size-eq) (build-payload-size-predicate = v)]\n      [else (error 'interp-match \"unknown\" k)])))",
    "and-all": "(define (all? preds b)\n  (let loop ([ps preds])\n    (or (null? ps)\n        (and ((car ps) b)\n             (loop (cdr ps))))))",
    "or-any": "(define (any? preds b)\n  (let loop ([ps preds])\n    (and (pair? ps)\n         (or ((car ps) b)\n             (loop (cdr ps))))))",
}

BUGGY_CASES = [
    {
        "fn": "build-tag-predicate",
        "buggy": """(define (build-tag-predicate tag)
  (lambda (block)
          (eq? (block-payload block) tag)))""",
        "note": "The predicate must compare tag symbols, not payload bytes.",
    },
    {
        "fn": "build-tag-predicate",
        "buggy": """(define (build-tag-predicate tag)
  (lambda (block)
          (eq? (block-tag block) 'tag)))""",
        "note": "The function should compare against parameter `tag`, not literal symbol `'tag`.",
    },
    {
        "fn": "build-has-refs-predicate",
        "buggy": """(define (build-has-refs-predicate)
  (lambda (block)
          (> (vector-length (block-refs block)) 1)))""",
        "note": "Blocks with exactly one reference must also satisfy has-refs.",
    },
    {
        "fn": "build-has-refs-predicate",
        "buggy": """(define (build-has-refs-predicate)
  (lambda (block)
          (>= (vector-length (block-refs block)) 0)))""",
        "note": "Empty ref vectors must return #f, not #t.",
    },
    {
        "fn": "build-refs-count-predicate",
        "buggy": """(define (build-refs-count-predicate n)
  (lambda (block)
          (>= (vector-length (block-refs block)) n)))""",
        "note": "refs-count must enforce exact equality, not lower bound.",
    },
    {
        "fn": "build-refs-count-predicate",
        "buggy": """(define (build-refs-count-predicate n)
  (lambda (block)
          (= (vector-length (block-refs block)) (+ n 1))))""",
        "note": "The comparison is off by one.",
    },
    {
        "fn": "build-refs-to-predicate",
        "buggy": """(define (build-refs-to-predicate target-hash)
  (lambda (block)
          (let ([refs (block-refs block)])
               (let check-refs ([i 0])
                    (cond
                     [(>= i (vector-length refs)) #t]
                     [(bytevector=? (vector-ref refs i) target-hash) #t]
                     [else (check-refs (+ i 1))])))))""",
        "note": "When scan reaches the end without match, result must be #f.",
    },
    {
        "fn": "build-refs-to-predicate",
        "buggy": """(define (build-refs-to-predicate target-hash)
  (lambda (block)
          (let ([refs (block-refs block)])
               (let check-refs ([i 0])
                    (cond
                     [(>= i (vector-length refs)) #f]
                     [(bytevector=? (vector-ref refs i) target-hash) #t]
                     [else (check-refs (+ i 2))])))))""",
        "note": "The scan skips every other reference and can miss valid matches.",
    },
    {
        "fn": "build-payload-size-predicate",
        "buggy": """(define (build-payload-size-predicate comparator size)
  (lambda (block)
          (comparator size (bytevector-length (block-payload block)))))""",
        "note": "Comparator argument order is reversed.",
    },
    {
        "fn": "build-payload-size-predicate",
        "buggy": """(define (build-payload-size-predicate comparator size)
  (lambda (block)
          (comparator (vector-length (block-refs block)) size)))""",
        "note": "Size predicate must use payload byte length, not reference count.",
    },
    {
        "fn": "interpret-match",
        "buggy": """(define (interpret-match pattern)
  (let ([key (car pattern)]
        [value (cdr pattern)])
       (case key
             [(tag)
              (build-tag-predicate value)]
             [(payload-contains)
              (build-payload-contains-predicate value)]
             [(payload-matches)
              (build-payload-matches-predicate value)]
             [(has-refs)
              (build-has-refs-predicate)]
             [(refs-count)
              (build-has-refs-predicate)]
             [(payload-size-gt)
              (build-payload-size-predicate > value)]
             [(payload-size-lt)
              (build-payload-size-predicate < value)]
             [(payload-size-eq)
              (build-payload-size-predicate = value)]
             [else
              (error 'interpret-match \"Unknown match pattern\" key)])))""",
        "note": "refs-count must dispatch to build-refs-count-predicate with the provided count.",
    },
    {
        "fn": "interpret-match",
        "buggy": """(define (interpret-match pattern)
  (let ([key (car pattern)]
        [value (cdr pattern)])
       (case key
             [(tag)
              (build-tag-predicate value)]
             [(payload-contains)
              (build-payload-contains-predicate value)]
             [(payload-matches)
              (build-payload-matches-predicate value)]
             [(has-refs)
              (build-has-refs-predicate)]
             [(refs-count)
              (build-refs-count-predicate value)]
             [(payload-size-gt)
              (build-payload-size-predicate > value)]
             [(payload-size-lt)
              (build-payload-size-predicate > value)]
             [(payload-size-eq)
              (build-payload-size-predicate = value)]
             [else
              (error 'interpret-match \"Unknown match pattern\" key)])))""",
        "note": "payload-size-lt must use < comparator.",
    },
    {
        "fn": "and-all",
        "buggy": """(define (and-all predicates block)
  (let loop ([preds predicates])
       (and (pair? preds)
            (and ((car preds) block)
                 (loop (cdr preds))))))""",
        "note": "and-all should return #t for empty predicate list (identity element).",
    },
    {
        "fn": "and-all",
        "buggy": """(define (and-all predicates block)
  (let loop ([preds predicates])
       (or (null? preds)
           (or ((car preds) block)
               (loop (cdr preds))))))""",
        "note": "Predicates must be combined conjunctively, not disjunctively.",
    },
    {
        "fn": "or-any",
        "buggy": """(define (or-any predicates block)
  (let loop ([preds predicates])
       (or (null? preds)
           (or ((car preds) block)
               (loop (cdr preds))))))""",
        "note": "or-any should return #f for empty predicate list.",
    },
    {
        "fn": "or-any",
        "buggy": """(define (or-any predicates block)
  (let loop ([preds predicates])
       (and (pair? preds)
            (and ((car preds) block)
                 (loop (cdr preds))))))""",
        "note": "or-any must short-circuit on first true predicate, not require all true.",
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
    sid = f"query_dsl_{family}_{family_counter[family]:03d}"
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
            available_functions=ALL_ORDER,
        ),
        "ground_truth": ground_truth.strip(),
        "verify_expr": verify_expr.strip(),
        "tags": tags,
    }
    for k in REQUIRED_KEYS:
        if k not in sample:
            raise ValueError(f"missing key {k}")
    samples.append(sample)


# Dependency-closure fix: scan verify expressions for referenced symbols.
def verify_refs(verify_expr: str) -> List[str]:
    tokens = set(TOKEN_RE.findall(verify_expr))
    names = FUNCTION_ORDER + SUPPORT_ORDER
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


DEPENDS: Dict[str, List[str]] = {
    "make-test-block": [],
    "block-tag": [],
    "block-payload": [],
    "block-refs": [],
    "safe-utf8->string": [],
    "query-string-contains?": [],
    "build-payload-contains-predicate": ["safe-utf8->string", "query-string-contains?", "block-payload"],
    "build-payload-matches-predicate": ["safe-utf8->string", "block-payload"],
    "build-tag-predicate": ["block-tag"],
    "build-has-refs-predicate": ["block-refs"],
    "build-refs-count-predicate": ["block-refs"],
    "build-refs-to-predicate": ["block-refs"],
    "build-payload-size-predicate": ["block-payload"],
    "interpret-match": [
        "build-tag-predicate",
        "build-payload-contains-predicate",
        "build-payload-matches-predicate",
        "build-has-refs-predicate",
        "build-refs-count-predicate",
        "build-payload-size-predicate",
    ],
    "and-all": [],
    "or-any": [],
}


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement this query DSL function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "query", "query-dsl", "spec-to-code", fn],
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
        tags=["tier1", "query", "query-dsl", "skeleton-completion", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement `{fn}` from this query-DSL contract.

Module: `{SOURCE_MODULE}`
Contract focus: {FUNCTION_SPECS[fn]}

Requirements:
1. Preserve query predicate semantics exactly.
2. Keep the exact function name/signature for `{fn}`.
3. Return one production-ready definition only.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "query", "query-dsl", "contract-implementation", fn],
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
        tags=["tier1", "query", "query-dsl", "python-to-scheme", fn],
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
        tags=["tier1", "query", "query-dsl", "chez-to-fold", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Translate this reference implementation into canonical Fold Scheme for `{fn}`.

Preserve observable query behavior exactly.
Keep the target function name/signature as `{fn}`.
Return only the final Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "query", "query-dsl", "reference-translation", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
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
        tags=["tier1", "query", "query-dsl", "bugfix", fn],
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
        verify_expr=build_verify(verify_check, [source_function]),
        tags=["tier1", "query", "query-dsl", "composition", source_function] + extra_tags,
    )


composition_cases = [
    {
        "fn": "build-tag-predicate",
        "prompt": "Build an entity tag predicate and return results for entity/relation/entity blocks as a list.",
        "gt": "(let ([pred (build-tag-predicate 'entity)]) (list (pred (make-test-block 'entity (string->utf8 \"a\") (vector))) (pred (make-test-block 'relation (string->utf8 \"r\") (vector))) (pred (make-test-block 'entity (string->utf8 \"b\") (vector)))))",
        "verify": "(equal? (let ([pred (build-tag-predicate 'entity)]) (list (pred (make-test-block 'entity (string->utf8 \"a\") (vector))) (pred (make-test-block 'relation (string->utf8 \"r\") (vector))) (pred (make-test-block 'entity (string->utf8 \"b\") (vector))))) '(#t #f #t))",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "build-tag-predicate",
        "prompt": "Return whether relation block fails the entity tag predicate.",
        "gt": "(let ([pred (build-tag-predicate 'entity)]) (pred (make-test-block 'relation (string->utf8 \"r\") (vector (string->utf8 \"h\")))))",
        "verify": "(equal? (let ([pred (build-tag-predicate 'entity)]) (pred (make-test-block 'relation (string->utf8 \"r\") (vector (string->utf8 \"h\"))))) #f)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "build-tag-predicate",
        "prompt": "Use and-all with entity tag + has-refs predicates on a block that should satisfy both.",
        "gt": "(let* ([tag-pred (build-tag-predicate 'entity)] [refs-pred (build-has-refs-predicate)] [blk (make-test-block 'entity (string->utf8 \"x\") (vector (string->utf8 \"h1\")))]) (and-all (list tag-pred refs-pred) blk))",
        "verify": "(equal? (let* ([tag-pred (build-tag-predicate 'entity)] [refs-pred (build-has-refs-predicate)] [blk (make-test-block 'entity (string->utf8 \"x\") (vector (string->utf8 \"h1\")))]) (and-all (list tag-pred refs-pred) blk)) #t)",
        "difficulty": "medium",
        "tags": ["integration"],
    },
    {
        "fn": "build-tag-predicate",
        "prompt": "Use or-any over entity/relation predicates on a collection block.",
        "gt": "(let* ([entity? (build-tag-predicate 'entity)] [relation? (build-tag-predicate 'relation)] [blk (make-test-block 'collection (string->utf8 \"c\") (vector))]) (or-any (list entity? relation?) blk))",
        "verify": "(equal? (let* ([entity? (build-tag-predicate 'entity)] [relation? (build-tag-predicate 'relation)] [blk (make-test-block 'collection (string->utf8 \"c\") (vector))]) (or-any (list entity? relation?) blk)) #f)",
        "difficulty": "medium",
        "tags": ["integration"],
    },

    {
        "fn": "build-has-refs-predicate",
        "prompt": "Apply has-refs predicate to blocks with zero and one reference and return the result list.",
        "gt": "(let ([pred (build-has-refs-predicate)]) (list (pred (make-test-block 'note (string->utf8 \"n\") (vector))) (pred (make-test-block 'note (string->utf8 \"n\") (vector (string->utf8 \"h1\"))))))",
        "verify": "(equal? (let ([pred (build-has-refs-predicate)]) (list (pred (make-test-block 'note (string->utf8 \"n\") (vector))) (pred (make-test-block 'note (string->utf8 \"n\") (vector (string->utf8 \"h1\")))))) '(#f #t))",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "build-has-refs-predicate",
        "prompt": "Evaluate has-refs on a block with two references.",
        "gt": "((build-has-refs-predicate) (make-test-block 'relation (string->utf8 \"rel\") (vector (string->utf8 \"a\") (string->utf8 \"b\"))))",
        "verify": "(equal? ((build-has-refs-predicate) (make-test-block 'relation (string->utf8 \"rel\") (vector (string->utf8 \"a\") (string->utf8 \"b\")))) #t)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "build-has-refs-predicate",
        "prompt": "Return whether a block satisfies both has-refs and refs-count=2.",
        "gt": "(let* ([blk (make-test-block 'relation (string->utf8 \"rel\") (vector (string->utf8 \"a\") (string->utf8 \"b\")))] [preds (list (build-has-refs-predicate) (build-refs-count-predicate 2))]) (and-all preds blk))",
        "verify": "(equal? (let* ([blk (make-test-block 'relation (string->utf8 \"rel\") (vector (string->utf8 \"a\") (string->utf8 \"b\")))] [preds (list (build-has-refs-predicate) (build-refs-count-predicate 2))]) (and-all preds blk)) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },
    {
        "fn": "build-has-refs-predicate",
        "prompt": "Check that has-refs is false for an empty refs vector.",
        "gt": "((build-has-refs-predicate) (make-test-block 'entity (string->utf8 \"x\") (vector)))",
        "verify": "(equal? ((build-has-refs-predicate) (make-test-block 'entity (string->utf8 \"x\") (vector))) #f)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },

    {
        "fn": "build-refs-count-predicate",
        "prompt": "Apply refs-count=2 predicate to blocks with 0/1/2 refs and return booleans.",
        "gt": "(let ([pred2 (build-refs-count-predicate 2)]) (list (pred2 (make-test-block 'x (string->utf8 \"\") (vector))) (pred2 (make-test-block 'x (string->utf8 \"\") (vector (string->utf8 \"a\")))) (pred2 (make-test-block 'x (string->utf8 \"\") (vector (string->utf8 \"a\") (string->utf8 \"b\"))))))",
        "verify": "(equal? (let ([pred2 (build-refs-count-predicate 2)]) (list (pred2 (make-test-block 'x (string->utf8 \"\") (vector))) (pred2 (make-test-block 'x (string->utf8 \"\") (vector (string->utf8 \"a\")))) (pred2 (make-test-block 'x (string->utf8 \"\") (vector (string->utf8 \"a\") (string->utf8 \"b\")))))) '(#f #f #t))",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "build-refs-count-predicate",
        "prompt": "Return whether a two-ref block satisfies count=2 and not count=1.",
        "gt": "(let* ([blk (make-test-block 'r (string->utf8 \"\") (vector (string->utf8 \"a\") (string->utf8 \"b\")))] [count2 (build-refs-count-predicate 2)] [count1 (build-refs-count-predicate 1)]) (and (count2 blk) (not (count1 blk))))",
        "verify": "(equal? (let* ([blk (make-test-block 'r (string->utf8 \"\") (vector (string->utf8 \"a\") (string->utf8 \"b\")))] [count2 (build-refs-count-predicate 2)] [count1 (build-refs-count-predicate 1)]) (and (count2 blk) (not (count1 blk)))) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },
    {
        "fn": "build-refs-count-predicate",
        "prompt": "Use or-any with predicates count=1 or count=3 on a three-ref block.",
        "gt": "(let* ([blk (make-test-block 'r (string->utf8 \"\") (vector (string->utf8 \"a\") (string->utf8 \"b\") (string->utf8 \"c\")))] [preds (list (build-refs-count-predicate 1) (build-refs-count-predicate 3))]) (or-any preds blk))",
        "verify": "(equal? (let* ([blk (make-test-block 'r (string->utf8 \"\") (vector (string->utf8 \"a\") (string->utf8 \"b\") (string->utf8 \"c\")))] [preds (list (build-refs-count-predicate 1) (build-refs-count-predicate 3))]) (or-any preds blk)) #t)",
        "difficulty": "medium",
        "tags": ["integration"],
    },
    {
        "fn": "build-refs-count-predicate",
        "prompt": "Evaluate refs-count=0 on a block with no references.",
        "gt": "((build-refs-count-predicate 0) (make-test-block 'x (string->utf8 \"\") (vector)))",
        "verify": "(equal? ((build-refs-count-predicate 0) (make-test-block 'x (string->utf8 \"\") (vector))) #t)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },

    {
        "fn": "build-refs-to-predicate",
        "prompt": "Return whether a relation block references target hash `h`.",
        "gt": "(let* ([h (string->utf8 \"h\")] [pred (build-refs-to-predicate h)] [blk (make-test-block 'relation (string->utf8 \"\") (vector (string->utf8 \"x\") h))]) (pred blk))",
        "verify": "(equal? (let* ([h (string->utf8 \"h\")] [pred (build-refs-to-predicate h)] [blk (make-test-block 'relation (string->utf8 \"\") (vector (string->utf8 \"x\") h))]) (pred blk)) #t)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "build-refs-to-predicate",
        "prompt": "Check that refs-to predicate returns false when target hash is absent.",
        "gt": "(let* ([h (string->utf8 \"h\")] [pred (build-refs-to-predicate h)] [blk (make-test-block 'relation (string->utf8 \"\") (vector (string->utf8 \"x\") (string->utf8 \"y\")))]) (pred blk))",
        "verify": "(equal? (let* ([h (string->utf8 \"h\")] [pred (build-refs-to-predicate h)] [blk (make-test-block 'relation (string->utf8 \"\") (vector (string->utf8 \"x\") (string->utf8 \"y\")))]) (pred blk)) #f)",
        "difficulty": "medium",
        "tags": ["edge-case"],
    },
    {
        "fn": "build-refs-to-predicate",
        "prompt": "Run refs-to predicate against two blocks and return boolean results.",
        "gt": "(let* ([h (string->utf8 \"h\")] [pred (build-refs-to-predicate h)]) (list (pred (make-test-block 'r (string->utf8 \"\") (vector h))) (pred (make-test-block 'r (string->utf8 \"\") (vector (string->utf8 \"z\"))))))",
        "verify": "(equal? (let* ([h (string->utf8 \"h\")] [pred (build-refs-to-predicate h)]) (list (pred (make-test-block 'r (string->utf8 \"\") (vector h))) (pred (make-test-block 'r (string->utf8 \"\") (vector (string->utf8 \"z\")))))) '(#t #f))",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "build-refs-to-predicate",
        "prompt": "Return whether a block satisfies both refs-to(target) and refs-count=2.",
        "gt": "(let* ([h (string->utf8 \"h\")] [blk (make-test-block 'r (string->utf8 \"\") (vector h (string->utf8 \"x\")))] [preds (list (build-refs-to-predicate h) (build-refs-count-predicate 2))]) (and-all preds blk))",
        "verify": "(equal? (let* ([h (string->utf8 \"h\")] [blk (make-test-block 'r (string->utf8 \"\") (vector h (string->utf8 \"x\")))] [preds (list (build-refs-to-predicate h) (build-refs-count-predicate 2))]) (and-all preds blk)) #t)",
        "difficulty": "hard",
        "tags": ["integration"],
    },

    {
        "fn": "build-payload-size-predicate",
        "prompt": "Apply payload-size>3 predicate to payloads \"ab\" and \"abcd\".",
        "gt": "(let ([pred (build-payload-size-predicate > 3)]) (list (pred (make-test-block 'doc (string->utf8 \"ab\") (vector))) (pred (make-test-block 'doc (string->utf8 \"abcd\") (vector)))))",
        "verify": "(equal? (let ([pred (build-payload-size-predicate > 3)]) (list (pred (make-test-block 'doc (string->utf8 \"ab\") (vector))) (pred (make-test-block 'doc (string->utf8 \"abcd\") (vector))))) '(#f #t))",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "build-payload-size-predicate",
        "prompt": "Evaluate payload-size=5 predicate on payload \"hello\".",
        "gt": "((build-payload-size-predicate = 5) (make-test-block 'doc (string->utf8 \"hello\") (vector)))",
        "verify": "(equal? ((build-payload-size-predicate = 5) (make-test-block 'doc (string->utf8 \"hello\") (vector))) #t)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "build-payload-size-predicate",
        "prompt": "Evaluate payload-size<4 predicate on payload \"abcd\".",
        "gt": "((build-payload-size-predicate < 4) (make-test-block 'doc (string->utf8 \"abcd\") (vector)))",
        "verify": "(equal? ((build-payload-size-predicate < 4) (make-test-block 'doc (string->utf8 \"abcd\") (vector))) #f)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "build-payload-size-predicate",
        "prompt": "Use and-all to require tag=doc and payload-size>2 on payload \"cat\".",
        "gt": "(let* ([blk (make-test-block 'doc (string->utf8 \"cat\") (vector))] [preds (list (build-tag-predicate 'doc) (build-payload-size-predicate > 2))]) (and-all preds blk))",
        "verify": "(equal? (let* ([blk (make-test-block 'doc (string->utf8 \"cat\") (vector))] [preds (list (build-tag-predicate 'doc) (build-payload-size-predicate > 2))]) (and-all preds blk)) #t)",
        "difficulty": "medium",
        "tags": ["integration"],
    },

    {
        "fn": "interpret-match",
        "prompt": "Interpret `(tag . entity)` and evaluate it on an entity block.",
        "gt": "((interpret-match '(tag . entity)) (make-test-block 'entity (string->utf8 \"x\") (vector)))",
        "verify": "(equal? ((interpret-match '(tag . entity)) (make-test-block 'entity (string->utf8 \"x\") (vector))) #t)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "interpret-match",
        "prompt": "Interpret `(payload-contains . \"ana\")` and evaluate on payloads \"banana\" and \"cider\".",
        "gt": "(let ([pred (interpret-match '(payload-contains . \"ana\"))]) (list (pred (make-test-block 'doc (string->utf8 \"banana\") (vector))) (pred (make-test-block 'doc (string->utf8 \"cider\") (vector)))))",
        "verify": "(equal? (let ([pred (interpret-match '(payload-contains . \"ana\"))]) (list (pred (make-test-block 'doc (string->utf8 \"banana\") (vector))) (pred (make-test-block 'doc (string->utf8 \"cider\") (vector))))) '(#t #f))",
        "difficulty": "hard",
        "tags": ["direct"],
    },
    {
        "fn": "interpret-match",
        "prompt": "Interpret a payload-matches pattern that checks for substring \"bc\" in decoded payload text.",
        "gt": "(let* ([m (lambda (s) (query-string-contains? s \"bc\"))] [pred (interpret-match (cons 'payload-matches m))]) (list (pred (make-test-block 'doc (string->utf8 \"abcde\") (vector))) (pred (make-test-block 'doc (string->utf8 \"ax\") (vector)))))",
        "verify": "(equal? (let* ([m (lambda (s) (query-string-contains? s \"bc\"))] [pred (interpret-match (cons 'payload-matches m))]) (list (pred (make-test-block 'doc (string->utf8 \"abcde\") (vector))) (pred (make-test-block 'doc (string->utf8 \"ax\") (vector))))) '(#t #f))",
        "difficulty": "hard",
        "tags": ["integration"],
    },
    {
        "fn": "interpret-match",
        "prompt": "Return whether interpreting an unknown pattern raises an exception.",
        "gt": "(guard (ex [else #t]) (begin (interpret-match '(bad-key . 1)) #f))",
        "verify": "(equal? (guard (ex [else #t]) (begin (interpret-match '(bad-key . 1)) #f)) #t)",
        "difficulty": "hard",
        "tags": ["edge-case"],
    },

    {
        "fn": "and-all",
        "prompt": "Use and-all with predicates (>0), even?, and (<10) on 8.",
        "gt": "(and-all (list (lambda (n) (> n 0)) (lambda (n) (even? n)) (lambda (n) (< n 10))) 8)",
        "verify": "(equal? (and-all (list (lambda (n) (> n 0)) (lambda (n) (even? n)) (lambda (n) (< n 10))) 8) #t)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "and-all",
        "prompt": "Use and-all with predicates (>0), even?, and (<10) on -2.",
        "gt": "(and-all (list (lambda (n) (> n 0)) (lambda (n) (even? n)) (lambda (n) (< n 10))) -2)",
        "verify": "(equal? (and-all (list (lambda (n) (> n 0)) (lambda (n) (even? n)) (lambda (n) (< n 10))) -2) #f)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "and-all",
        "prompt": "Evaluate and-all with an empty predicate list and any value.",
        "gt": "(and-all '() 99)",
        "verify": "(equal? (and-all '() 99) #t)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "and-all",
        "prompt": "Return whether a block passes tag=entity and refs-count=1 using and-all.",
        "gt": "(let* ([blk (make-test-block 'entity (string->utf8 \"x\") (vector (string->utf8 \"h\")))] [preds (list (build-tag-predicate 'entity) (build-refs-count-predicate 1))]) (and-all preds blk))",
        "verify": "(equal? (let* ([blk (make-test-block 'entity (string->utf8 \"x\") (vector (string->utf8 \"h\")))] [preds (list (build-tag-predicate 'entity) (build-refs-count-predicate 1))]) (and-all preds blk)) #t)",
        "difficulty": "hard",
        "tags": ["integration"],
    },

    {
        "fn": "or-any",
        "prompt": "Use or-any with predicates (>10) and even? on 11.",
        "gt": "(or-any (list (lambda (n) (> n 10)) (lambda (n) (even? n))) 11)",
        "verify": "(equal? (or-any (list (lambda (n) (> n 10)) (lambda (n) (even? n))) 11) #t)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "or-any",
        "prompt": "Use or-any with predicates (<0) and even? on 7.",
        "gt": "(or-any (list (lambda (n) (< n 0)) (lambda (n) (even? n))) 7)",
        "verify": "(equal? (or-any (list (lambda (n) (< n 0)) (lambda (n) (even? n))) 7) #f)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "or-any",
        "prompt": "Evaluate or-any with an empty predicate list.",
        "gt": "(or-any '() 'unused)",
        "verify": "(equal? (or-any '() 'unused) #f)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "or-any",
        "prompt": "Return whether a block satisfies tag=entity or refs-to(target) using or-any.",
        "gt": "(let* ([target (string->utf8 \"target\")] [blk (make-test-block 'relation (string->utf8 \"\") (vector target))] [preds (list (build-tag-predicate 'entity) (build-refs-to-predicate target))]) (or-any preds blk))",
        "verify": "(equal? (let* ([target (string->utf8 \"target\")] [blk (make-test-block 'relation (string->utf8 \"\") (vector target))] [preds (list (build-tag-predicate 'entity) (build-refs-to-predicate target))]) (or-any preds blk)) #t)",
        "difficulty": "hard",
        "tags": ["integration"],
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

    wrapped_checks = [build_verify(check, [fn]) for check in selected_checks]
    sample["verify_expr"] = (
        f"(and {str(sample['verify_expr']).strip()} {' '.join(wrapped_checks)})"
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

# -----------------------------------------------------------------------------
# Split train/eval
# -----------------------------------------------------------------------------
by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

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

id_to_sample: Dict[str, Dict[str, object]] = {str(s["id"]): s for s in samples}
all_source_functions = sorted({str(s["source_function"]) for s in samples})

missing_after = [
    fn
    for fn in all_source_functions
    if not any(str(id_to_sample[sid]["source_function"]) == fn for sid in eval_ids)
]
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

if not train_rows or not eval_rows:
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
    "difficulty": dict(sorted(Counter(str(s["difficulty"]) for s in samples).items())),
    "source_functions": dict(sorted(Counter(str(s["source_function"]) for s in samples).items())),
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
