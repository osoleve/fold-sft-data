#!/usr/bin/env python3
"""Generate Tier-0 SFT samples for lattice/data/collection-protocol.ss."""

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

SOURCE_MODULE = "lattice/data/collection-protocol.ss"
SOURCE_TEST = "lattice/data/test-collection-protocol.ss"

DEFS: Dict[str, str] = {
    "coll-count": """(define (coll-count coll pred)
  (coll-fold coll
             (lambda (acc elem)
               (if (pred elem) (+ acc 1) acc))
             0))""",
    "coll-any?": """(define (coll-any? coll pred)
  (if (coll-fold coll
                 (lambda (acc elem)
                   (or acc (pred elem)))
                 #f)
      #t #f))""",
    "coll-all?": """(define (coll-all? coll pred)
  (if (coll-fold coll
                 (lambda (acc elem)
                   (and acc (pred elem)))
                 #t)
      #t #f))""",
    "coll-filter-list": """(define (coll-filter-list coll pred)
  (reverse
   (coll-fold coll
              (lambda (acc elem)
                (if (pred elem) (cons elem acc) acc))
              '())))""",
    "coll-map-list": """(define (coll-map-list coll fn)
  (reverse
   (coll-fold coll
              (lambda (acc elem)
                (cons (fn elem) acc))
              '())))""",
    "coll-find": """(define (coll-find coll pred)
  (coll-fold coll
             (lambda (acc elem)
               (if acc acc
                   (if (pred elem) elem #f)))
             #f))""",
    "coll-partition": """(define (coll-partition coll pred)
  (let ([result (coll-fold coll
                           (lambda (acc elem)
                             (if (pred elem)
                                 (cons (cons elem (car acc)) (cdr acc))
                                 (cons (car acc) (cons elem (cdr acc)))))
                           (cons '() '()))])
    (values (reverse (car result)) (reverse (cdr result)))))""",
    "coll-protocols": """(define (coll-protocols type-tag)
  (filter (lambda (proto)
            (type-implements? type-tag proto))
          '(coll-empty? coll-size coll-fold coll-to-list
            keyed-lookup keyed-insert keyed-delete keyed-contains?
            keyed-keys keyed-values
            spatial-nearest spatial-knn spatial-range spatial-radius spatial-contains?
            prio-peek prio-pop prio-insert prio-merge)))""",
}

SUPPORT_DEFS: Dict[str, str] = {
    "build-avl": """(define (build-avl pairs)
  (fold-left (lambda (tree pair)
               (avl-insert (car pair) (cdr pair) tree))
             avl-empty
             pairs))""",
    "pair-keys": """(define (pair-keys pairs)
  (map car pairs))""",
    "pair-values": """(define (pair-values pairs)
  (map cdr pairs))""",
}

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **DEFS}

DEPENDS: Dict[str, List[str]] = {
    "build-avl": [],
    "pair-keys": [],
    "pair-values": [],
    "coll-count": [],
    "coll-any?": [],
    "coll-all?": [],
    "coll-filter-list": [],
    "coll-map-list": [],
    "coll-find": [],
    "coll-partition": [],
    "coll-protocols": [],
}

FUNCTION_ORDER = [
    "coll-count",
    "coll-any?",
    "coll-all?",
    "coll-filter-list",
    "coll-map-list",
    "coll-find",
    "coll-partition",
    "coll-protocols",
]

SUPPORT_ORDER = [
    "build-avl",
    "pair-keys",
    "pair-values",
]

FUNCTION_SPECS = {
    "coll-count": "Count collection elements that satisfy predicate `pred`.",
    "coll-any?": "Return #t iff at least one element satisfies predicate `pred`.",
    "coll-all?": "Return #t iff every element satisfies predicate `pred`.",
    "coll-filter-list": "Return a list of elements that satisfy predicate `pred` preserving fold traversal order.",
    "coll-map-list": "Map `fn` over collection elements and return a list in traversal order.",
    "coll-find": "Return first element satisfying `pred`, otherwise #f.",
    "coll-partition": "Partition into two lists `(matching, non-matching)` while preserving traversal order.",
    "coll-protocols": "Return protocol symbols implemented by `type-tag` via `type-implements?` over the canonical protocol list.",
}

SKELETONS = {
    "coll-count": """(define (coll-count coll pred)
  ;; TODO: count elements where pred returns #t
  <TODO>)""",
    "coll-any?": """(define (coll-any? coll pred)
  ;; TODO: return #t if any element satisfies pred
  <TODO>)""",
    "coll-all?": """(define (coll-all? coll pred)
  ;; TODO: return #t only when all elements satisfy pred
  <TODO>)""",
    "coll-filter-list": """(define (coll-filter-list coll pred)
  ;; TODO: collect matching elements into a list preserving traversal order
  <TODO>)""",
    "coll-map-list": """(define (coll-map-list coll fn)
  ;; TODO: map each element with fn and return a list in traversal order
  <TODO>)""",
    "coll-find": """(define (coll-find coll pred)
  ;; TODO: return first matching element or #f
  <TODO>)""",
    "coll-partition": """(define (coll-partition coll pred)
  ;; TODO: return (values matching-list non-matching-list)
  <TODO>)""",
    "coll-protocols": """(define (coll-protocols type-tag)
  ;; TODO: filter canonical protocol symbols by (type-implements? type-tag proto)
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "coll-count": "(let* ([heap (list->heap '(3 1 4 1 5 9 2 6))] [tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (5 . \"five\")))]) (and (= (coll-count heap (lambda (x) (> x 4))) 3) (= (coll-count tree (lambda (kv) (<= (car kv) 3))) 2)))",
    "coll-any?": "(let ([heap (list->heap '(3 1 4 1 5))]) (and (coll-any? heap (lambda (x) (= x 4))) (not (coll-any? heap (lambda (x) (= x 8))))))",
    "coll-all?": "(let ([heap (list->heap '(3 1 4 1 5))]) (and (coll-all? heap (lambda (x) (> x 0))) (not (coll-all? heap (lambda (x) (> x 2)))) (coll-all? heap-empty (lambda (x) #f))))",
    "coll-filter-list": "(let* ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (5 . \"five\") (2 . \"two\")))] [out (coll-filter-list tree (lambda (kv) (odd? (car kv))))]) (and (equal? (pair-keys out) '(1 3 5)) (equal? (pair-values out) '(\"one\" \"three\" \"five\"))))",
    "coll-map-list": "(let* ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\")))] [keys (coll-map-list tree (lambda (kv) (car kv)))]) (equal? keys '(1 3 4)))",
    "coll-find": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (5 . \"five\")))]) (and (equal? (coll-find tree (lambda (kv) (> (car kv) 3))) '(4 . \"four\")) (not (coll-find tree (lambda (kv) (= (car kv) 99))))))",
    "coll-partition": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\")))]) (call-with-values (lambda () (coll-partition tree (lambda (kv) (even? (car kv))))) (lambda (yes no) (and (equal? (pair-keys yes) '(2 4)) (equal? (pair-keys no) '(1 3))))))",
    "coll-protocols": "(let ([heap-protos (coll-protocols 'heap-node)] [avl-protos (coll-protocols 'avl-node)] [kdtree-protos (coll-protocols 'kdtree-node)]) (and (and (member 'coll-size heap-protos) #t) (and (member 'prio-peek heap-protos) #t) (not (member 'keyed-lookup heap-protos)) (and (member 'keyed-lookup avl-protos) #t) (and (member 'spatial-nearest kdtree-protos) #t)))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "coll-count": "def coll_count(coll, pred):\n    acc = 0\n    for elem in coll_fold_iter(coll):\n        if pred(elem):\n            acc += 1\n    return acc",
    "coll-any?": "def coll_any(coll, pred):\n    acc = False\n    for elem in coll_fold_iter(coll):\n        acc = acc or pred(elem)\n    return True if acc else False",
    "coll-all?": "def coll_all(coll, pred):\n    acc = True\n    for elem in coll_fold_iter(coll):\n        acc = acc and pred(elem)\n    return True if acc else False",
    "coll-filter-list": "def coll_filter_list(coll, pred):\n    acc = []\n    for elem in coll_fold_iter(coll):\n        if pred(elem):\n            acc.insert(0, elem)\n    return list(reversed(acc))",
    "coll-map-list": "def coll_map_list(coll, fn):\n    acc = []\n    for elem in coll_fold_iter(coll):\n        acc.insert(0, fn(elem))\n    return list(reversed(acc))",
    "coll-find": "def coll_find(coll, pred):\n    acc = None\n    for elem in coll_fold_iter(coll):\n        if acc is not None:\n            acc = acc\n        elif pred(elem):\n            acc = elem\n        else:\n            acc = None\n    return acc",
    "coll-partition": "def coll_partition(coll, pred):\n    yes, no = [], []\n    for elem in coll_fold_iter(coll):\n        if pred(elem):\n            yes.insert(0, elem)\n        else:\n            no.insert(0, elem)\n    return (list(reversed(yes)), list(reversed(no)))",
    "coll-protocols": "def coll_protocols(type_tag):\n    protos = [\n        'coll-empty?', 'coll-size', 'coll-fold', 'coll-to-list',\n        'keyed-lookup', 'keyed-insert', 'keyed-delete', 'keyed-contains?',\n        'keyed-keys', 'keyed-values',\n        'spatial-nearest', 'spatial-knn', 'spatial-range', 'spatial-radius', 'spatial-contains?',\n        'prio-peek', 'prio-pop', 'prio-insert', 'prio-merge',\n    ]\n    return [p for p in protos if type_implements(type_tag, p)]",
}

CHEZ_SNIPPETS = {
    "coll-count": "(define (count0 coll pred)\n  (coll-fold coll\n             (lambda (acc elem)\n               (if (pred elem) (+ acc 1) acc))\n             0))",
    "coll-any?": "(define (any0 coll pred)\n  (if (coll-fold coll\n                 (lambda (acc elem)\n                   (or acc (pred elem)))\n                 #f)\n      #t\n      #f))",
    "coll-all?": "(define (all0 coll pred)\n  (if (coll-fold coll\n                 (lambda (acc elem)\n                   (and acc (pred elem)))\n                 #t)\n      #t\n      #f))",
    "coll-filter-list": "(define (filter-list0 coll pred)\n  (reverse\n   (coll-fold coll\n              (lambda (acc elem)\n                (if (pred elem) (cons elem acc) acc))\n              '())))",
    "coll-map-list": "(define (map-list0 coll fn)\n  (reverse\n   (coll-fold coll\n              (lambda (acc elem)\n                (cons (fn elem) acc))\n              '())))",
    "coll-find": "(define (find0 coll pred)\n  (coll-fold coll\n             (lambda (acc elem)\n               (if acc\n                   acc\n                   (if (pred elem) elem #f)))\n             #f))",
    "coll-partition": "(define (partition0 coll pred)\n  (let ([result (coll-fold coll\n                           (lambda (acc elem)\n                             (if (pred elem)\n                                 (cons (cons elem (car acc)) (cdr acc))\n                                 (cons (car acc) (cons elem (cdr acc)))))\n                           (cons '() '()))])\n    (values (reverse (car result)) (reverse (cdr result)))))",
    "coll-protocols": "(define (protocols0 type-tag)\n  (filter (lambda (proto)\n            (type-implements? type-tag proto))\n          '(coll-empty? coll-size coll-fold coll-to-list\n            keyed-lookup keyed-insert keyed-delete keyed-contains?\n            keyed-keys keyed-values\n            spatial-nearest spatial-knn spatial-range spatial-radius spatial-contains?\n            prio-peek prio-pop prio-insert prio-merge)))",
}

BUGGY_CASES = [
    {
        "fn": "coll-count",
        "buggy": """(define (coll-count coll pred)
  (coll-fold coll
             (lambda (acc elem)
               (+ acc 1))
             0))""",
        "note": "The predicate must gate whether the counter increments.",
    },
    {
        "fn": "coll-count",
        "buggy": """(define (coll-count coll pred)
  (coll-fold coll
             (lambda (acc elem)
               (if (pred elem) (+ acc 1) acc))
             1))""",
        "note": "Fold initialization should start at 0, not 1.",
    },
    {
        "fn": "coll-any?",
        "buggy": """(define (coll-any? coll pred)
  (if (coll-fold coll
                 (lambda (acc elem)
                   (and acc (pred elem)))
                 #f)
      #t #f))""",
        "note": "Any-match logic must use OR accumulation, not AND.",
    },
    {
        "fn": "coll-any?",
        "buggy": """(define (coll-any? coll pred)
  (if (coll-fold coll
                 (lambda (acc elem)
                   (or acc (pred elem)))
                 #t)
      #t #f))""",
        "note": "Starting accumulator at #t makes non-empty and empty collections incorrectly return #t.",
    },
    {
        "fn": "coll-all?",
        "buggy": """(define (coll-all? coll pred)
  (if (coll-fold coll
                 (lambda (acc elem)
                   (or acc (pred elem)))
                 #t)
      #t #f))""",
        "note": "All-match logic must use AND accumulation, not OR.",
    },
    {
        "fn": "coll-all?",
        "buggy": """(define (coll-all? coll pred)
  (if (coll-fold coll
                 (lambda (acc elem)
                   (and acc (pred elem)))
                 #f)
      #t #f))""",
        "note": "Starting accumulator at #f forces false even when all elements satisfy predicate.",
    },
    {
        "fn": "coll-filter-list",
        "buggy": """(define (coll-filter-list coll pred)
  (coll-fold coll
             (lambda (acc elem)
               (if (pred elem) (cons elem acc) acc))
             '()))""",
        "note": "Missing final reverse returns elements in backward traversal order.",
    },
    {
        "fn": "coll-filter-list",
        "buggy": """(define (coll-filter-list coll pred)
  (reverse
   (coll-fold coll
              (lambda (acc elem)
                (if (pred elem) acc (cons elem acc)))
              '())))""",
        "note": "Predicate branch is inverted; this keeps rejected elements instead of matches.",
    },
    {
        "fn": "coll-map-list",
        "buggy": """(define (coll-map-list coll fn)
  (coll-fold coll
             (lambda (acc elem)
               (cons (fn elem) acc))
             '()))""",
        "note": "Missing final reverse breaks expected traversal order.",
    },
    {
        "fn": "coll-map-list",
        "buggy": """(define (coll-map-list coll fn)
  (reverse
   (coll-fold coll
              (lambda (acc elem)
                (cons elem acc))
              '())))""",
        "note": "Mapping function must be applied to each element.",
    },
    {
        "fn": "coll-find",
        "buggy": """(define (coll-find coll pred)
  (coll-fold coll
             (lambda (acc elem)
               (if (pred elem) elem acc))
             #f))""",
        "note": "This returns the last matching element; function should keep the first match.",
    },
    {
        "fn": "coll-find",
        "buggy": """(define (coll-find coll pred)
  (coll-fold coll
             (lambda (acc elem)
               (if acc acc
                   (if (pred elem) #t #f)))
             #f))""",
        "note": "On match, function should return the matching element, not #t.",
    },
    {
        "fn": "coll-partition",
        "buggy": """(define (coll-partition coll pred)
  (let ([result (coll-fold coll
                           (lambda (acc elem)
                             (if (pred elem)
                                 (cons (car acc) (cons elem (cdr acc)))
                                 (cons (cons elem (car acc)) (cdr acc))))
                           (cons '() '()))])
    (values (reverse (car result)) (reverse (cdr result)))))""",
        "note": "Matching and non-matching buckets are swapped in the fold step.",
    },
    {
        "fn": "coll-partition",
        "buggy": """(define (coll-partition coll pred)
  (let ([result (coll-fold coll
                           (lambda (acc elem)
                             (if (pred elem)
                                 (cons (cons elem (car acc)) (cdr acc))
                                 (cons (car acc) (cons elem (cdr acc)))))
                           (cons '() '()))])
    (values (car result) (cdr result))))""",
        "note": "Final lists must be reversed to restore traversal order.",
    },
    {
        "fn": "coll-protocols",
        "buggy": """(define (coll-protocols type-tag)
  (filter (lambda (proto)
            (type-implements? proto type-tag))
          '(coll-empty? coll-size coll-fold coll-to-list
            keyed-lookup keyed-insert keyed-delete keyed-contains?
            keyed-keys keyed-values
            spatial-nearest spatial-knn spatial-range spatial-radius spatial-contains?
            prio-peek prio-pop prio-insert prio-merge)))""",
        "note": "`type-implements?` argument order is `(type-tag proto)`.",
    },
    {
        "fn": "coll-protocols",
        "buggy": """(define (coll-protocols type-tag)
  (filter (lambda (proto)
            (type-implements? type-tag proto))
          '(coll-empty? coll-size coll-fold coll-to-list
            keyed-lookup keyed-insert keyed-delete keyed-contains?
            keyed-keys keyed-values
            spatial-nearest spatial-knn spatial-range spatial-radius spatial-contains?)))""",
        "note": "Canonical protocol set must include priority protocols too.",
    },
]

DIFFICULTY = {
    "coll-count": "easy",
    "coll-any?": "medium",
    "coll-all?": "medium",
    "coll-filter-list": "medium",
    "coll-map-list": "medium",
    "coll-find": "medium",
    "coll-partition": "hard",
    "coll-protocols": "hard",
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
    sid = f"collection_protocol_{family}_{family_counter[family]:03d}"
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


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement this collection protocol helper in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "collection-protocol", "spec-to-code", fn],
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
        tags=["tier0", "data", "collection-protocol", "skeleton-completion", fn],
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
        tags=["tier0", "data", "collection-protocol", "python-to-scheme", fn],
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
        tags=["tier0", "data", "collection-protocol", "chez-to-fold", fn],
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
        tags=["tier0", "data", "collection-protocol", "bugfix", fn],
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
        verify_expr=build_verify(verify_check, [source_function]),
        tags=["tier0", "data", "collection-protocol", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # coll-count
    {
        "fn": "coll-count",
        "prompt": "Count how many heap elements are strictly greater than 4.",
        "gt": "(let ([heap (list->heap '(3 1 4 1 5 9 2 6))]) (coll-count heap (lambda (x) (> x 4))))",
        "verify": "(equal? (let ([heap (list->heap '(3 1 4 1 5 9 2 6))]) (coll-count heap (lambda (x) (> x 4)))) 3)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "coll-count",
        "prompt": "Build an AVL map and count entries with odd keys.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (coll-count tree (lambda (kv) (odd? (car kv)))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (coll-count tree (lambda (kv) (odd? (car kv))))) 3)",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "coll-count",
        "prompt": "Return whether counting all elements equals coll-size for a heap.",
        "gt": "(let ([heap (list->heap '(7 1 8 2 8))]) (= (coll-count heap (lambda (x) #t)) (coll-size heap)))",
        "verify": "(equal? (let ([heap (list->heap '(7 1 8 2 8))]) (= (coll-count heap (lambda (x) #t)) (coll-size heap))) #t)",
        "difficulty": "medium",
        "tags": ["property"],
    },
    {
        "fn": "coll-count",
        "prompt": "Count elements in heap-empty with a predicate that always returns #t.",
        "gt": "(coll-count heap-empty (lambda (x) #t))",
        "verify": "(equal? (coll-count heap-empty (lambda (x) #t)) 0)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },

    # coll-any?
    {
        "fn": "coll-any?",
        "prompt": "Check whether heap '(3 1 4 1 5 9) contains the value 9.",
        "gt": "(let ([heap (list->heap '(3 1 4 1 5 9))]) (coll-any? heap (lambda (x) (= x 9))))",
        "verify": "(equal? (let ([heap (list->heap '(3 1 4 1 5 9))]) (coll-any? heap (lambda (x) (= x 9)))) #t)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "coll-any?",
        "prompt": "Check whether any heap element is greater than 10.",
        "gt": "(let ([heap (list->heap '(3 1 4 1 5 9))]) (coll-any? heap (lambda (x) (> x 10))))",
        "verify": "(equal? (let ([heap (list->heap '(3 1 4 1 5 9))]) (coll-any? heap (lambda (x) (> x 10)))) #f)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "coll-any?",
        "prompt": "Return whether an AVL map has any entry with key 4.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\")))]) (coll-any? tree (lambda (kv) (= (car kv) 4))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\")))]) (coll-any? tree (lambda (kv) (= (car kv) 4)))) #t)",
        "difficulty": "medium",
        "tags": ["integration"],
    },
    {
        "fn": "coll-any?",
        "prompt": "Check coll-any? on avl-empty with predicate that always returns #t.",
        "gt": "(coll-any? avl-empty (lambda (kv) #t))",
        "verify": "(equal? (coll-any? avl-empty (lambda (kv) #t)) #f)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },

    # coll-all?
    {
        "fn": "coll-all?",
        "prompt": "Check whether all values in heap '(3 1 4 1 5 9) are positive.",
        "gt": "(let ([heap (list->heap '(3 1 4 1 5 9))]) (coll-all? heap (lambda (x) (> x 0))))",
        "verify": "(equal? (let ([heap (list->heap '(3 1 4 1 5 9))]) (coll-all? heap (lambda (x) (> x 0)))) #t)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "coll-all?",
        "prompt": "Check whether all values in heap '(3 1 4 1 5 9) are even.",
        "gt": "(let ([heap (list->heap '(3 1 4 1 5 9))]) (coll-all? heap (lambda (x) (even? x))))",
        "verify": "(equal? (let ([heap (list->heap '(3 1 4 1 5 9))]) (coll-all? heap (lambda (x) (even? x)))) #f)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "coll-all?",
        "prompt": "Return whether every key in an AVL map is <= 5.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (coll-all? tree (lambda (kv) (<= (car kv) 5))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (coll-all? tree (lambda (kv) (<= (car kv) 5)))) #t)",
        "difficulty": "medium",
        "tags": ["integration"],
    },
    {
        "fn": "coll-all?",
        "prompt": "Check coll-all? on empty KDTree with a predicate that always returns #f.",
        "gt": "(coll-all? kdtree-empty (lambda (pt) #f))",
        "verify": "(equal? (coll-all? kdtree-empty (lambda (pt) #f)) #t)",
        "difficulty": "medium",
        "tags": ["edge-case"],
    },

    # coll-filter-list
    {
        "fn": "coll-filter-list",
        "prompt": "Filter AVL entries to keys greater than 2, then return the keys.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (pair-keys (coll-filter-list tree (lambda (kv) (> (car kv) 2)))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (pair-keys (coll-filter-list tree (lambda (kv) (> (car kv) 2))))) '(3 4 5))",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "coll-filter-list",
        "prompt": "Filter AVL entries whose value string length is 4, then return keys.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (pair-keys (coll-filter-list tree (lambda (kv) (= (string-length (cdr kv)) 4)))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (pair-keys (coll-filter-list tree (lambda (kv) (= (string-length (cdr kv)) 4))))) '(4 5))",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "coll-filter-list",
        "prompt": "Filter an AVL map with a predicate that never matches.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\")))]) (coll-filter-list tree (lambda (kv) #f)))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\")))]) (coll-filter-list tree (lambda (kv) #f))) '())",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "coll-filter-list",
        "prompt": "Return the number of AVL entries with keys <= 3 using coll-filter-list.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (length (coll-filter-list tree (lambda (kv) (<= (car kv) 3)))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (length (coll-filter-list tree (lambda (kv) (<= (car kv) 3))))) 3)",
        "difficulty": "medium",
        "tags": ["integration"],
    },

    # coll-map-list
    {
        "fn": "coll-map-list",
        "prompt": "Map an AVL map to its sorted key list.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (coll-map-list tree (lambda (kv) (car kv))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (coll-map-list tree (lambda (kv) (car kv)))) '(1 2 3 4 5))",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "coll-map-list",
        "prompt": "Double each heap element with coll-map-list and sum the resulting list.",
        "gt": "(apply + (coll-map-list (list->heap '(3 1 4)) (lambda (x) (* 2 x))))",
        "verify": "(equal? (apply + (coll-map-list (list->heap '(3 1 4)) (lambda (x) (* 2 x)))) 16)",
        "difficulty": "medium",
        "tags": ["integration"],
    },
    {
        "fn": "coll-map-list",
        "prompt": "Map over heap-empty and return the resulting list.",
        "gt": "(coll-map-list heap-empty (lambda (x) (+ x 1)))",
        "verify": "(equal? (coll-map-list heap-empty (lambda (x) (+ x 1))) '())",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },
    {
        "fn": "coll-map-list",
        "prompt": "Map AVL entries to value-string lengths.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (coll-map-list tree (lambda (kv) (string-length (cdr kv)))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (coll-map-list tree (lambda (kv) (string-length (cdr kv))))) '(3 3 5 4 4))",
        "difficulty": "medium",
        "tags": ["direct"],
    },

    # coll-find
    {
        "fn": "coll-find",
        "prompt": "Find the first AVL entry whose key is greater than 2.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (coll-find tree (lambda (kv) (> (car kv) 2))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (coll-find tree (lambda (kv) (> (car kv) 2)))) '(3 . \"three\"))",
        "difficulty": "medium",
        "tags": ["direct"],
    },
    {
        "fn": "coll-find",
        "prompt": "Attempt to find AVL entry with key 42.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\")))]) (coll-find tree (lambda (kv) (= (car kv) 42))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\")))]) (coll-find tree (lambda (kv) (= (car kv) 42)))) #f)",
        "difficulty": "easy",
        "tags": ["direct"],
    },
    {
        "fn": "coll-find",
        "prompt": "Find the first AVL entry whose value-string length is 4.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (coll-find tree (lambda (kv) (= (string-length (cdr kv)) 4))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (coll-find tree (lambda (kv) (= (string-length (cdr kv)) 4)))) '(4 . \"four\"))",
        "difficulty": "medium",
        "tags": ["integration"],
    },
    {
        "fn": "coll-find",
        "prompt": "Run coll-find on heap-empty with predicate (> x 0).",
        "gt": "(coll-find heap-empty (lambda (x) (> x 0)))",
        "verify": "(equal? (coll-find heap-empty (lambda (x) (> x 0))) #f)",
        "difficulty": "easy",
        "tags": ["edge-case"],
    },

    # coll-partition
    {
        "fn": "coll-partition",
        "prompt": "Partition AVL entries by even keys and return key lists as '(yes no).",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (call-with-values (lambda () (coll-partition tree (lambda (kv) (even? (car kv))))) (lambda (yes no) (list (pair-keys yes) (pair-keys no)))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (call-with-values (lambda () (coll-partition tree (lambda (kv) (even? (car kv))))) (lambda (yes no) (list (pair-keys yes) (pair-keys no))))) '((2 4) (1 3 5)))",
        "difficulty": "hard",
        "tags": ["direct"],
    },
    {
        "fn": "coll-partition",
        "prompt": "Partition heap values by (> x 4) and return '(match-count non-match-count).",
        "gt": "(call-with-values (lambda () (coll-partition (list->heap '(3 1 4 1 5 9 2 6)) (lambda (x) (> x 4)))) (lambda (yes no) (list (length yes) (length no))))",
        "verify": "(equal? (call-with-values (lambda () (coll-partition (list->heap '(3 1 4 1 5 9 2 6)) (lambda (x) (> x 4)))) (lambda (yes no) (list (length yes) (length no)))) '(3 5))",
        "difficulty": "hard",
        "tags": ["integration"],
    },
    {
        "fn": "coll-partition",
        "prompt": "Check that partitioning avl-empty yields two empty lists.",
        "gt": "(call-with-values (lambda () (coll-partition avl-empty (lambda (kv) #t))) (lambda (yes no) (and (null? yes) (null? no))))",
        "verify": "(equal? (call-with-values (lambda () (coll-partition avl-empty (lambda (kv) #t))) (lambda (yes no) (and (null? yes) (null? no)))) #t)",
        "difficulty": "medium",
        "tags": ["edge-case"],
    },
    {
        "fn": "coll-partition",
        "prompt": "Check that partition output sizes add up to coll-size for an AVL map.",
        "gt": "(let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (call-with-values (lambda () (coll-partition tree (lambda (kv) (<= (car kv) 3)))) (lambda (yes no) (= (+ (length yes) (length no)) (coll-size tree)))))",
        "verify": "(equal? (let ([tree (build-avl '((3 . \"three\") (1 . \"one\") (4 . \"four\") (2 . \"two\") (5 . \"five\")))]) (call-with-values (lambda () (coll-partition tree (lambda (kv) (<= (car kv) 3)))) (lambda (yes no) (= (+ (length yes) (length no)) (coll-size tree))))) #t)",
        "difficulty": "hard",
        "tags": ["property"],
    },

    # coll-protocols
    {
        "fn": "coll-protocols",
        "prompt": "Check that heap-node protocols include coll-size and prio-pop, but not keyed-lookup.",
        "gt": "(let ([protos (coll-protocols 'heap-node)]) (and (and (member 'coll-size protos) #t) (and (member 'prio-pop protos) #t) (not (member 'keyed-lookup protos))))",
        "verify": "(equal? (let ([protos (coll-protocols 'heap-node)]) (and (and (member 'coll-size protos) #t) (and (member 'prio-pop protos) #t) (not (member 'keyed-lookup protos)))) #t)",
        "difficulty": "hard",
        "tags": ["direct"],
    },
    {
        "fn": "coll-protocols",
        "prompt": "Check AVL protocol coverage: keyed-lookup + keyed-values present, spatial-nearest absent.",
        "gt": "(let ([protos (coll-protocols 'avl-node)]) (and (and (member 'keyed-lookup protos) #t) (and (member 'keyed-values protos) #t) (not (member 'spatial-nearest protos))))",
        "verify": "(equal? (let ([protos (coll-protocols 'avl-node)]) (and (and (member 'keyed-lookup protos) #t) (and (member 'keyed-values protos) #t) (not (member 'spatial-nearest protos)))) #t)",
        "difficulty": "hard",
        "tags": ["direct"],
    },
    {
        "fn": "coll-protocols",
        "prompt": "Check KDTree protocol coverage: spatial-range present and prio-peek absent.",
        "gt": "(let ([protos (coll-protocols 'kdtree-node)]) (and (and (member 'spatial-range protos) #t) (and (member 'coll-fold protos) #t) (not (member 'prio-peek protos))))",
        "verify": "(equal? (let ([protos (coll-protocols 'kdtree-node)]) (and (and (member 'spatial-range protos) #t) (and (member 'coll-fold protos) #t) (not (member 'prio-peek protos)))) #t)",
        "difficulty": "hard",
        "tags": ["integration"],
    },
    {
        "fn": "coll-protocols",
        "prompt": "Return whether an unknown type tag reports no implemented protocols.",
        "gt": "(null? (coll-protocols 'not-a-type))",
        "verify": "(equal? (null? (coll-protocols 'not-a-type)) #t)",
        "difficulty": "medium",
        "tags": ["edge-case"],
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
        if swapped:
            break

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
    "difficulty": dict(sorted(Counter(str(s["difficulty"]) for s in samples).items())),
    "source_functions": dict(sorted(Counter(str(s["source_function"]) for s in samples).items())),
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
