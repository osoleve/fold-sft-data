#!/usr/bin/env python3
"""Generate Tier-0 SFT samples for lattice/data/dict.ss."""

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

SOURCE_MODULE = "lattice/data/dict.ss"
SOURCE_TEST = "lattice/data/test-data-structures.ss"

DEFS: Dict[str, str] = {
    "dict-empty?": """(define (dict-empty? dict)
  (null? dict))""",
    "dict-lookup": """(define (dict-lookup key dict)
  (let ([pair (assoc key dict)])
    (if pair
        (cdr pair)
        #f)))""",
    "dict-has-key?": """(define (dict-has-key? key dict)
  (if (assoc key dict) #t #f))""",
    "dict-assoc": """(define (dict-assoc key value dict)
  (cons (cons key value)
        (dict-dissoc key dict)))""",
    "dict-dissoc": """(define (dict-dissoc key dict)
  (let loop ([remaining dict]
             [acc '()])
    (cond
      [(null? remaining) (reverse acc)]
      [(equal? key (car (car remaining)))
       (append (reverse acc) (cdr remaining))]
      [else (loop (cdr remaining) (cons (car remaining) acc))])))""",
    "dict-keys": """(define (dict-keys dict)
  (map car dict))""",
    "dict-values": """(define (dict-values dict)
  (map cdr dict))""",
    "dict-entries": """(define (dict-entries dict)
  dict)""",
    "dict-merge": """(define (dict-merge dict1 dict2)
  (let loop ([remaining dict2]
             [result dict1])
    (if (null? remaining)
        result
        (loop (cdr remaining)
              (dict-assoc (car (car remaining))
                          (cdr (car remaining))
                          result)))))""",
    "dict-map-values": """(define (dict-map-values f dict)
  (map (lambda (pair)
         (cons (car pair) (f (cdr pair))))
       dict))""",
    "dict-filter": """(define (dict-filter pred dict)
  (let loop ([remaining dict]
             [acc '()])
    (cond
      [(null? remaining) (reverse acc)]
      [(pred (car (car remaining)) (cdr (car remaining)))
       (loop (cdr remaining) (cons (car remaining) acc))]
      [else (loop (cdr remaining) acc)])))""",
    "dict-size": """(define (dict-size dict)
  (length dict))""",
    "dict->alist": """(define (dict->alist dict)
  dict)""",
    "alist->dict": """(define (alist->dict alist)
  alist)""",
}

DEPENDS: Dict[str, List[str]] = {
    "dict-empty?": [],
    "dict-lookup": [],
    "dict-has-key?": [],
    "dict-assoc": ["dict-dissoc"],
    "dict-dissoc": [],
    "dict-keys": [],
    "dict-values": [],
    "dict-entries": [],
    "dict-merge": ["dict-assoc", "dict-dissoc"],
    "dict-map-values": [],
    "dict-filter": [],
    "dict-size": [],
    "dict->alist": [],
    "alist->dict": [],
}

FUNCTION_ORDER = [
    "dict-empty?",
    "dict-lookup",
    "dict-has-key?",
    "dict-assoc",
    "dict-dissoc",
    "dict-keys",
    "dict-values",
    "dict-entries",
    "dict-merge",
    "dict-map-values",
    "dict-filter",
    "dict-size",
    "dict->alist",
    "alist->dict",
]

FUNCTION_SPECS = {
    "dict-empty?": "Return #t iff dictionary has zero entries.",
    "dict-lookup": "Lookup key and return value, else #f if key is absent.",
    "dict-has-key?": "Return #t iff key exists in dictionary.",
    "dict-assoc": "Associate key with value; replace existing key binding if present.",
    "dict-dissoc": "Remove key from dictionary, leaving other entries unchanged.",
    "dict-keys": "Return list of dictionary keys.",
    "dict-values": "Return list of dictionary values.",
    "dict-entries": "Return list of key-value pairs.",
    "dict-merge": "Merge dictionaries; for overlapping keys, dict2 values win.",
    "dict-map-values": "Apply function to each value while preserving keys.",
    "dict-filter": "Keep entries where predicate (key value) returns true.",
    "dict-size": "Return number of key-value pairs.",
    "dict->alist": "Convert dictionary to association list (identity).",
    "alist->dict": "Convert association list to dictionary (identity).",
}

SKELETONS = {
    "dict-empty?": """(define (dict-empty? dict)
  ;; TODO: return whether dictionary is empty
  <TODO>)""",
    "dict-lookup": """(define (dict-lookup key dict)
  ;; TODO: return value for key, else #f
  <TODO>)""",
    "dict-has-key?": """(define (dict-has-key? key dict)
  ;; TODO: return whether key exists
  <TODO>)""",
    "dict-assoc": """(define (dict-assoc key value dict)
  ;; TODO: upsert key-value pair
  <TODO>)""",
    "dict-dissoc": """(define (dict-dissoc key dict)
  ;; TODO: remove key while preserving other entries
  <TODO>)""",
    "dict-keys": """(define (dict-keys dict)
  ;; TODO: extract keys
  <TODO>)""",
    "dict-values": """(define (dict-values dict)
  ;; TODO: extract values
  <TODO>)""",
    "dict-entries": """(define (dict-entries dict)
  ;; TODO: return entries
  <TODO>)""",
    "dict-merge": """(define (dict-merge dict1 dict2)
  ;; TODO: merge dictionaries with dict2 precedence
  <TODO>)""",
    "dict-map-values": """(define (dict-map-values f dict)
  ;; TODO: map function over values
  <TODO>)""",
    "dict-filter": """(define (dict-filter pred dict)
  ;; TODO: keep entries satisfying pred
  <TODO>)""",
    "dict-size": """(define (dict-size dict)
  ;; TODO: return entry count
  <TODO>)""",
    "dict->alist": """(define (dict->alist dict)
  ;; TODO: convert dict to alist
  <TODO>)""",
    "alist->dict": """(define (alist->dict alist)
  ;; TODO: convert alist to dict
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "dict-empty?": "(and (dict-empty? '()) (not (dict-empty? '((a . 1)))))",
    "dict-lookup": "(and (= (dict-lookup 'a '((a . 10) (b . 20))) 10) (not (dict-lookup 'z '((a . 10) (b . 20)))))",
    "dict-has-key?": "(and (dict-has-key? 'a '((a . 1))) (dict-has-key? 'a '((a . #f))) (not (dict-has-key? 'b '((a . 1)))))",
    "dict-assoc": "(let* ([d1 (dict-assoc 'a 1 '((b . 2)))] [d2 (dict-assoc 'b 9 '((b . 2) (a . 1)))]) (and (= (dict-lookup 'a d1) 1) (= (dict-size d1) 2) (= (dict-lookup 'b d2) 9) (= (dict-size d2) 2)))",
    "dict-dissoc": "(let ([d (dict-dissoc 'b '((a . 1) (b . 2) (c . 3)))]) (and (not (dict-has-key? 'b d)) (= (dict-lookup 'a d) 1) (= (dict-lookup 'c d) 3)))",
    "dict-keys": "(equal? (dict-keys '((a . 1) (b . 2) (c . 3))) '(a b c))",
    "dict-values": "(equal? (dict-values '((a . 1) (b . 2) (c . 3))) '(1 2 3))",
    "dict-entries": "(equal? (dict-entries '((x . 7) (y . 8))) '((x . 7) (y . 8)))",
    "dict-merge": "(let* ([d1 '((a . 1) (b . 2))] [d2 '((b . 99) (c . 3))] [m (dict-merge d1 d2)]) (and (= (dict-size m) 3) (= (dict-lookup 'a m) 1) (= (dict-lookup 'b m) 99) (= (dict-lookup 'c m) 3)))",
    "dict-map-values": "(equal? (dict-map-values (lambda (v) (* 2 v)) '((x . 10) (y . 20))) '((x . 20) (y . 40)))",
    "dict-filter": "(equal? (dict-filter (lambda (k v) (> v 1)) '((a . 1) (b . 2) (c . 3))) '((b . 2) (c . 3)))",
    "dict-size": "(and (= (dict-size '()) 0) (= (dict-size '((a . 1) (b . 2))) 2))",
    "dict->alist": "(equal? (dict->alist '((a . 1) (b . 2))) '((a . 1) (b . 2)))",
    "alist->dict": "(equal? (alist->dict '((a . 1) (b . 2))) '((a . 1) (b . 2)))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "dict-empty?": "def dict_empty(d):\n    return len(d) == 0",
    "dict-lookup": "def dict_lookup(key, d):\n    return d[key] if key in d else None",
    "dict-has-key?": "def dict_has_key(key, d):\n    return key in d",
    "dict-assoc": "def dict_assoc(key, value, d):\n    out = {k: v for k, v in d.items() if k != key}\n    out[key] = value\n    return out",
    "dict-dissoc": "def dict_dissoc(key, d):\n    return {k: v for k, v in d.items() if k != key}",
    "dict-keys": "def dict_keys(d):\n    return list(d.keys())",
    "dict-values": "def dict_values(d):\n    return list(d.values())",
    "dict-entries": "def dict_entries(d):\n    return list(d.items())",
    "dict-merge": "def dict_merge(d1, d2):\n    out = dict(d1)\n    out.update(d2)\n    return out",
    "dict-map-values": "def dict_map_values(f, d):\n    return {k: f(v) for k, v in d.items()}",
    "dict-filter": "def dict_filter(pred, d):\n    return {k: v for k, v in d.items() if pred(k, v)}",
    "dict-size": "def dict_size(d):\n    return len(d)",
    "dict->alist": "def dict_to_alist(d):\n    return d",
    "alist->dict": "def alist_to_dict(xs):\n    return xs",
}

CHEZ_SNIPPETS = {
    "dict-empty?": "(define (empty? d)\n  (null? d))",
    "dict-lookup": "(define (lookup k d)\n  (let ([p (assoc k d)])\n    (if p (cdr p) #f)))",
    "dict-has-key?": "(define (has-key? k d)\n  (if (assoc k d) #t #f))",
    "dict-assoc": "(define (put k v d)\n  (cons (cons k v) (dict-dissoc k d)))",
    "dict-dissoc": "(define (drop k d)\n  (filter (lambda (p) (not (equal? k (car p)))) d))",
    "dict-keys": "(define (keys d)\n  (map car d))",
    "dict-values": "(define (values0 d)\n  (map cdr d))",
    "dict-entries": "(define (entries d)\n  d)",
    "dict-merge": "(define (merge0 d1 d2)\n  (fold-left (lambda (acc p) (dict-assoc (car p) (cdr p) acc)) d1 d2))",
    "dict-map-values": "(define (map-values f d)\n  (map (lambda (p) (cons (car p) (f (cdr p)))) d))",
    "dict-filter": "(define (filter-d pred d)\n  (filter (lambda (p) (pred (car p) (cdr p))) d))",
    "dict-size": "(define (size0 d)\n  (length d))",
    "dict->alist": "(define (to-alist d)\n  d)",
    "alist->dict": "(define (from-alist xs)\n  xs)",
}

BUGGY_CASES = [
    {
        "fn": "dict-empty?",
        "buggy": "(define (dict-empty? dict)\n  (null? (cdr dict)))",
        "note": "A one-entry dictionary should not be empty.",
    },
    {
        "fn": "dict-empty?",
        "buggy": "(define (dict-empty? dict)\n  #f)",
        "note": "Empty dictionary must return #t.",
    },
    {
        "fn": "dict-lookup",
        "buggy": "(define (dict-lookup key dict)\n  (let ([pair (assoc key dict)])\n    (if pair pair #f)))",
        "note": "Lookup should return value, not key-value pair.",
    },
    {
        "fn": "dict-lookup",
        "buggy": "(define (dict-lookup key dict)\n  0)",
        "note": "Missing keys should yield #f, and present keys must return stored values.",
    },
    {
        "fn": "dict-has-key?",
        "buggy": "(define (dict-has-key? key dict)\n  (if (dict-lookup key dict) #t #f))",
        "note": "A key with value #f should still count as present.",
    },
    {
        "fn": "dict-has-key?",
        "buggy": "(define (dict-has-key? key dict)\n  #t)",
        "note": "Absent keys must return #f.",
    },
    {
        "fn": "dict-assoc",
        "buggy": "(define (dict-assoc key value dict)\n  (cons (cons key value) dict))",
        "note": "Existing keys must be replaced, not duplicated.",
    },
    {
        "fn": "dict-assoc",
        "buggy": "(define (dict-assoc key value dict)\n  dict)",
        "note": "Association should insert or update the key.",
    },
    {
        "fn": "dict-dissoc",
        "buggy": "(define (dict-dissoc key dict)\n  dict)",
        "note": "Key should be removed when present.",
    },
    {
        "fn": "dict-dissoc",
        "buggy": "(define (dict-dissoc key dict)\n  '())",
        "note": "Removing one key must not erase unrelated entries.",
    },
    {
        "fn": "dict-keys",
        "buggy": "(define (dict-keys dict)\n  (map cdr dict))",
        "note": "This returns values, not keys.",
    },
    {
        "fn": "dict-keys",
        "buggy": "(define (dict-keys dict)\n  '())",
        "note": "Keys list should reflect dictionary entries.",
    },
    {
        "fn": "dict-values",
        "buggy": "(define (dict-values dict)\n  (map car dict))",
        "note": "This returns keys, not values.",
    },
    {
        "fn": "dict-values",
        "buggy": "(define (dict-values dict)\n  '())",
        "note": "Values list should reflect dictionary entries.",
    },
    {
        "fn": "dict-entries",
        "buggy": "(define (dict-entries dict)\n  (reverse dict))",
        "note": "Entries should be returned unchanged.",
    },
    {
        "fn": "dict-entries",
        "buggy": "(define (dict-entries dict)\n  '())",
        "note": "Entries should not be dropped.",
    },
    {
        "fn": "dict-merge",
        "buggy": "(define (dict-merge dict1 dict2)\n  dict1)",
        "note": "Merge must include keys from dict2.",
    },
    {
        "fn": "dict-merge",
        "buggy": "(define (dict-merge dict1 dict2)\n  (let loop ([remaining dict2] [result dict1])\n    (if (null? remaining) result (loop (cdr remaining) (dict-assoc (car (car remaining)) (cdr (car remaining)) result)))))",
        "note": "Ensure overlap precedence matches spec (dict2 values win).",
    },
    {
        "fn": "dict-map-values",
        "buggy": "(define (dict-map-values f dict)\n  (map (lambda (pair) (cons (f (car pair)) (cdr pair))) dict))",
        "note": "Function should transform values, not keys.",
    },
    {
        "fn": "dict-map-values",
        "buggy": "(define (dict-map-values f dict)\n  dict)",
        "note": "Values must be transformed.",
    },
    {
        "fn": "dict-filter",
        "buggy": "(define (dict-filter pred dict)\n  dict)",
        "note": "Entries failing predicate must be removed.",
    },
    {
        "fn": "dict-filter",
        "buggy": "(define (dict-filter pred dict)\n  (let loop ([remaining dict] [acc '()])\n    (cond [(null? remaining) (reverse acc)]\n          [(pred (car (car remaining)) (cdr (car remaining))) (loop (cdr remaining) acc)]\n          [else (loop (cdr remaining) (cons (car remaining) acc))])))",
        "note": "Predicate logic is inverted.",
    },
    {
        "fn": "dict-size",
        "buggy": "(define (dict-size dict)\n  (+ 1 (length dict)))",
        "note": "Size must equal entry count exactly.",
    },
    {
        "fn": "dict-size",
        "buggy": "(define (dict-size dict)\n  0)",
        "note": "Non-empty dictionaries must have positive size.",
    },
    {
        "fn": "dict->alist",
        "buggy": "(define (dict->alist dict)\n  (reverse dict))",
        "note": "Conversion is identity in this representation.",
    },
    {
        "fn": "dict->alist",
        "buggy": "(define (dict->alist dict)\n  '())",
        "note": "Entries should be preserved.",
    },
    {
        "fn": "alist->dict",
        "buggy": "(define (alist->dict alist)\n  (reverse alist))",
        "note": "Conversion is identity in this representation.",
    },
    {
        "fn": "alist->dict",
        "buggy": "(define (alist->dict alist)\n  '())",
        "note": "Entries should be preserved.",
    },
]

# Replace one dict-merge buggy case with explicit wrong precedence.
BUGGY_CASES[17] = {
    "fn": "dict-merge",
    "buggy": "(define (dict-merge dict1 dict2)\n  (let loop ([remaining dict1] [result dict2])\n    (if (null? remaining) result (loop (cdr remaining) (dict-assoc (car (car remaining)) (cdr (car remaining)) result)))))",
    "note": "This gives dict1 precedence; dict2 should win on overlap.",
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
    sid = f"dict_{family}_{family_counter[family]:03d}"
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
# Family 1: spec_to_code (28)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    diff = "easy" if fn in {"dict-empty?", "dict-lookup", "dict-has-key?", "dict-size", "dict-keys", "dict-values", "dict-entries", "dict->alist", "alist->dict"} else "medium"

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=diff,
        source_function=fn,
        prompt=f"""You are implementing Tier-0 dictionary code in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "dict", "spec-to-code", fn],
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
        tags=["tier0", "data", "dict", "skeleton-completion", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (28)
# -----------------------------------------------------------------------------
for fn in TRANSLATION_FUNCTIONS:
    diff = "easy" if fn in {"dict-empty?", "dict-lookup", "dict-has-key?", "dict-size", "dict-keys", "dict-values", "dict-entries", "dict->alist", "alist->dict"} else "medium"

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
        tags=["tier0", "data", "dict", "python-to-scheme", fn],
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
        tags=["tier0", "data", "dict", "chez-to-fold", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (28)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = case["fn"]
    diff = "easy" if fn in {"dict-empty?", "dict-lookup", "dict-has-key?", "dict-size", "dict-keys", "dict-values", "dict-entries", "dict->alist", "alist->dict"} else "medium"

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
        tags=["tier0", "data", "dict", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition/use (34)
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
        tags=["tier0", "data", "dict", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # Direct
    ("dict-empty?", "Return whether `dict-empty` is empty.", "(dict-empty? dict-empty)", "(equal? (dict-empty? dict-empty) #t)", "easy", ["direct"]),
    ("dict-empty?", "Return whether dictionary `'((a . 1))` is empty.", "(dict-empty? '((a . 1)))", "(equal? (dict-empty? '((a . 1))) #f)", "easy", ["direct"]),
    ("dict-lookup", "Lookup key `'b` in `'((a . 1) (b . 2) (c . 3))`.", "(dict-lookup 'b '((a . 1) (b . 2) (c . 3)))", "(equal? (dict-lookup 'b '((a . 1) (b . 2) (c . 3))) 2)", "easy", ["direct"]),
    ("dict-lookup", "Lookup missing key `'z` in `'((a . 1) (b . 2))`.", "(dict-lookup 'z '((a . 1) (b . 2)))", "(equal? (dict-lookup 'z '((a . 1) (b . 2))) #f)", "easy", ["direct"]),
    ("dict-has-key?", "Check whether `'name` exists in `'((name . \"Ada\") (age . 36))`.", "(dict-has-key? 'name '((name . \"Ada\") (age . 36)))", "(equal? (dict-has-key? 'name '((name . \"Ada\") (age . 36))) #t)", "easy", ["direct"]),
    ("dict-assoc", "Associate `'city` -> \"NYC\" into `'((name . \"Ada\"))`.", "(dict-assoc 'city \"NYC\" '((name . \"Ada\")))", "(equal? (dict-lookup 'city (dict-assoc 'city \"NYC\" '((name . \"Ada\")))) \"NYC\")", "easy", ["direct"]),
    ("dict-dissoc", "Remove key `'b` from `'((a . 1) (b . 2) (c . 3))`.", "(dict-dissoc 'b '((a . 1) (b . 2) (c . 3)))", "(equal? (dict-dissoc 'b '((a . 1) (b . 2) (c . 3))) '((a . 1) (c . 3)))", "medium", ["direct"]),
    ("dict-keys", "Return keys of `'((x . 24) (y . 25) (z . 26))`.", "(dict-keys '((x . 24) (y . 25) (z . 26)))", "(equal? (dict-keys '((x . 24) (y . 25) (z . 26))) '(x y z))", "easy", ["direct"]),
    ("dict-values", "Return values of `'((x . 24) (y . 25) (z . 26))`.", "(dict-values '((x . 24) (y . 25) (z . 26)))", "(equal? (dict-values '((x . 24) (y . 25) (z . 26))) '(24 25 26))", "easy", ["direct"]),
    ("dict-size", "Return size of `'((a . 1) (b . 2) (c . 3))`.", "(dict-size '((a . 1) (b . 2) (c . 3)))", "(equal? (dict-size '((a . 1) (b . 2) (c . 3))) 3)", "easy", ["direct"]),
    ("dict-entries", "Return entries of `'((id . 7) (ok . #t))`.", "(dict-entries '((id . 7) (ok . #t)))", "(equal? (dict-entries '((id . 7) (ok . #t))) '((id . 7) (ok . #t)))", "easy", ["direct"]),
    ("dict-merge", "Merge `d1='((a . 1) (b . 2))` with `d2='((b . 99) (c . 3))` and return lookup of `'b`.", "(dict-lookup 'b (dict-merge '((a . 1) (b . 2)) '((b . 99) (c . 3))))", "(equal? (dict-lookup 'b (dict-merge '((a . 1) (b . 2)) '((b . 99) (c . 3)))) 99)", "medium", ["direct"]),

    # Properties
    ("dict-assoc", "Return #t iff associating an existing key updates value without changing size.", "(let ([d (dict-assoc 'b 9 '((a . 1) (b . 2)))]) (and (= (dict-size d) 2) (= (dict-lookup 'b d) 9)))", "(equal? (let ([d (dict-assoc 'b 9 '((a . 1) (b . 2)))]) (and (= (dict-size d) 2) (= (dict-lookup 'b d) 9))) #t)", "medium", ["property"]),
    ("dict-dissoc", "Return #t iff dissociating a missing key leaves dictionary unchanged.", "(equal? (dict-dissoc 'z '((a . 1) (b . 2))) '((a . 1) (b . 2)))", "(equal? (equal? (dict-dissoc 'z '((a . 1) (b . 2))) '((a . 1) (b . 2))) #t)", "medium", ["property"]),
    ("dict-merge", "Return #t iff merge includes keys from both dictionaries.", "(let ([m (dict-merge '((a . 1)) '((b . 2)))]) (and (dict-has-key? 'a m) (dict-has-key? 'b m)))", "(equal? (let ([m (dict-merge '((a . 1)) '((b . 2)))]) (and (dict-has-key? 'a m) (dict-has-key? 'b m))) #t)", "medium", ["property"]),
    ("dict-merge", "Return #t iff dict2 precedence holds for overlapping key `'k`.", "(= (dict-lookup 'k (dict-merge '((k . 1)) '((k . 2)))) 2)", "(equal? (= (dict-lookup 'k (dict-merge '((k . 1)) '((k . 2)))) 2) #t)", "medium", ["property"]),
    ("dict-keys", "Return #t iff key/value/entry lengths match dictionary size.", "(let ([d '((x . 1) (y . 2) (z . 3))]) (and (= (length (dict-keys d)) (dict-size d)) (= (length (dict-values d)) (dict-size d)) (= (length (dict-entries d)) (dict-size d))))", "(equal? (let ([d '((x . 1) (y . 2) (z . 3))]) (and (= (length (dict-keys d)) (dict-size d)) (= (length (dict-values d)) (dict-size d)) (= (length (dict-entries d)) (dict-size d)))) #t)", "medium", ["property"]),
    ("dict-filter", "Return #t iff filtering by `v > 1` removes key `'a` from `'((a . 1) (b . 2) (c . 3))`.", "(let ([d (dict-filter (lambda (k v) (> v 1)) '((a . 1) (b . 2) (c . 3)))]) (and (not (dict-has-key? 'a d)) (dict-has-key? 'b d) (dict-has-key? 'c d)))", "(equal? (let ([d (dict-filter (lambda (k v) (> v 1)) '((a . 1) (b . 2) (c . 3)))]) (and (not (dict-has-key? 'a d)) (dict-has-key? 'b d) (dict-has-key? 'c d))) #t)", "medium", ["property"]),
    ("dict-map-values", "Return #t iff dict-map-values preserves keys while changing values.", "(let ([d (dict-map-values (lambda (v) (+ v 10)) '((a . 1) (b . 2)))]) (and (dict-has-key? 'a d) (dict-has-key? 'b d) (= (dict-lookup 'a d) 11) (= (dict-lookup 'b d) 12)))", "(equal? (let ([d (dict-map-values (lambda (v) (+ v 10)) '((a . 1) (b . 2)))]) (and (dict-has-key? 'a d) (dict-has-key? 'b d) (= (dict-lookup 'a d) 11) (= (dict-lookup 'b d) 12))) #t)", "medium", ["property"]),
    ("dict->alist", "Return #t iff alist->dict and dict->alist round-trip preserves entries.", "(equal? (dict->alist (alist->dict '((a . 1) (b . 2)))) '((a . 1) (b . 2)))", "(equal? (equal? (dict->alist (alist->dict '((a . 1) (b . 2)))) '((a . 1) (b . 2))) #t)", "easy", ["property"]),
    ("dict-assoc", "Return #t iff associating then dissociating same key recovers original dict when key absent initially.", "(equal? (dict-dissoc 'k (dict-assoc 'k 99 '((a . 1) (b . 2)))) '((a . 1) (b . 2)))", "(equal? (equal? (dict-dissoc 'k (dict-assoc 'k 99 '((a . 1) (b . 2)))) '((a . 1) (b . 2))) #t)", "medium", ["property"]),
    ("dict-size", "Return #t iff dict size after dissoc is one less for existing key.", "(= (dict-size (dict-dissoc 'b '((a . 1) (b . 2) (c . 3)))) 2)", "(equal? (= (dict-size (dict-dissoc 'b '((a . 1) (b . 2) (c . 3)))) 2) #t)", "easy", ["property"]),

    # Fold/list/loop
    ("dict-assoc", "Build dict from alist `'((a . 1) (b . 2) (c . 3))` using fold-left and dict-assoc.", "(fold-left (lambda (d p) (dict-assoc (car p) (cdr p) d)) dict-empty '((a . 1) (b . 2) (c . 3)))", "(let ([d (fold-left (lambda (d p) (dict-assoc (car p) (cdr p) d)) dict-empty '((a . 1) (b . 2) (c . 3)))]) (and (= (dict-size d) 3) (= (dict-lookup 'a d) 1) (= (dict-lookup 'b d) 2) (= (dict-lookup 'c d) 3)))", "hard", ["fold"]),
    ("dict-map-values", "Double values in `'((x . 10) (y . 20))` then return values list.", "(dict-values (dict-map-values (lambda (v) (* 2 v)) '((x . 10) (y . 20))))", "(equal? (dict-values (dict-map-values (lambda (v) (* 2 v)) '((x . 10) (y . 20)))) '(20 40))", "medium", ["list"]),
    ("dict-filter", "Filter keys with even values from `'((a . 1) (b . 2) (c . 4) (d . 5))`.", "(dict-filter (lambda (k v) (even? v)) '((a . 1) (b . 2) (c . 4) (d . 5)))", "(equal? (dict-filter (lambda (k v) (even? v)) '((a . 1) (b . 2) (c . 4) (d . 5))) '((b . 2) (c . 4)))", "medium", ["list"]),
    ("dict-assoc", "Use a named-let to insert keys `(k1 k2 k3)` with values `(1 2 3)`.", "(let loop ([ks '(k1 k2 k3)] [vs '(1 2 3)] [d dict-empty]) (if (null? ks) d (loop (cdr ks) (cdr vs) (dict-assoc (car ks) (car vs) d))))", "(let ([d (let loop ([ks '(k1 k2 k3)] [vs '(1 2 3)] [d dict-empty]) (if (null? ks) d (loop (cdr ks) (cdr vs) (dict-assoc (car ks) (car vs) d))))]) (and (= (dict-size d) 3) (= (dict-lookup 'k1 d) 1) (= (dict-lookup 'k2 d) 2) (= (dict-lookup 'k3 d) 3)))", "hard", ["loop"]),
    ("dict-size", "Count how many keys in `'(a b c d)` exist in dict `'((a . 1) (c . 3))`.", "(let loop ([ks '(a b c d)] [n 0]) (if (null? ks) n (loop (cdr ks) (if (dict-has-key? (car ks) '((a . 1) (c . 3))) (+ n 1) n))))", "(equal? (let loop ([ks '(a b c d)] [n 0]) (if (null? ks) n (loop (cdr ks) (if (dict-has-key? (car ks) '((a . 1) (c . 3))) (+ n 1) n)))) 2)", "medium", ["loop"]),
    ("dict-size", "Map `dict-size` over dict list `'(((a . 1)) ((a . 1) (b . 2)) ())`.", "(map dict-size '(((a . 1)) ((a . 1) (b . 2)) ()))", "(equal? (map dict-size '(((a . 1)) ((a . 1) (b . 2)) ())) '(1 2 0))", "easy", ["list"]),

    # Integration
    ("dict-merge", "Merge user metadata and score maps, then lookup `'score`.", "(dict-lookup 'score (dict-merge '((user . \"u1\") (age . 30)) '((score . 88) (age . 31))))", "(equal? (dict-lookup 'score (dict-merge '((user . \"u1\") (age . 30)) '((score . 88) (age . 31)))) 88)", "medium", ["integration"]),
    ("dict-merge", "Merge overlapping dicts and return `(list age city)`.", "(let ([d (dict-merge '((name . \"Ada\") (age . 36)) '((age . 37) (city . \"London\")))]) (list (dict-lookup 'age d) (dict-lookup 'city d)))", "(equal? (let ([d (dict-merge '((name . \"Ada\") (age . 36)) '((age . 37) (city . \"London\")))]) (list (dict-lookup 'age d) (dict-lookup 'city d))) '(37 \"London\"))", "hard", ["integration"]),
    ("dict-map-values", "Apply +1 to values, then filter for values >= 3.", "(dict-filter (lambda (k v) (>= v 3)) (dict-map-values (lambda (v) (+ v 1)) '((a . 1) (b . 2) (c . 3))))", "(equal? (dict-filter (lambda (k v) (>= v 3)) (dict-map-values (lambda (v) (+ v 1)) '((a . 1) (b . 2) (c . 3)))) '((b . 3) (c . 4)))", "hard", ["integration"]),
    ("dict-dissoc", "Insert `'x` into dict then remove it; return resulting size.", "(dict-size (dict-dissoc 'x (dict-assoc 'x 5 '((a . 1) (b . 2)))))", "(equal? (dict-size (dict-dissoc 'x (dict-assoc 'x 5 '((a . 1) (b . 2))))) 2)", "medium", ["integration"]),
    ("dict-filter", "Filter entries whose key is in allowed set `'(a c)`.", "(dict-filter (lambda (k v) (or (equal? k 'a) (equal? k 'c))) '((a . 1) (b . 2) (c . 3)))", "(equal? (dict-filter (lambda (k v) (or (equal? k 'a) (equal? k 'c))) '((a . 1) (b . 2) (c . 3))) '((a . 1) (c . 3)))", "medium", ["integration"]),
    ("dict-lookup", "Return #t iff every key in `'((a . 1) (b . 2) (c . 3))` is found by dict-lookup.", "(let ([d '((a . 1) (b . 2) (c . 3))]) (and (= (dict-lookup 'a d) 1) (= (dict-lookup 'b d) 2) (= (dict-lookup 'c d) 3)))", "(equal? (let ([d '((a . 1) (b . 2) (c . 3))]) (and (= (dict-lookup 'a d) 1) (= (dict-lookup 'b d) 2) (= (dict-lookup 'c d) 3))) #t)", "medium", ["integration"]),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

if sum(1 for s in samples if s["family"] == "composition") != 34:
    raise ValueError("composition family must contain exactly 34 samples")

# -----------------------------------------------------------------------------
# Split train/eval
# -----------------------------------------------------------------------------
if len(samples) != 118:
    raise ValueError(f"expected 118 samples, got {len(samples)}")

by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

EVAL_QUOTA = {
    "spec_to_code": 4,
    "translation": 4,
    "bugfix": 4,
    "composition": 8,
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
            removable.sort(key=lambda r: (fn_counts[str(r["source_function"])] , str(r["id"])), reverse=True)
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

if len(train_rows) != 98 or len(eval_rows) != 20:
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
