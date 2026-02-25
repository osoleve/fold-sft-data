#!/usr/bin/env python3
"""Generate Tier-0 SFT samples for lattice/data/alist.ss."""

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

SOURCE_MODULE = "lattice/data/alist.ss"
SOURCE_TEST = "lattice/data/test-alist.ss"

DEFS: Dict[str, str] = {
    "alist-ref": """(define (alist-ref alist key)
  (let ([pair (assq key alist)])
    (if pair (cdr pair) #f)))""",
    "alist-ref/default": """(define (alist-ref/default alist key default)
  (let ([pair (assq key alist)])
    (if pair (cdr pair) default)))""",
    "alist-set": """(define (alist-set alist key value)
  (cons (cons key value) (alist-remove alist key)))""",
    "alist-update": """(define (alist-update alist key fn default)
  (let ([current (alist-ref/default alist key default)])
    (alist-set alist key (fn current))))""",
    "alist-remove": """(define (alist-remove alist key)
  (let loop ([remaining alist] [acc '()])
    (cond
      [(null? remaining) (reverse acc)]
      [(eq? key (car (car remaining)))
       (append (reverse acc) (cdr remaining))]
      [else (loop (cdr remaining) (cons (car remaining) acc))])))""",
    "alist-merge": """(define (alist-merge a b)
  (let loop ([remaining b] [result a])
    (if (null? remaining)
        result
        (loop (cdr remaining)
              (alist-set result
                         (car (car remaining))
                         (cdr (car remaining)))))))""",
    "alist-keys": """(define (alist-keys alist)
  (map car alist))""",
    "alist-map": """(define (alist-map fn alist)
  (map (lambda (pair)
         (cons (car pair) (fn (car pair) (cdr pair))))
       alist))""",
}

FUNCTION_ORDER = [
    "alist-ref",
    "alist-ref/default",
    "alist-set",
    "alist-update",
    "alist-remove",
    "alist-merge",
    "alist-keys",
    "alist-map",
]

FUNCTION_SPECS = {
    "alist-ref": "Look up key with assq/eq? semantics, returning value or #f when key is missing.",
    "alist-ref/default": "Look up key and return default when key is absent.",
    "alist-set": "Functionally set key to value: replace existing first entry for key or prepend new pair.",
    "alist-update": "Update key by applying fn to current value (or default when missing), then storing result.",
    "alist-remove": "Remove only the first occurrence of key from alist while preserving order of remaining entries.",
    "alist-merge": "Merge two alists where entries from b override entries from a on key conflict.",
    "alist-keys": "Extract keys in current alist order.",
    "alist-map": "Map values with function (key value -> new-value), preserving keys and pair structure.",
}

SKELETONS = {
    "alist-ref": """(define (alist-ref alist key)
  ;; TODO: return value for key, or #f when missing
  <TODO>)""",
    "alist-ref/default": """(define (alist-ref/default alist key default)
  ;; TODO: return value for key, or default when missing
  <TODO>)""",
    "alist-set": """(define (alist-set alist key value)
  ;; TODO: set key to value functionally without duplicate key entries
  <TODO>)""",
    "alist-update": """(define (alist-update alist key fn default)
  ;; TODO: read current/default value, apply fn, and store updated value
  <TODO>)""",
    "alist-remove": """(define (alist-remove alist key)
  ;; TODO: remove first entry matching key and keep order for others
  <TODO>)""",
    "alist-merge": """(define (alist-merge a b)
  ;; TODO: merge with b taking precedence on conflicts
  <TODO>)""",
    "alist-keys": """(define (alist-keys alist)
  ;; TODO: extract key list
  <TODO>)""",
    "alist-map": """(define (alist-map fn alist)
  ;; TODO: map values using fn(key, value) while preserving keys
  <TODO>)""",
}

DEPENDS: Dict[str, List[str]] = {
    "alist-ref": [],
    "alist-ref/default": [],
    "alist-set": ["alist-remove"],
    "alist-update": ["alist-ref/default", "alist-set"],
    "alist-remove": [],
    "alist-merge": ["alist-set"],
    "alist-keys": [],
    "alist-map": [],
}

VERIFY_BY_FUNCTION = {
    "alist-ref": """(and
  (equal? (alist-ref '((a . 1) (b . 2) (c . 3)) 'b) 2)
  (equal? (alist-ref '((a . 10) (a . 20)) 'a) 10)
  (equal? (alist-ref '() 'z) #f))""",
    "alist-ref/default": """(and
  (equal? (alist-ref/default '((a . 1) (b . 2)) 'b 99) 2)
  (equal? (alist-ref/default '((a . 1)) 'z 99) 99)
  (equal? (alist-ref/default '() 'x 0) 0))""",
    "alist-set": """(and
  (let ([r (alist-set '((a . 1)) 'b 2)])
    (and (equal? (alist-ref r 'a) 1)
         (equal? (alist-ref r 'b) 2)))
  (let ([r (alist-set '((a . 1) (b . 2)) 'a 10)])
    (and (equal? (alist-ref r 'a) 10)
         (equal? (length r) 2))))""",
    "alist-update": """(and
  (equal? (alist-ref (alist-update '((a . 1) (b . 2)) 'a (lambda (v) (+ v 10)) 0) 'a) 11)
  (equal? (alist-ref (alist-update '((a . 1)) 'b (lambda (v) (+ v 10)) 0) 'b) 10)
  (equal? (alist-ref (alist-update '() 'x (lambda (v) (* v 2)) 5) 'x) 10))""",
    "alist-remove": """(and
  (equal? (alist-remove '((a . 1) (b . 2) (c . 3)) 'b) '((a . 1) (c . 3)))
  (equal? (alist-remove '((a . 1)) 'z) '((a . 1)))
  (equal? (alist-remove '((a . 1) (b . 2) (a . 3)) 'a) '((b . 2) (a . 3))))""",
    "alist-merge": """(let ([r (alist-merge '((a . 1) (b . 2)) '((a . 10) (c . 3)))])
  (and (equal? (alist-ref r 'a) 10)
       (equal? (alist-ref r 'b) 2)
       (equal? (alist-ref r 'c) 3)))""",
    "alist-keys": """(and
  (equal? (alist-keys '((a . 1) (b . 2) (c . 3))) '(a b c))
  (equal? (alist-keys '()) '()))""",
    "alist-map": """(and
  (equal? (alist-map (lambda (k v) (* v 2)) '((a . 1) (b . 2) (c . 3))) '((a . 2) (b . 4) (c . 6)))
  (equal? (alist-map (lambda (k v) (list k v)) '((x . 1))) '((x . (x 1))))
  (equal? (alist-map (lambda (k v) v) '()) '()))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "alist-ref": """def alist_ref(alist, key):
    for k, v in alist:
        if k is key:
            return v
    return False""",
    "alist-ref/default": """def alist_ref_default(alist, key, default):
    for k, v in alist:
        if k is key:
            return v
    return default""",
    "alist-set": """def alist_set(alist, key, value):
    return [(key, value)] + alist_remove(alist, key)""",
    "alist-update": """def alist_update(alist, key, fn, default):
    cur = alist_ref_default(alist, key, default)
    return alist_set(alist, key, fn(cur))""",
    "alist-remove": """def alist_remove(alist, key):
    out = []
    removed = False
    for pair in alist:
        if (not removed) and pair[0] is key:
            removed = True
            continue
        out.append(pair)
    return out""",
    "alist-merge": """def alist_merge(a, b):
    result = a
    for k, v in b:
        result = alist_set(result, k, v)
    return result""",
    "alist-keys": """def alist_keys(alist):
    return [k for (k, v) in alist]""",
    "alist-map": """def alist_map(fn, alist):
    return [(k, fn(k, v)) for (k, v) in alist]""",
}

CHEZ_SNIPPETS = {
    "alist-ref": """(define (ref a k)
  (let ([p (assq k a)])
    (if p (cdr p) #f)))""",
    "alist-ref/default": """(define (ref/default a k d)
  (let ([p (assq k a)])
    (if p (cdr p) d)))""",
    "alist-set": """(define (set-entry a k v)
  (cons (cons k v) (alist-remove a k)))""",
    "alist-update": """(define (update-entry a k f d)
  (let ([cur (alist-ref/default a k d)])
    (alist-set a k (f cur))))""",
    "alist-remove": """(define (remove-entry a k)
  (let loop ([rest a] [acc '()])
    (cond
      [(null? rest) (reverse acc)]
      [(eq? k (caar rest)) (append (reverse acc) (cdr rest))]
      [else (loop (cdr rest) (cons (car rest) acc))])))""",
    "alist-merge": """(define (merge-alists a b)
  (let loop ([rest b] [res a])
    (if (null? rest)
        res
        (loop (cdr rest)
              (alist-set res (caar rest) (cdar rest))))))""",
    "alist-keys": """(define (keys a)
  (map car a))""",
    "alist-map": """(define (map-values f a)
  (map (lambda (p) (cons (car p) (f (car p) (cdr p)))) a))""",
}

BUGGY_CASES = [
    {
        "fn": "alist-ref",
        "buggy": """(define (alist-ref alist key)
  (let ([pair (assq key alist)])
    pair))""",
        "note": "Return the value (cdr), not the whole pair.",
    },
    {
        "fn": "alist-ref",
        "buggy": """(define (alist-ref alist key)
  (let ([pair (assq key alist)])
    (if pair (cdr pair) '())))""",
        "note": "Missing keys must return #f, not the empty list.",
    },
    {
        "fn": "alist-ref/default",
        "buggy": """(define (alist-ref/default alist key default)
  (let ([pair (assq key alist)])
    (if pair (cdr pair) #f)))""",
        "note": "Must return the provided default when key is missing.",
    },
    {
        "fn": "alist-ref/default",
        "buggy": """(define (alist-ref/default alist key default)
  default)""",
        "note": "Existing keys must return their stored value, not default.",
    },
    {
        "fn": "alist-set",
        "buggy": """(define (alist-set alist key value)
  (cons (cons key value) alist))""",
        "note": "Existing key entries must be removed to avoid duplicates.",
    },
    {
        "fn": "alist-set",
        "buggy": """(define (alist-set alist key value)
  (cons (cons value key) (alist-remove alist key)))""",
        "note": "Construct pairs as (key . value), not (value . key).",
    },
    {
        "fn": "alist-update",
        "buggy": """(define (alist-update alist key fn default)
  (alist-set alist key default))""",
        "note": "Must apply fn to current/default value before storing.",
    },
    {
        "fn": "alist-update",
        "buggy": """(define (alist-update alist key fn default)
  (let ([current (alist-ref alist key)])
    (alist-set alist key (fn current))))""",
        "note": "Missing keys must use the explicit default value, not #f.",
    },
    {
        "fn": "alist-remove",
        "buggy": """(define (alist-remove alist key)
  (filter (lambda (pair) (not (eq? key (car pair)))) alist))""",
        "note": "Only the first matching key should be removed, not all matches.",
    },
    {
        "fn": "alist-remove",
        "buggy": """(define (alist-remove alist key)
  (let loop ([remaining alist] [acc '()])
    (cond
      [(null? remaining) (reverse acc)]
      [(eq? key (car (car remaining))) (reverse acc)]
      [else (loop (cdr remaining) (cons (car remaining) acc))])))""",
        "note": "After removing first match, the remaining tail must be preserved.",
    },
    {
        "fn": "alist-merge",
        "buggy": """(define (alist-merge a b)
  (let loop ([remaining b] [result a])
    (if (null? remaining)
        result
        (loop (cdr remaining) result))))""",
        "note": "Entries from b are ignored; merge must upsert each pair from b.",
    },
    {
        "fn": "alist-merge",
        "buggy": """(define (alist-merge a b)
  (let loop ([remaining a] [result b])
    (if (null? remaining)
        result
        (loop (cdr remaining)
              (alist-set result (car (car remaining)) (cdr (car remaining)))))))""",
        "note": "Conflict resolution is reversed; b must win over a.",
    },
    {
        "fn": "alist-keys",
        "buggy": """(define (alist-keys alist)
  (map cdr alist))""",
        "note": "Extract keys with car, not values.",
    },
    {
        "fn": "alist-keys",
        "buggy": """(define (alist-keys alist)
  (reverse (map car alist)))""",
        "note": "Key order should match alist order.",
    },
    {
        "fn": "alist-map",
        "buggy": """(define (alist-map fn alist)
  (map (lambda (pair)
         (fn (car pair) (cdr pair)))
       alist))""",
        "note": "Must preserve key/value pair structure in output alist.",
    },
    {
        "fn": "alist-map",
        "buggy": """(define (alist-map fn alist)
  (map (lambda (pair)
         (cons (car pair) (fn (cdr pair) (car pair))))
       alist))""",
        "note": "Function arguments are reversed; fn expects (key value).",
    },
]

DIFFICULTY = {
    "alist-ref": "easy",
    "alist-ref/default": "easy",
    "alist-set": "medium",
    "alist-update": "medium",
    "alist-remove": "medium",
    "alist-merge": "medium",
    "alist-keys": "easy",
    "alist-map": "medium",
}

REQUIRED_KEYS = {
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
}

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
    sid = f"alist_{family}_{family_counter[family]:03d}"
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
            if dep in DEFS and dep not in seen:
                seen.add(dep)
                visit(dep)
                ordered.append(dep)

    for dep in DEPENDS.get(fn, []):
        if dep == fn:
            continue
        if dep in DEFS and dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    for dep in verify_refs(fn):
        if dep == fn:
            continue
        if dep in DEFS and dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    return ordered


def def_verify(fn: str) -> str:
    parts = [DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    return "(let ()\n  " + "\n  ".join(parts) + ")"


def wrap_verify_expr(expr: str) -> str:
    parts = [DEFS[name] for name in FUNCTION_ORDER] + [expr]
    return "(let ()\n  " + "\n  ".join(parts) + ")"


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
        tags=["tier0", "data", "alist", "spec-to-code", fn],
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
        tags=["tier0", "data", "alist", "spec-to-code", "skeleton", fn],
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
Preserve behavior exactly.

Target function name: `{fn}`

```python
{PYTHON_SNIPPETS[fn]}
```

Return only the Scheme definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "alist", "translation", "python", fn],
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
        tags=["tier0", "data", "alist", "translation", "chez", fn],
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
        tags=["tier0", "data", "alist", "bugfix", fn],
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
        tags=["tier0", "data", "alist", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # alist-ref
    (
        "alist-ref",
        "Look up key 'b in the alist '((a . 1) (b . 2) (c . 3)).",
        "(alist-ref '((a . 1) (b . 2) (c . 3)) 'b)",
        "(equal? (alist-ref '((a . 1) (b . 2) (c . 3)) 'b) 2)",
        "easy",
        ["direct"],
    ),
    (
        "alist-ref",
        "Look up missing key 'z and return the not-found sentinel.",
        "(alist-ref '((a . 1)) 'z)",
        "(equal? (alist-ref '((a . 1)) 'z) #f)",
        "easy",
        ["missing"],
    ),
    (
        "alist-ref",
        "Check duplicate-key lookup behavior for '((a . 10) (a . 20)); return first value.",
        "(alist-ref '((a . 10) (a . 20)) 'a)",
        "(equal? (alist-ref '((a . 10) (a . 20)) 'a) 10)",
        "medium",
        ["duplicates"],
    ),
    (
        "alist-ref",
        "Set key 'a to 9, then read it back.",
        "(alist-ref (alist-set '((a . 1) (b . 2)) 'a 9) 'a)",
        "(equal? (alist-ref (alist-set '((a . 1) (b . 2)) 'a 9) 'a) 9)",
        "medium",
        ["integration"],
    ),

    # alist-ref/default
    (
        "alist-ref/default",
        "Read key 'b with default 99 from '((a . 1) (b . 2)).",
        "(alist-ref/default '((a . 1) (b . 2)) 'b 99)",
        "(equal? (alist-ref/default '((a . 1) (b . 2)) 'b 99) 2)",
        "easy",
        ["direct"],
    ),
    (
        "alist-ref/default",
        "Read missing key 'z with default 99.",
        "(alist-ref/default '((a . 1)) 'z 99)",
        "(equal? (alist-ref/default '((a . 1)) 'z 99) 99)",
        "easy",
        ["missing"],
    ),
    (
        "alist-ref/default",
        "Read key from empty alist with fallback default 7.",
        "(alist-ref/default '() 'x 7)",
        "(equal? (alist-ref/default '() 'x 7) 7)",
        "easy",
        ["edge-case"],
    ),
    (
        "alist-ref/default",
        "Remove key 'a then read it with default 0.",
        "(alist-ref/default (alist-remove '((a . 1) (b . 2)) 'a) 'a 0)",
        "(equal? (alist-ref/default (alist-remove '((a . 1) (b . 2)) 'a) 'a 0) 0)",
        "medium",
        ["integration"],
    ),

    # alist-set
    (
        "alist-set",
        "Insert key 'b => 2 into '((a . 1)) and read back 'b.",
        "(alist-ref (alist-set '((a . 1)) 'b 2) 'b)",
        "(equal? (alist-ref (alist-set '((a . 1)) 'b 2) 'b) 2)",
        "easy",
        ["direct"],
    ),
    (
        "alist-set",
        "Replace existing key 'a with value 10 and return resulting length.",
        "(length (alist-set '((a . 1) (b . 2)) 'a 10))",
        "(equal? (length (alist-set '((a . 1) (b . 2)) 'a 10)) 2)",
        "medium",
        ["replace"],
    ),
    (
        "alist-set",
        "Apply two sets (a=9 then c=3) and return keys order.",
        "(alist-keys (alist-set (alist-set '((a . 1) (b . 2)) 'a 9) 'c 3))",
        "(equal? (alist-keys (alist-set (alist-set '((a . 1) (b . 2)) 'a 9) 'c 3)) '(c a b))",
        "medium",
        ["order"],
    ),
    (
        "alist-set",
        "Return #t iff setting duplicated key 'k makes lookup return the newly written value.",
        "(equal? (alist-ref (alist-set '((k . 1) (x . 2) (k . 3)) 'k 8) 'k) 8)",
        "(equal? (alist-ref (alist-set '((k . 1) (x . 2) (k . 3)) 'k 8) 'k) 8)",
        "medium",
        ["property"],
    ),

    # alist-update
    (
        "alist-update",
        "Increment existing key 'a by 10 and return its updated value.",
        "(alist-ref (alist-update '((a . 1) (b . 2)) 'a (lambda (v) (+ v 10)) 0) 'a)",
        "(equal? (alist-ref (alist-update '((a . 1) (b . 2)) 'a (lambda (v) (+ v 10)) 0) 'a) 11)",
        "medium",
        ["direct"],
    ),
    (
        "alist-update",
        "Update missing key 'b using default 5 and fn (* v 2); return new value.",
        "(alist-ref (alist-update '((a . 1)) 'b (lambda (v) (* v 2)) 5) 'b)",
        "(equal? (alist-ref (alist-update '((a . 1)) 'b (lambda (v) (* v 2)) 5) 'b) 10)",
        "medium",
        ["default"],
    ),
    (
        "alist-update",
        "Update key 'n twice (+1 then *3) from default 0; return final value.",
        "(alist-ref (alist-update (alist-update '() 'n (lambda (v) (+ v 1)) 0) 'n (lambda (v) (* v 3)) 0) 'n)",
        "(equal? (alist-ref (alist-update (alist-update '() 'n (lambda (v) (+ v 1)) 0) 'n (lambda (v) (* v 3)) 0) 'n) 3)",
        "hard",
        ["composition"],
    ),
    (
        "alist-update",
        "Use update with string append on key 'msg.",
        "(alist-ref (alist-update '((msg . \"hi\")) 'msg (lambda (v) (string-append v \"!\")) \"\") 'msg)",
        "(equal? (alist-ref (alist-update '((msg . \"hi\")) 'msg (lambda (v) (string-append v \"!\")) \"\") 'msg) \"hi!\")",
        "medium",
        ["string"],
    ),

    # alist-remove
    (
        "alist-remove",
        "Remove key 'b from '((a . 1) (b . 2) (c . 3)).",
        "(alist-remove '((a . 1) (b . 2) (c . 3)) 'b)",
        "(equal? (alist-remove '((a . 1) (b . 2) (c . 3)) 'b) '((a . 1) (c . 3)))",
        "easy",
        ["direct"],
    ),
    (
        "alist-remove",
        "Remove key 'a from duplicate list and keep only first removal.",
        "(alist-remove '((a . 1) (b . 2) (a . 3)) 'a)",
        "(equal? (alist-remove '((a . 1) (b . 2) (a . 3)) 'a) '((b . 2) (a . 3)))",
        "medium",
        ["duplicates"],
    ),
    (
        "alist-remove",
        "Try removing missing key 'z and return the unchanged alist.",
        "(alist-remove '((a . 1)) 'z)",
        "(equal? (alist-remove '((a . 1)) 'z) '((a . 1)))",
        "easy",
        ["missing"],
    ),
    (
        "alist-remove",
        "Set 'a to 9, then remove 'a, and confirm lookup returns #f.",
        "(alist-ref (alist-remove (alist-set '((a . 1) (b . 2)) 'a 9) 'a) 'a)",
        "(equal? (alist-ref (alist-remove (alist-set '((a . 1) (b . 2)) 'a 9) 'a) 'a) #f)",
        "medium",
        ["integration"],
    ),

    # alist-merge
    (
        "alist-merge",
        "Merge disjoint alists and return value for key 'b.",
        "(alist-ref (alist-merge '((a . 1)) '((b . 2))) 'b)",
        "(equal? (alist-ref (alist-merge '((a . 1)) '((b . 2))) 'b) 2)",
        "easy",
        ["direct"],
    ),
    (
        "alist-merge",
        "Merge with conflict on 'a and return merged value for 'a (b must win).",
        "(alist-ref (alist-merge '((a . 1) (b . 2)) '((a . 10) (c . 3))) 'a)",
        "(equal? (alist-ref (alist-merge '((a . 1) (b . 2)) '((a . 10) (c . 3))) 'a) 10)",
        "medium",
        ["conflict"],
    ),
    (
        "alist-merge",
        "Return #t iff merge keeps 'b from a and adds 'c from b in this example.",
        "(let ([r (alist-merge '((a . 1) (b . 2)) '((a . 10) (c . 3)))]) (and (equal? (alist-ref r 'b) 2) (equal? (alist-ref r 'c) 3)))",
        "(equal? (let ([r (alist-merge '((a . 1) (b . 2)) '((a . 10) (c . 3)))]) (and (equal? (alist-ref r 'b) 2) (equal? (alist-ref r 'c) 3))) #t)",
        "medium",
        ["property"],
    ),
    (
        "alist-merge",
        "Merge three small alists left-to-right and return final value at 'x.",
        "(alist-ref (alist-merge (alist-merge '((x . 1)) '((y . 2))) '((x . 7))) 'x)",
        "(equal? (alist-ref (alist-merge (alist-merge '((x . 1)) '((y . 2))) '((x . 7))) 'x) 7)",
        "hard",
        ["composition"],
    ),

    # alist-keys
    (
        "alist-keys",
        "Extract keys from '((a . 1) (b . 2) (c . 3)).",
        "(alist-keys '((a . 1) (b . 2) (c . 3)))",
        "(equal? (alist-keys '((a . 1) (b . 2) (c . 3))) '(a b c))",
        "easy",
        ["direct"],
    ),
    (
        "alist-keys",
        "Extract keys from empty alist.",
        "(alist-keys '())",
        "(equal? (alist-keys '()) '())",
        "easy",
        ["edge-case"],
    ),
    (
        "alist-keys",
        "Set key 'z in '((a . 1) (b . 2)) and return resulting key order.",
        "(alist-keys (alist-set '((a . 1) (b . 2)) 'z 9))",
        "(equal? (alist-keys (alist-set '((a . 1) (b . 2)) 'z 9)) '(z a b))",
        "medium",
        ["order"],
    ),
    (
        "alist-keys",
        "Remove 'b then return keys.",
        "(alist-keys (alist-remove '((a . 1) (b . 2) (c . 3)) 'b))",
        "(equal? (alist-keys (alist-remove '((a . 1) (b . 2) (c . 3)) 'b)) '(a c))",
        "medium",
        ["integration"],
    ),

    # alist-map
    (
        "alist-map",
        "Double all values in '((a . 1) (b . 2) (c . 3)).",
        "(alist-map (lambda (k v) (* v 2)) '((a . 1) (b . 2) (c . 3)))",
        "(equal? (alist-map (lambda (k v) (* v 2)) '((a . 1) (b . 2) (c . 3))) '((a . 2) (b . 4) (c . 6)))",
        "medium",
        ["direct"],
    ),
    (
        "alist-map",
        "Map each pair to list (key value).",
        "(alist-map (lambda (k v) (list k v)) '((x . 1) (y . 2)))",
        "(equal? (alist-map (lambda (k v) (list k v)) '((x . 1) (y . 2))) '((x . (x 1)) (y . (y 2))))",
        "medium",
        ["key-aware"],
    ),
    (
        "alist-map",
        "Map over empty alist and return result.",
        "(alist-map (lambda (k v) v) '())",
        "(equal? (alist-map (lambda (k v) v) '()) '())",
        "easy",
        ["edge-case"],
    ),
    (
        "alist-map",
        "Map values by adding key-dependent offset, then read value for key 'c.",
        "(alist-ref (alist-map (lambda (k v) (+ v (if (eq? k 'c) 10 0))) '((a . 1) (c . 3))) 'c)",
        "(equal? (alist-ref (alist-map (lambda (k v) (+ v (if (eq? k 'c) 10 0))) '((a . 1) (c . 3))) 'c) 13)",
        "hard",
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
            "total": len(fam_samples),
            "eval": sum(1 for s in fam_samples if s["id"] in eval_ids),
            "train": sum(1 for s in fam_samples if s["id"] not in eval_ids),
        }
        for fam, fam_samples in sorted(by_family.items())
    },
    "difficulty": dict(sorted(Counter(str(r["difficulty"]) for r in samples).items())),
    "source_functions": len(all_source_functions),
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
