#!/usr/bin/env python3
"""Generate Tier-0 SFT samples for lattice/data/set.ss."""

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

SOURCE_MODULE = "lattice/data/set.ss"
SOURCE_TEST = "lattice/data/test-data-structures.ss"

DEFS: Dict[str, str] = {
    "set-empty?": """(define (set-empty? set)
  (null? set))""",
    "set-member?": """(define (set-member? elem set)
  (cond
    [(null? set) #f]
    [(equal? elem (car set)) #t]
    [else (set-member? elem (cdr set))]))""",
    "set-add": """(define (set-add elem set)
  (if (set-member? elem set)
      set
      (cons elem set)))""",
    "set-remove": """(define (set-remove elem set)
  (let loop ([remaining set]
             [acc '()])
    (cond
      [(null? remaining) (reverse acc)]
      [(equal? elem (car remaining))
       (append (reverse acc) (cdr remaining))]
      [else (loop (cdr remaining) (cons (car remaining) acc))])))""",
    "set-union": """(define (set-union set1 set2)
  (let loop ([remaining set1]
             [result set2])
    (if (null? remaining)
        result
        (loop (cdr remaining)
              (set-add (car remaining) result)))))""",
    "set-intersection": """(define (set-intersection set1 set2)
  (let loop ([remaining set1]
             [acc '()])
    (cond
      [(null? remaining) (reverse acc)]
      [(set-member? (car remaining) set2)
       (loop (cdr remaining) (cons (car remaining) acc))]
      [else (loop (cdr remaining) acc)])))""",
    "set-difference": """(define (set-difference set1 set2)
  (let loop ([remaining set1]
             [acc '()])
    (cond
      [(null? remaining) (reverse acc)]
      [(set-member? (car remaining) set2)
       (loop (cdr remaining) acc)]
      [else (loop (cdr remaining) (cons (car remaining) acc))])))""",
    "set-subset?": """(define (set-subset? set1 set2)
  (let loop ([remaining set1])
    (cond
      [(null? remaining) #t]
      [(set-member? (car remaining) set2)
       (loop (cdr remaining))]
      [else #f])))""",
    "set-size": """(define (set-size set)
  (length set))""",
    "set->list": """(define (set->list set)
  set)""",
    "list->set": """(define (list->set lst)
  (let loop ([remaining lst]
             [acc '()])
    (if (null? remaining)
        acc
        (loop (cdr remaining) (set-add (car remaining) acc)))))""",
}

DEPENDS: Dict[str, List[str]] = {
    "set-empty?": [],
    "set-member?": [],
    "set-add": ["set-member?"],
    "set-remove": [],
    "set-union": ["set-add", "set-member?"],
    "set-intersection": ["set-member?"],
    "set-difference": ["set-member?"],
    "set-subset?": ["set-member?"],
    "set-size": [],
    "set->list": [],
    "list->set": ["set-add", "set-member?"],
}

FUNCTION_ORDER = [
    "set-empty?",
    "set-member?",
    "set-add",
    "set-remove",
    "set-union",
    "set-intersection",
    "set-difference",
    "set-subset?",
    "set-size",
    "set->list",
    "list->set",
]

FUNCTION_SPECS = {
    "set-empty?": "Return #t iff the set has no elements.",
    "set-member?": "Return #t iff elem is present in set.",
    "set-add": "Add elem to set if absent; keep set unchanged if elem already exists.",
    "set-remove": "Remove elem from set if present; otherwise return the original set.",
    "set-union": "Return a set containing elements from both set1 and set2 without duplicates.",
    "set-intersection": "Return elements common to both sets.",
    "set-difference": "Return elements in set1 that are not in set2.",
    "set-subset?": "Return #t iff every element of set1 appears in set2.",
    "set-size": "Return number of elements in set.",
    "set->list": "Convert set representation to list (arbitrary order).",
    "list->set": "Convert list to set by removing duplicates.",
}

SKELETONS = {
    "set-empty?": """(define (set-empty? set)
  ;; TODO: return whether set is empty
  <TODO>)""",
    "set-member?": """(define (set-member? elem set)
  ;; TODO: recursive membership test
  <TODO>)""",
    "set-add": """(define (set-add elem set)
  ;; TODO: add only if elem is not already in set
  <TODO>)""",
    "set-remove": """(define (set-remove elem set)
  ;; TODO: remove first matching elem and preserve order
  <TODO>)""",
    "set-union": """(define (set-union set1 set2)
  ;; TODO: union without duplicates
  <TODO>)""",
    "set-intersection": """(define (set-intersection set1 set2)
  ;; TODO: keep only elements present in both sets
  <TODO>)""",
    "set-difference": """(define (set-difference set1 set2)
  ;; TODO: keep elements in set1 not in set2
  <TODO>)""",
    "set-subset?": """(define (set-subset? set1 set2)
  ;; TODO: check that every element of set1 is in set2
  <TODO>)""",
    "set-size": """(define (set-size set)
  ;; TODO: return element count
  <TODO>)""",
    "set->list": """(define (set->list set)
  ;; TODO: convert set to list
  <TODO>)""",
    "list->set": """(define (list->set lst)
  ;; TODO: build set from list values
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "set-empty?": "(and (set-empty? '()) (not (set-empty? '(x))))",
    "set-member?": "(and (set-member? 'b '(a b c)) (not (set-member? 'z '(a b c))))",
    "set-add": "(let ([s1 (set-add 'a '(b c))] [s2 (set-add 'b '(b c))]) (and (set-member? 'a s1) (= (set-size s1) 3) (equal? s2 '(b c))))",
    "set-remove": "(and (equal? (set-remove 'b '(a b c)) '(a c)) (equal? (set-remove 'z '(a b c)) '(a b c)))",
    "set-union": "(let ([u (set-union '(1 2) '(2 3))]) (and (= (set-size u) 3) (set-member? 1 u) (set-member? 2 u) (set-member? 3 u)))",
    "set-intersection": "(equal? (set-intersection '(1 2 3) '(2 3 4)) '(2 3))",
    "set-difference": "(equal? (set-difference '(1 2 3 4) '(2 4 6)) '(1 3))",
    "set-subset?": "(and (set-subset? '(1 2) '(3 2 1)) (not (set-subset? '(1 4) '(1 2 3))))",
    "set-size": "(and (= (set-size '()) 0) (= (set-size '(a b c)) 3))",
    "set->list": "(and (equal? (set->list '(c b a)) '(c b a)) (equal? (set->list '()) '()))",
    "list->set": "(let ([s (list->set '(a b a c b))]) (and (= (set-size s) 3) (set-member? 'a s) (set-member? 'b s) (set-member? 'c s)))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "set-empty?": "def set_empty(s):\n    return len(s) == 0",
    "set-member?": "def set_member(elem, s):\n    return elem in s",
    "set-add": "def set_add(elem, s):\n    return s if elem in s else [elem] + s",
    "set-remove": "def set_remove(elem, s):\n    out = []\n    removed = False\n    for x in s:\n        if not removed and x == elem:\n            removed = True\n        else:\n            out.append(x)\n    return out",
    "set-union": "def set_union(s1, s2):\n    out = list(s2)\n    for x in s1:\n        if x not in out:\n            out.insert(0, x)\n    return out",
    "set-intersection": "def set_intersection(s1, s2):\n    return [x for x in s1 if x in s2]",
    "set-difference": "def set_difference(s1, s2):\n    return [x for x in s1 if x not in s2]",
    "set-subset?": "def set_subset(s1, s2):\n    return all(x in s2 for x in s1)",
    "set-size": "def set_size(s):\n    return len(s)",
    "set->list": "def set_to_list(s):\n    return s",
    "list->set": "def list_to_set(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.insert(0, x)\n    return out",
}

CHEZ_SNIPPETS = {
    "set-empty?": "(define (empty? s)\n  (null? s))",
    "set-member?": "(define (member0 x s)\n  (cond [(null? s) #f] [(equal? x (car s)) #t] [else (member0 x (cdr s))]))",
    "set-add": "(define (add x s)\n  (if (set-member? x s) s (cons x s)))",
    "set-remove": "(define (remove1 x s)\n  (cond [(null? s) '()] [(equal? x (car s)) (cdr s)] [else (cons (car s) (remove1 x (cdr s)))]))",
    "set-union": "(define (union0 s1 s2)\n  (fold-left (lambda (acc x) (set-add x acc)) s2 s1))",
    "set-intersection": "(define (intersect0 s1 s2)\n  (filter (lambda (x) (set-member? x s2)) s1))",
    "set-difference": "(define (diff0 s1 s2)\n  (filter (lambda (x) (not (set-member? x s2))) s1))",
    "set-subset?": "(define (subset0 s1 s2)\n  (fold-left (lambda (ok x) (and ok (set-member? x s2))) #t s1))",
    "set-size": "(define (size0 s)\n  (length s))",
    "set->list": "(define (to-list s)\n  s)",
    "list->set": "(define (from-list xs)\n  (fold-left (lambda (acc x) (set-add x acc)) '() xs))",
}

BUGGY_CASES = [
    {
        "fn": "set-empty?",
        "buggy": "(define (set-empty? set)\n  (null? (cdr set)))",
        "note": "A one-element set should not be empty.",
    },
    {
        "fn": "set-empty?",
        "buggy": "(define (set-empty? set)\n  #f)",
        "note": "Empty set must return #t.",
    },
    {
        "fn": "set-member?",
        "buggy": "(define (set-member? elem set)\n  (and (not (null? set)) (equal? elem (car set))))",
        "note": "Membership must search entire set, not just the head.",
    },
    {
        "fn": "set-member?",
        "buggy": "(define (set-member? elem set)\n  #t)",
        "note": "Absent elements must return #f.",
    },
    {
        "fn": "set-add",
        "buggy": "(define (set-add elem set)\n  (cons elem set))",
        "note": "Adding an existing element must not create duplicates.",
    },
    {
        "fn": "set-add",
        "buggy": "(define (set-add elem set)\n  set)",
        "note": "Missing element should actually be added.",
    },
    {
        "fn": "set-remove",
        "buggy": "(define (set-remove elem set)\n  set)",
        "note": "Existing element should be removed.",
    },
    {
        "fn": "set-remove",
        "buggy": "(define (set-remove elem set)\n  (if (null? set) '() (cdr set)))",
        "note": "Removal should match by value, not always drop head.",
    },
    {
        "fn": "set-union",
        "buggy": "(define (set-union set1 set2)\n  (append set1 set2))",
        "note": "Union must avoid duplicates.",
    },
    {
        "fn": "set-union",
        "buggy": "(define (set-union set1 set2)\n  set1)",
        "note": "Union should include elements unique to set2.",
    },
    {
        "fn": "set-intersection",
        "buggy": "(define (set-intersection set1 set2)\n  set1)",
        "note": "Intersection must keep only common elements.",
    },
    {
        "fn": "set-intersection",
        "buggy": "(define (set-intersection set1 set2)\n  (set-union set1 set2))",
        "note": "Union is not intersection.",
    },
    {
        "fn": "set-difference",
        "buggy": "(define (set-difference set1 set2)\n  set2)",
        "note": "Difference should come from set1.",
    },
    {
        "fn": "set-difference",
        "buggy": "(define (set-difference set1 set2)\n  (filter (lambda (x) (set-member? x set2)) set1))",
        "note": "This computes intersection, not difference.",
    },
    {
        "fn": "set-subset?",
        "buggy": "(define (set-subset? set1 set2)\n  (let loop ([remaining set2])\n    (cond\n      [(null? remaining) #t]\n      [(set-member? (car remaining) set1)\n       (loop (cdr remaining))]\n      [else #f])))",
        "note": "Subset direction is reversed.",
    },
    {
        "fn": "set-subset?",
        "buggy": "(define (set-subset? set1 set2)\n  #t)",
        "note": "Non-subsets must return #f.",
    },
    {
        "fn": "set-size",
        "buggy": "(define (set-size set)\n  (if (null? set) 1 (length set)))",
        "note": "Empty set size must be 0.",
    },
    {
        "fn": "set-size",
        "buggy": "(define (set-size set)\n  (- (length set) 1))",
        "note": "Size should match length exactly.",
    },
    {
        "fn": "set->list",
        "buggy": "(define (set->list set)\n  (reverse set))",
        "note": "Set->list is identity in this representation.",
    },
    {
        "fn": "set->list",
        "buggy": "(define (set->list set)\n  '())",
        "note": "Conversion must preserve elements.",
    },
    {
        "fn": "list->set",
        "buggy": "(define (list->set lst)\n  lst)",
        "note": "Duplicates should be removed.",
    },
    {
        "fn": "list->set",
        "buggy": "(define (list->set lst)\n  (reverse lst))",
        "note": "Reversing alone does not enforce set semantics.",
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
    sid = f"set_{family}_{family_counter[family]:03d}"
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
# Family 1: spec_to_code (22)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    diff = "easy" if fn in {"set-empty?", "set-member?", "set-size", "set->list", "list->set"} else "medium"

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=diff,
        source_function=fn,
        prompt=f"""You are implementing Tier-0 data-structure code in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "data", "set", "spec-to-code", fn],
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
        tags=["tier0", "data", "set", "skeleton-completion", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (22)
# -----------------------------------------------------------------------------
for fn in TRANSLATION_FUNCTIONS:
    diff = "easy" if fn in {"set-empty?", "set-member?", "set-size", "set->list", "list->set"} else "medium"

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
        tags=["tier0", "data", "set", "python-to-scheme", fn],
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
        tags=["tier0", "data", "set", "chez-to-fold", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (22)
# -----------------------------------------------------------------------------
for case in BUGGY_CASES:
    fn = case["fn"]
    diff = "easy" if fn in {"set-empty?", "set-member?", "set-size", "set->list", "list->set"} else "medium"

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
        tags=["tier0", "data", "set", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition/use (30)
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
        tags=["tier0", "data", "set", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # Direct
    ("set-empty?", "Return whether `set-empty` is empty.", "(set-empty? set-empty)", "(equal? (set-empty? set-empty) #t)", "easy", ["direct"]),
    ("set-empty?", "Return whether set `'(a)` is empty.", "(set-empty? '(a))", "(equal? (set-empty? '(a)) #f)", "easy", ["direct"]),
    ("set-member?", "Check membership of `'b` in set `'(a b c)`.", "(set-member? 'b '(a b c))", "(equal? (set-member? 'b '(a b c)) #t)", "easy", ["direct"]),
    ("set-member?", "Check membership of `'z` in set `'(a b c)`.", "(set-member? 'z '(a b c))", "(equal? (set-member? 'z '(a b c)) #f)", "easy", ["direct"]),
    ("set-add", "Add `'x` to set `'(a b)`.", "(set-add 'x '(a b))", "(equal? (set-add 'x '(a b)) '(x a b))", "easy", ["direct"]),
    ("set-remove", "Remove `'b` from set `'(a b c)`.", "(set-remove 'b '(a b c))", "(equal? (set-remove 'b '(a b c)) '(a c))", "easy", ["direct"]),
    ("set-union", "Return union of sets `'(1 2)` and `'(2 3)`.", "(set-union '(1 2) '(2 3))", "(let ([u (set-union '(1 2) '(2 3))]) (and (= (set-size u) 3) (set-member? 1 u) (set-member? 2 u) (set-member? 3 u)))", "medium", ["direct"]),
    ("set-intersection", "Return intersection of `'(1 2 3)` and `'(2 3 4)`.", "(set-intersection '(1 2 3) '(2 3 4))", "(equal? (set-intersection '(1 2 3) '(2 3 4)) '(2 3))", "medium", ["direct"]),
    ("set-difference", "Return difference `set1 - set2` for `'(1 2 3 4)` and `'(2 4)`.", "(set-difference '(1 2 3 4) '(2 4))", "(equal? (set-difference '(1 2 3 4) '(2 4)) '(1 3))", "medium", ["direct"]),
    ("set-subset?", "Check whether `'(1 2)` is subset of `'(3 2 1)`.", "(set-subset? '(1 2) '(3 2 1))", "(equal? (set-subset? '(1 2) '(3 2 1)) #t)", "easy", ["direct"]),

    # Properties
    ("set-add", "Return #t iff adding an existing element does not change set size.", "(= (set-size (set-add 'b '(a b c))) (set-size '(a b c)))", "(equal? (= (set-size (set-add 'b '(a b c))) (set-size '(a b c))) #t)", "medium", ["property"]),
    ("set-remove", "Return #t iff removing an absent element leaves the set unchanged.", "(equal? (set-remove 'z '(a b c)) '(a b c))", "(equal? (equal? (set-remove 'z '(a b c)) '(a b c)) #t)", "medium", ["property"]),
    ("set-subset?", "Check subset reflexivity for `'(x y z)`.", "(set-subset? '(x y z) '(x y z))", "(equal? (set-subset? '(x y z) '(x y z)) #t)", "easy", ["property"]),
    ("set-subset?", "Check that empty set is subset of `'(a b c)`.", "(set-subset? '() '(a b c))", "(equal? (set-subset? '() '(a b c)) #t)", "easy", ["property"]),
    ("set-union", "Return #t iff union is equivalent regardless of argument order for `'(1 2)` and `'(2 3)`.", "(let ([u1 (set-union '(1 2) '(2 3))] [u2 (set-union '(2 3) '(1 2))]) (and (set-subset? u1 u2) (set-subset? u2 u1)))", "(equal? (let ([u1 (set-union '(1 2) '(2 3))] [u2 (set-union '(2 3) '(1 2))]) (and (set-subset? u1 u2) (set-subset? u2 u1))) #t)", "hard", ["property"]),
    ("set-intersection", "Return #t iff intersection is subset of the left set.", "(let ([i (set-intersection '(1 2 3) '(2 3 4))]) (set-subset? i '(1 2 3)))", "(equal? (let ([i (set-intersection '(1 2 3) '(2 3 4))]) (set-subset? i '(1 2 3))) #t)", "medium", ["property"]),
    ("set-difference", "Return #t iff removed elements are absent after difference.", "(let ([d (set-difference '(1 2 3 4) '(2 4))]) (and (not (set-member? 2 d)) (not (set-member? 4 d))))", "(equal? (let ([d (set-difference '(1 2 3 4) '(2 4))]) (and (not (set-member? 2 d)) (not (set-member? 4 d)))) #t)", "medium", ["property"]),
    ("list->set", "Return #t iff list->set followed by set->list then list->set preserves set size.", "(let ([s (list->set '(a b a c b))]) (= (set-size s) (set-size (list->set (set->list s)))))", "(equal? (let ([s (list->set '(a b a c b))]) (= (set-size s) (set-size (list->set (set->list s))))) #t)", "medium", ["property"]),

    # List/fold/loop
    ("set-add", "Build a set from `'(1 2 3 2 1 4)` using `fold-left` and `set-add`.", "(fold-left (lambda (s x) (set-add x s)) set-empty '(1 2 3 2 1 4))", "(let ([s (fold-left (lambda (s x) (set-add x s)) set-empty '(1 2 3 2 1 4))]) (and (= (set-size s) 4) (set-member? 1 s) (set-member? 2 s) (set-member? 3 s) (set-member? 4 s)))", "medium", ["fold"]),
    ("set-member?", "Map membership of `'a` across sets `'((a b) (b c) (a c) ())`.", "(map (lambda (s) (set-member? 'a s)) '((a b) (b c) (a c) ()))", "(equal? (map (lambda (s) (set-member? 'a s)) '((a b) (b c) (a c) ())) '(#t #f #t #f))", "medium", ["list"]),
    ("set-union", "Fold union over set list `'((1 2) (2 3) (3 4))`.", "(fold-left (lambda (acc s) (set-union acc s)) set-empty '((1 2) (2 3) (3 4)))", "(let ([u (fold-left (lambda (acc s) (set-union acc s)) set-empty '((1 2) (2 3) (3 4)))]) (and (= (set-size u) 4) (set-member? 1 u) (set-member? 2 u) (set-member? 3 u) (set-member? 4 u)))", "hard", ["fold"]),
    ("set-intersection", "Use a loop to intersect `'(1 2 3 4)` with each set in `'((2 3 5) (3 4) (3 6))`.", "(let loop ([cur '(1 2 3 4)] [sets '((2 3 5) (3 4) (3 6))]) (if (null? sets) cur (loop (set-intersection cur (car sets)) (cdr sets))))", "(equal? (let loop ([cur '(1 2 3 4)] [sets '((2 3 5) (3 4) (3 6))]) (if (null? sets) cur (loop (set-intersection cur (car sets)) (cdr sets)))) '(3))", "hard", ["loop"]),
    ("set-size", "Count how many elements in `'(a b c d)` are missing from set `'(a c)`.", "(let loop ([xs '(a b c d)] [n 0]) (if (null? xs) n (loop (cdr xs) (if (set-member? (car xs) '(a c)) n (+ n 1)))))", "(equal? (let loop ([xs '(a b c d)] [n 0]) (if (null? xs) n (loop (cdr xs) (if (set-member? (car xs) '(a c)) n (+ n 1))))) 2)", "medium", ["loop"]),
    ("set-size", "Map set sizes over `'((a) (a b) (a b c) ())`.", "(map set-size '((a) (a b) (a b c) ()))", "(equal? (map set-size '((a) (a b) (a b c) ())) '(1 2 3 0))", "easy", ["list"]),

    # Integration
    ("set-difference", "Compute symmetric difference of `'(1 2 3)` and `'(3 4)` as union of two differences.", "(set-union (set-difference '(1 2 3) '(3 4)) (set-difference '(3 4) '(1 2 3)))", "(let ([s (set-union (set-difference '(1 2 3) '(3 4)) (set-difference '(3 4) '(1 2 3)))]) (and (= (set-size s) 3) (set-member? 1 s) (set-member? 2 s) (set-member? 4 s)))", "hard", ["integration"]),
    ("set-intersection", "Return overlap size between `'(a b c d)` and `'(c d e)`.", "(set-size (set-intersection '(a b c d) '(c d e)))", "(equal? (set-size (set-intersection '(a b c d) '(c d e))) 2)", "medium", ["integration"]),
    ("set-difference", "From `users='(u1 u2 u3 u4)` remove banned `'(u2 u4)` and return remaining users.", "(set-difference '(u1 u2 u3 u4) '(u2 u4))", "(equal? (set-difference '(u1 u2 u3 u4) '(u2 u4)) '(u1 u3))", "medium", ["integration"]),
    ("list->set", "Return #t iff deduplicating `'(a a b c c c)` yields set size 3.", "(= (set-size (list->set '(a a b c c c))) 3)", "(equal? (= (set-size (list->set '(a a b c c c))) 3) #t)", "easy", ["integration"]),
    ("set-subset?", "Check if intersection is subset of each original set for `A='(1 2 3)` and `B='(2 3 4)`.", "(let ([i (set-intersection '(1 2 3) '(2 3 4))]) (and (set-subset? i '(1 2 3)) (set-subset? i '(2 3 4))))", "(equal? (let ([i (set-intersection '(1 2 3) '(2 3 4))]) (and (set-subset? i '(1 2 3)) (set-subset? i '(2 3 4)))) #t)", "medium", ["integration"]),
    ("set-union", "Build union of three sets and test that element `5` appears.", "(let ([u (set-union (set-union '(1 2) '(2 3)) '(3 4 5))]) (set-member? 5 u))", "(equal? (let ([u (set-union (set-union '(1 2) '(2 3)) '(3 4 5))]) (set-member? 5 u)) #t)", "medium", ["integration"]),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

if sum(1 for s in samples if s["family"] == "composition") != 30:
    raise ValueError("composition family must contain exactly 30 samples")

# -----------------------------------------------------------------------------
# Split train/eval
# -----------------------------------------------------------------------------
if len(samples) != 96:
    raise ValueError(f"expected 96 samples, got {len(samples)}")

by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

EVAL_QUOTA = {
    "spec_to_code": 4,
    "translation": 4,
    "bugfix": 4,
    "composition": 6,
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

if len(train_rows) != 78 or len(eval_rows) != 18:
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
