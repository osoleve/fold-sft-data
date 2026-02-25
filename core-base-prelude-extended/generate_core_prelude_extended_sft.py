#!/usr/bin/env python3
"""Generate SFT samples for additional core/base/prelude.ss functions."""

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

SOURCE_MODULE = "core/base/prelude.ss"
SOURCE_TEST = "core/base/test-prelude.ss"

SUPPORT_DEFS: Dict[str, str] = {
    "filter": """(define (filter pred lst)
  (cond
    [(null? lst) '()]
    [(pred (car lst)) (cons (car lst) (filter pred (cdr lst)))]
    [else (filter pred (cdr lst))]))""",
}

DEFS: Dict[str, str] = {
    "unique-simple": """(define (unique-simple lst)
  (let loop ([lst lst] [seen '()] [acc '()])
       (cond
        [(null? lst) (reverse acc)]
        [(memq (car lst) seen) (loop (cdr lst) seen acc)]
        [else (loop (cdr lst) (cons (car lst) seen) (cons (car lst) acc))])))""",
    "unique-fast": """(define (unique-fast lst)
  (let ([seen (make-hashtable equal-hash equal?)])
       (let loop ([items lst] [acc '()])
            (if (null? items)
                (reverse acc)
                (let ([x (car items)])
                     (if (hashtable-contains? seen x)
                         (loop (cdr items) acc)
                         (begin
                          (hashtable-set! seen x #t)
                          (loop (cdr items) (cons x acc)))))))))""",
    "cons*": """(define (cons* . args)
  (cond
   [(null? args) (error 'cons* "requires at least one argument")]
   [(null? (cdr args)) (car args)]
   [else (cons (car args) (apply cons* (cdr args)))]))""",
    "assoc-ref": """(define (assoc-ref alist key)
  (let ([pair (assoc key alist)])
    (and pair (cdr pair))))""",
    "assq-ref": """(define (assq-ref alist key)
  (let ([pair (assq key alist)])
    (and pair (cdr pair))))""",
    "alist-update": """(define (alist-update alist key value)
  (cons (cons key value)
        (filter (lambda (pair) (not (equal? (car pair) key))) alist)))""",
    "ok?": """(define (ok? result)
  (and (pair? result) (eq? (car result) 'ok)))""",
    "error?": """(define (error? result)
  (and (pair? result) (eq? (car result) 'error)))""",
    "unwrap-ok": """(define (unwrap-ok result)
  (cadr result))""",
    "unwrap-error": """(define (unwrap-error result)
  (cdr result))""",
    "result-map": """(define (result-map f result)
  (if (ok? result)
      `(ok ,(f (unwrap-ok result)))
      result))""",
    "result-bind": """(define (result-bind result f)
  (if (ok? result)
      (f (unwrap-ok result))
      result))""",
    "result-sequence": """(define (result-sequence results)
  (if (null? results)
      '(ok ())
      (let ([first (car results)])
           (if (error? first)
               first
               (let ([rest (result-sequence (cdr results))])
                    (if (error? rest)
                        rest
                        `(ok ,(cons (unwrap-ok first) (unwrap-ok rest)))))))))""",
}

FUNCTION_ORDER = [
    "unique-simple",
    "unique-fast",
    "cons*",
    "assoc-ref",
    "assq-ref",
    "alist-update",
    "ok?",
    "error?",
    "unwrap-ok",
    "unwrap-error",
    "result-map",
    "result-bind",
    "result-sequence",
]

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **DEFS}

DEPENDS: Dict[str, List[str]] = {
    "filter": [],
    "unique-simple": [],
    "unique-fast": [],
    "cons*": [],
    "assoc-ref": [],
    "assq-ref": [],
    "alist-update": ["filter"],
    "ok?": [],
    "error?": [],
    "unwrap-ok": [],
    "unwrap-error": [],
    "result-map": ["ok?", "unwrap-ok"],
    "result-bind": ["ok?", "unwrap-ok"],
    "result-sequence": ["error?", "unwrap-ok"],
}

FUNCTION_SPECS = {
    "unique-simple": "Remove duplicates using memq (eq?) while preserving first occurrence order.",
    "unique-fast": "Remove duplicates using a hashtable (equal?) while preserving first occurrence order.",
    "cons*": "Build an improper list from arguments; with one arg return it directly.",
    "assoc-ref": "Lookup key using assoc/equal? and return value or #f.",
    "assq-ref": "Lookup symbol key using assq/eq? and return value or #f.",
    "alist-update": "Return new alist with key updated/replaced at front and old entries removed.",
    "ok?": "Return #t iff value is tagged `(ok ...)`.",
    "error?": "Return #t iff value is tagged `(error ...)`.",
    "unwrap-ok": "Extract ok payload from `(ok value)`.",
    "unwrap-error": "Extract error payload tail from `(error tag ...)`.",
    "result-map": "Apply function to ok payload and pass through errors unchanged.",
    "result-bind": "Monadic bind: apply f to ok payload, short-circuit errors.",
    "result-sequence": "Convert list of results to result of list, stopping at first error.",
}

SKELETONS = {
    "unique-simple": """(define (unique-simple lst)
  ;; TODO: O(n^2) duplicate removal using memq
  <TODO>)""",
    "unique-fast": """(define (unique-fast lst)
  ;; TODO: O(n) duplicate removal using hashtable
  <TODO>)""",
    "cons*": """(define (cons* . args)
  ;; TODO: improper-list constructor with variadic args
  <TODO>)""",
    "assoc-ref": """(define (assoc-ref alist key)
  ;; TODO: assoc lookup by equal?
  <TODO>)""",
    "assq-ref": """(define (assq-ref alist key)
  ;; TODO: assq lookup by eq?
  <TODO>)""",
    "alist-update": """(define (alist-update alist key value)
  ;; TODO: insert new key/value and remove stale entries
  <TODO>)""",
    "ok?": """(define (ok? result)
  ;; TODO: tag check for ok
  <TODO>)""",
    "error?": """(define (error? result)
  ;; TODO: tag check for error
  <TODO>)""",
    "unwrap-ok": """(define (unwrap-ok result)
  ;; TODO: extract ok payload
  <TODO>)""",
    "unwrap-error": """(define (unwrap-error result)
  ;; TODO: extract error payload
  <TODO>)""",
    "result-map": """(define (result-map f result)
  ;; TODO: map only successful results
  <TODO>)""",
    "result-bind": """(define (result-bind result f)
  ;; TODO: bind successful result through f
  <TODO>)""",
    "result-sequence": """(define (result-sequence results)
  ;; TODO: sequence list of results with short-circuiting
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "unique-simple": "(equal? (unique-simple '(a b a c b a)) '(a b c))",
    "unique-fast": "(equal? (unique-fast '(\"a\" \"b\" \"a\" \"c\" \"b\")) '(\"a\" \"b\" \"c\"))",
    "cons*": "(and (equal? (cons* 'a) 'a) (equal? (cons* 'a 'b) '(a . b)) (equal? (cons* 'a 'b 'c) '(a b . c)))",
    "assoc-ref": "(and (equal? (assoc-ref '((a . 1) (b . 2)) 'b) 2) (not (assoc-ref '((a . 1)) 'z)))",
    "assq-ref": "(and (equal? (assq-ref '((a . 1) (b . 2)) 'a) 1) (not (assq-ref '((a . 1)) 'z)))",
    "alist-update": "(let ([updated (alist-update '((a . 1) (b . 2)) 'a 9)] [inserted (alist-update '((a . 1)) 'c 7)]) (and (equal? (assoc-ref updated 'a) 9) (equal? (assoc-ref inserted 'c) 7) (= (length (filter (lambda (p) (equal? (car p) 'a)) updated)) 1)))",
    "ok?": "(and (ok? '(ok 42)) (not (ok? '(error bad))))",
    "error?": "(and (error? '(error bad detail)) (not (error? '(ok 1))))",
    "unwrap-ok": "(equal? (unwrap-ok '(ok (1 2))) '(1 2))",
    "unwrap-error": "(equal? (unwrap-error '(error bad detail)) '(bad detail))",
    "result-map": "(and (equal? (result-map (lambda (x) (* x 2)) '(ok 5)) '(ok 10)) (equal? (result-map (lambda (x) (* x 2)) '(error bad)) '(error bad)))",
    "result-bind": "(and (equal? (result-bind '(ok 5) (lambda (x) `(ok ,(+ x 1)))) '(ok 6)) (equal? (result-bind '(error bad) (lambda (x) `(ok ,(+ x 1)))) '(error bad)))",
    "result-sequence": "(and (equal? (result-sequence '((ok 1) (ok 2) (ok 3))) '(ok (1 2 3))) (equal? (result-sequence '((ok 1) (error bad) (ok 3))) '(error bad)) (equal? (result-sequence '()) '(ok ())))",
}

PYTHON_SNIPPETS = {
    "unique-simple": "def unique_simple(xs):\n    seen=[]\n    out=[]\n    for x in xs:\n        if x not in seen:\n            seen.append(x)\n            out.append(x)\n    return out",
    "unique-fast": "def unique_fast(xs):\n    seen=set()\n    out=[]\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
    "cons*": "def cons_star(*args):\n    if len(args)==0:\n        raise ValueError('requires at least one argument')\n    if len(args)==1:\n        return args[0]\n    return (args[0], cons_star(*args[1:]))",
    "assoc-ref": "def assoc_ref(alist, key):\n    for k,v in alist:\n        if k == key:\n            return v\n    return None",
    "assq-ref": "def assq_ref(alist, key):\n    for k,v in alist:\n        if k is key:\n            return v\n    return None",
    "alist-update": "def alist_update(alist, key, value):\n    return [(key, value)] + [(k,v) for (k,v) in alist if k != key]",
    "ok?": "def is_ok(result):\n    return isinstance(result, (list, tuple)) and len(result) > 0 and result[0] == 'ok'",
    "error?": "def is_error(result):\n    return isinstance(result, (list, tuple)) and len(result) > 0 and result[0] == 'error'",
    "unwrap-ok": "def unwrap_ok(result):\n    return result[1]",
    "unwrap-error": "def unwrap_error(result):\n    return result[1:]",
    "result-map": "def result_map(f, result):\n    if result[0] == 'ok':\n        return ('ok', f(result[1]))\n    return result",
    "result-bind": "def result_bind(result, f):\n    if result[0] == 'ok':\n        return f(result[1])\n    return result",
    "result-sequence": "def result_sequence(results):\n    out=[]\n    for r in results:\n        if r[0] == 'error':\n            return r\n        out.append(r[1])\n    return ('ok', out)",
}

BUGGY_CASES = [
    {"fn": "unique-simple", "buggy": "(define (unique-simple lst)\n  (reverse lst))", "note": "Must remove duplicates while preserving first occurrence."},
    {"fn": "unique-fast", "buggy": "(define (unique-fast lst)\n  (unique-simple (reverse lst)))", "note": "Order and complexity behavior are wrong."},
    {"fn": "cons*", "buggy": "(define (cons* . args)\n  (apply list args))", "note": "cons* builds improper lists, not always proper lists."},
    {"fn": "assoc-ref", "buggy": "(define (assoc-ref alist key)\n  (let ([pair (assq key alist)])\n    (and pair (cdr pair))))", "note": "assoc-ref must use assoc/equal?, not assq/eq?."},
    {"fn": "assq-ref", "buggy": "(define (assq-ref alist key)\n  (let ([pair (assoc key alist)])\n    (and pair (cdr pair))))", "note": "assq-ref must use assq/eq? semantics."},
    {"fn": "alist-update", "buggy": "(define (alist-update alist key value)\n  (append alist (list (cons key value))))", "note": "Must replace existing key entries and put new mapping at front."},
    {"fn": "ok?", "buggy": "(define (ok? result)\n  (and (pair? result) (eq? (car result) 'error)))", "note": "Tag check is inverted."},
    {"fn": "error?", "buggy": "(define (error? result)\n  (and (pair? result) (eq? (car result) 'ok)))", "note": "Tag check is inverted."},
    {"fn": "unwrap-ok", "buggy": "(define (unwrap-ok result)\n  (car result))", "note": "Should return payload, not tag."},
    {"fn": "unwrap-error", "buggy": "(define (unwrap-error result)\n  (cadr result))", "note": "Should return full error payload tail."},
    {"fn": "result-map", "buggy": "(define (result-map f result)\n  `(ok ,(f (unwrap-ok result))))", "note": "Errors must pass through unchanged."},
    {"fn": "result-bind", "buggy": "(define (result-bind result f)\n  (result-map f result))", "note": "Bind should return f's result directly for ok case."},
    {"fn": "result-sequence", "buggy": "(define (result-sequence results)\n  '(ok ()))", "note": "Must preserve ok values and short-circuit on first error."},
]

DIFFICULTY = {
    "unique-simple": "easy",
    "unique-fast": "medium",
    "cons*": "medium",
    "assoc-ref": "easy",
    "assq-ref": "easy",
    "alist-update": "medium",
    "ok?": "easy",
    "error?": "easy",
    "unwrap-ok": "easy",
    "unwrap-error": "easy",
    "result-map": "medium",
    "result-bind": "medium",
    "result-sequence": "hard",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
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
    sid = f"core_prelude_ext_{family}_{family_counter[family]:03d}"
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
    return [name for name in FUNCTION_ORDER + list(SUPPORT_DEFS.keys()) if name != fn and name in tokens]


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
    parts = [ALL_DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (26)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""You are implementing additional core prelude utilities in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["core", "base", "prelude", "extended", "spec-to-code", fn],
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
        tags=["core", "base", "prelude", "extended", "skeleton-completion", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (13)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
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
        tags=["core", "base", "prelude", "extended", "python-to-scheme", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (13)
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
        tags=["core", "base", "prelude", "extended", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition (26)
# -----------------------------------------------------------------------------

def add_composition(source_function: str, prompt: str, ground_truth: str, verify_expr: str, difficulty: str, extra_tags: List[str]) -> None:
    add_sample(
        family="composition",
        category="usage",
        difficulty=difficulty,
        source_function=source_function,
        prompt=prompt,
        ground_truth=ground_truth,
        verify_expr=verify_expr,
        tags=["core", "base", "prelude", "extended", "composition", source_function] + extra_tags,
    )


composition_cases = [
    ("unique-simple", "Remove duplicates from `(a b a c b a)` with unique-simple.", "(unique-simple '(a b a c b a))", "(equal? (unique-simple '(a b a c b a)) '(a b c))", "easy", ["direct"]),
    ("unique-simple", "Return length of unique-simple over `(x x y y z)`.", "(length (unique-simple '(x x y y z)))", "(equal? (length (unique-simple '(x x y y z))) 3)", "easy", ["integration"]),
    ("unique-fast", "Remove duplicates from string list with unique-fast.", "(unique-fast '(\"a\" \"b\" \"a\" \"c\" \"b\"))", "(equal? (unique-fast '(\"a\" \"b\" \"a\" \"c\" \"b\")) '(\"a\" \"b\" \"c\"))", "medium", ["direct"]),
    ("unique-fast", "Return #t iff unique-fast preserves first occurrence order.", "(equal? (unique-fast '(3 1 3 2 1)) '(3 1 2))", "(equal? (unique-fast '(3 1 3 2 1)) '(3 1 2))", "medium", ["property"]),
    ("cons*", "Build `(a b . c)` using cons*.", "(cons* 'a 'b 'c)", "(equal? (cons* 'a 'b 'c) '(a b . c))", "medium", ["direct"]),
    ("cons*", "Call cons* with one argument `q`.", "(cons* 'q)", "(equal? (cons* 'q) 'q)", "easy", ["edge-case"]),
    ("assoc-ref", "Lookup key `b` in `((a . 1) (b . 2))`.", "(assoc-ref '((a . 1) (b . 2)) 'b)", "(equal? (assoc-ref '((a . 1) (b . 2)) 'b) 2)", "easy", ["direct"]),
    ("assoc-ref", "Return #f when assoc-ref key is missing.", "(assoc-ref '((a . 1)) 'z)", "(equal? (assoc-ref '((a . 1)) 'z) #f)", "easy", ["edge-case"]),
    ("assq-ref", "Lookup symbol `a` via assq-ref.", "(assq-ref '((a . 1) (b . 2)) 'a)", "(equal? (assq-ref '((a . 1) (b . 2)) 'a) 1)", "easy", ["direct"]),
    ("assq-ref", "Return #f when assq-ref key is missing.", "(assq-ref '((a . 1)) 'z)", "(equal? (assq-ref '((a . 1)) 'z) #f)", "easy", ["edge-case"]),
    ("alist-update", "Update existing key `a` to 9.", "(assoc-ref (alist-update '((a . 1) (b . 2)) 'a 9) 'a)", "(equal? (assoc-ref (alist-update '((a . 1) (b . 2)) 'a 9) 'a) 9)", "medium", ["direct"]),
    ("alist-update", "Insert new key `c` with value 7.", "(assoc-ref (alist-update '((a . 1)) 'c 7) 'c)", "(equal? (assoc-ref (alist-update '((a . 1)) 'c 7) 'c) 7)", "medium", ["direct"]),
    ("ok?", "Check ok? on `(ok 42)`.", "(ok? '(ok 42))", "(equal? (ok? '(ok 42)) #t)", "easy", ["direct"]),
    ("ok?", "Count ok? results in a mixed list.", "(length (filter ok? '((ok 1) (error bad) (ok 2))))", "(equal? (length (filter ok? '((ok 1) (error bad) (ok 2)))) 2)", "medium", ["integration"]),
    ("error?", "Check error? on `(error bad detail)`.", "(error? '(error bad detail))", "(equal? (error? '(error bad detail)) #t)", "easy", ["direct"]),
    ("error?", "Count error? results in a mixed list.", "(length (filter error? '((ok 1) (error bad) (error x))))", "(equal? (length (filter error? '((ok 1) (error bad) (error x)))) 2)", "medium", ["integration"]),
    ("unwrap-ok", "Unwrap `(ok (1 2 3))`.", "(unwrap-ok '(ok (1 2 3)))", "(equal? (unwrap-ok '(ok (1 2 3))) '(1 2 3))", "easy", ["direct"]),
    ("unwrap-ok", "Map unwrap-ok across only ok values.", "(map unwrap-ok '((ok 1) (ok 2) (ok 3)))", "(equal? (map unwrap-ok '((ok 1) (ok 2) (ok 3))) '(1 2 3))", "medium", ["integration"]),
    ("unwrap-error", "Unwrap `(error bad detail)` payload.", "(unwrap-error '(error bad detail))", "(equal? (unwrap-error '(error bad detail)) '(bad detail))", "easy", ["direct"]),
    ("unwrap-error", "Unwrap error with one tag.", "(unwrap-error '(error oops))", "(equal? (unwrap-error '(error oops)) '(oops))", "easy", ["edge-case"]),
    ("result-map", "Double an ok result using result-map.", "(result-map (lambda (x) (* x 2)) '(ok 5))", "(equal? (result-map (lambda (x) (* x 2)) '(ok 5)) '(ok 10))", "medium", ["direct"]),
    ("result-map", "Return unchanged error through result-map.", "(result-map (lambda (x) (* x 2)) '(error bad))", "(equal? (result-map (lambda (x) (* x 2)) '(error bad)) '(error bad))", "medium", ["edge-case"]),
    ("result-bind", "Bind ok result through increment function.", "(result-bind '(ok 5) (lambda (x) `(ok ,(+ x 1))))", "(equal? (result-bind '(ok 5) (lambda (x) `(ok ,(+ x 1)))) '(ok 6))", "medium", ["direct"]),
    ("result-bind", "Return unchanged error through result-bind.", "(result-bind '(error bad) (lambda (x) `(ok ,(+ x 1))))", "(equal? (result-bind '(error bad) (lambda (x) `(ok ,(+ x 1)))) '(error bad))", "medium", ["edge-case"]),
    ("result-sequence", "Sequence three ok results.", "(result-sequence '((ok 1) (ok 2) (ok 3)))", "(equal? (result-sequence '((ok 1) (ok 2) (ok 3))) '(ok (1 2 3)))", "hard", ["direct"]),
    ("result-sequence", "Short-circuit result-sequence on first error.", "(result-sequence '((ok 1) (error bad) (ok 3)))", "(equal? (result-sequence '((ok 1) (error bad) (ok 3))) '(error bad))", "hard", ["direct"]),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

if len([s for s in samples if s["family"] == "spec_to_code"]) != 26:
    raise ValueError("spec_to_code family must contain exactly 26 samples")
if len([s for s in samples if s["family"] == "translation"]) != 13:
    raise ValueError("translation family must contain exactly 13 samples")
if len([s for s in samples if s["family"] == "bugfix"]) != 13:
    raise ValueError("bugfix family must contain exactly 13 samples")
if len([s for s in samples if s["family"] == "composition"]) != 26:
    raise ValueError("composition family must contain exactly 26 samples")
if len(samples) != 78:
    raise ValueError(f"expected 78 samples, got {len(samples)}")


# -----------------------------------------------------------------------------
# Split
# -----------------------------------------------------------------------------
by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

EVAL_QUOTA = {
    "spec_to_code": 5,
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

if len(train_rows) != 62 or len(eval_rows) != 16:
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
