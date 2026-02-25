#!/usr/bin/env python3
"""Generate Tier-1 optics profunctor SFT samples for lattice/optics/profunctor-optics.ss."""

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

SOURCE_MODULE = "lattice/optics/profunctor-optics.ss"
SOURCE_TEST = "lattice/optics/test-profunctor-optics.ss"

SOURCE_DEFS: Dict[str, str] = {
    "make-profunctor": """(define (make-profunctor dimap-fn)
  (list 'profunctor
        dimap-fn
        (lambda (f pa) (dimap-fn f identity pa))
        (lambda (g pa) (dimap-fn identity g pa))))""",
    "profunctor?": """(define (profunctor? x)
  (and (pair? x) (eq? (car x) 'profunctor)))""",
    "profunctor-dimap": """(define (profunctor-dimap p)
  (cadr p))""",
    "profunctor-lmap": """(define (profunctor-lmap p)
  (caddr p))""",
    "profunctor-rmap": """(define (profunctor-rmap p)
  (cadddr p))""",
    "dimap": """(define (dimap prof f g pa)
  ((profunctor-dimap prof) f g pa))""",
    "lmap": """(define (lmap prof f pa)
  ((profunctor-lmap prof) f pa))""",
    "rmap": """(define (rmap prof g pa)
  ((profunctor-rmap prof) g pa))""",
}

SUPPORT_DEFS: Dict[str, str] = {
    "identity": """(define (identity x)
  x)""",
}

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **SOURCE_DEFS}

FUNCTION_ORDER = [
    "make-profunctor",
    "profunctor?",
    "profunctor-dimap",
    "profunctor-lmap",
    "profunctor-rmap",
    "dimap",
    "lmap",
    "rmap",
]

SUPPORT_ORDER = [
    "identity",
]

DEPENDS: Dict[str, List[str]] = {
    "identity": [],
    "make-profunctor": ["identity"],
    "profunctor?": [],
    "profunctor-dimap": [],
    "profunctor-lmap": [],
    "profunctor-rmap": [],
    "dimap": ["profunctor-dimap"],
    "lmap": ["profunctor-lmap"],
    "rmap": ["profunctor-rmap"],
}

FUNCTION_SPECS = {
    "make-profunctor": "Build a profunctor dictionary record with tag, dimap function, and derived lmap/rmap.",
    "profunctor?": "Return #t iff value is a pair tagged with symbol 'profunctor in car position.",
    "profunctor-dimap": "Extract the dimap function from a profunctor dictionary.",
    "profunctor-lmap": "Extract the lmap function from a profunctor dictionary.",
    "profunctor-rmap": "Extract the rmap function from a profunctor dictionary.",
    "dimap": "Apply the profunctor's dimap implementation to pre-map f, post-map g, and payload pa.",
    "lmap": "Apply the profunctor's contravariant map on the input side.",
    "rmap": "Apply the profunctor's covariant map on the output side.",
}

SKELETONS = {
    "make-profunctor": """(define (make-profunctor dimap-fn)
  ;; TODO: return tagged profunctor dictionary with dimap/lmap/rmap
  <TODO>)""",
    "profunctor?": """(define (profunctor? x)
  ;; TODO: recognize profunctor tagged pairs
  <TODO>)""",
    "profunctor-dimap": """(define (profunctor-dimap p)
  ;; TODO: select dimap slot from profunctor record
  <TODO>)""",
    "profunctor-lmap": """(define (profunctor-lmap p)
  ;; TODO: select lmap slot from profunctor record
  <TODO>)""",
    "profunctor-rmap": """(define (profunctor-rmap p)
  ;; TODO: select rmap slot from profunctor record
  <TODO>)""",
    "dimap": """(define (dimap prof f g pa)
  ;; TODO: dispatch through profunctor-dimap
  <TODO>)""",
    "lmap": """(define (lmap prof f pa)
  ;; TODO: dispatch through profunctor-lmap
  <TODO>)""",
    "rmap": """(define (rmap prof g pa)
  ;; TODO: dispatch through profunctor-rmap
  <TODO>)""",
}

DIFFICULTY = {
    "make-profunctor": "hard",
    "profunctor?": "easy",
    "profunctor-dimap": "medium",
    "profunctor-lmap": "medium",
    "profunctor-rmap": "medium",
    "dimap": "medium",
    "lmap": "medium",
    "rmap": "medium",
}

VERIFY_BY_FUNCTION = {
    "make-profunctor": """(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))]
       [d (cadr p)]
       [lm (caddr p)]
       [rm (cadddr p)]
       [h (lambda (n) (+ n 3))]
       [d-fn (d (lambda (n) (* n 2)) (lambda (n) (- n 1)) h)]
       [lm-fn (lm (lambda (n) (* n 2)) h)]
       [rm-fn (rm (lambda (n) (* n 10)) h)])
  (and (= (d-fn 4) 10)
       (= (lm-fn 1) 5)
       (= (rm-fn 1) 40)))""",
    "profunctor?": """(let ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))])
  (and (profunctor? p)
       (profunctor? (cons 'profunctor 'tail))
       (not (profunctor? '(foo bar)))
       (not (profunctor? 'profunctor))
       (not (profunctor? #f))))""",
    "profunctor-dimap": """(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))]
       [d (profunctor-dimap p)]
       [fn (d (lambda (x) (+ x 1)) (lambda (y) (* y 2)) (lambda (z) (- z 3)))])
  (= (fn 5) 6))""",
    "profunctor-lmap": """(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))]
       [lm (profunctor-lmap p)]
       [fn (lm (lambda (x) (+ x 10)) (lambda (z) (* z 2)))])
  (= (fn 3) 26))""",
    "profunctor-rmap": """(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))]
       [rm (profunctor-rmap p)]
       [fn (rm (lambda (y) (- y 4)) (lambda (z) (* z 3)))])
  (= (fn 6) 14))""",
    "dimap": """(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))]
       [fn (dimap p (lambda (x) (+ x 2)) (lambda (y) (* y 3)) (lambda (z) (- z 1)))]
       [lhs (fn 5)]
       [rhs ((rmap p (lambda (y) (* y 3))
                  (lmap p (lambda (x) (+ x 2)) (lambda (z) (- z 1))))
             5)])
  (and (= lhs 18) (= lhs rhs)))""",
    "lmap": """(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))]
       [fn (lmap p (lambda (x) (* x 2)) (lambda (z) (+ z 1)))]
       [cmp (dimap p (lambda (x) (* x 2)) identity (lambda (z) (+ z 1)))])
  (and (= (fn 4) 9)
       (= (fn 7) (cmp 7))))""",
    "rmap": """(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))]
       [fn (rmap p (lambda (y) (+ y 100)) (lambda (z) (* z 2)))]
       [cmp (dimap p identity (lambda (y) (+ y 100)) (lambda (z) (* z 2)))])
  (and (= (fn 3) 106)
       (= (fn 8) (cmp 8))))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "make-profunctor": """def build_profunctor(dimap_fn):
    return [
        "profunctor",
        dimap_fn,
        lambda f, pa: dimap_fn(f, identity, pa),
        lambda g, pa: dimap_fn(identity, g, pa),
    ]""",
    "profunctor?": """def is_profunctor(x):
    return isinstance(x, (list, tuple)) and len(x) > 0 and x[0] == 'profunctor'""",
    "profunctor-dimap": """def get_dimap(p):
    return p[1]""",
    "profunctor-lmap": """def get_lmap(p):
    return p[2]""",
    "profunctor-rmap": """def get_rmap(p):
    return p[3]""",
    "dimap": """def apply_dimap(prof, f, g, pa):
    return get_dimap(prof)(f, g, pa)""",
    "lmap": """def apply_lmap(prof, f, pa):
    return get_lmap(prof)(f, pa)""",
    "rmap": """def apply_rmap(prof, g, pa):
    return get_rmap(prof)(g, pa)""",
}

CHEZ_SNIPPETS = {
    "make-profunctor": """(define (mk-profunctor dimap-fn)
  (list 'profunctor
        dimap-fn
        (lambda (f pa) (dimap-fn f identity pa))
        (lambda (g pa) (dimap-fn identity g pa))))""",
    "profunctor?": """(define (profunctor-tagged? x)
  (and (pair? x) (eq? (car x) 'profunctor)))""",
    "profunctor-dimap": """(define (dimap-slot p)
  (cadr p))""",
    "profunctor-lmap": """(define (lmap-slot p)
  (caddr p))""",
    "profunctor-rmap": """(define (rmap-slot p)
  (cadddr p))""",
    "dimap": """(define (do-dimap prof f g pa)
  ((profunctor-dimap prof) f g pa))""",
    "lmap": """(define (do-lmap prof f pa)
  ((profunctor-lmap prof) f pa))""",
    "rmap": """(define (do-rmap prof g pa)
  ((profunctor-rmap prof) g pa))""",
}

BUGGY_CASES = [
    {
        "fn": "make-profunctor",
        "buggy": """(define (make-profunctor dimap-fn)
  (list 'profunctor
        dimap-fn
        (lambda (f pa) (dimap-fn identity f pa))
        (lambda (g pa) (dimap-fn identity g pa))))""",
        "note": "Derived lmap must pre-map with f and keep identity on output; arguments are swapped.",
    },
    {
        "fn": "make-profunctor",
        "buggy": """(define (make-profunctor dimap-fn)
  (list 'profunctor
        dimap-fn
        (lambda (f pa) (dimap-fn f identity pa))
        (lambda (g pa) (dimap-fn g identity pa))))""",
        "note": "Derived rmap must post-map with g and keep identity on input, not the reverse.",
    },
    {
        "fn": "profunctor?",
        "buggy": """(define (profunctor? x)
  (and (pair? x) (eq? (car x) 'profunct)))""",
        "note": "Tag symbol is misspelled; it must match 'profunctor exactly.",
    },
    {
        "fn": "profunctor?",
        "buggy": """(define (profunctor? x)
  (and (list? x) (eq? (car x) 'profunctor)))""",
        "note": "Recognizer must accept any tagged pair, including improper pairs created with cons.",
    },
    {
        "fn": "profunctor-dimap",
        "buggy": """(define (profunctor-dimap p)
  (caddr p))""",
        "note": "dimap slot is the second element, not the third.",
    },
    {
        "fn": "profunctor-dimap",
        "buggy": """(define (profunctor-dimap p)
  (cadddr p))""",
        "note": "dimap accessor is reading the rmap slot instead of dimap.",
    },
    {
        "fn": "profunctor-lmap",
        "buggy": """(define (profunctor-lmap p)
  (cadr p))""",
        "note": "lmap accessor should return the third slot, not dimap.",
    },
    {
        "fn": "profunctor-lmap",
        "buggy": """(define (profunctor-lmap p)
  (cadddr p))""",
        "note": "lmap accessor is pointing at rmap; use caddr.",
    },
    {
        "fn": "profunctor-rmap",
        "buggy": """(define (profunctor-rmap p)
  (caddr p))""",
        "note": "rmap accessor should return the fourth slot, not lmap.",
    },
    {
        "fn": "profunctor-rmap",
        "buggy": """(define (profunctor-rmap p)
  (cadr p))""",
        "note": "rmap accessor is reading dimap; use cadddr.",
    },
    {
        "fn": "dimap",
        "buggy": """(define (dimap prof f g pa)
  ((profunctor-lmap prof) f pa))""",
        "note": "dimap must apply both input and output mappings through profunctor-dimap.",
    },
    {
        "fn": "dimap",
        "buggy": """(define (dimap prof f g pa)
  ((profunctor-dimap prof) g f pa))""",
        "note": "Input and output maps are reversed; keep order (f g pa).",
    },
    {
        "fn": "lmap",
        "buggy": """(define (lmap prof f pa)
  ((profunctor-rmap prof) f pa))""",
        "note": "lmap must dispatch to profunctor-lmap, not profunctor-rmap.",
    },
    {
        "fn": "lmap",
        "buggy": """(define (lmap prof f pa)
  ((profunctor-lmap prof) pa f))""",
        "note": "lmap argument order is (f pa); this version passes them reversed.",
    },
    {
        "fn": "rmap",
        "buggy": """(define (rmap prof g pa)
  ((profunctor-lmap prof) g pa))""",
        "note": "rmap must dispatch to profunctor-rmap, not profunctor-lmap.",
    },
    {
        "fn": "rmap",
        "buggy": """(define (rmap prof g pa)
  ((profunctor-rmap prof) pa g))""",
        "note": "rmap argument order is (g pa); this version swaps them.",
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
    *,
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
    sample_id = f"optics_profunctor_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sample_id,
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
        ),
        "ground_truth": ground_truth.strip(),
        "verify_expr": verify_expr.strip(),
        "tags": tags,
    }
    for key in REQUIRED_KEYS:
        if key not in sample:
            raise ValueError(f"missing key {key}")
    samples.append(sample)


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
        prompt=f"""Implement this function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "profunctor", "spec-to-code", fn],
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
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "profunctor", "skeleton", fn],
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
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "profunctor", "translation", "python", fn],
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
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "profunctor", "translation", "chez", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
# -----------------------------------------------------------------------------
if len(BUGGY_CASES) != 16:
    raise ValueError(f"expected 16 bugfix cases, got {len(BUGGY_CASES)}")

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
        ground_truth=SOURCE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "profunctor", "bugfix", fn],
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
        tags=["tier1", "optics", "profunctor", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # make-profunctor
    {
        "fn": "make-profunctor",
        "prompt": "Construct a profunctor dictionary with standard dimap semantics and return its tag symbol.",
        "gt": "(car (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x)))))))",
        "verify": "(equal? (car (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))) 'profunctor)",
        "difficulty": "easy",
        "tags": ["record"],
    },
    {
        "fn": "make-profunctor",
        "prompt": "Build a profunctor with canonical dimap behavior, extract dimap, and evaluate the transformed function at 5.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [d (profunctor-dimap p)] [fn (d (lambda (x) (+ x 1)) (lambda (y) (* y 2)) (lambda (z) (- z 3)))]) (fn 5))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [d (profunctor-dimap p)] [fn (d (lambda (x) (+ x 1)) (lambda (y) (* y 2)) (lambda (z) (- z 3)))]) (fn 5)) 6)",
        "difficulty": "medium",
        "tags": ["execution"],
    },
    {
        "fn": "make-profunctor",
        "prompt": "Using a profunctor built by make-profunctor, verify derived lmap matches dimap with identity output mapping.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (+ x 4))] [lhs ((profunctor-lmap p) (lambda (x) (* x 2)) h)] [rhs ((profunctor-dimap p) (lambda (x) (* x 2)) identity h)]) (= (lhs 3) (rhs 3)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (+ x 4))] [lhs ((profunctor-lmap p) (lambda (x) (* x 2)) h)] [rhs ((profunctor-dimap p) (lambda (x) (* x 2)) identity h)]) (= (lhs 3) (rhs 3))) #t)",
        "difficulty": "hard",
        "tags": ["law"],
    },
    {
        "fn": "make-profunctor",
        "prompt": "Using a profunctor built by make-profunctor, verify derived rmap matches dimap with identity input mapping.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (* x 5))] [lhs ((profunctor-rmap p) (lambda (y) (- y 2)) h)] [rhs ((profunctor-dimap p) identity (lambda (y) (- y 2)) h)]) (= (lhs 6) (rhs 6)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (* x 5))] [lhs ((profunctor-rmap p) (lambda (y) (- y 2)) h)] [rhs ((profunctor-dimap p) identity (lambda (y) (- y 2)) h)]) (= (lhs 6) (rhs 6))) #t)",
        "difficulty": "hard",
        "tags": ["law"],
    },

    # profunctor?
    {
        "fn": "profunctor?",
        "prompt": "Classify a valid profunctor dictionary and a plain list using profunctor?.",
        "gt": "(list (profunctor? (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))) (profunctor? '(foo bar)))",
        "verify": "(equal? (list (profunctor? (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))) (profunctor? '(foo bar))) '(#t #f))",
        "difficulty": "easy",
        "tags": ["classification"],
    },
    {
        "fn": "profunctor?",
        "prompt": "Count how many entries in a mixed list are recognized by profunctor?.",
        "gt": "(length (filter profunctor? (list (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x)))))) '(profunctor) (cons 'profunctor 'tail) 7 '())))",
        "verify": "(equal? (length (filter profunctor? (list (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x)))))) '(profunctor) (cons 'profunctor 'tail) 7 '()))) 3)",
        "difficulty": "medium",
        "tags": ["filter"],
    },
    {
        "fn": "profunctor?",
        "prompt": "Check whether an improper pair tagged with 'profunctor is accepted by profunctor?.",
        "gt": "(profunctor? (cons 'profunctor 'tail))",
        "verify": "(equal? (profunctor? (cons 'profunctor 'tail)) #t)",
        "difficulty": "medium",
        "tags": ["pair-shape"],
    },
    {
        "fn": "profunctor?",
        "prompt": "Return whether profunctor? rejects three non-pair inputs.",
        "gt": "(and (not (profunctor? 'profunctor)) (not (profunctor? #f)) (not (profunctor? 0)))",
        "verify": "(equal? (and (not (profunctor? 'profunctor)) (not (profunctor? #f)) (not (profunctor? 0))) #t)",
        "difficulty": "easy",
        "tags": ["negative-cases"],
    },

    # profunctor-dimap
    {
        "fn": "profunctor-dimap",
        "prompt": "Extract dimap from a profunctor dictionary and apply it to evaluate at x=5.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [d (profunctor-dimap p)] [fn (d (lambda (x) (+ x 2)) (lambda (y) (* y 3)) (lambda (z) (- z 1)))]) (fn 5))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [d (profunctor-dimap p)] [fn (d (lambda (x) (+ x 2)) (lambda (y) (* y 3)) (lambda (z) (- z 1)))]) (fn 5)) 18)",
        "difficulty": "medium",
        "tags": ["execution"],
    },
    {
        "fn": "profunctor-dimap",
        "prompt": "Verify profunctor-dimap extraction agrees with the top-level dimap wrapper on the same inputs.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (+ x 7))] [a ((profunctor-dimap p) (lambda (x) (* x 2)) (lambda (y) (- y 5)) h)] [b (dimap p (lambda (x) (* x 2)) (lambda (y) (- y 5)) h)]) (= (a 4) (b 4)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (+ x 7))] [a ((profunctor-dimap p) (lambda (x) (* x 2)) (lambda (y) (- y 5)) h)] [b (dimap p (lambda (x) (* x 2)) (lambda (y) (- y 5)) h)]) (= (a 4) (b 4))) #t)",
        "difficulty": "hard",
        "tags": ["agreement"],
    },
    {
        "fn": "profunctor-dimap",
        "prompt": "Use extracted dimap to build a transformed function and map it over '(1 2 3).",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [d (profunctor-dimap p)] [fn (d identity (lambda (y) (+ y 1)) (lambda (z) (* z 2)))]) (map fn '(1 2 3)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [d (profunctor-dimap p)] [fn (d identity (lambda (y) (+ y 1)) (lambda (z) (* z 2)))]) (map fn '(1 2 3))) '(3 5 7))",
        "difficulty": "medium",
        "tags": ["map"],
    },
    {
        "fn": "profunctor-dimap",
        "prompt": "Apply profunctor-dimap with both nontrivial pre-map and post-map and return the numeric result at input 2.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn ((profunctor-dimap p) (lambda (x) (+ x 10)) (lambda (y) (- y 4)) (lambda (z) (* z 3)))]) (fn 2))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn ((profunctor-dimap p) (lambda (x) (+ x 10)) (lambda (y) (- y 4)) (lambda (z) (* z 3)))]) (fn 2)) 32)",
        "difficulty": "hard",
        "tags": ["pre-post"],
    },

    # profunctor-lmap
    {
        "fn": "profunctor-lmap",
        "prompt": "Extract lmap from a profunctor dictionary and evaluate the transformed function at input 3.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [lm (profunctor-lmap p)] [fn (lm (lambda (x) (+ x 10)) (lambda (z) (* z 2)))]) (fn 3))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [lm (profunctor-lmap p)] [fn (lm (lambda (x) (+ x 10)) (lambda (z) (* z 2)))]) (fn 3)) 26)",
        "difficulty": "medium",
        "tags": ["execution"],
    },
    {
        "fn": "profunctor-lmap",
        "prompt": "Show profunctor-lmap is equivalent to dimap with identity post-map for one input.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (+ x 1))] [a ((profunctor-lmap p) (lambda (x) (* x 3)) h)] [b ((profunctor-dimap p) (lambda (x) (* x 3)) identity h)]) (= (a 6) (b 6)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (+ x 1))] [a ((profunctor-lmap p) (lambda (x) (* x 3)) h)] [b ((profunctor-dimap p) (lambda (x) (* x 3)) identity h)]) (= (a 6) (b 6))) #t)",
        "difficulty": "hard",
        "tags": ["law"],
    },
    {
        "fn": "profunctor-lmap",
        "prompt": "Extract lmap, build a transformed function, and map it over '(1 2 3).",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [lm (profunctor-lmap p)] [fn (lm (lambda (x) (* x 3)) (lambda (z) (- z 1)))]) (map fn '(1 2 3)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [lm (profunctor-lmap p)] [fn (lm (lambda (x) (* x 3)) (lambda (z) (- z 1)))]) (map fn '(1 2 3))) '(2 5 8))",
        "difficulty": "medium",
        "tags": ["map"],
    },
    {
        "fn": "profunctor-lmap",
        "prompt": "Use profunctor-lmap with a pair-producing base function and return the transformed pair at input 3.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [lm (profunctor-lmap p)] [fn (lm (lambda (x) (* x 2)) (lambda (x) (cons x (+ x 1))))]) (fn 3))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [lm (profunctor-lmap p)] [fn (lm (lambda (x) (* x 2)) (lambda (x) (cons x (+ x 1))))]) (fn 3)) '(6 . 7))",
        "difficulty": "hard",
        "tags": ["shape-change"],
    },

    # profunctor-rmap
    {
        "fn": "profunctor-rmap",
        "prompt": "Extract rmap from a profunctor dictionary and evaluate the transformed function at input 6.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [rm (profunctor-rmap p)] [fn (rm (lambda (y) (- y 4)) (lambda (z) (* z 3)))]) (fn 6))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [rm (profunctor-rmap p)] [fn (rm (lambda (y) (- y 4)) (lambda (z) (* z 3)))]) (fn 6)) 14)",
        "difficulty": "medium",
        "tags": ["execution"],
    },
    {
        "fn": "profunctor-rmap",
        "prompt": "Show profunctor-rmap is equivalent to dimap with identity input map for one input.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (+ x 2))] [a ((profunctor-rmap p) (lambda (y) (* y 5)) h)] [b ((profunctor-dimap p) identity (lambda (y) (* y 5)) h)]) (= (a 3) (b 3)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (+ x 2))] [a ((profunctor-rmap p) (lambda (y) (* y 5)) h)] [b ((profunctor-dimap p) identity (lambda (y) (* y 5)) h)]) (= (a 3) (b 3))) #t)",
        "difficulty": "hard",
        "tags": ["law"],
    },
    {
        "fn": "profunctor-rmap",
        "prompt": "Extract rmap, build an output-transformed function, and map it over '(1 2 3).",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [rm (profunctor-rmap p)] [fn (rm (lambda (y) (* y y)) (lambda (x) (+ x 1)))]) (map fn '(1 2 3)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [rm (profunctor-rmap p)] [fn (rm (lambda (y) (* y y)) (lambda (x) (+ x 1)))]) (map fn '(1 2 3))) '(4 9 16))",
        "difficulty": "medium",
        "tags": ["map"],
    },
    {
        "fn": "profunctor-rmap",
        "prompt": "Use profunctor-rmap to post-process a numeric result into a tagged pair.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [rm (profunctor-rmap p)] [fn (rm (lambda (y) (cons 'out y)) (lambda (x) (* x 4)))]) (fn 3))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [rm (profunctor-rmap p)] [fn (rm (lambda (y) (cons 'out y)) (lambda (x) (* x 4)))]) (fn 3)) '(out . 12))",
        "difficulty": "hard",
        "tags": ["shape-change"],
    },

    # dimap
    {
        "fn": "dimap",
        "prompt": "Apply dimap through a function profunctor and evaluate at input 5.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn (dimap p (lambda (x) (+ x 2)) (lambda (y) (* y 3)) (lambda (z) (- z 1)))]) (fn 5))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn (dimap p (lambda (x) (+ x 2)) (lambda (y) (* y 3)) (lambda (z) (- z 1)))]) (fn 5)) 18)",
        "difficulty": "medium",
        "tags": ["execution"],
    },
    {
        "fn": "dimap",
        "prompt": "Check that dimap with identity/identity leaves behavior unchanged for a sample function.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (- x 2))] [a ((dimap p identity identity h) 9)] [b (h 9)]) (= a b))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (- x 2))] [a ((dimap p identity identity h) 9)] [b (h 9)]) (= a b)) #t)",
        "difficulty": "easy",
        "tags": ["identity-law"],
    },
    {
        "fn": "dimap",
        "prompt": "Verify dimap equals rmap-after-lmap for the same pre and post maps.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [f (lambda (x) (* x 2))] [g (lambda (y) (+ y 1))] [h (lambda (x) (- x 3))] [a (dimap p f g h)] [b (rmap p g (lmap p f h))]) (= (a 10) (b 10)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [f (lambda (x) (* x 2))] [g (lambda (y) (+ y 1))] [h (lambda (x) (- x 3))] [a (dimap p f g h)] [b (rmap p g (lmap p f h))]) (= (a 10) (b 10))) #t)",
        "difficulty": "hard",
        "tags": ["factorization"],
    },
    {
        "fn": "dimap",
        "prompt": "Build a dimap-transformed function and map it over '(0 1 2).",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn (dimap p (lambda (x) (+ x 1)) (lambda (y) (- y 1)) (lambda (z) (* z 2)))]) (map fn '(0 1 2)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn (dimap p (lambda (x) (+ x 1)) (lambda (y) (- y 1)) (lambda (z) (* z 2)))]) (map fn '(0 1 2))) '(1 3 5))",
        "difficulty": "medium",
        "tags": ["map"],
    },

    # lmap
    {
        "fn": "lmap",
        "prompt": "Apply lmap through a function profunctor and evaluate at input 4.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn (lmap p (lambda (x) (* x 2)) (lambda (z) (+ z 1)))]) (fn 4))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn (lmap p (lambda (x) (* x 2)) (lambda (z) (+ z 1)))]) (fn 4)) 9)",
        "difficulty": "easy",
        "tags": ["execution"],
    },
    {
        "fn": "lmap",
        "prompt": "Check that lmap equals dimap with identity output mapping on a shared example.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [f (lambda (x) (+ x 5))] [h (lambda (x) (* x 2))] [a (lmap p f h)] [b (dimap p f identity h)]) (= (a 3) (b 3)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [f (lambda (x) (+ x 5))] [h (lambda (x) (* x 2))] [a (lmap p f h)] [b (dimap p f identity h)]) (= (a 3) (b 3))) #t)",
        "difficulty": "medium",
        "tags": ["agreement"],
    },
    {
        "fn": "lmap",
        "prompt": "Verify sequential lmap applications match one combined pre-map.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (+ x 1))] [f1 (lmap p (lambda (x) (+ x 2)) h)] [f2 (lmap p (lambda (x) (* x 3)) f1)] [f3 (lmap p (lambda (x) (+ (* 3 x) 2)) h)]) (= (f2 4) (f3 4)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (+ x 1))] [f1 (lmap p (lambda (x) (+ x 2)) h)] [f2 (lmap p (lambda (x) (* x 3)) f1)] [f3 (lmap p (lambda (x) (+ (* 3 x) 2)) h)]) (= (f2 4) (f3 4))) #t)",
        "difficulty": "hard",
        "tags": ["composition-law"],
    },
    {
        "fn": "lmap",
        "prompt": "Use lmap to build a transformed function and map it over '(2 4 6).",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn (lmap p (lambda (x) (/ x 2)) (lambda (z) (+ z 7)))]) (map fn '(2 4 6)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn (lmap p (lambda (x) (/ x 2)) (lambda (z) (+ z 7)))]) (map fn '(2 4 6))) '(8 9 10))",
        "difficulty": "medium",
        "tags": ["map"],
    },

    # rmap
    {
        "fn": "rmap",
        "prompt": "Apply rmap through a function profunctor and evaluate at input 3.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn (rmap p (lambda (y) (+ y 100)) (lambda (z) (* z 2)))]) (fn 3))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn (rmap p (lambda (y) (+ y 100)) (lambda (z) (* z 2)))]) (fn 3)) 106)",
        "difficulty": "easy",
        "tags": ["execution"],
    },
    {
        "fn": "rmap",
        "prompt": "Check that rmap equals dimap with identity input mapping on a shared example.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [g (lambda (y) (* y 4))] [h (lambda (x) (+ x 1))] [a (rmap p g h)] [b (dimap p identity g h)]) (= (a 5) (b 5)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [g (lambda (y) (* y 4))] [h (lambda (x) (+ x 1))] [a (rmap p g h)] [b (dimap p identity g h)]) (= (a 5) (b 5))) #t)",
        "difficulty": "medium",
        "tags": ["agreement"],
    },
    {
        "fn": "rmap",
        "prompt": "Verify sequential rmap applications match one combined post-map.",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (* x 2))] [r1 (rmap p (lambda (y) (+ y 3)) h)] [r2 (rmap p (lambda (y) (* y 5)) r1)] [r3 (rmap p (lambda (y) (* (+ y 3) 5)) h)]) (= (r2 4) (r3 4)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [h (lambda (x) (* x 2))] [r1 (rmap p (lambda (y) (+ y 3)) h)] [r2 (rmap p (lambda (y) (* y 5)) r1)] [r3 (rmap p (lambda (y) (* (+ y 3) 5)) h)]) (= (r2 4) (r3 4))) #t)",
        "difficulty": "hard",
        "tags": ["composition-law"],
    },
    {
        "fn": "rmap",
        "prompt": "Use rmap to build a transformed function and map it over '(1 2 3).",
        "gt": "(let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn (rmap p (lambda (y) (+ y 1)) (lambda (x) (* x 3)))]) (map fn '(1 2 3)))",
        "verify": "(equal? (let* ([p (make-profunctor (lambda (f g h) (lambda (x) (g (h (f x))))))] [fn (rmap p (lambda (y) (+ y 1)) (lambda (x) (* x 3)))]) (map fn '(1 2 3))) '(4 7 10))",
        "difficulty": "medium",
        "tags": ["map"],
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
for family, family_samples in by_family.items():
    picked = spread_indices(len(family_samples), EVAL_QUOTA[family])
    for i, s in enumerate(family_samples):
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
    missing_functions = [fn for fn in all_source_functions if fn_counts[fn] == 0]
    if not missing_functions:
        break

    for fn in missing_functions:
        candidates = [s for s in samples if str(s["source_function"]) == fn and str(s["id"]) not in eval_ids]
        swapped = False
        for cand in candidates:
            fam = str(cand["family"])
            fam_eval = [id_to_sample[sid] for sid in eval_ids if str(id_to_sample[sid]["family"]) == fam]
            removable = [r for r in fam_eval if fn_counts[str(r["source_function"])] > 1]
            if not removable:
                continue
            removable.sort(
                key=lambda r: (fn_counts[str(r["source_function"])], str(r["id"])),
                reverse=True,
            )
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
all_rows: List[Dict[str, object]] = []
for s in samples:
    row = dict(s)
    row["split"] = "eval" if s["id"] in eval_ids else "train"
    all_rows.append(row)
    if row["split"] == "eval":
        eval_rows.append(row)
    else:
        train_rows.append(row)

if len(train_rows) != 66 or len(eval_rows) != 14:
    raise ValueError(f"split mismatch: train={len(train_rows)}, eval={len(eval_rows)}")


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


write_jsonl(ALL_PATH, all_rows)
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
