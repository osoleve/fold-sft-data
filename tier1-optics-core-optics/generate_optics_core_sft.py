#!/usr/bin/env python3
"""Generate Tier-1 optics core SFT samples for lattice/optics/optics.ss."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_ROOT.parent
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from sft_split_utils import compute_leakage_aware_eval_ids

ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"
PRE_DIVERSIFY_PATH = OUT_DIR / ".pre_diversify.jsonl"
SFT_GENERATOR_PATH = REPO_ROOT / "user" / "sft" / "generate.ss"

SOURCE_MODULE = "lattice/optics/optics.ss"
SOURCE_TEST = "lattice/optics/test-optics.ss"
SOURCE_PATH = REPO_ROOT / SOURCE_MODULE

FUNCTION_ORDER = [
    "make-iso",
    "iso-over",
    "iso-compose",
    "prism-over",
    "affine-compose",
    "traversal-compose",
    "fold-preview",
    "optic-compose",
]

FUNCTION_SPECS = {
    "make-iso": "Construct an iso record tagged 'iso with forward/backward functions in order.",
    "iso-over": "Map over an isomorphic view: forward, transform, then backward.",
    "iso-compose": "Compose two isos preserving forward/backward directionality.",
    "prism-over": "Apply a function through prism focus only when preview succeeds.",
    "affine-compose": "Compose two affines by chaining Maybe-aware getter/setter behavior.",
    "traversal-compose": "Compose traversals by nesting traverse semantics and flattening folded targets.",
    "fold-preview": "Return the first fold target as Just, or nothing when no targets exist.",
    "optic-compose": "Dispatch optic composition to the most specific compatible optic type.",
}

SKELETONS = {
    "make-iso": """(define (make-iso forward backward)
  ;; TODO: construct an iso record with the canonical tag and slots
  <TODO>)""",
    "iso-over": """(define (iso-over iso f s)
  ;; TODO: transform via forward view and rebuild via backward direction
  <TODO>)""",
    "iso-compose": """(define (iso-compose outer inner)
  ;; TODO: compose forward/backward paths with correct directionality
  <TODO>)""",
    "prism-over": """(define (prism-over prism f s)
  ;; TODO: modify only when preview produces a focus
  <TODO>)""",
    "affine-compose": """(define (affine-compose outer inner)
  ;; TODO: compose affine getter/setter while preserving missing-focus behavior
  <TODO>)""",
    "traversal-compose": """(define (traversal-compose outer inner)
  ;; TODO: compose traversal traverse/fold behaviors
  <TODO>)""",
    "fold-preview": """(define (fold-preview fold s)
  ;; TODO: return Just first target or nothing if empty
  <TODO>)""",
    "optic-compose": """(define (optic-compose outer inner)
  ;; TODO: dispatch by optic-type and return the most specific composed optic
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "make-iso": """(let* ([i (make-iso cdr (lambda (x) (cons 'k x)))])
  (and (iso? i)
       (= (iso-view i '(a . 7)) 7)
       (equal? (iso-review i 9) '(k . 9))))""",
    "iso-over": """(and (= (iso-over iso-id (lambda (x) (+ x 1)) 4) 5)
     (equal? (iso-over iso-reversed (lambda (xs) (cons 0 xs)) '(1 2 3)) '(1 2 3 0))
     (equal? (iso-over iso-swapped
                       (lambda (p) (cons (+ (car p) 1) (+ (cdr p) 1)))
                       '(1 . 2))
             '(2 . 3))
     (equal? (let ([i (make-iso car (lambda (x) (cons x 'ignored)))])
               (iso-over i (lambda (n) (* n 3)) '(4 . tail)))
             '(12 . ignored)))""",
    "iso-compose": """(let* ([a (make-iso (lambda (x) (+ x 1)) (lambda (y) (- y 1)))]
       [b (make-iso (lambda (x) (* x 2)) (lambda (y) (/ y 2)))]
       [c (iso-compose a b)]
       [swap2 (iso-compose iso-swapped iso-swapped)])
  (and (= (iso-view c 3) 8)
       (= (iso-review c 8) 3)
       (equal? (iso-view swap2 '(1 . 2)) '(1 . 2))))""",
    "prism-over": """(and (equal? (prism-over prism-just (lambda (x) (+ x 1)) (just 4)) (just 5))
     (equal? (prism-over prism-just (lambda (x) (+ x 1)) nothing) nothing)
     (equal? (prism-over prism-left (lambda (s) (string-append s "!")) (left "err")) (left "err!"))
     (equal? (prism-over prism-left (lambda (s) (string-append s "!")) (right 9)) (right 9)))""",
    "affine-compose": """(let* ([a (affine-compose (affine-nth 1) (affine-nth 0))]
       [s '((10 11) (20 21) (30 31))])
  (and (equal? (affine-preview a s) (just 20))
       (equal? (affine-set a 99 s) '((10 11) (99 21) (30 31)))
       (equal? (affine-preview a '((1 2))) nothing)))""",
    "traversal-compose": """(let* ([t (traversal-compose traversal-each traversal-each)]
       [s '((1 2) (3 4))])
  (and (equal? (traversal-to-list t s) '(1 2 3 4))
       (equal? (traversal-over t (lambda (x) (+ x 1)) s) '((2 3) (4 5)))
       (equal? (traversal-set t 0 s) '((0 0) (0 0)))))""",
    "fold-preview": """(and (equal? (fold-preview fold-each '(1 2 3)) (just 1))
     (equal? (fold-preview fold-each '()) nothing)
     (equal? (fold-preview (fold-filtered even?) '(1 3 4 6)) (just 4)))""",
    "optic-compose": """(let* ([lp (optic-compose lens-fst prism-just)]
       [ll (optic-compose lens-fst lens-fst)]
       [gg (optic-compose getter-fst lens-fst)])
  (and (affine? lp)
       (lens? ll)
       (fold-optic? gg)
       (equal? (affine-preview lp (cons (just 7) 'tail)) (just 7))
       (equal? (set-lens ll 99 '((1 . 2) . (3 . 4))) '((99 . 2) . (3 . 4)))
       (equal? (fold-to-list gg '((1 . 2) . z)) '(1))))""",
}

PYTHON_SNIPPETS = {
    "make-iso": """def make_iso(forward, backward):
    return ["iso", forward, backward]""",
    "iso-over": """def iso_over(iso_obj, fn, s):
    forward = iso_obj[1]
    backward = iso_obj[2]
    return backward(fn(forward(s)))""",
    "iso-compose": """def iso_compose(outer, inner):
    f = lambda s: inner[1](outer[1](s))
    b = lambda d: outer[2](inner[2](d))
    return ["iso", f, b]""",
    "prism-over": """def prism_over(prism, fn, s):
    maybe_a = preview(prism, s)
    if maybe_a is nothing:
        return s
    return review(prism, fn(from_just(maybe_a)))""",
    "affine-compose": """def affine_compose(outer, inner):
    def get(s):
        ma = affine_preview(outer, s)
        if ma is nothing:
            return nothing
        return affine_preview(inner, from_just(ma))

    def set_(d, s):
        ma = affine_preview(outer, s)
        if ma is nothing:
            return s
        return affine_set(outer, affine_set(inner, d, from_just(ma)), s)

    return make_affine(get, set_)""",
    "traversal-compose": """def traversal_compose(outer, inner):
    def traverse(f, s):
        return traversal_traverse(outer)(lambda a: traversal_traverse(inner)(f, a), s)

    def fold(s):
        return append_map(traversal_fold(inner), traversal_fold(outer)(s))

    return make_traversal(traverse, fold)""",
    "fold-preview": """def fold_preview(fold_obj, s):
    targets = fold_optic_fn(fold_obj)(s)
    if len(targets) == 0:
        return nothing
    return just(targets[0])""",
    "optic-compose": """def optic_compose(outer, inner):
    t1 = optic_type(outer)
    t2 = optic_type(inner)

    if t1 == "lens" and t2 == "lens":
        return lens_compose(outer, inner)
    if t1 == "prism" and t2 == "prism":
        return prism_compose(outer, inner)
    if t1 == "lens" and t2 == "prism":
        return affine_compose(lens_to_affine(outer), prism_to_affine(inner))

    return traversal_compose(to_traversal(outer), to_traversal(inner))""",
}

CHEZ_SNIPPETS = {
    "make-iso": """(define (build-iso fwd bwd)
  (list 'iso fwd bwd))""",
    "iso-over": """(define (iso-over iso fn s)
  (let* ([fw (iso-forward iso)]
         [bw (iso-backward iso)]
         [a (fw s)])
    (bw (fn a))))""",
    "iso-compose": """(define (iso-compose outer inner)
  (let ([fw (compose2 (iso-forward inner) (iso-forward outer))]
        [bw (compose2 (iso-backward outer) (iso-backward inner))])
    (make-iso fw bw)))""",
    "prism-over": """(define (prism-over prism fn s)
  (let ([hit (preview prism s)])
    (cond
      [(nothing? hit) s]
      [else (review prism (fn (from-just hit)))])))""",
    "affine-compose": """(define (affine-compose outer inner)
  (make-affine
   (lambda (s)
     (let ([oa (affine-preview outer s)])
       (if (nothing? oa)
           nothing
           (affine-preview inner (from-just oa)))))
   (lambda (d s)
     (let ([oa (affine-preview outer s)])
       (if (nothing? oa)
           s
           (affine-set outer
                       (affine-set inner d (from-just oa))
                       s))))))""",
    "traversal-compose": """(define (traversal-compose outer inner)
  (make-traversal
   (lambda (f s)
     ((traversal-traverse outer)
      (lambda (a)
        ((traversal-traverse inner) f a))
      s))
   (lambda (s)
     (append-map (traversal-fold inner)
                 ((traversal-fold outer) s)))))""",
    "fold-preview": """(define (fold-preview fold s)
  (let ([targets ((fold-optic-fn fold) s)])
    (if (pair? targets)
        (just (car targets))
        nothing)))""",
    "optic-compose": """(define (optic-compose outer inner)
  (let ([t1 (optic-type outer)]
        [t2 (optic-type inner)])
    (cond
      [(and (eq? t1 'lens) (eq? t2 'lens))
       (lens-compose outer inner)]
      [(and (eq? t1 'prism) (eq? t2 'prism))
       (prism-compose outer inner)]
      [(and (eq? t1 'lens) (eq? t2 'prism))
       (affine-compose (lens->affine outer) (prism->affine inner))]
      [else
       (traversal-compose (->traversal outer) (->traversal inner))])))""",
}

BUGGY_CASES = [
    {
        "fn": "make-iso",
        "buggy": """(define (make-iso forward backward)
  (list 'is0 forward backward))""",
        "note": "Iso records must use the canonical 'iso tag.",
    },
    {
        "fn": "make-iso",
        "buggy": """(define (make-iso forward backward)
  (list 'iso backward forward))""",
        "note": "Forward/backward slots are swapped.",
    },
    {
        "fn": "iso-over",
        "buggy": """(define (iso-over iso f s)
  (f ((iso-forward iso) s)))""",
        "note": "Result must be reconstructed with iso-backward.",
    },
    {
        "fn": "iso-over",
        "buggy": """(define (iso-over iso f s)
  ((iso-backward iso) (f ((iso-backward iso) s))))""",
        "note": "Input should be transformed through iso-forward, not iso-backward.",
    },
    {
        "fn": "iso-compose",
        "buggy": """(define (iso-compose outer inner)
  (make-iso
   (compose2 (iso-forward outer) (iso-forward inner))
   (compose2 (iso-backward inner) (iso-backward outer))))""",
        "note": "Composition order is reversed for both directions.",
    },
    {
        "fn": "iso-compose",
        "buggy": """(define (iso-compose outer inner)
  (make-iso
   (compose2 (iso-forward inner) (iso-forward outer))
   (compose2 (iso-backward outer) (iso-backward outer))))""",
        "note": "Backward path must use inner backward then outer backward.",
    },
    {
        "fn": "prism-over",
        "buggy": """(define (prism-over prism f s)
  (review prism (f (from-just (preview prism s)))))""",
        "note": "Must preserve the source when preview fails.",
    },
    {
        "fn": "prism-over",
        "buggy": """(define (prism-over prism f s)
  (let ([maybe-a (preview prism s)])
    (if (nothing? maybe-a)
        s
        (review prism (from-just maybe-a)))))""",
        "note": "Transformation function f is ignored on successful matches.",
    },
    {
        "fn": "affine-compose",
        "buggy": """(define (affine-compose outer inner)
  (make-affine
   (lambda (s)
     (let ([oa (affine-preview outer s)])
       (if (nothing? oa)
           nothing
           (affine-preview inner (from-just oa)))))
   (lambda (_d s) s)))""",
        "note": "Setter path must update nested focus, not always return original source.",
    },
    {
        "fn": "affine-compose",
        "buggy": """(define (affine-compose outer inner)
  (make-affine
   (lambda (s) ((affine-getter outer) s))
   (lambda (d s) ((affine-setter outer) d s))))""",
        "note": "Composed affine getter/setter must route through both optics.",
    },
    {
        "fn": "traversal-compose",
        "buggy": """(define (traversal-compose outer inner)
  (make-traversal
   (lambda (f s)
     ((traversal-traverse outer) f s))
   (lambda (s)
     (append-map (traversal-fold inner)
                 ((traversal-fold outer) s)))))""",
        "note": "Traverse path must apply inner traversal to each outer focus.",
    },
    {
        "fn": "traversal-compose",
        "buggy": """(define (traversal-compose outer inner)
  (make-traversal
   (lambda (f s)
     ((traversal-traverse outer)
      (lambda (a) a)
      s))
   (lambda (s)
     ((traversal-fold outer) s))))""",
        "note": "Fold path must flatten inner folds, and traverse path must not ignore f.",
    },
    {
        "fn": "fold-preview",
        "buggy": """(define (fold-preview fold s)
  (let ([targets ((fold-optic-fn fold) s)])
    (if (null? targets) nothing (car targets))))""",
        "note": "fold-preview must return a Maybe value, not a raw target.",
    },
    {
        "fn": "fold-preview",
        "buggy": """(define (fold-preview fold s)
  (let ([targets ((fold-optic-fn fold) s)])
    (if (null? targets)
        nothing
        (let loop ([xs targets])
          (if (null? (cdr xs))
              (just (car xs))
              (loop (cdr xs)))))))""",
        "note": "fold-preview should return the first target, not the last.",
    },
    {
        "fn": "optic-compose",
        "buggy": """(define (optic-compose outer inner)
  (traversal-compose (->traversal outer) (->traversal inner)))""",
        "note": "optic-compose must preserve specific optic kinds (lens/prism/affine/fold/setter), not always return traversal.",
    },
    {
        "fn": "optic-compose",
        "buggy": """(define (optic-compose outer inner)
  (let ([t1 (optic-type outer)]
        [t2 (optic-type inner)])
    (cond
      [(and (eq? t1 'lens) (eq? t2 'prism))
       (prism-compose inner prism-id)]
      [else
       (traversal-compose (->traversal outer) (->traversal inner))])))""",
        "note": "lens+prism composition should produce affine behavior focused through both optics.",
    },
]

BASE_DIFFICULTY = {
    "make-iso": "medium",
    "iso-over": "medium",
    "iso-compose": "hard",
    "prism-over": "medium",
    "affine-compose": "hard",
    "traversal-compose": "hard",
    "fold-preview": "easy",
    "optic-compose": "hard",
}

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
DIFFICULTY_INDEX = {name: idx for idx, name in enumerate(DIFFICULTY_LEVELS)}

REQUIRED_KEYS = [
    "id",
    "family",
    "category",
    "difficulty",
    "source_module",
    "source_test",
    "source_function",
    "prompt_body",
    "ground_truth",
    "verify_expr",
    "tags",
]


def read_source_module() -> str:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"source module missing: {SOURCE_PATH}")
    return SOURCE_PATH.read_text(encoding="utf-8")


def extract_define(module_text: str, fn: str) -> str:
    pattern = re.compile(rf"\(define\s+\({re.escape(fn)}(?:\s|\))")
    m = pattern.search(module_text)
    if not m:
        raise ValueError(f"could not find definition for {fn}")

    start = m.start()
    i = start
    depth = 0
    in_string = False
    escaped = False
    in_comment = False

    while i < len(module_text):
        ch = module_text[i]
        if in_comment:
            if ch == "\n":
                in_comment = False
        elif in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == ";":
                in_comment = True
            elif ch == '"':
                in_string = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return module_text[start : i + 1]
        i += 1

    raise ValueError(f"unterminated definition for {fn}")


def strip_doc_forms(defn: str) -> str:
    lines = [line for line in defn.splitlines() if not line.strip().startswith("(doc ")]
    return "\n".join(lines)


MODULE_TEXT = read_source_module()
DEFS: Dict[str, str] = {fn: extract_define(MODULE_TEXT, fn) for fn in FUNCTION_ORDER}
DOC_FREE_DEFS: Dict[str, str] = {fn: strip_doc_forms(code) for fn, code in DEFS.items()}

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
    sid = f"optics_core_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
        "prompt_body": prompt.strip(),
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


def bump_difficulty(level: str, delta: int) -> str:
    idx = DIFFICULTY_INDEX[level] + delta
    idx = max(0, min(idx, len(DIFFICULTY_LEVELS) - 1))
    return DIFFICULTY_LEVELS[idx]


def task_difficulty(fn: str, family: str, task_kind: str, override: str | None = None) -> str:
    if override:
        return override

    base = BASE_DIFFICULTY[fn]
    if family == "spec_to_code":
        if task_kind == "skeleton":
            return bump_difficulty(base, -1)
        if task_kind == "contract":
            return bump_difficulty(base, +1)
        return base

    if family == "translation":
        if task_kind == "chez":
            return bump_difficulty(base, -1)
        if task_kind == "excerpt":
            return base
        return base

    if family == "bugfix":
        return base

    return base


FUNCTION_SECTION = {
    "make-iso": "iso",
    "iso-over": "iso",
    "iso-compose": "iso",
    "prism-over": "prism",
    "affine-compose": "affine",
    "traversal-compose": "traversal",
    "fold-preview": "fold",
    "optic-compose": "unified",
}


def make_source_excerpt(fn: str, snippet: str) -> str:
    section = FUNCTION_SECTION[fn]
    indented = "\n".join(f"  {line}" for line in snippet.splitlines())
    return (
        ";;; lattice/optics/optics.ss excerpt\n"
        "(require 'prelude)\n"
        "(require 'combinators)\n"
        "(require 'templates)\n"
        "\n"
        "(doc 'module 'optics)\n"
        f"(doc 'section '{section})\n"
        "\n"
        "(define (local-helper x) x)\n"
        "\n"
        f"{indented}\n"
    )


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (24)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=task_difficulty(fn, "spec_to_code", "direct"),
        source_function=fn,
        prompt=f"""Implement this optics function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "core", "spec-to-code", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=task_difficulty(fn, "spec_to_code", "skeleton"),
        source_function=fn,
        prompt=f"""Complete this Fold Scheme skeleton.

```scheme
{SKELETONS[fn]}
```

Replace `<TODO>` and return only the completed definition for `{fn}`.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "core", "skeleton-completion", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=task_difficulty(fn, "spec_to_code", "contract"),
        source_function=fn,
        prompt=f"""Implement `{fn}` from this contract.

Module: `{SOURCE_MODULE}`
Contract focus: {FUNCTION_SPECS[fn]}

Requirements:
1. Keep the exact function name/signature.
2. Preserve semantics for optic laws and edge cases.
3. Return only one production-ready definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "core", "contract-implementation", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (24)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="translation",
        category="translation",
        difficulty=task_difficulty(fn, "translation", "python"),
        source_function=fn,
        prompt=f"""Translate this Python function into Fold-native Scheme.
Preserve behavior and keep the function name `{fn}`.
Return only the Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "core", "python-to-scheme", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=task_difficulty(fn, "translation", "chez"),
        source_function=fn,
        prompt=f"""Convert this Chez-style snippet to canonical Fold style.
The target function must be named `{fn}`.
Return only the final Fold definition.

```scheme
{CHEZ_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "core", "chez-to-fold", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=task_difficulty(fn, "translation", "excerpt"),
        source_function=fn,
        prompt=f"""Extract and translate the target function from this source-style module excerpt.
Return only a single Fold definition for `{fn}`.
Drop metadata doc forms from the output and keep executable behavior unchanged.

```scheme
{make_source_excerpt(fn, CHEZ_SNIPPETS[fn])}
```""",
        ground_truth=DOC_FREE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "core", "source-excerpt-to-fold", "doc-free-target", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (16)
# -----------------------------------------------------------------------------
if len(BUGGY_CASES) != 16:
    raise ValueError(f"expected 16 bugfix cases, found {len(BUGGY_CASES)}")

for case in BUGGY_CASES:
    fn = case["fn"]
    add_sample(
        family="bugfix",
        category="debugging",
        difficulty=task_difficulty(fn, "bugfix", "bugfix", str(case.get("difficulty", "")) or None),
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
        tags=["tier1", "optics", "core", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition/use (40)
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
        f"Ensure `{source_function}` is part of the composed solution.\n"
        "Return only the final Fold expression."
    )
    add_sample(
        family="composition",
        category="usage",
        difficulty=difficulty,
        source_function=source_function,
        prompt=composition_prompt,
        ground_truth=ground_truth,
        verify_expr=verify_check.strip(),
        tags=["tier1", "optics", "core", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # make-iso
    {
        "fn": "make-iso",
        "prompt": "Create an identity iso via make-iso and return its tag symbol.",
        "gt": "(car (make-iso identity identity))",
        "verify": "(equal? (car (make-iso identity identity)) 'iso)",
        "difficulty": "easy",
        "tags": ["record"],
    },
    {
        "fn": "make-iso",
        "prompt": "Build an iso from pair tail to scalar and read from '(a . 7).",
        "gt": "(iso-view (make-iso cdr (lambda (x) (cons 'k x))) '(a . 7))",
        "verify": "(= (iso-view (make-iso cdr (lambda (x) (cons 'k x))) '(a . 7)) 7)",
        "difficulty": "easy",
        "tags": ["view"],
    },
    {
        "fn": "make-iso",
        "prompt": "Build the same pair-tail iso and review value 9 back into a pair.",
        "gt": "(iso-review (make-iso cdr (lambda (x) (cons 'k x))) 9)",
        "verify": "(equal? (iso-review (make-iso cdr (lambda (x) (cons 'k x))) 9) '(k . 9))",
        "difficulty": "easy",
        "tags": ["review"],
    },
    {
        "fn": "make-iso",
        "prompt": "Compose iso-view and iso-review through a custom numeric iso and return the roundtrip result for 5.",
        "gt": "(let* ([i (make-iso (lambda (x) (+ x 10)) (lambda (y) (- y 10)))]) (iso-review i (iso-view i 5)))",
        "verify": "(= (let* ([i (make-iso (lambda (x) (+ x 10)) (lambda (y) (- y 10)))]) (iso-review i (iso-view i 5))) 5)",
        "difficulty": "medium",
        "tags": ["roundtrip"],
    },
    {
        "fn": "make-iso",
        "prompt": "Use make-iso with reverse/reverse and append 0 through iso-over on '(1 2 3).",
        "gt": "(iso-over (make-iso reverse reverse) (lambda (xs) (cons 0 xs)) '(1 2 3))",
        "verify": "(equal? (iso-over (make-iso reverse reverse) (lambda (xs) (cons 0 xs)) '(1 2 3)) '(1 2 3 0))",
        "difficulty": "medium",
        "tags": ["iso-over"],
    },

    # iso-over
    {
        "fn": "iso-over",
        "prompt": "Increment through iso-id with iso-over on 4.",
        "gt": "(iso-over iso-id (lambda (x) (+ x 1)) 4)",
        "verify": "(= (iso-over iso-id (lambda (x) (+ x 1)) 4) 5)",
        "difficulty": "easy",
        "tags": ["identity"],
    },
    {
        "fn": "iso-over",
        "prompt": "Apply iso-over via iso-reversed to append 0 to list '(1 2 3).",
        "gt": "(iso-over iso-reversed (lambda (xs) (cons 0 xs)) '(1 2 3))",
        "verify": "(equal? (iso-over iso-reversed (lambda (xs) (cons 0 xs)) '(1 2 3)) '(1 2 3 0))",
        "difficulty": "medium",
        "tags": ["reversed"],
    },
    {
        "fn": "iso-over",
        "prompt": "Modify both pair components through iso-swapped and return the reconstructed pair.",
        "gt": "(iso-over iso-swapped (lambda (p) (cons (+ (car p) 1) (+ (cdr p) 1))) '(1 . 2))",
        "verify": "(equal? (iso-over iso-swapped (lambda (p) (cons (+ (car p) 1) (+ (cdr p) 1))) '(1 . 2)) '(2 . 3))",
        "difficulty": "medium",
        "tags": ["swapped"],
    },
    {
        "fn": "iso-over",
        "prompt": "Apply iso-over on iso-id to append ! to the string \"ok\".",
        "gt": "(iso-over iso-id (lambda (s) (string-append s \"!\")) \"ok\")",
        "verify": "(equal? (iso-over iso-id (lambda (s) (string-append s \"!\")) \"ok\") \"ok!\")",
        "difficulty": "easy",
        "tags": ["string"],
    },
    {
        "fn": "iso-over",
        "prompt": "Run iso-over with a custom car/cons iso and triple the focused numeric component.",
        "gt": "(let ([i (make-iso car (lambda (x) (cons x 'ignored)))]) (iso-over i (lambda (n) (* n 3)) '(4 . tail)))",
        "verify": "(equal? (let ([i (make-iso car (lambda (x) (cons x 'ignored)))]) (iso-over i (lambda (n) (* n 3)) '(4 . tail))) '(12 . ignored))",
        "difficulty": "hard",
        "tags": ["custom-iso"],
    },

    # iso-compose
    {
        "fn": "iso-compose",
        "prompt": "Compose iso-swapped with itself and view pair '(1 . 2).",
        "gt": "(let ([c (iso-compose iso-swapped iso-swapped)]) (iso-view c '(1 . 2)))",
        "verify": "(equal? (let ([c (iso-compose iso-swapped iso-swapped)]) (iso-view c '(1 . 2))) '(1 . 2))",
        "difficulty": "medium",
        "tags": ["self-inverse"],
    },
    {
        "fn": "iso-compose",
        "prompt": "Compose numeric add-then-double isos and evaluate composed view at 3.",
        "gt": "(let* ([a (make-iso (lambda (x) (+ x 1)) (lambda (y) (- y 1)))] [b (make-iso (lambda (x) (* x 2)) (lambda (y) (/ y 2)))] [c (iso-compose a b)]) (iso-view c 3))",
        "verify": "(= (let* ([a (make-iso (lambda (x) (+ x 1)) (lambda (y) (- y 1)))] [b (make-iso (lambda (x) (* x 2)) (lambda (y) (/ y 2)))] [c (iso-compose a b)]) (iso-view c 3)) 8)",
        "difficulty": "hard",
        "tags": ["numeric"],
    },
    {
        "fn": "iso-compose",
        "prompt": "Using the same composed iso, review value 8 back to source space.",
        "gt": "(let* ([a (make-iso (lambda (x) (+ x 1)) (lambda (y) (- y 1)))] [b (make-iso (lambda (x) (* x 2)) (lambda (y) (/ y 2)))] [c (iso-compose a b)]) (iso-review c 8))",
        "verify": "(= (let* ([a (make-iso (lambda (x) (+ x 1)) (lambda (y) (- y 1)))] [b (make-iso (lambda (x) (* x 2)) (lambda (y) (/ y 2)))] [c (iso-compose a b)]) (iso-review c 8)) 3)",
        "difficulty": "hard",
        "tags": ["numeric"],
    },
    {
        "fn": "iso-compose",
        "prompt": "Compose iso-id with iso-reversed and view '(1 2 3).",
        "gt": "(let ([c (iso-compose iso-id iso-reversed)]) (iso-view c '(1 2 3)))",
        "verify": "(equal? (let ([c (iso-compose iso-id iso-reversed)]) (iso-view c '(1 2 3))) '(3 2 1))",
        "difficulty": "medium",
        "tags": ["identity-compose"],
    },
    {
        "fn": "iso-compose",
        "prompt": "Compose iso-reversed with iso-id and review '(3 2 1).",
        "gt": "(let ([c (iso-compose iso-reversed iso-id)]) (iso-review c '(3 2 1)))",
        "verify": "(equal? (let ([c (iso-compose iso-reversed iso-id)]) (iso-review c '(3 2 1))) '(1 2 3))",
        "difficulty": "medium",
        "tags": ["identity-compose"],
    },

    # prism-over
    {
        "fn": "prism-over",
        "prompt": "Increment a Just value through prism-over prism-just.",
        "gt": "(prism-over prism-just (lambda (x) (+ x 1)) (just 4))",
        "verify": "(equal? (prism-over prism-just (lambda (x) (+ x 1)) (just 4)) (just 5))",
        "difficulty": "easy",
        "tags": ["just"],
    },
    {
        "fn": "prism-over",
        "prompt": "Apply the same prism-over transformation to nothing and keep it unchanged.",
        "gt": "(prism-over prism-just (lambda (x) (+ x 1)) nothing)",
        "verify": "(equal? (prism-over prism-just (lambda (x) (+ x 1)) nothing) nothing)",
        "difficulty": "easy",
        "tags": ["nothing"],
    },
    {
        "fn": "prism-over",
        "prompt": "Modify Left payload text through prism-left using string-append.",
        "gt": "(prism-over prism-left (lambda (s) (string-append s \"!\")) (left \"err\"))",
        "verify": "(equal? (prism-over prism-left (lambda (s) (string-append s \"!\")) (left \"err\")) (left \"err!\"))",
        "difficulty": "medium",
        "tags": ["either-left"],
    },
    {
        "fn": "prism-over",
        "prompt": "Run prism-over prism-left on a Right value and preserve the original value.",
        "gt": "(prism-over prism-left (lambda (s) (string-append s \"!\")) (right 9))",
        "verify": "(equal? (prism-over prism-left (lambda (s) (string-append s \"!\")) (right 9)) (right 9))",
        "difficulty": "medium",
        "tags": ["either-right"],
    },
    {
        "fn": "prism-over",
        "prompt": "Compose prism-just with itself and increment nested Just payload.",
        "gt": "(prism-over (prism-compose prism-just prism-just) (lambda (x) (+ x 1)) (just (just 4)))",
        "verify": "(equal? (prism-over (prism-compose prism-just prism-just) (lambda (x) (+ x 1)) (just (just 4))) (just (just 5)))",
        "difficulty": "hard",
        "tags": ["composed-prism"],
    },

    # affine-compose
    {
        "fn": "affine-compose",
        "prompt": "Compose two affine-nth optics to preview the nested focus in '((10 11) (20 21) (30 31)).",
        "gt": "(let* ([a (affine-compose (affine-nth 1) (affine-nth 0))] [s '((10 11) (20 21) (30 31))]) (affine-preview a s))",
        "verify": "(equal? (let* ([a (affine-compose (affine-nth 1) (affine-nth 0))] [s '((10 11) (20 21) (30 31))]) (affine-preview a s)) (just 20))",
        "difficulty": "hard",
        "tags": ["preview"],
    },
    {
        "fn": "affine-compose",
        "prompt": "Set the same composed affine focus to 99 in nested list data.",
        "gt": "(let* ([a (affine-compose (affine-nth 1) (affine-nth 0))] [s '((10 11) (20 21) (30 31))]) (affine-set a 99 s))",
        "verify": "(equal? (let* ([a (affine-compose (affine-nth 1) (affine-nth 0))] [s '((10 11) (20 21) (30 31))]) (affine-set a 99 s)) '((10 11) (99 21) (30 31)))",
        "difficulty": "hard",
        "tags": ["set"],
    },
    {
        "fn": "affine-compose",
        "prompt": "Check that the composed affine returns nothing when outer focus is missing.",
        "gt": "(let* ([a (affine-compose (affine-nth 1) (affine-nth 0))]) (affine-preview a '((1 2))))",
        "verify": "(equal? (let* ([a (affine-compose (affine-nth 1) (affine-nth 0))]) (affine-preview a '((1 2)))) nothing)",
        "difficulty": "medium",
        "tags": ["missing"],
    },
    {
        "fn": "affine-compose",
        "prompt": "Compose affine-id with affine-nth 0 and preview from '(5 6).",
        "gt": "(let* ([a (affine-compose affine-id (affine-nth 0))]) (affine-preview a '(5 6)))",
        "verify": "(equal? (let* ([a (affine-compose affine-id (affine-nth 0))]) (affine-preview a '(5 6))) (just 5))",
        "difficulty": "medium",
        "tags": ["identity"],
    },
    {
        "fn": "affine-compose",
        "prompt": "Use affine-over through composed affine to add 10 to the nested focus.",
        "gt": "(let* ([a (affine-compose (affine-nth 1) (affine-nth 0))] [s '((10 11) (20 21) (30 31))]) (affine-over a (lambda (x) (+ x 10)) s))",
        "verify": "(equal? (let* ([a (affine-compose (affine-nth 1) (affine-nth 0))] [s '((10 11) (20 21) (30 31))]) (affine-over a (lambda (x) (+ x 10)) s)) '((10 11) (30 21) (30 31)))",
        "difficulty": "hard",
        "tags": ["over"],
    },

    # traversal-compose
    {
        "fn": "traversal-compose",
        "prompt": "Compose traversal-each with traversal-each and collect flattened targets from nested list input.",
        "gt": "(let* ([t (traversal-compose traversal-each traversal-each)] [s '((1 2) (3 4))]) (traversal-to-list t s))",
        "verify": "(equal? (let* ([t (traversal-compose traversal-each traversal-each)] [s '((1 2) (3 4))]) (traversal-to-list t s)) '(1 2 3 4))",
        "difficulty": "hard",
        "tags": ["to-list"],
    },
    {
        "fn": "traversal-compose",
        "prompt": "Compose traversal-each with traversal-each and increment all nested numeric targets.",
        "gt": "(let* ([t (traversal-compose traversal-each traversal-each)] [s '((1 2) (3 4))]) (traversal-over t (lambda (x) (+ x 1)) s))",
        "verify": "(equal? (let* ([t (traversal-compose traversal-each traversal-each)] [s '((1 2) (3 4))]) (traversal-over t (lambda (x) (+ x 1)) s)) '((2 3) (4 5)))",
        "difficulty": "hard",
        "tags": ["over"],
    },
    {
        "fn": "traversal-compose",
        "prompt": "Use traversal-set through composed traversal to zero all nested values.",
        "gt": "(let* ([t (traversal-compose traversal-each traversal-each)] [s '((1 2) (3 4))]) (traversal-set t 0 s))",
        "verify": "(equal? (let* ([t (traversal-compose traversal-each traversal-each)] [s '((1 2) (3 4))]) (traversal-set t 0 s)) '((0 0) (0 0)))",
        "difficulty": "hard",
        "tags": ["set"],
    },
    {
        "fn": "traversal-compose",
        "prompt": "Compose traversal-each with filtered odd? and scale focused values by 10.",
        "gt": "(let* ([t (traversal-compose traversal-each (filtered odd?))]) (traversal-over t (lambda (x) (* x 10)) '(1 2 3 4)))",
        "verify": "(equal? (let* ([t (traversal-compose traversal-each (filtered odd?))]) (traversal-over t (lambda (x) (* x 10)) '(1 2 3 4))) '(10 2 30 4))",
        "difficulty": "medium",
        "tags": ["filtered"],
    },
    {
        "fn": "traversal-compose",
        "prompt": "Compose traversal-each with traversal-just and collect present Maybe payloads.",
        "gt": "(let* ([t (traversal-compose traversal-each traversal-just)]) (traversal-to-list t (list (just 1) nothing (just 3))))",
        "verify": "(equal? (let* ([t (traversal-compose traversal-each traversal-just)]) (traversal-to-list t (list (just 1) nothing (just 3)))) '(1 3))",
        "difficulty": "medium",
        "tags": ["maybe"],
    },

    # fold-preview
    {
        "fn": "fold-preview",
        "prompt": "Preview the first element through fold-each on '(1 2 3).",
        "gt": "(fold-preview fold-each '(1 2 3))",
        "verify": "(equal? (fold-preview fold-each '(1 2 3)) (just 1))",
        "difficulty": "easy",
        "tags": ["first"],
    },
    {
        "fn": "fold-preview",
        "prompt": "Preview through fold-each on an empty list and confirm nothing.",
        "gt": "(fold-preview fold-each '())",
        "verify": "(equal? (fold-preview fold-each '()) nothing)",
        "difficulty": "easy",
        "tags": ["empty"],
    },
    {
        "fn": "fold-preview",
        "prompt": "Preview first even element through fold-filtered even? on mixed input.",
        "gt": "(fold-preview (fold-filtered even?) '(1 3 4 6))",
        "verify": "(equal? (fold-preview (fold-filtered even?) '(1 3 4 6)) (just 4))",
        "difficulty": "medium",
        "tags": ["filtered"],
    },
    {
        "fn": "fold-preview",
        "prompt": "Compose fold-each with fold-filtered odd? over nested lists and preview the first odd value.",
        "gt": "(let* ([f (fold-compose fold-each (fold-filtered odd?))]) (fold-preview f '((1 2) (4 6) (7 8))))",
        "verify": "(equal? (let* ([f (fold-compose fold-each (fold-filtered odd?))]) (fold-preview f '((1 2) (4 6) (7 8)))) (just 1))",
        "difficulty": "hard",
        "tags": ["composed-fold"],
    },
    {
        "fn": "fold-preview",
        "prompt": "Use lens->fold lens-fst and preview from pair '(9 . 2).",
        "gt": "(fold-preview (lens->fold lens-fst) '(9 . 2))",
        "verify": "(equal? (fold-preview (lens->fold lens-fst) '(9 . 2)) (just 9))",
        "difficulty": "medium",
        "tags": ["lens-fold"],
    },

    # optic-compose
    {
        "fn": "optic-compose",
        "prompt": "Compose lens-fst with lens-fst and set nested pair focus to 99.",
        "gt": "(let* ([o (optic-compose lens-fst lens-fst)]) (set-lens o 99 '((1 . 2) . (3 . 4))))",
        "verify": "(equal? (let* ([o (optic-compose lens-fst lens-fst)]) (set-lens o 99 '((1 . 2) . (3 . 4)))) '((99 . 2) . (3 . 4)))",
        "difficulty": "hard",
        "tags": ["lens-lens"],
    },
    {
        "fn": "optic-compose",
        "prompt": "Compose lens-fst with prism-just and preview the focused payload from (cons (just 7) 'tail).",
        "gt": "(let* ([o (optic-compose lens-fst prism-just)]) (affine-preview o (cons (just 7) 'tail)))",
        "verify": "(equal? (let* ([o (optic-compose lens-fst prism-just)]) (affine-preview o (cons (just 7) 'tail))) (just 7))",
        "difficulty": "hard",
        "tags": ["lens-prism"],
    },
    {
        "fn": "optic-compose",
        "prompt": "Compose prism-just with prism-just and preview nested maybe focus.",
        "gt": "(let* ([o (optic-compose prism-just prism-just)]) (preview o (just (just 5))))",
        "verify": "(equal? (let* ([o (optic-compose prism-just prism-just)]) (preview o (just (just 5)))) (just 5))",
        "difficulty": "hard",
        "tags": ["prism-prism"],
    },
    {
        "fn": "optic-compose",
        "prompt": "Compose iso-swapped with lens-fst and view pair '(1 . 2).",
        "gt": "(let* ([o (optic-compose iso-swapped lens-fst)]) (view o '(1 . 2)))",
        "verify": "(= (let* ([o (optic-compose iso-swapped lens-fst)]) (view o '(1 . 2))) 2)",
        "difficulty": "medium",
        "tags": ["iso-lens"],
    },
    {
        "fn": "optic-compose",
        "prompt": "Compose getter-fst with lens-fst and collect the resulting read-only fold output.",
        "gt": "(let* ([o (optic-compose getter-fst lens-fst)]) (fold-to-list o '((1 . 2) . z)))",
        "verify": "(equal? (let* ([o (optic-compose getter-fst lens-fst)]) (fold-to-list o '((1 . 2) . z))) '(1))",
        "difficulty": "hard",
        "tags": ["getter-fold"],
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

expected_family_counts = {
    "spec_to_code": len(FUNCTION_ORDER) * 3,
    "translation": len(FUNCTION_ORDER) * 3,
    "bugfix": len(BUGGY_CASES),
    "composition": len(composition_cases),
}

for family, expected_count in expected_family_counts.items():
    actual = sum(1 for s in samples if s["family"] == family)
    if actual != expected_count:
        raise ValueError(f"{family} must contain {expected_count} samples, found {actual}")

if len(samples) != sum(expected_family_counts.values()):
    raise ValueError(f"unexpected total samples: {len(samples)}")

EVAL_RATIO = 0.18
EVAL_MIN_BY_FAMILY = {
    "spec_to_code": 4,
    "translation": 4,
    "bugfix": 3,
    "composition": 7,
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


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


pre_diversify_rows = [dict(sample, split=("eval" if sample["id"] in eval_ids else "train")) for sample in samples]
write_jsonl(PRE_DIVERSIFY_PATH, pre_diversify_rows)

env = os.environ.copy()
env["SFT_EVAL_RATIO"] = str(EVAL_RATIO)
proc = subprocess.run(
    [
        "scheme",
        "--script",
        str(SFT_GENERATOR_PATH),
        str(PRE_DIVERSIFY_PATH),
        str(OUT_DIR),
    ],
    cwd=str(REPO_ROOT),
    env=env,
    text=True,
    capture_output=True,
)
if proc.returncode != 0:
    raise RuntimeError(
        "prompt diversification failed:\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )

all_rows = read_jsonl(ALL_PATH)
train_rows = read_jsonl(TRAIN_PATH)
eval_rows = read_jsonl(EVAL_PATH)

if len(all_rows) != len(samples):
    raise ValueError(f"all.jsonl row count mismatch: expected {len(samples)} got {len(all_rows)}")
if len(train_rows) + len(eval_rows) != len(samples):
    raise ValueError("train/eval row count mismatch")
if any("prompt" not in row or not str(row["prompt"]).strip() for row in all_rows):
    raise ValueError("dsl output missing prompt in one or more rows")
if any("prompt_body" not in row or not str(row["prompt_body"]).strip() for row in all_rows):
    raise ValueError("dsl output missing prompt_body in one or more rows")

summary = {
    "total": len(all_rows),
    "train": len(train_rows),
    "eval": len(eval_rows),
    "families": {},
    "difficulty": dict(sorted(Counter(str(r["difficulty"]) for r in all_rows).items())),
    "source_functions": dict(sorted(Counter(str(r["source_function"]) for r in all_rows).items())),
}

for family in sorted(expected_family_counts.keys()):
    fam_rows = [r for r in all_rows if str(r["family"]) == family]
    summary["families"][family] = {
        "total": len(fam_rows),
        "eval": sum(1 for r in fam_rows if str(r["split"]) == "eval"),
        "train": sum(1 for r in fam_rows if str(r["split"]) == "train"),
    }

SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
