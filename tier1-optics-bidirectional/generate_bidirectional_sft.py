#!/usr/bin/env python3
"""Generate Tier-1 bidirectional optics SFT samples for lattice/optics/bidirectional.ss."""

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

SOURCE_MODULE = "lattice/optics/bidirectional.ss"
SOURCE_TEST = "lattice/optics/test-bidirectional.ss"
SOURCE_PATH = REPO_ROOT / SOURCE_MODULE

FUNCTION_ORDER = [
    "make-migration",
    "migrate",
    "rollback",
    "migration-apply",
    "migration-compose",
    "migration-chain",
    "migration-flip",
    "make-migration-from-functions",
    "make-identity-migration",
    "verify-migration-laws",
]

FUNCTION_SPECS = {
    "make-migration": "Create a migration record tagged 'migration with name/from/to/iso slots in canonical order.",
    "migrate": "Apply the migration's forward direction by reading p-iso-forward from the migration iso.",
    "rollback": "Apply the migration's backward direction by reading p-iso-backward from the migration iso.",
    "migration-apply": "Dispatch by direction symbol: 'forward uses migrate, 'backward uses rollback, else signal an error.",
    "migration-compose": "Compose two compatible migrations: forward m1 then m2, backward m2 then m1, preserving version endpoints.",
    "migration-chain": "Compose a non-empty migration list; reject empty input and return the sole item for singleton lists.",
    "migration-flip": "Invert migration direction by swapping from/to versions and exchanging forward/backward transforms.",
    "make-migration-from-functions": "Lift raw forward/backward functions into a migration via make-p-iso and make-migration.",
    "make-identity-migration": "Create a migration that leaves data unchanged and keeps from/to at the same version.",
    "verify-migration-laws": "Delegate to verify-p-iso-laws on a migration's iso using source-domain and target-domain test samples.",
}

SKELETONS = {
    "make-migration": """(define (make-migration name from-version to-version iso)
  ;; TODO: construct a canonical migration record
  <TODO>)""",
    "migrate": """(define (migrate m data)
  ;; TODO: apply migration in forward direction
  <TODO>)""",
    "rollback": """(define (rollback m data)
  ;; TODO: apply migration in backward direction
  <TODO>)""",
    "migration-apply": """(define (migration-apply m data direction)
  ;; TODO: dispatch on direction with explicit error for invalid symbols
  <TODO>)""",
    "migration-compose": """(define (migration-compose m1 m2)
  ;; TODO: validate version boundary and compose forward/backward transforms
  <TODO>)""",
    "migration-chain": """(define (migration-chain migrations)
  ;; TODO: reject empty input; compose the list into a single migration
  <TODO>)""",
    "migration-flip": """(define (migration-flip m)
  ;; TODO: swap endpoints and invert iso direction
  <TODO>)""",
    "make-migration-from-functions": """(define (make-migration-from-functions name from-ver to-ver forward backward)
  ;; TODO: wrap raw functions into a migration
  <TODO>)""",
    "make-identity-migration": """(define (make-identity-migration name version)
  ;; TODO: build an identity migration at one version
  <TODO>)""",
    "verify-migration-laws": """(define (verify-migration-laws m test-as test-bs)
  ;; TODO: verify underlying iso invertibility laws on both sample sets
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "make-migration": """(let* ([iso (make-p-iso add1 sub1)]
       [m (make-migration 'inc 'v1 'v2 iso)])
  (and (migration? m)
       (equal? (car m) 'migration)
       (equal? (migration-name m) 'inc)
       (equal? (migration-from m) 'v1)
       (equal? (migration-to m) 'v2)
       (= (migrate m 10) 11)
       (= (rollback m 11) 10)))""",
    "migrate": """(let* ([m (make-migration-from-functions
           'prefix 'raw 'tagged
           (lambda (s) (string-append "v2:" s))
           (lambda (s) (substring s 3 (string-length s))))])
  (and (equal? (migrate m "abc") "v2:abc")
       (equal? (rollback m (migrate m "x")) "x")))""",
    "rollback": """(let* ([m (make-migration-from-functions
           'prefix 'raw 'tagged
           (lambda (s) (string-append "v2:" s))
           (lambda (s) (substring s 3 (string-length s))))])
  (and (equal? (rollback m "v2:abc") "abc")
       (equal? (migrate m (rollback m "v2:x")) "v2:x")))""",
    "migration-apply": """(let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))])
  (and (= (migration-apply m 10 'forward) 11)
       (= (migration-apply m 10 'backward) 9)
       (guard (ex [else #t])
         (begin
           (migration-apply m 10 'sideways)
           #f))))""",
    "migration-compose": """(let* ([m1 (make-migration 'add1 'v1 'v2 (make-p-iso add1 sub1))]
       [m2 (make-migration 'double 'v2 'v3
                           (make-p-iso (lambda (x) (* x 2))
                                       (lambda (x) (/ x 2))))]
       [m3 (migration-compose m1 m2)])
  (and (equal? (migration-from m3) 'v1)
       (equal? (migration-to m3) 'v3)
       (= (migrate m3 5) 12)
       (= (rollback m3 12) 5)
       (guard (ex [else #t])
         (begin
           (migration-compose m1 (make-migration 'bad 'x 'y p-iso-id))
           #f))))""",
    "migration-chain": """(let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))]
       [m2 (make-migration 'b 'v2 'v3
                           (make-p-iso (lambda (x) (* x 2))
                                       (lambda (x) (/ x 2))))]
       [m3 (make-migration 'c 'v3 'v4
                           (make-p-iso (lambda (x) (- x 3))
                                       (lambda (x) (+ x 3))))]
       [chain (migration-chain (list m1 m2 m3))]
       [single (migration-chain (list m1))])
  (and (= (migrate chain 5) 9)
       (= (rollback chain 9) 5)
       (= (migrate single 5) 6)
       (guard (ex [else #t])
         (begin
           (migration-chain '())
           #f))))""",
    "migration-flip": """(let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))]
       [flipped (migration-flip m)]
       [double-flipped (migration-flip flipped)])
  (and (equal? (migration-from flipped) 'v2)
       (equal? (migration-to flipped) 'v1)
       (= (migrate flipped 10) 9)
       (= (rollback flipped 10) 11)
       (= (migrate double-flipped 7) (migrate m 7))))""",
    "make-migration-from-functions": """(let* ([m (make-migration-from-functions
           'triple 'v1 'v2
           (lambda (x) (* x 3))
           (lambda (y) (/ y 3)))])
  (and (migration? m)
       (= (migrate m 4) 12)
       (= (rollback m 12) 4)
       (equal? (migration-to m) 'v2)))""",
    "make-identity-migration": """(let ([m (make-identity-migration 'id 'v9)])
  (and (migration? m)
       (equal? (migration-from m) 'v9)
       (equal? (migration-to m) 'v9)
       (equal? (migrate m '(1 2)) '(1 2))
       (equal? (rollback m '(a . b)) '(a . b))))""",
    "verify-migration-laws": """(let* ([good-prefix (make-migration-from-functions
                     'good-prefix 'v1 'v2
                     (lambda (s) (string-append "v2:" s))
                     (lambda (s)
                       (if (and (<= 3 (string-length s))
                                (string=? (substring s 0 3) "v2:"))
                           (substring s 3 (string-length s))
                           (error 'good-prefix-backward "bad payload"))))]
       [bad-one-sided (make-migration-from-functions
                       'bad-one-sided 'v1 'v2
                       abs
                       identity)]
       [prefix-ok (guard (ex [else #f])
                    (verify-migration-laws good-prefix
                                           '("a" "bc")
                                           '("v2:a" "v2:bc")))]
       [bad-ok (guard (ex [else #f])
                 (verify-migration-laws bad-one-sided
                                        '(1 2 3)
                                        '(-1 2 -3)))])
  (and prefix-ok (not bad-ok)))""",
}

PYTHON_SNIPPETS = {
    "make-migration": """def build_migration(name, from_version, to_version, iso_obj):
    return ["migration", name, from_version, to_version, iso_obj]""",
    "migrate": """def migrate_step(migration, payload):
    iso_obj = migration_iso(migration)
    fwd = p_iso_forward(iso_obj)
    return fwd(payload)""",
    "rollback": """def rollback_step(migration, payload):
    iso_obj = migration_iso(migration)
    bwd = p_iso_backward(iso_obj)
    return bwd(payload)""",
    "migration-apply": """def migration_apply(migration, payload, direction):
    if direction == "forward":
        return migrate(migration, payload)
    if direction == "backward":
        return rollback(migration, payload)
    raise ValueError(("migration-apply", "Invalid direction", direction))""",
    "migration-compose": """def migration_compose(first, second):
    if migration_to(first) != migration_from(second):
        raise ValueError("version mismatch")

    iso1 = migration_iso(first)
    iso2 = migration_iso(second)

    combo = make_p_iso(
        lambda x: p_iso_forward(iso2)(p_iso_forward(iso1)(x)),
        lambda y: p_iso_backward(iso1)(p_iso_backward(iso2)(y)),
    )

    name = symbol(f"{migration_from(first)}->{migration_to(second)}")
    return make_migration(name, migration_from(first), migration_to(second), combo)""",
    "migration-chain": """def migration_chain(ms):
    if len(ms) == 0:
        raise ValueError(("migration-chain", "Empty migration list"))
    if len(ms) == 1:
        return ms[0]

    acc = ms[0]
    for m in ms[1:]:
        acc = migration_compose(acc, m)
    return acc""",
    "migration-flip": """def migration_flip(migration):
    iso_obj = migration_iso(migration)
    flipped = make_p_iso(p_iso_backward(iso_obj), p_iso_forward(iso_obj))
    name = symbol(f"{migration_name(migration)}-reversed")
    return make_migration(name, migration_to(migration), migration_from(migration), flipped)""",
    "make-migration-from-functions": """def make_migration_from_functions(name, from_ver, to_ver, forward, backward):
    return make_migration(name, from_ver, to_ver, make_p_iso(forward, backward))""",
    "make-identity-migration": """def make_identity_migration(name, version):
    return make_migration(name, version, version, p_iso_id)""",
    "verify-migration-laws": """def verify_migration_laws(migration, test_as, test_bs):
    return verify_p_iso_laws(migration_iso(migration), test_as, test_bs)""",
}

CHEZ_SNIPPETS = {
    "make-migration": """(define (build-migration tag src dst conversion)
  (list 'migration tag src dst conversion))""",
    "migrate": """(define (run-forward migration payload)
  (let* ([iso (migration-iso migration)]
         [forward (p-iso-forward iso)])
    (forward payload)))""",
    "rollback": """(define (run-backward migration payload)
  (let* ([iso (migration-iso migration)]
         [backward (p-iso-backward iso)])
    (backward payload)))""",
    "migration-apply": """(define (apply-migration migration payload dir)
  (cond
    [(eq? dir 'forward)
     (migrate migration payload)]
    [(eq? dir 'backward)
     (rollback migration payload)]
    [else
     (error 'migration-apply "Invalid direction" dir)]))""",
    "migration-compose": """(define (compose-migrations first second)
  (unless (equal? (migration-to first) (migration-from second))
    (error 'migration-compose "Version mismatch"))

  (let* ([forward-1 (p-iso-forward (migration-iso first))]
         [forward-2 (p-iso-forward (migration-iso second))]
         [backward-1 (p-iso-backward (migration-iso first))]
         [backward-2 (p-iso-backward (migration-iso second))]
         [combined (make-p-iso
                    (lambda (x) (forward-2 (forward-1 x)))
                    (lambda (y) (backward-1 (backward-2 y))))]
         [name (string->symbol
                (format "~a->~a"
                        (migration-from first)
                        (migration-to second)))])
    (make-migration name
                    (migration-from first)
                    (migration-to second)
                    combined)))""",
    "migration-chain": """(define (compose-chain ms)
  (cond
    [(null? ms)
     (error 'migration-chain "Empty migration list")]
    [(null? (cdr ms))
     (car ms)]
    [else
     (fold-left migration-compose (car ms) (cdr ms))]))""",
    "migration-flip": """(define (invert-migration m)
  (let* ([iso (migration-iso m)]
         [reversed (make-p-iso (p-iso-backward iso)
                               (p-iso-forward iso))]
         [name (string->symbol
                (format "~a-reversed" (migration-name m)))])
    (make-migration name
                    (migration-to m)
                    (migration-from m)
                    reversed)))""",
    "make-migration-from-functions": """(define (migration-from-fns label src dst fw bw)
  (let ([iso (make-p-iso fw bw)])
    (make-migration label src dst iso)))""",
    "make-identity-migration": """(define (identity-migration label v)
  (let ([same v]
        [iso p-iso-id])
    (make-migration label same same iso)))""",
    "verify-migration-laws": """(define (migration-laws-hold? migration source-samples target-samples)
  (let ([iso (migration-iso migration)])
    (verify-p-iso-laws iso source-samples target-samples)))""",
}

BUGGY_CASES = [
    {
        "fn": "make-migration",
        "buggy": """(define (make-migration name from-version to-version iso)
  (list 'migratoin name from-version to-version iso))""",
        "note": "Migration records must be tagged with the exact symbol 'migration.",
    },
    {
        "fn": "make-migration",
        "buggy": """(define (make-migration name from-version to-version iso)
  (list 'migration name to-version from-version iso))""",
        "note": "from-version and to-version are swapped.",
    },
    {
        "fn": "migrate",
        "buggy": """(define (migrate m data)
  ((p-iso-backward (migration-iso m)) data))""",
        "note": "migrate must use the iso's forward function.",
    },
    {
        "fn": "migrate",
        "buggy": """(define (migrate m data)
  data)""",
        "note": "migrate cannot ignore the migration transform.",
    },
    {
        "fn": "rollback",
        "buggy": """(define (rollback m data)
  ((p-iso-forward (migration-iso m)) data))""",
        "note": "rollback must use the iso's backward function.",
    },
    {
        "fn": "rollback",
        "buggy": """(define (rollback m data)
  data)""",
        "note": "rollback cannot return payload unchanged in general.",
    },
    {
        "fn": "migration-apply",
        "buggy": """(define (migration-apply m data direction)
  (case direction
    [(forward) (migrate m data)]
    [(backward) (migrate m data)]
    [else (error 'migration-apply "Invalid direction" direction)]))""",
        "note": "Backward direction must dispatch to rollback.",
    },
    {
        "fn": "migration-apply",
        "buggy": """(define (migration-apply m data direction)
  (case direction
    [(forward) (migrate m data)]
    [(backward) (rollback m data)]
    [else (migrate m data)]))""",
        "note": "Invalid direction must raise an error, not silently fallback.",
    },
    {
        "fn": "migration-compose",
        "buggy": """(define (migration-compose m1 m2)
  (let* ([iso1 (migration-iso m1)]
         [iso2 (migration-iso m2)]
         [combined-iso (make-p-iso
                        (compose2 (p-iso-forward iso2) (p-iso-forward iso1))
                        (compose2 (p-iso-backward iso2) (p-iso-backward iso1)))]
         [combined-name (string->symbol
                         (format "~a->~a"
                                 (migration-from m1)
                                 (migration-to m2)))])
    (make-migration combined-name
                    (migration-from m1)
                    (migration-to m2)
                    combined-iso)))""",
        "note": "Backward composition order must be iso1.backward after iso2.backward.",
    },
    {
        "fn": "migration-compose",
        "buggy": """(define (migration-compose m1 m2)
  (let* ([iso1 (migration-iso m1)]
         [iso2 (migration-iso m2)]
         [combined-iso (make-p-iso
                        (compose2 (p-iso-forward iso2) (p-iso-forward iso1))
                        (compose2 (p-iso-backward iso1) (p-iso-backward iso2)))])
    (make-migration (migration-name m1)
                    (migration-from m1)
                    (migration-to m2)
                    combined-iso)))""",
        "note": "Composed migration name should reflect endpoint versions, not reuse m1 name.",
    },
    {
        "fn": "migration-chain",
        "buggy": """(define (migration-chain migrations)
  (if (null? migrations)
      #f
      (fold-left migration-compose (car migrations) (cdr migrations))))""",
        "note": "Empty migration lists must raise an error, not return #f.",
    },
    {
        "fn": "migration-chain",
        "buggy": """(define (migration-chain migrations)
  (cond
    [(null? migrations) (error 'migration-chain "Empty migration list")]
    [else (fold-right migration-compose (car migrations) (cdr migrations))]))""",
        "note": "Using fold-right with this seed changes composition semantics and can break endpoints.",
    },
    {
        "fn": "migration-flip",
        "buggy": """(define (migration-flip m)
  (let* ([iso (migration-iso m)]
         [flipped-iso (make-p-iso
                       (p-iso-backward iso)
                       (p-iso-forward iso))])
    (make-migration (migration-name m)
                    (migration-from m)
                    (migration-to m)
                    flipped-iso)))""",
        "note": "Flipped migrations must swap from/to versions and derive a reversed name.",
    },
    {
        "fn": "migration-flip",
        "buggy": """(define (migration-flip m)
  (let* ([iso (migration-iso m)]
         [flipped-name (string->symbol
                        (format "~a-reversed" (migration-name m)))])
    (make-migration flipped-name
                    (migration-to m)
                    (migration-from m)
                    iso)))""",
        "note": "Flipped migration must invert the iso direction, not reuse original iso.",
    },
    {
        "fn": "make-migration-from-functions",
        "buggy": """(define (make-migration-from-functions name from-ver to-ver forward backward)
  (make-migration name from-ver to-ver (make-p-iso backward forward)))""",
        "note": "Forward/backward arguments are reversed when building the iso.",
    },
    {
        "fn": "make-migration-from-functions",
        "buggy": """(define (make-migration-from-functions name from-ver to-ver forward backward)
  (make-migration name from-ver to-ver (make-p-iso forward identity)))""",
        "note": "Backward function must come from argument, not identity.",
    },
    {
        "fn": "make-identity-migration",
        "buggy": """(define (make-identity-migration name version)
  (make-migration name 'v0 version p-iso-id))""",
        "note": "Identity migration should keep from/to at the provided version.",
    },
    {
        "fn": "make-identity-migration",
        "buggy": """(define (make-identity-migration name version)
  (make-migration name version version
                  (make-p-iso add1 sub1)))""",
        "note": "Identity migration must use p-iso-id, not a non-identity transform.",
    },
    {
        "fn": "verify-migration-laws",
        "buggy": """(define (verify-migration-laws m test-as test-bs)
  (let ([iso (migration-iso m)])
    (andmap (lambda (a) (equal? ((p-iso-backward iso) ((p-iso-forward iso) a)) a))
            test-as)))""",
        "note": "Must verify both iso directions (source and target domains), not one side only.",
    },
    {
        "fn": "verify-migration-laws",
        "buggy": """(define (verify-migration-laws m test-as test-bs)
  (let ([iso (migration-iso m)])
    (verify-p-iso-laws iso test-bs test-as)))""",
        "note": "Argument order to verify-p-iso-laws is swapped.",
    },
]

BASE_DIFFICULTY = {
    "make-migration": "medium",
    "migrate": "medium",
    "rollback": "medium",
    "migration-apply": "medium",
    "migration-compose": "hard",
    "migration-chain": "hard",
    "migration-flip": "medium",
    "make-migration-from-functions": "medium",
    "make-identity-migration": "easy",
    "verify-migration-laws": "hard",
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
    out: List[str] = []
    i = 0
    n = len(defn)
    while i < n:
        ch = defn[i]
        if ch != "(":
            out.append(ch)
            i += 1
            continue

        j = i + 1
        while j < n and defn[j].isspace():
            j += 1
        if not defn.startswith("doc", j):
            out.append(ch)
            i += 1
            continue

        k = j + 3
        if k < n and (defn[k].isspace() or defn[k] in "()"):
            depth = 0
            in_string = False
            escaped = False
            in_comment = False
            while i < n:
                c = defn[i]
                if in_comment:
                    if c == "\n":
                        in_comment = False
                elif in_string:
                    if escaped:
                        escaped = False
                    elif c == "\\":
                        escaped = True
                    elif c == '"':
                        in_string = False
                else:
                    if c == ";":
                        in_comment = True
                    elif c == '"':
                        in_string = True
                    elif c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                i += 1
            continue

        out.append(ch)
        i += 1

    text = "".join(out)
    return "\n".join(line for line in text.splitlines() if line.strip())


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
    sid = f"optics_bidirectional_{family}_{family_counter[family]:03d}"
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
    "make-migration": "migration-type",
    "migrate": "core-operations",
    "rollback": "core-operations",
    "migration-apply": "core-operations",
    "migration-compose": "composition",
    "migration-chain": "composition",
    "migration-flip": "reversal",
    "make-migration-from-functions": "builders",
    "make-identity-migration": "builders",
    "verify-migration-laws": "verification",
}



def make_source_excerpt(fn: str, snippet: str) -> str:
    section = FUNCTION_SECTION[fn]
    indented = "\n".join(f"  {line}" for line in snippet.splitlines())
    return (
        ";;; lattice/optics/bidirectional.ss excerpt\n"
        "(require 'profunctor-optics)\n"
        "\n"
        "(doc 'module 'bidirectional)\n"
        f"(doc 'section '{section})\n"
        "\n"
        "(define (passthrough x) x)\n"
        "\n"
        f"{indented}\n"
    )


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (30)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=task_difficulty(fn, "spec_to_code", "direct"),
        source_function=fn,
        prompt=f"""Implement this bidirectional optics function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "bidirectional", "spec-to-code", fn],
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
        tags=["tier1", "optics", "bidirectional", "skeleton-completion", fn],
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
2. Preserve edge-case behavior and migration semantics.
3. Return only one production-ready definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "optics", "bidirectional", "contract-implementation", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (30)
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
        tags=["tier1", "optics", "bidirectional", "python-to-scheme", fn],
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
        tags=["tier1", "optics", "bidirectional", "chez-to-fold", fn],
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
        tags=["tier1", "optics", "bidirectional", "source-excerpt-to-fold", "doc-free-target", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (20)
# -----------------------------------------------------------------------------
if len(BUGGY_CASES) != 20:
    raise ValueError(f"expected 20 bugfix cases, found {len(BUGGY_CASES)}")

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
        tags=["tier1", "optics", "bidirectional", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition/use (50)
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
        tags=["tier1", "optics", "bidirectional", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # make-migration
    {
        "fn": "make-migration",
        "prompt": "Construct a migration and return its record tag symbol.",
        "gt": "(car (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)))",
        "verify": "(equal? (car (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))) 'migration)",
        "difficulty": "easy",
        "tags": ["record"],
    },
    {
        "fn": "make-migration",
        "prompt": "Create an increment migration and apply it forward to 10.",
        "gt": "(migrate (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 10)",
        "verify": "(= (migrate (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 10) 11)",
        "difficulty": "easy",
        "tags": ["forward"],
    },
    {
        "fn": "make-migration",
        "prompt": "Create an increment migration and apply rollback to 11.",
        "gt": "(rollback (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 11)",
        "verify": "(= (rollback (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 11) 10)",
        "difficulty": "easy",
        "tags": ["backward"],
    },
    {
        "fn": "make-migration",
        "prompt": "Build a prefixing migration and roundtrip the string \"abc\".",
        "gt": "(let* ([m (make-migration 'prefix 'raw 'tagged (make-p-iso (lambda (s) (string-append \"v2:\" s)) (lambda (s) (substring s 3 (string-length s)))))]) (rollback m (migrate m \"abc\")))",
        "verify": "(equal? (let* ([m (make-migration 'prefix 'raw 'tagged (make-p-iso (lambda (s) (string-append \"v2:\" s)) (lambda (s) (substring s 3 (string-length s)))))]) (rollback m (migrate m \"abc\"))) \"abc\")",
        "difficulty": "medium",
        "tags": ["roundtrip"],
    },
    {
        "fn": "make-migration",
        "prompt": "Create two migrations and test whether they are composition-compatible.",
        "gt": "(let* ([m1 (make-migration 'a 'v1 'v2 p-iso-id)] [m2 (make-migration 'b 'v2 'v3 p-iso-id)]) (migration-compatible? m1 m2))",
        "verify": "(equal? (let* ([m1 (make-migration 'a 'v1 'v2 p-iso-id)] [m2 (make-migration 'b 'v2 'v3 p-iso-id)]) (migration-compatible? m1 m2)) #t)",
        "difficulty": "medium",
        "tags": ["compatibility"],
    },

    # migrate
    {
        "fn": "migrate",
        "prompt": "Run migrate on a +1 migration with input 5.",
        "gt": "(migrate (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 5)",
        "verify": "(= (migrate (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 5) 6)",
        "difficulty": "easy",
        "tags": ["numeric"],
    },
    {
        "fn": "migrate",
        "prompt": "Compose +1 then *2 migrations and migrate input 4 through the composed migration.",
        "gt": "(let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (migration-compose m1 m2)]) (migrate m3 4))",
        "verify": "(= (let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (migration-compose m1 m2)]) (migrate m3 4)) 10)",
        "difficulty": "hard",
        "tags": ["compose"],
    },
    {
        "fn": "migrate",
        "prompt": "Map migrate across three numbers with an increment migration.",
        "gt": "(let ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))]) (map (lambda (n) (migrate m n)) '(1 2 3)))",
        "verify": "(equal? (let ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))]) (map (lambda (n) (migrate m n)) '(1 2 3))) '(2 3 4))",
        "difficulty": "medium",
        "tags": ["map"],
    },
    {
        "fn": "migrate",
        "prompt": "Flip an increment migration and migrate 10 through the flipped migration.",
        "gt": "(let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (migrate f 10))",
        "verify": "(= (let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (migrate f 10)) 9)",
        "difficulty": "medium",
        "tags": ["flip"],
    },
    {
        "fn": "migrate",
        "prompt": "Use migrate with an identity migration on list '(a b c).",
        "gt": "(migrate (make-identity-migration 'id 'v3) '(a b c))",
        "verify": "(equal? (migrate (make-identity-migration 'id 'v3) '(a b c)) '(a b c))",
        "difficulty": "easy",
        "tags": ["identity"],
    },

    # rollback
    {
        "fn": "rollback",
        "prompt": "Run rollback on a +1 migration with input 6.",
        "gt": "(rollback (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 6)",
        "verify": "(= (rollback (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 6) 5)",
        "difficulty": "easy",
        "tags": ["numeric"],
    },
    {
        "fn": "rollback",
        "prompt": "Roundtrip through migrate then rollback for value 12.",
        "gt": "(let ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))]) (rollback m (migrate m 12)))",
        "verify": "(= (let ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))]) (rollback m (migrate m 12))) 12)",
        "difficulty": "medium",
        "tags": ["roundtrip"],
    },
    {
        "fn": "rollback",
        "prompt": "Map rollback across prefixed strings using a prefix migration.",
        "gt": "(let ([m (make-migration-from-functions 'prefix 'raw 'tagged (lambda (s) (string-append \"v2:\" s)) (lambda (s) (substring s 3 (string-length s))))]) (map (lambda (s) (rollback m s)) '(\"v2:a\" \"v2:bc\")))",
        "verify": "(equal? (let ([m (make-migration-from-functions 'prefix 'raw 'tagged (lambda (s) (string-append \"v2:\" s)) (lambda (s) (substring s 3 (string-length s))))]) (map (lambda (s) (rollback m s)) '(\"v2:a\" \"v2:bc\"))) '(\"a\" \"bc\"))",
        "difficulty": "medium",
        "tags": ["map"],
    },
    {
        "fn": "rollback",
        "prompt": "Compose +1 then *2 migrations and rollback 10 through the composed migration.",
        "gt": "(let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (migration-compose m1 m2)]) (rollback m3 10))",
        "verify": "(= (let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (migration-compose m1 m2)]) (rollback m3 10)) 4)",
        "difficulty": "hard",
        "tags": ["compose"],
    },
    {
        "fn": "rollback",
        "prompt": "Flip an increment migration and rollback 10 through the flipped migration.",
        "gt": "(let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (rollback f 10))",
        "verify": "(= (let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (rollback f 10)) 11)",
        "difficulty": "medium",
        "tags": ["flip"],
    },

    # migration-apply
    {
        "fn": "migration-apply",
        "prompt": "Call migration-apply in forward direction for an increment migration.",
        "gt": "(migration-apply (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 8 'forward)",
        "verify": "(= (migration-apply (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 8 'forward) 9)",
        "difficulty": "easy",
        "tags": ["forward"],
    },
    {
        "fn": "migration-apply",
        "prompt": "Call migration-apply in backward direction for an increment migration.",
        "gt": "(migration-apply (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 8 'backward)",
        "verify": "(= (migration-apply (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 8 'backward) 7)",
        "difficulty": "easy",
        "tags": ["backward"],
    },
    {
        "fn": "migration-apply",
        "prompt": "Evaluate migration-apply with an unsupported direction and return a caught marker symbol.",
        "gt": "(guard (ex [else 'bad-direction]) (migration-apply (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 8 'sideways))",
        "verify": "(equal? (guard (ex [else 'bad-direction]) (migration-apply (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1)) 8 'sideways)) 'bad-direction)",
        "difficulty": "medium",
        "tags": ["error-path"],
    },
    {
        "fn": "migration-apply",
        "prompt": "Compose +1 then *2 migrations and apply the composed migration forward to 3.",
        "gt": "(let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (migration-compose m1 m2)]) (migration-apply m3 3 'forward))",
        "verify": "(= (let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (migration-compose m1 m2)]) (migration-apply m3 3 'forward)) 8)",
        "difficulty": "hard",
        "tags": ["compose"],
    },
    {
        "fn": "migration-apply",
        "prompt": "Use migration-apply in backward mode on a flipped migration.",
        "gt": "(let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (migration-apply f 9 'backward))",
        "verify": "(= (let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (migration-apply f 9 'backward)) 10)",
        "difficulty": "medium",
        "tags": ["flip"],
    },

    # migration-compose
    {
        "fn": "migration-compose",
        "prompt": "Compose +1 then *2 migrations and migrate 5 through the composition.",
        "gt": "(let* ([m1 (make-migration 'add1 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'double 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (migration-compose m1 m2)]) (migrate m3 5))",
        "verify": "(= (let* ([m1 (make-migration 'add1 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'double 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (migration-compose m1 m2)]) (migrate m3 5)) 12)",
        "difficulty": "hard",
        "tags": ["forward"],
    },
    {
        "fn": "migration-compose",
        "prompt": "Using the same composed migration, rollback 12 to the source version value.",
        "gt": "(let* ([m1 (make-migration 'add1 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'double 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (migration-compose m1 m2)]) (rollback m3 12))",
        "verify": "(= (let* ([m1 (make-migration 'add1 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'double 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (migration-compose m1 m2)]) (rollback m3 12)) 5)",
        "difficulty": "hard",
        "tags": ["backward"],
    },
    {
        "fn": "migration-compose",
        "prompt": "Compose two compatible migrations and return their boundary versions as a pair.",
        "gt": "(let* ([m1 (make-migration 'a 'v1 'v2 p-iso-id)] [m2 (make-migration 'b 'v2 'v3 p-iso-id)] [m3 (migration-compose m1 m2)]) (cons (migration-from m3) (migration-to m3)))",
        "verify": "(equal? (let* ([m1 (make-migration 'a 'v1 'v2 p-iso-id)] [m2 (make-migration 'b 'v2 'v3 p-iso-id)] [m3 (migration-compose m1 m2)]) (cons (migration-from m3) (migration-to m3))) '(v1 . v3))",
        "difficulty": "medium",
        "tags": ["versions"],
    },
    {
        "fn": "migration-compose",
        "prompt": "Attempt to compose incompatible migrations and catch the failure as a symbol.",
        "gt": "(guard (ex [else 'mismatch]) (migration-compose (make-migration 'a 'v1 'v2 p-iso-id) (make-migration 'b 'x 'y p-iso-id)))",
        "verify": "(equal? (guard (ex [else 'mismatch]) (migration-compose (make-migration 'a 'v1 'v2 p-iso-id) (make-migration 'b 'x 'y p-iso-id))) 'mismatch)",
        "difficulty": "medium",
        "tags": ["error-path"],
    },
    {
        "fn": "migration-compose",
        "prompt": "Check associative behavior by comparing left- and right-associated migration-compose results on input 6.",
        "gt": "(let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (make-migration 'c 'v3 'v4 (make-p-iso sub1 add1))] [left (migration-compose (migration-compose m1 m2) m3)] [right (migration-compose m1 (migration-compose m2 m3))]) (= (migrate left 6) (migrate right 6)))",
        "verify": "(equal? (let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (make-migration 'c 'v3 'v4 (make-p-iso sub1 add1))] [left (migration-compose (migration-compose m1 m2) m3)] [right (migration-compose m1 (migration-compose m2 m3))]) (= (migrate left 6) (migrate right 6))) #t)",
        "difficulty": "hard",
        "tags": ["associative"],
    },

    # migration-chain
    {
        "fn": "migration-chain",
        "prompt": "Chain +1, *2, and -3 migrations, then migrate input 5.",
        "gt": "(let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (make-migration 'c 'v3 'v4 (make-p-iso (lambda (x) (- x 3)) (lambda (x) (+ x 3))))] [chain (migration-chain (list m1 m2 m3))]) (migrate chain 5))",
        "verify": "(= (let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (make-migration 'c 'v3 'v4 (make-p-iso (lambda (x) (- x 3)) (lambda (x) (+ x 3))))] [chain (migration-chain (list m1 m2 m3))]) (migrate chain 5)) 9)",
        "difficulty": "hard",
        "tags": ["forward"],
    },
    {
        "fn": "migration-chain",
        "prompt": "Chain a singleton migration list and apply migrate to 7.",
        "gt": "(let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [chain (migration-chain (list m1))]) (migrate chain 7))",
        "verify": "(= (let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [chain (migration-chain (list m1))]) (migrate chain 7)) 8)",
        "difficulty": "easy",
        "tags": ["singleton"],
    },
    {
        "fn": "migration-chain",
        "prompt": "Invoke migration-chain on an empty list and catch the expected error marker.",
        "gt": "(guard (ex [else 'empty]) (migration-chain '()))",
        "verify": "(equal? (guard (ex [else 'empty]) (migration-chain '())) 'empty)",
        "difficulty": "medium",
        "tags": ["error-path"],
    },
    {
        "fn": "migration-chain",
        "prompt": "Chain two migrations (+1 then *2) and rollback 10.",
        "gt": "(let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [chain (migration-chain (list m1 m2))]) (rollback chain 10))",
        "verify": "(= (let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [chain (migration-chain (list m1 m2))]) (rollback chain 10)) 4)",
        "difficulty": "hard",
        "tags": ["backward"],
    },
    {
        "fn": "migration-chain",
        "prompt": "Chain identity followed by increment migration and migrate input 3.",
        "gt": "(let* ([m1 (make-identity-migration 'id 'v1)] [m2 (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [chain (migration-chain (list m1 m2))]) (migrate chain 3))",
        "verify": "(= (let* ([m1 (make-identity-migration 'id 'v1)] [m2 (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [chain (migration-chain (list m1 m2))]) (migrate chain 3)) 4)",
        "difficulty": "medium",
        "tags": ["identity-compose"],
    },

    # migration-flip
    {
        "fn": "migration-flip",
        "prompt": "Flip an increment migration and return its version endpoints as a pair.",
        "gt": "(let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (cons (migration-from f) (migration-to f)))",
        "verify": "(equal? (let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (cons (migration-from f) (migration-to f))) '(v2 . v1))",
        "difficulty": "easy",
        "tags": ["versions"],
    },
    {
        "fn": "migration-flip",
        "prompt": "Flip an increment migration and migrate value 10.",
        "gt": "(let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (migrate f 10))",
        "verify": "(= (let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (migrate f 10)) 9)",
        "difficulty": "medium",
        "tags": ["forward"],
    },
    {
        "fn": "migration-flip",
        "prompt": "Flip an increment migration and rollback value 10.",
        "gt": "(let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (rollback f 10))",
        "verify": "(= (let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (rollback f 10)) 11)",
        "difficulty": "medium",
        "tags": ["backward"],
    },
    {
        "fn": "migration-flip",
        "prompt": "Double-flip a migration and compare migrate output against the original on input 4.",
        "gt": "(let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [ff (migration-flip (migration-flip m))]) (= (migrate ff 4) (migrate m 4)))",
        "verify": "(equal? (let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [ff (migration-flip (migration-flip m))]) (= (migrate ff 4) (migrate m 4))) #t)",
        "difficulty": "medium",
        "tags": ["involution"],
    },
    {
        "fn": "migration-flip",
        "prompt": "Compose a migration with its flipped counterpart and migrate input 7.",
        "gt": "(let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)] [id-like (migration-compose m f)]) (migrate id-like 7))",
        "verify": "(= (let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)] [id-like (migration-compose m f)]) (migrate id-like 7)) 7)",
        "difficulty": "hard",
        "tags": ["compose"],
    },

    # make-migration-from-functions
    {
        "fn": "make-migration-from-functions",
        "prompt": "Build a prefix migration from raw functions and migrate \"abc\".",
        "gt": "(migrate (make-migration-from-functions 'prefix 'raw 'tagged (lambda (s) (string-append \"v2:\" s)) (lambda (s) (substring s 3 (string-length s)))) \"abc\")",
        "verify": "(equal? (migrate (make-migration-from-functions 'prefix 'raw 'tagged (lambda (s) (string-append \"v2:\" s)) (lambda (s) (substring s 3 (string-length s)))) \"abc\") \"v2:abc\")",
        "difficulty": "easy",
        "tags": ["forward"],
    },
    {
        "fn": "make-migration-from-functions",
        "prompt": "Build the same prefix migration and rollback \"v2:abc\".",
        "gt": "(rollback (make-migration-from-functions 'prefix 'raw 'tagged (lambda (s) (string-append \"v2:\" s)) (lambda (s) (substring s 3 (string-length s)))) \"v2:abc\")",
        "verify": "(equal? (rollback (make-migration-from-functions 'prefix 'raw 'tagged (lambda (s) (string-append \"v2:\" s)) (lambda (s) (substring s 3 (string-length s)))) \"v2:abc\") \"abc\")",
        "difficulty": "easy",
        "tags": ["backward"],
    },
    {
        "fn": "make-migration-from-functions",
        "prompt": "Create a migration from functions and test its version endpoints against v1->v2.",
        "gt": "(let ([m (make-migration-from-functions 'num 'v1 'v2 add1 sub1)]) (migration-versions-match? m 'v1 'v2))",
        "verify": "(equal? (let ([m (make-migration-from-functions 'num 'v1 'v2 add1 sub1)]) (migration-versions-match? m 'v1 'v2)) #t)",
        "difficulty": "medium",
        "tags": ["versions"],
    },
    {
        "fn": "make-migration-from-functions",
        "prompt": "Compose a function-built migration with an identity migration and migrate input 2.",
        "gt": "(let* ([m1 (make-migration-from-functions 'num 'v1 'v2 add1 sub1)] [m2 (make-identity-migration 'id 'v2)] [m3 (migration-compose m1 m2)]) (migrate m3 2))",
        "verify": "(= (let* ([m1 (make-migration-from-functions 'num 'v1 'v2 add1 sub1)] [m2 (make-identity-migration 'id 'v2)] [m3 (migration-compose m1 m2)]) (migrate m3 2)) 3)",
        "difficulty": "hard",
        "tags": ["compose"],
    },
    {
        "fn": "make-migration-from-functions",
        "prompt": "Run verify-migration-laws over a migration created from add1/sub1.",
        "gt": "(verify-migration-laws (make-migration-from-functions 'num 'v1 'v2 add1 sub1) '(1 2 3) '(2 3 4))",
        "verify": "(equal? (verify-migration-laws (make-migration-from-functions 'num 'v1 'v2 add1 sub1) '(1 2 3) '(2 3 4)) #t)",
        "difficulty": "medium",
        "tags": ["laws"],
    },

    # make-identity-migration
    {
        "fn": "make-identity-migration",
        "prompt": "Migrate numeric value 42 through an identity migration.",
        "gt": "(migrate (make-identity-migration 'id 'v5) 42)",
        "verify": "(= (migrate (make-identity-migration 'id 'v5) 42) 42)",
        "difficulty": "easy",
        "tags": ["forward"],
    },
    {
        "fn": "make-identity-migration",
        "prompt": "Rollback list '(a b) through an identity migration.",
        "gt": "(rollback (make-identity-migration 'id 'v5) '(a b))",
        "verify": "(equal? (rollback (make-identity-migration 'id 'v5) '(a b)) '(a b))",
        "difficulty": "easy",
        "tags": ["backward"],
    },
    {
        "fn": "make-identity-migration",
        "prompt": "Compose identity then increment migration and migrate input 5.",
        "gt": "(let* ([m1 (make-identity-migration 'id 'v1)] [m2 (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [m3 (migration-compose m1 m2)]) (migrate m3 5))",
        "verify": "(= (let* ([m1 (make-identity-migration 'id 'v1)] [m2 (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [m3 (migration-compose m1 m2)]) (migrate m3 5)) 6)",
        "difficulty": "medium",
        "tags": ["compose-left"],
    },
    {
        "fn": "make-identity-migration",
        "prompt": "Compose increment then identity migration and rollback input 9.",
        "gt": "(let* ([m1 (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-identity-migration 'id 'v2)] [m3 (migration-compose m1 m2)]) (rollback m3 9))",
        "verify": "(= (let* ([m1 (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-identity-migration 'id 'v2)] [m3 (migration-compose m1 m2)]) (rollback m3 9)) 8)",
        "difficulty": "medium",
        "tags": ["compose-right"],
    },
    {
        "fn": "make-identity-migration",
        "prompt": "Chain two identity migrations and migrate input '(x y).",
        "gt": "(let* ([m1 (make-identity-migration 'id1 'v1)] [m2 (make-identity-migration 'id2 'v1)] [chain (migration-chain (list m1 m2))]) (migrate chain '(x y)))",
        "verify": "(equal? (let* ([m1 (make-identity-migration 'id1 'v1)] [m2 (make-identity-migration 'id2 'v1)] [chain (migration-chain (list m1 m2))]) (migrate chain '(x y))) '(x y))",
        "difficulty": "medium",
        "tags": ["chain"],
    },

    # verify-migration-laws
    {
        "fn": "verify-migration-laws",
        "prompt": "Check laws for add1/sub1 migration on sample domains.",
        "gt": "(verify-migration-laws (make-migration-from-functions 'num 'v1 'v2 add1 sub1) '(1 2 3) '(2 3 4))",
        "verify": "(equal? (verify-migration-laws (make-migration-from-functions 'num 'v1 'v2 add1 sub1) '(1 2 3) '(2 3 4)) #t)",
        "difficulty": "medium",
        "tags": ["good-laws"],
    },
    {
        "fn": "verify-migration-laws",
        "prompt": "Check laws for a broken migration whose backward function is identity.",
        "gt": "(verify-migration-laws (make-migration-from-functions 'bad 'v1 'v2 add1 identity) '(1 2 3) '(2 3 4))",
        "verify": "(equal? (verify-migration-laws (make-migration-from-functions 'bad 'v1 'v2 add1 identity) '(1 2 3) '(2 3 4)) #f)",
        "difficulty": "medium",
        "tags": ["bad-laws"],
    },
    {
        "fn": "verify-migration-laws",
        "prompt": "Compose +1 and *2 migrations, then verify iso laws for the composition.",
        "gt": "(let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (migration-compose m1 m2)]) (verify-migration-laws m3 '(1 2 3) '(4 6 8)))",
        "verify": "(equal? (let* ([m1 (make-migration 'a 'v1 'v2 (make-p-iso add1 sub1))] [m2 (make-migration 'b 'v2 'v3 (make-p-iso (lambda (x) (* x 2)) (lambda (x) (/ x 2))))] [m3 (migration-compose m1 m2)]) (verify-migration-laws m3 '(1 2 3) '(4 6 8))) #t)",
        "difficulty": "hard",
        "tags": ["compose"],
    },
    {
        "fn": "verify-migration-laws",
        "prompt": "Flip an increment migration and verify its laws on reversed domains.",
        "gt": "(let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (verify-migration-laws f '(2 3 4) '(1 2 3)))",
        "verify": "(equal? (let* ([m (make-migration 'inc 'v1 'v2 (make-p-iso add1 sub1))] [f (migration-flip m)]) (verify-migration-laws f '(2 3 4) '(1 2 3))) #t)",
        "difficulty": "hard",
        "tags": ["flip"],
    },
    {
        "fn": "verify-migration-laws",
        "prompt": "Verify laws for an identity migration over symbol samples.",
        "gt": "(verify-migration-laws (make-identity-migration 'id 'v0) '(a b c) '(a b c))",
        "verify": "(equal? (verify-migration-laws (make-identity-migration 'id 'v0) '(a b c) '(a b c)) #t)",
        "difficulty": "easy",
        "tags": ["identity"],
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
    "spec_to_code": 5,
    "translation": 5,
    "bugfix": 4,
    "composition": 9,
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
