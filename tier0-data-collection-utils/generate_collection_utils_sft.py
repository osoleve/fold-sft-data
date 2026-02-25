#!/usr/bin/env python3
"""Generate SFT samples for lattice/data/collection-utils.ss (pure subset)."""

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

SOURCE_MODULE = "lattice/data/collection-utils.ss"
SOURCE_TEST = "lattice/data/test-collection-utils.ss"

DEFS: Dict[str, str] = {
    "foldr": """(define (foldr f init lst)
  (if (null? lst)
      init
      (f (car lst) (foldr f init (cdr lst)))))""",
    "collection-hashes": """(define (collection-hashes collection)
  (let ([refs (block-refs collection)])
       (let loop ([i 0]
                  [result '()])
            (if (>= i (vector-length refs))
                (reverse result)
                (loop (+ i 1)
                      (cons (vector-ref refs i) result))))))""",
    "collection-size": """(define (collection-size collection)
  (vector-length (block-refs collection)))""",
    "collection-empty?": """(define (collection-empty? collection)
  (= (vector-length (block-refs collection)) 0))""",
    "make-collection-from-blocks": """(define (make-collection-from-blocks tag name blocks)
  (let ([hashes (map hash-block blocks)])
       (make-block tag
                   (string->utf8 (format "~a (~a members)" name (length blocks)))
                   (list->vector hashes))))""",
    "collection-add": """(define (collection-add collection new-member)
  (let* ([old-refs (block-refs collection)]
         [new-hash (hash-block new-member)]
         [new-refs (list->vector (cons new-hash (vector->list old-refs)))])
        (make-block (block-tag collection)
                    (block-payload collection)
                    new-refs)))""",
    "collection-remove": """(define (collection-remove collection member-hash)
  (let* ([old-refs (block-refs collection)]
         [new-refs (list->vector
                    (filter (lambda (h) (not (equal? h member-hash)))
                            (vector->list old-refs)))])
        (make-block (block-tag collection)
                    (block-payload collection)
                    new-refs)))""",
    "collection-merge": """(define (collection-merge coll1 coll2)
  (let* ([refs1 (block-refs coll1)]
         [refs2 (block-refs coll2)]
         [combined (append (vector->list refs1) (vector->list refs2))]
         [new-refs (list->vector combined)])
        (make-block (block-tag coll1)
                    (string->utf8 (format "merged (~a members)" (length combined)))
                    new-refs)))""",
}

DEPENDS: Dict[str, List[str]] = {
    "foldr": [],
    "collection-hashes": [],
    "collection-size": [],
    "collection-empty?": [],
    "make-collection-from-blocks": [],
    "collection-add": [],
    "collection-remove": [],
    "collection-merge": [],
}

FUNCTION_ORDER = [
    "foldr",
    "collection-hashes",
    "collection-size",
    "collection-empty?",
    "make-collection-from-blocks",
    "collection-add",
    "collection-remove",
    "collection-merge",
]

FUNCTION_SPECS = {
    "foldr": "Right-associative fold over lists: f x1 (f x2 (... init)).",
    "collection-hashes": "Extract collection ref hashes as a list in vector order.",
    "collection-size": "Return number of refs in a collection block.",
    "collection-empty?": "Return #t iff collection has zero refs.",
    "make-collection-from-blocks": "Build a collection block from a list of blocks by hashing each member.",
    "collection-add": "Return a new collection with new-member hash prepended to existing refs.",
    "collection-remove": "Return a new collection with all refs equal to member-hash removed.",
    "collection-merge": "Return a new collection combining refs from coll1 then coll2.",
}

SKELETONS = {
    "foldr": """(define (foldr f init lst)
  ;; TODO: right-associative list fold
  <TODO>)""",
    "collection-hashes": """(define (collection-hashes collection)
  ;; TODO: return refs as a list preserving vector order
  <TODO>)""",
    "collection-size": """(define (collection-size collection)
  ;; TODO: number of references in collection
  <TODO>)""",
    "collection-empty?": """(define (collection-empty? collection)
  ;; TODO: check whether collection has zero members
  <TODO>)""",
    "make-collection-from-blocks": """(define (make-collection-from-blocks tag name blocks)
  ;; TODO: hash each block and create a new collection block
  <TODO>)""",
    "collection-add": """(define (collection-add collection new-member)
  ;; TODO: create new immutable collection with one additional member hash
  <TODO>)""",
    "collection-remove": """(define (collection-remove collection member-hash)
  ;; TODO: remove matching hash values from refs
  <TODO>)""",
    "collection-merge": """(define (collection-merge coll1 coll2)
  ;; TODO: concatenate refs from coll1 then coll2 into a new block
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "foldr": "(and (equal? (foldr cons '() '(1 2 3)) '(1 2 3)) (= (foldr + 0 '(1 2 3 4)) 10))",
    "collection-hashes": "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [coll (make-block 'collection (string->utf8 \"c\") (vector (hash-block b1) (hash-block b2)))]) (equal? (collection-hashes coll) (list (hash-block b1) (hash-block b2))))",
    "collection-size": "(let ([c1 (make-block 'collection (string->utf8 \"empty\") (vector))] [c2 (make-block 'collection (string->utf8 \"full\") (vector 1 2 3))]) (and (= (collection-size c1) 0) (= (collection-size c2) 3)))",
    "collection-empty?": "(and (collection-empty? (make-block 'collection (string->utf8 \"e\") (vector))) (not (collection-empty? (make-block 'collection (string->utf8 \"n\") (vector 1)))))",
    "make-collection-from-blocks": "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2))]) (and (eq? (block-tag c) 'group) (= (collection-size c) 2)))",
    "collection-add": "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-add c b2)]) (and (= (collection-size c) 1) (= (collection-size c2) 2) (equal? (car (collection-hashes c2)) (hash-block b2))))",
    "collection-remove": "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [b3 (make-block 'entity (string->utf8 \"c\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2 b3))] [c2 (collection-remove c (hash-block b2))]) (and (= (collection-size c2) 2) (not (member (hash-block b2) (collection-hashes c2)))))",
    "collection-merge": "(let* ([a (make-block 'entity (string->utf8 \"a\") (vector))] [b (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-block 'entity (string->utf8 \"c\") (vector))] [c1 (make-collection-from-blocks 'group \"x\" (list a b))] [c2 (make-collection-from-blocks 'group \"y\" (list c))] [m (collection-merge c1 c2)]) (and (= (collection-size m) 3) (eq? (block-tag m) 'group)))",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")
TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "foldr": "def foldr(f, init, xs):\n    if not xs:\n        return init\n    return f(xs[0], foldr(f, init, xs[1:]))",
    "collection-hashes": "def collection_hashes(collection):\n    refs = block_refs(collection)\n    out = []\n    for i in range(len(refs)):\n        out.append(refs[i])\n    return out",
    "collection-size": "def collection_size(collection):\n    return len(block_refs(collection))",
    "collection-empty?": "def collection_empty(collection):\n    return len(block_refs(collection)) == 0",
    "make-collection-from-blocks": "def make_collection_from_blocks(tag, name, blocks):\n    hashes = [hash_block(b) for b in blocks]\n    payload = f\"{name} ({len(blocks)} members)\"\n    return make_block(tag, utf8(payload), list_to_vector(hashes))",
    "collection-add": "def collection_add(collection, new_member):\n    old_refs = block_refs(collection)\n    new_hash = hash_block(new_member)\n    new_refs = list_to_vector([new_hash] + vector_to_list(old_refs))\n    return make_block(block_tag(collection), block_payload(collection), new_refs)",
    "collection-remove": "def collection_remove(collection, member_hash):\n    old_refs = block_refs(collection)\n    new_refs = list_to_vector([h for h in vector_to_list(old_refs) if h != member_hash])\n    return make_block(block_tag(collection), block_payload(collection), new_refs)",
    "collection-merge": "def collection_merge(coll1, coll2):\n    refs1 = block_refs(coll1)\n    refs2 = block_refs(coll2)\n    combined = vector_to_list(refs1) + vector_to_list(refs2)\n    return make_block(block_tag(coll1), utf8(f\"merged ({len(combined)} members)\"), list_to_vector(combined))",
}

CHEZ_SNIPPETS = {
    "foldr": "(define (fold-right f z xs)\n  (if (null? xs)\n      z\n      (f (car xs) (fold-right f z (cdr xs)))))",
    "collection-hashes": "(define (hashes-of coll)\n  (let ((refs (block-refs coll)))\n    (let loop ((i 0) (acc '()))\n      (if (>= i (vector-length refs))\n          (reverse acc)\n          (loop (+ i 1) (cons (vector-ref refs i) acc))))))",
    "collection-size": "(define (coll-size coll)\n  (vector-length (block-refs coll)))",
    "collection-empty?": "(define (coll-empty? coll)\n  (= (vector-length (block-refs coll)) 0))",
    "make-collection-from-blocks": "(define (mk-collection tag name blocks)\n  (let ((hs (map hash-block blocks)))\n    (make-block tag\n                (string->utf8 (format \"~a (~a members)\" name (length blocks)))\n                (list->vector hs))))",
    "collection-add": "(define (coll-add coll member)\n  (let* ((old (block-refs coll))\n         (h (hash-block member))\n         (new (list->vector (cons h (vector->list old)))))\n    (make-block (block-tag coll) (block-payload coll) new)))",
    "collection-remove": "(define (coll-remove coll h)\n  (let* ((old (block-refs coll))\n         (new (list->vector (filter (lambda (x) (not (equal? x h))) (vector->list old)))))\n    (make-block (block-tag coll) (block-payload coll) new)))",
    "collection-merge": "(define (coll-merge a b)\n  (let* ((r1 (block-refs a))\n         (r2 (block-refs b))\n         (combined (append (vector->list r1) (vector->list r2))))\n    (make-block (block-tag a)\n                (string->utf8 (format \"merged (~a members)\" (length combined)))\n                (list->vector combined))))",
}

BUGGY_CASES = [
    {"fn": "foldr", "buggy": "(define (foldr f init lst)\n  (fold-left f init lst))", "note": "foldr must be right-associative, not left-associative."},
    {"fn": "foldr", "buggy": "(define (foldr f init lst)\n  init)", "note": "Non-empty lists must apply f recursively."},
    {"fn": "collection-hashes", "buggy": "(define (collection-hashes collection)\n  (reverse (vector->list (block-refs collection))))", "note": "Hashes must be returned in collection ref order, not reversed."},
    {"fn": "collection-hashes", "buggy": "(define (collection-hashes collection)\n  '())", "note": "Must return all member hashes from refs."},
    {"fn": "collection-size", "buggy": "(define (collection-size collection)\n  (+ 1 (vector-length (block-refs collection))))", "note": "Size should equal the exact vector length."},
    {"fn": "collection-size", "buggy": "(define (collection-size collection)\n  0)", "note": "Non-empty collections must report their actual size."},
    {"fn": "collection-empty?", "buggy": "(define (collection-empty? collection)\n  #f)", "note": "Empty collections must return #t."},
    {"fn": "collection-empty?", "buggy": "(define (collection-empty? collection)\n  (= (vector-length (block-refs collection)) 1))", "note": "Predicate should check for zero members."},
    {"fn": "make-collection-from-blocks", "buggy": "(define (make-collection-from-blocks tag name blocks)\n  (make-block tag (string->utf8 name) (vector)))", "note": "Must include hashes of all provided blocks in refs."},
    {"fn": "make-collection-from-blocks", "buggy": "(define (make-collection-from-blocks tag name blocks)\n  (let ([hashes (map hash-block blocks)])\n       (make-block 'collection (string->utf8 name) (list->vector hashes))))", "note": "Result should preserve the caller-supplied tag and payload format."},
    {"fn": "collection-add", "buggy": "(define (collection-add collection new-member)\n  collection)", "note": "Must return a new collection with the new member hash inserted."},
    {"fn": "collection-add", "buggy": "(define (collection-add collection new-member)\n  (let* ([old-refs (block-refs collection)]\n         [new-hash (hash-block new-member)]\n         [new-refs (list->vector (append (vector->list old-refs) (list new-hash)))])\n        (make-block (block-tag collection)\n                    (block-payload collection)\n                    new-refs)))", "note": "Module semantics prepend new member hash to refs."},
    {"fn": "collection-remove", "buggy": "(define (collection-remove collection member-hash)\n  collection)", "note": "Must remove matching hashes from refs."},
    {"fn": "collection-remove", "buggy": "(define (collection-remove collection member-hash)\n  (let* ([old-refs (block-refs collection)]\n         [new-refs (list->vector (filter (lambda (h) (equal? h member-hash)) (vector->list old-refs)))])\n        (make-block (block-tag collection)\n                    (block-payload collection)\n                    new-refs)))", "note": "Predicate is inverted; this keeps only the removed hash."},
    {"fn": "collection-merge", "buggy": "(define (collection-merge coll1 coll2)\n  coll1)", "note": "Merge must include refs from both collections."},
    {"fn": "collection-merge", "buggy": "(define (collection-merge coll1 coll2)\n  (let* ([refs1 (block-refs coll1)]\n         [refs2 (block-refs coll2)]\n         [combined (append (vector->list refs2) (vector->list refs1))]\n         [new-refs (list->vector combined)])\n        (make-block (block-tag coll1)\n                    (string->utf8 \"merged\")\n                    new-refs)))", "note": "Module ordering is refs1 then refs2, with merged payload including member count."},
]

DIFFICULTY = {
    "foldr": "easy",
    "collection-hashes": "medium",
    "collection-size": "easy",
    "collection-empty?": "easy",
    "make-collection-from-blocks": "medium",
    "collection-add": "medium",
    "collection-remove": "medium",
    "collection-merge": "medium",
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
    sid = f"collection_utils_{family}_{family_counter[family]:03d}"
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
    return [name for name in FUNCTION_ORDER if name != fn and name in tokens]


def dependency_closure(fn: str) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    def visit(name: str) -> None:
        for dep in DEPENDS.get(name, []):
            if dep == fn:
                continue
            if dep not in seen:
                seen.add(dep)
                visit(dep)
                ordered.append(dep)

    for dep in DEPENDS.get(fn, []):
        if dep == fn:
            continue
        if dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    for dep in verify_refs(fn):
        if dep == fn:
            continue
        if dep not in seen:
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    return ordered


def def_verify(fn: str) -> str:
    parts = [DEFS[d] for d in dependency_closure(fn)] + [DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement this collection utility in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["data", "collection-utils", "spec-to-code", fn],
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
        tags=["data", "collection-utils", "skeleton-completion", fn],
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
        tags=["data", "collection-utils", "python-to-scheme", fn],
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
        tags=["data", "collection-utils", "chez-to-fold", fn],
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
        tags=["data", "collection-utils", "bugfix", fn],
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
        verify_expr=verify_expr,
        tags=["data", "collection-utils", "composition", source_function] + extra_tags,
    )


composition_cases = [
    # Direct
    ("foldr", "Compute sum of '(1 2 3 4) using foldr.", "(foldr + 0 '(1 2 3 4))", "(equal? (foldr + 0 '(1 2 3 4)) 10)", "easy", ["direct"]),
    ("foldr", "Use foldr with cons to reconstruct '(a b c).", "(foldr cons '() '(a b c))", "(equal? (foldr cons '() '(a b c)) '(a b c))", "easy", ["direct"]),
    ("collection-hashes", "Return number of hashes in a 2-member collection.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2))]) (length (collection-hashes c)))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2))]) (length (collection-hashes c))) 2)", "medium", ["direct"]),
    ("collection-hashes", "Return first hash after collection-add prepends a member.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-add c b2)]) (equal? (car (collection-hashes c2)) (hash-block b2)))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-add c b2)]) (equal? (car (collection-hashes c2)) (hash-block b2))) #t)", "medium", ["direct"]),
    ("collection-size", "Return size of a collection with refs vector '(1 2 3).", "(collection-size (make-block 'collection (string->utf8 \"x\") (vector 1 2 3)))", "(equal? (collection-size (make-block 'collection (string->utf8 \"x\") (vector 1 2 3))) 3)", "easy", ["direct"]),
    ("collection-size", "Return size of an empty collection block.", "(collection-size (make-block 'collection (string->utf8 \"x\") (vector)))", "(equal? (collection-size (make-block 'collection (string->utf8 \"x\") (vector))) 0)", "easy", ["direct"]),
    ("collection-empty?", "Check whether empty collection block is empty.", "(collection-empty? (make-block 'collection (string->utf8 \"empty\") (vector)))", "(equal? (collection-empty? (make-block 'collection (string->utf8 \"empty\") (vector))) #t)", "easy", ["direct"]),
    ("collection-empty?", "Check whether non-empty collection block is empty.", "(collection-empty? (make-block 'collection (string->utf8 \"full\") (vector 1)))", "(equal? (collection-empty? (make-block 'collection (string->utf8 \"full\") (vector 1))) #f)", "easy", ["direct"]),
    ("make-collection-from-blocks", "Create a collection from three blocks and return its size.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [b3 (make-block 'entity (string->utf8 \"c\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2 b3))]) (collection-size c))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [b3 (make-block 'entity (string->utf8 \"c\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2 b3))]) (collection-size c)) 3)", "medium", ["direct"]),
    ("make-collection-from-blocks", "Create collection and return whether tag is preserved.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [c (make-collection-from-blocks 'custom \"demo\" (list b1))]) (eq? (block-tag c) 'custom))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [c (make-collection-from-blocks 'custom \"demo\" (list b1))]) (eq? (block-tag c) 'custom)) #t)", "medium", ["direct"]),
    ("collection-add", "Add a block to a 1-member collection and return new size.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-add c b2)]) (collection-size c2))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-add c b2)]) (collection-size c2)) 2)", "medium", ["direct"]),
    ("collection-add", "Add a member and check that its hash appears first in refs.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-add c b2)]) (equal? (car (collection-hashes c2)) (hash-block b2)))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-add c b2)]) (equal? (car (collection-hashes c2)) (hash-block b2))) #t)", "medium", ["direct"]),
    ("collection-remove", "Remove a known member hash and return resulting size.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2))] [c2 (collection-remove c (hash-block b1))]) (collection-size c2))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2))] [c2 (collection-remove c (hash-block b1))]) (collection-size c2)) 1)", "medium", ["direct"]),
    ("collection-remove", "Remove a missing hash and confirm size stays unchanged.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-remove c 123456)]) (= (collection-size c2) (collection-size c)))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-remove c 123456)]) (= (collection-size c2) (collection-size c))) #t)", "medium", ["direct"]),
    ("collection-merge", "Merge 2-member and 1-member collections and return size.", "(let* ([a (make-block 'entity (string->utf8 \"a\") (vector))] [b (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-block 'entity (string->utf8 \"c\") (vector))] [c1 (make-collection-from-blocks 'group \"x\" (list a b))] [c2 (make-collection-from-blocks 'group \"y\" (list c))] [m (collection-merge c1 c2)]) (collection-size m))", "(equal? (let* ([a (make-block 'entity (string->utf8 \"a\") (vector))] [b (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-block 'entity (string->utf8 \"c\") (vector))] [c1 (make-collection-from-blocks 'group \"x\" (list a b))] [c2 (make-collection-from-blocks 'group \"y\" (list c))] [m (collection-merge c1 c2)]) (collection-size m)) 3)", "medium", ["direct"]),
    ("collection-merge", "Merge two collections and check resulting tag comes from first collection.", "(let* ([a (make-block 'entity (string->utf8 \"a\") (vector))] [b (make-block 'entity (string->utf8 \"b\") (vector))] [c1 (make-collection-from-blocks 'left \"x\" (list a))] [c2 (make-collection-from-blocks 'right \"y\" (list b))] [m (collection-merge c1 c2)]) (eq? (block-tag m) 'left))", "(equal? (let* ([a (make-block 'entity (string->utf8 \"a\") (vector))] [b (make-block 'entity (string->utf8 \"b\") (vector))] [c1 (make-collection-from-blocks 'left \"x\" (list a))] [c2 (make-collection-from-blocks 'right \"y\" (list b))] [m (collection-merge c1 c2)]) (eq? (block-tag m) 'left)) #t)", "medium", ["direct"]),

    # Properties
    ("foldr", "Return #t iff foldr subtraction is right-associative on '(1 2 3).", "(= (foldr - 0 '(1 2 3)) 2)", "(equal? (= (foldr - 0 '(1 2 3)) 2) #t)", "medium", ["property"]),
    ("collection-hashes", "Return #t iff collection-hashes length matches collection-size.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2))]) (= (length (collection-hashes c)) (collection-size c)))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2))]) (= (length (collection-hashes c)) (collection-size c))) #t)", "medium", ["property"]),
    ("collection-size", "Return #t iff collection-size is non-negative for any constructed collection.", "(let ([c (make-block 'collection (string->utf8 \"x\") (vector 1 2 3))]) (>= (collection-size c) 0))", "(equal? (let ([c (make-block 'collection (string->utf8 \"x\") (vector 1 2 3))]) (>= (collection-size c) 0)) #t)", "easy", ["property"]),
    ("collection-empty?", "Return #t iff collection-empty? agrees with size==0.", "(let ([c (make-block 'collection (string->utf8 \"x\") (vector))]) (equal? (collection-empty? c) (= (collection-size c) 0)))", "(equal? (let ([c (make-block 'collection (string->utf8 \"x\") (vector))]) (equal? (collection-empty? c) (= (collection-size c) 0))) #t)", "easy", ["property"]),
    ("make-collection-from-blocks", "Return #t iff making a collection from empty block list yields an empty collection.", "(collection-empty? (make-collection-from-blocks 'group \"empty\" '()))", "(equal? (collection-empty? (make-collection-from-blocks 'group \"empty\" '())) #t)", "medium", ["property"]),
    ("collection-add", "Return #t iff collection-add preserves original collection size (immutability check).", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-add c b2)]) (= (collection-size c) 1))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-add c b2)]) (= (collection-size c) 1)) #t)", "medium", ["property"]),
    ("collection-remove", "Return #t iff removing an existing hash decreases size by one.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2))] [c2 (collection-remove c (hash-block b1))]) (= (collection-size c2) (- (collection-size c) 1)))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2))] [c2 (collection-remove c (hash-block b1))]) (= (collection-size c2) (- (collection-size c) 1))) #t)", "medium", ["property"]),
    ("collection-merge", "Return #t iff merged size equals sum of input sizes.", "(let* ([a (make-block 'entity (string->utf8 \"a\") (vector))] [b (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-block 'entity (string->utf8 \"c\") (vector))] [c1 (make-collection-from-blocks 'group \"x\" (list a b))] [c2 (make-collection-from-blocks 'group \"y\" (list c))]) (= (collection-size (collection-merge c1 c2)) (+ (collection-size c1) (collection-size c2))))", "(equal? (let* ([a (make-block 'entity (string->utf8 \"a\") (vector))] [b (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-block 'entity (string->utf8 \"c\") (vector))] [c1 (make-collection-from-blocks 'group \"x\" (list a b))] [c2 (make-collection-from-blocks 'group \"y\" (list c))]) (= (collection-size (collection-merge c1 c2)) (+ (collection-size c1) (collection-size c2)))) #t)", "medium", ["property"]),

    # Integration/fold
    ("foldr", "Map doubling over '(1 2 3) via foldr and cons.", "(foldr (lambda (x acc) (cons (* 2 x) acc)) '() '(1 2 3))", "(equal? (foldr (lambda (x acc) (cons (* 2 x) acc)) '() '(1 2 3)) '(2 4 6))", "medium", ["fold"]),
    ("collection-hashes", "Build a collection from three blocks, then return hashes count via foldr over collection-hashes.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [b3 (make-block 'entity (string->utf8 \"c\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2 b3))]) (foldr (lambda (h acc) (+ acc 1)) 0 (collection-hashes c)))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [b3 (make-block 'entity (string->utf8 \"c\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1 b2 b3))]) (foldr (lambda (h acc) (+ acc 1)) 0 (collection-hashes c))) 3)", "hard", ["integration"]),
    ("collection-size", "Use named-let to add two members and return final collection-size.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [b3 (make-block 'entity (string->utf8 \"c\") (vector))] [c0 (make-collection-from-blocks 'group \"demo\" (list b1))]) (let loop ([members (list b2 b3)] [acc c0]) (if (null? members) (collection-size acc) (loop (cdr members) (collection-add acc (car members))))))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [b3 (make-block 'entity (string->utf8 \"c\") (vector))] [c0 (make-collection-from-blocks 'group \"demo\" (list b1))]) (let loop ([members (list b2 b3)] [acc c0]) (if (null? members) (collection-size acc) (loop (cdr members) (collection-add acc (car members)))))) 3)", "hard", ["loop"]),
    ("collection-empty?", "Add one block to an empty collection and test emptiness.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [c0 (make-collection-from-blocks 'group \"e\" '())] [c1 (collection-add c0 b1)]) (collection-empty? c1))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [c0 (make-collection-from-blocks 'group \"e\" '())] [c1 (collection-add c0 b1)]) (collection-empty? c1)) #f)", "medium", ["integration"]),
    ("collection-add", "Add then remove same member and compare resulting size to original.", "(let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-remove (collection-add c b2) (hash-block b2))]) (= (collection-size c2) (collection-size c)))", "(equal? (let* ([b1 (make-block 'entity (string->utf8 \"a\") (vector))] [b2 (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-collection-from-blocks 'group \"demo\" (list b1))] [c2 (collection-remove (collection-add c b2) (hash-block b2))]) (= (collection-size c2) (collection-size c))) #t)", "hard", ["integration"]),
    ("collection-remove", "Remove all duplicate hashes from a manually-crafted refs vector.", "(let* ([h 42] [c (make-block 'collection (string->utf8 \"dup\") (vector h h 7))] [c2 (collection-remove c h)]) (collection-hashes c2))", "(equal? (let* ([h 42] [c (make-block 'collection (string->utf8 \"dup\") (vector h h 7))] [c2 (collection-remove c h)]) (collection-hashes c2)) '(7))", "medium", ["integration"]),
    ("collection-merge", "Merge, then prepend one member with collection-add, then report size.", "(let* ([a (make-block 'entity (string->utf8 \"a\") (vector))] [b (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-block 'entity (string->utf8 \"c\") (vector))] [c1 (make-collection-from-blocks 'group \"x\" (list a))] [c2 (make-collection-from-blocks 'group \"y\" (list b))] [m (collection-merge c1 c2)] [m2 (collection-add m c)]) (collection-size m2))", "(equal? (let* ([a (make-block 'entity (string->utf8 \"a\") (vector))] [b (make-block 'entity (string->utf8 \"b\") (vector))] [c (make-block 'entity (string->utf8 \"c\") (vector))] [c1 (make-collection-from-blocks 'group \"x\" (list a))] [c2 (make-collection-from-blocks 'group \"y\" (list b))] [m (collection-merge c1 c2)] [m2 (collection-add m c)]) (collection-size m2)) 3)", "hard", ["integration"]),
    ("collection-merge", "Merge two empty collections and test collection-empty?.", "(let* ([c1 (make-collection-from-blocks 'group \"x\" '())] [c2 (make-collection-from-blocks 'group \"y\" '())] [m (collection-merge c1 c2)]) (collection-empty? m))", "(equal? (let* ([c1 (make-collection-from-blocks 'group \"x\" '())] [c2 (make-collection-from-blocks 'group \"y\" '())] [m (collection-merge c1 c2)]) (collection-empty? m)) #t)", "medium", ["edge-case"]),
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
            removable.sort(key=lambda r: (fn_counts[str(r["source_function"])] , str(r["id"])), reverse=True)
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
