#!/usr/bin/env python3
"""Generate Tier-0 geometry SFT samples for lattice/geometry/geometry.ss."""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_ROOT.parent
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"
PRE_DIVERSIFY_PATH = OUT_DIR / ".pre_diversify.jsonl"
SFT_GENERATOR_PATH = REPO_ROOT / "user" / "sft" / "generate.ss"
MAPS_PATH = OUT_DIR / "_bootstrap_geometry_maps.json"

SOURCE_MODULE = "lattice/geometry/geometry.ss"
SOURCE_TEST = "lattice/geometry/test-geometry.ss"
SOURCE_PATH = REPO_ROOT / SOURCE_MODULE

TOTAL_BUGFIX_TARGET = 100
TOTAL_COMPOSITION_TARGET = 150

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


def load_bootstrap_maps() -> Tuple[List[str], Dict[str, str], Dict[str, Dict[str, str]], Dict[str, str]]:
    if not MAPS_PATH.exists():
        raise FileNotFoundError(f"missing bootstrap maps: {MAPS_PATH}")
    data = json.loads(MAPS_PATH.read_text(encoding="utf-8"))
    function_order = list(data["function_order"])
    difficulty = {str(k): str(v) for k, v in data["difficulty"].items()}
    function_specs = {
        str(k): {str(sk): str(sv) for sk, sv in v.items()} for k, v in data["function_specs"].items()
    }
    verify_by_function = {str(k): str(v) for k, v in data["verify_by_function"].items()}
    return function_order, difficulty, function_specs, verify_by_function


FUNCTION_ORDER, DIFFICULTY, FUNCTION_SPECS, VERIFY_BY_FUNCTION = load_bootstrap_maps()

# Normalize a few legacy verifiers that depended on vec2 constructors not
# available in the default geometry validation load path.
VERIFY_BY_FUNCTION.update(
    {
        "circle": "(let* ([c (vec3 1 2 0)] [r 5] [ci (circle c r)]) (and (circle? ci) (equal? (circle-center ci) c) (= (circle-radius ci) r)))",
        "circle?": "(and (circle? (circle (vec3 0 0 0) 1)) (not (circle? '(not-a-circle))) (not (circle? 42)))",
        "circle-center": "(equal? (circle-center (circle (vec3 3 4 0) 10)) (vec3 3 4 0))",
        "circle-radius": "(= (circle-radius (circle (vec3 0 0 0) 7)) 7)",
    }
)


def read_source_module() -> str:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"source module missing: {SOURCE_PATH}")
    return SOURCE_PATH.read_text(encoding="utf-8")


MODULE_TEXT = read_source_module()


def extract_define(module_text: str, fn: str) -> str:
    patterns = [
        re.compile(rf"\(define\s+\({re.escape(fn)}(?:\s|\))"),
        re.compile(rf"\(define\s+{re.escape(fn)}(?:\s|\))"),
    ]
    m = None
    for pat in patterns:
        m = pat.search(module_text)
        if m:
            break
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


DEFS: Dict[str, str] = {fn: extract_define(MODULE_TEXT, fn) for fn in FUNCTION_ORDER}
DOC_FREE_DEFS: Dict[str, str] = {fn: strip_doc_forms(code) for fn, code in DEFS.items()}


def parse_signature_args(defn: str, fn: str) -> List[str]:
    m = re.match(r"\(define\s+\(([^)]*)\)", defn.strip())
    if not m:
        return []
    sig = m.group(1).strip()
    if not sig:
        return []
    tokens = sig.split()
    if not tokens or tokens[0] != fn:
        return []
    return tokens[1:]


def parse_signature_form(defn: str, fn: str) -> str:
    args = parse_signature_args(defn, fn)
    if args:
        return f"({fn} {' '.join(args)})"
    return f"{fn}"


def quote_scheme_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def python_snippet_for(fn: str, args: List[str], desc: str) -> str:
    if not args:
        return f"# {desc}\n{fn} = None"
    py_args = ", ".join(a.replace("-", "_").replace("?", "_p") for a in args)
    return (
        f"def {fn.replace('-', '_').replace('?', '_p')}({py_args}):\n"
        f"    \"\"\"{desc}\"\"\"\n"
        "    raise NotImplementedError"
    )


def _rename_internals(body: str, suffix: str = "-val") -> str:
    """Rename internal let/let* bound variables by adding a suffix.

    Only renames bindings introduced by let/let* — not top-level args or
    standard Fold library names.
    """
    binding_pat = re.compile(r'\[\s*([a-z][a-z0-9_-]*)\s+', re.IGNORECASE)
    names_in_bindings: set[str] = set()
    for m in binding_pat.finditer(body):
        name = m.group(1)
        # Skip names that look like standard library calls / accessors.
        if name.startswith(("vec3-", "matrix-", "aabb-", "sphere-", "ray3-",
                            "plane3-", "quat-", "line3-", "triangle3-",
                            "transform-", "distance-", "intersect-",
                            "closest-", "point-", "else")):
            continue
        names_in_bindings.add(name)

    if not names_in_bindings:
        return body

    # Sort longest-first to avoid partial-match replacement issues.
    for name in sorted(names_in_bindings, key=len, reverse=True):
        new_name = name + suffix
        # Word-boundary-aware replacement: preceded and followed by delimiters.
        body = re.sub(
            r'(?<=[(\[\s,])' + re.escape(name) + r'(?=[)\]\s,])',
            new_name,
            body,
        )
    return body


def _let_star_to_nested_lets(body: str) -> str:
    """Convert the outermost let* to nested let forms.

    Splits (let* ([a ...] [b ...] [c ...]) body) into
    (let ([a ...]) (let ([b ...]) (let ([c ...]) body))).
    """
    stripped = body.strip()
    if not stripped.startswith("(let* ("):
        return body

    # Find the bindings list opening paren.
    bindings_start = stripped.index("(", 5)  # The ( after "let* "
    # Parse individual [name expr] bindings.
    i = bindings_start + 1
    bindings: list[str] = []
    while i < len(stripped):
        while i < len(stripped) and stripped[i].isspace():
            i += 1
        if i >= len(stripped) or stripped[i] == ")":
            i += 1
            break
        if stripped[i] == "[":
            depth = 0
            start = i
            in_str = False
            esc = False
            while i < len(stripped):
                c = stripped[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c in "[(" :
                    depth += 1
                elif c in "])":
                    depth -= 1
                    if depth == 0:
                        bindings.append(stripped[start : i + 1])
                        i += 1
                        break
                i += 1
        else:
            i += 1

    if len(bindings) < 2:
        return body

    # Everything after the bindings list up to matching close paren is the body.
    rest_start = i
    while rest_start < len(stripped) and stripped[rest_start].isspace():
        rest_start += 1
    depth = 1
    j = rest_start
    in_str = False
    esc = False
    in_comment = False
    while j < len(stripped) and depth > 0:
        c = stripped[j]
        if in_comment:
            if c == "\n":
                in_comment = False
        elif in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == ";":
                in_comment = True
            elif c == '"':
                in_str = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
        j += 1
    inner_body = stripped[rest_start : j - 1].strip()

    # Build nested lets from inside out.
    result = inner_body
    for binding in reversed(bindings):
        result = f"(let ({binding})\n         {result})"
    return result


def _if_to_cond(body: str) -> str:
    """Convert a top-level (if test then else) to (cond [test then] [else alt])."""
    stripped = body.strip()
    if not stripped.startswith("(if "):
        return body

    i = 4
    parts: list[str] = []
    for _ in range(3):
        while i < len(stripped) and stripped[i].isspace():
            i += 1
        if i >= len(stripped):
            return body
        start = i
        if stripped[i] == "(":
            depth = 0
            in_str = False
            esc = False
            while i < len(stripped):
                c = stripped[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            parts.append(stripped[start:i])
        else:
            while i < len(stripped) and stripped[i] not in " \t\n\r)":
                i += 1
            parts.append(stripped[start:i])

    if len(parts) != 3:
        return body

    test, then, els = parts
    return f"(cond\n      [{test}\n       {then}]\n      [else\n       {els}])"


# Functions whose lambda-wrap snippets are >0.97 similarity to the original.
_HIGH_SIMILARITY_FNS = {
    "transform-rotation-axis",
    "transform-from-quaternion",
    "transform-point",
    "intersect-ray-sphere",
    "intersect-ray-aabb",
    "barycentric-coords",
}


def chez_snippet_for(fn: str, defn: str) -> str:
    src = strip_doc_forms(defn)

    # Prefer a lambda-style form to avoid trivial near-identity translation pairs.
    if src.startswith(f"(define ({fn} ") or src.startswith(f"(define ({fn})"):
        header_start = len("(define (")
        header_end = src.find(")", header_start)
        if header_end > header_start:
            sig = src[header_start:header_end].strip()
            tokens = sig.split()
            args = tokens[1:] if len(tokens) >= 1 else []
            body = src[header_end + 1 :].strip()
            if body.endswith(")"):
                body = body[:-1].rstrip()

            # Extra structural transforms for high-similarity functions:
            # rename internal bindings, decompose let* chains, convert if->cond.
            if fn in _HIGH_SIMILARITY_FNS:
                body = _rename_internals(body)
                body = _let_star_to_nested_lets(body)
                body = _if_to_cond(body)

            body_lines = body.splitlines() or ["<missing-body>"]
            indented_body = "\n".join("    " + line for line in body_lines)
            return (
                f"(define legacy-{fn}\n"
                f"  (lambda ({' '.join(args)})\n"
                f"{indented_body}\n"
                "  ))"
            )

    if src.startswith(f"(define {fn} "):
        return src.replace(f"(define {fn} ", f"(define legacy-{fn} ", 1)
    return src


def make_source_excerpt(fn: str, snippet: str) -> str:
    indented = "\n".join(f"  {line}" for line in snippet.splitlines())
    return (
        ";;; lattice/geometry/geometry.ss excerpt\n"
        "(require 'vec3)\n"
        "(require 'matrix)\n"
        "\n"
        "(doc 'module 'geometry)\n"
        "(doc 'layer 'lattice)\n"
        "\n"
        "(define (legacy-helper x) x)\n"
        "\n"
        f"{indented}\n"
    )


def validate_verifier_map() -> None:
    missing = [fn for fn in FUNCTION_ORDER if fn not in VERIFY_BY_FUNCTION]
    if missing:
        raise ValueError(f"missing verify_expr for functions: {missing}")


validate_verifier_map()


MUTATION_RULES: List[Tuple[str, str, str]] = [
    ("cadr", "caddr", "Accessor index shifted from second slot to third slot."),
    ("caddr", "cadr", "Accessor index shifted from third slot to second slot."),
    ("cadddr", "caddr", "Accessor index shifted from fourth slot to third slot."),
    ("vec3-add", "vec3-sub", "Vector combination operator uses subtraction instead of addition."),
    ("vec3-sub", "vec3-add", "Vector difference operator uses addition instead of subtraction."),
    ("<=", "<", "Boundary check became strict (<) and drops edge inclusions."),
    (">=", ">", "Boundary check became strict (>) and drops edge inclusions."),
    (" 0.5", " 1.0", "Midpoint/half-scale factor was changed from 0.5 to 1.0."),
    (" 1.0", " 0.0", "Important numeric constant changed to 0.0."),
]


def replace_once(text: str, old: str, new: str) -> str | None:
    i = text.find(old)
    if i < 0:
        return None
    return text[:i] + new + text[i + len(old) :]


def fallback_bug_stub(fn: str, args: List[str], mode: int) -> Tuple[str, str]:
    if args:
        if mode == 0:
            return (
                f"(define ({fn} {' '.join(args)})\n  (error '{fn} \"intentional bug\"))",
                "Function raises an error instead of implementing required behavior.",
            )
        first = args[0]
        return (
            f"(define ({fn} {' '.join(args)})\n  {first})",
            "Function incorrectly returns the first argument unchanged.",
        )
    if mode == 0:
        return (
            f"(define {fn}\n  #f)",
            "Constant/function alias replaced with #f.",
        )
    return (
        f"(define {fn}\n  (lambda args (error '{fn} \"intentional bug\")))",
        "Alias replaced with an erroring lambda.",
    )


def bug_variants_for_function(fn: str) -> List[Tuple[str, str]]:
    base = DOC_FREE_DEFS[fn]
    args = parse_signature_args(base, fn)
    variants: List[Tuple[str, str]] = []

    for old, new, note in MUTATION_RULES:
        candidate = replace_once(base, old, new)
        if candidate and candidate != base:
            variants.append((candidate, note))

    # Tag mutation for tagged records/predicates
    list_tag = re.search(r"\(list\s+'([A-Za-z0-9-]+)", base)
    if list_tag:
        tag = list_tag.group(1)
        candidate = replace_once(base, f"'{tag}", f"'{tag}-broken")
        if candidate and candidate != base:
            variants.append((candidate, "Tag symbol was changed, breaking representation compatibility."))

    eq_tag = re.search(r"\(eq\?\s+\(car\s+[^)]*\)\s+'([A-Za-z0-9-]+)\)", base)
    if eq_tag:
        tag = eq_tag.group(1)
        candidate = replace_once(base, f"'{tag}", f"'{tag}-broken")
        if candidate and candidate != base:
            variants.append((candidate, "Predicate checks the wrong tag symbol."))

    variants.append(fallback_bug_stub(fn, args, 0))
    variants.append(fallback_bug_stub(fn, args, 1))

    uniq: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for code, note in variants:
        k = code.strip()
        if k not in seen:
            seen.add(k)
            uniq.append((k, note))
    return uniq


def candidate_is_semantic_bug(fn: str, buggy_code: str) -> bool:
    verify_expr = VERIFY_BY_FUNCTION[fn]
    script = "\n".join(
        [
            '(load "core/lang/module.ss")',
            '(load "lattice/geometry/geometry.ss")',
            "(define result",
            "  (guard (ex [else 'exception])",
            "    (begin",
            buggy_code,
            verify_expr,
            "    )))",
            "(if (equal? result #t)",
            '    (begin (display "BAD\n") (exit 1))',
            '    (begin (display "GOOD\n")))',
        ]
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".ss", delete=False) as tf:
        tf.write(script)
        path = Path(tf.name)
    try:
        proc = subprocess.run(
            ["scheme", "--quiet", "--script", str(path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        # Only accept syntactically valid candidates that are not passing the verifier.
        return proc.returncode == 0 and "GOOD" in proc.stdout
    finally:
        path.unlink(missing_ok=True)


def build_bug_cases(target_total: int) -> List[Dict[str, str]]:
    cases: List[Dict[str, str]] = []
    used_by_fn: Dict[str, set[str]] = defaultdict(set)

    # One high-confidence bug per function first.
    for fn in FUNCTION_ORDER:
        chosen = None
        for buggy, note in bug_variants_for_function(fn):
            if buggy in used_by_fn[fn]:
                continue
            if candidate_is_semantic_bug(fn, buggy):
                chosen = {"fn": fn, "buggy": buggy, "note": note}
                break
        if chosen is None:
            fallback_code, fallback_note = fallback_bug_stub(fn, parse_signature_args(DOC_FREE_DEFS[fn], fn), 0)
            chosen = {"fn": fn, "buggy": fallback_code, "note": fallback_note}
        used_by_fn[fn].add(chosen["buggy"])
        cases.append(chosen)

    # Add extra bugfixes until target_total, prioritizing non-easy functions.
    priority = [fn for fn in FUNCTION_ORDER if DIFFICULTY.get(fn, "medium") != "easy"] + FUNCTION_ORDER
    i = 0
    while len(cases) < target_total and i < len(priority):
        fn = priority[i]
        i += 1
        chosen = None
        for buggy, note in bug_variants_for_function(fn):
            if buggy in used_by_fn[fn]:
                continue
            if candidate_is_semantic_bug(fn, buggy):
                chosen = {"fn": fn, "buggy": buggy, "note": note}
                break
        if chosen is None:
            fallback_code, fallback_note = fallback_bug_stub(fn, parse_signature_args(DOC_FREE_DEFS[fn], fn), 1)
            if fallback_code in used_by_fn[fn]:
                continue
            chosen = {"fn": fn, "buggy": fallback_code, "note": fallback_note}
        used_by_fn[fn].add(chosen["buggy"])
        cases.append(chosen)

    if len(cases) < target_total:
        raise ValueError(f"unable to synthesize enough bug cases: {len(cases)} < {target_total}")

    return cases[:target_total]


def build_composition_cases(target_total: int) -> List[Dict[str, str]]:
    cases: List[Dict[str, str]] = []
    for fn in FUNCTION_ORDER:
        base_verify = VERIFY_BY_FUNCTION[fn]

        cases.append(
            {
                "fn": fn,
                "prompt": f"Compose geometry APIs to check canonical behavior for `{fn}` on representative inputs.",
                "solution": base_verify,
                "verify": f"(equal? {base_verify} #t)",
                "difficulty": DIFFICULTY.get(fn, "medium"),
                "tags": ["behavior-check", "auto-composition"],
            }
        )

        cases.append(
            {
                "fn": fn,
                "prompt": f"Build an expression using `{fn}` that returns `'ok` when expected behavior holds.",
                "solution": f"(if {base_verify} 'ok 'bad)",
                "verify": f"(equal? (if {base_verify} 'ok 'bad) 'ok)",
                "difficulty": DIFFICULTY.get(fn, "medium"),
                "tags": ["behavior-signal", "auto-composition"],
            }
        )

    # Two extra explicit geometry-integration compositions to hit 150 exactly.
    cases.append(
        {
            "fn": "intersect-ray-aabb",
            "prompt": "Cast a +Z ray from z=-10 into an AABB and return the entry distance.",
            "solution": "(car (intersect-ray-aabb (ray3 (vec3 0 0 -10) (vec3 0 0 1)) (aabb (vec3 -1 -1 -1) (vec3 1 1 1))))",
            "verify": "(< (abs (- (car (intersect-ray-aabb (ray3 (vec3 0 0 -10) (vec3 0 0 1)) (aabb (vec3 -1 -1 -1) (vec3 1 1 1)))) 9.0)) 0.001)",
            "difficulty": "hard",
            "tags": ["ray-box", "entry-distance"],
        }
    )
    cases.append(
        {
            "fn": "triangle-area",
            "prompt": "Create a 3-4-5 right triangle in the XY plane and return its area.",
            "solution": "(triangle-area (triangle3 (vec3 0 0 0) (vec3 3 0 0) (vec3 0 4 0)))",
            "verify": "(< (abs (- (triangle-area (triangle3 (vec3 0 0 0) (vec3 3 0 0) (vec3 0 4 0))) 6.0)) 0.001)",
            "difficulty": "medium",
            "tags": ["triangle", "area"],
        }
    )

    if len(cases) < target_total:
        raise ValueError(f"insufficient composition cases: {len(cases)} < {target_total}")
    return cases[:target_total]


samples: List[Dict[str, object]] = []
family_counter: Dict[str, int] = defaultdict(int)


def add_sample(
    family: str,
    category: str,
    difficulty: str,
    source_function: str,
    prompt_body: str,
    ground_truth: str,
    verify_expr: str,
    tags: List[str],
) -> None:
    family_counter[family] += 1
    sid = f"geometry_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
        "prompt_body": prompt_body.strip(),
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


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (3 variants per function = 222)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    spec = FUNCTION_SPECS[fn]
    signature = parse_signature_form(DEFS[fn], fn)
    args = parse_signature_args(DEFS[fn], fn)

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt_body=f"""Implement this geometry function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Signature: `{signature}`
Description: {spec['desc']}

Write exactly one definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "geometry", "spec-to-code", fn],
    )

    if args:
        skeleton = f"(define ({fn} {' '.join(args)})\n  ;; TODO: implement\n  <TODO>)"
    else:
        skeleton = f"(define {fn}\n  ;; TODO: implement\n  <TODO>)"

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt_body=f"""Complete this Fold Scheme skeleton.

```scheme
{skeleton}
```

Replace `<TODO>` and return only the completed definition for `{fn}`.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "geometry", "skeleton-completion", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt_body=f"""Implement this geometry function without doc metadata.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Signature: `{signature}`

Return only executable Scheme code for `{fn}` without `(doc ...)` forms.""",
        ground_truth=DOC_FREE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "geometry", "doc-free", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (3 variants per function = 222)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    spec = FUNCTION_SPECS[fn]
    args = parse_signature_args(DEFS[fn], fn)
    py = python_snippet_for(fn, args, spec["desc"])
    chez = chez_snippet_for(fn, DEFS[fn])

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt_body=f"""Translate this Python-like snippet into Fold-native Scheme.
Preserve behavior and use the target name `{fn}`.
Return only the Scheme definition.

```python
{py}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "geometry", "python-to-scheme", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt_body=f"""Convert this Chez-style snippet to canonical Fold style.
The target function must be named `{fn}`.
Return only the final Fold definition.

```scheme
{chez}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "geometry", "chez-to-fold", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt_body=f"""Extract and normalize this geometry source excerpt.
Return only one Fold definition for `{fn}`.
Drop metadata doc forms and preserve behavior.

```scheme
{make_source_excerpt(fn, chez)}
```""",
        ground_truth=DOC_FREE_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "geometry", "source-excerpt-to-fold", "doc-free-target", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (100)
# -----------------------------------------------------------------------------
BUGGY_CASES = build_bug_cases(TOTAL_BUGFIX_TARGET)

for case in BUGGY_CASES:
    fn = case["fn"]
    add_sample(
        family="bugfix",
        category="debugging",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt_body=f"""Fix the bug in this Fold Scheme function with minimal semantic changes.
Target: `{fn}` in `{SOURCE_MODULE}`.
Known issue: {case['note']}

```scheme
{case['buggy']}
```

Return only the corrected definition.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier0", "geometry", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition (150)
# -----------------------------------------------------------------------------
COMPOSITION_CASES = build_composition_cases(TOTAL_COMPOSITION_TARGET)

for case in COMPOSITION_CASES:
    fn = str(case["fn"])
    add_sample(
        family="composition",
        category="usage",
        difficulty=str(case["difficulty"]),
        source_function=fn,
        prompt_body=f"""Compose geometry functions to solve this task.

{case['prompt']}

Ensure `{fn}` is used in the expression.
Return only one executable Scheme expression.""",
        ground_truth=str(case["solution"]),
        verify_expr=str(case["verify"]),
        tags=["tier0", "geometry", "composition", fn] + list(case.get("tags", [])),
    )


# -----------------------------------------------------------------------------
# Split + DSL diversification + write artifacts
# -----------------------------------------------------------------------------
expected_family_counts = {
    "spec_to_code": len(FUNCTION_ORDER) * 3,
    "translation": len(FUNCTION_ORDER) * 3,
    "bugfix": TOTAL_BUGFIX_TARGET,
    "composition": TOTAL_COMPOSITION_TARGET,
}

for family, expected_count in expected_family_counts.items():
    actual = sum(1 for s in samples if s["family"] == family)
    if actual != expected_count:
        raise ValueError(f"{family} must contain {expected_count} samples, found {actual}")

if len(samples) != sum(expected_family_counts.values()):
    raise ValueError(f"unexpected total samples: {len(samples)}")

EVAL_RATIO = 0.18
all_source_functions = sorted({str(s["source_function"]) for s in samples})


def stable_hash_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def choose_eval_functions(functions: List[str], eval_ratio: float) -> set[str]:
    target_fn_count = max(1, round(len(functions) * eval_ratio))
    by_difficulty: Dict[str, List[str]] = defaultdict(list)
    for fn in functions:
        by_difficulty[DIFFICULTY.get(fn, "medium")].append(fn)

    for fn_list in by_difficulty.values():
        fn_list.sort(key=stable_hash_int)

    bucket_count = len(by_difficulty)
    min_quota = 1 if target_fn_count >= bucket_count else 0

    quotas: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    total = 0
    for diff, fn_list in by_difficulty.items():
        raw = len(fn_list) * eval_ratio
        quota = int(raw)
        if quota < min_quota:
            quota = min_quota
        quota = min(quota, len(fn_list))
        quotas[diff] = quota
        total += quota
        remainders.append((raw - quota, diff))

    if total < target_fn_count:
        for _, diff in sorted(remainders, key=lambda x: (x[0], x[1]), reverse=True):
            while total < target_fn_count and quotas[diff] < len(by_difficulty[diff]):
                quotas[diff] += 1
                total += 1
    elif total > target_fn_count:
        for _, diff in sorted(remainders, key=lambda x: (x[0], x[1])):
            while total > target_fn_count and quotas[diff] > min_quota:
                quotas[diff] -= 1
                total -= 1

    selected: List[str] = []
    for diff in sorted(by_difficulty.keys()):
        selected.extend(by_difficulty[diff][: quotas[diff]])
    selected = sorted(set(selected), key=stable_hash_int)

    if len(selected) < target_fn_count:
        remaining = [fn for fn in sorted(functions, key=stable_hash_int) if fn not in selected]
        selected.extend(remaining[: target_fn_count - len(selected)])
    elif len(selected) > target_fn_count:
        selected = selected[:target_fn_count]

    return set(selected)


eval_functions = choose_eval_functions(all_source_functions, EVAL_RATIO)
eval_ids = {str(s["id"]) for s in samples if str(s["source_function"]) in eval_functions}



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
    "source_functions": len({str(r["source_function"]) for r in all_rows}),
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
