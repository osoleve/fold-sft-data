#!/usr/bin/env python3
"""Deterministic prompt diversification helpers for SFT generators."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List


FAMILY_PREFIXES: Dict[str, List[str]] = {
    "spec_to_code": [
        "Task mode: implement the API contract in canonical Fold Scheme.",
        "Task mode: behavior-first function implementation.",
        "Task mode: complete the target function to match module semantics.",
    ],
    "translation": [
        "Task mode: semantic translation into idiomatic Fold Scheme.",
        "Task mode: preserve behavior while translating syntax and naming.",
        "Task mode: convert source-language logic to Fold-native form.",
    ],
    "bugfix": [
        "Task mode: minimal patch bug repair.",
        "Task mode: localize and fix the behavioral defect.",
        "Task mode: surgical correction with unchanged intended API.",
    ],
    "composition": [
        "Task mode: compose existing APIs into one expression.",
        "Task mode: small integration task across module primitives.",
        "Task mode: solve by expression synthesis over available functions.",
    ],
}

FAMILY_SUFFIXES: Dict[str, List[str]] = {
    "spec_to_code": [
        "Prioritize edge-case behavior when specified by the contract.",
        "Keep the implementation idiomatic and dependency-aware.",
        "Ensure the definition is production-ready for module integration.",
    ],
    "translation": [
        "Semantic equivalence is more important than token-level similarity.",
        "Preserve boundary conditions and error behavior from the source snippet.",
        "Prefer Fold conventions while keeping the same observable behavior.",
    ],
    "bugfix": [
        "Keep unrelated logic unchanged unless needed for correctness.",
        "Fix root-cause behavior rather than masking symptoms.",
        "Retain public contract while repairing the implementation fault.",
    ],
    "composition": [
        "Favor direct API use over ad-hoc reimplementation.",
        "Keep the answer as a concise executable expression.",
        "Use provided module operations to satisfy the requested behavior.",
    ],
}

CATEGORY_HINTS: Dict[str, List[str]] = {
    "implementation": [
        "Focus on correctness first; keep structure straightforward.",
        "Match the stated contract exactly, including edge cases.",
    ],
    "transpile": [
        "Translate semantics faithfully; adapt names to the requested target.",
        "Keep behavior exact while adopting Fold syntax.",
    ],
    "translation": [
        "Translate semantics faithfully; adapt names to the requested target.",
        "Keep behavior exact while adopting Fold syntax.",
    ],
    "repair": [
        "Apply the smallest coherent fix that restores expected behavior.",
        "Repair the defect without broad refactoring.",
    ],
    "debugging": [
        "Apply the smallest coherent fix that restores expected behavior.",
        "Repair the defect without broad refactoring.",
    ],
    "usage": [
        "Compose from existing module functions where appropriate.",
        "Solve with an expression that can be evaluated directly.",
    ],
}

LET_FORMS = {"let", "let*", "letrec", "letrec*"}


def _stable_index(key: str, size: int) -> int:
    if size <= 0:
        return 0
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def _pick(options: List[str], key: str) -> str:
    if not options:
        return ""
    return options[_stable_index(key, len(options))]


def _append_if_missing(text: str, addition: str) -> str:
    if not addition or addition in text:
        return text
    return f"{text}\n\n{addition}"


def _find_matching_paren(text: str, open_index: int) -> int:
    depth = 0
    in_string = False
    escape = False
    i = open_index
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _split_top_level_forms(text: str) -> List[str]:
    forms: List[str] = []
    n = len(text)
    i = 0
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        depth = 0
        in_string = False
        escape = False
        while i < n:
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue

            if ch == '"':
                in_string = True
                i += 1
                continue

            if ch == ";":
                while i < n and text[i] != "\n":
                    i += 1
                continue

            if ch == "(":
                depth += 1
                i += 1
                continue

            if ch == ")":
                if depth > 0:
                    depth -= 1
                    i += 1
                    if depth == 0:
                        while i < n and text[i].isspace():
                            i += 1
                        break
                    continue
                break

            if depth == 0 and ch.isspace():
                break

            i += 1

        form = text[start:i].strip()
        if form:
            forms.append(form)
        while i < n and text[i].isspace():
            i += 1
    return forms


def _extract_last_let_body_form(expr: str) -> str:
    expr = expr.strip()
    if not expr.startswith("("):
        return expr
    close = _find_matching_paren(expr, 0)
    if close != len(expr) - 1:
        return expr

    inner = expr[1:close].strip()
    parts = _split_top_level_forms(inner)
    if not parts:
        return expr
    if parts[0] not in LET_FORMS:
        return expr

    if len(parts) < 3:
        return expr

    # Support both (let (<bindings>) body...) and named-let variants.
    if parts[1].startswith("("):
        body = parts[2:]
    elif len(parts) >= 4 and parts[2].startswith("("):
        body = parts[3:]
    else:
        body = parts[2:]

    if not body:
        return expr
    return body[-1].strip()


def _normalize_check(expr: str) -> str:
    flat = re.sub(r"\s+", " ", expr).strip()
    return flat


def _extract_checks(verify_expr: str, max_checks: int = 2, max_check_len: int = 420) -> List[str]:
    if not verify_expr.strip():
        return []

    core = _extract_last_let_body_form(verify_expr)
    if not core.startswith("("):
        candidate = _normalize_check(core)
        return [candidate] if len(candidate) <= max_check_len else []

    close = _find_matching_paren(core, 0)
    if close != len(core) - 1:
        candidate = _normalize_check(core)
        return [candidate] if len(candidate) <= max_check_len else []

    inner = core[1:close].strip()
    parts = _split_top_level_forms(inner)
    if not parts:
        candidate = _normalize_check(core)
        return [candidate] if len(candidate) <= max_check_len else []

    op = parts[0]
    args = parts[1:]
    candidates = args if op in {"and", "or"} and args else [core]

    checks: List[str] = []
    seen = set()
    for cand in candidates:
        cand = cand.strip()
        if not cand or cand.startswith("(define"):
            continue
        normalized = _normalize_check(cand)
        if len(normalized) > max_check_len:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        checks.append(normalized)
        if len(checks) >= max_checks:
            break

    if checks:
        return checks
    candidate = _normalize_check(core)
    if len(candidate) <= max_check_len:
        return [candidate]
    return []


def _format_checks_block(title: str, checks: List[str]) -> str:
    if not checks:
        return ""
    lines = "\n".join(checks)
    return f"{title}:\n```scheme\n{lines}\n```"


def _build_task_type_frame(family: str, key: str, verify_expr: str) -> str:
    checks = _extract_checks(verify_expr)
    if not checks:
        return ""

    mode = _stable_index(f"{key}|task_type", 3)
    if mode == 0:
        return ""

    if family == "spec_to_code":
        if mode == 1:
            return _format_checks_block("Behavior examples your implementation must satisfy", checks)
        return _format_checks_block("Acceptance checks to pass", checks)

    if family == "translation":
        if mode == 1:
            return _format_checks_block("Preserve these observable behaviors in translation", checks)
        return _format_checks_block("Semantic checks for the translated function", checks)

    if family == "bugfix":
        if mode == 1:
            return _format_checks_block("Regression checks after fixing the bug", checks)
        return (
            "Keep the original function signature unchanged.\n\n"
            + _format_checks_block("Regression checks after fixing the bug", checks)
        )

    if family == "composition":
        if mode == 1:
            return _format_checks_block("Target properties for your expression", checks)
        return (
            "Expression-only output is required (no helper definitions).\n\n"
            + _format_checks_block("Target properties for your expression", checks)
        )

    return ""


def diversify_prompt(
    prompt: str,
    family: str,
    source_function: str,
    sample_index: int,
    category: str,
    verify_expr: str = "",
) -> str:
    """Add deterministic wording/task-frame diversity while preserving semantics."""
    text = prompt.strip()
    key = f"{family}|{category}|{source_function}|{sample_index}"

    prefix = _pick(FAMILY_PREFIXES.get(family, []), f"{key}|prefix")
    if prefix and not text.startswith(prefix):
        text = f"{prefix}\n\n{text}"

    hint = _pick(CATEGORY_HINTS.get(category, []), f"{key}|hint")
    text = _append_if_missing(text, hint)

    task_type_frame = _build_task_type_frame(family, key, verify_expr)
    text = _append_if_missing(text, task_type_frame)

    suffix = _pick(FAMILY_SUFFIXES.get(family, []), f"{key}|suffix")
    text = _append_if_missing(text, suffix)

    return text
