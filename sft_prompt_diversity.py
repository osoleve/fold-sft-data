#!/usr/bin/env python3
"""Deterministic prompt diversification helpers for SFT generators."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Sequence


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
CALL_HEAD_RE = re.compile(r"\(\s*([^\s()]+)")
CALLABLE_HEAD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-+*/<>=!?._:]*$")
SPECIAL_FORMS = {
    "begin",
    "call-with-values",
    "case",
    "cond",
    "define",
    "define-record-type",
    "define-syntax",
    "do",
    "else",
    "if",
    "lambda",
    "let",
    "let*",
    "let-values",
    "letrec",
    "letrec*",
    "or",
    "and",
    "quote",
    "quasiquote",
    "set!",
    "unquote",
}


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


def _parse_top_level_let(expr: str):
    expr = expr.strip()
    if not expr.startswith("("):
        return None
    close = _find_matching_paren(expr, 0)
    if close != len(expr) - 1:
        return None

    inner = expr[1:close].strip()
    parts = _split_top_level_forms(inner)
    if not parts or parts[0] not in LET_FORMS:
        return None
    if len(parts) < 3:
        return None

    form = parts[0]
    if parts[1].startswith("("):
        name = ""
        bindings = parts[1]
        body = parts[2:]
        return form, name, bindings, body

    if len(parts) >= 4 and parts[2].startswith("("):
        name = parts[1]
        bindings = parts[2]
        body = parts[3:]
        return form, name, bindings, body

    return None


def _normalize_check(expr: str) -> str:
    flat = re.sub(r"\s+", " ", expr).strip()
    return flat


def _extract_checks(verify_expr: str, max_checks: int = 2, max_check_len: int = 420) -> List[str]:
    if not verify_expr.strip():
        return []

    let_ctx = _parse_top_level_let(verify_expr)
    if let_ctx:
        form, name, bindings, body = let_ctx
        core = body[-1].strip() if body else verify_expr.strip()
    else:
        core = verify_expr.strip()

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
        if let_ctx:
            if name:
                cand = f"({form} {name} {bindings} {cand})"
            else:
                cand = f"({form} {bindings} {cand})"
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


def _extract_known_issue(prompt: str) -> str:
    match = re.search(r"Known issue:\s*(.+)", prompt)
    if not match:
        return ""
    issue = " ".join(match.group(1).split())
    if len(issue) > 220:
        return issue[:217].rstrip() + "..."
    return issue


def _extract_call_heads(expr: str) -> List[str]:
    seen = set()
    names: List[str] = []
    for match in CALL_HEAD_RE.finditer(expr):
        head = match.group(1).strip()
        head = head.lstrip("',`")
        if not head:
            continue
        if head[0] in {'"', "[", "]"}:
            continue
        if head.startswith("#"):
            continue
        if head in SPECIAL_FORMS:
            continue
        if not CALLABLE_HEAD_RE.fullmatch(head):
            continue
        if not any(ch.isalpha() for ch in head):
            continue
        if len(head) == 1 and head.isalpha():
            continue
        if head in seen:
            continue
        seen.add(head)
        names.append(head)
    return names


def _infer_composition_functions(
    source_function: str,
    ground_truth: str,
    available_functions: Sequence[str] | None,
    max_items: int = 5,
) -> List[str]:
    calls = _extract_call_heads(ground_truth)
    selected: List[str] = []
    seen = set()

    def push(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            selected.append(name)

    push(source_function)

    if available_functions:
        allowed = set(available_functions)
        for name in calls:
            if name in allowed:
                push(name)
            if len(selected) >= max_items:
                break
    else:
        for name in calls:
            push(name)
            if len(selected) >= max_items:
                break

    return selected[:max_items]


def _build_bugfix_overlay(key: str, prompt: str, verify_expr: str) -> str:
    checks = _extract_checks(verify_expr, max_checks=1, max_check_len=420)
    if not checks:
        return ""
    if _stable_index(f"{key}|bugfix_overlay", 2) == 0:
        return ""

    issue = _extract_known_issue(prompt)
    issue_line = (
        f"Bug report summary: {issue}"
        if issue
        else "Bug report summary: the current implementation violates required behavior."
    )
    return (
        f"{issue_line}\n\n"
        + _format_checks_block("Expected behavior after patch", checks)
        + "\n\nActual behavior: the provided implementation fails the expectation above."
    )


def _build_composition_overlay(
    key: str,
    source_function: str,
    ground_truth: str,
    available_functions: Sequence[str] | None,
) -> str:
    if _stable_index(f"{key}|composition_overlay", 2) == 0:
        return ""

    funcs = _infer_composition_functions(
        source_function=source_function,
        ground_truth=ground_truth,
        available_functions=available_functions,
        max_items=5,
    )
    if len(funcs) < 2:
        return ""

    lines = "\n".join(f"- `{name}`" for name in funcs)
    return (
        "Available functions you may compose:\n"
        f"{lines}\n\n"
        "Prefer using this API inventory directly instead of re-implementing behavior."
    )


def _build_family_overlay(
    prompt: str,
    family: str,
    key: str,
    source_function: str,
    verify_expr: str,
    ground_truth: str,
    available_functions: Sequence[str] | None,
) -> str:
    if family == "bugfix":
        return _build_bugfix_overlay(key, prompt, verify_expr)
    if family == "composition":
        return _build_composition_overlay(key, source_function, ground_truth, available_functions)
    return ""


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
    ground_truth: str = "",
    available_functions: Sequence[str] | None = None,
) -> str:
    """Add deterministic wording/task-frame diversity while preserving semantics."""
    text = prompt.strip()
    key = f"{family}|{category}|{source_function}|{sample_index}"

    prefix = _pick(FAMILY_PREFIXES.get(family, []), f"{key}|prefix")
    if prefix and not text.startswith(prefix):
        text = f"{prefix}\n\n{text}"

    hint = _pick(CATEGORY_HINTS.get(category, []), f"{key}|hint")
    text = _append_if_missing(text, hint)

    family_overlay = _build_family_overlay(
        prompt=text,
        family=family,
        key=key,
        source_function=source_function,
        verify_expr=verify_expr,
        ground_truth=ground_truth,
        available_functions=available_functions,
    )
    text = _append_if_missing(text, family_overlay)

    task_type_frame = _build_task_type_frame(family, key, verify_expr)
    text = _append_if_missing(text, task_type_frame)

    suffix = _pick(FAMILY_SUFFIXES.get(family, []), f"{key}|suffix")
    text = _append_if_missing(text, suffix)

    return text
