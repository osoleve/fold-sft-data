#!/usr/bin/env python3
"""Generate SFT samples for core/base/error.ss."""

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

SOURCE_MODULE = "core/base/error.ss"
SOURCE_TEST = "core/base/test-error.ss"

SUPPORT_DEFS: Dict[str, str] = {
    "*parse-errors*": """(define *parse-errors*
  '((unexpected-eof    . "Unexpected end of input")
    (unexpected-char   . "Unexpected character")
    (unclosed-string   . "Unclosed string literal")
    (unclosed-list     . "Unclosed list - missing )")
    (invalid-number    . "Invalid number format")
    (invalid-escape    . "Invalid escape sequence")))""",
    "*infer-errors*": """(define *infer-errors*
  '((unbound-variable  . "Variable is not defined")
    (type-mismatch     . "Types do not match")
    (arity-mismatch    . "Wrong number of arguments")
    (not-a-function    . "Attempting to call a non-function")
    (occurs-check      . "Infinite type detected")
    (unknown-primitive . "Unknown primitive operation")
    (if-test-not-bool  . "If condition must be boolean")))""",
    "*eval-errors*": """(define *eval-errors*
  '((unbound-variable   . "Variable is not defined")
    (invalid-expression . "Cannot evaluate this expression")
    (not-a-closure      . "Attempting to call a non-function")
    (invalid-arguments  . "Invalid arguments to function")
    (division-by-zero   . "Division by zero")
    (out-of-bounds      . "Index out of bounds")
    (type-error         . "Runtime type error")))""",
    "*block-errors*": """(define *block-errors*
  '((invalid-tag       . "Invalid block tag")
    (invalid-payload   . "Invalid block payload")
    (invalid-refs      . "Invalid block references")
    (hash-mismatch     . "Content hash does not match")
    (not-found         . "Block not found in store")))""",
}

TARGET_DEFS: Dict[str, str] = {
    "make-error": """(define (make-error phase code context . details)
  `(error ,phase ,code ,context ,@details))""",
    "error?": """(define (error? x)
  (and (pair? x)
       (eq? (car x) 'error)
       (>= (length x) 4)))""",
    "error-phase": """(define (error-phase err)
  (and (error? err) (cadr err)))""",
    "error-code": """(define (error-code err)
  (and (error? err) (caddr err)))""",
    "error-context": """(define (error-context err)
  (and (error? err) (cadddr err)))""",
    "error-details": """(define (error-details err)
  (and (error? err) (cddddr err)))""",
    "lookup-error-message": """(define (lookup-error-message phase code)
  (let* ([table (case phase
                  [(parse) *parse-errors*]
                  [(infer) *infer-errors*]
                  [(eval) *eval-errors*]
                  [(block cas) *block-errors*]
                  [else '()])]
         [entry (assq code table)])
    (if entry
        (cdr entry)
        (symbol->string code))))""",
    "format-phase": """(define (format-phase phase)
  (format "[~a] " phase))""",
    "format-location": """(define (format-location ctx)
  (cond
    [(span? ctx)
     (let ([file (span-file ctx)]
           [line (span-line ctx)]
           [col (span-column ctx)])
       (if (and (> line 0) (> col 0))
           (format "~a:~a:~a: " file line col)
           ""))]
    [(string? ctx) (format "~a: " ctx)]
    [else ""]))""",
    "format-details": """(define (format-details code details)
  (if (null? details)
      ""
      (case code
        [(unbound-variable)
         (format ": '~a'" (car details))]
        [(type-mismatch)
         (if (>= (length details) 2)
             (format "\n  expected: ~a\n  actual:   ~a"
                     (car details) (cadr details))
             "")]
        [(arity-mismatch)
         (if (>= (length details) 2)
             (format " (expected ~a, got ~a)"
                     (car details) (cadr details))
             "")]
        [(unknown-primitive)
         (format ": '~a'" (car details))]
        [else
         (if (pair? details)
             (format ": ~a" (car details))
             "")])))""",
    "similar-to?": """(define (similar-to? s1 s2)
  (and (symbol? s1) (symbol? s2)
       (let ([str1 (symbol->string s1)]
             [str2 (symbol->string s2)])
         (<= (edit-distance str1 str2) 2))))""",
    "get-suggestion": """(define (get-suggestion phase code details)
  (case code
    [(unbound-variable)
     (cond
       [(and (pair? details) (similar-to? (car details) 'define))
        "Did you mean 'fn' for function definition?"]
       [(and (pair? details) (similar-to? (car details) 'lambda))
        "Use 'fn' instead of 'lambda' in The Fold"]
       [else
        "Check spelling or add a binding with 'let' or 'fix'"])]
    [(type-mismatch)
     "Ensure the expression returns the expected type"]
    [(not-a-function)
     "Only closures can be called. Check that the first element is a function."]
    [(unknown-primitive)
     (if (and (pair? details)
              (memq (car details) '(+ - * /)))
         (format "Use 'add', 'sub', 'mul', 'div' instead of ~a" (car details))
         "See (help 'primitives) for available operations")]
    [(unclosed-list)
     "Count your parentheses - every ( needs a matching )"]
    [(unclosed-string)
     "Add a closing \\\" to complete the string"]
    [else #f]))""",
    "format-suggestion": """(define (format-suggestion phase code details)
  (let ([suggestion (get-suggestion phase code details)])
    (if suggestion
        (format "\n  hint: ~a" suggestion)
        "")))""",
    "format-error": """(define (format-error err)
  (if (not (error? err))
      (format "~a" err)
      (let* ([phase (error-phase err)]
             [code (error-code err)]
             [ctx (error-context err)]
             [details (error-details err)]
             [message (lookup-error-message phase code)]
             [location (format-location ctx)])
        (string-append
          location
          (format-phase phase)
          message
          (format-details code details)
          (format-suggestion phase code details)))))""",
    "unbound-error": """(define (unbound-error var span)
  (make-error 'infer 'unbound-variable span var))""",
    "type-error": """(define (type-error expected actual span)
  (make-error 'infer 'type-mismatch span expected actual))""",
    "parse-error": """(define (parse-error code expected position)
  (make-error 'parse code (make-span "<input>" 1 position 1 position) expected))""",
    "eval-error": """(define (eval-error code value span)
  (make-error 'eval code span value))""",
}

FUNCTION_ORDER = [
    "make-error",
    "error?",
    "error-phase",
    "error-code",
    "error-context",
    "error-details",
    "lookup-error-message",
    "format-phase",
    "format-location",
    "format-details",
    "similar-to?",
    "get-suggestion",
    "format-suggestion",
    "format-error",
    "unbound-error",
    "type-error",
    "parse-error",
    "eval-error",
]

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **TARGET_DEFS}
DEF_ORDER = list(SUPPORT_DEFS.keys()) + FUNCTION_ORDER

DEPENDS: Dict[str, List[str]] = {
    "*parse-errors*": [],
    "*infer-errors*": [],
    "*eval-errors*": [],
    "*block-errors*": [],
    "make-error": [],
    "error?": [],
    "error-phase": ["error?"],
    "error-code": ["error?"],
    "error-context": ["error?"],
    "error-details": ["error?"],
    "lookup-error-message": ["*parse-errors*", "*infer-errors*", "*eval-errors*", "*block-errors*"],
    "format-phase": [],
    "format-location": [],
    "format-details": [],
    "similar-to?": [],
    "get-suggestion": ["similar-to?"],
    "format-suggestion": ["get-suggestion", "similar-to?"],
    "format-error": [
        "error?",
        "error-phase",
        "error-code",
        "error-context",
        "error-details",
        "lookup-error-message",
        "format-location",
        "format-phase",
        "format-details",
        "format-suggestion",
        "get-suggestion",
        "similar-to?",
        "*parse-errors*",
        "*infer-errors*",
        "*eval-errors*",
        "*block-errors*",
    ],
    "unbound-error": ["make-error"],
    "type-error": ["make-error"],
    "parse-error": ["make-error"],
    "eval-error": ["make-error"],
}

FUNCTION_SPECS = {
    "make-error": "Build structured error list `(error phase code context details...)`.",
    "error?": "Return #t iff value is an error list with tag `error` and arity >= 4.",
    "error-phase": "Extract phase symbol from error; return #f for non-errors.",
    "error-code": "Extract code symbol from error; return #f for non-errors.",
    "error-context": "Extract context payload from error; return #f for non-errors.",
    "error-details": "Extract details tail from error; return #f for non-errors.",
    "lookup-error-message": "Lookup message text by phase+code, else fallback to code symbol string.",
    "format-phase": "Format phase as `[phase] `.",
    "format-location": "Format source context: `file:line:col: ` for spans, `name: ` for strings, else empty string.",
    "format-details": "Format code-specific detail suffixes for diagnostics.",
    "similar-to?": "Return #t when two symbols have edit distance <= 2.",
    "get-suggestion": "Return targeted hint string for known error codes, else #f.",
    "format-suggestion": "Format suggestion as newline-prefixed hint, else empty string.",
    "format-error": "Render error object to user-facing string with location, phase, message, details, and hint.",
    "unbound-error": "Construct infer/unbound-variable error helper.",
    "type-error": "Construct infer/type-mismatch error helper.",
    "parse-error": "Construct parse-phase error with `<input>` span at given position.",
    "eval-error": "Construct eval-phase error helper.",
}

SKELETONS = {
    "make-error": """(define (make-error phase code context . details)
  ;; TODO: construct canonical error tuple
  <TODO>)""",
    "error?": """(define (error? x)
  ;; TODO: detect canonical error representation
  <TODO>)""",
    "error-phase": """(define (error-phase err)
  ;; TODO: return phase for valid errors
  <TODO>)""",
    "error-code": """(define (error-code err)
  ;; TODO: return code for valid errors
  <TODO>)""",
    "error-context": """(define (error-context err)
  ;; TODO: return context for valid errors
  <TODO>)""",
    "error-details": """(define (error-details err)
  ;; TODO: return details tail for valid errors
  <TODO>)""",
    "lookup-error-message": """(define (lookup-error-message phase code)
  ;; TODO: phase/code lookup with symbol->string fallback
  <TODO>)""",
    "format-phase": """(define (format-phase phase)
  ;; TODO: produce bracketed phase prefix
  <TODO>)""",
    "format-location": """(define (format-location ctx)
  ;; TODO: format span/string context into location prefix
  <TODO>)""",
    "format-details": """(define (format-details code details)
  ;; TODO: format code-specific detail suffix
  <TODO>)""",
    "similar-to?": """(define (similar-to? s1 s2)
  ;; TODO: symbol typo heuristic using edit distance <= 2
  <TODO>)""",
    "get-suggestion": """(define (get-suggestion phase code details)
  ;; TODO: return hint string for known cases, else #f
  <TODO>)""",
    "format-suggestion": """(define (format-suggestion phase code details)
  ;; TODO: prepend hint label when suggestion exists
  <TODO>)""",
    "format-error": """(define (format-error err)
  ;; TODO: render non-error as ~a; render structured errors with location/phase/message/details/hint
  <TODO>)""",
    "unbound-error": """(define (unbound-error var span)
  ;; TODO: construct unbound variable helper error
  <TODO>)""",
    "type-error": """(define (type-error expected actual span)
  ;; TODO: construct type mismatch helper error
  <TODO>)""",
    "parse-error": """(define (parse-error code expected position)
  ;; TODO: build parse error with <input> span anchored at position
  <TODO>)""",
    "eval-error": """(define (eval-error code value span)
  ;; TODO: construct eval helper error
  <TODO>)""",
}

DIFFICULTY = {
    "make-error": "easy",
    "error?": "easy",
    "error-phase": "easy",
    "error-code": "easy",
    "error-context": "easy",
    "error-details": "easy",
    "lookup-error-message": "medium",
    "format-phase": "easy",
    "format-location": "medium",
    "format-details": "medium",
    "similar-to?": "medium",
    "get-suggestion": "hard",
    "format-suggestion": "medium",
    "format-error": "hard",
    "unbound-error": "easy",
    "type-error": "easy",
    "parse-error": "medium",
    "eval-error": "easy",
}

VERIFY_BY_FUNCTION = {
    "make-error": "(equal? (make-error 'infer 'unbound-variable \"ctx\" 'x) '(error infer unbound-variable \"ctx\" x))",
    "error?": "(and (error? '(error infer unbound-variable \"ctx\" x)) (not (error? '(ok 1))) (not (error? '(error short))))",
    "error-phase": "(equal? (error-phase '(error eval division-by-zero no-span 0)) 'eval)",
    "error-code": "(equal? (error-code '(error eval division-by-zero no-span 0)) 'division-by-zero)",
    "error-context": "(equal? (error-context '(error infer type-mismatch \"ctx\" Int Bool)) \"ctx\")",
    "error-details": "(equal? (error-details '(error infer type-mismatch no-span Int Bool)) '(Int Bool))",
    "lookup-error-message": "(and (equal? (lookup-error-message 'infer 'unbound-variable) \"Variable is not defined\") (equal? (lookup-error-message 'infer 'unknown-code) \"unknown-code\"))",
    "format-phase": "(equal? (format-phase 'parse) \"[parse] \")",
    "format-location": "(let ([s (make-span \"file.ss\" 3 9 3 12)]) (and (equal? (format-location s) \"file.ss:3:9: \") (equal? (format-location \"repl\") \"repl: \") (equal? (format-location no-span) \"\")))",
    "format-details": "(and (equal? (format-details 'unbound-variable '(foo)) \": 'foo'\") (equal? (format-details 'arity-mismatch '(2 3)) \" (expected 2, got 3)\") (equal? (format-details 'type-mismatch '(Int Bool)) \"\\n  expected: Int\\n  actual:   Bool\") (equal? (format-details 'other '()) \"\"))",
    "similar-to?": "(and (similar-to? 'lambda 'lamdba) (not (similar-to? 'lambda 'completely-different)) (not (similar-to? \"x\" 'x)))",
    "get-suggestion": "(and (equal? (get-suggestion 'infer 'type-mismatch '()) \"Ensure the expression returns the expected type\") (equal? (get-suggestion 'parse 'unclosed-string '()) \"Add a closing \\\" to complete the string\") (not (get-suggestion 'infer 'unknown-code '())))",
    "format-suggestion": "(and (equal? (format-suggestion 'infer 'type-mismatch '()) \"\\n  hint: Ensure the expression returns the expected type\") (equal? (format-suggestion 'infer 'unknown-code '()) \"\"))",
    "format-error": "(equal? (format-error (make-error 'infer 'unbound-variable no-span 'x)) \"[infer] Variable is not defined: 'x'\\n  hint: Check spelling or add a binding with 'let' or 'fix'\")",
    "unbound-error": "(let ([e (unbound-error 'foo no-span)]) (and (equal? (error-phase e) 'infer) (equal? (error-code e) 'unbound-variable) (equal? (error-details e) '(foo))))",
    "type-error": "(let ([e (type-error 'Int 'Bool no-span)]) (and (equal? (error-code e) 'type-mismatch) (equal? (error-details e) '(Int Bool))))",
    "parse-error": "(let ([e (parse-error 'unexpected-char \"digit\" 7)]) (and (equal? (error-phase e) 'parse) (equal? (error-code e) 'unexpected-char) (span? (error-context e)) (= (span-column (error-context e)) 7)))",
    "eval-error": "(let ([e (eval-error 'division-by-zero 0 no-span)]) (and (equal? (error-phase e) 'eval) (equal? (error-code e) 'division-by-zero) (equal? (error-details e) '(0))))",
}

PYTHON_SNIPPETS = {
    "make-error": "def make_error(phase, code, context, *details):\n    return ('error', phase, code, context, *details)",
    "error?": "def is_error(x):\n    return isinstance(x, tuple) and len(x) >= 4 and x[0] == 'error'",
    "error-phase": "def error_phase(err):\n    return err[1] if is_error(err) else None",
    "error-code": "def error_code(err):\n    return err[2] if is_error(err) else None",
    "error-context": "def error_context(err):\n    return err[3] if is_error(err) else None",
    "error-details": "def error_details(err):\n    return list(err[4:]) if is_error(err) else None",
    "lookup-error-message": "def lookup_error_message(phase, code):\n    table = PHASE_TABLES.get(phase, {})\n    return table.get(code, str(code))",
    "format-phase": "def format_phase(phase):\n    return f'[{phase}] '",
    "format-location": "def format_location(ctx):\n    if is_span(ctx) and ctx.line > 0 and ctx.col > 0:\n        return f'{ctx.file}:{ctx.line}:{ctx.col}: '\n    if isinstance(ctx, str):\n        return f'{ctx}: '\n    return ''",
    "format-details": """def format_details(code, details):
    if not details:
        return ''
    if code == 'unbound-variable':
        return f": '{details[0]}'"
    if code == 'type-mismatch':
        if len(details) >= 2:
            return f"\\n  expected: {details[0]}\\n  actual:   {details[1]}"
        return ''
    if code == 'arity-mismatch':
        if len(details) >= 2:
            return f' (expected {details[0]}, got {details[1]})'
        return ''
    if code == 'unknown-primitive':
        return f": '{details[0]}'"
    return f': {details[0]}'""",
    "similar-to?": "def similar_to(a, b):\n    return isinstance(a, str) and isinstance(b, str) and edit_distance(a, b) <= 2",
    "get-suggestion": """def get_suggestion(phase, code, details):
    if code == 'unbound-variable':
        if details and similar_to(details[0], 'define'):
            return \"Did you mean 'fn' for function definition?\"
        if details and similar_to(details[0], 'lambda'):
            return \"Use 'fn' instead of 'lambda' in The Fold\"
        return \"Check spelling or add a binding with 'let' or 'fix'\"
    if code == 'type-mismatch':
        return 'Ensure the expression returns the expected type'
    if code == 'not-a-function':
        return 'Only closures can be called. Check that the first element is a function.'
    if code == 'unknown-primitive':
        if details and details[0] in ['+', '-', '*', '/']:
            return f\"Use 'add', 'sub', 'mul', 'div' instead of {details[0]}\"
        return \"See (help 'primitives) for available operations\"
    if code == 'unclosed-list':
        return 'Count your parentheses - every ( needs a matching )'
    if code == 'unclosed-string':
        return 'Add a closing \" to complete the string'
    return None""",
    "format-suggestion": "def format_suggestion(phase, code, details):\n    s = get_suggestion(phase, code, details)\n    return f'\\n  hint: {s}' if s else ''",
    "format-error": "def format_error(err):\n    if not is_error(err):\n        return str(err)\n    phase, code, ctx, *details = err[1:]\n    return format_location(ctx) + format_phase(phase) + lookup_error_message(phase, code) + format_details(code, details) + format_suggestion(phase, code, details)",
    "unbound-error": "def unbound_error(var, span):\n    return make_error('infer', 'unbound-variable', span, var)",
    "type-error": "def type_error(expected, actual, span):\n    return make_error('infer', 'type-mismatch', span, expected, actual)",
    "parse-error": "def parse_error(code, expected, position):\n    return make_error('parse', code, make_span('<input>', 1, position, 1, position), expected)",
    "eval-error": "def eval_error(code, value, span):\n    return make_error('eval', code, span, value)",
}

BUGGY_CASES = [
    {
        "fn": "make-error",
        "buggy": "(define (make-error phase code context . details)\n  `(err ,phase ,code ,context ,@details))",
        "note": "Tag must be `error`, not `err`.",
    },
    {
        "fn": "error?",
        "buggy": "(define (error? x)\n  (and (pair? x) (eq? (car x) 'error) (> (length x) 4)))",
        "note": "Valid errors have length >= 4, not strictly greater than 4.",
    },
    {
        "fn": "error-phase",
        "buggy": "(define (error-phase err)\n  (and (error? err) (car err)))",
        "note": "Phase is the second field, not the tag.",
    },
    {
        "fn": "error-code",
        "buggy": "(define (error-code err)\n  (and (error? err) (cadddr err)))",
        "note": "Code is the third field.",
    },
    {
        "fn": "error-context",
        "buggy": "(define (error-context err)\n  (and (error? err) (caddr err)))",
        "note": "Context is the fourth field.",
    },
    {
        "fn": "error-details",
        "buggy": "(define (error-details err)\n  (and (error? err) (cdddr err)))",
        "note": "Details should drop the first four fields, not three.",
    },
    {
        "fn": "lookup-error-message",
        "buggy": "(define (lookup-error-message phase code)\n  (let ([entry (assq code *infer-errors*)])\n    (and entry (cdr entry))))",
        "note": "Must select table by phase and fallback to symbol string when missing.",
    },
    {
        "fn": "format-phase",
        "buggy": "(define (format-phase phase)\n  (format \"~a \" phase))",
        "note": "Formatted phase must include square brackets.",
    },
    {
        "fn": "format-location",
        "buggy": "(define (format-location ctx)\n  (if (string? ctx) ctx \"\"))",
        "note": "String locations need trailing `: `; spans must format file/line/col.",
    },
    {
        "fn": "format-details",
        "buggy": "(define (format-details code details)\n  (if (null? details) \"\" (format \": ~a\" details)))",
        "note": "Details should format code-specific content, not print raw list object.",
    },
    {
        "fn": "similar-to?",
        "buggy": "(define (similar-to? s1 s2)\n  (and (symbol? s1) (symbol? s2) (<= (edit-distance (symbol->string s1) (symbol->string s2)) 1)))",
        "note": "Threshold should be <= 2 edits.",
    },
    {
        "fn": "get-suggestion",
        "buggy": "(define (get-suggestion phase code details)\n  #f)",
        "note": "Known codes should return useful suggestions.",
    },
    {
        "fn": "format-suggestion",
        "buggy": "(define (format-suggestion phase code details)\n  (format \"hint: ~a\" (get-suggestion phase code details)))",
        "note": "When suggestion is missing, this must return empty string.",
    },
    {
        "fn": "format-error",
        "buggy": "(define (format-error err)\n  (if (not (error? err))\n      \"error\"\n      (lookup-error-message (error-phase err) (error-code err))))",
        "note": "Non-errors should print as `~a`, and full formatted output must include location/phase/details/suggestions.",
    },
    {
        "fn": "unbound-error",
        "buggy": "(define (unbound-error var span)\n  (make-error 'parse 'unbound-variable span var))",
        "note": "Helper should use infer phase.",
    },
    {
        "fn": "type-error",
        "buggy": "(define (type-error expected actual span)\n  (make-error 'infer 'type-error span expected actual))",
        "note": "Code should be `type-mismatch`.",
    },
    {
        "fn": "parse-error",
        "buggy": "(define (parse-error code expected position)\n  (make-error 'parse code no-span expected))",
        "note": "Parse error should include `<input>` span at the given position.",
    },
    {
        "fn": "eval-error",
        "buggy": "(define (eval-error code value span)\n  (make-error 'infer code span value))",
        "note": "Helper should use eval phase.",
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

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

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
    sid = f"core_error_{family}_{family_counter[family]:03d}"
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
    return [name for name in DEF_ORDER if name != fn and name in tokens]


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
    parts = [ALL_DEFS[dep] for dep in dependency_closure(fn)] + [TARGET_DEFS[fn], VERIFY_BY_FUNCTION[fn]]
    body = "\n  ".join(parts)
    return f"(let ()\n  {body})"


# -----------------------------------------------------------------------------
# Family 1: spec_to_code (36)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    diff = DIFFICULTY[fn]

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=diff,
        source_function=fn,
        prompt=f"""You are implementing core runtime utilities in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=TARGET_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["core", "base", "error", "spec-to-code", fn],
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
        ground_truth=TARGET_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["core", "base", "error", "skeleton-completion", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (18)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
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
        ground_truth=TARGET_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["core", "base", "error", "python-to-scheme", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (18)
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
        ground_truth=TARGET_DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["core", "base", "error", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition/use (28)
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
        tags=["core", "base", "error", "composition", source_function] + extra_tags,
    )


composition_cases = [
    ("make-error", "Construct an eval division-by-zero error with value 0.", "(make-error 'eval 'division-by-zero no-span 0)", "(let ([e (make-error 'eval 'division-by-zero no-span 0)]) (and (equal? (car e) 'error) (equal? (error-phase e) 'eval) (equal? (error-code e) 'division-by-zero) (equal? (car (error-details e)) 0)))", "easy", ["direct"]),
    ("error?", "Return whether `(make-error 'infer 'unbound-variable no-span 'x)` is an error.", "(error? (make-error 'infer 'unbound-variable no-span 'x))", "(equal? (error? (make-error 'infer 'unbound-variable no-span 'x)) #t)", "easy", ["direct"]),
    ("error-phase", "Extract phase from `(make-error 'parse 'unexpected-eof no-span)`.", "(error-phase (make-error 'parse 'unexpected-eof no-span))", "(equal? (error-phase (make-error 'parse 'unexpected-eof no-span)) 'parse)", "easy", ["direct"]),
    ("error-code", "Extract code from `(make-error 'eval 'division-by-zero no-span 0)`.", "(error-code (make-error 'eval 'division-by-zero no-span 0))", "(equal? (error-code (make-error 'eval 'division-by-zero no-span 0)) 'division-by-zero)", "easy", ["direct"]),
    ("error-context", "Extract context from `(make-error 'infer 'type-mismatch \"ctx\" 'Int 'Bool)`.", "(error-context (make-error 'infer 'type-mismatch \"ctx\" 'Int 'Bool))", "(equal? (error-context (make-error 'infer 'type-mismatch \"ctx\" 'Int 'Bool)) \"ctx\")", "easy", ["direct"]),
    ("error-details", "Extract details from `(make-error 'infer 'type-mismatch no-span 'Int 'Bool)`.", "(error-details (make-error 'infer 'type-mismatch no-span 'Int 'Bool))", "(equal? (error-details (make-error 'infer 'type-mismatch no-span 'Int 'Bool)) '(Int Bool))", "easy", ["direct"]),
    ("lookup-error-message", "Lookup infer message for `unbound-variable`.", "(lookup-error-message 'infer 'unbound-variable)", "(equal? (lookup-error-message 'infer 'unbound-variable) \"Variable is not defined\")", "medium", ["direct"]),
    ("lookup-error-message", "Lookup unknown infer code and return fallback string.", "(lookup-error-message 'infer 'not-real)", "(equal? (lookup-error-message 'infer 'not-real) \"not-real\")", "medium", ["direct"]),
    ("format-phase", "Format phase symbol `infer`.", "(format-phase 'infer)", "(equal? (format-phase 'infer) \"[infer] \")", "easy", ["direct"]),
    ("format-location", "Format span location for `core.ss:2:7`.", "(format-location (make-span \"core.ss\" 2 7 2 9))", "(equal? (format-location (make-span \"core.ss\" 2 7 2 9)) \"core.ss:2:7: \")", "medium", ["direct"]),
    ("format-details", "Format details for `unbound-variable` with symbol `foo`.", "(format-details 'unbound-variable '(foo))", "(equal? (format-details 'unbound-variable '(foo)) \": 'foo'\")", "medium", ["direct"]),
    ("format-details", "Format arity mismatch details `(2 5)`.", "(format-details 'arity-mismatch '(2 5))", "(equal? (format-details 'arity-mismatch '(2 5)) \" (expected 2, got 5)\")", "medium", ["direct"]),
    ("similar-to?", "Check if `lambda` and `lamdba` are considered similar.", "(similar-to? 'lambda 'lamdba)", "(equal? (similar-to? 'lambda 'lamdba) #t)", "medium", ["property"]),
    ("get-suggestion", "Get suggestion for unknown primitive `+`.", "(get-suggestion 'infer 'unknown-primitive '(+))", "(equal? (get-suggestion 'infer 'unknown-primitive '(+)) \"Use 'add', 'sub', 'mul', 'div' instead of +\")", "hard", ["direct"]),
    ("get-suggestion", "Return #f for unknown suggestion code.", "(get-suggestion 'infer 'no-such-code '())", "(equal? (get-suggestion 'infer 'no-such-code '()) #f)", "medium", ["edge-case"]),
    ("format-suggestion", "Format suggestion for type mismatch.", "(format-suggestion 'infer 'type-mismatch '())", "(equal? (format-suggestion 'infer 'type-mismatch '()) \"\\n  hint: Ensure the expression returns the expected type\")", "medium", ["direct"]),
    ("format-suggestion", "Format suggestion for unknown code (should be empty).", "(format-suggestion 'infer 'nope '())", "(equal? (format-suggestion 'infer 'nope '()) \"\")", "easy", ["edge-case"]),
    ("format-error", "Format a canonical unbound-variable error for symbol `x`.", "(format-error (make-error 'infer 'unbound-variable no-span 'x))", "(equal? (format-error (make-error 'infer 'unbound-variable no-span 'x)) \"[infer] Variable is not defined: 'x'\\n  hint: Check spelling or add a binding with 'let' or 'fix'\")", "hard", ["integration"]),
    ("unbound-error", "Build `unbound-error` for `foo` and return its `(phase code details)`.", "(let ([e (unbound-error 'foo no-span)]) (list (error-phase e) (error-code e) (error-details e)))", "(equal? (let ([e (unbound-error 'foo no-span)]) (list (error-phase e) (error-code e) (error-details e))) '(infer unbound-variable (foo)))", "medium", ["integration"]),
    ("type-error", "Build `type-error` and return details tuple.", "(let ([e (type-error 'Int 'Bool no-span)]) (list (error-phase e) (error-code e) (error-details e)))", "(equal? (let ([e (type-error 'Int 'Bool no-span)]) (list (error-phase e) (error-code e) (error-details e))) '(infer type-mismatch (Int Bool)))", "medium", ["integration"]),
    ("parse-error", "Build parse error at position 11 and return `(phase code col)`.", "(let ([e (parse-error 'unexpected-char \"digit\" 11)]) (list (error-phase e) (error-code e) (span-column (error-context e))))", "(equal? (let ([e (parse-error 'unexpected-char \"digit\" 11)]) (list (error-phase e) (error-code e) (span-column (error-context e)))) '(parse unexpected-char 11))", "medium", ["integration"]),
    ("eval-error", "Build eval error and return `(phase code detail)`.", "(let ([e (eval-error 'division-by-zero 0 no-span)]) (list (error-phase e) (error-code e) (car (error-details e))))", "(equal? (let ([e (eval-error 'division-by-zero 0 no-span)]) (list (error-phase e) (error-code e) (car (error-details e)))) '(eval division-by-zero 0))", "easy", ["integration"]),
    ("error-code", "Map `error-code` over three helper errors.", "(map error-code (list (unbound-error 'x no-span) (type-error 'Int 'Bool no-span) (eval-error 'division-by-zero 0 no-span)))", "(equal? (map error-code (list (unbound-error 'x no-span) (type-error 'Int 'Bool no-span) (eval-error 'division-by-zero 0 no-span))) '(unbound-variable type-mismatch division-by-zero))", "medium", ["list"]),
    ("error-phase", "Count parse-phase errors in a list of mixed errors.", "(fold-left (lambda (n e) (if (eq? (error-phase e) 'parse) (+ n 1) n)) 0 (list (parse-error 'unexpected-eof \"x\" 1) (eval-error 'type-error 'bad no-span) (parse-error 'unclosed-list \"list\" 5)))", "(equal? (fold-left (lambda (n e) (if (eq? (error-phase e) 'parse) (+ n 1) n)) 0 (list (parse-error 'unexpected-eof \"x\" 1) (eval-error 'type-error 'bad no-span) (parse-error 'unclosed-list \"list\" 5))) 2)", "hard", ["fold"]),
    ("lookup-error-message", "Map lookup over infer codes `(unbound-variable type-mismatch unknown-code)`.", "(map (lambda (c) (lookup-error-message 'infer c)) '(unbound-variable type-mismatch unknown-code))", "(equal? (map (lambda (c) (lookup-error-message 'infer c)) '(unbound-variable type-mismatch unknown-code)) '(\"Variable is not defined\" \"Types do not match\" \"unknown-code\"))", "medium", ["list"]),
    ("format-location", "Map `format-location` over contexts `[no-span, \"repl\", span]`.", "(list (format-location no-span) (format-location \"repl\") (format-location (make-span \"f.ss\" 1 2 1 4)))", "(equal? (list (format-location no-span) (format-location \"repl\") (format-location (make-span \"f.ss\" 1 2 1 4))) '(\"\" \"repl: \" \"f.ss:1:2: \"))", "medium", ["list"]),
    ("format-details", "Return #t iff type mismatch detail formatting matches multiline layout.", "(equal? (format-details 'type-mismatch '(Int Bool)) \"\\n  expected: Int\\n  actual:   Bool\")", "(equal? (format-details 'type-mismatch '(Int Bool)) \"\\n  expected: Int\\n  actual:   Bool\")", "hard", ["property"]),
    ("format-error", "Format parse error at position 7 for expected `digit`.", "(format-error (parse-error 'unexpected-char \"digit\" 7))", "(equal? (format-error (parse-error 'unexpected-char \"digit\" 7)) \"<input>:1:7: [parse] Unexpected character: digit\")", "hard", ["integration"]),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

if sum(1 for s in samples if s["family"] == "spec_to_code") != 36:
    raise ValueError("spec_to_code family must contain exactly 36 samples")
if sum(1 for s in samples if s["family"] == "translation") != 18:
    raise ValueError("translation family must contain exactly 18 samples")
if sum(1 for s in samples if s["family"] == "bugfix") != 18:
    raise ValueError("bugfix family must contain exactly 18 samples")
if sum(1 for s in samples if s["family"] == "composition") != 28:
    raise ValueError("composition family must contain exactly 28 samples")
if len(samples) != 100:
    raise ValueError(f"expected 100 samples, got {len(samples)}")


# -----------------------------------------------------------------------------
# Split train/eval
# -----------------------------------------------------------------------------
by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

EVAL_QUOTA = {
    "spec_to_code": 7,
    "translation": 4,
    "bugfix": 4,
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

if len(train_rows) != 80 or len(eval_rows) != 20:
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
