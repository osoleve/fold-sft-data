#!/usr/bin/env python3
"""Generate Tier-1 SFT samples for lattice/query/patterns-parse.ss."""

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
from sft_split_utils import compute_leakage_aware_eval_ids

ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

SOURCE_MODULE = "lattice/query/patterns-parse.ss"
SOURCE_TEST = "lattice/query/test-patterns-parse.ss"

DEFS: Dict[str, str] = {
    "extract-tags": """(define (extract-tags text)
  (let ([positions (extract-tag-positions text)])
       (map (lambda (pos)
                    (let ([key (car pos)]
                          [val (cadr pos)])
                         (cons key (or val #t))))
            positions)))""",
    "extract-tag-positions": """(define (extract-tag-positions text)
  (let ([len (string-length text)])
       (let loop ([i 0] [results '()])
            (if (>= i len)
                (reverse results)
                (if (and (char=? (string-ref text i) #\\@)
                         (valid-tag-start? text i))
                    (let ([tag-info (parse-tag-at text i)])
                         (if tag-info
                             (loop (cadddr tag-info)
                                   (cons tag-info results))
                             (loop (+ i 1) results)))
                    (loop (+ i 1) results))))))""",
    "parse-tag-at": """(define (parse-tag-at text i)
  (let ([len (string-length text)]
        [start i])
       (let ([j (+ i 1)])
            (if (>= j len)
                #f
                (let ([first-char (string-ref text j)])
                     (if (not (char-key-start? first-char))
                         #f
                         (let key-loop ([k j])
                              (if (or (>= k len)
                                      (not (char-key? (string-ref text k))))
                                  (if (= k j)
                                      #f
                                      (let ([key-str (substring text j k)]
                                            [key-sym (string->symbol
                                                      (string-downcase
                                                       (substring text j k)))])
                                           (if (and (< k len)
                                                    (char=? (string-ref text k) #\\:))
                                               (let value-loop ([v (+ k 1)])
                                                    (if (or (>= v len)
                                                            (not (char-value? (string-ref text v))))
                                                        (if (= v (+ k 1))
                                                            (list key-sym #f start k)
                                                            (list key-sym
                                                                  (substring text (+ k 1) v)
                                                                  start v))
                                                        (value-loop (+ v 1))))
                                               (list key-sym #f start k))))
                                  (key-loop (+ k 1))))))))))""",
    "valid-tag-key?": """(define (valid-tag-key? s)
  (and (> (string-length s) 0)
       (char-key-start? (string-ref s 0))
       (let loop ([i 1])
            (if (>= i (string-length s))
                #t
                (and (char-key? (string-ref s i))
                     (loop (+ i 1)))))))""",
    "format-tag": """(define (format-tag key value)
  (if (eq? value #t)
      (string-append "@" (symbol->string key))
      (string-append "@" (symbol->string key) ":" value)))""",
    "tags->string": """(define (tags->string tags)
  (if (null? tags)
      ""
      (let loop ([ts tags] [acc ""])
           (if (null? ts)
               acc
               (loop (cdr ts)
                     (string-append acc
                                    (if (string=? acc "") "" " ")
                                    (format-tag (caar ts) (cdar ts))))))))""",
    "has-path-traversal?": """(define (has-path-traversal? s)
  (let ([len (string-length s)])
       (cond
        [(= len 0) #f]
        [(char=? (string-ref s 0) #\\/) #t]
        [else
         (let loop ([i 0])
              (cond
               [(>= i (- len 1)) #f]
               [(and (char=? (string-ref s i) #\\.)
                     (char=? (string-ref s (+ i 1)) #\\.))
                (let ([at-start (= i 0)]
                      [at-end (>= (+ i 2) len)]
                      [after-slash (and (> i 0) (char=? (string-ref s (- i 1)) #\\/))]
                      [before-slash (and (< (+ i 2) len)
                                         (or (char=? (string-ref s (+ i 2)) #\\/)
                                             (char=? (string-ref s (+ i 2)) #\\\\)))])
                     (if (or at-start at-end after-slash before-slash)
                         #t
                         (loop (+ i 1))))]
               [else (loop (+ i 1))]))])))""",
    "safe-extract-tags": """(define (safe-extract-tags text)
  (map (lambda (pair)
               (let ([key (car pair)]
                     [val (cdr pair)])
                    (cons key (if (string? val)
                                  (sanitize-tag-value val)
                                  val))))
       (extract-tags text)))""",
}

FUNCTION_ORDER = [
    "extract-tags",
    "extract-tag-positions",
    "parse-tag-at",
    "valid-tag-key?",
    "format-tag",
    "tags->string",
    "has-path-traversal?",
    "safe-extract-tags",
]

SUPPORT_DEFS: Dict[str, str] = {
    "char-key-start?": """(define (char-key-start? c)
  (and (char>=? c #\\a) (char<=? c #\\z)))""",
    "char-key?": """(define (char-key? c)
  (or (and (char>=? c #\\a) (char<=? c #\\z))
      (and (char>=? c #\\0) (char<=? c #\\9))
      (char=? c #\\-)))""",
    "char-value?": """(define (char-value? c)
  (or (and (char>=? c #\\a) (char<=? c #\\z))
      (and (char>=? c #\\A) (char<=? c #\\Z))
      (and (char>=? c #\\0) (char<=? c #\\9))
      (char=? c #\\-)
      (char=? c #\\.)
      (char=? c #\\/)
      (char=? c #\\_)
      (char=? c #\\:)))""",
    "char-whitespace?": """(define (char-whitespace? c)
  (or (char=? c #\\space)
      (char=? c #\\newline)
      (char=? c #\\tab)
      (char=? c #\\return)))""",
    "valid-tag-start?": """(define (valid-tag-start? text i)
  (or (= i 0)
      (let ([prev-char (string-ref text (- i 1))])
           (char-whitespace? prev-char))))""",
    "string-downcase": """(define (string-downcase s)
  (list->string
   (map char-downcase (string->list s))))""",
    "valid-tag-value?": """(define (valid-tag-value? s)
  (and (> (string-length s) 0)
       (let loop ([i 0])
            (if (>= i (string-length s))
                #t
                (and (char-value? (string-ref s i))
                     (loop (+ i 1)))))))""",
    "has-tag?": """(define (has-tag? tags key)
  (and (assq key tags) #t))""",
    "get-tag": """(define (get-tag tags key)
  (let ([pair (assq key tags)])
       (if pair (cdr pair) #f)))""",
    "filter-tags-by-key": """(define (filter-tags-by-key tags pred)
  (filter (lambda (pair) (pred (car pair))) tags))""",
    "filter": """(define (filter pred lst)
  (cond
   [(null? lst) '()]
   [(pred (car lst)) (cons (car lst) (filter pred (cdr lst)))]
   [else (filter pred (cdr lst))]))""",
    "sanitize-tag-value": """(define (sanitize-tag-value s)
  (if (has-path-traversal? s)
      ""
      (list->string
       (filter (lambda (c)
                       (and (char>=? c #\\space)
                            (char<=? c #\\~)
                            (not (memv c '(#\\$ #\\` #\\& #\\| #\\; #\\\\)))))
               (string->list s)))))""",
}

SUPPORT_ORDER = [
    "char-key-start?",
    "char-key?",
    "char-value?",
    "char-whitespace?",
    "valid-tag-start?",
    "string-downcase",
    "valid-tag-value?",
    "has-tag?",
    "get-tag",
    "filter-tags-by-key",
    "filter",
    "sanitize-tag-value",
]

ALL_DEFS: Dict[str, str] = dict(SUPPORT_DEFS)
ALL_DEFS.update(DEFS)
ALL_ORDER = list(dict.fromkeys(FUNCTION_ORDER + SUPPORT_ORDER))

FUNCTION_SPECS = {
    "extract-tags": "Extract every valid @tag from text and return alist pairs `(symbol . value-or-#t)`.",
    "extract-tag-positions": "Scan text and return `(key value start end)` rows for every valid tag occurrence.",
    "parse-tag-at": "Parse one tag beginning at index `i` where `text[i]` is `@`; return parsed tuple or `#f`.",
    "valid-tag-key?": "Validate tag keys: non-empty, starts with lowercase letter, then lowercase/digit/hyphen.",
    "format-tag": "Render a single tag pair as `@key` for flags or `@key:value` for valued tags.",
    "tags->string": "Render a list of tags into one space-separated string using `format-tag`.",
    "has-path-traversal?": "Detect unsafe path payloads including absolute paths and `..` traversal contexts.",
    "safe-extract-tags": "Extract tags and sanitize string values while preserving flag tags as `#t`.",
}

SKELETONS = {
    "extract-tags": """(define (extract-tags text)
  ;; TODO: map tag-position rows to `(key . value-or-#t)` pairs
  <TODO>)""",
    "extract-tag-positions": """(define (extract-tag-positions text)
  ;; TODO: scan the string, parse each valid @tag, and collect `(key value start end)` rows
  <TODO>)""",
    "parse-tag-at": """(define (parse-tag-at text i)
  ;; TODO: parse one @tag at index i and return `(key value start end)` or #f
  <TODO>)""",
    "valid-tag-key?": """(define (valid-tag-key? s)
  ;; TODO: enforce tag-key lexical rules
  <TODO>)""",
    "format-tag": """(define (format-tag key value)
  ;; TODO: format @key or @key:value
  <TODO>)""",
    "tags->string": """(define (tags->string tags)
  ;; TODO: join formatted tags with spaces
  <TODO>)""",
    "has-path-traversal?": """(define (has-path-traversal? s)
  ;; TODO: detect absolute/traversal paths in tag values
  <TODO>)""",
    "safe-extract-tags": """(define (safe-extract-tags text)
  ;; TODO: sanitize string values returned by extract-tags
  <TODO>)""",
}

DIFFICULTY = {
    "extract-tags": "medium",
    "extract-tag-positions": "hard",
    "parse-tag-at": "hard",
    "valid-tag-key?": "easy",
    "format-tag": "easy",
    "tags->string": "medium",
    "has-path-traversal?": "hard",
    "safe-extract-tags": "medium",
}

VERIFY_BY_FUNCTION = {
    "extract-tags": """(let ([a (extract-tags "Fix @status:done @urgent")]
      [b (extract-tags "email@example.com @ok")]
      [c (extract-tags "@scope:core:query")])
  (and (equal? a '((status . "done") (urgent . #t)))
       (equal? b '((ok . #t)))
       (equal? c '((scope . "core:query")))))""",
    "extract-tag-positions": """(let ([ps (extract-tag-positions "x @todo:fix y @done")]
      [mid (extract-tag-positions "mail user@example.com")])
  (and (equal? ps '((todo "fix" 2 11) (done #f 14 19)))
       (equal? mid '())))""",
    "parse-tag-at": """(let* ([r1 (parse-tag-at "@topic:lang:scheme" 0)]
       [r2 (parse-tag-at "x @ok" 2)]
       [r3 (parse-tag-at "@9bad" 0)]
       [r4 (parse-tag-at "@todo:" 0)])
  (and (equal? r1 '(topic "lang:scheme" 0 18))
       (equal? r2 '(ok #f 2 5))
       (equal? r3 #f)
       (equal? r4 '(todo #f 0 5))))""",
    "valid-tag-key?": """(and (valid-tag-key? "status")
     (valid-tag-key? "high-priority2")
     (not (valid-tag-key? "1status"))
     (not (valid-tag-key? "Bad"))
     (not (valid-tag-key? "")))""",
    "format-tag": """(and (string=? (format-tag 'todo #t) "@todo")
     (string=? (format-tag 'status "done") "@status:done")
     (string=? (format-tag 'scope "lang:scheme") "@scope:lang:scheme"))""",
    "tags->string": """(let ([s1 (tags->string '((status . "done") (urgent . #t) (scope . "core:query")))]
      [s2 (tags->string '())])
  (and (string=? s1 "@status:done @urgent @scope:core:query")
       (string=? s2 "")))""",
    "has-path-traversal?": """(and (has-path-traversal? "../secret")
     (has-path-traversal? "foo/../bar")
     (has-path-traversal? "/etc/passwd")
     (not (has-path-traversal? "file..name"))
     (not (has-path-traversal? "./current/dir")))""",
    "safe-extract-tags": """(let ([tags (safe-extract-tags "@path:../secret @cmd:rm&rf @ok:clean @flag")])
  (and (equal? tags '((path . "") (cmd . "rm") (ok . "clean") (flag . #t)))
       (string=? (cdr (assq 'ok tags)) "clean")
       (eq? (cdr (assq 'flag tags)) #t)))""",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9!$%&*+./:<=>?@^_~-]+")

TRANSLATION_FUNCTIONS = FUNCTION_ORDER

PYTHON_SNIPPETS = {
    "extract-tags": """def extract_tags(text):
    positions = extract_tag_positions(text)
    out = []
    for key, value, _start, _end in positions:
        out.append((key, True if value is None else value))
    return out""",
    "extract-tag-positions": """def extract_tag_positions(text):
    i = 0
    out = []
    n = len(text)
    while i < n:
        if text[i] == '@' and valid_tag_start(text, i):
            row = parse_tag_at(text, i)
            if row is not None:
                out.append(row)
                i = row[3]
                continue
        i += 1
    return out""",
    "parse-tag-at": """def parse_tag_at(text, i):
    n = len(text)
    start = i
    j = i + 1
    if j >= n:
        return None
    if not char_key_start(text[j]):
        return None

    k = j
    while k < n and char_key(text[k]):
        k += 1
    if k == j:
        return None

    key = text[j:k].lower()
    if k < n and text[k] == ':':
        v = k + 1
        while v < n and char_value(text[v]):
            v += 1
        if v == k + 1:
            return (key, None, start, k)
        return (key, text[k + 1:v], start, v)

    return (key, None, start, k)""",
    "valid-tag-key?": """def valid_tag_key(s):
    if len(s) == 0:
        return False
    if not char_key_start(s[0]):
        return False
    return all(char_key(ch) for ch in s[1:])""",
    "format-tag": """def format_tag(key, value):
    if value is True:
        return f"@{key}"
    return f"@{key}:{value}" """,
    "tags->string": """def tags_to_string(tags):
    return " ".join(format_tag(k, v) for (k, v) in tags)""",
    "has-path-traversal?": """def has_path_traversal(s):
    if len(s) == 0:
        return False
    if s[0] == '/':
        return True
    for i in range(len(s) - 1):
        if s[i] == '.' and s[i + 1] == '.':
            at_start = (i == 0)
            at_end = (i + 2 >= len(s))
            after_slash = (i > 0 and s[i - 1] == '/')
            before_slash = (i + 2 < len(s) and s[i + 2] in ('/', '\\\\'))
            if at_start or at_end or after_slash or before_slash:
                return True
    return False""",
    "safe-extract-tags": """def safe_extract_tags(text):
    out = []
    for key, value in extract_tags(text):
        if isinstance(value, str):
            out.append((key, sanitize_tag_value(value)))
        else:
            out.append((key, value))
    return out""",
}

CHEZ_SNIPPETS = {
    "extract-tags": """(define (collect-tags txt)
  (map (lambda (row)
         (let ([k (car row)] [v (cadr row)])
           (cons k (if v v #t))))
       (extract-tag-positions txt)))""",
    "extract-tag-positions": """(define (collect-tag-rows txt)
  (let ([n (string-length txt)])
    (let loop ([i 0] [acc '()])
      (if (>= i n)
          (reverse acc)
          (if (and (char=? (string-ref txt i) #\\@)
                   (valid-tag-start? txt i))
              (let ([row (parse-tag-at txt i)])
                (if row
                    (loop (cadddr row) (cons row acc))
                    (loop (+ i 1) acc)))
              (loop (+ i 1) acc))))))""",
    "parse-tag-at": """(define (parse-one-tag txt i)
  (let* ([n (string-length txt)] [j (+ i 1)])
    (if (or (>= j n) (not (char-key-start? (string-ref txt j))))
        #f
        (let key-loop ([k j])
          (if (and (< k n) (char-key? (string-ref txt k)))
              (key-loop (+ k 1))
              (let ([ks (substring txt j k)]
                    [sym (string->symbol (string-downcase (substring txt j k)))])
                (if (and (< k n) (char=? (string-ref txt k) #\\:))
                    (let val-loop ([v (+ k 1)])
                      (if (and (< v n) (char-value? (string-ref txt v)))
                          (val-loop (+ v 1))
                          (if (= v (+ k 1))
                              (list sym #f i k)
                              (list sym (substring txt (+ k 1) v) i v))))
                    (list sym #f i k)))))))""",
    "valid-tag-key?": """(define (tag-key-ok? s)
  (and (> (string-length s) 0)
       (char-key-start? (string-ref s 0))
       (let loop ([i 1])
         (or (>= i (string-length s))
             (and (char-key? (string-ref s i))
                  (loop (+ i 1)))))))""",
    "format-tag": """(define (render-tag key value)
  (if (eq? value #t)
      (string-append "@" (symbol->string key))
      (string-append "@" (symbol->string key) ":" value)))""",
    "tags->string": """(define (render-tags tags)
  (let loop ([ts tags] [acc ""])
    (if (null? ts)
        acc
        (loop (cdr ts)
              (string-append acc
                             (if (string=? acc "") "" " ")
                             (format-tag (caar ts) (cdar ts)))))))""",
    "has-path-traversal?": """(define (path-traversal? s)
  (let ([n (string-length s)])
    (cond
      [(= n 0) #f]
      [(char=? (string-ref s 0) #\\/) #t]
      [else
       (let loop ([i 0])
         (cond
           [(>= i (- n 1)) #f]
           [(and (char=? (string-ref s i) #\\.)
                 (char=? (string-ref s (+ i 1)) #\\.))
            (let ([at-start (= i 0)]
                  [at-end (>= (+ i 2) n)]
                  [after-slash (and (> i 0) (char=? (string-ref s (- i 1)) #\\/))]
                  [before-slash (and (< (+ i 2) n)
                                     (or (char=? (string-ref s (+ i 2)) #\\/)
                                         (char=? (string-ref s (+ i 2)) #\\\\)))])
              (if (or at-start at-end after-slash before-slash)
                  #t
                  (loop (+ i 1))))]
           [else (loop (+ i 1))]))])))""",
    "safe-extract-tags": """(define (collect-safe-tags txt)
  (map (lambda (kv)
         (let ([k (car kv)] [v (cdr kv)])
           (cons k (if (string? v) (sanitize-tag-value v) v))))
       (extract-tags txt)))""",
}

BUGGY_CASES = [
    {
        "fn": "extract-tags",
        "buggy": """(define (extract-tags text)
  (let ([positions (extract-tag-positions text)])
       (map (lambda (pos)
                    (let ([key (car pos)]
                          [val (cadr pos)])
                         (cons key val)))
            positions)))""",
        "note": "Flag tags like `@todo` should map to `#t`, not `#f`.",
    },
    {
        "fn": "extract-tags",
        "buggy": """(define (extract-tags text)
  (let ([positions (extract-tag-positions text)])
       (map (lambda (pos)
                    (let ([key (car pos)]
                          [val (cadr pos)])
                         (cons key (or val #t))))
            (cdr positions))))""",
        "note": "The current logic silently drops the first parsed tag.",
    },
    {
        "fn": "extract-tag-positions",
        "buggy": """(define (extract-tag-positions text)
  (let ([len (string-length text)])
       (let loop ([i 0] [results '()])
            (if (>= i len)
                (reverse results)
                (if (char=? (string-ref text i) #\\@)
                    (let ([tag-info (parse-tag-at text i)])
                         (if tag-info
                             (loop (cadddr tag-info)
                                   (cons tag-info results))
                             (loop (+ i 1) results)))
                    (loop (+ i 1) results))))))""",
        "note": "`@` in the middle of words (for example emails) must not start tags.",
    },
    {
        "fn": "extract-tag-positions",
        "buggy": """(define (extract-tag-positions text)
  (let ([len (string-length text)])
       (let loop ([i 0] [results '()])
            (if (>= i len)
                (reverse results)
                (if (and (char=? (string-ref text i) #\\@)
                         (valid-tag-start? text i))
                    (let ([tag-info (parse-tag-at text i)])
                         (if tag-info
                             (loop (+ i 1)
                                   (cons (list (car tag-info)
                                               (cadr tag-info)
                                               (+ (caddr tag-info) 1)
                                               (cadddr tag-info))
                                         results))
                             (loop (+ i 1) results)))
                    (loop (+ i 1) results))))))""",
        "note": "Recorded start index must point at `@`; shifting it by +1 corrupts reported tag positions.",
    },
    {
        "fn": "parse-tag-at",
        "buggy": """(define (parse-tag-at text i)
  (let ([len (string-length text)]
        [start i])
       (let ([j (+ i 1)])
            (if (>= j len)
                #f
                (let ([first-char (string-ref text j)])
                     (if (not (char-key? first-char))
                         #f
                         (let key-loop ([k j])
                              (if (or (>= k len)
                                      (not (char-key? (string-ref text k))))
                                  (if (= k j)
                                      #f
                                      (let ([key-sym (string->symbol
                                                      (string-downcase
                                                       (substring text j k)))])
                                           (if (and (< k len)
                                                    (char=? (string-ref text k) #\\:))
                                               (let value-loop ([v (+ k 1)])
                                                    (if (or (>= v len)
                                                            (not (char-value? (string-ref text v))))
                                                        (if (= v (+ k 1))
                                                            (list key-sym #f start k)
                                                            (list key-sym
                                                                  (substring text (+ k 1) v)
                                                                  start v))
                                                        (value-loop (+ v 1))))
                                               (list key-sym #f start k))))
                                  (key-loop (+ k 1))))))))))""",
        "note": "Tag keys must start with lowercase letters only; leading digits are invalid.",
    },
    {
        "fn": "parse-tag-at",
        "buggy": """(define (parse-tag-at text i)
  (let ([len (string-length text)]
        [start i])
       (let ([j (+ i 1)])
            (if (>= j len)
                #f
                (let ([first-char (string-ref text j)])
                     (if (not (char-key-start? first-char))
                         #f
                         (let key-loop ([k j])
                              (if (or (>= k len)
                                      (not (char-key? (string-ref text k))))
                                  (if (= k j)
                                      #f
                                      (let ([key-sym (string->symbol
                                                      (string-downcase
                                                       (substring text j k)))])
                                           (if (and (< k len)
                                                    (char=? (string-ref text k) #\\:))
                                               (let value-loop ([v (+ k 1)])
                                                    (if (or (>= v len)
                                                            (not (char-value? (string-ref text v))))
                                                        (if (= v (+ k 1))
                                                            (list key-sym "" start k)
                                                            (list key-sym
                                                                  (substring text (+ k 1) v)
                                                                  start v))
                                                        (value-loop (+ v 1))))
                                               (list key-sym #f start k))))
                                  (key-loop (+ k 1))))))))))""",
        "note": "`@key:` with an empty value should be treated as a flag (`#f` internal value), not empty string.",
    },
    {
        "fn": "valid-tag-key?",
        "buggy": """(define (valid-tag-key? s)
  (and (>= (string-length s) 0)
       (char-key-start? (string-ref s 0))
       (let loop ([i 1])
            (if (>= i (string-length s))
                #t
                (and (char-key? (string-ref s i))
                     (loop (+ i 1)))))))""",
        "note": "The empty-string guard is wrong and can crash on `(string-ref s 0)`; empty keys must return #f.",
    },
    {
        "fn": "valid-tag-key?",
        "buggy": """(define (valid-tag-key? s)
  (and (> (string-length s) 0)
       (char-key? (string-ref s 0))
       (let loop ([i 1])
            (if (>= i (string-length s))
                #t
                (and (char-key? (string-ref s i))
                     (loop (+ i 1)))))))""",
        "note": "The first character has stricter rules than subsequent characters.",
    },
    {
        "fn": "format-tag",
        "buggy": """(define (format-tag key value)
  (if (eq? value #t)
      (string-append "@" (symbol->string key) ":")
      (string-append "@" (symbol->string key) ":" value)))""",
        "note": "Flag tags should render as `@key` without a trailing colon.",
    },
    {
        "fn": "format-tag",
        "buggy": """(define (format-tag key value)
  (if (eq? value #t)
      (string-append "@" (symbol->string key))
      (string-append "@" value ":" (symbol->string key))))""",
        "note": "Value tags must keep `key` before `value` in output.",
    },
    {
        "fn": "tags->string",
        "buggy": """(define (tags->string tags)
  (if (null? tags)
      ""
      (let loop ([ts tags] [acc ""])
           (if (null? ts)
               acc
               (loop (cdr ts)
                     (string-append acc
                                    (format-tag (caar ts) (cdar ts))))))))""",
        "note": "Tags should be separated by spaces in the joined output.",
    },
    {
        "fn": "tags->string",
        "buggy": """(define (tags->string tags)
  (if (null? tags)
      ""
      (let loop ([ts tags] [acc " "])
           (if (null? ts)
               acc
               (loop (cdr ts)
                     (string-append acc
                                    (if (string=? acc "") "" " ")
                                    (format-tag (caar ts) (cdar ts))))))))""",
        "note": "The accumulator should not inject a leading space before the first tag.",
    },
    {
        "fn": "has-path-traversal?",
        "buggy": """(define (has-path-traversal? s)
  (let ([len (string-length s)])
       (cond
        [(= len 0) #f]
        [else
         (let loop ([i 0])
              (cond
               [(>= i (- len 1)) #f]
               [(and (char=? (string-ref s i) #\\.)
                     (char=? (string-ref s (+ i 1)) #\\.))
                (let ([at-start (= i 0)]
                      [at-end (>= (+ i 2) len)]
                      [after-slash (and (> i 0) (char=? (string-ref s (- i 1)) #\\/))]
                      [before-slash (and (< (+ i 2) len)
                                         (or (char=? (string-ref s (+ i 2)) #\\/)
                                             (char=? (string-ref s (+ i 2)) #\\\\)))])
                     (if (or at-start at-end after-slash before-slash)
                         #t
                         (loop (+ i 1))))]
               [else (loop (+ i 1))]))])))""",
        "note": "Absolute Unix-style paths should be rejected immediately.",
    },
    {
        "fn": "has-path-traversal?",
        "buggy": """(define (has-path-traversal? s)
  (let ([len (string-length s)])
       (cond
        [(= len 0) #f]
        [(char=? (string-ref s 0) #\\/) #t]
        [else
         (let loop ([i 0])
              (cond
               [(>= i (- len 1)) #f]
               [(and (char=? (string-ref s i) #\\.)
                     (char=? (string-ref s (+ i 1)) #\\.))
                (let ([at-start (= i 0)]
                      [at-end (>= (+ i 2) len)]
                      [after-slash (and (> i 0) (char=? (string-ref s (- i 1)) #\\/))]
                      [before-slash (and (< (+ i 2) len)
                                         (char=? (string-ref s (+ i 2)) #\\/))])
                     (if (or at-start at-end after-slash before-slash)
                         #t
                         (loop (+ i 1))))]
               [else (loop (+ i 1))]))])))""",
        "note": "Traversal checks should also handle Windows separator `\\` after `..`.",
    },
    {
        "fn": "safe-extract-tags",
        "buggy": """(define (safe-extract-tags text)
  (extract-tags text))""",
        "note": "Safe extraction must sanitize string values instead of returning raw extracted tags.",
    },
    {
        "fn": "safe-extract-tags",
        "buggy": """(define (safe-extract-tags text)
  (map (lambda (pair)
               (let ([key (car pair)]
                     [val (cdr pair)])
                    (cons key (sanitize-tag-value (if (string? val) val "")))))
       (extract-tags text)))""",
        "note": "Flag tags must remain `#t` and should not be coerced into empty strings.",
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
    sid = f"patterns_parse_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
        "prompt": diversify_prompt(
            prompt.strip(),
            family,
            source_function,
            family_counter[family],
            category,
            verify_expr,
            ground_truth=ground_truth,
            available_functions=ALL_ORDER,
        ),
        "ground_truth": ground_truth.strip(),
        "verify_expr": verify_expr.strip(),
        "tags": tags,
    }
    for k in REQUIRED_KEYS:
        if k not in sample:
            raise ValueError(f"missing key {k}")
    samples.append(sample)


def verify_refs(verify_expr: str) -> List[str]:
    tokens = set(TOKEN_RE.findall(verify_expr))
    return [name for name in ALL_ORDER if name in tokens]


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


DEPENDS: Dict[str, List[str]] = {
    "char-key-start?": [],
    "char-key?": [],
    "char-value?": [],
    "char-whitespace?": [],
    "valid-tag-start?": ["char-whitespace?"],
    "string-downcase": [],
    "parse-tag-at": ["char-key-start?", "char-key?", "char-value?", "string-downcase"],
    "extract-tag-positions": ["valid-tag-start?", "parse-tag-at"],
    "extract-tags": ["extract-tag-positions"],
    "valid-tag-key?": ["char-key-start?", "char-key?"],
    "valid-tag-value?": ["char-value?"],
    "format-tag": [],
    "tags->string": ["format-tag"],
    "has-tag?": [],
    "get-tag": [],
    "filter": [],
    "filter-tags-by-key": ["filter"],
    "has-path-traversal?": [],
    "sanitize-tag-value": ["has-path-traversal?", "filter"],
    "safe-extract-tags": ["extract-tags", "sanitize-tag-value"],
}

# -----------------------------------------------------------------------------
# Family 1: spec_to_code (16)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement this metadata parser function in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme function definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "query", "patterns-parse", "spec-to-code", fn],
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
        tags=["tier1", "query", "patterns-parse", "skeleton-completion", fn],
    )

    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Implement `{fn}` directly from this API contract.

Module: `{SOURCE_MODULE}`
Contract focus: {FUNCTION_SPECS[fn]}

Requirements:
1. Keep the exact function name/signature for `{fn}`.
2. Match module behavior on boundary/edge cases.
3. Return exactly one definition, with no extra helpers or commentary.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "query", "patterns-parse", "contract-implementation", fn],
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
Preserve behavior exactly and use target function name `{fn}`.
Return only the Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "query", "patterns-parse", "python-to-scheme", fn],
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
        tags=["tier1", "query", "patterns-parse", "chez-to-fold", fn],
    )

    add_sample(
        family="translation",
        category="translation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""Translate this reference logic into canonical Fold Scheme for `{fn}`.

Preserve observable behavior and edge-case handling exactly.
Keep the target function name/signature as `{fn}`.
Return only the final Scheme definition.

```python
{PYTHON_SNIPPETS[fn]}
```""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["tier1", "query", "patterns-parse", "reference-translation", fn],
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
        tags=["tier1", "query", "patterns-parse", "bugfix", fn],
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
    composition_prompt = (
        f"{prompt.rstrip()}\n\n"
        f"Ensure `{source_function}` is part of the composed solution."
    )
    add_sample(
        family="composition",
        category="usage",
        difficulty=difficulty,
        source_function=source_function,
        prompt=composition_prompt,
        ground_truth=ground_truth,
        verify_expr=build_verify(verify_check, [source_function]),
        tags=["tier1", "query", "patterns-parse", "composition", source_function] + extra_tags,
    )


composition_cases = [
    {
        "fn": "extract-tags",
        "prompt": "Parse a deployment line and return `(status urgent?)` using tag lookup helpers.",
        "gt": "(let* ([tags (extract-tags \"Deploy @status:ready @urgent\")] [status (get-tag tags 'status)] [urgent? (has-tag? tags 'urgent)]) (list status urgent?))",
        "verify": "(equal? (let* ([tags (extract-tags \"Deploy @status:ready @urgent\")] [status (get-tag tags 'status)] [urgent? (has-tag? tags 'urgent)]) (list status urgent?)) '(\"ready\" #t))",
        "difficulty": "medium",
        "tags": ["lookup"],
    },
    {
        "fn": "extract-tags",
        "prompt": "Extract tags, keep only keys starting with `p`, and return canonical text.",
        "gt": "(let* ([tags (extract-tags \"Refs @priority:high @owner:alice @phase:beta\")] [p-tags (filter-tags-by-key tags (lambda (k) (char=? (string-ref (symbol->string k) 0) #\\p)))]) (tags->string p-tags))",
        "verify": "(equal? (let* ([tags (extract-tags \"Refs @priority:high @owner:alice @phase:beta\")] [p-tags (filter-tags-by-key tags (lambda (k) (char=? (string-ref (symbol->string k) 0) #\\p)))]) (tags->string p-tags)) \"@priority:high @phase:beta\")",
        "difficulty": "hard",
        "tags": ["filtering"],
    },
    {
        "fn": "extract-tags",
        "prompt": "Parse a mixed line with an email token and return `(count rendered)` for valid tags only.",
        "gt": "(let* ([tags (extract-tags \"mail user@example.com @ok @kind:note\")] [count (length tags)] [as-text (tags->string tags)]) (list count as-text))",
        "verify": "(equal? (let* ([tags (extract-tags \"mail user@example.com @ok @kind:note\")] [count (length tags)] [as-text (tags->string tags)]) (list count as-text)) '(2 \"@ok @kind:note\"))",
        "difficulty": "medium",
        "tags": ["render"],
    },
    {
        "fn": "extract-tags",
        "prompt": "Roundtrip parse/format and report whether the first parsed key satisfies key validation.",
        "gt": "(let* ([tags (extract-tags \"@status:open @todo\")] [rendered (tags->string tags)] [status-ok (valid-tag-key? (symbol->string (caar tags)))]) (list rendered status-ok))",
        "verify": "(equal? (let* ([tags (extract-tags \"@status:open @todo\")] [rendered (tags->string tags)] [status-ok (valid-tag-key? (symbol->string (caar tags)))]) (list rendered status-ok)) '(\"@status:open @todo\" #t))",
        "difficulty": "medium",
        "tags": ["roundtrip"],
    },

    {
        "fn": "extract-tag-positions",
        "prompt": "Extract tag rows, then re-parse each start offset and return only parsed keys.",
        "gt": "(let* ([text \"x @todo:fix y @done\"] [positions (extract-tag-positions text)]) (map (lambda (p) (car (parse-tag-at text (caddr p)))) positions))",
        "verify": "(equal? (let* ([text \"x @todo:fix y @done\"] [positions (extract-tag-positions text)]) (map (lambda (p) (car (parse-tag-at text (caddr p)))) positions)) '(todo done))",
        "difficulty": "hard",
        "tags": ["alignment"],
    },
    {
        "fn": "extract-tag-positions",
        "prompt": "Parse rows from mixed text and render each parsed key back as a flag tag.",
        "gt": "(let* ([text \"mail user@example.com @ok @phase:beta\"] [positions (extract-tag-positions text)] [keys (map car positions)]) (map (lambda (k) (format-tag k #t)) keys))",
        "verify": "(equal? (let* ([text \"mail user@example.com @ok @phase:beta\"] [positions (extract-tag-positions text)] [keys (map car positions)]) (map (lambda (k) (format-tag k #t)) keys)) '(\"@ok\" \"@phase\"))",
        "difficulty": "medium",
        "tags": ["formatting"],
    },
    {
        "fn": "extract-tag-positions",
        "prompt": "Inspect the second parsed row and report its key plus whether its value is lexically valid.",
        "gt": "(let* ([positions (extract-tag-positions \"@a @topic:core:query\")] [second (cadr positions)]) (list (car second) (valid-tag-value? (cadr second))))",
        "verify": "(equal? (let* ([positions (extract-tag-positions \"@a @topic:core:query\")] [second (cadr positions)]) (list (car second) (valid-tag-value? (cadr second)))) '(topic #t))",
        "difficulty": "medium",
        "tags": ["validation"],
    },
    {
        "fn": "extract-tag-positions",
        "prompt": "Cross-check each extracted position by reparsing at the same offset and verifying key/value agreement.",
        "gt": "(let* ([text \"A @kind:doc and @flag\"] [positions (extract-tag-positions text)]) (map (lambda (p) (let ([parsed (parse-tag-at text (caddr p))]) (and parsed (eq? (car parsed) (car p)) (equal? (cadr parsed) (cadr p))))) positions))",
        "verify": "(equal? (let* ([text \"A @kind:doc and @flag\"] [positions (extract-tag-positions text)]) (map (lambda (p) (let ([parsed (parse-tag-at text (caddr p))]) (and parsed (eq? (car parsed) (car p)) (equal? (cadr parsed) (cadr p))))) positions)) '(#t #t))",
        "difficulty": "hard",
        "tags": ["consistency"],
    },

    {
        "fn": "parse-tag-at",
        "prompt": "Parse a tag at a known index and re-render it with `format-tag`.",
        "gt": "(let* ([text \"x @status:done\"] [p (parse-tag-at text 2)]) (format-tag (car p) (or (cadr p) #t)))",
        "verify": "(equal? (let* ([text \"x @status:done\"] [p (parse-tag-at text 2)]) (format-tag (car p) (or (cadr p) #t))) \"@status:done\")",
        "difficulty": "medium",
        "tags": ["render"],
    },
    {
        "fn": "parse-tag-at",
        "prompt": "Parse three candidate offsets and summarize key validity plus parsed outputs.",
        "gt": "(let* ([text \"@ok not-tag @1bad @todo\"] [r1 (parse-tag-at text 0)] [r2 (parse-tag-at text 12)] [r3 (parse-tag-at text 18)]) (list (valid-tag-key? (symbol->string (car r1))) r2 (car r3) (cadr r3)))",
        "verify": "(equal? (let* ([text \"@ok not-tag @1bad @todo\"] [r1 (parse-tag-at text 0)] [r2 (parse-tag-at text 12)] [r3 (parse-tag-at text 18)]) (list (valid-tag-key? (symbol->string (car r1))) r2 (car r3) (cadr r3))) '(#t #f todo #f))",
        "difficulty": "hard",
        "tags": ["offsets"],
    },
    {
        "fn": "parse-tag-at",
        "prompt": "Show that both `@todo:` and `@todo` normalize to flag rendering.",
        "gt": "(let* ([p1 (parse-tag-at \"@todo:\" 0)] [p2 (parse-tag-at \"@todo\" 0)]) (list (format-tag (car p1) (or (cadr p1) #t)) (format-tag (car p2) (or (cadr p2) #t))))",
        "verify": "(equal? (let* ([p1 (parse-tag-at \"@todo:\" 0)] [p2 (parse-tag-at \"@todo\" 0)]) (list (format-tag (car p1) (or (cadr p1) #t)) (format-tag (car p2) (or (cadr p2) #t)))) '(\"@todo\" \"@todo\"))",
        "difficulty": "medium",
        "tags": ["flags"],
    },
    {
        "fn": "parse-tag-at",
        "prompt": "Parse a hierarchical value and return `(value valid-value?)`.",
        "gt": "(let* ([p (parse-tag-at \"@scope:lang:scheme:chez\" 0)] [v (cadr p)]) (list v (valid-tag-value? v)))",
        "verify": "(equal? (let* ([p (parse-tag-at \"@scope:lang:scheme:chez\" 0)] [v (cadr p)]) (list v (valid-tag-value? v))) '(\"lang:scheme:chez\" #t))",
        "difficulty": "medium",
        "tags": ["hierarchical"],
    },

    {
        "fn": "valid-tag-key?",
        "prompt": "Filter candidate keys to only valid tag keys.",
        "gt": "(filter valid-tag-key? '(\"status\" \"1bad\" \"high-priority\" \"Bad\"))",
        "verify": "(equal? (filter valid-tag-key? '(\"status\" \"1bad\" \"high-priority\" \"Bad\")) '(\"status\" \"high-priority\"))",
        "difficulty": "easy",
        "tags": ["filtering"],
    },
    {
        "fn": "valid-tag-key?",
        "prompt": "Parse tags and return key-validation flags for each parsed symbol.",
        "gt": "(let* ([tags (extract-tags \"@status:ok @todo @x-1:yes\")] [keys (map (lambda (p) (symbol->string (car p))) tags)]) (map valid-tag-key? keys))",
        "verify": "(equal? (let* ([tags (extract-tags \"@status:ok @todo @x-1:yes\")] [keys (map (lambda (p) (symbol->string (car p))) tags)]) (map valid-tag-key? keys)) '(#t #t #t))",
        "difficulty": "medium",
        "tags": ["parser-integration"],
    },
    {
        "fn": "valid-tag-key?",
        "prompt": "Format only valid keys as flags and mark invalid ones explicitly.",
        "gt": "(map (lambda (k) (if (valid-tag-key? k) (format-tag (string->symbol k) #t) \"invalid\")) '(\"priority\" \"2bad\" \"owner\"))",
        "verify": "(equal? (map (lambda (k) (if (valid-tag-key? k) (format-tag (string->symbol k) #t) \"invalid\")) '(\"priority\" \"2bad\" \"owner\")) '(\"@priority\" \"invalid\" \"@owner\"))",
        "difficulty": "medium",
        "tags": ["conditional"],
    },
    {
        "fn": "valid-tag-key?",
        "prompt": "Run safe extraction, then validate every parsed key symbol.",
        "gt": "(let* ([tags (safe-extract-tags \"@path:../secret @owner:alice\")] [keys (map (lambda (p) (symbol->string (car p))) tags)]) (map valid-tag-key? keys))",
        "verify": "(equal? (let* ([tags (safe-extract-tags \"@path:../secret @owner:alice\")] [keys (map (lambda (p) (symbol->string (car p))) tags)]) (map valid-tag-key? keys)) '(#t #t))",
        "difficulty": "medium",
        "tags": ["safe-pipeline"],
    },

    {
        "fn": "format-tag",
        "prompt": "Render the first tag directly and also render the full tag list with `tags->string`.",
        "gt": "(let* ([tags '((status . \"done\") (urgent . #t) (scope . \"core:query\"))] [joined (tags->string tags)] [first (format-tag (caar tags) (cdar tags))]) (list first joined))",
        "verify": "(equal? (let* ([tags '((status . \"done\") (urgent . #t) (scope . \"core:query\"))] [joined (tags->string tags)] [first (format-tag (caar tags) (cdar tags))]) (list first joined)) '(\"@status:done\" \"@status:done @urgent @scope:core:query\"))",
        "difficulty": "medium",
        "tags": ["render"],
    },
    {
        "fn": "format-tag",
        "prompt": "Format a key/value pair and verify it roundtrips through `extract-tags`.",
        "gt": "(let* ([rendered (format-tag 'owner \"alice\")] [roundtrip (extract-tags rendered)]) (cdr (car roundtrip)))",
        "verify": "(equal? (let* ([rendered (format-tag 'owner \"alice\")] [roundtrip (extract-tags rendered)]) (cdr (car roundtrip))) \"alice\")",
        "difficulty": "easy",
        "tags": ["roundtrip"],
    },
    {
        "fn": "format-tag",
        "prompt": "Conditionally format a tag only when the key string is valid.",
        "gt": "(let ([k \"priority\"]) (if (valid-tag-key? k) (format-tag (string->symbol k) \"high\") \"invalid\"))",
        "verify": "(equal? (let ([k \"priority\"]) (if (valid-tag-key? k) (format-tag (string->symbol k) \"high\") \"invalid\")) \"@priority:high\")",
        "difficulty": "easy",
        "tags": ["conditional"],
    },
    {
        "fn": "format-tag",
        "prompt": "Safe-extract two path tags, then format both normalized results.",
        "gt": "(let* ([tags (safe-extract-tags \"@path:../secret @path2:docs/readme\")] [p1 (format-tag (caar tags) (cdar tags))] [p2 (format-tag (caadr tags) (cdadr tags))]) (list p1 p2))",
        "verify": "(equal? (let* ([tags (safe-extract-tags \"@path:../secret @path2:docs/readme\")] [p1 (format-tag (caar tags) (cdar tags))] [p2 (format-tag (caadr tags) (cdadr tags))]) (list p1 p2)) '(\"@path:\" \"@path2:docs/readme\"))",
        "difficulty": "medium",
        "tags": ["safe-pipeline"],
    },

    {
        "fn": "tags->string",
        "prompt": "Roundtrip tags by parsing then rendering the same sequence.",
        "gt": "(tags->string (extract-tags \"@status:done @urgent\"))",
        "verify": "(equal? (tags->string (extract-tags \"@status:done @urgent\")) \"@status:done @urgent\")",
        "difficulty": "easy",
        "tags": ["roundtrip"],
    },
    {
        "fn": "tags->string",
        "prompt": "Render safely extracted tags where an unsafe path is sanitized away.",
        "gt": "(tags->string (safe-extract-tags \"@path:../secret @owner:alice\"))",
        "verify": "(equal? (tags->string (safe-extract-tags \"@path:../secret @owner:alice\")) \"@path: @owner:alice\")",
        "difficulty": "medium",
        "tags": ["safe-pipeline"],
    },
    {
        "fn": "tags->string",
        "prompt": "Parse tags, keep only status/priority, and render the filtered subset.",
        "gt": "(let* ([tags (extract-tags \"@status:draft @priority:high @owner:bob\")] [filtered (filter-tags-by-key tags (lambda (k) (or (eq? k 'status) (eq? k 'priority))))]) (tags->string filtered))",
        "verify": "(equal? (let* ([tags (extract-tags \"@status:draft @priority:high @owner:bob\")] [filtered (filter-tags-by-key tags (lambda (k) (or (eq? k 'status) (eq? k 'priority))))]) (tags->string filtered)) \"@status:draft @priority:high\")",
        "difficulty": "hard",
        "tags": ["filtering"],
    },
    {
        "fn": "tags->string",
        "prompt": "Append one formatted tag to the rendered base tag string.",
        "gt": "(let* ([tags (extract-tags \"@status:done\")] [base (tags->string tags)] [extra (format-tag 'source \"api\")]) (string-append base \" \" extra))",
        "verify": "(equal? (let* ([tags (extract-tags \"@status:done\")] [base (tags->string tags)] [extra (format-tag 'source \"api\")]) (string-append base \" \" extra)) \"@status:done @source:api\")",
        "difficulty": "medium",
        "tags": ["concatenation"],
    },

    {
        "fn": "has-path-traversal?",
        "prompt": "Classify several paths for traversal risk and also include sanitized control value.",
        "gt": "(let* ([paths '(\"../secret\" \"docs/readme\" \"/etc/passwd\" \"foo/../bar\" \"file..name\")] [flags (map has-path-traversal? paths)] [safe (sanitize-tag-value \"docs/readme\")]) (list flags safe))",
        "verify": "(equal? (let* ([paths '(\"../secret\" \"docs/readme\" \"/etc/passwd\" \"foo/../bar\" \"file..name\")] [flags (map has-path-traversal? paths)] [safe (sanitize-tag-value \"docs/readme\")]) (list flags safe)) '((#t #f #t #t #f) \"docs/readme\"))",
        "difficulty": "medium",
        "tags": ["classification"],
    },
    {
        "fn": "has-path-traversal?",
        "prompt": "Sanitize two path-like values and report sanitized outputs plus a traversal check.",
        "gt": "(let* ([v1 (sanitize-tag-value \"../secret\")] [v2 (sanitize-tag-value \"docs/readme\")]) (list v1 v2 (has-path-traversal? v2)))",
        "verify": "(equal? (let* ([v1 (sanitize-tag-value \"../secret\")] [v2 (sanitize-tag-value \"docs/readme\")]) (list v1 v2 (has-path-traversal? v2))) '(\"\" \"docs/readme\" #f))",
        "difficulty": "medium",
        "tags": ["sanitization"],
    },
    {
        "fn": "has-path-traversal?",
        "prompt": "Safe-extract tags and check whether sanitized path values still look like traversal.",
        "gt": "(let* ([tags (safe-extract-tags \"@path:../../etc/passwd @ok:docs/readme\")] [p1 (cdr (assq 'path tags))] [p2 (cdr (assq 'ok tags))]) (list (has-path-traversal? p1) (has-path-traversal? p2)))",
        "verify": "(equal? (let* ([tags (safe-extract-tags \"@path:../../etc/passwd @ok:docs/readme\")] [p1 (cdr (assq 'path tags))] [p2 (cdr (assq 'ok tags))]) (list (has-path-traversal? p1) (has-path-traversal? p2))) '(#f #f))",
        "difficulty": "hard",
        "tags": ["safe-pipeline"],
    },
    {
        "fn": "has-path-traversal?",
        "prompt": "Format a path tag as a flag when traversal is detected, otherwise keep the value.",
        "gt": "(let ([p \"../x\"]) (if (has-path-traversal? p) (format-tag 'path #t) (format-tag 'path p)))",
        "verify": "(equal? (let ([p \"../x\"]) (if (has-path-traversal? p) (format-tag 'path #t) (format-tag 'path p))) \"@path\")",
        "difficulty": "medium",
        "tags": ["conditional"],
    },

    {
        "fn": "safe-extract-tags",
        "prompt": "Compare raw extraction vs safe extraction for a path value.",
        "gt": "(let* ([raw (extract-tags \"@path:../secret @owner:alice\")] [safe (safe-extract-tags \"@path:../secret @owner:alice\")]) (list (cdr (assq 'path raw)) (cdr (assq 'path safe))))",
        "verify": "(equal? (let* ([raw (extract-tags \"@path:../secret @owner:alice\")] [safe (safe-extract-tags \"@path:../secret @owner:alice\")]) (list (cdr (assq 'path raw)) (cdr (assq 'path safe)))) '(\"../secret\" \"\"))",
        "difficulty": "medium",
        "tags": ["comparison"],
    },
    {
        "fn": "safe-extract-tags",
        "prompt": "Safe-extract tags and also render the sanitized result string.",
        "gt": "(let ([tags (safe-extract-tags \"@path:/etc/passwd @todo\")]) (list tags (tags->string tags)))",
        "verify": "(equal? (let ([tags (safe-extract-tags \"@path:/etc/passwd @todo\")]) (list tags (tags->string tags))) '(((path . \"\") (todo . #t)) \"@path: @todo\"))",
        "difficulty": "medium",
        "tags": ["render"],
    },
    {
        "fn": "safe-extract-tags",
        "prompt": "Render safe tags from text that includes a traversal sequence.",
        "gt": "(tags->string (safe-extract-tags \"@path:foo/../bar @owner:bob\"))",
        "verify": "(equal? (tags->string (safe-extract-tags \"@path:foo/../bar @owner:bob\")) \"@path: @owner:bob\")",
        "difficulty": "medium",
        "tags": ["safe-render"],
    },
    {
        "fn": "safe-extract-tags",
        "prompt": "Safe-extract, keep only cmd/status tags, then return their values.",
        "gt": "(let* ([tags (safe-extract-tags \"@cmd:rm&rf @status:ok @scope:core:query\")] [keep (filter-tags-by-key tags (lambda (k) (or (eq? k 'cmd) (eq? k 'status))))]) (list (get-tag keep 'cmd) (get-tag keep 'status)))",
        "verify": "(equal? (let* ([tags (safe-extract-tags \"@cmd:rm&rf @status:ok @scope:core:query\")] [keep (filter-tags-by-key tags (lambda (k) (or (eq? k 'cmd) (eq? k 'status))))]) (list (get-tag keep 'cmd) (get-tag keep 'status))) '(\"rm\" \"ok\"))",
        "difficulty": "hard",
        "tags": ["selective"],
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

# Strengthen non-composition rows with an independent behavior check selected
# from composition verifies of the same source function.
composition_verify_by_fn: Dict[str, List[str]] = defaultdict(list)
for case in composition_cases:
    fn = str(case["fn"])
    check = str(case["verify"]).strip()
    if check not in composition_verify_by_fn[fn]:
        composition_verify_by_fn[fn].append(check)

for sample in samples:
    family = str(sample["family"])
    if family == "composition":
        continue

    fn = str(sample["source_function"])
    checks = composition_verify_by_fn.get(fn, [])
    if not checks:
        continue

    sid = str(sample["id"])
    base = sum(ord(ch) for ch in sid)
    pick1 = base % len(checks)
    selected_checks = [checks[pick1]]
    if len(checks) > 1:
        pick2 = (base * 7 + len(sid)) % len(checks)
        if pick2 == pick1:
            pick2 = (pick1 + 1) % len(checks)
        selected_checks.append(checks[pick2])

    wrapped_checks = [build_verify(check, [fn]) for check in selected_checks]
    sample["verify_expr"] = (
        f"(and {str(sample['verify_expr']).strip()} {' '.join(wrapped_checks)})"
    )
    checks_block = "\n\n".join(
        f"Check {idx + 1}:\n```scheme\n{check}\n```"
        for idx, check in enumerate(selected_checks)
    )
    sample["prompt"] = (
        f"{str(sample['prompt']).rstrip()}\n\n"
        "Independent behavior checks to satisfy:\n"
        f"{checks_block}"
    )

if sum(1 for s in samples if s["family"] == "composition") != 32:
    raise ValueError("composition family must contain exactly 32 samples")

# -----------------------------------------------------------------------------
# Split train/eval (deterministic, with eval source-function coverage)
# -----------------------------------------------------------------------------
by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

EVAL_RATIO = 0.18
EVAL_MIN_BY_FAMILY = {
    "spec_to_code": 3,
    "translation": 3,
    "bugfix": 2,
    "composition": 5,
}

eval_ids = compute_leakage_aware_eval_ids(
    samples,
    eval_ratio=EVAL_RATIO,
    eval_min_by_family=EVAL_MIN_BY_FAMILY,
    enforce_source_function_coverage=True,
)

id_to_sample: Dict[str, Dict[str, object]] = {str(s["id"]): s for s in samples}
all_source_functions = sorted({str(s["source_function"]) for s in samples})
missing_after = [
    fn
    for fn in all_source_functions
    if not any(str(id_to_sample[sid]["source_function"]) == fn for sid in eval_ids)
]
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

if not train_rows or not eval_rows:
    raise ValueError(f"split mismatch: train={len(train_rows)}, eval={len(eval_rows)}")


def quality_gate(rows: List[Dict[str, object]]) -> None:
    local_symbols = set(ALL_ORDER)

    for i, row in enumerate(rows, start=1):
        family = str(row["family"])
        source_function = str(row["source_function"])
        prompt = str(row["prompt"])
        gt = str(row["ground_truth"]).strip()
        verify = str(row["verify_expr"]).strip()

        if gt == verify:
            raise ValueError(f"row {i}: tautological verify_expr equals ground_truth")

        if family in {"spec_to_code", "translation", "bugfix"}:
            expected_prefix = f"(define ({source_function}"
            if not gt.lstrip().startswith(expected_prefix):
                raise ValueError(
                    f"row {i}: source function mismatch; expected ground_truth to start with {expected_prefix}"
                )
            if source_function not in prompt:
                raise ValueError(f"row {i}: prompt does not mention source_function {source_function}")

        if family == "composition":
            tokens = set(TOKEN_RE.findall(gt + "\n" + verify))
            if source_function not in tokens:
                raise ValueError(f"row {i}: composition source_function not used in solution/verify")
            used_local = {name for name in local_symbols if name in tokens}
            if len(used_local) < 2:
                raise ValueError(f"row {i}: composition sample too weak (needs at least 2 composed local functions)")

        if family == "bugfix":
            if "Known issue:" not in prompt:
                raise ValueError(f"row {i}: bugfix prompt missing issue description")
            if "```scheme" not in prompt:
                raise ValueError(f"row {i}: bugfix prompt missing code block")
            if "<TODO>" in prompt:
                raise ValueError(f"row {i}: weak bugfix prompt contains TODO placeholder")


quality_gate(train_rows + eval_rows)


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
    f.write("\n")

print(json.dumps(summary, indent=2))
print(f"Wrote: {ALL_PATH}")
print(f"Wrote: {TRAIN_PATH}")
print(f"Wrote: {EVAL_PATH}")
