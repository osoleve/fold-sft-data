#!/usr/bin/env python3
"""Generate SFT samples for core/base/sha256.ss."""

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

SOURCE_MODULE = "core/base/sha256.ss"
SOURCE_TEST = "core/base/test-sha256.ss"

SUPPORT_DEFS: Dict[str, str] = {
    "u32": """(define-syntax u32
  (syntax-rules ()
    [(_ x) (fxand x #xFFFFFFFF)]))""",
    "u32+": """(define-syntax u32+
  (syntax-rules ()
    [(_ . args) (fxand (fx+ . args) #xFFFFFFFF)]))""",
    "rotr32": """(define-syntax rotr32
  (syntax-rules ()
    [(_ x n)
     (fxior (fxarithmetic-shift-right x n)
            (fxarithmetic-shift-left (fxand x (fx- (fxarithmetic-shift-left 1 n) 1))
                                     (fx- 32 n)))]))""",
    "shr": """(define-syntax shr
  (syntax-rules ()
    [(_ x n) (fxarithmetic-shift-right x n)]))""",
    "Ch": """(define-syntax Ch
  (syntax-rules ()
    [(_ x y z) (fxxor (fxand x y)
                      (fxand (fxnot x) z))]))""",
    "Maj": """(define-syntax Maj
  (syntax-rules ()
    [(_ x y z) (fxxor (fxand x y)
                      (fxxor (fxand x z)
                             (fxand y z)))]))""",
    "Sigma0": """(define-syntax Sigma0
  (syntax-rules ()
    [(_ x) (fxxor (rotr32 x 2)
                  (fxxor (rotr32 x 13)
                         (rotr32 x 22)))]))""",
    "Sigma1": """(define-syntax Sigma1
  (syntax-rules ()
    [(_ x) (fxxor (rotr32 x 6)
                  (fxxor (rotr32 x 11)
                         (rotr32 x 25)))]))""",
    "sigma0": """(define-syntax sigma0
  (syntax-rules ()
    [(_ x) (fxxor (rotr32 x 7)
                  (fxxor (rotr32 x 18)
                         (shr x 3)))]))""",
    "sigma1": """(define-syntax sigma1
  (syntax-rules ()
    [(_ x) (fxxor (rotr32 x 17)
                  (fxxor (rotr32 x 19)
                         (shr x 10)))]))""",
    "H-init": """(define H-init
  (fxvector #x6a09e667 #xbb67ae85 #x3c6ef372 #xa54ff53a
            #x510e527f #x9b05688c #x1f83d9ab #x5be0cd19))""",
    "K": """(define K
  (fxvector #x428a2f98 #x71374491 #xb5c0fbcf #xe9b5dba5
            #x3956c25b #x59f111f1 #x923f82a4 #xab1c5ed5
            #xd807aa98 #x12835b01 #x243185be #x550c7dc3
            #x72be5d74 #x80deb1fe #x9bdc06a7 #xc19bf174
            #xe49b69c1 #xefbe4786 #x0fc19dc6 #x240ca1cc
            #x2de92c6f #x4a7484aa #x5cb0a9dc #x76f988da
            #x983e5152 #xa831c66d #xb00327c8 #xbf597fc7
            #xc6e00bf3 #xd5a79147 #x06ca6351 #x14292967
            #x27b70a85 #x2e1b2138 #x4d2c6dfc #x53380d13
            #x650a7354 #x766a0abb #x81c2c92e #x92722c85
            #xa2bfe8a1 #xa81a664b #xc24b8b70 #xc76c51a3
            #xd192e819 #xd6990624 #xf40e3585 #x106aa070
            #x19a4c116 #x1e376c08 #x2748774c #x34b0bcb5
            #x391c0cb3 #x4ed8aa4a #x5b9cca4f #x682e6ff3
            #x748f82ee #x78a5636f #x84c87814 #x8cc70208
            #x90befffa #xa4506ceb #xbef9a3f7 #xc67178f2))""",
}

DEFS: Dict[str, str] = {
    "iota": """(define (iota n)
  (let loop ([i 0] [acc '()])
       (if (= i n)
           (reverse acc)
           (loop (+ i 1) (cons i acc)))))""",
    "pad-message": """(define (pad-message msg)
  (let* ([len (bytevector-length msg)]
         [bit-len (* 8 len)]
         [padded-len (let ([rem (modulo (+ len 9) 64)])
                          (if (= rem 0)
                              (+ len 9)
                              (+ len 9 (- 64 rem))))]
         [result (make-bytevector padded-len 0)])
        (bytevector-copy! msg 0 result 0 len)
        (bytevector-u8-set! result len #x80)
        (bytevector-u64-set! result (- padded-len 8) bit-len 'big)
        result))""",
    "make-schedule": """(define (make-schedule msg offset)
  (let ([W (make-fxvector 64)])
       (do ([i 0 (fx+ i 1)])
           ((fx= i 16))
           (fxvector-set! W i (bytevector-u32-ref msg (fx+ offset (fx* i 4)) 'big)))
       (do ([i 16 (fx+ i 1)])
           ((fx= i 64))
           (fxvector-set! W i
                        (u32+ (sigma1 (fxvector-ref W (fx- i 2)))
                              (fxvector-ref W (fx- i 7))
                              (sigma0 (fxvector-ref W (fx- i 15)))
                              (fxvector-ref W (fx- i 16)))))
       W))""",
    "compress": """(define (compress H W)
  (let ([a (fxvector-ref H 0)]
        [b (fxvector-ref H 1)]
        [c (fxvector-ref H 2)]
        [d (fxvector-ref H 3)]
        [e (fxvector-ref H 4)]
        [f (fxvector-ref H 5)]
        [g (fxvector-ref H 6)]
        [h (fxvector-ref H 7)])
       (do ([i 0 (fx+ i 1)])
           ((fx= i 64))
           (let* ([T1 (u32+ h (Sigma1 e) (Ch e f g)
                            (fxvector-ref K i) (fxvector-ref W i))]
                  [T2 (u32+ (Sigma0 a) (Maj a b c))])
                 (set! h g)
                 (set! g f)
                 (set! f e)
                 (set! e (u32+ d T1))
                 (set! d c)
                 (set! c b)
                 (set! b a)
                 (set! a (u32+ T1 T2))))
       (fxvector (u32+ (fxvector-ref H 0) a)
                 (u32+ (fxvector-ref H 1) b)
                 (u32+ (fxvector-ref H 2) c)
                 (u32+ (fxvector-ref H 3) d)
                 (u32+ (fxvector-ref H 4) e)
                 (u32+ (fxvector-ref H 5) f)
                 (u32+ (fxvector-ref H 6) g)
                 (u32+ (fxvector-ref H 7) h))))""",
    "sha256": """(define (sha256 msg)
  (let* ([padded (pad-message msg)]
         [num-blocks (quotient (bytevector-length padded) 64)]
         [H (fxvector-copy H-init)])
        (do ([i 0 (fx+ i 1)])
            ((fx= i num-blocks))
            (let ([W (make-schedule padded (fx* i 64))])
                 (set! H (compress H W))))
        (let ([result (make-bytevector 32)])
             (do ([i 0 (fx+ i 1)])
                 ((fx= i 8))
                 (bytevector-u32-set! result (fx* i 4) (fxvector-ref H i) 'big))
             result)))""",
    "sha256-hex": """(define (sha256-hex msg)
  (let ([hash (sha256 msg)]
        [hex-chars "0123456789abcdef"])
       (apply string-append
              (map (lambda (i)
                           (let ([b (bytevector-u8-ref hash i)])
                                (string
                                 (string-ref hex-chars (quotient b 16))
                                 (string-ref hex-chars (modulo b 16)))))
                   (iota 32)))))""",
    "hash->hex": """(define (hash->hex hash)
  (let ([hex-chars "0123456789abcdef"])
       (apply string-append
              (map (lambda (i)
                           (let ([b (bytevector-u8-ref hash i)])
                                (string
                                 (string-ref hex-chars (quotient b 16))
                                 (string-ref hex-chars (modulo b 16)))))
                   (iota (bytevector-length hash))))))""",
    "hex->hash": """(define (hex->hash hex)
  (let* ([len (string-length hex)]
         [result (make-bytevector (quotient len 2))])
        (do ([i 0 (+ i 2)])
            ((>= i len))
            (bytevector-u8-set! result
                                (quotient i 2)
                                (+ (* 16 (hex-digit (string-ref hex i)))
                                   (hex-digit (string-ref hex (+ i 1))))))
        result))""",
    "hex-digit": """(define (hex-digit c)
  (cond
   [(char<=? #\\0 c #\\9) (- (char->integer c) (char->integer #\\0))]
   [(char<=? #\\a c #\\f) (+ 10 (- (char->integer c) (char->integer #\\a)))]
   [(char<=? #\\A c #\\F) (+ 10 (- (char->integer c) (char->integer #\\A)))]
   [else 0]))""",
}

FUNCTION_ORDER = [
    "iota",
    "pad-message",
    "make-schedule",
    "compress",
    "sha256",
    "sha256-hex",
    "hash->hex",
    "hex->hash",
    "hex-digit",
]

ALL_DEFS: Dict[str, str] = {**SUPPORT_DEFS, **DEFS}
SUPPORT_ORDER = list(SUPPORT_DEFS.keys())

DEPENDS: Dict[str, List[str]] = {
    "u32": [],
    "u32+": [],
    "rotr32": [],
    "shr": [],
    "Ch": [],
    "Maj": [],
    "Sigma0": ["rotr32"],
    "Sigma1": ["rotr32"],
    "sigma0": ["rotr32", "shr"],
    "sigma1": ["rotr32", "shr"],
    "H-init": [],
    "K": [],
    "iota": [],
    "pad-message": [],
    "make-schedule": ["u32+", "sigma0", "sigma1"],
    "compress": ["u32+", "Sigma1", "Ch", "Sigma0", "Maj", "K"],
    "sha256": ["pad-message", "make-schedule", "compress", "H-init"],
    "sha256-hex": ["sha256", "iota"],
    "hash->hex": ["iota"],
    "hex->hash": ["hex-digit"],
    "hex-digit": [],
}

FUNCTION_SPECS = {
    "iota": "Return list `(0 ... n-1)`.",
    "pad-message": "SHA-256 pad: append 0x80, zeroes, and 64-bit big-endian bit length to 64-byte boundary.",
    "make-schedule": "Build 64-word message schedule from one 64-byte block.",
    "compress": "Run one SHA-256 compression round over state H and schedule W.",
    "sha256": "Compute 32-byte SHA-256 digest bytevector.",
    "sha256-hex": "Compute lowercase hex SHA-256 digest string.",
    "hash->hex": "Convert digest bytevector to lowercase hex string.",
    "hex->hash": "Convert even-length hex string to bytevector.",
    "hex-digit": "Convert a hexadecimal character to integer 0..15, else 0.",
}

SKELETONS = {
    "iota": """(define (iota n)
  ;; TODO: generate 0..n-1
  <TODO>)""",
    "pad-message": """(define (pad-message msg)
  ;; TODO: SHA-256 message padding
  <TODO>)""",
    "make-schedule": """(define (make-schedule msg offset)
  ;; TODO: build 64-word schedule
  <TODO>)""",
    "compress": """(define (compress H W)
  ;; TODO: 64-round SHA-256 compression
  <TODO>)""",
    "sha256": """(define (sha256 msg)
  ;; TODO: hash bytevector message
  <TODO>)""",
    "sha256-hex": """(define (sha256-hex msg)
  ;; TODO: hex digest from sha256
  <TODO>)""",
    "hash->hex": """(define (hash->hex hash)
  ;; TODO: bytes to lowercase hex
  <TODO>)""",
    "hex->hash": """(define (hex->hash hex)
  ;; TODO: lowercase/uppercase hex string to bytes
  <TODO>)""",
    "hex-digit": """(define (hex-digit c)
  ;; TODO: decode a single hex char
  <TODO>)""",
}

VERIFY_BY_FUNCTION = {
    "iota": "(and (equal? (iota 0) '()) (equal? (iota 5) '(0 1 2 3 4)))",
    "pad-message": "(let* ([m (string->utf8 \"abc\")] [p (pad-message m)]) (and (= (bytevector-length p) 64) (= (bytevector-u8-ref p 0) 97) (= (bytevector-u8-ref p 1) 98) (= (bytevector-u8-ref p 2) 99) (= (bytevector-u8-ref p 3) #x80) (= (bytevector-u64-ref p 56 'big) 24)))",
    "make-schedule": "(let* ([p (pad-message (string->utf8 \"abc\"))] [w (make-schedule p 0)]) (and (= (fxvector-length w) 64) (= (fxvector-ref w 0) #x61626380) (= (fxvector-ref w 15) 24) (= (fxvector-ref w 16) #x61626380)))",
    "compress": "(let* ([p (pad-message (string->utf8 \"\"))] [w (make-schedule p 0)] [h (compress H-init w)]) (and (= (fxvector-length h) 8) (= (fxvector-ref h 0) #xe3b0c442) (= (fxvector-ref h 1) #x98fc1c14) (= (fxvector-ref h 2) #x9afbf4c8) (= (fxvector-ref h 3) #x996fb924) (= (fxvector-ref h 4) #x27ae41e4) (= (fxvector-ref h 5) #x649b934c) (= (fxvector-ref h 6) #xa495991b) (= (fxvector-ref h 7) #x7852b855)))",
    "sha256": "(let* ([h1 (sha256 (string->utf8 \"abc\"))] [h2 (sha256 (string->utf8 \"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq\"))]) (and (= (bytevector-length h1) 32) (= (bytevector-u8-ref h1 0) #xba) (= (bytevector-u8-ref h1 31) #xad) (equal? (hash->hex h2) \"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1\")))",
    "sha256-hex": "(and (equal? (sha256-hex (string->utf8 \"\")) \"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\") (equal? (sha256-hex (string->utf8 \"abc\")) \"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\") (equal? (sha256-hex (string->utf8 \"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq\")) \"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1\"))",
    "hash->hex": "(equal? (hash->hex (bytevector #x00 #x01 #xfe #xff)) \"0001feff\")",
    "hex->hash": "(let ([b (hex->hash \"0001feff\")]) (and (= (bytevector-length b) 4) (= (bytevector-u8-ref b 0) 0) (= (bytevector-u8-ref b 1) 1) (= (bytevector-u8-ref b 2) 254) (= (bytevector-u8-ref b 3) 255)))",
    "hex-digit": "(and (= (hex-digit #\\0) 0) (= (hex-digit #\\9) 9) (= (hex-digit #\\a) 10) (= (hex-digit #\\f) 15) (= (hex-digit #\\A) 10) (= (hex-digit #\\F) 15) (= (hex-digit #\\x) 0))",
}

PYTHON_SNIPPETS = {
    "iota": "def iota(n):\n    return list(range(0, n))",
    "pad-message": "def pad_message(msg: bytes) -> bytes:\n    bit_len = len(msg) * 8\n    padded = bytearray(msg)\n    padded.append(0x80)\n    while (len(padded) + 8) % 64 != 0:\n        padded.append(0)\n    padded.extend(bit_len.to_bytes(8, 'big'))\n    return bytes(padded)",
    "make-schedule": "def make_schedule(msg: bytes, offset: int):\n    W = [0] * 64\n    for i in range(16):\n        W[i] = int.from_bytes(msg[offset + 4*i:offset + 4*i + 4], 'big')\n    for i in range(16, 64):\n        W[i] = (sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16]) & 0xffffffff\n    return W",
    "compress": "def compress(H, W):\n    a,b,c,d,e,f,g,h = H\n    for i in range(64):\n        T1 = (h + Sigma1(e) + Ch(e,f,g) + K[i] + W[i]) & 0xffffffff\n        T2 = (Sigma0(a) + Maj(a,b,c)) & 0xffffffff\n        h,g,f,e,d,c,b,a = g,f,e,(d + T1) & 0xffffffff,c,b,a,(T1 + T2) & 0xffffffff\n    return [\n        (H[0] + a) & 0xffffffff, (H[1] + b) & 0xffffffff,\n        (H[2] + c) & 0xffffffff, (H[3] + d) & 0xffffffff,\n        (H[4] + e) & 0xffffffff, (H[5] + f) & 0xffffffff,\n        (H[6] + g) & 0xffffffff, (H[7] + h) & 0xffffffff,\n    ]",
    "sha256": "def sha256(msg: bytes) -> bytes:\n    padded = pad_message(msg)\n    H = H_INIT[:]\n    for i in range(0, len(padded), 64):\n        H = compress(H, make_schedule(padded, i))\n    out = bytearray()\n    for word in H:\n        out.extend(word.to_bytes(4, 'big'))\n    return bytes(out)",
    "sha256-hex": "def sha256_hex(msg: bytes) -> str:\n    return sha256(msg).hex()",
    "hash->hex": "def hash_to_hex(hash_bytes: bytes) -> str:\n    return hash_bytes.hex()",
    "hex->hash": "def hex_to_hash(hex_str: str) -> bytes:\n    return bytes.fromhex(hex_str)",
    "hex-digit": "def hex_digit(c: str) -> int:\n    if '0' <= c <= '9':\n        return ord(c) - ord('0')\n    if 'a' <= c <= 'f':\n        return 10 + ord(c) - ord('a')\n    if 'A' <= c <= 'F':\n        return 10 + ord(c) - ord('A')\n    return 0",
}

BUGGY_CASES = [
    {"fn": "iota", "buggy": "(define (iota n)\n  (let loop ([i 1] [acc '()])\n       (if (> i n)\n           (reverse acc)\n           (loop (+ i 1) (cons i acc)))))", "note": "Sequence must start at 0 and end at n-1."},
    {"fn": "pad-message", "buggy": "(define (pad-message msg)\n  (let* ([len (bytevector-length msg)] [result (make-bytevector (+ len 1) 0)])\n        (bytevector-copy! msg 0 result 0 len)\n        (bytevector-u8-set! result len #x80)\n        result))", "note": "Must pad to 64-byte boundary and append 64-bit length."},
    {"fn": "make-schedule", "buggy": "(define (make-schedule msg offset)\n  (let ([W (make-fxvector 64 0)])\n       (do ([i 0 (+ i 1)])\n           ((= i 16))\n           (fxvector-set! W i (bytevector-u32-ref msg (+ offset (* i 4)) 'big)))\n       W))", "note": "Schedule must extend words 16..63 using sigma functions."},
    {"fn": "compress", "buggy": "(define (compress H W)\n  H)", "note": "Must run 64 rounds and add working state back into H."},
    {"fn": "sha256", "buggy": "(define (sha256 msg)\n  (let* ([padded (pad-message msg)] [H (fxvector-copy H-init)] [W (make-schedule padded 0)])\n        (set! H (compress H W))\n        (let ([result (make-bytevector 32)])\n             (do ([i 0 (+ i 1)]) ((= i 8))\n                 (bytevector-u32-set! result (* i 4) (fxvector-ref H i) 'big))\n             result)))", "note": "Must process every 64-byte block, not only the first."},
    {"fn": "sha256-hex", "buggy": "(define (sha256-hex msg)\n  (string-upcase (hash->hex (sha256 msg))))", "note": "Hex output must be lowercase."},
    {"fn": "hash->hex", "buggy": "(define (hash->hex hash)\n  (apply string-append (map number->string (bytevector->u8-list hash))))", "note": "Must emit fixed-width hexadecimal pairs."},
    {"fn": "hex->hash", "buggy": "(define (hex->hash hex)\n  (string->utf8 hex))", "note": "Must decode hexadecimal digits into bytes."},
    {"fn": "hex-digit", "buggy": "(define (hex-digit c)\n  (if (char<=? #\\0 c #\\9) (- (char->integer c) (char->integer #\\0)) 0))", "note": "Must support both lowercase and uppercase A-F."},
]

DIFFICULTY = {
    "iota": "easy",
    "pad-message": "medium",
    "make-schedule": "hard",
    "compress": "hard",
    "sha256": "hard",
    "sha256-hex": "medium",
    "hash->hex": "easy",
    "hex->hash": "medium",
    "hex-digit": "easy",
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
    sid = f"core_sha256_{family}_{family_counter[family]:03d}"
    sample = {
        "id": sid,
        "family": family,
        "category": category,
        "difficulty": difficulty,
        "source_module": SOURCE_MODULE,
        "source_test": SOURCE_TEST,
        "source_function": source_function,
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
    names = FUNCTION_ORDER + SUPPORT_ORDER
    return [name for name in names if name != fn and name in tokens]


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
# Family 1: spec_to_code (18)
# -----------------------------------------------------------------------------
for fn in FUNCTION_ORDER:
    add_sample(
        family="spec_to_code",
        category="implementation",
        difficulty=DIFFICULTY[fn],
        source_function=fn,
        prompt=f"""You are implementing SHA-256 utilities in Fold-native Scheme.

Target module: {SOURCE_MODULE}
Function: `{fn}`
Spec: {FUNCTION_SPECS[fn]}

Write exactly one Scheme definition for `{fn}`.
Return only code, no explanation.""",
        ground_truth=DEFS[fn],
        verify_expr=def_verify(fn),
        tags=["core", "base", "sha256", "spec-to-code", fn],
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
        tags=["core", "base", "sha256", "skeleton-completion", fn],
    )


# -----------------------------------------------------------------------------
# Family 2: translation (9)
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
        tags=["core", "base", "sha256", "python-to-scheme", fn],
    )


# -----------------------------------------------------------------------------
# Family 3: bugfix (9)
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
        tags=["core", "base", "sha256", "bugfix", fn],
    )


# -----------------------------------------------------------------------------
# Family 4: composition (18)
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
        tags=["core", "base", "sha256", "composition", source_function] + extra_tags,
    )


composition_cases = [
    ("iota", "Generate iota for 6.", "(iota 6)", "(equal? (iota 6) '(0 1 2 3 4 5))", "easy", ["direct"]),
    ("iota", "Return length of iota 10.", "(length (iota 10))", "(equal? (length (iota 10)) 10)", "easy", ["integration"]),
    ("pad-message", "Return padded byte length for `abc`.", "(bytevector-length (pad-message (string->utf8 \"abc\")))", "(equal? (bytevector-length (pad-message (string->utf8 \"abc\"))) 64)", "medium", ["direct"]),
    ("pad-message", "Check empty-message pad length is one block.", "(bytevector-length (pad-message (string->utf8 \"\")))", "(equal? (bytevector-length (pad-message (string->utf8 \"\"))) 64)", "medium", ["edge-case"]),
    ("make-schedule", "Read W[0] for the padded `abc` block.", "(fxvector-ref (make-schedule (pad-message (string->utf8 \"abc\")) 0) 0)", "(equal? (fxvector-ref (make-schedule (pad-message (string->utf8 \"abc\")) 0) 0) #x61626380)", "hard", ["direct"]),
    ("make-schedule", "Read W[15] for the padded `abc` block.", "(fxvector-ref (make-schedule (pad-message (string->utf8 \"abc\")) 0) 15)", "(equal? (fxvector-ref (make-schedule (pad-message (string->utf8 \"abc\")) 0) 15) 24)", "hard", ["direct"]),
    ("compress", "Compute first state word after compressing empty block.", "(fxvector-ref (compress H-init (make-schedule (pad-message (string->utf8 \"\")) 0)) 0)", "(equal? (fxvector-ref (compress H-init (make-schedule (pad-message (string->utf8 \"\")) 0)) 0) #xe3b0c442)", "hard", ["direct"]),
    ("compress", "Return compression state length.", "(fxvector-length (compress H-init (make-schedule (pad-message (string->utf8 \"\")) 0)))", "(equal? (fxvector-length (compress H-init (make-schedule (pad-message (string->utf8 \"\")) 0))) 8)", "hard", ["property"]),
    ("sha256", "Compute SHA-256 byte length for `abc`.", "(bytevector-length (sha256 (string->utf8 \"abc\")))", "(equal? (bytevector-length (sha256 (string->utf8 \"abc\"))) 32)", "medium", ["direct"]),
    ("sha256", "Hash empty message and convert to hex.", "(hash->hex (sha256 (string->utf8 \"\")))", "(equal? (hash->hex (sha256 (string->utf8 \"\"))) \"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\")", "hard", ["integration"]),
    ("sha256-hex", "Compute SHA-256 hex digest for `abc`.", "(sha256-hex (string->utf8 \"abc\"))", "(equal? (sha256-hex (string->utf8 \"abc\")) \"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\")", "medium", ["direct"]),
    ("sha256-hex", "Hash the 448-bit NIST vector in hex.", "(sha256-hex (string->utf8 \"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq\"))", "(equal? (sha256-hex (string->utf8 \"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq\")) \"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1\")", "hard", ["direct"]),
    ("hash->hex", "Convert bytes 00 01 fe ff to hex.", "(hash->hex (bytevector #x00 #x01 #xfe #xff))", "(equal? (hash->hex (bytevector #x00 #x01 #xfe #xff)) \"0001feff\")", "easy", ["direct"]),
    ("hash->hex", "Roundtrip bytes through hash->hex then hex->hash.", "(hash->hex (hex->hash \"deadbeef\"))", "(equal? (hash->hex (hex->hash \"deadbeef\")) \"deadbeef\")", "medium", ["roundtrip"]),
    ("hex->hash", "Decode `deadbeef` and read byte length.", "(bytevector-length (hex->hash \"deadbeef\"))", "(equal? (bytevector-length (hex->hash \"deadbeef\")) 4)", "medium", ["direct"]),
    ("hex->hash", "Decode uppercase hex and re-encode lowercase.", "(hash->hex (hex->hash \"DEADBEEF\"))", "(equal? (hash->hex (hex->hash \"DEADBEEF\")) \"deadbeef\")", "medium", ["integration"]),
    ("hex-digit", "Decode uppercase hex digit A.", "(hex-digit #\\A)", "(equal? (hex-digit #\\A) 10)", "easy", ["direct"]),
    ("hex-digit", "Decode invalid hex digit x as 0.", "(hex-digit #\\x)", "(equal? (hex-digit #\\x) 0)", "easy", ["edge-case"]),
]

for fn, prompt, gt, verify, diff, tags in composition_cases:
    add_composition(fn, prompt, gt, verify, diff, tags)

if len([s for s in samples if s["family"] == "spec_to_code"]) != 18:
    raise ValueError("spec_to_code family must contain exactly 18 samples")
if len([s for s in samples if s["family"] == "translation"]) != 9:
    raise ValueError("translation family must contain exactly 9 samples")
if len([s for s in samples if s["family"] == "bugfix"]) != 9:
    raise ValueError("bugfix family must contain exactly 9 samples")
if len([s for s in samples if s["family"] == "composition"]) != 18:
    raise ValueError("composition family must contain exactly 18 samples")
if len(samples) != 54:
    raise ValueError(f"expected 54 samples, got {len(samples)}")


# -----------------------------------------------------------------------------
# Split
# -----------------------------------------------------------------------------
by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
for s in samples:
    by_family[str(s["family"])].append(s)

EVAL_QUOTA = {
    "spec_to_code": 3,
    "translation": 2,
    "bugfix": 2,
    "composition": 4,
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

if len(train_rows) != 43 or len(eval_rows) != 11:
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
