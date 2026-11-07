"""Prompts (METHODOLOGY 2.1) and read positions (2.2)."""
from __future__ import annotations

import gzip
import json

import numpy as np

from . import config as C

_C1 = None


def c1_seed42():
    """The stored E10 copy prompts (seed 42), identical for all models (same dataset hash)."""
    global _C1
    if _C1 is None:
        _C1 = {int(k): v for k, v in json.load(gzip.open(C.C1_PROMPTS, "rt")).items()}
    return _C1


def c1_generated(seed, n_max=C.N_MAX):
    """Copy prompts in the E10 format: A,B,C,D uniform 1-,2-,3-,4-digit numbers, never equal to n."""
    rng = np.random.default_rng(seed); out = {}
    for n in range(1, n_max + 1):
        demo = []
        for lo, hi in ((1, 10), (10, 100), (100, 1000), (1000, 10000)):
            v = int(rng.integers(lo, hi))
            while v == n:
                v = int(rng.integers(lo, hi))
            demo.append(v)
        out[n] = ",".join(f"{v}={v}" for v in demo) + f",{n}="
    return out


def build(ctx, n, c1s7=None):
    """Return (prompt, start, end) where prompt[start:end] == str(n) is the target number."""
    s = str(n)
    if ctx == "C1":
        p = c1_seed42()[n]
    elif ctx == "C1s7":
        p = (c1s7 or c1_generated(C.SEED_C1B))[n]
    else:
        tpl = C.CONTEXTS.get(ctx) or C.VARIANTS[ctx]
        p = tpl.replace("{n}", s)
        i = tpl.index("{n}")
        return p, i, i + len(s)
    assert p.endswith(s + "="), (ctx, n, p)
    return p, len(p) - 1 - len(s), len(p) - 1


def all_contexts(with_variants=True):
    return list(C.CONTEXTS) + (list(C.VARIANTS) if with_variants else [])


def positions(tok_fast, prompt, start, end, ids_model=None, add_special_tokens=True):
    """Token positions of the number via the fast tokenizer's offsets.

    Returns dict(ids, first, last, next, task, n_tokens, ok, reason). `ok` is False when the number's last token
    runs past the number (merged with the next character), when there is no next token, or when the model's own
    tokenizer gives different ids (e.g. slow vs fast)."""
    enc = tok_fast(prompt, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    ids = list(enc["input_ids"]); off = enc["offset_mapping"]
    idx = [i for i, (a, b) in enumerate(off) if a < end and b > start and b > a]
    out = dict(ids=ids, first=-1, last=-1, next=-1, task=len(ids) - 1, n_tokens=0, ok=False, reason="")
    if not idx:
        out["reason"] = "number not found"; return out
    first, last = idx[0], idx[-1]
    out.update(first=first, last=last, next=last + 1, n_tokens=len(idx))
    if off[last][1] != end:
        out["reason"] = "last number token runs past the number"; return out
    if off[first][0] < start and prompt[off[first][0]:start].strip():
        out["reason"] = "first number token starts inside the prefix"; return out
    if last + 1 >= len(ids):
        out["reason"] = "no token after the number"; out["next"] = -1; return out
    if ids_model is not None and list(ids_model) != ids:
        out["reason"] = "model tokenizer ids differ from fast tokenizer"; return out
    out["ok"] = True
    return out
