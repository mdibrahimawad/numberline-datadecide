"""B4 data statistics of the 64-token text (brief section 6.1)."""

from __future__ import annotations

import zlib

import numpy as np

B4_NAMES = ["zlib_ratio", "frac_unique_tokens", "longest_repeated_ngram", "cont_unigram_logfreq"]


def unigram_logfreq(store, steps: tuple[int, int], vocab_size: int) -> np.ndarray:
    """log relative frequency of every token id over all tokens of ``steps`` (add-one)."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    for st in range(steps[0], steps[1] + 1):
        counts += np.bincount(store.read_step(st).ravel(), minlength=vocab_size)[:vocab_size]
    return np.log((counts + 1) / (counts.sum() + vocab_size))


def longest_repeated_ngram(tokens: np.ndarray) -> int:
    """Largest n such that some n-gram occurs at least twice (overlaps allowed)."""
    t = [int(x) for x in tokens]
    lo, hi = 0, len(t) - 1
    while lo < hi:                   # repetition is monotone in n: binary search
        n = (lo + hi + 1) // 2
        seen, rep = set(), False
        for i in range(len(t) - n + 1):
            g = tuple(t[i:i + n])
            if g in seen:
                rep = True
                break
            seen.add(g)
        lo, hi = (n, hi) if rep else (lo, n - 1)
    return lo


def b4_row(tokens: np.ndarray, text: str, logfreq: np.ndarray, prompt_len: int) -> list[float]:
    raw = text.encode("utf-8", errors="replace")
    ratio = len(zlib.compress(raw, 9)) / max(len(raw), 1)
    return [ratio, len(set(tokens.tolist())) / len(tokens), float(longest_repeated_ngram(tokens)),
            float(logfreq[tokens[prompt_len:]].mean())]
