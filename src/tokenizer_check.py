"""Do two models split numbers into the same tokens? (E10 stage D2 pre-condition)

For every 4-digit number n, takes the saved prompt text `...,n=` and records how the tokenizer splits
the characters of n (e.g. "198|7"). Two models "read numbers the same way" if the splits agree for all
9,000 numbers. Only tokenizers are downloaded.

  python -m src.tokenizer_check --prompts <tokenization.json.gz> --models EleutherAI/pythia-1.4b,togethercomputer/RedPajama-INCITE-Base-3B-v1
"""
from __future__ import annotations

import argparse
import json

from src.e11_tier1_laptop import original_prompts


def splits(tok, prompts):
    out = {}
    for n in range(1000, 10000):
        p = prompts[n]
        start = p.rfind(",") + 1                          # the target number starts after the last comma
        enc = tok(p, return_offsets_mapping=True, add_special_tokens=False)
        pieces = [p[a:b] for a, b in enc["offset_mapping"] if a >= start and b <= len(p) - 1 and b > a]
        out[n] = "|".join(x.strip() for x in pieces if x.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--models", required=True, help="comma-separated Hugging Face ids (2 or more)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    prompts = original_prompts(a.prompts)
    ids = a.models.split(",")
    S = {m: splits(AutoTokenizer.from_pretrained(m), prompts) for m in ids}
    ref = ids[0]
    res = {}
    for m in ids[1:]:
        diff = [n for n in range(1000, 10000) if S[m][n] != S[ref][n]]
        res[m] = dict(same_as=ref, n_different=len(diff), examples=[(n, S[ref][n], S[m][n]) for n in diff[:10]])
        print(f"{m} vs {ref}: {9000 - len(diff)} of 9000 four-digit numbers split identically", flush=True)
    for m in ids:
        print(m, "examples:", {n: S[m][n] for n in (1987, 5371, 9999, 2048, 1000)}, flush=True)
    if a.out:
        json.dump(dict(results=res, examples={m: {n: S[m][n] for n in (1987, 5371, 9999, 2048, 1000)} for m in ids}),
                  open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
