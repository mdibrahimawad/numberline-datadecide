"""E15 exploratory extension (not pre-registered): 3-digit numbers padded to 4 digits (123 -> 0123).

    python -m src.e15_extra_pad3 --model EleutherAI/pythia-1.4b --out <dir>

Same prompt, contexts (paired by value), read position and averaging as src/e15_length_mechanism.py.
Saves <dir>/<slug>_pad3.npz with the final-'=' hidden states of the plain and padded strings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.e15_length_mechanism import N_CTX, contexts


def items() -> list[dict]:
    rng = np.random.default_rng(2)
    out = []
    for n in sorted(rng.choice(np.arange(100, 1000), 100, replace=False).tolist()):
        out.append({"set": "pad3_plain", "x": str(n), "n": n, "k": 0})
        out.append({"set": "pad3", "x": str(n).zfill(4), "n": n, "k": 1})
    return out


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    its = items()
    H = None
    ntok = np.zeros(len(its), int)
    with torch.inference_mode():
        for i, it in enumerate(its):
            for a, b, c in contexts(it["n"]):
                prefix = f"{a}={a},{b}={b},{c}={c},"
                enc = tok(prefix + it["x"] + "=", return_offsets_mapping=True)
                q0, q1 = len(prefix), len(prefix) + len(it["x"])
                ntok[i] = sum(1 for s, e in enc["offset_mapping"] if e > s and s < q1 and e > q0)
                res = model(input_ids=torch.tensor([enc["input_ids"]]), output_hidden_states=True)
                hs = torch.stack(res.hidden_states)[:, 0, -1, :].double().numpy()
                if H is None:
                    H = np.zeros((len(its),) + hs.shape)
                H[i] += hs / N_CTX
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = args.model.split("/")[-1]
    np.savez_compressed(out / f"{slug}_pad3.npz", H=H.astype(np.float32), ntok=ntok, items=json.dumps(its))
    print("saved", out / f"{slug}_pad3.npz")


if __name__ == "__main__":
    main()
