"""E12 T9 positive control: days of the week and months, same prompt form as the numbers (PREREG T9).

Extract (CPU is enough: 19 items x 20 prompts):
  python -m src.e12_positive_control --extract --out results/e12_manifold
Analyse:
  python -m src.e12_positive_control --out results/e12_manifold
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

from src.e12_manifold import DEPTH, pca, standardise
from src.extract_number_vectors import levels_for

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
ITEMS = DAYS + MONTHS
MODELS = {"pythia-1.4b": "EleutherAI/pythia-1.4b", "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B"}


def prompts(seed=12, per_item=20):
    rng = random.Random(seed); out = []
    for it in ITEMS:
        for _ in range(per_item):
            demos = rng.sample([x for x in ITEMS if x != it], 4)
            out.append((it, ",".join(f"{d}={d}" for d in demos) + f",{it}="))
    return out


def extract(out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    P = prompts()
    for name, mid in MODELS.items():
        tok = AutoTokenizer.from_pretrained(mid); model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32).eval()
        levels = levels_for(model.config.num_hidden_layers); H = {L: [] for L in levels}
        with torch.no_grad():
            for _, p in P:
                hs = model(**tok(p, return_tensors="pt"), output_hidden_states=True).hidden_states
                for L in levels:
                    H[L].append(hs[L][0, -1].float().numpy())
        np.savez_compressed(os.path.join(out, f"positive_control_{name}.npz"), items=np.array([i for i, _ in P]),
                            **{f"L{L}": np.stack(v) for L, v in H.items()})
        print(name, "extracted", levels, flush=True)
        del model


def polygon_in_order(C):
    K = len(C); D = squareform(pdist(C)); np.fill_diagonal(D, np.inf); nn2 = np.argsort(D, 1)[:, :2]
    share = float(np.mean([set(nn2[i]) == {(i - 1) % K, (i + 1) % K} for i in range(K)]))
    return share if (K - 1 in nn2[0] and 0 in nn2[K - 1]) else 0.0


def analyse(out):
    rows = []
    for name in MODELS:
        z = np.load(os.path.join(out, f"positive_control_{name}.npz")); items = z["items"]
        for d, key in enumerate(sorted([k for k in z.files if k.startswith("L")], key=lambda k: int(k[1:]))):
            S = pca(standardise(z[key].astype(np.float64)))
            for cat, names in (("days", DAYS), ("months", MONTHS)):
                C = np.stack([S[items == it].mean(0) for it in names])
                dg = ripser(squareform(pdist(C)), distance_matrix=True, maxdim=1)["dgms"][1]
                med = float(np.median(pdist(C))); life = (dg[:, 1] - dg[:, 0]) / med if len(dg) else np.zeros(1)
                poly = polygon_in_order(C)
                rows.append(dict(model=name, depth=DEPTH[d], category=cat, long_H1=int(np.sum(life > 0.25)),
                                 max_H1_ratio=float(life.max()) if len(life) else 0.0, polygon_in_order=poly,
                                 passed=bool(np.sum(life > 0.25) >= 1 or poly >= 0.75)))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(out, "positive_control.csv"), index=False)
    print(T.round(3).to_string(index=False))
    for name in MODELS:
        m = T[(T.model == name) & (T.category == "months")]
        print(name, "control works (months pass at >= 2 of 5 depths):", int(m.passed.sum()) >= 2, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e12_manifold")
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.extract:
        extract(a.out)
    analyse(a.out)


if __name__ == "__main__":
    main()
