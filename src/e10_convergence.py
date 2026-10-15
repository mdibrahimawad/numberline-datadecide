"""E10 stage D: pre-registered test of "models converge on the topology of numbers in late
layers" (stage C, observation 2) on new models. Locked in experiments/E10_frequency_topology/
README.md, section "Stage D". Do not change after the new vectors are seen.

For every pair of models and each of the 5 relative depths: RTD on the same 120 ordinary 4-digit
numbers (2 per block of 100, 60 random blocks; no 1500-2099, no multiples of 10, no
special-pattern numbers), divided by RTD with model B's points shuffled (broken correspondence).
K = 10 random sets. Per pair: early = mean ratio at ~25 % and ~50 % depth, late = mean ratio at
~84 % and last. Prediction: late < early.

Run:  python -m src.e10_convergence --data <topology_number_line>/data --models crystal,...
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from math import comb

import numpy as np
import pandas as pd

import src.e10_explore as ex
from src.e10_topology import rtd

DEPTH = ["~25%", "~50%", "~75%", "~84%", "last"]
ORIGINAL = ["crystal", "openllama-3b", "starcoderbase-3b"]


def model_files(data, name):
    folder = os.path.join(data, f"{name}_vectors")
    files = sorted(f for f in os.listdir(folder) if f.startswith("layer_"))
    assert len(files) == 5, f"{name}: expected 5 layer files, found {files}"
    return [os.path.join(folder, f) for f in files]           # sorted by level = by depth


def ordinary_pool():
    n = ex.FOUR[~ex.year_zone(ex.FOUR) & (ex.FOUR % 10 != 0)]
    return n[~ex.is_special_pattern(n)]


def run(data, models, K=10, seed=7):
    rng = np.random.default_rng(seed)
    pool = ordinary_pool(); blocks = pool // 100; ub = np.unique(blocks)
    sets = []
    for _ in range(K):
        idx = []
        for b in rng.choice(ub, 60, replace=False):
            idx.extend(rng.choice(pool[blocks == b], 2, replace=False))
        sets.append(np.array(idx))
    perms = [rng.permutation(120) for _ in range(K)]
    rows = []
    for d in range(5):
        X = {m: np.load(model_files(data, m)[d])["X"].astype(np.float32) for m in models}
        for a, b in itertools.combinations(models, 2):
            for k, idx in enumerate(sets):
                Pa, Pb = X[a][idx - 1], X[b][idx - 1]
                rows.append(dict(depth=DEPTH[d], a=a, b=b, rep=k, rtd=rtd(Pa, Pb), rtd_shuffled=rtd(Pa, Pb[perms[k]])))
        print("depth", DEPTH[d], "done", flush=True)
    return pd.DataFrame(rows)


def score(df):
    g = df.groupby(["a", "b", "depth"])[["rtd", "rtd_shuffled"]].mean()
    ratio = (g["rtd"] / g["rtd_shuffled"]).unstack()[DEPTH]
    ratio["early"] = ratio[["~25%", "~50%"]].mean(1)
    ratio["late"] = ratio[["~84%", "last"]].mean(1)
    ratio["late_lower"] = ratio["late"] < ratio["early"]
    ratio = ratio.reset_index()
    ratio["new_pair"] = ~(ratio["a"].isin(ORIGINAL) & ratio["b"].isin(ORIGINAL))
    new = ratio[ratio.new_pair]
    n, s = len(new), int(new["late_lower"].sum())
    p = sum(comb(n, i) for i in range(s, n + 1)) / 2 ** n            # one-sided sign test
    return ratio, dict(new_pairs=n, late_lower=s, sign_test_p=p, passed=bool(p < 0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", default="results/e10_convergence")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = run(a.data, a.models.split(","))
    df.to_csv(os.path.join(a.out, "rtd_pairs.csv"), index=False)
    ratio, res = score(df)
    ratio.to_csv(os.path.join(a.out, "ratio_by_pair.csv"), index=False)
    json.dump(dict(models=a.models.split(","), **res), open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(ratio.round(2).to_string()); print(res)


if __name__ == "__main__":
    main()
