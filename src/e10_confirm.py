"""E10 stage B-A: the pre-registered replication tests (P1-P5), and nothing else.

Definitions, layers, predictions and pass rules are locked in
experiments/E10_frequency_topology/README.md, section "Stage B-A". Do not change this file after
the new vectors are seen; any extra analysis goes elsewhere and is labelled exploratory.

Run:  python -m src.e10_confirm --model openllama-3b --vectors <dir> --layers 7,13,20,22,26 \
          --out results/e10_confirm/openllama-3b
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import src.e10_explore as ex
from src.topo_features import barcode_features, diagrams

CONTROL = "block_digits_plain"
SPECIAL = ["thousands x000", "repdigits 1111-9999", "powers of 2", "years 1950-2025"]


def norm_test(cnt, vec_dir, layers, n_boot=500, seed=0):
    """P3: Spearman(frequency left after block + digits, vector length), block bootstrap CI."""
    mt = ex.matched_tiers(cnt, CONTROL)
    rel = (mt["key"] - mt.groupby("block")["key"].transform("mean")).values
    blocks = mt["block"].values; ub = np.unique(blocks)
    by_b = {b: np.where(blocks == b)[0] for b in ub}
    rng = np.random.default_rng(seed); rows = []
    for L in layers:
        X = np.load(os.path.join(vec_dir, f"layer_{L:02d}.npz"))["X"].astype(np.float32)
        nrm = np.linalg.norm(X[mt["n"].values - 1], axis=1)
        r = spearmanr(rel, nrm).correlation
        bs = []
        for _ in range(n_boot):
            pick = np.concatenate([by_b[b] for b in rng.choice(ub, len(ub))])
            bs.append(spearmanr(rel[pick], nrm[pick]).correlation)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(layer=L, rho=r, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def years_test(vec_dir, layer):
    """P5: years 1900-2025 vs every window of 126 consecutive ordinary 4-digit numbers starting
    at 2100, 2150, ... (overall spread = median pairwise distance / layer scale; H0 total with
    distances divided by the set's own median distance, i.e. scale-free)."""
    X = np.load(os.path.join(vec_dir, f"layer_{layer:02d}.npz"))["X"].astype(np.float32)
    scale = float(np.median(pdist(X[np.random.default_rng(0).choice(ex.FOUR, 1000, replace=False) - 1])))

    def f(idx):
        D = squareform(pdist(X[idx - 1].astype(np.float64))); med = np.median(D[np.triu_indices(len(idx), 1)])
        h0 = barcode_features(diagrams(X[idx - 1], med))
        return med / scale, h0["H0_total"]

    yrs = np.arange(1900, 2026)
    g = f(yrs)
    null = np.array([f(np.arange(s, s + 126)) for s in range(2100, 10000 - 126, 50)])
    z = (np.array(g) - null.mean(0)) / null.std(0)
    return dict(layer=layer, spread_z=float(z[0]), h0_total_scalefree_z=float(z[1]), n_null=len(null))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    ex.LAYERS[:] = layers
    os.makedirs(a.out, exist_ok=True)
    cnt = ex.load_counts()
    res = {}
    for unit, name in ((False, "P1_raw"), (True, "P2_direction_only")):
        with Pool(a.workers, initializer=ex._init, initargs=(a.vectors, unit, layers)) as pool:
            B = ex.tier_test(pool, cnt, CONTROL)
        B.to_csv(os.path.join(a.out, f"{name}_tier_test.csv"), index=False)
        z = B[B.feature == "H0_death_mean"].set_index("layer")["z"]
        res[name] = dict(z=z.round(2).to_dict(), passed=bool((z >= 2).sum() >= 3))
        print(name, res[name], flush=True)
    N = norm_test(cnt, a.vectors, layers); N.to_csv(os.path.join(a.out, "P3_norm.csv"), index=False)
    res["P3_norm"] = dict(rho=N.set_index("layer")["rho"].round(3).to_dict(),
                          passed=bool((N["ci_lo"] > 0).sum() >= 3))
    print("P3_norm", res["P3_norm"], flush=True)
    F = ex.special_isolation(cnt, a.vectors); F.to_csv(os.path.join(a.out, "P4_special.csv"), index=False)
    first = F[(F.layer == layers[0]) & F.category.isin(SPECIAL)].set_index("category")["mean_z"]
    res["P4_special"] = dict(z_first_layer=first.round(2).to_dict(), passed=bool((first > 1).sum() >= 3))
    print("P4_special", res["P4_special"], flush=True)
    Y = years_test(a.vectors, layers[0]); res["P5_years"] = dict(**Y, passed=bool(Y["spread_z"] > 2 and Y["h0_total_scalefree_z"] < -2))
    print("P5_years", res["P5_years"], flush=True)
    json.dump(dict(model=a.model, layers=layers, control=CONTROL, results=res), open(os.path.join(a.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
