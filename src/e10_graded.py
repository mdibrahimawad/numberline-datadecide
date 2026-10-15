"""E10 stage E (exploratory): frequency in 3, 5 and 10 levels, 41 barcode features, 7 models.

Levels: ordinary 4-digit numbers (no 1500-2099, no multiples of 10, no special-pattern numbers),
ranked inside their block of 100 by the count left after log10 count ~ block + tens digit +
last digit (control "block_digits_plain"), then cut into Q equal levels per block
(level 0 = rarest). Every level has the same magnitudes and digit forms.

For each model, depth, Q and level: K subsamples of 120 numbers (2 per block, the same 60 blocks
for every level of a subsample), PH (Rips, Euclidean / layer scale), 41 features. Also the same
with the levels shuffled inside each block (null).

  python -m src.e10_graded --data <topology_number_line>/data --out results/e10_graded
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

import src.e10_explore as ex
from src.topo_features import FEATURE_NAMES, barcode_features, diagrams

MODELS = ["crystal", "openllama-3b", "starcoderbase-3b", "qwen2.5-1.5b", "smollm2-1.7b", "pythia-1.4b", "olmo2-1b"]
DEPTH = ["~25%", "~50%", "~75%", "~84%", "last"]
QS = (3, 5, 10)
M = 120


def layer_files(data, model):
    folder = os.path.join(data, f"{model}_vectors")
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("layer_"))


def levels(cnt, Q):
    mt = ex.matched_tiers(cnt, "block_digits_plain")
    size = mt.groupby("block")["n"].transform("size")
    mt["level"] = (mt["rank"] * Q // size).astype(int)
    return mt


def subsample(mt, lab, Q, rng, per_block=2, block_pool=None):
    pool_b = np.unique(mt["block"].values) if block_pool is None else np.asarray(block_pool)
    blocks = rng.choice(pool_b, M // per_block, replace=False)
    out = {q: [] for q in range(Q)}
    n, b = mt["n"].values, mt["block"].values
    for blk in blocks:
        inb = b == blk
        for q in range(Q):
            out[q].extend(rng.choice(n[inb & (lab == q)], per_block, replace=False))
    return {q: np.array(v) for q, v in out.items()}


_X: dict = {}
_S: dict = {}


def _init(data):
    for m in MODELS:
        for d, f in enumerate(layer_files(data, m)):
            X = np.load(f)["X"]                                  # kept as float16 (memory: 7 models x 5 layers)
            _X[(m, d)] = X
            rs = np.random.default_rng(0).choice(ex.FOUR, 1000, replace=False)
            _S[(m, d)] = float(np.median(pdist(X[rs - 1].astype(np.float32))))


def _feat(job):
    m, d, idx = job
    return barcode_features(diagrams(_X[(m, d)][idx - 1].astype(np.float32), _S[(m, d)]))


def compute(pool, cnt, K=30, seed=11):
    rng = np.random.default_rng(seed)
    jobs, meta = [], []
    for Q in QS:
        mt = levels(cnt, Q)
        blocks = mt["block"].values
        for shuffled in (False, True):
            for k in range(K):
                lab = mt["level"].values.copy()
                if shuffled:
                    for blk in np.unique(blocks):
                        i = np.where(blocks == blk)[0]; lab[i] = rng.permutation(lab[i])
                sub = subsample(mt, lab, Q, rng)
                for m in MODELS:
                    for d in range(5):
                        for q in range(Q):
                            jobs.append((m, d, sub[q]))
                            meta.append(dict(Q=Q, shuffled=shuffled, rep=k, model=m, depth=DEPTH[d], level=q,
                                             median_logcount=float(np.median(mt.set_index("n").loc[sub[q], "logc"]))))
    feats = pool.map(_feat, jobs, chunksize=64)
    return pd.DataFrame([{**a, **b} for a, b in zip(meta, feats)])


def compute_clf(pool, cnt, K=24, seed=12):
    """Sets for the classifiers: the blocks are split into two fixed halves and every set is
    drawn from one half only (40 blocks x 3 numbers), so train and test never share a number."""
    rng = np.random.default_rng(seed)
    jobs, meta = [], []
    for Q in (3, 10):
        mt = levels(cnt, Q)
        ub = np.unique(mt["block"].values); rng.shuffle(ub)
        halves = (ub[: len(ub) // 2], ub[len(ub) // 2:])
        for h, hb in enumerate(halves):
            for k in range(K):
                sub = subsample(mt, mt["level"].values, Q, rng, per_block=3, block_pool=hb)
                for m in MODELS:
                    for d in range(5):
                        for q in range(Q):
                            jobs.append((m, d, sub[q]))
                            meta.append(dict(Q=Q, half=h, rep=k, model=m, depth=DEPTH[d], level=q))
    feats = pool.map(_feat, jobs, chunksize=64)
    return pd.DataFrame([{**a, **b} for a, b in zip(meta, feats)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/e10_graded")
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="features,clf")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cnt = ex.load_counts()
    for Q in QS:
        lv = levels(cnt, Q)
        print(f"Q={Q}: median count per level", lv.groupby("level")["logc"].median().map(lambda x: round(10 ** x)).tolist(), flush=True)
    with Pool(a.workers, initializer=_init, initargs=(a.data,)) as pool:
        if "features" in a.only:
            df = compute(pool, cnt, K=a.K)
            df.to_csv(os.path.join(a.out, "features.csv.gz"), index=False, compression="gzip")
            print("features done", len(df), flush=True)
        if "clf" in a.only:
            c = compute_clf(pool, cnt)
            c.to_csv(os.path.join(a.out, "clf_features.csv.gz"), index=False, compression="gzip")
            print("clf features done", len(c), flush=True)
    json.dump(dict(models=MODELS, depth=DEPTH, Qs=QS, K=a.K, m=M, features=FEATURE_NAMES), open(os.path.join(a.out, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
