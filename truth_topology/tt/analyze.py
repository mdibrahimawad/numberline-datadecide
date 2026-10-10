"""Analysis of extracted states (CPU; resumable; uses all cores).

    python -m tt.analyze --model pythia-1.4b-deduped --input P1 [--stages cells velocity mst aggregate floor]

results/<model>/<input>/
  config.json, subsamples.npz
  cells/step<N>_L<l>.json|.npz   per-cell metrics and diagrams (resume unit)
  summaries/step<N>_L<l>.npz     41-feature vectors (true, false, mixed, null pseudo-classes)
  metrics.csv     long: step, layer, seed, metric, value, null_mean, null_sd, z, p (+ ci for effect sizes)
  velocity.csv    consecutive-checkpoint velocities + geometric baselines
  concentration.csv
  onset.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

from . import config as C
from . import topology as T
from .cells import make_subsamples, run_cell, unpack
from .linear import b1, _m

log = logging.getLogger("analyze")


class Paths:
    def __init__(self, model, inp):
        self.states = C.STATES / model / inp
        self.items = C.STATES / inp / "items.csv"
        self.res = C.RESULTS / model / inp
        self.cells = self.res / "cells"
        self.summ = self.res / "summaries"
        for p in (self.res, self.cells, self.summ):
            p.mkdir(parents=True, exist_ok=True)

    def steps(self):
        return sorted(int(p.stem[4:]) for p in self.states.glob("step*.npz") if ".tmp" not in p.name)


def load_items(P):
    it = pd.read_csv(P.items)
    return it, it.label.to_numpy().astype(int), it.topic.to_numpy()


def fixed_indices(P, y, topics):
    f = P.res / "subsamples.npz"
    if f.exists():
        z = np.load(f)
        subs = {s: {k[len(f"s{s}_"):]: z[k] for k in z.files if k.startswith(f"s{s}_")} for s in C.SEEDS_SUBSAMPLE}
        return subs, z["ref_idx"]
    subs = {s: make_subsamples(y, topics, s) for s in C.SEEDS_SUBSAMPLE}
    ref_idx = np.sort(np.random.default_rng(C.SEED_DATA).choice(len(y), min(C.NORM_REF_N, len(y)), replace=False))
    np.savez(f, ref_idx=ref_idx, **{f"s{s}_{k}": v for s, d in subs.items() for k, v in d.items()})
    return subs, ref_idx


def _cell_job(X16, y, topics, subs, ref_idx, step, layer, P, linear):
    t0 = time.time()
    recs, summ = run_cell(X16, y, topics, subs, ref_idx, seed=C.SEED_PERM + step * 1000 + layer,
                          out_npz=P.cells / f"step{step}_L{layer}.npz", linear=linear)
    np.savez(P.summ / f"step{step}_L{layer}.npz", feature_names=np.array(T.FEATURE_NAMES), **summ)
    for r in recs:
        r.update(step=step, layer=layer)
    json.dump(recs, open(P.cells / f"step{step}_L{layer}.json", "w"), default=float)
    return time.time() - t0


def stage_cells(P, model, inp, layers=None, n_jobs=-1, linear=True):
    it, y, topics = load_items(P)
    subs, ref_idx = fixed_indices(P, y, topics)
    rng = np.random.default_rng(C.SEED_PERM)
    for step in P.steps():
        z = np.load(P.states / f"step{step}.npz")
        H = z["H"]
        Ls = range(H.shape[0]) if layers is None else layers
        todo = [l for l in Ls if not (P.cells / f"step{step}_L{l}.json").exists()]
        bfile = P.cells / f"step{step}_behaviour.json"
        if not bfile.exists():
            json.dump(behaviour(z, it, y, topics, rng, step), open(bfile, "w"), default=float)
        if not todo:
            continue
        t0 = time.time()
        times = Parallel(n_jobs=n_jobs)(delayed(_cell_job)(H[l], y, topics, subs, ref_idx, step, l, P, linear)
                                        for l in todo)
        log.info("%s %s step%d: %d cells in %.0fs (mean cell %.0fs)", model, inp, step, len(todo),
                 time.time() - t0, np.mean(times))


def behaviour(z, it, y, topics, rng, step):
    recs = []
    if "loglik" in z.files:
        recs.append(b1(z["loglik"], y, topics, 1000, rng))
    if "top1_correct" in z.files:
        tc, ent = z["top1_correct"], z["entropy"]
        recs.append(_m("B2_memorization_truectx", float(tc[y == 1].mean())))
        recs.append(_m("B2_memorization_all", float(tc.mean())))
        recs.append(_m("B2_p_correct_truectx", float(z["p_correct"][y == 1].mean())))
        pid = it.pair_id.to_numpy()
        e = pd.DataFrame({"pid": pid, "y": y, "ent": ent}).pivot(index="pid", columns="y", values="ent")
        dh = (e[0] - e[1]).to_numpy()
        null = []
        for _ in range(1000):  # sign-flip null for the paired difference
            null.append(float((dh * rng.choice([-1, 1], len(dh))).mean()))
        recs.append(_m("B2_deltaH", float(dh.mean()), null))
    for r in recs:
        r.update(step=step, layer=-1, seed=-1)
    return recs


# ---------------------------------------------------------------- velocity

def _velocity_layer(P, layer, steps):
    rows, pair = [], {}
    dg, geo = {}, {}
    for st in steps:
        zc = np.load(P.cells / f"step{st}_L{layer}.npz")
        geo[st] = (zc["centroid"].astype(np.float64), float(zc["scale"]))
        for s in C.SEEDS_SUBSAMPLE:
            for k in ("true", "false", "mixed"):
                dg[(st, s, k)] = unpack(zc, f"s{s}_{k}")
    met = pd.concat([pd.DataFrame(json.load(open(P.cells / f"step{st}_L{layer}.json"))) for st in steps])
    tv = met[met.metric == "geo_total_var"].set_index("step").value
    mpd = met[met.metric == "geo_mean_pair_dist"].set_index("step").value

    def vel(a, b, s, k, h, prep):
        A, Bd = dg[(a, s, k)], dg[(b, s, k)]
        fa = 1.0 if prep == "norm" else geo[a][1]
        fb = 1.0 if prep == "norm" else geo[b][1]
        return float(np.mean([T.wasserstein2(x[h] * fa, w[h] * fb) for x, w in zip(A, Bd)]))

    for a, b in zip(steps[:-1], steps[1:]):
        base = {"layer": layer, "step_from": a, "step_to": b,
                "geo_centroid_drift_raw": float(np.linalg.norm(geo[b][0] - geo[a][0])),
                "geo_centroid_drift_norm": float(np.linalg.norm(geo[b][0] / geo[b][1] - geo[a][0] / geo[a][1])),
                "geo_total_var_change_raw": float(abs(tv[b] - tv[a])),
                "geo_total_var_change_norm": float(abs(tv[b] / geo[b][1] ** 2 - tv[a] / geo[a][1] ** 2)),
                "geo_mean_pair_dist_change_raw": float(abs(mpd[b] - mpd[a])),
                "geo_mean_pair_dist_change_norm": float(abs(mpd[b] / geo[b][1] - mpd[a] / geo[a][1]))}
        for s in C.SEEDS_SUBSAMPLE:
            for k in ("mixed", "true", "false"):
                for h in (0, 1):
                    for prep in ("norm", "raw"):
                        rows.append({**base, "seed": s, "kind": k, "hom": h, "prep": prep,
                                     "velocity": vel(a, b, s, k, h, prep)})
    # all-pairs distances for the checkpoint-order permutation null (mixed clouds)
    for s in C.SEEDS_SUBSAMPLE:
        for h in (0, 1):
            for prep in ("norm", "raw"):
                Dm = np.zeros((len(steps), len(steps)))
                for i in range(len(steps)):
                    for j in range(i + 1, len(steps)):
                        Dm[i, j] = Dm[j, i] = vel(steps[i], steps[j], s, "mixed", h, prep)
                pair[(s, h, prep)] = Dm
    return rows, {f"L{layer}_s{s}_h{h}_{p}": v for (s, h, p), v in pair.items()}


def concentration(Dm, steps, n_perm, rng, start=0):
    """C = share of path length up to FIRST_THIRD_STEP; null = random checkpoint orders."""
    st = np.array(steps)[start:]
    Dm = Dm[start:, start:]
    k = int(np.sum(st[1:] <= C.FIRST_THIRD_STEP))
    path = lambda o: np.array([Dm[o[i], o[i + 1]] for i in range(len(o) - 1)])
    v = path(np.arange(len(st)))
    c = v[:k].sum() / v.sum()
    null = []
    for _ in range(n_perm):
        pv = path(rng.permutation(len(st)))
        null.append(pv[:k].sum() / pv.sum())
    null = np.array(null)
    return {"C": float(c), "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p": float((np.sum(null >= c) + 1) / (n_perm + 1)), "n_first": k,
            "peak_step_to": int(st[1:][np.argmax(v)]), "peak_ratio": float(v.max() / np.median(v))}


def stage_velocity(P, n_jobs=-1):
    steps = P.steps()
    layers = sorted({int(p.stem.split("_L")[1]) for p in P.cells.glob(f"step{steps[0]}_L*.npz")})
    out = Parallel(n_jobs=n_jobs)(delayed(_velocity_layer)(P, l, steps) for l in layers)
    vel = pd.DataFrame([r for rows, _ in out for r in rows])
    vel.to_csv(P.res / "velocity.csv", index=False)
    pairs = {k: v for _, d in out for k, v in d.items()}
    np.savez(P.res / "velocity_pairwise.npz", steps=np.array(steps), **pairs)
    rng = np.random.default_rng(C.SEED_PERM)
    rows = []
    for key, Dm in pairs.items():
        layer, s, h, prep = key.split("_")
        for start, tag in ((0, "all"), (1, "excl_init")):
            rows.append({"layer": int(layer[1:]), "seed": int(s[1:]), "hom": int(h[1:]), "prep": prep,
                         "transitions": tag, **concentration(Dm, steps, C.N_PERM_CONC, rng, start)})
    pd.DataFrame(rows).to_csv(P.res / "concentration.csv", index=False)


# ---------------------------------------------------------------- MST max-statistic

def stage_mst(P):
    """Max-statistic correction over layers. Permutation k is the same relabelling at every layer
    (cells.t3_cell seeds it identically), so per permutation we take the max z over layers and
    compare the observed best layer's z to that distribution."""
    rows = []
    names = ["T3a_topic_homophily", "T3b_within_truth_homophily", "T3c_cross_truth_homophily"]
    for step in P.steps():
        files = sorted(P.cells.glob(f"step{step}_L*.npz"), key=lambda p: int(p.stem.split("_L")[1]))
        if not files:
            continue
        nulls = np.stack([np.load(f)["t3_null"] for f in files])            # (L, perm, 3)
        recs = pd.concat([pd.DataFrame(json.load(open(f.with_suffix(".json")))) for f in files])
        for i, n in enumerate(names):
            obs = recs[recs.metric == n].set_index("layer").value.reindex(range(len(files))).to_numpy()
            mu, sd = np.nanmean(nulls[:, :, i], 1), np.nanstd(nulls[:, :, i], 1)
            sd = np.where(sd > 0, sd, np.nan)
            zobs = (obs - mu) / sd
            zperm = (nulls[:, :, i] - mu[:, None]) / sd[:, None]
            maxnull = np.nanmax(zperm, 0)
            best = int(np.nanargmax(zobs)) if np.isfinite(zobs).any() else -1
            p = float((np.sum(maxnull >= zobs[best]) + 1) / (len(maxnull) + 1)) if best >= 0 else np.nan
            rows.append({"step": step, "layer": best, "seed": -1, "metric": f"{n}_maxz_bestlayer",
                         "value": float(zobs[best]) if best >= 0 else np.nan, "p": p})
    return rows


# ---------------------------------------------------------------- aggregate, onset

def stage_aggregate(P):
    recs = []
    for f in P.cells.glob("*.json"):
        recs += json.load(open(f))
    recs += stage_mst(P)
    m = pd.DataFrame(recs)
    cols = ["step", "layer", "seed", "metric", "value", "null_mean", "null_sd", "z", "p"]
    m = m[cols + [c for c in m.columns if c not in cols]].sort_values(["metric", "seed", "layer", "step"])
    m.to_csv(P.res / "metrics.csv", index=False)
    ref = reference_layer(m)
    json.dump({"reference_layer": ref, "rule": f"argmax over layers of mean L1_auc over the last "
               f"{C.REF_LAYER_LAST_K} checkpoints"}, open(P.res / "reference_layer.json", "w"))
    onset(m, ref).to_csv(P.res / "onset.csv", index=False)
    return m, ref


def reference_layer(m):
    l1 = m[(m.metric == "L1_auc")]
    if l1.empty:
        return None
    last = sorted(l1.step.unique())[-C.REF_LAYER_LAST_K:]
    return int(l1[l1.step.isin(last)].groupby("layer").value.mean().idxmax())


ONSET_METRICS = ["B1_loglik_auc", "B2_deltaH", "L1_auc", "L2_auc", "L1_within_auc", "L3_auc", "T1_acc",
                 "T3a_topic_homophily", "T3b_within_truth_homophily", "T3c_cross_truth_homophily"]


def onset(m, ref):
    """First checkpoint at which the metric exceeds its null 95th percentile (one-sided, normal fit)
    for ONSET_CONSECUTIVE consecutive checkpoints, at the reference layer (behaviour: layer -1)."""
    rows = []
    q = norm.ppf(0.95)
    for name in ONSET_METRICS:
        d = m[m.metric == name]
        if d.empty:
            continue
        lay = -1 if (d.layer == -1).all() else ref
        for seed in sorted(d.seed.unique()):
            s = d[(d.layer == lay) & (d.seed == seed)].sort_values("step")
            sig = (s.value > s.null_mean + q * s.null_sd).to_numpy()
            steps = s.step.to_numpy()
            first = next((steps[i] for i in range(len(sig) - C.ONSET_CONSECUTIVE + 1)
                          if sig[i:i + C.ONSET_CONSECUTIVE].all()), None)
            rows.append({"metric": name, "layer": lay, "seed": seed, "onset_step": first,
                         "onset_index": None if first is None else int(np.where(steps == first)[0][0]),
                         "sig_at_step0": bool(sig[0]) if len(sig) and steps[0] == 0 else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Gaussian floor

def stage_floor(P):
    z = np.load(P.states / f"step{P.steps()[-1]}.npz")
    d = z["H"].shape[-1]
    rng = np.random.default_rng(C.SEED_PERM)
    F = []
    for _ in range(C.B):
        X = rng.standard_normal((C.M, d))
        D = T.distances(X)
        D /= np.median(D[np.triu_indices(C.M, 1)])
        F.append(T.summary(T.diagrams(D)))
    F = np.array(F)
    out = {"m": C.M, "d": d, **{f: {"mean": float(F[:, T.FEATURE_NAMES.index(f)].mean()),
                                   "sd": float(F[:, T.FEATURE_NAMES.index(f)].std())}
                               for f in T.FEATURE_NAMES}}
    json.dump(out, open(P.res / "gaussian_floor.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-1.4b-deduped")
    ap.add_argument("--input", default="P1")
    ap.add_argument("--stages", nargs="+", default=["cells", "velocity", "aggregate", "floor"])
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    P = Paths(a.model, a.input)
    (C.WORK / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(C.WORK / "logs" / f"analyze_{a.model}_{a.input}.log"),
                                  logging.StreamHandler()])
    json.dump({**C.as_dict(), "model": a.model, "input": a.input}, open(P.res / "config.json", "w"), indent=1, default=str)
    for st in a.stages:
        t0 = time.time()
        if st == "cells":
            stage_cells(P, a.model, a.input, a.layers, a.jobs)
        elif st == "velocity":
            stage_velocity(P, a.jobs)
        elif st == "aggregate":
            stage_aggregate(P)
        elif st == "floor":
            stage_floor(P)
        log.info("stage %s done in %.0fs", st, time.time() - t0)


if __name__ == "__main__":
    main()
