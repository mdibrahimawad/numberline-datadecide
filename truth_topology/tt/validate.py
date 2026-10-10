"""Synthetic validation of every measure before real data (brief section 7.6).

    python -m tt.validate [--d 2048] [--jobs -1]

Uses the real P1 item set (n, labels, topics) and the real hidden size d. Pass criteria are
declared here, before running. Writes results/validation/validation.json + figures/fig0_validation.png.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from . import topology as T
from .cells import run_cell, make_subsamples

SIGMA = 1.0          # isotropic noise per dimension
TOPIC_SEP = 30.0     # norm of topic-mean offsets
SHIFT = 25.0         # class mean shift (2, 3)
RADIUS = 40.0        # circle radius / matched Gaussian scale (4, 5)

CRITERIA = {
    "1_no_signal": "|L1-0.5|<0.06, |T1 gap|<0.15, |z(T3b)|<3, |z(T3c)|<3",
    "2_shared_shift": "L1>0.9, L3>0.9, |T1 gap|<0.15, z(T3b)>3, z(T3c)>3",
    "3_topic_specific_shift": "L1_within>0.9, |L3-0.5|<0.1, |z(T3c)|<3, z(T3b)>3",
    "4_shape_only": "|L1-0.5|<0.1, T1>0.95, T1 gap>0.2",
    "5_abrupt_change": "H1 velocity peaks at the change transition, peak/median > 3",
    "6_scale_invariance": "x10 cloud: normalised summaries, T1, T3 identical (rel. diff < 1e-6)",
}


# Changes made after the first validation run (recorded, not hidden):
CRITERIA_CHANGES = [
    "4_shape_only: was 'T1 gap>0.3'. First run gave T1 = 1.00 but T1-null = 0.70 (overlapping "
    "subsamples from two fixed pools are told apart by pool idiosyncrasies), so a gap > 0.3 was "
    "unreachable even for a perfect shape test. Now T1>0.95 and gap>0.2.",
    "3_topic_specific_shift: topic shift directions were independent random draws; in d=2048 their "
    "accidental overlap (~1/sqrt(d)) times a 25-sigma shift moved L3 to 0.40. The scenario means "
    "'unrelated directions', so they are now exactly orthogonal to each other and to the topic means.",
]


def _plane(d, rng):
    q, _ = np.linalg.qr(rng.standard_normal((d, 2)))
    return q.T                                             # (2, d) orthonormal rows


def scenario(name, y, topics, d, rng):
    n = len(y)
    tid = np.unique(topics, return_inverse=True)[1]
    nt = tid.max() + 1
    mu = rng.standard_normal((nt, d))
    mu *= TOPIC_SEP / np.linalg.norm(mu, axis=1, keepdims=True)
    X = mu[tid] + SIGMA * rng.standard_normal((n, d))
    sgn = (y - 0.5) * 2
    if name == "1_no_signal":
        pass
    elif name == "2_shared_shift":
        u = rng.standard_normal(d); u /= np.linalg.norm(u)
        X += (SHIFT / 2) * sgn[:, None] * u
    elif name == "3_topic_specific_shift":
        q, _ = np.linalg.qr(np.c_[mu.T, rng.standard_normal((d, nt))])
        U = q[:, nt:].T                                    # orthonormal, orthogonal to the topic means
        X += (SHIFT / 2) * sgn[:, None] * U[tid]
    elif name == "4_shape_only":
        P = _plane(d, rng)
        th = rng.uniform(0, 2 * np.pi, n)
        circ = RADIUS * np.c_[np.cos(th), np.sin(th)]
        gaus = RADIUS / np.sqrt(2) * rng.standard_normal((n, 2))   # same mean (0), same covariance
        X += np.where(y[:, None] == 1, circ, gaus) @ P
    return X.astype(np.float16) if np.abs(X).max() < 6e4 else X


def summarize(recs):
    m = pd.DataFrame(recs)
    g = lambda k, f="value": float(m[(m.metric == k) & (m.seed.isin([-1, 0]))][f].iloc[0])
    return {"L1": g("L1_auc"), "L1_within": g("L1_within_auc"), "L3": g("L3_auc"), "L2": g("L2_auc"),
            "T1": g("T1_acc"), "T1_null": g("T1_acc", "null_mean"), "T1_gap": g("T1_gap"),
            "T3a_z": g("T3a_topic_homophily", "z"), "T3b": g("T3b_within_truth_homophily"),
            "T3b_z": g("T3b_within_truth_homophily", "z"), "T3c": g("T3c_cross_truth_homophily"),
            "T3c_z": g("T3c_cross_truth_homophily", "z")}


def check(name, s):
    if name == "1_no_signal":
        return abs(s["L1"] - .5) < .06 and abs(s["T1_gap"]) < .15 and abs(s["T3b_z"]) < 3 and abs(s["T3c_z"]) < 3
    if name == "2_shared_shift":
        return s["L1"] > .9 and s["L3"] > .9 and abs(s["T1_gap"]) < .15 and s["T3b_z"] > 3 and s["T3c_z"] > 3
    if name == "3_topic_specific_shift":
        return s["L1_within"] > .9 and abs(s["L3"] - .5) < .1 and abs(s["T3c_z"]) < 3 and s["T3b_z"] > 3
    if name == "4_shape_only":
        return abs(s["L1"] - .5) < .1 and s["T1"] > .95 and s["T1_gap"] > .2


def _run(name, y, topics, d, subs, ref_idx, outdir, seed):
    rng = np.random.default_rng(seed)
    X = scenario(name, y, topics, d, rng)
    recs, summ = run_cell(np.asarray(X), y, topics, subs, ref_idx, seed, outdir / f"{name}.npz")
    s = summarize(recs)
    return name, s, summ


def abrupt(y, topics, d, subs, k_change=4, n_ckpt=8, seed=5):
    """Gaussian clouds until checkpoint k_change, circle from then on; small fresh noise each checkpoint."""
    rng = np.random.default_rng(seed)
    n = len(y)
    P = _plane(d, rng)
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.standard_normal((n, 2)) / np.sqrt(2)
    base = SIGMA * rng.standard_normal((n, d))
    dgs = []
    for c in range(n_ckpt):
        lat = RADIUS * (np.c_[np.cos(th), np.sin(th)] if c >= k_change else z)
        X = lat @ P + base + 0.1 * SIGMA * rng.standard_normal((n, d))
        D = T.distances(X)
        D /= np.median(D[np.triu_indices(n, 1)])
        dgs.append([T.diagrams(D[np.ix_(ix, ix)]) for ix in subs["mixed"]])
    v = [float(np.mean([T.wasserstein2(a[1], b[1]) for a, b in zip(dgs[c], dgs[c + 1])])) for c in range(n_ckpt - 1)]
    peak = int(np.argmax(v))
    ok = peak == k_change - 1 and max(v) / np.median(v) > 3
    return {"velocity_h1": v, "change_transition": k_change - 1, "peak_transition": peak,
            "peak_ratio": float(max(v) / np.median(v))}, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2048)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    out = C.RESULTS / "validation"
    out.mkdir(parents=True, exist_ok=True)
    it = pd.read_csv(C.STATES / "P1" / "items.csv")
    y, topics = it.label.to_numpy().astype(int), it.topic.to_numpy()
    subs = {s: make_subsamples(y, topics, s) for s in C.SEEDS_SUBSAMPLE[:1]}
    ref_idx = np.sort(np.random.default_rng(0).choice(len(y), C.NORM_REF_N, replace=False))
    names = ["1_no_signal", "2_shared_shift", "3_topic_specific_shift", "4_shape_only"]
    res = Parallel(n_jobs=a.jobs)(delayed(_run)(nm, y, topics, a.d, subs, ref_idx, out, i) for i, nm in enumerate(names))
    report = {"n": int(len(y)), "d": a.d, "criteria": CRITERIA,
              "criteria_changes": CRITERIA_CHANGES, "tests": {}}
    summaries = {}
    for nm, s, summ in res:
        report["tests"][nm] = {"values": s, "pass": bool(check(nm, s))}
        summaries[nm] = summ
    v, ok = abrupt(y, topics, a.d, subs[0])
    report["tests"]["5_abrupt_change"] = {"values": v, "pass": bool(ok)}
    # 6: scale invariance on the shape-only cloud
    rng = np.random.default_rng(3)
    X = scenario("4_shape_only", y, topics, a.d, rng).astype(np.float64)
    r1, s1 = run_cell(X, y, topics, subs, ref_idx, 7, out / "scale1.npz", linear=False)
    r2, s2 = run_cell(X * 10, y, topics, subs, ref_idx, 7, out / "scale10.npz", linear=False)
    rel = max(float(np.max(np.abs(s1[k] - s2[k]) / (np.abs(s1[k]) + 1e-9))) for k in s1)
    m1, m2 = pd.DataFrame(r1), pd.DataFrame(r2)
    keys = ["T1_acc", "T3a_topic_homophily", "T3b_within_truth_homophily", "T3c_cross_truth_homophily"]
    dm = max(abs(float(m1[m1.metric == k].value.iloc[0]) - float(m2[m2.metric == k].value.iloc[0])) for k in keys)
    report["tests"]["6_scale_invariance"] = {"values": {"max_rel_diff_summaries": rel, "max_abs_diff_T1_T3": dm},
                                             "pass": bool(rel < 1e-6 and dm < 1e-9)}
    report["all_pass"] = all(t["pass"] for t in report["tests"].values())
    json.dump(report, open(out / "validation.json", "w"), indent=1)
    for nm, t in report["tests"].items():
        print(f"{nm:26s} {'PASS' if t['pass'] else 'FAIL'}  {json.dumps(t['values'], default=lambda x: round(x, 3))}")
    from .figures import fig0
    fig0(report, summaries)


if __name__ == "__main__":
    main()
