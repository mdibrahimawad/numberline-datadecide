"""E13-1D causal tests T7 (necessary), T8 (sufficient), T9 (explains behaviour) - PREREG_1D.md.
Uses the curve built from the "Sentences" reading at the dev-chosen layer; edits the number's last token."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from . import causal as CZ
from .manifold1d import SETS, dev_split, fit_curve, place, prep_apply, prep_fit


def to_raw(T, vec50):
    """A direction in PCA-50 (standardised) space -> raw hidden-state space."""
    d = np.zeros_like(T["mu"]); d[T["keep"]] = (vec50 @ T["Vt"]) * T["sd"][T["keep"]]
    return d


def build(X, n, s):
    tr = ~dev_split(n); T = prep_fit(X[tr]); Z = prep_apply(T, X); lo, hi = SETS[s]
    cur = fit_curve(Z[tr], n[tr], lo, hi, 0)
    Q = np.stack([to_raw(T, v) for v in cur["V"]], 1); Q, _ = np.linalg.qr(Q)
    nh = place(cur, Z)
    return dict(T=T, cur=cur, Q=Q, mu_raw=X[tr].mean(0), nhat={int(a): float(b) for a, b in zip(n, nh)}, lo=lo, hi=hi)


def boot_ci(x, B=1000, seed=0):
    rng = np.random.default_rng(seed); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def partner(it):
    a, b = [int(t) for t in (it["cand_text"][0], it["cand_text"][1])]
    return b if a == it["n"] else a


def run(model, tok, family, X, n, s, layer, n_items=200, seed=0, log=print):
    rng = np.random.default_rng(seed); K = build(X, n, s); layer = max(1, int(layer))
    test = n[~dev_split(n)]
    b1 = CZ.build_items(tok, family, test, "B1", n_items, rng)
    for it in b1:
        it["cand_text"] = [tok.decode(c).strip() for c in it["cands"]]
    b1 = [it for it in b1 if all(t.isdigit() for t in it["cand_text"])]
    b2 = CZ.build_items(tok, family, test, "B2", n_items, rng)
    log(f"    causal-1D {s}: layer {layer}, curve dim m={K['cur']['m']}, items B1={len(b1)} B2={len(b2)}")
    out = dict(set=s, layer=layer, m=K["cur"]["m"], n_B1=len(b1), n_B2=len(b2))
    base1 = CZ.score(model, tok, b1); base2 = CZ.score(model, tok, b2)
    # ---- T7 necessary: ablate the curve subspace vs 3 random subspaces of the same dimension
    k = K["Q"].shape[1]; d = K["Q"].shape[0]
    abl = lambda Q: CZ.make_edit("ablate", Q, K["mu_raw"])
    e1 = base1 - CZ.score(model, tok, b1, abl(K["Q"]), layer); e2 = base2 - CZ.score(model, tok, b2, abl(K["Q"]), layer)
    r1, r2 = [], []
    for r in range(3):
        G = rng.standard_normal((d, k)); G -= K["Q"] @ (K["Q"].T @ G); G, _ = np.linalg.qr(G)
        r1.append(base1 - CZ.score(model, tok, b1, abl(G), layer)); r2.append(base2 - CZ.score(model, tok, b2, abl(G), layer))
    r1 = np.mean(r1, 0); r2 = np.mean(r2, 0)
    t_lo, _ = boot_ci(e1 - 2 * r1); _, o_hi = boot_ci(e2 - 1.2 * r2)
    out["T7"] = dict(B1_drop=float(e1.mean()), B1_random=float(r1.mean()), B2_drop=float(e2.mean()), B2_random=float(r2.mean()),
                     target_ci_lo=t_lo, offtarget_ci_hi=o_hi, passed=bool(t_lo > 0 and o_hi < 0))
    # ---- T8 sufficient: move the number along the curve to the other side of its partner
    ok = base1 > 0; items = [it for it, g in zip(b1, ok) if g]; deltas = []; rand = []
    g0 = K["cur"]["gamma"]; lo, hi = K["lo"], K["hi"]
    for it in items:
        b = partner(it); n2 = int(np.clip(2 * b - it["n"], lo, hi))
        if n2 == b:
            n2 = int(np.clip(b + (1 if it["n"] < b else -1), lo, hi))
        dv = to_raw(K["T"], (g0[n2 - lo] - g0[it["n"] - lo]) @ K["cur"]["V"]); deltas.append(dv)
        u = rng.standard_normal(d); rand.append(u / np.linalg.norm(u) * np.linalg.norm(dv))
    if items:
        steer = lambda D: CZ.make_edit("steer", None, None, delta=np.array(D))
        f_curve = CZ.score(model, tok, items, steer(deltas), layer) < 0
        f_rand = CZ.score(model, tok, items, steer(rand), layer) < 0
        diff = f_curve.astype(float) - 2 * f_rand.astype(float); lo_ci, _ = boot_ci(diff)
        out["T8"] = dict(n_items=len(items), flip_curve=float(f_curve.mean()), flip_random=float(f_rand.mean()),
                         ci_lo_curve_minus_2random=lo_ci, passed=bool(lo_ci > 0))
    # ---- T9 behaviour: B1 margin vs distance along the curve, controlling the log ratio
    a = np.array([it["n"] for it in b1]); bb = np.array([partner(it) for it in b1])
    dcur = np.abs(np.array([K["nhat"].get(int(x), np.nan) for x in a]) - np.array([K["nhat"].get(int(x), np.nan) for x in bb]))
    lr = np.abs(np.log(a) - np.log(bb)); keep = np.isfinite(dcur)
    rm, rd, rl = (rankdata(v[keep]) for v in (base1, dcur, lr))

    def pcor(i):
        X1 = np.c_[np.ones(len(i)), rl[i]]
        res_m = rm[i] - X1 @ np.linalg.lstsq(X1, rm[i], rcond=None)[0]; res_d = rd[i] - X1 @ np.linalg.lstsq(X1, rd[i], rcond=None)[0]
        return float(np.corrcoef(res_m, res_d)[0, 1])
    idx = np.arange(keep.sum()); pc = pcor(idx); bs = [pcor(rng.choice(idx, len(idx))) for _ in range(1000)]
    out["T9"] = dict(partial_spearman=pc, ci_lo=float(np.nanpercentile(bs, 2.5)), ci_hi=float(np.nanpercentile(bs, 97.5)),
                     passed=bool(np.nanpercentile(bs, 2.5) > 0), n=int(keep.sum()))
    return out
