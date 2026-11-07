"""H0 invariance support (METHODOLOGY Part 3 H0 + user additions): noise ceilings (M3), variance decomposition
number / context / interaction, cross-context transfer of f + digits, and distance-matrix agreement.
Run per (model, stored layer, set)."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import spearmanr

from src import e13_stage1 as E1
from src.e12c_heldout_r2 import fit_cd, pred_cd

from . import config as C
from .panel import choose_h, n_digits, positions_for

MAIN = ["C1", "C1s7"] + [c for c in C.CONTEXTS if c not in ("C1", "C1s7")]


def _space(arrs, fit_rows=None, D=50):
    """Common standardisation + PCA fitted on the stacked arrays (rows fit_rows of each), applied to each."""
    S = np.vstack([a if fit_rows is None else a[fit_rows] for a in arrs])
    mu = S.mean(0); sd = S.std(0); keep = sd > 1e-8
    A = (S[:, keep] - mu[keep]) / sd[keep]; D = min(D, A.shape[0] - 1, A.shape[1])
    Vt = np.linalg.svd(A, full_matrices=False)[2][:D]
    return [((a[:, keep] - mu[keep]) / sd[keep]) @ Vt.T for a in arrs]


def reliability(A, B):
    """Split-half reliability between two noisy measurements of the same numbers: tr Cov(A,B) / sqrt(trVarA trVarB),
    plus the R2 of using A as a prediction of B (M3 wording)."""
    za, zb = _space([A, B]); za = za - za.mean(0); zb = zb - zb.mean(0)
    cov = (za * zb).sum(); va = (za ** 2).sum(); vb = (zb ** 2).sum()
    return dict(r=float(cov / np.sqrt(va * vb)), R2_half=float(1 - ((zb - za) ** 2).sum() / vb))


def ceilings(md, slot, n):
    out = {}
    for c in C.NATURAL:
        if f"{c}b" in C.VARIANTS:
            out[c] = reliability(md.X((c, "num"), slot, n), md.X((f"{c}b", "num"), slot, n))
    out["C1"] = reliability(md.X(("C1", "num"), slot, n), md.X(("C1s7", "num"), slot, n))
    out["C1_next"] = reliability(md.X(("C1", "next"), slot, n), md.X(("C1s7", "next"), slot, n))
    rs, r2s = [], []
    for k in (2,):
        for half in itertools.combinations(C.NEUTRAL, k):
            other = [c for c in C.NEUTRAL if c not in half]
            rr = reliability(md.half_average(list(half), slot, n), md.half_average(other, slot, n))
            rs.append(rr["r"]); r2s.append(rr["R2_half"])
    r = float(np.mean(rs)); out["AVG5"] = dict(r_half=r, r=2 * r / (1 + r), R2_half=float(np.mean(r2s)))
    return out


def variance_decomposition(md, slot, n, contexts=C.NATURAL):
    """Two-way decomposition of Z(n, c) (common standardised PCA space of all contexts): number main effect,
    context mean shift, number x context interaction; the second wordings give a wording-noise estimate."""
    arrs = [md.X((c, "num"), slot, n) for c in contexts]
    var = [md.X((f"{c}b", "num"), slot, n) for c in contexts if f"{c}b" in C.VARIANTS]
    Zs = _space(arrs + var); Z = np.stack(Zs[:len(contexts)], 1)           # [N, C, D]
    mu = Z.mean((0, 1)); a = Z.mean(1) - mu; b = Z.mean(0) - mu
    inter = Z - mu - a[:, None] - b[None]
    ss_n = Z.shape[1] * (a ** 2).sum(); ss_c = Z.shape[0] * (b ** 2).sum(); ss_i = (inter ** 2).sum()
    tot = ((Z - mu) ** 2).sum(); out = dict(share_number=float(ss_n / tot), share_context=float(ss_c / tot),
                                            share_interaction=float(ss_i / tot), n=int(len(n)), contexts=list(contexts))
    if len(var) == len(contexts):
        W = np.stack(Zs[len(contexts):], 1)
        dz = (Z - Z.mean(0)) - (W - W.mean(0)); noise = 0.5 * (dz ** 2).sum()
        out["share_wording_noise"] = float(noise / tot)
        out["share_interaction_beyond_noise"] = float(max(ss_i - noise, 0) / tot)
    return out


def transfer(md, slot, n, s, ceil, contexts=None, seed=0):
    """Centre each context; fit f + digits on context A (training numbers), predict held-out numbers of context B
    in A's standardised PCA space. R2 and R2 / ceiling(B)."""
    contexts = contexts or (MAIN + ["AVG5"])
    L = n_digits(s); pos = positions_for(L)
    te = np.random.default_rng(seed).random(len(n)) < 0.2; tr = ~te; out = []
    raw = {c: md.X((c, "num"), slot, n) for c in contexts}
    raw = {c: v - v[tr].mean(0) for c, v in raw.items()}
    for A in contexts:
        XA = raw[A]; mu = XA[tr].mean(0); sd = XA[tr].std(0); keep = sd > 1e-8
        AA = (XA[tr][:, keep] - mu[keep]) / sd[keep]; D = min(50, AA.shape[0] - 1)
        Vt = np.linalg.svd(AA - AA.mean(0), full_matrices=False)[2][:D]; m0 = AA.mean(0)
        proj = lambda X: ((X[:, keep] - mu[keep]) / sd[keep] - m0) @ Vt.T
        ZA = proj(XA); h = choose_h(ZA[tr], n[tr], seed); mdl = fit_cd(ZA[tr], n[tr], h, pos)
        P = pred_cd(mdl, n[tr], n[te], h, pos)
        for B in contexts:
            ZB = proj(raw[B]); r2 = float(1 - ((ZB[te] - P) ** 2).sum() / ((ZB[te] - ZB[tr].mean(0)) ** 2).sum())
            cb = ceil.get(B if B != "AVG5" else "AVG5", {}); cv = cb.get("r") if cb else None
            out.append(dict(A=A, B=B, R2=r2, ceiling_B=cv, R2_frac=(r2 / cv) if cv else None))
    return out


def distance_agreement(md, slot, n, contexts=None, n_pairs=20000, seed=0):
    contexts = contexts or (MAIN + list(C.VARIANTS) + ["AVG5"])
    rng = np.random.default_rng(seed); N = len(n)
    i = rng.integers(0, N, n_pairs); j = rng.integers(0, N, n_pairs); k = i != j; i, j = i[k], j[k]
    dist = {}
    for c in contexts:
        Z = E1.prep(md.X((c, "num"), slot, n)); dist[c] = np.linalg.norm(Z[i] - Z[j], axis=1)
    out = []
    for a, b in itertools.combinations(contexts, 2):
        out.append(dict(A=a, B=b, spearman=float(spearmanr(dist[a], dist[b])[0]),
                        same_context=(b == a + "b") or ({a, b} == {"C1", "C1s7"})))
    return out


def run(md, slot, s):
    ctxs = MAIN + list(C.VARIANTS)
    n = md.set_numbers(s, ctxs)
    if len(n) < 20:
        return dict(skipped=f"only {len(n)} numbers valid in every context")
    if s == "S4":
        from .data import subsample_s4
        n = subsample_s4(n)
    ceil = ceilings(md, slot, n)
    return dict(N=int(len(n)), ceilings=ceil, decomposition=variance_decomposition(md, slot, n),
                decomposition_with_C1=variance_decomposition(md, slot, n, ["C1"] + C.NATURAL),
                transfer=transfer(md, slot, n, s, ceil), distance_agreement=distance_agreement(md, slot, n))
