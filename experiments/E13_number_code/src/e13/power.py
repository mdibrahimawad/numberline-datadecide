"""Planted-structure power checks at the real noise level (G4). For a real dataset Z(n) (standardised PCA scores),
noise = the real residual rows permuted across numbers; the planted signal has the real variance of the part it
replaces. Each test returns the recovery rate (planted present) and the false-positive rate (planted absent).
H1 (templates, decay) and H2 (carries, tree vs line) power come from the full-tier cells (E13 Stage-1 code) and are
copied into power_check.csv by `rows_from_cell`."""
from __future__ import annotations

import numpy as np

from src import e13_stage1 as E1
from src.e12c_heldout_r2 import fit_cd, kernel_pred, pred_cd

from .panel import choose_h, knee_curve, knee_stats, latent_features, n_digits, positions_for, r2


def _split(n, seed):
    te = np.random.default_rng(seed).random(len(n)) < 0.2
    return ~te, te


def _m1_pass(Z, n, seed):
    tr, te = _split(n, seed); mu = Z[tr].mean(0); h = choose_h(Z[tr], n[tr], seed)
    a = r2(Z[te], kernel_pred(Z[tr], n[tr], n[te], h), mu)
    ns = np.random.default_rng(seed + 5).permutation(n[tr]); b = r2(Z[te], kernel_pred(Z[tr], ns, n[te], h), mu)
    return a - b > 0.05


def _m2_pass(Z, n, s, seed):
    L = n_digits(s); pos = positions_for(L); tr, te = _split(n, seed); mu = Z[tr].mean(0); h = choose_h(Z[tr], n[tr], seed)
    a = r2(Z[te], kernel_pred(Z[tr], n[tr], n[te], h), mu)
    b = r2(Z[te], pred_cd(fit_cd(Z[tr], n[tr], h, pos), n[tr], n[te], h, pos), mu)
    return b - a > 0.02


def parts(Z, n, s, seed=0):
    """Real smooth part f, digit part D and residual E (in-sample backfit), used to build planted data."""
    L = n_digits(s); pos = positions_for(L); h = choose_h(Z, n, seed); m = fit_cd(Z, n, h, pos)
    from src.e12b_smooth_digits import digit_design
    f = kernel_pred(m["base"], n, n, h); D = digit_design(n, pos, m["means"])[0] @ m["coef"]
    return f, D, Z - f - D


def power_m1_m2_p8_h3(Z, n, s, tag, reps=10):
    f, D, E = parts(Z, n, s); L = n_digits(s); rows = []; mu = Z.mean(0)
    hit = dict(m1=0, m1_fp=0, m2=0, m2_fp=0, p8=0, h3=0, h3_fp=0)
    for rep in range(reps):
        r = np.random.default_rng(E1.seed_of(tag, "power", rep)); noise = E[r.permutation(len(n))]
        hit["m1"] += _m1_pass(mu + f + noise, n, rep)
        hit["m1_fp"] += _m1_pass((mu + f + noise)[r.permutation(len(n))], n, rep)
        hit["m2"] += _m2_pass(mu + f + D + noise, n, s, rep)
        hit["m2_fp"] += _m2_pass(mu + f + noise, n, s, rep)
        # P8: plant a rank-3 smooth latent (first 3 PCs of the fitted f + D) at the real noise
        F = f + D; U, S, Vt = np.linalg.svd(F - F.mean(0), full_matrices=False); F3 = (U[:, :3] * S[:3]) @ Vt[:3]
        tr, te = _split(n, rep); cur = knee_curve(mu + F3 + noise, n, L, tr, te); ks = knee_stats(cur)
        hit["p8"] += ks["knee"] is not None and 2 <= ks["knee"] <= 4
        # H3: a lexicon shared by both halves with the residual's variance (true reliability 0.5, above the 0.30
        # threshold) + independent real-noise rows in each half
        lex = E[r.permutation(len(n))]; e2 = E[r.permutation(len(n))]
        from .h3 import residual_reliability
        A = mu + f + D + lex + noise; B = mu + f + D + lex + e2
        hit["h3"] += residual_reliability(A, B, n, s, tr, te, rep) >= 0.30
        hit["h3_fp"] += residual_reliability(mu + f + D + noise, mu + f + D + e2, n, s, tr, te, rep) >= 0.30
    rate = {k: v / reps for k, v in hit.items()}
    for test, k, fp in (("M1 smooth curve", "m1", "m1_fp"), ("M2 digit-additive", "m2", "m2_fp"),
                        ("P8 knee (rank-3 planted)", "p8", None), ("H3 lexicon reliability", "h3", "h3_fp")):
        rec = rate[k]; f_ = rate[fp] if fp else None
        rows.append(dict(test=test, recovered=rec, false_positive=f_, reps=reps,
                         has_power=bool(rec >= 0.8 and (f_ is None or f_ <= 0.2))))
    return rows


def rows_from_cell(cell, reps_h1=20, reps_ab=20, reps_c=10):
    """H1 / H2 power from a full-tier panel cell."""
    rows = []; h1 = cell.get("h1") or {}
    if h1.get("template_recovery"):
        for k, v in h1["template_recovery"].items():
            rows.append(dict(test=f"H1 template {k}", recovered=v, false_positive=None, reps=reps_h1, has_power=bool(v >= 0.8)))
        rows.append(dict(test="H1 decay rule", recovered=h1["decay_power"], false_positive=h1["decay_fp"], reps=reps_h1,
                         has_power=h1["has_power"]))
    h2 = cell.get("h2") or {}
    if h2.get("power_ab"):
        p = h2["power_ab"]
        rows.append(dict(test="H2a transient carry spike", recovered=p["a_transient"], false_positive=p["a_none"], reps=reps_ab,
                         has_power=bool(p["a_transient"] >= 0.8 and p["a_none"] <= 0.2)))
        rows.append(dict(test="H2a cumulative carry jump", recovered=p["a_jump"], false_positive=p["a_none"], reps=reps_ab,
                         has_power=bool(p["a_jump"] >= 0.8 and p["a_none"] <= 0.2)))
        rows.append(dict(test="H2b successor break (cumulative jump)", recovered=p["b_jump"], false_positive=p["b_none"],
                         reps=reps_ab, has_power=bool(p["b_jump"] >= 0.8 and p["b_none"] <= 0.2)))
    if h2.get("power_c"):
        p = h2["power_c"]
        rows.append(dict(test="H2c tree vs line", recovered=p["tree"], false_positive=p["line"], reps=reps_c,
                         has_power=bool(p["tree"] >= 0.8 and p["line"] <= 0.2)))
    return rows
