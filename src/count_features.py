"""Features of each recipe's exact number counts (N = 0..10000) that might explain beta
beyond alpha, and how they relate to the beta left over after alpha.

    python -m src.count_features      # -> results/datadecide/count_features.csv

Exploratory: ~25 features vs 25 models, so about one will look significant by chance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.beta_fine import CENTRES, band
from src.confounder_analysis import source_shares

RARE_K = 4000   # the locked threshold of R (docs/project_overview.md); do not re-tune

DECADES = {"d1_0_9": (0, 10), "d2_10_99": (10, 100), "d3_100_999": (100, 1000), "d4_1000_9999": (1000, 10000)}


def load_counts(roots: list[Path]) -> dict[str, np.ndarray]:
    out = {}
    for root in roots:
        for d in sorted(root.glob("exact_100b_*")):
            f = d / "counts_seed_2.csv"
            if f.exists():
                c = np.zeros(10001)
                t = pd.read_csv(f)
                c[t["number"].to_numpy()] = t["count"].to_numpy()
                out[d.name.removeprefix("exact_100b_")] = c
    return out


def _slope(c: np.ndarray, lo: int, hi: int) -> float:
    n = np.arange(lo, hi)
    y = c[lo:hi]
    ok = y > 0
    return float(np.polyfit(np.log(n[ok]), np.log(y[ok]), 1)[0])


def _excess(c: np.ndarray, special: np.ndarray, width: int = 3) -> float:
    """Mean log ratio of special numbers to their +-width non-special neighbours."""
    sset = set(special.tolist())
    r = []
    for s in special:
        nb = [s + k for k in range(-width, width + 1) if k and 1 <= s + k <= 10000 and s + k not in sset]
        base = np.mean(c[nb]) if nb else 0
        if c[s] > 0 and base > 0:
            r.append(np.log(c[s] / base))
    return float(np.mean(r))


def features(c: np.ndarray) -> dict[str, float]:
    tot = c.sum()
    mid = c[10:10000].sum()
    f: dict[str, float] = {}
    # amount
    f["F01_total_numbers_B"] = tot / 1e9
    f["F02_numbers_10_9999_B"] = mid / 1e9
    # share per decade
    for i, (k, (lo, hi)) in enumerate(DECADES.items()):
        f[f"F0{3 + i}_share_{k}"] = c[lo:hi].sum() / tot
    f["F07_big_vs_small_d4_over_d2"] = c[1000:10000].sum() / c[10:100].sum()
    # local slopes (alpha inside each decade) and curvature
    f["F08_slope_10_99"] = _slope(c, 10, 100)
    f["F09_slope_100_999"] = _slope(c, 100, 1000)
    f["F10_slope_1000_9999"] = _slope(c, 1000, 10000)
    f["F11_curvature_slope3_minus_slope1"] = f["F10_slope_1000_9999"] - f["F08_slope_10_99"]
    # frequency tiers (same thresholds for every recipe), over 10..9999
    x = c[10:10000]
    for name, lo, hi in (("low_lt1k", 0, 1e3), ("mid_1k_100k", 1e3, 1e5), ("high_ge100k", 1e5, np.inf)):
        sel = (x >= lo) & (x < hi)
        f[f"F12_n_numbers_{name}"] = int(sel.sum())
        f[f"F13_mass_share_{name}"] = x[sel].sum() / mid
    f["F14_n_seen_lt_4000"] = int((x < RARE_K).sum())        # R: the rarely-seen count
    # shape statistics
    p = x / mid
    f["F15_entropy_norm"] = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p)))
    s = np.sort(x)
    f["F16_gini"] = float(1 - 2 * np.sum(np.cumsum(s) / s.sum()) / len(s) + 1 / len(s))
    n = np.arange(10, 10000)
    ok = x > 0
    a, b = np.polyfit(np.log(n[ok]), np.log(x[ok]), 1)
    f["F17_spikiness_rms_resid"] = float(np.sqrt(np.mean((np.log(x[ok]) - (a * np.log(n[ok]) + b)) ** 2)))
    # special numbers
    f["F18_round10_excess"] = _excess(c, np.arange(20, 10000, 10))
    f["F19_round100_excess"] = _excess(c, np.arange(200, 10000, 100), width=10)
    f["F20_year_share_of_d4"] = c[1900:2031].sum() / c[1000:10000].sum()
    f["F21_year_excess"] = float(np.log(c[1950:2031].mean() / np.r_[c[1800:1950], c[2031:2200]].mean()))
    f["F22_pow2_excess"] = _excess(c, np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]))
    f["F23_small_1_20_share"] = c[1:21].sum() / tot
    ld = np.array([c[10:10000][(np.arange(10, 10000) % 10) == k].sum() for k in range(10)]) / mid
    f["F24_last_digit_0_or_5"] = float(ld[0] + ld[5])
    # what the probe actually tests: counts in the prompt bands (counts stop at 10000, so the
    # top band 9000-11000 is cut to 9000-10000 and rescaled by its width)
    bc = np.array([c[band(g).start:min(band(g).stop, 10001)].sum() * len(band(g))
                   / (min(band(g).stop, 10001) - band(g).start) for g in range(len(CENTRES))])
    f["F25_probe_band_top_over_bottom"] = float(bc[-1] / bc[0])
    f["F26_probe_band_slope"] = float(np.polyfit(np.log(CENTRES), np.log(bc), 1)[0])
    return f


def partial_corr(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> tuple[float, float]:
    """Pearson correlation of x and y after regressing both on the controls (+ intercept)."""
    Z = np.column_stack([np.ones(len(y))] + list(controls))
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = len(y) - 2 - len(controls)
    return r, float(2 * stats.t.sf(abs(r * np.sqrt(dof / (1 - r * r))), dof))


def rare_count_checks(d: pd.DataFrame, counts: dict[str, np.ndarray], beta: str,
                      shares: dict[str, dict[str, float]]) -> None:
    """Robustness of R (F14) as a predictor of beta: within family, leave one family out,
    together with alpha, other thresholds, and the code/math-content confounder."""
    R, a, y, fam = d.F14_n_seen_lt_4000, d.alpha_ols, d[beta], d.family

    def demean(s):
        return s - s.groupby(fam).transform("mean")

    print(f"\nR = #{{n in 10..9999 : c(n) < {RARE_K}}} vs {beta}  (alpha_ols for comparison)")
    for name, x in (("R", R), ("alpha_ols", a)):
        r, p = stats.pearsonr(x, y)
        rho, prho = stats.spearmanr(x, y)
        rw, pw = stats.spearmanr(demean(x), demean(y))
        drop = [stats.spearmanr(x[fam != f], y[fam != f])[0] for f in fam.unique()]
        print(f"  {name:9s} pearson r={r:+.2f} (p={p:.1e})  spearman {rho:+.2f} (p={prho:.1e})  "
              f"within-family {rw:+.2f} (p={pw:.1e})  weakest leave-one-family-out {min(drop, key=abs):+.2f}")
    X = np.column_stack([np.ones(len(y)), a, R])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    se = np.sqrt(np.diag(resid @ resid / (len(y) - 3) * np.linalg.inv(X.T @ X)))
    pv = 2 * stats.t.sf(np.abs(coef / se), len(y) - 3)
    print(f"  together: beta ~ alpha + R   p(alpha)={pv[1]:.3f}  p(R)={pv[2]:.4f}")
    fit = stats.linregress(R, y)
    print(f"  fitted line: beta = {fit.intercept:.2f} {fit.slope * 1000:+.3f} x R/1000")
    print("  other thresholds K: " + "  ".join(
        f"{k}: {stats.spearmanr([(counts[r][10:10000] < k).sum() for r in d.index], y)[0]:+.2f}"
        for k in (1000, 2000, 4000, 8000)))
    code = np.array([shares.get(r, {}).get("code_share", 0.0) for r in d.index])
    math = np.array([shares.get(r, {}).get("math_share", 0.0) for r in d.index])
    print(f"  content: code share vs beta r={np.corrcoef(code, y)[0, 1]:+.2f}, "
          f"math share r={np.corrcoef(math, y)[0, 1]:+.2f}, code share vs R r={np.corrcoef(code, R)[0, 1]:+.2f}")
    for label, ctrl in (("code share", [code]), ("math share", [math]),
                        ("contains Dolma (code > 0)", [(code > 0).astype(float)])):
        (rr, pr), (ra, pa) = partial_corr(R.values, y.values, ctrl), partial_corr(a.values, y.values, ctrl)
        print(f"    controlling {label:26s} R: partial r={rr:+.2f} (p={pr:.3f})   alpha: {ra:+.2f} (p={pa:.3f})")
    sub = code == 0
    (rs, ps), (as_, pas) = stats.spearmanr(R[sub], y[sub]), stats.spearmanr(a[sub], y[sub])
    print(f"    {sub.sum()} recipes without code: R spearman {rs:+.2f} (p={ps:.3f})   alpha {as_:+.2f} (p={pas:.3f})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--counts", nargs="+", default=["results/corpus_alpha_datadecide"])
    p.add_argument("--table", default="results/beta_fine/fine_vs_data.csv")
    p.add_argument("--beta", default="old_direct")
    p.add_argument("--out", default="results/datadecide/count_features.csv")
    args = p.parse_args(argv)

    counts = load_counts([Path(r) for r in args.counts])
    m = pd.read_csv(args.table).set_index("recipe")
    rows = {r: features(c) for r, c in counts.items() if r in m.index}
    F = pd.DataFrame(rows).T
    d = m.join(F, how="inner")
    print(f"{len(d)} recipes with counts (missing: {sorted(set(m.index) - set(d.index))})")
    s, i, *_ = stats.linregress(d.alpha_ols, d[args.beta])
    d["beta_resid"] = d[args.beta] - (i + s * d.alpha_ols)
    out = d[["family", "alpha_ols", args.beta, "beta_resid"] + list(F.columns)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out)

    res = []
    for col in F.columns:
        r_b, p_b = stats.spearmanr(d[col], d[args.beta])
        r_res, p_res = stats.spearmanr(d[col], d.beta_resid)
        r_a = stats.spearmanr(d[col], d.alpha_ols)[0]
        res.append((col, r_b, p_b, r_res, p_res, r_a))
    t = pd.DataFrame(res, columns=["feature", "rho_beta", "p_beta", "rho_resid", "p_resid", "rho_alpha"])
    pd.set_option("display.width", 200)
    print(f"\nSpearman rho with {args.beta}, with beta left over after alpha (resid), and with alpha:")
    print(t.sort_values("p_resid").round(3).to_string(index=False))
    rare_count_checks(d, counts, args.beta, source_shares(Path(args.counts[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
