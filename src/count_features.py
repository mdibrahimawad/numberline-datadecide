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
from src.decade_metrics import FAMILY

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
    f["F14_n_seen_lt_4000"] = int((x < 4000).sum())          # below the saturation K
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
