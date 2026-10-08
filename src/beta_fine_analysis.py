"""Fine beta (src.beta_fine, 25 models) vs exact-100B alpha and numbers seen, next to the
old 4-group beta.

    python -m src.beta_fine_analysis     # -> results/beta_fine/fine_vs_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.decade_metrics import FAMILY

BETAS = ["old_direct", "old_log", "beta_coarse", "beta_cont", "beta_fine",
         "beta_coarse_oldlayer", "beta_cont_oldlayer"]
PREDICTORS = ["alpha_ols", "alpha_mle", "log_numbers"]


def merge_tables(fine: Path, alpha: Path, old: Path) -> pd.DataFrame:
    """One row per recipe: fine beta, exact alpha / numbers seen, old 4-group beta, family."""
    old_beta = pd.read_csv(old)[["recipe", "beta_direct_mean", "beta_log_mean"]].rename(
        columns={"beta_direct_mean": "old_direct", "beta_log_mean": "old_log"})
    m = pd.read_csv(fine).merge(pd.read_csv(alpha), on="recipe").merge(old_beta, on="recipe")
    m["family"] = m.recipe.map(lambda r: FAMILY.get(r, r))
    m["log_numbers"] = np.log(m.integer_matches)
    return m


def _demean(m: pd.DataFrame, col: str) -> pd.Series:
    return m[col] - m.groupby("family")[col].transform("mean")


def correlations(m: pd.DataFrame) -> pd.DataFrame:
    """Pearson / Spearman, within-family, without Dolma, and the weakest r when any one
    family is dropped, for every (beta version, predictor) pair."""
    rows = []
    no_dolma = m[m.family != "dolma"]
    for b in BETAS:
        for x in PREDICTORS:
            r, p = stats.pearsonr(m[x], m[b])
            rw, pw = stats.pearsonr(_demean(m, x), _demean(m, b))
            rnd, pnd = stats.pearsonr(no_dolma[x], no_dolma[b])
            drop = [stats.pearsonr(m[m.family != f][x], m[m.family != f][b])[0] for f in m.family.unique()]
            rows.append((b, x, r, p, stats.spearmanr(m[x], m[b])[0], rw, pw, rnd, pnd, min(drop)))
    return pd.DataFrame(rows, columns=["beta", "predictor", "r", "p", "spearman", "within_r", "within_p",
                                       "r_noDolma", "p_noDolma", "min_r_dropfamily"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fine", default="results/beta_fine/step69369-seed-default/summary.csv")
    p.add_argument("--alpha", default="results/corpus_alpha_datadecide/alpha_exact_25.csv")
    p.add_argument("--old", default="results/datadecide/summary.csv")
    p.add_argument("--out", default="results/beta_fine/fine_vs_data.csv")
    args = p.parse_args(argv)

    m = merge_tables(Path(args.fine), Path(args.alpha), Path(args.old))
    pd.set_option("display.width", 250)
    print(len(m), "models;", m.family.value_counts().to_dict())
    print(m[["recipe", "family", "layer", "old_layer", "old_direct", "beta_coarse", "beta_cont", "r2_cont",
             "beta_fine", "err_fine", "beta_coarse_oldlayer", "beta_cont_oldlayer", "beta_cont_std"]]
          .round(3).to_string(index=False))
    print(f"\nsame layer as old: {(m.layer == m.old_layer).sum()}/{len(m)}; "
          f"layers chosen: {m.layer.value_counts().sort_index().to_dict()}")
    print(f"median seed std  coarse {m.beta_coarse_std.median():.3f} cont {m.beta_cont_std.median():.3f} "
          f"fine {m.beta_fine_std.median():.3f} | median err coarse {m.err_coarse.median():.2f} "
          f"fine {m.err_fine.median():.2f} | median r2_cont {m.r2_cont.median():.2f}")
    print("\ncorrelation between beta versions:")
    print(m[BETAS].corr().round(2).to_string())
    print()
    print(correlations(m).round(3).to_string(index=False))

    print("\nwithin each family, r(alpha_ols, beta):")
    for fam, g in m.groupby("family"):
        if len(g) >= 4:
            print(f"  {fam:8s} n={len(g)} " + "  ".join(
                f"{b}={stats.pearsonr(g.alpha_ols, g[b])[0]:+.2f}"
                for b in ["old_direct", "beta_coarse", "beta_cont", "beta_cont_oldlayer"]))

    X = np.column_stack([np.ones(len(m)), m.alpha_ols, m.log_numbers])
    for b in ["old_direct", "beta_coarse", "beta_cont"]:
        coef, *_ = np.linalg.lstsq(X, m[b], rcond=None)
        r2 = 1 - ((m[b] - X @ coef) ** 2).sum() / ((m[b] - m[b].mean()) ** 2).sum()
        print(f"{b}: beta ~ alpha_ols + log_numbers  R2={r2:.2f}  "
              f"coefs alpha {coef[1]:+.3f} lognum {coef[2]:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
