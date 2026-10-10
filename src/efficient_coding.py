"""Efficient-coding test at one fixed layer: does the number line the training counts
predict (with no fitting) match the β the model has?

    python -m src.efficient_coding            # layer 8 -> results/beta_fine/efficient_coding_L8.csv
    python -m src.efficient_coding --layer 9

Efficient coding (Ganguli & Simoncelli 2014): an optimal code gives each value space in
proportion to p(n)^q, with q set by what the code optimises, not fitted:
    q = 1    infomax (histogram equalisation)
    q = 1/2  minimum mean absolute error
    q = 1/3  minimum mean squared error
For each model the predicted line is F_q(n) = sum_{m <= n} c(m)^q, with c the exact counts of
its 100B training stream. F_q is read at the same prompt numbers β is measured on (the 10
bands) and fitted with the same curve a + s * B**log10(n); B is the predicted β. Also shown:
the closed form for q = 1 if the counts were an exact power law c ∝ n^α: β = 10**(1 + α).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.beta_fine import N_GROUPS, band, fit_continuous
from src.decade_metrics import load_counts

QS = {"infomax_q1": 1.0, "mae_q1/2": 0.5, "mse_q1/3": 1 / 3}
COUNTS = Path("results/corpus_alpha_datadecide")
TABLE = Path("results/beta_fine/fixed_layer_beta.csv")


def prompt_numbers() -> np.ndarray:
    """The integers of the 10 β bands that have counts (n <= 10000)."""
    return np.array(sorted({n for g in range(N_GROUPS) for n in band(g) if n <= 10000}))


def coded_beta(counts: np.ndarray, q: float, ns: np.ndarray) -> float:
    """β of the line F_q(n) = sum_{m<=n} c(m)^q (unseen numbers count as 0.5)."""
    c = np.maximum(np.asarray(counts[:10001], float), 0.0)
    space = np.where(c > 0, c, 0.5) ** q
    return fit_continuous(np.log10(ns) - 1, np.cumsum(space)[ns])[0]


def loo_mae(x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out error of a straight line y = a + b x (for comparison with the α line)."""
    err = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        b, a = np.polyfit(x[keep], y[keep], 1)
        err.append(abs(a + b * x[i] - y[i]))
    return float(np.mean(err))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    d = pd.read_csv(TABLE, index_col=0)
    counts = load_counts(COUNTS)
    ns = prompt_numbers()
    beta = d[f"beta_cont_L{args.layer}"]
    t = pd.DataFrame({"beta": beta, "alpha_ols": d["alpha_ols"]})
    for name, q in QS.items():
        t[name] = [coded_beta(counts[r], q, ns) for r in t.index]
    t["powerlaw_q1"] = 10 ** (1 + t["alpha_ols"])
    out = Path(args.out or f"results/beta_fine/efficient_coding_L{args.layer}.csv")
    t.to_csv(out)
    print(f"layer {args.layer}, {len(t)} models, measured β mean {beta.mean():.3f}")
    print(f"{'prediction':12s} {'mean':>6s} {'bias':>7s} {'MAE':>6s} {'r':>6s} {'p':>8s} {'rho':>6s} {'LOO-MAE':>8s}")
    for col in [*QS, "powerlaw_q1", "alpha_ols"]:
        x = t[col].to_numpy()
        r, pr = stats.pearsonr(x, beta)
        s, _ = stats.spearmanr(x, beta)
        direct = col != "alpha_ols"   # α is not a β prediction by itself: no level to compare
        print(f"{col:12s} {x.mean() if direct else float('nan'):6.3f} "
              f"{(x - beta).mean() if direct else float('nan'):+7.3f} "
              f"{np.abs(x - beta).mean() if direct else float('nan'):6.3f} "
              f"{r:+6.2f} {pr:8.1e} {s:+6.2f} {loo_mae(x, beta.to_numpy()):8.3f}")
    print(f"[efficient-coding] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
