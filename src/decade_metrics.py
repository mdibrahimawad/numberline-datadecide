"""Decade-count metrics of the training data, shaped like beta, compared with model beta.

beta compares the gaps between the probe's group means at 10, 100, 1000, 10000:
Delta_{k+1} = beta * Delta_k, and with three gaps the direct fit is beta = sqrt(Delta_3 / Delta_1).
Each metric here predicts those gaps from the exact training-stream counts n_N
(N = 10..9999, from exact_100b_<recipe>/counts_seed_2.csv) and turns them into a
"data beta" the same way:

  power q   Delta_k = sum_{N in decade k} n_N^q              shape only (scale-free)
            q=0: every number gets equal space (linear line, beta=10)
            q=1: space proportional to frequency (infomax / histogram equalisation)
  sat K     Delta_k = sum_{N in decade k} n_N / (n_N + K)     shape + amount
            a number seen n times takes n/(n+K) of a full slot: well-learned numbers
            (n >> K) get a full slot, rare ones (n << K) a share proportional to n.
            More numbers seen -> more of the large ones saturate -> beta rises.
            K -> inf reduces to q=1; K -> 0 to q=0.

q and K are free; each is chosen to maximise the Pearson r with beta, and a nested
leave-one-out (choose q / K and the linear map on the other models, predict the
held-out one) scores it honestly.

    python -m src.decade_metrics
    python -m src.decade_metrics --exclude dclm --out results/datadecide/decade_metrics_no_dclm.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

from src.confounder_analysis import BETAS, PRIMARY_FAMILY

FAMILY = {**PRIMARY_FAMILY, "c4": "c4", "fineweb-pro": "fineweb", "fineweb-edu": "fineweb"}
DECADES = ((10, 100), (100, 1000), (1000, 10000))
QS = np.linspace(0.0, 1.0, 101)
KS = np.logspace(0, 9, 181)


def load_counts(root: Path, seed: int = 2) -> dict[str, np.ndarray]:
    """recipe -> counts indexed by N (0..10000) of that seed's exact 100B stream."""
    out = {}
    for d in sorted(root.glob("exact_100b_*")):
        path = d / f"counts_seed_{seed}.csv"
        if not path.exists():
            continue
        c = np.zeros(10001)
        for row in csv.DictReader(open(path)):
            n = int(row["number"])
            if n <= 10000:
                c[n] = float(row["count"])
        out[d.name.removeprefix("exact_100b_")] = c
    return out


def data_beta(counts: np.ndarray, g) -> float:
    gaps = [g(counts[lo:hi]).sum() for lo, hi in DECADES]
    return float(np.sqrt(gaps[2] / gaps[0]))


def power_beta(counts, q):
    return data_beta(counts, lambda n: n ** q)


def sat_beta(counts, k):
    return data_beta(counts, lambda n: n / (n + k))


def _within(x, y, fams):
    x, y = np.array(x, float), np.array(y, float)
    for f in set(fams):
        m = np.array(fams) == f
        x[m] -= x[m].mean()
        y[m] -= y[m].mean()
    keep = np.array([sum(1 for g in fams if g == f) > 1 for f in fams])
    return stats.pearsonr(x[keep], y[keep])


def _fit_param(counts_list, y, grid, fn):
    """Grid value whose log data-beta has the highest Pearson r with y."""
    best, best_r = grid[0], -np.inf
    for p in grid:
        x = np.log10([fn(c, p) for c in counts_list])
        if np.ptp(x) == 0:
            continue
        r = stats.pearsonr(x, y)[0]
        if r > best_r:
            best, best_r = p, r
    return best


def nested_loo(counts_list, y, grid, fn) -> float:
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        tr = [c for c, keep in zip(counts_list, m) if keep]
        p = _fit_param(tr, y[m], grid, fn)
        x = np.log10([fn(c, p) for c in counts_list])
        b = np.polyfit(x[m], y[m], 1)
        pred[i] = np.polyval(b, x[i])
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alpha-root", default="results/corpus_alpha_datadecide")
    p.add_argument("--beta", default="results/datadecide/summary.csv")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--exclude", default="", help="comma-separated recipe prefixes to leave out, e.g. dclm")
    p.add_argument("--out", default="results/datadecide/decade_metrics.csv")
    args = p.parse_args(argv)

    counts = load_counts(Path(args.alpha_root), args.seed)
    beta = {r["recipe"]: r for r in csv.DictReader(open(args.beta))}
    skip = tuple(x for x in args.exclude.split(",") if x)
    recipes = [r for r in counts if r in beta and not (skip and r.startswith(skip))]
    if len(recipes) < 4:
        raise SystemExit(f"[decade] need >= 4 recipes with counts and beta, have {recipes}")
    C = [counts[r] for r in recipes]
    fams = [FAMILY.get(r, r) for r in recipes]
    total = np.array([c[1:].sum() for c in C])
    print(f"[decade] {len(recipes)} recipes: {recipes}")

    rows = {r: {"recipe": r, "family": f, "numbers_seen": t} for r, f, t in zip(recipes, fams, total)}
    for b in BETAS:
        y = np.array([float(beta[r][b]) for r in recipes])
        print(f"\n===== {b}")
        print(f"{'metric':34s} {'param':>9s} {'r':>6s} {'p':>6s} {'within r':>9s} {'p':>6s} {'LOO R2':>7s}")
        cands = [("log10 numbers seen", None, np.log10(total), None),
                 ("infomax (power q=1)", 1.0, np.log10([power_beta(c, 1.0) for c in C]), None),
                 ("power q=1/3", 1 / 3, np.log10([power_beta(c, 1 / 3) for c in C]), None)]
        q = _fit_param(C, y, QS, power_beta)
        cands.append(("power, best q (shape)", q, np.log10([power_beta(c, q) for c in C]), (QS, power_beta)))
        k = _fit_param(C, y, KS, sat_beta)
        cands.append(("saturation, best K (shape+amount)", k,
                      np.log10([sat_beta(c, k) for c in C]), (KS, sat_beta)))
        for name, param, x, fit in cands:
            r, pr = stats.pearsonr(x, y)
            wr, wp = _within(x, y, fams)
            if fit is None:
                xs = x.reshape(-1, 1)
                pred = [np.polyval(np.polyfit(np.delete(xs[:, 0], i), np.delete(y, i), 1), xs[i, 0])
                        for i in range(len(y))]
                loo = 1 - ((y - np.array(pred)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            else:
                loo = nested_loo(C, y, *fit)
            ps = "" if param is None else (f"{param:.3g}" if param < 1e3 else f"{param:.2e}")
            print(f"{name:34s} {ps:>9s} {r:+6.2f} {pr:6.3f} {wr:+9.2f} {wp:6.3f} {loo:+7.2f}")
        tag = b.removesuffix("_mean")
        for rcp, c in zip(recipes, C):
            rows[rcp][f"{tag}"] = float(beta[rcp][b])
            rows[rcp][f"q_best_{tag}"] = q
            rows[rcp][f"data_beta_power_{tag}"] = power_beta(c, q)
            rows[rcp][f"K_best_{tag}"] = k
            rows[rcp][f"data_beta_sat_{tag}"] = sat_beta(c, k)
        print(f"\n{'recipe':28s} {'family':8s} {'seen':>6s} {'beta':>6s} {'infomax':>8s} "
              f"{'pow q':>6s} {'sat K':>6s}")
        for rcp, c, yi in sorted(zip(recipes, C, y), key=lambda t: t[2]):
            print(f"{rcp:28s} {FAMILY.get(rcp, rcp):8s} {c[1:].sum() / 1e9:5.2f}B {yi:6.3f} "
                  f"{power_beta(c, 1.0):8.3f} {power_beta(c, q):6.3f} {sat_beta(c, k):6.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(next(iter(rows.values())))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows.values())
    print(f"\n[decade] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
