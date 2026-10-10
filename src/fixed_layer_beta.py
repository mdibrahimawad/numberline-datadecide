"""β at fixed layers for every model (from the E06 per-seed, per-layer metrics), and the
frozen DataDecide calibration lines used by the Paloma prediction test (E09).

    python -m src.fixed_layer_beta     # -> results/beta_fine/fixed_layer_beta.csv, results/beta_fine/frozen_lines.json

Comparing models at one common layer removes the layer-selection near-ties of E06.
Both DataDecide 1B and the Paloma 1B baselines have 16 transformer layers, so a layer
index means the same depth in both.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.count_features import RARE_K, load_counts

PRIMARY_LAYER = 8          # the layer the joint EV·|ρ| rule picks most often on DataDecide (E01 and E06)
METRICS = ("beta_cont", "beta_coarse", "rho", "ev")


def per_layer_table(run_dir: Path) -> pd.DataFrame:
    """Median over prompt seeds of each metric at every layer, one row per model."""
    rows = {}
    for f in sorted(run_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if "per_seed" not in d:
            continue
        seeds = d["per_seed"]
        layers = sorted(seeds[0]["layers"], key=int)
        with warnings.catch_warnings():   # the embedding layer has no beta (constant input)
            warnings.simplefilter("ignore", RuntimeWarning)
            rows[d["recipe"]] = {f"{m}_L{l}": float(np.nanmedian([s["layers"][l].get(m, np.nan) for s in seeds]))
                                 for l in layers for m in METRICS}
    return pd.DataFrame(rows).T


def data_predictors(counts: np.ndarray) -> dict[str, float]:
    """S (shrinkage sum) and R (rarely-seen count) from integer counts c(n), n = 0..10000."""
    x = counts[10:10000]
    return {"S": float((x / (x + RARE_K)).sum()), "R": int((x < RARE_K).sum())}


def frozen_line(x: pd.Series, y: pd.Series) -> dict:
    fit = stats.linregress(x, y)
    loo = []
    for r in x.index:
        keep = x.index != r
        g = stats.linregress(x[keep], y[keep])
        loo.append(g.intercept + g.slope * x[r])
    loo = np.array(loo)
    return {"intercept": float(fit.intercept), "slope": float(fit.slope), "r": float(fit.rvalue),
            "p": float(fit.pvalue), "n": int(len(x)), "loo_r": float(np.corrcoef(loo, y)[0, 1]),
            "loo_mae": float(np.abs(loo - y).mean()), "resid_sd": float(np.std(y - (fit.intercept + fit.slope * x), ddof=2))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", default="results/beta_fine/step69369-seed-default")
    p.add_argument("--counts", default="results/corpus_alpha_datadecide")
    p.add_argument("--alpha", default="results/corpus_alpha_datadecide/alpha_seed2.csv")
    p.add_argument("--table", default="results/beta_fine/fine_vs_data.csv", help="for the family column")
    p.add_argument("--out", default="results/beta_fine/fixed_layer_beta.csv")
    p.add_argument("--lines", default="results/beta_fine/frozen_lines.json")
    args = p.parse_args(argv)

    t = per_layer_table(Path(args.run_dir))
    counts = load_counts([Path(args.counts)])
    pred = pd.DataFrame({r: data_predictors(counts[r]) for r in t.index}).T
    alpha = pd.read_csv(args.alpha).set_index("recipe")[["alpha_ols", "alpha_mle", "integer_matches"]]
    fam = pd.read_csv(args.table).set_index("recipe")["family"]
    t = t.join(pred).join(alpha).join(fam)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out)

    def demean(s):
        return s - s.groupby(t.family).transform("mean")

    print(f"{len(t)} models; correlation of each predictor with beta_cont at each fixed layer")
    for layer in range(1, 17):
        col = f"beta_cont_L{layer}"
        if col not in t or t[col].isna().any():
            continue
        y = t[col]
        cells = [f"{k} r {stats.pearsonr(t[k], y)[0]:+.2f} (within {stats.pearsonr(demean(t[k]), demean(y))[0]:+.2f})"
                 for k in ("alpha_ols", "R", "S")]
        print(f"  L{layer:2d} median|rho| {t[f'rho_L{layer}'].abs().median():.2f}  " + "  ".join(cells))

    lines = {"layer": PRIMARY_LAYER, "K": RARE_K, "fitted_on": "25 DataDecide 1B models, E06 protocol",
             "lines": {}}
    for target in ("beta_cont", "beta_coarse"):
        y = t[f"{target}_L{PRIMARY_LAYER}"]
        lines["lines"][target] = {k: frozen_line(t[k], y) for k in ("alpha_ols", "S", "R")}
    Path(args.lines).write_text(json.dumps(lines, indent=1))
    print(f"\nfrozen lines at layer {PRIMARY_LAYER} (beta = intercept + slope * predictor):")
    for target, d in lines["lines"].items():
        for k, v in d.items():
            print(f"  {target:11s} ~ {k:9s} intercept {v['intercept']:+.4f} slope {v['slope']:+.6f}  r {v['r']:+.2f}  "
                  f"LOO r {v['loo_r']:.2f}  LOO MAE {v['loo_mae']:.3f}")
    print(f"-> {args.out}, {args.lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
