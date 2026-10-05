"""Join corpus alpha (training data) with model beta (number-line geometry) for the DataDecide 1B models.

Inputs (all keyed by recipe slug, as in configs/datadecide_models.json):
  results/corpus_alpha_datadecide/alpha_seed2.csv   exact 100B training stream (::exact_all)
  results/corpus_alpha_datadecide/alpha_summary.csv cheap ~1B-token window sample (::sweep),
                                                    used for recipes without an exact alpha
  results/datadecide/summary.csv                    beta (modal_app/datadecide_app.py)

The window sampler's alpha_MLE matches the exact value (dolma1_7: 0.0003) but its
alpha_OLS is biased by ~0.05, so OLS correlations use exact rows only.

Output:
  results/datadecide/alpha_beta.csv                 one row per recipe with both
  results/datadecide/alpha_beta.png                 alpha vs beta scatter plots
and the Pearson / Spearman correlations printed to the terminal.

    python -m src.join_alpha_beta
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ALPHA_COLS = ("alpha_ols", "alpha_mle", "r2")
BETA_COLS = ("selected_layer", "rho_mean", "ev_mean", "beta_direct_mean", "beta_direct_std",
             "r2_direct", "beta_log_mean", "beta_log_std", "failed_groups")


def _read(path: Path) -> dict[str, dict]:
    with open(path) as fh:
        return {row["recipe"]: row for row in csv.DictReader(fh)}


def join(alpha_csv: Path, beta_csv: Path, window_csv: Path | None = None) -> list[dict]:
    alpha = _read(alpha_csv) if alpha_csv.exists() else {}
    for row in alpha.values():
        row["alpha_source"] = "exact_100b"
    if window_csv is not None and window_csv.exists():
        for recipe, row in _read(window_csv).items():
            if recipe not in alpha:
                alpha[recipe] = {**row, "seed": "", "alpha_source": "window_1b"}
    beta = _read(beta_csv)
    order = json.loads((Path(__file__).resolve().parent.parent / "configs" / "datadecide_models.json")
                       .read_text())["recipes"]
    rows = []
    for recipe in order:
        if recipe in alpha and recipe in beta:
            row = {"recipe": recipe, "alpha_source": alpha[recipe]["alpha_source"],
                   "alpha_seed": alpha[recipe]["seed"]}
            row.update({k: alpha[recipe][k] for k in ALPHA_COLS})
            row["alpha_r2"] = row.pop("r2")
            row.update({k: beta[recipe].get(k, "") for k in BETA_COLS})
            rows.append(row)
    missing = [r for r in order if r not in alpha or r not in beta]
    if missing:
        print(f"[join] missing alpha or beta for: {missing}")
    return rows


def correlations(rows: list[dict]) -> dict:
    out = {}
    for a in ("alpha_ols", "alpha_mle"):
        # window-sample OLS is biased; only exact rows enter the OLS correlations
        use = rows if a == "alpha_mle" else [r for r in rows if r["alpha_source"] == "exact_100b"]
        for b in ("beta_direct_mean", "beta_log_mean"):
            pairs = [(float(r[a]), float(r[b])) for r in use if r[a] and r[b]]
            pairs = [(x, y) for x, y in pairs if np.isfinite(x) and np.isfinite(y)]
            if len(pairs) < 3:
                continue
            x, y = map(np.asarray, zip(*pairs))
            pr, pp = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
            out[f"{a} vs {b}"] = {"n": len(x), "pearson_r": float(pr), "pearson_p": float(pp),
                                  "spearman_rho": float(sr), "spearman_p": float(sp)}
    return out


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, a in zip(axes, ("alpha_ols", "alpha_mle")):
        use = rows if a == "alpha_mle" else [r for r in rows if r["alpha_source"] == "exact_100b"]
        for source, color in (("exact_100b", "#c0392b"), ("window_1b", "#2471a3")):
            sel = [r for r in use if r["alpha_source"] == source]
            if sel:
                ax.errorbar([float(r[a]) for r in sel], [float(r["beta_direct_mean"]) for r in sel],
                            yerr=[float(r["beta_direct_std"] or 0) for r in sel], fmt="o", ms=4,
                            capsize=2, color=color, label=source)
        ax.legend(fontsize=7)
        x = [float(r[a]) for r in use]
        y = [float(r["beta_direct_mean"]) for r in use]
        for xi, yi, r in zip(x, y, use):
            ax.annotate(r["recipe"], (xi, yi), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel(f"corpus {a}")
        ax.set_ylabel("model beta (direct fit, frozen layer)")
        ax.set_title(f"DataDecide 1B: {a} vs beta")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alpha", default="results/corpus_alpha_datadecide/alpha_seed2.csv")
    p.add_argument("--window-alpha", default="results/corpus_alpha_datadecide/alpha_summary.csv")
    p.add_argument("--beta", default="results/datadecide/summary.csv")
    p.add_argument("--out", default="results/datadecide/alpha_beta.csv")
    args = p.parse_args(argv)

    rows = join(Path(args.alpha), Path(args.beta), Path(args.window_alpha))
    if not rows:
        raise SystemExit("[join] no recipe has both alpha and beta yet")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, out.with_suffix(".png"))
    corr = correlations(rows)
    out.with_name("alpha_beta_correlations.json").write_text(json.dumps(corr, indent=2) + "\n")
    print(f"[join] {len(rows)} recipes -> {out}, {out.with_suffix('.png')}")
    for k, v in corr.items():
        print(f"  {k:36s} n={v['n']:2d}  pearson r={v['pearson_r']:+.3f} (p={v['pearson_p']:.3g})  "
              f"spearman rho={v['spearman_rho']:+.3f} (p={v['spearman_p']:.3g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
