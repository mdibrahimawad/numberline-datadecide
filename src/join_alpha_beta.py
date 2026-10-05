"""Join corpus alpha (training data) with model beta (number-line geometry) for the DataDecide 1B models.

Inputs (both keyed by recipe slug, as in configs/datadecide_models.json):
  results/corpus_alpha_datadecide/alpha_seed2.csv   from modal_app/datadecide_alpha_app.py::exact_all
  results/datadecide/summary.csv                    from modal_app/datadecide_app.py

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


def join(alpha_csv: Path, beta_csv: Path) -> list[dict]:
    alpha, beta = _read(alpha_csv), _read(beta_csv)
    order = json.loads((Path(__file__).resolve().parent.parent / "configs" / "datadecide_models.json")
                       .read_text())["recipes"]
    rows = []
    for recipe in order:
        if recipe in alpha and recipe in beta:
            row = {"recipe": recipe, "alpha_seed": alpha[recipe]["seed"]}
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
        for b in ("beta_direct_mean", "beta_log_mean"):
            pairs = [(float(r[a]), float(r[b])) for r in rows if r[a] and r[b]]
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
        x = np.asarray([float(r[a]) for r in rows])
        y = np.asarray([float(r["beta_direct_mean"]) for r in rows])
        err = np.asarray([float(r["beta_direct_std"] or 0) for r in rows])
        ax.errorbar(x, y, yerr=err, fmt="o", ms=4, capsize=2, color="#2471a3")
        for xi, yi, r in zip(x, y, rows):
            ax.annotate(r["recipe"], (xi, yi), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel(f"corpus {a} (exact 100B training stream)")
        ax.set_ylabel("model beta (direct fit, frozen layer)")
        ax.set_title(f"DataDecide 1B: {a} vs beta")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alpha", default="results/corpus_alpha_datadecide/alpha_seed2.csv")
    p.add_argument("--beta", default="results/datadecide/summary.csv")
    p.add_argument("--out", default="results/datadecide/alpha_beta.csv")
    args = p.parse_args(argv)

    rows = join(Path(args.alpha), Path(args.beta))
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
