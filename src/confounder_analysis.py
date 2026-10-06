"""Is the alpha -> beta link explained by other differences between DataDecide recipes?

Per recipe it gathers alpha (exact 100B stream where available, else the ~1B
window sample), beta (frozen-layer geometry), and candidate confounders:

  number_density  integer matches per 1,000 tokens of training text
  code_share      token share of StarCoder files in the recipe
  math_share      token share of Proof-Pile-2 (algebraic-stack + open-web-math)

and reports
  1. plain correlations of every predictor with beta;
  2. partial correlations of alpha with beta controlling for density (and code);
  3. within-family tables (Dolma ablations, DCLM/Dolma mixes, DCLM and Falcon
     quality filters) and a family-demeaned ("fixed effects") correlation, which
     removes everything that differs between families.

alpha_OLS from the window sample is biased (~0.05), so OLS rows are exact only.

    python -m src.confounder_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.join_alpha_beta import join

REPO = Path(__file__).resolve().parent.parent
ALPHA_ROOT = Path("results/corpus_alpha_datadecide")

FAMILIES = {
    "dolma_ablations": ["dolma1_7", "dolma1_7-no-code", "dolma1_7-no-math-code",
                        "dolma1_7-no-reddit", "dolma1_7-no-flan"],
    "dclm_dolma_mix": ["dclm-baseline", "dclm-baseline-75p-dolma1.7-25p",
                       "dclm-baseline-50p-dolma1.7-50p", "dclm-baseline-25p-dolma1.7-75p",
                       "dolma1_7"],
    "dclm_quality": ["dclm-baseline", "dclm-baseline-qc-7p-fw2", "dclm-baseline-qc-7p-fw3",
                     "dclm-baseline-qc-fw-3p", "dclm-baseline-qc-fw-10p",
                     "dclm-baseline-qc-10p", "dclm-baseline-qc-20p"],
    "falcon_quality": ["falcon", "falcon-and-cc", "falcon-and-cc-qc-10p", "falcon-and-cc-qc-20p",
                       "falcon-and-cc-qc-orig-10p", "falcon-and-cc-qc-tulu-10p"],
}
# Dolma share of the DCLM/Dolma mixes, for the dose-response view
DOLMA_SHARE = {"dclm-baseline": 0.0, "dclm-baseline-75p-dolma1.7-25p": 0.25,
               "dclm-baseline-50p-dolma1.7-50p": 0.5, "dclm-baseline-25p-dolma1.7-75p": 0.75,
               "dolma1_7": 1.0}
# primary family for the fixed-effects analysis (each recipe counted once)
PRIMARY_FAMILY = {
    **{r: "dolma" for r in FAMILIES["dolma_ablations"] + ["dolma1_6plus"]},
    **{r: "dclm" for r in FAMILIES["dclm_quality"]},
    **{r: "mix" for r in FAMILIES["dclm_dolma_mix"][1:4]},
    **{r: "falcon" for r in FAMILIES["falcon_quality"]},
    "c4": "web_other", "fineweb-pro": "web_other", "fineweb-edu": "web_other",
}
BETAS = ("beta_direct_mean", "beta_log_mean")


def _float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def number_density(root: Path) -> dict[str, float]:
    """Integer matches per 1,000 tokens: exact stream first, else the window sample."""
    out = {}
    window = root / "alpha_summary.csv"
    if window.exists():
        for r in csv.DictReader(open(window)):
            out[r["recipe"]] = 1000 * _float(r["integer_matches"]) / _float(r["sampled_tokens"])
    exact = root / "alpha_seed2.csv"
    if exact.exists():
        for r in csv.DictReader(open(exact)):
            out[r["recipe"]] = 1000 * _float(r["integer_matches"]) / _float(r["sample_tokens"])
    return out


def source_shares(root: Path) -> dict[str, dict[str, float]]:
    """Code / math token shares from each recipe's mixture.json (written by ::sweep);
    a recipe without StarCoder / Proof-Pile-2 files has share exactly 0."""
    data_map = json.loads((REPO / "configs" / "datadecide_data_map.json").read_text())["recipes"]
    out = {}
    for slug, rec in data_map.items():
        has_code = any("/starcoder/" in f"/{p}" for p in rec["paths"])
        has_math = any("/proof-pile-2/" in f"/{p}" for p in rec["paths"])
        mix_path = root / slug / "mixture.json"
        mix = json.loads(mix_path.read_text()) if mix_path.exists() else None

        def share(prefix, present):
            if not present:
                return 0.0
            if mix is None:
                return float("nan")
            return sum(v["share"] for k, v in mix.items() if k.startswith(prefix))

        out[slug] = {"code_share": share("starcoder/", has_code),
                     "math_share": share("proof-pile-2/", has_math)}
    return out


def _corr(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(len(x))}
    r, p = stats.pearsonr(x, y)
    rho, sp = stats.spearmanr(x, y)
    return {"n": int(len(x)), "pearson_r": float(r), "p": float(p),
            "spearman_rho": float(rho), "spearman_p": float(sp)}


def _partial(x, y, covariates) -> dict:
    """Correlation of x and y after regressing both on the covariates (OLS)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.column_stack([np.asarray(c, float) for c in covariates])
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[ok], y[ok], Z[ok]
    k = Z.shape[1]
    n = len(x)
    if n - k - 2 < 1:
        return {"n": int(n), "controls": k}
    A = np.column_stack([np.ones(n), Z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    if np.std(rx) == 0 or np.std(ry) == 0:
        return {"n": int(n), "controls": k}
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = n - 2 - k
    t = r * np.sqrt(df / max(1e-12, 1 - r * r))
    p = float(2 * stats.t.sf(abs(t), df))
    return {"n": int(n), "controls": k, "partial_r": r, "p": p}


def _demeaned(rows, key, families):
    out = np.full(len(rows), np.nan)
    for fam in set(families):
        idx = [i for i, f in enumerate(families) if f == fam]
        vals = np.asarray([_float(rows[i][key]) for i in idx])
        if len(idx) >= 2 and np.isfinite(vals).sum() >= 2:
            out[idx] = vals - np.nanmean(vals)
    return out


def analyze(rows: list[dict]) -> dict:
    exact = [r for r in rows if r["alpha_source"] == "exact_100b"]
    report = {"n_recipes": len(rows), "n_exact": len(exact), "plain": {}, "partial": {},
              "within_family": {}, "fixed_effects": {}}

    def col(rs, k):
        return [_float(r[k]) for r in rs]

    predictors = {
        "alpha_ols (exact only)": (exact, "alpha_ols"),
        "alpha_mle": (rows, "alpha_mle"),
        "number_density": (rows, "number_density"),
        "code_share": (rows, "code_share"),
        "math_share": (rows, "math_share"),
    }
    for name, (rs, k) in predictors.items():
        for b in BETAS:
            report["plain"][f"{name} vs {b}"] = _corr(col(rs, k), col(rs, b))

    for alpha_name, (rs, k) in (("alpha_ols (exact only)", (exact, "alpha_ols")),
                                ("alpha_mle", (rows, "alpha_mle"))):
        for b in BETAS:
            report["partial"][f"{alpha_name} vs {b} | density"] = _partial(
                col(rs, k), col(rs, b), [col(rs, "number_density")])
            report["partial"][f"{alpha_name} vs {b} | density + code"] = _partial(
                col(rs, k), col(rs, b), [col(rs, "number_density"), col(rs, "code_share")])

    by_recipe = {r["recipe"]: r for r in rows}
    for fam, members in FAMILIES.items():
        present = [by_recipe[m] for m in members if m in by_recipe]
        entry = {"rows": [{k: present_r.get(k) for k in (
            "recipe", "alpha_source", "alpha_ols", "alpha_mle", "number_density",
            "code_share", "math_share", "beta_direct_mean", "beta_log_mean", "r2_direct")}
            for present_r in present]}
        ex = [r for r in present if r["alpha_source"] == "exact_100b"]
        for b in BETAS:
            entry[f"alpha_mle vs {b}"] = _corr(col(present, "alpha_mle"), col(present, b))
            entry[f"alpha_ols (exact) vs {b}"] = _corr(col(ex, "alpha_ols"), col(ex, b))
        if fam == "dclm_dolma_mix":
            share = [DOLMA_SHARE[r["recipe"]] for r in present]
            for k in ("alpha_mle", "alpha_ols", "number_density", "beta_direct_mean", "beta_log_mean"):
                entry[f"dolma_share vs {k}"] = _corr(share, col(present, k))
        report["within_family"][fam] = entry

    fams = [PRIMARY_FAMILY.get(r["recipe"], "other") for r in rows]
    ex_fams = [PRIMARY_FAMILY.get(r["recipe"], "other") for r in exact]
    for b in BETAS:
        report["fixed_effects"][f"alpha_mle vs {b}"] = _corr(
            _demeaned(rows, "alpha_mle", fams), _demeaned(rows, b, fams))
        report["fixed_effects"][f"alpha_ols (exact) vs {b}"] = _corr(
            _demeaned(exact, "alpha_ols", ex_fams), _demeaned(exact, b, ex_fams))
        report["fixed_effects"][f"number_density vs {b}"] = _corr(
            _demeaned(rows, "number_density", fams), _demeaned(rows, b, fams))
    return report


def _fmt(d: dict) -> str:
    if "pearson_r" in d:
        return f"n={d['n']:2d}  r={d['pearson_r']:+.2f} (p={d['p']:.3f})  rho={d['spearman_rho']:+.2f}"
    if "partial_r" in d:
        return f"n={d['n']:2d}  partial r={d['partial_r']:+.2f} (p={d['p']:.3f})"
    return f"n={d.get('n', 0):2d}  (too few points)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alpha-root", default=str(ALPHA_ROOT))
    p.add_argument("--beta", default="results/datadecide/summary.csv")
    p.add_argument("--out", default="results/datadecide/confounders.json")
    args = p.parse_args(argv)
    root = Path(args.alpha_root)

    rows = join(root / "alpha_seed2.csv", Path(args.beta), root / "alpha_summary.csv")
    dens, shares = number_density(root), source_shares(root)
    for r in rows:
        r["number_density"] = dens.get(r["recipe"], float("nan"))
        r.update(shares.get(r["recipe"], {"code_share": float("nan"), "math_share": float("nan")}))
    report = analyze(rows)
    report["recipes"] = rows
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")

    print(f"[confounders] {report['n_recipes']} recipes ({report['n_exact']} with exact alpha)\n")
    print(f"{'recipe':32s} {'src':6s} {'a_OLS':>7s} {'a_MLE':>7s} {'num/1k':>6s} {'code':>5s} "
          f"{'math':>5s} {'b_dir':>6s} {'b_log':>6s}")
    for r in rows:
        print(f"{r['recipe']:32s} {r['alpha_source'][:6]:6s} {_float(r['alpha_ols']):7.3f} "
              f"{_float(r['alpha_mle']):7.3f} {r['number_density']:6.2f} {r['code_share']:5.2f} "
              f"{r['math_share']:5.2f} {_float(r['beta_direct_mean']):6.3f} {_float(r['beta_log_mean']):6.3f}")
    for section in ("plain", "partial", "fixed_effects"):
        print(f"\n== {section.replace('_', ' ')} ==")
        for k, v in report[section].items():
            print(f"  {k:52s} {_fmt(v)}")
    print("\n== within family ==")
    for fam, entry in report["within_family"].items():
        names = ", ".join(f"{r['recipe']}{'*' if r['alpha_source'] == 'exact_100b' else ''}"
                          for r in entry["rows"])
        print(f"  [{fam}] {names}")
        for k, v in entry.items():
            if k != "rows":
                print(f"    {k:48s} {_fmt(v)}")
    print(f"\n(* = exact alpha)  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
