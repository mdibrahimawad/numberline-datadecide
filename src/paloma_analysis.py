"""E09 analysis: the frozen DataDecide predictions vs the measured β of the Paloma models,
exactly as pre-registered in experiments/E09_paloma_out_of_sample/README.md.

    python -m src.paloma_analysis      # -> results/paloma_counts/scores.json, validity.csv

Validity (locked, per model, at the layer analysed): acceptance >= 80 % in every group (pooled
over the 3 prompt seeds), median |rho| >= 0.9, EV >= 0.1, err_coarse <= 0.4. Failing models
are reported and excluded, never dropped silently. Models without counts (RedPajama) are
measured but not scored.

Criteria per predictor (α, S, R): 1. Spearman ρ of predicted vs measured; 2. MAE vs 1.5 x
its DataDecide LOO MAE; 3. lowest MAE wins (ties < 0.02); 4. amount effect: mean residual
(measured - predicted) of the α line. Secondary: shift-corrected MAE, β_coarse, the mean over
layers 6-10, each model's own selected layer.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RUN = Path("results/beta_fine/step35000-unsharded")
PRED = Path("results/paloma_counts/predictions.json")
FROZEN = Path("results/beta_fine/frozen_lines.json")
OUT = Path("results/paloma_counts")
PREDICTORS = ("alpha_ols", "S", "R")
CHECKS = {"acceptance": 0.80, "rho": 0.90, "ev": 0.10, "err_coarse": 0.40}
TIE = 0.02


def corpus_of(path: Path) -> str:
    return path.stem.replace("allenai_paloma-1b-baseline-", "")


def layer_value(d: dict, layer: int, key: str) -> float:
    return float(np.nanmedian([s["layers"][str(layer)].get(key, np.nan) for s in d["per_seed"]]))


def acceptance(d: dict) -> float:
    """Lowest acceptance over the groups, each pooled over the prompt seeds."""
    g = defaultdict(lambda: [0, 0])
    for groups in d["filter"].values():
        for k, v in groups.items():
            g[k][0] += v["accepted"]
            g[k][1] += v["tried"]
    return min(a / t for a, t in g.values())


def validity(d: dict, layer: int) -> dict:
    v = {"acceptance": acceptance(d), "rho": layer_value(d, layer, "rho"),
         "ev": layer_value(d, layer, "ev"), "err_coarse": layer_value(d, layer, "err_coarse")}
    fails = [k for k, lim in CHECKS.items() if (v[k] > lim if k == "err_coarse" else v[k] < lim)]
    return {**v, "valid": not fails, "fails": fails}


def score(pred: dict, meas: dict, target: str, frozen: dict) -> dict:
    """The four criteria for each predictor over the models in `meas`."""
    names = sorted(meas)
    y = np.array([meas[n] for n in names])
    out = {"models": names, "measured": dict(zip(names, y.tolist()))}
    for p in PREDICTORS:
        x = np.array([pred[n][f"pred_{target}_from_{p}"] for n in names])
        res = y - x
        loo = frozen["lines"][target][p]["loo_mae"]
        rho = stats.spearmanr(x, y)[0] if len(y) >= 3 and np.ptp(x) > 0 else float("nan")
        out[p] = {"predicted": dict(zip(names, x.tolist())), "spearman": float(rho),
                  "mae": float(np.abs(res).mean()), "limit_1.5x_loo": 1.5 * loo,
                  "transfers": bool(np.abs(res).mean() <= 1.5 * loo),
                  "mean_residual": float(res.mean()),
                  "shift_corrected_mae": float(np.abs(res - res.mean()).mean())}
    maes = {p: out[p]["mae"] for p in PREDICTORS}
    best = min(maes, key=maes.get)
    out["winner"] = [p for p in PREDICTORS if maes[p] - maes[best] < TIE]
    out["amount_effect_alpha_mean_residual"] = out["alpha_ols"]["mean_residual"]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layer", type=int, default=8)
    args = p.parse_args(argv)
    pred = {r["corpus"]: r for r in json.loads(PRED.read_text())["predictions"]}
    frozen = json.loads(FROZEN.read_text())
    models = {corpus_of(f): json.loads(f.read_text()) for f in sorted(RUN.glob("*.json"))}

    rows, analyses = [], {}
    for name, d in models.items():
        v = validity(d, args.layer)
        rows.append({"corpus": name, "layer": args.layer, **{k: round(v[k], 4) for k in CHECKS},
                     "valid": v["valid"], "fails": ";".join(v["fails"]), "has_counts": name in pred,
                     **{f"{t}_L{args.layer}": layer_value(d, args.layer, t) for t in ("beta_cont", "beta_coarse")},
                     "own_layer": d["selection"]["selected_layer"] if "selection" in d else None})
    with open(OUT / "validity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    valid = [r["corpus"] for r in rows if r["valid"] and r["has_counts"]]
    scored = [r["corpus"] for r in rows if r["has_counts"]]
    for target in ("beta_cont", "beta_coarse"):
        analyses[f"primary_{target}_L{args.layer}_valid"] = score(
            pred, {n: layer_value(models[n], args.layer, target) for n in valid}, target, frozen)
        analyses[f"all5_{target}_L{args.layer}_incl_invalid"] = score(
            pred, {n: layer_value(models[n], args.layer, target) for n in scored}, target, frozen)
        analyses[f"secondary_{target}_mean_L6-10"] = score(
            pred, {n: float(np.nanmean([layer_value(models[n], l, target) for l in range(6, 11)]))
                   for n in scored}, target, frozen)
        own = {n: models[n]["selection"]["selected_layer"] for n in scored}
        analyses[f"secondary_{target}_own_layer"] = score(
            pred, {n: layer_value(models[n], int(own[n]), target) for n in scored}, target, frozen)
        analyses[f"secondary_{target}_own_layer"]["own_layers"] = own
    (OUT / "scores.json").write_text(json.dumps({"validity": rows, "analyses": analyses}, indent=2) + "\n")

    print(f"validity at layer {args.layer}:")
    for r in rows:
        print(f"  {r['corpus']:18s} {'VALID' if r['valid'] else 'invalid':8s} {r['fails']:28s} "
              f"rho {r['rho']:.3f} acc {r['acceptance']:.2f} beta_cont {r[f'beta_cont_L{args.layer}']:.3f}"
              + ("" if r["has_counts"] else "  (no counts: not scored)"))
    for key, a in analyses.items():
        print(f"\n{key}  (n = {len(a['models'])}: {', '.join(a['models'])})")
        print("  measured: " + ", ".join(f"{n} {v:.3f}" for n, v in a["measured"].items()))
        for pr in PREDICTORS:
            s = a[pr]
            print(f"  {pr:9s} spearman {s['spearman']:+.2f}  MAE {s['mae']:.3f} (limit {s['limit_1.5x_loo']:.3f}, "
                  f"{'transfers' if s['transfers'] else 'fails'})  mean residual {s['mean_residual']:+.3f}  "
                  f"shift-corrected MAE {s['shift_corrected_mae']:.3f}")
        print(f"  winner (lowest MAE, ties < {TIE}): {a['winner']}")
    print(f"\n[paloma] wrote {OUT / 'validity.csv'} and {OUT / 'scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
