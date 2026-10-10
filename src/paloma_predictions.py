"""E09 step 1: the β each frozen DataDecide line predicts for the Paloma models, written
down BEFORE any Paloma β is measured.

    python -m src.paloma_predictions     # -> results/paloma_counts/predictions.{csv,json}

Inputs: results/beta_fine/frozen_lines.json (fitted on DataDecide only, layer 8) and
results/paloma_counts/<corpus>/summary.json (α_OLS, S, R of each training corpus at its
146.8B-token budget). Each prediction is flagged when its predictor lies outside the
DataDecide range the line was fitted on (an extrapolation).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

FROZEN = Path("results/beta_fine/frozen_lines.json")
DATADECIDE = Path("results/beta_fine/fixed_layer_beta.csv")
COUNTS = Path("results/paloma_counts")
MODEL = "allenai/paloma-1b-baseline-{}"
PREDICTORS = {"alpha_ols": "alpha_ols", "S": "S_K4000", "R": "R_K4000"}   # line name -> summary key


def datadecide_range() -> dict[str, tuple[float, float]]:
    rows = list(csv.DictReader(open(DATADECIDE)))
    return {p: (min(float(r[p]) for r in rows), max(float(r[p]) for r in rows)) for p in PREDICTORS}


def predictions(frozen: dict, corpora: dict[str, dict], ranges: dict) -> list[dict]:
    out = []
    for corpus, s in sorted(corpora.items()):
        row = {"corpus": corpus, "model": MODEL.format(corpus)}
        for name, key in PREDICTORS.items():
            x = float(s[key])
            lo, hi = ranges[name]
            row[name] = x
            row[f"{name}_outside_datadecide"] = not lo <= x <= hi
            for target, lines in frozen["lines"].items():
                line = lines[name]
                row[f"pred_{target}_from_{name}"] = line["intercept"] + line["slope"] * x
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--counts", default=str(COUNTS))
    args = p.parse_args(argv)
    root = Path(args.counts)
    frozen = json.loads(FROZEN.read_text())
    corpora = {d.name: json.loads((d / "summary.json").read_text())
               for d in sorted(root.iterdir()) if (d / "summary.json").exists()}
    rows = predictions(frozen, corpora, datadecide_range())
    with open(root / "predictions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit, "layer": frozen["layer"], "K": frozen["K"],
            "note": "written before any Paloma beta was measured (E09 pre-registration)",
            "datadecide_range": datadecide_range(), "predictions": rows}
    (root / "predictions.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{'corpus':18s} {'alpha':>8s} {'S':>6s} {'R':>6s} | beta_cont from: {'alpha':>6s} {'S':>6s} {'R':>6s}")
    for r in rows:
        flag = lambda n: "*" if r[f"{n}_outside_datadecide"] else " "
        print(f"{r['corpus']:18s} {r['alpha_ols']:8.4f}{flag('alpha_ols')} {r['S']:6.0f}{flag('S')} {r['R']:6.0f}{flag('R')} |"
              f"                 {r['pred_beta_cont_from_alpha_ols']:6.3f} {r['pred_beta_cont_from_S']:6.3f} "
              f"{r['pred_beta_cont_from_R']:6.3f}")
    print("* = outside the DataDecide range the line was fitted on (extrapolation)")
    print(f"[paloma] wrote {root / 'predictions.csv'} and predictions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
