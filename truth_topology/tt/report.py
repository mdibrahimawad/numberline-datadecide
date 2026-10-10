"""Collect the numbers REPORT.md needs into results/REPORT_tables.md (verdicts are written by hand).

    python -m tt.report
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C


def _fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def run_tables(model, inp):
    res = C.RESULTS / model / inp
    if not (res / "metrics.csv").exists():
        return ""
    m = pd.read_csv(res / "metrics.csv")
    ref = json.load(open(res / "reference_layer.json"))["reference_layer"]
    out = [f"## {model} / {inp}  (reference layer {ref})\n"]
    on = pd.read_csv(res / "onset.csv")
    out.append("### Onset (first of 2 consecutive checkpoints above the null 95th percentile)\n")
    out.append(on.to_markdown(index=False) if hasattr(on, "to_markdown") else on.to_string())
    # step-0 control over all layers, Bonferroni over layers
    s0 = m[(m.step == 0) & m.p.notna()]
    if not s0.empty:
        rows = []
        for met, d in s0.groupby("metric"):
            k = len(d)
            rows.append({"metric": met, "cells": k, "min_p": d.p.min(), "bonferroni_sig": bool((d.p * k < .05).any()),
                         "ref_layer_value": _fmt(float(d[d.layer.isin([ref, -1])].value.mean()) if len(d[d.layer.isin([ref, -1])]) else np.nan)})
        out.append("\n### Step-0 control (untrained model): any metric significant after Bonferroni over layers?\n")
        out.append(pd.DataFrame(rows).to_markdown(index=False))
    # trajectory at the reference layer
    keys = ["B1_loglik_auc", "B2_deltaH", "B2_memorization_truectx", "L1_auc", "L1_split_acc", "L2_auc",
            "L1_within_auc", "L3_auc", "T1_acc", "T1_gap", "T3a_topic_homophily",
            "T3b_within_truth_homophily", "T3c_cross_truth_homophily"]
    d = m[m.metric.isin(keys) & (m.layer.isin([ref, -1])) & (m.seed.isin([-1, 0]))]
    piv = d.pivot_table(index="step", columns="metric", values="value")
    out.append("\n### Trajectory at the reference layer (seed 0)\n")
    out.append(piv.round(3).to_markdown())
    g = m[m.metric.str.startswith("T1g_") & (m.layer == ref) & (m.seed == 0)]
    if not g.empty:
        g = g.assign(txt=g.apply(lambda r: f"{r.value:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]", axis=1))
        out.append("\n### Hedges' g (true − false), key barcode features, reference layer, 95% bootstrap CI\n")
        out.append(g.pivot(index="step", columns="metric", values="txt").to_markdown())
    if (res / "concentration.csv").exists():
        c = pd.read_csv(res / "concentration.csv")
        c = c[(c.layer == ref)]
        out.append("\n### T2-C concentration at the reference layer\n")
        out.append(c.round(3).to_markdown(index=False))
    if (res / "gaussian_floor.json").exists():
        f = json.load(open(res / "gaussian_floor.json"))
        out.append(f"\n### Gaussian floor (m={f['m']}, d={f['d']}): H1 bars {f['h1_n_bars']['mean']:.1f} ± "
                   f"{f['h1_n_bars']['sd']:.1f}, H0 entropy {f['h0_entropy']['mean']:.3f}\n")
    return "\n".join(out) + "\n"


def main():
    parts = ["# Auto-generated tables (do not edit; rerun `python -m tt.report`)\n"]
    v = C.RESULTS / "validation" / "validation.json"
    if v.exists():
        r = json.load(open(v))
        parts.append("## Synthetic validation\n")
        parts.append("| test | criterion | pass |\n|---|---|---|")
        for k, t in r["tests"].items():
            parts.append(f"| {k} | {r['criteria'][k]} | {'yes' if t['pass'] else '**NO**'} |")
        parts.append("")
    for f in sorted(C.RESULTS.glob("confounds_*.json")):
        parts.append(f"## Confound check {f.stem[10:]}\n\n```\n{json.dumps(json.load(open(f)), indent=1)[:3000]}\n```\n")
    for model in C.MODELS:
        for inp in ("P1", "P2"):
            parts.append(run_tables(model, inp))
    (C.RESULTS / "REPORT_tables.md").write_text("\n".join(parts))
    print("wrote", C.RESULTS / "REPORT_tables.md")


if __name__ == "__main__":
    main()
