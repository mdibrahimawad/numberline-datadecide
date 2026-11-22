"""E14 report: applies the pre-registered verdict rules (experiments/E14_length_vs_log/PREREG.md §6) to the CSVs."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

SMOOTH = ("smooth, log inside", "smooth, linear inside", "smooth, shape unclear")


def _f(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:+.{d}f}"


def verdict(classes):
    c = [k for k in classes if k != "no number line"]
    if not c:
        return "no data", {}
    n = len(c); counts = {k: c.count(k) for k in sorted(set(c))}
    if sum(k == "length staircase" for k in c) >= 2 * n / 3:
        v = "PROFESSOR RIGHT (length staircase)"
    elif sum(k in SMOOTH for k in c) >= 2 * n / 3:
        v = "PROFESSOR WRONG (smooth line)"
    else:
        v = "PARTLY RIGHT (mixed)"
    return v, counts


def tok_test(df):
    inp = ((df.LS_100_hi <= 0.25) & (df.LS_1000_lo >= 0.5)).sum()
    learned = ((df.LS_100_lo >= 0.5) & (df.LS_1000_lo >= 0.5)).sum()
    return int(inp), int(learned), len(df)


def table(df, cols):
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fn(r) for _, fn in cols) + " |")
    return "\n".join(lines)


def ci_str(r, k):
    return f"{_f(r[k])} [{_f(r[k + '_lo'])}, {_f(r[k + '_hi'])}]"


def write_report(out):
    md = ["# E14 results: is the log number line just digit length?", "",
          "Pre-registration: `experiments/E14_length_vs_log/PREREG.md`. LS = share of each decade's rise that happens as a "
          "jump at the digit-length boundary (0 = smooth log line, 1 = pure length staircase). λ = shape inside a length "
          "(0 = log, 1 = linear). 95 % bootstrap CIs in brackets.", ""]
    cols = [("model", lambda r: str(r.model).replace("allenai_paloma-1b-baseline-", "paloma-")),
            ("layer", lambda r: str(int(r.layer))), ("LS", lambda r: ci_str(r, "LS")),
            ("LS at 100", lambda r: ci_str(r, "LS_100")), ("LS at 1000", lambda r: ci_str(r, "LS_1000")),
            ("J", lambda r: _f(r.J)), ("λ inside", lambda r: ci_str(r, "lam")), ("ρ", lambda r: f"{r.rho:.2f}"),
            ("class", lambda r: r["class"])]
    summary = []
    for tag, title in (("E06_DataDecide", "D1: 25 DataDecide models (β protocol) — main verdict"),
                       ("E06_Paloma", "D2: 6 Paloma models (replication)")):
        p = os.path.join(out, f"{tag}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        for role, lab in (("selected", "primary: E06-selected layer"), ("fixed8", "secondary: fixed layer 8")):
            d = df[df.role.fillna("").str.contains(role)]
            v, counts = verdict(list(d["class"]))
            ti, tl, tn = tok_test(d)
            md += [f"## {title} — {lab}", "", f"**Verdict: {v}.** Classes: {counts}. Median LS = {np.nanmedian(d.LS):+.2f} "
                   f"(LS at 100: {np.nanmedian(d.LS_100):+.2f}, at 1000: {np.nanmedian(d.LS_1000):+.2f}); median λ = "
                   f"{np.nanmedian(d.lam):+.2f}.", "",
                   f"Tokenizer sub-test: input-format pattern (no step at 100, step at 1000) in {ti}/{tn}; "
                   f"learned-length pattern (step at both) in {tl}/{tn}.", "", table(d, cols), ""]
            summary.append(dict(data=tag, layer=lab, verdict=v, median_LS=np.nanmedian(d.LS), n=len(d)))
        # all layers, median over models
        g = df.groupby("layer")[["LS", "LS_100", "LS_1000", "J", "lam", "rho"]].median().reset_index()
        md += [f"### {tag}: all layers (median over models, point estimates, exploratory)", "",
               table(g, [("layer", lambda r: str(int(r.layer))), ("LS", lambda r: _f(r.LS)),
                         ("LS 100", lambda r: _f(r.LS_100)), ("LS 1000", lambda r: _f(r.LS_1000)),
                         ("J", lambda r: _f(r.J)), ("λ", lambda r: _f(r.lam)), ("ρ", lambda r: f"{r.rho:.2f}")]), ""]
    p = os.path.join(out, "E13.csv")
    if os.path.exists(p):
        df = pd.read_csv(p); d = df[df.role == "selected"]
        cols3 = [("model", lambda r: r.model), ("reading", lambda r: r.reading)] + cols[1:] + \
                [("R 3-digit", lambda r: _f(r.R_3, 1)), ("R 4-digit", lambda r: _f(r.R_4, 1))]
        md += ["## D3: Pythia and Qwen, every number 10–9999 (replication + tokenizer contrast)", "", table(d, cols3), ""]
        for _, r in d.iterrows():
            summary.append(dict(data=f"E13 {r.model}", layer=r.reading, verdict=r["class"], median_LS=r.LS, n=1))
        g = df.groupby(["model", "reading", "layer"])[["LS", "LS_100", "LS_1000", "J", "lam", "rho", "R_3", "R_4"]].first().reset_index()
        md += ["### D3: all layers (point estimates, exploratory)", "",
               table(g, [("model", lambda r: r.model), ("reading", lambda r: r.reading), ("layer", lambda r: str(int(r.layer))),
                         ("LS", lambda r: _f(r.LS)), ("LS 100", lambda r: _f(r.LS_100)), ("LS 1000", lambda r: _f(r.LS_1000)),
                         ("λ", lambda r: _f(r.lam)), ("R3", lambda r: _f(r.R_3, 1)), ("ρ", lambda r: f"{r.rho:.2f}")]), ""]
    p = os.path.join(out, "power_synthetic.csv")
    if os.path.exists(p):
        pw = pd.read_csv(p)
        md += ["## Power (synthetic plants)", "", table(pw, [("design", lambda r: r.design), ("noise", lambda r: str(r.noise)),
               ("plant", lambda r: r.plant), ("correct", lambda r: f"{r.correct_rate:.2f}"), ("LS", lambda r: _f(r.LS_median)),
               ("λ", lambda r: _f(r.lam_median)), ("classes", lambda r: r.classes)]), ""]
    if summary:
        md = md[:4] + ["## Summary", "", table(pd.DataFrame(summary), [("data", lambda r: r.data), ("layer / reading", lambda r: r.layer),
                       ("verdict / class", lambda r: r.verdict), ("median LS", lambda r: _f(r.median_LS)), ("n", lambda r: str(r.n))]), ""] + md[4:]
    open(os.path.join(out, "RESULTS.md"), "w").write("\n".join(md))
    return os.path.join(out, "RESULTS.md")
