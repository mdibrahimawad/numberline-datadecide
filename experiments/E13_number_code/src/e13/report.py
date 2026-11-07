"""Aggregate every model's cells into tables, apply the pre-registered decision rules (PREREG_E13.md), draw figures."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from . import config as C

DEPTH_GE50 = {4, 5, 6, 7, 8}          # stored-layer slots at >= 50 % depth
OLD_SLOTS = [2, 4, 6, 7, 8]


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def model_dirs(out_root):
    return sorted(d for d in glob.glob(os.path.join(out_root, "*")) if os.path.exists(os.path.join(d, "config.json"))
                  and not os.path.basename(d).startswith(("ckpt_", "dd-")))


def panel_table(d):
    key = os.path.basename(d); fam = C.MODELS.get(key, {}).get("family"); rows = []
    for f in glob.glob(os.path.join(d, "cells", "panel__*.json")):
        J = _load(f)
        if not J or "result" not in J or not isinstance(J["result"], dict) or "m12" not in J["result"]:
            continue
        R = J["result"]; m = R["m12"]; p4 = R["m4"]; row = dict(model=key, family=fam, stage=C.MODELS.get(key, {}).get("stage"),
                                                               cond="_".join(R["condition"]), slot=R["layer_slot"], layer=R["layer"],
                                                               set=R["set"], tier=R["tier"], N=R["N"])
        row.update({k: m.get(k) for k in ("R2_curve", "R2_curve_digits", "R2_curve_units", "R2_shuffled", "block_R2_curve",
                                           "block_R2_curve_digits", "share_f", "share_digits", "share_resid")})
        row.update({k: v for k, v in p4.items() if not isinstance(v, dict)})
        h1 = R.get("h1")
        if h1:
            for q, st in enumerate(h1["positions"]):
                row[f"O1_p{q + 1}"] = st["O1"]; row[f"tmpl_p{q + 1}"] = st["template"]
            row.update(delta=h1["delta"], delta_lo=h1["delta_lo"], decay_pass=h1["decay_pass"], h1_power=h1.get("has_power"),
                       Ra=h1["Ra"], Rb=h1["Rb"], Rc=h1.get("Rc"), units_template=h1["positions"][-1]["template"])
        h2 = R.get("h2") or {}
        a = h2.get("h2a"); b = h2.get("h2b"); c = h2.get("h2c")
        if a:
            row.update(e1=a["e"][0], e1_lo=a["e_lo"][0], e2=a["e"][1] if len(a["e"]) > 1 else None, h2a_pass=a["pass"],
                       h2a_power=a.get("has_power"), has_level2=a["has_level2"])
        if b:
            row.update(h2b_drop=b["drop"], h2b_pass=b["pass"], h2b_power=b.get("has_power"),
                       **{k: v for k, v in b.items() if k.startswith("R2_")})
        if c:
            row.update(dR2g=c["dR2g"], dR2g_lo=c["dR2g_lo"], h2c_pass=c["pass"], h2c_power=c.get("has_power"),
                       within_group_line_R2=c.get("within_group_line_R2"))
        p8 = R.get("p8") or {}
        if p8:
            row.update(knee=p8["random"]["knee"], sharpness=p8["random"]["sharpness"], sharper=p8["sharper_than_controls"],
                       knee_expected=p8["expected_H1"], knee_block=(p8.get("block") or {}).get("knee"))
        rows.append(row)
    T = pd.DataFrame(rows)
    if len(T):                                           # light cells inherit power from the full AVG5 cell (same slot, set)
        full = T[(T.cond == "AVG5_num") & (T.tier == "full")].set_index(["slot", "set"])
        for col in ("h1_power", "h2a_power", "h2b_power", "h2c_power"):
            if col in T and col in full:
                inh = [full[col].get((sl, s)) if (sl, s) in full.index else None for sl, s in zip(T.slot, T.set)]
                T[col + "_used"] = [own if own is not None and own == own else i for own, i in zip(T[col], inh)]
    return T


def family_status(df, pass_col, power_col=None):
    n = len(df)
    if n == 0:
        return dict(n=0, powered=0, share=None, status="no cells")
    pw = df[df[power_col].fillna(False).astype(bool)] if power_col else df; npw = len(pw)
    share = float(pw[pass_col].fillna(False).astype(bool).mean()) if npw else None
    st = "cannot tell" if (power_col and npw < 0.5 * n) else ("holds" if share >= 0.70 else ("does not hold" if share <= 0.30 else "mixed"))
    return dict(n=n, powered=npw, share=share, status=st)


def verdict(T, pass_col, power_col, label, scope_models=None):
    rows = []
    for fam in ("digit", "chunk"):
        sub = T[T.family == fam]
        if scope_models is not None:
            sub = sub[sub.model.isin(scope_models)]
        rows.append(dict(test=label, family=fam, **family_status(sub, pass_col, power_col)))
    st = [r["status"] for r in rows]
    v = "supported" if all(s == "holds" for s in st) else ("cannot tell" if "cannot tell" in st and "holds" not in st
                                                           else ("one family only" if "holds" in st else "not supported"))
    return rows, v


def verdicts(T, stage_models=None):
    out, summ = [], []
    num_reading = T[T.cond.isin(["AVG5_num", "C1_num"]) & T.slot.isin(DEPTH_GE50) & T.set.isin(C.MAIN_SETS)]
    for cond in ("AVG5_num", "C1_num", "C1_next"):
        X = T[(T.cond == cond) & T.set.isin(C.MAIN_SETS)]
        g50 = X[X.slot.isin(DEPTH_GE50)]
        late = X[X.slot.isin([7, 8])]
        tests = [("H1 decay", g50, "decay_pass", "h1_power_used"), ("R-a", g50, "Ra", None), ("R-b", g50, "Rb", None),
                 ("H2a carry spikes", X[X.get("has_level2", False) == True] if "has_level2" in X else X.iloc[0:0], "h2a_pass", "h2a_power_used"),
                 ("H2b successor breaks", X, "h2b_pass", "h2b_power_used"),
                 ("H2c tree beats line (late)", late, "h2c_pass", "h2c_power_used"),
                 ("P8 knee sharper than controls", g50, "sharper", None)]
        for lab, df, pc, pw in tests:
            if pc not in df:
                continue
            if pw and pw not in df:
                pw = None
            rows, v = verdict(df, pc, pw, f"{lab} [{cond}]", stage_models)
            out += rows; summ.append(dict(test=lab, condition=cond, verdict=v))
    return pd.DataFrame(out), pd.DataFrame(summ)


KEY_METRICS = ["R2_curve", "R2_curve_digits", "O1_p1", "delta", "e1", "dR2g", "probe_n_R2", "lin_acc_units"]


def invariance(T):
    """Per (model, slot, set, metric): value of each condition as a fraction of C1/P_next; the H0 rule."""
    rows = []; ref = T[T.cond == "C1_next"].set_index(["model", "slot", "set"])
    for (m, sl, s), g in T.groupby(["model", "slot", "set"]):
        if (m, sl, s) not in ref.index:
            continue
        r = ref.loc[(m, sl, s)]
        for met in KEY_METRICS:
            if met not in g or r.get(met) is None or not np.isfinite(r[met]) or r[met] == 0:
                continue
            vals = g.set_index("cond")[met]; fr = (vals / r[met]).to_dict()
            nat = [fr.get(f"{c}_num") for c in C.NATURAL]
            okn = sum(1 for v in nat if v is not None and np.isfinite(v) and v >= 0.5)
            c1 = fr.get("C1_num")
            rows.append(dict(model=m, slot=sl, set=s, metric=met, ref=float(r[met]), frac_C1_num=c1, frac_AVG5=fr.get("AVG5_num"),
                             natural_ge50=okn, holds=bool(c1 is not None and np.isfinite(c1) and c1 >= 0.5 and okn >= 3),
                             **{f"frac_{k}": v for k, v in fr.items()}))
    I = pd.DataFrame(rows); summ = []
    if len(I):
        for met, g in I.groupby("metric"):
            summ.append(dict(metric=met, cells=len(g), share_holds=float(g.holds.mean()), invariant=bool(g.holds.mean() >= 0.70)))
    return I, pd.DataFrame(summ)


def side_tables(d):
    key = os.path.basename(d); out = {k: [] for k in ("h0_ceilings", "h0_decomp", "h0_transfer", "h0_distance", "h3", "units", "expo", "power", "errors")}
    for f in glob.glob(os.path.join(d, "cells", "*.json")):
        J = _load(f); base = os.path.basename(f)
        if not J:
            continue
        if "error" in J:
            out["errors"].append(dict(model=key, file=base, error=J["error"])); continue
        R = J.get("result"); job = J.get("job", [None] * 5)
        if base.startswith("h0__") and R and "ceilings" in R:
            for c, v in R["ceilings"].items():
                out["h0_ceilings"].append(dict(model=key, slot=job[2], set=job[3], context=c, **v))
            for nm in ("decomposition", "decomposition_with_C1"):
                out["h0_decomp"].append(dict(model=key, slot=job[2], set=job[3], which=nm, **{k: v for k, v in R[nm].items() if k != "contexts"}))
            out["h0_transfer"] += [dict(model=key, slot=job[2], set=job[3], **t) for t in R["transfer"]]
            out["h0_distance"] += [dict(model=key, slot=job[2], set=job[3], **t) for t in R["distance_agreement"]]
        elif base.startswith("h3__") and R and "reliability" in R:
            row = dict(model=key, slot=job[2], set=job[3], reliability=R["reliability"]["reliability"])
            for nm, v in (R.get("frequency") or {}).items():
                row.update({f"{nm}_slope": v["slope_logfreq"], f"{nm}_lo": v["lo"], f"{nm}_hi": v["hi"]})
            cr = R.get("crowding") or {}
            row.update({f"crowd_{k}": v for k, v in cr.items() if not isinstance(v, list)})
            out["h3"].append(row)
        elif base.startswith("units__") and R:
            out["units"].append(dict(model=key, slot=job[2], set=job[3], probe_acc=R["probe_acc"], n_k_pass=R["n_k_pass"],
                                     **{f"order_k{k}": v for k, v in R["order_score"].items()}))
        elif base.startswith("expo__") and R and R.get("result"):
            out["expo"].append(dict(model=key, cond="_".join(R["condition"]), slot=R["layer_slot"], **R["result"]))
        elif base.startswith("panel__") and R and R.get("power_rows"):
            out["power"] += [dict(model=key, cond="_".join(R["condition"]), slot=R["layer_slot"], set=R["set"], **p) for p in R["power_rows"]]
    p7 = _load(os.path.join(d, "cells", "p7.json"))
    return {k: pd.DataFrame(v) for k, v in out.items()}, (p7 or {}).get("result")


def build(out_root, report_dir=None):
    report_dir = report_dir or os.path.join(out_root, "report"); os.makedirs(report_dir, exist_ok=True)
    dirs = model_dirs(out_root); Ts, sides, p7s = [], {}, {}
    for d in dirs:
        Ts.append(panel_table(d)); S, p7 = side_tables(d); p7s[os.path.basename(d)] = p7
        for k, v in S.items():
            sides.setdefault(k, []).append(v)
    T = pd.concat(Ts) if Ts else pd.DataFrame()
    T.to_csv(os.path.join(report_dir, "panel.csv"), index=False)
    for k, v in sides.items():
        pd.concat(v).to_csv(os.path.join(report_dir, f"{k}.csv"), index=False)
    p7rows = []
    for m, r in p7s.items():
        if not r:
            continue
        for cat in ("days", "months"):
            if cat in r:
                for sl, v in r[cat]["by_slot"].items():
                    p7rows.append(dict(model=m, category=cat, slot=int(sl), probe_acc=v["probe_acc"], n_k_pass=v["n_k_pass"],
                                       passed=v["passed"], works=r[cat]["works"], **{f"order_k{k}": s for k, s in v["order_score"].items()}))
    pd.DataFrame(p7rows).to_csv(os.path.join(report_dir, "p7_positive_control.csv"), index=False)
    res = dict(models=[os.path.basename(d) for d in dirs], p7_control_works={m: (r or {}).get("control_works") for m, r in p7s.items()})
    if len(T):
        V, S = verdicts(T); V.to_csv(os.path.join(report_dir, "verdict_families.csv"), index=False)
        S.to_csv(os.path.join(report_dir, "verdicts.csv"), index=False)
        I, IS = invariance(T); I.to_csv(os.path.join(report_dir, "invariance_cells.csv"), index=False)
        IS.to_csv(os.path.join(report_dir, "invariance_summary.csv"), index=False)
        res.update(verdicts=S.to_dict("records"), invariance=IS.to_dict("records"))
        try:
            figures(T, I, pd.DataFrame(p7rows), report_dir)
        except Exception as e:                       # figures never block the tables
            res["figure_error"] = f"{type(e).__name__}: {e}"
    json.dump(res, open(os.path.join(report_dir, "summary.json"), "w"), indent=2, default=str)
    return res


def figures(T, I, P7, rd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted(T.model.unique())
    # (1) ordinality by digit position: AVG5/num vs C1/num vs C1/next, 5 old depths
    for m in models:
        fig, axs = plt.subplots(3, 5, figsize=(16, 9), sharey=True)
        for r, s in enumerate(C.MAIN_SETS):
            for k, sl in enumerate(OLD_SLOTS):
                ax = axs[r, k]
                for cond, col in (("C1_next", "k"), ("C1_num", "tab:blue"), ("AVG5_num", "tab:red")):
                    q = T[(T.model == m) & (T.cond == cond) & (T.slot == sl) & (T.set == s)]
                    if len(q):
                        ys = [q.iloc[0].get(f"O1_p{p}") for p in range(1, 5)]; ys = [y for y in ys if y is not None and y == y]
                        ax.plot(range(1, len(ys) + 1), ys, "o-", color=col, label=cond if (r, k) == (0, 0) else None)
                ax.axhline(0, color="0.6", lw=0.5); ax.set_ylim(-1, 1.05)
                if r == 0:
                    ax.set_title(C.DEPTH_NAMES[sl])
                if k == 0:
                    ax.set_ylabel(f"{s}\nO1")
        fig.legend(loc="lower center", ncol=3, frameon=False); fig.suptitle(f"{m}: ordinality by digit position (1 = leading)")
        fig.tight_layout(rect=(0, 0.04, 1, 0.96)); fig.savefig(os.path.join(rd, f"fig_ordinality_{m}.png"), dpi=110); plt.close(fig)
    # (2) invariance matrix: median fraction of C1/P_next per metric x condition
    if len(I):
        cols = [c for c in I.columns if c.startswith("frac_") and c not in ("frac_C1_num", "frac_AVG5")]
        M = I.groupby("metric")[cols].median()
        fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.6), 4))
        im = ax.imshow(M.values.astype(float), cmap="RdBu", vmin=-1, vmax=2, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[5:] for c in cols], rotation=70, fontsize=7)
        ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=8); plt.colorbar(im, ax=ax, label="fraction of C1/P_next")
        ax.set_title("Invariance matrix (median over models, layers, sets)"); fig.tight_layout()
        fig.savefig(os.path.join(rd, "fig_invariance_matrix.png"), dpi=120); plt.close(fig)
    # (3) positive control
    if len(P7):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for ax, cat in zip(axs, ("days", "months")):
            for m, g in P7[P7.category == cat].groupby("model"):
                ax.plot(g.slot, g[[f"order_k{k}" for k in range(2, 7)]].median(1), "o-", label=m)
            ax.axhline(0.75, color="g", ls=":"); ax.set_title(f"P7 {cat}: circular-order score (median over k)"); ax.set_xlabel("stored layer slot")
        axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(rd, "fig_positive_control.png"), dpi=120); plt.close(fig)
    # (4) tree vs line and successor R2 by level, number reading (AVG5/num)
    A = T[T.cond == "AVG5_num"]
    if "dR2g" in A:
        fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
        for ax, s in zip(axs, C.MAIN_SETS):
            for m in models:
                q = A[(A.model == m) & (A.set == s)].sort_values("slot"); ax.plot(q.slot, q.dR2g, "o-", label=m)
            ax.axhline(0.10, color="g", ls=":"); ax.axhline(0, color="k", lw=0.5); ax.set_title(f"{s}: R²(tree) − R²(line), held-out groups")
        axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(rd, "fig_tree_vs_line.png"), dpi=120); plt.close(fig)
    lv = [c for c in ("R2_0", "R2_1", "R2_2", "R2_3") if c in A]
    if lv:
        fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
        for ax, s in zip(axs, C.MAIN_SETS):
            for m in models:
                q = A[(A.model == m) & (A.set == s) & (A.slot.isin([4, 8]))]
                for _, r in q.iterrows():
                    ax.plot(range(len(lv)), [r[c] for c in lv], "o-", alpha=0.7, label=f"{m} slot {r.slot}")
            ax.set_xticks(range(len(lv))); ax.set_xticklabels(["level 0", "1", "2", "3"][:len(lv)]); ax.set_title(f"{s}: successor operator R²")
        axs[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(rd, "fig_successor_by_level.png"), dpi=120); plt.close(fig)
