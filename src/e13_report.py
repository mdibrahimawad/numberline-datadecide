"""E13 Stage 1 report: tables, pre-registered verdicts, figures S1-S4.

  python -m src.e13_report --out results/e13_stage1 --data <topology_number_line>/data
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import e13_stage1 as E

ORDER = list(E.MODELS)
FAM_COL = {"digit": "#1f77b4", "chunk": "#d62728"}
MODEL_COL = dict(zip(ORDER, plt.cm.tab10(np.linspace(0, 1, 10))))


def load(out):
    cells = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "cells", "*__S*__*.json")))]
    expo = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "cells", "*__exponent__*.json")))]
    return cells, expo


def tables(cells, expo):
    pos, h1c, h2a, h2b, h2c, rob, kn = [], [], [], [], [], [], []
    for c in cells:
        key = dict(model=c["model"], family=c["family"], subfamily=c["subfamily"], set=c["set"], depth=c["depth"],
                   N=c["N"], token_count=str(c["token_count"]), discovery=c["discovery"], secondary=c["secondary"])
        h = c["h1"]; L = len(h["positions"])
        for p, st in enumerate(h["positions"]):
            pos.append(dict(key, position=p + 1, position_name=["leading", "2nd", "3rd", "4th"][p] if p < L - 1 else "units",
                            **{k: st.get(k) for k in ["K", "O1", "O1_lo", "O1_hi", "O1_label_shuffle_p", "PR", "A1", "V",
                                                       "V_shuffle_p", "template", "R2_line", "R2_circle", "R2_simplex"]},
                            fake_ms_O1=h["fake_ms"]["O1"][p], fake_ms_template=h["fake_ms"]["template"][p],
                            rec_line=h["template_recovery"][f"p{p}_line"], rec_circle=h["template_recovery"][f"p{p}_circle"],
                            rec_simplex=h["template_recovery"][f"p{p}_simplex"]))
        h1c.append(dict(key, O1=",".join(f"{st['O1']:.2f}" if st["O1"] is not None else "nan" for st in h["positions"]),
                        delta=h["delta"], delta_lo=h["delta_lo"], delta_hi=h["delta_hi"], decay_pass=h["decay_pass"],
                        decay_power=h["decay_power"], decay_fp=h["decay_fp"], has_power=h["has_power"], Ra=h["Ra"], Rb=h["Rb"],
                        Rc=h["Rc"], rc_power=h["template_recovery"][f"p{L-1}_circle"] >= 0.8,
                        fake_ms_delta=h["fake_ms"]["delta"], beyond_smooth=h["beyond_smooth"],
                        units_simplex_signal=h["units_simplex_signal"], units_template=h["positions"][-1]["template"]))
        H = c["h2"]
        if H["h2a"]:
            a = H["h2a"]; row = dict(key, levels=str(a["levels"]), n_steps=str(a["n_steps"]), passed=a["pass"],
                                     has_power=a["has_power"], has_power_cumulative=a["has_power_cumulative"],
                                     has_level2=a["has_level2"], b_freeb=a["b_freeb"], **{f"pw_{k}": v for k, v in H["power_ab"].items()})
            for q, lv in enumerate(a["levels"]):
                row.update({f"e{lv}": a["e"][q], f"e{lv}_lo": a["e_lo"][q], f"e{lv}_hi": a["e_hi"][q], f"e{lv}_freeb": a["e_freeb"][q],
                            f"e{lv}_s": a["e_s"][q]})
            for q in range(len(a["diff"])):
                row.update({f"d{q+2}{q+1}": a["diff"][q], f"d{q+2}{q+1}_lo": a["diff_lo"][q], f"d{q+2}{q+1}_hi": a["diff_hi"][q]})
            row.update(perm_p_e1=a.get("perm_p_e1"), perm_p_diff12=a.get("perm_p_diff12"), mean_y_level0=a.get("mean_y_level0"),
                       ctrl_additive_pass=H["controls"]["additive"]["h2a_pass"], ctrl_ms_pass=H["controls"]["matched_smooth"]["h2a_pass"])
            h2a.append(row)
        if H["h2b"]:
            b = H["h2b"]; row = dict(key, passed=b["pass"], has_power=b["has_power"], drop=b["drop"], drop_lo=b["drop_lo"],
                                     drop_hi=b["drop_hi"], beyond_digits=b.get("beyond_digits"), beyond_digits_lo=b.get("beyond_digits_lo"),
                                     beyond_digits_pass=b.get("beyond_digits_pass"),
                                     ctrl_additive_drop=H["controls"]["additive"]["h2b_drop"],
                                     ctrl_ms_drop=H["controls"]["matched_smooth"]["h2b_drop"])
            for k, v in b.items():
                if k.startswith("R2"):
                    row[k] = v
            h2b.append(row)
        cc = H["h2c"]
        h2c.append(dict(key, passed=cc["pass"], has_power=cc["has_power"], **{k: cc.get(k) for k in
                        ["R2g_line", "R2g_tree", "R2g_both", "dR2g", "dR2g_lo", "dR2g_hi", "R2r_line", "R2r_tree", "R2r_both",
                         "within_group_line_R2"]},
                        UVI_lead=cc["tree_lead"]["UVI"], delta_lead=cc["tree_lead"]["delta_mean"],
                        UVI_pref2=cc.get("tree_pref2", {}).get("UVI"), delta_pref2=cc.get("tree_pref2", {}).get("delta_mean"),
                        pw_tree=H["power_c"]["tree"], pw_line=H["power_c"]["line"],
                        ctrl_ms_dR2g=H["controls"]["matched_smooth"]["h2c_dR2g"], ctrl_add_dR2g=H["controls"]["additive"]["h2c_dR2g"]))
        for v, d in c["robust"].items():
            rob.append(dict(key, variant=v, O1=",".join(f"{x:.2f}" if x is not None else "nan" for x in d["O1"]),
                            h2a_e=str(d["h2a_e"]), h2b_drop=d["h2b_drop"], h2c_dR2g=d["h2c_dR2g"]))
        kn.append(dict(key, knee=c["knee"]["knee"], R2_k50=c["knee"]["R2_by_k"]["50"], R2_k1=c["knee"]["R2_by_k"]["1"],
                       R2_k10=c["knee"]["R2_by_k"]["10"]))
    ex = pd.DataFrame([dict(model=e["model"], family=E.MODELS[e["model"]][0], depth=e["depth"], lengths=str(e["lengths"]),
                            counts=str(e["counts"]), bal_acc=e["bal_acc"], bal_acc_shuffled=e["bal_acc_shuffled"],
                            all_ordered=e["all_ordered"], explicit=e["explicit"],
                            heldout_pred=json.dumps(e["heldout_pred"])) for e in expo])
    return {k: pd.DataFrame(v) for k, v in dict(h1_positions=pos, h1_cells=h1c, h2a=h2a, h2b=h2b, h2c=h2c,
                                                   robustness=rob, knee=kn).items()} | {"exponent": ex}


def family_status(df, pass_col, power_col=None):
    n = len(df)
    if n == 0:
        return dict(n=0, powered=0, passed=0, share=np.nan, status="no cells")
    pw = df[df[power_col]] if power_col else df; npw = len(pw)
    share = pw[pass_col].mean() if npw else np.nan
    if power_col and npw < 0.5 * n:
        st = "cannot tell"
    elif share >= 0.70:
        st = "holds"
    elif share <= 0.30:
        st = "does not hold"
    else:
        st = "mixed"
    return dict(n=n, powered=npw, passed=int(pw[pass_col].sum()) if npw else 0, share=share, status=st)


def verdict(df, pass_col, power_col=None, label=""):
    rows = []
    for scope, sub in (("all", df), ("replication", df[~df.discovery]), ("discovery", df[df.discovery])):
        for fam in ("digit", "chunk"):
            rows.append(dict(test=label, scope=scope, family=fam, **family_status(sub[sub.family == fam], pass_col, power_col)))
    R = pd.DataFrame(rows)
    st = {(r.scope, r.family): r.status for r in R.itertuples()}
    both_all = st[("all", "digit")] == "holds" and st[("all", "chunk")] == "holds"
    both_rep = st[("replication", "digit")] == "holds" and st[("replication", "chunk")] == "holds"
    one = (st[("all", "digit")] == "holds") ^ (st[("all", "chunk")] == "holds")
    if both_all and both_rep:
        v = "supported"
    elif any(st[("all", f)] == "cannot tell" for f in ("digit", "chunk")) and not one and not both_all:
        v = "cannot tell"
    elif one or (both_all and not both_rep):
        v = "one family only" if one else "holds in all 9 but not in replication alone"
    else:
        v = "not supported"
    return R, v


def all_verdicts(T):
    h1 = T["h1_cells"]; h1 = h1[(~h1.secondary) & (h1.depth != "~25%")]
    out, summ = [], []
    for lab, df, pc, pw in [("H1 decay", h1, "decay_pass", "has_power"), ("R-a all ordinal", h1, "Ra", None),
                            ("R-b all categorical", h1, "Rb", None), ("R-c units circle", h1[h1.rc_power], "Rc", None)]:
        R, v = verdict(df, pc, pw, lab); out.append(R); summ.append(dict(test=lab, verdict=v))
    a = T["h2a"]; a = a[(~a.secondary) & (a.has_level2)]
    R, v = verdict(a, "passed", "has_power", "H2a carry spikes"); out.append(R); summ.append(dict(test="H2a carry spikes", verdict=v))
    b = T["h2b"]; b = b[~b.secondary]
    R, v = verdict(b, "passed", "has_power", "H2b successor breaks"); out.append(R); summ.append(dict(test="H2b successor breaks", verdict=v))
    c = T["h2c"]; c = c[(~c.secondary) & (c.depth.isin(["~84%", "last"]))]
    R, v = verdict(c, "passed", "has_power", "H2c tree beats line (late)"); out.append(R); summ.append(dict(test="H2c tree beats line (late)", verdict=v))
    return pd.concat(out), pd.DataFrame(summ)


# --------------------------------------------------------------------------- figures
def fig_s1(T, path):
    P = T["h1_positions"]; P = P[~P.secondary]
    fig, axs = plt.subplots(3, 5, figsize=(17, 9), sharey=True)
    for r, s in enumerate(["S2", "S3", "S4"]):
        for k, d in enumerate(E.DEPTH):
            ax = axs[r, k]; sub = P[(P.set == s) & (P.depth == d)]
            for m in ORDER:
                q = sub[sub.model == m].sort_values("position")
                if len(q):
                    ax.errorbar(q.position + (ORDER.index(m) - 4) * 0.03, q.O1, yerr=[q.O1 - q.O1_lo, q.O1_hi - q.O1], marker="o",
                                ms=3, lw=1, capsize=1.5, color=MODEL_COL[m], ls="-" if E.MODELS[m][0] == "digit" else "--",
                                label=m if (r, k) == (0, 0) else None)
            ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(1, 5)); ax.set_ylim(-1, 1.05)
            if r == 0:
                ax.set_title(d)
            if k == 0:
                ax.set_ylabel(f"{s}\nO1 = Spearman(dist, |a-b|)")
            if r == 2:
                ax.set_xlabel("digit position (1 = leading, last = units)")
    fig.legend(loc="lower center", ncol=9, fontsize=8, frameon=False)
    fig.suptitle("S1  Ordinality of the digit code by position (debiased centroid distances; 95% bootstrap CI). "
                 "Solid = digit-splitting, dashed = chunk tokenizers", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96)); fig.savefig(path, dpi=130); plt.close(fig)


def fig_s2(T, data, path):
    """Offset step residual y(n) = s^2 - debiased additive step^2 against n, S3, discovery models x 5 depths."""
    fig, axs = plt.subplots(2, 5, figsize=(18, 6.5))
    for r, m in enumerate(["qwen2.5-1.5b", "pythia-1.4b"]):
        n = E.set_numbers(m, "S3")
        for k in range(5):
            ax = axs[r, k]; X = np.load(E.layer_files(data, m)[k])["X"][n - 1].astype(float); Z = E.prep(X)
            i, j, c = E.steps_of(n); add = E.additive(Z, E.digits(n, 3)); lv = E.h2_levels(c)
            Xr, y, _, _ = E.h2a_fit(Z, n, 3, i, j, c, lv, add); keep = (c == 0) | np.isin(c, lv); nn = n[i][keep]; cc = c[keep]
            for lev, col, sz in ((0, "0.7", 3), (1, "tab:orange", 10), (2, "tab:red", 28)):
                ax.scatter(nn[cc == lev], y[cc == lev], s=sz, color=col, label=f"carry level {lev}" if (r, k) == (0, 0) else None, lw=0)
            ax.set_title(f"{m}  {E.DEPTH[k]}", fontsize=9)
            if k == 0:
                ax.set_ylabel("s²(n) − additive-model step²")
            if r == 1:
                ax.set_xlabel("n (step n → n+1), S3")
    fig.legend(loc="lower center", ncol=3, frameon=False)
    fig.suptitle("S2  Step length beyond the digit code; carries marked (orange = n ends in 9, red = ends in 99)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)


def fig_s2b(T, path):
    A = T["h2a"]; A = A[~A.secondary]
    fig, axs = plt.subplots(1, 3, figsize=(17, 6), sharey=False)
    for k, s in enumerate(["S2", "S3", "S4"]):
        ax = axs[k]; sub = A[A.set == s].copy(); sub["ix"] = [ORDER.index(m) * 6 + E.DEPTH.index(d) for m, d in zip(sub.model, sub.depth)]
        for lv, col, off in ((1, "tab:orange", -0.2), (2, "tab:red", 0.0), (3, "darkred", 0.2)):
            if f"e{lv}" in sub and sub[f"e{lv}"].notna().any():
                q = sub[sub[f"e{lv}"].notna()]; sc = q.mean_y_level0.abs()
                ax.errorbar(q.ix + off, q[f"e{lv}"] / sc, yerr=[(q[f"e{lv}"] - q[f"e{lv}_lo"]) / sc, (q[f"e{lv}_hi"] - q[f"e{lv}"]) / sc],
                            fmt="o", ms=3, color=col, capsize=1.5, lw=0.8, label=f"e{lv} (level {lv})")
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks([ORDER.index(m) * 6 + 2 for m in ORDER]); ax.set_xticklabels(ORDER, rotation=60, fontsize=7)
        ax.set_title(f"{s} (5 depths per model, left→right)"); ax.set_ylabel("carry spike / mean level-0 offset residual")
        ax.legend(fontsize=8)
    fig.suptitle("S2b  Carry spikes beyond the additive digit code (95% cluster-bootstrap CI)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=130); plt.close(fig)


def fig_s3(T, path):
    B = T["h2b"]; B = B[~B.secondary]
    fig, axs = plt.subplots(5, 3, figsize=(13, 15), sharey=True)
    for r, d in enumerate(E.DEPTH):
        for k, s in enumerate(["S2", "S3", "S4"]):
            ax = axs[r, k]; sub = B[(B.set == s) & (B.depth == d)]
            for m in ORDER:
                q = sub[sub.model == m]
                if not len(q):
                    continue
                q = q.iloc[0]; xs = [lv for lv in (0, 1, 2, 3) if f"R2_{lv}" in q and pd.notna(q[f"R2_{lv}"])]
                ax.plot(xs, [q[f"R2_{lv}"] for lv in xs], marker="o", ms=3, color=MODEL_COL[m],
                        ls="-" if E.MODELS[m][0] == "digit" else "--", label=m if (r, k) == (0, 0) else None)
            ax.set_xticks([0, 1, 2, 3])
            if r == 0:
                ax.set_title(s)
            if k == 0:
                ax.set_ylabel(f"{d}\nheld-out R² of z(n+1)")
            if r == 4:
                ax.set_xlabel("carry level of the step")
    fig.legend(loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("S3  Linear successor operator (trained on carry-free steps): held-out R² by carry level", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(path, dpi=130); plt.close(fig)


def fig_s4(T, path):
    C = T["h2c"]; C = C[~C.secondary]
    fig, axs = plt.subplots(3, 1, figsize=(16, 11))
    for k, s in enumerate(["S2", "S3", "S4"]):
        ax = axs[k]; sub = C[C.set == s]; x0 = 0; ticks = []
        for m in ORDER:
            q = sub[sub.model == m].set_index("depth").reindex(E.DEPTH)
            if q.dR2g.isna().all():
                continue
            xs = x0 + np.arange(5); ticks.append((x0 + 2, m))
            ax.bar(xs, q.dR2g, color=FAM_COL[E.MODELS[m][0]], alpha=0.75,
                   yerr=[q.dR2g - q.dR2g_lo, q.dR2g_hi - q.dR2g], capsize=2)
            ax.scatter(xs, q.ctrl_ms_dR2g, marker="x", color="k", s=14, zorder=3, label="matched-smoothness fake" if x0 == 0 else None)
            for xx, hp in zip(xs, q.has_power):
                if hp is False or hp == 0:
                    ax.text(xx, 0, "np", ha="center", va="bottom", fontsize=6)
            x0 += 6
        ax.axhline(0.10, color="g", ls=":", lw=1, label="pre-registered threshold 0.10"); ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([m for _, m in ticks], fontsize=8)
        ax.set_ylabel(f"{s}\nR²(tree) − R²(line)\non held-out leading-digit groups"); ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("S4  Tree vs line on held-out leading-digit groups (bars = 5 depths left→right; blue = digit-splitting, red = chunk; "
                 "np = no power)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e13_stage1")
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    cells, expo = load(a.out); T = tables(cells, expo)
    for k, df in T.items():
        df.to_csv(os.path.join(a.out, f"{k}.csv"), index=False)
    V, S = all_verdicts(T); V.to_csv(os.path.join(a.out, "verdict_families.csv"), index=False)
    S.to_csv(os.path.join(a.out, "verdicts.csv"), index=False)
    print(S.to_string(index=False)); print(V.round(2).to_string(index=False))
    fig_s1(T, os.path.join(a.out, "fig_S1_ordinality_by_position.png"))
    fig_s2(T, a.data, os.path.join(a.out, "fig_S2_step_length_carries.png"))
    fig_s2b(T, os.path.join(a.out, "fig_S2b_carry_spikes.png"))
    fig_s3(T, os.path.join(a.out, "fig_S3_successor_R2_by_level.png"))
    fig_s4(T, os.path.join(a.out, "fig_S4_tree_vs_line.png"))


if __name__ == "__main__":
    main()
