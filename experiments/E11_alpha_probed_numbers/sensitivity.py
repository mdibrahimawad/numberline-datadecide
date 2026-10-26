"""E11 sensitivity: alpha-beta correlation (old vs new alpha) under many removal scenarios.

    python experiments/E11_alpha_probed_numbers/sensitivity.py    (after compute.py)
Outputs: sensitivity.csv, leave_one_out.csv in this folder.
"""
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr

OUT = Path(__file__).resolve().parent
d = pd.read_csv(OUT / "per_model.csv")
dolma_family = d.family == "dolma"
dolma_any = d.family.isin(["dolma", "mix"])
code_math = (d.code_share > 0) | (d.math_share > 0)
flags = ["dolma1_7-no-reddit", "falcon-and-cc-qc-10p", "dolma1_7", "fineweb-edu"]
base = ["c4", "fineweb-pro", "fineweb-edu", "falcon", "falcon-and-cc", "dclm-baseline", "dolma1_7", "dolma1_6plus"]
scen = {
    "All 25 models": d.model == d.model,
    "Remove the 4 E10 measurement flags": ~d.model.isin(flags),
    "Remove dolma1_7-no-reddit only (beta does not replicate)": d.model != "dolma1_7-no-reddit",
    "Remove the 6 Dolma models": ~dolma_family,
    "Remove the 3 DCLM-Dolma mixes": d.family != "mix",
    "Remove all 9 Dolma-containing models (Dolma + mixes)": ~dolma_any,
    "Remove all recipes with code or math (8)": ~code_math,
    "Remove the DCLM family (7)": d.family != "dclm",
    "Remove the Falcon family (6)": d.family != "falcon",
    "Remove FineWeb (2)": d.family != "fineweb",
    "Remove C4 (1)": d.family != "c4",
    "Only the 6 Dolma models": dolma_family,
    "Only Dolma-containing models (9)": dolma_any,
    "Only base corpora, no filtered/ablated variants (8)": d.model.isin(base),
    "Only quality-filtered variants (DCLM-qc + Falcon-qc, 10)": d.model.str.contains("-qc"),
}
rows = []
for name, m in scen.items():
    s = d[m]
    row = {"scenario": name, "n": int(m.sum())}
    for a in ("alpha_old", "alpha_new"):
        r, p = pearsonr(s[a], s.beta); rho, q = spearmanr(s[a], s.beta)
        row.update({f"{a}_r": round(r, 2), f"{a}_p": float(f"{p:.2g}"), f"{a}_rho": round(rho, 2), f"{a}_rho_p": float(f"{q:.2g}")})
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "sensitivity.csv", index=False)
loo = []
for i, mname in enumerate(d.model):
    s = d.drop(index=i)
    loo.append({"removed": mname, "r_new": round(pearsonr(s.alpha_new, s.beta)[0], 3), "rho_new": round(spearmanr(s.alpha_new, s.beta)[0], 3),
                "r_old": round(pearsonr(s.alpha_old, s.beta)[0], 3), "rho_old": round(spearmanr(s.alpha_old, s.beta)[0], 3)})
loo = pd.DataFrame(loo); loo.to_csv(OUT / "leave_one_out.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
print(loo.sort_values("r_new").to_string(index=False))
print("LOO new r range", loo.r_new.min(), loo.r_new.max(), "rho range", loo.rho_new.min(), loo.rho_new.max())
