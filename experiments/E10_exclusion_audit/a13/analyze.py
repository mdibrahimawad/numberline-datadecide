import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

R = "/home/user/numberline-datadecide/results"
OUT = sys.argv[1]
recipes = pd.read_csv(f"{R}/datadecide/alpha_beta.csv")["recipe"].tolist()  # recipe list only
a2 = pd.read_csv(f"{R}/corpus_alpha_datadecide/alpha_seed2.csv").set_index("recipe")

# E01 groups: utils/prompts.py default_interval, half_width=20
G = {1: np.arange(1, 40), 2: np.arange(80, 120), 3: np.arange(980, 1020), 4: np.arange(9980, 10020)}
G4obs = np.arange(9980, 10001)  # counts only exist for 0..10000

def ols(x, y):
    m = y > 0
    return np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]

rows = []
for r in recipes:
    c = pd.read_csv(f"{R}/corpus_alpha_datadecide/exact_100b_{r}/counts_seed_2.csv").set_index("number")["count"]
    c = c.reindex(np.arange(0, 10001), fill_value=0).to_numpy().astype(float)
    n = np.arange(1, 10001)
    row = {"recipe": r}
    row["alpha_ols_recomp"] = ols(n, c[1:])
    row["alpha_ols_file"] = a2.loc[r, "alpha_ols"]
    row["total_0_10000"] = c.sum()
    for g, idx in G.items():
        idx = idx[idx <= 10000]
        v = c[idx]
        row[f"g{g}_median"] = np.median(v)
        row[f"g{g}_mean"] = v.mean()
        row[f"g{g}_min"] = v.min()
        row[f"g{g}_gmean"] = np.exp(np.mean(np.log(np.maximum(v, 1))))
        row[f"g{g}_share"] = v.sum() / c.sum()
    # group-4 without the round 10000 spike
    v4 = c[np.arange(9980, 10000)]
    row["g4_median_ex10000"] = np.median(v4)
    row["c10000_over_g4median"] = c[10000] / np.median(v4)
    row["c1000_over_g3median"] = c[1000] / np.median(c[G[3]])
    row["c100_over_g2median"] = c[100] / np.median(c[G[2]])
    # ratios between consecutive groups (medians)
    for a, b in [(1, 2), (2, 3), (3, 4)]:
        row[f"log10_med_ratio_g{a}g{b}"] = np.log10(row[f"g{a}_median"] / row[f"g{b}_median"])
    # local slopes
    xs = np.concatenate([G[1], G[2], G[3], G4obs]).astype(float)
    row["slope_tested_pooled"] = ols(xs, c[xs.astype(int)])
    centers = np.array([20, 100, 1000, 10000.0])
    meds = np.array([row[f"g{g}_median"] for g in (1, 2, 3, 4)])
    row["slope_group_medians"] = np.polyfit(np.log(centers), np.log(meds), 1)[0]
    row["slope_g2_g4_medians"] = np.polyfit(np.log(centers[1:]), np.log(meds[1:]), 1)[0]
    # per-decade log-slopes between group medians
    row["slope_g1g2"] = np.log(meds[1] / meds[0]) / np.log(100 / 20)
    row["slope_g2g3"] = np.log(meds[2] / meds[1]) / np.log(10)
    row["slope_g3g4"] = np.log(meds[3] / meds[2]) / np.log(10)
    # E01 filter rejection rates
    js = json.load(open(f"{R}/datadecide/{r}.json"))
    fs = js["filter_summary"]
    row["max_reject_rate"] = max(v["rejection_rate"] for v in fs.values())
    row["g4_reject_rate"] = fs["4"]["rejection_rate"]
    # Poisson CV of the group median count
    row["g4_poisson_cv"] = 1 / np.sqrt(row["g4_median_ex10000"])
    rows.append(row)

df = pd.DataFrame(rows)
df["delta_pooled_minus_alpha"] = df.slope_tested_pooled - df.alpha_ols_file
df["delta_medians_minus_alpha"] = df.slope_group_medians - df.alpha_ols_file
df.to_csv(OUT, index=False)
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 60)
print(df[["recipe", "alpha_ols_file", "alpha_ols_recomp", "g1_median", "g2_median", "g3_median", "g4_median_ex10000", "g4_min", "c10000_over_g4median", "c1000_over_g3median"]].to_string())
print(df[["recipe", "log10_med_ratio_g1g2", "log10_med_ratio_g2g3", "log10_med_ratio_g3g4", "slope_tested_pooled", "slope_group_medians", "slope_g1g2", "slope_g2g3", "slope_g3g4", "delta_pooled_minus_alpha", "delta_medians_minus_alpha", "max_reject_rate"]].to_string())
print("min g4 median over recipes", df.g4_median_ex10000.min(), df.loc[df.g4_median_ex10000.idxmin(), "recipe"])
print("min any tested count", df[["g1_min", "g2_min", "g3_min", "g4_min"]].min().to_dict())
print("spearman alpha_ols vs slope_tested_pooled", spearmanr(df.alpha_ols_file, df.slope_tested_pooled))
print("spearman alpha_ols vs slope_group_medians", spearmanr(df.alpha_ols_file, df.slope_group_medians))
print("spearman alpha_ols vs slope_g3g4", spearmanr(df.alpha_ols_file, df.slope_g3g4))
print("spearman alpha_ols vs slope_g1g2", spearmanr(df.alpha_ols_file, df.slope_g1g2))
print("pearson alpha_ols vs slope_tested_pooled", np.corrcoef(df.alpha_ols_file, df.slope_tested_pooled)[0, 1])
print("range alpha", df.alpha_ols_file.min(), df.alpha_ols_file.max())
print("range slope pooled", df.slope_tested_pooled.min(), df.slope_tested_pooled.max())
print("range slope medians", df.slope_group_medians.min(), df.slope_group_medians.max())
print("max |alpha recomp - file|", (df.alpha_ols_recomp - df.alpha_ols_file).abs().max())
