import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
R="/home/user/numberline-datadecide/results"
df=pd.read_csv("/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/a13/per_recipe.csv")
rej=[];e06=[]
for r in df.recipe:
    fs=json.load(open(f"{R}/datadecide/{r}.json"))["filter_summary"]
    rej.append({"recipe":r,**{f"rej_g{g}":fs[g]["rejection_rate"] for g in fs},**{f"fail_g{g}":fs[g]["failed_slots"] for g in fs}})
    z=np.load(f"{R}/beta_fine/step69369-seed-default/{r}.npz")
    t=np.concatenate([z[f"seed{s}_targets"] for s in (45,46,47)]); g=np.concatenate([z[f"seed{s}_groups"] for s in (45,46,47)])
    e06.append({"recipe":r,"e06_frac_targets_gt10000_all":(t>10000).mean(),"e06_frac_gt10000_g9":(t[g==9]>10000).mean()})
rj=pd.DataFrame(rej); e=pd.DataFrame(e06)
print(rj.to_string()); print(e.describe().to_string())
m=df.merge(rj,on="recipe")
print("spearman g4 median exposure vs g4 rejection", spearmanr(m.g4_median_ex10000, m.rej_g4))
print("spearman g3 median exposure vs g3 rejection", spearmanr(m.g3_median, m.rej_g3))
print("median of g4 medians", df.g4_median_ex10000.median(), " 0.5x =", 0.5*df.g4_median_ex10000.median())
for g in (1,2,3):
    print(g, "median of medians", df[f"g{g}_median"].median(), "min ratio", (df[f"g{g}_median"]/df[f"g{g}_median"].median()).min(), df.loc[(df[f"g{g}_median"]/df[f"g{g}_median"].median()).idxmin(),"recipe"])
print("g4 ratio min", (df.g4_median_ex10000/df.g4_median_ex10000.median()).min())
print(df[["recipe","g1_mean","g4_mean","g1_gmean","g4_gmean","g4_share"]].to_string())
# robustness: G1 centre = geometric mean of 1..39
gc=np.exp(np.mean(np.log(np.arange(1,40)))); print("geo centre G1",gc)
s=np.log(df.g2_median/df.g1_median)/np.log(100/gc)
print("slope_g1g2 with geo centre range", s.min(), s.max(), "spearman vs alpha", spearmanr(df.alpha_ols_file,s))
