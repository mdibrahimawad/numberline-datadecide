"""E11: alpha measured on the numbers that beta is measured on (E01, 3 gaps, best layer).

    python experiments/E11_alpha_probed_numbers/compute.py      (from the repo root)

Inputs: results/datadecide/alpha_beta.csv and <recipe>.json (E01 beta at the joint-score layer),
results/corpus_alpha_datadecide/exact_100b_<recipe>/counts_seed_2.csv (exact training-stream counts).
Outputs: per_model.csv and results.json in this folder.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
# E01 magnitude groups (utils/prompts.py, half-width 20); counts exist only up to 10000
GROUPS = [np.arange(1, 40), np.arange(80, 120), np.arange(980, 1020), np.arange(9980, 10001)]

a = pd.read_csv(ROOT / "results/datadecide/alpha_beta.csv")
fam = pd.read_csv(ROOT / "results/beta_fine/fine_vs_data.csv")[["recipe", "family"]]
conf = pd.DataFrame(json.load(open(ROOT / "results/datadecide/confounders.json"))["recipes"])[["recipe", "code_share", "math_share"]]


def counts(recipe):
    c = pd.read_csv(ROOT / f"results/corpus_alpha_datadecide/exact_100b_{recipe}/counts_seed_2.csv")
    return c.set_index(c.columns[0])[c.columns[1]].reindex(range(0, 10001), fill_value=0).to_numpy(float)


def alpha_probed(c):
    """Slope of log(median count) on log(group centre), one point per probed group."""
    x = [np.mean(np.log(g)) for g in GROUPS]
    y = [np.log(np.median(c[g])) for g in GROUPS]
    return float(np.polyfit(x, y, 1)[0])


rows = []
for r in a.itertuples():
    j = json.load(open(ROOT / f"results/datadecide/{r.recipe}.json"))
    Q = {int(k): v["q_median"] for k, v in j["selection"]["layer_scores"].items()}
    assert max(Q, key=Q.get) == int(r.selected_layer)          # beta is at the joint-score layer
    c = counts(r.recipe)
    rows.append(dict(model=r.recipe, layer=int(r.selected_layer), beta=r.beta_direct_mean,
                     alpha_old=r.alpha_ols, alpha_new=alpha_probed(c),
                     **{f"median_g{i+1}": float(np.median(c[g])) for i, g in enumerate(GROUPS)}))
d = pd.DataFrame(rows).merge(fam, left_on="model", right_on="recipe").drop(columns="recipe") \
    .merge(conf, left_on="model", right_on="recipe").drop(columns="recipe")
for col in ("beta", "alpha_old", "alpha_new"):
    d[f"rank_{col}"] = d[col].rank(ascending=False).astype(int)
d.round(4).to_csv(OUT / "per_model.csv", index=False)


def corr(x, y):
    r, p = pearsonr(x, y); s, q = spearmanr(x, y)
    return dict(n=int(len(x)), pearson_r=round(float(r), 3), pearson_p=float(f"{p:.2g}"),
                spearman_rho=round(float(s), 3), spearman_p=float(f"{q:.2g}"))


res = {"old alpha (1-10000), all 25": corr(d.alpha_old, d.beta),
       "new alpha (probed numbers), all 25": corr(d.alpha_new, d.beta)}
flags = ["dolma1_7-no-reddit", "falcon-and-cc-qc-10p", "dolma1_7", "fineweb-edu"]   # E10 measurement flags
m = ~d.model.isin(flags); res["new alpha, without the 4 E10 measurement flags"] = corr(d.alpha_new[m], d.beta[m])
m = (d.code_share == 0) & (d.math_share == 0); res["new alpha, only recipes without code/math"] = corr(d.alpha_new[m], d.beta[m])
m = (d.code_share == 0) & (d.math_share == 0); res["old alpha, only recipes without code/math"] = corr(d.alpha_old[m], d.beta[m])
dm = d.assign(x=d.alpha_new - d.groupby("family").alpha_new.transform("mean"), y=d.beta - d.groupby("family").beta.transform("mean"))
res["new alpha, within families (family means removed)"] = corr(dm.x, dm.y)
dm = d.assign(x=d.alpha_old - d.groupby("family").alpha_old.transform("mean"), y=d.beta - d.groupby("family").beta.transform("mean"))
res["old alpha, within families (family means removed)"] = corr(dm.x, dm.y)
fm = d.groupby("family")[["alpha_new", "beta"]].mean(); res["new alpha, family means (6 points)"] = corr(fm.alpha_new, fm.beta)
rng = np.random.default_rng(0); fams = d.family.unique(); boot = []
for _ in range(20000):
    s = pd.concat([d[d.family == f] for f in rng.choice(fams, len(fams))])
    if s.alpha_new.std() > 0 and s.beta.std() > 0:
        boot.append(pearsonr(s.alpha_new, s.beta)[0])
res["new alpha, family-cluster bootstrap 95% CI of r"] = [round(float(v), 2) for v in np.percentile(boot, [2.5, 97.5])]
res["corr(old alpha, new alpha) spearman"] = round(float(spearmanr(d.alpha_old, d.alpha_new)[0]), 3)
json.dump(res, open(OUT / "results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print(d.sort_values("beta", ascending=False)[["model", "family", "layer", "beta", "rank_beta", "alpha_old", "rank_alpha_old", "alpha_new", "rank_alpha_new"]].round(3).to_string(index=False))
