"""E11 cross-check: every alpha definition against every beta measurement, plus partial correlations.

    python experiments/E11_alpha_probed_numbers/cross_check.py      (from the repo root; after compute.py)

alpha versions
  old        alpha_OLS over all integers 1-10000 (paper)
  new        E11: log(median count) on log(group centre), the 4 E01 groups
  pooled     OLS of log c(n) on log n over every probed E01 number individually (E10 auditor 13's variant)
  new_e06    same as `new`, but on E06's 10 groups (centres 10*10^(g/3), +-10%)
beta versions
  e01        E01 beta_direct at E01's joint-score layer (seeds 42-44 select, 45-47 evaluate)
  e06_e01L   E06 beta_cont at E01's layer
  e06_own    E06 beta_cont at E06's own joint-score layer
  e06_L8     E06 beta_cont with every model at layer 8
Outputs: cross_check.csv (r, rho and p for each pair), partial.csv (controlling for code+math share).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t as tdist

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

pm = pd.read_csv(OUT / "per_model.csv").set_index("model")
fine = pd.read_csv(ROOT / "results/beta_fine/fine_vs_data.csv").set_index("recipe")
fixed = pd.read_csv(ROOT / "results/beta_fine/fixed_layer_beta.csv", index_col=0)

E01_GROUPS = [np.arange(1, 40), np.arange(80, 120), np.arange(980, 1020), np.arange(9980, 10001)]
E06_GROUPS = []
for g in range(10):
    c = 10 * 10 ** (g / 3)
    E06_GROUPS.append(np.arange(int(np.ceil(c * 0.9)), int(np.floor(min(c * 1.1, 10000))) + 1))


def counts(recipe):
    c = pd.read_csv(ROOT / f"results/corpus_alpha_datadecide/exact_100b_{recipe}/counts_seed_2.csv")
    return c.set_index(c.columns[0])[c.columns[1]].reindex(range(0, 10001), fill_value=0).to_numpy(float)


def slope_medians(c, groups):
    x = [np.mean(np.log(g)) for g in groups]
    y = [np.log(np.median(c[g])) for g in groups]
    return float(np.polyfit(x, y, 1)[0])


def slope_pooled(c, groups):
    n = np.concatenate(groups)
    keep = c[n] > 0
    return float(np.polyfit(np.log(n[keep]), np.log(c[n][keep]), 1)[0])


rows = []
for m in pm.index:
    c = counts(m)
    rows.append({"model": m, "old": pm.loc[m, "alpha_old"], "new": pm.loc[m, "alpha_new"],
                 "pooled": slope_pooled(c, E01_GROUPS), "new_e06": slope_medians(c, E06_GROUPS),
                 "e01": pm.loc[m, "beta"], "e06_e01L": fine.loc[m, "beta_cont_oldlayer"],
                 "e06_own": fine.loc[m, "beta_cont"], "e06_L8": fixed.loc[m, "beta_cont_L8"],
                 "codemath": pm.loc[m, "code_share"] + pm.loc[m, "math_share"]})
d = pd.DataFrame(rows).set_index("model")
assert np.allclose(d.new, [slope_medians(counts(m), E01_GROUPS) for m in d.index], atol=1e-3)

out = []
for a in ["old", "new", "pooled", "new_e06"]:
    for b in ["e01", "e06_e01L", "e06_own", "e06_L8"]:
        r, p = pearsonr(d[a], d[b])
        s, ps = spearmanr(d[a], d[b])
        out.append({"alpha": a, "beta": b, "n": len(d), "pearson_r": round(r, 3), "pearson_p": float(f"{p:.2g}"),
                    "spearman_rho": round(s, 3), "spearman_p": float(f"{ps:.2g}")})
pd.DataFrame(out).to_csv(OUT / "cross_check.csv", index=False)


def partial(x, y, z):
    X = np.c_[np.ones(len(z)), z]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = len(x) - 3
    p = float(2 * tdist.sf(abs(r) * np.sqrt(df / (1 - r * r)), df))
    return round(r, 3), float(f"{p:.2g}")


part = []
for a in ["old", "new", "pooled"]:
    r, p = partial(d[a].to_numpy(), d.e01.to_numpy(), d.codemath.to_numpy())
    part.append({"alpha": a, "beta": "e01", "control": "code_share + math_share", "partial_r": r, "p": p})
pd.DataFrame(part).to_csv(OUT / "partial.csv", index=False)
d.round(4).to_csv(OUT / "cross_check_inputs.csv")
print(pd.DataFrame(out).to_string(index=False))
print(pd.DataFrame(part).to_string(index=False))
