# 15 - Experimental design in alpha space (leverage, outliers, collinearity, support)

Scope: data side only. Inputs: `alpha_ols`, `alpha_mle`, `alpha_r2` from `results/datadecide/alpha_beta.csv` (beta columns not loaded);
`family` and the `F*` columns of `results/datadecide/count_features.csv` (`old_direct`, `beta_resid` dropped on load);
`number_density`, `code_share`, `math_share` from the `recipes` list of `results/datadecide/confounders.json` (no other key opened);
`results/corpus_alpha_datadecide/alpha_seed2.csv` (stream sizes). No beta value was read, plotted or related to alpha.

Code: `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/s15/design.py` and `design2.py`;
full output tables: `.../agents/s15/design_table.csv`, `.../agents/s15/design_table2.csv`.

## A-priori thresholds (fixed before looking at the values)

| Id | Rule | Threshold | Justification |
|---|---|---|---|
| L1 | Leverage in a simple regression on alpha_ols (p = 2) | h_ii > 2p/n = 0.16 (flag); > 3p/n = 0.24 (high) | Hoaglin & Welsch (1978) / Belsley-Kuh-Welsch convention |
| L2 | Same for alpha_mle | same | alternative alpha definition |
| L3 | Leverage in the confounder design alpha + code + math + density (p = 5) | h_ii > 2p/n = 0.40 | same convention |
| L4 | Leverage in the family fixed-effects design (6 family dummies + alpha, p = 7) | h_ii > 2p/n = 0.56 | same convention |
| O1 | Alpha outlier: Grubbs test and externally studentized (leave-one-out) deviation | Bonferroni alpha = 0.05: G > 2.82, abs(t_del) > 3.48 | standard single-outlier tests for n = 25 |
| M1 | Alpha measurement precision | recipe gaps smaller than the seed-to-seed spread (0.003 OLS, E04 README lines 39-42) count as ties | measurement resolution |
| C1 | Common support (hull): keep alpha in [2nd-lowest family min, 2nd-highest family max] | outcome-blind | standard overlap/common-support trimming (Crump et al. 2009 logic): no extrapolation from one family |
| C2 | Common support (strict): keep alpha inside the [min,max] range of >= 2 families | outcome-blind | stricter version of C1 |
| B1 | Family balance: weight each recipe 1/n_family (or use family means / within-family FE) | outcome-blind | clustered design, 6 families of size 1-7 |

## (1) Per-model table (all 25)

h_ols, h_mle: leverage in y ~ alpha. share = (x - mean)^2 / Sxx (that point's share of the alpha spread that identifies a slope). t_del: studentized leave-one-out deviation. h_multi: leverage in [1, alpha, code, math, density]. h_FE: leverage in family-FE design. h_bal: leverage with 1/n_f weights. cover: number of family ranges containing the point. C1/C2: kept (K) / dropped (D).

| recipe | family | alpha_ols | alpha_mle | alpha_r2 | t_del | h_ols | share | h_mle | h_multi | h_FE | h_bal | cover | C1 | C2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dolma1_6plus | dolma | -1.4290 | -1.1230 | 0.801 | 2.01 | **0.183** | 0.143 | 0.041 | 0.216 | 0.234 | 0.133 | 1 | D | D |
| dolma1_7-no-reddit | dolma | -1.4692 | -1.1268 | 0.806 | 1.49 | 0.125 | 0.085 | 0.043 | 0.261 | 0.178 | 0.090 | 1 | D | D |
| dolma1_7 | dolma | -1.4703 | -1.1284 | 0.806 | 1.48 | 0.123 | 0.083 | 0.045 | 0.234 | 0.177 | 0.089 | 1 | D | D |
| dolma1_7-no-flan | dolma | -1.4763 | -1.1294 | 0.806 | 1.40 | 0.116 | 0.076 | 0.046 | 0.242 | 0.173 | 0.084 | 1 | D | D |
| falcon | falcon | -1.5052 | -1.0897 | 0.789 | 1.07 | 0.085 | 0.045 | 0.067 | 0.355 | 0.393 | 0.062 | 2 | K | K |
| dclm-25p-dolma1.7-75p | mix | -1.5130 | -1.1296 | 0.806 | 0.98 | 0.078 | 0.038 | 0.046 | 0.085 | 0.347 | 0.113 | 3 | K | K |
| dclm-50p-dolma1.7-50p | mix | -1.5408 | -1.1290 | 0.803 | 0.67 | 0.058 | 0.018 | 0.045 | 0.074 | 0.333 | 0.083 | 3 | K | K |
| c4 | c4 | -1.5625 | -1.0743 | 0.770 | 0.43 | 0.048 | 0.008 | 0.105 | 0.153 | **1.000** | 0.203 | 4 | K | K |
| dolma1_7-no-code | dolma | -1.5689 | -1.1123 | 0.794 | 0.36 | 0.046 | 0.006 | 0.041 | **0.995** | 0.240 | 0.032 | 3 | K | K |
| dolma1_7-no-math-code | dolma | -1.5720 | -1.0956 | 0.790 | 0.33 | 0.045 | 0.005 | 0.056 | 0.115 | 0.246 | 0.031 | 3 | K | K |
| dclm-75p-dolma1.7-25p | mix | -1.5770 | -1.1299 | 0.799 | 0.28 | 0.043 | 0.003 | 0.046 | 0.067 | 0.349 | 0.061 | 2 | K | K |
| falcon-and-cc | falcon | -1.5813 | -1.0858 | 0.792 | 0.23 | 0.042 | 0.002 | 0.075 | 0.102 | 0.202 | 0.030 | 1 | K | D |
| dclm-baseline | dclm | -1.5965 | -1.1298 | 0.798 | 0.07 | 0.040 | 0.000 | 0.046 | 0.086 | 0.210 | 0.024 | 2 | K | K |
| dclm-baseline-qc-20p | dclm | -1.6060 | -1.1694 | 0.783 | -0.03 | 0.040 | 0.000 | 0.138 | 0.074 | 0.192 | 0.024 | 2 | K | K |
| dclm-baseline-qc-10p | dclm | -1.6142 | -1.1946 | 0.787 | -0.12 | 0.041 | 0.001 | **0.256** | 0.092 | 0.180 | 0.024 | 2 | K | K |
| falcon-and-cc-qc-orig-10p | falcon | -1.6482 | -1.1572 | 0.795 | -0.48 | 0.050 | 0.010 | 0.098 | 0.184 | 0.171 | 0.034 | 2 | K | K |
| dclm-baseline-qc-7p-fw2 | dclm | -1.6634 | -1.1227 | 0.783 | -0.65 | 0.057 | 0.017 | 0.041 | 0.063 | 0.143 | 0.034 | 2 | K | K |
| falcon-and-cc-qc-10p | falcon | -1.6786 | -1.1600 | 0.773 | -0.82 | 0.067 | 0.027 | 0.106 | 0.328 | 0.200 | 0.046 | 2 | K | K |
| falcon-and-cc-qc-20p | falcon | -1.6794 | -1.1405 | 0.793 | -0.82 | 0.068 | 0.028 | 0.060 | 0.247 | 0.201 | 0.047 | 2 | K | K |
| falcon-and-cc-qc-tulu-10p | falcon | -1.6907 | -1.0997 | 0.816 | -0.95 | 0.076 | 0.036 | 0.051 | 0.099 | 0.219 | 0.053 | 2 | K | K |
| fineweb-edu | fineweb | -1.6988 | -1.0541 | 0.778 | -1.04 | 0.083 | 0.043 | **0.182** | 0.101 | 0.514 | 0.173 | 2 | K | K |
| dclm-baseline-qc-7p-fw3 | dclm | -1.7193 | -1.1090 | 0.760 | -1.28 | 0.104 | 0.064 | 0.042 | 0.177 | 0.186 | 0.062 | 2 | K | K |
| dclm-baseline-qc-fw-10p | dclm | -1.7230 | -1.0911 | 0.766 | -1.32 | 0.108 | 0.068 | 0.064 | 0.182 | 0.192 | 0.064 | 2 | K | K |
| dclm-baseline-qc-fw-3p | dclm | -1.7307 | -1.0924 | 0.746 | -1.42 | 0.117 | 0.077 | 0.062 | 0.250 | 0.205 | 0.070 | 2 | K | K |
| fineweb-pro | fineweb | -1.7617 | -1.0503 | 0.795 | -1.80 | 0.159 (just under 0.16) | 0.119 | **0.199** | 0.220 | 0.514 | 0.334 | 1 | D | D |

(mix rows abbreviate `dclm-baseline-XXp-dolma1.7-YYp`. Bold marks a value over its threshold.)

## (2) Findings per criterion, with evidence

### Distribution, clusters, gaps (alpha_ols)
- n = 25, mean -1.6030, SD 0.0940, median -1.5965. Shapiro-Wilk W = 0.964, p = 0.49: no departure from normality, so no heavy tail (design.py output).
- Gaps between sorted neighbours: median 0.0089, mean 0.0139, largest 0.0402 (dolma1_7-no-reddit to dolma1_6plus), next 0.0340 (falcon-and-cc-qc-orig-10p to dclm-baseline-qc-10p) and 0.0310 (fineweb-pro to dclm-baseline-qc-fw-3p). No gap is large enough to split the design into separate clusters. The sample is close to a continuum.
- Two loose end clusters. Top: the 4 dolma recipes that keep code+math (dolma1_6plus, dolma1_7, -no-reddit, -no-flan; alpha -1.43 to -1.48) sit alone above every non-dolma recipe (the highest non-dolma value is falcon at -1.505). Together they carry **38.6 % of Sxx** (design2.py: `4 code-dolma share Sxx 0.386`). Bottom: fineweb-pro (-1.762) plus the three fastText-filtered DCLM recipes qc-fw-3p / qc-fw-10p / qc-7p-fw3 (-1.72 to -1.73). The top 8 points by share hold 71 % of Sxx.
- Ties within measurement precision (M1, gap < 0.003): falcon-and-cc-qc-10p vs -qc-20p (0.0008), dolma1_7 vs -no-reddit (0.0011). no-math-code vs no-code (0.0031) and qc-fw-10p vs qc-7p-fw3 (0.0038) are borderline. This matters only for rank-based methods. It does not change values.

### O1 outliers: none
Grubbs G = 1.85 against a critical value of 2.82. Max abs(t_del) = 2.01 (dolma1_6plus) against 3.48. No recipe's alpha is an extreme outlier.

### L1/L2 leverage
- alpha_ols: **dolma1_6plus h = 0.183 > 0.16** (alone in its share: 14.3 % of Sxx). fineweb-pro h = 0.1588 is right at the threshold (11.9 % of Sxx). Nothing exceeds 3p/n = 0.24. Dropping either point shrinks SD(alpha) by 5.8 % and 4.4 % respectively (`sd_loo_pct`), which is moderate.
- alpha_mle ranks recipes differently (r(OLS, MLE) = -0.25, Spearman -0.20). Its flags are dclm-baseline-qc-10p (0.256 > 0.24), fineweb-pro (0.199) and fineweb-edu (0.182). Only fineweb-pro is flagged under both definitions. Which points carry the leverage therefore depends on which alpha is used.

### L3 confounder design: a near-singular point
**dolma1_7-no-code has h_multi = 0.995** (threshold 0.40). The cause is a near-collinearity between the controls themselves: code_share and math_share have r = 0.83, and both are nonzero only for dolma and the mix recipes (design2.py listing). dolma1_7-no-code is the only recipe with math > 0 and code = 0 (0.0177, 0). So any model containing both code_share and math_share identifies their separate effects from this single recipe. That is a design property of the control set, not a defect of the recipe. The remedy is to merge code+math into one share or use only one of them, not to drop the recipe.

### Collinearity of alpha with family and content
- Family explains **eta^2 = 0.672** of alpha_ols variance (one-way ANOVA F = 7.79, p = 0.0004). VIF of alpha given family = 3.05. Only 32.8 % of the alpha spread is within-family.
- Of that within-family spread, falcon holds 38.5 %, dclm 30.8 %, dolma 24.9 %, mix 3.0 %, fineweb 2.8 % and c4 0 % (singleton). So a within-family (FE) analysis effectively rests on 3 families. The single largest contributor is base `falcon` at 22.6 % of within-family SS (h_FE = 0.393 < 0.56, not flagged).
- Content: r(alpha, code_share) = 0.75, r(alpha, math_share) = 0.71, r(alpha, number_density) = 0.30. R^2 of alpha on code+math+density = 0.58 (VIF 2.4); family+content = 0.76 (VIF 4.2). That is below the conventional VIF 5/10 alarm, but high. The top of the alpha range is essentially "dolma with code".
- count_features (non-beta): alpha_ols is nearly the same quantity as F10_slope_1000_9999 (r = 0.97, rho = 0.99), as expected because OLS gives the ~9000 points above 1000 equal weight. It is also strongly related to F17 spikiness (-0.90), F11 curvature (0.87), F21 year excess (0.86) and F25 probe band ratio (0.83). Those features cannot be treated as independent controls for alpha. F14 (R): r = -0.67. F01 numbers seen: r = 0.30.

### L4 family-FE design
No point exceeds 0.56 except **c4 (h = 1.000)**: as a single-recipe family it is fully absorbed by its own dummy and contributes nothing to a within-family alpha slope. That is automatic and harmless. The fineweb pair has h = 0.514 each (2 points per family, near the threshold).

### Measurement validity of alpha
alpha comes from exact counts on each model's actual 100B-token stream. Seed and sample spread is at most 0.003 OLS (E04 README lines 39-42), against a recipe SD of 0.094. alpha_r2 ranges 0.746 (dclm-qc-fw-3p) to 0.816, with no isolated low value. Three recipes trained for more than one epoch because their recipe is smaller than 100B (`alpha_seed2.csv` recipe_tokens: dclm-baseline-qc-10p 82.1B, fineweb-pro 82.7B, dclm-baseline-qc-fw-10p 98.5B). Their counts include the repeated reshuffled epoch, which E04 handles, so their alpha is still the alpha of what the model saw. **No recipe has a measurement-validity problem in alpha.**

## Design rules and kept/dropped lists (all outcome-blind)

| Rule | Dropped | Kept | Resulting design |
|---|---|---|---|
| L1 leverage > 2p/n (alpha_ols) | dolma1_6plus (fineweb-pro borderline at 0.159) | 24 | Do not drop. Report leave-one-out sensitivity. |
| L1 at 3p/n | none | 25 | |
| O1 outlier | none | 25 | |
| C1 common-support hull [-1.7307 (dclm min), -1.5052 (falcon max)] | dolma1_6plus, dolma1_7, dolma1_7-no-reddit, dolma1_7-no-flan, fineweb-pro | 20 (dclm 7, falcon 6, mix 3, dolma 2, c4 1, fineweb 1) | SD 0.071, eta^2 family 0.48, max h 0.195 < 2p/n = 0.20 (no high-leverage point left) |
| C2 strict >= 2-family overlap | C1 list + falcon-and-cc (lies in the 0.019-wide interior gap between dclm max and mix min) | 19 | SD 0.073, eta^2 0.50, max h 0.206 < 0.211. Dropping an interior point is an artefact of using ranges, so C1 is preferred. |
| B1 balanced weights 1/n_f | none (reweights) | 25, effective n = 15.6 | Leverage then concentrates on fineweb-pro (0.334) and c4 (0.203). |
| B1' family means (k = 6) | none | 6 points | fineweb 0.591, dolma 0.477 (threshold 0.67). With 6 points nothing is reliable. |
| FE (within-family) | c4 contributes 0 automatically | 24 effective | Identification comes 94 % from falcon, dclm, dolma. |

## (4) Validity statement
- **Nothing here is a measurement-validity problem. No recipe should be excluded on design grounds.** alpha is exact, precise to well within the recipe spacing, has no statistical outlier (Grubbs, t_del) and no isolated fit-quality failure.
- High leverage (dolma1_6plus; fineweb-pro under both alpha definitions; dclm-qc-10p and fineweb-edu under alpha_mle) and the near-singularity of dolma1_7-no-code in the code+math design are **genuine design properties**. They should be reported and controlled with leave-one-out or leave-one-family-out runs, cluster-robust or family-FE inference, and code+math merged into one control. They are not reasons to delete data.
- The main structural fact: alpha is 67 % family-determined and 58 % content-determined (code/math). The upper alpha tail exists only as the four code-containing dolma recipes, and the lower tail is anchored by fineweb-pro. Any pooled alpha inference therefore relies heavily on dolma-vs-rest contrasts.
- Standard practice (common-support trimming, C1) recommends a **sensitivity analysis**, not a primary exclusion: drop {dolma1_6plus, dolma1_7, dolma1_7-no-reddit, dolma1_7-no-flan, fineweb-pro}, leaving 20 recipes within [-1.731, -1.505]. Report it alongside the full-25 and the within-family (FE) analyses. Fix it a priori, as here, before the outcome is examined.
