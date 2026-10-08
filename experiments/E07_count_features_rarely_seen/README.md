# E07: Count features and R, the rarely-seen count

**Question:** models with the same α can have very different β. What in the exact
number counts explains β beyond α?
**Status:** done, **exploratory** (2026-10-08). The confirmation test is E08.
**Outputs:** `results/datadecide/count_features.csv` (26 features × 25 recipes, plus α, β
and the β left over after α).

## Method
For each recipe's exact counts c(n), n = 0..10000 (E04, seed 2), compute 26 features in
7 groups (`src/count_features.py`):

| Group | Features |
|---|---|
| A amount | total numbers; numbers in 10–9999 |
| B where numbers sit | share per decade (0–9, 10–99, 100–999, 1000–9999); big/small ratio |
| C shape per decade | local slope in each decade; curvature |
| D frequency tiers | how many numbers are rare (< 1k sightings), mid (1k–100k), frequent (≥ 100k), and their mass; **R = # seen < 4000 times** |
| E evenness | entropy, Gini, spikiness around the power law |
| F special numbers | round-10 and round-100 excess, years (1900–2030), powers of 2, 1–20, last digit 0/5 |
| G probe ranges | counts in the 10 E06 prompt bands: top/bottom ratio and slope |

Each feature is compared with β (Spearman), with the part of β that α doesn't explain,
within families, leaving each family out, and with E06's β.

## How to run
```bash
python -m src.count_features      # feature table + robustness checks for R (seconds)
python -m pytest -q tests/test_count_features.py
```

## Headline result
**R = #{n ∈ 10..9999 : c(n) < 4000}**, the number of integers the model saw
fewer than 4,000 times. More rarely-seen numbers go with more compression (lower β).

| | α_OLS | **R** |
|---|---|---|
| Spearman ρ with β | +0.46 | **−0.73** (p = 3e-5) |
| within families | +0.24 (n.s.) | **−0.70** (p = 8e-5) |
| weakest result with one family left out | +0.11 | −0.59 |
| with E06 β (old layer) | +0.52 | −0.63 |
| β variance explained | 32 % | 52 % |
| leave-one-out prediction r | 0.45 | 0.64 |
| in one regression | p = 0.47 | **p = 0.004** |

- **R absorbs α:** α correlates with R (r = −0.67, p = 0.0003). Once R is known, α
  adds nothing.
- **Not a tuned threshold:** K = 1000 / 2000 / 4000 / 8000 give ρ = −0.68 / −0.67 /
  −0.73 / −0.67.
- **It passes Bonferroni** for 26 features (p ≪ 0.05/26).
- **Stable across data seeds:** R varies by ±2 (c4) and ±17 (dolma1_7) across 5 seeds,
  against a range of ~2,000 between recipes.
- **Main caveat, content:** code/math share (Dolma-only) correlates with β at r = 0.85 and
  with R at −0.79. Controlling for code share removes R's signal (partial r = −0.17). R
  still works among the 18 recipes without code (ρ = −0.51, p = 0.03), where α doesn't.
- **Interpretation (hypothesis):** Bayesian shrinkage. Rarely-seen numbers are pulled
  toward a shared default and crowd together. Details and formulas: [DETAILS.md](DETAILS.md).
