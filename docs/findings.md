# Findings: what we know so far

The current state of knowledge, newest first in each section. Every number comes from a
file in `results/` and can be regenerated with the command shown (see each experiment's
README). n = 25 DataDecide 1B models unless stated. Last updated: 2026-10-08.

## Headline

1. **β is real and measurable.** All 25 models have a clear number line (ρ = 0.90–0.96).
   β differs strongly between recipes (0.55–1.38), while repeating the measurement with
   new prompts moves it by only about ±0.015 (E06). Measured at the same layer, the
   old 4-group β and the new 10-group β agree (r = 0.96).
2. **α (the paper's α_OLS) correlates with β across the 25 models**: r = +0.56
   (p = 0.003) with the paper-protocol β, and r = +0.63 (p = 0.001) with the
   10-group β at the same layer. The direction is the predicted one: a flatter α (more
   large numbers) goes with less compression. α_MLE shows no relation (r ≈ −0.14).
3. **But α is not the direct driver.** The relation is carried mostly by the Dolma
   family (without it, r = 0.22–0.31, not significant). It fails inside the DCLM and
   Falcon families, and models with the *same* α can have very different β
   (dolma1_7 vs dolma1_7-no-reddit: α −1.470 vs −1.469, β 0.98 vs 1.36).
4. **The best predictor found so far is R, the number of integers 10–9999 seen fewer than
   ~4000 times** in training (E07):

   | | α_OLS | R (rarely-seen count) |
   |---|---|---|
   | Spearman ρ with β | +0.46 | **−0.73** (p = 3e-5) |
   | Within families (Spearman, family-demeaned) | +0.24 (n.s.) | **−0.70** (p < 0.001) |
   | Variance of β explained | 32 % | **52 %** |
   | Leave-one-out prediction r | 0.45 | **0.64** |
   | In one regression together | p = 0.47 | **p = 0.004** |

   Once R is known, α adds nothing. α correlates with β mainly because it correlates
   with R (r = −0.67, p = 0.0003). The fitted line is β ≈ 2.79 − 0.275 × R/1000.
5. **Interpretation (hypothesis): Bayesian shrinkage / under-exposure.** A number
   seen *c* times is learned with weight c/(c+K). Rarely-seen numbers are pulled toward
   a shared default, crowd together, and compress the regions of the line where they
   are dense. This fits the data better than "efficient coding" (space ∝ frequency),
   which predicts that amount doesn't matter (infomax metric r ≈ 0).
6. **Biggest confounder: Dolma content.** The share of code (StarCoder) and math
   (Proof-Pile-2) files is non-zero only in Dolma-based recipes, and it correlates with β
   even more strongly (code r = +0.85, math r = +0.76). It also moves together with R
   (r = −0.79). After controlling for code share, neither α (partial r = −0.19) nor R
   (−0.17) stays significant. With a simple "contains Dolma" indicator, R survives
   (partial r = −0.48, p = 0.02) and α doesn't (0.00). Among the **18 recipes with no code
   at all**, R still predicts β (ρ = −0.51, p = 0.03) and α doesn't (ρ = −0.28, n.s.).
   So R's signal is not only "Dolma vs the rest", but the Dolma family carries much of
   it, and content (code, math) remains a live alternative explanation.
7. **Layer matters for "R beats α".** Items 2–4 use the paper-protocol β, with each model at its
   own selected layer. With **every model at the same layer** (E06 data,
   `python -m src.fixed_layer_beta`), α_OLS, R and S predict β about equally well at layers 6–11
   (r ≈ 0.6–0.75). At layer 8, α_OLS is the best (r = +0.75 vs S +0.62, R −0.70). So on DataDecide
   alone we can't say which formula is right. The pre-registered Paloma test (E09) is designed
   to decide: Paloma trained on 150B tokens, and α predicts no effect of the extra data, while S/R
   predict less compression.
8. **Status of the claim:** a strong exploratory finding, **not yet confirmed**. R was
   the best of 26 features tried on the same 25 models (it passes Bonferroni, and the
   threshold can be anywhere in 1000–8000). The confirmation test is planned
   (E08: training checkpoints, where α and content are fixed but R changes).

9. **Out-of-sample test on the 6 Paloma models (E09, pre-registered): no formula
   transfers.** Only 3 models pass the locked layer-8 validity checks. On them, α_OLS has the
   lowest error (MAE 0.19) but misses the pre-set bar (0.15); S overpredicts by ~0.5; R
   predicts β ≈ 2–2.6 against a measured 0.8–1.0 and is **falsified**. The "more data → less
   compression" prediction of S/R goes the wrong way. Across all 5 counted models none of
   the predictors ranks them. Details: `experiments/E09_paloma_out_of_sample/README.md` → Result.

## What holds up (with evidence)

| Claim | Evidence | Experiment |
|---|---|---|
| A 100B training sample has the same α as its whole recipe | c4: 5 seeds within 0.0003 (OLS) of the full 138B recipe; dolma1_7: 5 nearly independent seeds agree to 0.003 (OLS) / 0.00015 (MLE) | E04 |
| The data-order seed doesn't matter | α differs < 0.003, numbers seen < 0.04 %, R by ±2 (c4) / ±17 (dolma1_7) across 5 seeds | E04, E07 |
| α_MLE is unbiased on samples; α_OLS isn't | SlimPajama: MLE within 0.006 at every size from 10M tokens; OLS still off by ~0.03 at 1B | E02 |
| The cheap ~1B-token window sampler gives α_MLE ≈ exact | window minus exact: MLE 0.0003 ± 0.0008; OLS −0.036 ± 0.014 (biased) | E03 vs E04 |
| β is reproducible | seed s.d. of β ≈ 0.015; old vs new protocol at the same layer r = 0.96 | E06 |
| Dose-response in the DCLM → Dolma mixes | α flattens and β rises as Dolma is mixed in (ρ = 0.90) | E05 |
| R beats α, also within families and with any family left out | see table above; threshold sweep K = 1000…8000 gives ρ = −0.67…−0.73 | E07 |
| A second count feature points the same way | "round-hundred excess" (how much 200, 300, … stand out from their neighbours): ρ = −0.76 with β, within families −0.63; correlated with R (ρ = 0.72) | E07 |

## What does *not* hold

| Idea | Result | Experiment |
|---|---|---|
| α_MLE predicts β | r = −0.14 (n = 25) | E03, E05 |
| The efficient-coding law log₁₀β = 1 + γα with one γ | the fitted slope is ~3× too small and the intercept far from 1; constrained fit R² < 0 | E05 (n = 10) |
| Pure efficient coding ("infomax", space ∝ count) | r ≈ 0.01 with β | E05 |
| Efficient coding at a fixed layer, no fitting (Ganguli & Simoncelli: space ∝ count^q; q = 1 infomax, ½ min. absolute error, ⅓ min. squared error) | no single q fits both the level and the differences. q = 1 gets the average β right (0.64 vs 0.72, MAE 0.14) but r = −0.01 across models; q = ½ and ⅓ track the differences (r = +0.80, +0.76 at layer 8; same at layers 6–11) but predict β 2–3× too large (1.34, 2.46). As a ranking, q = ½ is as good as α_OLS (they correlate 0.99; LOO MAE 0.090 vs 0.101 after a fitted line) | `python -m src.efficient_coding`, `results/beta_fine/efficient_coding_L8.csv` |
| R with several cutoffs (counts below 1k / 3k / 10k / 30k / 100k, fixed ×√10 apart, one linear fit) | LOO MAE 0.076 vs 0.101 for α, but the 6-parameter fit is overfit: coefficients alternate in sign (−0.12, −0.25, +1.0, −0.95, +2.1 per 1000 numbers, intercept −16) and it predicts β = −4.6 (mc4) and −2.2 (pile) for Paloma, which is impossible. Not usable as a formula | exploratory, 2026-10-10 |
| The share of large numbers (1000–9999), years, powers of 2, numbers ending in 0/5 | no relation with β | E07 |
| Splitting β into 9 fine gaps (neighbouring ⅓-decade groups) | too noisy (fit error ≈ 65 %); use the continuous fit | E06 |
| Re-selecting the layer with 10 groups | 8/25 models switch layer on near-ties (scores within 0.7–7.7 %), which adds noise; compare at a fixed layer | E06 |

## Open questions

1. **Does R cause β?** Correlation only, so far, and it is confounded with code/math
   content (Dolma). The checkpoint test (E08) varies R inside one model with α and
   content fixed.
2. **Training noise:** how much does β change across training seeds of the *same*
   recipe? (DataDecide released seeds `-4` and `-5` for the 1B models.) This decides
   whether pairs like dolma1_7 vs no-reddit are noise.
3. **dolma1_7 vs dolma1_7-no-reddit** is unexplained by any count feature (R 6,384 vs
   6,279; β 0.98 vs 1.36). Is it noise, or the *context* numbers appear in?
4. **Removing math** (dolma1_7-no-code → no-math-code) drops β from 0.86 to 0.62 with
   almost unchanged counts. Does the math/arithmetic context of numbers matter beyond
   their frequency?
5. **Layer dependence:** with the newly selected layers, the α–β correlation drops to
   0.20–0.37. R was checked on the old-layer β; a fixed-layer analysis from the saved
   per-prompt PCA scores is free to run.

## How the picture evolved

The conclusions changed as more recipes were counted. Kept here so old messages and
notes can be understood in context:

| When | n | What we thought | Why it changed |
|---|---|---|---|
| α_MLE from ~1B windows | 25 | α_MLE doesn't predict β (r = −0.14) | still true |
| First exact α_OLS | 5 | promising (r = +0.86 with β_log, p = 0.06) | small n |
| | 10 | α_OLS fades (r = 0.33–0.49); numbers per token wins within families (r = 0.92) | Falcon recipes have equal α but different β |
| | 14 | "amount beats shape"; family-relative formula β_log ~ α + (numbers − family mean): R² 0.62, LOO 0.41; saturation metric (K ≈ 4000) best single metric | |
| All 25 exact | 25 | α_OLS significant (r = 0.56) but Dolma-driven; family-relative formula drops to R² 0.39, LOO 0.18; decade "power q→0" shape metric r = 0.67 | the 5 Dolma ablations have the flattest α and the highest β |
| New β (E06) | 25 | β reproducible; same-layer α link r = 0.63; re-selected layers weaken it | layer near-ties |
| Count features (E07) | 25 | R (rarely-seen count) is the best predictor and absorbs α | 26 features tested |
| Fixed-layer β (2026-10-10) | 25 | at one common layer α, R and S are about equal (r 0.6–0.75); "R absorbs α" was layer-dependent | removes the per-model layer choice |
