# 13 — Exposure to the exact numbers that beta is measured on

Auditor aspect: how often the tested β targets occur in each model's exact 100B-token training stream, the exposure ratios between groups, whether any group is too rare for its representation to have formed, and whether α_OLS (fit over 1..10000) describes the count slope over the numbers that are actually probed.

Blindness: I used only the `recipe` column of `results/datadecide/alpha_beta.csv` and the α values in `results/corpus_alpha_datadecide/alpha_seed2.csv`. I did not read or compute any β value or any α–β relationship. The only model-side quantity used is the E01 correct-output filter rejection rate from `filter_summary`, which is a property of the prompt filter, not of β.

## 0. What is tested, and what can be counted

- E01 targets: `utils/prompts.py:11-17`, `DEFAULT_HALF_WIDTH = 20`, `default_interval(i) = range(10**i - 20, 10**i + 20)` for i > 1 and `range(1, 40)` for i = 1. E01 config (`results/datadecide/<recipe>.json` → `config.groups = [1,2,3,4]`, `k = 40`) gives:
  - G1 = 1..39
  - G2 = 80..119
  - G3 = 980..1019
  - G4 = 9980..10019
  Targets are drawn uniformly within each group.
- Counts: `results/corpus_alpha_datadecide/exact_100b_<recipe>/counts_seed_2.csv` covers the exact 100.016B-token stream (data-order seed 2) for numbers **0..10000 only**. The file has 10002 lines, and its last row is `10000`. Recomputing α_OLS from these files reproduces `alpha_seed2.csv` to 7e-15 for all 25 recipes.
- **Coverage gap:** 10001..10019 are never counted, which is 19 of the 40 G4 targets (47.5 %). The G4 statistics below therefore use 9980..10000, and the "G4 median" uses 9980..9999 so that the round-number spike at 10000 is left out. E06 has the same problem in its top group: 43.7–46.7 % of E06 group-9 targets are > 10000 (`results/beta_fine/step69369-seed-default/<recipe>.npz`, `seed4x_targets`; mean 0.459 over recipes).
- Script and outputs:
  - `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/a13/analyze.py`
  - `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/a13/extra.py`
  - `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/a13/per_recipe.csv`
  - `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/a13/per_recipe_flags.csv`

Definitions used in the table:
- **"slope tested (pooled)"**: OLS of log c(n) on log n over the 139 countable targets (G1 ∪ G2 ∪ G3 ∪ 9980..10000).
- **"slope group medians"**: OLS of log(group median) on log(group centre), with centres 20, 100, 1000, 10000.
- **Δ** = pooled tested slope − α_OLS.
- **c(10000)/G4med**: size of the 10000 spike relative to the typical G4 number.

## 1. Per-model table (all 25)

| recipe | α_OLS | G1 med | G2 med | G3 med | G4 med (9980–9999) | G4 min | c(10000)/G4med | log10 G1/G2 | log10 G2/G3 | log10 G3/G4 | slope tested (pooled) | slope group medians | Δ pooled−α | E01 G4 reject | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dolma1_7 | -1.470 | 14,295,882 | 1,273,016 | 48,747 | 3,013.0 | 1,836 | 112 | 1.05 | 1.42 | 1.20 | -1.266 | -1.360 | +0.204 | 0.000 | - |
| dolma1_7-no-code | -1.569 | 14,536,710 | 1,263,822 | 45,112 | 1,823.0 | 1,092 | 79 | 1.06 | 1.45 | 1.39 | -1.332 | -1.443 | +0.237 | 0.000 | - |
| dolma1_7-no-math-code | -1.572 | 13,750,964 | 1,171,610 | 43,391 | 1,622.0 | 888 | 75 | 1.07 | 1.43 | 1.43 | -1.337 | -1.451 | +0.235 | 0.000 | - |
| dolma1_7-no-reddit | -1.469 | 14,763,352 | 1,311,984 | 50,739 | 3,015.0 | 1,783 | 114 | 1.05 | 1.41 | 1.21 | -1.266 | -1.362 | +0.204 | 0.000 | - |
| dolma1_7-no-flan | -1.476 | 14,333,338 | 1,276,040 | 48,628 | 2,740.5 | 1,807 | 124 | 1.05 | 1.42 | 1.24 | -1.271 | -1.374 | +0.206 | 0.000 | - |
| dolma1_6plus | -1.429 | 15,004,249 | 1,312,250 | 53,344 | 3,207.0 | 2,044 | 111 | 1.06 | 1.39 | 1.20 | -1.257 | -1.353 | +0.173 | 0.000 | - |
| c4 | -1.563 | 11,443,422 | 934,462 | 34,050 | 1,453.0 | 810 | 65 | 1.09 | 1.44 | 1.36 | -1.334 | -1.438 | +0.228 | 0.016 | - |
| fineweb-pro | -1.762 | 12,006,867 | 1,184,086 | 40,181 | 1,290.5 | 550 | 53 | 1.01 | 1.47 | 1.47 | -1.345 | -1.462 | +0.417 | 0.098 | C5b |
| fineweb-edu | -1.699 | 13,375,806 | 1,293,666 | 51,386 | 1,876.0 | 920 | 65 | 1.01 | 1.40 | 1.43 | -1.297 | -1.421 | +0.402 | 0.080 | C5b |
| falcon | -1.505 | 16,909,958 | 1,309,834 | 54,667 | 2,171.0 | 1,173 | 53 | 1.11 | 1.38 | 1.39 | -1.327 | -1.428 | +0.178 | 0.004 | - |
| falcon-and-cc | -1.581 | 14,313,318 | 1,172,050 | 43,926 | 1,566.0 | 845 | 73 | 1.09 | 1.43 | 1.45 | -1.344 | -1.461 | +0.238 | 0.004 | - |
| falcon-and-cc-qc-10p | -1.679 | 7,252,215 | 609,652 | 20,875 | 760.5 | 405 | 214 | 1.08 | 1.47 | 1.42 | -1.347 | -1.464 | +0.332 | 0.070 | C1b,C2 |
| falcon-and-cc-qc-20p | -1.679 | 8,586,506 | 732,516 | 24,874 | 930.5 | 493 | 149 | 1.07 | 1.47 | 1.41 | -1.345 | -1.462 | +0.334 | 0.036 | C1b |
| falcon-and-cc-qc-orig-10p | -1.648 | 9,123,335 | 718,772 | 27,366 | 1,012.5 | 586 | 200 | 1.10 | 1.42 | 1.42 | -1.330 | -1.454 | +0.319 | 0.020 | - |
| falcon-and-cc-qc-tulu-10p | -1.691 | 13,781,174 | 1,406,300 | 50,969 | 1,761.0 | 1,045 | 92 | 0.99 | 1.44 | 1.46 | -1.311 | -1.442 | +0.379 | 0.000 | C5b |
| dclm-baseline | -1.596 | 12,917,551 | 901,494 | 33,180 | 1,630.0 | 738 | 89 | 1.16 | 1.43 | 1.29 | -1.337 | -1.433 | +0.260 | 0.124 | - |
| dclm-baseline-qc-7p-fw2 | -1.663 | 13,038,184 | 1,066,631 | 44,257 | 1,735.5 | 895 | 123 | 1.09 | 1.38 | 1.39 | -1.307 | -1.423 | +0.357 | 0.091 | - |
| dclm-baseline-qc-7p-fw3 | -1.719 | 13,029,048 | 1,183,880 | 54,060 | 1,877.5 | 711 | 115 | 1.04 | 1.34 | 1.46 | -1.289 | -1.413 | +0.430 | 0.051 | C5b |
| dclm-baseline-qc-fw-3p | -1.731 | 13,109,768 | 1,315,202 | 57,338 | 2,054.5 | 900 | 96 | 1.00 | 1.36 | 1.44 | -1.281 | -1.401 | +0.450 | 0.073 | C5b |
| dclm-baseline-qc-fw-10p | -1.723 | 13,648,003 | 1,292,173 | 54,524 | 1,797.5 | 793 | 105 | 1.02 | 1.37 | 1.48 | -1.299 | -1.429 | +0.424 | 0.108 | C5b |
| dclm-baseline-qc-10p | -1.614 | 18,149,472 | 786,602 | 35,265 | 1,655.5 | 683 | 222 | 1.36 | 1.35 | 1.32 | -1.339 | -1.467 | +0.276 | 0.016 | - |
| dclm-baseline-qc-20p | -1.606 | 15,810,191 | 828,284 | 35,732 | 1,651.0 | 739 | 167 | 1.28 | 1.37 | 1.30 | -1.314 | -1.441 | +0.292 | 0.028 | - |
| dclm-baseline-25p-dolma1.7-75p | -1.513 | 13,813,450 | 1,117,346 | 41,728 | 2,497.0 | 1,539 | 104 | 1.09 | 1.43 | 1.21 | -1.289 | -1.383 | +0.224 | 0.000 | - |
| dclm-baseline-50p-dolma1.7-50p | -1.541 | 13,432,429 | 1,023,174 | 37,438 | 1,968.0 | 1,084 | 105 | 1.12 | 1.44 | 1.26 | -1.312 | -1.411 | +0.229 | 0.000 | - |
| dclm-baseline-75p-dolma1.7-25p | -1.577 | 13,141,323 | 954,954 | 34,662 | 1,804.0 | 858 | 92 | 1.14 | 1.44 | 1.27 | -1.328 | -1.424 | +0.249 | 0.000 | - |

Group means and geometric means are in `per_recipe.csv`. The G4 mean (9980..10000) is 5.4k–25.8k, against medians of 0.76k–3.2k, because the single number 10000 dominates it.

Minimum count of any countable tested number over all 25 recipes:
- G1: 1,826,732
- G2: 185,814
- G3: 8,233
- G4: 405 (falcon-and-cc-qc-10p)

## 2. Exclusion criteria (rule, threshold, justification) and flagged models

I fixed these thresholds from first principles, not by tuning them to any outcome. I set them after seeing the count table but without any β information.

**C1 – absolute rarity (hard).** Rule: the median count of any tested group is below 100 in the 100B-token stream. That is fewer than 1 occurrence per 10⁹ tokens. At that level the Poisson CV of the count is above 10 %, and a single-epoch 1B model has only a handful of gradient updates on the number's own token sequence.
- **Flags: none.** The lowest group median is 760.5 (G4, falcon-and-cc-qc-10p) and the lowest single count is 405.

**C1b – absolute rarity (soft).** Rule: the G4 median is below 1,000. Justification: about 1 occurrence per 10⁸ tokens, an order of magnitude above C1, as a "thin exposure" warning.
- **Flags: falcon-and-cc-qc-10p (760.5) and falcon-and-cc-qc-20p (930.5).** falcon-and-cc-qc-orig-10p is just above, at 1,012.5.

**C2 – relative under-exposure.** Rule: a recipe's group median is below 0.5× the cross-recipe median for that group. The cross-recipe medians are G1 13.65M, G2 1.17M, G3 43.9k and G4 1,797.5.
- **Flags: falcon-and-cc-qc-10p** (G4 ratio 0.42, G3 0.475).
- Its other groups are also low: G1 0.53 and G2 0.52.

**C3 – count measurement noise.** Rule: Poisson CV of the G4 median above 10 %, or seed-to-seed spread of a group median above 10 %.
- **Flags: none.**
- The worst Poisson CV is 1/√760 = 3.6 %.
- Spread across data-order seeds 2/4/5/6198/14:
  - c4: G4 median 1399–1455 (4 %), G3 33.9–34.1k, G1 11.437–11.444M
  - dolma1_7: G4 median 2801–3013 (7 %), G3 48.7–49.6k, c(10000) 335–338k

**C4 – target coverage of the count files (pipeline-level, not per model).** Rule: if more than 10 % of a tested group's targets have no count, the group's exposure is not measured.
- **Flags: all 25 models equally.** 47.5 % of E01 G4 (10001..10019) and about 46 % of E06 group 9 have no count, because counting stops at 10000 (`results/corpus_alpha_datadecide/exact_100b_*/counts_seed_2.csv` ends at 10000; the counting rule in `docs/project_overview.md` keeps 0..10000).
- A related point: the round number 10000 alone is 53×–222× the typical G4 count, and this ratio varies 4× across recipes. Any "mean G4 exposure" is therefore a measure of the 10000 spike. Medians or geometric means should be used.

**C5 – does α_OLS summarise the probed numbers?**
- **C5a (study-level).** Rule: flag if the cross-recipe Spearman ρ between α_OLS and the tested-range slope is below 0.7, or if the tested-range slope differs from α_OLS by more than 0.2 for most recipes.
  - **Flagged.** Spearman(α_OLS, pooled tested slope) = 0.33 (p = 0.11), and Spearman(α_OLS, group-median slope) = 0.49 (p = 0.014).
  - Δ(pooled − α) is +0.17 to +0.45 (median +0.26) and exceeds 0.2 for 23/25 recipes. The tested-range slope is systematically shallower.
  - The tested-range slope spans only 0.09 (pooled, −1.35 to −1.26) or 0.11 (group medians), against 0.33 for α_OLS (−1.76 to −1.43).
  - By segment, α_OLS tracks the G3→G4 slope (Spearman 0.85, p = 6e-8) but ranks recipes the opposite way on the G1→G2 slope (Spearman −0.36, p = 0.08; the same with the G1 centre at its geometric mean, 15.4).
  - Reason: OLS over 1..10000 gives the about 9000 points above 1000 about 90 % of the weight, so α_OLS is in effect the 1000–10000 slope. These are correlations between two data-side measures, not with β.
- **C5b (recipe-level).** Rule: a recipe's Δ deviates from the cross-recipe median Δ (0.26) by more than 0.1. For these recipes α is an especially poor summary of the probed range.
  - **Flags: fineweb-pro (+0.417), fineweb-edu (+0.402), falcon-and-cc-qc-tulu-10p (+0.379), dclm-baseline-qc-7p-fw3 (+0.430), dclm-baseline-qc-fw-3p (+0.450), dclm-baseline-qc-fw-10p (+0.424).**
  - The group-median version flags the same set except tulu-10p.
  - These recipes have a steep 1000–10000 tail (log10 G3/G4 = 1.43–1.48) but a shallow small-number end (log10 G1/G2 = 0.99–1.04).

Unflagged observation: dclm-baseline-qc-10p and -qc-20p have an unusually large G1/G2 ratio (log10 1.36 and 1.28, against 0.99–1.16 for the rest). The cause is excess counts of 1..39, mostly 1..9. This is a data property, recorded here for anyone modelling the first gap.

## 3. Additional evidence for "the representation formed"

The E01 correct-output filter shows that every model reproduced every tested group:
- `failed_slots = 0` for all groups in all 25 models (`results/datadecide/<recipe>.json` → `filter_summary`).
- The G4 rejection rate is 0–12.4 %. The maximum over all groups is 41 % (fineweb-pro G3), and it is still filled.
- For the C1b/C2-flagged models, G4 rejection is 7.0 % (qc-10p) and 3.6 % (qc-20p).

Across recipes, G4 exposure and G4 rejection rate are moderately anti-correlated (Spearman −0.49, p = 0.013). Less exposure gives slightly more copy failures, which fits a real but modest effect of exposure. Because the filter resamples within the group, those failures do not leave holes in the measurement. The filter does make the accepted set conditional on the model being correct, which is a known design property covered in other audits.

## 4. Measurement-validity problem or data property?

| Flag | Models | Type | Exclude? |
|---|---|---|---|
| C1 (median < 100) | none | — | — |
| C1b (G4 median < 1000) | falcon-and-cc-qc-10p, falcon-and-cc-qc-20p | **Data property.** The model really did see the 9980s rarely, and that is the independent variable under study (it is also what R/S measure). Counts are precise (CV < 4 %) and the models pass the copy filter in G4 with no failed slots. | **No.** Excluding on low exposure would truncate the predictor range (selection on X). Report it, and control for it if needed. |
| C2 (< 0.5× typical) | falcon-and-cc-qc-10p | **Data property.** The reduction is near-uniform across groups (0.42–0.53×), so it is less total number mass rather than a different shape. | **No.** |
| C3 (count noise) | none | — | — |
| C4 (10001..10019 uncounted; 10000 spike) | all 25 | **Measurement-validity gap in the data-side pipeline**, the same for every model. Any per-group exposure feature for G4 or E06-g9 is measured on only about 53 % of the targets. | **No model exclusion.** Extend counting to at least 11000, and use medians or geometric means rather than means. |
| C5a (α ≠ tested-range slope) | study-level | **Construct-validity property of α**: α_OLS mostly encodes the 1000–10000 tail and ranks recipes differently from the tested-range slope (ρ = 0.33). | **No.** Report it, and consider tested-range features alongside α. |
| C5b (α especially unrepresentative) | fineweb-pro, fineweb-edu, falcon-and-cc-qc-tulu-10p, dclm-baseline-qc-7p-fw3, dclm-baseline-qc-fw-3p, dclm-baseline-qc-fw-10p | **Data property** (curvature of these recipes' count curves). | **No.** |

**Honest bottom line: none of the 25 models should be excluded on these grounds.**
- Every tested number that can be counted was seen at least 405 times, and every tested group's median is at least 760 occurrences.
- Count noise is below 4 % (Poisson) and below 7 % across seeds.
- Every model passes the copy filter in every group.

The real issues are not about particular models:
1. The count files do not cover 47.5 % of G4 targets (and about 46 % of E06's top group). The 10000 spike dominates any mean G4 exposure.
2. α_OLS is mainly a 1000–10000 tail statistic and is only weakly rank-related to the count slope across the numbers β actually probes. Its range is also about 3× wider than the tested-range slope's.
