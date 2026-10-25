# Auditor 03: is α a valid measurement? (25 DataDecide recipes)

Scope: whether α (α_OLS, α_MLE) is a valid one-number summary of each recipe's integer-count
distribution c(n), n = 1..10000, counted on the exact 100B-token seed-2 training stream (E04).
I judged this on measurement grounds only. I did **not** compute or look up any α–β relation,
and I did not open `alpha_beta_correlations.json`. From `results/datadecide/alpha_beta.csv` I read
only `recipe, alpha_source, alpha_ols, alpha_mle, alpha_r2`. From `count_features.csv` I read only
`alpha_ols` and F01, F08–F11 and F20 (to cross-check).
*Disclosure:* the E03 README, which I was told to read, has a headline line on window-α_MLE vs β
(r = −0.14), and E03/DETAILS.md repeats it. I saw that line. It was not used in any criterion here.

Reproduce: `python3 /tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/alpha_audit.py /tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/alpha_audit.json`. The script is
`/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/alpha_audit.py`.
Raw output (all diagnostics per recipe) is
`/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/alpha_audit.json`.
Input: `results/corpus_alpha_datadecide/exact_100b_<recipe>/counts_seed_2.csv`. These are the exact
counts. The `<recipe>/counts_0_to_10000_*.csv` files are the ~1B-token E03 *window* samples, and I
used them only to check pipeline agreement (via `alpha_summary.csv`).

## 1. Pipeline integrity (no problems found)
- **α recomputes exactly.** Recomputing α_OLS, R² and α_MLE from `counts_seed_2.csv` with the
  repo's definitions (`_fit_alpha` in `modal_app/corpus_alpha_app.py`; `fit_alpha_mle` in
  `src/sampling_validation.py`) reproduces `alpha_seed2.csv` to within 7e-15 for all 25 recipes.
- **The tables agree.** `alpha_beta.csv` columns alpha_ols, alpha_mle and alpha_r2 are identical to
  `alpha_seed2.csv` (max diff 0), and every row has `alpha_source = exact_100b`.
- **Counts add up.** Σ_{n=0..10000} c(n) equals `integer_matches` in alpha_seed2.csv and in
  summary.json `per_seed.2` for all 25. Every count file has 10,001 rows. Support is 1.0: every
  n = 1..10000 occurs at least 175 times (minimum: falcon-and-cc-qc-10p), so the OLS zero-dropping
  bias described in E02 does not apply.
- **Slopes agree with count_features.csv.** Its F08/F09/F10, alpha_ols and F01 match my values to
  within 4e-15.
- **Sampling noise is small.** E04/DETAILS.md reports α_OLS s.d. across seeds of 0.0002 (c4) and
  0.0012 (dolma1_7), and α_MLE s.d. of 1e-5 and 6e-5. Window α_MLE minus exact α_MLE is at most
  0.0016 per recipe. α is therefore a precise, reproducible statistic of each stream, and the
  problem described below is **validity, not precision**.
- **Three recipes are smaller than 100B tokens:** fineweb-pro, dclm-baseline-qc-10p and
  dclm-baseline-qc-fw-10p. They train for a second epoch, so their counts include repeated chunks.
  The summary.json `union_of_samples.integer_matches` is lower (1.230B vs 1.487B for fineweb-pro),
  as it should be. This is correct, because it matches what the model saw. α of the union and of the
  stream differ by only 1e-4 (fineweb-pro: −1.76179 vs −1.76170).
- **Documented oddities do not affect counts.** The only ones are the RunPod pod that did not delete
  itself (E04) and the window-OLS bias, which is irrelevant because only exact α_OLS is used.

## 2. Criteria (fixed from documented reference quantities before reading the per-recipe table)
Reference scales:
- **Noise floor:** the RMS of log c(n) differences between two independent seeds, 0.024 (c4, seed 2
  vs 4) and 0.046 (dolma1_7). Command: in alpha_audit, compare `counts_seed_2.csv` with
  `counts_seed_4.csv`.
- **Δ_between = 0.333:** the full spread of α_OLS across the 25 recipes (−1.429 to −1.762), which
  is what α is used to tell apart.

| ID | Criterion | Threshold and justification |
|---|---|---|
| C1 | Goodness of fit | RMS residual of log c(n) around the global power law ≤ 3× the noise floor (≈ 0.15). A real power law fits to within noise. R² ≥ 0.95 is the equivalent check. |
| C2 | Curvature | Spread (max − min) of the local OLS slopes over 10–99, 100–999 and 1000–9999 ≤ Δ_between = 0.333. If the exponent changes inside one recipe by more than the whole between-recipe range, "the" exponent is not one quantity. |
| C3 | Estimator agreement | \|α_OLS − α_MLE\| ≤ Δ_between/2 = 0.167. For a true power law, both estimators target the same exponent. |
| C4 | Fit-range sensitivity | \|α_OLS(1..10000) − α_OLS(1..1000)\| ≤ 0.167. Moving the upper end of the fit range by one decade should move α by less than half the range that separates recipes. |
| C5 | Special-number dominance | Years 1900–2030 make up ≤ 50 % of the counts in 1000–9999, and removing years and multiples of 10 moves α_OLS by ≤ 0.167. |
| R | Recipe-specific outlier | Robust z = (x − median)/(1.4826·MAD) with \|z\| > 3.5 (Iglewicz–Hoaglin) on any diagnostic. This only flags; it does not exclude anything on its own. |

## 3. Results

### 3a. Universal failures: every recipe fails
- **C1 (fit) fails for all 25.** RMS residual is 0.711–1.008 in log units, 15–40× the seed noise
  floor. R² is 0.746–0.816 (alpha_beta.csv `alpha_r2`; min dclm-baseline-qc-fw-3p, max
  falcon-and-cc-qc-tulu-10p). Excluding years and multiples of 10 raises R² only to 0.82–0.91
  (`r2_clean`).
- **C3 (estimator agreement) fails for all 25.** |α_OLS − α_MLE| is 0.306 (dolma1_6plus) to 0.711
  (fineweb-pro). The two estimators also **rank recipes differently**: Spearman ρ(α_OLS, α_MLE) =
  −0.20. fineweb-pro is the steepest by OLS (rank 25/25) and the flattest by MLE (rank 1/25). The
  rank shift between the two is ≥ 12 places for 10 of 25 recipes.
- **The local slope does not change monotonically.** It is about −0.79 on 1–9, −1.27…−1.79 on
  10–99, *flatter* on 100–999 (−0.99…−1.26), then steeper again on 1000–9999 (−1.35…−2.02). The
  10–99 slope ranks recipes almost unrelated to, or opposite from, α_OLS (ρ = −0.26). The 100–999
  slope gives ρ = 0.76.
- **α_OLS is in practice the slope of the last decade.** ρ(α_OLS, slope 1000–9999) = 0.99 and the
  Pearson correlation is 0.97, because 9,000 of the 10,000 fit points lie in 1000–9999. A
  log-binned fit (equal weight per log interval, `a_logbin`) compresses the between-recipe range from
  0.333 to 0.117 and has ρ = 0.19 with α_OLS.
- **MLE depends strongly on the fit range.** α_MLE fitted on 1..10000 gives −1.05…−1.19. Fitted on
  10..9999 it gives −1.12…−1.35, and on 1000..9999 it gives −1.96…−2.23. MLE is dominated by n < 100.
- **C5 (special numbers) partly fails for all 25.** Years 1900–2030 hold 64–82 % of all counts in
  1000–9999 (F20 in count_features.csv, matching my `year_share_d4`), which is 146–246× what a power
  law fitted to non-year 1000–9999 predicts. Even so, removing years and multiples of 10 changes
  α_OLS by only −0.040 to −0.055, and the ranking is unchanged (ρ = 1.00). So years distort the
  *shape* but not α_OLS comparisons.
- **Other spikes are small.** Recurring point spikes are round numbers (5000 at 140–524× its local
  median, then 3000, 4000, 6000), 1024 and 2048, and web or code artifacts: 8080 (dolma family),
  8217 (dclm family, most likely the HTML entity `&#8217;`), 8859 (ISO-8859) and 9001. Each is a
  single point among 10,000, so its effect on α is negligible.

### 3b. Recipe-level criteria C2 and C4: these separate the recipes
Only 4 recipes pass both C2 and C4: **dolma1_7, dolma1_7-no-reddit, dolma1_7-no-flan,
dolma1_6plus**. Their decade-slope spread is 0.305–0.314 and their range sensitivity 0.143–0.157.

**21 recipes fail both C2 and C4** (the two criteria agree exactly):
- dolma1_7-no-code, dolma1_7-no-math-code, c4, falcon, falcon-and-cc
- dclm-baseline, the 3 dclm/dolma mixes, dclm-baseline-qc-20p (spread 0.389–0.558)
- falcon-and-cc-qc-10p, -qc-20p, -qc-orig-10p (spread 0.578–0.612)
- the strongest violations, with spread more than 2× Δ_between: **fineweb-pro (0.850),
  fineweb-edu (0.809), dclm-baseline-qc-7p-fw3 (0.776), dclm-baseline-qc-fw-10p (0.765),
  dclm-baseline-qc-fw-3p (0.757), falcon-and-cc-qc-tulu-10p (0.694),
  dclm-baseline-qc-10p (0.678), dclm-baseline-qc-7p-fw2 (0.670)**. These have very steep
  1000–9999 tails (−1.85…−2.02) together with ordinary 10–99 slopes. Their α_OLS therefore
  describes the tail only.

Robust-z outliers (R):
- **10–99 slope:** dclm-baseline-qc-10p −1.790 (z = −10.5), dclm-baseline-qc-20p −1.679 (z = −7.7)
  and dclm-baseline −1.540 (z = −4.2), against a median of −1.374. The counts in the 90s are
  depleted relative to 10–20, so these recipes are unusual at small n, which α_OLS barely sees.
- **Multiple-of-10 excess:** falcon-and-cc-qc-10p 2.45 (z = 7.4), -qc-20p 2.16, c4 2.12 and
  -qc-orig-10p 2.08, against a median of 1.67. For multiples of 100: falcon-and-cc-qc-10p 24.4 and
  -qc-20p 19.3, against a median of 10.2.
- **c(0) share:** 0.089–0.090 in dolma1_7, -no-reddit and -no-flan, against a median of 0.050.
  n = 0 is not in any α fit, so this does not matter for α.
- **Numbers seen, R², α_OLS − MLE gap, year share and year excess:** no outliers at |z| > 3.5. The
  smallest number of numbers seen is falcon-and-cc-qc-10p with 0.999B, which is still large enough
  that its minimum count is 175.

## 4. Per-recipe table (all 25; exact seed-2 100B counts)
| Recipe | Numbers seen (B) | α_OLS | R² | α_MLE | \|OLS−MLE\| | slope 10–99 | slope 100–999 | slope 1000–9999 | decade-slope spread | \|α_OLS(1..10⁴) − α_OLS(1..10³)\| | α_OLS w/o years & ×10 | year share of 1000–9999 | years ÷ power-law pred. | ×10 excess (median) | C2 (>0.333) | C4 (>0.167) | other notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dolma1_7 | 1.848 | -1.4703 | 0.806 | -1.1284 | 0.342 | -1.351 | -1.115 | -1.420 | 0.306 | 0.150 | -1.4267 | 0.770 | 156× | 1.60 | pass | pass | high c(0) share (not in fit) |
| dolma1_7-no-code | 1.743 | -1.5689 | 0.794 | -1.1123 | 0.457 | -1.355 | -1.065 | -1.546 | 0.481 | 0.251 | -1.5245 | 0.812 | 220× | 1.84 | FAIL | FAIL |  |
| dolma1_7-no-math-code | 1.619 | -1.5720 | 0.790 | -1.0956 | 0.476 | -1.374 | -1.049 | -1.558 | 0.509 | 0.274 | -1.5275 | 0.814 | 225× | 1.83 | FAIL | FAIL |  |
| dolma1_7-no-reddit | 1.897 | -1.4692 | 0.806 | -1.1268 | 0.342 | -1.349 | -1.112 | -1.417 | 0.305 | 0.152 | -1.4259 | 0.772 | 157× | 1.59 | pass | pass | high c(0) share (not in fit) |
| dolma1_7-no-flan | 1.851 | -1.4763 | 0.806 | -1.1294 | 0.347 | -1.351 | -1.114 | -1.428 | 0.314 | 0.157 | -1.4332 | 0.770 | 156× | 1.64 | pass | pass | high c(0) share (not in fit) |
| dolma1_6plus | 1.930 | -1.4290 | 0.801 | -1.1230 | 0.306 | -1.369 | -1.056 | -1.351 | 0.313 | 0.143 | -1.3886 | 0.762 | 147× | 1.74 | pass | pass | high c(0) share (not in fit) |
| c4 | 1.396 | -1.5625 | 0.770 | -1.0743 | 0.488 | -1.389 | -1.059 | -1.543 | 0.484 | 0.271 | -1.5166 | 0.813 | 246× | 2.12 | FAIL | FAIL | round-×10 excess outlier |
| fineweb-pro | 1.487 | -1.7617 | 0.795 | -1.0503 | 0.711 | -1.270 | -1.171 | -2.020 | 0.850 | 0.435 | -1.7078 | 0.762 | 203× | 1.62 | FAIL | FAIL | 2nd epoch (recipe<100B); spread >2× between-recipe range |
| fineweb-edu | 1.720 | -1.6988 | 0.778 | -1.0541 | 0.645 | -1.315 | -1.143 | -1.952 | 0.809 | 0.396 | -1.6463 | 0.733 | 182× | 1.56 | FAIL | FAIL | spread >2× between-recipe range |
| falcon | 1.890 | -1.5052 | 0.789 | -1.0897 | 0.416 | -1.453 | -0.989 | -1.431 | 0.464 | 0.242 | -1.4643 | 0.815 | 214× | 1.81 | FAIL | FAIL |  |
| falcon-and-cc | 1.628 | -1.5813 | 0.792 | -1.0858 | 0.495 | -1.401 | -1.055 | -1.572 | 0.517 | 0.283 | -1.5386 | 0.821 | 231× | 1.89 | FAIL | FAIL |  |
| falcon-and-cc-qc-10p | 0.999 | -1.6786 | 0.773 | -1.1600 | 0.519 | -1.352 | -1.155 | -1.733 | 0.578 | 0.277 | -1.6313 | 0.719 | 171× | 2.45 | FAIL | FAIL | round-×10 excess outlier |
| falcon-and-cc-qc-20p | 1.101 | -1.6794 | 0.793 | -1.1405 | 0.539 | -1.354 | -1.153 | -1.761 | 0.608 | 0.311 | -1.6344 | 0.743 | 164× | 2.16 | FAIL | FAIL | round-×10 excess outlier |
| falcon-and-cc-qc-orig-10p | 1.168 | -1.6482 | 0.795 | -1.1572 | 0.491 | -1.382 | -1.101 | -1.712 | 0.612 | 0.294 | -1.6025 | 0.731 | 150× | 2.08 | FAIL | FAIL | round-×10 excess outlier |
| falcon-and-cc-qc-tulu-10p | 1.749 | -1.6907 | 0.816 | -1.0997 | 0.591 | -1.275 | -1.155 | -1.849 | 0.694 | 0.368 | -1.6442 | 0.782 | 175× | 1.70 | FAIL | FAIL | spread >2× between-recipe range |
| dclm-baseline | 1.485 | -1.5965 | 0.798 | -1.1298 | 0.467 | -1.540 | -1.115 | -1.657 | 0.542 | 0.236 | -1.5516 | 0.786 | 187× | 1.65 | FAIL | FAIL | 10–99 slope robust-z outlier |
| dclm-baseline-qc-7p-fw2 | 1.658 | -1.6634 | 0.783 | -1.1227 | 0.541 | -1.436 | -1.191 | -1.862 | 0.670 | 0.299 | -1.6126 | 0.720 | 154× | 1.58 | FAIL | FAIL | spread >2× between-recipe range |
| dclm-baseline-qc-7p-fw3 | 1.869 | -1.7193 | 0.760 | -1.1090 | 0.610 | -1.382 | -1.209 | -1.986 | 0.776 | 0.372 | -1.6650 | 0.670 | 149× | 1.60 | FAIL | FAIL | spread >2× between-recipe range |
| dclm-baseline-qc-fw-3p | 1.975 | -1.7307 | 0.746 | -1.0924 | 0.638 | -1.341 | -1.260 | -2.017 | 0.757 | 0.380 | -1.6760 | 0.644 | 146× | 1.67 | FAIL | FAIL | spread >2× between-recipe range |
| dclm-baseline-qc-fw-10p | 1.868 | -1.7230 | 0.766 | -1.0911 | 0.632 | -1.359 | -1.223 | -1.988 | 0.765 | 0.378 | -1.6703 | 0.683 | 155× | 1.58 | FAIL | FAIL | 2nd epoch (recipe<100B); spread >2× between-recipe range |
| dclm-baseline-qc-10p | 1.766 | -1.6142 | 0.787 | -1.1946 | 0.420 | -1.790 | -1.112 | -1.688 | 0.678 | 0.186 | -1.5656 | 0.753 | 172× | 1.68 | FAIL | FAIL | 10–99 slope robust-z outlier; 2nd epoch (recipe<100B); spread >2× between-recipe range |
| dclm-baseline-qc-20p | 1.636 | -1.6060 | 0.783 | -1.1694 | 0.437 | -1.679 | -1.125 | -1.683 | 0.558 | 0.201 | -1.5577 | 0.758 | 174× | 1.67 | FAIL | FAIL | 10–99 slope robust-z outlier |
| dclm-baseline-25p-dolma1.7-75p | 1.701 | -1.5130 | 0.806 | -1.1296 | 0.383 | -1.426 | -1.117 | -1.506 | 0.389 | 0.178 | -1.4691 | 0.775 | 163× | 1.60 | FAIL | FAIL |  |
| dclm-baseline-50p-dolma1.7-50p | 1.602 | -1.5408 | 0.803 | -1.1290 | 0.412 | -1.475 | -1.115 | -1.560 | 0.445 | 0.196 | -1.4961 | 0.779 | 170× | 1.60 | FAIL | FAIL |  |
| dclm-baseline-75p-dolma1.7-25p | 1.534 | -1.5770 | 0.799 | -1.1299 | 0.447 | -1.511 | -1.117 | -1.619 | 0.501 | 0.221 | -1.5323 | 0.785 | 182× | 1.64 | FAIL | FAIL |  |

Column notes:
- Slopes are OLS fits of log c(n) on log n over each decade (the same as count_features F08–F10).
- "×10 excess" is the median over non-year multiples of 10 in 110..9990 of
  c(n)/√(c(n−1)·c(n+1)).
- "years ÷ power-law pred." is the year mass divided by the extrapolation of an OLS fit to
  non-year n in 1000–9999.

## 5. Verdict
1. **A single power-law exponent is not a valid description of the number distribution for any of
   the 25 recipes.** All 25 fail C1 (fit at 15–40× the noise floor), C3 (OLS and MLE differ by
   0.31–0.71 and rank the recipes uncorrelatedly) and the year part of C5. This is a property of
   number text, not a pipeline fault. The counts are exact and reproducible to about 0.001.
2. **On the recipe-specific criteria C2 and C4, 21 of 25 fail.** Only dolma1_7, dolma1_7-no-reddit,
   dolma1_7-no-flan and dolma1_6plus come close (they still fail C1 and C3). The most severe cases
   are fineweb-pro, fineweb-edu, the three dclm-baseline-qc-fw* / qc-7p-fw3 recipes, dclm-qc-7p-fw2,
   dclm-qc-10p and falcon-and-cc-qc-tulu-10p.
3. **My honest recommendation is not to exclude recipes on α-validity grounds.** The failure is
   universal, so excluding the 21 would leave 4 near-identical Dolma variants and only move the
   problem. The right remedy is to re-label what is measured:
   - α_OLS is a precise **"1000–9999 tail slope (with a year bump)"** (ρ = 0.99 with the
     1000–9999 slope).
   - α_MLE is a **"small-number (n < 100) decay rate"**.
   - Any result stated in terms of "the" α should say which one, and should be checked for
     robustness with the decade slopes (count_features F08–F10), because rankings change with the
     fit range (ρ = −0.20 to 0.99 depending on the variant).
4. **Recipes to treat with care in any analysis that relies on α as a shape summary:** the 8
   strong-curvature recipes listed in 3b. Also dclm-baseline-qc-10p and -qc-20p, which are
   outliers in the 10–99 decade that α_OLS ignores.
