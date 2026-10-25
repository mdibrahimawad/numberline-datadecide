# 19: Variance structure of beta alone (auditor 19)

Scope: β only. No α value was used, looked up or related to β anywhere. Families come from
`results/beta_fine/fine_vs_data.csv` column `family` (dolma 6, dclm 7, falcon 6, mix 3, fineweb 2, c4 1).
Code: `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/var19/analysis.py`
(full output in `.../var19/out.txt`, per-model frame in `.../var19/D.pkl`). Every number below comes from that script
or from the short follow-up snippets logged in this report. No repo files were edited.

Inputs (per-seed values, not the stored summaries):
- E01 `beta_direct`: `results/datadecide/<recipe>.json` → `evaluation.runs[*].beta` (3 eval seeds 45–47, frozen layer
  `evaluation.target_layer`). The repo's `beta_direct_std` uses ddof=0. Check on c4: runs 0.552/0.625/0.683, np.std=0.0536
  (equals the CSV value), ddof=1 gives 0.0656. **Below I use ddof=1**.
- E06: `results/beta_fine/step69369-seed-default/<recipe>.json` → `per_seed[k].layers[L]` (`beta_coarse`, `beta_cont`) at
  (a) the E01 layer (`report.old_layer.layer`, which equals the E01 target layer for all 25; asserted) and (b) the E06-selected layer.

## Thresholds, fixed before any outlier statistic was computed
(I had already seen the E01/E06 β tables in the experiment READMEs. The thresholds are textbook defaults, not tuned.)
- T1 seed-variance anomaly: per-model 2·s²/σ²_pooled ~ χ²(2), two-sided p < 0.05/25 = 0.002 (Bonferroni over 25).
- T2 cross-protocol disagreement: d = E01 β_direct − E06 β_coarse at the same layer. Flag if the Iglewicz–Hoaglin modified z
  0.6745·(d−med)/MAD satisfies |M| > 3.5, or if the noise z (d−mean)/√(SE01²+SE06²) satisfies |z| > 3.09.
- T3 within-family outlier, robust: modified z inside the family, |M| > 3.5. This is only defined for families with n ≥ 5
  (dolma, dclm, falcon).
- T4 within-family outlier against measurement noise only: (y − median of the other family members)/√(SE² + var(median)),
  |z| > 3.09.
- T5 within-family outlier, predictive: leave-one-out z that also includes the genuine within-family variance σ²_m from
  REML, |z| > 3.09.
- T6 layer near-tie: in E01 selection (seeds 42–44), the runner-up √(EV·|ρ|) is within 5 % of the top layer. It counts as
  "unstable" only if, in addition, |β(runner-up) − β(top)| > 3.09·√2·SE01.

## 1. Measurement noise is small at a fixed layer, large if the layer is re-chosen

| Component (E01 β_direct scale) | variance | s.d. | source |
|---|---|---|---|
| Prompt-seed noise of a 3-seed mean (pooled s²/3) | 0.00147 | 0.038 | per-seed E01 runs; pooled seed s.d. 0.066 |
| Protocol × model interaction (E01 vs E06 coarse at the same layer, beyond seed noise) | 0.00172 | 0.042 | sd(d)=0.058 vs 0.040 expected from seeds; mean offset +0.054 |
| Layer re-selection (E06 coarse, selected layer vs E01 layer; mean(Δ²)/2) | 0.00319 | 0.056 | 8/25 switch; Δcoarse up to +0.28 (fw-3p) |
| Layer re-selection, on the β_cont scale | 0.0139 | 0.118 | Δcont up to +0.57 (tulu-10p) |
| **Total observed between-model variance of E01 means** | 0.0348 | 0.187 | |

- Seed variance is homogeneous across models. E01 Bartlett χ²=17.0, p=0.85, and no model passes T1. The smallest p is
  dclm-baseline-qc-fw-3p, p=0.003: its E01 runs are 0.782/0.783/0.787, s.d. 0.002 against a pooled 0.066. That is
  suspiciously tight but not significant after Bonferroni, and with n=3 it is consistent with chance.
- In E06, seed noise of the mean is 0.010–0.012 (pooled seed s.d. 0.017–0.021). At the E06-selected layer, T1 flags fw-3p
  (s.d. 0.053 at layer 2, p=0.001).
- Reliability of E01 β across models:
  - seed noise only: 0.958 (separation G=4.8, ≈6.7 distinguishable strata);
  - plus protocol interaction: 0.908 (4.5 strata);
  - plus layer re-selection (coarse scale): 0.817 (3.2 strata);
  - using the β_cont layer term: 0.51 (1.7 strata).

  For E06 coarse at the E01 layer the numbers are 0.994 (seed only) and 0.856 with the layer term (3.6 strata).

Conclusion: once the layer is fixed, prompt noise explains only 4 % of E01's between-model variance and under 1 % of E06's.
The dominant measurement-validity risk is layer choice, which concentrates in the 8 near-tie switchers. The E01 seed s.d.
does not capture it.

## 2. Nested decomposition (REML, y = μ + family + model(family) + seed; 300-draw parametric bootstrap 95 % CI)

| Metric | σ²_family | σ²_model(fam) | σ²_seed (per run) | share family / within / seed-noise of a model mean | ICC_true | Kish n_eff |
|---|---|---|---|---|---|---|
| E01 β_direct | 0.0115 (0–0.035) | 0.0235 (0.011–0.041) | 0.0044 | 32 % / 64 % / 4 % | 0.33 | 11.2 |
| E06 β_coarse @E01 layer | 0.0083 (0–0.024) | 0.0151 (0.006–0.024) | 0.0004 | 35 % / 64 % / 0.6 % | 0.36 | 10.8 |
| E06 β_cont @E01 layer | 0.0135 (0–0.063) | 0.0241 (0.011–0.045) | 0.0003 | 36 % / 64 % / 0.3 % | 0.36 | 10.7 |
| E06 β_coarse @sel layer | 0.0026 (0–0.016) | 0.0203 | 0.0005 | 11 % / 88 % / 0.7 % | 0.11 | 17.1 |
| E06 β_cont @sel layer | 0.0000 (0–0.021) | 0.0481 | 0.0004 | 0 % / 99.8 % / 0.2 % | 0.00 | 25.0 |

- The family component is real in fixed-effect terms but poorly determined with only 6 families:
  - η²(between families) is 0.43 (E01) and 0.47 (E06 at the E01 layer);
  - one-way ANOVA gives F=2.90, p=0.041 (E01) and F=3.34, p=0.025 (E06), but Kruskal–Wallis p is 0.077 / 0.099;
  - every bootstrap CI of σ²_family includes 0.
- Family means (E01): dolma 1.01, mix 0.97, fineweb 0.78, falcon 0.77, dclm 0.76, c4 0.62.
- The within-family component is not noise. Within-family variance divided by noise SE² is:
  - E01: dolma 46, falcon 6.9, dclm 2.3 (p=0.035), mix 12, fineweb 18;
  - E06 coarse at the E01 layer: 244 / 32 / 26 / 115 / 233.
- Within-family spread differs strongly by family type (E01 true s.d.: dolma 0.26, mix 0.13, falcon 0.09, dclm 0.04; Levene
  on dolma/falcon/dclm p=0.022 for E01 and p=0.015 for E06 cont). Source ablations (dolma) and blends (mix) move β far more
  than quality filtering of one pool (dclm, falcon). This is a data effect, and a single pooled "within-family" variance
  understates dolma and overstates dclm.
- Using the E06-selected layer shifts variance from "family" to "within family" (ICC 0.36 → 0.00). That shift is created
  by the near-tie layer switches (§1), not by the data. **Compare at a fixed layer.**
- The within-family component also contains training-seed randomness, which no file in this repo measures: only
  `seed-default` checkpoints were used (docs/infrastructure.md:36–38; docs/handover.md:58–59 lists it as open). So
  σ²_model(fam) is "genuine data effect + training noise" and cannot be split further. After removing the protocol and
  layer terms, the E01 within-family variance is about 0.0186 (s.d. 0.14). Training-seed noise would have to be about that
  large to explain within-family differences away. This is untested.

## 3. Per-model table (all 25)
L = E01 layer / E06-selected layer. β ± s.d. is across 3 prompt seeds (ddof=1). Columns:
- XP = cross-protocol d (modified z);
- Mz = within-family modified z (E01 / E06coarse@L01 / E06cont@L01 / E06cont@sel);
- zN = T4 noise-only z (E01);
- zP = T5 predictive z (E01);
- tie = E01 selection margin to the runner-up layer;
- pV = T1 seed-variance p (E01).

| recipe | fam | L | E01 β_direct | E06 coarse@L01 | E06 cont@L01 | E06 cont@sel | XP d (M) | Mz E01 | Mz co | Mz ct | Mz ct@sel | zN E01 | zP E01 | tie | pV | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dolma1_7 | dolma | 6/4 | 0.978 ± 0.094 | 0.938 ± 0.016 | 0.916 ± 0.018 | 0.831 | 0.040 (-0.21) | -0.19 | -0.16 | -0.15 | -0.30 | -2.2 | -0.25 | 0.7% | 0.271 | LAYER-SWITCH |
| dolma1_7-no-code | dolma | 8/8 | 0.856 ± 0.073 | 0.826 ± 0.018 | 0.773 ± 0.017 | 0.773 | 0.030 (-0.46) | -0.68 | -0.67 | -0.63 | -0.49 | -5.0 | -1.10 | 3.7% | 0.595 | - |
| dolma1_7-no-math-code | dolma | 4/4 | 0.618 ± 0.046 | 0.632 ± 0.017 | 0.552 ± 0.013 | 0.552 | -0.013 (-1.49) | -1.62 | -1.56 | -1.37 | -1.23 | -10.4 | -2.75 | 2.5% | 0.758 | - |
| dolma1_7-no-reddit | dolma | 8/8 | 1.360 ± 0.073 | 1.130 ± 0.015 | 1.176 ± 0.018 | 1.176 | 0.229 (4.29) | 1.31 | 0.73 | 0.72 | 0.86 | 8.7 | 2.39 | 5.0% | 0.595 | X-PROT |
| dolma1_7-no-flan | dolma | 9/9 | 1.197 ± 0.069 | 1.120 ± 0.026 | 1.191 ± 0.022 | 1.191 | 0.077 (0.67) | 0.67 | 0.68 | 0.77 | 0.91 | 5.0 | 1.27 | 3.0% | 0.683 | - |
| dolma1_6plus | dolma | 9/9 | 1.077 ± 0.077 | 1.006 ± 0.011 | 1.008 ± 0.019 | 1.008 | 0.071 (0.52) | 0.19 | 0.16 | 0.15 | 0.30 | 2.2 | 0.43 | 7.7% | 0.521 | - |
| c4 | c4 | 7/3 | 0.620 ± 0.066 | 0.668 ± 0.045 | 0.566 ± 0.028 | 0.596 | -0.048 (-2.30) |  |  |  |  |  |  | 1.2% | 0.753 | LAYER-SWITCH |
| fineweb-pro | fineweb | 7/7 | 0.662 ± 0.015 | 0.634 ± 0.043 | 0.577 ± 0.020 | 0.577 | 0.028 (-0.51) |  |  |  |  | -4.3 | -1.04 | 6.6% | 0.100 | - |
| fineweb-edu | fineweb | 9/9 | 0.895 ± 0.049 | 0.897 ± 0.021 | 0.872 ± 0.017 | 0.872 | -0.002 (-1.23) |  |  |  |  | 4.3 | 1.04 | 0.0% | 0.845 | - |
| falcon | falcon | 9/8 | 0.792 ± 0.061 | 0.747 ± 0.017 | 0.678 ± 0.005 | 0.672 | 0.045 (-0.10) | 0.64 | 0.70 | 0.25 | -0.30 | 1.2 | 0.16 | 0.3% | 0.856 | LAYER-SWITCH |
| falcon-and-cc | falcon | 9/9 | 0.649 ± 0.049 | 0.694 ± 0.044 | 0.626 ± 0.027 | 0.626 | -0.045 (-2.24) | -1.81 | -0.65 | -1.10 | -1.05 | -2.8 | -0.83 | 2.1% | 0.850 | - |
| falcon-and-cc-qc-10p | falcon | 8/3 | 0.713 ± 0.082 | 0.676 ± 0.022 | 0.601 ± 0.014 | 0.876 | 0.038 (-0.27) | -0.71 | -1.11 | -1.74 | 3.08 | -1.3 | -0.38 | 1.1% | 0.427 | LAYER-SWITCH |
| falcon-and-cc-qc-20p | falcon | 8/8 | 0.739 ± 0.070 | 0.718 ± 0.021 | 0.678 ± 0.020 | 0.678 | 0.021 (-0.67) | -0.28 | -0.05 | 0.25 | -0.19 | -0.7 | -0.21 | 2.2% | 0.668 | - |
| falcon-and-cc-qc-orig-10p | falcon | 8/3 | 0.771 ± 0.068 | 0.722 ± 0.016 | 0.659 ± 0.012 | 0.701 | 0.049 (0.00) | 0.28 | 0.05 | -0.25 | 0.19 | 0.7 | 0.02 | 3.5% | 0.701 | LAYER-SWITCH |
| falcon-and-cc-qc-tulu-10p | falcon | 9/2 | 0.948 ± 0.050 | 0.868 ± 0.012 | 0.810 ± 0.015 | 1.380 | 0.079 (0.72) | 3.29 | 3.73 | 3.65 | 11.42 | 4.8 | 1.24 | 2.4% | 0.853 | FAM-OUT, LAYER-SWITCH |
| dclm-baseline | dclm | 8/8 | 0.666 ± 0.091 | 0.606 ± 0.019 | 0.584 ± 0.016 | 0.584 | 0.059 (0.24) | -1.49 | -0.67 | -0.82 | -1.50 | -2.5 | -0.65 | 5.7% | 0.301 | - |
| dclm-baseline-qc-7p-fw2 | dclm | 8/8 | 0.838 ± 0.053 | 0.753 ± 0.012 | 0.669 ± 0.009 | 0.669 | 0.085 (0.86) | 1.23 | 0.93 | 0.13 | -0.54 | 1.9 | 0.53 | 3.9% | 0.931 | - |
| dclm-baseline-qc-7p-fw3 | dclm | 9/3 | 0.750 ± 0.097 | 0.662 ± 0.009 | 0.605 ± 0.003 | 0.806 | 0.088 (0.92) | -0.16 | -0.06 | -0.59 | 0.99 | -0.5 | -0.07 | 7.9% | 0.241 | LAYER-SWITCH |
| dclm-baseline-qc-fw-3p | dclm | 9/2 | 0.784 ± 0.002 | 0.649 ± 0.008 | 0.570 ± 0.015 | 1.065 | 0.135 (2.05) | 0.38 | -0.21 | -0.98 | 3.89 | 0.7 | 0.16 | 5.8% | 0.003 | FAM-OUT@sel only, SEEDVAR@sel, LAYER-SWITCH |
| dclm-baseline-qc-fw-10p | dclm | 3/3 | 0.809 ± 0.042 | 0.766 ± 0.014 | 0.756 ± 0.008 | 0.756 | 0.043 (-0.14) | 0.77 | 1.07 | 1.10 | 0.43 | 1.3 | 0.33 | 1.2% | 0.652 | - |
| dclm-baseline-qc-10p | dclm | 8/8 | 0.717 ± 0.068 | 0.668 ± 0.003 | 0.657 ± 0.017 | 0.657 | 0.049 (0.00) | -0.67 | 0.00 | 0.00 | -0.67 | -1.3 | -0.30 | 7.5% | 0.694 | - |
| dclm-baseline-qc-20p | dclm | 9/9 | 0.760 ± 0.068 | 0.749 ± 0.010 | 0.717 ± 0.011 | 0.717 | 0.011 (-0.92) | 0.00 | 0.89 | 0.67 | 0.00 | -0.2 | -0.00 | 5.6% | 0.702 | - |
| dclm-baseline-25p-dolma1.7-75p | mix | 8/8 | 1.108 ± 0.070 | 1.027 ± 0.007 | 1.101 ± 0.027 | 1.101 | 0.081 (0.76) |  |  |  |  | 4.5 | 1.10 | 2.9% | 0.665 | - |
| dclm-baseline-50p-dolma1.7-50p | mix | 8/8 | 0.952 ± 0.056 | 0.823 ± 0.015 | 0.831 ± 0.018 | 0.831 | 0.130 (1.91) |  |  |  |  | -0.5 | -0.12 | 0.8% | 0.978 | - |
| dclm-baseline-75p-dolma1.7-25p | mix | 10/10 | 0.841 ± 0.081 | 0.785 ± 0.012 | 0.762 ± 0.018 | 0.762 | 0.056 (0.15) |  |  |  |  | -4.0 | -0.98 | 4.4% | 0.445 | - |
## 4. Flags, evidence and classification

**F1. falcon-and-cc-qc-tulu-10p: within-family outlier (T3). Genuine data effect, do not exclude.**
- It is the highest β in the falcon family in all five metrics. Mz: E01 3.29 (just under 3.5), E06 coarse@L01 3.73,
  cont@L01 3.65. It is not an outlier against the pooled within-family variance (T5 z=1.24), and it is not a
  cross-protocol outlier (XP M=0.72).
- So it reproduces across two independent prompt sets and three fit types at the same layer. It is not a measurement
  artifact at a fixed layer.
- Documented data reason: the recipe is the falcon+CC pool filtered to the top 10 % by a Tulu-trained quality classifier
  (`falcon_and_cc_tulu_qc_top10`, docs/datadecide_sampling.md:108; mixture
  `results/corpus_alpha_datadecide/falcon-and-cc-qc-tulu-10p/mixture.json`: olmo-mix tulu_qc_top10 0.649,
  falcon tulu_qc_top10 0.351). The other three QC siblings use an ELI5/OH classifier.
- Its stream also contains far more integers (`integer_matches` 1.75e9, against 1.00e9, 1.10e9 and 1.17e9 for qc-10p,
  qc-20p and qc-orig-10p in fine_vs_data.csv). I report that only as a documented data difference; I did not test it
  against β.
- Separate measurement caveat: its E06-selected layer 2 gives β_cont 1.38 (Mz 11.4). That value is a layer artifact
  (near tie, E06 README). Use the fixed-layer value.

**F2. dolma1_7-no-reddit: cross-protocol outlier (T2). Measurement problem in E01's magnitude only, do not exclude.**
- d = 1.360 − 1.130 = 0.229, against a median of 0.049 (M=4.29, noise z>3.09). It is the only T2 flag.
- E01's own β_log is 1.72 (alpha_beta.csv `beta_log_mean`), which shows its 3 gaps are very unequal. The E01 3-gap fit
  overstates this model by about 0.18 beyond the usual protocol offset.
- Its rank as the highest-β dolma model replicates (E06 coarse 1.130, cont 1.176). Within the dolma family it is not an
  outlier (Mz 0.73/0.72, T5 2.39).
- Data reason: it is a documented ablation that removes Reddit (4.7 % share; dolma1_7 mixture.json). docs/findings.md:107
  calls the dolma1_7 vs no-reddit gap unexplained by count features. The model property is genuine; the E01 magnitude is
  not reliable. Recommendation: report the E06 value, or a sensitivity analysis.

**F3. dclm-baseline-qc-fw-3p: outlier only at the E06-selected layer (Mz 3.89, T1 p=0.001). Measurement problem.**
- Layer 2 wins over layer 9 by 0.7 % (E06 DETAILS.md:50). β_cont jumps from 0.570 to 1.065, and the seed s.d. at layer 2
  is 0.053, against 0.015 at layer 9.
- At the E01 layer it is unremarkable (Mz −0.21 / −0.98).
- Its E01 seed s.d. of 0.002 is implausibly tight (p=0.003) but passes T1.
- Recommendation: use a fixed layer. This is not a reason to exclude the model.

**F4. The 8 layer switchers.** dolma1_7, c4, falcon, falcon-and-cc-qc-10p, -qc-orig-10p, -qc-tulu-10p, dclm-qc-7p-fw3 and
dclm-qc-fw-3p. Δβ_cont from E06 selected minus E01 layer: −0.085, 0.030, −0.007, 0.275, 0.042, 0.570, 0.201, 0.495.
- This is a measurement-validity problem of the layer rule, not of the models.
- 18/25 models have an E01 runner-up within 5 %. Under T6, no model's E01 β changes by more than 3.09·√2·SE between its
  top two E01 layers: the largest are falcon-qc-10p +0.161 and fw-3p +0.143, against a threshold of 0.167.
- So E01's frozen-layer values are stable to their own runner-up. E06's switches go to distant early layers (2–3). The
  remedy is a fixed or pre-registered layer, not exclusion.

**F5. Noise-only within-family test (T4) flags most models.** 9 models in E01; 18–20 in E06, where noise is 3× smaller.
- Examples: no-math-code z=−10.4, no-code −5.0, no-flan +5.0, fineweb-edu vs -pro ±4.3, mix 75p +4.5, tulu +4.8.
- This means within-family differences are far larger than prompt noise. They are **genuine data (or training-seed)
  effects, not outliers to remove.**
- T5, which allows for the family's genuine spread, flags none. The documented reasons are whole-source ablations:
  - no-code removes starcoder (15 % share) and stackexchange;
  - no-math-code also removes proof-pile-2 and most of olmo-mix v1_6 (mixture.json files).

  The no-math-code drop replicates in every metric (E01 0.618, E06 0.632/0.552).

**F6. The mix family is not a linear blend of its endpoints. Genuine effect.**
- Taking dclm-baseline (0 %) and dolma1_7 (100 %) as endpoints, all three mixes lie above linear interpolation: E01 z = +2.0,
  +2.8, +4.3; E06 coarse@L01 z = +6.2, +3.4, +11.1.
- The 75 %-Dolma mix exceeds pure dolma1_7 in every metric, so the series is not monotone.
- Part of this may come from dolma1_7's own layer ambiguity. Its E01 margin is 0.7 % to layer 10; it is 0.938 at L6 but
  0.847 at L4 (E06 coarse). That is a measurement caveat on the anchor, not on the mixes.

## 5. How many independent units of β information?
- **Family-clustering view (Kish design effect, at a fixed layer):** n_eff ≈ 11 (10.7–11.2 across E01/E06 metrics) out of 25.
  Within the σ²_family CI, the range is about 8 (upper CI, ICC≈0.6) to 25 (σ²_family = 0). The floor if ICC → 1 is 6 (number
  of families).
- **Resolution view:** at a fixed layer, E01 separates about 4.5–6.7 statistically distinct β levels and E06 about 17. If
  layer re-selection noise is counted, this falls to about 3–3.6 levels.
- **Structural redundancy (documentation):**
  - dclm QC variants are subsets of one DCLM pool;
  - the falcon-and-cc QC variants are subsets of the falcon+CC pool;
  - dolma ablations are dolma1_7 minus a source;
  - mixes are blends of dolma1_7 and dclm-baseline files.

  The mixes still carry non-redundant information (F6). dclm (true within-family s.d. 0.04–0.07) is the most redundant
  family: its 7 models carry close to 2–3 effective units.
- Bottom line: about 10–11 effective units at a fixed layer, with at most 6 truly independent data "worlds". Any
  across-model test with N=25 should account for this clustering, for example with family random effects or cluster-robust
  errors.

## 6. Exclusion verdict
- **No model should be excluded on variance-structure grounds.** No model has anomalous seed variance in E01 (T1). The one
  T3 within-family outlier (tulu-10p) replicates across protocols and has a documented data difference, so it is a genuine
  data effect to report or control.
- Measurement problems to fix rather than exclude:
  - E01 overstates dolma1_7-no-reddit's β (F2): use E06 or a sensitivity analysis;
  - near-tie layer switching inflates E06-selected-layer values (F3, F4): use a fixed layer;
  - E01 seed s.d. underestimates true measurement error by a factor of about 1.5 (protocol term) to 2 (with the layer term).
- Unmeasured: training-seed variance. It is confounded with the within-family "data effect" (64 % of variance).
