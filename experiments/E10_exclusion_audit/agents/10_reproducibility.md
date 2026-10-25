# 10 Reproducibility of beta across independent prompt samples (E01 vs E01 vs E06)

Auditor aspect: cross-run reproducibility of beta for the 25 DataDecide 1B models. No alpha-beta relationship was computed, looked up or used. alpha columns of `alpha_beta.csv` were not read. Only `recipe, selected_layer, beta_direct_mean, beta_direct_std` were loaded.

Scripts and intermediate outputs are in `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/a10/`:
`extract.py` (reads the JSONs and checks them against the CSVs), `analyze.py`, `seedeff.py`, `analyze2.py` (main z-scores), `repca.py` (re-analysis of the E06 npz), `filt.py` (filter rejection), `per_model3.csv` (all numbers), `table.md`.

## 0. Data and consistency checks (all passed)

| Quantity | Source | Check |
|---|---|---|
| E01 selection-seed beta at the selected layer L | `results/datadecide/<recipe>.json` -> `selection.per_seed[s].layers[L].beta_direct`, seeds 42,43,44 | L == `evaluation.target_layer` == `alpha_beta.csv:selected_layer` for all 25 |
| E01 eval-seed beta | `evaluation.runs[*].beta`, seeds 45,46,47 | mean equals `alpha_beta.csv:beta_direct_mean` (max abs diff 1e-16) |
| E06 beta_coarse / beta_cont at E01's layer | `results/beta_fine/step69369-seed-default/<recipe>.json` -> `per_seed[s].layers[L]`, seeds 45,46,47 | mean equals `fine_vs_data.csv:beta_coarse_oldlayer` / `beta_cont_oldlayer` (diff 0). `report.old_layer.layer == L` for all 25. beta_coarse recomputed from the npz PC1 scores reproduces the JSON exactly (max diff 0.0, `repca.py`) |

Side finding (documentation level): **E01's reported `beta_direct_std` uses ddof=0** (it matches `np.std(ddof=0)` to 1e-16, not ddof=1, where the max diff is 0.018). **E06's std uses ddof=1** (exact match). So the two experiments' "s.d." columns are not on the same footing: E01's per-model s.d. are 18% too small for a 3-seed sample s.d. (factor sqrt(2/3)). All s.d. below are recomputed with ddof=1.

## 1. Key methodological finding: the seed s.d. is mostly a SHARED prompt-set effect, not model-specific noise

The prompt RNG is `random.Random(seed)`, the same for every model. So seed s gives (almost) the **same prompts to all 25 models**. Resampling happens only for rejected slots.
- E06 seed 45: the fraction of targets identical to dolma1_7's is 1.000 for 9 models and >=0.94 for all but fineweb-pro (0.827) and dclm-qc-fw-3p (0.896) (`seedeff.py`).
- E01 eval: for each seed, at most 17 distinct `target_hash` values across 25 models, and the most common hash is shared by 9-10 models.

So "seed noise" contains a component common to all models. A two-way decomposition beta[model, seed] = mu_model + gamma_seed + e (`seedeff.py`) gives:

| Block | Seed means across models | within-model s.d. (what the docs call seed s.d.) | model x seed interaction s.d. (true model-specific noise) | F(seed) |
|---|---|---|---|---|
| E01 seeds 42-47 | 0.844, 0.791, 0.792, 0.780, 0.875, 0.878 | 0.058 | **0.037** | 35, p~0 |
| E01 sel 42-44 | 0.844, 0.791, 0.792 | 0.038 | 0.024 | 40, p=6e-11 |
| E01 eval 45-47 | 0.780, 0.875, 0.878 | 0.066 | 0.037 | 59, p=1e-13 |
| E06 coarse 45-47 | 0.803, 0.780, 0.789 | 0.021 | **0.018** | 11, p=1e-4 |
| E06 cont 45-47 | 0.767, 0.742, 0.764 | 0.017 | 0.011 | 41, p=5e-11 |

Consequences:
- The absolute level of E01 beta shifts by up to ~0.1 depending only on which prompt set was drawn (seed 45 vs 46/47). Between-model comparisons cancel this shift, so the relevant noise for per-model reproducibility is the interaction s.d. (0.037 per seed for E01, 0.018 for E06 coarse). The per-model seed s.d. reported in `alpha_beta.csv` / E01 DETAILS.md (~0.05-0.09) mixes the two and overstates model-specific noise.
- The mean selection-minus-eval difference over models, -0.035 (t=-3.77, p=0.001), equals exactly the difference of seed-set means (0.809 vs 0.844). It is a prompt-set effect, not an effect of layer selection, and it is removed before scoring.
- Within the same E01 protocol, eval seeds are noisier than selection seeds (interaction s.d. 0.037 vs 0.024, F=2.40 on 48/48 df, p=0.0015). An independent check: subsampling E06 to E01's k=40 per coarse group gives an interaction s.d. of 0.026 (`repca.py`), which matches the selection seeds. So the E01 eval run is noisier than expected from k=40 sampling alone.
- Noise is homogeneous across models: Bartlett p = 0.99 / 0.85 / 0.33 / 0.96 (sel / eval / coarse / cont), and Spearman(per-model sd, beta) is n.s. (p>=0.24). The interaction s.d. does not depend on beta (Spearman 0.03, p=0.88). A pooled noise estimate is therefore justified. With 2 df each, per-model s.d.s are far too unstable to use (e.g. qc-fw-3p eval s.d. 0.002).

## 2. Exclusion / flag criteria (fixed before looking at per-model results)

All z-scores use the **pooled model x seed interaction s.d.** (section 1) for the noise of a 3-seed mean (s/sqrt(3)). Threshold: **|z| > 3** (two-sided p = 0.0027, so ~0.07 false flags expected per criterion over 25 models).

- **A. Same protocol, independent prompt sets (E01 sel 42-44 vs E01 eval 45-47, same layer).** d = beta_sel - beta_eval, centred by its mean over models (the shared prompt-set shift). SE = s_int,E01(6 seeds) * sqrt(2/3) = 0.0306. Sensitivity: separate sel/eval s.d.s give SE 0.0251. Justification: both runs use an identical protocol, so any model-specific disagreement beyond noise means the E01 beta is not reproducible.
- **B (primary cross-experiment test). E01 eval beta_direct vs E06 beta_coarse at the same layer.** The protocols differ systematically: E01 groups are 1-39 / 80-119 / 980-1019 / 9980-10019 (`utils/prompts.py:default_interval`, half-width 20), while E06 uses +-10% bands around 10/100/1000/10000. E01 fits PCA on 4 groups x 40 prompts, E06 on 10 groups x 100. A global calibration is therefore allowed: a Theil-Sen line E06 = a + b*E01, which gives b=0.817, a=0.102, Pearson 0.962, Spearman 0.929. z = residual / sqrt(s_c6^2/3 + b^2 s_ev^2/3), noise = 0.020. Robustness checks: an offset-only calibration (median offset -0.049), and E01 6-seed means.
- **B-robust.** The same residuals, scaled by 1.4826*MAD of the 25 residuals. This asks whether a model is an outlier relative to the typical cross-protocol disagreement, rather than relative to pure sampling noise.
- **C (secondary). E01 eval vs E06 beta_cont at the same layer.** This is a different estimand: a continuous fit over all prompts. Same calibration and noise construction (b=0.984, a=-0.101, noise 0.022), plus a robust variant.

## 3. Per-model table (all 25)

beta values are 3-seed means (ddof=1 s.d. in brackets). Flags: A, B, C = |z|>3 on that criterion, R = |robust z|>3 on B or C.

| recipe | L | E01 sel 42-44 (sd) | E01 eval 45-47 (sd) | E06 coarse@L (sd) | E06 cont@L (sd) | zA sel-eval | zB E01ev vs E06coarse (noise) | zB robust | zC E01ev vs E06cont (noise) | zC robust | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dolma1_7 | 6 | 0.934 (0.009) | 0.978 (0.094) | 0.938 (0.016) | 0.916 (0.018) | -0.29 | +1.84 | +0.98 | +2.52 | +0.36 | - |
| dolma1_7-no-code | 8 | 0.833 (0.038) | 0.856 (0.073) | 0.826 (0.018) | 0.773 (0.017) | +0.40 | +1.23 | +0.67 | +1.48 | -0.08 | - |
| dolma1_7-no-math-code | 4 | 0.670 (0.032) | 0.618 (0.046) | 0.632 (0.017) | 0.552 (0.013) | +2.85 | +1.22 | +0.67 | +2.09 | +0.18 | - |
| dolma1_7-no-reddit | 8 | 1.220 (0.030) | 1.360 (0.073) | 1.130 (0.015) | 1.176 (0.018) | -3.42 | -4.11 | -2.04 | -2.77 | -1.86 | A,B |
| dolma1_7-no-flan | 9 | 1.138 (0.030) | 1.197 (0.069) | 1.120 (0.026) | 1.191 (0.022) | -0.81 | +1.99 | +1.06 | +5.29 | +1.52 | C |
| dolma1_6plus | 9 | 0.992 (0.026) | 1.077 (0.077) | 1.006 (0.011) | 1.008 (0.019) | -1.62 | +1.21 | +0.66 | +2.29 | +0.26 | - |
| c4 | 7 | 0.665 (0.047) | 0.620 (0.066) | 0.668 (0.045) | 0.566 (0.028) | +2.62 | +2.94 | +1.55 | +2.66 | +0.42 | - |
| fineweb-pro | 7 | 0.638 (0.020) | 0.662 (0.015) | 0.634 (0.043) | 0.577 (0.020) | +0.38 | -0.44 | -0.17 | +1.25 | -0.17 | - |
| fineweb-edu | 9 | 0.890 (0.025) | 0.895 (0.049) | 0.897 (0.021) | 0.872 (0.017) | +0.99 | +3.20 | +1.68 | +4.28 | +1.10 | B,C |
| falcon | 9 | 0.778 (0.055) | 0.792 (0.061) | 0.747 (0.017) | 0.678 (0.005) | +0.67 | -0.10 | +0.00 | +0.00 | -0.70 | - |
| falcon-and-cc | 9 | 0.629 (0.039) | 0.649 (0.049) | 0.694 (0.044) | 0.626 (0.027) | +0.48 | +3.07 | +1.61 | +4.10 | +1.02 | B,C |
| falcon-and-cc-qc-10p | 8 | 0.697 (0.054) | 0.713 (0.082) | 0.676 (0.022) | 0.601 (0.014) | +0.61 | -0.46 | -0.18 | +0.02 | -0.69 | - |
| falcon-and-cc-qc-20p | 8 | 0.700 (0.046) | 0.739 (0.070) | 0.718 (0.021) | 0.678 (0.020) | -0.12 | +0.61 | +0.36 | +2.43 | +0.32 | - |
| falcon-and-cc-qc-orig-10p | 8 | 0.708 (0.040) | 0.771 (0.068) | 0.722 (0.016) | 0.659 (0.012) | -0.92 | -0.50 | -0.21 | +0.08 | -0.67 | - |
| falcon-and-cc-qc-tulu-10p | 9 | 0.859 (0.054) | 0.948 (0.050) | 0.868 (0.012) | 0.810 (0.015) | -1.75 | -0.40 | -0.15 | -0.97 | -1.11 | - |
| dclm-baseline | 8 | 0.634 (0.022) | 0.666 (0.091) | 0.606 (0.019) | 0.584 (0.016) | +0.10 | -1.96 | -0.95 | +1.40 | -0.11 | - |
| dclm-baseline-qc-7p-fw2 | 8 | 0.781 (0.049) | 0.838 (0.053) | 0.753 (0.012) | 0.669 (0.009) | -0.71 | -1.69 | -0.81 | -2.49 | -1.75 | - |
| dclm-baseline-qc-7p-fw3 | 9 | 0.719 (0.023) | 0.750 (0.097) | 0.662 (0.009) | 0.605 (0.003) | +0.14 | -2.61 | -1.28 | -1.47 | -1.32 | - |
| dclm-baseline-qc-fw-3p | 9 | 0.775 (0.021) | 0.784 (0.002) | 0.649 (0.008) | 0.570 (0.015) | +0.86 | -4.67 | -2.33 | -4.63 | -2.65 | B,C |
| dclm-baseline-qc-fw-10p | 3 | 0.879 (0.037) | 0.809 (0.042) | 0.766 (0.014) | 0.756 (0.008) | +3.44 | +0.14 | +0.12 | +2.82 | +0.48 | A |
| dclm-baseline-qc-10p | 8 | 0.688 (0.028) | 0.717 (0.068) | 0.668 (0.003) | 0.657 (0.017) | +0.19 | -1.00 | -0.46 | +2.44 | +0.33 | - |
| dclm-baseline-qc-20p | 9 | 0.727 (0.053) | 0.760 (0.068) | 0.749 (0.010) | 0.717 (0.011) | +0.05 | +1.32 | +0.72 | +3.27 | +0.67 | C |
| dclm-baseline-25p-dolma1.7-75p | 8 | 1.027 (0.035) | 1.108 (0.070) | 1.027 (0.007) | 1.101 (0.027) | -1.53 | +1.00 | +0.56 | +5.13 | +1.46 | C |
| dclm-baseline-50p-dolma1.7-50p | 8 | 0.872 (0.034) | 0.952 (0.056) | 0.823 (0.015) | 0.831 (0.018) | -1.48 | -2.86 | -1.40 | -0.21 | -0.79 | - |
| dclm-baseline-75p-dolma1.7-25p | 10 | 0.772 (0.045) | 0.841 (0.081) | 0.785 (0.012) | 0.762 (0.018) | -1.12 | -0.19 | -0.05 | +1.67 | +0.00 | - |

Rank agreement (Spearman over 25 models; `analyze.py`): sel-eval 0.962, eval-coarse 0.929, sel-coarse 0.906, eval-cont 0.934, coarse-cont 0.972. Between-model s.d. of beta: 0.187 (E01 eval), 0.152 (E06 coarse). Residual cross-protocol disagreement is ~0.043 (see below), so the reliability of the between-model ordering is high (about 1 - 0.043^2/0.15^2 = 0.92).

## 4. Global result: disagreement exceeds sampling noise for all models, not just a few

- A: centred sel-eval differences have s.d. 0.046 vs expected 0.031. chi2 = 55.3 on 24 df, p = 0.0003.
- B: residual s.d. 0.043 vs noise 0.020. chi2 = 105 on 23 df. Excess protocol x model s.d. tau = 0.038 (method of moments). The 6-seed E01 mean gives the same picture: tau = 0.036, chi2 = 129.
- C: residual s.d. 0.055 vs 0.022, tau = 0.050. Much of this is estimand difference: even within E06, on the same prompts, the coarse-vs-cont residual s.d. is 0.030.
- **Robust z (relative to the typical disagreement): no model exceeds |z| = 3 on B or C.** The maxima are fw-3p -2.33 (B) and -2.65 (C), and no-reddit -2.04 (B).

So beta carries a model-specific dependence on probe details of about +-0.04 (one s.d.) that pure prompt resampling does not capture. The flagged models are the tail of this general effect, not qualitatively distinct failures.

Tested explanations (`repca.py`, `filt.py`), none of which removes the flags:
- **PCA population.** Re-fitting PC1 on only the 4 coarse groups, inside E06's stored top-5 PC subspace, leaves the B flags at no-reddit -3.78, fineweb-edu +3.79 and fw-3p -4.16 (residual s.d. 0.040).
- **E01-sized samples.** Using k=40 per group in E06 leaves fw-3p -4.88 and c4 +3.34.
- **Filter rejection.** It is not the driver: no-reddit has 0% rejection in both runs, and fineweb-pro (41% rejection in E01 group 3) reproduces fine (zB -0.44).
- **Layer gradient.** The local slope of E06 beta_coarse over L-1..L+1 (`fixed_layer_beta.csv`) is unrelated to |zB| (Spearman -0.15, p=0.48) or |zA| (0.22, p=0.29).
- **Not testable.** The remaining candidate is the different group definitions, especially E01 group 1 = 1-39, which mixes 1- and 2-digit numbers. E06 has no prompts below 9, and E01 saved no per-prompt states.

## 5. Flags, with evidence and validity verdict

1. **dolma1_7-no-reddit: A (zA = -3.42; -4.16 with separate s.d.s) and B (zB = -4.11; offset-only -7.68; vs 6-seed mean -3.82). The hint is confirmed.**
   - Per-seed E01 at L8: 42-44 = 1.254, 1.206, 1.200; 45-47 = 1.277, 1.385, 1.417. Model x seed residuals are -0.05 on every selection seed and +0.03 to +0.08 on every eval seed. E06 coarse 1.144 / 1.132 / 1.114 (residuals within +-0.014). E06 cont 1.176.
   - So the published value 1.36 is the high outlier among three independent measurements at the same layer (1.22, 1.36, 1.13). The best estimate is ~1.13-1.22.
   - The **rank is not affected**: it is the highest-beta model in all four estimates (rank 1 in sel, eval and coarse). Robust zB = -2.04.
   - Verdict: **measurement-reliability problem of the E01 eval number for this model, not grounds to exclude the model.** Its beta is clearly the largest by every measurement. Recommendation: replace 1.36 with the E06 value (or the 6-seed E01 mean 1.29) and report the sensitivity.
2. **dclm-baseline-qc-fw-3p: B (zB = -4.67; with 4-group PCA -4.16; with k=40 -4.88) and C (-4.63).**
   - E01 is very stable across all 6 seeds (0.762-0.800) at 0.78, but both E06 estimands give 0.65 / 0.57. Its rank moves from 13-14 to 22 of 25. Robust z = -2.33 / -2.65.
   - Both runs are internally reproducible, so this is **protocol sensitivity**, not prompt noise: this model's beta depends on how the magnitude groups are defined. Group 3 rejection is 19% in E01 and 12.5% in E06 at 1000.
   - Verdict: borderline. It is a real property of how this model's representation responds to the probe (not a broken measurement), so **not a valid exclusion reason by itself**. Report it and run a sensitivity analysis (with and without the model, and E06 vs E01 beta).
3. **dclm-baseline-qc-fw-10p: A (zA = +3.44; 4.19 with separate s.d.s).**
   - Selection seeds 0.898 / 0.903 / 0.837 vs eval 0.773 / 0.855 / 0.800. Its per-model interaction s.d. is 0.068, the largest of the 25.
   - The selected layer is 3, an early layer where beta changes fastest with depth (gradient -0.092 per layer, the steepest of the 25).
   - The eval value (0.809) agrees with E06 coarse (zB = +0.14), so the published E01 number is reproduced. The selection seeds are the odd ones, and they also chose the layer.
   - Verdict: **no action on beta**. It is a noisy early-layer measurement, but the value used is cross-validated. It is not a validity problem.
4. **fineweb-edu (zB = +3.20, zC = +4.28) and falcon-and-cc (zB = +3.07, zC = +4.10).**
   - Just above threshold. E06 is higher relative to the calibration line.
   - falcon-and-cc: E06 coarse seed 47 = 0.745 vs 0.67 on the other seeds. Its E06 coarse s.d. (0.044) is the second-highest, so the pooled noise likely understates its uncertainty.
   - Robust z 1.6-1.7. Both are within the general +-0.04 protocol x model scatter.
   - Verdict: **not a validity problem**. No exclusion.
5. **C-only flags: dolma1_7-no-flan (+5.29), dclm-25p-dolma1.7-75p (+5.13), dclm-baseline-qc-20p (+3.27).**
   - beta_cont is a different estimand: a continuous curve over all magnitudes, sensitive to curvature, which is largest in high-beta Dolma models. These models agree with E06 coarse (zB 2.0, 1.0, 1.3).
   - Verdict: **estimand difference, not a reproducibility failure**. No exclusion.
6. **Near misses**: dolma1_7-no-math-code zA +2.85 (+3.47 with separate s.d.s), c4 zA +2.62 / zB +2.94. Not flagged under the primary rule.

## 6. Honest bottom line

- Under pure prompt-resampling noise, 2 models fail the within-E01 test (A) and 4 fail the cross-experiment test (B).
- However, the excess disagreement is global: chi2 is 2-4.5x its df, and the protocol x model s.d. is ~0.04 against a between-model spread of ~0.15-0.19. Relative to that global effect, **no model is an outlier** (all robust |z| < 3).
- **No model should be excluded on reproducibility grounds.**
- Recommendations:
  - (a) Treat dolma1_7-no-reddit's E01 eval beta = 1.36 as an overestimate; it is ~1.13-1.22 in two independent re-measurements. Report the sensitivity of any conclusion to using 1.13 / 1.22 / 1.29.
  - (b) Report dclm-baseline-qc-fw-3p as protocol-sensitive (0.78 vs 0.65).
  - (c) Prefer E06 beta (k=100, interaction s.d. 0.018 vs 0.037) or average across experiments.
  - (d) Propagate a per-model beta uncertainty of about sqrt(0.02^2 + 0.04^2) ~ 0.045. The +-0.015 quoted in E06 and the +-0.05 quoted in E01 are both wrong for this purpose: the first ignores the protocol x model term, and the second is inflated by the shared prompt-set term.
  - (e) Fix the ddof inconsistency in the E01 s.d.
- Also note that the E01 eval and E06 runs reuse seeds 45-47. Their target sets differ because the group definitions and code paths differ, so they are effectively independent samples. Within each experiment, however, all models share the same prompts.
