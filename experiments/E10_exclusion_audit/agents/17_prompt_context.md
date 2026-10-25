# 17 — Sensitivity of beta to the prompt context (in-context examples, prompt seeds)

Auditor aspect: does the 3-example context `c1=c1,c2=c2,c3=c3,n=` leak into the last-token state, are contexts balanced across magnitude groups, and how much does beta move with prompt sampling compared with the differences between models. Blind to alpha: no alpha-beta relationship was computed or looked at.

Scripts and intermediate outputs (all outside the repo), in `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/a17/`:
`ctx.py` (context regression and adjusted beta, output `ctx_rows.csv`), `e01.py` (E01 per-seed betas, `e01_seeds.csv`), `anova.py` (model x seed decomposition, `seed_table.csv`), `sel.py` (filter selection versus context, `first_try.csv`), `table.py` (`final_table.csv`, the table below).

## 0. What the code does (facts the analysis relies on)

- **E01 context generation.** `utils/prompts.py::generate_numeral_prompts` with `context="random"` draws the 3 examples as `rng.randint(0, upper_bound)`, independently of the target group. `upper_bound` is None in every E01 config, so `src/geometry.py::_build_prompts` sets it to `10**max(groups) = 10000`. The examples are therefore uniform on 0..10000, so about 90 % of them are 4-digit. In expectation the contexts are balanced across groups by construction.
- **Shared prompts across models.** `src/geometry.py:871` builds `rng = random.Random(cfg.seed + run_idx)`, and E06 `src/beta_fine.py::filtered_prompts` builds `rng = random.Random(seed)`. All 25 models therefore get the same initial prompt set (same targets and same contexts) for a given seed. Only slots rejected by the correct-output filter are redrawn. E01 `evaluation.runs[*].target_hash` is identical for 10, 9 and 9 models at seeds 45, 46 and 47. These are exactly the models with near-zero rejections (dolma family and the mixes).
- **Filter redraws the context too.** In both E01 (`_resample_numeral_prompt`) and E06 (`make_prompt`), a rejected slot gets a new target and 3 new context numbers. The filter can therefore select on the context.
- **Per-prompt context is not saved.** Neither the E01 json (12 sample prompts only) nor the E06 npz saves it. The npz keys are `seed*_targets, seed*_groups, seed*_pc_scores (17x1000x5), seed*_pc_evr`. **But the E06 contexts can be reconstructed exactly.** `filtered_prompts` draws all 1000 initial prompts (slot order: group 0..9 x 100) from `random.Random(seed)` before any model call. The kept arrays keep slot order, and every model/seed has 1000 kept prompts. A slot whose regenerated target equals the saved target was accepted on its first try, so its regenerated context is its real context. These "matched" slots are 81.1 %–100 % of prompts (median 99.4 %). The number of unmatched slots never exceeds the logged rejections (`filter[seed][g].tried - accepted`), which is consistent. A resampled slot whose redraw happens to hit the same target would be mislabelled, but this is rare because groups 0–2, where bands are narrow, have rejection rates of only 0.7–1.4 %. All context analyses below use matched slots, at E06's selected layer ("sel") and at the E01 frozen layer ("E01L"). The PC scores are taken from the npz, already oriented to increase with n.
- **Unit note.** The E01 `beta_direct_std` in `results/datadecide/alpha_beta.csv` is the **population SD (ddof=0)** over the 3 eval seeds. `e01.py` checks that `np.std(..., ddof=0)` reproduces it and `ddof=1` does not. The sample SD is 1.22x larger. This report uses ddof=1.

## 1. Findings

### 1a. The context leaks into the last-token state, strongly in deep layers
For each model and seed, I fit PC1 ~ group fixed effects + within-group log10(target) + context features. The context features are the mean log10(c+1) of the 3 examples, log10 of the example next to the target, and the number of 4-digit examples. I then took the partial R² of the context block (`ctx.py`, column `pr2_ctx`).
- At the E06 layer the partial R² is 0.003–0.40 (table column `ctx_pR2`). The F-test gives p < 1e-4 in 16/25 models (22/25 at the E01 layer).
- At the E01 layer it is 0.004–0.40. In dolma-family and mix models at layers 8–10 it is 0.32–0.40, i.e. up to 40 % of the within-group PC1 variance is explained by the in-context magnitudes. At the shallow layers 2–4 it is about 0.01. Across models, partial R² correlates with the selected E06 layer at r = 0.66.
- The sign is negative (column `ctx_slope`: change in PC1, as a fraction of the group-0 to group-9 span, per +1 log10 of mean context magnitude). The slope ranges from -0.25 to +0.04. Larger in-context numbers push the target's PC1 down: the state at '=' is a target-versus-context contrast, not the target alone.

### 1b. Contexts are balanced across groups (no detectable imbalance)
- One-way ANOVA of mean log-context across the 10 groups (matched slots): minimum p = 0.32 over all 150 model x seed x layer cases, and 0/150 have p < 0.05 (`ctx.py` output).
- The range of the group means is 0.035–0.105 log10 units. Per-prompt SD is 0.245, so the SE of a group mean of 100 is about 0.025 and this range is what random draws give. Example for seed 45, groups 0/3/6/9: 3.555 / 3.547 / 3.545 / 3.545.
- **Filter selection on context (`sel.py`).** First-try rejected prompts have smaller contexts than accepted ones (pooled over models, by group: Δmean log-context = -0.02 to -0.22, Δ#4-digit examples = -0.18 to -0.73). This matches the logged failure mode of copying a short context number, e.g. E01 c4 `6653=6653,8977=8977,6=6,1017=` → `6`. The filter therefore tilts kept contexts upward in heavily filtered groups. The worst case is fineweb-pro group 5 (61 % first-try rejection), where accepted first-try prompts sit +0.035 log10 above the group's unfiltered mean. Multiplied by its slope (-0.08 span per log10), that is a group-mean shift of about 0.003 of the span, which is negligible.

### 1c. Because the prompt sets are shared, the leakage causes a small systematic, model-dependent beta bias
In expectation the bias would average out over seeds. But every model sees the same 3 seeds' contexts, so the small random imbalance in 1b is the **same** for all models. Each model turns it into a beta shift in proportion to its own context sensitivity. Adjusting the group means for the context covariates (matched slots, coefficients from 1a) changes beta_coarse by `dB_adj` = -0.002 to +0.019 (mean over seeds), with a per-seed maximum of 0.036. The shift is almost always positive: raw beta is biased slightly **down**, and the bias tracks partial R² (r = 0.94 across models). It is largest for dolma1_7-no-reddit (+0.019), dolma1_6plus / dolma1_7-no-flan / 25p-mix (+0.015), and the 50p and 75p mixes (+0.012–0.013). For beta_fine the change is ≤ 0.009 on average. This is about 0.1x the between-model SD (0.15), but it is not random across models: it is larger in the context-sensitive (dolma-heavy, deep-layer) models.

### 1d. Seed (prompt-sampling) noise: mostly a shared prompt-set effect, small relative to between-model spread
I decomposed per-seed betas as model x seed (`anova.py`). "Model-specific SD" is the residual after removing each seed's common effect. That residual is the noise that matters for comparing models.

| measurement | between-model SD | seed main effects | pooled raw within-model SD | model-specific (interaction) SD | interaction / between | reliability ICC(C,k) |
|---|---|---|---|---|---|---|
| E01 beta_direct, eval seeds 45–47 | 0.1865 | -0.065, +0.031, +0.034 | 0.066 | 0.037 | 0.20 | 0.987 |
| E01 beta_direct, seeds 42–47 at frozen layer | 0.171 | +0.017,-0.035,-0.034,-0.047,+0.048,+0.051 | 0.058 | 0.037 | 0.22 | 0.992 |
| E06 beta_coarse, seeds 45–47 | 0.150 | +0.007,-0.006,-0.002 | 0.022 | 0.021 | 0.14 | 0.994 |
| E06 beta_fine | 0.146 | -0.005,-0.001,+0.006 | 0.023 | 0.023 | 0.16 | 0.992 |
| E06 beta_cont | 0.220 | +0.004,-0.011,+0.007 | 0.019 | 0.017 | 0.08 | 0.998 |

- In E01 (k=40 prompts per group), more than half of the per-model seed SD is a common shift: seed 45 gives the lowest of the 3 eval betas for 23/25 models. This is what shared prompts predict. It moves the absolute beta level of every model together (SE of a 3-seed mean ≈ 0.04 for the absolute level) and cancels in between-model comparisons. The 3 models without the seed-45 dip (fineweb-pro, dclm-qc-fw-3p, dclm-qc-7p-fw3) are heavily filtered models whose prompt sets diverge from the shared set. Because of that, their raw SD (0.015, 0.002, 0.097) is a poor guide.
- The E06 design (100 prompts x 10 groups) cuts seed noise by about 3x, and the common component almost disappears.
- Consequence for interpretation: absolute claims such as "beta ≈ 1, a log line" for dolma1_7 (0.978 ± 0.094 raw seed SD) depend on the prompt set at the ±0.05 level. The ranking of models does not.

## 2. Exclusion / flag criteria (set before looking at which models they hit; thresholds chosen on measurement grounds)

- **S1 – model-specific seed SD > 0.5 x between-model SD** of the same measurement (E01 eval, E06 coarse, E06 fine). *Justification:* with 3 seeds the SE of the model mean would then be ≥ 0.29 between-model SD, so the per-model reliability is < 0.92 and the model's rank is uncertain by about a third of the population spread. Thresholds: 0.093 (E01), 0.075 (E06c), 0.073 (E06f).
  → **dclm-baseline-qc-7p-fw3** (E01: 0.110; seeds 0.708 / 0.682 / 0.860). Borderline: falcon-and-cc-qc-tulu-10p E06 fine SD 0.0727 versus 0.0731 (not flagged).
- **S2 – outlier seed sensitivity: model-specific SD > 3 x median of the 25 models** in a measurement set. For a 2-df SD this is p ≈ 0.002 under homogeneous noise, so about 5 % chance of one false hit per set. A flag counts as robust only if it appears in **both independent experiments** (E01 and E06, different prompts and layers).
  → E01 only: dclm-baseline-qc-7p-fw3 (3.7x). E06 only: falcon-and-cc (coarse 3.5x, fine 3.4x), dclm-baseline-qc-fw-3p (3.8x, 4.6x), falcon-and-cc-qc-tulu-10p (fine 6.1x). **No model is flagged in both experiments.** The E06-only cases are 2 of 3 models where E06 picked a shallow layer (2) far from E01's layer (9). That points to layer-choice instability (another auditor's aspect), not to context.
- **C1 – context imbalance across groups:** ANOVA p < 0.01 for mean log-context across groups, in any seed or layer. → **none** (min p = 0.32).
- **C2 – context-induced beta bias larger than random error:** |mean context-adjusted minus raw beta_coarse| > the median model-specific E06 seed SD (0.0158), at the E06 or E01 layer.
  → **dolma1_7-no-reddit** (+0.019; per-seed max 0.036). Just below the threshold: dolma1_6plus (0.0152), dolma1_7-no-flan (0.0146), 25p mix (0.0145).
- **C3 – strong context leakage:** partial R² > 0.2. This is reported, not used for exclusion: it raises within-group noise, which is already counted in S1/S2, and bias only arises through C2. 11 models at the E01 layer (all dolma-family and mix models except no-math-code, plus dclm-qc-10p, dclm-qc-7p-fw2 and falcon-and-cc-qc-orig-10p).
- **C4 – filter selection on context:** any E06 group with first-try rejection > 30 % (selection can tilt that group's context distribution). → **fineweb-pro** (group 5: 61 %; E01 group 3: 41 %). Estimated effect is about 0.003 of span, negligible.

## 3. Per-model table (all 25)
Columns: E01_L / E06_L = selected layer. E01_beta = beta_direct_mean (csv). E01_sd_raw = SD over eval seeds (ddof=1). E01_sd_model / E01_6seed_model = model-specific (interaction) SD, 3 and 6 seeds. E06_bcoarse = recomputed mean beta_coarse at the E06 layer. E06_c_sd / E06_c_model = raw and model-specific seed SD of beta_coarse. E06_f_sd = seed SD of beta_fine. ctx_pR2 = partial R² of context features on within-group PC1 (E06 layer). ctx_slope = PC1 change (fraction of 10→10000 span) per +1 log10 context. dB_adj = context-adjusted minus raw beta_coarse (seed mean), dB_adj_max = largest |per-seed| value. *_E01L = same at the E01 layer. filt_dmlog = mean log-context of first-try rejected minus accepted prompts (blank if < 10 rejections). max_grp_rej = maximum first-try rejection rate over E06 groups.

| recipe | E01_L | E01_beta | E01_sd_raw | E01_sd_model | E01_6seed_model | E06_L | E06_bcoarse | E06_c_sd | E06_c_model | E06_f_sd | ctx_pR2 | ctx_slope | dB_adj | dB_adj_max | ctx_pR2_E01L | dB_adj_E01L | filt_dmlog | max_grp_rej | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dolma1_7 | 6 | 0.978 | 0.094 | 0.057 | 0.039 | 4 | 0.847 | 0.013 | 0.009 | 0.011 | 0.173 | -0.079 | 0.004 | 0.007 | 0.391 | 0.011 |  | 0.003 | - |
| dolma1_7-no-code | 8 | 0.856 | 0.073 | 0.021 | 0.018 | 8 | 0.826 | 0.018 | 0.016 | 0.013 | 0.273 | -0.156 | 0.008 | 0.017 | 0.273 | 0.008 |  | 0.003 | - |
| dolma1_7-no-math-code | 4 | 0.618 | 0.046 | 0.016 | 0.057 | 4 | 0.632 | 0.017 | 0.013 | 0.001 | 0.005 | -0.015 | 0.001 | 0.002 | 0.005 | 0.001 | 0.011 | 0.013 | - |
| dolma1_7-no-reddit | 8 | 1.360 | 0.073 | 0.026 | 0.064 | 8 | 1.130 | 0.015 | 0.016 | 0.011 | 0.404 | -0.248 | 0.019 | 0.036 | 0.404 | 0.019 |  | 0.003 | C2 |
| dolma1_7-no-flan | 9 | 1.197 | 0.069 | 0.061 | 0.040 | 9 | 1.120 | 0.026 | 0.033 | 0.017 | 0.355 | -0.221 | 0.015 | 0.026 | 0.355 | 0.015 |  | 0.000 | - |
| dolma1_6plus | 9 | 1.077 | 0.077 | 0.026 | 0.034 | 9 | 1.006 | 0.011 | 0.020 | 0.024 | 0.321 | -0.228 | 0.015 | 0.028 | 0.321 | 0.015 |  | 0.000 | - |
| c4 | 7 | 0.620 | 0.066 | 0.034 | 0.055 | 3 | 0.650 | 0.015 | 0.014 | 0.012 | 0.007 | -0.005 | 0.001 | 0.001 | 0.086 | 0.006 | -0.119 | 0.117 | - |
| fineweb-pro | 7 | 0.662 | 0.015 | 0.083 | 0.048 | 7 | 0.634 | 0.043 | 0.046 | 0.018 | 0.045 | -0.080 | 0.004 | 0.006 | 0.045 | 0.004 | -0.056 | 0.610 | C4 |
| fineweb-edu | 9 | 0.895 | 0.049 | 0.042 | 0.042 | 9 | 0.897 | 0.021 | 0.019 | 0.010 | 0.077 | -0.049 | 0.001 | 0.008 | 0.077 | 0.001 | -0.193 | 0.170 | - |
| falcon | 9 | 0.792 | 0.061 | 0.045 | 0.036 | 8 | 0.749 | 0.015 | 0.011 | 0.014 | 0.058 | -0.090 | 0.005 | 0.008 | 0.097 | 0.008 | -0.035 | 0.010 | - |
| falcon-and-cc | 9 | 0.649 | 0.049 | 0.019 | 0.032 | 9 | 0.694 | 0.044 | 0.056 | 0.040 | 0.005 | -0.025 | 0.001 | 0.002 | 0.005 | 0.001 | -0.290 | 0.007 | S2(E06c,E06f) |
| falcon-and-cc-qc-10p | 8 | 0.713 | 0.082 | 0.039 | 0.031 | 3 | 0.836 | 0.018 | 0.019 | 0.007 | 0.003 | 0.010 | -0.000 | 0.001 | 0.185 | 0.010 | -0.145 | 0.053 | - |
| falcon-and-cc-qc-20p | 8 | 0.739 | 0.070 | 0.030 | 0.020 | 8 | 0.718 | 0.021 | 0.018 | 0.022 | 0.139 | -0.140 | 0.006 | 0.012 | 0.139 | 0.006 | -0.133 | 0.040 | - |
| falcon-and-cc-qc-orig-10p | 8 | 0.771 | 0.068 | 0.023 | 0.023 | 3 | 0.773 | 0.014 | 0.019 | 0.012 | 0.018 | -0.005 | -0.001 | 0.001 | 0.236 | 0.010 | -0.174 | 0.010 | - |
| falcon-and-cc-qc-tulu-10p | 9 | 0.948 | 0.050 | 0.014 | 0.038 | 2 | 1.030 | 0.028 | 0.035 | 0.073 | 0.005 | 0.013 | -0.000 | 0.004 | 0.114 | 0.010 | -0.262 | 0.027 | S2(E06f) |
| dclm-baseline | 8 | 0.666 | 0.091 | 0.048 | 0.030 | 8 | 0.606 | 0.019 | 0.017 | 0.003 | 0.116 | -0.153 | 0.009 | 0.014 | 0.116 | 0.009 | -0.090 | 0.107 | - |
| dclm-baseline-qc-7p-fw2 | 8 | 0.838 | 0.053 | 0.027 | 0.025 | 8 | 0.753 | 0.012 | 0.009 | 0.009 | 0.205 | -0.132 | 0.009 | 0.014 | 0.205 | 0.009 | -0.076 | 0.013 | - |
| dclm-baseline-qc-7p-fw3 | 9 | 0.750 | 0.097 | 0.110 | 0.063 | 3 | 0.790 | 0.016 | 0.027 | 0.014 | 0.012 | 0.016 | -0.002 | 0.003 | 0.152 | 0.009 | -0.121 | 0.263 | S1-E01;S2(E01) |
| dclm-baseline-qc-fw-3p | 9 | 0.784 | 0.002 | 0.066 | 0.041 | 2 | 0.932 | 0.042 | 0.060 | 0.055 | 0.010 | 0.043 | -0.001 | 0.002 | 0.039 | 0.002 | -0.080 | 0.210 | S2(E06c,E06f) |
| dclm-baseline-qc-fw-10p | 3 | 0.809 | 0.042 | 0.046 | 0.073 | 3 | 0.766 | 0.014 | 0.012 | 0.009 | 0.004 | -0.011 | 0.001 | 0.001 | 0.004 | 0.001 | -0.028 | 0.133 | - |
| dclm-baseline-qc-10p | 8 | 0.717 | 0.068 | 0.017 | 0.014 | 8 | 0.668 | 0.003 | 0.005 | 0.003 | 0.218 | -0.175 | 0.010 | 0.017 | 0.218 | 0.010 | 0.012 | 0.117 | - |
| dclm-baseline-qc-20p | 9 | 0.760 | 0.068 | 0.017 | 0.019 | 9 | 0.749 | 0.010 | 0.007 | 0.003 | 0.127 | -0.128 | 0.008 | 0.013 | 0.127 | 0.008 | -0.022 | 0.037 | - |
| dclm-baseline-25p-dolma1.7-75p | 8 | 1.108 | 0.070 | 0.018 | 0.031 | 8 | 1.027 | 0.007 | 0.012 | 0.015 | 0.386 | -0.230 | 0.015 | 0.028 | 0.386 | 0.015 |  | 0.003 | - |
| dclm-baseline-50p-dolma1.7-50p | 8 | 0.952 | 0.056 | 0.008 | 0.028 | 8 | 0.823 | 0.015 | 0.012 | 0.002 | 0.357 | -0.201 | 0.013 | 0.022 | 0.357 | 0.013 |  | 0.000 | - |
| dclm-baseline-75p-dolma1.7-25p | 10 | 0.841 | 0.081 | 0.032 | 0.029 | 10 | 0.785 | 0.012 | 0.007 | 0.006 | 0.374 | -0.196 | 0.012 | 0.022 | 0.374 | 0.012 |  | 0.000 | - |

## 4. Validity versus property

| flag | models | nature | exclude? |
|---|---|---|---|
| S1/S2 (E01 seed sensitivity) | dclm-baseline-qc-7p-fw3 | **Measurement precision.** One E01 seed (47: 0.860) is off by +0.11 against the shared seed pattern. It is not reproduced in E06 at the same E01 layer 9 (E06 coarse-at-E01-layer seed SD 0.009, `ctx_rows.csv`) or at E06's layer 3 (0.016). So it is E01's small-k sampling noise for this model, not a property. | **No.** Report a wider uncertainty, or use the E06 value (or the 6-seed E01 mean 0.735). It does not invalidate the model. |
| S2 (E06 only) | falcon-and-cc, dclm-baseline-qc-fw-3p, falcon-and-cc-qc-tulu-10p | Measurement precision of E06 at shallow or unstable layers. Not reproduced in E01 (model-specific SD 0.019, 0.066, 0.014). | No. Defer to the layer-selection audit. |
| C2 (context bias) | dolma1_7-no-reddit (and the near-threshold dolma1_6plus, no-flan, 25p mix) | **Measurement artifact** caused by shared, finite prompt sets combined with a genuine property, context sensitivity. The bias is small (+0.01 to +0.02, raw beta too low) and correctable by covariate adjustment or by balancing contexts across groups (e.g. same context per target slot across groups). | **No.** Correct or report it. It is systematic and family-structured, so state it as a known bias of up to 0.02 in dolma-heavy models. Excluding them would remove a family, not a broken measurement. |
| C3 (leakage) | 11 deep-layer dolma/mix and some dclm/falcon models | **Genuine model property**: deep-layer '=' states encode the context numbers as well as the target. | No. Report it as a covariate. It also explains part of the within-group spread. |
| C4 (filter selection) | fineweb-pro | Selection mechanism exists, effect about 0.003 of span. Genuine model behaviour (copying failures), negligible bias. | No. |
| C1 (imbalance) | none | — | — |

**Honest bottom line:** nothing in the prompt-context aspect justifies excluding any of the 25 models. Prompt sampling moves every model's beta together by about ±0.05 in E01, a shared prompt-set effect that cancels in between-model comparisons. The model-specific part (SD ≈ 0.037 in E01, ≈ 0.02 in E06) is 0.08–0.22 of the between-model SD, with reliability ≥ 0.987 everywhere. The in-context examples do leak strongly into deep-layer states, but contexts are balanced across groups, so the resulting group-mean bias is ≤ 0.02 in beta. The bias is consistent in sign and concentrated in context-sensitive (dolma-heavy) models, and should be reported or corrected rather than used to exclude. Two documentation fixes are recommended: `beta_direct_std` is ddof=0, and absolute beta levels carry a shared prompt-set uncertainty (about 0.04 SE for E01) that the per-model SDs do not convey.
