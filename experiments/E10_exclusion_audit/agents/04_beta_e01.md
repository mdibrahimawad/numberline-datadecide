# Auditor 04: quality of the E01 beta measurement (paper protocol)

Scope: the beta measurement of experiment E01 for all 25 DataDecide-1B recipes, at the layer each model selected.
Sources: `results/datadecide/<recipe>.json`, the E01 code (`src/datadecide_sweep.py`, `src/frozen_layer_evaluation.py`, `src/robust_geometry.py::_projection_metrics`, `src/spacing_fit.py`, `src/geometry.py`, `utils/prompts.py`). I also used E06's per-prompt files (`results/beta_fine/step69369-seed-default/<recipe>.{json,npz}`), read only at E01's own layer, as an independent test-retest check.
Blindness: I did not open `alpha_beta_correlations.json`, `confounders.json` or any alpha column, and I computed no alpha–beta statistic. Disclosure: the E06 README has a one-line alpha–beta sentence that I saw by accident while looking for its file format. I did not use it. I also read the E01 DETAILS.md beta table before setting thresholds (see "Threshold provenance").

Scripts and intermediate data: `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/work/` (`e01.py`, `e06chk.py`, `final.py`, `e01_rows.json`, `e06_rows.json`, `table.json`, `table.md`).

## 1. What E01 actually stores, and how I recovered the gaps

* E01 does **not** save group means or per-prompt PC1 scores. For each evaluation seed (45, 46, 47) it saves only `beta` (direct), `beta_scale`, `r2_beta`, `beta_log` and `r2_beta_log`.
* **The three gaps can be recovered exactly.** The direct fit minimizes Σ(d_i − s·b^i)² over i = 0, 1, 2. At the optimum, both normal equations hold: Σ d_i b^i = s Σ b^{2i} and Σ i·d_i b^i = s Σ i·b^{2i}. With 3 gaps, beta_log = exp((ln d3 − ln d1)/2), so d3 = beta_log²·d1. That is 3 linear equations in (d1, d2, d3). I solved them for all 75 runs. As a check, the recovered gaps reproduce the stored `beta`, `r2_beta` and `r2_beta_log` to < 3e-5 in every run (`chk` in `e01.py`).
* **Caveat about the code:** both fits use `np.abs(np.diff(means))` (`_projection_metrics`, line 136). **The sign of each gap is discarded**, so a reversed pair of group means would be folded into a positive gap, and E01 itself cannot show non-monotonicity. What I could do instead:
  * (a) flag gaps that are near zero, where a reversal could hide;
  * (b) use ρ. With 4 equal groups, ρ ≈ 0.94 already corresponds to clean group separation. A fully reversed adjacent pair would cap ρ near 0.81, and every model has ρ ≥ 0.899 (per seed ≥ 0.886);
  * (c) check signed decade-group means at E01's layer in E06. They are monotone in all 25 models × 3 seeds (0 non-positive gaps; per-gap t at n = 40 is ≥ 5.6 everywhere).

## 2. Exclusion criteria (thresholds and justification)

"Hard" criteria mean the beta value itself is not trustworthy as a measurement. "Soft" flags mean the value is valid under the protocol but needs a caveat or a sensitivity check.

| ID | Criterion | Threshold | Justification |
|---|---|---|---|
| H1 | Protocol integrity | any failed group/slot, or a final token other than `=` | A failed group means fewer than 40 prompts or a missing group. A wrong final token means a different readout position. |
| H2 | Near-zero gap (possible hidden reversal) | any gap < 0.10 × that seed's mean gap, in any eval seed | Such a gap is within ≈ 1 single-seed SE of zero (median single-seed SE ≈ 0.1 × gap). Because of `abs()` its sign is unknown, and beta_log (which uses ln d3) is ill-defined. |
| H3 | Seed irreproducibility | beta_direct s.d. over eval seeds > 0.10 | Twice the typical 0.05 for k = 40 (DETAILS.md). Above this, the seed noise is more than half the between-model s.d. of beta (0.19). |
| H4 | No clear number line | ρ < 0.85 or EV(PC1) < 0.15 | Below these, PC1 is not a dominant, magnitude-ordered axis, so "group means on PC1" is not a number line. |
| H5 | Filter-biased sample | correct-output rejection ≥ 33 % in any group | More than one third of the candidates discarded, so beta is measured on a selected subset. If this triggers, a pre-specified check follows: exclude only if the estimated composition bias exceeds 0.03 (≈ the SE of the 3-seed mean) or H6 also fails. |
| H6 | Test-retest failure at the same layer | \|(β_E06,coarse,oldlayer − β_E01) − median offset\| > 3 robust s.d. (MAD × 1.4826) | Same model, same layer, independent prompts (100 per group). The median offset (−0.049) absorbs the design difference between E06's ±10 % bands and E01's 1–39 / ±20 bands. 3 robust s.d. is a standard outlier cutoff. |
| S1 | Layer near-tie | top-2 q_median within 2 % **and** \|Δβ\| between those layers ≥ 0.10 (selection seeds) | The per-seed relative s.d. of q is ≈ 1.3 % (median), so a margin under 2 % is a coin flip. Δβ ≥ 0.10 is about 2 seed s.d. |
| S1b | Steep layer profile | \|β(L±1) − β(L)\| > 0.30 | Beta depends strongly on exactly which layer was chosen. |
| S2 | Weakly resolved gap | smallest single-seed t (mean gap / s.d. over 3 seeds) < 3 | One gap carries most of the noise. |
| S3 | Non-geometric gaps | \|ln(β_log/β_direct)\| > 0.30 | For geometric gaps the two fits coincide. A large divergence means the 3 gaps do not follow d·β^i, so "beta" depends on the estimator. |
| S4 | Unstable β_log | β_log s.d. > 0.15 | Only matters if β_log is used. |
| S5 | Moderate filter selection | 15–33 % rejection in a group, or ≥ 5 % in group 1 | The accepted sample is somewhat non-random (see §4). |
| S6 | Selection-vs-evaluation disagreement at the same layer | \|z\| > 3 | Internal replication on seeds 42–44 against 45–47. |

**Threshold provenance (honesty note):** I set the thresholds from SE and outlier conventions, not by tuning. I had, however, already seen the DETAILS.md summary table (β, s.d., max rejection) when I set them. H3, H4 and H1 are far from every model, so they are not sensitive to this. H5 at 33 % catches only fineweb-pro (41 %); the next highest model is at 19 %, so any cutoff from 20 % to 40 % gives the same result. H6 at 3 robust s.d. catches only no-reddit (−4.3); the next largest is c4 at +2.3.

## 3. Per-model table (all 25, frozen layer, eval seeds 45–47)

Columns:
* **gaps:** mean recovered gaps d1/d2/d3 in PC1 units;
* **min gap rel:** smallest gap / seed mean gap, over all seeds;
* **min t:** smallest single-seed t and which gap;
* **rej %:** rejection rate per group 1/2/3/4 over all 6 seeds (`filter_summary`);
* **g1 1-digit share:** estimated share of 1–9 among accepted group-1 targets (23 % if unfiltered);
* **1-tok:** 1-token targets out of 160 (seed 42; 106 for zero-rejection models);
* **margin:** q margin to the runner-up layer, and Δβ between the two layers;
* **nbr Δβ:** largest β change to an adjacent layer;
* **sel β (z):** β on selection seeds at the same layer, and its z against eval;
* **E06 β (z):** E06 β_coarse at E01's layer, and the robust z of its offset.

| recipe | L | EV | ρ | β_direct ± s.d. | β_log ± s.d. | ln(βlog/βd) | gaps d1/d2/d3 | min gap rel | min t | rej % g1/2/3/4 | g1 1-digit share | 1-tok | margin (runner-up, Δβ) | nbr Δβ | sel β (z) | E06 β (z) | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dolma1_7 | 6 | 0.37 | 0.933 | 0.978 ± 0.077 | 0.969 ± 0.120 | -0.01 | 19.1 / 37.3 / 17.7 | 0.62 | 5.6 (d1) | 0/0/0/0 | 23% | 106 | 0.7% (L10, Δβ 0.10) | 0.14 | 0.934 (-0.8) | 0.938 (+0.2) | - |
| dolma1_7-no-code | 8 | 0.41 | 0.935 | 0.856 ± 0.060 | 0.774 ± 0.080 | -0.10 | 31.9 / 49.7 / 18.9 | 0.49 | 6.8 (d1) | 0/0/0/0 | 23% | 106 | 3.7% (L9, Δβ 0.05) | 0.05 | 0.833 (-0.5) | 0.826 (+0.5) | - |
| dolma1_7-no-math-code | 4 | 0.34 | 0.945 | 0.618 ± 0.037 | 0.372 ± 0.141 | -0.51 | 13.6 / 13.5 / 2.2 | 0.04 | 1.3 (d3) | 1/0/0/0 | 23% | 106 | 2.5% (L10, Δβ 0.06) | 0.05 | 0.670 (+1.6) | 0.632 (+1.5) | H2, S2, S3 |
| dolma1_7-no-reddit | 8 | 0.52 | 0.912 | 1.360 ± 0.060 | 1.724 ± 0.278 | +0.24 | 18.0 / 59.9 / 49.8 | 0.27 | 2.8 (d1) | 0/0/0/0 | 23% | 106 | 5.0% (L9, Δβ 0.05) | 0.13 | 1.220 (-3.1) | 1.130 (-4.3) | H6, S2, S4, S6 |
| dolma1_7-no-flan | 9 | 0.47 | 0.931 | 1.197 ± 0.056 | 1.465 ± 0.166 | +0.20 | 18.3 / 68.1 / 38.0 | 0.36 | 3.7 (d1) | 0/0/0/0 | 23% | 106 | 3.0% (L7, Δβ 0.09) | 0.11 | 1.138 (-1.4) | 1.120 (-0.7) | S4 |
| dolma1_6plus | 9 | 0.43 | 0.923 | 1.077 ± 0.063 | 1.175 ± 0.137 | +0.09 | 24.4 / 64.4 / 32.3 | 0.49 | 3.3 (d1) | 0/0/0/0 | 23% | 106 | 7.7% (L10, Δβ 0.03) | 0.06 | 0.992 (-1.8) | 1.006 (-0.5) | - |
| c4 | 7 | 0.27 | 0.929 | 0.620 ± 0.054 | 0.544 ± 0.061 | -0.13 | 27.8 / 21.4 / 8.2 | 0.33 | 5.0 (d3) | 3/3/14/2 | 24% | 105 | 1.2% (L5, Δβ 0.12) | 0.10 | 0.665 (+1.0) | 0.668 (+2.3) | S1 |
| fineweb-pro | 7 | 0.30 | 0.925 | 0.662 ± 0.012 | 0.419 ± 0.034 | -0.46 | 26.8 / 31.7 / 4.8 | 0.18 | 4.2 (d3) | 9/10/41/10 | 22% | 100 | 6.6% (L5, Δβ 0.08) | 0.09 | 0.638 (-1.6) | 0.634 (+0.5) | H5, S3, S5 |
| fineweb-edu | 9 | 0.29 | 0.957 | 0.895 ± 0.040 | 0.856 ± 0.055 | -0.04 | 32.6 / 44.3 / 23.8 | 0.66 | 9.5 (d3) | 0/2/10/8 | 23% | 106 | 0.0% (L7, Δβ 0.11) | 0.13 | 0.890 (-0.1) | 0.897 (+1.2) | S1 |
| falcon | 9 | 0.38 | 0.924 | 0.792 ± 0.050 | 0.630 ± 0.081 | -0.23 | 35.1 / 55.0 / 14.1 | 0.27 | 3.1 (d3) | 1/0/3/0 | 23% | 105 | 0.3% (L8, Δβ 0.00) | 0.03 | 0.778 (-0.3) | 0.747 (+0.1) | - |
| falcon-and-cc | 9 | 0.33 | 0.933 | 0.649 ± 0.040 | 0.530 ± 0.046 | -0.20 | 43.0 / 39.3 / 12.0 | 0.33 | 6.9 (d3) | 0/2/2/0 | 23% | 105 | 2.1% (L11, Δβ 0.01) | 0.04 | 0.629 (-0.6) | 0.694 (+2.2) | - |
| falcon-and-cc-qc-10p | 8 | 0.31 | 0.922 | 0.713 ± 0.067 | 0.535 ± 0.062 | -0.29 | 27.2 / 34.3 / 7.6 | 0.28 | 5.6 (d1) | 0/1/7/7 | 23% | 104 | 1.1% (L3, Δβ 0.16) | 0.02 | 0.697 (-0.3) | 0.676 (+0.3) | S1 |
| falcon-and-cc-qc-20p | 8 | 0.35 | 0.932 | 0.739 ± 0.057 | 0.601 ± 0.057 | -0.21 | 32.3 / 40.0 / 11.6 | 0.36 | 6.0 (d1) | 1/2/4/4 | 23% | 105 | 2.2% (L9, Δβ 0.02) | 0.02 | 0.700 (-0.8) | 0.718 (+0.7) | - |
| falcon-and-cc-qc-orig-10p | 8 | 0.36 | 0.938 | 0.771 ± 0.056 | 0.655 ± 0.055 | -0.16 | 27.9 / 36.2 / 11.8 | 0.42 | 6.3 (d1) | 0/1/1/2 | 23% | 106 | 3.5% (L3, Δβ 0.05) | 0.03 | 0.708 (-1.4) | 0.722 (+0.0) | - |
| falcon-and-cc-qc-tulu-10p | 9 | 0.34 | 0.933 | 0.948 ± 0.040 | 0.918 ± 0.052 | -0.03 | 29.1 / 55.0 / 24.3 | 0.59 | 5.3 (d1) | 0/1/2/0 | 23% | 105 | 2.4% (L7, Δβ 0.10) | 0.14 | 0.859 (-2.1) | 0.868 (-0.7) | - |
| dclm-baseline | 8 | 0.35 | 0.899 | 0.666 ± 0.075 | 0.439 ± 0.101 | -0.42 | 34.0 / 39.4 / 6.6 | 0.13 | 2.3 (d3) | 0/1/16/12 | 23% | 103 | 5.7% (L9, Δβ 0.00) | 0.03 | 0.634 (-0.6) | 0.606 (-0.2) | S2, S3, S5 |
| dclm-baseline-qc-7p-fw2 | 8 | 0.38 | 0.951 | 0.838 ± 0.043 | 0.752 ± 0.052 | -0.11 | 29.3 / 43.0 / 16.4 | 0.52 | 7.7 (d1) | 0/0/3/9 | 23% | 107 | 3.9% (L6, Δβ 0.09) | 0.09 | 0.781 (-1.4) | 0.753 (-0.9) | - |
| dclm-baseline-qc-7p-fw3 | 9 | 0.26 | 0.925 | 0.750 ± 0.079 | 0.585 ± 0.059 | -0.25 | 36.8 / 51.2 / 12.1 | 0.35 | 3.3 (d1) | 3/4/19/5 | 22% | 101 | 7.9% (L3, Δβ 0.11) | 0.03 | 0.719 (-0.5) | 0.662 (-0.9) | S5 |
| dclm-baseline-qc-fw-3p | 9 | 0.22 | 0.931 | 0.784 ± 0.002 | 0.642 ± 0.028 | -0.20 | 39.4 / 56.9 / 16.4 | 0.37 | 5.8 (d3) | 11/8/19/7 | 17% | 106 | 5.8% (L5, Δβ 0.14) | 0.08 | 0.775 (-0.7) | 0.649 (-2.0) | S5 |
| dclm-baseline-qc-fw-10p | 3 | 0.27 | 0.939 | 0.809 ± 0.034 | 0.689 ± 0.068 | -0.16 | 7.3 / 10.6 / 3.5 | 0.42 | 4.6 (d3) | 3/1/18/11 | 22% | 101 | 1.2% (L7, Δβ 0.13) | 0.66 | 0.879 (+2.2) | 0.766 (+0.1) | S1, S1b, S5 |
| dclm-baseline-qc-10p | 8 | 0.32 | 0.932 | 0.717 ± 0.056 | 0.551 ± 0.054 | -0.26 | 39.0 / 48.2 / 11.7 | 0.31 | 6.1 (d1) | 2/4/14/2 | 21% | 103 | 7.5% (L9, Δβ 0.03) | 0.11 | 0.688 (-0.7) | 0.668 (-0.0) | - |
| dclm-baseline-qc-20p | 9 | 0.31 | 0.951 | 0.760 ± 0.055 | 0.687 ± 0.054 | -0.10 | 40.5 / 43.5 / 19.0 | 0.51 | 7.1 (d1) | 0/1/5/3 | 23% | 103 | 5.6% (L8, Δβ 0.11) | 0.11 | 0.727 (-0.7) | 0.749 (+0.9) | - |
| dclm-baseline-25p-dolma1.7-75p | 8 | 0.47 | 0.934 | 1.108 ± 0.057 | 1.195 ± 0.115 | +0.08 | 23.1 / 51.5 / 32.2 | 0.52 | 4.5 (d1) | 0/0/0/0 | 23% | 106 | 2.9% (L9, Δβ 0.02) | 0.02 | 1.027 (-1.8) | 1.027 (-0.8) | - |
| dclm-baseline-50p-dolma1.7-50p | 8 | 0.39 | 0.953 | 0.952 ± 0.046 | 0.926 ± 0.067 | -0.03 | 27.9 / 51.2 / 23.7 | 0.65 | 6.6 (d1) | 0/0/0/0 | 23% | 106 | 0.8% (L9, Δβ 0.06) | 0.06 | 0.872 (-2.1) | 0.823 (-1.9) | - |
| dclm-baseline-75p-dolma1.7-25p | 10 | 0.38 | 0.943 | 0.841 ± 0.066 | 0.741 ± 0.089 | -0.13 | 42.5 / 67.4 / 22.9 | 0.44 | 5.7 (d1) | 0/0/0/0 | 23% | 106 | 4.4% (L9, Δβ 0.04) | 0.04 | 0.772 (-1.3) | 0.785 (-0.2) | - |
H1, H3 and H4 pass for every model: failed slots = 0 everywhere, `final_token_is_equals` = True everywhere, max β s.d. = 0.079, min ρ = 0.899, min EV = 0.22.

## 4. Findings with evidence

### 4.1 Ordering / near-zero gaps (H2)
* No recovered gap is ≤ 0. Since the code takes `abs()`, that is guaranteed by construction and proves nothing on its own.
* **dolma1_7-no-math-code** has seed-47 gaps of 11.8 / 15.7 / **0.40**: the third gap is 4 % of that seed's mean gap. Its d3 across seeds is 2.6 / 3.6 / 0.4, with t = 1.3.
  * **β_log is unusable** here: per seed 0.41 / 0.52 / 0.19, s.d. 0.14, and it is driven entirely by ln d3.
  * **β_direct is robust:** 0.566 / 0.650 / 0.639. Flipping the sign of the 0.40 gap moves seed-47 β_direct only from 0.639 to 0.617. E06 at the same layer finds positive, well-resolved decade gaps (15.8 / 11.0 / 5.7, t ≥ 10) and β_coarse 0.632 vs E01's 0.618.
  * Verdict: **exclude from any β_log analysis; keep for β_direct.**
* The next-closest cases are dclm-baseline (0.13; d3 = 3.6 in seed 45) and fineweb-pro (0.18). Both are above the threshold.

### 4.2 Test-retest (H6) and internal replication (S6)
**dolma1_7-no-reddit fails both:**
* E01 eval β = 1.360, but the same model at the same layer 8 gives β = 1.220 on selection seeds 42–44 (z = −3.1) and β_coarse = 1.130 ± 0.015 in E06 (offset residual −0.18, robust z = −4.3; the next-worst model is at |z| = 2.3).
* The cause is a small, noisy first gap. d1 is 23.9 / 19.0 / 11.1 across eval seeds (single-seed t = 2.8, the lowest d1 t of all models). That drives β_direct 1.28 → 1.42 and β_log 1.42 → 2.09 (β_log s.d. 0.28, the largest of all models).
* ρ in seed 47 is 0.893, the lowest single-seed ρ among the Dolma models.
* This is the largest β in the set, so it has high leverage on any downstream analysis.
* Verdict: **exclude, or replace with a replicated value.** The 6-seed mean at the frozen layer is 1.29; E06 at the same layer gives 1.13 + 0.05 offset ≈ 1.18.

Borderline, kept (|z| between 2 and 3):
* c4 (+2.3);
* falcon-and-cc (+2.2);
* dclm-baseline-qc-fw-3p (−2.0; its tiny β s.d. of 0.002 is luck with 3 seeds: 0.782 / 0.783 / 0.787);
* dclm-baseline-50p-dolma1.7-50p (−1.9; selection z −2.1).

Note on the S6 z: it uses the stored s.d. (ddof = 0), so it is somewhat inflated. The selection-minus-eval differences are also partly common-mode for the 9 zero-rejection models (see 4.6).

### 4.3 Seed consistency (H3)
* β_direct s.d. ranges from 0.002 to 0.079, median 0.056. That is in line with what k = 40 is expected to give, and no model exceeds 0.10.
* The s.d. is dominated by a shared seed effect: seed 45 gives the lowest β in 21 of 25 models, because d1 is larger in seed 45. The group-1 targets (1–39, drawn first) are identical across models up to the first rejection, so this is prompt-sample noise common to all models. It is not model-specific unreliability.

### 4.4 β_direct vs β_log (S3, S4)
* The two fits agree (|ln ratio| ≤ 0.13) for models with near-geometric gaps.
* They diverge strongly when d3 is small relative to d1 and d2: no-math-code −0.51, fineweb-pro −0.46, dclm-baseline −0.42.
* Here β_log = √(d3/d1) ignores d2, while β_direct is dominated by the two big gaps.
* For these 3 models "beta" depends on the estimator. That is a property of the gap shape, not necessarily an error. **β_direct (the paper protocol) is the one to use.** β_log should not be used for these three models or for no-reddit / no-flan (β_log s.d. 0.28 / 0.17).
* General shape issue, the same for all models: d2 > d1 in 19 of 25 models. Part of this is design. Group 1 is 1–39 (geometric mean ≈ 15), so the "first decade" spans only ≈ 0.8 decade, against 1.0 for the other gaps. This biases d1 low for every model, and the bias varies between models wherever small numbers saturate. It is a caveat for the whole protocol, not grounds for exclusion.

### 4.5 Signal-to-noise
* E01 saves no within-group spread. I used two proxies:
  * (i) the single-seed t of each gap across the 3 eval seeds: the median minimum is 5.6. The lowest are no-math-code d3 (1.3), dclm-baseline d3 (2.3) and no-reddit d1 (2.8); d3 of falcon (3.1), qc-7p-fw3 d1 (3.3) and dolma1_6plus d1 (3.3) are just above 3.
  * (ii) the E06 per-prompt PC1 at E01's layer, giving gap / SE with n = 40 per group: minimum 5.6 (dclm-baseline d3), typically 9–15.
* Every model therefore has clearly resolved decade gaps. The weakest gap is d3 (1000 → 10000) in web-trained models with low β, and d1 in high-β Dolma models.

### 4.6 Correct-output filter (H5, S5)
* **Zero-rejection models share prompts.** The 6 dolma, 3 mix and partly falcon-and-cc-qc-orig-10p / falcon models have exactly 0 rejections. Their eval `target_hash` values are identical (10 models share seed-45 targets, 9 share seed 46 and seed 47), so they are measured on identical prompts and are paired. The other 16 models get desynchronised random streams after their first rejection. **Measurement noise is therefore not exchangeable across models:** differences within the paired set are free of prompt-sampling noise, and differences involving the other 16 are not.
* **The rejections are strongly target-specific.** Pooling the stored example rejections over all models:
  * group 3: 83 % of rejected targets are 3-digit 980–999. The failure modes are digit duplication (`997→9997`, `988→9888`) and truncation (`1016→1010`);
  * group 4: 75 % are 5-digit 10000–10019, often written with a thousands comma (`10000→"10,000"`);
  * group 1: in fineweb-pro and dclm-qc-fw-3p, almost all rejections are single digits 1–8.

  So the filter shifts group composition: group 3 toward 4-digit, group 4 toward 4-digit, group 1 away from single digits.
* **fineweb-pro (41 % in group 3, H5 triggered):**
  * The estimated 3-digit share in accepted group-3 targets is about 34 % instead of 50 %. The 1-token target count drops to 100 of 160, against 106 for unfiltered models.
  * To size the effect, I measured the digit-boundary step in E06 per-prompt data at E01's layer: median 0.29 × d3 at 1000 and 0.14 × d3 at 10000. Plugging that composition bias into the fit changes β_direct by only **+0.003** for fineweb-pro (≤ 0.005 for every model), because β_direct is dominated by d1 and d2.
  * E06 at the same layer (with its own filter) agrees: robust z +0.5.
  * By the pre-specified H5 rule (bias < 0.03 and H6 passes): **keep, with a caveat.** The bias that cannot be checked is a selection on how the model represents the number (accepted prompts may be "easier"), and these data cannot quantify it.
* **S5 (moderate):**
  * dclm-baseline (16 % / 12 % in groups 3 / 4);
  * dclm-baseline-qc-7p-fw3 (19 % g3);
  * dclm-baseline-qc-fw-3p (19 % g3, plus 11 % g1 concentrated on 1–9, so single-digit share 23 % → ≈ 17 %);
  * dclm-baseline-qc-fw-10p (18 % / 11 %);
  * fineweb-pro (g1 9 %).

  The group-1 shift is the most consequential direction, because group 1 spans 1.6 decades: dropping single digits raises the group-1 mean, which shrinks d1 and raises β. I cannot size it from E01. Weak, non-decisive consistency: qc-fw-3p has E01 > E06 by 0.086 beyond the offset (z = −2.0).

### 4.7 Tokenization diagnostics
* All 25 models use the same OLMo tokenizer: `OLMoTokenizerFast`, no BOS, final token id 30 (`=`) in 160 of 160 prompts for every model.
* The target-token-length histogram varies only through the filter: 100–107 one-token targets out of 160, against 106 when nothing is rejected. It tracks the group-3/4 rejection rate (fineweb-pro 100, qc-7p-fw3 101, qc-fw-10p 101).
* I found no tokenization defect. The histogram is stored for selection seed 42 only.

### 4.8 Explained variance and ρ
* EV(PC1) ranges from 0.22 (qc-fw-3p) to 0.52 (no-reddit); ρ from 0.899 (dclm-baseline) to 0.957.
* All are well above H4. EV is lowest in several web/dclm models (0.22–0.27), meaning PC1 carries less of the variance there. The β values still replicate (E06 z within ±2), so this is not grounds for exclusion.

### 4.9 Layer selection (S1, S1b)
These are protocol ambiguities, not measurement errors.

| model | top-2 q tie | β at the two layers |
|---|---|---|
| fineweb-edu | L9 and L7 tied (< 0.05 %) | 0.89 vs 0.78 |
| c4 | L7 vs L5 (1.2 %) | 0.67 vs 0.78 |
| falcon-and-cc-qc-10p | L8 vs L3 (1.1 %) | 0.70 vs 0.86 |
| dclm-baseline-qc-fw-10p | L3 vs L7 (1.2 %) | 0.88 vs 0.75 |

* dclm-baseline-qc-fw-10p also has a very steep profile: L2 β = 1.54 vs L3 β = 0.88. It is the only model whose selected layer is ≤ 4 apart from no-math-code (L4).
* dolma1_7 (0.7 %, Δβ 0.096), falcon (0.3 %, Δβ 0.00) and 50p (0.8 %, Δβ 0.06) are near-ties with little β change.
* Recommendation: run a sensitivity analysis with a common fixed layer, or the runner-up layer, for these models rather than excluding them.

## 5. Flagged models and verdict

| model | criterion | verdict |
|---|---|---|
| **dolma1_7-no-reddit** | H6 (E06 retest z −4.3), S6 (selection vs eval z −3.1), S2 (d1 t 2.8), S4 (β_log s.d. 0.28) | **Exclude** its E01 β_direct, or replace it with a replicated value (≈ 1.18–1.29 instead of 1.36). It is the only model where the E01 number fails replication. |
| **dolma1_7-no-math-code** | H2 (d3 = 4 % of mean gap in seed 47), S2, S3 | **Exclude for β_log only**; keep β_direct (stable 0.57–0.65; E06 agrees). |
| fineweb-pro | H5 (41 % group-3 rejection), S3, S5 | Keep. Bias check passed (estimated Δβ +0.003; E06 z +0.5). Caveat: non-random subset. |
| dclm-baseline | S2 (d3 t 2.3), S3, S5 | Keep. Do not use β_log. |
| dclm-baseline-qc-fw-3p | S5 (g1 single-digit rejections), E06 z −2.0 | Keep, with caveat. |
| dclm-baseline-qc-7p-fw3 | S5 | Keep. |
| dclm-baseline-qc-fw-10p | S1, S1b (L3 with steep layer profile), S5 | Keep; include in a layer-sensitivity analysis. |
| c4, fineweb-edu, falcon-and-cc-qc-10p | S1 (layer near-tie, Δβ 0.11–0.16) | Keep; include in a layer-sensitivity analysis. |
| dolma1_7-no-flan | S4 (β_log s.d. 0.17) | Keep for β_direct. |

**Honest bottom line:**
* Measured on their own terms, 23 of the 25 E01 β_direct values are sound: ordered groups, resolved gaps, reproducible, and confirmed by an independent same-layer retest.
* One value (dolma1_7-no-reddit) fails replication and should be excluded or replaced.
* β_log should not be used for dolma1_7-no-math-code (and is generally the less reliable estimator).
* The other flags call for caveats or sensitivity analyses, not exclusion.
* The systematic design limits affect all models: only 3 gaps, the 1–39 first group, sign-blind `abs()` gaps, and unpaired prompt sets for the 16 models with any rejection. Whole-protocol remedies (E06-style) are preferable to dropping models.
