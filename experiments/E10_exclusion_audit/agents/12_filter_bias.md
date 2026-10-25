# 12 — Selection bias from the correct-output filter

Auditor scope: the correct-output filter (a prompt is kept only if the greedy continuation starts with the target n; a rejected slot is resampled from the same group). Blind audit: no alpha–beta relationship was computed or consulted. Every flag below uses only filter statistics, the per-prompt E06 data, and beta itself (no data-side quantity).

Scripts (all read-only on the repo; outputs are CSVs next to them), directory `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/fb/`:
`e01.py` (E01 filter_summary + per-run filter_stats + logged rejections), `e01b.py` (where E01 rejections fall), `e01c.py` (E01 composition skew carried over to E06 PC1), `e06.py` (E06 rejection per band; reweighted (IPW) re-fit of beta), `e06b.py` (band coverage / sub-bin tests), `e06c.py` + `e06d.py` (complete first-try rejection map from slot matching), `sens.py` (worst-case bound).

## 0. How the filter works (code facts)

* E01 groups (`utils/prompts.py:14 default_interval`, half-width 20): g1 = 1..39, g2 = 80..119, g3 = 980..1019, g4 = 9980..10019. **Groups 2–4 straddle a power of ten**: half of each band has one digit fewer (80–99, 980–999, 9980–9999) than the other half (100–119, 1000–1019, 10000–10019). g1 = 9 one-digit + 30 two-digit numbers.
* E01 filter (`src/geometry.py:250-312 _filter_correct_prompts`): 40 slots per group per run, up to 100 candidates per slot, uniform redraw inside the same interval. Stats are kept per group, with **at most 5 logged rejections per group per run** (`geometry.py:280`). `filter_summary` pools the 6 runs (selection seeds 42–44 + evaluation seeds 45–47). I checked that the per-run sums equal filter_summary for every model and group (assert in `e01.py`).
* E06 filter (`src/beta_fine.py:190-228 filtered_prompts`): 100 slots per band per seed, bands ±10 % around 10^(1+g/3) (g0 = 9..11, g3 = 90..110, g6 = 900..1100, g9 = 9000..11000 straddle a power of ten). Targets are uniform within the band, and the RNG is seeded per seed only (`random.Random(seed)`, the same for every model).
* **Consequence used below.** The initial draw of every slot is identical across the 25 models. `dolma1_7-no-flan` rejected 0 of 3000 E06 prompts (tried = 1000 per seed). Its saved targets are therefore exactly the unfiltered first draws. Comparing any model's saved target in slot i with no-flan's shows whether that model **rejected the first-try (uniform) draw**. This gives a complete, unbiased map of first-try rejections, not the 5-per-group sample. Check: I rebuilt the prompts with `make_prompt` and they match all 75,000 saved n0 (`e06d.py`: "prompts reconstructed OK"). The first-try reject counts are ≤ the total reject counts for every model (`e06c.py` output, e.g. fineweb-pro 538 first-try / 1025 total).
* No slot ever ran out of candidates: `failed_slots = 0` in every group of every model, in both E01 and E06 (`e01.py`, `e06.py`: "failed slots total 0").

## 1. Per-model table (all 25)

Columns:
* **E01 rej g1/g2/g3/g4 (%)**: `filter_summary[g].rejection_rate` in `results/datadecide/<recipe>.json`.
* **E01 max-run (%)**: worst single run in any group.
* **E06 max band rej (%)**: pooled over the 3 seeds, from `filter` in `results/beta_fine/step69369-seed-default/<recipe>.json`.
* **E06 first-try rej 3-digit / 2+4-digit (%)**: from the slot matching.
* **E01 4-digit-share shift g3 / g4 (pp)**: estimated change in the share of accepted g3 targets that are ≥1000, and of g4 targets that are ≥10000, against the 50 % the sampler proposes. It is computed from tried counts plus the side of the boundary each logged rejection falls on.
* **ΔβIPW coarse / cont**: E06 beta at E01's layer, re-fitted with the group means reweighted back to the proposal distribution (strata = digit length × 4 sub-bins), minus the unweighted re-fit. The unweighted re-fit reproduces the reported `beta_coarse_oldlayer` / `beta_cont_oldlayer` to ≤1e-4 in every model (`e06.py` output).
* **ΔβE01-skew**: E01's estimated digit-composition skew carried over to E06's PC1 group means at E01's layer: β(skewed) − β(50/50).
* **Worst-case bound E06 / E01**: largest |Δβ| if the rejected prompts' PC1 had sat a full within-group SD above or below the accepted mean (worst sign pattern over 4 groups), using E06 or E01 rejection rates.
* **E01 β sd**: `beta_direct_std` in `results/datadecide/alpha_beta.csv`, the spread over 3 evaluation seeds.

| recipe | E01 rej g1/g2/g3/g4 (%) | E01 max-run (%) | E06 max band rej (%) | E06 first-try rej 3-dig / 2+4-dig (%) | E01 4-digit-share shift g3 / g4 (pp) | ΔβIPW coarse | ΔβIPW cont | ΔβE01-skew | worst-case bound E06 | worst-case bound E01 | E01 β sd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dolma1_7 | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 | 0.3 (g4) | 0.1 / 0.0 | +0.0 / +0.0 | -0.000 | -0.000 | +0.000 | 0.000 | 0.000 | 0.077 |
| dolma1_7-no-code | 0.4 / 0.0 / 0.0 / 0.0 | 2.4 | 0.3 (g3) | 0.1 / 0.1 | +0.0 / +0.0 | -0.002 | -0.000 | +0.000 | 0.001 | 0.000 | 0.060 |
| dolma1_7-no-math-code | 0.8 / 0.0 / 0.0 / 0.0 | 2.4 | 1.3 (g3) | 0.2 / 0.3 | +0.0 / +0.0 | +0.009 | +0.005 | -0.000 | 0.004 | 0.000 | 0.037 |
| dolma1_7-no-reddit | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 | 0.3 (g7) | 0.0 / 0.1 | +0.0 / +0.0 | -0.002 | -0.001 | +0.000 | 0.000 | 0.000 | 0.060 |
| dolma1_7-no-flan | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 | 0.0 (g0) | 0.0 / 0.0 | +0.0 / +0.0 | -0.004 | -0.003 | +0.000 | 0.000 | 0.000 | 0.056 |
| dolma1_6plus | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 | 0.0 (g0) | 0.0 / 0.0 | +0.0 / +0.0 | -0.007 | -0.005 | +0.000 | 0.000 | 0.000 | 0.063 |
| c4 | 3.2 / 3.2 / 14.3 / 1.6 | 20.0 | 12.3 (g4) | 10.3 / 2.5 | +5.4 / -0.4 | -0.001 | +0.002 | +0.008 | 0.017 | 0.021 | 0.054 |
| fineweb-pro | 9.4 / 10.4 / 41.3 / 9.8 | 45.9 | 64.2 (g5) | 41.5 / 7.5 | +16.4 / -4.1 | -0.015 | -0.002 | +0.019 | 0.040 | 0.056 | 0.012 |
| fineweb-edu | 0.4 / 2.0 / 9.8 / 8.0 | 13.0 | 16.7 (g5) | 11.2 / 1.9 | +4.5 / -3.1 | +0.005 | +0.002 | -0.007 | 0.019 | 0.023 | 0.040 |
| falcon | 1.2 / 0.4 / 2.8 / 0.4 | 4.8 | 1.0 (g0) | 0.2 / 0.3 | -0.2 / -0.2 | -0.003 | -0.002 | -0.001 | 0.003 | 0.005 | 0.050 |
| falcon-and-cc | 0.4 / 1.6 / 2.4 / 0.4 | 4.8 | 0.7 (g3) | 0.4 / 0.3 | +0.4 / -0.2 | -0.006 | -0.002 | -0.000 | 0.003 | 0.007 | 0.040 |
| falcon-and-cc-qc-10p | 0.4 / 1.2 / 7.0 / 7.0 | 14.9 | 5.1 (g6) | 3.0 / 1.6 | +3.3 / +0.2 | +0.000 | +0.002 | +0.001 | 0.007 | 0.010 | 0.067 |
| falcon-and-cc-qc-20p | 0.8 / 2.0 / 3.6 / 3.6 | 11.1 | 3.8 (g5) | 2.8 / 1.3 | -0.6 / -0.6 | +0.005 | +0.003 | +0.001 | 0.009 | 0.009 | 0.057 |
| falcon-and-cc-qc-orig-10p | 0.4 / 0.8 / 1.2 / 2.0 | 7.0 | 1.6 (g2) | 0.7 / 0.4 | +0.2 / +0.2 | -0.001 | +0.001 | +0.001 | 0.003 | 0.004 | 0.056 |
| falcon-and-cc-qc-tulu-10p | 0.0 / 0.8 / 2.0 / 0.0 | 2.4 | 2.9 (g5) | 1.8 / 0.4 | +1.0 / +0.0 | -0.003 | -0.002 | +0.000 | 0.003 | 0.004 | 0.040 |
| dclm-baseline | 0.0 / 0.8 / 16.1 / 12.4 | 20.0 | 10.2 (g6) | 6.5 / 1.0 | +7.6 / -6.6 | -0.002 | +0.000 | +0.003 | 0.010 | 0.015 | 0.075 |
| dclm-baseline-qc-7p-fw2 | 0.0 / 0.4 / 2.8 / 9.1 | 11.1 | 1.6 (g2) | 0.7 / 0.4 | +0.6 / -4.6 | +0.000 | -0.000 | -0.003 | 0.002 | 0.008 | 0.043 |
| dclm-baseline-qc-7p-fw3 | 3.2 / 4.0 / 18.9 / 5.1 | 23.1 | 27.0 (g5) | 16.5 / 2.7 | +10.9 / -1.0 | +0.001 | -0.003 | +0.007 | 0.016 | 0.027 | 0.079 |
| dclm-baseline-qc-fw-3p | 11.4 / 8.4 / 18.9 / 7.3 | 24.5 | 20.4 (g4) | 19.3 / 6.0 | +7.8 / -0.7 | +0.004 | +0.003 | +0.005 | 0.048 | 0.043 | 0.002 |
| dclm-baseline-qc-fw-10p | 2.8 / 0.8 / 17.8 / 10.8 | 23.1 | 14.5 (g5) | 9.4 / 1.1 | +7.2 / -5.6 | +0.010 | +0.007 | -0.005 | 0.013 | 0.027 | 0.034 |
| dclm-baseline-qc-10p | 2.4 / 4.4 / 13.7 / 1.6 | 24.5 | 11.8 (g5) | 8.8 / 1.1 | +6.2 / +0.4 | +0.005 | +0.003 | +0.001 | 0.008 | 0.017 | 0.056 |
| dclm-baseline-qc-20p | 0.0 / 1.2 / 4.8 / 2.8 | 7.0 | 3.5 (g5) | 3.0 / 0.6 | +2.1 / -1.5 | +0.001 | +0.001 | +0.000 | 0.005 | 0.008 | 0.055 |
| dclm-baseline-25p-dolma1.7-75p | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 | 0.3 (g4) | 0.1 / 0.0 | +0.0 / +0.0 | -0.005 | -0.004 | +0.000 | 0.000 | 0.000 | 0.057 |
| dclm-baseline-50p-dolma1.7-50p | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 | 0.0 (g0) | 0.0 / 0.0 | +0.0 / +0.0 | -0.000 | +0.000 | +0.000 | 0.000 | 0.000 | 0.046 |
| dclm-baseline-75p-dolma1.7-25p | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 | 0.0 (g0) | 0.0 / 0.0 | +0.0 / +0.0 | -0.000 | -0.000 | +0.000 | 0.000 | 0.000 | 0.066 |

Noise floor for the IPW column: models with no rejections at all (dolma1_6plus, dolma1_7-no-flan, the dclm–dolma mixes) still move by up to 0.007. That movement is the reweighting correcting ordinary sampling noise in the uniform draw, not the filter. So |ΔβIPW| ≲ 0.007 means "no filter effect".

## 2. Findings

### 2.1 Rejection rates by model and group
* 9 models reject essentially nothing: all 6 dolma recipes and the 3 dclm–dolma mixes. E01 rejection is 0 in groups 2–4 and ≤0.8 % in g1; E06 rejection is ≤1.3 % in every band (table). Rejection is clearly a property of the model, and through the model of its training data.
* The highest rejection is in the web-only recipes. **fineweb-pro**: E01 g3 = 41.3 % (409 tried for 240 accepted), worst run 45.9 % (eval or selection seed, `e01_filter.csv` col max3). E06 band 4 (194–236) = 38.4 %, band 5 (418–510) = 64.2 %, band 6 (900–1100) = 23.1 %. **dclm-baseline-qc-7p-fw3**: E06 band 5 = 27.0 %. **dclm-baseline-qc-fw-3p**: E01 max 18.9 %, E06 max 20.4 %. All other models are ≤ 20 % in every group of both experiments (E01 max-run ≤ 24.5 %).
* Pooled over all 25 models, the rejected E01 candidates are g1 97, g2 109, **g3 550**, g4 215 (`e01.py`). The 980–1019 group is the hardest.

### 2.2 Where the rejections fall
* **Inside the E01 boundary groups, rejections pile up on one digit length** (687 of the 971 rejections are logged; `e01b.py`):
  * g3 (980–1019): only 16.7 % of logged rejections are ≥1000; 83 % are the 3-digit 980–999. The most-rejected targets are 997 (35), 988 (26), 989 (24), 992 (23), 991 (18).
  * g4 (9980–10019): 80.5 % of logged rejections are the 5-digit 10000–10019.
  * g2 (80–119): 77 % are ≥100.
  * The proposal is 50/50 in all three groups.
  * Failure modes: mostly a wrong-length number (g3: 207/294 "other length", e.g. `988 -> 9888`, `997 -> 9997`). In g4, 5-digit targets are split, truncated or comma-formatted (`10004 -> "4,000,000,000"`, `10016 -> "1616,000,..."`, `10000 -> "10,000,000"`).
* **The complete E06 first-try map confirms this** (`e06d.py`, the 8 highest-rejection models pooled):
  * first-try rejection by digit count: 1-digit 7.2 %, 2-digit 2.6 %, **3-digit 15.4 %**, 4-digit 3.4 %, 5-digit 4.1 %;
  * band 6: 3-digit (900–999) 21.2 % against 4-digit (1000–1100) 2.3 %;
  * band 9: 4-digit 2.0 % against 5-digit 4.1 %.
  * The target simply occurring inside a context number does not explain it (ctx_contains 6.1 % against 7.0 %).
* **Rejections are tied to particular prompts, not random noise.** 112 of 3000 slots are rejected by ≥4 of the 8 high-rejection models. Independent failures would give 2.6 (simulation in `e06d.py`). Repeatedly rejected targets include 9, 44, 902, 233, 222, 228.
* Sub-bin coverage (10 sub-bins per band, `e06b.py`/`e06_subbin.csv`): 5 bands have χ² p < 0.01, against 2.5 expected by chance. Only 2 bands have a sub-bin below half its nominal share with p < 0.001:
  * fineweb-pro band 6, p = 6e-5, worst sub-bin at 0.40 of nominal. First-try rejection is 25–53 % in every 20-wide sub-bin of 900–999 and 0–12 % in 1000–1100.
  * dclm-baseline-qc-7p-fw3 band 4, p = 4e-4. Rejection is concentrated on 218–233: 15–35 % against 0–9 % elsewhere.
* **No sub-band is empty.** The worst digit-stratum acceptance relative to the other stratum is 0.59 (fineweb-pro band 6, `ratio_lo_hi`). Every stratum keeps ≥ 67 % of its nominal share; E01 fineweb-pro g3 has an estimated 33.6 % 3-digit share against 50 %. This is what makes reweighting possible (positivity holds).

### 2.3 Does the filter bias the measured group positions and β?
* **Position in log n is unaffected.** The accepted targets' mean log10 n differs from the proposal mean by at most **1.7 % of one ⅓-decade spacing** (fineweb-pro band 6). It is < 1 % for every other model (`e06d.py` "max |shift|"). The E01 groups are 40 integers wide (log width ≤ 0.017 decades in g3/g4), so a target-mean shift cannot move β there either.
* **The real channel is digit length.** In g3 the filter enriches 4-digit targets (fineweb-pro estimated +16.4 pp; qc-7p-fw3 +10.9; dclm-baseline +7.6; qc-fw-3p +7.8; qc-fw-10p +7.2). In g4 it enriches 4-digit too, by depleting 10000+ (dclm-baseline −6.6 pp of 5-digit; qc-fw-10p −5.6). If PC1 jumps at a digit-length boundary, both groups shift toward "4-digit" and d₂/d₃ change. Measured effect:
  * IPW re-fit of E06 (strata = digit length × sub-bins): |ΔβIPW coarse| ≤ **0.016** (fineweb-pro −0.015) and |ΔβIPW cont| ≤ **0.008**, for every model at both E01's layer and the newly selected layer (`e06.py` "old" and "sel" blocks). The noise floor is 0.007.
  * E01 skew carried over to E06 PC1: |Δβ| ≤ **0.019** (fineweb-pro +0.019; next largest c4 +0.008, qc-7p-fw3 +0.007). The E06 strata span 900–999 against 1000–1100, much wider than E01's 980–999 against 1000–1019, so the within-group PC1 contrast is overstated and this is an upper estimate.
  * Both are below the median prompt-seed s.d. of E06 beta_coarse (0.0154) or about that size, and well below the E01 3-seed s.d. (median 0.056, `alpha_beta.csv`).
* **What reweighting cannot fix** is selection on the hidden state itself: rejected prompts might have atypical PC1 values. That cannot be tested, because hidden states of rejected prompts are not saved. The worst-case bound (rejected prompts' PC1 a full within-group SD away, worst sign pattern) stays < 0.03 for 23 models. It exceeds 0.03 for **fineweb-pro** (E06 0.040 / E01 0.056) and **dclm-baseline-qc-fw-3p** (0.048 / 0.043). This is a bound, not an observed bias.
* E01-specific caveat: E01 does not store per-prompt targets (only `target_hash`) or hidden states, and logs ≤5 rejections per group per run. The E01 composition estimates rest on 24–30 logged rejections per affected group: fineweb-pro g3 = 30, dclm-baseline g3/g4 = 29/29. They are adequate but approximate.

### 2.4 Rejection is partly a numeracy property of the model
* Whether a model copies "997" correctly after three random 4-digit examples is a behavioural number skill. It is zero-failure in every Dolma-containing model and fails up to 64 % of the time in fineweb-pro (3-digit 418–510). The filter therefore conditions every model on the numbers it handles. The resulting β is **β over correctly copied numbers**, and that estimand is defined differently in effect for each model.
* That is a definitional choice of the paper's protocol, not an error. Removing the filter would mix in prompts where the model is not representing the target at all. It should be described as part of the measurement, and the rejection rate reported as a covariate.
* Excluding models because they reject a lot would amount to selecting on a model trait that very plausibly comes from the training data, the same data property the study is about. That would be selecting on the outcome side.
* Rejection also changes the prompt sets: zero-rejection models see byte-identical E06 prompts, while high-rejection models see a partly different set. The effect of a different prompt set is the prompt-seed noise (~0.015), already part of the β uncertainty.

## 3. Exclusion criteria (fixed before looking at any α–β result; based only on filter / β measurement properties)

| # | Rule | Threshold | Justification | Models flagged |
|---|---|---|---|---|
| F1 | High rejection in any group/band (E01 pooled filter_summary, or E06 pooled over seeds) | > 25 % | At rejection rate r, the worst-case shift in a 50/50 two-stratum band (all rejections in one half) is r/(2(1−r)) of composition. At r = 0.25 that is 17 pp, the point where digit-length composition could plausibly move a group mean by more than seed noise. 25 % is also the threshold the brief suggested; I adopted it before looking at the data. | **fineweb-pro** (E01 g3 41.3 %; E06 b4 38.4 %, b5 64.2 %); **dclm-baseline-qc-7p-fw3** (E06 b5 27.0 %) |
| F2 | Observed filter bias in β | abs(ΔβIPW) (coarse or cont, E06 at E01 or selected layer) or abs(ΔβE01-skew) > 0.03 | 0.03 = 2 × the median E06 prompt-seed s.d. of beta_coarse (0.0154, `fine_vs_data.csv`). A bias smaller than that cannot be told apart from prompt noise and cannot change a model's rank among the 25 by more than the noise already does. | **none** (max 0.016 IPW; 0.019 skew transfer, fineweb-pro) |
| F3 | Coverage failure: a digit stratum or a 1/10 sub-bin of a band accepted at < 50 % of its nominal share, with band χ² p < 0.001 (or any empty stratum) | as stated | Below half the nominal share, the group mean represents that part of the band poorly and reweighting by a factor > 2 gets unstable. The p-value guards against chance: zero-rejection models reach 0.63 of nominal from sampling alone. | **fineweb-pro** (E06 band 6, sub-bin ratio 0.40, p = 6e-5); **dclm-baseline-qc-7p-fw3** (E06 band 4, 0.43, p = 4e-4). No stratum is empty; the worst stratum share is 0.67 of nominal. |
| F4 | Untestable hidden-state selection, worst case | worst-case bound > 0.03 | Same 0.03 noise yardstick, applied to the most pessimistic assumption that the rejected prompts' PC1 sat 1 SD away. A sensitivity flag only. | **fineweb-pro** (0.040 / 0.056), **dclm-baseline-qc-fw-3p** (0.048 / 0.043) |
| F5 | Filter failure (a slot exhausts its 100 candidates) | any | Without it a group mean would rest on < k prompts or be missing. | **none** (failed_slots = 0 everywhere) |

## 4. Is each flag a measurement-validity problem or a genuine property?

* **F1 (fineweb-pro, qc-7p-fw3): genuine model property, not a valid reason to exclude.** A high rejection rate is the model's copying/numeracy behaviour. F2 shows that it does not turn into a measurable β bias: fineweb-pro ΔβIPW −0.015 coarse / −0.002 cont; qc-7p-fw3 +0.001 / −0.003. Report rejection rates alongside β and use them as a covariate / sensitivity variable.
* **F2: no model fails.** There is no measurement-validity case for excluding any model on filter-bias grounds. The largest observed corrections (fineweb-pro ≈ 0.015–0.019) are about one prompt-seed s.d. For fineweb-pro specifically they are larger than its own E01 seed s.d. (0.012), but far below the 0.55–1.38 spread across models.
* **F3 (fineweb-pro band 6, qc-7p-fw3 band 4): partly measurement-relevant, still not grounds to exclude.**
  * fineweb-pro band 6 is a coarse (β-bearing) group, and its under-coverage of 900–999 is exactly the digit-length skew that the IPW re-fit corrects (no empty stratum). The corrected β differs by −0.015.
  * qc-7p-fw3 band 4 (215) is a fine-only band. It does not enter beta_coarse or E01, and it changes beta_cont by −0.003.
  * Recommendation: report IPW-corrected β next to the raw value for these two models.
* **F4 (fineweb-pro, qc-fw-3p): an untestable sensitivity, not a demonstrated problem.** Any downstream result should be shown to survive (a) leaving these two out and (b) replacing their β by the IPW value. Treat that as a robustness check, not as a pre-registered exclusion.
* **Honest bottom line:** on filter grounds, **no model should be excluded**. The filter selects strongly and non-randomly: it rejects 3-digit numbers (980–999 in E01 g3, 194–510 in E06) and 5-digit 10000+ numbers far more than 2- and 4-digit ones, and the same prompts fail across models. But it never empties a stratum, never fails a slot, and moves β by ≤ 0.016 (IPW) / ≤ 0.019 (skew transfer) after correction, which is at or below seed noise.
  * Two improvements for future runs: save the per-prompt targets in E01, and stratify sampling by digit length.
  * The between-model differences in rejection are a real numeracy property and should be reported, not removed.
