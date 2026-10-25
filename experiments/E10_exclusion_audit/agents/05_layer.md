# Auditor 05: reliability of the per-model layer selection in E01

Scope: whether "β at the selected layer" (E01, `results/datadecide/<recipe>.json`) is a
well-defined quantity for each of the 25 models. Only measurement-side quantities were used:
q = √(EV·|ρ|) per layer and seed, the selected layer, per-layer β. I did not compute, open or
reason about any α–β relation. From `results/beta_fine/fixed_layer_beta.csv` I read **only**
the `beta_cont_L<k>` columns, and only to check whether tied layers give different β.
Scripts: `layer_audit.py`, `layer_rule.py`, `layer_table.py` in this folder.

## Data available
- `selection.layer_scores[l]`: `q_per_seed` (seeds 42/43/44) and `q_median` for layers 1–16.
- `selection.layers_direct[l].beta_mean`: per-layer β_direct (3-gap fit) averaged over the
  **selection** seeds 42–44, so it is on the same prompts as the selection. This is the E01-internal per-layer β.
- `evaluation`: **only the frozen layer** on seeds 45–47 (`run_frozen_layer_evaluation`).
  There are **no per-layer evaluation metrics**, so "which layer would be chosen on the eval
  seeds" cannot be computed from E01. As a partial check, q at L* on the eval seeds reproduces the
  selection value (median difference −0.007; range −0.018 to +0.011). That is a mild
  winner's-curse shrinkage and gives no sign of a spurious pick.
- Independent per-layer β: E06 `beta_cont_L<k>` (fresh prompts, 10 groups, ~100 prompts/group,
  median over seeds). Its absolute level differs from E01 (different fit), but the
  **differences between layers within a model** can be compared.

## Noise scales (computed before the rule was applied)
- Seed-to-seed s.d. of q at a layer: median 0.0074 (all models, layers ≥ 2). The s.d. of the paired
  per-seed difference q(L*)−q(l) is 0.0071, so the SE of a 3-seed paired difference is ≈ 0.004
  and 2·SE ≈ 0.008.
- Seed-to-seed s.d. of β at the frozen layer (eval seeds): median 0.055. The SE of a 3-seed mean is ≈ 0.03,
  and a difference of two such means has SE ≈ 0.045, so 2·SE ≈ 0.09.

## A-priori rule
1. **Tie set T** = {L*} ∪ {l ≥ 2 : q_med(L*) − q_med(l) < δ} ∪ {argmax layer of any single
   selection seed}, with **δ = 0.01** (≈ 2·SE of a 3-seed paired q difference, rounded up). Layer 1
   and the embedding layer are excluded because q < 0.2 there everywhere.
2. **"β at the selected layer" is ill-defined (FLAG)** if the β range across T exceeds
   **τ = 0.10** (≈ 2·SE of a difference of two 3-seed β means) in **both** independent
   per-layer β sources: E01 selection-seed β_direct **and** E06 β_cont. If the range exceeds τ in only one
   source, the model is **borderline**: the disagreement is not reproduced, so it is probably
   fit noise.
3. Not flagging criteria by themselves: (a) two peaks (early 2–5 vs middle 6–10), since early
   peaks are genuine number lines (prior finding); (b) a selected layer outside the modal band
   (6–10, where 23/25 models peak), unless the selection is also tied by rule 1–2;
   (c) a small margin whose tied layers agree on β (e.g. falcon, 50p-dolma mix).

## Per-model table (all 25)
q = √(EV·|ρ|). β_E01sel = per-layer β_direct mean over selection seeds 42–44. β_E06 = `beta_cont_L<k>`.
"Two peaks" = the early (2–5) and middle (6–10) maxima are separated by a dip ≥ 0.01 below the
lower peak. "Competitive" = the two peaks are also within 0.01 of each other.

|recipe|L*|q(L*)|runner-up|margin|margin %|argmax per seed 42/43/44|tie set T (δ=0.01)|β_E01sel over T|range E01|β_E06 over T|range E06|early pk (2-5)|mid pk (6-10)|two peaks?|in band 6-10|β_eval (45-47)|verdict|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|dolma1_7|6|0.5925|10|0.0039|0.7|6/6/4|{4, 6, 10}|L4:0.79, L6:0.93, L10:1.03|0.24|L4:0.84, L6:0.92, L10:1.02|0.18|L4 0.584|L6 0.592|yes, competitive (dip 0.556)|yes|0.978±0.077|FLAG|
|dolma1_7-no-code|8|0.6224|9|0.0229|3.7|8/8/8|{8}|L8:0.83|0.00|L8:0.78|0.00|L4 0.581|L8 0.622|yes (dip 0.565)|yes|0.856±0.060|ok|
|dolma1_7-no-math-code|4|0.5698|10|0.0143|2.5|4/4/4|{4}|L4:0.67|0.00|L4:0.55|0.00|L4 0.570|L10 0.555|yes (dip 0.520)|NO|0.618±0.037|ok|
|dolma1_7-no-reddit|8|0.6939|9|0.0345|5.0|8/8/8|{8}|L8:1.22|0.00|L8:1.18|0.00|L5 0.626|L8 0.694|yes (dip 0.594)|yes|1.360±0.060|ok|
|dolma1_7-no-flan|9|0.6660|7|0.0200|3.0|9/9/9|{9}|L9:1.14|0.00|L9:1.19|0.00|L4 0.634|L9 0.666|yes (dip 0.591)|yes|1.197±0.056|ok|
|dolma1_6plus|9|0.6411|10|0.0491|7.7|9/9/9|{9}|L9:0.99|0.00|L9:1.00|0.00|L4 0.587|L9 0.641|yes (dip 0.543)|yes|1.077±0.063|ok|
|c4|7|0.4928|5|0.0060|1.2|7/7/7|{5, 6, 7}|L5:0.78, L6:0.77, L7:0.67|0.12|L5:0.58, L6:0.59, L7:0.56|0.03|L5 0.487|L7 0.493|no (dip 0.483)|yes|0.620±0.054|borderline|
|fineweb-pro|7|0.5160|5|0.0342|6.6|7/7/7|{7}|L7:0.64|0.00|L7:0.58|0.00|L5 0.482|L7 0.516|no (dip 0.479)|yes|0.662±0.012|ok|
|fineweb-edu|9|0.5233|7|0.0000|0.0|9/9/9|{7, 9}|L7:0.78, L9:0.89|0.11|L7:0.70, L9:0.87|0.18|L3 0.491|L9 0.523|yes (dip 0.459)|yes|0.895±0.040|FLAG|
|falcon|9|0.6049|8|0.0019|0.3|9/9/9|{8, 9}|L8:0.78, L9:0.78|0.00|L8:0.67, L9:0.68|0.01|L3 0.505|L9 0.605|yes (dip 0.459)|yes|0.792±0.050|ok|
|falcon-and-cc|9|0.5495|11|0.0118|2.1|9/9/9|{9}|L9:0.63|0.00|L9:0.62|0.00|L5 0.482|L9 0.549|no (dip 0.482)|yes|0.649±0.040|ok|
|falcon-and-cc-qc-10p|8|0.5448|3|0.0057|1.1|3/8/3|{3, 8}|L3:0.86, L8:0.70|0.16|L3:0.88, L8:0.60|0.28|L3 0.539|L8 0.545|yes, competitive (dip 0.485)|yes|0.713±0.067|FLAG|
|falcon-and-cc-qc-20p|8|0.5873|9|0.0127|2.2|8/8/8|{8}|L8:0.70|0.00|L8:0.67|0.00|L5 0.551|L8 0.587|yes (dip 0.505)|yes|0.739±0.057|ok|
|falcon-and-cc-qc-orig-10p|8|0.5914|3|0.0209|3.5|8/8/8|{8}|L8:0.71|0.00|L8:0.66|0.00|L3 0.570|L8 0.591|yes (dip 0.529)|yes|0.771±0.056|ok|
|falcon-and-cc-qc-tulu-10p|9|0.5756|7|0.0139|2.4|9/9/9|{9}|L9:0.86|0.00|L9:0.81|0.00|L5 0.547|L9 0.576|yes (dip 0.476)|yes|0.948±0.040|ok|
|dclm-baseline|8|0.5762|9|0.0327|5.7|8/8/8|{8}|L8:0.63|0.00|L8:0.59|0.00|L5 0.511|L8 0.576|no (dip 0.506)|yes|0.666±0.075|ok|
|dclm-baseline-qc-7p-fw2|8|0.6151|6|0.0239|3.9|8/8/8|{8}|L8:0.78|0.00|L8:0.66|0.00|L5 0.490|L8 0.615|no (dip 0.490)|yes|0.838±0.043|ok|
|dclm-baseline-qc-7p-fw3|9|0.5106|3|0.0403|7.9|9/9/9|{9}|L9:0.72|0.00|L9:0.61|0.00|L3 0.470|L9 0.511|yes (dip 0.384)|yes|0.750±0.079|ok|
|dclm-baseline-qc-fw-3p|9|0.4686|5|0.0273|5.8|9/9/9|{9}|L9:0.78|0.00|L9:0.58|0.00|L5 0.441|L9 0.469|yes (dip 0.412)|yes|0.784±0.002|ok|
|dclm-baseline-qc-fw-10p|3|0.5000|7|0.0062|1.2|7/3/3|{3, 7}|L3:0.88, L7:0.75|0.13|L3:0.76, L7:0.67|0.09|L3 0.500|L7 0.494|yes, competitive (dip 0.455)|NO|0.809±0.034|borderline|
|dclm-baseline-qc-10p|8|0.5647|9|0.0425|7.5|8/8/8|{8}|L8:0.69|0.00|L8:0.66|0.00|L3 0.522|L8 0.565|yes (dip 0.495)|yes|0.717±0.056|ok|
|dclm-baseline-qc-20p|9|0.5611|8|0.0313|5.6|9/9/9|{9}|L9:0.73|0.00|L9:0.72|0.00|L5 0.462|L9 0.561|no (dip 0.462)|yes|0.760±0.055|ok|
|dclm-baseline-25p-dolma1.7-75p|8|0.6667|9|0.0196|2.9|8/8/8|{8}|L8:1.03|0.00|L8:1.10|0.00|L5 0.564|L8 0.667|no (dip 0.564)|yes|1.108±0.057|ok|
|dclm-baseline-50p-dolma1.7-50p|8|0.6117|9|0.0049|0.8|8/8/8|{8, 9}|L8:0.87, L9:0.81|0.06|L8:0.84, L9:0.79|0.06|L5 0.521|L8 0.612|no (dip 0.521)|yes|0.952±0.046|ok|
|dclm-baseline-75p-dolma1.7-25p|10|0.6069|9|0.0268|4.4|10/10/10|{10}|L10:0.77|0.00|L10:0.77|0.00|L5 0.543|L10 0.607|no (dip 0.543)|yes|0.841±0.066|ok|

Distribution of selected layers: L8 ×10, L9 ×9, L7 ×2, L6, L10, L4, L3 ×1 each.

## Flagged models (β at the selected layer not well-defined)
| recipe | why | evidence |
|---|---|---|
| **falcon-and-cc-qc-10p** | Layer 3 wins **2 of 3** selection seeds (argmax 3/8/3). L8 wins only on the median, by 0.0057 (1.1 %). | β at L3 vs L8: E01 0.86 vs 0.70 (Δ 0.16), E06 0.88 vs 0.60 (Δ 0.28). This is the strongest case. The reported β (0.71) is the low end of what the selection cannot tell apart. |
| **dolma1_7** | Three layers within 0.009: L4 0.584, L6 0.592, L10 0.589. Seed 44 picks L4. | β across {4,6,10}: E01 0.79/0.93/1.03 (range 0.24), E06 0.84/0.92/1.02 (range 0.18). The reported β 0.98 sits in the middle of a ±0.1 band set by layer choice alone. |
| **fineweb-edu** | An exact tie: q_med(L9) − q_med(L7) = 8.5·10⁻⁶. L9 is ahead by < 10⁻⁴ in 2 of 3 seeds. | β L7 vs L9: E01 0.78 vs 0.89 (Δ 0.11), E06 0.70 vs 0.87 (Δ 0.18). |

## Borderline (reported, not flagged)
- **dclm-baseline-qc-fw-10p**: L3 is selected, outside the 6–10 band. Seed 42 picks L7 (margin 0.0062).
  β L3 vs L7: E01 0.88 vs 0.75 (0.13 > τ), E06 0.76 vs 0.67 (0.09, just under τ). At τ = 0.08,
  or with δ ≥ 0.015, it becomes a FLAG. This is the closest call. Its β is probably overstated by ≈0.1
  compared with the in-band layer.
- **c4**: T = {5,6,7}, with all seeds picking L7. The E01 range is 0.12, but the E06 range is 0.03. The independent
  measurement shows a flat β across the tied layers, so the E01 spread is 3-gap fit noise. c4's β is
  effectively well-defined.

## Not flagged despite surface warning signs
- **dolma1_7-no-math-code** (L4, outside the band): a clear, seed-consistent winner (4/4/4), 0.014 above the best
  in-band layer (L10). β at L4 (E01 0.67, E06 0.55) lies within its own in-band β range
  (E01 0.60–0.74, E06 0.56–0.69). The quantity is well-defined. The depth differs from other models, but the value would
  not change much at a band layer (E06 L10 is +0.14 higher, E01 only +0.06). Comparability caveat only.
- **falcon** (margin 0.0019, L8≈L9) and **dclm-50p-dolma-50p** (margin 0.0049, L8≈L9): the tied layers give
  the same β (Δ ≤ 0.06 in both sources), so the near-tie is harmless.
- Two-peak profiles (yes in 15/25) are not a problem in themselves. Only the three with a
  *competitive* early peak (dolma1_7, qc-10p, qc-fw-10p) show up above, and they appear because their β differs by layer.

## Sensitivity of the rule
| δ \ τ | result (τ = 0.10 unless stated) |
|---|---|
| δ = 0.005 / 0.0075 / 0.01 | FLAG {dolma1_7, fineweb-edu, falcon-and-cc-qc-10p} |
| δ = 0.015 / 0.02 | + falcon-and-cc-qc-tulu-10p (L7, 0.014 below; Δβ 0.10/0.15), + dclm-baseline-qc-fw-10p |
| τ = 0.08 (δ 0.01) | + dclm-baseline-qc-fw-10p |
| τ = 0.12 or 0.15 | {dolma1_7, falcon-and-cc-qc-10p} only (fineweb-edu drops: E01 Δ = 0.11) |

The robust core is **falcon-and-cc-qc-10p and dolma1_7**. **fineweb-edu** is flagged under
the stated rule (exact q tie, E06 Δβ 0.18) but is marginal on the E01 side.

## Side observation (outside my aspect, for the coordinator)
Even at a fixed layer, β on eval seeds 45–47 minus β on selection seeds 42–44 is +0.04 on
median (range −0.07 to +0.14). **dolma1_7-no-reddit** has 1.36 (eval) vs 1.22 (selection) at the same L8, a difference of
≈ 3–4 SE. That is a prompt-sample effect, not a layer effect. It suggests the E01 3-gap β has more
seed sensitivity than the ±0.06 s.d. implies for that model.

## Honest bottom line
For 22 of 25 models the layer selection is reliable enough for this purpose: the selected layer is
seed-consistent, or any tied layer gives the same β. For three models (falcon-and-cc-qc-10p,
dolma1_7, fineweb-edu), the E01 selection cannot distinguish layers whose β differs by
0.11–0.28, so a single "β at the selected layer" is not well-defined for them. Possible remedies: excluding them,
or replacing β by a fixed-layer or tie-averaged β. A fixed common layer (e.g. L8, as in
`src/fixed_layer_beta.py`) removes this problem entirely for all 25.
