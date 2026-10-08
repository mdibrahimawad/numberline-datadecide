# E01 details

## Design decisions
- **Correct-output filter** (`src/geometry.py`, `collect_hidden_states`): greedy decode up
  to 8 new tokens; **only the new tokens** are decoded; accept if the first integer
  equals the target. A rejected slot resamples a target from the same group with
  context built the same way, up to 100 candidates per slot; a slot that exhausts them
  marks the group as failed. Per group the JSON stores candidates tried, accepted,
  rejection rate, failed slots and 5 example rejections (`filter_stats`).
  The decoding logic follows `reference/llm_natural_log_paperfaithful/main_lab.py`.
- **Loading DataDecide checkpoints:** `hf_olmo.OLMoForCausalLM` (ai2-olmo). It only has
  a working `.generate()` on `transformers < 4.50`, hence the pin
  `transformers==4.49.0 ai2-olmo==0.6.0`. Hidden states come from
  `output_hidden_states=True`. The final prompt token was verified to be `=`
  (`final_token_is_equals` in each JSON).
- **Layer selection, then frozen evaluation** (`src/datadecide_sweep.py`): selecting
  and evaluating on the same prompts would overfit the layer, so selection uses seeds
  42–44 and evaluation uses 45–47. The score √(EV·|ρ|) needs both a dominant direction
  (EV) and one ordered by magnitude (ρ).
- **Both β fits** are reported because the paper's direct geometric fit and the
  repo's earlier log-linear approximation differ (`src/spacing_fit.py`).

## Results (frozen layer, evaluation seeds 45–47; `results/datadecide/summary.csv`)

| Recipe | Layer | ρ | EV | β_direct ± s.d. | R² (direct) | β_log | Highest rejection rate |
|---|---|---|---|---|---|---|---|
| dolma1_7 | 6 | 0.933 | 0.37 | 0.978 ± 0.077 | 0.05 | 0.969 | 0% (group 1) |
| dolma1_7-no-code | 8 | 0.935 | 0.41 | 0.856 ± 0.060 | 0.17 | 0.774 | 0% (group 1) |
| dolma1_7-no-math-code | 4 | 0.945 | 0.34 | 0.618 ± 0.037 | 0.63 | 0.372 | 1% (group 1) |
| dolma1_7-no-reddit | 8 | 0.912 | 0.52 | 1.360 ± 0.060 | 0.44 | 1.724 | 0% (group 1) |
| dolma1_7-no-flan | 9 | 0.931 | 0.47 | 1.197 ± 0.056 | 0.12 | 1.465 | 0% (group 1) |
| dolma1_6plus | 9 | 0.923 | 0.43 | 1.077 ± 0.063 | 0.03 | 1.175 | 0% (group 1) |
| c4 | 7 | 0.929 | 0.27 | 0.620 ± 0.054 | 0.87 | 0.544 | 14% (group 3) |
| fineweb-pro | 7 | 0.925 | 0.30 | 0.662 ± 0.012 | 0.47 | 0.419 | 41% (group 3) |
| fineweb-edu | 9 | 0.957 | 0.29 | 0.895 ± 0.040 | 0.17 | 0.856 | 10% (group 3) |
| falcon | 9 | 0.924 | 0.38 | 0.792 ± 0.050 | 0.25 | 0.630 | 3% (group 3) |
| falcon-and-cc | 9 | 0.933 | 0.33 | 0.649 ± 0.040 | 0.72 | 0.530 | 2% (group 3) |
| falcon-and-cc-qc-10p | 8 | 0.922 | 0.31 | 0.713 ± 0.067 | 0.41 | 0.535 | 7% (group 3) |
| falcon-and-cc-qc-20p | 8 | 0.932 | 0.35 | 0.739 ± 0.057 | 0.42 | 0.601 | 4% (group 3) |
| falcon-and-cc-qc-orig-10p | 8 | 0.938 | 0.36 | 0.771 ± 0.056 | 0.36 | 0.655 | 2% (group 4) |
| falcon-and-cc-qc-tulu-10p | 9 | 0.933 | 0.34 | 0.948 ± 0.040 | 0.05 | 0.918 | 2% (group 3) |
| dclm-baseline | 8 | 0.899 | 0.35 | 0.666 ± 0.075 | 0.48 | 0.439 | 16% (group 3) |
| dclm-baseline-qc-7p-fw2 | 8 | 0.951 | 0.38 | 0.838 ± 0.043 | 0.21 | 0.752 | 9% (group 4) |
| dclm-baseline-qc-7p-fw3 | 9 | 0.925 | 0.26 | 0.750 ± 0.079 | 0.37 | 0.585 | 19% (group 3) |
| dclm-baseline-qc-fw-3p | 9 | 0.931 | 0.22 | 0.784 ± 0.002 | 0.26 | 0.642 | 19% (group 3) |
| dclm-baseline-qc-fw-10p | 3 | 0.939 | 0.27 | 0.809 ± 0.034 | 0.22 | 0.689 | 18% (group 3) |
| dclm-baseline-qc-10p | 8 | 0.932 | 0.32 | 0.717 ± 0.056 | 0.41 | 0.551 | 14% (group 3) |
| dclm-baseline-qc-20p | 9 | 0.951 | 0.31 | 0.760 ± 0.055 | 0.55 | 0.687 | 5% (group 3) |
| dclm-baseline-25p-dolma1.7-75p | 8 | 0.934 | 0.47 | 1.108 ± 0.057 | 0.08 | 1.195 | 0% (group 1) |
| dclm-baseline-50p-dolma1.7-50p | 8 | 0.953 | 0.39 | 0.952 ± 0.046 | 0.04 | 0.926 | 0% (group 1) |
| dclm-baseline-75p-dolma1.7-25p | 10 | 0.943 | 0.38 | 0.841 ± 0.066 | 0.18 | 0.741 | 0% (group 1) |

No group failed in any model (`failed_groups` is empty everywhere).

## Caveats
- **R² of the direct fit is not a quality measure near β ≈ 1.** Equal gaps leave nothing
  to explain, so a perfect β = 1 fit scores R² ≈ 0. The low R² of the Dolma models (≈1)
  therefore does **not** show their β is noisy. This was believed at first and corrected
  in E06, which reports a relative error instead.
- **Only 3 gaps per fit**, with group bands of fixed absolute width (±20), so the bands
  are much narrower in log terms at 10000 than at 100. E06 fixes both (9 gaps, ±10 %).
- **Per-prompt hidden states were not saved**, only per-layer summaries. Regrouping
  requires a new run (done in E06).
- β s.d. across evaluation seeds is ≈ 0.05 here (40 prompts per group); E06 with 100 per
  group gets ≈ 0.015.
