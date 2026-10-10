# Pre-registration (written 2026-10-10, before any real model data was extracted)

The git commit that adds this file is the timestamp. Nothing below may be changed after real
results exist; changes go to "Changes after seeing data" at the end, with the reason.

## Hypotheses

- **H1, timing.** The onset of topological truth structure (T1, T3b or T3c) differs from the onset
  of linear separability (L1) and of behaviour (B1) by ≥ 2 checkpoints (leads or lags).
  *Test:* `onset.csv`, reference layer, seed 0; supported if |onset_index(T) − onset_index(L1 or B1)| ≥ 2
  for at least one topological measure, and this holds for seed 1 too.
- **H2, abrupt vs smooth.** Probe AUC rises smoothly, but topological velocity is concentrated in a
  narrow window. *Test:* `concentration.csv`, reference layer, H1, normalised clouds: C exceeds its
  checkpoint-order permutation null (p < 0.05) **with the init transition excluded** (the step-0 → 512
  transition is a change from random weights and would trivially make every measure "concentrated"),
  or a single velocity peak with peak/median > 3. Both seeds.
- **H3, shape.** True-only and false-only clouds differ in shape: T1 accuracy above its
  mixed-vs-mixed null (z > 1.645, both seeds) at the reference layer at checkpoints where L1 AUC > 0.8.
- **H4, local truth.** Within-topic truth structure (T3b) is significant (max-statistic corrected
  p < 0.05) while cross-topic truth structure (T3c) is weak (z < 2 at the reference layer), and/or
  this pattern changes over training.

A null on all four (topology rises in lockstep with the probe) is a legitimate result.

## Fixed analysis choices

All in `tt/config.py`: checkpoints (Ravfogel et al.'s 15), P1 topics `cities`, `sp_en_trans`,
`larger_than` (≤ 300 per class per topic), B = 64 subsamples of m = 160, subsample seeds 0
(primary) and 1, N_NULL_T1 = 5 mixed-vs-mixed splits, 1000 within-topic permutations for T3, 2000
checkpoint-order permutations for T2-C, onset = 2 consecutive checkpoints above the null 95th
percentile.

- **Primary preprocessing for PH:** centre the cloud, divide by its median pairwise distance over a
  fixed set of 500 items. Secondary: raw activations. Note: VR diagrams scale linearly with the
  cloud, so within one cell the raw diagrams are the normalised ones times one constant; T1 (on
  standardised summaries) and T3 (MST, scale-free) are therefore identical under both, and only
  velocity and absolute feature values differ. Both velocities are reported.
- **Reference layer:** the layer with the highest mean L1 AUC over the last three checkpoints.
- **Read position:** P1 the last token (the final period); P2 the token before the final attribute.
- **Layer indexing:** 0 = token embeddings, l = output of block l (forward hooks, so the last layer
  is taken *before* the final layer norm, like all others).

## Validation (synthetic, before real data)

All six tests pass (`validation/validation.json`, `figures/fig0_validation.png`). Two pass
criteria were revised after the first validation run, before any real data, and the reasons are
recorded in `validation.json` → `criteria_changes`. In short: the mixed-vs-mixed T1 null is about
0.70, not 0.5, because overlapping subsamples from two fixed pools are easy to tell apart. So only
the T1 gap can be interpreted, as the brief says, and its maximum is about 0.3.

## Changes after seeing data

(none yet)
