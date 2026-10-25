# 14 - Health of the PCA itself (across layers, seeds and models)

Auditor aspect: is PC1 at the layer used for β a clean number axis, or could it be an outlier/massive-activation artefact, a PC1/PC2 swap, a sign flip, or a seed-unstable direction? Blind to any α-β relationship: α was never loaded (only the columns `recipe, selected_layer, rho_mean, ev_mean` of `results/datadecide/alpha_beta.csv` were read).

## 0. What was checked, and how the PCA is computed

* **E06** (`src/beta_fine.py`, `layer_metrics`, l.105-130): sklearn `PCA(n_components=5)` on the raw last-token hidden states (centred, **not** standardised) of 1000 prompts per seed (seeds 45/46/47, 10 groups x 100), at each of 17 layers. PC1 is oriented so that it has positive Pearson correlation with log10 n (l.117). PCs 2-5 are not oriented. Saved: `seed*_pc_evr` [17,5] and `seed*_pc_scores` [17,1000,5] in `results/beta_fine/step69369-seed-default/<recipe>.npz`.
* **E01** (`src/geometry.py` `_transform_layer` / `analyze_layer`; `src/datadecide_sweep.py` l.209 and `select_joint_layer`): one-component PCA on 160 prompts per seed (4 groups x 40). Stored ρ is `abs(Spearman)` (sweep l.209), and β uses `abs(np.diff(group_means))` (`_compression_rate_direct`), so **E01 β and the layer rule are invariant to PC1 sign by construction**. E01 keeps only PC1 EV/ρ, so PC2-swap checks rely on E06 (which is evaluated at both the E01-selected layer `L_E01` and the E06-selected layer `L_E06`).
* Because the PCA is unstandardised, a dimension with a huge, prompt-varying activation (massive activations / outlier features) would dominate PC1 and drive EV1 toward 1 with low ρ. A dimension with a huge but constant value is removed by centring and is harmless.

Script (all numbers below come from it): `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/14_work/pca_health.py`. Outputs: `.../14_work/models.csv` (per model), `.../14_work/layers.csv` (per model x layer, medians over the 3 seeds), `.../14_work/table.md`.

For each model, seed and layer it computes: EVR of PCs 1-5; |Spearman(n, PC_k)| for k=1..5; λ1/λ2 = EVR1/EVR2; R² of log10 n regressed on PC1 alone vs on PCs 1-5 ("R² share" = R²(PC1)/R²(PC1..5), the fraction of linearly decodable log-n signal within the top-5 subspace carried by PC1); the 10 group means of PC1, and the minimum pairwise Pearson correlation of these 10-vectors across the 3 seeds; whether the 4 decade-group means (10, 100, 1000, 10000) are strictly increasing.

## 1. A-priori criteria (rule, threshold, justification)

Thresholds were fixed from general PCA/measurement considerations before reading the per-model results; P3's 0.80 cut is the only one chosen while the first table was on screen, and no model lies near it (min 0.85), so the choice does not change any outcome.

| ID | Rule (evaluated at the layer actually used for β: L_E01, and also L_E06) | Threshold | Justification | Kind of problem |
|---|---|---|---|---|
| P1 | Artefact-dominated PC1 (massive activation / outlier dims) | EV1 ≥ 0.70 at the selected layer, or at any layer with abs ρ1 < 0.5 | Outlier-dimension domination shows as one direction taking most of the variance with no relation to n; a 2048-d residual stream carrying a number manifold plus prompt noise has EV1 well below this | measurement validity |
| P2 | Number axis is not PC1 | in **any** seed, argmax_k abs ρ(n, PC_k) ≠ 1; or median max_{k≥2} abs ρ_k ≥ abs ρ1 | β is read off PC1, so PC1 must be the direction most ordered by n | measurement validity |
| P3 | Number signal spread across PCs (PC1 is a mixture) | R²(log n \| PC1) / R²(log n \| PC1..5) < 0.80 | if PC1 carries < 80 % of the decodable log-n signal, PC1 gaps mix magnitude with another factor | measurement validity |
| P4 | Near-degenerate leading eigenvalues (PC1/PC2 swap risk) | min over seeds λ1/λ2 < 1.5 | with a small eigen-gap the PC1 direction is ill-defined (Davis-Kahan) and can rotate or swap with PC2 between prompt sets | measurement validity (risk; confirm with P2/P6) |
| P5 | EV instability across seeds | CV of EV1 over 3 seeds > 0.10 | the same layer should give the same spectrum on new prompts from the same distribution | measurement validity |
| P6 | Sign flip / axis swap between seeds | min cross-seed corr of the 10 PC1 group means < 0.95, or decade means not monotone in any seed | a flipped or swapped axis gives a different group-mean profile; E01's abs(diff) would silently fold a non-monotone profile | measurement validity |

## 2. Per-model table (all 25 models)

EV, ρ, λ1/λ2, R² share are E06 values (medians over seeds 45-47 unless "min"/"CV"); "E01 EV1 seeds 42-44" is from `results/datadecide/<recipe>.json` → `selection.per_seed[*].layers[L_E01].ev`. "argmax-PC" gives, for each of the 3 seeds, which of PCs 1-5 has the largest abs ρ with n ("111" = PC1 in all three seeds).

| recipe | L E01 / E06 | EV1 @L_E01 (E06 med) | E01 EV1 seeds 42-44 (CV) | λ1/λ2 min @L_E01 / @L_E06 | EV1 CV @L_E01 / @L_E06 | abs ρ(PC1) @L_E01 / @L_E06 | max abs ρ(PC2..5) @L_E01 / @L_E06 | argmax-PC per seed (E01L/E06L) | R² share PC1 @L_E01 / @L_E06 | min cross-seed corr of 10 group means (E01L/E06L) | monotone decade means (seeds) | max EV1 any layer (layer) | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dolma1_7 | 6 / 4 | 0.334 | 0.375, 0.379, 0.366 (0.018) | 4.70 / 5.90 | 0.002 / 0.001 | 0.960 / 0.983 | 0.076 / 0.259 | 111 / 111 | 0.993 / 0.995 | 0.9993 / 0.9998 | 3/3, 3/3 | 0.383 (L4, ρ1=0.98) | none |
| dolma1_7-no-code | 8 / 8 | 0.405 | 0.427, 0.416, 0.414 (0.017) | 7.18 / 7.18 | 0.003 / 0.003 | 0.948 / 0.948 | 0.182 / 0.182 | 111 / 111 | 0.963 / 0.963 | 0.9996 / 0.9996 | 3/3, 3/3 | 0.405 (L8, ρ1=0.95) | none |
| dolma1_7-no-math-code | 4 / 4 | 0.351 | 0.352, 0.329, 0.344 (0.034) | 3.92 / 3.92 | 0.010 / 0.010 | 0.924 / 0.924 | 0.360 / 0.360 | 111 / 111 | 0.871 / 0.871 | 0.9997 / 0.9997 | 3/3, 3/3 | 0.351 (L4, ρ1=0.92) | none |
| dolma1_7-no-reddit | 8 / 8 | 0.511 | 0.527, 0.518, 0.515 (0.012) | 10.54 / 10.54 | 0.011 / 0.011 | 0.929 / 0.929 | 0.236 / 0.236 | 111 / 111 | 0.891 / 0.891 | 0.9986 / 0.9986 | 3/3, 3/3 | 0.511 (L8, ρ1=0.93) | none |
| dolma1_7-no-flan | 9 / 9 | 0.443 | 0.478, 0.476, 0.462 (0.018) | 7.18 / 7.18 | 0.011 / 0.011 | 0.939 / 0.939 | 0.208 / 0.208 | 111 / 111 | 0.932 / 0.932 | 0.9988 / 0.9988 | 3/3, 3/3 | 0.443 (L9, ρ1=0.94) | none |
| dolma1_6plus | 9 / 9 | 0.409 | 0.453, 0.445, 0.447 (0.009) | 7.41 / 7.41 | 0.006 / 0.006 | 0.934 / 0.934 | 0.204 / 0.204 | 111 / 111 | 0.911 / 0.911 | 0.9989 / 0.9989 | 3/3, 3/3 | 0.409 (L9, ρ1=0.93) | none |
| c4 | 7 / 3 | 0.252 | 0.273, 0.263, 0.255 (0.034) | 3.12 / 2.69 | 0.018 / 0.006 | 0.895 / 0.925 | 0.400 / 0.309 | 111 / 111 | 0.862 / 0.925 | 0.9987 / 0.9996 | 3/3, 3/3 | 0.279 (L3, ρ1=0.93) | none |
| fineweb-pro | 7 / 7 | 0.292 | 0.306, 0.289, 0.289 (0.033) | 4.07 / 4.07 | 0.007 / 0.007 | 0.929 / 0.929 | 0.397 / 0.397 | 111 / 111 | 0.876 / 0.876 | 0.9986 / 0.9986 | 3/3, 3/3 | 0.292 (L7, ρ1=0.93) | none |
| fineweb-edu | 9 / 9 | 0.278 | 0.300, 0.286, 0.288 (0.026) | 4.16 / 4.16 | 0.003 / 0.003 | 0.956 / 0.956 | 0.118 / 0.118 | 111 / 111 | 0.984 / 0.984 | 0.9995 / 0.9995 | 3/3, 3/3 | 0.279 (L3, ρ1=0.95) | none |
| falcon | 9 / 8 | 0.380 | 0.405, 0.394, 0.388 (0.022) | 7.69 / 7.47 | 0.010 / 0.010 | 0.925 / 0.934 | 0.210 / 0.233 | 111 / 111 | 0.936 / 0.941 | 0.9992 / 0.9995 | 3/3, 3/3 | 0.382 (L8, ρ1=0.93) | none |
| falcon-and-cc | 9 / 9 | 0.315 | 0.350, 0.322, 0.321 (0.050) | 4.93 / 4.93 | 0.016 / 0.016 | 0.929 / 0.929 | 0.283 / 0.283 | 111 / 111 | 0.930 / 0.930 | 0.9977 / 0.9977 | 3/3, 3/3 | 0.315 (L9, ρ1=0.93) | none |
| falcon-and-cc-qc-10p | 8 / 3 | 0.312 | 0.323, 0.325, 0.301 (0.042) | 4.21 / 4.35 | 0.021 / 0.003 | 0.910 / 0.968 | 0.374 / 0.068 | 111 / 111 | 0.879 / 0.992 | 0.9989 / 0.9996 | 3/3, 3/3 | 0.333 (L3, ρ1=0.97) | none |
| falcon-and-cc-qc-20p | 8 / 8 | 0.359 | 0.371, 0.378, 0.360 (0.025) | 5.03 / 5.03 | 0.028 / 0.028 | 0.942 / 0.942 | 0.266 / 0.266 | 111 / 111 | 0.913 / 0.913 | 0.9984 / 0.9984 | 3/3, 3/3 | 0.359 (L8, ρ1=0.94) | none |
| falcon-and-cc-qc-orig-10p | 8 / 3 | 0.358 | 0.378, 0.376, 0.358 (0.030) | 6.09 / 5.02 | 0.014 / 0.008 | 0.941 / 0.970 | 0.282 / 0.215 | 111 / 111 | 0.934 / 0.966 | 0.9992 / 0.9997 | 3/3, 3/3 | 0.379 (L3, ρ1=0.97) | none |
| falcon-and-cc-qc-tulu-10p | 9 / 2 | 0.332 | 0.363, 0.352, 0.332 (0.045) | 5.47 / 3.79 | 0.006 / 0.010 | 0.951 / 0.912 | 0.154 / 0.267 | 111 / 111 | 0.978 / 0.910 | 0.9990 / 0.9965 | 3/3, 3/3 | 0.377 (L2, ρ1=0.91) | none |
| dclm-baseline | 8 / 8 | 0.353 | 0.375, 0.371, 0.350 (0.037) | 5.05 / 5.05 | 0.012 / 0.012 | 0.919 / 0.919 | 0.313 / 0.313 | 111 / 111 | 0.911 / 0.911 | 0.9986 / 0.9986 | 3/3, 3/3 | 0.353 (L8, ρ1=0.92) | none |
| dclm-baseline-qc-7p-fw2 | 8 / 8 | 0.382 | 0.397, 0.399, 0.376 (0.033) | 5.18 / 5.18 | 0.004 / 0.004 | 0.950 / 0.950 | 0.276 / 0.276 | 111 / 111 | 0.931 / 0.931 | 0.9996 / 0.9996 | 3/3, 3/3 | 0.382 (L8, ρ1=0.95) | none |
| dclm-baseline-qc-7p-fw3 | 9 / 3 | 0.254 | 0.291, 0.281, 0.275 (0.029) | 4.28 / 2.90 | 0.038 / 0.008 | 0.896 / 0.946 | 0.306 / 0.362 | 111 / 111 | 0.853 / 0.985 | 0.9978 / 0.9991 | 3/3, 3/3 | 0.255 (L3, ρ1=0.95) | none |
| dclm-baseline-qc-fw-3p | 9 / 2 | 0.218 | 0.231, 0.235, 0.225 (0.022) | 2.76 / 1.48 | 0.025 / 0.018 | 0.929 / 0.939 | 0.291 / 0.122 | 111 / 111 | 0.908 / 0.942 | 0.9994 / 0.9986 | 3/3, 3/3 | 0.218 (L2, ρ1=0.94) | P4-watch (L_E06 only) |
| dclm-baseline-qc-fw-10p | 3 / 3 | 0.282 | 0.262, 0.266, 0.267 (0.010) | 3.34 / 3.34 | 0.004 / 0.004 | 0.959 / 0.959 | 0.118 / 0.118 | 111 / 111 | 0.980 / 0.980 | 0.9998 / 0.9998 | 3/3, 3/3 | 0.282 (L3, ρ1=0.96) | none |
| dclm-baseline-qc-10p | 8 / 8 | 0.311 | 0.345, 0.340, 0.332 (0.019) | 4.49 / 4.49 | 0.017 / 0.017 | 0.943 / 0.943 | 0.217 / 0.217 | 111 / 111 | 0.937 / 0.937 | 0.9987 / 0.9987 | 3/3, 3/3 | 0.312 (L3, ρ1=0.94) | none |
| dclm-baseline-qc-20p | 9 / 9 | 0.302 | 0.334, 0.332, 0.324 (0.016) | 3.83 / 3.83 | 0.008 / 0.008 | 0.956 / 0.956 | 0.169 / 0.169 | 111 / 111 | 0.966 / 0.966 | 0.9995 / 0.9995 | 3/3, 3/3 | 0.302 (L9, ρ1=0.96) | none |
| dclm-baseline-25p-dolma1.7-75p | 8 / 8 | 0.463 | 0.478, 0.473, 0.470 (0.009) | 8.55 / 8.55 | 0.007 / 0.007 | 0.943 / 0.943 | 0.197 / 0.197 | 111 / 111 | 0.928 / 0.928 | 0.9987 / 0.9987 | 3/3, 3/3 | 0.463 (L8, ρ1=0.94) | none |
| dclm-baseline-50p-dolma1.7-50p | 8 / 8 | 0.382 | 0.396, 0.393, 0.391 (0.006) | 7.46 / 7.46 | 0.009 / 0.009 | 0.954 / 0.954 | 0.277 / 0.277 | 111 / 111 | 0.959 / 0.959 | 0.9991 / 0.9991 | 3/3, 3/3 | 0.382 (L8, ρ1=0.95) | none |
| dclm-baseline-75p-dolma1.7-25p | 10 / 10 | 0.366 | 0.394, 0.394, 0.380 (0.021) | 8.11 / 8.11 | 0.000 / 0.000 | 0.955 / 0.955 | 0.214 / 0.214 | 111 / 111 | 0.960 / 0.960 | 0.9990 / 0.9990 | 3/3, 3/3 | 0.377 (L2, ρ1=0.89) | none |
## 3. Results per criterion (with evidence)

Ranges over the 25 models (from `14_work/models.csv`):

| quantity | at L_E01: min (model) - max (model) | at L_E06: min - max |
|---|---|---|
| λ1/λ2 (min over seeds) | 2.76 (dclm-baseline-qc-fw-3p) - 10.54 (dolma1_7-no-reddit) | **1.48 (dclm-baseline-qc-fw-3p, L2)** - 10.54 |
| CV of EV1 over seeds | 0.0004 - 0.038 (dclm-baseline-qc-7p-fw3) | 0.0004 - 0.028 |
| abs ρ(n, PC1) | 0.895 (c4) - 0.960 (dolma1_7) | 0.912 (falcon-and-cc-qc-tulu-10p) - 0.983 |
| max abs ρ(n, PC2..5) | 0.076 - 0.400 (c4) | 0.069 - 0.397 (fineweb-pro) |
| argmax PC | "111" for all 25 | "111" for all 25 |
| R² share of PC1 | 0.854 (dclm-baseline-qc-7p-fw3) - 0.994 | 0.871 - 0.995 |
| min cross-seed corr of 10 group means | 0.9977 - 0.9998 | 0.9965 - 0.9998 |
| decade means monotone | 3/3 seeds for all 25 | 3/3 for all 25 |
| E01 CV of EV1 over selection seeds 42-44 at L_E01 | max 0.050 | - |

* **P1 (artefact domination): no model flagged.** The highest EV1 at **any** layer of any model is 0.527 (E01, dolma1_7-no-reddit L8) / 0.518 (E06, same model and layer), and at that point abs ρ1 = 0.93: the largest-variance direction anywhere in any network is the number axis. No EV1 spike, no EV1 near 1, and nothing that looks like massive-activation domination. The only "high-EV, no-ρ" layer is **layer 1** (e.g. falcon EV1 0.293 with abs ρ1 0.034; c4 0.274 / 0.048; fineweb-pro 0.269 / 0.016; `14_work/layers.csv`). This is a format/token-identity direction right after the embedding, which is why the rule uses √(EV·abs ρ) and not EV. Layer 1 is never selected (selected layers: E01 3-10, E06 2-10). For c4 the E01 EV1 peak is at layer 1 (0.31), but the √(EV·abs ρ) rule correctly rejects it.
* **P2 (number axis not PC1): no model flagged.** At both L_E01 and L_E06, PC1 has the largest abs ρ with n in every seed for all 25 models. PC2 has a moderate abs ρ in some models (up to 0.40: c4 at L7, fineweb-pro at L7). This is the expected second-order (arch/"horseshoe") component of a curved number manifold, which is monotone-ish in n, not a competing number axis. Even there, PC1 has abs ρ ≥ 0.89 and carries ≥ 0.86 of the decodable signal.
  - Layers where the number axis is **not** PC1 do exist, but only outside the selectable band. Counts of (model, layer) cells with median max_{k≥2} abs ρ_k > abs ρ1: layer 1: 21 models; layer 12: 3; layer 13: 15; layer 14: 22; layer 15: 19; layer 16: 8; layers 2-11: **0**. In layers 13-16 the spectrum is flat (median λ1/λ2 = 1.22 vs 4.14 in layers 3-10), the number signal is spread over PCs 2-5 (median abs ρ1 0.30-0.57), and the argmax PC changes between seeds, e.g. dolma1_7 L13 "321" and dolma1_6plus L14 "424". This is a real property of late layers (prediction-oriented, not magnitude-organised) and does not touch any β estimate.
* **P3 (signal spread): no model flagged** (min 0.854, dclm-baseline-qc-7p-fw3 at L9; c4 at L7 0.862).
* **P4 (near-degenerate eigenvalues): no model flagged at L_E01** (min 2.76). **At L_E06, dclm-baseline-qc-fw-3p (layer 2) has min λ1/λ2 = 1.48**, just under 1.5 (EV1 0.218 vs EV2 0.139). The swap risk does **not** materialise: argmax-PC "111", abs ρ(PC2) = 0.08, abs ρ1 = 0.94, group-mean cross-seed corr 0.9988, CV(EV1) 0.018. Status: **watch only, not an exclusion**. It is also another sign that layer 2 is a transitional, less well-conditioned layer, which matters for the separate question of E06's layer switches to layers 2-3 (owned by the layer-selection audit). falcon-and-cc-qc-tulu-10p at L_E06 = 2 has λ1/λ2 = 3.79, which is fine.
* **P5 (EV instability): no model flagged.** The worst seed CV at a selected layer is 0.038 (E06) / 0.050 (E01). Across all layers the largest E06 CV is 0.099 (dclm-baseline-50p-dolma1.7-50p, L1) and the largest E01 CV is 0.129 (dclm-baseline-qc-20p, L1). Both are at layer 1, below threshold and irrelevant to β.
* **P6 (sign / axis swap between seeds): no model flagged.** Min cross-seed corr of the 10 PC1 group means is 0.9965. Decade means are strictly increasing in all 3 seeds for all models at both layers, so E01's abs(diff) never folds a non-monotone profile at these layers. E01 cannot show sign flips at all, because ρ is stored as abs and β uses abs gaps, so sign is irrelevant to its numbers.
* **Cross-experiment consistency of the spectrum:** E01 EV1 at L_E01 (seeds 42-44, 160 prompts) exceeds E06 EV1 at the same layer (1000 prompts) by a median of +0.013 (range −0.017 to +0.039). This is expected: E01 has 4 widely separated groups rather than 10, and its smaller sample inflates the top eigenvalue. Every model keeps the same EV ordering, so it is not a defect.

## 4. Validity verdict per flag

| flag | models | measurement-validity problem? | action |
|---|---|---|---|
| P1-P3, P5, P6 | none of 25 | - | nothing to exclude |
| P4 (λ1/λ2 = 1.48 < 1.5) | dclm-baseline-qc-fw-3p, **only at the E06-selected layer 2** (not at E01's layer 9) | potential, but checked directly and not realised (no swap, PC1 = number axis in all seeds, stable profile) | do not exclude; report as a conditioning caveat if E06 new-layer β is used |
| late-layer (13-16) and layer-1 PCA disorder | all models | genuine property of the networks; never at a selected layer | report, do not exclude; supports restricting selection to layers 2-11 |

**Honest bottom line:** on PCA health alone, **no model should be excluded**. In all 25 models, at the layers used for β (both E01 and E06 choices), PC1 is the number axis in every seed. It explains 22-51 % of the variance with no sign of outlier or massive-activation domination, is well separated from PC2 (λ1/λ2 ≥ 2.76 at E01 layers), and is reproducible across prompt seeds (EV CV ≤ 0.05, group-mean profile corr ≥ 0.996). The only near-threshold item is dclm-baseline-qc-fw-3p's E06 layer 2 (λ1/λ2 1.48), a caveat rather than a validity failure.
