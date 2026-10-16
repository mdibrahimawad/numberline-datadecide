# results/

Committed outputs. Each folder belongs to one experiment (`experiments/E##_*`). The
results are small (≈ 50 MB) and kept in git so every analysis can be re-run with no cloud
access. Raw training data and model weights are **not** stored here.

| Folder / file | Experiment | Contents | Written by |
|---|---|---|---|
| `corpus_alpha_full/` | E00 → E02 | full-pass integer counts of SlimPajama-627B (reupload) and LLM-JP v3: `counts_0_to_10000.csv`, `summary.json` | `modal_app/corpus_alpha_full_app.py` |
| `sampling_validation/ground_truth.json` | E02 | α_OLS / α_MLE recomputed from the full-pass counts | `python -m src.sampling_validation` |
| `corpus_alpha_datadecide/<recipe>/` | E03 | ~1B-token window sample: `counts_0_to_10000_{drop_partial_docs,keep_partial}.csv`, `mixture.json` (source shares), `windows.json`, `summary.json` | `datadecide_alpha_app.py::sweep` |
| `corpus_alpha_datadecide/alpha_summary.csv` | E03 | window α for all 25 recipes (α_OLS biased; use α_MLE) | same |
| `corpus_alpha_datadecide/exact_100b_<recipe>/` | E04 | exact 100B training stream: `counts_seed_2.csv` (c(n), n = 0..10000), `summary.json`; c4/dolma1_7 also have other seeds, `counts_full_recipe.csv` / `counts_union_of_samples.csv` | `::exact_all`, `::exact_samples`, `runpod_jobs/exact_alpha.py` |
| `corpus_alpha_datadecide/alpha_seed2.csv` | E04 | **the α table**: one row per recipe (α_OLS, R², α_MLE, numbers seen, tokens) | rebuilt by `::exact_all --dry-run` or `runpod_jobs.fetch_results` |
| `datadecide/<recipe>.json`, `datadecide/summary.csv` | E01 | β, paper protocol: per-layer metrics, frozen-layer evaluation, filter stats | `modal_app/datadecide_app.py` |
| `datadecide/alpha_beta.{csv,png}`, `alpha_beta_correlations.json` | E05 | α joined with β, correlations | `python -m src.join_alpha_beta` |
| `datadecide/confounders.json` | E05 | number density, code/math share, partial and within-family correlations | `python -m src.confounder_analysis` |
| `datadecide/decade_metrics.csv` | E05 | data-side decade metrics (infomax, power q, saturation K) | `python -m src.decade_metrics` |
| `datadecide/count_features.csv` | E07 | 26 count features (incl. R = `F14_n_seen_lt_4000`), α, β, residual | `python -m src.count_features` |
| `beta_fine/step69369-seed-default/<recipe>.{json,npz}`, `summary.csv` | E06 | 10-group β; `.npz` = per-prompt targets, groups and top-5 PCA scores at every layer, per seed | `runpod_jobs/beta.py --fine` |
| `beta_fine/fixed_layer_beta.csv`, `beta_fine/frozen_lines.json` | E06, E09 | β at every fixed layer with α, S, R; the frozen DataDecide calibration lines (layer 8) used by E09 | `python -m src.fixed_layer_beta` |
| `paloma_counts/<corpus>/counts.csv`, `summary.json` | E09 | c(n) for the 6 Paloma training corpora at the training budget (146.8B tokens), with α, S and R (K = 4000); 5 random samples + Dolma's exact stream | `runpod_jobs/paloma_counts.sh`, then `runpod_jobs.fetch_results --kind paloma` |
| `e10_explore/` | E10 | stage A exploration on Crystal vectors: `A_profiles.csv` (41 barcode features per group × layer × subsample), `B_tier_test_*.csv`, `C_classifier_*.csv`, `D_per_number.csv`, `E_layer_change.csv`, `F_special.csv`, `matched_tiers.csv` | `python -m src.e10_explore --vectors <dir>` |
| `e10_explore/figures/` | E10, E11 | explanatory figures (gaps, where z comes from, special numbers, years, length vs direction, per-model mean gap raw vs length 1, reference shapes) | `python -m src.e10_explain_figures --vectors <dir> --only …` |
| `e10_confirm/<model>/` | E10 B-A, F | pre-registered P1–P5 per model (`summary.json` + tables) | `python -m src.e10_confirm` |
| `e10_topology/` | E10 C | depth profile, digit circle, years, RTD | `python -m src.e10_topology` |
| `e10_convergence/`, `e10_convergence_d2/` | E10 D, D2 | RTD / shuffled RTD per model pair and depth; D2 adds RedPajama-3B and Llama-3.2-1B, with per-depth matrices (`fig_rtd_depth*.png`) | `python -m src.e10_convergence`, `python -m src.e10_d2_figure` |
| `e10_graded/` | E10 E | 3/5/10-level features, dose-response, classifiers, SHAP, figures | `python -m src.e10_graded`, `src.e10_graded_analysis`, `src.e10_figures` |
| `e11_tier0/` | E11 | one CSV per hypothesis (H1–H29), `LID_per_number.csv` | `python -m src.e11_tier0`, `experiments/E11_topology_hypotheses/scratch/lid.py` |
| `e11_ckpt/` | E11 checkpoints | C1/C2/secondary summaries, per-checkpoint neighbourhoods and crowding, features, Wasserstein matrices, `fig_trajectories.png`, LaTeX tables | `python -m src.e11_ckpt_analysis`, `src.e11_ckpt_figure` |
| `e11_shape_search/` | E11 shape search | `discovery.csv`, `locked.json`, `test.csv`, per-cell cache, `fig_mod_clusters.png` | `python -m src.e11_shape_search --phase discover/test` |
| `beta_fine/fine_vs_data.csv` | E06 | fine β joined with exact α, numbers seen, E01 β, family | `python -m src.beta_fine_analysis` |

## Conventions
- Recipe slugs are those in `configs/datadecide_models.json`.
- All counts use the same rule (`_count_text`: `\b\d+\b`, leading zeros stripped, 0..10000).
- "seed 2" = the data-order seed of the released `seed-default` models (see E04).
- Re-running an analysis command overwrites its output file. Commit the change together
  with the doc update that cites it.

## Not here
- E02's SlimPajama run outputs (`sampling_validation/slimpajama/`) are on the owner's
  laptop and still need to be committed.
- E00's MLflow store (`mlflow.db`) and Modal volumes.
- Hugging Face copies of the RunPod outputs: `mdibrahimawad/numberline-alpha-results` and
  `mdibrahimawad/numberline-beta-results` (private; backups of folders above).
