# Code map

Every tracked file and folder, which experiment it belongs to, and its status.
**Active** = used in the DataDecide work (E01–E07) and covered by tests or a recent run.
**Prior work** = from before the DataDecide phase (E00); kept working but not exercised
since. Experiment folders: `experiments/E##_*/README.md`.

## Layout

```
README.md            start here
docs/                project-wide documentation (this file, findings, infrastructure, …)
experiments/         one folder per experiment: README (summary) + DETAILS (full write-up)
configs/             DataDecide model list and recipe → training-file map
src/                 library + local CLIs (analysis, geometry probe, samplers)
modal_app/           Modal cloud entrypoints (E00–E04)
runpod_jobs/         RunPod runners (E04 last 11 recipes, E06)
utils/               small helpers used by src/ (prompts, regex, dataset streaming, MLflow)
tests/               offline tests (python -m pytest -q tests)
results/             committed outputs, one subfolder per experiment (see results/README.md)
reference/           the original paper's code, unchanged copies (read-only reference)
paper/               LaTeX appendix + figures from the prior-work write-up
```

Python packages (`src`, `modal_app`, `runpod_jobs`, `utils`) stay at the top level so
every documented `python -m …` / `modal run …` command and every import keeps working.

## configs/

| File | Purpose | Exp |
|---|---|---|
| `datadecide_models.json` | the 25 recipe slugs, model repo template `allenai/DataDecide-{recipe}-1B`, revision `step69369-seed-default` | all |
| `datadecide_data_map.json` | recipe → ordered list of training `.npy` files (duplicates kept) + constants; rebuilt by `python -m src.datadecide_sampling` | E03, E04 |
| `paloma/` | the 6 Paloma baseline training configs (OLMo commit 1f2f020) + `paloma_data_map.json` (files, seed, batch, steps) | E09 |

## src/ — active

| File | Purpose | Exp |
|---|---|---|
| `geometry.py` | the number-line probe: prompts, correct-output filter, hidden states, PCA/PLS, EV/ρ/β per layer | E00, E01 |
| `spacing_fit.py` | β estimators from group means (direct geometric fit, log fit) | E01 |
| `datadecide_sweep.py` | per-model protocol: layer selection on seeds 42–44 (√(EV·\|ρ\|)), frozen evaluation on 45–47; `select_joint_layer` | E01, E06 |
| `frozen_layer_evaluation.py` | evaluate a fixed layer on unseen prompt seeds | E01 |
| `sampling_validation.py` | α_MLE, bootstrap, sampling schemes vs full pass | E02 |
| `datadecide_sampling.py` | DataDecide training-order reproduction: data map, window sampler, chunk shuffle, seed membership | E03, E04 |
| `join_alpha_beta.py` | join α (exact, else window) with β → `results/datadecide/alpha_beta.*` | E05 |
| `confounder_analysis.py` | number density, code/math share, partial and within-family correlations | E05 |
| `decade_metrics.py` | data-side "β" from decade counts: infomax, power q, saturation K; also defines `FAMILY` | E05 |
| `beta_fine.py` | 10-group batched β, saves per-prompt PCA scores | E06 |
| `beta_fine_analysis.py` | fine β vs exact α / numbers seen → `results/beta_fine/fine_vs_data.csv` | E06 |
| `count_features.py` | 26 count-shape features incl. R (`F14_n_seen_lt_4000`) → `results/datadecide/count_features.csv` | E07 |
| `fixed_layer_beta.py` | β at every fixed layer + the frozen DataDecide lines for E09 → `results/beta_fine/{fixed_layer_beta.csv,frozen_lines.json}` | E06, E09 |

## src/ — prior work (E00)

| File | Purpose |
|---|---|
| `cli.py`, `multi_counter.py`, `number_analysis.py`, `pile_splitted_counter.py` | Stage 1: count integers in a corpus stream (The Pile), log to MLflow, frequency figures |
| `frequency_compare_analysis.py` | compare frequency tables across corpora (log-log fits, Zipf plots) |
| `geometry_cli.py`, `geometry_analysis.py`, `per_seed_geometry.py`, `heldout_geometry.py`, `joint_layer_selection_figure.py`, `normal_pca_layer_grid.py`, `all_models_layer_ci_figures.py` | Stage 2: run the probe locally with MLflow logging; layer-selection variants; figures |
| `robust_geometry.py`, `robust_geometry_analysis.py` | robust-PCA (Principal Component Pursuit) check |
| `manifold_geometry.py` | is the number trajectory straight or curved in full hidden space |
| `circuits.py`, `circuits_analysis.py` | Stage 3: attention-head / MLP attribution to the PC1 direction (TransformerLens) |
| `number_comparison*.py` (4 files) | "which is larger?" comparison probe, few-shot bootstrap, group-gap analysis, publication figures (`paper/appendix_number_comparison.tex`) |
| `motivation_bias.py`, `motivation_scale.py`, `comparison_motivation_analysis.py`, `family_comparison_motivation.py`, `domain_group_comparison_figure.py`, `candidate_screen_analysis.py` | motivation experiments relating number-line geometry to downstream numeric behaviour across model families |
| `alpha_beta_analysis.py`, `paper_figures.py`, `paper_magnitude.py` | earlier α–β analysis across public model families and paper figures |

## modal_app/

| File | Purpose | Exp | Status |
|---|---|---|---|
| `datadecide_app.py` | β sweep of the 25 DataDecide models on Modal GPUs | E01 | active |
| `sampling_validation_app.py` | sampling validation on SlimPajama / LLM-JP | E02 | active (LLM-JP part never run) |
| `datadecide_alpha_app.py` | `::validate`, `::sweep` (window α), `::exact_samples`, `::exact_all` (exact 100B α) | E03, E04 | active |
| `corpus_alpha_app.py` | **defines `_count_text` and `_fit_alpha`**, the counting rule used everywhere | all data-side | active (as a library) |
| `corpus_alpha_full_app.py` | full-pass counts of SlimPajama and LLM-JP (ground truth for E02) | E02 | prior work, results committed |
| `app.py`, `aggregate_local.py`, `dolma_app.py`, `redpajama_app.py`, `refinedweb_app.py`, `stack_app.py`, `pile_splitted_app.py` | Stage 1 corpus counts on Modal | E00 | prior work |
| `geometry_app.py`, `robust_geometry_app.py`, `manifold_geometry_app.py`, `circuits_app.py`, `natural_log_app.py`, `number_comparison_app.py`, `motivation_app.py`, `motivation_bias_app.py`, `paper_magnitude_app.py`, `run_paper_bins_manifold_all.sh` | Stage 2/3 and probes on Modal GPUs | E00 | prior work |
| `README.md` | the original Pile-sweep guide | E00 | |

## runpod_jobs/ (active)

| File | Purpose | Exp |
|---|---|---|
| `setup.sh {cpu,gpu}` | one-shot pod setup: tools, Python ≥ 3.10, venv in /workspace, pinned packages | E04, E06 |
| `exact_alpha.py` | exact 100B α on CPU pods: training order, HTTP range reads, decode, count; HF upload, claims, self-delete | E04 |
| `beta.py` | β of many models on one GPU pod (`--fine` = E06), prefetch, upload, self-delete | E06 (E01 protocol also possible) |
| `fetch_results.py` | download pod results from the private HF datasets to `results/` | E04, E06 |
| `pod.py` | stop / terminate the current pod safely | E04, E06 |

## utils/

`prompts.py` (numeral / symbol prompt generators), `text.py` (digit regex), `dataset.py`
(HF streaming loader), `mlflow_utils.py` (MLflow helpers). Used by `src/geometry.py`
and the E00 counters.

## tests/

| File | Covers |
|---|---|
| `test_filter_correct.py` | correct-output filter with a fake model and a tiny random OLMo |
| `test_datadecide_smoke.py` | **online**: downloads `allenai/DataDecide-dolma1_7-60M`; fails without Hugging Face access |
| `test_sampling_validation.py` | α_MLE, sampling schemes on tiny fake parquet / jsonl.gz |
| `test_datadecide_sampling.py` | training-order reproduction on tiny fake `.npy` files |
| `test_runpod_exact_alpha.py` | the RunPod runner end to end with a fake HF (claims, crash/resume, upload, self-delete) |
| `test_decade_metrics.py` | decade metrics |
| `test_beta_fine.py` | 10-group β: bands, fits recover known β, batching = one-by-one, full run, R² blind spot |
| `test_count_features.py` | the locked definition of R and the count features |
| `test_fixed_layer_beta.py` | fixed-layer table, S/R definitions, frozen-line fit |
| `test_runpod_beta.py` | the GPU runner end to end with a fake worker and fake HF (full repo ids -> slug file names, upload, self-delete) |
