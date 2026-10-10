# E09: Paloma out-of-sample test (pre-registered)

**Question:** do formulas fitted only on the 25 DataDecide models predict number-line
compression β in 6 **new** models they never saw?
**Status:** **planned, not run.** Everything under "Locked" was written before any Paloma
data was collected. It must not change after the first Paloma β or count is computed.
Results go in a separate `RESULTS.md`, with every deviation from this plan stated.
**Written:** 2026-10-10.

## Why Paloma
The 6 Paloma baselines (Ai2) are 1B models trained with the same OLMo training code, the
same tokenizer (`allenai/gpt-neox-olmo-dolma-v1_5`) and 16 layers like DataDecide.
Each was trained on one corpus: C4, mC4-en, Falcon RefinedWeb, RedPajama, The Pile, Dolma.
They trained for **~150B tokens** instead of DataDecide's 100B. That lets the test separate
two kinds of formula:
- **α (shape only)** predicts that the extra data changes nothing;
- **S and R (shape + amount)** predict that more data means fewer rarely-seen numbers,
  hence less compression.

## Locked

### Frozen predictors (fitted on DataDecide only; `results/beta_fine/frozen_lines.json`)
Three competing formulas, all frozen. The test decides between them. No re-fitting on Paloma.

| Target (DataDecide, layer 8) | Predictor | β̂ = intercept + slope × predictor | DataDecide r | DataDecide LOO MAE |
|---|---|---|---|---|
| **β_cont (primary)** | α_OLS | 3.0752 + 1.469087 × α | +0.75 | 0.101 |
| | S | −0.1584 + 0.000203564 × S | +0.62 | 0.135 |
| | R | 2.5905 − 0.000264359 × R | −0.70 | 0.113 |
| β_coarse (secondary) | α_OLS | 2.5426 + 1.107539 × α | +0.72 | 0.087 |
| | S | 0.0240 + 0.000172196 × S | +0.67 | 0.098 |
| | R | 2.3473 − 0.000223352 × R | −0.76 | 0.080 |

- **S** = Σ_{n=10}^{9999} c(n)/(c(n)+K), **R** = #{n ∈ 10..9999 : c(n) < K}, with **K = 4000**
  (`RARE_K`, never re-tuned). **α_OLS** = least-squares slope of log c(n) vs log n,
  n = 1..10000, c(n) > 0.
- c(n) = count of integer n (`_count_text`) in that model's training data.

### β measurement
- Protocol E06 (`src/beta_fine.py`, unchanged): 10 groups, 100 prompts per group,
  prompt seeds 45–47, correct-output filter.
- **Primary β = `beta_cont` at fixed layer 8** (median over the 3 seeds).
  Layer 8 is the layer the joint EV·|ρ| rule selects most often on DataDecide; it was
  fixed before seeing Paloma. Secondary: `beta_coarse` at layer 8, and both at the mean
  over layers 6–10.
- Checkpoint: the final released ~150B-token checkpoint of each model.
- Validity checks (as in E08): acceptance ≥ 80 % per group, median |ρ| ≥ 0.9 at layer 8,
  EV ≥ 0.1, `err_coarse` ≤ 0.4. A model failing a check is reported and excluded, never
  dropped silently.

### Counts
- For each corpus, a **uniform random sample of 150B tokens** (tokenized with
  `allenai/gpt-neox-olmo-dolma-v1_5`), the same size as the model's training data,
  counted with `_count_text`. **No scaling.**
- Use Paloma's decontaminated version of each corpus if it is public; otherwise the
  public release of the same corpus version, noted as a deviation.
- If Paloma's exact training order turns out to be reproducible (tokenized files + config +
  seed), the exact stream is used instead, and that is noted.

### Success criteria (per predictor, on the 6 Paloma models)
1. **Ordering:** Spearman ρ between predicted and measured β. Reported with all 6 points;
   with n = 6, ρ ≥ 0.83 is needed for one-sided p < 0.05 (≥ 0.89 two-sided).
2. **Accuracy:** mean absolute error of the frozen prediction. "Transfers" if MAE ≤ 1.5 × the
   predictor's DataDecide LOO MAE (α 0.152, S 0.203, R 0.170 for β_cont).
3. **Which formula wins:** the lowest MAE on primary β wins. Differences below 0.02 count as
   a tie.
4. **Amount effect (the discriminating check):** mean measured β of the Paloma models vs the
   mean predicted by the α line. S/R predict Paloma β **above** the α prediction (more
   data, less compression). Reported as the mean residual of the α prediction, with its
   sign.

Secondary analyses (labelled as such): a shift-corrected MAE (subtract the mean
residual, in case Paloma's unfinished learning-rate schedule moves every β by the same
amount), the β_coarse and layer-6–10 versions, and α_MLE.

## Steps
1. **List the models** (laptop):
   `python -c "from huggingface_hub import list_models; [print(m.id) for m in list_models(author='allenai', search='paloma')]"`
   and their revisions (`list_repo_refs`).
2. **Check whether the training data is reproducible:** Paloma's training config (OLMo commit
   `1f2f020`), tokenized files, seed.
3. **β** (GPU, ~10 min, < $0.20): `python -m runpod_jobs.beta --fine --models <6 repo ids> …`
   (`src/beta_fine.py` accepts full repo ids).
4. **Counts** (CPU pods, roughly $5–15): a new runner for "first 150B tokens of a uniform
   shuffle" of each corpus, with tests, built after step 2.
5. **Analysis:** `src/paloma_analysis.py`, implementing exactly the criteria above.

## Context for reading the result
- Measured at fixed layers, α, S and R predict DataDecide β about equally well
  (r ≈ 0.6–0.75 at layers 6–11; see `python -m src.fixed_layer_beta`). DataDecide alone can't
  tell them apart. This test can.
- n = 6 is small: the ordering test has little power, so the accuracy and amount-effect
  checks carry most of the information.
