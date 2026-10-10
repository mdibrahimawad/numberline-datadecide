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
  fixed before seeing Paloma. Secondary: `beta_coarse` at layer 8; both at the mean
  over layers 6–10; and both at each model's own selected layer (joint EV·|ρ| rule), so
  the result is shown with and without a fixed layer.
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

## Facts found after pre-registration (do not change the plan above)
- Models: `allenai/paloma-1b-baseline-{c4,mc4,redpajama,pile,falcon-refinedweb,dolma}`.
- Training configs (OLMo commit `1f2f020`) are saved in `configs/paloma/`: data seed **6198**,
  batch 2048 × 2048 tokens, 35,000 steps ≈ 146.8B tokens, full lists of tokenized training
  files. So an **exact** reproduction of each training stream (as in E04) is possible if those
  files are downloadable; otherwise the random-sample plan applies.
- `runpod_jobs/beta.py` now names outputs by slug, so full repo ids work
  (`tests/test_runpod_beta.py`).
- The models are gated (accept the terms on each model page; done 2026-10-10). Final checkpoint
  revision: **`step35000-unsharded`** for all 6 (also step5000 … step30000 every 5k steps).
- The checkpoints are in the 2023 OLMo format (`model_type: "olmo"`, `pytorch_model.bin`).
  transformers 4.49 would read that as its native Olmo class and build a default-sized model
  with random weights; `src/geometry.py` now loads them with ai2-olmo's `hf_olmo` classes and
  refuses to run if any weight fails to load (`tests/test_legacy_olmo_loading.py`). The beta
  prefetch no longer skips `*.bin`.
- Training data: only Dolma's tokenized files are public (`olmo-data.org`, HTTP 200); the five
  decontaminated corpora return 404, so they use the random-sample plan (a stated deviation).

- How the counts are made (built 2026-10-10, before any Paloma β was measured):
  - **Dolma: exact.** `runpod_jobs/exact_alpha.py` with `configs/paloma/paloma_dolma_exact_map.json`:
    the 249 files from olmo-data.org, seed 6198, 71,680,000 sequences of 2,048 tokens, EOS 0,
    decoded with the tokenizer the files were made with (`allenai/gpt-neox-olmo-dolma-v1_5`).
    Split into 8 file-disjoint parts that share one training order, so several pods share it;
    the parts are added up and must hold exactly 146,800,640,000 tokens.
  - **The other five: random sample** (`runpod_jobs/corpus_sample.py`,
    `configs/paloma/corpus_sources.json`): files in a seeded random order (seed 6198), within
    each stratum (RedPajama's subsets) in proportion to its estimated tokens, until 146.8B
    tokens; the last file counts with the fraction needed. Tokens are estimated by tokenizing
    every 50th document (+1 EOS per document). Uncompressed .jsonl files over 2 GB are read
    as 1 GB pieces (lines starting in each piece), so they are sampled in random pieces too.
  - **Deviations (state them in the paper):** these are the public releases, not Paloma's
    decontaminated training versions (removed documents are a tiny fraction); RedPajama
    lacks its 'book' subset (no longer distributed) and the few files its server now refuses (listed in `summary.json` → `unavailable_files`; the run stops if more than 5 % of a subset is missing); the Pile is `pile-uncopyrighted` (no
    Books3 and a few other sets); RefinedWeb is the whole public release (Paloma held out
    5 %). The sample is random, not the model's exact order; between data seeds E04 found R
    to move by ±2 (c4) and ±17 (dolma1_7).

- **RedPajama on hold (2026-10-10):** 536 of the 859 common_crawl files in RedPajama's URL list
  are refused by data.together.xyz (the other subsets are served). The other five counts run
  first; RedPajama gets a decision later (sample the remaining common_crawl files with the
  official subset shares, use SlimPajama as a stated substitute, or test on 5 models).

## Steps
1. **List the models** (laptop):
   `python -c "from huggingface_hub import list_models; [print(m.id) for m in list_models(author='allenai', search='paloma')]"`
   and their revisions (`list_repo_refs`).
2. **Check whether the training data is reproducible:** Paloma's training config (OLMo commit
   `1f2f020`), tokenized files, seed.
3. **β** (GPU, ~10 min, < $0.20): `python -m runpod_jobs.beta --fine --models <6 repo ids> …`
   (`src/beta_fine.py` accepts full repo ids).
4. **Counts** (CPU pods, roughly $5–15, ~1 h): `bash runpod_jobs/paloma_counts.sh <corpus>`
   on one pod per corpus (3 pods for Dolma), see `docs/runpod.md` § 4b; then
   `python -m runpod_jobs.fetch_results --kind paloma`.
4b. **Write the predictions down before measuring β:** `python -m src.paloma_predictions`
   → `results/paloma_counts/predictions.{csv,json}` (frozen lines × each corpus's α_OLS, S, R;
   flags predictors outside the DataDecide range), committed before step 3 runs.
5. **Analysis:** `src/paloma_analysis.py`, implementing exactly the criteria above.

## Context for reading the result
- Measured at fixed layers, α, S and R predict DataDecide β about equally well
  (r ≈ 0.6–0.75 at layers 6–11; see `python -m src.fixed_layer_beta`). DataDecide alone can't
  tell them apart. This test can.
- n = 6 is small: the ordering test has little power, so the accuracy and amount-effect
  checks carry most of the information.
