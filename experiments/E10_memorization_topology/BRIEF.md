# Experiment brief: forecasting LLM memorization early in pretraining with topological features

**For:** Claude Code on a RunPod GPU pod (one A100/H100 80 GB is enough; ask for ≥ 32 vCPUs and ≥ 300 GB of volume disk).
**Goal of this run:** credible preliminary evidence, strong enough to show a professor, that topological features of hidden states at an early checkpoint predict which training sequences the model will memorize later, **beyond** the strongest simple baselines.

Read the whole file before writing code. §1–3 are context. §4–9 are the specification. §10–12 are how to execute and report. Where this brief states a fact about a dataset, repo or file layout, **verify it** before relying on it, and record what you found in `REPORT.md`.

---

## 1. Background

### 1.1 The task and its known weakness
**Biderman et al., "Emergent and Predictable Memorization in Large Language Models" (NeurIPS 2023; arXiv 2304.11158).**
- Memorization = **32-extractability**: give the model the first 32 tokens of a training sequence; it is memorized if the greedy 32-token continuation matches the true next 32 tokens exactly.
- Evaluated on Pythia over all ~146M training sequences (each training sequence is 2049 tokens; only the first 64 are used).
- **Forecasting from an early checkpoint of the same model fails on recall.** For Pythia-12B, using "memorized at the checkpoint after 23M sequences" (≈ step 22.5k) to predict "memorized at the end": precision 0.918, recall 0.500. Recall only becomes high (0.916) at the 126M-sequence checkpoint, i.e. after most of the compute. Pythia-6.9B: precision 0.919, recall 0.513 at 23M. Their conclusion: recall is low for all predictors that use much less compute than the final model.
- Labels released: Hugging Face dataset **`EleutherAI/pythia-memorized-evals`** (MIT). One split per model (e.g. `deduped.1b`, `deduped.410m`, …, plus duped variants and 13 intermediate-checkpoint splits for 12B). Columns: `index` (position of the sequence in the training order, 0 … ~146M), `tokens` (the 64 tokens). Only **memorized** sequences are listed. Code: `EleutherAI/pythia`, folder `predictable-memorization/`.

### 1.2 Memories that appear later without re-exposure
**"Uncovering Latent Memories in Large Language Models" (ICLR 2025; arXiv 2406.14549).**
- Pythia-1B, checkpoints every 3k steps from 10k to 43k (also Amber-7B). Memorization metric: kl-Levenshtein distance with 32-token context and 64-token target.
- Some sequences **seen once** that were not memorized at step 10k **became memorized by step 19k with no further exposure** ("latent memorization").
- Adding small Gaussian noise to the weights (scale 2×10⁻³, 200 perturbations) recovers latent memories.
- **Diagnostic:** latent-memorized sequences have much lower cross-entropy than unmemorized or unseen ones. → This cross-entropy diagnostic is a **baseline we must beat.**
- Data statistics: memorization probability rises log-linearly with repeat count and with compressibility ("z-complexity").
- No public code found.

### 1.3 Hidden states carry memorization information (final models only)
**"A Multi-Perspective Analysis of Memorization in Large Language Models" (EMNLP 2024).**
- Pythia 70M–12B, deduped Pile, final models only (no checkpoints).
- A small transformer on last-layer embeddings + entropy + unigram frequency predicts memorization: **token-level accuracy ~78–82%, sequence-level only ~5–11%.**
- Sequences with different memorization scores **cluster in embedding space** — a geometric signal.
- "Boundary effect": at the first generated token, unigram frequency dips for memorized sequences; entropy shows the inverse.

### 1.4 The newest competitor (preprint)
**"Don't Forget! Decomposing the Training Dynamics of Memorization in Language Models" (arXiv 2609.34933, Sept 2026).**
- Deduped Pythia 160M–6.9B; main model **Pythia-1B**; checkpoints 0–130k.
- Parameter-gradient decomposition features (sensitivity, update magnitude, gradient–update alignment).
- Ridge classifier, 6-fold CV, **only 128 balanced examples**: 90.6% accuracy at step 10k with decomposition + cross-entropy vs **84.4% for cross-entropy alone**; peak 96.1% at step 40k. Pythia-6.9B: 88.3% vs 77.3%.
- Notes 10–20% of "memorized" sequences failed re-verification (numerical fragility of exact-match extractability).
- Uses parameter-space geometry, **no hidden-state geometry or topology**. No code found.

### 1.5 Memorization timing (useful for design)
**"Causal Estimation of Memorisation Profiles" (Lesci et al., arXiv 2406.04327; code `pietrolesci/memorisation-profiles`, data on HF collection `pietrolesci/memorisation-profiles`).**
- Uses Pythia data order + checkpoints with a difference-in-differences estimator.
- Memorisation is stronger and more persistent in larger models; **persistent memorisation plateaus after ~25k steps**; many batches are forgotten by the end of the first epoch.
- **They excluded Pythia-2.8B because of a possible mismatch between its checkpoints and the data order. Do not use 2.8B.**
- Their repo likely contains working loaders for Pythia's data order — read it before writing your own.

### 1.6 Topology methodology we copy
**Fay et al., "The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology" (ICLR 2026 Oral; arXiv 2505.20435).**
- Point cloud → Vietoris–Rips persistent homology (Euclidean, H0 and H1, Ripser) → **41-number barcode summary**: {mean, min, Q1, median, Q3, max, std} × {H0 deaths, H1 births, H1 deaths, H1 persistences, H1 birth/death ratios} = 35, plus {total persistence, number of bars, persistent entropy} for H0 and for H1 = 6.
- Feature pruning (drop one of any pair with |corr| > 0.5), logistic regression, SHAP. Compared against linear classifiers on raw activations.
- Finding: adversarial/backdoored inputs cause **"topological compression"**: higher mean H0 death, fewer H1 bars that are born later and persist longer, lower H1 entropy. They link this to memorization reducing the effective dimensionality of representations.

**Malhotra et al., "Tracking Representation Dynamics in Large Language Models with Persistent Homology" (arXiv 2606.19542; code `malhotranaman/tracking-representation-dynamics-with-persistent-homology`, MIT).**
- Topology tracked **across fine-tuning checkpoints**; same 41-feature summary (`src/phlatent/topology/summary.py`, `FEATURE_NAMES`; `topology/` and `analysis/` are numpy/scipy/sklearn only).
- **Topological velocity** v_t = Wasserstein distance between H1 barcodes at consecutive checkpoints.
- Controls: isotropic Gaussian clouds matched in n and d (they produce artifactual H1 bars); geometric baselines (centroid drift, total variance, mean pairwise distance); effect sizes (Hedges' g, Cliff's δ with bootstrap CIs); permutation nulls (≥ 2000).
- Finding: topological peaks **never lag** behavioural peaks and can lead them.

## 2. The gap

- Memorization forecasting has a documented failure (recall ≈ 0.5 early) on an A*-published benchmark with public labels.
- Hidden states are known to carry memorization information (EMNLP 2024), and memorization can be latent before it is visible (ICLR 2025).
- **Nobody has used hidden-state geometry or topology across pretraining checkpoints to forecast memorization.** The only checkpoint-based predictor beyond loss uses expensive per-sequence gradients on 128 examples.
- The topology literature links "compression" to memorization but has never tested it on memorization itself.

## 3. Aim and pre-registered hypotheses

**Aim:** For training sequences seen once early in pretraining, use features computed at an early checkpoint S_f to predict whether the sequence is memorized at a later target checkpoint T. Show whether topological features add predictive power beyond (a) loss-based signals, (b) data statistics, and (c) a supervised probe on the same hidden states.

Write these into `REPORT.md` **before** looking at real results:

- **H1 (main).** Adding topological features to the strongest baseline set raises AUROC on the **hard subset** (sequences *not* memorized at S_f) with a paired-bootstrap 95% CI for ΔAUROC excluding 0, at S_f = 10000, on the out-of-time test split.
- **H2 (direction).** Sequences that will become memorized show topological compression at S_f relative to matched non-memorized ones (fewer H1 bars, higher H1 persistence, lower H1 entropy, higher mean H0 death). Report effect sizes; direction is secondary to H1.
- **H3 (dynamics).** Per-sequence topological velocity between the checkpoints just before exposure and S_f is larger for future-memorized sequences.
- **H4 (learned, not text).** The effect in H1 survives adding the same topological features computed at **step 0** (random model) as a baseline — i.e. it is not just a property of the text.

**Pre-registered success criterion for "strong preliminary evidence":** H1 holds at S_f = 10000 **and** replicates at a second forecast checkpoint (S_f = 23000) **and** on a second model size (pythia-410m-deduped), with all controls in §8 clean. Anything less is reported as partial.

---

## 4. Models and checkpoints

| Role | Model | Notes |
|---|---|---|
| **Primary** | `EleutherAI/pythia-1b-deduped` | Same main model as "Don't Forget"; the non-deduped `pythia-1b` was trained in bf16 after an fp16 loss spike, the deduped one is standard |
| Replication | `EleutherAI/pythia-410m-deduped` | Same pipeline |
| Scale check (stage 2, optional) | `EleutherAI/pythia-6.9b-deduped` | Direct comparison to Biderman's 6.9B numbers |
| **Never use** | any `pythia-2.8b*`, any `*-v0` model | Data-order mismatch / inconsistent step labels |

Checkpoints: HF `revision="step{N}"`; available: 0, 1, 2, 4, …, 512, then every 1000 up to 143000. Load in **fp32**.

**Epoch boundary — verify first.** The deduped Pile (~207B tokens) is shorter than the 300B-token run, so the deduped models see roughly 1.5 epochs; sequences from early steps are seen **again** in the second epoch (expected start around step 98–99k; one step = 1024 sequences × 2049 tokens ≈ 2.1M tokens). **Determine the exact boundary from the data** (e.g. find the first step whose sequences duplicate step-0 sequences). Then:
- **Primary target T = the last multiple of 1000 before the second epoch starts** (expected ≈ 95000–98000). This measures memorization with **no re-exposure**, the clean version of the question.
- Secondary target T = 143000 (final; Biderman's definition; labels already released).

## 5. Data

### 5.1 Training-order tokens
- Source: HF dataset **`EleutherAI/pile-deduped-pythia-preshuffled`** (602 GB in total; sharded). Pythia README: download with `hf_hub_download`, shards merge with `utils/unshard_memmap.py`, and `utils/batch_viewer.py` reads batches. **Download only the shard(s) covering the steps you need** (see 5.2) — do not download 602 GB.
- Each training sequence is 2049 tokens; sequence index i belongs to step `i // 1024`.
- **Alignment check (mandatory):** for 200 indices listed in `pythia-memorized-evals` split `deduped.1b` within your step range, confirm that the first 64 tokens you read from the shard equal that dataset's `tokens`. If they don't, stop and fix the indexing. Read `pietrolesci/memorisation-profiles` for a working loader if needed.

### 5.2 Sequence pool
- **Pool = sequences trained in steps 1000–5999** (5.12M sequences; all seen once, before every forecast checkpoint).
- Sample **300,000** uniformly at random (seed 0) → the labelled pool.
- **Time split:** steps 1000–3999 → train/CV; steps 4000–5999 → out-of-time test. All headline numbers come from the out-of-time test.

### 5.3 Labels
For each pooled sequence, using the first 64 tokens (32 prompt + 32 continuation):
- `mem_T` = 32-extractable at the primary target T (compute yourself: greedy decode 32 tokens, batched, fp32 or bf16 if needed for speed — state which).
- `mem_final` = present in `pythia-memorized-evals` `deduped.1b` (no decoding needed). Sanity check: recompute on 2000 random pooled sequences at step 143000 and report agreement (expect high but not perfect; "Don't Forget" reports 10–20% fragility).
- `mem_Sf` = 32-extractable at each forecast checkpoint S_f (below).
- `match_len_Sf` = number of leading continuation tokens matched greedily at S_f (0–32).

### 5.4 Forecast checkpoints
`S_f ∈ {10000, 23000}` (10000 = "Don't Forget" comparison; 23000 ≈ Biderman's 23M-sequence checkpoint). Pilot uses 10000 only.

### 5.5 Evaluation sets
- **Full set:** all pooled sequences (natural prevalence, expected ~1%).
- **Hard subset (main):** `mem_Sf = 0`. Here Biderman's predictor has recall 0 by construction; this is where forecasting is actually needed.
- **Case–control feature set:** to keep feature extraction cheap, compute expensive features for **all positives** (`mem_T = 1`) plus **5× as many random negatives**, stratified by split. Report precision/recall at natural prevalence by **reweighting** negatives by (true negative count / sampled negative count). Cheap features (loss family, data stats) are computed on the full 300k.

## 6. Features

All features are computed at S_f unless stated. "Cloud" = the hidden states of the 64 tokens of one sequence at one layer (64 points in R^d).

### 6.1 Baseline feature families
- **B0 Biderman:** `mem_Sf` (binary).
- **B1 Extraction:** `match_len_Sf`; greedy-match fraction.
- **B2 Likelihood (latent-memories diagnostic):** mean cross-entropy of the 32 continuation tokens; also of the prompt; Min-K% (k = 20%) log-prob; mean token entropy; first-continuation-token entropy and log-prob (EMNLP boundary effect). Reuse implementations from **`iamgroot42/mimir`** (MIA library with loss / zlib / Min-K% attacks) where convenient — verify the repo.
- **B3 Loss trajectory:** continuation cross-entropy at the checkpoint just **before** the sequence's step (floor to 1000s) and at the next 1000 after it, and at S_f; derived: exposure drop and post-exposure slope. (Counterfactual idea from Lesci et al.)
- **B4 Data statistics:** zlib compression ratio of the 64-token text (z-complexity proxy), fraction of unique tokens, longest repeated n-gram, mean unigram log-frequency of continuation tokens (estimate frequencies from 1M sequences of the shard).
- **B5 Supervised hidden-state probe:** mean-pooled continuation-token hidden states at 6 layers (see 6.2) → PCA to 256 → logistic regression. **This is the most important baseline:** topology must add information beyond a learned linear readout of the same states.
- **B6 Same-cloud geometry (Malhotra's geometric baselines):** per cloud: total variance, mean pairwise distance, participation ratio / effective rank, mean cosine between consecutive tokens, norm of the centroid.
- **B7 Weight-noise recovery (latent-memories, optional, subset only):** 20 perturbations with N(0, (2×10⁻³)²) added to all weights at S_f; fraction of perturbations under which the sequence becomes 32-extractable. Run only on the hard-subset test sequences (cost). If it's prohibitively slow, skip and say so.

### 6.2 Topological features (ours)
- **Layers:** 6 evenly spaced block outputs (for 1b-deduped with 16 layers: 3, 5, 8, 10, 13, 16 — confirm `num_hidden_layers` and adjust). Same layers for every model, by relative depth.
- **Preprocessing (fixed in advance):** centre the cloud; divide by its median pairwise distance. Rationale: hidden-state scale drifts during pretraining; without this, velocity measures scale. Also run raw clouds as a secondary analysis.
- **T1 Static topology:** Vietoris–Rips PH (Ripser, Euclidean, maxdim 1) of the 64-point cloud → 41-feature summary. Do it for the full 64 tokens and separately for the 32 continuation tokens. Reuse `summary.py` from the Malhotra repo after checking its feature list against §1.6.
- **T2 Topological velocity (Malhotra):** per sequence and layer, Wasserstein-2 distance (`persim`) between H1 diagrams (and separately H0) of the same cloud at (pre-exposure checkpoint → post-exposure checkpoint) and (post-exposure → S_f).
- **T3 Cross-layer change:** Wasserstein distance between H1 diagrams of adjacent selected layers at S_f, summarised by mean and max (a light version of Gardinazzi et al.'s zigzag idea).
- **T0 Text-only control:** T1 computed at **step 0** (random model). Used as a baseline in H4.

## 7. Models and analysis

- **Classifiers:** logistic regression (standardise; L2; C chosen by inner CV) as primary; gradient-boosted trees (LightGBM or sklearn HistGradientBoosting) as robustness. Prune features with |corr| > 0.5 within the topological family (Fay et al.) for interpretability analyses only; predictive comparisons use the full set.
- **Feature-set ladder (each evaluated identically):**
  1. B0  2. B0+B1  3. B0–B2  4. B0–B4 (**"strong non-representation baseline"**)  5. B0–B5 (**"strong baseline"**)  6. B0–B6  7. **B0–B6 + T1–T3 (full)**  8. full + T0  9. T1–T3 alone
- **Primary comparison (H1):** ladder 7 vs ladder 6, hard subset, out-of-time test, S_f = 10000.
- **Metrics:** AUROC (primary), AUPRC, recall at precision 0.9 (prevalence-reweighted; Biderman framing), balanced accuracy on a balanced 128-example draw (to compare with "Don't Forget"; repeat 100 draws, report mean ± sd).
- **Uncertainty:** paired bootstrap over test sequences (2000 resamples) for every ΔAUROC; also DeLong test.
- **Residual analysis:** bin test sequences into deciles of continuation cross-entropy at S_f; within each decile compute the AUROC of the topology-only model. If topology only works because it re-encodes loss, within-decile AUROC ≈ 0.5.
- **H2:** for the pruned interpretable topological features, Hedges' g (future-memorized vs not, hard subset, loss-decile-matched) with bootstrap CIs.
- **H3:** compare T2 velocity distributions; Cliff's δ with CI.
- **Reproduce the literature number:** report B0's precision and recall for `mem_final` at S_f = 23000 and compare to Biderman's 12B/6.9B numbers (different model size — compare qualitatively).

## 8. Controls (all mandatory)

1. **Shuffled labels:** the full pipeline with permuted labels → AUROC ≈ 0.5 for every ladder rung.
2. **Step-0 (text-only):** T0 as a feature set, and ladder 8 vs 7 (H4).
3. **Gaussian floor (Malhotra):** isotropic Gaussian clouds of 64 points matched in d → their H1 counts/entropy; ensures real H1 counts exceed the artifact level.
4. **Leakage:** no checkpoint later than S_f may be used for features. Grep the code for it and state it in the report.
5. **Time split:** all headline numbers from steps 4000–5999, models fit on 1000–3999 only.
6. **Two subsample seeds** for the negative sampling; report both.
7. **Label fragility:** report the `mem_final` recomputation agreement (5.3).

## 9. Outputs

```
data/      pool.parquet (index, step, split, tokens64, labels)
features/  <model>/S<Sf>/{loss,data,probe,geom,topo_T1,topo_T2,topo_T3,topo_T0}.parquet
results/   <model>/S<Sf>/ladder.csv   (rung, subset, AUROC, CI, AUPRC, recall@P0.9, bal_acc128)
           <model>/S<Sf>/deltas.csv   (comparison, ΔAUROC, CI, DeLong p)
           <model>/S<Sf>/effects.csv  (feature, Hedges g, CI)
figures/
  fig1_ladder.png        AUROC by feature-set rung, hard subset, with CIs (one panel per model × S_f)
  fig2_pr_curves.png     precision–recall, strong baseline vs full, natural prevalence, with Biderman's single point marked
  fig3_residual.png      topology-only AUROC within cross-entropy deciles
  fig4_effects.png       Hedges' g of interpretable topological features (H2), with CIs
  fig5_velocity.png      T2 velocity distributions, future-memorized vs not (H3)
  fig0_checks.png        alignment check, Gaussian floor, shuffled-label AUROCs
REPORT.md
config.yaml              every seed, model, revision, layer list, normalisation, S_f, T
```
Figures: one y-axis per panel, small multiples rather than dual axes, CIs as error bars or bands, colour-blind-safe palette in this order: `#2a78d6`, `#eb6834`, `#1baf7a`, `#eda100`; direct labels where few series.

## 10. Execution plan (stop at each gate)

1. **Setup.** Python 3.11 venv on `/workspace`; `HF_HOME=/workspace/hf`. Install: `torch transformers accelerate datasets huggingface_hub numpy scipy scikit-learn pandas pyarrow matplotlib ripser persim joblib zstandard lightgbm shap`. Clone into `/workspace/refs/`: `EleutherAI/pythia`, `malhotranaman/tracking-representation-dynamics-with-persistent-homology`, `pietrolesci/memorisation-profiles`, `iamgroot42/mimir`. Read the relevant files; note in `REPORT.md` what you reused. If a repo doesn't exist or differs from this description, say so and continue.
2. **Data + alignment.** Download only the needed preshuffled shard(s); determine the epoch boundary (§4); build the pool; run the alignment check. *Gate:* 200/200 token matches; epoch boundary found.
3. **Labels.** `mem_T`, `mem_Sf`, `match_len_Sf`, `mem_final`, fragility check. *Gate:* report prevalence of `mem_T`, size of hard subset with `mem_T = 1` (need **≥ 300** positives in the hard-subset **test** split for meaningful AUROC CIs; if fewer, enlarge the pool before continuing).
4. **Validation of the topology code** on synthetic 64-point clouds: (a) noisy circle → one long H1 bar; (b) Gaussian → only short bars; (c) scaling a cloud ×10 leaves normalised features unchanged; (d) Wasserstein(diagram, itself) = 0. *Gate:* all pass.
5. **Pilot.** 1b-deduped, S_f = 10000, all features, ladder, controls. *Gate (go/no-go):* report ladder 7 vs 6 on the hard subset. If ΔAUROC CI includes 0 **and** the topology-only residual AUROC within loss deciles is ≈ 0.5, stop and report a negative result with the evidence; do not tune toward a positive result.
6. **If go:** S_f = 23000 on 1b; then 410m at both S_f; then secondary target T = 143000.
7. **Optional stage 2:** 6.9b-deduped at S_f = 23000; B7 noise baseline.
8. **Report.**

Engineering:
- Make every stage resumable (skip existing outputs); log to `logs/`.
- Batch greedy decoding with KV cache; left-pad prompts; all prompts are exactly 32 tokens so no padding is actually needed — verify.
- Don't store full hidden states for all sequences: compute per-cloud features on the fly (GPU forward → CPU Ripser via joblib) and store only features plus the mean-pooled states needed for B5.
- PH on 64 points is fast (milliseconds); the cost is volume (sequences × layers × checkpoints × 2 clouds). Parallelise over CPU cores.
- Delete each checkpoint's weights after use unless reused.

## 11. Report (REPORT.md)

1. What was run (models, S_f, T, sizes, compute used).
2. Pre-registered hypotheses H1–H4 → supported / not supported / inconclusive, each with its figure and numbers.
3. The ladder table for the hard subset and full set; the primary Δ with CI.
4. Comparison points: B0 precision/recall vs Biderman; balanced-accuracy-128 vs "Don't Forget" (84.4% loss-only, 90.6% their best, at step 10k on Pythia-1B — note they used different sequence selection).
5. Controls: all clean? If any failed, say so first.
6. Effect sizes, not just p-values; describe small effects as small.
7. Deviations from this brief and why.
8. One-line next step.

## 12. Novelty check (do alongside, ~1 hour)

On Semantic Scholar, list papers citing (a) Biderman et al. 2023, (b) "Uncovering Latent Memories", (c) Fay et al. 2026, newest first. Flag any that predict memorization from hidden-state geometry or topology, or across checkpoints. Put them in `REPORT.md` under "Related work found".
