# Experiment brief: the topology of truth representations across pretraining

**For:** Claude Code, running on a RunPod GPU pod.
**Goal of this run:** preliminary results to show a professor within a few days. A pilot, not the final paper — but built so that it can grow into one without redoing work.

Read this whole file before writing any code. Sections 1–3 are context, 4–9 are the specification, 10–12 are how to execute and report.

---

## 1. Background

It is well established that LLM hidden states separate **true** from **false** statements, and that a *linear* probe can read this out (Marks & Tegmark, *The Geometry of Truth*, COLM 2024; Bürger et al., *Truth is Universal*, NeurIPS 2024, who found a 2-D truth subspace once negations are included).

Two papers have tracked this **across pretraining checkpoints**, both with linear probes only:

1. **Ravfogel, Yehudai, Linzen, Bruna, Bietti — "Emergence of Linear Truth Encodings in Language Models"** (arXiv 2510.15804; code: `github.com/shauli-ravfogel/truth-encoding-neurips`).
   - Toy model: truth encoding emerges in two phases — fast memorization of facts, then a slower, abrupt emergence of a linear truth direction. Mechanism hinges on layer norm.
   - Real-model check (their Appendix E.4): **Pythia-6.9B checkpoints**, CounterFact data, inputs built by concatenating **K = 4 statements whose context is all-true or all-false**. On the final statement they measure (a) *memorization* = share of cases where the correct attribute is top-1, (b) *ΔH* = entropy of the next-token distribution under false context minus under true context, (c) *probe AUC* = linear classifier predicting the truth of the context.
   - Their numbers (use as a replication target): probe AUC 0.383 at step 0, 0.467 at step 1000, 0.587 at 3000, 0.667 at 10000, 0.754 at 20000, 0.802 at 60000, 0.831 at 143000. Memorization 0.000 → 0.242 (3k) → 0.547 (10k) → 0.875 (143k). ΔH ≈ 0 until step 1000, 0.219 at 3000, 0.518 at 143000.
   - Their reading: memorization jumps then plateaus; ΔH and probe accuracy climb **steadily**. They found no evidence that layer norm alone induces separability in Pythia; the truth signal grows gradually with depth.
   - Checkpoints they used: 0, 512, 1000, 3000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 110000, 120000, 130000, 143000.

2. **Qian et al. — "Towards Tracing Trustworthiness Dynamics: Revisiting Pre-training Period of LLMs"** (arXiv 2402.19465; code: `github.com/ChnQ/TracingLLM`).
   - 360 checkpoints of LLM360 Amber-7B (+ OLMo-7B), five dimensions incl. truthfulness (TruthfulQA questions + candidate answers turned into true/false statements).
   - Method: **last-token activation at every layer**, balanced data, **random 4:1 train/test split, binary linear classifier, report test accuracy**.
   - Findings: middle layers separate best; probe accuracy rises early in pretraining then fluctuates. With mutual information (HSIC estimator) they see two phases: I(T;Y) rises throughout while I(T;X) rises then falls ("fitting" then "compression").

Two papers apply **persistent homology (PH)** to LLM activations, and supply our topological methodology:

3. **Fay, García-Redondo, Wang, Dubossarsky, Monod — "The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology"** (ICLR 2026 Oral; arXiv 2505.20435; no public code found).
   - One point = **last-token hidden state of one input at one layer**. Two conditions (clean vs adversarial) are compared.
   - **Subsampling:** K = 64 subsamples of k = 4096 points per condition per layer (their ablation: later layers were already perfectly separated with 30 subsamples of 100 points).
   - **PH:** Vietoris–Rips, Euclidean, H0 and H1, Ripser.
   - **41-number barcode summary** per barcode: {mean, min, Q1, median, Q3, max, std} × {H0 deaths, H1 births, H1 deaths, H1 persistence, H1 birth/death ratio} = 35, plus total persistence, number of bars and persistent entropy for H0 and for H1 = 6.
   - **Analysis:** drop features correlated > 0.5; PCA of summaries; logistic regression on summaries (70/30 split + 5-fold CV); SHAP for feature importance; CCA loadings. Compared against linear classifiers on raw activations.
   - **Controls:** clean-vs-clean, adversarial-vs-adversarial and mixed-vs-mixed random splits.
   - **Finding:** adversarial inputs → "topological compression": higher mean H0 death (more dispersed), fewer H1 bars that are born later and live longer, lower H1 entropy.

4. **Malhotra et al. — "Tracking Representation Dynamics in Large Language Models with Persistent Homology"** (arXiv 2606.19542; code: `github.com/malhotranaman/tracking-representation-dynamics-with-persistent-homology`, MIT).
   - Tracks topology **across fine-tuning checkpoints** of 4 models (1B–7B) on a **fixed evaluation set** (750 prompts), final-token states, five evenly spaced layers.
   - **B = 64 overlapping subsamples of size m = 160** per cell, Vietoris–Rips PH up to H1 with Ripser, the same 41-feature summary as Fay et al.; summaries averaged per cell; **inference at the cell level** (overlapping subsamples are not independent).
   - **Topological velocity** v_t = Wasserstein distance between H1 barcodes of consecutive checkpoints.
   - **Concentration** C = fraction of total velocity in the first third of training; null = permute checkpoint order (≥ 2000 permutations).
   - **Controls/baselines:** isotropic Gaussian clouds matched in n and d (they produce many artifactual H1 bars), geometric baselines (centroid drift, total variance, mean pairwise distance), effect sizes (Hedges' g, Cliff's δ with bootstrap CIs), energy distance between summary distributions.
   - **Behaviour comparison:** behavioural velocity (JS divergence of next-token distributions between checkpoints) and lead–lag of peaks: topological peaks never lagged behavioural ones.
   - Repo layout: `src/phlatent/topology/summary.py` holds the 41-feature summary (`FEATURE_NAMES`), `topology/` and `analysis/` are pure numpy/scipy/sklearn.

Data source for statements: **`github.com/saprmarks/geometry-of-truth`** (`datasets/` folder: true/false CSVs such as `cities`, `sp_en_trans`, `larger_than`, `neg_cities`, `companies_true_false`, `common_claim_true_false`, `counterfact_true_false`; each has at least `statement` and `label` columns — **verify the exact file names and columns after cloning**).

## 2. Motivation and the gap

Linear probes answer one question: *is there a flat boundary between true and false?* They are blind to **how** the representation is organised. Two concrete blind spots:

- **Position vs shape.** PH on a point cloud is translation-invariant: shifting a cloud changes nothing. A linear probe detects shifts (difference of means). So the probe measures *where* true and false sit; class-wise PH measures the *shape* of each class's cloud. These are complementary, not redundant.
- **Global vs local truth.** True and false could form one global split shared across topics, or be separated only *within* each topic with topic-specific directions. A pooled linear probe can score highly in both cases. The "LLM Knowledge is Brittle" paper (arXiv 2510.11905) found truth separability collapses for out-of-distribution surface forms, which hints at local organisation.

As far as we could find, **no published work has applied topological methods to true/false representations, at one checkpoint or across pretraining.** (Not certain — see §12 for the novelty check to run.)

## 3. Aim and hypotheses

**Aim:** track the topology of true and false statement representations across Pythia pretraining checkpoints, side by side with the linear-probe and behavioural measures of the prior papers, and find where topology shows something probes do not.

Pre-registered hypotheses (write them into the results README *before* looking at real data):

- **H1 — timing.** The onset of topological truth structure differs from the onset of linear separability and of behaviour by ≥ 2 checkpoints (leads or lags).
- **H2 — abrupt vs smooth.** Probe AUC rises smoothly (as Ravfogel et al. report), but topological velocity is concentrated in a narrow window (C exceeds its permutation null, or a single velocity peak).
- **H3 — shape.** True-only and false-only clouds differ in shape (barcode-summary classifier beats the mixed-vs-mixed control) at layers/checkpoints where the probe is high.
- **H4 — local truth.** Within-topic truth structure is significant while cross-topic truth structure (after removing topic means) is weak — and/or this changes over training.

A null on all four (topology rises in lockstep with the probe) is a legitimate result; report it and stop.

---

## 4. Models and checkpoints

| Role | Model | Why |
|---|---|---|
| **Primary** | `EleutherAI/pythia-1.4b-deduped` (24 layers, d = 2048) | Public step checkpoints, public data order, cheap on one GPU |
| Size check | `EleutherAI/pythia-410m-deduped` (24 layers, d = 1024) | Does the pattern hold at smaller scale? |
| Replication (stage 2, optional) | `EleutherAI/pythia-6.9b` | Direct comparison to Ravfogel et al.'s Table 1 |

**Checkpoints:** exactly Ravfogel et al.'s 15 steps, so results are comparable:
`0, 512, 1000, 3000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 110000, 120000, 130000, 143000` (HF `revision="step{N}"`).
Optional dense early window (Malhotra et al. found most topological change early): add `64, 128, 256` if time allows.

- Load in **fp32** (there is a known fp16 issue with Pythia checkpoints). For 6.9B, fp32 needs ~28 GB → use an 80 GB GPU, or bf16 and say so in the report.
- Step 0 is a randomly initialised model: it is the **untrained control** for every measure.
- Save hidden states for **all** layers (index 0 = embeddings, 1..L = block outputs).

## 5. Inputs (prompt styles)

Two input formats, one from each probe paper. **P1 is primary.**

### P1 — single statements (Marks & Tegmark / Qian et al. style)
- Topics: `cities`, `sp_en_trans`, `larger_than` (three different kinds of fact: geography, translation, numeric comparison).
- Balance within topic: up to 300 true + 300 false per topic (fewer if the file has fewer; `sp_en_trans` is small). Fixed seed.
- Input = the statement alone, no template, no BOS added beyond the tokenizer default.
- Read position = **last token** (the final period).
- Label = statement truth; also keep the topic label.
- **Confound check (required, report it):** distribution of token lengths per class per topic; and, for `cities`, that each country appears as an object in both true and false statements. If a class can be predicted from length alone (logistic regression on length, AUC > 0.6), rebalance.

### P2 — truth-context sequences (Ravfogel et al. style; replication)
- CounterFact facts (subject, relation, true attribute). Build negatives by swapping the attribute for another attribute of the same relation. Use a handful of frequent relations (Ravfogel used the 25 most frequent).
- Sequence = 4 statements, the first 3 all true or all false (the *context*), the 4th is a fresh true fact whose attribute is to be predicted.
- Read position = **the token immediately before the final attribute** (where the attribute is predicted).
- Label = truth of the context.
- Behaviour at that position: memorization (correct attribute's first token is top-1) and ΔH (entropy under false context − under true context, on matched pairs sharing the same 4th statement).
- ~600 sequences per class is plenty.

## 6. What to extract per checkpoint

For each model × checkpoint × input set, save `states/<model>/<input>/step<N>.npz` with:
- `H`: float16, shape (n_layers + 1, n_items, d_model) — read-position hidden state, all layers
- P1: `loglik` — mean per-token log-probability of the statement
- P2: `p_correct`, `top1_correct`, `entropy` at the read position
- Plus an `items.csv` once per input set (statement text, label, topic, ids).

Delete each downloaded checkpoint after extraction (each is several GB). Put `HF_HOME` on the pod's persistent volume (`/workspace`).

## 7. Measures

Compute at **every layer of every checkpoint** unless stated. "Cell" = (model, input set, checkpoint, layer).

**Preprocessing for PH (decide once, do not change after seeing data).** Primary: centre the cloud and divide by its median pairwise distance (computed on a fixed subset of 500 points). Hidden-state scale changes a lot during pretraining; without this, velocity mostly measures scale. Secondary: raw activations, to match the PH papers. Report both; interpret the primary.

### 7.1 Behaviour (per checkpoint)
- **B1 (P1):** AUC of `loglik` for true vs false, computed within each topic and averaged.
- **B2 (P2):** memorization and ΔH as Ravfogel et al.

### 7.2 Linear (the prior papers' measures)
- **L1 Logistic probe:** standardise → `LogisticRegression` (L2, C = 1.0, max_iter 2000). Report held-out AUC with 5-fold stratified CV (also report Qian et al.'s single 4:1 split for comparability). **Null:** same with shuffled labels.
- **L2 Mass-mean probe** (Marks & Tegmark): direction = mean(true) − mean(false) from training folds; AUC of the projection on test folds.
- **L3 Cross-topic generalisation (P1):** train L1 on one topic, test on each other topic; report the mean over ordered pairs. This is the *linear* version of the global-vs-local question.
- *(Optional)* **L4 HSIC estimates of I(T;Y) and I(T;X)** as Qian et al. (Gaussian kernel, σ by grid search, X = layer-1 activations). Skip if short on time.

### 7.3 Topology — class shape (Fay et al.)
- Subsamples: per class, B = 64 subsamples of m = 160 points, **stratified by topic** so true and false subsamples have identical topic mix. Use the **same subsample indices for every checkpoint and layer** (fixed seed).
- PH: Vietoris–Rips, Euclidean, H0 + H1, `ripser` (pass a precomputed distance matrix computed in float64; GPU `torch.cdist` is fine for speed).
- 41-feature summary per barcode. **Reuse `src/phlatent/topology/summary.py` from the Malhotra repo** (MIT) after checking its feature list matches Fay et al.'s definition above; otherwise implement it.
- **T1 Shape-test accuracy:** logistic regression on the 2B standardised summaries (true-subsample vs false-subsample), 5-fold CV accuracy.
- **T1-null:** identical pipeline on two pseudo-classes made by randomly splitting the pooled data (mixed vs mixed, Fay's control). Because subsamples overlap, the absolute accuracy is optimistic; **only the gap to T1-null is interpretable.**
- **T1-features:** for the interpretable features — mean H0 death, number of H1 bars, mean H1 persistence, H1 entropy — report the true-minus-false difference and Hedges' g with a bootstrap 95% CI (bootstrap over subsamples). After correlation pruning (> 0.5, as Fay et al.), report SHAP or logistic coefficients at the reference layer only.

### 7.4 Topology — dynamics (Malhotra et al.)
- Fixed set of B = 64 mixed subsamples (m = 160, stratified by topic and class), same indices at every checkpoint.
- **T2 Velocity:** at each layer, Wasserstein-2 distance (L∞ ground metric) between H1 diagrams of the same subsample at consecutive checkpoints, averaged over subsamples. Also H0 velocity, and velocity of true-only and false-only clouds separately.
- **T2-C Concentration:** fraction of total velocity accumulated by step 40000 (≈ first third of 143k steps). Null: permute checkpoint order, ≥ 2000 permutations.
- **T2-baselines:** centroid drift, total variance, mean pairwise distance between consecutive checkpoints (Malhotra's geometric baselines). Topology only adds information where it departs from these.
- **Gaussian floor:** isotropic Gaussian clouds with the same m and d, through the same pipeline (number of H1 bars, H0 entropy) — so artifactual H1 counts are not mistaken for structure.

### 7.5 Topology — global vs local truth (our addition)
The finite H0 death times of a Vietoris–Rips filtration are exactly the edge lengths of the Euclidean minimum spanning tree (MST): each H0 merge *is* an MST edge. So we can ask which points each merge joins.
- **T3a Topic homophily:** MST of the full (normalised) cloud; fraction of edges joining same-topic points.
- **T3b Within-topic truth homophily:** MST of each topic's points separately; fraction of edges joining same-truth points; average over topics.
- **T3c Cross-topic truth homophily:** subtract each topic's mean, MST of the pooled centred cloud; among edges joining *different* topics, fraction joining same-truth points.
- **Nulls:** permute truth labels within topic (1000 permutations; MST is fixed so this is cheap). Report z-scores. Correct for the layer search with the **max-statistic** method: for each permutation take the max z over layers, compare the observed best layer to that distribution.
- Reading: T3b high + T3c at chance → truth organised locally per topic. Both high → a shared truth structure. Compare against L3 (linear cross-topic transfer).

### 7.6 Validation before real data (mandatory)
Build synthetic clouds with the real n and d and confirm each measure behaves as expected. Do not run real analyses until these pass:
1. **No signal** (labels random): L1, T1 gap, T3b, T3c all at chance.
2. **Pure mean shift**, same covariance, shared direction across topics: L1 high, L3 high, **T1 gap ≈ 0**, T3b and T3c high.
3. **Topic-specific shift directions:** L1 within-topic high, **L3 ≈ chance, T3c ≈ chance**, T3b high.
4. **Shape-only difference** (equal means; e.g. one class on a noisy circle in a random 2-D plane, the other Gaussian with matched variance): **L1 ≈ chance, T1 gap high.**
5. **Abrupt change** over a synthetic checkpoint sequence (Gaussian → circle at step k): velocity peaks at k, flat elsewhere.
6. **Scale invariance:** multiplying a cloud by 10 leaves all primary (normalised) topological measures unchanged.

## 8. Controls (summary)
- Step-0 checkpoint: every measure must be at chance. **If any is significant at step 0, the pipeline is broken; stop and debug.**
- Shuffled labels (L1, L2), mixed-vs-mixed (T1), within-topic label permutation (T3), checkpoint-order permutation (T2-C), Gaussian floor (PH counts).
- Two different subsample seeds for T1 and T2; report both.
- Length/lexical confound check (§5).

## 9. Outputs

```
results/<model>/<input>/
  metrics.csv            long format: step, layer, metric, value, null_mean, null_sd, z, p
  summaries/step<N>_L<l>.npz   41-feature vectors (true, false, null pseudo-classes)
  velocity.csv
  onset.csv              per metric: first step exceeding its null 95th percentile for 2 consecutive checkpoints
figures/
  fig1_timeline_<model>.png    reference layer: B1, L1, T1 gap, T3b, T2 velocity vs step (log-x)
  fig2_heatmaps_<model>.png    layer × step heatmaps: L1, L3, T1 gap, T3b z, T3c z, T2 velocity
  fig3_global_local_<model>.png  T3a/T3b/T3c z and L3 vs step at reference layer
  fig4_replication.png         our P2 probe AUC / memorization / ΔH vs Ravfogel Table 1 (if P2 or 6.9B run)
  fig0_validation.png          synthetic tests
REPORT.md
```

**Reference layer (declared now):** the layer with the highest mean L1 AUC over the last three checkpoints. All other layers still appear in the heatmaps.

**Figures:** log-scale x for steps (step 0 plotted at a small offset and labelled "init"); one y-axis per panel (small multiples, never dual axes); null band (mean ± 2 sd) shaded grey; direct-label each line; colour-blind-safe palette (`#2a78d6`, `#eb6834`, `#1baf7a`, `#eda100` in that order); sequential single-hue colormap for heatmaps.

## 10. Execution plan

Work in phases and stop at each gate.

1. **Setup.** Python 3.11 venv on `/workspace`. `pip install torch transformers accelerate numpy scipy scikit-learn pandas matplotlib ripser persim shap`. Clone the four repos into `/workspace/refs/` and read the relevant files (Malhotra `topology/summary.py` and analysis code; Marks & Tegmark `generate_acts.py`, `probes.py`, dataset CSVs; Ravfogel repo for any CounterFact/Pythia code; Qian `src/generate_activations.py`). Note in `REPORT.md` what was reused.
2. **Data.** Build P1 (and P2 if time). Run the confound check. *Gate:* balanced, no length shortcut.
3. **Validation.** Implement measures; run §7.6. *Gate:* all six synthetic tests pass. Save `fig0_validation.png`.
4. **Pilot slice.** pythia-410m-deduped, P1, steps {0, 3000, 143000}, all layers. *Gate:* step 0 at chance everywhere; L1 at 143000 reproduces the known pattern (high in middle layers); runtime estimate for the full run.
5. **Full primary run.** pythia-1.4b-deduped, P1, 15 checkpoints. Then pythia-410m-deduped, P1.
6. **Replication (if time).** P2 on pythia-1.4b; then P2 on pythia-6.9b for a direct comparison to Ravfogel's Table 1.
7. **Report.**

Engineering notes:
- Extraction is GPU-bound and fast; the bottleneck is downloading checkpoints. Extraction and analysis are separate scripts; analysis runs on CPU and should use all cores (`joblib`) for Ripser.
- Make every script resumable (skip cells whose outputs exist) and log progress to a file, so a pod restart loses nothing.
- Rough sizes: 1.4b states ≈ 25 layers × ~1.5k items × 2048 × 2 bytes ≈ 160 MB per checkpoint.
- Fix all seeds; write the full config (model, steps, seeds, B, m, normalisation) into each results folder.

## 11. What to report (REPORT.md)

1. One paragraph: what was run.
2. The pre-registered hypotheses H1–H4 and, for each, **supported / not supported / inconclusive**, with the figure that shows it.
3. Onset table (from `onset.csv`): which measure becomes significant first.
4. Replication: does our L1 trajectory match Ravfogel et al.'s qualitative trend (steady rise)?
5. Controls: step 0, shuffled labels, Gaussian floor — all clean? If not, say so first.
6. Effect sizes, not just p-values. A significant but tiny effect must be described as tiny.
7. What failed or was skipped, and why.
8. Suggested next experiment, in one line.

Keep claims to what the data show. Do not tune thresholds, normalisation or the reference layer after seeing results; if you must change something, record the change and the reason.

## 12. Novelty check (run alongside, ~1 hour)

On Semantic Scholar, list the papers citing (a) *The Geometry of Truth*, (b) *Truth is Universal*, (c) *Emergence of Linear Truth Encodings* (2510.15804), (d) *The Shape of Adversarial Influence* (2505.20435), newest first. Flag any that apply topology/persistent homology to truth or factuality representations, or track truth representations across pretraining with non-linear methods. Put the list in `REPORT.md` under "Related work found".
