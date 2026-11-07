# E13 pre-registration: all stages (written 2026-11-07, before any E13 extraction was analysed)

This file fixes the hypotheses, metrics, thresholds and decision rules for every stage of E13. The sources are:
- `PLAN.md`: the owner's hypotheses and decision rules;
- `METHODOLOGY.md`: the owner's work order;
- the Stage-1 pre-registration (`PREREG_stage1.md`, including Amendment 1).

Where the owner's text left a choice open, the choice is fixed here and listed in `DEVIATIONS.md`. Code:
`src/e13/` and `run_all.py` (committed together with this file). Results are labelled **PRIMARY** (a rule below) or
**EXPLORATORY** (anything else); exploratory results are never headline claims.

**Laptop pilot.** Numbers 1–500, Pythia-1.4B on CPU. It is a code test only. Its numbers are not results and are
not used for any claim or for setting any threshold.

## 1. Data
- **Models.**
  - Stage A, discovery: Pythia-1.4B (chunk) and Qwen2.5-1.5B (digit-splitting).
  - Stage B, replication: Crystal, OpenLLaMA-3B, StarCoderBase-3B, SmolLM2-1.7B (digit-splitting); OLMo-2-1B,
    RedPajama-3B, Llama-3.2-1B (chunk).
- **Contexts and read positions** are as in METHODOLOGY 2.1–2.2:
  - C1 (stored E10 copy prompts, seed 42), C1s7 (same format, seed 7), and C2–C8;
  - each read at P_num and P_next, with P_task for C5 and C6;
  - second wordings C2b–C8b at P_num, for within-context reliability only.
- **Layers.** 9 stored layers, at fractions 0, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 0.84375 and 1 of depth.
- **Neutral average AVG5.** The mean over C2, C3, C4, C7 and C8 at P_num. It is the **main "number reading"**.
  C1 (copy), C5 (compare) and C6 (successor) are task contexts and are reported separately.
- **Sets.** S2 = 10–99, S3 = 100–999, S4 = 1000–9999, never mixed. S1 = 1–9 is used for the exponent probe only.
  - A number enters a (set, condition) only if its token count equals the set's modal count in every context the
    condition uses, and its position flag is ok.
  - Chunk tokenizers: the count must be exactly 1.
  - Token counts are reported for every set and context.
- **Dev split.** 20 % of numbers (seed 13). It is the only data used to choose anything: the causal layer, the
  subspace size m, and the steering α. All layers are reported for every other test.

## 2. Preparation and tiers
- **Preparation.** Centre, z-score, PCA to 50, fitted on training numbers wherever a train/test split exists.
- **Robustness.** The Stage-1 module's no-z-score and drop-top-3 variants are kept.
- **FULL tier** (every control and the planted power checks): C1/P_next (the old reading), C1/P_num and AVG5/P_num,
  at all 9 layers × 3 sets.
- **LIGHT tier:** every other condition (point estimates + shuffled-n and digit-label controls). LIGHT cells use the
  planted power of the FULL AVG5 cell with the same model, layer and set.

## 3. Metrics (every cell)
- **M1 smooth part.** Gaussian kernel f(n), bandwidth chosen by 5-fold CV from {1, 2, 5, 10, 20} % of the range
  (h ≥ 1).
  - Held-out R² on random 80/20 splits (3 repeats) and on the block 70–79 / 700–799 / 7000–7999.
  - R² = 1 − Σ‖Z_test − pred‖² / Σ‖Z_test − mean(Z_train)‖².
  - Shuffled-n control.
- **M2 additive digits.** Curve + centred one-hot digits (all positions), backfitted; also the units-only variant.
  Reports held-out R² and the variance shares of f, digits and residual.
- **M3 noise ceiling.**
  - **Single context c:** the split-half reliability between c and its second wording cb,
    r = tr Cov(c, cb) / √(trVar c · trVar cb). C1 uses C1s7 as its second set.
  - **AVG5:** the mean reliability over the 10 splits of the five neutral contexts into 2 + 3, Spearman–Brown
    corrected.
  - Every R² is also reported as a fraction of this ceiling.
- **M4 probes** (one 80/20 split; ridge with CV α):
  - linear probes for n and for log n (R², Spearman);
  - a sinusoidal probe: fixed periods {2, 5, 10, 100, 1000, 10000} decoded jointly; exact accuracy and MAE; plus a
    25-period scan;
  - a circular probe per digit position (Levy & Geva; accuracy);
  - a linear classifier per digit position (accuracy).
- **P8 knee.** A reduced-rank (k = 1…30) version of "spline(n, df 10) + one-hot digits"; held-out R² (random and
  block).
  - Knee = the smallest k ≤ 10 with R² ≥ 90 % of R²(k = 10).
  - Sharpness = the mean over k = 1…10 of R²(k) / R²(10).
  - Controls: shuffled n, and a flat 2-D sheet with the real residual as noise.
  - Expected under H1: 1 + 9·(digits − 1).

## 4. Decision rules (PRIMARY)
**Families.** For each family (digit-splitting, chunk), over the cells named in each rule:
- **holds** if ≥ 70 % of powered cells pass;
- **does not hold** if ≤ 30 % pass;
- **mixed** otherwise;
- **cannot tell** if < 50 % of the cells have power.

A hypothesis is **supported** if it holds in both families (and, at Stage B, in the replication models alone);
**one family only**; **not supported**; or **cannot tell**. Rules are applied separately to AVG5/P_num, C1/P_num
and C1/P_next.

- **H1 position-resolved digit code.**
  - **Cells:** depth ≥ 50 % (slots 50, 62.5, 75, 84 % and last) × sets.
  - **Cell rule:** O1_leading − O1_units > 0.30 with its 95 % bootstrap CI above 0, and O1 non-increasing from
    leading to units (tolerance 0.05).
  - **Power:** a planted line at the leading digit and a simplex at the units must be detected in ≥ 80 % of
    replicates, with ≤ 20 % false positives.
  - **Rivals:**
    - R-a: all O1 > 0.5 and range < 0.1;
    - R-b: all O1 < 0.3;
    - R-c: units template = circle, in cells with circle power.
  - All other metrics follow `PREREG_stage1.md` (debiased centroid distances, O1, PR, templates, A1).
  - **Exponent (secondary):** a ridge probe of digit count trained on lengths {2, 3} and tested on 1 and 4. "Ordered"
    if pred(1) < mean(2) and pred(4) > mean(3).
- **H2 carries** (Stage-1 rules with Amendment 1).
  - **H2a:** offset model, e_1 > 0 and e_2 − e_1 > 0 with CIs above 0. Power uses the transient carry plant.
  - **H2b:** R²(level 0) − R²(level ≥ 1) > 0.20 with CI above 0. Power uses the cumulative plant. The beyond-digits
    contrast with the additive fake is secondary.
  - **H2c:** tree − line R² on held-out leading-digit groups ≥ 0.10 with CI above 0. Late layers only (84 %, last).
- **H0 invariance**, per (model, layer, set) and per key metric (R2_curve, R2_curve_digits, O1_leading, Δ, e_1,
  ΔR²_group, probe_n_R², lin_acc_units):
  - **Cell rule:** the metric at C1/P_num is ≥ 50 % of its value at C1/P_next with the same sign, and so is the
    metric in ≥ 3 of the 7 natural contexts C2–C8 at P_num.
  - A metric is **invariant** if the cell rule holds in ≥ 70 % of cells.
  - **Across models:** the ordering of the models on the metric must not reverse. The Spearman correlation between
    models' C1/P_next and AVG5/P_num values must be > 0 (≥ 3 models).
  - **If not invariant:** the conclusion is "the old structure was mostly the copy plan / output geometry", and the
    parts that survive are listed.
  - **Also reported** (secondary):
    - the number / context / interaction variance decomposition, with the wording-noise share from the second
      wordings;
    - cross-context transfer of f + digits (context A → B) as a fraction of the B ceiling;
    - distance-matrix agreement between contexts, vs between the two wordings of one context.
- **H3 lexicon** (AVG5 halves).
  - **Residual** r = z − f − D, fitted on training numbers; held-out numbers only.
  - **Reliability** = the mean per-number correlation of r between two context halves, after removing the mean
    residual; averaged over 5 splits.
  - **Lexicon exists** if, in ≥ 70 % of cells at depth ≥ 50 % in both families, reliability ≥ 0.30 AND the
    log-frequency slope of ‖r‖ has a 95 % bootstrap CI excluding 0. The regression uses SlimPajama counts plus the
    flags: year 1900–2030, multiple of 100, multiple of 1000, repdigit, power of 2, famous {7, 404, 911, 666, 1337}.
  - **Crowding (S4)** is "lexical" if its effect, the relative median-H0-death difference between frequent and rare
    tiers, falls by ≥ 50 % once the residual is removed.
  - **Power:** a planted lexicon at true reliability 0.5, recovered (r ≥ 0.30) in ≥ 80 % of replicates with ≤ 20 %
    false positives.
- **P7 positive control.**
  - **Data:** days (7) and months (12) in 20 templates each, at the item's last token, averaged over templates.
  - **Projection:** the top-k PCs of the item centroids, k = 2…6.
  - **Circular-order score:** the share of items whose two nearest items are their calendar neighbours (wrap).
  - **Circular probe:** leave-one-template-out.
  - **(model, layer, category) passes** if the score ≥ 0.75 for ≥ 3 of the 5 k values AND probe accuracy ≥ 0.5.
  - **A category works** if it passes at ≥ 2 of the 5 E10 depths. The control works for a model if both categories
    work.
  - If it does not work, persistent-homology and loop claims are dropped. The same detector is run on the units
    digit and reported.
- **P8.** The knee is "sharper than the controls" if sharpness(real) > sharpness(shuffled) and > sharpness(sheet).
  This must hold in ≥ 70 % of cells at depth ≥ 50 %, both families.
- **Causal (Stage C).**
  - **Models:** Pythia-1.4B, Qwen2.5-1.5B, OLMo-2-1B. The stage runs only if Gate A passes.
  - **Layer:** the one maximising the dev effect of S_mag ablation on B1 minus its random control.
  - **m:** the smallest m with ≥ 90 % of f's variance (≤ 5), chosen on dev.
  - **Steering α:** each of {0.5, 1, 2} is reported.
  - **Subspaces:** S_mag, S_dig(p), S_carry, 5 random subspaces of equal dimension orthogonal to all of them, and
    S_mag fitted on shuffled n.
  - **Interventions** at the number's last token: mean-projection ablation, subspace patching from another
    number's clean state, and steering along f(n′) − f(n).
  - **Behaviours:** B1–B6 (`causal.py`), scored by log-probability margins, ≥ 500 test items each, balanced for
    carries.
  - **Prediction:** S_mag → B1; S_dig(units) → B2 and B6; S_carry → the carry items of B4 and B5.
  - **Pass:** the target effect exceeds 2 × the random effect (CI of target − 2·random above 0) AND the off-target
    effect is below 1.2 × random (CI of off − 1.2·random below 0). This must hold in ≥ 2 models.
  - The full effect matrix is reported.
- **Training (Stage D).**
  - **Checkpoints:** Pythia (16) and OLMo-2 (12).
  - **Emergence time:** the step at which a metric reaches 50 % of its final value (log-step interpolation).
  - **Predicted order:** units-digit categories (linear accuracy of the units digit) → leading-digit ordinality
    (O1_leading) → carry structure (|e_1|) → lexicon (H3 reliability).
  - **Test:** Kendall τ with the predicted order, averaged over (layer, set) cells; null = metric labels permuted
    within cells (10,000 permutations). It **holds** if p < 0.05 in both Pythia and OLMo-2.
  - **DataDecide (25 recipes):** the slope of each metric on recipe α (alpha_ols), with a recipe bootstrap.
    - **Predicted:** ordinality decay and lexicon vary with α (CI excludes 0).
    - **Predicted not to vary:** the units template (simplex share) and carry spikes (CI includes 0).
- **Gate A** (after Stage A2):
  - (i) the positive control works in both discovery models;
  - (ii) H1 or H2 is "supported" or "one family only" on the number reading (AVG5/P_num or C1/P_num).
  
  The causal stage runs only if (ii) holds. If neither holds, the outcome is the negative-result paper.

## 5. Order and stopping
`run_all.py` runs A1 → A2 → Gate A → B → C (if allowed) → D → report, and saves every result. The owner reviews the
results after the run; stopping between stages is replaced by the automatic Gate A (`DEVIATIONS.md` D16).
