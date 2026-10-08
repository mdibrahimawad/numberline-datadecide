# E08: Checkpoint test (planned; definitions and predictions locked)

**Question:** inside one model, as training proceeds, does β follow R, the
rarely-seen count?
**Status:** **planned, not run.** This file is the pre-registration. Everything under
"Locked" was written **before** any checkpoint data was collected. It may still be edited
until the first checkpoint data is collected (review it first), then must not change.
Results go in a separate `RESULTS.md`, including any deviation from this plan, with the reason.
**Written:** 2026-10-08 (git history shows every later edit).

## Why this test
Across the 25 final models, R predicts β better than α (E07), but recipes also differ
in content (code, math), which is a rival explanation. Within **one** recipe during
training:
- the data mixture is fixed, so **α and content stay the same**;
- every number keeps being seen, so **R falls** as training goes on (up to ~3,300 within
  a model, more than the ~2,000 spread between final models).

If shrinkage is right, β rises as R falls, and α and content cannot explain that,
because they don't change.

Estimated R at earlier stages (final counts × fraction; to be replaced by exact prefix
counts):

| Model | 5 % | 10 % | 25 % | 50 % | 75 % | 100 % |
|---|---|---|---|---|---|---|
| dolma1_6plus | 9,046 | 8,663 | 8,014 | 7,467 | 6,669 | 5,707 |
| dolma1_7 | 9,119 | 8,674 | 8,087 | 7,633 | 7,148 | 6,384 |
| dclm-baseline-qc-fw-10p | 8,922 | 8,286 | 7,864 | 7,700 | 7,427 | 7,060 |
| dclm-baseline | 9,276 | 8,804 | 8,275 | 7,793 | 7,619 | 7,392 |
| falcon-and-cc-qc-10p | 9,500 | 9,222 | 8,607 | 8,266 | 7,946 | 7,735 |

## Locked

**Definition of R(t):**

$$R(t) = \#\{\, n \in \{10,\dots,9999\} : c_t(n) < 4000 \,\}$$

where c_t(n) is the exact count of integer n in the training tokens consumed up to
checkpoint step t (seed-2 training order, `_count_text`, same as E04). K = 4000 and the
range 10–9999 are fixed (`RARE_K` in `src/count_features.py`). Nothing is re-tuned.

**β(t):** E06 protocol (`src/beta_fine.py`: 10 groups, 100 prompts per group, seeds 45–47,
correct-output filter). Primary value: `beta_cont` at a **fixed layer per recipe**, namely the
layer E06 selected for that recipe's final checkpoint. Secondary (reported, not primary):
`beta_coarse` at the same layer, and both at the per-checkpoint selected layer.

**Validity checks** (a checkpoint is analysed only if it passes all; failures are reported
as "too early", never dropped silently):
1. correct-output acceptance ≥ 80 % in every group;
2. median |ρ| ≥ 0.9 at the analysed layer;
3. EV of PC1 at the analysed layer ≥ 0.1;
4. `err_coarse` ≤ 0.4.

**Checkpoints:** ≈ 10 %, 25 %, 50 %, 75 % and 100 % of training (the nearest released
revisions to steps 6,900, 17,300, 34,700, 52,000 and 69,369), plus 5 % if it passes the
checks. **Recipes:** all 25 if the checkpoints exist for them; otherwise every recipe that
has all 5.

**Predictions:**
- **P1:** within each recipe, β(t) rises as R(t) falls (Spearman over checkpoints < 0 in
  the majority of recipes; sign test across recipes, one-sided p < 0.05).
- **P2:** across all (recipe, checkpoint) pairs, R predicts β with log(step) also in the
  model: β ~ R + log(step) + recipe fixed effects, coefficient of R < 0, p < 0.05.
- **P3:** where two recipes swap order in R between checkpoints, they also swap order in β
  more often than not.
- **P4 (strongest):** final-model points (E07 line β = 2.79 − 0.275·R/1000) and checkpoint
  points fall on one shared curve. Checked by predicting checkpoint β from the E07
  line fitted on final models only, and reporting the R² of those predictions.

**Falsified if:** P1 and P2 both fail. A rise of β with training that R does not
explain beyond log(step) would point to "training time" or content, not exposure.

## Steps (when run)
1. **List checkpoints** (laptop, free):
   ```bash
   python -c "from huggingface_hub import list_repo_refs; b=sorted(x.name for x in list_repo_refs('allenai/DataDecide-c4-1B').branches if 'seed-default' in x.name); print(len(b)); print(b)"
   ```
2. **Exact prefix counts** (CPU): extend `runpod_jobs/exact_alpha.py` so the counter saves
   cumulative counts at the chosen steps (one pass over the first 69,369 × 704 chunks).
   This needs a new option plus a test, before any run.
3. **β per checkpoint** (GPU, ~$0.50): `python -m runpod_jobs.beta --fine --revisions <list>`
   (the runner already accepts several revisions). One change is needed first: the runner frees
   a model's disk cache only when a single revision is run, so with 5 revisions per recipe
   it must delete each revision after use instead, or the 80 GB disk fills up.
4. **Analysis:** a new `src/checkpoint_analysis.py` implementing P1–P4 exactly as written
   above, with a test.

## Not part of this test (separate questions)
Training-seed noise (aux-seed final checkpoints), and model size (smaller DataDecide
models).
