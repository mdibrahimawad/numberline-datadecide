# E10 report: forecasting memorization from early-checkpoint topology

## 0. Pre-registration (locked)

Written **2026-10-11, before any real data, label or feature was computed** (git history shows
every edit). This section must not change after the first label is computed. Deviations go in §7
with the reason.

**Aim.** Take training sequences seen once in steps 1000–5999. Use features computed at an
early checkpoint S_f to predict whether each sequence is 32-extractable at the primary target
T. T is the last multiple of 1000 before the second epoch, so there is no re-exposure.
Measure what topology adds beyond (a) loss signals, (b) data statistics and (c) a supervised
probe on the same hidden states.

- **H1 (main).** Adding topological features (T1–T3) to the strongest baseline set (B0–B6)
  raises AUROC on the **hard subset** (`mem_Sf = 0`), on the out-of-time test split (steps
  4000–5999), at S_f = 10000. Criterion: the paired-bootstrap 95% CI (2000 resamples) for
  ΔAUROC (rung 7 − rung 6, logistic regression) excludes 0 on the positive side, for both
  negative-sampling seeds.
- **H2 (direction).** Sequences that will become memorized show topological compression at
  S_f compared with loss-decile-matched non-memorized ones: fewer H1 bars, higher mean H1
  persistence, lower H1 entropy, higher mean H0 death. Reported as Hedges' g with bootstrap
  CIs. The direction is secondary to H1.
- **H3 (dynamics).** Per-sequence topological velocity (W2 between H1 diagrams) from the
  pre-exposure checkpoint to S_f is larger for future-memorized sequences: Cliff's δ > 0,
  with a CI that excludes 0.
- **H4 (learned, not text).** H1 survives adding step-0 topology (T0) to the baseline: the
  CI of ΔAUROC for rung 8 (full + T0) − rung "6+T0" (B0–B6 + T0) excludes 0 on the positive
  side.

**Success criterion for "strong preliminary evidence":** H1 holds at S_f = 10000 **and** at
S_f = 23000 on pythia-1b-deduped, **and** at S_f = 10000 on pythia-410m-deduped, with all
controls in §5 clean. Anything less is reported as partial.

**Go/no-go after the pilot (1b, S_f = 10000):** stop and report a negative result if the
ΔAUROC CI includes 0 **and** the topology-only AUROC within loss deciles is ≈ 0.5
(every decile CI includes 0.5).

**Fixed analysis choices:** see `configs/memtopo.yaml` and the module docstring of
`memtopo/analysis.py`. Normalisation is to centre each cloud and divide by its median
pairwise distance (primary); raw clouds are secondary. Layers are 3, 5, 8, 10, 13, 16 for
1b, at the same relative depths for other sizes.

---

## 1. What was run
_To fill: models, S_f, T (from `gates.json: epoch_boundary`), pool and case-control sizes, GPU hours._

## 2. Hypotheses → verdicts
_To fill from `results/summary.md` and figures 1, 3, 4, 5._

| | Verdict | Numbers | Figure |
|---|---|---|---|
| H1 | | | fig1 |
| H2 | | | fig4 |
| H3 | | | fig5 |
| H4 | | | fig1 |

## 3. Ladder (hard subset and full set) and the primary Δ

## 4. Comparison points
- B0 precision/recall vs Biderman et al. (12B: 0.918 / 0.500 at 23M sequences; 6.9B: 0.919 / 0.513).
- Balanced accuracy, 128 examples, vs "Don't Forget" (84.4% loss only, 90.6% best, step 10k,
  Pythia-1B; their sequence selection differs).

## 5. Controls (state any failure first)
1. Shuffled labels: AUROC ≈ 0.5 for every rung.
2. Step-0 (text-only) topology: T0 rung, and H4.
3. Gaussian floor: real vs matched-Gaussian H1 counts and entropy.
4. Leakage: no feature uses a checkpoint after S_f (asserted in code; listed in `summary.md`).
5. Time split: train on steps 1000–3999, test on 4000–5999.
6. Two negative-subsample seeds.
7. Label fragility: `mem_final` recomputed at step 143000.

## 6. Effect sizes

## 7. Deviations from the brief
Decided before running; see the experiment README, "Design decisions":
teacher-forced extractability (checked against greedy decoding), the epoch boundary found
by shingle overlap, our own Wasserstein-2, the extra rung "6+T0" for H4, hard-subset models
fit on hard rows, the probe stored as npz.

## 8. Related work found (section 12)
_To fill from `results/related_work.csv` (`python -m memtopo.cli novelty`)._

## 9. Next step
