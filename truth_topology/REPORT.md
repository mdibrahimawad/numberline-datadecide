# Report: topology of truth representations across Pythia pretraining (pilot)

Run 2026-10-11 on one RunPod L40S. All numbers are from `results/REPORT_tables.md` and the per-run
`results/<model>/<input>/*.csv`; figures are in `figures/`. Hypotheses and every analysis choice
were fixed in `results/README.md` before any real data; the two changes made after the pilot are
listed there under "Changes after seeing data" (and in §7 below).

## 1. What was run

- **Models:** Pythia-1.4B-deduped (primary) and Pythia-410M-deduped, each at Ravfogel et al.'s
  15 checkpoints (0 … 143k), fp32, all 25 layers.
- **P1 (single statements):** 1,504 statements from three topics: `cities` (275 + 275),
  `sp_en_trans` (177 + 177) and `companies_true_false` (300 + 300). Read position: the final period.
- **P2 (Ravfogel replication):** run on 1.4B only, with 1,200 CounterFact sequences. Each is three
  context facts, all true or all false, then a fourth true fact cut before its attribute.
- **Measures:**
  - behaviour (B1, B2);
  - linear probes (L1 logistic, L2 mass-mean, L3 cross-topic);
  - class-shape persistent homology (T1: 64 subsamples × 160 points, Vietoris–Rips H0/H1, the 41
    Fay et al. features);
  - topological velocity (T2, W2/L∞);
  - MST truth homophily (T3).
- **Pythia-6.9B** (stage-2 replication) was not run.

## 2. Hypotheses

Reference layer (pre-registered rule: best mean L1 over the last three checkpoints) = **layer 13**
in both models.

### H1, timing: **inconclusive**
- **1.4B:** the pre-registered rule is met. T3b's onset is 10k, two checkpoints after L1 and B1
  (both 3k).
- **410M:** the rule is not met. T3b's onset is 5k, the same as L1 (B1 is at 3k).
- **The lag looks like sensitivity, not different timing.** T3b and L1 rise together on the same
  curve from 10k onwards (fig1, fig2).
- **T1's onset is unstable across subsample seeds:** 1k vs 10k (1.4B), 100k vs 10k (410M).

**No measure leads behaviour or the linear probe.**

### H2, abrupt vs smooth: **not supported** for topology as pre-registered
- **H1 velocity is not concentrated** (reference layer, normalised, init transition excluded):
  - 1.4B: C = 0.49 vs null 0.46, p = 0.13 / 0.17 (seeds 0 / 1).
  - 410M: C = 0.53 vs 0.46, p = 0.02 / 0.02. Significant but tiny, and the peak/median ratio is
    1.7 (rule: > 3).
- **H1 velocity is highest at the very start** (init → 3k) and roughly flat after that (fig1, bottom).
- **H0 velocity *is* sharply concentrated** (C = 0.92–0.96, p < 0.001, peak at 10k–20k, peak/median
  6–38). But it tracks the plain geometric baseline almost exactly: Spearman ρ = 0.93 with centroid
  drift and 0.87 with change in mean pairwise distance (1.4B, layer 13). So it adds nothing beyond
  geometry.
- **What it reflects is a large geometric event at 10k–20k.** Raw total variance at layer 13 grows
  about 100× between 5k and 20k, and the normalised cloud changes most in exactly those two
  transitions. The probe's steepest rise (L1 0.55 → 0.83) is in the same window.

### H3, shape: **inconclusive (weak, and confounded at initialisation)**
- **At checkpoints where L1 > 0.8,** T1 beats its mixed-vs-mixed null (z > 1.645 for both seeds) at
  4 of 10 checkpoints for 1.4B and 3 of 6 for 410M.
- **The T1 gap is small and noisy:** 0.09–0.19 for 1.4B after 10k, against a ceiling of about 0.3.
- **The individual features show a consistent late pattern.** From 100k on, true-statement clouds
  have **fewer H1 bars and lower H1 entropy** than false ones:
  - 1.4B: Hedges' g ≈ −0.4 to −0.7, 95% CIs excluding 0 at 100k, 120k, 130k and 143k.
  - 410M: g ≈ −0.6 to −0.8, CIs excluding 0 at 100k–143k.
  - 1.4B also has slightly smaller mean H0 death (g ≈ −0.4) at 120k–143k.
- **But the same sign is already present in the untrained model:** g(H1 bars) = −0.66 (1.4B) and
  −1.12 (410M) at step 0. Part of the shape difference therefore comes from the two item pools
  themselves, not from learning. With 64 overlapping subsamples per class, these effect sizes are
  also optimistic.

### H4, local truth: **partly supported (clear in 410M, mixed in 1.4B)**
- **Within-topic truth structure is strong and grows with training.** T3b is significant after
  max-statistic correction over layers from 1k–3k on. At layer 13 it rises from 0.44 to 0.75 (1.4B)
  and from 0.46 to 0.67 (410M), z ≈ 19 / 12 at the end (fig3).
- **Cross-topic truth structure after removing topic means (T3c) is weak:**
  - 410M: z < 2 at layer 13 at every checkpoint after 512.
  - 1.4B: mixed, with z between 0.3 and 3.4 after 20k and above 2 at 4 of 10 checkpoints.
- **The linear version disagrees in degree.** L3 (train on one topic, test on another) reaches
  0.75–0.78 (1.4B) and 0.59–0.64 (410M), against 0.93 / 0.86 within topic. So a partly shared linear
  truth direction exists, especially in the larger model, while nearest-neighbour structure is
  mostly within topic.
- **Reading:** truth is organised mainly locally, with a weaker global component that is larger in
  the larger model.

## 3. Onset table (reference layer 13, first of two consecutive checkpoints above the null 95th percentile)

| measure | 1.4B | 410M |
|---|---|---|
| B1 behaviour (log-likelihood AUC) | 3k | 3k |
| L1 logistic probe | 3k | 5k |
| L2 mass-mean probe | 512* | 512* |
| L3 cross-topic | 3k | 5k |
| T1 shape (seed 0 / 1) | 1k / 10k | 100k / 10k |
| T3b within-topic homophily | 10k | 5k |
| T3c cross-topic homophily | 130k | never |

\* L2 "onsets" at 512 sit at AUC ≈ 0.50–0.56. The null comes from only 5 label shuffles, so this is
probably a null-width artefact. Treat onsets that close to the null as unreliable.

## 4. Replication of the prior work

- **P1:** the L1 trajectory has the qualitative shape Ravfogel et al. report. It is at chance until
  about 3k, rises steadily through 10k–40k and plateaus at 0.93 (1.4B) and 0.85 (410M), with
  behaviour (B1) rising alongside.
- **P2, 1.4B vs their 6.9B numbers (fig4):**

  | at step 143k | ours (1.4B) | Ravfogel (6.9B) |
  |---|---|---|
  | probe AUC | 0.78 | 0.83 |
  | memorization | 0.24 | 0.88 |
  | ΔH | 0.29 | 0.52 |

  - The *shapes* match. ΔH is ≈ 0 until 512–1k and then rises; memorization rises and flattens; the
    probe climbs steadily.
  - The levels are lower, as expected for a model 5× smaller, with our relation subset and a
    first-token top-1 criterion.
  - **One difference:** our P2 probe is already above chance at step 0 (AUC 0.55, p < 10⁻¹⁷), while
    theirs is below chance (0.38). The true- and false-context attributes differ in their word
    distribution, which a random network can pick up.

## 5. Controls

- **Step 0 (untrained model), P1, after Bonferroni over layers:** L1, L2, L1-within, B1, T3b and T3c
  are at chance in both models. Exceptions:
  - L3 at one layer in 1.4B (min p = 1e-4) and T1 at one layer in 410M (min p = 5e-9). Both nulls
    rest on few repeats (5 shuffles or 5 pseudo-class splits) fitted with a normal distribution, so
    these p-values are fragile.
  - **T3b at layer 13 is significantly *below* chance at step 0** (0.44, z = −4.8 for 1.4B; −3.2 for
    410M). True and false statements come in near-identical pairs ("X is in France" / "X is in
    Japan"). An untrained model puts each pair together, which links opposite labels. The
    within-topic permutation null ignores this pairing, so T3b's baseline is below 0.5. The later
    rise to z ≈ 19 is far larger than this offset.
- **Topic structure (T3a)** is ≈ 1.0 at every checkpoint, including step 0. That is expected,
  because topics differ in their words.
- **Shuffled labels:** the L1/L2 null bands (grey in fig1) stay around 0.5 throughout.
- **Gaussian floor:** isotropic Gaussian clouds of the same size (m = 160, d = 2048) produce
  255 ± 44 H1 bars. Most H1 bars at this sample size and dimension are therefore not "structure",
  which is why H1 counts are only compared *between* classes, never read in absolute terms.
- **Lexical confounds:** the length-only probe gives AUC 0.51–0.54 per topic, and the words-only
  probe at most 0.40, so no length or word shortcut. `larger_than` failed this check and was removed
  (§7).

## 6. Effect sizes, in plain terms

- **Linear separability within topic** ends large: AUC 0.93 (1.4B).
- **Within-topic MST homophily** ends moderate: 75% of nearest-neighbour merges join same-truth
  statements, vs about 50% by chance.
- **Class-shape (T1) differences** are small: accuracy gap ≤ 0.19 against a noisy null, with feature
  effects of |g| ≈ 0.4–0.8 that are partly present at initialisation.
- **H1 velocity concentration**, where significant, is tiny: C 0.53 vs 0.46.

## 7. What failed, changed or was skipped

- **`larger_than` was replaced by `companies_true_false` after the first pilot.** A words-only probe
  solves it (AUC 0.97), and the untrained model reached AUC ≈ 1.0. Recorded in `results/README.md`.
- **The automatic step-0 gate no longer includes T3a,** because topic structure is expected at init.
- **Two bugs were fixed during the run:**
  - extraction failed on the pod's `transformers` version (padding token); it now pads by hand;
  - stale pilot files are now discarded automatically via an `items.csv` hash.
- **P2 topological measures (T1, T3) cannot be interpreted.** Each P2 sequence has a matched
  partner with the opposite label and the same context facts, so T3b is 0.30 (partners are nearest
  neighbours) and T1 shows no gap.
- **Not run:** Pythia-6.9B (stage 2), the optional HSIC measure (L4), and the dense early
  checkpoints (64, 128, 256).
- **Weak nulls:** the linear nulls use 5 shuffles and the T1 null 5 pseudo-class splits. That is
  enough for the trajectories, but marginal for p-values and onsets.

## 8. Suggested next experiment

Rerun T1 and T3b with **pair-aware nulls**: permute labels only *within* each true/false pair and
use ≥ 100 pseudo-class splits. Also run the 6.9B model. That would decide whether the late "true
clouds have fewer loops" effect and the T3b lag behind L1 are real.

## Related work found

Novelty check: Semantic Scholar citations of the four anchor papers, flagged automatically by title
and abstract keywords (`results/related_work.md`; 28 flagged, **titles only, not read**). None of
the flagged titles describes persistent homology of true/false representations or tracks truth
representations across pretraining with topological methods. Closest by title, worth reading
before any claim of novelty:
- "The Confidence Manifold: Geometric Structure of Correctness Representations" (arXiv 2602.08159)
- "The Shape of Beliefs: Geometry, Dynamics, and Interventions along Representation Manifolds"
  (arXiv 2602.02315)
- "Semantic Alignment of AI Models: Concept Collapse, Checkpoint Dynamics, and Cross-Lingual
  Transfer" (arXiv 2608.01585)
- Malhotra et al. (arXiv 2606.19542), whose design this pilot reuses.
