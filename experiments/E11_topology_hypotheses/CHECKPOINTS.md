# E11 tier 2: number topology across pretraining checkpoints (pre-registered 2026-10-15)

**Written before any checkpoint was downloaded.** Code: `src/e11_checkpoints.py` (extraction + copy
behaviour, run on the owner's laptop) and `src/e11_ckpt_analysis.py` (written before the data
arrives; implements exactly the measures below).

## Novelty checks (3 literature searches per hypothesis, 2026-10-15)
| Candidate | Closest prior work found | Kept? |
|---|---|---|
| Cross-model convergence grows during training | "Global Divergence, Local Convergence" (2026): local similarity converges in the first ~4 % of Pythia training, global by ~25 % | dropped (known) |
| Early topological shock in pretraining | Malhotra et al. 2026 (fine-tuning only); Li et al. 2025 (spectral phases over pretraining) | secondary only (partly known) |
| **C1 developmental order digit → value for numbers** | no study tracks number representations / number-line emergence over checkpoints (3 searches); our E11 H6 found the digit → value switch over *depth* | **kept** |
| **C2 do rare numbers catch up?** | frequency → earlier learning of surprisal (Chang et al., TACL); Bernas et al. (ICML 2026) report frequent tokens in tighter regions; no direct checkpoint test of rare-token geometry catching up (3 searches) | **kept** |

## Data
- **Pythia-1.4B** (Pile, fixed data order), 16 checkpoints: steps 0, 64, 256, 512, 1k, 2k, 4k, 8k, 13k,
  23k, 33k, 53k, 73k, 103k, 123k, 143k (final).
- **OLMo-2-1B**, 12 stage-1 checkpoints spread log-uniformly over the released list (chosen by the
  script, not by hand).
- Numbers: the 2,100-number subset (1–100 and 200 ordinary 4-digit numbers per frequency tenth, SlimPajama
  counts as the frequency proxy; the Pile order of numbers by frequency is assumed similar). Original
  prompt texts, last token, levels at the E10 relative depths (~25 %, ~50 %, ~75 %, ~84 %, last).
- Behaviour: teacher-forced log-probability of the correct number after `=`, and first-token accuracy
  (300 prompts).

## C1: developmental order (primary layer: ~84 % depth; others reported)
Measures per checkpoint, on the 2,000 four-digit numbers (10 nearest neighbours among them):
`share_same_last` (digit topology), `share_within_50` and `share_same_leading` (value topology),
`ari_leading` (average-linkage, 9 clusters, purity by leading digit).
- **C1a** value topology rises over training: Spearman(log step, share_same_leading) > 0 and
  Spearman(log step, share_same_last) < 0 at the ~84 % layer, both models.
- **C1b** the change is abrupt rather than gradual: a two-segment (step) fit over log step beats a
  straight line (lower residual, AIC) for `share_same_leading`; the step location is reported.
- **C1c** value topology appears before the copy behaviour saturates: the step at which
  `share_same_leading` reaches half of its final value precedes the step at which copy log-probability
  reaches half of its final gain.

## C2: do rare numbers catch up? (all layers; primary ~84 %)
Per checkpoint: half-merge radius ratio r½(rarest tenth) / r½(most frequent tenth), 2 × 120 numbers,
K = 20 subsamples, raw and scale-free (as E11 H1/H2), with the within-checkpoint permutation of tenth
labels (100×) giving p per checkpoint.
- **C2a** the crowding (ratio < 1) is present at the final checkpoint (replicates H1 in Pythia).
- **C2b** shape of the trajectory, two-sided: "catch-up" = the ratio moves toward 1 over the second
  half of training (Spearman(log step, ratio) > 0 over checkpoints ≥ 10 % of training);
  "divergence" = moves away from 1. Either is a finding; no change is a finding too.
- **C2c** the crowding appears only after the number line exists: the first checkpoint with p < 0.05 is not
  earlier than the C1 half-way step.

## Secondary (reported, not decisive)
H1 Wasserstein velocity of the number cloud between consecutive checkpoints and its early
concentration C (first third of log-step range) against a checkpoint-order permutation null
(Malhotra-style); 41-feature probe of training step (hidden-progress).

## Decision rules
C1a passes if both Spearman signs hold in both models. C1b passes if the step fit wins in both models.
C1c passes if the order holds in both models. C2 outcomes are reported as observed. Checkpoints that fail
to download are reported, not replaced.

## Results (run 2026-10-16 with `src/e11_ckpt_analysis.py` as committed in `a79a73d`; no deviations)
All 28 checkpoints extracted (Pythia 16, OLMo-2 12; copy first-token accuracy 0.00 at step 0 → 1.00 / 0.99 final).
Tables: `results/e11_ckpt/` (`C1_summary.csv`, `C2_summary.csv`, `secondary_summary.csv`, per-checkpoint `*_knn.csv`,
`*_C2_per_checkpoint.csv`); figure `fig_trajectories.png` (~84 % depth).

**Pre-registered decisions (primary ~84 % layer):**
| Test | Pythia-1.4B | OLMo-2-1B | Decision |
|---|---|---|---|
| C1a value topology up (ρ same leading > 0) **and** digit topology down (ρ same last < 0) | +0.69 / **+0.87** | +0.22 / **+0.62** | **FAIL** (the last-digit share does not fall: at step 0 every share is at chance, so there is no early digit stage to lose; it rises late in training) |
| C1b step fit beats a line (AIC) | yes, break 2,000–4,000 | yes, break 300–10,000 | **PASS**: the number line appears abruptly |
| C1c value topology half-way before copy log-prob half-way | 2,307 vs 492 | 1,660 vs 835 | **FAIL**: copy log-probability improves first |
| C2a crowding at the final checkpoint (p < 0.05) | 0.982, p = 0.26 | 0.985, p = 0.08 | **not at ~84 %**; present at ~25 % and ~50 % in both (0.96–0.97, p ≤ 0.02) and ~75 % in OLMo-2 |
| C2b trend over late training (≥ 10 % of steps) | ρ +0.89: toward 1 (**catch-up**) | ρ −0.2 (ratios ≈ 0.98–1.00) | Pythia: rare numbers catch up late; OLMo-2: at ~25–75 % ρ −0.5 to −0.8 (crowding grows) |
| C2c crowding first significant after the C1 half-way step | first at step 256 (before the number line) | first at 10,000 (the first checkpoint with a number line) | Pythia no; OLMo-2 yes (coarse spacing) |

**Secondary:** early concentration of the H1 Wasserstein path is not above the checkpoint-order null in Pythia
(p 0.29–0.52); in OLMo-2 it is at ~25 % and ~50 % (p ≈ 0.02). The 41-feature probe predicts the training step
(leave-one-checkpoint-out Spearman 0.88–0.90 at ~75 %–last in Pythia; 0.49–0.78 in OLMo-2 except its last layer).

**Exploratory observations (not tested, from the same tables):**
1. In Pythia the number line, the leading-digit neighbourhoods and copy accuracy all appear in the same window,
   steps 2,000 → 4,000 (value share 0.14 → 0.55, same leading digit 0.36 → 0.95, first-token accuracy 0.12 → 0.78);
   the log-probability (C1c) rises gradually before it.
2. Leading-digit *clusters* (ARI, 9 groups) form and then dissolve: Pythia ARI 0.65 at step 23,000 → ≈ 0 from
   33,000 on, while neighbours keep the same leading digit (a clustered line becomes a smooth line); OLMo-2 ARI 0.75 at
   40,000 → 0.17 at the end.
3. OLMo-2's number line *weakens* after step 40,000 (value share 0.86 → 0.54, same leading 0.95 → 0.77) while the
   last-digit share rises from chance to 0.39; copy accuracy stays ≈ 1. Pythia shows a milder late rise of the
   last-digit share (0.13 → 0.19–0.26).
4. Pythia's rare-number crowding is largest mid-training (ratio 0.91–0.92 at 13,000–23,000) and gone at the end.
