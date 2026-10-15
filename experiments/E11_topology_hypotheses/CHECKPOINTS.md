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
