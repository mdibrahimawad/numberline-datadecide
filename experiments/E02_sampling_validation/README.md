# E02: Sampling validation (does a random sample give the full-corpus α?)

**Question:** before counting DataDecide data, check that α from a random sample of a
corpus equals α from counting the whole corpus.
**Status:** SlimPajama done (2026-10-05, Modal CPU, ~$1). The LLM-JP part is built but was
**never run**.
**Outputs:** `results/sampling_validation/ground_truth.json` (committed).
The SlimPajama run outputs (`summary_table.csv`, `summary.json`, `runs.csv`,
`alpha_vs_size.png` under `results/sampling_validation/slimpajama/`) are on the owner's
laptop. **To do:** commit them
(`git add results/sampling_validation && git commit && git push`).

## Method
- **Ground truth:** full-pass counts that already existed (E00):
  SlimPajama-627B reupload (`gmongaras/SlimPajama-627B_Reupload`) and LLM-JP corpus v3,
  in `results/corpus_alpha_full/`.
- **Samples:** about 10M, 30M, 100M, 300M and 1B tokens (tokens ≈ text bytes / 4),
  5 independent replicates each, nested within a replicate.
  - Scheme A `uniform_files`: pick a file uniformly, then a random row slice.
  - Scheme B `size_proportional`: pick row groups (cut to ≤ 16 MB) with probability ∝
    size, weighted by 1/size (Hansen–Hurwitz) so every byte counts equally. An
    unweighted variant is reported as a check.
- **Per sample:** α_OLS, α_MLE, and a 500× bootstrap over row groups for a 95 % CI.
- **Acceptance** (scheme B): mean |α − α_full| ≤ 0.01, and CIs cover α_full in ≥ 60 % of replicates.

## How to run
```bash
python -m src.sampling_validation                                         # ground truth from the full-pass CSVs (seconds)
modal run modal_app/sampling_validation_app.py --corpus slimpajama --dry-run   # prints row-group sizes, download, cost
modal run modal_app/sampling_validation_app.py --corpus slimpajama
modal run modal_app/sampling_validation_app.py --corpus llmjp --dry-run        # not run yet
python tests/test_sampling_validation.py                                   # offline
```

## Headline result
- **α_MLE passes at every size, down to 10M tokens.** At 1B, all 5 samples are within
  0.006 of the full value (−1.0647).
- **α_OLS fails at every size, even 1B** (all 5 samples ≈ 0.03 steeper than −1.5567).
  This is a bias of the OLS fit itself on small counts, not a sampling problem: OLS drops
  numbers with zero count and takes logs of small counts.
- **Consequence:** report α_MLE for any sample, and compute α_OLS only from exact,
  large streams (≥ 100B tokens; this became E04).

Numbers and the explanation of the bias: [DETAILS.md](DETAILS.md).
