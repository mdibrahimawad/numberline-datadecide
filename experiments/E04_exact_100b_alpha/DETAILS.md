# E04 details

## c4: 5 training streams vs the full recipe (138.4B tokens)

| Data | Tokens | Numbers seen | α_OLS | Δ_OLS vs full | α_MLE | Δ_MLE vs full |
|---|---|---|---|---|---|---|
| Full c4 recipe | 138.44B | 1,932,366,147 | −1.56222 | — | −1.074342 | — |
| Seed 2 (released `seed-default`) | 100.02B | 1,396,279,342 | −1.56253 | −0.00031 | −1.074332 | +0.000011 |
| Seed 4 (aux seed) | 100.02B | 1,395,879,273 | −1.56201 | +0.00022 | −1.074358 | −0.000016 |
| Seed 5 (aux seed) | 100.02B | 1,395,978,330 | −1.56236 | −0.00014 | −1.074351 | −0.000009 |
| Seed 6198 | 100.02B | 1,395,966,441 | −1.56245 | −0.00023 | −1.074347 | −0.000005 |
| Seed 14 | 100.02B | 1,396,023,501 | −1.56247 | −0.00024 | −1.074354 | −0.000012 |
| Mean ± s.d. | | | −1.56236 ± 0.00021 | | −1.074349 ± 0.000010 | |

Each c4 seed reads 72 % of the recipe, so the samples overlap heavily. Even so, the 28 % each
run skipped didn't move α. Compute cost: $1.31 on Modal.

## dolma1_7: 5 nearly independent streams (recipe ≈ 1.7T tokens)

| Seed | Numbers seen | α_OLS | α_MLE |
|---|---|---|---|
| 2 | 1,848,383,391 | −1.47029 | −1.128354 |
| 4 | 1,848,400,951 | −1.47303 | −1.128359 |
| 5 | 1,848,694,511 | −1.47181 | −1.128502 |
| 6198 | 1,848,755,847 | −1.47158 | −1.128397 |
| 14 | 1,848,084,563 | −1.47309 | −1.128369 |
| Mean ± s.d. | | −1.47196 ± 0.00116 | −1.128396 ± 0.000062 |
| Union of the 5 (445B tokens, 26 % of the recipe) | 8,225,596,455 | −1.47161 | −1.128400 |

Any two seeds share only 5.83 % of their chunks, so the agreement can't come from
overlap. Compute cost: $5.67 on Modal. The full dolma1_7 recipe was not counted; the
union is the best direct estimate.

## All 25 recipes, seed 2 (`alpha_seed2.csv`)

| Recipe | α_OLS | R² | α_MLE | Numbers seen | Recipe size (tokens) | Share of recipe read | Counted on |
|---|---|---|---|---|---|---|---|
| dolma1_7 | -1.4703 | 0.806 | -1.12835 | 1.848B | 1,715B | 6% | Modal |
| dolma1_7-no-code | -1.5689 | 0.794 | -1.11232 | 1.743B | 1,432B | 7% | Modal |
| dolma1_7-no-math-code | -1.5720 | 0.790 | -1.09562 | 1.619B | 1,349B | 7% | RunPod |
| dolma1_7-no-reddit | -1.4692 | 0.806 | -1.12677 | 1.897B | 1,635B | 6% | RunPod |
| dolma1_7-no-flan | -1.4763 | 0.806 | -1.12941 | 1.851B | 1,699B | 6% | RunPod |
| dolma1_6plus | -1.4290 | 0.801 | -1.12304 | 1.930B | 1,885B | 5% | RunPod |
| c4 | -1.5625 | 0.770 | -1.07433 | 1.396B | 138B | 72% | Modal |
| fineweb-pro | -1.7617 | 0.795 | -1.05033 | 1.487B | 83B | 121% | Modal |
| fineweb-edu | -1.6988 | 0.778 | -1.05409 | 1.720B | 192B | 52% | Modal |
| falcon | -1.5052 | 0.789 | -1.08966 | 1.890B | 456B | 22% | RunPod |
| falcon-and-cc | -1.5813 | 0.792 | -1.08582 | 1.628B | 1,055B | 9% | RunPod |
| falcon-and-cc-qc-10p | -1.6786 | 0.773 | -1.15997 | 0.999B | 146B | 68% | Modal |
| falcon-and-cc-qc-20p | -1.6794 | 0.793 | -1.14048 | 1.101B | 286B | 35% | Modal |
| falcon-and-cc-qc-orig-10p | -1.6482 | 0.795 | -1.15723 | 1.168B | 148B | 68% | Modal |
| falcon-and-cc-qc-tulu-10p | -1.6907 | 0.816 | -1.09972 | 1.749B | 129B | 78% | Modal |
| dclm-baseline | -1.5965 | 0.798 | -1.12978 | 1.485B | 3,857B | 3% | RunPod |
| dclm-baseline-qc-7p-fw2 | -1.6634 | 0.783 | -1.12272 | 1.658B | 508B | 20% | RunPod |
| dclm-baseline-qc-7p-fw3 | -1.7193 | 0.760 | -1.10897 | 1.869B | 218B | 46% | Modal |
| dclm-baseline-qc-fw-3p | -1.7307 | 0.746 | -1.09245 | 1.975B | 118B | 85% | Modal |
| dclm-baseline-qc-fw-10p | -1.7230 | 0.766 | -1.09112 | 1.868B | 98B | 102% | Modal |
| dclm-baseline-qc-10p | -1.6142 | 0.787 | -1.19456 | 1.766B | 82B | 122% | Modal |
| dclm-baseline-qc-20p | -1.6060 | 0.783 | -1.16939 | 1.636B | 171B | 58% | Modal |
| dclm-baseline-25p-dolma1.7-75p | -1.5130 | 0.806 | -1.12957 | 1.701B | 2,246B | 4% | RunPod |
| dclm-baseline-50p-dolma1.7-50p | -1.5408 | 0.803 | -1.12904 | 1.602B | 2,814B | 4% | RunPod |
| dclm-baseline-75p-dolma1.7-25p | -1.5770 | 0.799 | -1.12994 | 1.534B | 3,336B | 3% | RunPod |

"Share of recipe read" above 100 % means the recipe is smaller than 100B tokens and
training went into a second epoch (reshuffled with seed + 1). The counter reproduces
that (`training_chunk_indices` in `src/datadecide_sampling.py`).

## Engineering notes
- **Membership:** for each chunk, a small integer code records which seeds use it
  (`membership_codes`), so all seeds are counted in one pass over the data.
  `--skip-unused` reads only chunks some requested seed uses.
- **Modal (`exact_all`)**: tasks of 256M tokens; resumes from saved `summary.json`;
  `--smallest-first` ran cheap recipes first under a fixed budget. The first workspace
  was disabled when its budget ran out after 8 recipes; a new workspace finished 14.
- **RunPod (`runpod_jobs/exact_alpha.py`)**, 11 largest recipes (≈ 42 TB of token files):
  - workers sized from the pod's real vCPU count and RAM (cgroup-aware);
  - CDN-direct HTTP range reads with 8 parallel connections per slice;
  - per-slice cache with atomic writes, so a crash resumes without recounting;
  - stall watchdog, retries, `--max-hours` hard limit;
  - each recipe uploaded to `numberline-alpha-results` and verified;
  - pods share the recipe list through claim files in that dataset;
  - a pod deletes itself only after all its recipes are verified.

  Two pods (16 and 32 vCPU) finished overnight. Pod 2's network was slower
  (0.23 GB/s vs 0.58 GB/s).
- **Known oddity:** pod 1 did not delete itself at the end. Its SSH session was closed by
  the remote host and it still showed as running; it was stopped by hand. Root cause not
  found.
