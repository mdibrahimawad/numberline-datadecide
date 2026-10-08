# E04: Exact 100B training-stream α

**Question:** what are α and the number counts of *exactly* the 100B tokens each
DataDecide model was trained on?
**Status:** done for all 25 recipes (2026-10-05 → 10-08). 14 recipes were counted on
Modal and the 11 largest on RunPod CPU pods.
**Outputs:** `results/corpus_alpha_datadecide/exact_100b_<recipe>/`
(`summary.json`, `counts_seed_2.csv` = c(n) for n = 0..10000; c4 and dolma1_7 also have
`counts_seed_{4,5,14,6198}.csv`, and `counts_full_recipe.csv` / `counts_union_of_samples.csv`),
plus the table `results/corpus_alpha_datadecide/alpha_seed2.csv` (one row per recipe).

## Method
Rebuild the training stream from E03's reproduction: the recipe's files in training
order, cut into 2048-token chunks per file, PCG64 shuffle with the data seed, first
69,369 × 704 chunks (with an epoch reshuffle at seed+1 for the 3 recipes smaller than
100B tokens). Download only the needed byte ranges of each `.npy` file from
`allenai/DataDecide-data-recipes`, decode each chunk with the recipe's tokenizer, and
count with `_count_text`. Then α_OLS, α_MLE, numbers seen.
- **c4:** 5 data seeds (2, 4, 5 = the three trained c4 runs; plus 6198 and 14) and the
  **full recipe** (138B tokens), in one pass.
- **dolma1_7:** 5 seeds (pairwise overlap only 5.8 %), plus their union (445B tokens).
- **All 25:** seed 2 only (`--skip-unused`: only chunks seed 2 trained on are read).

## How to run
```bash
# Modal
modal run modal_app/datadecide_alpha_app.py::exact_samples --recipe c4 --dry-run     # cost estimate
modal run modal_app/datadecide_alpha_app.py::exact_samples --recipe c4
modal run modal_app/datadecide_alpha_app.py::exact_all --smallest-first --dry-run    # per-recipe size + cost
modal run modal_app/datadecide_alpha_app.py::exact_all --smallest-first              # resumes; --dry-run alone rebuilds alpha_seed2.csv
# RunPod (step-by-step guide: docs/runpod.md)
python -m runpod_jobs.exact_alpha --recipes <list> --usd-per-hour 0.48 \
    --upload-hf numberline-alpha-results --delete-pod-when-done --max-hours 16
python -m runpod_jobs.fetch_results          # laptop: download + rebuild alpha_seed2.csv
python tests/test_datadecide_sampling.py; python -m pytest -q tests/test_runpod_exact_alpha.py
```

## Headline result
- **A 100B training sample has the same α as the whole recipe.** On c4, the 5 seeds are
  within 0.0003 (OLS) and 0.00002 (MLE) of the full recipe. On dolma1_7, 5 nearly
  independent seeds agree to 0.003 (OLS) and 0.00015 (MLE).
- **The data seed doesn't matter:** seeds 2 and 6198 differ by ≤ 0.0013 (OLS),
  0.00004 (MLE) and 0.04 % in numbers seen.
- **Differences between recipes are 75–900× the sampling noise**, so α is a real
  property of each recipe: α_OLS −1.43 … −1.76, α_MLE −1.05 … −1.19, numbers seen
  1.0 … 2.0 billion.

Per-recipe table, seed tables and cost notes: [DETAILS.md](DETAILS.md).
