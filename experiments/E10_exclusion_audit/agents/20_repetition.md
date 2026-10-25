# 20 — Data repetition audit (epochs, file upsampling, effect on comparability and on counting)

Blind audit: no alpha–beta relationship was computed or consulted. I only used data-side files (`results/corpus_alpha_datadecide/*`, `configs/datadecide_data_map.json`) and the loader code `src/datadecide_sampling.py`. I did not open any beta file. `results/datadecide/count_features.csv` contains beta columns, so I used it only for its header and recomputed every feature here from raw counts.

Prior claim checked: fineweb-pro, dclm-baseline-qc-10p and dclm-baseline-qc-fw-10p train for more than one epoch. **Verified independently. They are the only 3 such recipes** (same numbers as `01_training.md` F1).

## 0. How repetition arises (evidence)

* Token budget: 69,369 steps × 704 seqs = **48,835,776 chunks** of 2048 tokens = 100.016B (`docs/datadecide_sampling.md` §1, `experiments/E03_datadecide_training_sampling/DETAILS.md:10`).
* Each file is cut separately into `floor(len/2048)` chunks. The order is a PCG64 shuffle of all N pool chunks, and "a new epoch reshuffles with seed+1" (`docs/datadecide_sampling.md` §4.2–4.4).
* The repo's reproduction is `src/datadecide_sampling.py:187-201` (`training_chunk_indices`). It takes the shuffled `indices[:need]` and, while `need > 0`, moves to the next epoch with `seed+epoch`. So when N < 48.8M:
  * epoch 1 uses all N chunks once;
  * epoch 2 uses the first 48,835,776 − N chunks of a fresh permutation.
  * Each chunk is therefore used 1 or 2 times.
* `membership_codes` (`:220-244`) stores the per-chunk multiplicity. Its `base` = max multiplicity + 1 is saved in `exact_100b_<recipe>/summary.json → membership`.
* File-level upsampling: `configs/datadecide_data_map.json → recipes.<r>.repeated_paths` lists
  * the two Wikipedia/Wikibooks files `preprocessed/olmo-mix/v1_6-decontaminated/wiki/.../part-{0,1}-00000.npy` ×2 for dolma1_7, -no-code, -no-math-code, -no-reddit and -no-flan (n_paths − n_unique_paths = 2);
  * part-0 only for dclm-baseline-25p-dolma1.7-75p (1016 vs 1015).
  * No other recipe has duplicate paths. This matches `docs/datadecide_sampling.md` §2 "Upsampling".

Definitions:
* N = `membership.n_chunks` (pool size in 2048-token chunks).
* **Epochs** e = 48,835,776 / N.
* **Repeated fraction (epoch)** = max(0, 48,835,776 − N) / 48,835,776. This is the share of trained tokens that are a second sighting of an identical chunk.
* For duplicated wiki files, identical chunks also occur at two different indices. Both copies are drawn with probability ≈ f², where f = min(e, 1). So the exact-repeat share is ≈ W·f²/2048/48.8M, with W = unique wiki tokens.
* W comes from window ends in `<recipe>/windows.json`: part-0 ≥ 2,404,749,957 tokens and part-1 ≥ 1,167,333,744, so W ≥ 3.57B. I used an upper estimate of 3.75B. The Hub file size could not be fetched (HTTP 403 via proxy).

## 1. Per-model table (all 25)

Values come from `results/corpus_alpha_datadecide/exact_100b_<r>/summary.json` (`membership.n_chunks`, `membership.base`) and `configs/datadecide_data_map.json`. The calculation script output is reproduced in §3.

| # | recipe | pool chunks N | pool tokens | epochs e | max chunk multiplicity | repeated share of 100B (epoch wrap) | duplicated files | exact-repeat share from file dup. | Muennighoff "unique-equivalent" efficiency | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | dolma1_7 | 837,476,349 | 1715.2B | 0.0583 | 1 | 0 | 2 (wiki) | 0.012–0.013 % | 1.000 | none |
| 2 | dolma1_7-no-code | 699,095,116 | 1431.7B | 0.0699 | 1 | 0 | 2 (wiki) | 0.017–0.018 % | 1.000 | none |
| 3 | dolma1_7-no-math-code | 658,779,407 | 1349.2B | 0.0741 | 1 | 0 | 2 (wiki) | 0.020–0.021 % | 1.000 | none |
| 4 | dolma1_7-no-reddit | 798,441,223 | 1635.2B | 0.0612 | 1 | 0 | 2 (wiki) | 0.013–0.014 % | 1.000 | none |
| 5 | dolma1_7-no-flan | 829,401,716 | 1698.6B | 0.0589 | 1 | 0 | 2 (wiki) | 0.012–0.013 % | 1.000 | none |
| 6 | dolma1_6plus | 920,525,492 | 1885.2B | 0.0531 | 1 | 0 | 0 | 0 | 1.000 | none |
| 7 | c4 | 67,598,149 | 138.4B | 0.7224 | 1 | 0 | 0 | 0 | 1.000 | none |
| 8 | **fineweb-pro** | 40,392,120 | 82.7B | **1.2090** | 2 (base 3) | **17.29 %** | 0 | 0 | 0.9988 | report (R-rep-report) |
| 9 | fineweb-edu | 93,848,413 | 192.2B | 0.5204 | 1 | 0 | 0 | 0 | 1.000 | none |
| 10 | falcon | 222,853,887 | 456.4B | 0.2191 | 1 | 0 | 0 | 0 | 1.000 | none |
| 11 | falcon-and-cc | 515,052,339 | 1054.8B | 0.0948 | 1 | 0 | 0 | 0 | 1.000 | none |
| 12 | falcon-and-cc-qc-10p | 71,324,912 | 146.1B | 0.6847 | 1 | 0 | 0 | 0 | 1.000 | none |
| 13 | falcon-and-cc-qc-20p | 139,574,889 | 285.8B | 0.3499 | 1 | 0 | 0 | 0 | 1.000 | none |
| 14 | falcon-and-cc-qc-orig-10p | 72,189,178 | 147.8B | 0.6765 | 1 | 0 | 0 | 0 | 1.000 | none |
| 15 | falcon-and-cc-qc-tulu-10p | 62,946,778 | 128.9B | 0.7758 | 1 | 0 | 0 | 0 | 1.000 | none |
| 16 | dclm-baseline | 1,883,359,931 | 3857.1B | 0.0259 | 1 | 0 | 0 | 0 | 1.000 | none |
| 17 | dclm-baseline-qc-7p-fw2 | 247,821,193 | 507.5B | 0.1971 | 1 | 0 | 0 | 0 | 1.000 | none |
| 18 | dclm-baseline-qc-7p-fw3 | 106,431,877 | 218.0B | 0.4588 | 1 | 0 | 0 | 0 | 1.000 | none |
| 19 | dclm-baseline-qc-fw-3p | 57,660,788 | 118.1B | 0.8469 | 1 | 0 | 0 | 0 | 1.000 | none |
| 20 | **dclm-baseline-qc-fw-10p** | 48,078,767 | 98.5B | **1.0157** | 2 (base 3) | **1.55 %** | 0 | 0 | 0.99999 | none (below report threshold) |
| 21 | **dclm-baseline-qc-10p** | 40,076,013 | 82.1B | **1.2186** | 2 (base 3) | **17.94 %** | 0 | 0 | 0.9987 | report (R-rep-report) |
| 22 | dclm-baseline-qc-20p | 83,526,938 | 171.1B | 0.5847 | 1 | 0 | 0 | 0 | 1.000 | none |
| 23 | dclm-baseline-25p-dolma1.7-75p | 1,096,671,134 | 2246.0B | 0.0445 | 1 | 0 | 1 (wiki part-0) | 0.005 % | 1.000 | none |
| 24 | dclm-baseline-50p-dolma1.7-50p | 1,373,849,100 | 2813.6B | 0.0355 | 1 | 0 | 0 | 0 | 1.000 | none |
| 25 | dclm-baseline-75p-dolma1.7-25p | 1,628,869,132 | 3335.9B | 0.0300 | 1 | 0 | 0 | 0 | 1.000 | none |

How to read the columns:
* "Epochs" for recipes below 1 equals the share of the pool read. It matches the "Share of recipe read" column in `experiments/E04_exact_100b_alpha/DETAILS.md` (121 %, 102 %, 122 % for the three repeaters).
* Effective epochs on Wikipedia in the Dolma recipes: each Wikipedia token's expected exposures = 2f = **0.09–0.15** (§3c). The second copy adds 0.10–0.28 % of the training tokens. This is upweighting of a source (a mixture property), not repetition in the data-constrained sense.
* "Muennighoff efficiency" = D′/D. Here D′ = U·(1 + R_D*·(1 − e^(−R/R_D*))), with R = e − 1 repeated epochs, R_D* = 15.39 (Muennighoff et al. 2023, Eq. 5 fit), and D = e·U tokens actually trained.
  * For fineweb-pro, 100B repeated tokens are worth about 99.88B unique tokens.
  * For dclm-baseline-qc-10p, they are worth about 99.87B.
  * So the shortfall is ≈ 0.12–0.13 % of the budget.
* Where the repeats fall: epoch 1 ends at step N/704 = 57,375 (fineweb-pro), 56,926 (qc-10p) and 68,294 (qc-fw-10p) of 69,369. The repeated data are the **last ~17 %** of steps for two models and the last ~1.5 % for the third.

## 2. Exclusion criteria

**R-epoch (exclusion; measurement validity / comparability).**
* Rule: exclude a model if e > **4.0** epochs.
* Justification:
  * The DataDecide comparison assumes every model got an equally useful 100B-token budget.
  * Muennighoff et al. (2023, NeurIPS, "Scaling Data-Constrained Language Models") show that up to ~4 epochs of repetition gives a negligible loss difference from unique data. Returns decay strongly only beyond that, with repeated tokens near worthless around 16+ epochs.
  * Below 4 epochs, a model is to first order equivalent to one trained on 100B fresh tokens from the same distribution. So it does not break the "same budget, different data" design.
* The threshold was set before looking at the epoch values, which max out at 1.22.
* **Flags: none.**

**R-rep-report (reporting / sensitivity, NOT exclusion).**
* Rule: list a model for a sensitivity analysis if more than **10 %** of its training tokens are second exposures (e > 1.11).
* Justification:
  * 10 % is a conventional "material share" cut.
  * Even with no loss penalty, repeated data reduce document diversity and sit at the end of training (§1), which could in principle shape final representations.
  * So a robustness check (refit without these models) is good practice. Muennighoff efficiency is ≥ 0.9987, so this is not grounds for exclusion.
* **Flags: fineweb-pro (17.29 %), dclm-baseline-qc-10p (17.94 %).**
* Not flagged: dclm-baseline-qc-fw-10p at 1.55 %.

**R-upsample (file-level duplication).**
* Rule: flag if exact-duplicate chunks are more than 1 % of training tokens.
* Justification: the same materiality logic as above, at one tenth of its threshold, since these are true exact repeats.
* **Flags: none.** The maximum is 0.021 % (dolma1_7-no-math-code).
* Wikipedia 2× upweighting is a designed mixture weight and belongs to the recipe definition.

**Is repetition a measurement-validity problem for α / counts?**
* No. Exposure counting should double-count repeated documents, because the model saw them twice.
* The E04 counter does exactly that: the `membership` multiplicity is used, and the per-seed counts include the epoch-2 chunks.

## 3. Evidence (code + output)

### (a) Pool sizes, multiplicities, epochs
Read from `results/corpus_alpha_datadecide/exact_100b_<r>/summary.json`. Example (fineweb-pro):

```
"membership": {"base": 3, "n_chunks": 40392120, "union_chunks": 40392120, "per_seed_chunks": [48835776]}
```

`base` = 3 means the maximum multiplicity is 2. `union_chunks` = n_chunks means every pool chunk was used at least once. dclm-baseline-qc-10p (40,076,013) and dclm-baseline-qc-fw-10p (48,078,767) show the same pattern. All other 22 recipes have base = 2 (multiplicity ≤ 1).

Script output (python, this audit):

```
fineweb-pro            N 40392120 epochs 1.2090 repeated frac 0.1729
dclm-baseline-qc-10p   N 40076013 epochs 1.2186 repeated frac 0.1794
dclm-baseline-qc-fw-10p N 48078767 epochs 1.0157 repeated frac 0.0155
```

### (b) What repetition does to the number counts
For the three repeaters, E04 also saved the counts of the unique pool once (`counts_union_of_samples.csv`), next to the trained stream (`counts_seed_2.csv`). Comparison (python, this audit; R = #{n in 10..9999 : c(n) < 4000}):

```
fineweb-pro: matches ratio stream/unique 1.2090 (= epochs); per-n count ratio median 1.2088, p1 1.1713, p99 1.2513
  alpha_ols stream -1.76170 unique -1.76179 d=+0.00009 ; alpha_mle stream -1.050330 unique -1.050312 d=-0.000018
  R stream 7433 | R unique pool 7606 | R unique×epochs 7432
dclm-baseline-qc-10p: ratio 1.2186; per-n median 1.2185 (p1 1.1739, p99 1.2679)
  alpha_ols d=-0.00012 ; alpha_mle d=0.000000 ; R stream 7548 | unique 7658 | unique×epochs 7546
dclm-baseline-qc-fw-10p: ratio 1.0157; per-n median 1.0157 (p1 1.0076, p99 1.0263)
  alpha_ols d=+0.00006 ; alpha_mle d=+0.000008 ; R stream 7060 | unique 7074 | unique×epochs 7061
```

What this shows:
* The second epoch is a uniform random subset. Every integer's count is scaled by ≈ e (spread ±3 %), so **α (a shape measure) is unchanged**. |Δα_OLS| ≤ 1.2e-4 is far below the between-recipe range (−1.43 to −1.76). |Δα_MLE| ≤ 2e-5.
* R depends on amount. Repetition lowers it by 14–173 relative to counting each unique chunk once (fineweb-pro 7606 → 7433, qc-10p 7658 → 7548, qc-fw-10p 7074 → 7060).
* R on the stream equals, within 1–2, R for "e × the unique counts". So the stream count is exactly what fresh data of the same distribution would have given.
* **Conclusion for counting:** counting the stream with multiplicity (what E04 does) is the correct exposure measure. Using unique-pool counts instead would wrongly understate exposure for these 3 models. No correction is needed.

### (c) Wikipedia file upsampling
* `configs/datadecide_data_map.json` `recipes.*.repeated_paths`: 2 wiki files ×2 in the five dolma1_7* recipes, and 1 in dclm-baseline-25p-dolma1.7-75p.
* Wiki size lower bound from the maximum window end in `results/corpus_alpha_datadecide/dolma1_7-no-reddit/windows.json`: part-0 ≥ 2,404,749,957 and part-1 ≥ 1,167,333,744 tokens.
* Script output (f = 48,835,776 / N):

```
dolma1_7            f=0.0583 2nd-copy share 0.208%-0.219%  exact repeat 0.0121%-0.0127%  wiki exposures/token 0.117
dolma1_7-no-code    f=0.0699 2nd-copy share 0.249%-0.262%  exact repeat 0.0174%-0.0183%  0.140
dolma1_7-no-math-code f=0.0741 0.265%-0.278%  0.0196%-0.0206%  0.148
dolma1_7-no-reddit  f=0.0612 0.218%-0.229%  0.0134%-0.0140%  0.122
dolma1_7-no-flan    f=0.0589 0.210%-0.221%  0.0124%-0.0130%  0.118
dclm-25p-dolma75p   f=0.0445 0.102%-0.109%  0.0046%-0.0049%  0.089
```

The second wiki copy is already in E04's counts, because the duplicated paths get their own chunk indices (`docs/datadecide_sampling.md` §5: "duplicated files count twice"). So the counts reflect true exposure. The sampler's `mixture.json` folds the wiki files into `olmo-mix/v1_6-decontaminated`, but this does not affect any count.

## 4. Classification of each flag

| flag | models | measurement-validity problem? | treatment |
|---|---|---|---|
| e > 1 (second epoch) | fineweb-pro (1.209), dclm-baseline-qc-10p (1.219), dclm-baseline-qc-fw-10p (1.016) | **No.** It is a genuine property of small recipes under DataDecide's fixed-token design. It is far below the 4-epoch threshold, and the effective-data loss is ≤ 0.13 %. | Keep. Report e in the model table. |
| >10 % repeated tokens | fineweb-pro, dclm-baseline-qc-10p | **No** (genuine data property). Possible diversity / late-training effect; no evidence it biases measurement. | Keep. Run a sensitivity analysis excluding these 2 (or with e as a covariate). |
| Wikipedia 2× upsampling | 5 dolma1_7*, dclm-25p-dolma75p | **No.** It is part of the recipe's mixture weights. Exact repeats are < 0.021 % of tokens. | Keep. Nothing to control. |
| Counting with multiplicity | the 3 repeaters (+ wiki recipes) | **No problem.** Double-counting is the correct exposure measure. α is unaffected (|Δ| ≤ 1.2e-4). R/amount features correctly include the extra exposure. | No correction. Do not substitute unique-pool counts. |

**Honest bottom line: repetition gives no valid reason to exclude any of the 25 models.**

Caveats:
* `01_training.md` notes that a restart during epoch 2 might reuse the epoch-1 permutation. That changes *which* chunks are repeated, not how many, so it would leave α unchanged and R essentially unchanged.
* Within-corpus near-duplicate documents (duplicates already inside a source's files) are not measured by this audit or the repo. They are part of each recipe's genuine data.
