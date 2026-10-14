# E10: Frequency and the topology of number representations (stage A, exploratory)

**Question:** do numbers that appear more often in the training data get a different shape
(topology / local geometry) inside the model than equally large numbers that appear rarely?
**Status:** stage A (exploration on existing data) in progress, 2026-10-14: A, D, F done; B, C, E running. Nothing here is a
confirmed result: measures that look real are listed under "Candidates to lock" and must be
pre-registered and tested on new data (stage B, checkpoints) before being claimed.

## Why
E09 showed that recipe-level summaries of the counts (α, S, R) do not predict β out of
sample. E10 moves from one number per model (β) to one point per *number*, and uses
persistent homology (PH) and local geometry, as in Fay et al. (ICLR 2026), Malhotra et al.
(2026) and Bazarova et al. (TOHA, ACL 2026).

## Data
| | |
|---|---|
| Model | LLM360 Crystal (final checkpoint). Its tokenizer splits every number into single digits, so 1999 and 5371 are both 4 tokens (no tokenizer confound). |
| Prompts | one per integer 1..10000: `A=A,B=B,C=C,D=D,n=` (random demonstrations, seed 42), last-token hidden state. From the `topology_number_line` repo. |
| Vectors | layers 8, 16, 24, 27, 32 of 33, centred float16 (distance error ≤ 0.02 %): `mdibrahimawad/topology_number_line`, `data/crystal_vectors/` |
| Frequency | SlimPajama-627B counts (`results/corpus_alpha_full/slimpajama-627b_reupload`), Crystal's main pretraining corpus (Crystal also saw StarCoder data; not counted). |

## Plan (what stage A tests)
Every comparison holds magnitude fixed; distances are Euclidean in the full 4096-d space,
divided by the layer's typical 4-digit distance (comparable across layers, group scale kept).

| ID | Test | Groups / unit | Null / control |
|---|---|---|---|
| A | Profiles: 41 barcode features (H0/H1 births, deaths, lifetimes, ratios, counts, total persistence, entropy; our reconstruction of Fay et al.'s 41) | 120-point subsamples × 50: three frequency thirds (naive, block-matched, block+digit-matched), years, year twins, round numbers, other tens | descriptive |
| B | High vs low third, every feature × layer | block-matched and block+digit-matched thirds | tier labels shuffled within blocks (200×); Benjamini-Hochberg |
| C | Fay-style classifier: logistic regression on 41 features, high vs low | train on one half of the blocks, test on the other | shuffled labels (30×) |
| D | Per number: local geometry (10-NN distance, step to n±1, TwoNN ratio, norm) vs frequency left over after block (+digits) | 7,560 ordinary numbers | block bootstrap; without "special-pattern" numbers |
| E | Layer-to-layer change: Wasserstein-1 between diagrams of the same numbers at consecutive layers | thirds, years, year twins | between groups |
| F | Special categories (years, x000, x00, x50, repdigits, powers of 2, digit sequences): isolation vs ordinary numbers of the same block | per number | block distribution |

Group definitions (code: `src/e10_explore.py`, tests: `tests/test_topo_features.py`):
- **Ordinary numbers:** 4-digit, not 1500–2099, not a multiple of 10 (7,560 numbers, 84 blocks of 100).
- **Block-matched thirds:** inside each block, rank by count: 30 low / 30 mid / 30 high per block.
- **Block+digit-matched thirds:** same, ranking the count left after fitting
  log10 count ~ block + tens digit + last digit (the plain ranking puts 24 % numbers ending in 1
  in the high third vs 1 % in the low third).
- **Naive thirds:** all 4-digit numbers by count, magnitude *not* held fixed (shown to expose the confound).

## Results so far (stage A, exploratory)
`python -m src.e10_explore --vectors <topology_number_line>/data/crystal_vectors --only ADF` → `results/e10_explore/`

1. **Frequency of ordinary 4-digit numbers is mostly magnitude + digit form:** block + tens digit +
   last digit explain 87 % of the variance of log10 count; the block+digit-matched thirds differ
   only ~1.6× in median count (13k vs 21k).
2. **Thirds (A, H0 mean death, block+digit-matched; larger = points further apart):**
   low 0.712 / 0.743 / 0.738 / 0.557 / 0.523, high 0.739 / 0.773 / 0.762 / 0.577 / 0.545 at
   L8 / 16 / 24 / 27 / 32; mid lies between at every layer. Significance: test B (running).
3. **Per number (D):** frequency beyond block + digits vs mean 10-NN distance: Spearman +0.08…+0.11
   (block-bootstrap 95 % CI excludes 0 at every layer; positive inside 83–90 % of blocks), but
   +0.03…+0.06 after removing 625 special-pattern numbers (repdigit-like, digit runs, powers of 2).
4. **Special numbers are isolated (F, z vs ordinary numbers of the same block, L8):** repdigits +6.7,
   x000 +4.4, powers of 2 +3.4, years 1950–2025 +2.5, digit runs +1.3, x00 +1.0, years 1800s +0.8,
   x50 +0.5. Not ordered by frequency alone (years are the most frequent, not the most isolated).
