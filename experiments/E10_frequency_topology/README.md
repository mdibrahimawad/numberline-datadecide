# E10: Frequency and the topology of number representations (stage A, exploratory)

**Question:** do numbers that appear more often in the training data get a different shape
(topology / local geometry) inside the model than equally large numbers that appear rarely?
**Status:** stage A (exploration on existing data) done, 2026-10-14. Nothing here is a
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

## Results (stage A, exploratory; all numbers from `results/e10_explore/`)
Commands: `python -m src.e10_explore --vectors <topology_number_line>/data/crystal_vectors`
(default: A–F with the `block` and `block_digits` thirds); robustness:
`--only BC --controls block_digits_plain` and `--only B --unit --controls block_digits_plain
--out results/e10_explore/unit_length`.

### 0. What frequency means for single numbers
Among ordinary 4-digit numbers, block + tens digit + last digit explain **87 %** of the
variance of log10 count. Ranking by raw count inside a block puts numbers ending in 1 into the
high third 24 % of the time vs 1 % in the low third. Once magnitude and digit form are held
fixed, the thirds differ only **~1.6×** in median count (13k vs 21k). The big frequency
contrasts at fixed magnitude are categorical: years (≈ 630× at fixed magnitude), x000, x00,
repdigits, powers of 2. The most "unexpectedly frequent" ordinary numbers are memorable ones
(9999, 9001, 4096, 1024, 8192, 5555, 1111, 5678).

### 1. Three frequency groups (B: high vs low third, z against 200 within-block label shuffles)
H0 mean death = typical merge distance = how far apart neighbouring points are (larger = less
compressed). Distances in units of the layer's typical 4-digit distance.

| High − low third, H0 mean death (z) | L8 | L16 | L24 | L27 | L32 |
|---|---|---|---|---|---|
| block-matched | +8.2 | +5.3 | +6.1 | +5.7 | +3.8 |
| block + digit-matched | **+9.5** | **+6.4** | **+5.2** | **+5.1** | **+2.8** |
| … and special-pattern numbers removed | +6.1 | +4.7 | +3.4 | +3.3 | +2.4 |
| … and direction only (vectors scaled to length 1) | +3.4 | +2.6 | +1.9 | +2.1 | +2.7 |

Profiles (A, block+digit thirds, H0 mean death; low / mid / high): L8 0.712 / 0.721 / 0.739,
L16 0.743 / 0.745 / 0.773, L24 0.738 / 0.743 / 0.762, L27 0.557 / 0.561 / 0.577,
L32 0.523 / 0.529 / 0.545 (ordered at every layer).
Tests passing Benjamini-Hochberg q < 0.05 (of 205 feature × layer): 82 (block), 75
(block+digit), 46 (without special numbers), **0 (direction only)**. Most passing features
(H0 deaths, H1 births/deaths/midpoints) measure the same thing, local spacing; it is one
finding, not 75. The H1 loop-count / entropy difference (fewer loops for frequent numbers,
L8 z ≈ −3) disappears without the special-pattern numbers.
Naive thirds (magnitude not held fixed) show much larger differences (e.g. L8 H1 count 134 vs
90): that is the size effect, not frequency.

### 2. Fay-style classifier (C: logistic regression on the 41 features, held-out blocks)
| Held-out AUC, high vs low third (sets of 120 numbers) | L8 | L16 | L24 | L27 | L32 |
|---|---|---|---|---|---|
| block-matched | 0.996 | 0.907 | 0.960 | 0.926 | 0.871 |
| block + digit-matched | 0.998 | 0.988 | 0.964 | 0.954 | 0.807 |
| … special-pattern numbers removed | 0.926 | 0.908 | 0.827 | 0.828 | 0.754 |
| shuffled labels (mean of 30) | ≈ 0.5 | ≈ 0.5 | ≈ 0.5 | ≈ 0.5 | ≈ 0.5 |
Every real AUC is above all 30 shuffles (p = 1/31, the smallest possible with 30).
It classifies *sets* of 120 numbers, not single numbers.

### 3. Per number (D, 7,560 ordinary numbers; Spearman with frequency left after block + digits)
Mean distance to the 10 nearest 4-digit numbers: +0.08…+0.11 (block bootstrap 95 % CI excludes 0
at every layer; positive inside 83–90 % of the 84 blocks). Without the 625 special-pattern
numbers: **+0.04…+0.06**. Of that, vector length itself correlates +0.02…+0.06, and with unit
vectors: −0.003 (L8), +0.040, +0.045, +0.027, +0.024. So at L8 the per-number effect is vector
length; at deeper layers a small part is direction.

### 4. Special numbers (F, isolation z vs ordinary numbers of the same block)
| Category | median count | L8 | L16 | L24 | L27 | L32 |
|---|---|---|---|---|---|---|
| repdigits 1111–9999 | 164k | **+6.7** | +3.0 | +2.9 | +3.0 | +2.0 |
| x000 | 2.1M | **+4.4** | +2.4 | +1.5 | +3.1 | +2.4 |
| powers of 2 | 467k | **+3.4** | +2.4 | +2.2 | +2.1 | +1.4 |
| years 1950–2025 | 11.4M | +2.5 | +1.3 | +0.4 | +0.4 | +0.3 |
| digit runs 1234, 4321, … | 66k | +1.3 | +2.9 | +2.3 | +1.9 | +1.1 |
| x00 | 160k | +1.0 | +1.3 | +0.6 | +1.2 | +0.8 |
| years 1800–1899 | 1.3M | +0.8 | +0.6 | +0.2 | +0.1 | +0.3 |
| x50 | 37k | +0.5 | +0.3 | +0.1 | +0.2 | +0.2 |
Special numbers get extra room, mostly at L8, but not in order of frequency: distinctive digit
form matters as well (repdigits beat years, which are ~70× more frequent).

### 5. Years (scratch scripts in `years_scripts/`; same data)
Years 1900–2025 vs ~150 windows of 126 ordinary consecutive numbers, L8: overall spread z +8.4,
scale-free H0 total −9.2, H1 count −4.7, H1 entropy −5.8 ("fewer, larger features", the shape
Fay et al. call topological compression), and decades cluster (ratio within/between decade
0.82 vs 0.90; 1 of 156 windows as strong). Weaker at L16–L32; partly the 1999→2000 jump
(1900–1999 alone: spread +3.0, H0 total −3.1). Years also reorganise most between layers
(E: H0 Wasserstein L8→L16 9.9 vs ~4–5 for ordinary thirds; L16→L24 8.4 vs ~2). Ordinary
thirds do not differ from each other in layer-to-layer change.

## Interpretation (exploratory)
- **One robust candidate:** frequent numbers sit slightly further apart than equally large,
  equally shaped rare numbers, at every layer. It survives magnitude, digit form and removing
  special-pattern numbers, but **much of it is vector length**: with direction only it is
  positive at all five layers but no longer clears the multiple-testing correction.
- **Special numbers are a separate, larger effect:** repdigits, x000, powers of 2 and years are
  isolated, and years have their own clustered, reorganising structure. Frequency and
  distinctiveness are mixed here; this data cannot separate them.
- **Loops (H1) carry no frequency signal** beyond the special numbers.

## Limitations
One model (Crystal), final checkpoint, one prompt per number (no prompt-noise floor), SlimPajama
counts (Crystal also trained on StarCoder data), frequency contrast among ordinary numbers only
~1.6×, correlation only. The 41 features are our reconstruction of Fay et al.'s list. Stage A
tried many measures, so its p-values are descriptive.

## Candidates to lock for stage B (confirmation on new data)
1. H0 mean death, high − low block+digit third without special-pattern numbers, raw and
   direction-only, per layer (predicted > 0).
2. Vector length vs frequency left after block + digits (predicted > 0).
3. Isolation z of x000, repdigits, powers of 2, years (predicted > 0, strongest early).
4. Years at an early-middle layer: larger spread, smaller scale-free H0 total, decade clustering.

Stage B options (to be chosen and pre-registered): (a) the same vectors for StarCoderBase-3B and
OpenLLaMA-3B (already extracted, on Modal; replication across models, $0); (b) Crystal
intermediate checkpoints (LLM360 released them with data order: exposure grows while
everything else is fixed); (c) several prompts per number to measure the noise floor.
