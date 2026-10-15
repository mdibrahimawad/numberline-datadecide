# E10: Frequency and the topology of number representations

**Question:** do numbers that appear more often in the training data get a different shape
(topology / local geometry) inside the model than equally large numbers that appear rarely?
**Status:** stage A (exploration, Crystal), stage B-A (pre-registered replication on OpenLLaMA-3B
and StarCoderBase-3B) and stage C (topology-only questions, exploratory) done, 2026-10-15. Stage A results are exploratory; only the stage B-A section
reports pre-registered tests.

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

## Stage B-A: replication on StarCoderBase-3B and OpenLLaMA-3B (pre-registered 2026-10-14)
**Written before any vector of these two models was exported or looked at.** Code: `src/e10_confirm.py`
(runs exactly P1–P5 below and prints pass / fail). Results go in a separate section below, with
any deviation from this plan and its reason. This section must not change after the export.

**Data.** The existing full-range extractions of the `topology_number_line` project (same 10,000
prompt texts as Crystal, last token, native tokenizers), Modal volume `numberline-zigzag-results`,
folders `openllama-3b_fullrange_1_10000_ctx1234_k4_seed42` and
`starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42`. Exported like Crystal: centred float16,
check of distance error ≤ 1 %. The export also copies each model's `dataset.json` (tokenization).

**Layers (same relative depths as Crystal 8, 16, 24, 27, 32 of 0–32; round(f × last level)):**
OpenLLaMA-3B (levels 0–26): **7, 13, 20, 22, 26**. StarCoderBase-3B (levels 0–36): **9, 18, 27, 30, 36**.

**Counts.** SlimPajama-627B for both. For OpenLLaMA (trained on RedPajama) this is the
deduplicated version of its corpus: **OpenLLaMA is the primary replication.** StarCoderBase was
trained on code (The Stack), for which we have no counts: its results are **secondary (proxy
counts)** and cannot decide the replication.

**Tokenization check.** From `dataset.json`: every 4-digit target must be 4 single-digit tokens.
Numbers that are not are listed and excluded from P1–P3 (reported, not silently dropped).

**Tests** (group definitions as in stage A; `CONTROL = block_digits_plain`: ordinary 4-digit numbers,
no 1500–2099, no multiples of 10, no special-pattern numbers, thirds by count left after
block + tens digit + last digit; 120-point subsamples, 2 per block per tier, K = 20; null = 200
within-block shuffles of the tier labels):

| ID | Measure | Predicted | Pass rule |
|---|---|---|---|
| P1 | H0 mean death, high − low third, raw vectors | > 0 | z ≥ 2 at ≥ 3 of the 5 layers |
| P2 | same, vectors scaled to length 1 (direction only) | > 0 | z ≥ 2 at ≥ 3 of the 5 layers |
| P3 | Spearman(frequency left after block + digits, vector length), same numbers | > 0 | block-bootstrap 95 % CI above 0 at ≥ 3 of 5 layers |
| P4 | isolation z (10-NN distance vs ordinary numbers of the same block) of x000, repdigits, powers of 2, years 1950–2025, at the first layer | > 0 | mean z > 1 for ≥ 3 of the 4 categories |
| P5 | years 1900–2025 vs 156 windows of 126 consecutive ordinary numbers, first layer: overall spread and scale-free H0 total | spread up, H0 total down | spread z > 2 and H0-total z < −2 |

**Decision rule.** The stage-A spacing finding "replicates" if **P1 passes in OpenLLaMA**. P2 decides
whether it is more than vector length. P3–P5 are reported for both models. Crystal's own stage-A
values on the same tests (for reference, not a test): P1 z 6.1 / 4.7 / 3.4 / 3.3 / 2.4; P2 3.4 / 2.6 /
1.9 / 2.1 / 2.7; P4 at L8 4.4 / 6.7 / 3.4 / 2.5; P5 at L8 +8.4 / −9.2.
Anything beyond P1–P5 run on the new models is exploratory and labelled so.

## Stage B-A: results (run 2026-10-14, exactly as pre-registered in commit `6a00de9`)
`python -m src.e10_confirm --model <m> --vectors <topology_number_line>/data/<m>_vectors --layers <…>
--out results/e10_confirm/<m>` → `results/e10_confirm/<m>/` (`summary.json` + one table per test).
**Deviations from the plan: none.** Tokenization check: in both models every 4-digit target is
4 single-digit tokens followed by `=`; no number excluded.

| Test | OpenLLaMA-3B (primary; levels 7/13/20/22/26) | StarCoderBase-3B (proxy counts; 9/18/27/30/36) | Crystal stage A (reference) |
|---|---|---|---|
| P1 spacing, raw (z) | +3.9 / +1.8 / +2.5 / +2.4 / +1.5 → **pass** (3 of 5) | +5.8 / +3.7 / +3.7 / +3.3 / +2.4 → **pass** | +6.1 / +4.7 / +3.4 / +3.3 / +2.4 |
| P2 spacing, direction only (z) | +0.0 / +0.3 / 0.0 / +0.2 / +0.5 → **fail** | +2.7 / +3.4 / +3.2 / +2.7 / +3.4 → **pass** | +3.4 / +2.6 / +1.9 / +2.1 / +2.7 |
| P3 frequency vs vector length (ρ) | +0.023 / +0.023 / +0.031 / +0.028 / +0.004; CI > 0 at 1 layer → **fail** | +0.053 / +0.031 / +0.045 / +0.031 / −0.010; CI > 0 at 3 → **pass** | +0.02…+0.06 (stage A, D) |
| P4 isolation z, first layer: x000 / years 1950–2025 / repdigits / powers of 2 | +3.8 / +3.2 / +0.6 / +1.05 → **pass** (3 of 4) | +0.6 / +1.4 / +4.3 / −0.3 → **fail** (2 of 4) | +4.4 / +2.5 / +6.7 / +3.4 |
| P5 years vs 156 windows, first layer: spread z / scale-free H0 total z | +4.4 / −8.2 → **pass** | +2.1 / **+2.1** → **fail** | +8.4 / −9.2 |

**Decision (pre-registered rule): the stage-A spacing finding replicates in OpenLLaMA (P1 passes),
but there it is not a change of direction (P2 fails) and not a clear per-number vector-length
effect either (P3 fails).** It is a small, set-level effect. StarCoder (proxy counts) shows it more
clearly, in both raw and direction-only form.

**Reading the five tests together:**
- *Frequent ordinary numbers get slightly more room* (P1): positive in all 15 model × layer cells,
  z ≥ 2 in 12 of 15. The most consistent result of E10, but small (thirds differ only ~1.6× in
  count; per number ρ ≈ 0.02–0.05).
- *Whether that room is direction or length* differs by model (P2: Crystal moderate, OpenLLaMA none,
  StarCoder clear). No general claim can be made.
- *Special numbers stand out* in every model, but **which** ones differs (x000 and years in
  OpenLLaMA, repdigits in StarCoder, all four in Crystal).
- *Years as clustered "fewer, larger features"* holds in the two models trained on web text
  (Crystal, OpenLLaMA) and not in the code model (StarCoder: years spread out but not clustered).

**What this means for E10.** The topological (H1, loops) view added nothing; the signal is in local
spacing (H0), which is ordinary geometry. The candidates for a paper are (i) the small but
reproducible exposure → spacing effect at fixed magnitude and digit form, and (ii) the
model-dependent structure of special numbers (years clustering in web-text models). Both are
correlational, from final checkpoints, with one prompt per number; the next tests that could make
them stronger are checkpoints (does the spacing grow as exposure grows?) and several prompts per
number (noise floor).

## Stage C: questions only topology can ask (exploratory, all three models, 2026-10-15)
`python -m src.e10_topology --data <topology_number_line>/data` → `results/e10_topology/`.
Same relative depths in every model (~25 %, ~50 %, ~75 %, ~84 %, last). Sanity checks on known
shapes (a ring of 10 points gives a loop in digit order that wraps 9 → 0; a line gives none;
RTD(A, A) = 0, small for a noisy copy, large for a shuffled copy) passed before running.

**T0, PH across depth (random sets of 300 four-digit numbers, scale-free):** in every model the
number of H1 loops roughly halves from ~25 % depth to the late layers (Crystal 386 → 165,
OpenLLaMA 308 → 181, StarCoder 372 → 159), H1 entropy falls, and the scale-free H0 spacing falls
(Crystal 0.70 → 0.49): deeper layers have fewer loops and tighter clusters. This is the
"networks simplify topology with depth" pattern (Naitzat et al. 2020), seen here for numbers, and
fits the leading-digit grouping of late layers found in `topology_number_line`.

**T1, last-digit circle:** centroids of the 10 last-digit classes (block means removed) form **no
loop** in any model or layer (relative H1 persistence 0.00–0.08, same as random labels) and are
mostly not in digit order. The tens digit is in order (share 0.5–0.9) but as a **line**, not a loop.
Frequent and rare thirds do not differ. So, after a 4-digit number read digit by digit, there is no
period-10 circle at the `=` position (the "helix" results are on single-token numbers).

**T2, years:** year endings (19x0…19x9) form no loop; the 13 decades 1900s…2020s form a **line**
(decade loop 0.00–0.06), like ordinary ranges. Years are an ordered line of decade clusters, not a circle.

**T3, RTD between models on the same 120 numbers (lower = more similar topology):** matched
numbers are always more similar than a shuffled correspondence, and the agreement is **strongest in
late layers**: matched / shuffled RTD

| Pair | ~25 % | ~50 % | ~75 % | ~84 % | last |
|---|---|---|---|---|---|
| Crystal vs OpenLLaMA | 0.67 | 0.80 | 0.83 | 0.27 | 0.49 |
| Crystal vs StarCoder | 0.58 | 0.72 | 0.83 | 0.28 | 0.30 |
| OpenLLaMA vs StarCoder | 0.81 | 0.66 | 0.27 | 0.34 | 0.38 |

(Crystal's ~75 % layer, L24, is the exception.) Frequency: models agree slightly *more* on frequent
numbers at ~25 % depth (high − low RTD −0.18, Wilcoxon p = 0.02) and slightly *less* at ~75–84 %
(+0.39, p = 0.003; +0.25, p = 0.0008): no consistent direction, no claim.

## Observations through the PH lens (all of E10)
| # | Observation | Status | Known? |
|---|---|---|---|
| 1 | Topology of numbers simplifies with depth: H1 loops halve, points gather in tighter clusters (all 3 models) | exploratory, consistent in 3 models | known for networks in general (Naitzat et al. 2020); new for numbers in LLMs as far as we found |
| 2 | Different models give the same numbers increasingly similar topology in late layers (RTD 0.27–0.38 of chance) | exploratory, 3 pairs | related to convergent number features (Fu et al. 2026); not shown with RTD |
| 3 | Years are "fewer, larger features" (spread out, clustered by decade, an ordered line, no loop) in web-text models, not in the code model | pre-registered (P5) in 2 models | the topological-compression signature of Fay et al., here from data, not attacks |
| 4 | Special numbers (x000, years, repdigits, powers of 2) are isolated points in H0; which ones depends on the model | pre-registered (P4) | partly known (frequent tokens are outliers); new as an H0 observation on numbers |
| 5 | Frequent ordinary numbers sit slightly further apart (H0) at fixed size and digit form | pre-registered (P1) in 2 models | new as far as we found; small |
| 6 | No last-digit circle (H1) after digit-by-digit numbers; tens digit is a line | exploratory, 3 models | contrasts with the helix of single-token numbers (Kantamneni & Tegmark 2025) |
| 7 | H1 carries no frequency signal | stage A + C | — |

## Stage D: do models converge on the topology of numbers in late layers? (pre-registered 2026-10-15)
**Written before any vector of the new models exists.** Tests stage C observation 2 on new models.
Code: extraction `src/extract_number_vectors.py` (run by the owner on a laptop), test
`src/e10_convergence.py` (exactly the procedure below). Must not change after the new vectors exist.

**New models** (ungated, ~1–2B, different number tokenization): `Qwen/Qwen2.5-1.5B` (single digits),
`HuggingFaceTB/SmolLM2-1.7B` (single digits), `EleutherAI/pythia-1.4b` (multi-digit BPE),
`allenai/OLMo-2-0425-1B` (up to 3-digit chunks). Same 10,000 prompt texts as stages A–C, last-token
hidden state, levels at the same relative depths (round(f × number of layers), f = 8/32, 16/32, 24/32,
27/32, 1), float32 inference, stored centred float16. If a model fails to download or run, it is
reported and dropped, not replaced by another chosen after seeing results.

**Measure.** For each pair of models and depth: RTD (`src/e10_topology.rtd`) on the same 120 ordinary
4-digit numbers (2 per block, 60 random blocks; no 1500–2099, no multiples of 10, no special-pattern
numbers), divided by RTD with the second model's points shuffled; mean over K = 10 sets (seed 7).
Per pair: early = mean ratio at ~25 % and ~50 %; late = mean ratio at ~84 % and last.

**Prediction:** late < early. **Primary test:** over the pairs that include at least one new model
(18 pairs with 4 new models), one-sided sign test p < 0.05 (with 18 pairs: late < early in ≥ 13).
Pairs share models, so they are not independent; the result is reported per pair as well.
Reference (stage C models, already seen, not part of the test): late 0.28–0.35 vs early 0.67–0.75 in
all 3 pairs.
