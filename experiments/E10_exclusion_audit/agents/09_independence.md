# Auditor 09: statistical independence of the 25 DataDecide recipes

Scope: how much training data the 25 recipes share, the nesting structure, and de-duplication or one-point-per-corpus rules justified on design grounds only.
No alpha or beta value, and no alpha-beta correlation, was read, computed or used. The only inputs were
`configs/datadecide_data_map.json` (file lists), `results/corpus_alpha_datadecide/<recipe>/mixture.json` (token sizes),
`docs/datadecide_sampling.md`, `docs/project_overview.md`, and the local DataDecide README and OLMo `named_data_mixes.py` clones
in `/tmp/claude-0/agentrefs/` (used for provenance text and source-size comments). The recipe list was taken from the `recipe` column of `results/datadecide/alpha_beta.csv`; no other column was read.

Scripts (rerunnable): `scratchpad/agents/09/overlap.py` writes `overlap.json`; `scratchpad/agents/09/rules.py {point|low|high}` writes `rules_*.txt`.

## 1. Provenance evidence

| Relation | Evidence | Confidence |
|---|---|---|
| dolma1_7-no-{code,math-code,reddit,flan} are dolma1_7 with whole sources removed (same files) | `named_data_mixes.py:5127-5135` (`build_collection_include` source lists); file lists in the data map are exact subsets of dolma1_7's files | exact |
| no-math-code is a subset of no-code | no-code drops starcoder and stackexchange; no-math-code also drops algebraic-stack, open-web-math and pes2o | exact |
| c4, falcon and falcon-and-cc are whole sources of dolma1_7 (identical files) | `DATA_PATHS["c4"/"falcon"/"falcon_and_cc"]` use the same source keys as `dolma17`; the files are identical in the data map | exact |
| The mixes are random file subsets of dolma1_7 and dclm-baseline | `:5173-5183`, `sample_factor`; checked: every mix file is a dolma1_7 or dclm-baseline file. The DCLM parts of the 3 mixes are **not nested** (25p and 50p share 118 of 236 files, the value expected by chance); the Dolma parts are also not nested | exact |
| falcon-and-cc-qc-* are quality-filtered subsets of falcon-and-cc | DataDecide README: "Falcon and Dolma 1.7's Common Crawl. We quality filter to top 10% or 20% documents with reproduced or original Li et al. filter or retrain filter on Tulu-v3" | documented; assumed doc-level subset |
| dclm-baseline-qc-* are filtered subsets of DCLM-Baseline | README: "We quality filter to top 7% of DCLM classified documents and further take 2+ or 3+ FineWeb-edu scores; or top 3%/10% with FineWeb-edu classifier; or top 10%/20% with reproduced DCLM classifier" | documented; assumed doc-level subset |
| qc-7p-fw3 is a subset of qc-7p-fw2 | same base (`v0_rep32_ft7percentile`), FW-edu ≥3 is a subset of ≥2 | high |
| dclm qc-10p is a subset of qc-20p; falcon-and-cc qc-10p is a subset of qc-20p | same classifier, top 10% of the pool is inside its top 20% (sizes 82/171 and 146/286 fit a shared pool) | high, assumed |
| dclm qc-fw-3p versus qc-fw-10p | top 3p is **larger** than top 10p (118 vs 98.5 B), so 10p must come from a ~0.99 T shard subset (~25.5% of baseline). Point estimate: the 25.5% of fw-3p that lies in that shard is inside fw-10p (30 B). Range 3 to 98.5 B | low |
| Other cross-filter pairs (e.g. 7p-fw2 vs fw-3p, qc-orig vs qc-tulu) | unknown. **low** = independent selection (\|S1\|·\|S2\|/\|pool\|), **high** = full nesting (min size). The point value uses low, which is anti-conservative for dependence | unknown |
| dolma1_6plus versus dolma1_7 | No shared files (1_6plus is a re-tokenization, `-tokenizers-0-19-1`), but shared documents: same starcoder (263.8 B), proof-pile-2, RedPajama arxiv+SE (47.6 B), pes2o/books/wiki, Reddit (79.99 vs 79.94 B). Dolma-1.7 CC (598 B) and C4 (138 B) are dedup/quality-filtered versions of the 1.6 CC (676 B) and C4 (174 B); RefinedWeb v2 (456 B) is a filtered subset of v0 (547 B). Treated as doc-level containment. 1_6plus-only mass is about 206 B | medium |
| fineweb-pro versus fineweb-edu | both derived from FineWeb (ProX-refined random subset vs edu≥3 then dedup). Doc overlap estimated at about 1 B (range 0 to 10); ProX also rewrites the text | low |
| Different web pipelines (Dolma CC, RefinedWeb, C4, DCLM, FineWeb) | all come from Common Crawl, so URL-level duplicates exist but are unmeasured; set to 0 | unmeasured (see section 6) |

## 2. Nesting tree

```
Common Crawl (shared ecological root; doc-level overlap across pipelines unmeasured)
|
+-- Dolma lineage
|   dolma1_6plus  (1885 B)  ~ older version; 98% of dolma1_7's tokens are docs also in 1_6plus, 89% the reverse
|   \-- dolma1_7  (1715 B)
|       +-- dolma1_7-no-flan       (99% of d17)
|       +-- dolma1_7-no-reddit     (95%)
|       +-- dolma1_7-no-code       (83%)
|       |   \-- dolma1_7-no-math-code (79% of d17, 94% of no-code)
|       +-- c4                     (8%, whole source)
|       \-- falcon-and-cc          (62%, two whole sources: Dolma CC + RefinedWeb)
|           +-- falcon             (43% of f&cc, whole source)
|           +-- falcon-and-cc-qc-20p  (filter)
|           |   \-- falcon-and-cc-qc-10p (same classifier, top 10 within top 20)
|           +-- falcon-and-cc-qc-orig-10p (original classifier; overlap with qc-10p unknown, 15-97%)
|           \-- falcon-and-cc-qc-tulu-10p (Tulu-trained classifier; overlap unknown)
|
+-- DCLM lineage
|   \-- dclm-baseline (3857 B)
|       +-- qc-7p-fw2 (13%)  \-- qc-7p-fw3 (nested)
|       +-- qc-20p (4%)      \-- qc-10p (nested)
|       +-- qc-fw-3p (3%)  ~partial~ qc-fw-10p (3%, from ~25% shard subset)
|       \-- cross-filter overlaps unknown (independence lower bound 2-13%)
|
+-- Two-parent mixtures (random file subsets, not nested with one another)
|   dclm-25p-dolma-75p = 0.43 DCLM (25% of files) + 0.57 Dolma1.7 (75% of files)
|   dclm-50p-dolma-50p = 0.69 DCLM + 0.31 Dolma1.7
|   dclm-75p-dolma-25p = 0.87 DCLM + 0.13 Dolma1.7
|
\-- FineWeb lineage (shared parent pipeline, ~1% shared docs)
    +-- fineweb-edu (192 B)
    \-- fineweb-pro (83 B)
```

Graph components (link = either-direction containment > 0.05, which gives the same result at > 0.5):
- **With the mixes:** 3 components, {23 recipes}, {fineweb-pro}, {fineweb-edu}. The mixes bridge the Dolma and DCLM trees.
- **Mixes removed:** 4 components, Dolma lineage (13), DCLM lineage (7), fineweb-edu, fineweb-pro.

## 3. Overlap measures (definitions)

The definitions use I = unique tokens whose documents lie in both recipes' pools. All recipes are concatenations with uniform per-token weight (Wikipedia is 2x in Dolma 1.7, which is handled).
- **C(A in B)** = share of A's training stream made of documents that also occur in B (containment; asymmetric).
- **1-TV** = sum of min(p_A, p_B) = share of the two training distributions that is literally identical.
- **cos** = I/sqrt(\|A\|\|B\|), with multiplicities. Under a model where documents carry iid random effects, this is the expected correlation, caused only by shared data, of any additive count statistic (c(n) counts are additive) computed on the two corpora. It is the overlapping-sample formula used in meta-analysis with overlapping subjects (Lin & Sullivan 2009, AJHG; Bhattacharjee et al. 2012). It is a **lower bound** on dependence because it ignores shared pipeline and crawl structure. For a nested filter S in B, cos = sqrt(\|S\|/\|B\|).
- **Shared trained tokens** = expected share of the two 100 B-token runs that are identical chunks (independent PCG64 shuffles; recipes under 100 B are seen fully). This is small everywhere except nested small filters (dclm q10/q20: 48%; f&cc q10/q20: 35%; 7fw2/7fw3: 20%).

## 4. Overlap table (point estimates; every pair with I > 0.05 B, sorted by cos; 157 of 300 pairs overlap, 143 pairs share nothing measurable)

Abbreviations: d17=dolma1_7, -nf/-nr/-nc/-nmc = no-flan/no-reddit/no-code/no-math-code, d16p=dolma1_6plus, fal=falcon, f&cc=falcon-and-cc, f&cc-q10/q20/qO/qT = qc-10p/20p/orig-10p/tulu-10p, dclm-7fw2/7fw3/fw3/fw10/q10/q20 = dclm-baseline-qc-*, mixNN = dclm-baseline-NNp-dolma1.7-*. All zero-overlap pairs are cross-lineage pairs (e.g. any DCLM-only recipe with any Dolma-only recipe, any FineWeb recipe with anything except each other).

| A | B | I (B tok) | C(A in B) | C(B in A) | 1-TV | cos | shared trained tokens |
|---|---|---|---|---|---|---|---|
| d17 | d17-nf | 1694.9 | 0.99 | 1.00 | 0.99 | 1.00 | 5.9% |
| d17 | d17-nr | 1631.5 | 0.95 | 1.00 | 0.95 | 0.98 | 5.9% |
| d17-nr | d17-nf | 1615.0 | 0.99 | 0.95 | 0.95 | 0.97 | 5.9% |
| d17-nc | d17-nmc | 1345.5 | 0.94 | 1.00 | 0.94 | 0.97 | 7.0% |
| d17-nf | d16p | 1676.1 | 0.99 | 0.89 | 0.89 | 0.94 | 5.2% |
| d17 | d16p | 1676.1 | 0.98 | 0.89 | 0.89 | 0.93 | 5.2% |
| d17 | d17-nc | 1428.0 | 0.83 | 1.00 | 0.83 | 0.91 | 5.9% |
| d17-nr | d16p | 1596.2 | 0.98 | 0.85 | 0.85 | 0.91 | 5.2% |
| d17-nc | d17-nf | 1411.5 | 0.99 | 0.83 | 0.83 | 0.91 | 5.8% |
| d17 | d17-nmc | 1345.5 | 0.79 | 1.00 | 0.79 | 0.89 | 5.9% |
| d17-nc | d17-nr | 1348.1 | 0.94 | 0.83 | 0.83 | 0.88 | 5.8% |
| d17-nmc | f&cc | 1054.8 | 0.78 | 1.00 | 0.78 | 0.88 | 7.4% |
| d17-nmc | d17-nf | 1328.9 | 0.99 | 0.78 | 0.78 | 0.88 | 5.8% |
| d17-nc | f&cc | 1054.8 | 0.74 | 1.00 | 0.74 | 0.86 | 7.0% |
| d17-nmc | d17-nr | 1265.5 | 0.94 | 0.78 | 0.78 | 0.86 | 5.8% |
| d17-nc | d16p | 1392.7 | 0.98 | 0.74 | 0.74 | 0.85 | 5.2% |
| d17-nmc | d16p | 1310.1 | 0.97 | 0.69 | 0.69 | 0.82 | 5.2% |
| dclm | mix75 | 2893.8 | 0.75 | 0.87 | 0.75 | 0.81 | 2.3% |
| d17-nr | f&cc | 1054.8 | 0.65 | 1.00 | 0.65 | 0.80 | 6.1% |
| d17-nf | f&cc | 1054.8 | 0.62 | 1.00 | 0.62 | 0.79 | 5.9% |
| d17 | f&cc | 1054.8 | 0.62 | 1.00 | 0.62 | 0.78 | 5.8% |
| d16p | f&cc | 1054.8 | 0.56 | 1.00 | 0.56 | 0.75 | 5.3% |
| f&cc-q10 | f&cc-q20 | 146.1 | 1.00 | 0.51 | 0.51 | 0.71 | 35.0% |
| dclm-q10 | dclm-q20 | 82.1 | 1.00 | 0.48 | 0.48 | 0.69 | 48.0% |
| fal | f&cc | 456.4 | 1.00 | 0.43 | 0.43 | 0.66 | 9.5% |
| d17 | mix25 | 1291.6 | 0.76 | 0.57 | 0.57 | 0.66 | 3.4% |
| dclm-7fw2 | dclm-7fw3 | 218.0 | 0.43 | 1.00 | 0.43 | 0.66 | 19.7% |
| d17-nf | mix25 | 1279.1 | 0.76 | 0.57 | 0.57 | 0.65 | 3.4% |
| d17-nr | mix25 | 1231.2 | 0.76 | 0.55 | 0.55 | 0.64 | 3.4% |
| d16p | mix25 | 1264.9 | 0.67 | 0.56 | 0.56 | 0.61 | 3.0% |
| d17-nc | mix25 | 1078.0 | 0.76 | 0.48 | 0.48 | 0.60 | 3.4% |
| dclm | mix50 | 1930.6 | 0.50 | 0.69 | 0.50 | 0.59 | 1.8% |
| d17-nmc | mix25 | 1014.7 | 0.75 | 0.45 | 0.45 | 0.58 | 3.4% |
| d17-nmc | fal | 456.4 | 0.34 | 1.00 | 0.34 | 0.58 | 7.4% |
| d17-nc | fal | 456.4 | 0.32 | 1.00 | 0.32 | 0.56 | 7.0% |
| mix50 | mix75 | 1631.0 | 0.58 | 0.49 | 0.49 | 0.53 | 1.8% |
| d17-nr | fal | 456.4 | 0.28 | 1.00 | 0.28 | 0.53 | 6.1% |
| f&cc | f&cc-q20 | 285.9 | 0.27 | 1.00 | 0.27 | 0.52 | 9.5% |
| d17-nf | fal | 456.4 | 0.27 | 1.00 | 0.27 | 0.52 | 5.9% |
| d17 | fal | 456.4 | 0.27 | 1.00 | 0.27 | 0.51 | 5.8% |
| f&cc | mix25 | 793.1 | 0.75 | 0.35 | 0.35 | 0.51 | 3.3% |
| d16p | fal | 456.4 | 0.24 | 1.00 | 0.24 | 0.49 | 5.3% |
| d17-nmc | f&cc-q20 | 285.9 | 0.21 | 1.00 | 0.21 | 0.46 | 7.4% |
| d17-nc | f&cc-q20 | 285.9 | 0.20 | 1.00 | 0.20 | 0.45 | 7.0% |
| mix25 | mix50 | 1116.9 | 0.49 | 0.40 | 0.40 | 0.44 | 1.8% |
| d17-nr | f&cc-q20 | 285.9 | 0.17 | 1.00 | 0.17 | 0.42 | 6.1% |
| d17-nf | f&cc-q20 | 285.9 | 0.17 | 1.00 | 0.17 | 0.41 | 5.9% |
| d17 | f&cc-q20 | 285.9 | 0.17 | 1.00 | 0.17 | 0.41 | 5.8% |
| d17 | mix50 | 862.4 | 0.50 | 0.31 | 0.31 | 0.39 | 1.8% |
| d17-nf | mix50 | 854.1 | 0.50 | 0.31 | 0.31 | 0.39 | 1.8% |
| d16p | f&cc-q20 | 285.9 | 0.15 | 1.00 | 0.15 | 0.39 | 5.3% |
| mix25 | mix75 | 1065.8 | 0.47 | 0.32 | 0.32 | 0.39 | 1.4% |
| d17-nr | mix50 | 822.4 | 0.51 | 0.29 | 0.29 | 0.39 | 1.8% |
| f&cc | f&cc-qO | 147.8 | 0.14 | 1.00 | 0.14 | 0.37 | 9.5% |
| f&cc | f&cc-q10 | 146.1 | 0.14 | 1.00 | 0.14 | 0.37 | 9.5% |
| d16p | mix50 | 844.4 | 0.45 | 0.30 | 0.30 | 0.37 | 1.6% |
| dclm | dclm-7fw2 | 507.5 | 0.13 | 1.00 | 0.13 | 0.36 | 2.6% |
| d17-nc | mix50 | 719.2 | 0.50 | 0.26 | 0.26 | 0.36 | 1.8% |
| d17-nmc | mix50 | 677.5 | 0.50 | 0.24 | 0.24 | 0.35 | 1.8% |
| f&cc | f&cc-qT | 128.9 | 0.12 | 1.00 | 0.12 | 0.35 | 9.5% |
| fal | mix25 | 342.3 | 0.75 | 0.15 | 0.15 | 0.34 | 3.3% |
| fal | f&cc-q20 | 119.4 | 0.26 | 0.42 | 0.26 | 0.33 | 9.2% |
| d17-nmc | f&cc-qO | 147.8 | 0.11 | 1.00 | 0.11 | 0.33 | 7.4% |
| d17-nmc | f&cc-q10 | 146.1 | 0.11 | 1.00 | 0.11 | 0.33 | 7.4% |
| dclm | mix25 | 967.3 | 0.25 | 0.43 | 0.25 | 0.33 | 1.1% |
| d17-nc | f&cc-qO | 147.8 | 0.10 | 1.00 | 0.10 | 0.32 | 7.0% |
| d17-nmc | c4 | 138.4 | 0.10 | 1.00 | 0.10 | 0.32 | 7.4% |
| d17-nc | f&cc-q10 | 146.1 | 0.10 | 1.00 | 0.10 | 0.32 | 7.0% |
| d17-nc | c4 | 138.4 | 0.10 | 1.00 | 0.10 | 0.31 | 7.0% |
| d17-nmc | f&cc-qT | 128.9 | 0.10 | 1.00 | 0.10 | 0.31 | 7.4% |
| f&cc | mix50 | 528.8 | 0.50 | 0.19 | 0.19 | 0.31 | 1.8% |
| d17-nr | f&cc-qO | 147.8 | 0.09 | 1.00 | 0.09 | 0.30 | 6.1% |
| d17-nc | f&cc-qT | 128.9 | 0.09 | 1.00 | 0.09 | 0.30 | 7.0% |
| d17-nr | f&cc-q10 | 146.1 | 0.09 | 1.00 | 0.09 | 0.30 | 6.1% |
| d17-nf | f&cc-qO | 147.8 | 0.09 | 1.00 | 0.09 | 0.29 | 5.9% |
| dclm-7fw2 | mix75 | 380.8 | 0.75 | 0.11 | 0.11 | 0.29 | 2.3% |
| d17 | f&cc-qO | 147.8 | 0.09 | 1.00 | 0.09 | 0.29 | 5.8% |
| d17-nf | f&cc-q10 | 146.1 | 0.09 | 1.00 | 0.09 | 0.29 | 5.9% |
| d17 | f&cc-q10 | 146.1 | 0.09 | 1.00 | 0.09 | 0.29 | 5.8% |
| d17-nr | c4 | 138.4 | 0.08 | 1.00 | 0.08 | 0.29 | 6.1% |
| d17-nf | c4 | 138.4 | 0.08 | 1.00 | 0.08 | 0.28 | 5.9% |
| d17 | c4 | 138.4 | 0.08 | 1.00 | 0.08 | 0.28 | 5.8% |
| d17-nr | f&cc-qT | 128.9 | 0.08 | 1.00 | 0.08 | 0.28 | 6.1% |
| d16p | f&cc-qO | 147.8 | 0.08 | 1.00 | 0.08 | 0.28 | 5.3% |
| dclm-fw3 | dclm-fw10 | 30.1 | 0.26 | 0.31 | 0.26 | 0.28 | 25.5% |
| d16p | f&cc-q10 | 146.1 | 0.08 | 1.00 | 0.08 | 0.28 | 5.3% |
| d17-nf | f&cc-qT | 128.9 | 0.08 | 1.00 | 0.08 | 0.27 | 5.9% |
| d17 | f&cc-qT | 128.9 | 0.08 | 1.00 | 0.08 | 0.27 | 5.8% |
| d16p | c4 | 138.4 | 0.07 | 1.00 | 0.07 | 0.27 | 5.3% |
| f&cc-q20 | mix25 | 215.0 | 0.75 | 0.10 | 0.10 | 0.27 | 3.3% |
| d16p | f&cc-qT | 128.9 | 0.07 | 1.00 | 0.07 | 0.26 | 5.3% |
| dclm | dclm-7fw3 | 218.0 | 0.06 | 1.00 | 0.06 | 0.24 | 2.6% |
| dclm-7fw2 | mix50 | 254.0 | 0.50 | 0.09 | 0.09 | 0.21 | 1.8% |
| dclm | dclm-q20 | 171.1 | 0.04 | 1.00 | 0.04 | 0.21 | 2.6% |
| fal | mix50 | 228.2 | 0.50 | 0.08 | 0.08 | 0.20 | 1.8% |
| f&cc-q20 | f&cc-qO | 40.3 | 0.14 | 0.27 | 0.14 | 0.20 | 9.5% |
| f&cc-qO | mix25 | 111.2 | 0.75 | 0.05 | 0.05 | 0.19 | 3.3% |
| dclm-7fw3 | mix75 | 163.6 | 0.75 | 0.05 | 0.05 | 0.19 | 2.3% |
| fal | f&cc-qO | 49.7 | 0.11 | 0.34 | 0.11 | 0.19 | 7.4% |
| f&cc-q10 | mix25 | 109.9 | 0.75 | 0.05 | 0.05 | 0.19 | 3.3% |
| c4 | mix25 | 104.4 | 0.75 | 0.05 | 0.05 | 0.19 | 3.3% |
| fal | f&cc-qT | 45.2 | 0.10 | 0.35 | 0.10 | 0.19 | 7.7% |
| f&cc-q20 | f&cc-qT | 35.1 | 0.12 | 0.27 | 0.12 | 0.18 | 9.5% |
| d17 | mix75 | 435.5 | 0.25 | 0.13 | 0.13 | 0.18 | 0.8% |
| d17-nf | mix75 | 431.2 | 0.25 | 0.13 | 0.13 | 0.18 | 0.8% |
| f&cc-qT | mix25 | 97.0 | 0.75 | 0.04 | 0.04 | 0.18 | 3.3% |
| d17-nr | mix75 | 415.0 | 0.25 | 0.12 | 0.12 | 0.18 | 0.8% |
| dclm | dclm-fw3 | 118.1 | 0.03 | 1.00 | 0.03 | 0.17 | 2.6% |
| dclm-q20 | mix75 | 128.4 | 0.75 | 0.04 | 0.04 | 0.17 | 2.3% |
| d16p | mix75 | 426.1 | 0.23 | 0.13 | 0.13 | 0.17 | 0.7% |
| fal | f&cc-q10 | 43.5 | 0.10 | 0.30 | 0.10 | 0.17 | 6.5% |
| d17-nc | mix75 | 362.1 | 0.25 | 0.11 | 0.11 | 0.17 | 0.8% |
| d17-nmc | mix75 | 339.6 | 0.25 | 0.10 | 0.10 | 0.16 | 0.8% |
| f&cc-q20 | mix50 | 143.3 | 0.50 | 0.05 | 0.05 | 0.16 | 1.8% |
| dclm | dclm-fw10 | 98.5 | 0.03 | 1.00 | 0.03 | 0.16 | 2.6% |
| f&cc-q10 | f&cc-qO | 21.6 | 0.15 | 0.15 | 0.15 | 0.15 | 10.0% |
| dclm | dclm-q10 | 82.1 | 0.02 | 1.00 | 0.02 | 0.15 | 2.1% |
| dclm-fw3 | mix75 | 88.6 | 0.75 | 0.03 | 0.03 | 0.14 | 2.3% |
| f&cc | mix75 | 264.4 | 0.25 | 0.08 | 0.08 | 0.14 | 0.8% |
| dclm-7fw3 | mix50 | 109.1 | 0.50 | 0.04 | 0.04 | 0.14 | 1.8% |
| f&cc-q10 | f&cc-qT | 18.7 | 0.13 | 0.14 | 0.13 | 0.14 | 9.9% |
| f&cc-qO | f&cc-qT | 18.6 | 0.13 | 0.14 | 0.13 | 0.14 | 9.8% |
| dclm-fw10 | mix75 | 73.9 | 0.75 | 0.02 | 0.02 | 0.13 | 2.2% |
| dclm-q20 | mix50 | 85.6 | 0.50 | 0.03 | 0.03 | 0.12 | 1.8% |
| dclm-7fw2 | mix25 | 127.3 | 0.25 | 0.06 | 0.06 | 0.12 | 1.1% |
| dclm-q10 | mix75 | 61.6 | 0.75 | 0.02 | 0.02 | 0.12 | 1.9% |
| f&cc-qO | mix50 | 74.1 | 0.50 | 0.03 | 0.03 | 0.12 | 1.8% |
| f&cc-q10 | mix50 | 73.3 | 0.50 | 0.03 | 0.03 | 0.11 | 1.8% |
| c4 | mix50 | 69.6 | 0.50 | 0.02 | 0.02 | 0.11 | 1.8% |
| f&cc-qT | mix50 | 64.6 | 0.50 | 0.02 | 0.02 | 0.11 | 1.8% |
| dclm-fw3 | mix50 | 59.1 | 0.50 | 0.02 | 0.02 | 0.10 | 1.8% |
| dclm-fw10 | mix50 | 49.3 | 0.50 | 0.02 | 0.02 | 0.09 | 1.8% |
| fal | mix75 | 114.1 | 0.25 | 0.03 | 0.03 | 0.09 | 0.8% |
| dclm-q10 | mix50 | 41.1 | 0.50 | 0.01 | 0.01 | 0.09 | 1.5% |
| dclm-7fw3 | mix25 | 54.7 | 0.25 | 0.02 | 0.02 | 0.08 | 1.1% |
| dclm-7fw2 | dclm-q20 | 22.5 | 0.04 | 0.13 | 0.04 | 0.08 | 2.6% |
| f&cc-q20 | mix75 | 71.7 | 0.25 | 0.02 | 0.02 | 0.07 | 0.8% |
| dclm-q20 | mix25 | 42.9 | 0.25 | 0.02 | 0.02 | 0.07 | 1.1% |
| dclm-7fw2 | dclm-fw3 | 15.5 | 0.03 | 0.13 | 0.03 | 0.06 | 2.6% |
| dclm-7fw2 | dclm-fw10 | 13.0 | 0.03 | 0.13 | 0.03 | 0.06 | 2.6% |
| dclm-fw3 | mix25 | 29.6 | 0.25 | 0.01 | 0.01 | 0.06 | 1.1% |
| dclm-7fw2 | dclm-q10 | 10.8 | 0.02 | 0.13 | 0.02 | 0.05 | 2.1% |
| f&cc-qO | mix75 | 37.1 | 0.25 | 0.01 | 0.01 | 0.05 | 0.8% |
| f&cc-q10 | mix75 | 36.6 | 0.25 | 0.01 | 0.01 | 0.05 | 0.8% |
| dclm-fw10 | mix25 | 24.7 | 0.25 | 0.01 | 0.01 | 0.05 | 1.1% |
| c4 | mix75 | 34.8 | 0.25 | 0.01 | 0.01 | 0.05 | 0.8% |
| dclm-7fw3 | dclm-q20 | 9.7 | 0.04 | 0.06 | 0.04 | 0.05 | 2.6% |
| f&cc-qT | mix75 | 32.3 | 0.25 | 0.01 | 0.01 | 0.05 | 0.8% |
| dclm-q10 | mix25 | 20.6 | 0.25 | 0.01 | 0.01 | 0.05 | 0.9% |
| dclm-7fw3 | dclm-fw3 | 6.7 | 0.03 | 0.06 | 0.03 | 0.04 | 2.6% |
| dclm-7fw3 | dclm-fw10 | 5.6 | 0.03 | 0.06 | 0.03 | 0.04 | 2.6% |
| dclm-fw3 | dclm-q20 | 5.2 | 0.04 | 0.03 | 0.03 | 0.04 | 2.6% |
| dclm-7fw3 | dclm-q10 | 4.6 | 0.02 | 0.06 | 0.02 | 0.03 | 2.1% |
| dclm-fw10 | dclm-q20 | 4.4 | 0.04 | 0.03 | 0.03 | 0.03 | 2.6% |
| dclm-fw3 | dclm-q10 | 2.5 | 0.02 | 0.03 | 0.02 | 0.03 | 2.1% |
| dclm-fw10 | dclm-q10 | 2.1 | 0.02 | 0.03 | 0.02 | 0.02 | 2.1% |
| fw-pro | fw-edu | 1.0 | 0.01 | 0.01 | 0.01 | 0.01 | 0.5% |


Low and high bounds for the uncertain filter-filter pairs are in `scratchpad/agents/09/rules_low.txt` and `rules_high.txt`. The high bound turns several pairs into near-duplicates: f&cc q10/qO cos 0.96, q10/qT 0.93, dclm fw3/fw10 0.91, q20/qO 0.72, 7fw3/fw10 0.67.

### Effective number of independent points (design-only, no outcome used)
For a correlation between two variables measured on N units whose dependence matrix is R, the Clifford, Richardson & Hémon (1989) / Dutilleul (1993) effective sample size is n_eff = 1 + N²/tr(R²).

| R used | point | high-overlap bound |
|---|---|---|
| cos (doc-iid; lower bound on dependence) | 8.2 | 6.8 |
| 1-TV | 10.7 | 9.0 |
| max(C(A in B), C(B in A)) (any nesting counts as full dependence) | 4.0 | 3.6 |

So the 25 points carry the information of roughly **4 to 11 independent points**, depending on whether nested selections count as dependent. With all 25 treated as independent, df is overstated by a factor of about 2.3 to 6.

## 5. Rules and kept/dropped lists

Pre-specified tie-break for every greedy rule: process the DataDecide "Original" recipes first (README labels: dclm-baseline, dolma1_7, dolma1_6plus, falcon-and-cc, falcon, fineweb-edu, c4, fineweb-pro, ordered by corpus size), then the derived recipes by decreasing size. A recipe is dropped if its measure against an already-kept recipe exceeds X. This order is fixed by provenance, not by any outcome. One sensitivity case uses pure size order; it shows that the order matters (mix25 becomes the Dolma representative), which is why the provenance order is pre-specified.

**Mixtures excluded a priori (R-mix).** For additive count data, a mix is (up to the random file subset) a convex combination of two other recipes in the set: c_mix(n) ≈ λ·c_DCLM(n) + (1-λ)·c_Dolma(n). It adds no new independent corpus. It is a legitimate dose-response treatment for the model side, but on the data side it is a derived unit. Dropping derived or linear-combination units is the standard fix (in Hurlbert's terms they are not independent experimental units). Drop: mix25, mix50, mix75.

| Rule | Measure, X | Justification | KEEP | n |
|---|---|---|---|---|
| **R0** all | none | baseline; pseudo-replicated | all 25 | 25 (n_eff 4-11) |
| **R1** one point per independent lineage (components, mixes removed) | containment > 0.05, any | Hurlbert 1984: experimental unit = independent source corpus | dclm-baseline, dolma1_7, fineweb-edu, fineweb-pro (strict variant with FineWeb merged: 3) | 4 (3) |
| **R2** keep base, drop subsets | C(A in kept) > 0.5, provenance order, R-mix | a recipe whose training docs are mostly (>50%) inside another kept recipe is a subsample of that unit. Any X in (0.01, 0.89) gives the same list, because filtered subsets and ablations have C = 1.0 by construction (the lowest relevant pair, fineweb-pro in fineweb-edu, is 0.01) | dclm-baseline, dolma1_7, fineweb-edu, fineweb-pro | 4 |
| R2b | same, size order, mixes kept | sensitivity | dclm-baseline, mix25, fineweb-edu, fineweb-pro | 4 |
| **R3** shared-variance rule | cos > 0.5, provenance order, R-mix | cos is the expected correlation induced by shared documents; > 0.5 means > 25% shared variance | dclm-baseline, dolma1_7, fineweb-edu, c4, fineweb-pro, dclm-7fw2, f&cc-q20, dclm-q20, f&cc-qO, f&cc-qT, dclm-fw3, dclm-fw10 | 12 |
| R3 under high-overlap bound | cos > 0.5 | sensitivity | dclm-baseline, dolma1_7, fineweb-edu, c4, fineweb-pro, dclm-7fw2, f&cc-q20, dclm-fw3 | 8 |
| R3c | cos > 0.3 (point) | sensitivity | dclm-baseline, dolma1_7, fineweb-edu, c4, fineweb-pro, dclm-7fw3, dclm-q20, f&cc-qO, f&cc-q10, f&cc-qT, dclm-fw3, dclm-fw10 | 12 (7 at high bound) |
| R3d | cos > 0.7 (point) | sensitivity | R3 list + falcon, dclm-7fw3, dclm-q10 | 15 (11 at high bound) |
| **R4** near-duplicate (strict pseudo-replicate) rule | 1-TV > 0.5, R-mix | drop a model whose training distribution is more than half identical to a kept one, i.e. the same treatment replicated | dclm-baseline, dolma1_7, falcon, fineweb-edu, c4, fineweb-pro, dclm-7fw2, f&cc-q20, dclm-7fw3, dclm-q20, f&cc-qO, f&cc-qT, dclm-fw3, dclm-fw10, dclm-q10 | 15 (11 at high bound) |
| R4b | 1-TV > 0.8 | only near-identical distributions | R4 + falcon-and-cc, dolma1_7-no-math-code, f&cc-q10 | 18 (15 at high bound) |

Drop reasons, point estimates, with the measure against the kept recipe that triggered the drop:
- **R2:** d16p (in d17: 0.89 / 0.98); f&cc, falcon, c4, all 4 dolma ablations, all 4 f&cc-qc (C = 1.00 in d17); all 6 dclm-qc (C = 1.00 in dclm-baseline); 3 mixes (R-mix).
- **R3 (cos > 0.5):** d16p (0.93), f&cc (0.78), falcon (0.51, borderline), d17-nf (1.00), -nr (0.98), -nc (0.91), -nmc (0.89), dclm-7fw3 (vs 7fw2: 0.66), f&cc-q10 (vs q20: 0.71), dclm-q10 (vs q20: 0.69), 3 mixes.
- **R4 (1-TV > 0.5):** d16p (0.89), f&cc (0.62), d17-nf (0.99), -nr (0.95), -nc (0.83), -nmc (0.79), f&cc-q10 (vs q20: 0.51, borderline), 3 mixes.

**Recipes kept by every rule:** dclm-baseline, dolma1_7, fineweb-edu, fineweb-pro. **Recipes dropped by every rule except R0:** dolma1_6plus, dolma1_7-no-flan, -no-reddit, -no-code, the 3 mixes. **Most fragile decisions:** falcon (cos 0.51 to d17), f&cc-q10 (1-TV 0.51 to q20), dclm-q10 (1-TV 0.48 to q20), and every dclm or f&cc cross-filter pair whose overlap is unknown.

### Recommended design (stated before seeing any outcome)
1. **Primary inferential analysis: keep all 25 and model the dependence; do not delete.** Use lineage (Dolma / DCLM / FineWeb; mixes assigned to both through multiway clustering, or excluded) as a blocking or random factor, or use GLS with R = the overlap matrix and report n_eff-adjusted tests. Remedies for pseudoreplication: Hurlbert 1984; Millar & Anderson 2004 (Fish. Res.); Lazic 2010 (BMC Neurosci.). Caveat: with 3-4 clusters, cluster-robust SEs are invalid; use a wild cluster bootstrap or exact within-block permutation (Cameron, Gelbach & Miller 2008).
2. **Identification comes from within-lineage contrasts** (filter or ablation versus its base, with lineage fixed effects). This is a randomized-block / split-unit design: valid for "does manipulating the data of one corpus move both quantities together", but per Hurlbert it does not generalize across corpora, because the treatments are applied to subsamples of a single unit.
3. **Robustness check: R2 (n = 4) and R3 (n = 12; 8 at the high bound), both reported whatever their result.** Design fact: the two-sided p < .05 critical |r| is 0.950 at n = 4, 0.707 at n = 8, 0.576 at n = 12, 0.514 at n = 15 and 0.396 at n = 25. R2 is therefore descriptive only. R3 or R4 is the largest set that is defensible as near-independent.
4. Report R4 alongside R3 because they differ in principle. R4 removes only strict pseudo-replicates (near-identical training distributions). R3 also removes nested filters of moderate size.

### Which rules are standard
- **Standard:** one unit per independent experimental unit, i.e. avoiding pseudo-replication (Hurlbert 1984, Ecol. Monogr. 54:187); treating subsamples, sub-filters and repeated measures of one unit as non-independent; excluding derived or linear-combination units (the mixes); effective sample size under correlated observations (Clifford et al. 1989; Dutilleul 1993); the overlapping-sample correlation n_shared/sqrt(n1·n2) (Lin & Sullivan 2009); blocking / mixed models / cluster bootstrap as the alternative to deletion.
- **Not standard (convention chosen here):** the thresholds X = 0.5 for containment, cos and 1-TV, and the provenance-first tie-break order. There is no canonical X. 0.5 was chosen as "majority shared" (for cos, 25% shared variance), with 0.3, 0.7 and 0.8 shown as sensitivity. R2's list does not depend on X for any X in (0.01, 0.89).

## 6. Honest notes and limitations
- **Two measures give opposite answers for filtered subsets.** By containment, every qc filter is 100% inside its base (a pseudo-replicate). By shared distribution (1-TV, cos), top-k% filters are mostly *different* data from their base (1-TV 0.02-0.27). The right view depends on whether the filter is the treatment (then the filter is a valid manipulation and the base is its block) or noise (then it is a sub-sample). Pseudo-replication in Hurlbert's strict sense applies to random subsamples; the ablations (1-TV 0.79-0.99) come closest to that.
- **Lower bound on dependence.** All cross-pipeline overlaps (C4, RefinedWeb, Dolma CC, DCLM and FineWeb all from Common Crawl, with overlapping snapshots and URLs) are set to 0 because they were not measured. Pipeline-level shared conventions (extraction, number formatting, dedup) are not captured by any document-overlap measure. True dependence is higher than the table shows.
- **Assumptions.** qc docs are spread uniformly over base files (needed for qc-vs-mix overlaps). Dolma-1.7 CC and C4 are doc-subsets of the Dolma-1.6 versions (CC re-binned head/middle/tail; 1.7 middle bin 242 B > 1.6 middle 190 B, so containment may be < 100%). The f&cc-qc CC part is 17% of Dolma-1.7 CC tokens at "top 10%", which suggests a doc-count percentile or a larger pre-filter pool, so containment in d17 may be < 1. Tulu v0 vs v2 is treated as disjoint (≤ 1.8 B). Per-file tokens are per-pool averages; mix sizes come out 0.6-0.7% off the mixture.json totals.
- **Unknown cross-filter overlaps** within DCLM and within falcon-and-cc dominate the uncertainty in R3 and R4 (12 vs 8, 15 vs 11 kept). Exact Jaccard on document IDs or URLs from the HF files (`allenai/DataDecide-data-recipes`) would settle them. HF and arXiv were blocked from this session.
- Realized token identity between trained streams is small (< 10%) except for nested small filters, so training-sample overlap is a minor issue compared with population overlap.
- No outcome was used: rule order, thresholds and the recommended primary analysis were fixed from provenance and design arguments only.
