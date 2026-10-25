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

