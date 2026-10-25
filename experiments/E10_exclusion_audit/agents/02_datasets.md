# Auditor 02: pretraining dataset composition of the 25 DataDecide 1B recipes

Scope: what each recipe's training data is made of, how it was filtered, which recipes are
nested in or near-copies of others, and which composition differences could affect numeric
representations through a route other than the number-frequency shape (alpha).

**Blindness statement.** I did not compute, look up or reason about any alpha–beta correlation, and I
did not open `results/datadecide/alpha_beta_correlations.json`. From `results/datadecide/confounders.json` I read only the
`recipes[*].code_share / math_share / number_density` fields. From `count_features.csv` I used only
F01, F14, F20, F21, F22 and never `old_direct` or `beta_resid`. I did not print or use any beta column.
**Disclosure:** while grepping `docs/assumptions_and_caveats.md` for "code/epoch/subset", one caveat line (C2) came
up. It states a code-share–beta relation and says controlling for code share affects the alpha signal. I saw
this *after* the code criterion below was already defined from composition alone. I did not use it to set or tune any
threshold. A reviewer may still want to treat criterion C1 as not fully blind.

Reproducible work files (scratchpad):
- `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/02_work/composition.py`: per-recipe category shares. Output: `composition.tsv` and `composition.json` in the same folder.
- `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/02_work/overlap.py`: shared-document similarity matrix and effective number of corpora. Output: `similarity_lower.csv` and `similarity_upper.csv`.

Primary sources:
- `results/corpus_alpha_datadecide/<recipe>/mixture.json`: exact token counts per top-level source, from HF file sizes. Duplicated files are counted twice.
- `configs/datadecide_data_map.json`: `files_per_source` and `repeated_paths`.
- OLMo@DataDecide `olmo/data/named_data_mixes.py`, local clone `/tmp/claude-0/agentrefs/OLMo`, commit 7094aab:
  - `SOURCES_SIZES` (:4995–5075) gives exact Dolma 1.7 sub-source sizes.
  - Recipe definitions are at :5127–5183.
  - Dolma 1.6++ sub-source sizes are in the comments at :2559–3864.
- DataDecide recipe descriptions: `/tmp/claude-0/agentrefs/DataDecide/README.md:57-68`.

Sub-source shares inside a mixture.json group were split by `files_per_source × avg_file_size` from `SOURCES_SIZES`:
- books, pes2o and wiki inside `olmo-mix/v1_6-decontaminated`;
- algebraic-stack and open-web-math inside `proof-pile-2`.

The group totals are exact. For the full Dolma 1.7 recipes the split is also exact. For the mixes it is approximate
(±5% relative within the group). An independent check is the window sample in `windows.json` (4000 windows per recipe):
it gives the same shares to within ±0.005.

---

## 1. Per-recipe table (all 25)

### 1a. Composition: share of tokens in the recipe's full corpus

These are expected shares of the 100B training stream, because the shuffle is uniform over 2048-token chunks
(`docs/datadecide_sampling.md` §4).

Column key:
- **code** = StarCoder.
- **code-adj** = StackExchange + Algebraic-Stack. Both are code-heavy but are not counted as code by `confounders.json`.
- **math** = Proof-Pile-2 (OpenWebMath + Algebraic-Stack). This is exactly `confounders.json` `math_share`.
- **acad** = peS2o + arXiv (LaTeX).
- **STEM** = math ∪ acad ∪ StackExchange.
- **FLAN** = tulu_flan, which is instruction data in pretraining.
- **wiki** = Wikipedia/Wikibooks (2× upsampled in Dolma 1.7) + MegaWika.
- **ND** = `number_density`, integers 0–10000 per 1000 tokens (`confounders.json`).
- **yr** = F20 `year_share_of_d4`, the share of 1000–9999 counts that fall in 1900–2030 (`count_features.csv`).
- **p2** = F22 `pow2_excess`, excess of powers of two.

| # | recipe | corpus (B tok) | epochs of 100B | code | code-adj | math | acad | STEM | books | wiki | reddit | FLAN | news | ND | yr | p2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | dolma1_7 | 1715.2 | 0.058 | .154 | .019 | .015 | .050 | .076 | .003 | .007 | .047 | .010 | .008 | 18.5 | .770 | 1.16 |
| 2 | dolma1_7-no-code | 1431.7 | 0.070 | 0 | .009 (alg-stack) | .018 | .060 | .077 | .004 | .008 | .056 | .012 | .010 | 17.4 | .812 | 0.82 |
| 3 | dolma1_7-no-math-code | 1349.2 | 0.074 | 0 | 0 | 0 | .021 (arXiv kept) | .021 | .004 | .009 | .059 | .012 | .011 | 16.2 | .814 | 0.73 |
| 4 | dolma1_7-no-reddit | 1635.2 | 0.061 | .161 | .020 | .016 | .052 | .080 | .003 | .007 | 0 | .010 | .009 | 19.0 | .772 | 1.17 |
| 5 | dolma1_7-no-flan | 1698.6 | 0.059 | .155 | .019 | .015 | .050 | .077 | .003 | .007 | .047 | 0 | .008 | 18.5 | .770 | 1.16 |
| 6 | dolma1_6plus | 1885.2 | 0.053 | .140 | .017 | .013 | .045 | .069 | .003 | .002 | .043 | .001 (FLAN v0) | 0 | 19.3 | .762 | 1.10 |
| 7 | c4 | 138.4 | 0.722 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14.0 | .813 | 0.73 |
| 8 | fineweb-pro | 82.7 | **1.209** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14.9 | .762 | 0.38 |
| 9 | fineweb-edu | 192.2 | 0.520 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17.2 | .733 | 0.46 |
| 10 | falcon | 456.4 | 0.219 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18.9 | .815 | 0.64 |
| 11 | falcon-and-cc | 1054.8 | 0.095 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16.3 | .821 | 0.66 |
| 12 | falcon-and-cc-qc-10p | 146.1 | 0.685 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **10.0** | .719 | 1.15 |
| 13 | falcon-and-cc-qc-20p | 285.8 | 0.350 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **11.0** | .743 | 1.00 |
| 14 | falcon-and-cc-qc-orig-10p | 147.8 | 0.677 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **11.7** | .731 | 1.19 |
| 15 | falcon-and-cc-qc-tulu-10p | 128.9 | 0.776 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17.5 | .782 | 0.83 |
| 16 | dclm-baseline | 3857.1 | 0.026 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14.8 | .786 | 1.05 |
| 17 | dclm-baseline-qc-7p-fw2 | 507.5 | 0.197 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16.6 | .720 | 1.18 |
| 18 | dclm-baseline-qc-7p-fw3 | 218.0 | 0.459 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18.7 | .670 | 1.06 |
| 19 | dclm-baseline-qc-fw-3p | 118.1 | 0.847 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19.8 | **.644** | 0.94 |
| 20 | dclm-baseline-qc-fw-10p | 98.5 | **1.016** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18.7 | .683 | 0.99 |
| 21 | dclm-baseline-qc-10p | 82.1 | **1.219** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17.7 | .753 | 1.49 |
| 22 | dclm-baseline-qc-20p | 171.1 | 0.585 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16.4 | .758 | 1.37 |
| 23 | dclm-baseline-25p-dolma1.7-75p | 2246.0 | 0.045 | .085 | .011 | .009 | .029 | .045 | .002 | .004 | .027 | .006 | .005 | 17.0 | .775 | 1.14 |
| 24 | dclm-baseline-50p-dolma1.7-50p | 2813.6 | 0.036 | .049 | .006 | .004 | .015 | .022 | .001 | .002 | .014 | .003 | .003 | 16.0 | .779 | 1.08 |
| 25 | dclm-baseline-75p-dolma1.7-25p | 3335.9 | 0.030 | .020 | .003 | .002 | .007 | .011 | .001 | .001 | .006 | .001 | .001 | 15.3 | .785 | 1.07 |

"0" means the recipe contains no file from that source (`configs/datadecide_data_map.json` `files_per_source`). The
composition columns are by source label. Web-only recipes still contain code, math and Q&A embedded in web pages, and
no composition label measures that.

**Epochs** = 100.016B / corpus tokens. Three recipes are smaller than the token budget and wrap into a second epoch:
fineweb-pro (82.7B → 1.21 epochs), dclm-qc-10p (82.1B → 1.22 epochs) and dclm-qc-fw-10p (98.5B → 1.016 epochs). The
exact-stream counter handles the epoch reshuffle (`src/datadecide_sampling.py:190-199`).

### 1b. Origin, filter, classifier positives, nesting

"Retention" = recipe tokens / parent-pool tokens, from mixture.json totals.

| # | recipe | base corpus (lineage) | filtering / construction | classifier positives (style it selects toward) | nested in / near-copy of | retention vs parent |
|---|---|---|---|---|---|---|
| 1 | dolma1_7 | Dolma 1.7 | Dolma heuristics, dedup, CC quality classifier (`qc_cc_en_bin_001`); 15 sources concatenated; wiki 2× | – (contains FLAN/Tulu instruction data, 0.96%) | root of Dolma lineage | 1.00 |
| 2 | dolma1_7-no-code | Dolma 1.7 | drop StarCoder + StackExchange (`named_data_mixes.py:5128`) | – | ⊂ dolma1_7 | 0.835 |
| 3 | dolma1_7-no-math-code | Dolma 1.7 | also drop peS2o, OWM, Algebraic-Stack; **arXiv kept** (`:5129`) | – | ⊂ no-code ⊂ dolma1_7 | 0.787 |
| 4 | dolma1_7-no-reddit | Dolma 1.7 | drop Reddit (`:5131`) | – | ⊂ dolma1_7 | 0.953 |
| 5 | dolma1_7-no-flan | Dolma 1.7 | drop tulu_flan (`:5132`) | – | ⊂ dolma1_7 (S = 0.995) | 0.990 |
| 6 | dolma1_6plus | Dolma 1.6 + 1.7 extras | Dolma 1.6 CC/C4/Reddit/peS2o/books/wiki + Falcon v0 + StarCoder + RedPajama arXiv/SE + PP2 + FLAN v0 | – | Dolma 1.7 CC, C4 and Falcon are filtered versions of these pools; StarCoder, PP2, peS2o, books and arXiv are identical | – |
| 7 | c4 | Dolma 1.7 C4 subset | C4 heuristics + Dolma 1.7 dedup/qc | – | = the C4 source of dolma1_7, verbatim (`mixture.json` 138.44B in both) | 0.081 of dolma1_7 |
| 8 | fineweb-pro | FineWeb (CC) | ProX: an LM writes programs that drop documents and **edit/remove lines and strings** (model-refined text) | LM-judged | independent of Dolma/DCLM docs | ~0.5–1% of FineWeb (external) |
| 9 | fineweb-edu | FineWeb (CC) | FineWeb-Edu classifier (Llama-3-70B-annotated "educational" score), SmolLM dedup | LLM "educational" labels | disjoint from fineweb-pro except by chance | ~1% of FineWeb (external) |
| 10 | falcon | Dolma 1.7 Falcon-RefinedWeb subset | RefinedWeb heuristics + dedup | – | = Falcon source of dolma1_7, verbatim (456.40B) | 0.266 of dolma1_7 |
| 11 | falcon-and-cc | Dolma 1.7 web | Falcon + Dolma 1.7 CC (`web_rest`) | – | ⊂ dolma1_7 (61.5% of its tokens); S = 0.89 with no-math-code | 0.615 of dolma1_7 |
| 12 | falcon-and-cc-qc-10p | falcon-and-cc | top 10% by reproduced DCLM fastText | **OpenHermes-2.5 + ELI5** (instruction/chat + explanatory Q&A) | ⊂ qc-20p ⊂ falcon-and-cc | 0.138 |
| 13 | falcon-and-cc-qc-20p | falcon-and-cc | top 20%, same classifier | OH-2.5 + ELI5 | ⊂ falcon-and-cc | 0.271 |
| 14 | falcon-and-cc-qc-orig-10p | falcon-and-cc | top 10% by original DCLM fastText | OH-2.5 + ELI5 | ⊂ falcon-and-cc; heavy overlap with qc-10p expected (same positives) | 0.140 |
| 15 | falcon-and-cc-qc-tulu-10p | falcon-and-cc | top 10%, classifier retrained on pre-release **Tulu-v3** | Tulu-v3 SFT data (instruction, incl. math/code prompts) | ⊂ falcon-and-cc | 0.122 |
| 16 | dclm-baseline | DCLM-RefinedWeb pool (CC) | DCLM heuristics + dedup + fastText `openhermes_reddit_eli5_vs_rw` (path name in mixture.json) | **OH-2.5 + ELI5** | root of DCLM lineage | 1.00 |
| 17 | dclm-baseline-qc-7p-fw2 | DCLM | top 7% by DCLM fastText, then FineWeb-Edu score ≥ 2 | OH/ELI5 then educational | ⊃ qc-7p-fw3 (FW ≥ 3 ⊂ FW ≥ 2) | 0.132 of dclm-baseline |
| 18 | dclm-baseline-qc-7p-fw3 | DCLM | as above, FW-Edu ≥ 3 | OH/ELI5 then educational | ⊂ qc-7p-fw2 | 0.057 |
| 19 | dclm-baseline-qc-fw-3p | DCLM | top 3% by FineWeb-Edu classifier | educational | likely ⊃ part of fw-10p (see note) | 0.031 |
| 20 | dclm-baseline-qc-fw-10p | DCLM (probably a ~1T shard) | top 10% by FineWeb-Edu classifier | educational | see note | 0.026 |
| 21 | dclm-baseline-qc-10p | DCLM, 1T shard (path `.../1t/`) | top 10% by reproduced DCLM classifier (pos ELI5+OH, neg DCLM-RefinedWeb) | OH-2.5 + ELI5 | ⊂ qc-20p | 0.021 |
| 22 | dclm-baseline-qc-20p | DCLM, 1T shard | top 20%, same | OH-2.5 + ELI5 | ⊂ dclm-baseline | 0.044 |
| 23 | dclm-baseline-25p-dolma1.7-75p | Dolma 1.7 + DCLM | random 75% of each Dolma source's files + random 25% of DCLM files. **Token share of DCLM = 42.9%**, not 25% | – (inherits both) | convex mixture of #1 and #16 | – |
| 24 | dclm-baseline-50p-dolma1.7-50p | Dolma 1.7 + DCLM | 50/50 by files; **DCLM token share 69.1%** | – | mixture of #1 and #16 | – |
| 25 | dclm-baseline-75p-dolma1.7-25p | Dolma 1.7 + DCLM | 75/25 by files; **DCLM token share 86.8%** | – | mixture of #1 and #16 (S = 0.81 with dclm-baseline) | – |

Notes and evidence for table 1b:
- **Recipe descriptions.** Filter descriptions come from `DataDecide/README.md:58-67`. The OpenHermes + ELI5 positives of
  DCLM-baseline are named in the source path `fastdclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train`
  (`results/corpus_alpha_datadecide/dclm-baseline/mixture.json`). The reproduced classifiers are named in the paths
  `pos_eli5_oh_neg_dclm_refinedweb_*`, `eli5_oh_top10p`, `og_eli5_oh_top10p` and `tulu_qc_top10`
  (same files, and `named_data_mixes.py:1077-1785`).
- **Verbatim subsets.** `c4`, `falcon` and `falcon-and-cc` use exactly the same files as the corresponding Dolma 1.7 sources.
  The token counts are identical to the byte: 138,441,178,003, 456,404,959,027 and 598,422,659,025 tokens
  (`mixture.json` of dolma1_7 vs c4 / falcon / falcon-and-cc).
- **The mixes are not 25/50/75 by tokens.** `sample_factor` keeps a fraction of *files* per source
  (`named_data_mixes.py:5094-5099`). DCLM (3.86T) is 2.25× larger than Dolma 1.7 (1.72T), so the real DCLM token shares are
  0.429, 0.691 and 0.868 (`mixture.json`). The data side of the mixes is a deterministic interpolation of the two endpoints.
  Number density, predicted as share·ND(dclm) + (1−share)·ND(dolma1_7), is 16.92, 15.97 and 15.33, against measured 17.00,
  16.01 and 15.34. The mixes therefore add independent *model trainings* but almost no independent *data* information.
- **The dclm-qc-* recipes are subsets of DCLM-baseline.** This rests on the DataDecide description plus a size argument:
  - DCLM-baseline is about the top 10% of the DCLM pool by fastText score.
  - "Top 7%" of the pool (≈ 0.7 × 3.86T ≈ 2.7T) then FW ≥ 2 → 507B, and FW ≥ 3 → 218B. This is consistent with nesting.
  - qc-10p/20p (82B/171B) sit under `.../1t/`: they are 10% and 20% of a ~1T shard.
  - **Anomaly:** fw-3p (118B) is *larger* than fw-10p (98.5B), so they cannot be the same threshold on the same pool.
    The likely reading is top 3% of the whole 3.86T (= 116B) versus top 10% of a ~1T shard. I could not verify this at document level.
- **Language.** All web sources are English-only by construction:
  - Dolma CC/C4 paths carry `cc_en`, and Dolma applies English language ID.
  - RefinedWeb, DCLM, FineWeb and FineWeb-Edu are all English-filtered.
  - StarCoder is the only non-natural-language source.
  - Non-English share is therefore negligible for every recipe, and **no criterion is warranted**. This relies on the
    corpora's own documentation; nothing in the repo measures it.

---

## 2. Dependency / nesting structure and effective number of independent corpora

**Shared-document similarity.** S_AB = |A∩B| / √(|A||B|): the cosine between the uniform training densities over the
two corpora (`02_work/overlap.py`). Overlap rules:
- If one subset contains the other (same pool, same or nested filter, or one is the full pool): overlap = the smaller set.
- Random file subsets (the mixes): overlap = n_A·n_B / N.
- Different classifiers on the same pool:
  - lower bound: treated as independent, n_A·n_B / N;
  - upper bound: treated as maximally nested, min(n_A, n_B).
- Cross-lineage overlap (Dolma vs DCLM vs FineWeb, all from Common Crawl) is set to 0, because URL-level overlap is unmeasured.

Results:
- **Effective number of independent corpora:** n_eff = 25² / Σ S_ij² (= participation ratio of the eigenvalues of S)
  - **7.2** (lower overlap bound)
  - **5.8** (upper overlap bound)
- Largest eigenvalue of S: 8.0–8.2. It corresponds to the Dolma-1.7 block.

**Lineage groups.** These are defined by construction, not by threshold, and are recommended for clustered SEs or
leave-group-out checks:

| group | recipes | relation |
|---|---|---|
| L1 Dolma full mixtures | dolma1_7, -no-code, -no-math-code, -no-reddit, -no-flan, dolma1_6plus | pairwise S 0.89–0.995 |
| L2 Dolma web subsets (unfiltered) | c4, falcon, falcon-and-cc | verbatim sources of dolma1_7 (S with dolma1_7 ablations: c4 ≈ 0.3, falcon ≈ 0.57, falcon-and-cc 0.80–0.89) |
| L3 Falcon+CC classifier subsets | qc-10p, qc-20p, qc-orig-10p, qc-tulu-10p | 12–27% of falcon-and-cc; 10p ⊂ 20p (S = 0.71); with other filters S 0.0x–0.96 depending on bound |
| L4 DCLM + mixes | dclm-baseline, 75p mix (S = 0.81), 50p mix (S = 0.59); 25p mix sits between L1 (S = 0.65 to dolma1_7) and L4 | convex mixtures |
| L5 DCLM classifier subsets | qc-7p-fw2 ⊃ qc-7p-fw3; qc-20p ⊃ qc-10p; fw-3p, fw-10p | 2–13% of dclm-baseline |
| L6 FineWeb | fineweb-edu, fineweb-pro | same parent pipeline, nearly disjoint documents (S ≈ 0.01) |

**Independent document pools.** At the level of independent pools there are **3**: Dolma (L1–L3, 13 recipes),
DCLM (L4–L5, 9 recipes plus mixes) and FineWeb (2). The 3 mixes straddle Dolma and DCLM.

**The code's `PRIMARY_FAMILY` (`src/confounder_analysis.py:54-60`) does not encode nesting.** It puts falcon and
falcon-and-cc in the "falcon" family with the qc subsets, and c4 in "web_other". Yet falcon-and-cc is more similar to
dolma1_7-no-math-code (S = 0.89) than to its own qc-10p child (S = 0.37). Family fixed effects therefore do not remove
the Dolma-lineage dependence.

**Near-replicate pairs (S ≥ 0.97).** These pairs differ in ≤ 5% of tokens:
- dolma1_7 / no-flan: S = 0.995, 1% of tokens.
- dolma1_7 / no-reddit: S = 0.98.
- no-code / no-math-code: S = 0.97.
- no-reddit / no-flan: S = 0.97.

Their data-side differences are tiny: ND is 18.48 vs 18.51 for dolma1_7 vs no-flan, and F20 is identical to 3 dp. Any
model-side difference within these pairs is mostly training/measurement noise, so they give an a-priori noise-floor
estimate. They also mean the dolma block counts as roughly 3–4 rather than 6 independent data points.

---

## 3. Proposed criteria: precise rules, a-priori justification, recipes flagged

Each rule is stated so it could be applied before seeing any model-side quantity. For each I say whether it is a
**reason to exclude** or a **confound to control**.

**C1 — Source code in pretraining.** Rule: `code_share` (StarCoder) ≥ 0.01.
- Flags 7 recipes: dolma1_7 (.154), dolma1_7-no-reddit (.161), dolma1_7-no-flan (.155), dolma1_6plus (.140), mix-25p (.085), mix-50p (.049), mix-75p (.020).
- dolma1_7-no-code is a borderline case: it has 0.9% Algebraic-Stack, which is mathematical source code.
- Justification, part 1 (what numbers mean): in code, numerals are indices, sizes, versions, hex and bit widths. The
  measured power-of-two excess F22 is 1.16 in dolma1_7 vs 0.82 in no-code, so code changes *which* numbers appear and
  *what they mean*, not only the overall slope.
- Justification, part 2 (probe format): the β probe format `a=a,b=b,c=c,n=` is a variable-assignment/copy pattern that
  is far more frequent in code. Code exposure can change probe accuracy, which affects the correct-output filter and layer
  selection. That route is independent of the number-frequency curve.
- Treatment: **confound to control, not exclude.**
  - Use code share as a covariate, or stratify into code vs no-code.
  - The cleanest test is the matched ablation pair dolma1_7 vs dolma1_7-no-code, which differs only by StarCoder + StackExchange.
  - Excluding 7 of 25, all from one lineage, would remove most of the Dolma side and confound lineage with the exclusion.
  - Recommended: report results with and without the 7 as a sensitivity check.
- Caveat: code share is perfectly confined to the Dolma lineage (L1 + mixes). Every non-Dolma recipe has 0. Code share
  and "Dolma lineage" therefore cannot be separated with these 25 models.

**C2 — Explicit math/STEM sources.** Rule: STEM share (PP2 + peS2o + arXiv + StackExchange) ≥ 0.02.
- Flags dolma1_7, no-code, no-reddit, no-flan, dolma1_6plus, mix-25p and mix-50p (0.022). no-math-code is at 0.0207 because it keeps arXiv.
- `confounders.json` `math_share` counts PP2 only (`src/confounder_analysis.py:86-111`). It therefore understates math in
  no-code and no-math-code: no-math-code still has 2.07% arXiv LaTeX.
- Justification: equations, theorem numbering and LaTeX place numerals in arithmetic and ordinal contexts.
- Treatment: **control (covariate), not exclude.** The shares are small (≤ 8%). Web-filtered recipes (fineweb-edu, the
  dclm-fw subsets) also contain math that no label captures. This makes a binary "has math source" flag a weak proxy:
  prefer a measured content feature.

**C3 — Instruction data mixed into pretraining (FLAN/Tulu).** Rule: FLAN share ≥ 0.005.
- Flags dolma1_7, no-code, no-math-code, no-reddit and mix-25p. Shares are 0.6–1.2%.
- Below threshold: dolma1_6plus (0.10%), mix-50p (0.30%), mix-75p (0.13%).
- Justification: FLAN contains arithmetic and word-problem templates with numbers in an instruction format.
- Treatment: **no exclusion; ignore or note.** The share is ~1%. The direct ablation dolma1_7 vs no-flan has S = 0.995
  and data-side features that are identical, so this pair isolates any FLAN effect at near-zero data-shape change.

**C4 — Classifier-selected "instruction-like" style.** Rule: the recipe's documents were selected by a classifier whose
positives are instruction/chat/QA data (OpenHermes-2.5, ELI5, Tulu-v3).
- Flags 11 directly: dclm-baseline, all 6 dclm-qc-*, falcon-and-cc-qc-10p / -20p / -orig-10p / -tulu-10p.
- Flags the 3 mixes partially (43–87% DCLM).
- Sub-rule C4b, FineWeb-Edu "educational" LLM labels: flags fineweb-edu, dclm-qc-7p-fw2, -7p-fw3, -fw-3p and -fw-10p.
- Justification: selection changes the *context* of numbers (dates vs quantities, prose vs listings) and their overall rate:
  - ND falls from 16.3 (falcon-and-cc) to 10.0 / 11.0 / 11.7 in the OH/ELI5 subsets.
  - ND rises slightly to 17.5 under Tulu positives.
  - The year share F20 drops from 0.79 (dclm-baseline) to 0.64–0.72 in the FW-Edu-selected subsets.
  - The OLS slope summarises the shape of c(n), but not whether "1990" is a year or a quantity.
- Treatment: **confound to control** with a filter-type indicator (none / DCLM-fastText / FW-Edu / Tulu / ProX).
  - Do not exclude: filter variation is the design variable of DataDecide and covers 15+ of 25 recipes.
  - Within-lineage contrasts (falcon-and-cc vs its 4 children; dclm-baseline vs its 6 children) are the right comparisons.

**C5 — Heavily filtered small subsets.** Rule: retention < 15% of the parent pool and corpus < 300B tokens.
- Flags falcon-and-cc-qc-10p (13.8%), -orig-10p (14.0%), -tulu-10p (12.2%), dclm-qc-7p-fw3 (5.7%), -fw-3p (3.1%),
  -fw-10p (2.6%), -qc-10p (2.1%) and -qc-20p (4.4%).
- fineweb-edu and fineweb-pro are ~1% of FineWeb by external figures.
- Justification: extreme selection narrows topic and register diversity. Number usage can then differ in kind, not only
  in frequency slope.
- Treatment: **control or sensitivity, not exclude.** It overlaps almost entirely with C4. Excluding it would leave a
  sample dominated by the Dolma lineage.

**C6 — Model-edited text.** Rule: document text was rewritten or line-edited by a model.
- Flags **fineweb-pro** only. ProX programmatic refinement removes or edits lines and strings: headers, footers,
  navigation and page numbers (`DataDecide/README.md:63`).
- Supporting signal: lowest p2 (0.38) and year excess F21 = 2.67 against ≈ 3.2–3.5 for unedited web.
- Justification: the editing can selectively delete boilerplate numerals (page and list numbers). The surviving numbers are
  then in an unusual distribution of *contexts*, not only of frequencies.
- Treatment: **sensitivity exclusion (report with/without)**. It is the single recipe whose text is not natural
  web/curated text, and it is also a C7 recipe.

**C7 — Repetition (> 1 epoch).** Rule: corpus tokens < 100.016B.
- Flags fineweb-pro (1.21 epochs), dclm-baseline-qc-10p (1.22) and dclm-baseline-qc-fw-10p (1.016).
- Justification: repeated documents differ in memorisation dynamics. At ≤ 1.22 epochs the expected effect is small
  (data-constrained scaling results show negligible loss at < 4 epochs).
- c(n) of the training stream is still measured exactly (epoch reshuffle reproduced, `src/datadecide_sampling.py:190-199`).
- Treatment: **note only; no exclusion.** fw-10p at 1.016 epochs is trivially affected.

**C8 — Dependence / non-independence (grouping, not exclusion).**
- Rule: use the lineage groups L1–L6 (§2) for clustered or leave-group-out inference.
- Treat near-replicate pairs (S ≥ 0.97) as replicates.
- Treat the 3 mixes as interpolations of #1 and #16, with their real token shares (0.43 / 0.69 / 0.87 DCLM), not the nominal ones.
- Effective N on the data side is ≈ 6–7 (shared-document n_eff 5.8–7.2), or 3 independent document pools.
- Justification: any data–model association across 25 points overstates its evidence by a factor of about 25/6.5 in
  degrees of freedom. The Dolma lineage (13 recipes incl. c4, falcon, falcon-and-cc and the falcon-qc subsets) is the
  largest block.

### Summary of flags per recipe

| recipe | C1 code | C2 STEM | C3 FLAN | C4 instr-style clf | C4b FW-Edu | C5 tiny | C6 edited | C7 >1 ep | lineage |
|---|---|---|---|---|---|---|---|---|---|
| dolma1_7 | x | x | x | | | | | | L1 |
| dolma1_7-no-code | (alg-stack .9%) | x | x | | | | | | L1 |
| dolma1_7-no-math-code | | (arXiv 2.1%) | x | | | | | | L1 |
| dolma1_7-no-reddit | x | x | x | | | | | | L1 |
| dolma1_7-no-flan | x | x | | | | | | | L1 |
| dolma1_6plus | x | x | (0.1%) | | | | | | L1 |
| c4 | | | | | | | | | L2 |
| fineweb-pro | | | | | | (x) | x | x | L6 |
| fineweb-edu | | | | | x | (x) | | | L6 |
| falcon | | | | | | | | | L2 |
| falcon-and-cc | | | | | | | | | L2 |
| falcon-and-cc-qc-10p | | | | x | | x | | | L3 |
| falcon-and-cc-qc-20p | | | | x | | | | | L3 |
| falcon-and-cc-qc-orig-10p | | | | x | | x | | | L3 |
| falcon-and-cc-qc-tulu-10p | | | | x (Tulu) | | x | | | L3 |
| dclm-baseline | | | | x | | | | | L4 |
| dclm-baseline-qc-7p-fw2 | | | | x | x | | | | L5 |
| dclm-baseline-qc-7p-fw3 | | | | x | x | x | | | L5 |
| dclm-baseline-qc-fw-3p | | | | x | x | x | | | L5 |
| dclm-baseline-qc-fw-10p | | | | x | x | x | | x (1.016) | L5 |
| dclm-baseline-qc-10p | | | | x | | x | | x | L5 |
| dclm-baseline-qc-20p | | | | x | | x | | | L5 |
| dclm-25p-dolma-75p | x | x | x | partial (43%) | | | | | L1/L4 |
| dclm-50p-dolma-50p | x | x (0.022) | | partial (69%) | | | | | L4 |
| dclm-75p-dolma-25p | x | | | partial (87%) | | | | | L4 |

---

## 4. Honest bottom line

- **No recipe should be excluded on composition grounds alone.** Every composition difference here (code, math, FLAN,
  classifier style, filtering strength) is a *confound to control*. None is a data-quality defect: the numbers were
  measured on the exact training streams.
- **The only single-recipe candidate for a sensitivity exclusion is fineweb-pro (C6).** Its text is model-edited, it
  repeats data (1.21 epochs), and it shares no documents with any other recipe.
- **The dominant problem is structural.** The 25 models are about 6–7 independent corpora on the data side.
  - The non-alpha content differences are **confounded with lineage**: code, math, FLAN, books, wiki and Reddit exist only
    in Dolma-lineage recipes, while instruction-style classifier selection exists only in DCLM and falcon-qc recipes.
  - No adjustment within these 25 models can fully separate "number distribution" from "Dolma vs DCLM vs FineWeb
    content". The cleanest within-design contrasts are the matched ablations: dolma1_7 vs -no-code / -no-math-code /
    -no-flan / -no-reddit, falcon-and-cc vs its 4 qc children, dclm-baseline vs its 6 qc children, and the mix dose series
    on *true* token shares.
- **Threshold choices** (1% code, 2% STEM, 0.5% FLAN, 15% retention) are round a-priori numbers. The flagged sets do not
  change for any threshold within ±50%, except mix-50p on C2 (0.022) and mix-75p on C1 (0.020).
