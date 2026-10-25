# Auditor 16: external knowledge (literature and DataDecide sources) on the 25 recipes

Blind rule respected. I did not open `alpha_beta_correlations.json` or `confounders.json`, and I computed no α–β relation. Nothing below uses β values. I ignored the code-share/β correlation quoted in `docs/assumptions_and_caveats.md` C2.

## 0. Sources and access

| Source | Status |
|---|---|
| `github.com/allenai/DataDecide` | Cloned at `/tmp/claude-0/agentrefs/DataDecide`, commit `68abc496` (2025-06-17) |
| `github.com/allenai/OLMo`, branch `DataDecide` | Cloned at `/tmp/claude-0/agentrefs/OLMo`, commit `7094aab0` |
| DataDecide LFS notebooks with outputs | Already present in `/tmp/claude-0/agentrefs/dd_lfs_07/` |
| `github.com/allenai/dolma` | Cloned by me at `/tmp/claude-0/agentrefs/dolma`, commit `669f534` (2025-09-18) |
| arXiv, HF, proceedings.mlr.press, allenai.org, openreview | **Blocked**: proxy 403 / DNS failure for both `curl` and WebFetch. I could not read the paper PDF. |
| HF datasets `DataDecide-eval-results` and `-data-recipes` | **Blocked**. Per-recipe 1B OLMES accuracies were not reachable. |
| WebSearch | Works, but returns only search snippets. Every literature claim marked **[snippet]** was checked against a snippet only, not the full text. Claims marked **[memory]** are from my own knowledge and were not verified here. |

Scripts and outputs:
- `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/16_work/ppl.py`, run on `DataDecide/perplexity_metrics_by_group.csv`. This is DataDecide's own released perplexity log, 29,665 rows.
- `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/16_nb.txt`: text dump of the notebook outputs.

## 1. What the recipes are (official definitions)

Verbatim from `DataDecide/README.md:57-68` (the same text is in `release/model_card.md:73-84`):

- **Dolma1.7** *Original, No code, No math/code, No Reddit, No Flan*: "A 2.3T-token corpus … We ablate code, math/code, Reddit, or Flan subsets."
- **Dolma1.6++**: "Dolma 1.6 plus additional sources from Dolma 1.7: RedPajama's arxiv subset, openwebmath, algebraic stack, flan, starcoder, falcon."
- **C4**: "The C4 dataset … as prepared in Dolma 1.7, heuristically filtered from the April 2019 Common Crawl."
- **FineWeb-Pro**: "FineWeb Pro corpus (Zhou et al., 2024), featuring model-driven data cleaning on FineWeb." This is ProX: an LM rewrites or cleans each document as a program.
- **FineWeb-Edu**: "The deduplicated FineWeb-Edu subset of SmoLLM-Corpus … focused on educational web pages."
- **Falcon**: "The Falcon RefinedWeb corpus … in Dolma 1.7 … more aggressively filtered/deduplicated than C4."
- **Falcon+CC** *Original, QC 10%, QC 20%, QC Orig 10%, QC Tulu 10%*: "Falcon and Dolma 1.7's Common Crawl. We quality filter to top 10% or 20% documents with reproduced or original Li et al. (2024) [DCLM] filter or retrain filter on pre-release version of Tulu-v3."
- **DCLM-Baseline** *Original, QC 7% FW2, QC 7% FW3, QC FW 10% [and 3%], QC 10%, QC 20%*: "We quality filter to top 7% of DCLM classified documents and further take 2+ or 3+ scores with FineWeb-edu classifier; or filter to top 3% or 10% with FineWeb-edu classifier; or take top 10% or 20% with reproduced DCLM classifier."
- **λ% DCLM-Baseline + (1–λ)% Dolma1.7**, λ ∈ {25, 50, 75}.

**What the variant suffixes mean**, decoded from the official text plus the OLMo source paths (`OLMo/olmo/data/named_data_mixes.py`):

| Suffix | Meaning | Evidence |
|---|---|---|
| `qc-10p` / `qc-20p` (falcon-and-cc) | Top 10% / 20% by Ai2's **reproduction** of the DCLM fastText classifier. Positives are ELI5 + OpenHermes ("eli5_oh"). | mix `falcon_and_cc_eli5_oh_top10p` = `cc_eli5_oh_top10p` + `falcon_eli5_oh_top10p` (`:5141`) |
| `qc-orig-10p` | Top 10% by the **original** DCLM-released classifier ("og_eli5_oh"). | `:5143`; paths `olmo-mix/og_eli5_oh_top10p`, `falcon-refinedweb/og_eli5_oh_top10p` (`:1540-1541`) |
| `qc-tulu-10p` | Top 10% by a classifier **retrained with pre-release Tulu-v3 (instruction / SFT) data as positives**. | `:5144`; README text above; WebSearch snippet of the HF data card: "classifier trained on pre-release Tulu-v3" |
| `qc-7p-fw2` / `-fw3` (dclm) | Top 7% by the DCLM classifier, then FineWeb-Edu classifier score ≥2 / ≥3. | README; path `dclm/v0_rep32_ft7percentile_fw2|fw3` (`:1786`, `:2288`) |
| `qc-fw-3p` / `-10p` (dclm) | Top 3% / 10% by the FineWeb-Edu classifier. | path `dclm/v0_fw_top3p|top10p` (`:2552`, `:1785`) |
| `qc-10p` / `qc-20p` (dclm) | Top 10% / 20% by a reproduced DCLM classifier: positives ELI5+OpenHermes, negatives DCLM-RefinedWeb, 2000 steps, lr 3e-4. | mix `pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p|20p` (`:1781-1782`) |

**Lineage caveat (unverified, data property).** The README calls the DCLM QC recipes filters "of DCLM-Baseline". But the released DCLM-Baseline files come from `preprocessed/fastdclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/…` (`:4049`), while the QC recipes come from separate `preprocessed/dclm/v0_*` and `…/1t/` trees. So they may have been re-filtered from a larger DCLM pool rather than strictly subsetted from the 941 DCLM-Baseline files. I could not resolve this without the paper. It does not affect measurement validity.

## 2. Documented issues, anomalies and retraining in the DataDecide sources

1. **No recipe is documented as retrained, broken or withdrawn.**
   - Nothing like this appears in the DataDecide repo or the release notebooks.
   - I could not read GitHub issues: the GitHub MCP is restricted to the project repo, and `gh` returns 403. A WebSearch for DataDecide issues returned nothing relevant.
2. **Seed naming differs for 4 recipes.**
   - The 1B default-seed runs of `dclm_fw_top3`, `dclm_fw_top10`, `pos_eli5…top10p` and `pos_eli5…top20p` are named `…-1B-5xC`. The other 21 are named `…-1B-5xC-2` (`DataDecide/checkpoints/final_defualt_seed_paths.jsonl`, 1B lines).
   - DataDecide's own release code merges the two: `df['group'].apply(lambda x: x[:-4] + '-5xC-2' if x.endswith('-5xC') else x)` (`dd_lfs_07/release_loss_curves.ipynb`, cell 2). So the authors treat both as the same "default" slot.
   - By OLMo's naming rule the 4 unsuffixed runs used seed 6198 and the 21 suffixed runs used seed 2. This matches auditor 01 (`01_training.md` §3).
   - This is a nuisance (training-seed noise), not a validity problem. Every model has one draw of seed noise either way.
3. **Many W&B segments per run** (resumptions): c4 49, dolma1_7-no-flan 58, falcon-and-cc-qc-orig-10p 40, versus 1 for several others (`release/matched_runs_by_s3.json`; table §4).
   - This is operational, not a documented problem.
   - The released perplexity curves show no divergence (item 5).
4. **Only one 1B seed exists for dolma1_7**: `baseline-1B-5xC-2` is the only 1B dolma17 entry in `checkpoints/weka_paths.jsonl` and in the perplexity log. Every other recipe has 3 seeds at the final step (`ppl.py` output, "seeds at final per recipe").
   - Irrelevant here, because only the default seed is used.
   - It does mean seed noise cannot be estimated for dolma1_7 from DataDecide.
5. **No loss spikes or divergence in any of the 25 default 1B runs.**
   - Metric: max over checkpoints after step 5000 of ppl(t) / min ppl before t (`ppl.py`).
   - On the web-like validation sets (c4_en, dolma_common-crawl) every recipe stays ≤ 1.002. On wikitext-103 every recipe stays ≤ 1.008.
   - All 25 logs reach step 69369.
   - The only large rebounds are on out-of-domain sets, mainly the Pile, which contains code:
     - c4: 1.119 at step 35000, driven by dolma_stack ppl wandering between 35.6 and 64.6.
     - falcon 1.035, fineweb-pro 1.036, fineweb-edu 1.023, falcon-and-cc-qc-20p 1.023.
   - These recipes see little or no code, so this reflects missing domain coverage (a data property), not a training failure.
6. **Final checkpoint is the same for all 25.**
   - Step 69369 = 100.0B tokens (`utils/constants.py` `FULL_SCHEDULE_LAST_STEP_PER_MODEL['1B']=69369`; the max step in the perplexity log is 69369 for all 25).
   - The release notebook notes that some runs "ran too long". It rounds the step filter up so that it captures the post-decay checkpoint (`release_loss_curves.ipynb`). This does not affect `step69369`.
7. **Repetition / unusual token counts.** The fixed 100B budget means the smallest pools are repeated. Pool sizes are from auditor 01 (`01_training.md` table) and auditor 20 (`20_repetition.md`):

   | Recipe | Epochs | Repeated tokens |
   |---|---|---|
   | fineweb-pro | 1.209 | 17.3% |
   | dclm-baseline-qc-10p | 1.219 | 17.9% |
   | dclm-baseline-qc-fw-10p | 1.016 | 1.6% |
   | All others | < 1 | 0% |

   On top of this, Dolma 1.7 upsamples Wikipedia 2×.
   - Literature: Muennighoff et al. (NeurIPS 2023 / JMLR 2025), "Scaling Data-Constrained Language Models", find that "up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data" **[snippet]**. 1.22 epochs is far below that.
   - Hernandez et al. (2022, Anthropic), "Scaling Laws and Interpretability of Learning from Repeated Data", report harm to copying and induction heads **[memory]**. That matters for an in-context copying probe. But the harm came from repeating a small subset ~100× or more. Here repetition is near-uniform, ≤2 exposures. So this does not apply.
8. **Dolma PII masking changes digit strings in Dolma-processed recipes (data property).**
   - Dolma replaces detected emails, phone numbers and IP addresses with special tokens. The tokenizer reserves IDs 50277 `|||EMAIL_ADDRESS|||`, 50278 `|||PHONE_NUMBER|||` and 50279 `|||IP_ADDRESS|||`. In the OLMo tokenizer, 50279 was swapped to serve as EOS (`dolma/scripts/fix_dolma_v15_tokenizer.py:10-15, 40-50`). The PII tagger is `dolma/python/dolma/taggers/pii.py:29-30`.
   - The Falcon and Reddit source paths say so explicitly: `falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii`, `reddit/v5-dedupe-pii-…`.
   - So phone-number and IP digits are largely absent from Dolma-derived text: dolma*, c4, falcon, falcon-and-cc* (likely, via its sources), and the 3 mixes. They are present in the DCLM and FineWeb recipes.
   - The project counts numbers on the decoded training tokens (A2), so its counts already reflect what the model actually saw. This is a genuine source difference, not a measurement error.
9. **Benchmark quality at 1B.**
   - Per-recipe OLMES numbers were not reachable (HF blocked). Auditor 07 reached the same conclusion: `07_competence.md:138`.
   - DataDecide's own notebook gives the **mean over recipes at 1B** (`dd_lfs_07/scaling_laws_notebooks_math_and_code.ipynb`, "Average scores across all models at the 1B scale"): OLMES-core9 0.589, MMLU 0.351, ARC-C 0.417, HellaSwag 0.607, GSM8K 0.025, Minerva-math 0.014, MBPP 0.017, HumanEval 0.025.
   - So math and code task accuracy is at floor for every 1B model. Benchmark math scores cannot serve as a numeracy-competence covariate or exclusion signal for these models.
   - As a substitute, the released final perplexities are in §4. By Pile perplexity the clear quality and coverage outliers are:
     - c4: Pile 22.47, stack 36.90, books 53.20.
     - fineweb-pro: Pile 17.23.
     - fineweb-edu: Pile 15.17.
     - dclm-baseline-qc-fw-3p: Pile 14.55, and the worst c4_en at 26.10.
   - All four are explained by domain coverage: no code or books in c4 or FineWeb, very aggressive filtering in fw-3p. These are data properties.

## 3. Literature: do data properties change numeric representations?

| Data property | Literature | What it implies here |
|---|---|---|
| Number frequency in pretraining | Razeghi et al. 2022 (EMNLP Findings), "Impact of Pretraining Term Frequencies on Few-Shot Reasoning": accuracy on arithmetic instances tracks how often the operands appear in pretraining **[snippet + memory]** | This is the project's own hypothesis class. It supports treating counts as the causal variable of interest, not as a reason to exclude. |
| Log compression of the number line | AlQuabeh et al. 2025 (arXiv 2502.16147), "Number Representations in LLMs: A Computational Parallel to Human Perception": PCA/PLS on hidden states, sublinear compression; this is the β protocol audited here **[snippet]**. Zhu et al. 2025: linear probes favour linear rather than log targets **[snippet]**. arXiv 2603.20642 (Weber's law in transformers): globally log, locally linear, and uses frequency-matched controls **[snippet]** | β depends on the readout (PCA versus probe). This is a measurement-design point, the same for all 25 models. |
| Digit-wise structure | Levy & Geva 2024/25 (arXiv 2410.11781): numbers are encoded per digit in base 10 (circular) **[snippet]**. Zhou et al. NeurIPS 2024 (arXiv 2406.03445): pretrained LLMs use Fourier features, and models trained from scratch on addition only use low-frequency features, so **pretraining data shapes number features** **[snippet]** | Multi-digit structure can make PC1 track digit count rather than magnitude. The project already guards against this with ρ in layer selection. The digit-count/magnitude confound is the same kind of issue in every recipe. |
| Data, architecture, optimizer and tokenizer all matter | "Convergent Evolution: How Different LMs Learn Similar Number Representations" (arXiv 2604.20817, 2026): data, architecture, optimizer and tokenizer all determine whether geometrically separable number features emerge **[snippet]** | Only data varies across DataDecide, so data-driven differences are expected and are the signal, not noise. |
| Code data | Petty, van Steenkiste & Linzen (TMLR 2025, arXiv 2409.04556): more code in pretraining **improves arithmetic**, more so as digit count grows, and hurts syntax and world knowledge **[snippet]**. Aryabumi et al. (ICLR 2025), "To Code or Not to Code": code raises natural-language reasoning by up to 8.2% (relative) at 470M–2.8B **[snippet]** | Code share is a documented, plausible causal driver of numeric processing. It differs strongly across recipes: only Dolma-based recipes contain StarCoder code; no-code / no-math-code ablate it. It is a **confounder to control or report**, not an exclusion reason. |
| Math data | Dolma's `algebraic_stack`, `openwebmath` and `pes20_stem_papers` are present only in Dolma-based recipes and are removed in -no-math-code (`named_data_mixes.py:5127-5134`). There is no direct study of math share versus number-line geometry that I could verify. | Data property; confounded with code share by design. |
| Instruction data | FLAN (`tulu_flan`) is in dolma1_7, absent in -no-flan, and in Dolma1.6++. The Tulu-v3-trained classifier selects instruction-like web pages (qc-tulu-10p). No verified study links instruction data to number-line geometry. | Data property; may change prompt-format familiarity (`a=a,b=b,…`). That is a competence and format effect, which the correct-output filter partly absorbs. |
| Tokenization of numbers | Singh & Strouse 2024, "Tokenization counts" **[memory]**: number tokenization strongly affects arithmetic | Same tokenizer for all 25 (`01_training.md`), so held constant. |
| Heavy repetition | Muennighoff 2023 **[snippet]**; Hernandez 2022 **[memory]** | Max 1.22 epochs here, so not "heavy". No literature basis for exclusion. |

## 4. Per-model table (all 25)

Column notes:
- Epochs: from `01_training.md`.
- W&B segments: from `release/matched_runs_by_s3.json`.
- Perplexities: final step 69369, default seed (`ppl.py` on `perplexity_metrics_by_group.csv`).
- Max rebound: worst of c4_en, wikitext and dolma-CC after step 5000.
- PII masking: Dolma-processed text with PII masking.
- Code/math/instruction content: from the mix definitions.

| # | recipe | OLMo mix | definition (short) | PII-masked (Dolma) | code / math / instr. in mix | epochs | seed (naming) | W&B segs | ppl c4_en | ppl Pile | ppl stack | max rebound | documented problem | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | dolma1_7 | dolma17 | Dolma 1.7 full | yes | code+math+FLAN | 0.058 | 2 | 34 | 14.95 | 8.29 | 2.80 | 0.999 | only 1 seed at 1B | none |
| 2 | dolma1_7-no-code | no_code | − starcoder, stackexchange | yes | math+FLAN | 0.070 | 2 | 26 | 14.61 | 8.89 | 4.25 | 1.001 | – | D (no code) |
| 3 | dolma1_7-no-math-code | no_math_no_code | − code − math (algebraic stack, OWM, pes2o) | yes | FLAN only | 0.074 | 2 | 23 | 14.55 | 9.47 | 4.78 | 1.000 | – | D (no code/math) |
| 4 | dolma1_7-no-reddit | no_reddit | − Reddit | yes | code+math+FLAN | 0.061 | 2 | 27 | 14.87 | 8.25 | 2.79 | 0.999 | – | none |
| 5 | dolma1_7-no-flan | no_flan | − tulu_flan | yes | code+math | 0.059 | 2 | 58 | 14.90 | 8.31 | 2.80 | 0.999 | most W&B resumptions (58) | D (no instr.) |
| 6 | dolma1_6plus | dolma-v1-6-and-sources-baseline | Dolma 1.6 + 1.7 sources | yes | code+math+FLAN | 0.053 | 2 | 10 | 14.06 | 8.50 | 2.84 | 0.999 | – | none |
| 7 | c4 | c4 | Dolma-1.7 C4 (Apr-2019 CC) | yes | none | 0.722 | 2 | 49 | 13.90 | **22.47** | **36.90** | 1.008 (Pile 1.119) | OOD ppl wanders (no code) | D (narrow domain) |
| 8 | fineweb-pro | prox_fineweb_pro | ProX LM-cleaned FineWeb | no | none (web) | **1.209** | 2 | 28 | 16.71 | **17.23** | 15.33 | 0.999 | 17.3% repeated tokens | D (repeat, LM-rewritten) |
| 9 | fineweb-edu | fineweb_edu_dedup | FineWeb-Edu dedup (SmolLM) | no | none (web) | 0.520 | 2 | 31 | 17.84 | **15.17** | 12.85 | 1.005 | – | D (edu filter) |
| 10 | falcon | falcon | RefinedWeb in Dolma 1.7 | yes | none | 0.219 | 2 | 26 | 14.87 | 12.38 | 8.85 | 1.002 | – | none |
| 11 | falcon-and-cc | falcon_and_cc | Falcon + Dolma CC | yes | none | 0.095 | 2 | 19 | 14.64 | 10.65 | 6.31 | 1.000 | – | none |
| 12 | falcon-and-cc-qc-10p | falcon_and_cc_eli5_oh_top10p | top 10% reproduced DCLM QC | likely | none | 0.685 | 2 | 31 | 16.90 | 11.14 | 5.94 | 0.999 | 1 eval ckpt missing in log (27/28) | none |
| 13 | falcon-and-cc-qc-20p | falcon_and_cc_eli5_oh_top20p | top 20% reproduced DCLM QC | likely | none | 0.350 | 2 | 4 | 16.01 | 10.58 | 5.93 | 1.001 | – | none |
| 14 | falcon-and-cc-qc-orig-10p | falcon_and_cc_og_eli5_oh_top10p | top 10% original DCLM QC | likely | none | 0.676 | 2 | 40 | 16.45 | 10.91 | 5.29 | 0.999 | – | none |
| 15 | falcon-and-cc-qc-tulu-10p | falcon_and_cc_tulu_qc_top10 | top 10% by Tulu-v3-trained QC | likely | instruction-like web | 0.776 | 2 | 1 | 16.05 | 9.86 | 5.29 | 1.001 | – | D (instr.-style selection) |
| 16 | dclm-baseline | DCLM-baseline | DCLM-Baseline | no | none (web) | 0.026 | 2 | 8 | 16.48 | 10.12 | 5.20 | 1.001 | – | none |
| 17 | dclm-baseline-qc-7p-fw2 | dclm_ft7percentile_fw2 | top 7% DCLM QC + FW-Edu ≥2 | no | none | 0.197 | 2 | 2 | 18.67 | 10.40 | 5.05 | 1.006 | lineage path differs (§1) | none |
| 18 | dclm-baseline-qc-7p-fw3 | dclm_ft7percentile_fw3 | top 7% DCLM QC + FW-Edu ≥3 | no | none | 0.459 | 2 | 1 | 22.57 | 12.35 | 5.98 | 1.002 | lineage path differs | none |
| 19 | dclm-baseline-qc-fw-3p | dclm_fw_top3 | top 3% by FW-Edu classifier | no | none | 0.847 | **6198** | 1 | **26.10** | 14.55 | 7.13 | 1.008 | seed-naming differs | S, D (most aggressive filter) |
| 20 | dclm-baseline-qc-fw-10p | dclm_fw_top10 | top 10% by FW-Edu classifier | no | none | **1.016** | **6198** | 4 | 20.63 | 11.33 | 5.49 | 1.005 | seed-naming differs; 1.6% repeated | S, D |
| 21 | dclm-baseline-qc-10p | pos_eli5…top10p | top 10% reproduced DCLM QC | no | none | **1.219** | **6198** | 1 | 19.42 | 11.07 | 4.98 | 1.000 | seed-naming differs; 17.9% repeated | S, D (repeat) |
| 22 | dclm-baseline-qc-20p | pos_eli5…top20p | top 20% reproduced DCLM QC | no | none | 0.585 | **6198** | 1 | 18.06 | 10.41 | 4.91 | 0.999 | seed-naming differs | S |
| 23 | dclm-baseline-25p-dolma1.7-75p | dolma17-75p-DCLM-baseline-25p | 25% DCLM / 75% Dolma (file subsample) | partly | code+math+FLAN (diluted) | 0.045 | 2 | 1 | 15.00 | 8.38 | 2.93 | 0.999 | – | none |
| 24 | dclm-baseline-50p-dolma1.7-50p | dolma17-50p-DCLM-baseline-50p | 50/50 | partly | diluted | 0.036 | 2 | 1 | 15.30 | 8.57 | 3.05 | 0.999 | – | none |
| 25 | dclm-baseline-75p-dolma1.7-25p | dolma17-25p-DCLM-baseline-75p | 75% DCLM / 25% Dolma | partly | diluted | 0.030 | 2 | 1 | 15.59 | 8.91 | 3.29 | 0.999 | – | none |

Flag key: **D** = documented data property (report or control); **S** = seed-naming heterogeneity (nuisance). No row carries an **M** (measurement-validity) flag.

Mix note: the DataDecide uploader explicitly drops the pure-DCLM runs `DCLM-baseline-{25,50,75}p` (`release/upload_checkpoints.py`, `model_name_to_repo_name`, `return None`). So the released mixes are the `dolma17-XXp-DCLM-baseline-YYp` runs, consistent with `configs/datadecide_data_map.json`. There is no mapping error.

## 5. Exclusion criteria (a priori, blind to β-outcome relations)

| ID | Rule | Threshold | Justification | Models flagged | Type |
|---|---|---|---|---|---|
| L1 | Documented training failure, retraining, withdrawn or replaced checkpoint in official DataDecide sources | any statement | A broken model does not represent its recipe | **none** (§2.1) | – |
| L2 | Training instability in DataDecide's own validation log | in-domain web/wiki ppl rebound > 1.05 after step 5000, or log not reaching 69369 | A spike or divergence near the end could leave an atypical final state | **none** (max 1.008; all reach 69369) | – |
| L3 | Final checkpoint ≠ step69369 / 100B tokens, or wrong recipe→repo mapping | any | Equal-budget comparison is the design premise | **none** | – |
| L4 | Repetition beyond the regime where the literature shows negligible effect | > 4 epochs (Muennighoff 2023) | Heavy repetition changes learning dynamics | **none** (max 1.219) | – |
| L5 | Different data-order/initialisation seed from the others | seed ≠ majority | Adds seed noise | qc-fw-3p, qc-fw-10p, qc-10p, qc-20p | **Nuisance, not validity.** Every model is one seed draw. DataDecide itself treats these as "default". Report it; optionally run a sensitivity check without these 4. |
| L6 | Benchmark-quality outlier | per-recipe OLMES > 2 s.d. from mean | Could indicate a degenerate model | **not evaluable** (HF blocked). Math/code benchmarks are at floor (~1–2.5%) for every 1B model, so they are uninformative. | – |
| D1 | Known data properties with literature support for affecting numeric processing: code, math, instruction share; PII masking of phone/IP digits; LM-rewritten text (ProX); aggressive quality filtering; mild repetition | – | Petty 2025, Aryabumi 2025, Razeghi 2022, Zhou 2024 | code (StarCoder): 7 (dolma1_7, -no-reddit, -no-flan, dolma1_6plus, 3 mixes); math without code: dolma1_7-no-code; neither: the other 17; FLAN: dolma1_7, -no-code, -no-math-code, -no-reddit, dolma1_6plus, 3 mixes; instruction-style selection: qc-tulu-10p; PII-masked: 16 Dolma-derived (6 dolma*, c4, falcon, falcon-and-cc + 4 QC, 3 mixes, the last 4+3 partly or likely); repetition: fineweb-pro, dclm-qc-10p, (qc-fw-10p) | **Genuine data properties.** They are the variation the study exploits. Do not exclude. Report them and use them as covariates. |

## 6. Bottom line

The external sources (DataDecide repo and release notebooks, OLMo mix code, Dolma code, literature snippets) contain **no evidence that any of the 25 models is broken, retrained or mis-measured**. All 25 have smooth validation curves to step 69369 and the same budget, tokenizer and architecture. Every documented anomaly is either a nuisance or a genuine data property:

- Four DCLM-QC models carry a different seed name (6198 versus 2).
- Three small recipes repeat 2–18% of tokens (≤1.22 epochs).
- c4 and the two FineWeb recipes have poor coverage outside their domain.
- Dolma-derived text masks phone and IP digits.
- Code, math and instruction data appear only in Dolma-based recipes.

The literature shows that code, math and number frequency plausibly change numeric processing. That makes these confounders to control or report, not exclusion grounds. **Nothing should be excluded on literature or provenance grounds.**

Limits: per-recipe OLMES accuracies and the paper's full text were unreachable (HF and arXiv blocked). Literature claims marked [snippet] or [memory] are not full-text verified.
