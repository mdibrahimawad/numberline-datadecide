# Auditor 06: tokenization and how numbers are presented to each model

Scope: whether every one of the 25 DataDecide-1B models (a) sees the β probe numbers tokenized the same way and (b) had its α counts produced by the same decode-and-count pipeline, so that the α counts match the text the model was trained on. I did not compute, look up or reason about any α–β correlation, and I did not open `alpha_beta_correlations.json`. The β values below are used only to measure how big a measurement artefact is.

## Bottom line
- **No model should be excluded on tokenization or presentation grounds.** All 25 pass every hard criterion (H1–H5 below).
- **Soft flag for 7 models: c4, fineweb-pro, fineweb-edu, dclm-baseline, dclm-baseline-qc-7p-fw3, dclm-baseline-qc-fw-3p and dclm-baseline-qc-fw-10p.** The correct-output filter rejects mostly 3-digit, single-token targets near 1000 in these models. As a result, their accepted "1000" group contains more 2-token targets than the other models' (up to 0.87 vs 0.69). At the E01 (old) layer this moves β_coarse by at most 0.011, which is less than the prompt-seed s.d. (≈0.015). At the newly selected early layers 2–3 it reaches +0.029 for qc-fw-3p and +0.014 for qc-7p-fw3. Those layers are already called fragile in B3, and the PC1 there partly tracks token count. Recommendation: report β at a fixed or old layer (as C5/B3 already advise) and note the composition skew. Do not drop these models.

## Hard criteria (a model is excluded if it fails any)
| id | criterion | measurement reason |
|---|---|---|
| H1 | Same tokenizer: class, vocabulary/ids of probe numerals, EOS id, `=` id | Different token ids would mean different inputs |
| H2 | Same prompt format: `prepend_bos`, BOS count, last token `=`, protocol config identical | Hidden state is read at the last token |
| H3 | Training token files decode to real text with the recipe tokenizer: EOS = 50279 separates documents (plausible tokens/doc, 1–3 % of tokens in cut documents) and number statistics look like natural text (year spike 1990–2020 ≫ 2190–2220, round-number spikes, c(1)/c(9) of about 4–6) | A vocabulary or special-token mismatch would produce garbage text, so the counts would not describe what the model saw |
| H4 | Identical counting code path (`_tokenizer` → `_decode_and_count_native` → `_count_text`, split on EOS, seed 2) on Modal and on RunPod | Same rule for every model |
| H5 | Reweighting the filter-induced 1-token/2-token mix of the "1000" group back to the uniform mix changes β_coarse by less than 0.03 (2 × prompt-seed s.d.) at the reported layer | The target composition must be comparable between models |

Every model passes H1–H5. qc-fw-3p at its new layer 2 comes closest to the H5 limit (+0.029). At the old layer 9 it is +0.001.

## Evidence

### 1. The β prompts are tokenized identically for all 25 models
- The `config` (minus model name) and the `evaluation.config` hash to the same value in all 25 JSONs: `prepend_bos=False`, `tokenizer_use_fast=True`, bf16, k=40, 3 random contexts, filter on, seeds 42–44 and 45–47. All files come from one commit (083b867).
- `tokenization_diagnostics` is the same in all 25 files: `OLMoTokenizerFast`, `bos_token_id=None`, `eos=50279`, `=` → [30], 0 prompts starting with BOS, 0 duplicate BOS, 160/160 prompts ending in the `=` token, and `final_token_is_equals=True`.
- The diagnostic sample prompts are the same in 23 models (fineweb-pro and dclm-baseline-qc-10p differ in 1 of 12, because of filter resampling). Every sampled target maps to the same token ids in every model.
- I loaded the training tokenizer from OLMo@DataDecide `olmo_data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json` (md5 3da55f3d…; `ladder.py:250`). It reproduces every sampled id (2→19, 8→25, 18→1093, …). Rebuilding the seed-42 prompts with no rejections gives exactly the 106/54 target histogram and the 198/922 numeral histogram that the zero-rejection models report.
- **Why the target histograms differ between models:** the tokenizer is the same, but the correct-output filter resamples different slots in each model. Uniform token mix per group: g1 (1–39) and g2 (80–119) are all 1 token. g3 (980–1019) is 23×1 and 17×2 tokens, and token count does not equal digit count there: 981–986 and 1003, 1009–1015, 1017–1019 are 2 tokens, while 987–1002 are 1 token. g4 (9980–10019) is 38×2 and 2×1 tokens (9999, 10000). Most group-3 rejections are 3-digit targets (e.g. `997=` → `9997`). For example, 22 of the 30 logged fineweb-pro group-3 rejections are below 1000. So the E01 README's "fails to copy 4-digit numbers" is inaccurate: the failures are mostly 3-digit numbers among 4-digit context numbers.
- E06 saves the accepted targets (npz), so the skew can be measured on 300 prompts per group. In the 1000 band (900–1100) the multi-token share is 0.69 when uniform; per model it is in the table. I reweighted each model's "1000" group mean to the uniform share at its layer and refitted β_coarse. My refit reproduces the reported β_coarse exactly for all 25 models. The resulting change is the Δβ column.
- Within the 1000 band, at layers 2–3, multi-token targets sit 1–2 coarse gaps away from single-token targets: PC1 there encodes token count. This is the measurement reason to prefer the old/fixed layer.

### 2. The counting pipeline decodes and counts the same way for every recipe
- 14 recipes were counted on Modal and 11 on RunPod (column "count runner"). RunPod imports `_tokenizer` and `_decode_and_count_native` from the Modal app (`runpod_jobs/exact_alpha.py:40-46`). Both download the recipe repo's `tokenizer.json`, decode with `skip_special_tokens=True`, split each 2048-token chunk on EOS 50279 and count with `\b\d+\b`.
- **Different tokenisation pipelines (dolma1_6plus, DCLM, QC pools).** dolma1_6plus uses only `gpt-neox-olmo-dolma-v1_5-tokenizers-0-19-1` files (a re-tokenization with a newer `tokenizers` library, same vocabulary). The `fastdclm/.../allenai` files (dclm-baseline and the 3 mixes), the DCLM-QC `/1t` files (qc-10p, qc-20p) and the falcon-QC pools (`eli5_oh_*`, `og_eli5_*`, `tulu_qc_*`) do not name a tokenizer in their path. Decoding is a pure id→bytes lookup, so a different encoder version can change how the model segmented the text (a property of the recipe, i.e. of what the model saw) but not the decoded text or the counts. The only harmful case would be a different vocabulary, e.g. the PII-special NeoX file where EOS = 0 and 50279 = `|||IP_ADDRESS|||`; I confirmed the two files differ only in that swap. H3 rules that case out empirically for every pool. Tokens per document are 533–1330 and only 0.8–3.4 % of tokens fall in cut documents, so EOS = 50279 does split documents. Every recipe also shows strong natural-text signatures (year spike 313–742×, c(1000) 7–19× its neighbours, c(1)/c(9) 3.9–6.4).
- I could not hash each model repo's own `tokenizer.json`: huggingface.co is blocked by the proxy. The diagnostics above are the substitute evidence.
- **The falcon-QC 10p, 20p and orig-10p recipes have low number density** (10.0, 11.0 and 11.7 per 1k tokens, against 16–19 elsewhere) and the weakest year spikes. The tulu-QC pool, which shares the same unlabelled pipeline, is normal (17.5). The ELI5/OpenHermes classifiers selecting explanatory, less date-heavy text explain this pattern; a decode fault would not (with a decode fault the round-number spikes would vanish, but they are the strongest in these recipes). So this is content, not a pipeline artefact.

### 3. Does the counting rule match how the model sees numbers? (applies to all models; no model is singled out)
The NeoX ByteLevel pre-tokenizer splits digit runs from letters and punctuation, and BPE then splits long runs. Concretely:
- `1,000` → `1` `,` `000`. The rule counts 1 and 0. This matches the tokens but loses the value 1000. Comma-grouped values ≥ 1000 are never counted as themselves, which inflates c(1)–c(999) and c(0). c(0) is unused by α and R.
- `3.14` → `3` `.` `14`. The rule counts 3 and 14, which is consistent with the tokens.
- **Numbers attached to letters or underscores are dropped by the rule** (`abc123`, `x_1`, `var2`, `1st`, `0x1F`): `\b` finds no boundary there. The model, however, sees standalone digit tokens (`abc` `123`, `1` `st`). The undercount is therefore larger in code- and math-heavy recipes (Dolma with starcoder, proof-pile, and the mixes) than in web-only recipes. This is a systematic, recipe-dependent difference in what "count" means, but it comes from the same rule for every recipe. It is a definitional caveat, not a reason to exclude a model; it could be checked by recounting with a token-aligned rule such as `(?<!\d)\d+(?!\d)`.
- Python's `\d` also matches non-ASCII decimal digits (`٣٤` → 34, `１２３` → 123). The model sees these as unrelated byte tokens. The effect is negligible for English-filtered recipes.
- Probe targets 10001–10019 (E01) and 10001–11000 (E06 top band) lie outside the counted range 0..10000. This is the same for all models.

## Per-model table (all 25)
"Δβ" = β_coarse (E06, seeds 45–47) after reweighting the 1000-band token mix to uniform, minus the reported value. "old layer" = the layer E01 selected. The E01 histogram comes from the first selection seed (42) only, 160 prompts.

| # | recipe | tok class / BOS / "=" id / final "=" | E01 seed-42 target tokens 1/2 (excess 2-tok vs no-filter 106/54) | E01 eval rej. grp3 (980-1019) | E06 multi-token share, 1000 band (uniform 0.69) | Δβ_coarse reweighted, old layer | Δβ_coarse reweighted, selected layer | training-token pipeline(s) | count runner | tok/doc | numbers/1k tok | year spike | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | dolma1_7 | OLMoTokenizerFast / False / 30 / True (160/160) | 106/54 (+0) | 0.000 | 0.69 | +0.000 (L6) | -0.000 (L4) | neox-v1_5 | Modal | 667 | 18.48 | 467 | keep |
| 2 | dolma1_7-no-code | OLMoTokenizerFast / False / 30 / True (160/160) | 106/54 (+0) | 0.000 | 0.69 | +0.000 (L8) | +0.000 (L8) | neox-v1_5 | Modal | 632 | 17.43 | 633 | keep |
| 3 | dolma1_7-no-math-code | OLMoTokenizerFast / False / 30 / True (160/160) | 106/54 (+0) | 0.000 | 0.69 | -0.000 (L4) | -0.000 (L4) | neox-v1_5 | RunPod | 599 | 16.19 | 644 | keep |
| 4 | dolma1_7-no-reddit | OLMoTokenizerFast / False / 30 / True (160/160) | 106/54 (+0) | 0.000 | 0.69 | +0.001 (L8) | +0.001 (L8) | neox-v1_5 | RunPod | 757 | 18.97 | 472 | keep |
| 5 | dolma1_7-no-flan | OLMoTokenizerFast / False / 30 / True (160/160) | 106/54 (+0) | 0.000 | 0.69 | +0.001 (L9) | +0.001 (L9) | neox-v1_5 | RunPod | 680 | 18.51 | 465 | keep |
| 6 | dolma1_6plus | OLMoTokenizerFast / False / 30 / True (160/160) | 106/54 (+0) | 0.000 | 0.69 | +0.002 (L9) | +0.002 (L9) | neox-v1_5 (tokenizers-0.19.1 retok) | RunPod | 608 | 19.30 | 413 | keep |
| 7 | c4 | OLMoTokenizerFast / False / 30 / True (160/160) | 105/55 (+1) | 0.143 | 0.75 | -0.004 (L7) | +0.000 (L3) | neox-v1_5 | Modal | 551 | 13.96 | 703 | keep; soft flag (filter-skewed token mix) |
| 8 | fineweb-pro | OLMoTokenizerFast / False / 30 / True (160/160) | 100/60 (+6) | 0.413 | 0.87 | +0.003 (L7) | +0.003 (L7) | neox-v1_5 | Modal | 1189 | 14.86 | 742 | keep; soft flag (filter-skewed token mix) |
| 9 | fineweb-edu | OLMoTokenizerFast / False / 30 / True (160/160) | 106/54 (+0) | 0.098 | 0.76 | +0.006 (L9) | +0.006 (L9) | neox-v1_5 | Modal | 1006 | 17.20 | 660 | keep; soft flag (filter-skewed token mix) |
| 10 | falcon | OLMoTokenizerFast / False / 30 / True (160/160) | 105/55 (+1) | 0.028 | 0.69 | -0.000 (L9) | -0.000 (L8) | neox-v1_5 | RunPod | 684 | 18.89 | 661 | keep |
| 11 | falcon-and-cc | OLMoTokenizerFast / False / 30 / True (160/160) | 105/55 (+1) | 0.024 | 0.70 | +0.000 (L9) | +0.000 (L9) | neox-v1_5 | RunPod | 682 | 16.28 | 672 | keep |
| 12 | falcon-and-cc-qc-10p | OLMoTokenizerFast / False / 30 / True (160/160) | 104/56 (+2) | 0.070 | 0.72 | +0.000 (L8) | +0.005 (L3) | QC pool (unlabelled); neox-v1_5 | Modal | 672 | 9.99 | 313 | keep |
| 13 | falcon-and-cc-qc-20p | OLMoTokenizerFast / False / 30 / True (160/160) | 105/55 (+1) | 0.036 | 0.72 | +0.001 (L8) | +0.001 (L8) | QC pool (unlabelled) | Modal | 773 | 11.01 | 373 | keep |
| 14 | falcon-and-cc-qc-orig-10p | OLMoTokenizerFast / False / 30 / True (160/160) | 106/54 (+0) | 0.012 | 0.70 | -0.000 (L8) | +0.000 (L3) | QC pool (unlabelled) | Modal | 612 | 11.68 | 356 | keep |
| 15 | falcon-and-cc-qc-tulu-10p | OLMoTokenizerFast / False / 30 / True (160/160) | 105/55 (+1) | 0.020 | 0.69 | -0.000 (L9) | +0.001 (L2) | QC pool (unlabelled) | Modal | 533 | 17.49 | 589 | keep |
| 16 | dclm-baseline | OLMoTokenizerFast / False / 30 / True (160/160) | 103/57 (+3) | 0.161 | 0.77 | +0.002 (L8) | +0.002 (L8) | fastdclm/allenai (unlabelled) | RunPod | 1298 | 14.85 | 633 | keep; soft flag (filter-skewed token mix) |
| 17 | dclm-baseline-qc-7p-fw2 | OLMoTokenizerFast / False / 30 / True (160/160) | 107/53 (-1) | 0.028 | 0.70 | +0.000 (L8) | +0.000 (L8) | neox-v1_5 | RunPod | 1275 | 16.58 | 535 | keep |
| 18 | dclm-baseline-qc-7p-fw3 | OLMoTokenizerFast / False / 30 / True (160/160) | 101/59 (+5) | 0.189 | 0.79 | +0.001 (L9) | +0.014 (L3) | neox-v1_5 | Modal | 1266 | 18.69 | 498 | keep; soft flag (filter-skewed token mix) |
| 19 | dclm-baseline-qc-fw-3p | OLMoTokenizerFast / False / 30 / True (160/160) | 106/54 (+0) | 0.189 | 0.76 | +0.001 (L9) | +0.029 (L2) | neox-v1_5 | Modal | 1329 | 19.75 | 512 | keep; soft flag (filter-skewed token mix) |
| 20 | dclm-baseline-qc-fw-10p | OLMoTokenizerFast / False / 30 / True (160/160) | 101/59 (+5) | 0.178 | 0.78 | +0.011 (L3) | +0.011 (L3) | neox-v1_5 | Modal | 1330 | 18.68 | 544 | keep; soft flag (filter-skewed token mix) |
| 21 | dclm-baseline-qc-10p | OLMoTokenizerFast / False / 30 / True (160/160) | 103/57 (+3) | 0.137 | 0.73 | +0.000 (L8) | +0.000 (L8) | dclm-qc /1t (unlabelled) | Modal | 990 | 17.66 | 594 | keep |
| 22 | dclm-baseline-qc-20p | OLMoTokenizerFast / False / 30 / True (160/160) | 103/57 (+3) | 0.048 | 0.71 | +0.001 (L9) | +0.001 (L9) | dclm-qc /1t (unlabelled) | Modal | 1047 | 16.36 | 603 | keep |
| 23 | dclm-baseline-25p-dolma1.7-75p | OLMoTokenizerFast / False / 30 / True (160/160) | 106/54 (+0) | 0.000 | 0.69 | +0.001 (L8) | +0.001 (L8) | fastdclm/allenai (unlabelled); neox-v1_5 | RunPod | 852 | 17.00 | 523 | keep |
| 24 | dclm-baseline-50p-dolma1.7-50p | OLMoTokenizerFast / False / 30 / True (160/160) | 106/54 (+0) | 0.000 | 0.69 | +0.000 (L8) | +0.000 (L8) | fastdclm/allenai (unlabelled); neox-v1_5 | RunPod | 996 | 16.01 | 551 | keep |
| 25 | dclm-baseline-75p-dolma1.7-25p | OLMoTokenizerFast / False / 30 / True (160/160) | 106/54 (+0) | 0.000 | 0.69 | +0.000 (L10) | +0.000 (L10) | fastdclm/allenai (unlabelled); neox-v1_5 | RunPod | 1159 | 15.34 | 613 | keep |

## Flagged models
- **Exclude: none.**
- **Soft flag (keep; report β at the fixed/old layer and mention the skew):** fineweb-pro (1000-band multi-token share 0.87, group-3 rejection 41 %), dclm-baseline-qc-7p-fw3 (0.79; Δβ +0.014 at new layer 3), dclm-baseline-qc-fw-10p (0.78; +0.011 at layer 3, which is also its old layer), dclm-baseline (0.77), dclm-baseline-qc-fw-3p (0.76; +0.029 at new layer 2), fineweb-edu (0.76), c4 (0.75).
- **Informational:** falcon-and-cc-qc-10p, -qc-20p and -qc-orig-10p have low number density, which reflects content, not the pipeline. dolma1_6plus is the only recipe using the 0.19.1 re-tokenized files, and its decode checks out. The unlabelled DCLM and QC pools all pass H3.

## Honest note
Tokenization and presentation are uniform across the 25 models. The one real model-dependent effect is caused by the correct-output filter, which makes the accepted target sets, and so their 1-token/2-token mix, differ between models. It is too small to matter at fixed layers. The counting rule's mismatch with model-visible digit tokens (numbers glued to letters, comma-grouped values) is a definitional caveat that affects all recipes, in different amounts. It does not justify excluding any single model. Limitation: I could not hash each repo's `tokenizer.json` or inspect raw training tokens (no HF access); H3 is indirect, statistical evidence.

Scripts: scratchpad `tok1.py`–`tok7.py`, `reweight.json`, `e06_comp.json`.
