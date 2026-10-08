# Assumptions and caveats

Everything a careful reader (or reviewer) should know before trusting a number from this
repo. Each item says whether it was **checked**, and how.

## Data side (counting numbers)

| # | Assumption | Status |
|---|---|---|
| A1 | The counting rule (`\b\d+\b`, leading zeros stripped, 0..10000) is identical in every experiment | **Checked**: all counters import `_count_text` from `modal_app/corpus_alpha_app.py` |
| A2 | Decoding the training token ids back to text with the recipe model's tokenizer reproduces the text the counts should be taken on | Assumed. The training data exists only as token ids, so there is no alternative. |
| A3 | Counting per 2048-token chunk is fine. A number cut by a chunk boundary (`20`\|`19`) counts as two pieces | Negligible: ~2 cuts per 2048 tokens vs 1–2 billion matches. It is also what the model sees in one context window. |
| A4 | The reconstructed training order (file order, per-file 2048 chunking, PCG64 shuffle, first 69,369 × 704 chunks) equals DataDecide's real order | Rebuilt line by line from OLMo's code (`docs/datadecide_sampling.md`); **not** checked against a log of real batches (none is public). It doesn't matter for α: any seed gives the same α (A5). |
| A5 | The data-order seed of the released models is 2 (run names `…-1B-5xC-2`), not the nominal default 6198 | Uncertain, but **irrelevant**: across 5 seeds α differs < 0.003, numbers seen < 0.04 %, R by ±17 (E04, E07) |
| A6 | Chunks failing OLMo's repetition filter are counted although their loss is masked | They are still model input; a tiny fraction |
| A7 | A random 100B sample has the same α as the whole recipe | **Checked** on c4 (vs the full 138B recipe) and dolma1_7 (5 nearly independent samples) (E04) |
| A8 | α_OLS from small samples is biased | **Checked** (E02): unreliable below ≈1B tokens. All α_OLS values used in E05/E07 come from exact 100B streams. |
| A9 | Counts at earlier training checkpoints ≈ final counts × fraction of training | Expected because the stream is a uniform shuffle (Poisson model: ±3; real clumping: ±tens). Only needed for E08, and to be verified by exact prefix counts. |

## Model side (β)

| # | Assumption / caveat | Status |
|---|---|---|
| B1 | PC1 of the last-token hidden state is "the number line" | Standard in this literature; ρ = 0.90–0.96 in every model supports it |
| B2 | Only prompts the model copies correctly are used (correct-output filter) | By design. Rejection rates are recorded per group; no group failed in any model. Some web models reject 10–41 % of 4-digit targets in E01. |
| B3 | The selected layer is the right one to compare | **Fragile**: 8/25 models switch layer between protocols on near-ties (E06). Report β at a fixed / the old layer, and say so. |
| B4 | β from 3 gaps (paper protocol) is a fair summary | Noisy but reproducible. Its R² is uninformative near β = 1 (see `docs/project_overview.md`). E06 adds 9-gap and continuous fits. |
| B5 | Prompt-sampling noise is the main noise in β | Measured: β s.d. ≈ 0.015 over prompt seeds. **Training-seed noise is not measured yet** (open question). |
| B6 | DataDecide checkpoints load correctly | ai2-olmo 0.6.0 needs transformers 4.49 (`.generate()` breaks on ≥ 4.50). Pinned in `modal_app/datadecide_app.py` and `runpod_jobs/setup.sh`. |

## Analysis

| # | Caveat |
|---|---|
| C1 | **n = 25 models, not independent**: the mixes are built from DCLM and Dolma, and families share data. Within-family tests and leave-one-family-out checks are reported for this reason. |
| C2 | **Correlation, not causation**: recipes differ in topic, quality and how numbers are used, not only in counts. In particular, code and math shares (non-zero only in Dolma-based recipes) correlate with β at r = 0.85 / 0.76, and controlling for code share removes both the α and the R signal (`docs/findings.md`, item 6). E08 (checkpoints) is designed to address this. |
| C3 | **Multiple testing**: E07 tried 26 features, so about one should look significant by chance. R passes Bonferroni (p < 0.05/26). It is still treated as exploratory until E08. |
| C4 | **Tuned threshold**: K ≈ 4000 was first fitted on β (saturation metric, E05). The R result holds for any K in 1000–8000 (E07), so it does not depend on that tuning. |
| C5 | **Which β**: E05/E07 correlations use the paper-protocol β (`old_direct`) or the new β at the old layer. With newly selected layers the α–β link weakens (0.20–0.37). Always state which β a number refers to. |
| C6 | **Families are small** (1–7 models). Per-family r values on 2–3 models give a direction, not a significance. |
| C7 | **Cherry-picking**: removing "bad" models or hand-picking one per family can push r anywhere from −0.8 to +0.9. Never report a picked subset as the result. |
