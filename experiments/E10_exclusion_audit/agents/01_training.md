# Auditor 01: training procedure and architecture of the 25 DataDecide 1B models

Scope: are the 25 `allenai/DataDecide-<recipe>-1B` @ `step69369-seed-default` models comparable in architecture, tokenizer, token budget, steps, seed, LR schedule and final checkpoint? Are there per-recipe training anomalies (repetition/epochs, restarts, different seeds)?
Blind rule respected: I did not open `alpha_beta_correlations.json` or `confounders.json`, and I did not compute or use any α–β relation. Every criterion below rests on training and measurement grounds only. `results/datadecide/<recipe>.json` was read only for structural fields: layer count, tokenizer class, EOS id and revision.

Sources:
- Project repo `/home/user/numberline-datadecide`.
- `git clone --depth 1 https://github.com/allenai/DataDecide` into `/tmp/claude-0/agentrefs/DataDecide`, commit `68abc49` (2025-06-17).
- `git clone --depth 1 -b DataDecide https://github.com/allenai/OLMo` into `/tmp/claude-0/agentrefs/OLMo`, commit `7094aab` (2025-03-13).
- Both commits match the ones listed in `docs/datadecide_sampling.md:3-8`.
- HF and arXiv are blocked (proxy 403), so the HF model configs and tokenizer files could not be checked directly.

## 1. Per-model table (all 25)

Column definitions:
- **pool tokens** = `membership.n_chunks` × 2048, from `results/corpus_alpha_datadecide/exact_100b_<recipe>/summary.json`. These are the 2048-token chunks the loader can form after dropping each file's tail.
- **epochs** = 48,835,776 (69,369 steps × 704 sequences) / `n_chunks`.
- **repeated share** = the fraction of the 100B training tokens that are a second exposure of a chunk.
- **seed by naming rule**: `ladder.py` appends `-<seed>` to the run name only when the seed is not 6198 (details in §3).
- **W&B run records**: how many W&B runs are logged under that checkpoint path in `DataDecide/release/matched_runs_by_s3.json`. This is a proxy for resumptions or preemptions.
- **layers**: transformer layers seen when the project loaded the model. It is `len(selection.layers_direct) - 1` in `results/datadecide/<recipe>.json`, where index 0 is the embeddings.
- **run dir**: the dolma1_7 run directory is named `baseline-1B-5xC-2`, because DataDecide maps `baseline` to `dolma17` (`release/upload_checkpoints.py:60-61`).

| # | recipe (slug) | OLMo mix | default-seed run dir (`benb/...`) | seed by naming rule | W&B run records | files (repeated) | pool tokens (2048-chunkable) | epochs = 48,835,776 / N_chunks | repeated share of the 100B | layers (E01 load) | tokenizer class / EOS | final ckpt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | dolma1_7 | `dolma17` | `<mix>-1B-5xC-2` | 2 | 34 | 1033 (2) | 1,715.2B | 0.058 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 2 | dolma1_7-no-code | `no_code` | `<mix>-1B-5xC-2` | 2 | 26 | 914 (2) | 1,431.7B | 0.070 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 3 | dolma1_7-no-math-code | `no_math_no_code` | `<mix>-1B-5xC-2` | 2 | 23 | 859 (2) | 1,349.2B | 0.074 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 4 | dolma1_7-no-reddit | `no_reddit` | `<mix>-1B-5xC-2` | 2 | 27 | 955 (2) | 1,635.2B | 0.061 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 5 | dolma1_7-no-flan | `no_flan` | `<mix>-1B-5xC-2` | 2 | 58 | 967 (2) | 1,698.6B | 0.059 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 6 | dolma1_6plus | `dolma-v1-6-and-sources-baseline` | `<mix>-1B-5xC-2` | 2 | 10 | 1441 (0) | 1,885.2B | 0.053 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 7 | c4 | `c4` | `<mix>-1B-5xC-2` | 2 | 49 | 171 (0) | 138.4B | 0.722 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 8 | fineweb-pro | `prox_fineweb_pro` | `<mix>-1B-5xC-2` | 2 | 28 | 128 (0) | 82.7B | **1.209** | 17.3% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 9 | fineweb-edu | `fineweb_edu_dedup` | `<mix>-1B-5xC-2` | 2 | 31 | 234 (0) | 192.2B | 0.520 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 10 | falcon | `falcon` | `<mix>-1B-5xC-2` | 2 | 26 | 188 (0) | 456.4B | 0.219 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 11 | falcon-and-cc | `falcon_and_cc` | `<mix>-1B-5xC-2` | 2 | 19 | 411 (0) | 1,054.8B | 0.095 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 12 | falcon-and-cc-qc-10p | `falcon_and_cc_eli5_oh_top10p` | `<mix>-1B-5xC-2` | 2 | 31 | 271 (0) | 146.1B | 0.685 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 13 | falcon-and-cc-qc-20p | `falcon_and_cc_eli5_oh_top20p` | `<mix>-1B-5xC-2` | 2 | 4 | 334 (0) | 285.8B | 0.350 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 14 | falcon-and-cc-qc-orig-10p | `falcon_and_cc_og_eli5_oh_top10p` | `<mix>-1B-5xC-2` | 2 | 40 | 240 (0) | 147.8B | 0.676 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 15 | falcon-and-cc-qc-tulu-10p | `falcon_and_cc_tulu_qc_top10` | `<mix>-1B-5xC-2` | 2 | 1 | 240 (0) | 128.9B | 0.776 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 16 | dclm-baseline | `DCLM-baseline` | `<mix>-1B-5xC-2` | 2 | 8 | 941 (0) | 3,857.1B | 0.026 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 17 | dclm-baseline-qc-7p-fw2 | `dclm_ft7percentile_fw2` | `<mix>-1B-5xC-2` | 2 | 2 | 500 (0) | 507.5B | 0.197 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 18 | dclm-baseline-qc-7p-fw3 | `dclm_ft7percentile_fw3` | `<mix>-1B-5xC-2` | 2 | 1 | 256 (0) | 218.0B | 0.459 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 19 | dclm-baseline-qc-fw-3p | `dclm_fw_top3` | `<mix>-1B-5xC` **(no `-2`)** | 6198 | 1 | 128 (0) | 118.1B | 0.847 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 20 | dclm-baseline-qc-fw-10p | `dclm_fw_top10` | `<mix>-1B-5xC` **(no `-2`)** | 6198 | 4 | 128 (0) | 98.5B | **1.016** | 1.6% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 21 | dclm-baseline-qc-10p | `pos_eli5_oh_neg_dclm_refinedweb_steps_20` | `<mix>-1B-5xC` **(no `-2`)** | 6198 | 1 | 128 (0) | 82.1B | **1.219** | 17.9% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 22 | dclm-baseline-qc-20p | `pos_eli5_oh_neg_dclm_refinedweb_steps_20` | `<mix>-1B-5xC` **(no `-2`)** | 6198 | 1 | 192 (0) | 171.1B | 0.585 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 23 | dclm-baseline-25p-dolma1.7-75p | `dolma17-75p-DCLM-baseline-25p` | `<mix>-1B-5xC-2` | 2 | 1 | 1016 (1) | 2,246.0B | 0.045 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 24 | dclm-baseline-50p-dolma1.7-50p | `dolma17-50p-DCLM-baseline-50p` | `<mix>-1B-5xC-2` | 2 | 1 | 991 (0) | 2,813.6B | 0.036 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |
| 25 | dclm-baseline-75p-dolma1.7-25p | `dolma17-25p-DCLM-baseline-75p` | `<mix>-1B-5xC-2` | 2 | 1 | 970 (0) | 3,335.9B | 0.030 | 0.0% | 16 | OLMoTokenizerFast / 50279 | step69369 |

**Common to all 25.** These values come from code, so they are identical by construction (OLMo@DataDecide `scripts/ladder.py`):

| Setting | Value | Source |
|---|---|---|
| Architecture | d_model 2048, 16 heads, 16 layers, mlp_ratio 8, SwiGLU, RoPE, RMSNorm, no weight tying, vocab 50280 / embedding 50304, seq len 2048 | `:48-78`, `:94` |
| LR | 0.0047·(1e9/1.08e8)^(-1/3) = 2.24e-3 | `:207` |
| Optimizer | AdamW, wd 0.05, β = (0.9, 0.95) | `:231-240` |
| Schedule | cosine with warmup, alpha_f 0.01, warmup = round(1e9/(704·2048)) = 694 steps | `:241-246` |
| Global batch | 704 | `:352-362` |
| Token budget | `max_duration = 100000000000T` | `:247` |
| Tokenizer | `tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json` | `:250` |
| Data loader | `drop_last=True`, same `InstanceFilterConfig` | `:342-346` |

How the final step is reached:
- `max_steps = ceil(1e11 / 1,441,792) = 69,359`. The code is `olmo/train.py:238-248`. I checked the arithmetic with `python3 -c "import math;print(math.ceil(1e11/1441792))"`, which prints 69359.
- Training stops at `stop_at = max_steps + 10 = 69,369` (`olmo/train.py:1013`).
- So `step69369` is the end of training for every model. Its last 10 steps run at the LR floor η_min = 0.01·LR (`olmo/optim.py:703-704`).
- The default-seed weka listings end at step 69369 for all 25 runs (`DataDecide/checkpoints/weka_paths.jsonl` lines 375-398 and 573-582 print `[67500, 69369]`).
- The README hyperparameter table lists 1B: batch 704, 69,369 steps, 100.0B tokens (`DataDecide/README.md:92-108`). It shows LR 2.1e-3, which is the formula applied to the real parameter count of 1176.8M. The code uses the nominal size 1e9 for every recipe, so this does not differ between models.

**What I checked empirically:**
- All 25 loaded with 16 layers, `OLMoTokenizerFast`, EOS 50279, `=` token id `[30]`, revision `step69369-seed-default`. The command was a python loop over `results/datadecide/<recipe>.json` (`tokenization_diagnostics`).
- The per-model numeral-length histograms in those files differ slightly. They are counted over model-specific filtered prompts, so they are not evidence of different tokenizers.
- Every model has the same budget of 100,015,669,248 training tokens: `per_seed["2"].tokens` in every `exact_100b_<recipe>/summary.json`.

## 2. Findings (per-recipe anomalies)

### F1. Data repetition (epochs)

Three recipes have pools smaller than 100B tokens, so they trained for more than one epoch:

| Recipe | n_chunks | Epochs | Pool share seen twice | Training tokens that are repeats |
|---|---|---|---|---|
| fineweb-pro | 40,392,120 | 1.209 | 20.9 % | 17.3 % |
| dclm-baseline-qc-10p | 40,076,013 | 1.219 | 21.9 % | 17.9 % |
| dclm-baseline-qc-fw-10p | 48,078,767 | 1.016 | 1.6 % | 1.6 % |

- **Proof:** `membership.n_chunks` and `membership.base = 3` in `results/corpus_alpha_datadecide/exact_100b_{fineweb-pro,dclm-baseline-qc-10p,dclm-baseline-qc-fw-10p}/summary.json`. Base 3 means a chunk can be used 0, 1 or 2 times.
- The project already reports this as "Share of recipe read" of 121 %, 122 % and 102 % in `experiments/E04_exact_100b_alpha/DETAILS.md:143,155,156`, with the explanation at `:162-164`.
- **Maximum repetition anywhere is 2 exposures of a chunk.**
- The next-largest repeaters are under one epoch: dclm-baseline-qc-fw-3p 0.847, falcon-and-cc-qc-tulu-10p 0.776, c4 0.722.
- Repetition through duplicated file lists is limited to Wikipedia/Wikibooks (2 files listed twice, OLMo `named_data_mixes.py:40-45`). It appears in the 5 Dolma-1.7 recipes and in dclm-baseline-25p-dolma1.7-75p. Those pools are read at only 0.045–0.074 epochs, so the expected exposure of a Wikipedia token is about 0.1–0.15. This is upweighting, not memorisation-level repetition.
- There is no heavy upsampling of a small pool. The "top-3 %" pool, dclm-baseline-qc-fw-3p, is 118B tokens and is read 0.85 times.

### F2. Seed provenance differs for 4 recipes

- **The rule:** `ladder.py:166-168` sets `run_name = f"{name}-{model}-{length}"` and appends `-{seed}` only if `seed != DEFAULT_SEED (6198)` (`:100`). The aux seeds are named `-4`, `-5`, `-14` and `-15` (`weka_paths.jsonl`).
- **21 recipes:** the released `seed-default` checkpoints come from `…-1B-5xC-2` runs, so by this rule they were launched with seed 2. This is what the project assumes (`docs/datadecide_sampling.md:156-166`).
- **4 recipes:** dclm-baseline-qc-fw-3p, qc-fw-10p, qc-10p and qc-20p come from runs named `…-1B-5xC` with no suffix (`DataDecide/checkpoints/final_defualt_seed_paths.jsonl:173,176,181,182`; also `matched_runs_by_s3.json`). By the same rule they used **seed 6198**.
- The seed drives both `seed_all(cfg.seed)` (weight init, `scripts/train.py:119`) and the data order (`olmo/data/__init__.py:108,113`).
- These 4 runs also look like a separate training batch. At smaller scales their final steps are off by 10 (150M: 38167 vs 38157; 300M: 45797; 750M: 63599 vs 63589), and `dclm_fw_top10-750M` stops at step 62500 (`final_defualt_seed_paths.jsonl`, lines around 170-182).
- At 1B their step count (69369) and every config value are the same as the other 21.
- **This is not in the project docs.** `grep` for unsuffixed `5xC` in `docs/` and `experiments/` finds nothing.
- **Effect on project data:** E04 counted seed 2 for all 25 (`experiments/E04_exact_100b_alpha/README.md:22`). For these 4 recipes, the counted stream is therefore probably not the exact trained stream. E04 shows that α and numbers-seen are insensitive to the data seed (Δα_OLS ≤ 0.0013, numbers seen < 0.04 %; `docs/assumptions_and_caveats.md` A5). The impact on the data side is negligible.
- **Effect on the model side:** each model still has exactly one random seed. "6198 vs 2" is a different draw of a nuisance variable, not a systematic difference in training.

### F3. Resumptions and restarts

- The number of W&B run records per default-seed 1B run ranges from 1 to 58 (table column; `matched_runs_by_s3.json`). This normally reflects Beaker preemptions and resumes.
- OLMo restores the optimizer, step and data position from the checkpoint (`olmo/train.py:334-372`). A resume is therefore not a protocol change.
- There is no evidence of a failed or diverged run: all 25 reached step 69369.
- **One subtle unverified caveat** concerns the >1-epoch recipes:
  - On resume, the data loader is rebuilt with `seed + (cfg.epoch or 0)` (`olmo/data/__init__.py:113`). `cfg.epoch` is `None` by default (`ladder.py:258`, `:483`).
  - `reshuffle()` (seed+1) is called only at a live epoch boundary (`olmo/train.py:1244-1248`), while the trainer epoch is restored from the checkpoint (`:356`).
  - So a restart *during epoch 2* would continue with the epoch-1 permutation instead of seed+1.
  - That changes *which* chunks are repeated, not how many. The repeats would still be a uniform random subset.
  - This can only apply to fineweb-pro (28 W&B records; epoch 2 = last ~12k steps) and dclm-baseline-qc-fw-10p (4 records; epoch 2 = last ~1,075 steps). dclm-baseline-qc-10p has a single W&B record.
  - It is unverifiable without run logs and immaterial to α (A5).

### F4. Tokenization provenance

- All 25 use the same tokenizer file and vocabulary (`ladder.py:250`). All decoded with `OLMoTokenizerFast`, EOS 50279 (table).
- The token files were produced by different pipelines (path components in `configs/datadecide_data_map.json`):
  - dolma1_6plus uses a re-tokenization `gpt-neox-olmo-dolma-v1_5-tokenizers-0-19-1` (`named_data_mixes.py:2560`).
  - dclm-baseline and the DCLM half of the mixes use `fastdclm/.../allenai/`.
  - The falcon-QC and dclm-QC-10/20p files sit in directories without a tokenizer name.
- I could not check that these pipelines tokenize digits identically, because HF is blocked. This is a caveat, not evidence of a problem.

### F5. Nominal and actual pool size (label anomaly, not a training anomaly)

- dclm-baseline-qc-fw-10p ("top 10 % by FineWeb-edu classifier") has a pool of 98.5B tokens. That is *smaller* than dclm-baseline-qc-fw-3p ("top 3 %") at 118.1B.
- 10 % of the 3,857B DCLM-baseline pool would be about 386B.
- Both file lists are the same generated template, `part-{i:02d}-0000{j}.npy for i in range(64) for j in range(2)` = 128 files (`named_data_mixes.py:1785` and `:2552`).
- So the qc-fw-10p model plausibly trained on a 128-file subset of its nominal top-10 % pool (roughly 25 % of it). That would explain why it is the one with about 1.02 epochs.
- The model's training is still exactly "100B tokens from those 128 files", like every other model. The anomaly is in what the label "10 %" means, not in comparability of training.
- I did not verify on the Hub whether more `v0_fw_top10p` files exist, because HF is blocked.

## 3. Proposed exclusion criterion (a priori)

**Rule R-epoch:** exclude a model if its training stream repeats its data pool more than **4 epochs**, where epochs = 48,835,776 / number of 2048-token chunks in its recipe.

Justification:
- Muennighoff et al. 2023, "Scaling Data-Constrained Language Models" (NeurIPS 2023), find that training for up to about 4 epochs of repeated data gives negligible change in loss compared with unique data. The value of repeated tokens decays strongly only well beyond that (16+ epochs approaches zero).
- Below 4 epochs, a model trained on a repeated pool is, to first order, equivalent to one trained on unique data from the same distribution. So it is a valid member of a "same budget, different data" comparison.
- The threshold is fixed from the literature, not from the data here.

**Models flagged by R-epoch: none.** The maximum is 1.219 (dclm-baseline-qc-10p).

Sensitivity labels (not exclusions):
- **S1, epochs > 1:** fineweb-pro (1.209), dclm-baseline-qc-10p (1.219), dclm-baseline-qc-fw-10p (1.016).
  - These are the only models whose stream contains any chunk twice. In practice only the first two matter (about 17–18 % repeated tokens); qc-fw-10p has 1.6 %.
  - A cautious analyst could run a robustness check without them. Under the Muennighoff result there is no scientific ground to exclude them.
- **S2, seed provenance (seed 6198 instead of 2 by naming rule):** dclm-baseline-qc-fw-3p, dclm-baseline-qc-fw-10p, dclm-baseline-qc-10p, dclm-baseline-qc-20p.
  - This is not a reason to exclude, because the seed is a random nuisance factor and every model has one draw.
  - It *is* a documentation fix: E04 counted seed 2 for these 4, but their trained stream is probably seed 6198. A5 says this is immaterial to α.
  - If exact streams are needed (for example E08 prefix counts), these 4 should be recounted with seed 6198, and the epoch-2 reshuffle with seed 6199 for qc-10p and qc-fw-10p.

Rules I considered and rejected:
- **Pool < 100B:** this is the same thing as S1, and as shown above it is not harmful.
- **Number of restarts:** resumes are deterministic, and all runs completed.
- **Separate training batch (the 4 DCLM-QC runs):** at 1B the config and step count are identical.
- **Different tokenization pipeline for dolma1_6plus:** same vocabulary, and no evidence of different behaviour.

## 4. Honest conclusion

I found **no valid training or architecture reason to exclude any of the 25 models**. All 25 share:
- the same architecture (16 layers, 2048 hidden, 16 heads; confirmed at load);
- the same tokenizer and EOS;
- the same 100.016B-token budget (69,369 × 704 × 2048);
- the same LR, optimizer and cosine schedule;
- the same final checkpoint, step69369, which is the end of training (`max_steps` + 10).

The only per-recipe training differences:
- (a) Mild repetition, at most 1.22 epochs, in three recipes. This is far below the about 4-epoch threshold where repetition starts to matter.
- (b) A different random seed (6198 instead of 2) for the four DCLM-QC recipes, which also seem to have been trained in a separate batch.
- (c) An unverifiable epoch-2 reshuffle subtlety on resume.
- (d) The qc-fw-10p pool is smaller than its label implies.

None of these breaks comparability. Items (a) and (b) are worth stating as sensitivity labels and as a correction to `docs/datadecide_sampling.md` and E04 (seed 2 is not right for 4 recipes).

## 5. Reproduction commands

```bash
# epochs per recipe
python3 -I -c "import json,csv;S=69369*704;[print(r,S/json.load(open(f'results/corpus_alpha_datadecide/exact_100b_{r}/summary.json'))['membership']['n_chunks']) for r in [x['recipe'] for x in csv.DictReader(open('results/datadecide/alpha_beta.csv'))]]"
# run dirs of the released default-seed 1B checkpoints (note lines 173,176,181,182 lack -2)
grep -n -- "-1B-6198" /tmp/claude-0/agentrefs/DataDecide/checkpoints/final_defualt_seed_paths.jsonl
# naming rule / default seed / stop_at
sed -n 100p /tmp/claude-0/agentrefs/OLMo/scripts/ladder.py; sed -n 164,169p /tmp/claude-0/agentrefs/OLMo/scripts/ladder.py; sed -n 1013p /tmp/claude-0/agentrefs/OLMo/olmo/train.py
# fw-top3 / fw-top10 file templates
sed -n 1785p /tmp/claude-0/agentrefs/OLMo/olmo/data/named_data_mixes.py; sed -n 2552p /tmp/claude-0/agentrefs/OLMo/olmo/data/named_data_mixes.py
```
