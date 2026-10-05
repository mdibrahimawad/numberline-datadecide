# How DataDecide 1B models sampled their training data

Sources read (cloned 2026-10-05):

| repo | ref | commit |
|---|---|---|
| github.com/allenai/DataDecide | `main` | `68abc496` (2025-06-17) |
| github.com/allenai/OLMo | branch `DataDecide` | `7094aab0` (2025-03-13) |
| github.com/allenai/OLMo | branch `datadecide-data-upload` (HF upload scripts) | `07c99f22` |
| github.com/allenai/OLMo | branch `benb/ladder-1xC` (same `ladder.py` seed/run-name logic) | — |

Paths below are `DataDecide:<file>:<line>` or `OLMo@DataDecide:<file>:<line>`.

## 1. The 1B training configuration

DataDecide launches every run with
`scripts/beaker/ladder-launch.sh 1 normal --model {scale} --data {mix} --length 5xC --name {mix} [--s3] [--seed]`
(`DataDecide:pretraining/create_ladder_over_scale_script.py:24`, scales in
`DataDecide:pretraining/eval-for-consistent-ranking-scales.txt:14` = `1B`).
`OLMo@DataDecide:scripts/ladder.py` turns that into a `TrainConfig`:

| setting | value | where |
|---|---|---|
| architecture | `MODEL_CONFIG_150M` updated to d_model 2048, 16 heads, 16 layers, mlp_ratio 8 | `scripts/ladder.py:48-78`, `:94` |
| sequence length | `max_sequence_length=2048` | `scripts/ladder.py:68` |
| vocab / EOS / pad | `vocab_size=50280`, `eos_token_id=50279`, `pad_token_id=1` | `scripts/ladder.py:69-72` |
| tokenizer | `tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json` (`<|endoftext|>` = 50279, `<|padding|>` = 1) | `scripts/ladder.py:250` |
| tokens | `--length 5xC` → 5 · 20 · 1e9 = **100B tokens** (`max_duration="100000000000T"`) | `scripts/ladder.py:145-160`, `:247` |
| global batch | `160·(1e9/1.08e8)^(2/3)` rounded to a multiple of 32 → **704** sequences = 1,441,792 tokens/step | `scripts/ladder.py:352-362` |
| final checkpoint | `step69369` → 69,369 × 704 = 48,835,776 sequences = 100.02B tokens | `DataDecide:README.md:51`, `checkpoints/final_defualt_seed_paths.jsonl:124` |
| seed | `TrainConfig.seed = args.seed`, default `DEFAULT_SEED = 6198` | `scripts/ladder.py:100`, `:218`; `olmo/config.py:858` |
| data paths | `[f"{prefix}/{p}" for p in named_data_mixes.DATA_PATHS[args.data]]` | `scripts/ladder.py:347` |
| drop_last / instance filter | `drop_last=True`, `instance_filter=InstanceFilterConfig()` (repetition filter) | `scripts/ladder.py:342`, `:346` |

**Seed caveat.** `ladder.py` appends `-{seed}` to the run name only when
`seed != 6198` (`scripts/ladder.py:166-168`; same on `benb/ladder-1xC`). The
released 1B "default seed" checkpoints come from runs named
`<mix>-1B-5xC-2` (e.g. `baseline-1B-5xC-2`, `DCLM-baseline-1B-5xC-2`,
`DataDecide:checkpoints/final_defualt_seed_paths.jsonl:124,130,140`;
`release/matched_runs_by_s3.json`), while the aux seeds are `-4`, `-5`
(`DataDecide:release/upload_checkpoints.py:15-21`). DataDecide relabels the
`-2` runs as seed 6198, but by the naming rule they were most likely launched
with `--seed 2`, i.e. **data order seed 2**. The W&B run config
(e.g. `wandb.ai/ai2-llm/olmo-ladder-benb/runs/fp23ljle`) would settle it; it
was not reachable from here. The exact-order check therefore runs both seeds.
The run name `baseline-1B` comes from `--name baseline`; DataDecide maps
`baseline` → `dolma17` (`DataDecide:release/upload_checkpoints.py:60-61`).

## 2. Recipe → tokenized files

* Mix names: `DataDecide:pretraining/eval-for-consistent-ranking-mix-names.jsonl:1-25`.
* Mix name → public slug (`allenai/DataDecide-<slug>-1B`):
  `recipe_display_name`, `DataDecide:release/upload_checkpoints.py:26-52`.
* Mix name → file list: `DATA_PATHS[mix]` in
  `OLMo@DataDecide:olmo/data/named_data_mixes.py`
  * source → files: `DATA_SOURCES` (`:6`), `DOLMA_1_6_TO_1_7_DATA_SOURCES` (`:2556`),
    `EXTRA_DATA_SOURCES` (`:4045`, holds `DCLM-baseline`);
  * `build_collection_include(corpora, sample_factor)` concatenates the
    sources' lists in the given order (`:5082-5103`); repeating a source or a
    path repeats its files (upsampling);
  * recipes: `:5127-5183` (e.g. `dolma17` `:5133`, `dolma-v1-6-and-sources-baseline` `:5170`).
* **Upsampling**: `wikipedia_wikibooks` lists its two files twice (`:40-45`), so every Dolma 1.7 recipe sees Wikipedia 2× (`dolma1_7`,
  `-no-code`, `-no-math-code`, `-no-reddit`, `-no-flan`; one of them survives
  in `dclm-baseline-25p-dolma1.7-75p`). No other recipe repeats files.
* **Mixture proportions** are implicit: files are concatenated, so each
  source contributes in proportion to its token count (× repeats). There are
  no sampling weights (`build_collection_with_weights`, `:5116`, is unused by
  the 25 recipes). `SOURCES_SIZES` (`:4995`) lists Dolma source sizes in
  tokens; actual per-recipe shares are written by the sampler to
  `results/corpus_alpha_datadecide/<recipe>/mixture.json` from HF file sizes.
* **Mixed recipes** `dclm-baseline-{25,50,75}p-dolma1.7-{75,50,25}p` use
  `sample_factor` (`:5173-5183`): `random.seed(62540); random.shuffle(paths)`
  **in place** on the shared source lists, then keep `ceil(len·factor)` files
  (`:5094-5099`). Because the shuffle mutates the lists and is re-seeded per
  call, the subsets depend on the call order in the module, so the only exact
  reproduction is executing the module — which is what
  `src/datadecide_sampling.py::build_data_map` does (`runpy`).
* **HF location**: `upload_to_hf.py` downloads each S3 key to
  `local_dir/<s3_key>` and uploads `local_dir` with `upload_large_folder`
  (`OLMo@datadecide-data-upload:scripts/beaker/upload_to_hf.py:35`, `:65`,
  `:76-79`), so the path inside `allenai/DataDecide-data-recipes` equals the
  S3 key in `named_data_mixes.py`. `dolma1_6plus` was uploaded with
  `--dolma-1-6-bypass` (`olmo/data/readme.md` on that branch); its files are
  the `...-tokenizers-0-19-1` re-tokenization (`named_data_mixes.py:2560`).
  `modal run modal_app/datadecide_alpha_app.py::sweep --dry-run` checks every
  path against the Hub.

The result is `configs/datadecide_data_map.json`: for all 25 slugs, the mix
name, model repo, ordered path list (duplicates kept), repeated paths and
files per source, plus the constants below.

| slug | OLMo mix | files | repeated |
|---|---|---|---|
| dolma1_7 | dolma17 | 1033 | 2 (wiki) |
| dolma1_7-no-code | no_code | 914 | 2 |
| dolma1_7-no-math-code | no_math_no_code | 859 | 2 |
| dolma1_7-no-reddit | no_reddit | 955 | 2 |
| dolma1_7-no-flan | no_flan | 967 | 2 |
| dolma1_6plus | dolma-v1-6-and-sources-baseline | 1441 | 0 |
| c4 | c4 | 171 | 0 |
| fineweb-pro | prox_fineweb_pro | 128 | 0 |
| fineweb-edu | fineweb_edu_dedup | 234 | 0 |
| falcon | falcon | 188 | 0 |
| falcon-and-cc | falcon_and_cc | 411 | 0 |
| falcon-and-cc-qc-10p | falcon_and_cc_eli5_oh_top10p | 271 | 0 |
| falcon-and-cc-qc-20p | falcon_and_cc_eli5_oh_top20p | 334 | 0 |
| falcon-and-cc-qc-orig-10p | falcon_and_cc_og_eli5_oh_top10p | 240 | 0 |
| falcon-and-cc-qc-tulu-10p | falcon_and_cc_tulu_qc_top10 | 240 | 0 |
| dclm-baseline | DCLM-baseline | 941 | 0 |
| dclm-baseline-qc-7p-fw2 | dclm_ft7percentile_fw2 | 500 | 0 |
| dclm-baseline-qc-7p-fw3 | dclm_ft7percentile_fw3 | 256 | 0 |
| dclm-baseline-qc-fw-3p | dclm_fw_top3 | 128 | 0 |
| dclm-baseline-qc-fw-10p | dclm_fw_top10 | 128 | 0 |
| dclm-baseline-qc-10p | pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p | 128 | 0 |
| dclm-baseline-qc-20p | pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p | 192 | 0 |
| dclm-baseline-25p-dolma1.7-75p | dolma17-75p-DCLM-baseline-25p | 1016 | 1 |
| dclm-baseline-50p-dolma1.7-50p | dolma17-50p-DCLM-baseline-50p | 991 | 0 |
| dclm-baseline-75p-dolma1.7-25p | dolma17-25p-DCLM-baseline-75p | 970 | 0 |

## 3. Token format

* dtype: `DataConfig.memmap_dtype = "uint16"` (`olmo/config.py:582`), passed to
  `MemMapDataset` (`olmo/data/__init__.py:38-39`).
* Files are raw headerless memmaps despite the `.npy` suffix: a chunk is read
  at byte `index * itemsize * chunk_size` (`olmo/data/memmap_dataset.py:150-160`).
* Documents are concatenated with EOS **50279** after each one
  (`scripts/ladder.py:71`); there is no BOS.

## 4. Loader: concatenation, chunking, shuffling, budget

1. `build_memmap_dataset` keeps `DataConfig.paths` in list order
   (`olmo/data/__init__.py:17-46`).
2. `MemMapDataset` cuts **each file separately** into
   `floor(file_tokens / 2048)` chunks — the tail remainder of every file is
   never trained on (`olmo/data/memmap_dataset.py:162-165`) — and gives the
   files consecutive global index ranges in path order (`offsets`, `:95-148`).
   Chunks ignore document boundaries; a chunk may span several documents.
3. `build_train_dataloader` wraps it in `IterableDataset` with
   `seed = data.seed or train.seed` (+ epoch) (`olmo/data/__init__.py:108`, `:113`).
4. `IterableDataset._build_global_indices`: `indices = arange(N, uint32)`,
   `np.random.Generator(np.random.PCG64(seed)).shuffle(indices)`, truncated
   for `drop_last` (`olmo/data/iterable_dataset.py:88-111`); a new epoch
   reshuffles with `seed + 1` (`:119-121`). Ranks take strided slices
   (`:138`), so global step *s* consumes `indices[704·s : 704·(s+1)]`.
5. Training stops after 100B tokens = the first **48,835,776** shuffled
   chunks (step 69,369). For Dolma 1.7 (~1.7T tokens, ~840M chunks) that is
   ~6% of the corpus; the uniform shuffle makes it a uniform random subset of
   chunks.
6. Chunks failing the repetition filter (`memmap_dataset.py:225-236`) are still
   drawn, but their labels are masked out of the loss (`olmo/train.py:640`).

## 5. What the sampler copies (src/datadecide_sampling.py)

* `plan_windows`: windows of 262,144 tokens (~1 MB of text) with **exactly
  uniform per-token coverage** of the concatenated stream (duplicated files
  count twice): a file is picked with probability ∝ `len + W − 1` and the
  start uniform on `[−(W−1), len−1]`, clipped to the file. This is the
  expectation of the training distribution (uniform chunk subset), the analogue
  of scheme B. Windows never cross files.
* `split_documents`: split on EOS 50279; `drop_partial_docs` (default, as
  specified) keeps only documents entirely inside the window; `keep_partial`
  keeps the cut edge documents minus 8 tokens at each cut. Both are counted
  for every window. Dropping partial documents under-weights documents
  longer than a fair fraction of the window (a document of L tokens survives
  with probability ≈ (W−L)/W; books of >262k tokens never do), so the
  difference between the two modes measures that bias.
* Decoding: the recipe model's `tokenizer.json` (`allenai/DataDecide-<slug>-1B`,
  revision `step69369-seed-default`, falling back to `main`), via
  `_decode_and_count_native` → `_count_text`.
* `training_chunk_indices` / `chunks_to_windows`: the optional exact training
  order (per-file `floor(len/2048)` chunks, PCG64 shuffle, first 69,369 × 704
  indices, epoch reshuffle with seed+1), used to compare 1000 trained chunks
  against 1000 uniform chunks for data seeds 2 and 6198. N ≈ 840M chunks for
  Dolma 1.7: a 3.4 GB uint32 array and ~1.5 min to shuffle, run in a 32 GB
  Modal container.
