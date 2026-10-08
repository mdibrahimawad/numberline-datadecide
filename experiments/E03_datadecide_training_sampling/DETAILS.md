# E03 details

The full reproduction of DataDecide's data pipeline (config table, recipe → file
counts, upsampling, mixed-recipe subsets, token format, loader steps, seed caveat) is in
[`docs/datadecide_sampling.md`](../../docs/datadecide_sampling.md). The key facts:

| Item | Value |
|---|---|
| Sequence length / global batch | 2048 tokens / 704 sequences |
| Final checkpoint | `step69369` = 48,835,776 chunks = 100.016B tokens |
| Token files | raw headerless uint16 `.npy`, EOS 50279 after each document, no BOS |
| Chunking | each file separately, `floor(len/2048)` chunks (tail never trained) |
| Shuffle | `np.random.Generator(PCG64(seed)).shuffle(arange(N))`, epoch 2 reshuffles with seed+1 |
| Upsampling | Dolma 1.7 recipes list the two Wikipedia files twice (2×) |
| Mixed recipes | `random.seed(62540); random.shuffle(paths)` in place, keep `ceil(len·factor)` files; order-dependent, so the module is executed |
| Data seed of released models | runs named `<mix>-1B-5xC-2` → most likely **seed 2**; DataDecide labels them 6198 |

## Window sweep design
- **Coverage:** a file is picked with probability ∝ `len + W − 1` and the window start is
  uniform on `[−(W−1), len−1]`, clipped to the file, so per-token coverage is exactly
  uniform over the concatenated stream (duplicated files count twice). Windows never
  cross files.
- **Edge modes:** `drop_partial_docs` (default) keeps only documents entirely inside
  the window and under-weights long documents (a document of L tokens survives with
  probability ≈ (W−L)/W); `keep_partial` keeps the cut edges minus 8 tokens. Both are
  saved, and the difference measures the bias.
- **Cost fix** (`cad83ec`): the first recipe cost ~$3 because every window was saved to
  the Modal volume separately and decoded twice. After batching (100 windows per save)
  and decoding once, a recipe costs ~$0.5. The counts were checked to be identical on 1,000
  windows.
- **Robust overnight run** (`1c9d2ad`): a failed recipe is logged and skipped, and
  `alpha_summary.csv` is rewritten after every recipe.

## Window vs exact (all 25 recipes; `alpha_summary.csv` vs `alpha_seed2.csv`)

| | mean (window − exact) | s.d. | max \|diff\| | correlation |
|---|---|---|---|---|
| α_MLE | −0.0003 | 0.0008 | 0.0016 | 1.000 |
| α_OLS | −0.036 | 0.014 | 0.072 | 0.989 |

Median window sample: 1.03B tokens. On dolma1_7, the validation run already showed MLE
within 0.0003 of the exact value and OLS off by 0.055.

## α_MLE vs β (25 models, before the exact counts existed)

| | n | Pearson r | p | Spearman ρ |
|---|---|---|---|---|
| α_MLE vs β_direct | 25 | −0.14 | 0.50 | −0.17 |
| α_MLE vs β_log | 25 | −0.14 | 0.51 | −0.19 |

An early 3-model "r = −0.999" was chance. α_MLE is dominated by small numbers (1–100),
while β spans 10 to 10,000.
