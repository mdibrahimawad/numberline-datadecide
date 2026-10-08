# E03: DataDecide training data: reproduction and cheap "window" α

**Question:** how exactly did DataDecide feed data to its 1B models, and can a cheap
sample (~1B tokens per recipe) give α for all 25 recipes?
**Status:** done (2026-10-05, Modal CPU). Superseded for α_OLS by the exact counts (E04);
the training-order reproduction is what E04 is built on.
**Outputs:**
- `configs/datadecide_data_map.json`: recipe → ordered training files;
- `docs/datadecide_sampling.md`: how training sampled data, with file and line references into Ai2's code;
- `results/corpus_alpha_datadecide/<recipe>/`: per-recipe window counts
  (`counts_0_to_10000_{drop_partial_docs,keep_partial}.csv`, `mixture.json`,
  `windows.json`, `summary.json`);
- `results/corpus_alpha_datadecide/alpha_summary.csv`: window α for all 25 recipes.

## Method
1. **Read Ai2's code** (DataDecide repo and the `DataDecide` branch of OLMo) and
   document the 1B config, recipe → file lists, token format and loader
   (`docs/datadecide_sampling.md`). The file lists are produced by *executing* OLMo's own
   `named_data_mixes.py`, because the mixed recipes use an order-dependent in-place
   shuffle.
2. **Window sampler** (`src/datadecide_sampling.py`): windows of 262,144 tokens
   (~1 MB of text) placed so that every token of the recipe's concatenated stream has the
   same chance of being covered, which equals the training distribution in expectation.
   Text is split on EOS, decoded with the recipe model's tokenizer and counted with
   `_count_text`. Both edge modes are counted (drop vs keep documents cut by the window).
3. **Validation on dolma1_7:** convergence over 50–1000 windows, split-half agreement,
   and an optional exact-training-order comparison.
4. **Sweep:** 4000 windows (~1B tokens) for each of the 25 recipes, overnight.

## How to run
```bash
python -m src.datadecide_sampling --olmo-repo <OLMo@DataDecide> --datadecide-repo <DataDecide>   # rebuild the data map
modal run modal_app/datadecide_alpha_app.py::validate --dry-run
modal run modal_app/datadecide_alpha_app.py::validate                 # dolma1_7 checks
modal run modal_app/datadecide_alpha_app.py::sweep --dry-run          # verifies every file path on the Hub
modal run modal_app/datadecide_alpha_app.py::sweep --windows 4000     # all 25 -> alpha_summary.csv
python tests/test_datadecide_sampling.py                               # offline, tiny fake .npy files
```

## Headline result
- **Training order reproduced:** 2048-token chunks per file (each file's tail dropped),
  one PCG64 shuffle of all chunks, the first 69,369 × 704 chunks = 100.016B tokens. Seed
  2 is the most likely data seed of the released models.
- **Window α_MLE ≈ exact α_MLE:** across all 25 recipes, window minus exact (E04) =
  0.0003 ± 0.0008 (max 0.0016); the correlation is 1.000.
- **Window α_OLS is biased** (as E02 predicted): −0.036 ± 0.014 (max 0.072); the
  correlation with exact is 0.989. Only exact α_OLS is used downstream.
- **α_MLE (window, n = 25) vs β:** r = −0.14 (p = 0.50). No relation.

Details: [DETAILS.md](DETAILS.md).
