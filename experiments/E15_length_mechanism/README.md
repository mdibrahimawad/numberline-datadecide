# E15: does token count make the log number line?

**Question.** Veli asked what plays the role of the "leading zeros" of the one-hot digit tables inside an LLM. Hilal
proposed **token count**: at the final `=`, each query token adds a roughly constant push while each digit multiplies the
value by 10, so equal pushes per ×10 look logarithmic. If so, the log pattern should follow the tokenizer.

**Status.** Done (2026-11-24). Pre-registration `PREREG.md`, frozen at commit `eff540e` before any real-model run (code
tested on pythia-14m only). No deviations. Full tables: `results/e15/RESULTS.md` and `results/e15/*.csv`.
Code: `src/e15_length_mechanism.py`. Vectors (~0.5 GB) are not committed; rerun with the commands below.

**Models** (CPU, fp32): Pythia-1.4B (same number tokenization as the DataDecide models: 10,000 / 10,000 identical),
Qwen2.5-1.5B (one token per digit), OLMo-2-1B (groups of up to 3 digits, like Llama 3). 3,690 prompts per model.

## Result (pre-registered overall reading: **token-count mechanism contradicted; digit length supported**)

| | Pythia-1.4B (L* 10) | OLMo-2-1B (L* 7) | Qwen2.5-1.5B (L* 10) |
|---|---|---|---|
| **T1** shift per leading zero, decades | **+1.06** [1.02, 1.11] → length | **+0.92** [0.89, 0.95] → length | **+1.11** [1.07, 1.14] → length |
| … tokens added by the zeros | **none** (`07`, `007`, `0007` are one token) | none for `07`, `007` | one per zero |
| **T3** `1 234` vs `1234`, decades | −1.06 [−1.09, −1.03] | −1.54 [−1.58, −1.50] | −0.39 [−0.42, −0.37] |
| **T3** `1_234` vs `1234`, decades | −0.79 [−0.81, −0.76] | −0.87 [−0.91, −0.83] | −0.28 [−0.30, −0.26] |
| **T4** jump 99→100 (token added?) | +0.51 [0.34, 0.68] (no) | +0.18 [0.07, 0.29] (no) | +0.56 [0.35, 0.77] (yes) |
| **T4** jump 999→1000 (token added?) | +0.20 [0.05, 0.35] (≈ no: 1.0 → 1.2) | +0.22 [0.08, 0.34] (yes) | +0.34 [0.21, 0.45] (yes) |
| T4 verdict | does not follow tokens | mixed | follows tokens |

Sources: `results/e15/summary.csv`, `*_T1_by_width.csv`, `*_T3.csv`, `*_T4.csv`. CIs: 95 % bootstrap over items.

**What it means**
1. **The LLM's "leading zeros" exist, but they are digit characters, not tokens.** Writing `7` as `0007` moves it about
   one decade up the number axis per zero in all three models, including Pythia, where `0007` is a single token, and
   OLMo-2, where `007` is a single token. The models encode *how many digits the string has*, even inside one token.
   This is the one-hot-table effect Veli described, but not by Hilal's token-counting route.
2. **More tokens can push the number down.** Splitting `1234` into `1 234` or `1_234` adds a token, but the number moves
   0.3–1.5 decades *down*, as if the model read mainly the last digit group (3 digits). The token account predicted up.
3. **Jumps appear where no token is added.** Pythia (the DataDecide tokenizer) jumps at 99→100 with no change in token
   count; so does OLMo-2. Only Qwen, where tokens = digits, "follows tokens", and there the two accounts can't be told
   apart.
4. **Attention (T2, descriptive):** the final `=` puts 1–24 % (Pythia), 3–18 % (OLMo-2) and 3–42 % (Qwen) of its
   attention on the query tokens, and more when there are more tokens (`*_T2_attention.csv`). That is compatible with
   Hilal's picture but cannot show that each token adds a constant push.

**Robustness (exploratory).** Across all layers with a clear number line (|Spearman| > 0.9), the shift per leading zero is
0.57–1.06 (Pythia), 0.71–0.95 (OLMo-2) and 0.72–1.54 (Qwen); spaces push down at every such layer in all three models,
underscores at all but one Pythia layer (`*_all_layers.csv`).

**How it fits E14.** E14 found digit length explains about a quarter of the log pattern in the DataDecide models, with
the jump at 99→100. E15 shows the length signal is a *digit-count* feature the model computes inside tokens, not a
by-product of how many tokens attention mixes.

## Caveats
- The axis is calibrated on real numbers, so "one decade up" means "lands where 10× larger numbers sit". It shows the
  axis responds to length, not that the model thinks 0007 is 7,000.
- Copy accuracy for padded inputs varies (first token correct: Pythia 0.85–1.00, OLMo-2 0.30–0.89, Qwen 0.96–1.00).
- Spaces and underscores also change how numbers look in text (e.g. `1 234` may read as two numbers), so T3 tests
  "more tokens" only together with a format change.
- Pythia stands in for the DataDecide models (same tokenization, different weights and data). Three models, one prompt
  format, correlational read-outs (no interventions).

## Run
```bash
python -m src.e15_length_mechanism extract --model EleutherAI/pythia-1.4b --out <vec_dir>   # ~8 min on 4 CPUs
python -m src.e15_length_mechanism extract --model allenai/OLMo-2-0425-1B --out <vec_dir>   # ~8 min
python -m src.e15_length_mechanism extract --model Qwen/Qwen2.5-1.5B --out <vec_dir>        # ~17 min
python -m src.e15_length_mechanism analyse --vectors <vec_dir> --out results/e15
```
