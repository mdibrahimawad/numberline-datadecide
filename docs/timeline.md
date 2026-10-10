# Timeline

What was done, in order, with the decision behind each step. Commit hashes are on
branch `alpha-sampling` (merged into `main` on 2026-10-10, now the only branch) unless noted. Dates are when the work landed in git.

## Before 2026-10-05: prior work (E00)
The original repository (uploaded to `main` on 2026-10-05) covered:
- **Stage 1**: integer frequencies in The Pile and other corpora (RedPajama,
  RefinedWeb, Dolma, The Stack), plus full-pass counts of SlimPajama and LLM-JP v3
  (`results/corpus_alpha_full/`, which later became E02's ground truth);
- **Stage 2**: the number-line geometry probe on public models (Pythia, GPT-2, Llama,
  Mistral, OLMo); layer-selection variants, robust PCA, manifold diagnostics;
- **Stage 3**: circuit attribution;
- a number-comparison probe with a LaTeX appendix (`paper/`);
- the paper's released code and a "paper-faithful" copy (`reference/`).

## 2026-10-05
- **E01 built** (branch `datadecide-sweep`, `027fe42`): correct-output filter with
  resampling; DataDecide β sweep on Modal (joint layer selection, frozen evaluation,
  direct + log β); 60M smoke test.
- **Cleanup** (`5cab307`): full-pass ground-truth files copied into place; `.gitignore`.
- **E02 built and run** (`3931117`): SlimPajama sampling validation.
  Result: α_MLE is accurate from 10M tokens; α_OLS is biased below ~1B.
  *Decision: report α_MLE for samples; use α_OLS only from exact streams.*
- **E03 built** (`3931117`, `cad83ec`): DataDecide training-data reproduction
  (`docs/datadecide_sampling.md`) and the window sampler.
- **E04 c4 and dolma1_7** (`2ceb652`, `79d3a8a`, `0d672db`): exact 100B training streams for
  5 data seeds. Result: 100B samples = full recipe; the seed doesn't matter.
  *Decision: count one seed (2) per recipe.*
- **E01 run by the user on Modal**: β for all 25 models (`results/datadecide/`).
- **E03 window sweep** for all 25 recipes, run overnight (`1c9d2ad` made it survive
  failures). Result: α_MLE vs β r = −0.14 (no relation).
- **E04 `exact_all`** started (`dd8aa74`, `014e71d`, `fc42ce4` smallest-first).

## 2026-10-06
- Exact α reaches 8 → 10 → 14 recipes over two Modal workspaces (the first was disabled
  when its budget ran out).
- **E05 analyses**: confounders (`40dd191`): number density beats α within families; the
  efficient-coding γ law is not supported (n = 10); decade metrics (`2b1b633`, `f7cb881`):
  the saturation metric (K ≈ 4000) is the best single data metric at n = 14.
- Asked: are intermediate checkpoints released? Yes, every 2,500 steps (the basis of E08).

## 2026-10-07
- **Moved compute to RunPod** (`157eddf` … `09bfaaa`): `runpod_jobs/exact_alpha.py` with
  cgroup-aware sizing, CDN range reads, atomic writes, HF upload + verify, multi-pod claims,
  `--max-hours`, self-delete; `docs/runpod.md` beginner guide.
- **Overnight run**: 2 CPU pods counted the last 11 (largest) recipes.

## 2026-10-08
- **All 25 exact α in** (`7e808f1`, `ce40188`). Full E05 rerun: α_OLS vs β r = +0.56
  (p = 0.003), Dolma-driven; within-family α fails in DCLM/Falcon.
- Found that the β-fit R² is blind near β = 1, which corrected an earlier claim that the
  Dolma β values were unreliable.
- **E06 built** (`493572d`) **and run** on 1 RunPod GPU (fixes `f8d5c0a`, `3a47683`,
  `f1f20bc`): 10-group β for all 25 models in 18 min. Result: β reproducible; same-layer
  α link r = 0.63; layer re-selection is fragile.
- **E07** (`6b429fb`, `f835e2b`): 26 count features. R (rarely-seen count) is the best
  predictor (ρ = −0.73) and absorbs α. Shrinkage interpretation; R is stable across data
  seeds.
- Code tidied and tested (`6728032`); result tables refreshed for all 25 recipes
  (`338c540`); repository reorganised into `experiments/` and `docs/` (this commit series).

## 2026-10-10
- Fixed-layer analysis (`src/fixed_layer_beta.py`): at one common layer α, R and S predict β about
  equally well; "R absorbs α" was layer-dependent.
- **E09 pre-registered**: three frozen predictors (α, S, R) to be tested on the 6 Paloma baselines.

- **E09 run end to end**: 146.8B-token counts of 5 Paloma corpora on RunPod CPU pods
  (`runpod_jobs/corpus_sample.py`, Dolma exact in 8 parts), frozen predictions committed before
  β, β of the 6 models on one GPU pod (`runpod_jobs/paloma_beta.sh`), scored by
  `src/paloma_analysis.py`: only 3 models valid at layer 8; no formula transfers; R falsified.
- Efficient-coding test at a fixed layer (`src/efficient_coding.py`); multi-cutoff R tried and
  rejected as overfit; `main` made the only branch.

## Next
- **E08**: the checkpoint test (R changes during training while α stays fixed);
  definitions and predictions are pre-registered in `experiments/E08_checkpoint_test/README.md`.
- Training-seed noise of β (seeds `-4`, `-5`).
- Fixed-layer β for all models from the saved per-prompt scores (free).
