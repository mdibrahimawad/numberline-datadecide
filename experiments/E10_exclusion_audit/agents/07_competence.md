# Auditor 07: basic numerical competence and training health of the 25 DataDecide 1B models

Scope: can each model reliably echo numbers in the `a=a,b=b,c=c,n=` prompt? Is there any other
numeracy, downstream or perplexity evidence per model? Is any model under-trained, broken, or so poor
with numbers that its measured number line is not comparable?

Blind-audit compliance: I did not compute, look up or reason about any alpha-beta correlation. I did not
open `results/datadecide/alpha_beta_correlations.json` or `paper/tables/alpha_beta_by_layer.tex`.
I read `alpha_beta.csv` only for the `recipe` column. While reading `experiments/E06_fine_beta/README.md`
for the filter method, I saw its headline alpha-beta r values. I did not use them, and none of my
criteria or verdicts refer to them. No β value is used in any criterion below. ρ and EV, which measure
whether the number line is present at all, are used as a sanity check only.

## Bottom line
- **No model is broken, under-trained, or invalid to measure.** All 25 pass every hard criterion.
- **One soft flag, fineweb-pro.** The model fails to copy most targets in one magnitude band
  (64% rejection around 464 in E06, 42% for 980-1019 in E01). This is a **property of the model/data
  (behaviour), not a measurement-validity failure**. It is **not a valid reason to exclude** the model.
  At most, use it for a pre-declared sensitivity analysis that is reported next to the full-n result.
- Seven more web-only models reject 10-27% in their worst band (c4, fineweb-edu, dclm-baseline,
  dclm-qc-7p-fw3, dclm-qc-fw-3p, dclm-qc-fw-10p, dclm-qc-10p). This is a note, not a flag.
- **The repo has no per-model number-comparison benchmark or downstream accuracy for DataDecide models.**
  The paper's number-comparison table covers 14 public models only. Per-model OLMES results are on
  Hugging Face, which is blocked here. Per-model validation perplexities are available in the DataDecide
  GitHub repo, and I used them for training health.

## Criteria (fixed before looking at per-model values except the E01 summary already in the docs; justified scientifically)

| ID | Criterion | Threshold | Kind | Why |
|---|---|---|---|---|
| H1 | Every magnitude group estimable | no group failed (no slot used all 100 candidates), in E01 (4 groups x 6 seeds) and E06 (10 bands x 3 seeds) | measurement validity, hard | If a group cannot be filled, its group mean is undefined and β cannot be computed on the same footing as for other models. This is the protocol's own pre-set failure rule. |
| H2 | The filter does not distort which numbers are measured | within each band, the mean log10 target of the accepted prompts differs from the uniform-band expectation by < 0.10 group spacing (1/3 decade) | measurement validity, hard | Rejected slots are redrawn from the same band. If rejection concentrated on part of a band (e.g. 997-999), the surviving targets would shift and bias group means, and so the gap ratios. 0.1 spacing is about 3x the sampling floor (0.021, the value seen in models with 0 rejections). |
| H3 | A number line exists | ρ ≥ 0.80 at the selected layer (E01 frozen eval and E06) | measurement validity, hard | β is a summary of PC1 spacing. If PC1 is not ordered by magnitude, β is meaningless. |
| H4 | Fully trained, intended checkpoint | revision `step69369-seed-default` (100B tokens) for both E01 and E06; final prompt token is `=` in 160/160 prompts; same tokenizer class | measurement validity, hard | Comparability needs the same training length and the same probe position. |
| H5 | Training health | the released (default-seed) run's final validation perplexity is within about 10% of its two auxiliary seed replicates (mean log-ppl deviation < 0.10), and in-domain perplexity (c4_en) falls steadily over the second half of training | measurement validity / health, hard | A diverged or corrupted run would stand out against replicates trained on the same data. Comparing to replicates of the *same recipe* removes the data effect, so a failure here is not a data property. |
| S1 | Copy competence | soft flag if rejection is ≥ 50% in any band (E01 group or E06 band) | **behaviour, soft** | When most attempts fail, the model's typical behaviour at that magnitude is failure. The measured line is then conditional on a minority of successful prompts. This is worth a sensitivity check, but it is an outcome plausibly caused by the training data (see caveat). |

**Data property versus measurement validity.** H1-H5 are measurement-validity criteria, and failing one
would justify exclusion. S1 is not. Copy failure is a downstream behaviour of the model, and the training
data is the very variable under study. Excluding models because of it would select on an outcome of the
treatment (post-treatment selection). That can create or erase a data-to-geometry relationship. A model
that fails S1 but passes H2 still has a measured line on an undistorted target set, so it stays comparable.

## Per-model table (all 25)

| # | Recipe | E01 rej. (sel / eval) | E01 worst group (eval) | E06 rej. | E06 worst band | Failed slots | rho E01 / E06 | EV E01 / E06 | Filter shift (max, group spacings) | Final ppl c4 / pile | Default vs aux seeds (mean log-ppl dev) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | dolma1_7 | 0.0% / 0.0% | 0% (1-39) | 0.0% | 0% (~215) | 0+0 | 0.933 / 0.983 | 0.37 / 0.38 | 0.021 | 14.9 / 8.3 | +0.001 | PASS |
| 2 | dolma1_7-no-code | 0.1% / 0.0% | 0% (1-39) | 0.1% | 0% (~100) | 0+0 | 0.935 / 0.947 | 0.41 / 0.40 | 0.021 | 14.6 / 8.9 | +0.001 | PASS |
| 3 | dolma1_7-no-math-code | 0.2% / 0.2% | 1% (1-39) | 0.4% | 1% (~100) | 0+0 | 0.945 / 0.920 | 0.34 / 0.35 | 0.021 | 14.5 / 9.5 | +0.001 | PASS |
| 4 | dolma1_7-no-reddit | 0.0% / 0.0% | 0% (1-39) | 0.0% | 0% (~2154) | 0+0 | 0.912 / 0.931 | 0.52 / 0.51 | 0.021 | 14.9 / 8.3 | -0.001 | PASS |
| 5 | dolma1_7-no-flan | 0.0% / 0.0% | 0% (1-39) | 0.0% | 0% (~10) | 0+0 | 0.931 / 0.937 | 0.47 / 0.44 | 0.021 | 14.9 / 8.3 | +0.000 | PASS |
| 6 | dolma1_6plus | 0.0% / 0.0% | 0% (1-39) | 0.0% | 0% (~10) | 0+0 | 0.923 / 0.934 | 0.43 / 0.41 | 0.021 | 14.1 / 8.5 | +0.003 | PASS |
| 7 | c4 | 5.9% / 6.4% | 15% (980-1019) | 5.0% | 12% (~215) | 0+0 | 0.929 / 0.926 | 0.27 / 0.28 | 0.024 | 13.9 / 22.5 | -0.068 | PASS (copy-rejection 10-25% in some band; note) |
| 8 | fineweb-pro | 20.5% / 20.8% | 42% (980-1019) | 25.5% | 64% (~464) | 0+0 | 0.925 / 0.927 | 0.30 / 0.29 | 0.031 | 16.7 / 17.2 | -0.027 | PASS; SOFT FLAG (majority failure in one band; data property) |
| 9 | fineweb-edu | 5.2% / 4.2% | 9% (980-1019) | 5.0% | 17% (~464) | 0+0 | 0.957 / 0.957 | 0.29 / 0.28 | 0.020 | 17.8 / 15.2 | -0.013 | PASS (copy-rejection 10-25% in some band; note) |
| 10 | falcon | 1.2% / 0.8% | 3% (980-1019) | 0.3% | 1% (~10) | 0+0 | 0.924 / 0.936 | 0.38 / 0.38 | 0.020 | 14.9 / 12.4 | -0.022 | PASS |
| 11 | falcon-and-cc | 1.2% / 1.0% | 2% (980-1019) | 0.4% | 1% (~100) | 0+0 | 0.933 / 0.928 | 0.33 / 0.32 | 0.020 | 14.6 / 10.7 | -0.004 | PASS |
| 12 | falcon-and-cc-qc-10p | 4.0% / 5.0% | 9% (980-1019) | 1.9% | 5% (~1000) | 0+0 | 0.922 / 0.968 | 0.31 / 0.33 | 0.020 | 16.9 / 11.1 | -0.004 | PASS |
| 13 | falcon-and-cc-qc-20p | 2.5% / 2.0% | 3% (9980-10019) | 1.7% | 4% (~464) | 0+0 | 0.932 / 0.944 | 0.35 / 0.35 | 0.021 | 16.0 / 10.6 | -0.012 | PASS |
| 14 | falcon-and-cc-qc-orig-10p | 1.1% / 1.6% | 4% (9980-10019) | 0.6% | 2% (~46) | 0+0 | 0.938 / 0.970 | 0.36 / 0.38 | 0.021 | 16.5 / 10.9 | +0.006 | PASS |
| 15 | falcon-and-cc-qc-tulu-10p | 0.7% / 0.8% | 2% (980-1019) | 0.9% | 3% (~464) | 0+0 | 0.933 / 0.913 | 0.34 / 0.38 | 0.021 | 16.1 / 9.9 | -0.004 | PASS |
| 16 | dclm-baseline | 7.9% / 8.4% | 17% (980-1019) | 3.2% | 10% (~1000) | 0+0 | 0.899 / 0.918 | 0.35 / 0.35 | 0.020 | 16.5 / 10.1 | +0.001 | PASS (copy-rejection 10-25% in some band; note) |
| 17 | dclm-baseline-qc-7p-fw2 | 3.2% / 3.4% | 10% (9980-10019) | 0.6% | 2% (~46) | 0+0 | 0.951 / 0.949 | 0.38 / 0.38 | 0.021 | 18.7 / 10.4 | +0.001 | PASS |
| 18 | dclm-baseline-qc-7p-fw3 | 8.3% / 8.2% | 19% (980-1019) | 7.7% | 27% (~464) | 0+0 | 0.925 / 0.946 | 0.26 / 0.25 | 0.022 | 22.6 / 12.4 | -0.004 | PASS (copy-rejection 10-25% in some band; note) |
| 19 | dclm-baseline-qc-fw-3p | 11.8% / 12.2% | 21% (980-1019) | 11.9% | 20% (~215) | 0+0 | 0.931 / 0.938 | 0.22 / 0.22 | 0.026 | 26.1 / 14.5 | +0.006 | PASS (copy-rejection 10-25% in some band; note) |
| 20 | dclm-baseline-qc-fw-10p | 8.6% / 10.3% | 21% (980-1019) | 4.6% | 15% (~464) | 0+0 | 0.939 / 0.959 | 0.27 / 0.28 | 0.028 | 20.6 / 11.3 | +0.004 | PASS (copy-rejection 10-25% in some band; note) |
| 21 | dclm-baseline-qc-10p | 5.8% / 7.2% | 18% (980-1019) | 3.7% | 12% (~464) | 0+0 | 0.932 / 0.944 | 0.32 / 0.31 | 0.021 | 19.4 / 11.1 | -0.005 | PASS (copy-rejection 10-25% in some band; note) |
| 22 | dclm-baseline-qc-20p | 2.2% / 2.2% | 5% (980-1019) | 1.3% | 4% (~464) | 0+0 | 0.951 / 0.955 | 0.31 / 0.30 | 0.021 | 18.1 / 10.4 | -0.004 | PASS |
| 23 | dclm-baseline-25p-dolma1.7-75p | 0.0% / 0.0% | 0% (1-39) | 0.0% | 0% (~215) | 0+0 | 0.934 / 0.942 | 0.47 / 0.46 | 0.021 | 15.0 / 8.4 | +0.000 | PASS |
| 24 | dclm-baseline-50p-dolma1.7-50p | 0.0% / 0.0% | 0% (1-39) | 0.0% | 0% (~10) | 0+0 | 0.953 / 0.954 | 0.39 / 0.38 | 0.021 | 15.3 / 8.6 | -0.001 | PASS |
| 25 | dclm-baseline-75p-dolma1.7-25p | 0.0% / 0.0% | 0% (1-39) | 0.0% | 0% (~10) | 0+0 | 0.943 / 0.954 | 0.38 / 0.37 | 0.021 | 15.6 / 8.9 | -0.002 | PASS |

Column notes:
- Rejection = 1 - accepted / candidates tried, pooled.
  - E01 "sel" is `filter_summary`, the selection seeds 42-44. E01 "eval" is pooled over the frozen-layer eval runs, seeds 45-47.
  - E01 groups are 1-39, 80-119, 980-1019 and 9980-10019. E06 bands are ±10% around 10, 22, 46, 100, 215, 464, 1000, 2154, 4642 and 10000.
- Failed slots: E01 + E06. Zero everywhere.
- Filter shift: the largest |mean log10 accepted target - uniform-band expectation| over 10 bands x 3 seeds, in units of a 1/3-decade spacing.
  - The floor is 0.021. Models with zero rejections reach it because their targets are just uniform draws: same RNG seeds, identical prompts.
  - fineweb-pro's worst value is 0.031, so the filter barely moves the measured targets.
- Perplexity comes from the DataDecide GitHub repo, `perplexity_metrics_by_group.csv`, at commit 68abc49 (2025-06-17).
  - Released default runs map to `<recipe>-1B-5xC-2`. Dolma1.7 is `baseline-1B-5xC-2`.
  - Four DCLM-QC recipes are `-1B-5xC` with no suffix: dclm-baseline-qc-fw-3p, qc-fw-10p, qc-10p and qc-20p. The mapping is from `checkpoints/final_defualt_seed_paths.jsonl` and `release/make_final_checkpoint_script.py`.
  - Aux seeds are `-5xC-4` and `-5xC-5`, at their final step 69369.
  - The deviation column is the mean over 11 validation sets of log(ppl_default) - mean log(ppl_aux). Negative means the released run is *better* than its replicates.

## Evidence

### 1. Copying competence (correct-output filter)
- **Dolma-containing recipes (6 Dolma variants + 3 DCLM/Dolma mixes) copy almost perfectly:** 0-0.4% rejection
  in both runs. This includes dolma1_7-no-math-code at 0.2-0.4%. So the near-zero rate is not only due to the
  code and math subsets.
- **Web-only recipes reject more, and mostly in the middle of the range.** Rejection peaks for 3-digit and
  ~1000 targets (E06 bands 215-1000, E01 group 980-1019). It is low at 10-46 and at 10000.
- In the stored examples, the dominant failure is **copying a context number, or mixing its digits**, after the
  4-digit context items. Examples: `...,6248=6248,477= -> 6247`, `...,7402=7402,993= -> 7993` and
  `...,997= -> 9997`. Across the stored examples (5/group/run E01, 3/band/seed E06):
  - fineweb-pro: 57/138 echo a context number and 25/138 are digit superstrings of the target.
  - dclm-qc-fw-3p: 61/138 echo a context number.
  - This is an in-context copy/induction failure for a shorter number after longer ones. It is not garbage output: the output is always a well-formed integer.
- E01 and E06 agree, with independent prompts and different band layouts. fineweb-pro is worst in both (21%/26% overall),
  followed by dclm-qc-fw-3p (12%/12%). The behaviour is a stable property of each checkpoint.
- **Reference point, from Paloma 1B baselines** in `results/beta_fine/step35000-unsharded`, same E06 protocol:
  - The Paloma C4 model rejects **31%** overall (54% in its worst band), against 5% for DataDecide-c4.
  - Paloma Falcon-RefinedWeb rejects 14%, against 0.3% for DataDecide-falcon.
  - Paloma Dolma, Pile, mC4 and RedPajama reject 0-1.4%.
  - So rejection of 20-30% happens in normally trained public 1B models. It is also **not purely a property of the corpus**: the same corpus under a different training setup (Paloma trains ~150B tokens, step35000) gives a very different rate. This weakens the case that S1 reflects data alone. It is partly run-specific, and that cannot be separated without seed replicates (see recommendations).

### 2. Is the measured line still comparable? (H2, H3)
- **H2: no failure in any model.** Even for fineweb-pro, with 64% rejection around 464, the accepted targets'
  mean log value moves by at most 0.031 spacing (sampling floor 0.021). Rejections are spread across each band,
  not concentrated at its edges, so the group means are not shifted by the filter.
- **H3: ρ = 0.899-0.957** in E01 frozen evaluation and **0.913-0.983** in E06. The lowest is dclm-baseline in E01, at 0.899.
  EV ranges from 0.22 (dclm-qc-fw-3p) to 0.52. A clear magnitude-ordered PC1 is present in all 25.
- **H4: passes for all 25.** The revision is `step69369-seed-default` in both E01 and E06. `final_token_is_equals` is true, with 160/160 prompts ending on the `=` token.
  The tokenizer is OLMoTokenizerFast everywhere, and the target-token-length histograms are near-identical (1-token 100-107 of 160).
  The dtype is bf16 everywhere.

### 3. Training health (H5)
- All 25 released runs have the full 28 or 27 evaluation points up to step 69369 (100B tokens).
- **Released runs versus their own seed replicates:**
  - Mean log-ppl deviation is within **[-0.068, +0.006]**. All but c4 are within ±0.03.
  - No released run is worse than its replicates by more than 0.6% on average.
  - c4's -0.068 means its released run is *better* than its aux seeds. The gap comes from the noisy out-of-domain sets (ICE, Dolma-Stack code).
    For example, c4 Stack ppl is 36.9 against 39.8 and 65.6 for the aux seeds, while its in-domain c4_en ppl is 13.90 against 13.86 and 13.90.
- **In-domain perplexity falls smoothly to the end** for every recipe; c4 is shown as an example:
  - c4_en 16.5 → 13.9 over steps 30k-69k for c4.
  - Late "increases" appear only on noisy out-of-domain Pile/Stack evaluations, at most +4.7% between adjacent checkpoints, and they recover.
  - No divergence, loss spike or plateau consistent with a broken run.
- Absolute perplexity differs strongly by recipe. The most aggressive quality filter, dclm-qc-fw-3p, has c4 ppl 26.1 and Pile 14.5, against 13.9-17.8 for most others.
  This is expected for a narrower data distribution and is a **data property**. The run is healthy against its own replicates, so it is not under-training.

### 4. Number-comparison and downstream evidence
- `paper/appendix_number_comparison.tex` reports number-comparison accuracy for **14 public models only**: Falcon, RPJ, OLMo-7B, StarCoderBase, Cerebras-GPT, GPT-Neo and OPT.
  There are **no DataDecide models**. The code exists (`src/number_comparison*.py`, `modal_app/number_comparison_app.py`), but it has not been run on DataDecide.
  E00 results lived in an MLflow store that is not in the repo.
- The DataDecide GitHub repo has no per-model OLMES accuracy. The README points to HF datasets `allenai/DataDecide-eval-results` and `-eval-instances`, which are blocked here (proxy 403).
  Its notebooks are Git LFS pointers. I fetched six from media.githubusercontent.com. They contain only **averages across all recipes**.
  At 1B, the mean GSM8K is 0.025 and the mean Minerva-math is 0.014 (`viz/math_and_code.ipynb`). So generative math benchmarks are at floor for every 1B model and would not separate them.
- So the only per-model numeracy evidence available is the copy filter (sections 1-2). The only per-model health evidence is validation perplexity (section 3).

## Flagged models
| Model | Flag | Type | Exclude? |
|---|---|---|---|
| fineweb-pro | S1: ≥ 50% rejection in one band (E06 ~464: 64%; ~215: 38%; E01 980-1019: 42%); 21-26% overall | behaviour; plausibly caused by training data (ProX-refined FineWeb), partly run-specific | **No.** Passes H1-H5, and the target set is undistorted (H2). Optional pre-declared sensitivity analysis only, always reported with the full n = 25 result. |
| c4, fineweb-edu, dclm-baseline, dclm-qc-7p-fw3, dclm-qc-fw-3p, dclm-qc-fw-10p, dclm-qc-10p | note: worst-band rejection 10-27% | behaviour / data | No |
| none | H1-H5 | measurement validity | none fails |

## Honest notes and limits
- I am auditing a single checkpoint per recipe. I cannot tell how much of the copy-rejection differences come from data and how much from training randomness. The Paloma-C4 comparison (31% against 5% on the same corpus) suggests the run-specific part is large.
- The stored rejection examples are capped (5 per group per run in E01, 3 per band per seed in E06), so the failure-type counts are indicative only.
- H2 checks only the *target* distribution after filtering. It cannot rule out that successful prompts in a high-rejection model have systematically different hidden states from the failed ones, which are never measured. I did not test whether rejection relates to β. That would be a data-mediated relationship, and it is not a sound a-priori exclusion criterion.
- Perplexity was checked only against the model's own seed replicates and its curve shape. There is no absolute "healthy" threshold for a 1B model on 100B tokens other than the replicate comparison.
- The DataDecide repo clone at `/tmp/claude-0/agentrefs/DataDecide` already existed (another auditor). I treated it as data and read it with `python3 -I`.

## Recommendations (no exclusions)
1. Keep all 25. Report fineweb-pro's high copy-rejection openly. If a robustness check drops it, declare that check in advance and show it next to the full result.
2. To separate data from run effects in numeracy, rerun the E06 filter (it is cheap) on the aux-seed 1B checkpoints (`-5xC-4`, `-5xC-5`). This is also handover item 2.
3. If a behavioural numeracy score is wanted, run the existing `src/number_comparison.py` benchmark on the 25 models, as a descriptive covariate and not as an exclusion criterion.

Files:
- per-model metrics: `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/07_work/filter_stats.json`
- perplexities: `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/07_work/ppl.json`
- scripts: `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/07_work/filter_stats.py`, `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/07_work/ppl.py`
