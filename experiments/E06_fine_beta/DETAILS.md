# E06 details

## Per-model results (`results/beta_fine/fine_vs_data.csv`)

"err" = RMS residual / mean gap of the 9-gap fit. β_cont R² is over all 3,000 prompts.

| Recipe | Family | Layer (new / old) | β old (E01) | β_coarse | β_cont (R²) | β_fine (err) | β_cont at old layer | seed s.d. (β_cont) |
|---|---|---|---|---|---|---|---|---|
| dolma1_7 | dolma | 4 / 6 | 0.978 | 0.847 | 0.831 (0.97) | 0.877 (39%) | 0.916 | 0.011 |
| dolma1_7-no-code | dolma | 8 / 8 | 0.856 | 0.826 | 0.773 (0.90) | 0.859 (68%) | 0.773 | 0.017 |
| dolma1_7-no-math-code | dolma | 4 / 4 | 0.618 | 0.632 | 0.552 (0.89) | 0.718 (79%) | 0.552 | 0.013 |
| dolma1_7-no-reddit | dolma | 8 / 8 | 1.360 | 1.130 | 1.176 (0.87) | 1.063 (50%) | 1.176 | 0.018 |
| dolma1_7-no-flan | dolma | 9 / 9 | 1.197 | 1.120 | 1.191 (0.89) | 1.102 (65%) | 1.191 | 0.022 |
| dolma1_6plus | dolma | 9 / 9 | 1.077 | 1.006 | 1.008 (0.87) | 0.933 (55%) | 1.008 | 0.019 |
| c4 | c4 | 3 / 7 | 0.620 | 0.650 | 0.596 (0.90) | 0.757 (89%) | 0.566 | 0.013 |
| fineweb-pro | fineweb | 7 / 7 | 0.662 | 0.634 | 0.577 (0.87) | 0.712 (98%) | 0.577 | 0.020 |
| fineweb-edu | fineweb | 9 / 9 | 0.895 | 0.897 | 0.872 (0.92) | 0.888 (54%) | 0.872 | 0.017 |
| falcon | falcon | 8 / 9 | 0.792 | 0.749 | 0.672 (0.89) | 0.749 (76%) | 0.678 | 0.010 |
| falcon-and-cc | falcon | 9 / 9 | 0.649 | 0.694 | 0.626 (0.87) | 0.743 (81%) | 0.626 | 0.027 |
| falcon-and-cc-qc-10p | falcon | 3 / 8 | 0.713 | 0.836 | 0.876 (0.95) | 0.896 (50%) | 0.601 | 0.006 |
| falcon-and-cc-qc-20p | falcon | 8 / 8 | 0.739 | 0.718 | 0.678 (0.88) | 0.770 (80%) | 0.678 | 0.020 |
| falcon-and-cc-qc-orig-10p | falcon | 3 / 8 | 0.771 | 0.773 | 0.701 (0.94) | 0.793 (57%) | 0.659 | 0.009 |
| falcon-and-cc-qc-tulu-10p | falcon | 2 / 9 | 0.948 | 1.030 | 1.380 (0.81) | 1.268 (102%) | 0.810 | 0.002 |
| dclm-baseline | dclm | 8 / 8 | 0.666 | 0.606 | 0.584 (0.87) | 0.675 (71%) | 0.584 | 0.016 |
| dclm-baseline-qc-7p-fw2 | dclm | 8 / 8 | 0.838 | 0.753 | 0.669 (0.90) | 0.724 (77%) | 0.669 | 0.009 |
| dclm-baseline-qc-7p-fw3 | dclm | 3 / 9 | 0.750 | 0.790 | 0.806 (0.92) | 0.935 (56%) | 0.605 | 0.017 |
| dclm-baseline-qc-fw-3p | dclm | 2 / 9 | 0.784 | 0.932 | 1.065 (0.87) | 1.046 (70%) | 0.570 | 0.053 |
| dclm-baseline-qc-fw-10p | dclm | 3 / 3 | 0.809 | 0.766 | 0.756 (0.94) | 0.809 (54%) | 0.756 | 0.008 |
| dclm-baseline-qc-10p | dclm | 8 / 8 | 0.717 | 0.668 | 0.657 (0.89) | 0.744 (67%) | 0.657 | 0.017 |
| dclm-baseline-qc-20p | dclm | 9 / 9 | 0.760 | 0.749 | 0.717 (0.91) | 0.808 (63%) | 0.717 | 0.011 |
| dclm-baseline-25p-dolma1.7-75p | mix | 8 / 8 | 1.108 | 1.027 | 1.101 (0.89) | 1.048 (55%) | 1.101 | 0.027 |
| dclm-baseline-50p-dolma1.7-50p | mix | 8 / 8 | 0.952 | 0.823 | 0.831 (0.91) | 0.864 (49%) | 0.831 | 0.018 |
| dclm-baseline-75p-dolma1.7-25p | mix | 10 / 10 | 0.841 | 0.785 | 0.762 (0.92) | 0.823 (46%) | 0.762 | 0.018 |

Summary: median seed s.d. coarse 0.015, cont 0.017, fine 0.012; median err coarse 0.25,
fine 0.65; median R² (cont) 0.89; same layer as E01 for 17/25 models.

## Correlation between β versions (Pearson r, n = 25)

| | old_direct | beta_coarse | beta_cont | beta_fine | coarse at old layer | cont at old layer |
|---|---|---|---|---|---|---|
| old_direct (E01) | 1 | 0.91 | 0.79 | 0.70 | **0.96** | **0.96** |
| beta_coarse | | 1 | 0.95 | 0.90 | 0.87 | 0.85 |
| beta_cont | | | 1 | 0.98 | 0.73 | 0.71 |

## Layer near-ties: the 8 models whose selected layer changed

| Model | New layer: score | Old layer: score | Score difference | β_cont new / old layer |
|---|---|---|---|---|
| dclm-baseline-qc-fw-3p | L2: 0.453 | L9: 0.450 | 0.7 % | 1.065 / 0.570 |
| falcon | L8: 0.598 | L9: 0.593 | 0.8 % | 0.672 / 0.678 |
| dclm-baseline-qc-7p-fw3 | L3: 0.491 | L9: 0.476 | 2.9 % | 0.806 / 0.605 |
| falcon-and-cc-qc-tulu-10p | L2: 0.587 | L9: 0.561 | 4.4 % | 1.380 / 0.810 |
| falcon-and-cc-qc-orig-10p | L3: 0.607 | L8: 0.578 | 4.8 % | 0.701 / 0.659 |
| falcon-and-cc-qc-10p | L3: 0.568 | L8: 0.530 | 6.7 % | 0.876 / 0.601 |
| c4 | L3: 0.510 | L7: 0.475 | 6.9 % | 0.596 / 0.566 |
| dolma1_7 | L4: 0.614 | L6: 0.566 | 7.7 % | 0.831 / 0.916 |

Score = median over seeds of √(EV·|ρ|).

## α vs β, by β version (`python -m src.beta_fine_analysis`)

| β version | vs α_OLS r (p) | within family r (p) | vs log numbers seen r (p) |
|---|---|---|---|
| old (E01, paper) | +0.56 (0.003) | +0.28 (0.17) | +0.45 (0.03) |
| cont, old layer | **+0.63 (0.001)** | **+0.40 (0.049)** | +0.41 (0.04) |
| coarse, old layer | +0.63 (0.001) | +0.37 (0.07) | +0.41 (0.04) |
| coarse, new layer | +0.37 (0.07) | +0.04 | +0.39 (0.05) |
| cont, new layer | +0.20 (0.33) | −0.09 | +0.36 (0.08) |

**Fairness note.** The layer rule was fixed in advance. The old-layer numbers are a
robustness check, not a replacement: picking whichever layer gives the best correlation
would be cherry-picking. The honest summary is that the α–β link depends on comparing
models at a consistent depth.

## Problems met during the run, and fixes
- **Disk:** 25 fp32 checkpoints (~4.7 GB each) don't fit on an 80 GB container disk, so
  each model's weights are deleted after it finishes and uploads (`f8d5c0a`).
- **Download:** `snapshot_download` also fetched the 10 GB optimizer state
  (`training/optim.pt`). The prefetch now ignores `training/*`, `*.pt` and `*.bin` (`3a47683`).
- **Template conflict:** the RunPod PyTorch 2.8 image ships torchvision built for torch 2.8.
  Seen through `--system-site-packages`, it broke `import transformers` under our torch
  2.6 ("operator torchvision::nms does not exist"). `setup.sh gpu` now pins
  `torchvision==0.21.0 torchaudio==2.6.0` and checks the OLMo import (`f1f20bc`).
- **R² near β = 1:** found while writing the tests; `err` replaces R² as the fit-quality
  measure.

## Free follow-ups (no GPU needed)
The `.npz` files keep the top-5 PCA scores of every prompt at every layer, so these can be
recomputed locally: β at one fixed layer for all models, other groupings, and
bootstrap confidence intervals.
