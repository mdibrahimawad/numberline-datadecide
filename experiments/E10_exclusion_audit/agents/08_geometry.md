# Auditor 08: number-line geometry (is a single PC1 axis a valid basis for beta?)

Data: E06 per-prompt top-5 PC scores (`results/beta_fine/step69369-seed-default/<recipe>.npz`), 3 seeds x 1000 prompts,
10 magnitude groups. Evaluated at each model's E01 selected layer (`alpha_beta.csv: selected_layer`, PRIMARY) and at layer 8.
All statistics are per seed, then the median over the 3 seeds. The alpha-beta correlation was not computed or looked at,
and the correlations/confounders json files were not opened.
Sanity check: my PC1 coarse beta (groups 10/100/1000/10000, the paper's direct fit) matches E06 `beta_coarse_oldlayer` to within 0.02
for all 25 models (median over seeds compared with the mean over seeds).

## A-priori rule (written to `08_geometry_work/00_prereg.md` before any geometry number was computed)
A model's single-axis beta is **not valid** at a layer if ANY of these hold:
- **C1 curvature**: |beta_arc5 - beta_pc1| > 0.10. beta_arc5 is the same direct fit applied to the Euclidean distances between consecutive coarse
  group means in PC1..PC5, with the sampling-noise bias tr(Sigma_a)/n_a + tr(Sigma_b)/n_b subtracted. If the line is straight but tilted,
  projecting it onto PC1 scales every gap by the same factor and leaves beta unchanged. Only bending changes beta, so this test
  measures what the PC1 projection actually does to beta.
- **C2 digit-count staircase**: the pooled within-digit-class slope of PC1 on log10 n (classes of 2, 3 and 4 digits, each with its own
  intercept), divided by the overall slope, is < 0.5. Below that, more than half of the PC1 rise comes from jumps at digit-count boundaries
  and not from a magnitude code.
- **C3 weak or unstable magnitude axis**: eta^2 of PC1 on the 10 groups < 0.5, OR the seed s.d. of beta_pc1 > 0.05 (3x the E06 median of 0.015), OR
  a coarse PC1 gap has the wrong sign in >= 2 of 3 seeds. The fit takes |gaps|, which hides a wrong-sign gap.
- **C4 outliers / skewed groups**: |beta_pc1 from group medians - beta_pc1 from group means| > 0.10.
- **C5 weak PC1**: EV1 < 0.5 x the median EV1 of the 25 models at the same layer.
- The threshold of 0.10 in beta is about 7x the seed s.d. and about 1/8 of the range between models.
- Reported but not used for flags: |rho(n, PC2)|, R2(log n ~ PC1) / R2(log n ~ PC1..5), the between-group variance that lies off the line,
  the PC1-PC2 arc, the digit-boundary step and EV1/EV2.

## Result
**Flagged at the E01 selected layer (primary, 4/25):**
- `dolma1_7-no-reddit` (C1: beta_pc1 1.132 vs beta_arc5 1.020)
- `dolma1_7-no-flan` (C1: 1.131 vs 0.998)
- `fineweb-pro` (C2: within-digit slope ratio 0.488)
- `falcon-and-cc` (C2: 0.466; C4: median-based beta 0.775 vs mean-based 0.671)

**Flagged at layer 8 (secondary, 4/25):**
- `dolma1_7-no-reddit` (C1)
- `dolma1_6plus` (C1: 1.063 vs 0.959)
- `falcon-and-cc` (C2 0.461; C3: seed s.d. 0.051)
- `dclm-baseline-qc-7p-fw3` (C5: EV1 0.153 = 0.49 x median)

**Borderline (within 10 % of a threshold):**
- At the selected layer: dolma1_7 (dC1 0.097), dolma1_6plus (0.095), dclm-baseline-qc-fw-10p (0.099, layer 3), falcon (slope ratio 0.51), c4 (seed s.d. 0.045).
- How many models are flagged depends heavily on the thresholds: making all thresholds 20 % stricter flags 18/25 at the selected layer and 12/25 at layer 8, mostly through C2.
  Making them 25 % more lenient flags only dolma1_7-no-flan.

## Per-model table (all 25)
dC1 = |beta_arc5 - beta_pc1|. The "off/PC1 gap" column gives the noise-corrected size of the group-mean step in PC2..PC5, divided by the step along PC1,
for the 10->100 and 1000->10000 steps.

| recipe | E01 layer | EV1 | EV1/EV2 | beta_pc1 | beta_arc5 | dC1 | within-digit slope ratio | |beta_med-beta| | seed sd | eta2 | rho(n,PC2) | off/PC1 gap 10-100, 1000-10000 | outliers % | flags (sel) | L8: beta_pc1 / arc5 / slope ratio / EV1 | flags (L8) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dolma1_7 | 6 | 0.334 | 4.9 | 0.931 | 0.834 | 0.097 | 0.66 | 0.001 | 0.016 | 0.94 | 0.01 | 1.29, 0.88 | 0.1 | - | 0.868 / 0.826 / 0.59 / 0.308 | - | ok |
| dolma1_7-no-code | 8 | 0.405 | 7.5 | 0.835 | 0.829 | 0.006 | 0.61 | 0.024 | 0.018 | 0.93 | 0.12 | 0.69, 0.84 | 0.5 | - | 0.835 / 0.829 / 0.61 / 0.405 | - | ok |
| dolma1_7-no-math-code | 4 | 0.351 | 3.9 | 0.629 | 0.647 | 0.017 | 0.60 | 0.013 | 0.017 | 0.94 | 0.34 | 0.83, 1.28 | 0.0 | - | 0.703 / 0.730 / 0.74 / 0.304 | - | ok |
| dolma1_7-no-reddit | 8 | 0.511 | 10.6 | 1.132 | 1.020 | 0.112 | 0.67 | 0.042 | 0.015 | 0.88 | 0.22 | 0.98, 0.44 | 0.5 | C1 | 1.132 / 1.020 / 0.67 / 0.511 | C1 | NOT VALID (sel) |
| dolma1_7-no-flan | 9 | 0.443 | 7.3 | 1.131 | 0.998 | 0.133 | 0.63 | 0.006 | 0.026 | 0.91 | 0.10 | 1.45, 0.57 | 0.5 | C1 | 0.986 / 0.928 / 0.76 / 0.387 | - | NOT VALID (sel) |
| dolma1_6plus | 9 | 0.409 | 7.6 | 1.002 | 0.907 | 0.095 | 0.62 | 0.012 | 0.011 | 0.89 | 0.14 | 1.05, 0.52 | 0.2 | - | 1.063 / 0.959 / 0.75 / 0.323 | C1 | NOT VALID at L8 only |
| c4 | 7 | 0.252 | 3.2 | 0.673 | 0.647 | 0.026 | 0.72 | 0.013 | 0.045 | 0.93 | 0.39 | 0.84, 0.67 | 0.1 | - | 0.664 / 0.695 / 0.68 / 0.206 | - | ok |
| fineweb-pro | 7 | 0.292 | 4.2 | 0.651 | 0.670 | 0.019 | 0.49 | 0.081 | 0.043 | 0.93 | 0.39 | 0.75, 1.40 | 0.2 | C2 | 0.656 / 0.700 / 0.59 / 0.255 | - | NOT VALID (sel) |
| fineweb-edu | 9 | 0.278 | 4.2 | 0.885 | 0.857 | 0.028 | 0.57 | 0.022 | 0.021 | 0.94 | 0.01 | 0.75, 0.72 | 0.1 | - | 0.761 / 0.807 / 0.68 / 0.217 | - | ok |
| falcon | 9 | 0.380 | 7.8 | 0.755 | 0.752 | 0.002 | 0.51 | 0.011 | 0.017 | 0.91 | 0.16 | 0.61, 0.86 | 0.2 | - | 0.743 / 0.756 / 0.55 / 0.382 | - | ok |
| falcon-and-cc | 9 | 0.315 | 5.3 | 0.671 | 0.702 | 0.032 | 0.47 | 0.104 | 0.044 | 0.90 | 0.28 | 0.60, 0.96 | 0.0 | C2,C4 | 0.682 / 0.692 / 0.46 / 0.286 | C2,C3 | NOT VALID (sel) |
| falcon-and-cc-qc-10p | 8 | 0.312 | 4.3 | 0.664 | 0.689 | 0.025 | 0.63 | 0.013 | 0.022 | 0.91 | 0.36 | 0.63, 1.04 | 0.2 | - | 0.664 / 0.689 / 0.63 / 0.312 | - | ok |
| falcon-and-cc-qc-20p | 8 | 0.359 | 5.3 | 0.711 | 0.738 | 0.027 | 0.62 | 0.003 | 0.021 | 0.92 | 0.24 | 0.62, 1.07 | 0.2 | - | 0.711 / 0.738 / 0.62 / 0.359 | - | ok |
| falcon-and-cc-qc-orig-10p | 8 | 0.358 | 6.4 | 0.730 | 0.737 | 0.007 | 0.68 | 0.017 | 0.016 | 0.92 | 0.24 | 0.62, 0.93 | 0.1 | - | 0.730 / 0.737 / 0.68 / 0.358 | - | ok |
| falcon-and-cc-qc-tulu-10p | 9 | 0.332 | 5.7 | 0.871 | 0.857 | 0.014 | 0.57 | 0.001 | 0.012 | 0.91 | 0.10 | 0.58, 0.59 | 0.5 | - | 0.722 / 0.754 / 0.65 / 0.237 | - | ok |
| dclm-baseline | 8 | 0.353 | 5.4 | 0.616 | 0.675 | 0.059 | 0.60 | 0.001 | 0.019 | 0.90 | 0.27 | 0.48, 2.09 | 0.1 | - | 0.616 / 0.675 / 0.60 / 0.353 | - | ok |
| dclm-baseline-qc-7p-fw2 | 8 | 0.382 | 5.3 | 0.750 | 0.756 | 0.006 | 0.65 | 0.003 | 0.012 | 0.94 | 0.27 | 0.66, 1.00 | 0.3 | - | 0.750 / 0.756 / 0.65 / 0.382 | - | ok |
| dclm-baseline-qc-7p-fw3 | 9 | 0.254 | 4.6 | 0.658 | 0.700 | 0.041 | 0.55 | 0.002 | 0.009 | 0.87 | 0.27 | 0.68, 1.49 | 0.0 | - | 0.673 / 0.699 / 0.79 / 0.153 | C5 | NOT VALID at L8 only |
| dclm-baseline-qc-fw-3p | 9 | 0.218 | 2.9 | 0.651 | 0.697 | 0.046 | 0.72 | 0.007 | 0.008 | 0.92 | 0.27 | 0.61, 1.41 | 0.1 | - | 0.633 / 0.685 / 0.90 / 0.171 | - | ok |
| dclm-baseline-qc-fw-10p | 3 | 0.282 | 3.4 | 0.773 | 0.674 | 0.099 | 0.73 | 0.007 | 0.014 | 0.96 | 0.11 | 1.22, 0.89 | 0.0 | - | 0.693 / 0.713 / 0.82 / 0.216 | - | ok |
| dclm-baseline-qc-10p | 8 | 0.311 | 5.0 | 0.669 | 0.686 | 0.017 | 0.71 | 0.004 | 0.003 | 0.92 | 0.19 | 0.59, 1.02 | 0.2 | - | 0.669 / 0.686 / 0.71 / 0.311 | - | ok |
| dclm-baseline-qc-20p | 9 | 0.302 | 3.9 | 0.746 | 0.736 | 0.010 | 0.62 | 0.019 | 0.010 | 0.93 | 0.15 | 0.67, 0.74 | 0.2 | - | 0.606 / 0.666 / 0.79 / 0.280 | - | ok |
| dclm-baseline-25p-dolma1.7-75p | 8 | 0.463 | 8.6 | 1.029 | 0.964 | 0.065 | 0.69 | 0.012 | 0.007 | 0.91 | 0.15 | 0.96, 0.58 | 0.5 | - | 1.029 / 0.964 / 0.69 / 0.463 | - | ok |
| dclm-baseline-50p-dolma1.7-50p | 8 | 0.382 | 7.5 | 0.814 | 0.811 | 0.003 | 0.77 | 0.016 | 0.015 | 0.93 | 0.03 | 0.80, 0.92 | 0.3 | - | 0.814 / 0.811 / 0.77 / 0.382 | - | ok |
| dclm-baseline-75p-dolma1.7-25p | 10 | 0.366 | 8.2 | 0.784 | 0.752 | 0.032 | 0.72 | 0.009 | 0.012 | 0.93 | 0.20 | 0.69, 0.69 | 0.2 | - | 0.777 / 0.764 / 0.74 / 0.342 | - | ok |

## Evidence: what the geometry looks like (ranges over the 25 models, selected layer; layer 8 is similar)
- **The line is bent in every model, not only in the flagged ones.** The PC1-PC2 group-mean path is an arc:
  - Fitting PC2 group means as a quadratic in PC1 adds a median of +0.65 R2 over a linear fit.
  - The PC2 excursion is 0.23-0.58 of the PC1 range.
  - 10-33 % of the bias-corrected between-group variance lies off the best straight line.
  - The off-axis movement sits at the two ends. For the 10->100 step it is 0.48-1.46x the PC1 step. For the 1000->10000 step it is 0.44-2.09x.
    The middle step (100->1000) is mostly along PC1 (0.04-0.81x, median 0.27).
  - The main direction of the group means is still aligned with PC1 (|cos| >= 0.995), and PC1 carries 85-99 % of the linearly decodable log n in the top 5 PCs.
- **Effect on beta.** The ends bend in a similar way in most models, so beta_arc5 differs from beta_pc1 by < 0.07 in 20/25 models.
  The correction is not neutral, though: it lowers the high betas and raises the low ones.
  - Range: beta_pc1 0.616-1.132 becomes beta_arc5 0.647-1.020.
  - C1 therefore flags models with high beta_pc1 (all C1 hits have beta_pc1 > 1.0). See the honest note.
- **Digit count is mixed into PC1 in every model.** The within-digit-class slope ratio is 0.47-0.77 (median 0.63).
  - A smooth exponential fit of each model's own PC1 (no digit effect) gives 1.00-1.06. Every model therefore shows a staircase component at the
    9|10, 99|100, 999|1000 and 9999|10000 boundaries.
  - Inside the groups that straddle a boundary, the jump in PC1 beyond what the local magnitude slope predicts has a median of 0.16 (up to 0.37)
    of the mean coarse gap.
  - The coarse groups 10/100/1000/10000 (+-10 %) all straddle a digit boundary. In the most step-like model (falcon-and-cc) this makes the groups
    bimodal, so the group mean and the group median give betas that differ by 0.10 (C4).
- **Spread within groups is moderate.**
  - eta^2 of PC1 on the groups is 0.87-0.96.
  - The pooled within-group s.d. is 0.24-0.41 of the mean coarse gap.
  - No wrong-sign coarse gap appears in any seed (at most 1 wrong-sign fine gap).
- **Outliers are negligible.** At most 0.6 % of prompts have a robust |z| > 5 on PC1, and at most 0.3 % have a 5-D Mahalanobis p < 2e-6.
  Where the median and mean differ, the cause is the digit-boundary bimodality, not outliers.
- **EV1** is 0.22-0.51 at the selected layer, and EV1/EV2 is 2.9-10.6. PC1 is never degenerate with PC2. Only dclm-baseline-qc-7p-fw3 at layer 8
  falls below half the median.

## Code
- `08_geometry_work/geom.py`: per-model, per-layer and per-seed metrics, written to `geom.csv`.
- `08_geometry_work/extra.py`: per-gap decomposition into PC1 and off-axis parts, and the smooth-curve null for the digit test, written to `extra.csv`.
- `08_geometry_work/flags.py`: applies the pre-registered rule and the ±20/25 % threshold sensitivity, written to `geom_flags.csv`.

Run with `/tmp/claude-0/venv/bin/python -I geom.py geom.csv` (each script takes the output path or work directory).
Key code for beta_arc5:
```python
sq = ((M[b, dims] - M[a, dims])**2).sum()
bias = trace(cov(S[g==a][:, dims]))/n[a] + trace(cov(S[g==b][:, dims]))/n[b]
gap = sqrt(max(sq - bias, 0));  beta_arc5 = fit_direct(gaps)   # same LSQ as src/spacing_fit.fit_spacing_direct
```

## Honest note
1. **The 1-D assumption is not cleanly met in any model.** The arc and the digit staircase are present in all 25 models. The flags pick out where
   the deviation crosses an a-priori threshold, not a qualitative break. The flags are sensitive to the thresholds (4 models at the chosen values,
   1 at more lenient ones, 18 at stricter ones).
2. **C1 is mechanically tied to beta.** The arc correction shrinks the beta range, so C1 preferentially flags the highest-beta models.
   Removing them would truncate the beta range, which is a selection on the outcome variable. I would report beta_arc5 as a sensitivity variant
   rather than drop these models.
3. **beta_arc5 is not ground truth.** Its distances include whatever else differs between groups in PC2-5, including digit-count and token-format
   directions. It uses only the top 5 PCs, which PCA fitted per seed on the 10-group set; that is not the E01 4-group PCA. E01's own beta_direct used
   different prompts, so my checks describe the E06 geometry at E01's layer.
4. **Token count was not checked.** No tokenizer was available offline, so digit count is the only format feature I tested. The tokenisation of n
   (for example, single- vs multi-token 4-digit numbers) is unchecked and could be part of the "staircase".
5. **The digit test is confounded by the group design.** The coarse groups are built so that each one straddles a digit boundary. A better protocol
   would use bands that do not cross boundaries (for example, centres at 15/150/1500 or 20/200/2000).
