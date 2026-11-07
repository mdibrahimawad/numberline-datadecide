# E12b: smooth part + additive digit part (pre-registered 2026-11-07, before running)

**Written before this decomposition was fitted on any real data.** Request (owner): fit every vector as
h(n) = f(n) + digit part, f smooth (bandwidth ≥ 1 % of the range), digit part additive (one-hot tens + one-hot units,
+ hundreds for Q4); remove the digit part; run T2, T3, T4 on f only and on fakes; power check at noise 0.1×, 0.3×, 1×;
report whether f is 1-D and how much variance it carries. Code: `src/e12b_smooth_digits.py`.

**Known in advance (stated so it is not over-read):** f is a kernel-smoothed function of the scalar n, so it is a
1-D curve by construction; a T2 pass on f alone is expected and is reported as a consistency check, not as evidence.
The informative quantities are f's variance share, whether f's geometry recovers order without n (T3), its spacing
(T4), and whether the **digit-removed data** Z − D (= f + residual) is detectably 1-D given the power check.

## Data and preparation
Exactly E12 (`PREREG.md`): the same 30 datasets (Qwen Q2, Q3, Q4; Pythia P2, P3, P1; 5 depths), centred, z-scored per
dimension, PCA to D = min(50, N − 1). Z denotes these PCA scores.

## Decomposition
Z(n) = f(n) + D(n) + E(n).
- **Digit part D:** additive one-hot effects (each centred to mean 0) of the tens digit and the units digit of n;
  for Q4 also the hundreds digit. Digits from n itself (P1 includes 1- and 2-digit numbers: tens digit 0 for them).
- **Smooth part f:** Gaussian kernel smoother over n with bandwidth h.
- **Fit:** backfitting, 30 alternations (f ← smooth(Z − D); D ← least-squares one-hot fit of Z − f).
- **Bandwidth:** h chosen from {0.01, 0.02, 0.05, 0.10, 0.20} × R (R = range of n) by 5-fold CV of the whole additive
  model (digit effects and f fitted on the training folds, f evaluated at held-out n). Smallest CV error wins.
- **Known confound:** for 2-digit sets (Q2, P2) the tens digit is the leading digit, so the tens one-hot and f compete
  for the same coarse trend. Primary analysis follows the request (tens + units); **secondary for Q2 and P2:** units
  digit only.
- **Variance shares** (sum over dimensions, of centred components): var(f)/var(Z), var(D)/var(Z), var(E)/var(Z), plus
  R² of f alone and of f + D.

## Conditions and tests
Conditions per dataset: **f** (the fitted smooth part at the dataset's n), **Z − D** (digit-removed data, the
primary real-data test), and fakes at noise scale c ∈ {0.1, 0.3, 1}: **curve_c** = f + c·G, **thin sheet_c**,
**fat sheet_c** (E12 construction: x = n linearly, y ~ U[0, W], W = 0.1 L / 0.5 L, scaled to var(f), + c·G); G is
Gaussian with the covariance of the real residual E; c scales its standard deviation.
- **T2** (E12 rule): Levina–Bickel MLE at k ∈ {5, 10, 20, 40, 80}, TwoNN; 1-D plateau = MLE in [0.8, 1.5] at three
  consecutive k.
- **T3** (E12): unsupervised principal curve, random 80/20 (3 repeats), held-out middle block, top-20 % extrapolation.
- **T4:** (a) for f only, the **exact** arc length along f (sorted by n), against n, log n, n^p (5-fold CV, p with a
  200-sample bootstrap 95 % interval); (b) for every condition, the E12 principal-curve arc length.
- **Power (T1-style):** 5 replicates of curve_c, thin sheet_c and fat sheet_c per noise level; power vs the fat sheet at
  level c if ≥ 4 / 5 curves pass the plateau and ≤ 1 / 5 fat sheets pass (thin sheet reported).

## Decision rules (per dataset; then over the 5 depths as in E12: ≥ 4 supported, ≤ 1 not, else mixed)
- **"f is 1-D"** = T2 plateau on f (expected by construction; reported).
- **"f recovers order"** = T3 on f: random |ρ| ≥ 0.9 and block ρ ≥ 0.8.
- **"Digit-removed data is 1-D"** = power vs fat sheet at c = 1 **and** T2 plateau on Z − D. Without power at c = 1:
  "cannot tell", and the lowest c with power is reported.
- Variance: f's share reported per dataset and as min–max per set.
Any change after real results are seen is listed under "Deviations" in the E12 README.
