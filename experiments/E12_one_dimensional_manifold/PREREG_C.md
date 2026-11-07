# E12c: held-out R² of smooth models of Z(n) (pre-registered 2026-11-07, before running)

**Written before any of these models was fitted on real data.** Request (owner): for each dataset and depth, fit a smooth
curve f(n) on a random 80 % of the numbers and report the held-out R² of predicting Z(n) for the other 20 %; also a 2-D
smooth surface over (n, n mod 10), curve + digits, and a shuffled-n control; repeat with a held-out block.
Code: `src/e12c_heldout_r2.py`.

## Data and preparation
The 30 datasets of E12 (Qwen Q2, Q3, Q4; Pythia P2, P3, P1; 5 depths). For every split, standardisation (per
dimension) and PCA to D = min(50, N_train − 1) are fitted **on the training numbers only**; held-out vectors are
projected with the training transform. Z = those PCA scores.

## Models (each predicts Z at held-out n from training data only)
- **curve:** Gaussian kernel smoother over n, bandwidth h_n.
- **curve + digits:** curve plus additive centred one-hot effects of the tens and units digits (+ hundreds for Q4),
  backfitted (15 alternations), bandwidth h_n.
- **2-D surface:** product-kernel smoother over (n, u = n mod 10): Gaussian in n (h_n) times Gaussian in the circular
  distance of u (min(|a − b|, 10 − |a − b|), bandwidth h_u).
- **shuffled:** the curve model fitted after randomly permuting n among the training numbers (held-out numbers keep
  their true n); 5 permutations, mean R².
Bandwidths: h_n from {0.005, 0.01, 0.02, 0.05, 0.10, 0.20} × R (R = range of n; h_n ≥ 1), h_u from {0.5, 1, 2, 5},
chosen per model by 5-fold CV inside the training set (lowest squared error). Kernel weights are normalised in log space;
a held-out point whose weights all underflow is predicted by its nearest training number.

## R²
R² = 1 − Σ‖Z_test − prediction‖² / Σ‖Z_test − mean(Z_train)‖², summed over the D dimensions (negative = worse than
predicting the training mean).

## Splits
- **Random:** 80 / 20, 3 repeats (seeds 0–2), mean R² (and the per-repeat values).
- **Block:** a contiguous block held out, everything else trains: 70–79 for Q2 and P2, 700–799 for Q3, P3 and P1,
  7000–7999 for Q4 (its 3,000-number subsample).

## Reported
Per dataset, depth and split: R² for curve, curve + digits, 2-D, shuffled; the chosen bandwidths; and the contrasts
curve + digits − curve, 2-D − curve, curve − shuffled, block − random. No pass/fail threshold is set; the table is the
result. Caveat known in advance: with one prompt per number there is no measure of prompt noise, so the maximum
attainable R² is unknown.
