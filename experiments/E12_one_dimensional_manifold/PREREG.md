# E12: Do number hidden states lie on a 1-D manifold? (pre-registered 2026-11-07)

**Written before any of the tests below was run on real data.** The question, hypotheses H1–H5, fake-data controls
and tests T1–T9 are the project owner's (professor's request); this file fixes every setting and threshold.
Code: `src/e12_manifold.py` (written and checked on synthetic shapes before the real run). Any change after real
results are seen is listed at the end under "Deviations" with its reason.

**Already known before this test (disclosed):** PH of the number clouds shows one blob without persistent holes;
TwoNN dimension of 4-digit numbers ≈ 17–33; early-layer neighbours share the last digit, late-layer neighbours the
leading digit and nearby values; two-digit endings are category-like and additive (E10, E11).

## Data
Vectors: `topology_number_line/data/<model>_vectors/` (last-token state at the final `=` of
`A=A,B=B,C=C,D=D,n=`), **all five depths** (~25, ~50, ~75, ~84 %, last), each reported separately.

| Model | Tokenizer | Sets (token count constant inside each set) |
|---|---|---|
| Qwen2.5-1.5B | single digits | Q2 = 10–99 (90 numbers, 2 tokens); Q3 = 100–999 (900, 3 tokens); Q4 = 1000–9999 (9,000, 4 tokens; a fixed random subsample of 3,000, seed 0, used for every test and fake) |
| Pythia-1.4B | GPT-NeoX chunks | P2 = 10–99 (90, all 1 token); P3 = the 777 numbers in 100–999 that are 1 token; P1 = all 876 one-token numbers in 1–999 (token counts in prompt context: `pythia_token_counts_1_10000.txt`) |
6 sets × 5 depths = **30 datasets**.

## Preparation (every dataset)
1. Centre; divide each dimension by its standard deviation (dimensions with sd < 1e-8 dropped).
2. PCA to D = min(50, N − 1) dimensions (**main analysis**, all tests).
3. Without PCA (standardised full dimension): T2, T5, T6 and T8 repeated on the real data only (robustness row).
4. Huge dimensions: before step 1, report the share of total variance in the largest 1 and 5 raw dimensions and the
   ratio max / median dimension variance. Standardisation (step 1) removes their dominance; nothing else is done.

## Fake-data controls (built in the same D-dimensional PCA space, same N and the same numbers n)
- **Trend and noise:** trend T(n) = Gaussian kernel smoother of the PCA scores over n, bandwidth h (in units of n)
  chosen by leave-one-out CV from {1, 2, 4, 8, 16, 32, 64, 128, 256} (only h ≤ range/4); residual E = Z − T;
  noise for all fakes = Gaussian with the covariance of E, independent across numbers.
- (a1) **curve** = T(n) + noise. (a2) **log-curve** = the same curve T, but point n placed at arc-length fraction
  (log n − log n_min)/(log n_max − log n_min) along it, + noise. (a3) **helix** = T(n) + periodic parts with periods
  {2, 5, 10, 100} (only periods ≤ range/3), each a circle (cos, sin; cos only for p = 2) in its own pair of random
  orthonormal directions, together carrying 50 % of the residual variance, + noise with the other 50 %.
- (b) **sheets**: x = n mapped linearly to [0, L], y ~ Uniform[0, W] independent of n, embedded with two random
  orthonormal directions, scaled to the total variance of T, + noise. **thin** W = 0.1 L, **fat** W = 0.5 L.
- (c) **shuffled**: real Z with each dimension permuted independently across numbers.
- (d) **matched smoothness**: for each PCA dimension j, empirical variogram γ_j(Δ) over |m − n| = Δ (binned to
  integer Δ, smoothed with a 5-bin moving average), covariance k_j(Δ) = var_j − γ_j(Δ); a stationary Gaussian process
  on the integer grid n_min…n_max is sampled by circulant embedding (negative eigenvalues set to 0), independently per
  dimension, then evaluated at the dataset's numbers. Same distance profile, no shared number line.
Seeds fixed (`seed = 1000·dataset + condition`). Every test T2–T8 is run on the real data and all 8 fakes
(curve, log-curve, helix, thin sheet, fat sheet, shuffled, matched smoothness; the curve doubles as the T1 curve).

## Tests
**T1 power (first).** 5 independent replicates each of fake curve, thin sheet and fat sheet. The plateau rule (T2) has
**power against a sheet** if ≥ 4 of 5 curves pass it and ≤ 1 of 5 sheets of that type pass it. Power vs the **fat** sheet
is required before any dimension claim for that dataset; power vs the thin sheet is reported. No power → "no power
here" for that dataset's dimension claims (T2, H1, H4).

**T2 dimension.** TwoNN (maximum likelihood) and Levina–Bickel MLE with k ∈ {5, 10, 20, 40, 80} (k < N), dataset value
= 1 / mean over points of 1/m_k(x). **1-D plateau:** the MLE lies in [0.8, 1.5] for three consecutive k of the grid
(a factor-4 span). Also reported: ID at each k and the ratio ID(k = 80) / ID(k = 5).

**T3 order recovery.** Unsupervised principal curve (Hastie–Stuetzle): start from the first PC, 10 iterations of
(Gaussian smoothing of each coordinate along the current order, σ = 2 % of training points; 200-knot polyline;
re-projection), fitted without n. Held-out points are projected onto the curve; the curve's direction is set so
that Spearman(position, n) > 0 on the training points. Splits: (i) random 80/20, 3 repeats, mean |ρ|; (ii) the
contiguous block [n_min + 0.45 R, n_min + 0.55 R) held out (R = range), signed ρ; (iii) the top 20 % of the range held
out (extrapolation), signed ρ among them, and the share projected beyond the training end. **Fine order:** median
Spearman inside windows of 5 % of the range (random split, held-out points, windows with ≥ 5 points).

**T4 spacing.** Curve fitted on all points; arc length s(n) against n with three models: a + b n, a + b log n,
a + b n^p (p on a grid 0.05…2.00, step 0.05, chosen inside each fold). 5-fold CV MSE picks the model; p reported with a
200-sample bootstrap 95 % interval.

**T5 distance rule.** 200,000 random pairs (all pairs if fewer): squared distance against Δ_lin = |m − n| and against
Δ_log = |log m − log n|, each by isotonic regression; 5-fold CV R² for both.

**T6 digits.** The same pairs: squared distance against a digit-additive model (per digit position, a one-hot of the
unordered digit pair; numbers zero-padded to the set's digit count; ridge α = 1) vs isotonic |m − n|, 5-fold CV R² for
both and for both together. Nearest neighbours (k = 10): share with the same last digit, same leading digit (same
digit count and first digit), and |m − n| ≤ 1 % of the range, each with its chance level.

**T7 helix.** Coarse trend = Gaussian kernel smoother over n with σ = 0.2 R; residual. For each period p in
{2, 5, 10, 100} (p ≤ R/3): share of residual variance explained by regressing every residual dimension on
cos(2πn/p), sin(2πn/p) (cos only for p = 2). Null: 200 permutations of the residual rows; p-value and z.

**T8 topology (secondary).** Ripser H0/H1 on the D-dimensional data (greedy landmarks, 400, if N > 400). Long H1 bar =
lifetime > 0.25 × the median pairwise distance of the points used. Reported: number of long H1 bars, largest lifetime
ratio. Cannot distinguish a line from a blob.

**T9 positive control.** Days (7) and months (12), same prompt form: `X1=X1,X2=X2,X3=X3,X4=X4,Y=` with 4 random other
items from the 19 as demonstrations, 20 prompts per item, last-token state at the same 5 depths; one centroid per item
(centred and standardised over the 19 × 20 states, PCA as above). Pass for a (model, depth, category) if the centroids
have a long H1 bar (T8 rule) **or** the polygon-in-order score (share of items whose two nearest items are their
calendar neighbours, wrap-around required, as in E11 Spoly) ≥ 0.75. The control "works" for a model if months pass at
≥ 2 of 5 depths. If it fails everywhere, a missing loop for numbers is uninformative and is reported as such.

## Decision rules (per dataset, then per model × set over the 5 depths)
- **H1 smooth 1-D curve:** T1 power vs fat sheet, T2 plateau passes, T3 random |ρ| ≥ 0.9 and block ρ ≥ 0.8, and the real
  data is distinguishable from the matched-smoothness fake (below).
- **H2 helix:** some period with T7 permutation p < 0.01 and share ≥ 0.02, and no T2 plateau at k = 5–20.
- **H3 digit-by-digit:** T6 digit R² − value R² ≥ 0.05, and same-last-digit neighbour share ≥ 3 × chance.
- **H4 multiscale:** ID(80) ≤ 0.5 · ID(5), T3 random |ρ| ≥ 0.8, and fine-order median ρ < 0.5.
- **H5 nothing special:** real ID(k = 20) within ±20 % of the matched-smoothness fake's and real T3 random |ρ| within
  ±0.05 of the fake's.
- Over depths: **supported** if the rule holds at ≥ 4 of 5 depths, **not supported** at ≤ 1, otherwise **mixed**;
  **cannot tell** where T1 shows no power (for H1 and H4).
