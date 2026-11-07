# E13 Stage 1: digit-position codes (H1) and carry structure (H2) (pre-registered 2026-11-07)

**Written and committed before any H1 or H2 statistic was computed on real vectors.** The hypotheses, rivals and
decision thresholds are the project owner's (E13 plan). This file fixes every setting. Code: `src/e13_stage1.py`,
which is validated on planted synthetic data before the real run. Deviations will be listed at the end of the README.

**Known before this test (disclosed):** see E10–E12. Additive digit models beat numeric closeness in 30/30 E12
datasets. Smooth f(n) interpolates random held-out numbers but not held-out leading-digit blocks. Dimension and PH
tests have no power. E11 notes: digit centroids are near-equidistant (sd/mean 0.06–0.19), with weak order (Spearman
up to +0.77 for the tens digit). These were computed on Qwen and Pythia only, with a different method (no debiasing,
no power check).

## Data
- **Vectors.** Existing vectors in `topology_number_line/data/<model>_vectors/`: last-token state at the final `=` of
  `A=A,B=B,C=C,D=D,n=`, one prompt per number (demonstrations: random 1/2/3/4-digit numbers, seed 42), at 5 depths
  (~25/50/75/84 %/last).
- **9 models in 2 tokenizer families.** Token counts of n in prompt context are in `token_counts_in_prompt.json`:

| Family | Models | S2 = 10–99 | S3 = 100–999 | S4 = 1000–9999 |
|---|---|---|---|---|
| digit-splitting (D) | Crystal, OpenLLaMA-3B, StarCoderBase-3B, Qwen2.5-1.5B, SmolLM2-1.7B | 90 (2 tok) | 900 (3 tok) | 9,000 (4 tok) |
| chunk, learned BPE (C-neox) | Pythia-1.4B, RedPajama-3B | 90 (1 tok) | 777 one-token | 139 one-token |
| chunk, 3-digit groups (C-3) | OLMo-2-1B, Llama-3.2-1B | 90 (1 tok) | 900 (1 tok) | none one-token → **S4x** = all 9,000 (2 tok, `abc|d`), secondary only |

- **Main cells.** 125 main cells (model × set × depth), plus 10 secondary S4x cells.
- **Discovery and replication.** Discovery = Qwen2.5 and Pythia; replication = the other 7. Thresholds are locked
  here for both, before any data is seen. Everything is reported for discovery, replication and all 9.

## Preparation
1. **Main preparation.** Centre; z-score each dimension (sd < 1e-8 dropped); PCA to D = min(50, N − 1).
   - Descriptive statistics (H1, H2a, the H2c centroid indices) fit this on all numbers of the set.
   - Held-out evaluations (H2b, H2c cross-validation, knee) fit it on training numbers only.
2. **Robustness (R8)**, point estimates only:
   - (a) centre only, no z-scoring;
   - (b) the 3 raw dimensions with largest variance removed, then step 1.
3. **Seeds.** seed = CRC32 of `"model|set|depth|component|replicate"`, so seeds do not depend on job order.

## Digits and the additive model (used by H1 and H2)
- **Digits.** d_p(n) for p = 1 (leading) … L (units); L = 2, 3, 4. The leading digit takes 9 values (1–9), the others 10.
- **Additive model.** Z = μ + Σ_p B_p[d_p(n)] + E, fitted by OLS with one-hot columns per position (reference level
  dropped); this is "each position with the others regressed out". Effects are centred within each position.
- **Debiased squared distance between digit values a, b at position p:**
  d²_ab = ‖ĉ_a − ĉ_b‖² − σ̂² q_ab, where
  - q_ab = (e_a − e_b)ᵀ(XᵀX)⁻¹(e_a − e_b) (exact OLS variance factor of the contrast),
  - σ̂² = ‖E‖²_F / (N − rank X).
  
  Distance d_ab = √max(d²_ab, 0). This removes the noise inflation that pushes noisy centroids towards "equidistant".

## H1: digit-position codes ("floating-point code")
**Per cell and position p:**
- **O1** = Spearman(d_ab, |a − b|) over the 45 pairs (36 for the leading digit).
- **PR** = participation ratio (Σλ)²/Σλ² of the eigenvalues of −½ J D² J (classical MDS of the debiased squared
  distances; negative eigenvalues set to 0). Line ≈ 1, circle ≈ 2, simplex ≈ K − 1.
- **Templates**, each fitted as d_ab ≈ s · t_ab with a free scale s:
  - line t = |a − b|;
  - circle t = 2 sin(π|a − b|/10);
  - simplex t = 1.
  
  The best template has the lowest 5-fold CV MSE over pairs.
- **A1** = (λ₁/Σλ) × |Spearman(MDS axis-1 coordinate, v)|.
- **Signal V_p** = mean debiased squared distance / 2 (between-digit variance), as a share of total variance. Its
  null comes from 200 shuffles of n (rows of Z permuted); the position "has signal" if p < 0.01.
- **Uncertainty.** 95 % CIs from 200 bootstrap resamples of numbers (everything refitted). The digit-label shuffle
  uses 200 random permutations of the digit values within the position and gives a one-sided p-value for O1.
- **Matched-smoothness fake.** A Gaussian process per PCA dimension with the set's own variogram over |m − n| (as in
  E12; for N > 3,000 the variogram uses 2 million random pairs). One replicate, with every H1 statistic. Δ_real −
  Δ_fake is reported ("beyond smoothness").
- **Planted power.** Fake = the real centroids of the other positions + a planted template at position p + the real
  residual rows permuted. The planted template has the same between-digit variance as real position p and sits in
  random orthonormal directions (line 1-D, circle 2-D, simplex K − 1-D). 20 replicates each.
  - (i) **Template recovery** = share of replicates whose best template is the planted one.
  - (ii) **Decay power** = share passing the cell decay rule (below) when line is planted at the leading digit and
    simplex at the units (other positions real). This uses 50 bootstraps per replicate instead of 200.
  - (iii) **Decay false positives** = share passing when line is planted at every position.
  - A cell **has power** for the decay rule if (ii) ≥ 0.80 and (iii) ≤ 0.20.

**Cell decay rule.** Δ = O1_leading − O1_units > 0.30, bootstrap 95 % CI of Δ above 0, and O1 non-increasing
from leading to units, with tolerance 0.05 (O1_{p+1} ≤ O1_p + 0.05 for every p).

**Rival cells:**
- **R-a:** all O1 > 0.5 and max − min < 0.1.
- **R-b:** all O1 < 0.3.
- **R-c:** the units best template is circle, and circle-template recovery at the units ≥ 0.80.

**Family verdict** (computed over cells at depth ≥ 50 %, i.e. 4 depths × sets × models of the family):
- **holds** if ≥ 70 % of powered cells pass;
- **does not hold** if ≤ 30 % pass;
- **mixed** otherwise;
- **cannot tell** if < 50 % of the family's cells have power.

**H1 verdict:**
- **supported** if it holds in both families AND in both families among the replication models alone;
- **one family only** if it holds in exactly one family;
- **not supported** otherwise.

R-a, R-b and R-c get the same family verdicts (≥ 70 % of cells; for R-c only cells with circle power count).
R-d (ordinality depends on the tokenizer, not the position) has no formal rule; it is judged from the
family-by-position table.

**"Beyond smoothness" check (added, secondary).** A smooth function of n alone already makes leading-digit centroids
ordered. So we also report:
- the share of passing cells where Δ_real − Δ_fake > 0.10;
- the share where the units digit has signal (p < 0.01) and its best template is simplex.

The full H1 picture ("trailing digits categorical") needs both. They are reported, but are not part of the main rule.

**Exponent (digit-count) code, secondary.**
- **Data:** numbers 1–9, 10–99, 100–999, 1000–9999 (single-token only in chunk models: C-neox lengths 1–4, C-3
  lengths 1–3); 4-digit subsampled to 900 (seed).
- **Probes:**
  - (a) 5-fold CV balanced accuracy of a class-weighted logistic regression (C = 1) on PCA-50 fitted within folds;
  - (b) leave-one-length-out ridge (α = 1) regression of length: a held-out length is "ordered" if its mean
    prediction lies strictly between its neighbours' training means, or beyond the nearest one on the correct side
    at the ends.
- **Explicit** = balanced accuracy ≥ 0.90 and every held-out length ordered.
- **Control:** permuted length labels.
- **Token-count confound.** In digit-splitting models length = token count, so only chunk models count as a clean test.

## H2: carries
**Steps.** A step is n → n + 1 with both n and n + 1 in the set. The carry level c(n) is the number of trailing 9s of n.
- Levels 1 (S2), 1–2 (S3), 1–3 (S4).
- H2a and H2b are run only where there are ≥ 30 level-0 steps and ≥ 5 level-1 steps. A level j is tested only with
  ≥ 5 steps at that level.
- C-neox S4 (139 scattered numbers) is therefore skipped for H2a/H2b.

**H2a: carry spikes beyond the digit code.**
- **Regression.** s²(n) = ‖z(n+1) − z(n)‖² is regressed on: intercept + a natural cubic spline in n (df 6) +
  b · ŝ²_add(n) + Σ_j e_j · 1[c(n) = j], where ŝ_add is the step of the fitted additive model.
- Squared step length is primary because noise adds a constant to E[s²] but not to s. The user's s is also reported.
- Exact-level dummies are used, so "a_2 > a_1" is read as "the level-2 spike exceeds the level-1 spike".
- **CIs.** Cluster bootstrap over decades (n // 10), 500 resamples.
- **Carry-shuffle control.** Carry labels permuted across steps 200 times; permutation p-values.
- **Cell rule:**
  - e_1 > 0 with CI above 0;
  - and, where level 2 exists, e_2 − e_1 > 0 with CI above 0;
  - S4 also reports e_3 − e_2.
- **Controls with the same rule:**
  - **additive fake:** fitted additive model + permuted residual rows; expected to fail;
  - **matched-smoothness fake;**
  - **planted fakes.** Smooth trend T(n) (Gaussian kernel, bandwidth by leave-one-out CV from {1, 2, 5, 10, 20} %
    of the range, minimum 1) + permuted residual rows (Z − T). They are run with and without planted cumulative
    carry jumps: at each level-j carry the trajectory jumps by κ_j along a fresh random unit vector, κ_j = 0.5 j ×
    the median real level-0 step length. 20 replicates each, 100 bootstraps each.
- A cell **has power** if planted-with-jumps passes ≥ 80 % and planted-without passes ≤ 20 %.

**H2b: linear successor operator.**
- **Fit.** z(n+1) ≈ A z(n) + b by ridge, α chosen by 5-fold CV from {0.1, 1, 10, 100, 1000}, on a random 80 % of
  the level-0 steps (3 repeats).
- **Held-out sets.**
  - the other 20 % of level-0 steps;
  - all level-j steps (never used for training).
- **Primary metric.** R²_j for predicting z(n+1), relative to the training mean of z(n+1). PCA is fitted on the
  numbers in training steps.
- **Secondary metric.** R² of the step Δz.
- **Cell rule.** R²_0 − R²_{≥1} > 0.20 with bootstrap CI (500 resamples of test steps) above 0. The hierarchy
  (R² falling with level) is reported.
- **Controls:** additive fake, matched-smoothness fake, and planted fakes with and without jumps.
- **Power:** as for H2a.
- **"Beyond digits" (secondary).** A purely additive categorical code also breaks a linear operator at carries,
  because the 9 → 0 move is never seen in training. So we also report (real drop − additive-fake drop); this must be
  > 0.10 with CI above 0 before a carry *event* beyond the digit code is claimed.

**H2c: tree vs line.**
- **Pairs.** All pairs, or 200,000 random pairs; for S4, 100,000 training and 50,000 test pairs per fold. The
  outcome is the squared distance in Z.
- **Models.**
  - (a) **line:** isotonic (increasing) in |m − n|;
  - (b) **tree:** OLS on indicators 1[prefixes of length l differ], l = 1 … L − 1, plus an intercept;
  - (c) **both:** backfitted, 10 alternations.
- **Cross-validation.** Standardisation and PCA are refitted on training numbers in every fold.
  - (i) **random numbers:** 5-fold; test pairs have both numbers held out.
  - (ii) **held-out groups:** leave one leading-digit group out (9 folds); test pairs involve ≥ 1 number of the
    held-out group, training pairs involve none.
  - R² is computed relative to the training-pair mean.
- **Also reported:**
  - isotonic line R² inside each leading-digit group (5-fold over within-group pairs, mean over groups);
  - ultrametric-violation index (mean over triples of (d_max − d_mid)/d_max, 20,000 random triples);
  - mean and 95th percentile of the four-point Gromov δ relative to the largest of the six distances (20,000
    quadruples).
  
  Both indices use debiased centroid distances (‖ḡ − h̄‖² − σ̂²(1/n_g + 1/n_h)) of the leading-digit groups and of
  the two-digit-prefix groups (S3, S4).
- **Cell metric.** ΔR²_group = R²_tree − R²_line on held-out groups. 95 % CI from 1,000 bootstrap resamples of the
  9 held-out groups.
- **Cell rule.** ΔR²_group ≥ 0.10 and CI above 0.
- **Controls:**
  - matched-smoothness fake;
  - **planted tree:** random vectors per leading-digit group and per two-digit prefix, each level with the real
    debiased between-group variance, + permuted residual;
  - **planted line:** smooth trend T(n) + permuted residual.
  
  10 replicates each. A cell **has power** if the planted tree passes ≥ 80 % and the planted line ≤ 20 %.
- **H2c verdict** uses late layers (~84 %, last). The chunk family is reported alongside; the owner's expectation that
  the line wins in Pythia mid layers is checked descriptively.

**H2 verdicts.**
- H2a and H2b use all 5 depths.
- Family and hypothesis verdicts follow the H1 scheme: holds ≥ 70 % of powered cells, does not hold ≤ 30 %, mixed
  otherwise, cannot tell if < 50 % have power; supported if it holds in both families and among the replication models.
- H2a uses cells with level 2 (S3, S4); S2 is reported for e_1 only.

## Secondary: the "knee" test
- **Model.** Held-out R² (random 80/20, seed fixed) of a smooth + additive digits model: OLS on a natural spline in
  n (df 10) + one-hot digits, fitted on training PCA scores.
- **Rank restriction.** Predictions are restricted to the top-k right singular vectors of the training fitted values,
  for k ∈ {1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50}.
- **Knee** = the smallest k reaching 95 % of the R² at k = 50.
- **Read-out.** Compared descriptively with 1 + 9 × (number of categorical positions). No rule.

## Labels
- All H1, H2a, H2b and H2c rules above are PRIMARY.
- The "beyond smoothness / beyond digits" checks, the exponent code, the knee, robustness and the Pythia mid-layer
  remark are SECONDARY.
- Anything else found while looking is EXPLORATORY.
