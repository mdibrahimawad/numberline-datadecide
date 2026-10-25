# Pre-registered rule (written before any geometry numbers were computed)
Unit: model x layer (E01 selected layer = PRIMARY decision; layer 8 = secondary), medians over seeds 45/46/47,
E06 per-prompt top-5 PC scores. beta_pc1 = paper direct fit on |gaps| of PC1 means of coarse groups 10/100/1000/10000.
A single-axis beta is NOT a valid description if ANY of:
 C1 curvature: |beta_arc5 - beta_pc1| > 0.10, beta_arc5 = same fit on noise-bias-corrected Euclidean distances
    between consecutive coarse group means in PC1..PC5 (a straight tilted line leaves beta unchanged; only bending changes it)
 C2 digit-count staircase: pooled within-digit-class slope of PC1 on log10 n / overall slope < 0.5
 C3 weak magnitude share of PC1: eta^2 (10 groups) of PC1 < 0.5, OR seed s.d. of beta_pc1 > 0.05 (3x E06 median)
    OR any coarse PC1 gap has the wrong sign in >=2 of 3 seeds (the fit takes |gaps| and hides this)
 C4 outliers: |beta_pc1(group medians) - beta_pc1(group means)| > 0.10
 C5 weak PC1: EV1 < 0.5 x median EV1 of the 25 models at the same layer
Thresholds: 0.10 in beta ~ 7x seed s.d. (0.015) and ~1/8 of the between-model range.
Reported only (no flag): |rho(n,PC2)|, R2(u~PC1)/R2(u~PC1..5), off-line between-group variance, digit-boundary step, EV1/EV2.
