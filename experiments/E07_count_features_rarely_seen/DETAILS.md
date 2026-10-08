# E07 details

β here = E01's paper-protocol β (`old_direct`) unless stated. "resid" = β minus the
straight-line prediction from α_OLS. Spearman correlations, n = 25.

## 1. Strongest features (sorted by p with β)

| Feature | ρ with β | p | ρ with E06 β (old layer) | within family | weakest leave-one-family-out | ρ with resid | Bonferroni |
|---|---|---|---|---|---|---|---|
| F19 round-100 excess | −0.76 | < 0.001 | −0.70 | −0.63 | −0.67 | −0.74 | ✓ |
| **F14 R (# seen < 4000)** | **−0.73** | < 0.001 | −0.63 | −0.70 | −0.59 | −0.59 | ✓ |
| F12 # numbers with 1k–100k sightings | +0.68 | < 0.001 | +0.67 | +0.57 | +0.57 | +0.35 | ✓ |
| F12 # numbers with < 1k sightings | −0.68 | < 0.001 | −0.66 | −0.55 | −0.57 | −0.38 | ✓ |
| F13 mass share of < 1k numbers | −0.67 | < 0.001 | −0.63 | −0.52 | −0.54 | −0.41 | ✓ |
| F16 Gini | −0.62 | 0.001 | −0.45 | −0.67 | −0.54 | −0.71 | ✓ |
| F25 probe-band top/bottom ratio | +0.61 | 0.001 | +0.59 | +0.72 | +0.45 | +0.26 | ✓ |
| F17 spikiness | −0.60 | 0.001 | −0.67 | −0.28 | −0.33 | −0.12 | ✓ |
| F04 share of 10–99 | −0.59 | 0.002 | −0.38 | −0.71 | −0.48 | −0.76 | |
| F01 total numbers | +0.55 | 0.005 | +0.42 | +0.50 | +0.41 | +0.49 | |
| α_OLS (reference) | +0.46 | 0.02 | +0.52 | +0.24 | | | |

**No relation:** share of 1000–9999 (ρ = −0.21), years (−0.07), powers of 2 (+0.34,
n.s.), 1–20 (+0.01), last digit 0/5 (−0.29), slope within 1000–9999 (+0.41, but it is
just α again: ρ with α = 0.99).

The round-100 feature and R are correlated (ρ = 0.72) and tell the same story: where big
numbers appear mostly as rough round values ("about 500"), exact big numbers are rare.
R has the clearer interpretation and a better linear fit (R² 0.52 vs 0.26), so it is the
feature carried forward. Both were chosen *after* looking (see section 6).

## 2. Robustness of R (`python -m src.count_features`, last block of output)

```
R         pearson r=-0.72 (p=4.5e-05)  spearman -0.73 (p=3.2e-05)  within-family -0.70  weakest leave-one-family-out -0.59
alpha_ols pearson r=+0.56 (p=3.3e-03)  spearman +0.46 (p=2.2e-02)  within-family +0.24  weakest leave-one-family-out +0.11
together: beta ~ alpha + R   p(alpha)=0.465  p(R)=0.0042
fitted line: beta = 2.79 -0.275 x R/1000
other thresholds K: 1000: -0.68  2000: -0.67  4000: -0.73  8000: -0.67
content: code share vs beta r=+0.85, math share r=+0.76, code share vs R r=-0.79
  controlling code share                 R: partial r=-0.17 (p=0.416)   alpha: -0.19 (p=0.368)
  controlling math share                 R: partial r=-0.44 (p=0.033)   alpha: +0.05 (p=0.817)
  controlling contains Dolma (code > 0)  R: partial r=-0.48 (p=0.018)   alpha: -0.00 (p=0.998)
  18 recipes without code: R spearman -0.51 (p=0.031)   alpha -0.28 (p=0.268)
```

R vs α: r = −0.67 (p = 0.0003); within families r = −0.54. Saturation sum
S = Σ c/(c+4000) vs β: r = +0.65 (p = 0.0005).

## 3. Pairs with (almost) the same α

| Pair | α_OLS | β (E01) | R | Explained by R? |
|---|---|---|---|---|
| dolma1_7-no-code vs **c4** | −1.569 vs −1.563 | 0.86 vs **0.62** | 7,177 vs **7,392** | yes; c4 also has fewer numbers in total (1.40B vs 1.74B) and stronger round-number spikes |
| falcon-qc-20p vs qc-10p | −1.679 vs −1.679 | 0.74 vs 0.71 | 7,668 vs 7,735 | yes (small difference, right direction) |
| dolma1_7-no-code vs **no-math-code** | −1.569 vs −1.572 | 0.86 vs **0.62** | 7,177 vs 7,227 | direction right, size far too small, so probably content (math) |
| dolma1_7 vs **no-reddit** | −1.470 vs −1.469 | 0.98 vs **1.36** | 6,384 vs 6,279 | **no**: almost identical counts |

## 4. Stability of R across data seeds (exact counts, E04)

| Recipe | R for seeds 2 / 4 / 5 / 14 / 6198 | spread |
|---|---|---|
| c4 | 7,392 / 7,394 / 7,397 / 7,393 / 7,397 | ±2 (range 5) |
| dolma1_7 | 6,384 / 6,423 / 6,416 / 6,407 / 6,425 | ±17 (range 41) |

c4's seeds each read 72 % of the recipe, so they overlap heavily; dolma1_7's seeds read
6 % each and overlap little. That is why dolma1_7's spread is larger. Only numbers
whose count sits near 4,000 can cross the threshold, and only a few dozen do, so the
spread is in the tens. ±17 in R ≈ ±0.005 in β.

## 5. Interpretation: Bayesian shrinkage (hypothesis)
A learner estimating the position μ_n of each number from c(n) noisy examples gets, under
empirical Bayes (James–Stein shrinkage):

$$\hat\mu_n = w_n\,\bar x_n + (1-w_n)\,\mu_0,\qquad w_n=\frac{c(n)}{c(n)+K},\qquad K=\sigma^2_\text{noise}/\tau^2_\text{prior}$$

- c ≫ K means w ≈ 1: the number keeps its own position.
- c ≪ K means w ≈ 0: the number is pulled toward a shared centre μ₀.
- Many rarely-seen numbers in a range crowd together, gaps shrink, the range is
  compressed, and β falls.
- R is the all-or-nothing version (count the numbers with w < ½). S = Σ w_n is the
  smooth version, "the effective number of learned numbers".
- With c(n) = T·p(n) (amount × shape), w = p/(p + K/T): **shape and amount enter
  together**, which is why α (shape only) can't capture it.
- **Efficient coding** (space ∝ probability, or ∝ √probability as in Piantadosi 2016) is
  scale-free: it predicts amount doesn't matter. The infomax metric gives r ≈ 0 (E05),
  so the data favour shrinkage.

Supporting literature (all LLM accuracy tracks pretraining frequency):
- Razeghi et al. 2022, *Impact of Pretraining Term Frequencies on Few-Shot Numerical Reasoning*;
- Kandpal et al. 2023, *Large Language Models Struggle to Learn Long-Tail Knowledge*;
- Allen-Zhu & Li 2024, *Physics of Language Models 3.3* (knowledge capacity; a third-party
  summary says ~1,000 exposures per fact, not verified in the paper).

Efficient-coding background: Ganguli & Simoncelli 2014 (*Neural Computation*);
Piantadosi 2016 (*Psychonomic Bulletin & Review*). Statistics background: Good–Turing
"frequency of frequencies" N_k (R = Σ_{k<K} N_k), James–Stein shrinkage.

## 6. What this does and doesn't show
- **Exploratory:** R is the best of 26 features chosen after looking at these same 25 models.
  It passes Bonferroni, and its threshold isn't tuned (any K in 1000–8000 works).
- **Correlation:** the Dolma content (code, math) is a strong rival explanation (section 2).
- **Mechanism untested:** shrinkage is an interpretation that fits. A direct test is that
  within one model, decades with more rarely-seen numbers should be the most compressed.
  This can be computed for free from the E06 `.npz` files.
- **Confirmation:** E08, with the definitions locked before running.
