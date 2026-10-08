# E05 details

All numbers are n = 25, β from E01 (`beta_direct_mean`, `beta_log_mean`), α from E04
(`alpha_seed2.csv`) unless stated.

## 1. Plain correlations (`python -m src.join_alpha_beta`)

| | β_direct | β_log |
|---|---|---|
| α_OLS | r = +0.564 (p = 0.0033), ρ = +0.456 (p = 0.022) | r = +0.596 (p = 0.0017), ρ = +0.478 (p = 0.016) |
| α_MLE | r = −0.141 (p = 0.50), ρ = −0.133 | r = −0.138 (p = 0.51), ρ = −0.155 |

## 2. Within families (Pearson r, β_direct)

| Family | n | α_OLS vs β | numbers seen vs β |
|---|---|---|---|
| dolma | 6 | +0.75 | +0.87 |
| mix | 3 | +0.99 | +1.00 |
| fineweb | 2 | same direction | same direction |
| falcon | 6 | **−0.26** (wrong way) | +0.41 |
| dclm | 7 | **−0.60** (wrong way) | +0.47 |
| pooled (family-demeaned) | 25 | +0.28 (p = 0.17) | **+0.41 (p = 0.04)** |

Without the Dolma family (n = 19): α_OLS vs β r = +0.22 (p = 0.36), β_log +0.31.

## 3. The DCLM → Dolma mixing series (dose-response)

| Mix | α_OLS | β_direct | β_log |
|---|---|---|---|
| DCLM 100 % | −1.596 | 0.666 | 0.439 |
| 75 % DCLM / 25 % Dolma | −1.577 | 0.841 | 0.741 |
| 50 / 50 | −1.541 | 0.952 | 0.926 |
| 25 % DCLM / 75 % Dolma | −1.513 | 1.108 | 1.195 |
| Dolma 1.7 100 % | −1.470 | 0.978 | 0.969 |

Dolma share vs α_OLS r = 0.99, vs β ρ = 0.90. Note that number density and code share
also rise with the Dolma share.

## 4. Confounders (`python -m src.confounder_analysis`)

| Predictor | r with β_direct | after controlling density | after density + code share |
|---|---|---|---|
| α_OLS | +0.56 | partial +0.50 (p = 0.01) | **−0.20 (p = 0.36)** |
| α_MLE | −0.14 | −0.30 | −0.10 |
| number density | +0.47 (p = 0.02) | | |
| code share | **+0.85** | | |
| math share | **+0.76** | | |

Code (StarCoder) and math (Proof-Pile-2) files exist only in Dolma 1.7-based recipes, so
these shares act as a Dolma indicator. Controlling for them removes α's signal; see also
E07, where the same check is applied to R.

## 5. Decade metrics (`python -m src.decade_metrics`)
β compares the gaps between groups at 10, 100, 1000, 10000; with three gaps the direct fit
is √(Δ₃/Δ₁). The matching data-side measure gives each decade (10–99, 100–999,
1000–9999) a "space" G_k from its counts and forms **data β = √(G₃/G₁)**:

| Metric | Space per integer n | Measures |
|---|---|---|
| infomax | c(n) | shape (efficient coding: space ∝ frequency) |
| power q | c(n)^q, q fitted in (0, 1] | shape; q → 0 compares typical log-frequencies per decade |
| saturation | c(n)/(c(n)+K), K fitted | shape + amount (Bayesian shrinkage weight; see E07) |

Results (n = 25):

| Metric | param | r (β_dir) | within r | LOO R² | r (β_log) | within r | LOO R² |
|---|---|---|---|---|---|---|---|
| log10 numbers seen | | +0.37 | +0.32 | +0.02 | +0.35 | +0.29 | +0.00 |
| infomax | q = 1 | +0.01 | +0.11 | −0.12 | −0.01 | +0.07 | −0.13 |
| power | q = 1/3 | +0.51 | +0.47 | +0.15 | +0.50 | +0.44 | +0.13 |
| power, best q | q → 0.01 | **+0.67** | **+0.53** | **+0.34** | **+0.67** | **+0.51** | **+0.33** |
| saturation, best K | K ≈ 4,000–4,500 | +0.61 | +0.43 | +0.21 | +0.60 | +0.41 | +0.19 |

At n = 14 (before the Dolma ablations), saturation was the best metric (r = 0.57, within
0.87). Leave-one-out R² re-fits q/K without the held-out model.

## 6. Formulas tried at n = 14 (history; superseded by E07)
- **Effective α** = α + w·log10(numbers per 1k tokens), then β = a + b·effective α. Best
  w = 0.86 (β_log): r 0.42 → 0.59, but leave-one-out R² ≈ 0.
- **Family-relative formula**: β_log = 2.31 + 0.99·α + 0.057·(numbers per 1k tokens −
  family mean). Both terms were significant (p = 0.021, 0.005), with R² 0.62 and LOO 0.41 at
  n = 14. At n = 25 it dropped to R² 0.39, LOO 0.18.
- **Hand-picking** (dropping 1–2 models, or one model per family): r ranges from −0.8 to
  +0.9 depending on the pick. That is cherry-picking and is not reported as a result.

## 7. The efficient-coding γ law (n = 10)
The proposed law log₁₀β = 1 + γα predicts β from α with one constant γ.
- γ computed per recipe as (log₁₀β − 1)/α comes out at 0.675 ± 0.046. It looks constant, but
  since |log₁₀β| ≤ 0.22 is small next to the 1, γ ≈ 1/|α| whatever β is. This is
  circular.
- The real test is whether β moves with α by the predicted amount. Free fit:
  log₁₀β_direct = 0.23 + 0.21α (slope ~3× too small, intercept far from 1). Forcing the
  law's form: R² = −0.36 (β_direct), 0.22 (β_log).
- The Falcon filters break it directly: α spans only 0.04, while β goes 0.71 → 0.95.
- The scale-free theory also predicts that the **amount** of numbers can't matter, yet
  amount does matter within families. This motivated the saturation and shrinkage version.
