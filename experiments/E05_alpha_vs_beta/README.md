# E05: α vs β (does the training data's number distribution predict compression?)

**Question:** does the α of each model's exact training stream (E04) predict its β (E01)?
If not, what other data measure does?
**Status:** done for all 25 models (2026-10-06 → 10-08). Analyses were rerun as more
recipes finished (n = 3 → 5 → 10 → 14 → 25); `docs/findings.md` keeps that history.
**Outputs:** `results/datadecide/alpha_beta.{csv,png}`, `alpha_beta_correlations.json`,
`confounders.json`, `decade_metrics.csv`.

## How to run (all local, seconds, no cloud)
```bash
python -m src.join_alpha_beta          # α (exact; window α as fallback) + β  -> alpha_beta.*
python -m src.confounder_analysis      # number density, code/math share, partial + within-family
python -m src.decade_metrics           # data-side "β" from decade counts (infomax / power q / saturation K)
python -m src.decade_metrics --exclude dclm --out results/datadecide/decade_metrics_no_dclm.csv
```

## Headline result (n = 25, β = paper protocol from E01)

| Predictor | r with β_direct (p) | r with β_log (p) | Notes |
|---|---|---|---|
| **α_OLS** (exact) | **+0.56 (0.003)** | **+0.60 (0.002)** | without the Dolma family: 0.22 / 0.31, n.s. |
| α_MLE | −0.14 (0.50) | −0.14 (0.51) | no relation |
| Numbers per 1k tokens | +0.47 (0.02) | +0.44 (0.03) | same direction in every family; pooled within-family r = +0.41 (p = 0.04) |
| Code share / math share | +0.85 / +0.76 | +0.86 / +0.77 | non-zero only in Dolma-based recipes, so effectively a Dolma indicator |
| Decade metric "power q → 0" (shape) | +0.67 (< 0.001) | +0.67 (< 0.001) | within-family +0.53; leave-one-out R² 0.34 |
| Decade metric "saturation", K ≈ 4000 (shape + amount) | +0.61 (0.001) | +0.60 (0.002) | within-family +0.43; LOO R² 0.21 |
| Infomax (efficient coding: space ∝ count) | +0.01 | −0.01 | fails |

- **Direction as predicted:** a flatter α (more large numbers) goes with less compression.
- **But not a direct law:** inside families α goes the wrong way in DCLM (r = −0.60) and
  Falcon (−0.26), and pairs with equal α have very different β.
- **Cleanest evidence:** the DCLM → Dolma mixing series, where α flattens and β rises
  with the Dolma share (ρ = 0.90). Content changes too, though.
- **The efficient-coding law** log₁₀β = 1 + γα with one shared γ is not supported (n = 10:
  fitted slope ~3× too small, constrained R² < 0). The apparent constancy of γ ≈ 0.675 is
  arithmetic, not evidence.
- This led to E07: what in the counts explains β beyond α?

Full tables, formulas and the efficient-coding analysis: [DETAILS.md](DETAILS.md).
