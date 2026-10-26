# E11: α measured on the numbers that β is measured on (E01, 3 gaps, best layer)

**Question:** does the α–β relation get stronger if α describes the training-data frequency of
the *same numbers* the model's number line is measured on?
**Status:** done (2026-10-26). **Exploratory:** this α definition was adopted after seeing the
original result (see Caveats). Reproduce with `python experiments/E11_alpha_probed_numbers/compute.py`;
the outputs are `per_model.csv` and `results.json`.

## 1. Setup (unchanged from E01)

- **Models:** the 25 DataDecide 1B models (same architecture, tokenizer and 100B tokens; only the data differs).
- **β (model side), exactly as in the paper:**
  - Prompts use 4 magnitude groups: **1–39, 80–119, 980–1019, 9980–10019**.
  - The layer is chosen by the variance–monotonicity joint score, Q_ℓ = median over seeds 42–44 of √(EV·|ρ|),
    with ℓ* = argmax Q_ℓ.
  - β is the direct fit on the 3 gaps between the group means at ℓ*, measured on held-out seeds 45–47.
- **Counts (data side):** exact counts c(n) of every integer n = 0…10000 in each model's 100B-token training stream (E04).

## 2. What changes: how α is computed

Both versions of α measure the same thing: **how fast numbers get rarer as they get bigger**, the slope of
log count against log n. Only the numbers that enter the slope differ.

| | Numbers used | Points in the fit |
|---|---|---|
| **Old α** (paper) | every integer 1–10000 | 10,000. Nine thousand of them are 4-digit numbers, so the slope is mostly the 1000–9999 range. |
| **New α** | only the probed groups: 1–39, 80–119, 980–1019, 9980–10000 | 4, one per group: log(median count in the group) against log(group centre) |

Counts stop at 10000, so the top group uses 9980–10000. The median keeps one-off spikes (e.g. the number
10000 itself) from dominating a group.

## 3. Why the new α is the more meaningful test

The hypothesis is that **how often numbers appear in training shapes how the model spaces them on its
internal number line.**

- β only measures how the model spaces the **4 probed groups**. It never sees 2000, 5000 or 7500.
- The old α is mostly about **1000–9999**, a range β does not probe.
- So with the old α, the model's spacing of one set of numbers is compared with the data's frequency of a
  different set. When they disagree, it is unclear whether the hypothesis failed or the two sides measured
  different numbers.
- The new α puts both sides on **the same numbers**. That is a direct test of the hypothesis.

**Example: fineweb-edu vs c4.** Median count in each probed group:

| | 1–39 | 80–119 | 980–1019 | 9980–10000 | Drop between groups | Old α | New α | β |
|---|---|---|---|---|---|---|---|---|
| fineweb-edu | 13.4 M | 1.29 M | 51 k | 1.9 k | ×10, ×25, ×27 | −1.70 | −1.37 | **0.89** |
| c4 | 11.4 M | 0.93 M | 34 k | 1.5 k | ×12, ×27, ×23 | −1.56 | −1.39 | **0.62** |

- **The old α predicts the wrong order.** It says fineweb-edu's data is much steeper than c4's, so it predicts
  fineweb-edu should be *more* compressed. Its β says the opposite.
- **The new α predicts the right order.** On the numbers the model is actually tested on, fineweb-edu thins
  out slightly *slower* (×10, ×25 vs ×12, ×27), so the new α predicts *less* compression, which matches β.
- **Where the old α went wrong:** its large difference between these two recipes comes from numbers between
  the probed groups, which β does not measure.

The old α is not wrong. It answers a broader question (does the overall number distribution matter?),
while the new α answers the question β is built for.

## 4. Results (all 25 models, nothing dropped)

| α used | Pearson r | Spearman ρ |
|---|---|---|
| Old α (1–10000) | 0.56 (p = 0.003) | 0.46 (p = 0.022) |
| **New α (probed numbers)** | **0.82 (p = 7×10⁻⁷)** | **0.77 (p = 7×10⁻⁶)** |

### Robustness

| Check | Old α: r / ρ | New α: r / ρ |
|---|---|---|
| All 25 | 0.56 / 0.46 | **0.82 / 0.77** |
| Without the 4 models with a measurement flag (E10) | 0.50 / 0.38 | **0.79 / 0.70** |
| Within families (family means removed) | 0.28 / 0.24 | **0.75 / 0.75** |
| Only the 17 recipes without code or math | −0.38 / −0.41 | 0.44 (p = 0.08) / 0.53 (p = 0.03) |
| Family means (6 points) | n/a | 0.84 (p = 0.035) / 0.54 |
| Family-cluster bootstrap, 95% CI of r | n/a | [0.39, 0.90] |

The 4 E10 measurement flags are dolma1_7-no-reddit (its β does not replicate) and falcon-and-cc-qc-10p,
dolma1_7 and fineweb-edu (layer near-ties).

What the robustness checks show:

- **The new α works within families.** r = 0.75 after removing family means, where the old α gave 0.28.
  So it is not only Dolma vs the rest.
- **In the 17 web-only recipes** (no code or math), the old α relation is *negative*. The new α turns it
  positive, but moderately: ρ = 0.53 (p = 0.03), r = 0.44 (p = 0.08). Part of the full-sample strength still
  comes from the code/math (Dolma) recipes.
- **The two α definitions rank recipes differently:** Spearman 0.47 between them.

## 5. Per model (sorted by β; rank 1 = highest β or flattest α)

| Model | Family | Layer ℓ* | β | β rank | Old α | Old rank | New α | New rank |
|---|---|---|---|---|---|---|---|---|
| dolma1_7-no-reddit | dolma | 8 | 1.360 | 1 | −1.469 | 2 | −1.315 | 3 |
| dolma1_7-no-flan | dolma | 9 | 1.197 | 2 | −1.476 | 4 | −1.328 | 4 |
| dclm-25p-dolma1.7-75p | mix | 8 | 1.108 | 3 | −1.513 | 6 | −1.336 | 5 |
| dolma1_6plus | dolma | 9 | 1.077 | 4 | −1.429 | 1 | −1.308 | 1 |
| dolma1_7 | dolma | 6 | 0.978 | 5 | −1.470 | 3 | −1.314 | 2 |
| dclm-50p-dolma1.7-50p | mix | 8 | 0.952 | 6 | −1.541 | 7 | −1.364 | 7 |
| falcon-and-cc-qc-tulu-10p | falcon | 9 | 0.948 | 7 | −1.691 | 20 | −1.391 | 16 |
| fineweb-edu | fineweb | 9 | 0.895 | 8 | −1.699 | 21 | −1.371 | 9 |
| dolma1_7-no-code | dolma | 8 | 0.856 | 9 | −1.569 | 9 | −1.393 | 17 |
| dclm-75p-dolma1.7-25p | mix | 10 | 0.841 | 10 | −1.577 | 11 | −1.376 | 11 |
| dclm-baseline-qc-7p-fw2 | dclm | 8 | 0.838 | 11 | −1.663 | 17 | −1.375 | 10 |
| dclm-baseline-qc-fw-10p | dclm | 3 | 0.809 | 12 | −1.723 | 23 | −1.379 | 12 |
| falcon | falcon | 9 | 0.792 | 13 | −1.505 | 5 | −1.379 | 13 |
| dclm-baseline-qc-fw-3p | dclm | 9 | 0.784 | 14 | −1.731 | 24 | −1.352 | 6 |
| falcon-and-cc-qc-orig-10p | falcon | 8 | 0.771 | 15 | −1.648 | 16 | −1.405 | 20 |
| dclm-baseline-qc-20p | dclm | 9 | 0.760 | 16 | −1.606 | 14 | −1.394 | 18 |
| dclm-baseline-qc-7p-fw3 | dclm | 9 | 0.750 | 17 | −1.719 | 22 | −1.364 | 8 |
| falcon-and-cc-qc-20p | falcon | 8 | 0.739 | 18 | −1.679 | 19 | −1.412 | 23 |
| dclm-baseline-qc-10p | dclm | 8 | 0.717 | 19 | −1.614 | 15 | −1.420 | 25 |
| falcon-and-cc-qc-10p | falcon | 8 | 0.713 | 20 | −1.679 | 18 | −1.414 | 24 |
| dclm-baseline | dclm | 8 | 0.666 | 21 | −1.596 | 13 | −1.385 | 14 |
| fineweb-pro | fineweb | 7 | 0.662 | 22 | −1.762 | 25 | −1.411 | 22 |
| falcon-and-cc | falcon | 9 | 0.649 | 23 | −1.581 | 12 | −1.410 | 21 |
| c4 | c4 | 7 | 0.620 | 24 | −1.563 | 8 | −1.388 | 15 |
| dolma1_7-no-math-code | dolma | 4 | 0.618 | 25 | −1.572 | 10 | −1.400 | 19 |

How the new α changes the ranking:

- **The Dolma models rank near the top under both α definitions.**
- **Recipes moved up (high β, flatter data on the probed numbers):** fineweb-edu, dclm-baseline-qc-fw-3p and
  dclm-baseline-qc-7p-fw3. The old α called them very steep (ranks 21–24).
- **Recipes moved down (low β, steeper data on the probed numbers):** c4, falcon-and-cc and
  dolma1_7-no-math-code. The old α called them flat.

## 6. Caveats

1. **Post-hoc choice.** This α was proposed by a blind auditor in E10 (agent 13), but it was adopted after the
   original α was known to correlate weakly, and several slope variants were looked at. A variant that pools
   all probed numbers individually gives r = 0.79, ρ = 0.80, so the two natural definitions agree. Still,
   treat it as a hypothesis to confirm, not a confirmed result.
2. **Only 4 points per model.** The new α is a coarse slope.
3. **It does not fully replicate in E06 (9 gaps).** With E06's own best layers it is r = 0.42, ρ = 0.43, mainly
   because E06's layer choice jumps to layers 2–4 for 7 models, and E06 selects and evaluates on the same seeds.
4. **Content confound.** Code/math (Dolma) recipes still drive part of the relation (see the web-only row).
5. **Counts stop at 10000,** so 10001–10019 of the top group are missing.
6. **Confirm before claiming.** Fix this α definition now and test it on the 6 Paloma models (E09) or on the
   DataDecide checkpoints (E08).

## 7. Suggested wording for a paper

> *Because β is estimated only from the four probed magnitude groups, we also compute the corpus exponent on
> those same numbers (slope of log median count against log group centre). This probed-range α correlates
> with β at each model's selected layer at r = 0.82 (ρ = 0.77, n = 25), compared with r = 0.56 (ρ = 0.46) for α
> fitted over 1–10000, and the relation holds within data families (r = 0.75). This analysis is exploratory and
> awaits out-of-sample confirmation.*
