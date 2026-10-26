# E11: α measured on the numbers that β is measured on

**Setup:** E01 (paper protocol, 3 gaps). β is taken at each model's layer chosen by the variance–monotonicity
joint score.
**Question:** does the α–β relation get stronger if α describes how often the training data contains the
*same numbers* that the model's number line is measured on?
**Status:** done (2026-10-26). **Exploratory:** this α definition was adopted after seeing the original result
(see §8).

**Reproduce (from the repo root):**
```bash
python experiments/E11_alpha_probed_numbers/compute.py       # per_model.csv, results.json
python experiments/E11_alpha_probed_numbers/sensitivity.py   # sensitivity.csv, leave_one_out.csv
```

## Headline

| α used | Pearson r | Spearman ρ | n |
|---|---|---|---|
| Old α: fitted on all integers 1–10000 | 0.56 (p = 0.003) | 0.46 (p = 0.022) | 25 |
| **New α: fitted on the probed numbers only** | **0.82 (p = 7×10⁻⁷)** | **0.77 (p = 7×10⁻⁶)** | 25 |

No model is dropped. Removing any single model leaves r between 0.79 and 0.84 (§6).

---

## 1. How β is measured (unchanged, as in the paper)

1. **Prompts.** The model sees prompts `a=a,b=b,c=c,n=`, where the target n comes from **four magnitude
   groups**: **1–39, 80–119, 980–1019 and 9980–10019** (40 prompts each, 3 random in-context examples).
2. **Hidden states.** The last-token hidden state at every layer is projected on its first principal
   component (PC1), giving the model's internal number line.
3. **Layer choice (variance–monotonicity joint score).** For each layer ℓ and selection seed s:
   - q_{s,ℓ} = √(EV_{s,ℓ} · |ρ_{s,ℓ}|), where EV is the PC1 explained variance and ρ the Spearman correlation
     of PC1 with n;
   - Q_ℓ = median over seeds 42–44 of q_{s,ℓ};
   - ℓ* = argmax_ℓ Q_ℓ, then frozen.
4. **β.** On held-out seeds 45–47, at ℓ*:
   - take the 4 group means on PC1 and the **3 gaps** between neighbouring groups;
   - β is the direct least-squares fit of how the gaps scale from decade to decade;
   - β = 1 is a logarithmic line, β < 1 compresses large numbers more.

**β therefore depends only on how the model places these four groups of numbers.** It never sees 50, 500,
2000 or 7500.

## 2. How α was measured before, and what changes now

Both α measure the same property of the training data: **how fast numbers become rarer as they get
larger**, i.e. the slope of log count c(n) against log n, from exact counts of every integer in each
model's 100B-token training stream (E04). The difference is **which numbers enter the slope**.

| | Numbers used | Points in the fit | What it mostly reflects |
|---|---|---|---|
| **Old α** (paper) | every integer 1–10000 | 10,000 | the **4-digit range 1000–9999**: 9,000 of the 10,000 points are there |
| **New α** | only the probed groups: 1–39, 80–119, 980–1019, 9980–10000 | 4, one per group: log(median count) vs log(group centre) | the frequency of **exactly the numbers β is measured on**, each magnitude weighted equally |

Implementation details:
- **Median count per group,** so a single spike (e.g. the number 10000 itself, 50–200× its neighbours)
  does not dominate a group.
- **Group centre** = geometric mean of the group's numbers.
- **Counts stop at 10000,** so the top group uses 9980–10000.

## 3. Why α should be measured on the numbers used for β

**The hypothesis.** How often numbers appear in pretraining shapes how the model spaces them on its
internal number line: numbers that thin out quickly in the data get squeezed together.

**The logic.**

1. **A test is only as good as the match between its two measurements.** If the hypothesis is "frequency of
   numbers → spacing of those numbers", the frequency and the spacing must refer to the **same numbers**.
2. **β measures the spacing of the four probed groups only.** Its value is set by three gaps: 1–39 → 80–119,
   80–119 → 980–1019, and 980–1019 → 9980–10019.
3. **The old α mostly measures something else.** Fitting over every integer from 1 to 10000 gives the
   thousands 90 % of the weight, so it describes how 4-digit numbers thin out, a region β does not probe
   except at its two edges.
4. **So the old test compares the model's spacing of set A with the data's frequency of set B.** When the two
   disagree, there is no way to tell whether the hypothesis is wrong or the measurements are simply about
   different numbers. Datasets that differ mainly in their 4-digit counts look like counter-examples even if
   the hypothesis is true.
5. **The new α removes that mismatch.** It asks how quickly, in this training data, the numbers that the model
   is tested on become rarer from one group to the next. That is the direct data-side counterpart of the
   three gaps that define β.
6. **Equal weight per magnitude matches β.** β gives each of its 4 groups equal weight. The new α also uses one
   point per group, so the 1–39 group counts as much as the 9980–10000 group, as it does in β.

**Analogy.** Students are tested only on fractions. To ask whether practice predicts performance, count their
*fraction* exercises, not all their math exercises, most of which are algebra.

### Worked example: fineweb-edu vs c4

Median count of the numbers in each probed group:

| | 1–39 | 80–119 | 980–1019 | 9980–10000 | Drop between groups | Old α | New α | β |
|---|---|---|---|---|---|---|---|---|
| fineweb-edu | 13.4 M | 1.29 M | 51 k | 1.9 k | ×10, ×25, ×27 | −1.70 | −1.37 | **0.89** |
| c4 | 11.4 M | 0.93 M | 34 k | 1.5 k | ×12, ×27, ×23 | −1.56 | −1.39 | **0.62** |

- **The old α predicts the wrong order.** It says fineweb-edu's numbers thin out much faster (−1.70 vs −1.56),
  so fineweb-edu should be more compressed. In fact its β is higher (less compressed).
- **The new α predicts the right order.** On the numbers the model is actually tested on, fineweb-edu thins
  out slightly *slower* in the first two steps (×10 and ×25 vs ×12 and ×27). The new α correctly predicts
  that it is less compressed.
- **Where the old α's difference comes from:** the thousands that lie between the probed groups. β never
  measures them.

### What this does and does not claim
- **The old α is not wrong.** It answers a broader question: does the overall number distribution, over
  1–10000, predict compression?
- **The new α answers a narrower, sharper question:** does exposure to the probed numbers predict how those
  numbers are spaced?
- **The hypothesis becomes more local.** What matters is the frequency of the numbers in question, not of
  all numbers. Report both α.

## 4. Results, all 25 models

| α used | Pearson r | Spearman ρ |
|---|---|---|
| Old α (1–10000) | 0.56 (p = 0.003) | 0.46 (p = 0.022) |
| **New α (probed numbers)** | **0.82 (p = 7×10⁻⁷)** | **0.77 (p = 7×10⁻⁶)** |

**Dependence between models.** The 25 recipes are not independent: some are subsets or mixes of others
(E10 estimates 8–11 effective units). Two checks:
- **Family-cluster bootstrap** (resampling whole families): 95 % CI of r = [0.39, 0.90].
- **Family means** (6 points): r = 0.84 (p = 0.035).

## 5. Results when models are removed

Each row removes or keeps a group of models; old and new α are shown side by side. p-values are in brackets;
\* marks p < 0.05.

| Scenario | n | Old α: r | Old α: ρ | **New α: r** | **New α: ρ** |
|---|---|---|---|---|---|
| All 25 models | 25 | 0.56\* | 0.46\* | **0.82\*** | **0.77\*** |
| **Measurement-quality removals (E10)** | | | | | |
| Remove dolma1_7-no-reddit (its β does not replicate) | 24 | 0.50\* | 0.39 | **0.79\*** | **0.75\*** |
| Remove all 4 E10 measurement flags¹ | 21 | 0.50\* | 0.38 | **0.79\*** | **0.70\*** |
| **Removing Dolma** | | | | | |
| Remove the 6 Dolma models | 19 | 0.22 | 0.11 | **0.68\*** | **0.66\*** |
| Remove the 3 DCLM–Dolma mixes | 22 | 0.53\* | 0.30 | **0.80\*** | **0.71\*** |
| Remove all 9 Dolma-containing models (Dolma + mixes) | 16 | −0.31 | −0.33 | 0.43 (p = 0.10) | 0.53\* |
| Remove all 8 recipes with code or math | 17 | −0.38 | −0.41 | 0.44 (p = 0.08) | 0.52\* |
| **Removing other families** | | | | | |
| Remove DCLM (7) | 18 | 0.59\* | 0.59\* | **0.86\*** | **0.82\*** |
| Remove Falcon (6) | 19 | 0.63\* | 0.62\* | **0.83\*** | **0.81\*** |
| Remove FineWeb (2) | 23 | 0.57\* | 0.48\* | **0.81\*** | **0.76\*** |
| Remove C4 (1) | 24 | 0.61\* | 0.52\* | **0.82\*** | **0.80\*** |
| **Subsets only** | | | | | |
| Only the 6 Dolma models | 6 | 0.75 | 0.71 | 0.80 (p = 0.054) | 0.54 |
| Only Dolma-containing models (9) | 9 | 0.74\* | 0.80\* | **0.82\*** | **0.73\*** |
| Only base corpora, no filtered/ablated variants² | 8 | 0.55 | 0.50 | **0.92\*** | **0.90\*** |
| Only quality-filtered variants (DCLM-qc + Falcon-qc) | 10 | −0.27 | −0.33 | 0.35 | 0.65\* |
| **Within families** (family means removed) | 25 | 0.28 | 0.24 | **0.75\*** | **0.75\*** |

¹ dolma1_7-no-reddit (β does not replicate), plus falcon-and-cc-qc-10p, dolma1_7 and fineweb-edu, whose
best layers are near-ties giving different β.
² c4, fineweb-pro, fineweb-edu, falcon, falcon-and-cc, dclm-baseline, dolma1_7, dolma1_6plus: one recipe per
original corpus, avoiding near-duplicate data.

### What the removals show

1. **With the old α, the relation is carried by Dolma.**
   - Removing the 6 Dolma models drops it to r = 0.22 (not significant).
   - Removing all Dolma-containing or code/math recipes makes it *negative* (r ≈ −0.3 to −0.4).
   - So the old α–β link was essentially "Dolma vs the rest", which E05 already flagged as a confound.
2. **With the new α, the relation survives removing Dolma.**
   - Without the 6 Dolma models: r = 0.68, ρ = 0.66, both significant.
   - Without all 9 Dolma-containing recipes it is weaker but still positive: ρ = 0.53 (p = 0.04),
     r = 0.43 (p = 0.10).
   - **The new α explains differences among web-only datasets that the old α got backwards.**
3. **It holds within families:** r = 0.75 after removing each family's mean, against 0.28 for the old α. It is
   not driven by any single family: removing DCLM, Falcon, FineWeb or C4 leaves r = 0.81–0.86.
4. **It is strongest on independent corpora.** On the 8 base corpora, with no near-duplicate filtered
   variants, r = 0.92 and ρ = 0.90. The old α gives 0.55 / 0.50 there.
5. **Measurement-quality removals change little.** Removing the 4 models flagged by the blind audit (E10)
   leaves r = 0.79, ρ = 0.70.
6. **The weakest subset is the quality-filtered variants alone** (10 models; r = 0.35, ρ = 0.65). These are
   classifier-filtered subsets of only two base corpora, with a narrow spread of α.

## 6. Removing one model at a time (leave-one-out)

| | Pearson r: min–max | Spearman ρ: min–max |
|---|---|---|
| New α | **0.79–0.84** | **0.75–0.80** |
| Old α | 0.50–0.61 | 0.39–0.52 |

No single model creates the new result.
- **Largest single drop:** removing dolma1_7-no-reddit (r 0.82 → 0.79).
- **Largest single rise:** removing falcon-and-cc-qc-tulu-10p (r → 0.84).

The full table is in `leave_one_out.csv`.

## 7. Per model (sorted by β; rank 1 = highest β or flattest α)

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

- **The Dolma models are at the top under both α definitions.**
- **Recipes that moved up** (high β; their data thins out slowly on the probed numbers): fineweb-edu,
  dclm-baseline-qc-fw-3p and dclm-baseline-qc-7p-fw3. The old α ranked them 21–24, so they looked like
  counter-examples; the new α ranks them 6–9.
- **Recipes that moved down** (low β; their data thins out quickly on the probed numbers): c4, falcon-and-cc
  and dolma1_7-no-math-code. The old α ranked them 8–12; the new α ranks them 15–21.

## 8. Caveats

1. **Post-hoc choice.** This α was suggested by a blind auditor in E10 (agent 13, who never saw any α–β
   correlation), but it was adopted after the original α was known to correlate weakly, and several slope
   variants were looked at.
   - A second natural variant agrees: pooling every probed number individually gives r = 0.79, ρ = 0.80.
   - Single-gap slopes do not: r = 0.05–0.66.
   - **Treat it as a hypothesis to confirm, not as a confirmed result.**
2. **Only 4 points per model,** so the new α is a coarse slope.
3. **It is weaker in E06** (10 groups, 9 gaps) at E06's own best layers: r = 0.42, ρ = 0.43. E06's layer choice
   jumps to layers 2–4 for 7 models, and E06 selects and evaluates on the same seeds. At E01's layers, E06
   gives r ≈ 0.70.
4. **The content confound is reduced, not gone.** Without any code/math recipes, the relation is positive but
   moderate (ρ = 0.52–0.53).
5. **Counts stop at 10000,** so 10001–10019 of the top group are missing.
6. **Next step:** fix this α definition now and test it on data it was not chosen on: the 6 Paloma models (E09)
   or the DataDecide checkpoints (E08). Alternatively, probe numbers spread over all of 1–10000, so that the old
   and new α describe the same range.

## 9. Suggested wording for a paper

> *Because β is estimated only from four probed magnitude groups, we measure the corpus exponent on those same
> numbers (slope of log median count against log group centre, one point per group). This probed-range α
> correlates with β at each model's selected layer at r = 0.82 (ρ = 0.77, n = 25), compared with r = 0.56
> (ρ = 0.46) for α fitted over 1–10000. Unlike the full-range α, it holds within data families (r = 0.75), on
> the 8 independent base corpora (r = 0.92), and without the Dolma models (r = 0.68). Every leave-one-out
> correlation lies between 0.79 and 0.84. This analysis is exploratory and awaits out-of-sample confirmation.*
