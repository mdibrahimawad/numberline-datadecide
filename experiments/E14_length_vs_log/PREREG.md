# E14 pre-registration: is the "log number line" just digit length?

Status: **frozen before any real data was analysed** (commit that adds this file). Synthetic power checks
(fake data only) were run before freezing and are reported in §7. Code: `src/e14_length_log.py`.

## 1. The claim being tested

A colleague (Prof. Hilal) showed that if every number is written as a one-hot digit table (8 slots × 11
symbols, right-aligned, blank symbol for empty slots) and PCA is run on numbers sampled from 1–9,
10–99, …, PC1 is close to linear in log10(n). The reason is that PC1 picks up **how many slots are
blank**, i.e. the digit length, and each ×10 adds one digit. He concludes that digit length "is
indeed the reason" for the logarithmic spacing seen in LLM number representations (our β).

**H_len (the professor's claim):** on the number-line axis, the log-like spacing comes from digit
length. The axis position is a staircase: a step at each length boundary, no log compression inside
a length.

**H_log (alternative):** the axis is a smooth log line. There is no extra step at length boundaries,
and there is log compression inside a length.

**H_pl (alternative):** smooth across boundaries (equal space per decade, no extra step), but linear
inside each length.

The three agree on how decades are spaced. They differ only (a) at the boundary 99→100 / 999→1000
and (b) inside a length. The tests look only there.

## 2. Data (all existing, nothing new is generated)

| ID | What | Models | Reading | Numbers |
|---|---|---|---|---|
| D1 | E06 β data, `results/beta_fine/step69369-seed-default/*.npz` | 25 DataDecide 1B | copy prompt `a=a,b=b,c=c,n=`, last token (the β protocol), correct-copy filter | 10 bands ±10 % around 10, 22, 46, 100, 215, 464, 1000, 2154, 4642, 10000; 100 prompts each; 3 seeds |
| D2 | E06 Paloma, `results/beta_fine/step35000-unsharded/*.npz` | 6 Paloma 1B baselines | same | same |
| D3 | E13 vectors (`mdibrahimawad/e13-number-code`) | Pythia-1.4B, Qwen2.5-1.5B | (i) copy `…,n=` at "=" (`vec_C1_next`, correct copies only), (ii) sentence reading AVG5 at the number's last token (`vec_avg_num_alllayers`) | every integer 10–9999 |

Tokenization (checked before freezing): the DataDecide/Paloma tokenizer and Pythia's split numbers identically.
90–99 and 100–110 are single tokens (100 %), 900–999 single 48 %, 1000–1100 single 14 %, 9000–11000 two
tokens. Qwen: one token per digit.

## 3. The coordinate

- D1/D2: the stored PC1 of each seed and layer (PCA fitted on that seed's 1,000 prompts, oriented to
  increase with log10 n), exactly as β is computed.
- D3: PCA fitted on a **length-balanced subset** (90 random numbers each of 2, 3 and 4 digits), then
  every number 10–9999 is projected on PC1 (oriented with log10 n). Repeated for 10 random subsets;
  the point estimate is the median. Each length gets equal total weight in all fits (weight 1/count).

## 4. Measures (exact formulas)

u = log10 n, L = number of digits, x = n / 10^(L−1) ∈ [1, 10).

1. **LS, length share (primary).** Weighted least squares y = c_L + s_in·u (one intercept per
   length, one common log slope). At the boundary L→L+1, the extra step is d = c_{L+1} − c_L, and
   LS = d / (s_in + d): the share of one decade's rise that happens as a jump at the boundary.
   **LS = mean of LS at 100 and at 1000** (primary boundaries). LS at 10 and 10000 (D1/D2 only)
   is secondary. Smooth line: LS ≈ 0. Pure staircase: LS ≈ 1.
2. **J, local jump (secondary, model-free check of LS).** Bands [0.9B, B) and [B, 1.1B];
   J_B = (ȳ_up − ȳ_low)/s_all − (ū_up − ū_low), with s_all the slope of y on u over all points.
   J = mean of J_100 and J_1000. Smooth: ≈ 0. Staircase: ≈ 1.
3. **λ, shape inside a length.** y = c_L + b·g_λ(x), with g_λ the Box-Cox transform; λ chosen by
   least squares on the grid −1.00, −0.95, …, 2.00. λ ≈ 0 means log inside a length; λ ≈ 1 means linear.
4. **R, squeeze ratio (descriptive, D3 only).** Within 3- and 4-digit numbers,
   (ȳ[2xx] − ȳ[1xx]) / (ȳ[9xx] − ȳ[8xx]). Log ≈ 4–5; linear ≈ 1.
5. **ρ**: |Spearman(PC1, n)| (D3: on the balanced subset). ρ < 0.5 → "no number line"; the
   cell is excluded from verdict counts but reported.

**Confidence intervals (95 % percentile bootstrap).**
- D1/D2: resample prompts within each band, separately per seed, average over the 3 seeds; B = 500.
- D3: resample numbers within each length and refit the PCA every time; B = 200.

## 5. Classification of each model × reading (at the primary layer)

| Class | Rule |
|---|---|
| no number line | ρ < 0.5 |
| can't tell | LS CI wider than 0.5 |
| **length staircase** (professor right) | LS CI lower bound ≥ 0.5 |
| **smooth** (professor wrong): log inside / linear inside / shape unclear | LS CI upper bound ≤ 0.25; then λ CI upper < 0.5 → log inside; λ CI lower > 0.5 → linear inside; else shape unclear |
| mixed | everything else (report LS as the length share) |

**Primary layers.**
- D1/D2: the E06-selected layer (√(EV·|ρ|) rule, stored in the json). Secondary: fixed layer 8 (the
  E09 convention).
- D3: the layer with the highest √(EV·ρ) among stored layers ≥ 1, for each reading.
- All layers are reported as point estimates (no class) in the CSVs.

## 6. Verdict rules (decided now)

**Main verdict (D1, the 25 β models, primary layer).** Among models with a number line:
- **Professor right**: ≥ 2/3 are "length staircase".
- **Professor wrong**: ≥ 2/3 are "smooth" (any subtype). "Log inside" vs "linear inside" says what
  the β log pattern is instead.
- **Partly right**: otherwise, with the median LS across models reported as "the share of the log
  pattern due to digit length".

**Replications.**
- D2 (6 Paloma models): same rule, read as a replication.
- D3: per model × reading, the class at the primary layer. Two models, so no count rule; stated as
  agreeing or disagreeing with D1.

**Tokenizer sub-test (where length comes from).** In the NeoX-tokenizer models (D1, D2, Pythia),
99→100 keeps one token on both sides, so the input does not show the extra digit. The input-format
version of H_len predicts LS_100 CI upper ≤ 0.25 while LS_1000 CI lower ≥ 0.5. If both are ≥ 0.5,
length is learned rather than given by the input. Reported per model as counts.

## 7. Power (synthetic, run before freezing; `results/e14/power_synthetic.csv`)

Fake hidden states (64-dim) = right-aligned one-hot digits (5 slots × 11 symbols) + a planted
structure × 3 along a random direction + Gaussian noise (sd 0.1, 0.3, 0.6 per dimension). Plants:
- professor's toy (no plant);
- length staircase;
- smooth log;
- smooth, linear inside;
- mixed (37.5 % of the rise as a length step).

Sampling copies D1 (bands, 3 seeds) and D3 (every number). 20 repetitions. A test counts as having
power at a noise level if each plant gets its expected class in ≥ 80 % of repetitions. Results are
filled in below at freezing time.

POWER_TABLE_PLACEHOLDER

## 8. What this can and cannot show

- It decides whether, on the β axis, the log pattern is a length staircase, a smooth log line,
  smooth but linear inside a length, or a mix (with the share).
- It does not show *why* a smooth log line exists, if one does (frequency, Benford, training).
- PC1 is the axis β and the professor's toy both use. If PC1 is dominated by something else in a
  model, ρ flags it.
- Anything outside the pre-registered layers and readings is labelled exploratory.
