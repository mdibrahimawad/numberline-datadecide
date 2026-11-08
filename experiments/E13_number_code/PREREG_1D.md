# E13-1D: is there a 1-D manifold that encodes the number line? (pre-registered 2026-11-08, before running)

**Owner's question (professor).** Prove that there is a 1-D manifold that encodes numbers internally.

**Claim tested** (precise version):
> Inside a model's hidden state there is a **1-D curve** along which numbers sit **in order**, that is **continuous**,
> **generalises** to numbers not used to build it, is the **same across contexts** and **across models**, and
> **forms during training**.

The claim is not that the whole vector is 1-D; that is already known to be false (E12, E13 Stage A).

Every test below can fail. All results are reported, passed or not. Thresholds are fixed here, before any test is run
on real data. The causal tests 7–9 (necessary, sufficient, explains behaviour) run in the same overnight laptop run, after the
shape tests (owner's decision, 2026-11-08). Code: `src/e13/manifold1d.py`, runner `run_1d.py`. Results are labelled PRIMARY (rules below) or
EXPLORATORY.

## Data ("two prompt types")
- **New prompts** (E13 extraction, Pythia-1.4B and Qwen2.5-1.5B):
  - **Sentences**: the number's last token, averaged over the 5 neutral sentences C2, C3, C4, C7, C8. Every layer.
  - **Copy · number**: the number's last token in the copy prompt. 9 layers.
  - For test 5, the 8 single contexts C1, C2…C8 at the number token.
- **Old prompt** (E10 vectors, all 9 models): the final "=" of the copy prompt `A=A,B=B,C=C,D=D,n=`. 5 layers
  (~25/50/75/84 %/last).
- **Checkpoints** (E11, old prompt): Pythia-1.4B (16 steps) and OLMo-2-1B (12 steps), numbers 1–100 and 2,000 4-digit
  numbers, 5 layers.
- **Sets** (never mixed):
  - 10–99, 100–999, 1000–9999 (4-digit sets use the fixed 3,000-number subsample, seed 0);
  - chunk tokenizers: one-token numbers only;
  - the checkpoint data has 10–99 and its 2,000 4-digit numbers.

## The curve (same construction in every test)
1. **Preparation.** Centre, z-score, PCA to 50, all fitted on the training numbers only.
2. **f(n)** = Gaussian kernel smoother of the training vectors over n. Bandwidth by 5-fold CV from
   {1, 2, 5, 10, 20} % of the range (minimum 1).
3. **Magnitude subspace S** = the top-m principal directions of f, where m is the smallest number with ≥ 90 % of f's
   variance, at most 5.
4. **The curve** γ(g) = f(g) projected into S, on the integer grid g = n_min … n_max.
5. **Position of a vector z** = n̂(z) = the grid value g with the nearest γ(g) to z (projected into S).

## Tests (PRIMARY)
| # | Test | Measure | Pass |
|---|---|---|---|
| T1 | **Order** | Spearman(n̂, n) on held-out numbers, random 80/20, 3 repeats | mean ≥ 0.90 |
| T2 | **One coordinate carries the magnitude** | R² of predicting held-out n from n̂ (one number per vector) vs the best of a 10-nearest-neighbour regressor and a ridge probe on all 50 dimensions | R²(n̂) ≥ 0.80 **and** ≥ 0.90 × the best flexible decoder |
| T3 | **Continuity (no tears)** | Consecutive blocks of b = max(3, 5 % of range) numbers; noise-debiased distance between neighbouring block centroids; tear ratio = median step across a leading-digit boundary / median step inside a leading digit | ratio ≤ 1.5, with bootstrap 95 % upper bound ≤ 2.0 (500 resamples) |
| T4 | **Generalises to an unseen block** | For each of three blocks (leading digit 3, 5, 7: e.g. 300–399), build the curve without the block and place its numbers | A block passes if within-block Spearman(n̂, n) ≥ 0.80 **and** ≥ 80 % of its numbers land within half a block-width of the block. T4 passes if ≥ 2 of 3 blocks pass |
| T5 | **Same curve across contexts** (new prompts) | Curve built in context A (8 contexts, each centred on its own training mean, A's standardisation); held-out numbers of context B placed on it | Spearman ≥ 0.90 for ≥ 70 % of the 56 ordered pairs (A ≠ B) |
| T6 | **Same across models** (old prompt, 9 models; new prompts, 2 models) | Per (model, set): "1-D manifold" = T1, T2 and T4 pass at the chosen layer | holds in ≥ 70 % of (model, set) cells in both tokenizer families (digit-splitting; chunk) |
| T10 | **Forms during training** (checkpoints) | T1 order at each checkpoint (layer fixed = the chosen layer of the final checkpoint); emergence = the first step with order ≥ 0.90. Compared with the first step where copy accuracy ≥ 0.50 (E11 behaviour log) | emergence at the same or an adjacent checkpoint as the behaviour, in both Pythia and OLMo-2 |

## Causal tests (PRIMARY; run on the owner's laptop GPU (Apple MPS) or CPU after T1–T4)
- **Models and data.** Pythia-1.4B and Qwen2.5-1.5B; the "Sentences" reading.
- **The curve's subspace S.** Built as above on training (non-dev) numbers at the dev-chosen layer. Interventions
  act at the number's last token, after that layer, in the behaviour prompts.
- **Items.** 200 test items per behaviour, from non-dev numbers of 10–99 and 100–999 (one-token numbers for Pythia).
- **Random control.** 3 random subspaces of the same dimension, orthogonal to S.

| # | Test | Measure | Pass |
|---|---|---|---|
| T7 | **Necessary** | Remove S (mean-projection ablation). Effect = the drop of the log-prob margin on "Which is larger, a or b? Answer:" (B1). Off-target: "The last digit of n is" (B2) | B1 drop ≥ 2 × the random drop (bootstrap CI of B1 − 2·random above 0), **and** B2 drop ≤ 1.2 × random (CI of B2 − 1.2·random below 0) |
| T8 | **Sufficient** | Move the number along the curve: add γ(n′) − γ(n) (mapped back to the model's space), where n′ is on the other side of the comparison partner. Score = the share of B1 items whose answer flips towards n′; control = a random direction of the same length | flip share ≥ 2 × the random flip share, bootstrap CI excluding equality |
| T9 | **Explains behaviour** (no intervention) | Spearman between the B1 margin and the distance between a and b **along the curve**, |n̂(a) − n̂(b)|, as a partial correlation controlling |log a − log b| | partial correlation > 0, bootstrap 95 % CI above 0 |

## Layer choice (no cherry-picking)
- **Dev split.** 20 % of the numbers (seed 13) is set aside. For each (model, reading, set), the layer with the highest
  T1 order **on dev numbers** is chosen.
- **Evaluation.** All tests are then evaluated on the other 80 % at that layer.
- **All layers.** T1, T2, T3 and T4 are also reported at every layer.

## Controls and power (G4)
- **Shuffled n.** The curve is built with n permuted among the training numbers; T1, T2 and T4 must fail.
- **Planted structure at the real noise level** (per model, reading and set, at the chosen layer, 10 replicates).
  T = the real smooth part f(n); noise = the real residual rows (Z − f) permuted across numbers.
  - **Planted curve** T + noise: T1, T2 and T4 should pass and T3 should find no tear.
  - **Planted tear** T + a jump at every leading-digit boundary (size 3 × the median within-digit block step) + noise:
    T3 should detect it.
- **Power rule.** A test has **power** if it recovers the planted curve (T1, T2, T4) or detects the planted tear (T3)
  in ≥ 80 % of replicates. Without power the result is "no power" and no claim is made from it.

## Verdict
- **Per model and reading.** "A 1-D number manifold is supported" if T1, T2 and T4 pass, with power, in ≥ 2 of the
  3 sets.
- **Reported alongside:** T3 (continuous or torn) and, for the new prompts, T5.
- **Across models:** T6.
- **Training:** T10.
- **Causal:** T7, T8 and T9 per model; the curve "encodes the number line for the model" if T7 and T8 pass in both models.

## Amendment 1 (2026-11-08, from the synthetic check, before any real data was tested)
- **Problem.** On planted data, a curve built with **shuffled** n still gave a high order (Spearman −0.74 to +0.93 over
  5 shuffles). The shuffled curve collapses to a few points, and which way they line up is luck. R² stayed ≤ 0.31, so
  T2 is not fooled, but "order ≥ 0.90" alone is not specific.
- **Fix.** T1 now passes only if order ≥ 0.90 **and** the order is above the 95th percentile of 20 shuffled-n curves,
  computed at the chosen layer. The shuffled curves also report how often T2 would pass by chance (`T2_false_pass`).
- Nothing else changed.
