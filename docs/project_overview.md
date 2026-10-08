# Project overview

## The research question

Language models place numbers along a "number line" inside their hidden states, and
that line is **compressed**: large numbers sit closer together than small ones,
roughly like a logarithmic scale. Our question is **why the compression is as strong
as it is in a given model**, and specifically:

> Is a model's number-line compression (**β**) set by the statistics of the numbers in
> the data it was trained on (**α**, and related measures)?

To answer it without other differences getting in the way, we use the **25 DataDecide
1B models** from Ai2 (`allenai/DataDecide-<recipe>-1B`). All 25 share the same OLMo
architecture, size (1B), tokenizer, training length (100B tokens) and code. The **only**
thing that differs between them is the pretraining data ("recipe"). So any difference in β
between them comes from the data or from training randomness.

The project has two sides:

| Side | What is measured | Where |
|---|---|---|
| **Data side** | How often each integer 0–10000 appears in the exact 100B tokens each model trained on: α, numbers seen, rarely-seen count R, … | experiments E02–E04, E07 |
| **Model side** | How compressed the model's internal number line is: β | experiments E01, E06 |
| **Link** | Which data measure predicts β, and how strongly | experiments E05, E07 (E08 planned) |

## Definitions (used everywhere in this repo)

### Counting numbers in text
- **Counting rule** (identical in every experiment): regex `\b\d+\b` on the text, leading
  zeros stripped, keep integers **0..10000**. Code: `_count_text` in
  `modal_app/corpus_alpha_app.py`; every other counter imports it.
  `count(3)` does **not** include the 3 inside `30` or `123`.
- **c(n)**: how many times integer *n* appears.
- **Numbers seen** (`integer_matches`): Σ c(n) over 0..10000 in a model's 100B-token
  training stream (1.0–2.0 billion for the 25 recipes).

### α: the slope of the number-frequency curve (data side)
Small numbers are far more common than large ones; on a log–log plot the counts fall
roughly on a line.
- **α_OLS** (the paper's definition): least-squares slope of log c(n) vs log n over
  n = 1..10000 with c(n) > 0. Code: `_fit_alpha`. Values for our recipes: −1.43 to −1.76.
  **Biased on small samples** (it drops zeros; see E02), reliable from ≈1B+ tokens.
- **α_MLE**: maximum-likelihood exponent of a discrete power law
  p(n) = n^a / Σ_{m=1..10000} m^a, fitted on all counts including zeros. Code:
  `src/sampling_validation.py`. Unbiased even on small samples. Values: −1.05 to −1.19.
- The two fits weight different parts of the curve (MLE is dominated by the very common
  small numbers, OLS gives the ~9000 numbers above 1000 equal weight), so they are not
  interchangeable and can even rank recipes differently.
- α measures **shape only**: multiplying every count by a constant leaves it unchanged.

### R: the rarely-seen count (data side; the current best predictor, E07)
$$R = \#\{\, n \in \{10,\dots,9999\} : c(n) < K \,\},\qquad K = 4000$$
How many of the numbers in the probed range the model saw fewer than 4000 times.
It combines **shape and amount**. Code: `src/count_features.py`, feature
`F14_n_seen_lt_4000`; the exact definition is pinned by
`tests/test_count_features.py`. Range over the 25 models: 5,707 to 7,735.
Its smooth cousin is the **saturation sum** S = Σ c(n)/(c(n)+K), the Bayesian-shrinkage
weight (see `docs/findings.md`).

### β: compression of the model's number line (model side)
1. Prompts like `a=a,b=b,c=c,n=` (in-context copying), with the target *n* drawn from
   magnitude groups.
2. A **correct-output filter** keeps a prompt only if the model's greedy continuation
   starts with *n* (a rejected slot is resampled from the same group).
3. Take the **last-token hidden state** at every layer, run **PCA**, keep PC1 (oriented
   to increase with *n*).
4. Average PC1 per group, giving group means; consecutive differences are the gaps
   d₁, d₂, d₃, …
5. Fit a geometric sequence **d_i = s·β^i**:
   - **β < 1**: each decade gets less space than the one before, i.e. compression;
   - **β = 1**: equal space per decade, i.e. a logarithmic line;
   - **β > 1**: large numbers spread out.

Variants used in this repo:

| Name | Groups | Fit | Experiment |
|---|---|---|---|
| `beta_direct` (paper protocol) | 4 groups near 10, 100, 1000, 10000 | least squares on the 3 gaps (paper Eq.) | E01 |
| `beta_log` | same | exp(slope of log gaps) = √(d₃/d₁) for 3 gaps | E01 |
| `beta_coarse` | the 4 decade groups inside the 10-group run (±10 % bands) | as `beta_direct`, reported per decade | E06 |
| `beta_fine` | 10 groups, one every ⅓ decade | geometric fit on 9 gaps, converted to per decade (b³) | E06 |
| `beta_cont` | all 3000 prompts individually | y = a + s·B^u, u = log10(n) − 1 | E06 |
| `*_oldlayer` | E06 values measured at the layer E01 selected | | E06 |

**Fit quality:** R² of the gap fit is meaningless near β = 1 (all gaps equal means
nothing to explain, so R² ≈ 0 even for a perfect fit). E06 therefore reports the
relative error `err` = RMS residual / mean gap instead.

### Layer selection
For every layer: EV = variance explained by PC1, ρ = Spearman(PC1, n).
Score = √(EV × |ρ|); take the median over seeds; **pick the layer with the highest
score** (`select_joint_layer`, `src/datadecide_sweep.py`). EV alone is not enough,
because in some layers the dominant direction tracks digit count or prompt format
rather than magnitude; ρ makes sure the direction is ordered by size.

## The 25 DataDecide recipes and their families

| Family | Recipes | What differs inside the family |
|---|---|---|
| dolma (6) | dolma1_7, dolma1_7-no-code, -no-math-code, -no-reddit, -no-flan, dolma1_6plus | whole sources removed (ablations) |
| mix (3) | dclm-baseline-{25,50,75}p-dolma1.7-{75,50,25}p | DCLM/Dolma blend ratio (dose-response) |
| dclm (7) | dclm-baseline + qc-7p-fw2, qc-7p-fw3, qc-fw-3p, qc-fw-10p, qc-10p, qc-20p | quality filter only (same web crawl) |
| falcon (6) | falcon, falcon-and-cc + qc-10p, qc-20p, qc-orig-10p, qc-tulu-10p | quality filter (mostly same pool) |
| fineweb (2) | fineweb-pro, fineweb-edu | processing of FineWeb |
| c4 (1) | c4 | — |

The family mapping used in code is `FAMILY` in `src/decade_metrics.py`. The list of
recipes, model repo template and revision (`step69369-seed-default` = final
checkpoint) is in `configs/datadecide_models.json`; the recipe → training-file map is
in `configs/datadecide_data_map.json` (see `docs/datadecide_sampling.md`).

## How DataDecide trained (needed to reproduce "the exact 100B tokens")
Full details with file and line references: `docs/datadecide_sampling.md`. In short:
- each recipe is a list of headerless uint16 token files (`.npy`), EOS = 50279;
- each file is cut into 2048-token chunks, and each file's leftover tail is dropped;
- all chunks are shuffled once with `numpy PCG64(seed)`;
- training reads the first **69,369 steps × 704 sequences** = 48.8M chunks = **100.016B
  tokens** (final checkpoint `step69369`);
- the data-order seed of the released models is **most likely 2** (run names end in
  `-2`); seeds 2 and 6198 give the same α to < 0.002 (E04), so the choice doesn't matter.
