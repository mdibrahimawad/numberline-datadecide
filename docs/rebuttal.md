# AAAI reviews of paper 42639: what the new experiments answer

The paper (*Pretraining Numerical Frequencies…*, submission 42639) was reviewed by H7EL, FoZw, VRQv and a1nj. This
page maps the reviewers' concerns to the experiments in this repository. Reviewer IDs follow our reading of the
reviews; check them against the reviews before quoting.

**Main evidence: E01, E06 and E11.** A 3-page write-up, one page per experiment, is in
[`paper/new_experiments_body.tex`](../paper/new_experiments_body.tex). It is body-only LaTeX to paste between
`\begin{document}` and `\end{document}` and needs only `amsmath`. A longer, self-contained version (all
experiments, figure, per-model appendix) is [`paper/new_experiments.tex`](../paper/new_experiments.tex).

**Framing.** The 25 DataDecide models are a *new* set of models, not a re-analysis of the paper's 22. Present
them as new controlled evidence.

## The three main experiments

| Experiment | What it does | Key result | Reviewer concern it answers |
|---|---|---|---|
| [E01](../experiments/E01_datadecide_beta_paper_protocol/) | Paper's exact β protocol on 25 models that differ only in training data (same architecture, tokenizer, size, 100B tokens) | α–β: r = 0.56 (p = 0.003), ρ = 0.46 (p = 0.022), n = 25; the paper had 6 corpora, r = 0.81, permutation p = 0.061 | Confounding by architecture / tokenizer / size (H7EL, VRQv, a1nj); "only 6 corpora"; partial correlation for scale (a1nj; scale is constant) |
| [E06](../experiments/E06_fine_beta/) | β from 10 magnitude groups (9 gaps), 100 prompts per group, 3 seeds, one continuous fit | β seed s.d. 0.015; median R² 0.89; agrees with the 3-gap β at r = 0.96. α–β: 0.63 at E01's layers, 0.20 with layers re-chosen (near-ties), 0.75 with all models at layer 8; significant at every layer 6–11 | β fitted from few points, stability, fit quality (a1nj point 3); sensitivity to the layer |
| [E11](../experiments/E11_alpha_probed_numbers/) | α measured on the same numbers β is measured on (log median count against log group centre, 4 groups) | r = 0.82, ρ = 0.77 (n = 25, none dropped); leave-one-out 0.79–0.84; without Dolma 0.68; within families 0.75; 8 base corpora 0.92; family bootstrap 95% CI [0.39, 0.90] | Sensitivity / omitting each corpus (VRQv Q2, H7EL); influence of Dolma; stratifying by family (a1nj Q1); uncertainty |

### Numbers to state alongside the main results

These are needed for an honest rebuttal.

- **E01 is driven by Dolma.** Without the 6 Dolma models, α_OLS–β drops to r = 0.22 (p = 0.36).
- **E06 layer choice.** For 8 models the two best layers score within 8% of each other and give different β;
  that is why α–β falls to 0.20 when E06 re-chooses layers.
- **E11 is exploratory.** This α definition was adopted after α_OLS was seen to correlate weakly.
- **E11 and content.** Without the 8 code/math recipes: r = 0.44 (p = 0.08), ρ = 0.52 (p = 0.03). Controlling
  for code + math share: partial r = 0.30 (p = 0.15).
- **E11 at a fixed layer.** At layer 8 the new α is no better than α_OLS (r = 0.75 for both; E11 README §8).

## Supporting experiments

| Experiment | Use in the rebuttal |
|---|---|
| [E04](../experiments/E04_exact_100b_alpha/) exact 100B counts | How counts were made (VRQv Q3). Counts come from each model's exact training stream up to the evaluated checkpoint (step 69,369), so repeated data counts as often as it was seen and no sampling weights are needed. Sampling noise in α ≤ 0.003, against 0.33 between recipes |
| [E10](../experiments/E10_exclusion_audit/) blind exclusion audit | Inclusion/exclusion criteria (H7EL, VRQv Q1). 21 blind auditors set rules before seeing α–β; only 4 models flagged; no valid exclusion strengthens the result |
| [E05](../experiments/E05_alpha_vs_beta/) confounders | Partial correlations (a1nj). Number density r = 0.47; code/math share r = 0.85/0.76 (a Dolma indicator); α_OLS controlling for density + code: −0.20. A limitation, and the motivation for E11 |
| [E09](../experiments/E09_paloma_out_of_sample/) Paloma | Out-of-sample test of the DataDecide formulas on 6 new models: only 3 valid at layer 8 and no formula transfers. A negative result; do not cite as support |

## Not addressed by any experiment

- **Causality / training dynamics (FoZw Q2).** Everything is correlational. E08 (the checkpoint test) is
  pre-registered but not run.
- **Novelty versus prior work, prompt clarity, notation.** Text changes only.
- **The paper's own 22 models,** including the ambiguous StarCoder controls. DataDecide does not touch them;
  only the paper's supplement does.

## Possible next step

The new α can be tested out of sample without new GPU runs. The Paloma counts already exist
(`results/paloma_counts/`), so α on the probed numbers can be computed for those corpora and compared with their
measured β (E09). That would be the first test of the E11 definition on data it was not chosen on.
