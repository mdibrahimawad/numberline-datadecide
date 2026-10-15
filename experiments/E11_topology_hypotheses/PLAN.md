# E11: Topology of numbers in LLMs: 29 hypotheses from recent findings (plan)

**Written 2026-10-15, before any E11 analysis was run.** Every hypothesis below starts from a
published observation (or one of ours) whose *why* is open, proposes what topology might show,
names the metric, and states the predicted direction. Status of each run goes into `README.md`.

Rules (as in `docs/handover.md`): hypotheses tested on the 7 models already seen in E10 are
**exploratory**; anything that looks real is locked and confirmed on new data (new prompts, new
models, checkpoints) before being claimed. Inference is done per *cell* (model × layer × condition),
not per overlapping subsample (Malhotra et al. 2026, App. C.1).

## Data tiers
| Tier | Data | Where | Cost |
|---|---|---|---|
| 0 | existing vectors: 7 models × 5 relative depths, numbers 1–10,000, one prompt each (`topology_number_line/data/`) | here | $0 |
| 1 | new extraction, final models: 5 prompts per number; random-initialised copies of the 7 architectures; a "which is larger?" behaviour probe; all layers | owner's laptop (MPS) | $0, ~1–2 h |
| 2 | training checkpoints: Pythia-1.4B (~16 of its 154 checkpoints, log-spaced) and OLMo-2-1B (~12) | RunPod GPU (much faster downloads) or laptop overnight | ~$2–5 on RunPod |

## Topological metrics used (all on Vietoris–Rips, Euclidean, H0 + H1)
- **Half-merge radius** r½ = median H0 death; **H0 total persistence** = MST length (spacing / compression).
- **41-feature barcode vector**: ours (`src/topo_features.py`) and Malhotra et al.'s list (`h0_death_p90`,
  `h1_pers_gap`, `h1_h0_bar_ratio`, …; MIT code, credited). Both reconstructions of Fay et al.'s
  unpublished list.
- **Betti curves**, **persistence landscapes** (L² distances), **Wasserstein velocity** between layers or
  checkpoints, **RTD** between models, **persistent cluster count and purity** (H0 gaps), **class-centroid
  loops** (H1 on 10 centroids), **TwoNN intrinsic dimension**, **isotropic / spectrum-matched Gaussian floor**.
- Effect sizes: Hedges' g and Cliff's δ with bootstrap CIs; permutation nulls on the labels (never on subsamples).

## Hypotheses
| # | Published / earlier observation (open "why") | Topological hypothesis | Metric | Prediction | Tier |
|---|---|---|---|---|---|
| H1 | Rare numbers are more crowded (E10, 7 models), but our 10-level z treats overlapping subsamples as independent | the crowding survives a cell-level label-permutation test | r½ and H0 mean death, rarest vs most frequent tenth, label permutation (200×) at cell level | r½(rare) < r½(frequent) in most cells, p < 0.05 | 0 |
| H2 | E10: much of the effect is vector length | it is a *scale* effect, not a *shape* effect | the same features on clouds rescaled to median distance 1 (scale-free) | effect shrinks to ~0 when scale-free | 0 |
| H3 | Malhotra et al.: an isotropic Gaussian cloud already gives many H1 bars | rare-number clouds are closer to structureless noise | energy distance of the 41-vector to a spectrum-matched Gaussian cloud | rarer → closer to the Gaussian floor | 0 |
| H4 | Merullo et al. (ICLR 2025): frequency decides whether a linear representation forms | frequent numbers occupy a higher intrinsic dimension (more learned features) | TwoNN ID per frequency level | ID rises with frequency | 0 |
| H5 | Merullo: threshold of ~1–4k co-occurrences | topology changes at a frequency knee, not smoothly | knee search over levels (needs numbers with counts below ~4k: 5-digit prompts) | a knee | 1 |
| H6 | Levy & Geva (NAACL 2025): errors are digit-wise, not value-based; others find distance effects | the neighbourhood structure decides it: neighbours share digits (digit topology) vs are numerically close (value topology), and this depends on tokenizer | kNN composition: share of 10 nearest neighbours with Hamming distance 1 vs with \|a − b\| ≤ 10, per model/depth | digit-splitting models: digit topology; chunk tokenizers: more value topology | 0 |
| H7 | Distance/ratio effects in LLM number comparison | comparison difficulty follows local crowding | logit margin of "which is larger" vs r½ around the pair | harder where more crowded | 1 |
| H8 | LLM "random numbers" favour 7-endings, avoid round numbers and repdigits | favoured numbers are topologically *typical* (central, many close neighbours), avoided ones are isolated | centrality (mean distance to all 2-digit numbers) and isolation of x7, x0, repdigits among 10–99 | x7 most central; x0 and repdigits isolated | 0 (+1 for behaviour) |
| H9 | topology_number_line: late layers group 4-digit numbers by leading digit | the number of persistent H0 clusters ≈ 9 and they are leading-digit pure in digit-splitting models | persistent cluster count (largest H0 gaps) and purity by leading digit | 9 pure clusters late in digit models, not in chunk models | 0 |
| H10 | E10 P5: years are "fewer, larger features" in web models, not in StarCoder | holds in all web-text models, absent in code models | P5 statistics on 7 models | pass in the 6 non-code models | 0 |
| H11 | PRH critique (2026): random networks already look aligned (CKA 0.86) | our late-layer convergence is learned, not architectural | RTD between randomly initialised copies | random-init RTD ratio ≈ 1 late; trained ≪ 1 | 1 |
| H12 | E10 D: convergence strongest among digit-splitting models | tokenizer type predicts pairwise topological similarity at fixed depth | regression of RTD ratio on same-tokenizer-type | same type → lower RTD | 0 |
| H13 | E10 C: convergence vs frequency was mixed | with 7 models, frequent numbers' topology is more shared | RTD ratio, high vs low tenth | lower for frequent | 0 |
| H14 | "Aristotelian" refinement (2026): local alignment persists, global weakens | local neighbourhoods converge more than global topology | kNN Jaccard overlap across models vs RTD | local overlap high where RTD is not | 0 |
| H15 | Valeriani et al. (NeurIPS 2023): ID expands then contracts with depth | topology simplification with depth (E10) tracks the ID profile; OLMo-2's opposite trend comes with rising ID | TwoNN ID per depth vs H1 count | positive correlation across depths | 0 |
| H16 | Massive activations / outlier dimensions favour frequent tokens (2024–2025) | the norm part of the frequency effect lives in a few outlier dimensions | frequency effect with the top-k variance dimensions removed | effect drops when outliers removed | 0 |
| H17 | Final normalisation changes geometry (topology_number_line) | last-layer anomalies (OLMo-2, Pythia) come from the final norm | compare ~84 % and last layer on scale-free features | the last-layer change is mostly scale | 0 |
| H18 | Kantamneni & Tegmark (2025): numbers form a helix with period-10 circles (single-token numbers) | the last-digit circle exists where numbers are single tokens (Pythia, OLMo-2 on 10–99) and not in digit-splitting models | loop persistence of 10 last-digit centroids among 10–99, vs label shuffles | loop in Pythia/OLMo-2, none in digit models | 0 |
| H19 | Fu et al. (2026): periods 2, 5, 10 in number features | persistent clusters by parity / mod 5 at fine scale | H0 cluster purity by n mod 2, mod 5, mod 10 | above chance, strongest for mod 10 in chunk models | 0 |
| H20 | Kadlčík et al. (EMNLP 2025): number embeddings are sinusoidal and precise | the 1–100 cloud has a long-lived H1 loop (a ring), not just a line | max H1 persistence of numbers 1–100 vs shuffled-order null | long-lived loop in single-token models | 0 |
| H21 | Malhotra et al.: an early transient topological shock in fine-tuning | pretraining shows the same early burst for number topology | H1 Wasserstein velocity across checkpoints; checkpoint-order null | early concentration C > null | 2 |
| H22 | Shrinkage view (E07–E10): rare numbers are under-exposed | the rare–frequent r½ gap changes with exposure (opens then closes as all numbers saturate) | r½ gap vs training step | non-monotonic or closing gap | 2 |
| H23 | Malhotra Fig. 28: a 41-feature probe shows "hidden progress" | number topology predicts training step before the copy task improves | CV accuracy of step from 41-vector vs copy accuracy | probe saturates earlier | 2 |
| H24 | Malhotra: representation velocity leads function velocity | for numbers, topology changes before the copy behaviour | velocity peaks: H1 Wasserstein vs JS of next-token distribution | topology peak ≤ behaviour peak | 2 |
| H25 | E10 D: models converge late | convergence grows over training (checkpoints move toward the common topology) | RTD(checkpoint, other final models) vs step | decreasing | 2 |
| H26 | Reviewer question: is the 3 % crowding above prompt noise? | the per-number position noise across prompts is small vs the frequency effect | 5 prompts per number: within-number spread vs between-level r½ difference | effect > noise | 1 |
| H27 | Malhotra: pooling the last 16 tokens erases the signal | the frequency signal sits at the `=` token, not in the digit tokens | last token vs mean of the number's tokens | signal drops with pooling | 1 |
| H28 | Malhotra Gaussian floor | number clouds have more structure than spectrum-matched noise | 41-vector energy distance to the matched Gaussian; H1 counts vs floor | far from the floor, nearest at the last layer | 0 |
| H29 | Our E10 C: no last-digit circle in digit-splitting models | the digit structure is a product (torus-like) of per-position circles across tokens, visible only when the *pair* of last two digits is used | H1/H2 of 100 two-digit centroids (00–99) | H1 rank ≥ 2 (torus) in digit models | 0 |

## Order of work
1. Tier 0 now (H1–H4, H6, H8–H10, H12–H20, H28, H29), exploratory, on all 7 models.
2. Tier 1 script for the laptop (H5, H7, H11, H26, H27; also "all layers").
3. Tier 2 (H21–H25) after choosing RunPod or laptop; checkpoints need their own pre-registration.
4. Every hypothesis that survives tier 0 gets a locked version for tier 1/2 data.
