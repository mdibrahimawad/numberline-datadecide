# E11: Topology of numbers in LLMs: hypotheses, training checkpoints, and a shape search

**Question:** beyond frequency (E10), what shape do numbers take inside LLMs, how does it form during training, and
which parts of it are general?
**Status (2026-10-16):** tier 0 (29 hypotheses on the 7 E10 models, exploratory) done; tier 2 (pre-registered
checkpoint study, Pythia-1.4B and OLMo-2-1B) done; shape search (pre-registered discover → lock → test) done.
Tier 1 (several prompts per number, random-init models, behaviour probe; `src/e11_tier1_laptop.py`) not run.

| File | Contents |
|---|---|
| [`PLAN.md`](PLAN.md) | the 29 hypotheses H1–H29, metrics and predictions (written before tier 0) |
| [`CHECKPOINTS.md`](CHECKPOINTS.md) | pre-registration and results of the checkpoint study (C1, C2, secondary) |
| [`SHAPE_SEARCH.md`](SHAPE_SEARCH.md) | pre-registration and results of the shape search |
| [`NOTES.md`](NOTES.md) | exploratory checks made while explaining results (corrections included) |
| `number_subset.txt` | the 2,100 numbers used for checkpoints (1–100 + 200 per frequency tenth) |
| `scratch/` | small scripts behind NOTES.md |

## Data
Vectors from `mdibrahimawad/topology_number_line`, `data/`: last-token hidden state at the final `=` of
`A=A,B=B,C=C,D=D,n=` for n = 1…10,000, five relative depths (~25 / 50 / 75 / 84 % / last), 9 models
(Crystal, OpenLLaMA-3B, StarCoderBase-3B, Qwen2.5-1.5B, SmolLM2-1.7B, Pythia-1.4B, OLMo-2-1B, and for E10 D2 /
the shape-search test RedPajama-3B and Llama-3.2-1B), plus `data/e11_ckpt/` (28 checkpoints, 2,100 numbers).
Frequencies: SlimPajama-627B counts. Persistent homology: Vietoris–Rips on Euclidean distances (`ripser`).

## 1. Tier 0: 29 hypotheses on the existing vectors (exploratory; `src/e11_tier0.py`, `results/e11_tier0/`)
| # | Hypothesis | Result |
|---|---|---|
| H1 | rare numbers more crowded, cell-level permutation | **supported**: median H0 death rarest < most frequent tenth in 31 / 35 model × depth cells (p < 0.05; the 4 misses are late layers) |
| H2 | the effect is scale (length), not shape | partly: scale-free version survives only in late layers of digit models |
| H3 / H28 | number clouds vs a spectrum-matched Gaussian | number clouds have **fewer** H1 loops than matched noise (e.g. Crystal last layer 54 vs 90): more orderly than noise |
| H4 | frequent numbers have higher intrinsic dimension | **rejected**; per-number LID shows no link with frequency (ρ −0.05…+0.06, NOTES.md) |
| H6 | neighbourhoods: digit vs value | **early layers: neighbours share the last digit (up to 78–100 %); late layers: same leading digit ≈ 100 %, value-close** (all 7 models); OLMo-2 late median gap to a neighbour = 4 |
| H8 | favourite "random" numbers (x7) are central | rejected |
| H9 | 9 leading-digit clusters late | **yes in the 5 digit-splitting models at ~84 % depth** (ARI 0.88–0.99, average linkage); single linkage (H0) shows 9 separate islands only in StarCoder (1.00) and Qwen2.5 (0.88); none in Pythia (smooth line), partial in OLMo-2 |
| H10 | years "fewer, larger features" in web models | 5 of 6 non-code models (also E10 P5) |
| H12 | tokenizer type predicts cross-model similarity | **supported**; tested properly in E10 D2 (pass) |
| H14 | local neighbourhoods shared across models | 1–4 of 10 nearest neighbours shared vs 0.01 by chance; Pythia–OLMo-2 share the most locally (3.8) while their global RTD is weak (0.70) |
| H16 | frequency effect lives in a few outlier dimensions | rejected |
| H18 / H20 | last-digit circle; ring for 1–100 | rejected (no persistent loop) |
| H29 | the 100 two-digit endings form a torus | **no torus, no loop, no void**; endings are high-dimensional (top-2 PCs 25–37 %, TwoNN 17–26) and category-like: each digit value its own nearly equidistant direction, tens and units additive (R² 0.64–0.99) |
Hypotheses that need new data (H5, H7, H11, H13, H15, H17, H19, H26, H27) were not run in tier 0; H21–H25 became the
checkpoint study.

Figures: `results/e10_explore/figures/fig_E11_shapes_circle_square_torus.png` (reference barcodes vs the endings).
PH of all 10,000 numbers at once (NOTES.md): one blob, no persistent loop or void at any depth; at ~25 % depth the
single digits 1–9 are isolated points.

## 2. Checkpoints (pre-registered in `CHECKPOINTS.md`; `src/e11_checkpoints.py`, `src/e11_ckpt_analysis.py`)
Pythia-1.4B, 16 checkpoints (step 0 → 143,000); OLMo-2-1B, 12 checkpoints (step 0 → 1,907,359).

| Test (~84 % depth) | Result |
|---|---|
| C1a value topology up, digit topology down | **fail**: at step 0 all neighbour shares are at chance; the number line appears directly (no early digit stage), and the last-digit share *rises* late |
| C1b the change is abrupt | **pass**: Pythia between steps 2,000 and 4,000 (value-close neighbours 14 % → 55 %, same leading digit 36 % → 95 %), OLMo-2 by step 10,000 |
| C1c shape before skill | **fail**: copy log-probability improves first; the abrupt part of the number line and of first-token accuracy (12 % → 78 %) happen together |
| C2a rare crowding at the end | not at ~84 % (p 0.26 / 0.08); yes at ~25–50 % in both models |
| C2b trend over late training | Pythia: rare numbers **catch up**; OLMo-2: crowding grows at ~25–75 % |
| C2c crowding only after the number line | Pythia no (first at step 256), OLMo-2 yes |

Descriptive (exploratory): rare ÷ frequent mean H0 death goes 1.00 (step 0) → 0.95–0.96 mid-training → 0.98 at the
end in both models; leading-digit clusters form and then melt (Pythia ARI 0.65 at step 23,000 → ≈ 0; OLMo-2 0.75 at
40,000 → 0.17); OLMo-2's number line weakens after step 40,000 (value-close 86 % → 54 %) while the last-digit share
rises to 39 %. In PH terms: a burst of H1 loops at the moment the number line forms (Pythia 329 → 374 at step 2,000),
H0 gap variability peaks with the clusters (Pythia step 23,000, OLMo-2 step 40,000) and falls when they melt; no
persistent hole at any stage. The 41 barcode features predict the training step (Spearman 0.88–0.90, Pythia late layers).

Figure: `results/e11_ckpt/fig_trajectories.png`. LaTeX: `results/e11_ckpt/rare_frequent_h0_document.tex`,
`results/e11_ckpt/number_line_tables_document.tex`.

## 3. Shape search (pre-registered in `SHAPE_SEARCH.md`; `src/e11_shape_search.py`, `results/e11_shape_search/`)
48 number sets × 6 shape scores (clusters, loop, second loop, void, polygon in numeric order, ordered line) × 5 depths,
discovered on Crystal and Pythia, top 10 per model locked, tested on held-out models with 1,000 nulls.
**All 20 locked shapes replicate in every held-out model** (p = 0.001). They are clusters among the class centroids of
n mod 6/12/24/60, i.e. the models' **last-digit (parity, mod 4) code**: mod 6 / 12 split into even vs odd (ARI 1.00),
mod 60 into 10 groups by last digit (ARI 0.89–1.00); moduli coprime with 10 show nothing over 1000–9999. Real but known.
No clock-like polygons (1 hit in 480 cells). Lead for later: mod 7 / 9 / 11 structure among Pythia's single-token
numbers 1–999 (z 22–41), absent in Crystal. Figure: `results/e11_shape_search/fig_mod_clusters.png`.

## What E11 says overall
1. Number topology has **no holes**: no circles, tori or voids, at any depth, in any model, at any training stage.
   Its structure is in **grouping and ordering**: last-digit categories early in the network, a number line with
   leading-digit groups late.
2. **The tokenizer decides the late-layer shape** (with E10 D/D2): digit-splitting models share 9 leading-digit regions
   and nearly the same topology; chunk tokenizers give smooth or very fine number lines; same tokenizer → same shape,
   whatever the training data.
3. **The number line appears abruptly during training, together with the copy skill**, and keeps reorganising
   afterwards (clusters melt; OLMo-2 drifts back toward last-digit neighbourhoods).
4. Frequency only modulates spacing: rare numbers are slightly more crowded, most in mid-training.
