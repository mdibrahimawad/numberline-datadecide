# E13: how LLMs code integers (staged study)

**Stage 1 is done. The E13 pipeline is built; a reduced first pod run (DEVIATIONS D26) produced Stage A data for Pythia-1.4B and
Qwen2.5-1.5B (discovery models only). Results below are discovery-only; Stage B (7 replication models), the causal stage and
checkpoints have not run.**

## Stage A results (Pythia-1.4B + Qwen2.5-1.5B), 2026-11-08
**What ran:**
- Pod run 2026-11-07, extraction for both models: all numbers 1–9999, 16 contexts, 9 layers. Checks 2.5 pass for both
  (cosine with the old vectors ≥ 0.9999998).
- Pythia analysis on the pod.
- Qwen: 102 cells on the pod; the rest here (2026-11-08). Here, the second wordings and the C1 seed-7 set were not
  downloaded, so the H0 ceilings, decomposition and transfer (secondary) are missing for Qwen.
- Tables and figures: `results/stageA_partial/`.

**The pod run failed overnight:**
- The pod container restarted at about 22:04. This killed the run and the auto-stop, and the empty pod stayed on for
  about 9 h.
- The cause is unknown: one heavy 4-digit cell peaks at 1.7 GB, so memory is unlikely.

| Pre-registered test (PREREG_E13), number reading AVG5/P_num | Digit (Qwen) | Chunk (Pythia) | Verdict (discovery only) |
|---|---|---|---|
| P7 positive control (days and months both work) | no | no (days yes, months no) | **fails**: no loop/PH claims |
| H1: order decays from the leading digit to the units | 12/15 powered cells | 10/10 | **holds in both families** |
| R-a / R-b | 0 | 0 | not supported |
| H2a carry spikes | 1/11 | no power | not supported / cannot tell |
| H2b, H2c | no power | no power | cannot tell |
| P8 knee sharper than controls | 0 | 0 | not supported |
| H3 lexicon: residual reliability ≥ 0.30 and frequency slope CI ≠ 0 (depth ≥ 50 %) | 11/15 | 13/15 | holds in both families. Crowding is "lexical" in only 2/9 Qwen cells (S4); not testable for Pythia (139 numbers) |
| H0 invariance (C1/P_num and ≥ 3 natural contexts ≥ 50 % of C1/P_next) | | | invariant: O1_leading (94 % of cells), smooth-curve R² (85 %), curve + digits R² (96 %), units-digit decoding (100 %), probe for n (96 %). Not invariant: decay Δ, carry e_1, tree − line |

**The copy confound, resolved for these two models (exploratory reading of the tables above):**
- Stage 1 found that the leading digit loses its order in late layers. This happens **only at the copy position "="**:
  - Qwen late layers: O1_leading −0.15 to 0.03 at "=";
  - Qwen at the number token: 0.77–0.85;
  - Qwen in neutral sentences: 0.83–0.88.
- Likewise, Qwen's high late-layer smooth-curve R² (0.64–0.75, the E12c headline) appears only at "="; the number
  readings give 0.06–0.23.

**PCA pictures by reading:** `results/pca_number_line_*.png`, `results/pca_number_line_by_reading.csv`. Exploratory:
the layer with the highest |Spearman(PC1, n)| is shown.
- Neutral-sentence reading: PC1 order 0.94–0.97, on 7–17 % of the variance.
- Old copy reading at "=": 0.78–0.92.


## E13 pipeline: build and laptop tests (2026-11-07)
| Item | Where |
|---|---|
| Work order | `METHODOLOGY.md`, `PLAN.md` |
| Pre-registration (all stages) | `PREREG_E13.md` |
| Choices and deviations | `DEVIATIONS.md` |
| Code | `run_all.py` (master), `src/e13/` (config, prompts, extract, checks, data, panel, h0, h3, p7, power, causal, training, analyse, report, upload) |
| Pod steps | `RUNPOD.md` |
| Tests | `tests/test_e13_pipeline.py` and `tests/test_e13_stage1.py` (15 pass): positions with the real Pythia and Qwen tokenizers; extraction, checks and resume with a tiny random model; causal hooks; planted-structure recovery (`results/power_check_synthetic.csv`) |

**Laptop pilot.** Pythia-1.4B on CPU, numbers 1–500, all 16 contexts. It is a code test only; its numbers are
not results.
- **Checks (2.5)** (`results/checks_pilot_pythia.json`):
  - batched vs single: max relative difference 5e-5;
  - copy accuracy: 0.995;
  - the stored old E10 vectors are reproduced (cosine ≥ 0.9999998);
  - greedy decoding via the output head matches the full model.
- **Timing:** extraction 9 min (8,000 prompts); 844 analysis jobs in 9.5 min on 3 CPU workers; report, figures and
  Gate A all built.
- **Causal stage:** ran end to end with 24 items per behaviour (848 matrix rows, the decision rule applied); about
  2 h on CPU.
- **Checkpoint runner:** tested on Pythia-160M at step 0 and step 143000 (numbers 1–150): extraction, behaviour,
  light analysis, and vectors purged after the run.
- **Real-noise power rows from the pilot:** `results/power_check_pilot.csv`.
- **Bugs found and fixed by the pilot:** constant layer-0 cells, a duplicate report column, Gate A not tolerating
  a failed report, errored cells not retried on resume, the steering edit crashing, and degenerate digit positions.


| Item | Where |
|---|---|
| Pre-registration | `PREREG_stage1.md` (9f91607), Amendment 1 before the real run (1627c32) |
| Code | `src/e13_stage1.py`, `src/e13_report.py` (f33b365; report fix in the results commit) |
| Tests | `tests/test_e13_stage1.py` (6 pass) |
| Results | `results/e13_stage1/` |
| Literature | `LITERATURE.md` |

The data are existing vectors: last token at the final `=` of the copy prompt, one prompt per number.

Cells: 9 models × 5 depths × 3 sets = 135 cells: 125 main plus 10 secondary (S4x). Token counts per set are in
`token_counts_in_prompt.json`. Families:
- **digit-splitting:** Crystal, OpenLLaMA, StarCoder, Qwen2.5, SmolLM2;
- **chunk:** Pythia, RedPajama (BPE), OLMo-2, Llama-3.2 (3-digit groups).

## Stage 1 verdicts (pre-registered rules)
| Test | Digit-splitting family | Chunk family | Verdict |
|---|---|---|---|
| **H1** ordinality decays leading → units (cells at depth ≥ 50 %) | mixed: 18/59 powered cells pass (31 %); replication 12/47 | mixed: 22/32 (69 %); replication 16/24 | **not supported** |
| R-a all positions equally ordinal | 0/60 | 0/40 | not supported |
| R-b all positions categorical | 0/60 | 2/40 | not supported |
| R-c units digit is a circle | 0/60 | 0/40 | not supported |
| **H2a** carry spikes beyond the digit code | 2/36 powered cells (both Qwen S4, late) | 0/11 | **not supported** |
| **H2b** successor operator breaks at carries | no power (0/75 cells) | no power (0/50) | **cannot tell** |
| **H2c** tree beats line on held-out leading-digit groups (late layers) | 8/30 (27 %) | no power (0/20) | **cannot tell** |

Power: template recovery is 1.00 for line, circle and simplex at every position in every cell, so the template tests
have full power. Decay power passes in 112 of 125 cells (all 10 C-neox S4 cells fail: 139 numbers).

## What the data show
1. **Every digit position is mainly a simplex (equidistant categories) with an ordinal tilt.**
   - **Template.** The best template is simplex in 274 of 292 position × cell cases at depth ≥ 50 %. It is never
     line-like except for a few leading digits, and the circle is never the units template.
   - **Participation ratio** is 3–7, far from 1 (line) or 2 (circle).
   - **Ordinal tilt.** Order is still present on top: O1 is 0.3–0.9 for most positions.
   - **Signal.** Every position carries real signal (shuffle p < 0.01 in 99 % of cases).
   - **Prior work.** This contradicts a literal reading of the per-digit *circles* of Levy & Geva (their probes were
     supervised and can map a simplex onto a circle).
2. **In digit-splitting models the leading digit loses its order in late layers.** This was not predicted.
   - At ~25–50 % depth the leading digit is the most ordinal position (O1 0.80–0.92 in all 5 models).
   - At ~84 % and last its order collapses: O1 −0.15 to 0.50; Qwen −0.15 to 0.03.
   - Meanwhile the second digit stays strongly ordinal (O1 0.81–0.94).
   - The leading digit also carries about half of the digit-code variance (V share 0.50–0.56).
   - **Reading.** In these models the next token to be written is the leading digit, so this fits the copy/output
     confound of plan §1: the late state becomes a categorical "which digit do I write next" code. Stage 2 (H0) is
     the test.
3. **Chunk tokenizers keep a leading-digit line longer.**
   - O1 for the leading digit is 0.60–0.97 up to ~84 %, then falls at the last layer (0.09–0.64).
   - The units digit is close to unordered (−0.07 to 0.42).
   - Inside a leading-digit group, a line explains more in chunk models: isotonic R² 0.11–0.52, vs 0.00–0.08 in
     digit-splitting models.
4. **The decay rule therefore depends on depth.** It passes in 80 % of cells at ~50 % depth in both families, but in
   0 % (digit-splitting) and 20–50 % (chunk) at ~84 % and last. H1 fails as pre-registered (depth ≥ 50 %, both
   families). A "mid-layer floating-point code" is an exploratory reading only.
5. **Carries are not events beyond the digit code. In chunk models they are bridged** (exploratory direction).
   - In digit-splitting models the step n → n + 1 across a carry is what the additive digit code predicts:
     median spike ≈ 0, with Qwen S4 late layers the only exception.
   - In chunk models at depth ≥ 75 % the x99 → x00 step is **smaller** than the digit code predicts, by about one
     ordinary step (95 % CI below 0 in 14/20 chunk S3 cells, all 4 models). For example 199 and 200 sit closer than
     "hundreds 1 → 2, tens 9 → 0, units 9 → 0" implies: the number line bridges the carry.
   - The pre-registered direction was a positive spike, so this is labelled exploratory.
6. **The successor operator falls at carries, but no more than a pure digit code predicts.**
   - R²(carry-free) − R²(carry steps) = 0.16–0.61, and the additive fake gives 0.15–0.22.
   - "Beyond digits" ≈ 0 except digit-splitting S2.
   - The planted cumulative jumps are too small to move R², so the formal verdict is "cannot tell".
7. **Tree vs line.**
   - In digit-splitting models the tree wins on held-out groups only in late layers and only modestly (ΔR² mean
     +0.10 at ~84 %; passes 8/30).
   - In chunk models the line wins at every depth (ΔR² −0.02 to −0.19), as the owner expected for Pythia mid layers,
     but the test has no power there (planted trees detected in ≤ 56 % of replicates).
   - Ultrametric violation of group centroids is low (0.08–0.20), as expected when groups are near-equidistant.
8. **Secondary checks.**
   - **Exponent (digit count)** is decodable (balanced accuracy 0.74–1.0; shuffled 0.2–0.37). It is "explicit" by
     the pre-registered rule in chunk models mostly at early depths (OLMo-2 4/5, Pythia 2/5, RedPajama 1/5,
     Llama 1/5). Held-out-length probes do not extrapolate to 4 digits.
   - **Knee** (rank needed for 95 % of held-out R²): median 8 (S2), 15 (S3), 15 (S4). The naive "1 + 9 per
     position" predicts 10 / 19 / 28.
   - **Robustness:** without z-scoring, and with the 3 largest raw dimensions removed, results are unchanged
     (Δ correlation 0.97–1.00).

## Deviations
- **Amendment 1** (before the real run, from synthetic validation):
  - H2a uses the additive prediction as a noise-debiased offset (the free-b version is kept as `e_freeb`);
  - H2a power uses a transient carry plant (cumulative is kept as `has_power_cumulative`).
- **H2a/H2b sets.** C-neox S4 (139 one-token numbers) has enough consecutive steps, so it was run, not skipped;
  its H2a cells have no power.
- **Report.** Error bars are clipped at 0 where the point estimate falls outside its percentile CI (figure only).

## Figures
- `fig_S1_ordinality_by_position.png`
- `fig_S2_step_length_carries.png` (Qwen / Pythia S3, all depths)
- `fig_S2b_carry_spikes.png`
- `fig_S3_successor_R2_by_level.png`
- `fig_S4_tree_vs_line.png`
