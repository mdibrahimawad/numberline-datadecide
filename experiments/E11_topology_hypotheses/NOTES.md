# E11 notes (exploratory checks made while explaining results, 2026-10-16)
Scripts in `scratch/` (run from the repo root with `PYTHONPATH=.`).

**H29 correction: the 100 two-digit endings are not a flat 2-D sheet.** PH finds no persistent loop or void
(not a torus, not a circle), but the top 2 principal components hold only 25–37 % of the variance and the
TwoNN dimension is 17–26 (`flat.py`). The 10 values of each digit are nearly equidistant (distance sd/mean
0.06–0.19; top PC 17–43 %), with a weak order (Spearman of distance with |a − b| up to +0.77 for the tens
digit) (`simplex.py`): digits behave mostly like categories (a simplex), combined additively (R² 0.64–0.99),
consistent with an ~18-dimensional product of two 9-simplices. "Grid" in H29 should read "no holes,
additive in tens and units".

**PH of all 10,000 numbers** (`whole.py`, 1,000 greedy landmarks for H0/H1, 300 for H2, vs a spectrum-matched
Gaussian cloud): Crystal L8 — largest H0 merges 1.83 / 1.76 / 1.53 vs noise ≈ 0.9 (isolated points: the single
digits 1–9, 100, 1971, 55, 61); longest loop 0.15 vs noise 0.08 (short; a ring would be ≈ 1.2); 1,826 loops vs
2,952 for noise; no void. L27: no outliers (largest merge 1.04 vs 0.93), longest loop 0.16, 1,210 loops vs
2,773, no void. The whole cloud is one connected blob with no persistent holes, more orderly than noise.

**Local intrinsic dimension per number vs frequency** (`scratch/lid.py`, Levina–Bickel MLE with 20 neighbours among
all 4-digit numbers; 7,027 ordinary numbers; frequency = count left after block + tens + last digit;
`results/e11_tier0/LID_per_number.csv`): Spearman ρ between −0.048 and +0.059 in the 35 model × depth cells, with
both signs (negative in 17, positive in 17, zero in 1); the same after removing vector length. Mean LID rarest vs most
frequent tenth differs by ≤ 1.8. No link between frequency and local dimension. The set-level TwoNN tendency
(frequent tenth ~1 dimension lower in 27/35 cells, H4) does not hold per number and is not claimed.

**Rare–rare, rare–frequent, frequent–frequent distances inside blocks of 100** (`scratch/rr_rf_ff.py`,
`results/e11_tier0/RR_RF_FF_distances.csv`; block + digit matched thirds, no special patterns; 5 models × 5 depths):
mean pairwise distance RR < RF ≤ FF in almost every cell (e.g. Crystal ~25 %: 0.801 / 0.821 / 0.824). Rare and frequent
numbers are **interleaved**, not in separate regions: of a rare number's 5 nearest numbers in its block (rare + frequent
pool), 41–48 % are frequent (50 % = fully mixed); rare vectors are 1–2 % shorter. Picture: one mixed cloud in which rare
numbers sit slightly closer to the centre. In Pythia and OLMo-2 late layers RF ≥ FF and frequent numbers' neighbours
are 51–54 % frequent: a slight segregation.

**Feasibility of a same-number, different-data test on DataDecide (E04 counts, 25 recipes, 100B tokens each):** a
4-digit ordinary number's absolute count varies ≈ 1.5× (SD of log10 0.18) across recipes (median count per recipe 529
in falcon-and-cc-qc-10p to 2,700 in dolma1_6plus; 5371: 495 to 2,666); its count relative to its own block varies less
(SD 0.067 in log10, ≈ 17 %; between-number SD 0.146; relative counts correlate 0.74 between recipes).
