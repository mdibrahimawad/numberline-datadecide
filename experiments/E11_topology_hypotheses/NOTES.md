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
