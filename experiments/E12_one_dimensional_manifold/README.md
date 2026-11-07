# E12: Do number hidden states lie on a 1-D manifold?

**Question (the professor's):** do the hidden states of numbers sit along one thin thread (a 1-D manifold) whose order
matches numeric order? Tested in a strict form: local dimension ≈ 1 over a range of scales (L), one piece that
recovers held-out order (G), and a spacing law (S), always against fake data with matched noise and smoothness.
**Status (2026-11-07):** done as pre-registered. **Answer: no evidence for a 1-D manifold, and no power to detect one
at the real noise level; the structure is better described by additive digit parts plus periodic (2/5/10/100)
components, and it is not reducible to smoothness. The positive control (days, months) also failed, so the absence
of loops is uninformative.**

| File | Contents |
|---|---|
| [`PREREG.md`](PREREG.md) | hypotheses H1–H5, data, preparation, fakes, tests T1–T9, thresholds, decision rules (commit `5307d7a`; clarifications `60f65d6`, before results) |
| `pythia_token_counts_1_10000.txt` | Pythia token count of each n in prompt context (selects one-token numbers) |
| Code | `src/e12_manifold.py` (prep, fakes, T1–T8), `src/e12_report.py` (verdicts, tables, figure), `src/e12_positive_control.py` (T9), `tests/test_e12_manifold.py` (5 synthetic-shape tests) |
| Results | `results/e12_manifold/`: `tests.csv` (all metrics, 30 datasets × 9 rows), `summary_table.csv`, `T1_power.csv`, `power_replicates.csv`, `verdicts.csv`, `verdicts_by_depth.csv`, `positive_control.csv`, `fig_dimension_vs_k.png` |

Run: `python -m src.e12_manifold --data <topology_number_line>/data` → `python -m src.e12_report` →
`python -m src.e12_positive_control --extract` (CPU; needs Hugging Face access).

## Data
Qwen2.5-1.5B (digit tokenizer): Q2 10–99, Q3 100–999, Q4 1000–9999 (fixed random 3,000). Pythia-1.4B (chunks): P2 10–99,
P3 = 777 one-token numbers in 100–999, P1 = 876 one-token numbers in 1–999. All 5 depths, 30 datasets. Largest raw
dimension 0.3–12 % of the variance (up to 264× the median dimension in Qwen); z-scoring removes its dominance.

## Results
**T1 power: none.** The fake curve (the real kernel-smoothed trend + Gaussian noise with the real residual covariance)
passed the 1-D plateau in 0 of 5 replicates in every dataset; so did every sheet. At the real noise level local
dimension cannot tell a curve from a sheet. The smooth trend carries 12–58 % of the variance in most sets (Q4 and late
Qwen layers 63–97 %, where the chosen bandwidth was tiny; see double-checks).

**T2:** real MLE dimension 4.3–17.0 at k = 20; plateau in 0 / 30 datasets; dimension falls only mildly with k.
**T3:** random-split order recovery ρ 0.11–0.99 (fake curves 0.14–0.99, sheets 0.44–1.00); held-out block ρ mostly < 0.5;
extrapolation fails. **T4:** spacing exponents mostly at the grid edges (0.05 or 2.0): not interpretable.
**T5:** squared distance follows |m − n| about as well as |log m − log n| (log better in 9 / 30). **T6:** an additive
per-digit model explains squared distance better than numeric closeness in **30 / 30** datasets (e.g. Qwen Q4 ~84 %:
R² 0.74 vs 0.49; Pythia one-token P1 ~84 %: 0.38 vs 0.13). **T7:** residual variance at periods 2, 5, 10 (and 100)
is significant in nearly every dataset (shares 0.5–10 %, permutation p ≤ 0.005), largest in early and middle layers.
**T8:** no long H1 bar in any real or fake dataset (including the fake helix).

Example (Qwen Q4, ~84 %; full table in `summary_table.csv`):
| condition | ID k=5 | k=20 | k=80 | T3 random ρ | T3 block ρ | T5 R² lin / log | T6 R² digits / value | T7 p10 share | T8 long H1 |
|---|---|---|---|---|---|---|---|---|---|
| real | 13.3 | 10.9 | 8.9 | 0.77 | −0.06 | 0.49 / 0.45 | 0.74 / 0.49 | 0.005 | 0 |
| curve | 8.7 | 9.3 | 8.3 | 0.81 | −0.23 | 0.51 / 0.47 | 0.77 / 0.51 | 0.004 | 0 |
| sheet (fat) | 15.2 | 11.1 | 6.7 | 0.98 | 0.77 | 0.94 / 0.74 | 0.90 / 0.94 | 0.001 | 0 |
| matched smooth | 12.8 | 13.4 | 11.9 | 0.09 | −0.03 | 0.13 / 0.11 | 0.20 / 0.13 | 0.002 | 0 |

**T9 positive control: failed.** Months pass at 0 / 5 depths in both models; days pass at 3 / 5 depths in Pythia, only
through the polygon-in-order score (0.86–1.00), never through a persistent H1 loop (largest lifetime ratio 0.14); Qwen
passes nothing. As pre-registered, the absence of loops for numbers is therefore uninformative.

## Verdicts (pre-registered rules, over 5 depths)
| | Q2 | Q3 | Q4 | P2 | P3 | P1 |
|---|---|---|---|---|---|---|
| H1 smooth 1-D curve | cannot tell (no power) | cannot tell | cannot tell | cannot tell | cannot tell | cannot tell |
| H2 helix (periodic residual) | supported 5/5 | supported 5/5 | mixed 3/5 | supported 5/5 | supported 5/5 | supported 5/5 |
| H3 digit-by-digit | not supported 1/5 | mixed 2/5 | mixed 2/5 | mixed 3/5 | mixed 3/5 | mixed 3/5 |
| H4 multiscale | cannot tell | cannot tell | cannot tell | cannot tell | cannot tell | cannot tell |
| H5 nothing special | not supported 0/5 | not supported 0/5 | not supported 0/5 | not supported 0/5 | not supported 0/5 | not supported 0/5 |
H2's "supported" means significant periodic residual structure, not a geometric helix (no loops, and the loop detector
fails its positive control). H3's second condition (neighbours share the last digit ≥ 3× chance) holds mainly in early
and middle layers; its first condition (digit-additive distances beat value) holds in all 30.

## Deviations from the pre-registration
None in settings or rules. Operational: Hugging Face hosts had to be allowed for T9 and the tokenizers (a new personal
cloud environment); T9 ran with single-threaded torch (no effect on results).

## To double-check / next
1. Power curve: rerun the fakes with the noise scaled to 0.1× and 0.3× to find where the tests regain power.
2. Q4's leave-one-out bandwidth was h = 1 (numbers ≈ 3 apart), so its "trend" is not smooth and its fake curve ≈ the
   real data; a minimum bandwidth (e.g. ≥ 1 % of the range) should be pre-registered for a rerun.
3. T4 exponents at the grid edges: do not link to β.
4. Per-dimension z-scoring amplifies low-variance dimensions; repeat without it.
5. The principal curve recovers order poorly even on fake curves; T3 is weak for curved trends.
6. T7 p-values floor at 1/201; judge by the share.
7. Pythia P1 mixes 1–3-digit numbers; its log advantage in T5 may be digit length.
8. A positive control in the subspace where days/months are known to be circular (Engels et al. 2024 use specific
   probes / PCA directions and many contexts) would tell whether the loop detector can work at all on LLM states.

---

# E12b: smooth part + additive digit part (pre-registered in `PREREG_B.md`, commit `689399e`)
Code `src/e12b_smooth_digits.py` (`7e9c2c7`), report `src/e12b_report.py` (`74fdd75`), both before results; synthetic
test `tests/test_e12b_smooth_digits.py`. Results `results/e12b_smooth_digits/` (`tests.csv`, `power.csv`,
`verdicts.csv`, `verdicts_by_depth.csv`, `secondary_units_only.csv`, `fig_dimension_vs_k.png`). Deviations: none.

Z(n) = f(n) + D(n) + E(n), f smooth over n (CV bandwidth, ≥ 1 % of the range; chosen 1–10 %), D additive one-hot tens +
units (+ hundreds for Q4), backfitted.

**Variance carried (share of total variance of the 50 PCA dimensions):**
| Set | f (smooth) by depth ~25 / 50 / 75 / 84 % / last | digit part | residual |
|---|---|---|---|
| Qwen Q2 (10–99) | 0.21 / 0.03 / 0.04 / 0.14 / 0.17 | 0.40–0.74 | 0.20–0.58 |
| Qwen Q3 (100–999) | 0.18 / 0.08 / 0.16 / **0.61 / 0.54** | 0.13–0.32 | 0.19–0.68 |
| Qwen Q4 (1000–9999) | 0.12 / 0.08 / 0.19 / **0.67 / 0.59** | 0.08–0.35 | 0.19–0.72 |
| Pythia P2 (10–99) | 0.23 / 0.21 / 0.37 / 0.17 / 0.37 | 0.43–0.52 | 0.28–0.43 |
| Pythia P3 (one-token 100–999) | 0.32 / 0.33 / 0.25 / 0.25 / 0.25 | 0.20–0.34 | 0.35–0.44 |
| Pythia P1 (one-token 1–999) | 0.37 / 0.36 / 0.25 / 0.25 / 0.25 | 0.18–0.33 | 0.36–0.40 |

**Is f 1-D?** Yes, as expected by construction: the T2 plateau passes in 5 / 5 depths for Q2, Q3, Q4, P3, P1 (MLE at
k = 20: 1.08–1.30); P2 mixed (2 / 5; MLE up to 2.2–2.6 at ~75 %–last, where f is strongly curved).
**Does f's geometry recover order without n (T3)?** Supported for Q2, Q3, P3, P1 (random ρ 0.80–1.00, held-out block
ρ 0.97–1.00); mixed for Q4 (2 / 5; block ρ 0.02 at ~50 %) and P2 (3 / 5).
**Spacing along f (T4, exact arc length):** close to linear in n: exponent p 0.75–1.20 (Qwen mostly 0.75–0.95, mild
compression; Pythia 0.80–1.10); log spacing never selected.
**Is the digit-removed data Z − D (f + residual) 1-D?** **Cannot tell: no power at any tested noise level.** The fake curve
(f + c × the real residual) passed the plateau in 0 / 5 replicates at c = 0.1, 0.3 and 1 in every dataset (at c = 0.1
its MLE is 1.4–2.0 at k = 20–80 but 2–11 at k = 5); the real Z − D has MLE 3.2–15 at k = 20. Even a tenth of the real
residual is larger than the spacing of neighbouring numbers along f, so local dimension sees the noise.
**Secondary (2-digit sets, units digit only):** f carries 0.10–0.55 (Qwen Q2) and 0.25–0.30 (Pythia P2); f passes the
plateau at ~25–75 % (MLE 1.13–1.31), not at ~84 % / last (1.57–2.63).

**Reading.** Removing an additive digit part leaves a smooth part that is a non-self-crossing curve ordering the numbers
with roughly linear spacing, but that part carries only 3–37 % of the variance except in Qwen's late layers for 3- and
4-digit numbers (54–67 %). Whether the full digit-removed representation is 1-D cannot be decided with local dimension:
the residual is too large relative to the spacing of neighbours.

**Double-check:** (1) f's bootstrap intervals for p are narrow because they resample points along a fixed f, not
uncertainty in f itself; (2) for 2-digit sets f and the tens one-hot compete (see the units-only secondary); (3) a power
check at c < 0.1 would locate the noise level where the plateau rule regains power.

---

# E12c: held-out R² of smooth models (pre-registered in `PREREG_C.md`, commit `3627f6a`)
Code `src/e12c_heldout_r2.py` (`e259a5d`, synthetic test `tests/test_e12c_heldout_r2.py`). Results
`results/e12c_heldout_r2/` (`r2_all.csv` per split and repeat with chosen bandwidths, `r2_summary.csv`). Deviations: none.
Standardisation and PCA fitted on training numbers only; bandwidths by 5-fold CV inside the training set.

**Random 80 / 20 (mean of 3 repeats), held-out R²:**
| Set | curve (~25 / 50 / 75 / 84 % / last) | curve + digits | 2-D (n, n mod 10) | shuffled |
|---|---|---|---|---|
| Qwen Q2 | 0.19 / 0.12 / 0.12 / 0.56 / 0.48 | 0.13 / 0.20 / 0.07 / 0.77 / 0.69 | 0.18 / 0.16 / 0.12 / 0.55 / 0.49 | −0.04 to −0.02 |
| Qwen Q3 | 0.18 / 0.09 / 0.17 / 0.74 / 0.68 | 0.39 / 0.38 / 0.25 / 0.81 / 0.75 | 0.29 / 0.29 / 0.17 / 0.74 / 0.68 | ≈ 0 |
| Qwen Q4 | 0.17 / 0.08 / 0.19 / 0.78 / 0.71 | 0.44 / 0.41 / 0.24 / 0.82 / 0.76 | 0.29 / 0.30 / 0.19 / 0.78 / 0.71 | ≈ 0 |
| Pythia P2 | 0.41 / 0.32 / 0.18 / 0.15 / 0.13 | 0.35 / 0.25 / 0.35 / 0.29 / 0.21 | 0.42 / 0.33 / 0.19 / 0.17 / 0.14 | −0.04 to −0.02 |
| Pythia P3 | 0.38 / 0.39 / 0.34 / 0.33 / 0.35 | 0.50 / 0.51 / 0.62 / 0.59 / 0.62 | 0.39 / 0.39 / 0.35 / 0.33 / 0.36 | ≈ 0 |
| Pythia P1 | 0.46 / 0.43 / 0.35 / 0.33 / 0.35 | 0.56 / 0.55 / 0.62 / 0.60 / 0.60 | 0.46 / 0.43 / 0.36 / 0.34 / 0.36 | ≈ 0 |

**Held-out block (70–79, 700–799, 7000–7999):**
| Set | curve | curve + digits | 2-D | shuffled |
|---|---|---|---|---|
| Qwen Q2 | 0.09 / 0.05 / 0.03 / −1.70 / −1.63 | −0.01 / 0.22 / 0.02 / −0.18 / −0.19 | 0.09 / 0.18 / 0.03 / −1.70 / −0.78 | ≈ −0.04 |
| Qwen Q3 | −0.38 / −0.12 / −0.43 / −1.57 / −1.74 | 0.18 / 0.31 / −0.06 / −1.27 / −1.32 | 0.22 / 0.26 / −0.43 / −1.57 / −1.74 | ≈ 0 |
| Qwen Q4 | −0.72 / −0.14 / −0.25 / −1.30 / −1.47 | 0.22 / 0.28 / −0.15 / −1.31 / −1.39 | −0.05 / 0.25 / −0.17 / −1.30 / −1.48 | ≈ 0 |
| Pythia P2 | 0.09 / 0.07 / −0.07 / −0.11 / −0.14 | 0.11 / 0.10 / 0.15 / 0.24 / −0.09 | 0.11 / 0.09 / 0.00 / −0.04 / −0.08 | ≈ −0.03 |
| Pythia P3 | −0.54 / −0.33 / −0.48 / −0.54 / −0.55 | 0.29 / 0.03 / 0.02 / −0.06 / −0.09 | −0.54 / −0.33 / −0.47 / −0.52 / −0.54 | ≈ 0 |
| Pythia P1 | −0.39 / −0.22 / −0.40 / −0.44 / −0.45 | 0.05 / 0.14 / 0.08 / 0.02 / −0.03 | −0.35 / −0.22 / −0.39 / −0.43 / −0.44 | ≈ 0 |

**Reading.**
1. A smooth function of n predicts held-out numbers well above the shuffled control everywhere, most in Qwen's late
   layers for 3- and 4-digit numbers (0.68–0.78); CV picked small bandwidths (median 1 % of the range), i.e. local
   interpolation from neighbouring numbers.
2. Adding the additive digit part helps in 26 of 30 random-split cells (largest gains in early Qwen layers and in Pythia's
   late layers, up to +0.33); the 2-D (n, n mod 10) surface helps only in Qwen Q3/Q4 early layers (up to +0.22) and is
   otherwise within 0.04 of the curve.
3. The curve does **not** interpolate across a held-out block: R² < 0 in 25 of 30 cells, down to −1.7 in Qwen's late
   layers. Each block is a whole leading-digit group (7x, 7xx, 7xxx), so a missing group is not where a smooth curve
   through its neighbours would put it, consistent with the leading-digit groups of E11. The digit part rescues early
   layers (R² 0.1–0.3) but not Qwen's late layers.
4. Caveat (pre-registered): one prompt per number, so the attainable R² is unknown.

## Amendment A results (Q4 minimum bandwidth; Pythia log vs linear on P3)
Pre-registered in `PREREG.md` (Amendment A) before the rerun. Output `results/e12_manifold_q4_minbw/` (same files as
`results/e12_manifold/`); the original Q4 rows are unchanged.

**Deviation (seeds):** the rerun was started with `--sets Q4`, so the per-dataset id in the seed (`1000·dataset +
condition`) was 1–5 instead of the original 11–15. Seeds were fixed, but not the same ones as in the main run. Effect on
the real data: only the random parts move (T3 random ρ by ≤ 0.09, e.g. 0.20 → 0.16 at ~25 %); dimensions, T5–T8 on real
data are identical to 3 decimals. The verdicts are identical to the original Q4.

**Q4: bandwidth and trend (h = 1 before → h = 1 % of the range now):**
| Depth | h (old → new) | trend share (old → new) | real MLE k = 20 | fake curve MLE k = 20 (old → new) | T1 curves passing plateau |
|---|---|---|---|---|---|
| ~25 % | 1 → 90 | 0.84 → 0.16 | 15.7 | 14.9 → 21.3 | 0 / 5 |
| ~50 % | 1 → 90 | 0.82 → 0.09 | 15.2 | 16.5 → 22.4 | 0 / 5 |
| ~75 % | 1 → 90 | 0.83 → 0.20 | 17.0 | 16.0 → 21.4 | 0 / 5 |
| ~84 % | 1 → 90 | 0.97 → 0.74 | 10.9 | 9.3 → 16.6 | 0 / 5 |
| last | 1 → 90 | 0.95 → 0.66 | 10.9 | 9.3 → 16.4 | 0 / 5 |

Leave-one-out CV chose the smallest allowed bandwidth (1 %) at every depth. With a smooth trend, the fake curve is no
longer ≈ the real data, but its noise (the residual covariance, 26–91 % of the variance) is still ~50-dimensional, so the
fake curve, both sheets and every other fake are 11–24-dimensional; T1 power is still 0 at every depth (curve 0/5, thin
0/5, fat 0/5). Real data vs fakes: the real MLE is now **below** the fake curve's at every depth (it was above before), and
the real data still has far more digit structure than the curve (T6 digit R² 0.12–0.74 vs value R² 0.05–0.48; fake curve
digit ≈ value at early depths), and a period-10 share (5–10 % at ~25/50 %) that only the helix fake reproduces.

**Q4 verdicts (unchanged):** H1 cannot tell (0/5 depths with power), H2 mixed (3/5), H3 mixed (2/5), H4 cannot tell,
H5 not supported (0/5).

**Pythia log vs linear on P3 (one-token 100–999), from the main run:**
| Depth | T5 R² linear / log (P3) | T4 best (P3) | T4 p [95 % CI] (P3) | P1 T5 linear / log (reference, confounded) |
|---|---|---|---|---|
| ~25 % | 0.31 / 0.34 | power | 0.75 [0.70, 0.80] | 0.28 / 0.66 |
| ~50 % | 0.33 / 0.25 | power | 0.85 [0.80, 0.90] | 0.36 / 0.43 |
| ~75 % | 0.17 / 0.10 | power | 0.60 [0.35, 0.85] | 0.15 / 0.07 |
| ~84 % | 0.14 / 0.08 | linear | 0.80 [0.30, 1.25] | 0.13 / 0.05 |
| last | 0.12 / 0.07 | power | 1.50 [1.30, 1.70] | 0.09 / 0.03 |

On P3, log beats linear only at ~25 % and only slightly (0.34 vs 0.31); at the other depths linear wins. Log is never the
best spacing model. P1's large early log advantage (0.66 vs 0.28) was mostly digit length (1- vs 2- vs 3-digit numbers).
