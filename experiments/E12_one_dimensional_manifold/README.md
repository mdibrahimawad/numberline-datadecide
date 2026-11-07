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
