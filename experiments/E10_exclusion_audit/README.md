# E10: Blind exclusion audit of the 25 DataDecide models (E01, 3-gap β)

**Question:** is there a scientifically valid reason to exclude any of the 25 models from the α–β analysis
(E01 paper protocol, β_direct at each model's own selected layer)?
**Status:** done (2026-10-25). Exploratory, personal analysis; not pre-registered.

## Method
21 independent auditors (Claude subagents), each covering one aspect of all 25 models. **None saw or computed
the α–β relationship**; each had to state exclusion criteria with a-priori thresholds and evidence, and say
whether each flag is a measurement failure (valid to exclude) or a property of the data (not valid to exclude).
Reports: `agents/NN_*.md` (one per aspect; 19 is still missing at the time of writing). Only after all criteria were
fixed were they applied to the α–β correlation (`final/rules.py`, `final/rules_out.csv`).

Aspects: training/architecture (01), dataset composition (02), α validity (03), β measurement (04), layer selection (05),
tokenization (06), numeracy/training health (07), number-line geometry (08), independence/nesting (09),
E01↔E06 reproducibility (10), gap structure (11), copy-filter bias (12), exposure to the tested numbers (13),
PCA health (14), α-space design/leverage (15), literature (16), prompt context (17), number contexts in data (18),
repetition/epochs (20), checkpoint maturity (21).

## Result
**Measurement-validity flags (the only valid exclusions):**
- `dolma1_7-no-reddit`: its E01 β (1.36) fails replication. It is 1.22 on the selection seeds and 1.13 in E06 at the
  same layer; gap 1 was under-measured in E01 (auditors 04, 10, 11). Fix the value or exclude it.
- `falcon-and-cc-qc-10p`, `dolma1_7`, `fineweb-edu`: β "at the selected layer" is ill-defined. Tied layers give β
  differing by more than 0.10 in both E01 and E06 (auditor 05).

Everything else is a data property to report or control, not a reason to exclude.

**α–β with the original α_OLS (1–10000):** no valid rule reaches 0.7. All 25: r = 0.56, ρ = 0.46. Removing the 4
measurement flags: r = 0.50, ρ = 0.38. Design rules (common α support, de-duplicating overlapping corpora) bring it
down to r ≈ 0.1.

**α measured over the numbers that β probes** (auditor 13, criterion C5a: α_OLS is effectively the 1000–9999 tail
slope and ranks recipes only weakly like the counts of the tested numbers 1–39, 80–119, 980–1019, 9980–10000):

| α_tested = OLS slope of log c(n) on log n over the tested numbers | n | r | ρ |
|---|---|---|---|
| all 25 | 25 | 0.79 | 0.80 |
| without the 4 measurement flags | 21 | 0.76 | 0.74 |
| only recipes with no code/math | 17 | 0.64 | 0.69 |
| within families (family means removed) | 25 | 0.79 | 0.82 |
| partial correlation, controlling for code+math share | 25 | 0.40 (p 0.045) | |

The rows above use the pooled variant (every probed number as its own point). E11 adopts the group-median
variant; its partial correlation is 0.30 (p = 0.15); see `../E11_alpha_probed_numbers/README.md` §8.

The variant using the 4 group medians gives the same picture (r = 0.82, ρ = 0.76).

**Caveats:** this α definition was proposed by a blind auditor, but it was adopted after α_OLS had been seen to
correlate weakly, and 6 slope variants were inspected. The pooled and group-median variants agree, while single-segment
slopes do not (g1→g2 r = 0.23, g2→g3 r = 0.05, g3→g4 r = 0.66). It needs confirmation on new data (E09 Paloma, E08
checkpoints). Counts stop at 10000, so 10001–10019 of the top group are uncounted.

**Dependence check** (auditors 09 and 19 estimate about 8–11 effective independent units):
- **Family-cluster bootstrap** of r for α_tested: 95% CI [0.54, 0.92].
- **Within-family permutation test** (β shuffled only inside families, 20,000 times): p < 0.0001.
- **Family means** (6 points): r = 0.83, p = 0.04.

Auditor 19 (variance structure) adds no exclusions. It confirms dolma1_7-no-reddit as the only cross-protocol outlier
and falcon-and-cc-qc-tulu-10p as a genuine within-family outlier (a data property).
