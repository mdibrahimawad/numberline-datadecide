
Low and high bounds for the uncertain filter-filter pairs are in `scratchpad/agents/09/rules_low.txt` and `rules_high.txt`. The high bound turns several pairs into near-duplicates: f&cc q10/qO cos 0.96, q10/qT 0.93, dclm fw3/fw10 0.91, q20/qO 0.72, 7fw3/fw10 0.67.

### Effective number of independent points (design-only, no outcome used)
For a correlation between two variables measured on N units whose dependence matrix is R, the Clifford, Richardson & Hémon (1989) / Dutilleul (1993) effective sample size is n_eff = 1 + N²/tr(R²).

| R used | point | high-overlap bound |
|---|---|---|
| cos (doc-iid; lower bound on dependence) | 8.2 | 6.8 |
| 1-TV | 10.7 | 9.0 |
| max(C(A in B), C(B in A)) (any nesting counts as full dependence) | 4.0 | 3.6 |

So the 25 points carry the information of roughly **4 to 11 independent points**, depending on whether nested selections count as dependent. With all 25 treated as independent, df is overstated by a factor of about 2.3 to 6.

## 5. Rules and kept/dropped lists

Pre-specified tie-break for every greedy rule: process the DataDecide "Original" recipes first (README labels: dclm-baseline, dolma1_7, dolma1_6plus, falcon-and-cc, falcon, fineweb-edu, c4, fineweb-pro, ordered by corpus size), then the derived recipes by decreasing size. A recipe is dropped if its measure against an already-kept recipe exceeds X. This order is fixed by provenance, not by any outcome. One sensitivity case uses pure size order; it shows that the order matters (mix25 becomes the Dolma representative), which is why the provenance order is pre-specified.

**Mixtures excluded a priori (R-mix).** For additive count data, a mix is (up to the random file subset) a convex combination of two other recipes in the set: c_mix(n) ≈ λ·c_DCLM(n) + (1-λ)·c_Dolma(n). It adds no new independent corpus. It is a legitimate dose-response treatment for the model side, but on the data side it is a derived unit. Dropping derived or linear-combination units is the standard fix (in Hurlbert's terms they are not independent experimental units). Drop: mix25, mix50, mix75.

| Rule | Measure, X | Justification | KEEP | n |
|---|---|---|---|---|
| **R0** all | none | baseline; pseudo-replicated | all 25 | 25 (n_eff 4-11) |
| **R1** one point per independent lineage (components, mixes removed) | containment > 0.05, any | Hurlbert 1984: experimental unit = independent source corpus | dclm-baseline, dolma1_7, fineweb-edu, fineweb-pro (strict variant with FineWeb merged: 3) | 4 (3) |
| **R2** keep base, drop subsets | C(A in kept) > 0.5, provenance order, R-mix | a recipe whose training docs are mostly (>50%) inside another kept recipe is a subsample of that unit. Any X in (0.01, 0.89) gives the same list, because filtered subsets and ablations have C = 1.0 by construction (the lowest relevant pair, fineweb-pro in fineweb-edu, is 0.01) | dclm-baseline, dolma1_7, fineweb-edu, fineweb-pro | 4 |
| R2b | same, size order, mixes kept | sensitivity | dclm-baseline, mix25, fineweb-edu, fineweb-pro | 4 |
| **R3** shared-variance rule | cos > 0.5, provenance order, R-mix | cos is the expected correlation induced by shared documents; > 0.5 means > 25% shared variance | dclm-baseline, dolma1_7, fineweb-edu, c4, fineweb-pro, dclm-7fw2, f&cc-q20, dclm-q20, f&cc-qO, f&cc-qT, dclm-fw3, dclm-fw10 | 12 |
| R3 under high-overlap bound | cos > 0.5 | sensitivity | dclm-baseline, dolma1_7, fineweb-edu, c4, fineweb-pro, dclm-7fw2, f&cc-q20, dclm-fw3 | 8 |
| R3c | cos > 0.3 (point) | sensitivity | dclm-baseline, dolma1_7, fineweb-edu, c4, fineweb-pro, dclm-7fw3, dclm-q20, f&cc-qO, f&cc-q10, f&cc-qT, dclm-fw3, dclm-fw10 | 12 (7 at high bound) |
| R3d | cos > 0.7 (point) | sensitivity | R3 list + falcon, dclm-7fw3, dclm-q10 | 15 (11 at high bound) |
| **R4** near-duplicate (strict pseudo-replicate) rule | 1-TV > 0.5, R-mix | drop a model whose training distribution is more than half identical to a kept one, i.e. the same treatment replicated | dclm-baseline, dolma1_7, falcon, fineweb-edu, c4, fineweb-pro, dclm-7fw2, f&cc-q20, dclm-7fw3, dclm-q20, f&cc-qO, f&cc-qT, dclm-fw3, dclm-fw10, dclm-q10 | 15 (11 at high bound) |
| R4b | 1-TV > 0.8 | only near-identical distributions | R4 + falcon-and-cc, dolma1_7-no-math-code, f&cc-q10 | 18 (15 at high bound) |

Drop reasons, point estimates, with the measure against the kept recipe that triggered the drop:
- **R2:** d16p (in d17: 0.89 / 0.98); f&cc, falcon, c4, all 4 dolma ablations, all 4 f&cc-qc (C = 1.00 in d17); all 6 dclm-qc (C = 1.00 in dclm-baseline); 3 mixes (R-mix).
- **R3 (cos > 0.5):** d16p (0.93), f&cc (0.78), falcon (0.51, borderline), d17-nf (1.00), -nr (0.98), -nc (0.91), -nmc (0.89), dclm-7fw3 (vs 7fw2: 0.66), f&cc-q10 (vs q20: 0.71), dclm-q10 (vs q20: 0.69), 3 mixes.
- **R4 (1-TV > 0.5):** d16p (0.89), f&cc (0.62), d17-nf (0.99), -nr (0.95), -nc (0.83), -nmc (0.79), f&cc-q10 (vs q20: 0.51, borderline), 3 mixes.

**Recipes kept by every rule:** dclm-baseline, dolma1_7, fineweb-edu, fineweb-pro. **Recipes dropped by every rule except R0:** dolma1_6plus, dolma1_7-no-flan, -no-reddit, -no-code, the 3 mixes. **Most fragile decisions:** falcon (cos 0.51 to d17), f&cc-q10 (1-TV 0.51 to q20), dclm-q10 (1-TV 0.48 to q20), and every dclm or f&cc cross-filter pair whose overlap is unknown.

### Recommended design (stated before seeing any outcome)
1. **Primary inferential analysis: keep all 25 and model the dependence; do not delete.** Use lineage (Dolma / DCLM / FineWeb; mixes assigned to both through multiway clustering, or excluded) as a blocking or random factor, or use GLS with R = the overlap matrix and report n_eff-adjusted tests. Remedies for pseudoreplication: Hurlbert 1984; Millar & Anderson 2004 (Fish. Res.); Lazic 2010 (BMC Neurosci.). Caveat: with 3-4 clusters, cluster-robust SEs are invalid; use a wild cluster bootstrap or exact within-block permutation (Cameron, Gelbach & Miller 2008).
2. **Identification comes from within-lineage contrasts** (filter or ablation versus its base, with lineage fixed effects). This is a randomized-block / split-unit design: valid for "does manipulating the data of one corpus move both quantities together", but per Hurlbert it does not generalize across corpora, because the treatments are applied to subsamples of a single unit.
3. **Robustness check: R2 (n = 4) and R3 (n = 12; 8 at the high bound), both reported whatever their result.** Design fact: the two-sided p < .05 critical |r| is 0.950 at n = 4, 0.707 at n = 8, 0.576 at n = 12, 0.514 at n = 15 and 0.396 at n = 25. R2 is therefore descriptive only. R3 or R4 is the largest set that is defensible as near-independent.
4. Report R4 alongside R3 because they differ in principle. R4 removes only strict pseudo-replicates (near-identical training distributions). R3 also removes nested filters of moderate size.

### Which rules are standard
- **Standard:** one unit per independent experimental unit, i.e. avoiding pseudo-replication (Hurlbert 1984, Ecol. Monogr. 54:187); treating subsamples, sub-filters and repeated measures of one unit as non-independent; excluding derived or linear-combination units (the mixes); effective sample size under correlated observations (Clifford et al. 1989; Dutilleul 1993); the overlapping-sample correlation n_shared/sqrt(n1·n2) (Lin & Sullivan 2009); blocking / mixed models / cluster bootstrap as the alternative to deletion.
- **Not standard (convention chosen here):** the thresholds X = 0.5 for containment, cos and 1-TV, and the provenance-first tie-break order. There is no canonical X. 0.5 was chosen as "majority shared" (for cos, 25% shared variance), with 0.3, 0.7 and 0.8 shown as sensitivity. R2's list does not depend on X for any X in (0.01, 0.89).

## 6. Honest notes and limitations
- **Two measures give opposite answers for filtered subsets.** By containment, every qc filter is 100% inside its base (a pseudo-replicate). By shared distribution (1-TV, cos), top-k% filters are mostly *different* data from their base (1-TV 0.02-0.27). The right view depends on whether the filter is the treatment (then the filter is a valid manipulation and the base is its block) or noise (then it is a sub-sample). Pseudo-replication in Hurlbert's strict sense applies to random subsamples; the ablations (1-TV 0.79-0.99) come closest to that.
- **Lower bound on dependence.** All cross-pipeline overlaps (C4, RefinedWeb, Dolma CC, DCLM and FineWeb all from Common Crawl, with overlapping snapshots and URLs) are set to 0 because they were not measured. Pipeline-level shared conventions (extraction, number formatting, dedup) are not captured by any document-overlap measure. True dependence is higher than the table shows.
- **Assumptions.** qc docs are spread uniformly over base files (needed for qc-vs-mix overlaps). Dolma-1.7 CC and C4 are doc-subsets of the Dolma-1.6 versions (CC re-binned head/middle/tail; 1.7 middle bin 242 B > 1.6 middle 190 B, so containment may be < 100%). The f&cc-qc CC part is 17% of Dolma-1.7 CC tokens at "top 10%", which suggests a doc-count percentile or a larger pre-filter pool, so containment in d17 may be < 1. Tulu v0 vs v2 is treated as disjoint (≤ 1.8 B). Per-file tokens are per-pool averages; mix sizes come out 0.6-0.7% off the mixture.json totals.
- **Unknown cross-filter overlaps** within DCLM and within falcon-and-cc dominate the uncertainty in R3 and R4 (12 vs 8, 15 vs 11 kept). Exact Jaccard on document IDs or URLs from the HF files (`allenai/DataDecide-data-recipes`) would settle them. HF and arXiv were blocked from this session.
- Realized token identity between trained streams is small (< 10%) except for nested small filters, so training-sample overlap is a minor issue compared with population overlap.
- No outcome was used: rule order, thresholds and the recommended primary analysis were fixed from provenance and design arguments only.
