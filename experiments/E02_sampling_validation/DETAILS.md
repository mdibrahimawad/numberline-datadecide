# E02 details

## Ground truth (full-pass counts, recomputed here)

| Corpus | α_OLS (recomputed) | summary value | R² | α_MLE |
|---|---|---|---|---|
| SlimPajama reupload | −1.5567 | −1.5567 | 0.794 | −1.0647 |
| LLM-JP v3 | −1.4598 | −1.4598 | 0.789 | −1.1149 |

The two estimators differ by 0.4–0.5 because a single power law fits poorly (R² ≈ 0.79).
MLE is dominated by the very high counts of small *n*; OLS gives each of the 10,000 values of *n*
equal weight, and 9,000 of them are above 1000. Report both; never compare one to the
other. The MLE's textbook standard error (~3e-6) is meaningless here; use the bootstrap.

## The 5 independent 1B samples (scheme B), SlimPajama

| | full corpus | mean of 5 | s.d. | mean − full | worst sample |
|---|---|---|---|---|---|
| α_MLE | −1.0647 | −1.0662 | 0.0026 | −0.0015 | 0.0055 |
| α_OLS | −1.5567 | −1.5889 | 0.0061 | **−0.032** | 0.039 |

Per replicate (from `runs.csv`, size 1B):

| scheme | rep | α_OLS | α_MLE |
|---|---|---|---|
| size_proportional | 0–4 | −1.5823, −1.5899, −1.5941, −1.5953, −1.5828 | −1.06615, −1.07020, −1.06332, −1.06470, −1.06673 |
| size_proportional_unweighted | 0–4 | −1.5855, −1.5901, −1.5972, −1.5973, −1.5831 | −1.06647, −1.07021, −1.06279, −1.06502, −1.06702 |
| uniform_files | 0–4 | −1.5958, −1.6044, −1.5946, −1.5865, −1.5586 | −1.06487, −1.06350, −1.06510, −1.06321, −1.06225 |

SlimPajama is well shuffled, so even scheme A matches. This corpus therefore can't show
that size-proportional sampling matters; mixtures like DataDecide's could.

## Why α_OLS is biased on samples (idealised check)
Thinning the full SlimPajama counts at random (a perfectly random sample) and refitting:

| sample | α_OLS | OLS bias | α_MLE | MLE bias | n in 1..10000 seen at least once |
|---|---|---|---|---|---|
| 10M | −1.060 | +0.497 | −1.0648 | 0.0000 | 42 % |
| 30M | −1.333 | +0.224 | −1.0647 | 0.0000 | 66 % |
| 100M | −1.545 | +0.012 | −1.0648 | 0.0000 | 93 % |
| 300M | −1.586 | −0.029 | −1.0648 | 0.0000 | 99.8 % |
| 1B | −1.566 | −0.009 | −1.0647 | 0.0000 | 100 % |
| 10B | | −0.001 | | ~0 | |
| 100B | | −0.0001 | | ~0 | |

- **Small samples:** about half of the numbers are missing; the surviving rare numbers are the
  lucky ones, so the tail looks too flat and the slope too shallow.
- **Medium samples:** almost everything is present, but log of small counts pulls them
  down, so the slope overshoots.
- **Real samples** are noisier than the ideal (numbers cluster within documents; the units were
  16 MB slices): at 1B the real OLS bias was −0.032 instead of −0.009.

MLE uses every count including zeros and is unbiased at all sizes.

## Design decisions
- Scheme B weights each draw by 1/size: drawing ∝ size and summing as-is would
  over-weight large row groups.
- LLM-JP uses 1,595 gzip files (~1.7 GB each) that can't be read from a random offset, so its scheme B is
  two-stage (32 files ∝ size, then random 1 MB pieces); its CIs are over files.
- The 60 % coverage threshold was a choice made here, not taken from the literature.
