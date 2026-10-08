# E06: Fine β (10 magnitude groups, 3 seeds, per-prompt data saved)

**Question:** can β be measured more precisely and reproducibly than with the paper's
3 gaps, and does the α–β link survive a better measurement?
**Status:** done (2026-10-08). All 25 models on one RunPod RTX A5000 in 18 minutes,
about $0.20 including setup and the pilot.
**Outputs:** `results/beta_fine/step69369-seed-default/<recipe>.json` (all metrics) and
`<recipe>.npz` (per-prompt targets, groups and the top-5 PCA scores at **every layer**, for
each seed), `summary.csv`, and `results/beta_fine/fine_vs_data.csv` (joined with α and E01 β).

## Method
- **Groups:** 10 centres one ⅓ decade apart, 10·10^(g/3): 10, 22, 46, 100, 215, 464, 1000,
  2154, 4642, 10000, each a ±10 % band (the same width on a log scale), giving 9 gaps.
  The 4 decade groups (10, 100, 1000, 10000) are a subset, so the same run also gives a
  paper-style β (`beta_coarse`).
- **Prompts:** 100 per group per seed, seeds 45, 46, 47 (3,000 prompts per model), same
  `a=a,b=b,c=c,n=` format and correct-output filter as E01, with rejected prompts
  resampled in batches.
- **Speed:** prompts of the same token length are batched (no padding, so results are
  identical to one-by-one; tested), the model is loaded once, the next models download
  while the GPU works, and 4 models run at once.
- **Per layer:** EV, ρ, and three β fits: `beta_coarse` (4 groups), `beta_fine` (9 gaps,
  reported per decade), and `beta_cont` (a continuous fit over all 3,000 prompts). Fit
  quality is the relative error (not R², which is blind near β = 1).
- **Layer:** selected by the same √(EV·|ρ|) rule, **and** every metric is also reported
  at the layer E01 selected (`*_oldlayer`).

## How to run
```bash
# on a RunPod GPU pod (24 GB card; full steps in docs/runpod.md section 4)
bash runpod_jobs/setup.sh gpu
python -m runpod_jobs.beta --fine --models c4 --parallel 1                 # pilot (~1 min)
python -m runpod_jobs.beta --fine --parallel 4 --upload-hf numberline-beta-results \
    --delete-pod-when-done --max-hours 1 2>&1 | tee beta.log              # all 25
# laptop
python -m runpod_jobs.fetch_results --kind beta
python -m src.beta_fine_analysis                                            # -> fine_vs_data.csv
python -m pytest -q tests/test_beta_fine.py
```

## Headline result
- **β is reproducible:** across prompt seeds, β moves by only about ±0.015 (median s.d.),
  against a spread of 0.55–1.38 between models.
- **The old β holds up:** at E01's layer, the new 4-group β matches the old one at r = 0.96.
- **The continuous fit works** (median R² 0.89). The 9 fine gaps are too noisy to use
  directly (fit error ≈ 65 %), because neighbouring ⅓-decade groups are too close together.
- **Layer choice is the fragile part:** 8 of 25 models switch layer, all on near-ties
  (scores within 0.7–7.7 %), mostly to early layers 2–3 with much higher β
  (falcon-and-cc-qc-tulu-10p: 0.81 at layer 9, 1.38 at layer 2).
- **α vs β:** at the old layer, the link is a bit stronger than in E01 (r = +0.63, p = 0.001;
  within families +0.40, p = 0.049). With the newly selected layers it weakens to
  0.37 / 0.20, because of those near-tie switches. Compare models at a fixed layer.

Per-model table and the near-tie table: [DETAILS.md](DETAILS.md).
