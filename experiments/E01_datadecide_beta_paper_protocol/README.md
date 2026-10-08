# E01: β of the 25 DataDecide models (paper protocol)

**Question:** how compressed is the number line inside each of the 25 DataDecide 1B
models, measured exactly as in the paper?
**Status:** done (2026-10-05, Modal GPU, run by the project owner).
**Outputs:** `results/datadecide/<recipe>.json` (25 files), `results/datadecide/summary.csv`.

## Method (one paragraph)
For each model (`allenai/DataDecide-<recipe>-1B`, revision `step69369-seed-default`):
prompts `a=a,b=b,c=c,n=` with targets in 4 magnitude groups (1–39, 80–119, 980–1019,
9980–10019), k = 40 prompts per group, 3 in-context examples drawn at random,
**correct-output filter** on (greedy decoding must reproduce *n*, otherwise the slot is
resampled, up to 100 candidates). The last-token hidden state at every layer goes
through PCA, with PC1 per layer. The **layer** is chosen on prompt seeds 42–44 by the
median of √(EV·|ρ|), then **frozen** and evaluated on fresh seeds 45–47. β is fitted
from the 3 gaps between group means, two ways: `beta_direct` (geometric least squares,
with R²) and `beta_log`. Definitions: `docs/project_overview.md`.

## How to run
```bash
modal run modal_app/datadecide_app.py --dry-run                              # check all 25 repos + revision
modal run modal_app/datadecide_app.py --models dolma1_7,c4 --output-dir results/datadecide   # pilot
modal run modal_app/datadecide_app.py --output-dir results/datadecide         # all 25
```
GPU L4 (`MODAL_DATADECIDE_GPU=A10G` to switch), bf16, ≤ 5 models in parallel, model cache
on the Modal volume `numberline-datadecide-hf-cache`. Needs `transformers==4.49.0`
+ `ai2-olmo==0.6.0` (pinned in the app's image). Without Modal, the same protocol runs
on a RunPod GPU: `python -m runpod_jobs.beta` (without `--fine`).

Tests: `python tests/test_filter_correct.py` (offline) and `python tests/test_datadecide_smoke.py`
(downloads the 60M model; needs Hugging Face access).

## Headline result
- **Every model has a clear number line:** ρ = 0.90–0.96, and no group failed in any model.
- **β ranges from 0.62 to 1.36.** Dolma-trained models compress least (β ≈ 1 or above);
  web-only recipes (c4, fineweb, dclm, falcon) sit at 0.62–0.95.
- **DCLM → Dolma mixes form a smooth series:** 0.67 → 0.84 → 0.95 → 1.11 as the Dolma
  share grows from 0 to 75 %.
- **Removing code, then math** from Dolma 1.7 lowers β step by step: dolma1_7 0.98 →
  no-code 0.86 → no-math-code 0.62, the largest single drop in the table.
- Some web models reject many 4-digit prompts (fineweb-pro 41 % in group 3), i.e. they
  often fail to copy 4-digit numbers.

Full table and caveats: [DETAILS.md](DETAILS.md).
