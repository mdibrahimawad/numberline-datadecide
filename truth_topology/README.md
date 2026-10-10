# The topology of truth representations across pretraining

A pilot study, separate from the number-line project in the rest of this repository. It tracks
how the hidden states of **true and false statements** are organised across **Pythia pretraining
checkpoints**, with linear probes (as in Ravfogel et al. 2025 and Qian et al. 2024) side by side
with **persistent homology** (Fay et al. 2026, Malhotra et al. 2026) and a minimum-spanning-tree
test of whether truth is organised globally or within each topic.

The full specification is [`EXPERIMENT_BRIEF.md`](EXPERIMENT_BRIEF.md). The pre-registered hypotheses
and every fixed choice are in [`results/README.md`](results/README.md). Results will go in
[`REPORT.md`](REPORT.md).

## Status (2026-10-10)

| Phase (brief §10) | State |
|---|---|
| 1 Setup: code, refs read | done (what was reused: below) |
| 2 Data: P1, P2 + confound check | done with word counts (all pass); **rerun on the pod with the Pythia tokenizer** (`run_pipeline.sh data`) |
| 3 Validation: synthetic tests 1–6 | **all pass** (n = 1504, d = 2048) → `figures/fig0_validation.png` |
| 4 Pilot: 410m, steps 0/3000/143000 | to run on a GPU pod (Hugging Face is not reachable from the machine where this was written) |
| 5 Full run: 1.4b then 410m, P1 | to run |
| 6 Replication: P2 on 1.4b (6.9b optional) | to run |
| 7 Report | after 4–6 |

## Run on a RunPod GPU pod

One 24 GB GPU (RTX A5000 / L4 / 3090 / 4090) is enough for 410m and 1.4b in fp32. Pick a pod with
many vCPUs (≥ 16), because the analysis runs on the CPU. Use ≥ 60 GB of volume disk on `/workspace`.
Each checkpoint is deleted after extraction; states are about 160 MB per checkpoint for 1.4b.
See [`../docs/runpod.md`](../docs/runpod.md) for RunPod basics.

```bash
git clone -b claude/relaxed-keller-a4rdiz https://github.com/mdibrahimawad/numberline-datadecide /workspace/repo
cd /workspace/repo/truth_topology
bash scripts/setup_pod.sh
source /workspace/venv-tt/bin/activate
export TT_WORK=/workspace/tt HF_HOME=/workspace/hf HF_TOKEN=hf_...   # token optional, raises rate limits
tmux new -s tt
bash scripts/run_pipeline.sh data       # gate: results/confounds_*.json "passes": true
bash scripts/run_pipeline.sh validate   # gate: all six PASS (takes ~3 min)
bash scripts/run_pipeline.sh pilot      # gate: step 0 at chance; L1 high in middle layers; see logs for timing
bash scripts/run_pipeline.sh full       # 1.4b + 410m, 15 checkpoints each
bash scripts/run_pipeline.sh p2         # replication; MODEL=pythia-6.9b DTYPE=bfloat16 for stage 2
bash scripts/run_pipeline.sh report     # results/REPORT_tables.md + results/related_work.md
```

Every step can be resumed: if the pod restarts, rerun the same command. Logs are in `$TT_WORK/logs/`.
Copy `results/` and `figures/` back (or commit them; the big per-cell files are git-ignored).

## Code

| File | What it does |
|---|---|
| `tt/config.py` | every fixed choice (models, checkpoints, seeds, B, m, nulls) |
| `tt/data.py` | builds P1 (single statements) and P2 (CounterFact truth-context sequences); length and country confound checks |
| `tt/extract.py` | GPU: read-position hidden states for all layers, log-likelihood (P1), memorization/entropy (P2); downloads the next checkpoint in the background, deletes each one after use |
| `tt/linear.py` | L1 logistic probe (5-fold CV AUC and Qian's 4:1 split), L2 mass-mean probe, L1 within topic, L3 cross-topic transfer, B1; shuffled-label nulls |
| `tt/topology.py` | normalisation, topic-stratified subsamples, Ripser VR H0/H1, the 41-feature summary, W2/L∞ diagram distance, Hedges' g |
| `tt/mst.py` | T3a/b/c MST homophily (H0 merges = MST edges) |
| `tt/cells.py` | everything for one (checkpoint, layer) cell: linear, T1 + T1-null, effect sizes, T3 + permutations, diagrams for velocity |
| `tt/analyze.py` | driver: cells (parallel over layers) → velocity + geometric baselines + concentration null → max-statistic over layers → metrics.csv, onset.csv; Gaussian floor |
| `tt/validate.py` | synthetic tests 1–6 with declared pass criteria |
| `tt/figures.py` | fig0–fig4 |
| `tt/report.py` | tables for REPORT.md (onset, step-0 control, trajectories, effect sizes, concentration) |
| `tt/novelty.py` | Semantic Scholar citations of the four anchor papers, with topology/truth/dynamics flags |
| `tests/` | `pytest tests` checks the PH and MST identities, the W2 distance, subsampling, and runs the whole analysis end to end on fake states |

## What was reused from the reference repos

- **geometry-of-truth** (Marks & Tegmark, commit `5d1c630`): the datasets `cities`, `sp_en_trans`,
  `larger_than` and `counterfact.json`, downloaded at runtime (the repo has no licence file, so the
  data are not copied here). The mass-mean probe follows their `MMProbe` (difference of class means).
- **Malhotra et al.** (MIT): B = 64 / m = 160, the velocity, concentration and Gaussian-floor design,
  and the convention of finite bars only with empty → 0. **Their `summary.py` was not reused.** Its 41
  features are a different set (skew, kurtosis, gaps, …), not Fay et al.'s
  {mean, min, Q1, median, Q3, max, std} × {H0 deaths, H1 births, H1 deaths, H1 persistence,
  H1 birth/death ratio} + {total persistence, bar count, entropy} × {H0, H1}, which is what
  `tt/topology.py` implements. Their velocity uses `persim.wasserstein`, which is W1 with a Euclidean
  ground metric. The brief asks for W2 with an L∞ ground metric, implemented in `topology.wasserstein2`.
- **Ravfogel et al.**: their repo has only the toy model and notebooks, with no Pythia/CounterFact
  code. P2 is built from the description in their App. E.4, with matched pairs: the same context
  facts with true or false attributes, followed by the same final fact.
- **Qian et al.**: the single 4:1 split accuracy (`L1_split_acc`) is reported alongside the 5-fold AUC.
  The HSIC measure (L4, optional) is not implemented.

## Things to know when reading results

- **The T1-null sits at about 0.70, not 0.5.** Subsamples overlap heavily (160 of about 750 points),
  so summaries from two fixed pools are easy to tell apart. Only the T1 *gap* means anything, and
  its ceiling is about 0.3.
- **Gaussian clouds of 160 points in 2048 dimensions have about 236 H1 bars.** H1 counts are only
  meaningful relative to `gaussian_floor.json`.
- **Layer 0 at P1 is degenerate.** Every statement ends in the same token ("."), so the embedding
  layer gives (almost) the same point for every item. The code handles distance-0 points; expect
  chance values there.
- **T3 also responds to shape, not only to mean shifts.** In the shape-only synthetic test T3b and
  T3c are high (z ≈ 16–18), because MST neighbours follow the circle. T3 asks "are nearest
  neighbours same-truth?" and is not a linear measure.
- **The init → 512 transition dominates velocity**, because it is a change from random weights.
  Concentration is reported both with and without it, and H2 is judged without it.
