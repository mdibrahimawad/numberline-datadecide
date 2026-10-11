# E10: Forecasting memorization early in pretraining with topological features

**Question:** do topological features of a training sequence's hidden states at an early
Pythia checkpoint (S_f = 10k, 23k) predict whether the model will have memorized it
(32-extractable) at a later checkpoint T? And do they add anything **beyond** loss signals,
data statistics and a linear probe on the same hidden states?
**Status:** **code ready and tested offline. Not run yet** (it needs a GPU pod). The hypotheses
are pre-registered in [`REPORT.md`](REPORT.md), and the full specification is [`BRIEF.md`](BRIEF.md).
**Written:** 2026-10-11.

This is a separate research line from the number-line work (E00–E09). It shares only the
RunPod setup ([`docs/runpod.md`](../../docs/runpod.md)).

## Files

| Path | What |
|---|---|
| `BRIEF.md` | the experiment brief (specification; sections 4–9) |
| `REPORT.md` | pre-registration (H1–H4 and the success criterion, locked) + the report skeleton |
| `configs/memtopo.yaml` | every seed, model, revision, layer list, normalisation, S_f, T |
| `memtopo/pile.py` | reader for the pre-shuffled training shards (local memmap + HTTP range reads); epoch-boundary search |
| `memtopo/scan.py` | one checkpoint pass: loss family, extractability, hidden-state hooks → Ripser on CPU workers |
| `memtopo/topology.py` | normalisation, Vietoris–Rips diagrams, the 41-feature summary, Wasserstein-p, B6 geometry, Gaussian floor |
| `memtopo/textstats.py` | B4 data statistics |
| `memtopo/pipeline.py` | stages: data, align, labels, case-control, scans, features |
| `memtopo/analysis.py` | ladder, deltas (bootstrap + DeLong), residual-by-loss-decile, H2/H3 effects, controls |
| `memtopo/figures.py`, `report.py`, `novelty.py`, `cli.py` | figures 0–5, `results/summary.md`, section 12, command line |
| `runpod_jobs/memtopo_setup.sh` | pod setup (venv, packages, GPU check, tests, reference repos) |
| `tests/test_memtopo.py` | offline tests (12; see below) |

## How to run (GPU pod: 1× A100/H100 80 GB, ≥ 32 vCPU, ≥ 300 GB volume)

```bash
cd /workspace && git clone -b claude/dazzling-fermat-1l665q https://github.com/mdibrahimawad/numberline-datadecide.git
cd numberline-datadecide && bash runpod_jobs/memtopo_setup.sh
export HF_TOKEN=hf_...            # read access is enough
tmux new -s e10                   # every stage resumes if interrupted: just rerun it

python -m memtopo.cli validate                 # gate 4: topology code on synthetic clouds
python -m memtopo.cli data                     # shards for steps 0-5999, epoch boundary, pool, B4
python -m memtopo.cli align --model 1b         # gate 2: must print 200/200
python -m memtopo.cli labels --model 1b        # gate 3: prevalence, >= 300 hard test positives
python -m memtopo.cli pilot --model 1b --sf 10000   # gate 5: go / no-go (look at results/summary.md)
# if go:
python -m memtopo.cli pilot --model 1b --sf 23000
python -m memtopo.cli align --model 410m && python -m memtopo.cli labels --model 410m
python -m memtopo.cli pilot --model 410m --sf 10000 && python -m memtopo.cli pilot --model 410m --sf 23000
python -m memtopo.cli analyze --model 1b --sf 10000 --target mem_final     # secondary target T = 143000
python -m memtopo.cli figures && python -m memtopo.cli report && python -m memtopo.cli novelty
```

Outputs go to `/workspace/memtopo_run/` (`data/ features/ results/ figures/ logs/ scans/ gates.json`).
`gates.json` collects every checked fact and gate: layout, epoch boundary, alignment, label
fragility, greedy check, prevalence, layers, and the checkpoints each feature set uses.
Copy `results/`, `figures/` and `gates.json` back into `results/memtopo/` in this repo.
Delete `scans/` and the shards at the end, because they are large.

**Stop at each gate.** If the alignment check is below 200/200, fix the indexing; `gates.json`
has a shift diagnosis. If there are fewer than 300 hard test positives, enlarge `pool_size`
before continuing. The pilot's go/no-go rule is in BRIEF §10.5: do not tune toward a positive result.

## Design decisions not spelled out in the brief (also listed in REPORT.md §7)

1. **Extractability by teacher forcing.** Greedy decoding reproduces the 32 true tokens
   exactly when the argmax at every continuation position, given the true prefix, is the
   true token. So `mem` and `match_len` come from one forward pass over the 64 tokens, not
   from 32 decoding steps. The two are equivalent in exact arithmetic, and a unit test
   checks this. Numerics can differ, so `labels` also runs real KV-cache greedy decoding on
   2000 sequences and records the agreement (`greedy_check_*`). The precision is fp32.
2. **Epoch boundary from text overlap, not identical sequences.** GPT-NeoX re-shuffles and
   re-packs documents in each epoch, so second-epoch sequences are generally not
   byte-identical to first-epoch ones. Instead we hash 32-token shingles from steps 0–5999,
   leaving out shingles that already repeat within that range. We then count matches per
   step: a coarse scan from 88k to 112k every 250 steps, then bisection. Epoch-1 steps
   should give about 0 matches. T = the last multiple of 1000 at or before the first
   matching step.
3. **Wasserstein-2 is our own implementation.** It uses a Euclidean ground metric and
   diagonal matching. `persim.wasserstein` only does p = 1; with p = 1 ours equals persim
   to 1e-15 (checked).
4. **The 41-feature summary is implemented from Fay et al.'s description.** `cli refs`
   imports Malhotra's `FEATURE_NAMES`, if that repo exists, so the two lists can be compared.
   Empty H1 gives zeros (`h1_n_bars` = 0 carries the fact).
5. **Hard-subset models are fit on hard train rows only** (B0 is constant there). The
   full-set models are fit on all rows.
6. **H4 gets an extra rung, "B0–B6 + T0".** H4 is then tested as rung 8 vs this rung (does
   topology still add something once step-0 topology is in the baseline). Rung 8 vs 7 is
   also reported, as the brief asks.
7. **Probe:** standardise, then PCA(256) fitted on the train rows only, then joined to the
   standardised columns in one L2 logistic regression. C is chosen by inner 5-fold CV.
   Mean-pooled states are stored as `probe.npz`, not parquet.
8. **Min-K%** (k = 20%) is computed over all 63 predicted tokens. All loss features are
   computed in-house, not taken from mimir.
9. **Case-control:** per target (`mem_T`, `mem_final`) and per seed (0, 1), all positives
   plus 5× random negatives per split. Expensive features are computed for the union.
   AUPRC and recall@P0.9 reweight the test negatives to the pool's natural prevalence.
10. **H2/H3 effect sizes** use both splits of the hard subset, matched within loss deciles
    (H2). They are descriptive; the predictive claims rest on the test split only.

## Tested offline (this container: no GPU, no Hugging Face access)

`python -m pytest -q tests/test_memtopo.py`: 12 tests pass. They cover the topology
summary, scale invariance, the circle/Gaussian checks (gate 4 also passes through `cli
validate`), Wasserstein, geometry, the layer mapping (1b → 3, 5, 8, 10, 13, 16; 410m → 5, 8,
12, 15, 20, 24), the metrics against sklearn (including weighted AUPRC), DeLong, and effect
sizes. They also cover the sharded reader on synthetic shards (including sequences that
straddle shards) and the epoch-boundary search on a synthetic two-epoch stream. With a
tiny random GPT-NeoX, they check that teacher-forced extractability equals real greedy
decoding and that a scan writes and resumes. A smoke run of the whole analysis and the
figures on synthetic features is included.

**Not yet checked against the real data** (each fails loudly at its gate): the layout of
the shard files, the indices in `pythia-memorized-evals`, the split names (`deduped.1b`,
`deduped.410m`, `deduped.6.9b`), the epoch boundary, and the reference repos.
