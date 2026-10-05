# modal_app — cloud fan-out for the full-Pile sweep

Runs the digit-token frequency counter across **`monology/pile-uncopyrighted`**
at scale by fanning out over Modal workers. Keeps the local
`python -m src.cli` pipeline completely unchanged so small reproducible
runs still work the same way.

## Why `monology/pile-uncopyrighted` and not the original Pile?

- The original `EleutherAI/pile` was withdrawn in 2023 over Books3
  copyright issues. It is no longer publicly redistributable.
- `monology/pile-uncopyrighted` is **the Pile minus Books3** and is the
  de-facto academic replacement used in current numeracy research.
- Crucially, it is **not deduplicated**, so the raw frequency
  distribution that LLMs actually see during pretraining is preserved.
  This matters for your research question: identifying which numbers
  cluster under Zipf-like vs. uniform vs. Gaussian-like behaviour is
  sensitive to dedup.
- Books3's removal is mostly prose fiction with low digit-token density;
  the resulting distributional bias on integers in `[0, 1000]` is
  negligible compared to the bias introduced by deduplication.

Shape on disk: 30 train shards as `train/00.jsonl.zst` … `train/29.jsonl.zst`,
~335 GB compressed total.

## What this does differently from the local pipeline

|  | Local (`python -m src.cli`) | Modal (`modal run modal_app/app.py`) |
|---|---|---|
| Counts | **one** integer at a time (`\bN\b`) | **all** digit tokens in a single regex pass (`\b\d+\b`), filtered to `[0, 1000]` at aggregation |
| Parallelism | single process, streaming | N parallel Modal containers, each processing a disjoint set of Pile shards |
| Dataset | whatever `utils/dataset.py` streams | `monology/pile-uncopyrighted` `.jsonl.zst` shards, downloaded once into a persistent Modal Volume |
| Typical scale | 10² – 10⁵ docs | 10⁶ – 2 · 10⁸ docs |
| MLflow | one metric per run | full frequency table in one run, logged as both per-N metrics and a CSV artifact |

The Modal workers reuse only `src/multi_counter.py` and `utils/text.py`;
`utils/dataset.py` (HF streaming loader) is not touched by the cloud path.

## One-time setup

```bash
.venv/bin/pip install modal huggingface_hub
.venv/bin/modal setup       # opens browser, stores token in ~/.modal.toml
```

No HF login is needed — `monology/pile-uncopyrighted` is a public,
ungated dataset.

## Architecture: two-stage pipeline

### Stage 1 — download into a Modal Volume

`download_shard` fetches each `.jsonl.zst` from HuggingFace Hub into a
persistent Modal Volume named `pile-uncopyrighted-shards`.

- Runs `.map()` across all 30 shards in parallel → full 335 GB lands on
  the Volume in ~10–30 minutes depending on HF bandwidth.
- **Re-entrant**: re-running the command skips shards that are already
  cached in the Volume.

### Stage 2 — stream-decompress + count

`count_shard` opens each `.jsonl.zst` directly from the Volume, streams
it through `zstandard.ZstdDecompressor`, parses one JSON object per
line, and feeds the raw `text` field to `run_multi_count`, which
applies a single `\b\d+\b` regex pass and accumulates into a `Counter`.

Workers return their `Counter` as a plain dict; the local entrypoint
sums them and writes a single MLflow run.

## Run

First, dry-run to verify shard discovery without any cloud spend:

```bash
.venv/bin/modal run modal_app/app.py --dry-run
```

Pre-populate the Volume (costs bandwidth only, no counting):

```bash
.venv/bin/modal run modal_app/app.py --download-only
```

Smoke test the counting path on 100k docs (2 workers):

```bash
.venv/bin/modal run modal_app/app.py \
  --max-docs 100000 \
  --n-workers 2 \
  --skip-download       # Volume already primed from previous step
```

Full 200M-doc sweep, one worker per shard:

```bash
.venv/bin/modal run modal_app/app.py \
  --max-docs 200000000 \
  --n-workers 32 \
  --skip-download \
  --experiment-name numberline_freq_count_expv1 \
  --run-name modal_full_sweep_200M
```

### Flags

- `--max-docs` — total documents across *all* workers. The app divides
  evenly, so each worker processes `max_docs // n_workers` docs.
- `--n-workers` — Modal parallelism. Capped at number of shards (30),
  since a single `.jsonl.zst` can't be split across workers without
  seekable decompression.
- `--download-only` — only run Stage 1, exit before counting.
- `--skip-download` — skip Stage 1, assume Volume already has shards.
- `--dataset-repo` — override the HF dataset repo (e.g. pass
  `EleutherAI/the_pile_deduplicated` for a dedup ablation).
- `--dry-run` — print the shard plan and exit without launching
  containers.

## What gets logged

The local entrypoint writes to your normal MLflow SQLite store
(`./mlflow.db`) as a *single run* containing:

- **params**: `target_max_docs`, `n_workers_requested`,
  `n_workers_effective`, `max_docs_per_worker`, `n_shards_total`,
  `log_every`, `dataset_repo`.
- **metrics**: `docs_seen`, `total_matches`, `elapsed_seconds`,
  `avg_matches_per_doc`, `n_integers_observed_in_range`, and one
  `count_N_0000` … `count_N_1000` entry per integer in `[0, 1000]`.
- **artifacts**:
  - `counts_0_to_1000.csv` — the authoritative frequency table, the
    input for your PCA / distribution-grouping analysis.
  - `worker_stats/worker_<i>.txt` — per-worker filenames / docs /
    matches for post-mortem load-balance analysis.
- **tags**: `stage=number_frequency`, `backend=modal`,
  `dataset=monology/pile-uncopyrighted`, `counter_mode=digit_boundary_multi`.

View locally with:

```bash
.venv/bin/mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" --port 5555
```

(Don't use port 5000 on macOS — Apple's AirPlay Receiver claims it.)

## Cost / time expectations

`monology/pile-uncopyrighted` is 30 train shards, ~11 GB compressed
each, ~335 GB total, ~200M documents total.

### Stage 1 (download)

One-time cost. ~30 shards × parallel download → mostly bounded by
HuggingFace CDN egress bandwidth.

| Parallelism | Wall-clock | Modal cost (≈ CPU-minute) |
|---|---|---|
| 30 (one per shard) | 15–45 min | $0.30 – $1 |

### Stage 2 (counting) — dominated by zstd + regex on CPU

| `--n-workers` | Wall-clock for 200M docs | Rough Modal cost |
|---|---|---|
| 8 | 3–5 h | $2–5 |
| 16 | 1.5–2.5 h | $2–6 |
| 32 (one per shard) | 45–90 min | $3–8 |

These are back-of-envelope numbers — actual cost depends on Modal's
current pricing and how much zstd-decompression bottlenecks each
worker. **Run the smoke test first** to calibrate on your account.

## Why underscore in `modal_app`?

Because a top-level directory named `modal/` shadows `import modal`
whenever the project root is on `PYTHONPATH`. The underscore avoids
the collision.
