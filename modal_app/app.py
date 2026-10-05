from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "pile-number-frequency"

DEFAULT_DATASET_REPO = "monology/pile-uncopyrighted"
DATA_VOLUME_NAME = "pile-uncopyrighted-shards"
DATA_PATH = "/data"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub>=0.20.0",
        "zstandard>=0.22.0",
        "orjson>=3.10.0",
    )
    .add_local_python_source("src", "utils")
)


app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)


@app.function(
    timeout=2 * 60 * 60,
    cpu=1.0,
    memory=2048,
    volumes={DATA_PATH: data_volume},
)
def download_shard(filename: str, repo: str = DEFAULT_DATASET_REPO) -> dict:
    from huggingface_hub import hf_hub_download

    local_path = Path(DATA_PATH) / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        print(
            f"[download] {filename} cached "
            f"({local_path.stat().st_size / 1e9:.2f} GB) — skip"
        )
        return {"filename": filename, "size": local_path.stat().st_size, "cached": True}

    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] fetching {filename} ...")
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        local_dir=DATA_PATH,
    )
    data_volume.commit()

    size = local_path.stat().st_size
    print(f"[download] done {filename}: {size / 1e9:.2f} GB")
    return {"filename": filename, "size": size, "cached": False}


@app.function(
    timeout=3 * 60 * 60,
    cpu=2.0,
    memory=8192,
    volumes={DATA_PATH: data_volume},
)
def count_shard(
    filenames: list[str],
    max_docs: int | None = None,
    log_every: int = 50_000,
    results_subdir: str = "_results/default",
) -> dict:
    # Worker also persists its result JSON to the Volume so the aggregation
    # step can recover after a local-client disconnect without recounting.
    import hashlib
    import io
    import json

    import orjson
    import zstandard as zstd

    from src.multi_counter import run_multi_count

    data_volume.reload()

    def _iter_examples():
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        for fname in filenames:
            path = Path(DATA_PATH) / fname
            print(f"[worker] opening {fname} ({path.stat().st_size / 1e9:.2f} GB)")
            with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    if not line:
                        continue
                    try:
                        obj = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        continue
                    yield obj

    cap = max_docs if max_docs is not None else 10**18

    def _progress(update):
        print(
            f"[worker] docs_seen={update.docs_seen:,} "
            f"total_matches={update.total_matches:,}"
        )

    result = run_multi_count(
        _iter_examples(),
        max_docs=cap,
        log_every=log_every,
        progress_callback=_progress,
    )

    payload = {
        "docs_seen": result.docs_seen,
        "total_matches": result.total_matches,
        "counts": dict(result.counts),
        "n_files": len(filenames),
        "filenames": filenames,
    }

    results_dir = Path(DATA_PATH) / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1("\n".join(filenames).encode()).hexdigest()[:16]
    out_path = results_dir / f"{key}.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
    data_volume.commit()
    print(f"[worker] persisted result to {out_path}")

    return payload


@app.function(
    timeout=30 * 60,
    cpu=1.0,
    memory=4096,
    volumes={DATA_PATH: data_volume},
)
def read_persisted_results(results_subdir: str) -> list[dict]:
    import json

    data_volume.reload()
    results_dir = Path(DATA_PATH) / results_subdir
    if not results_dir.exists():
        return []
    out: list[dict] = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as fh:
            out.append(json.load(fh))
    print(f"[recover] loaded {len(out)} result files from {results_dir}")
    return out


@app.function(
    timeout=30 * 60,
    cpu=2.0,
    memory=8192,
    volumes={DATA_PATH: data_volume},
)
def aggregate_on_cloud_range(
    results_subdir: str, n_min: int = 0, n_max: int = 1000,
) -> dict:
    # Same aggregation as `aggregate_on_cloud` but with a configurable
    # output range. Workers persist *unfiltered* counters to the Volume,
    # so any window we want is one re-aggregation away.
    import json
    from collections import Counter

    from src.multi_counter import filter_to_range

    data_volume.reload()
    results_dir = Path(DATA_PATH) / results_subdir
    if not results_dir.exists():
        raise FileNotFoundError(f"no results dir at {results_dir}")

    combined: Counter[str] = Counter()
    total_docs = 0
    total_matches = 0
    n_files_total = 0
    worker_summaries: list[dict] = []

    paths = sorted(results_dir.glob("*.json"))
    for i, p in enumerate(paths, 1):
        with open(p) as fh:
            r = json.load(fh)
        combined.update(r["counts"])
        total_docs += r["docs_seen"]
        total_matches += r["total_matches"]
        n_files_total += r["n_files"]
        worker_summaries.append(
            {
                "filenames": r["filenames"],
                "n_files": r["n_files"],
                "docs_seen": r["docs_seen"],
                "total_matches": r["total_matches"],
            }
        )
        if i % 5 == 0 or i == len(paths):
            print(f"[aggregate-range] merged {i}/{len(paths)} worker files")

    filtered = filter_to_range(combined, n_min, n_max)
    print(
        f"[aggregate-range] window=[{n_min},{n_max}]  "
        f"docs={total_docs:,}  matches={total_matches:,}  "
        f"in_range={len(filtered)}  total_distinct_strings={len(combined):,}"
    )
    return {
        "n_min": n_min,
        "n_max": n_max,
        "total_docs": total_docs,
        "total_matches": total_matches,
        "n_integers_observed_in_range": len(filtered),
        "n_distinct_integers_total": len(combined),
        "n_shards_total_from_workers": n_files_total,
        "n_workers": len(worker_summaries),
        "counts_in_range": {str(k): v for k, v in filtered.items()},
        "worker_summaries": worker_summaries,
    }


@app.function(
    timeout=30 * 60,
    cpu=2.0,
    memory=8192,
    volumes={DATA_PATH: data_volume},
)
def aggregate_on_cloud(results_subdir: str) -> dict:
    # Sum every worker Counter on the Volume side and return only the
    # compact summary (counts in [0,1000] + per-worker stats) so the
    # client doesn't have to pull ~1 GB of raw Counter JSONs.
    import json
    from collections import Counter

    from src.multi_counter import filter_to_range

    data_volume.reload()
    results_dir = Path(DATA_PATH) / results_subdir
    if not results_dir.exists():
        raise FileNotFoundError(f"no results dir at {results_dir}")

    combined: Counter[str] = Counter()
    total_docs = 0
    total_matches = 0
    n_files_total = 0
    worker_summaries: list[dict] = []

    paths = sorted(results_dir.glob("*.json"))
    for i, p in enumerate(paths, 1):
        with open(p) as fh:
            r = json.load(fh)
        combined.update(r["counts"])
        total_docs += r["docs_seen"]
        total_matches += r["total_matches"]
        n_files_total += r["n_files"]
        worker_summaries.append(
            {
                "filenames": r["filenames"],
                "n_files": r["n_files"],
                "docs_seen": r["docs_seen"],
                "total_matches": r["total_matches"],
            }
        )
        if i % 5 == 0 or i == len(paths):
            print(f"[aggregate] merged {i}/{len(paths)} worker files")

    filtered = filter_to_range(combined, 0, 1000)
    print(
        f"[aggregate] total_docs={total_docs:,} "
        f"total_matches={total_matches:,} "
        f"integers_in_range={len(filtered)}"
    )
    return {
        "total_docs": total_docs,
        "total_matches": total_matches,
        "n_integers_observed_in_range": len(filtered),
        "n_distinct_integers_total": len(combined),
        "n_shards_total_from_workers": n_files_total,
        "n_workers": len(worker_summaries),
        "counts_in_range": {str(k): v for k, v in filtered.items()},
        "worker_summaries": worker_summaries,
    }


def _discover_shards(repo: str = DEFAULT_DATASET_REPO) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    return sorted(
        f for f in files if f.endswith(".jsonl.zst") and f.startswith("train/")
    )


@app.local_entrypoint()
def main(
    max_docs: int = 200_000_000,
    n_workers: int = 32,
    experiment_name: str = "numberline_freq_count_expv1",
    run_name: str = "modal_sweep",
    log_every: int = 50_000,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    download_only: bool = False,
    skip_download: bool = False,
    aggregate_only: bool = False,
    results_subdir: str = "",
    dry_run: bool = False,
) -> None:
    # If the local client disconnects mid-count, re-run with
    # --aggregate-only --results-subdir <same> to rebuild from persisted
    # per-worker JSONs on the Volume.
    import csv
    import io
    import time
    from collections import Counter
    from datetime import datetime

    import mlflow

    from src.multi_counter import filter_to_range
    from utils.mlflow_utils import (
        log_metric,
        log_metrics,
        log_params,
        setup_tracking,
        start_run,
    )

    if n_workers < 1:
        raise SystemExit("--n-workers must be >= 1")
    if max_docs < 1 and not download_only and not aggregate_only:
        raise SystemExit("--max-docs must be >= 1 (unless --download-only or --aggregate-only)")

    if not results_subdir:
        results_subdir = f"_results/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    print(f"[modal] results_subdir = {results_subdir} (on volume {DATA_VOLUME_NAME})")

    shards = _discover_shards(dataset_repo)
    if not shards:
        raise SystemExit(
            f"no train-split .jsonl.zst shards discovered in {dataset_repo}"
        )
    print(f"[modal] discovered {len(shards)} shards in {dataset_repo}:")
    for s in shards[:5]:
        print(f"  {s}")
    if len(shards) > 5:
        print(f"  ... ({len(shards) - 5} more)")

    if dry_run:
        print("[modal] --dry-run set; exiting before any cloud work")
        return

    if not skip_download and not aggregate_only:
        print(f"[modal] stage 1: download {len(shards)} shards into Volume ...")
        dl_start = time.time()
        dl_results = list(
            download_shard.map(shards, kwargs={"repo": dataset_repo})
        )
        dl_elapsed = time.time() - dl_start
        total_bytes = sum(r["size"] for r in dl_results)
        n_cached = sum(1 for r in dl_results if r["cached"])
        print(
            f"[modal] stage 1 done in {dl_elapsed:.1f}s: "
            f"{total_bytes / 1e9:.1f} GB on volume "
            f"({n_cached}/{len(dl_results)} already cached)"
        )

    if download_only:
        print("[modal] --download-only set; skipping counting stage")
        return

    effective_workers = min(n_workers, len(shards))
    chunks: list[list[str]] = [
        shards[i::effective_workers] for i in range(effective_workers)
    ]
    chunks = [c for c in chunks if c]
    max_per_worker = max(1, max_docs // len(chunks)) if max_docs >= 1 else 1

    if aggregate_only:
        print(
            f"[modal] aggregate-only: reading persisted results from {results_subdir}"
        )
        worker_results = read_persisted_results.remote(results_subdir)
        if not worker_results:
            raise SystemExit(
                f"no persisted results found under {results_subdir}; "
                "nothing to aggregate."
            )
        print(f"[modal] recovered {len(worker_results)} worker results from volume")
        elapsed = 0.0
    else:
        print(
            f"[modal] stage 2: counting with {len(chunks)} workers "
            f"(n_workers={n_workers} capped to n_shards={len(shards)}), "
            f"{max_per_worker:,} docs/worker, target total={max_docs:,}"
        )

    setup_tracking(None, experiment_name)

    tags = {
        "stage": "number_frequency",
        "backend": "modal",
        "dataset": dataset_repo,
        "split": "train",
        "counter_mode": "digit_boundary_multi",
        "results_subdir": results_subdir,
    }

    with start_run(run_name=run_name, tags=tags):
        log_params(
            {
                "target_max_docs": max_docs,
                "n_workers_requested": n_workers,
                "n_workers_effective": len(chunks),
                "max_docs_per_worker": max_per_worker,
                "n_shards_total": len(shards),
                "log_every": log_every,
                "dataset_repo": dataset_repo,
                "results_subdir": results_subdir,
                "recovered_from_volume": aggregate_only,
            }
        )

        if not aggregate_only:
            start = time.time()
            worker_results = list(
                count_shard.map(
                    chunks,
                    kwargs={
                        "max_docs": max_per_worker,
                        "log_every": log_every,
                        "results_subdir": results_subdir,
                    },
                )
            )
            elapsed = time.time() - start

        total_docs = sum(r["docs_seen"] for r in worker_results)
        total_matches = sum(r["total_matches"] for r in worker_results)

        combined: Counter[str] = Counter()
        for r in worker_results:
            combined.update(r["counts"])
        filtered = filter_to_range(combined, 0, 1000)

        print(
            f"[modal] done. docs_seen={total_docs:,} "
            f"total_matches={total_matches:,} elapsed={elapsed:.1f}s"
        )
        print(
            f"[modal] observed counts in [0,1000]: "
            f"{len(filtered)} distinct integers"
        )

        log_metric("docs_seen", total_docs)
        log_metric("total_matches", total_matches)
        log_metric("elapsed_seconds", elapsed)
        if total_docs > 0:
            log_metric("avg_matches_per_doc", total_matches / total_docs)
        log_metric("n_integers_observed_in_range", len(filtered))

        batch: dict[str, float] = {}
        for n, c in sorted(filtered.items()):
            batch[f"count_N_{n:04d}"] = float(c)
            if len(batch) >= 500:
                log_metrics(batch)
                batch = {}
        if batch:
            log_metrics(batch)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["number", "count"])
        for n in sorted(filtered):
            writer.writerow([n, filtered[n]])
        mlflow.log_text(buf.getvalue(), "counts_0_to_1000.csv")

        for i, r in enumerate(worker_results):
            mlflow.log_text(
                "\n".join(
                    [
                        f"filenames: {', '.join(r['filenames'])}",
                        f"n_files: {r['n_files']}",
                        f"docs_seen: {r['docs_seen']}",
                        f"total_matches: {r['total_matches']}",
                    ]
                ),
                f"worker_stats/worker_{i:03d}.txt",
            )


@app.local_entrypoint()
def aggregate_cloud(
    results_subdir: str,
    experiment_name: str = "numberline_freq_count_expv1",
    run_name: str = "modal_sweep_cloud_agg",
    dataset_repo: str = DEFAULT_DATASET_REPO,
) -> None:
    # Aggregation path that keeps the 1 GB of per-worker Counter JSONs on
    # the Volume; only the small summary (~30 KB) crosses the network.
    import csv
    import io

    import mlflow

    from utils.mlflow_utils import (
        log_metric,
        log_metrics,
        log_params,
        setup_tracking,
        start_run,
    )

    print(f"[cloud-agg] aggregating {results_subdir} on the Volume ...")
    summary = aggregate_on_cloud.remote(results_subdir)
    print(
        f"[cloud-agg] got summary: {summary['n_workers']} workers, "
        f"{summary['total_docs']:,} docs, "
        f"{summary['total_matches']:,} matches, "
        f"{summary['n_integers_observed_in_range']} integers in [0,1000]"
    )

    counts_in_range: dict[int, int] = {
        int(k): int(v) for k, v in summary["counts_in_range"].items()
    }

    setup_tracking(None, experiment_name)

    tags = {
        "stage": "number_frequency",
        "backend": "modal",
        "dataset": dataset_repo,
        "split": "train",
        "counter_mode": "digit_boundary_multi",
        "results_subdir": results_subdir,
        "aggregation": "cloud_on_volume",
    }

    with start_run(run_name=run_name, tags=tags):
        log_params(
            {
                "n_workers_effective": summary["n_workers"],
                "n_shards_total": summary["n_shards_total_from_workers"],
                "dataset_repo": dataset_repo,
                "results_subdir": results_subdir,
                "recovered_from_volume": True,
            }
        )

        log_metric("docs_seen", summary["total_docs"])
        log_metric("total_matches", summary["total_matches"])
        if summary["total_docs"] > 0:
            log_metric(
                "avg_matches_per_doc",
                summary["total_matches"] / summary["total_docs"],
            )
        log_metric("n_integers_observed_in_range", summary["n_integers_observed_in_range"])
        log_metric("n_distinct_integers_total", summary["n_distinct_integers_total"])

        batch: dict[str, float] = {}
        for n, c in sorted(counts_in_range.items()):
            batch[f"count_N_{n:04d}"] = float(c)
            if len(batch) >= 500:
                log_metrics(batch)
                batch = {}
        if batch:
            log_metrics(batch)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["number", "count"])
        for n in sorted(counts_in_range):
            writer.writerow([n, counts_in_range[n]])
        mlflow.log_text(buf.getvalue(), "counts_0_to_1000.csv")

        for i, w in enumerate(summary["worker_summaries"]):
            mlflow.log_text(
                "\n".join(
                    [
                        f"filenames: {', '.join(w['filenames'])}",
                        f"n_files: {w['n_files']}",
                        f"docs_seen: {w['docs_seen']}",
                        f"total_matches: {w['total_matches']}",
                    ]
                ),
                f"worker_stats/worker_{i:03d}.txt",
            )

        active = mlflow.active_run()
        print(f"[cloud-agg] done. MLflow run_id = {active.info.run_id}")


@app.local_entrypoint()
def aggregate_cloud_range(
    results_subdir: str,
    n_min: int = 0,
    n_max: int = 5000,
    experiment_name: str = "numberline_freq_count_expv1",
    run_name: str = "",
    dataset_repo: str = DEFAULT_DATASET_REPO,
    results_base_dir: str = "results/pile_uncopyrighted",
) -> None:
    """Re-aggregate workers' unfiltered counters with a *configurable* range
    and log a new MLflow run. No re-counting; just a new filter window over
    the same per-worker JSONs already on the Volume."""
    import csv
    import io
    import json

    import mlflow

    from utils.mlflow_utils import (
        log_metric,
        log_metrics,
        log_params,
        setup_tracking,
        start_run,
    )

    if n_max <= n_min:
        raise SystemExit(f"need n_max ({n_max}) > n_min ({n_min})")
    if not run_name:
        run_name = f"modal_sweep_cloud_agg_{n_min}_to_{n_max}"

    print(
        f"[cloud-agg-range] window=[{n_min}, {n_max}]  "
        f"results_subdir={results_subdir}"
    )
    summary = aggregate_on_cloud_range.remote(results_subdir, n_min=n_min, n_max=n_max)
    print(
        f"[cloud-agg-range] got summary: {summary['n_workers']} workers, "
        f"{summary['total_docs']:,} docs, {summary['total_matches']:,} matches, "
        f"{summary['n_integers_observed_in_range']} integers in [{n_min},{n_max}], "
        f"{summary['n_distinct_integers_total']:,} total distinct strings."
    )

    counts_in_range: dict[int, int] = {
        int(k): int(v) for k, v in summary["counts_in_range"].items()
    }

    out_dir = Path(results_base_dir) / f"{n_min}_to_{n_max}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"counts_{n_min}_to_{n_max}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["number", "count"])
        for n in range(n_min, n_max + 1):
            writer.writerow([n, counts_in_range.get(n, 0)])
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[cloud-agg-range] wrote {csv_path} and {out_dir / 'summary.json'}")

    setup_tracking(None, experiment_name)

    tags = {
        "stage": "number_frequency",
        "backend": "modal",
        "dataset": dataset_repo,
        "split": "train",
        "counter_mode": "digit_boundary_multi",
        "results_subdir": results_subdir,
        "aggregation": "cloud_on_volume_range",
        "n_min": str(n_min),
        "n_max": str(n_max),
    }

    pad = max(4, len(str(n_max)))
    artifact_name = f"counts_{n_min}_to_{n_max}.csv"

    with start_run(run_name=run_name, tags=tags):
        log_params(
            {
                "n_workers_effective": summary["n_workers"],
                "n_shards_total": summary["n_shards_total_from_workers"],
                "dataset_repo": dataset_repo,
                "results_subdir": results_subdir,
                "recovered_from_volume": True,
                "n_min": n_min,
                "n_max": n_max,
            }
        )

        log_metric("docs_seen", summary["total_docs"])
        log_metric("total_matches", summary["total_matches"])
        if summary["total_docs"] > 0:
            log_metric(
                "avg_matches_per_doc",
                summary["total_matches"] / summary["total_docs"],
            )
        log_metric("n_integers_observed_in_range", summary["n_integers_observed_in_range"])
        log_metric("n_distinct_integers_total", summary["n_distinct_integers_total"])
        log_metric("n_min", float(n_min))
        log_metric("n_max", float(n_max))

        batch: dict[str, float] = {}
        for n, c in sorted(counts_in_range.items()):
            batch[f"count_N_{n:0{pad}d}"] = float(c)
            if len(batch) >= 500:
                log_metrics(batch)
                batch = {}
        if batch:
            log_metrics(batch)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["number", "count"])
        for n in range(n_min, n_max + 1):
            writer.writerow([n, counts_in_range.get(n, 0)])
        mlflow.log_text(buf.getvalue(), artifact_name)

        for i, w in enumerate(summary["worker_summaries"]):
            mlflow.log_text(
                "\n".join(
                    [
                        f"filenames: {', '.join(w['filenames'])}",
                        f"n_files: {w['n_files']}",
                        f"docs_seen: {w['docs_seen']}",
                        f"total_matches: {w['total_matches']}",
                    ]
                ),
                f"worker_stats/worker_{i:03d}.txt",
            )

        active = mlflow.active_run()
        print(f"[cloud-agg-range] done. MLflow run_id = {active.info.run_id}")
