from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "refinedweb-number-frequency"

DEFAULT_DATASET_REPO = "tiiuae/falcon-refinedweb"
RESULTS_VOLUME_NAME = "refinedweb-number-frequency-results"
RESULTS_PATH = "/results"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub>=0.24.0",
        "pyarrow>=16.0.0",
    )
    .add_local_python_source("utils")
)

app = modal.App(APP_NAME, image=image)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def _discover_parquet_shards(repo: str = DEFAULT_DATASET_REPO) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    return sorted(
        f for f in files
        if f.startswith("data/train-") and f.endswith(".parquet")
    )


@app.function(
    timeout=12 * 60 * 60,
    cpu=4.0,
    memory=8192,
    volumes={RESULTS_PATH: results_volume},
)
def count_parquet_shards(
    filenames: list[str],
    *,
    repo: str = DEFAULT_DATASET_REPO,
    text_key: str = "content",
    n_min: int = 0,
    n_max: int = 1000,
    max_docs: int = -1,
    batch_size: int = 1024,
    results_subdir: str = "_results/refinedweb_default",
) -> dict:
    import hashlib
    import json
    import os
    import shutil
    import tempfile
    from collections import Counter
    from pathlib import Path

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from utils.text import DIGIT_TOKEN_PATTERN

    counts: Counter[int] = Counter()
    docs_seen = 0
    total_matches = 0
    cache_dir = Path(tempfile.mkdtemp(prefix="refinedweb_"))

    def _count_text(text: str) -> None:
        nonlocal total_matches
        toks = DIGIT_TOKEN_PATTERN.findall(text.lower())
        total_matches += len(toks)
        for tok in toks:
            try:
                n = int(tok)
            except ValueError:
                continue
            if n_min <= n <= n_max:
                counts[n] += 1

    try:
        for file_idx, filename in enumerate(filenames, 1):
            if 0 <= max_docs <= docs_seen:
                break

            local_path = hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=filename,
                local_dir=str(cache_dir),
            )
            print(f"[worker] {file_idx}/{len(filenames)} opened {filename}")

            parquet_file = pq.ParquetFile(local_path)
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=[text_key],
            ):
                column = batch.column(0).to_pylist()
                for text in column:
                    if 0 <= max_docs <= docs_seen:
                        break
                    docs_seen += 1
                    if isinstance(text, str):
                        _count_text(text)
                if 0 <= max_docs <= docs_seen:
                    break

            try:
                os.remove(local_path)
            except OSError:
                pass

            print(
                f"[worker] progress files={file_idx}/{len(filenames)} "
                f"docs={docs_seen:,} matches={total_matches:,}"
            )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    payload = {
        "docs_seen": docs_seen,
        "total_matches": total_matches,
        "counts_in_range": {str(k): int(v) for k, v in sorted(counts.items())},
        "n_min": n_min,
        "n_max": n_max,
        "n_files": len(filenames),
        "filenames": filenames,
    }

    results_dir = Path(RESULTS_PATH) / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1("\n".join(filenames).encode()).hexdigest()[:16]
    out_path = results_dir / f"{key}.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
    results_volume.commit()
    print(f"[worker] persisted result to {out_path}")

    return payload


@app.function(
    timeout=60 * 60,
    cpu=2.0,
    memory=8192,
    volumes={RESULTS_PATH: results_volume},
)
def aggregate_persisted(
    results_subdir: str,
    *,
    n_min: int = 0,
    n_max: int = 1000,
) -> dict:
    import json
    from collections import Counter
    from pathlib import Path

    results_volume.reload()
    results_dir = Path(RESULTS_PATH) / results_subdir
    if not results_dir.exists():
        raise FileNotFoundError(f"no results dir at {results_dir}")

    counts: Counter[int] = Counter()
    docs_seen = 0
    total_matches = 0
    n_files = 0
    workers = []

    paths = sorted(results_dir.glob("*.json"))
    insufficient = 0
    for i, path in enumerate(paths, 1):
        with open(path) as fh:
            payload = json.load(fh)
        payload_n_max = payload.get("n_max")
        if payload_n_max is not None and int(payload_n_max) < n_max:
            insufficient += 1
        elif payload_n_max is None and n_max > 1000:
            max_key = max((int(k) for k in payload.get("counts_in_range", {})), default=-1)
            if max_key <= 1000:
                insufficient += 1
        docs_seen += int(payload["docs_seen"])
        total_matches += int(payload["total_matches"])
        n_files += int(payload["n_files"])
        counts.update(
            {
                int(k): int(v)
                for k, v in payload["counts_in_range"].items()
                if n_min <= int(k) <= n_max
            }
        )
        workers.append(
            {
                "filenames": payload["filenames"],
                "n_files": int(payload["n_files"]),
                "docs_seen": int(payload["docs_seen"]),
                "total_matches": int(payload["total_matches"]),
            }
        )
        if i % 20 == 0 or i == len(paths):
            print(f"[aggregate] merged {i}/{len(paths)} worker files")

    if insufficient:
        raise RuntimeError(
            f"{insufficient}/{len(paths)} worker JSONs do not contain counts up to "
            f"n_max={n_max}. Re-run counting with --n-max {n_max}; aggregate-only "
            "cannot recover numbers that were filtered out by the worker."
        )

    return {
        "n_min": n_min,
        "n_max": n_max,
        "docs_seen": docs_seen,
        "total_matches": total_matches,
        "n_files": n_files,
        "n_workers": len(paths),
        "counts_in_range": {str(k): int(v) for k, v in sorted(counts.items())},
        "worker_summaries": workers,
    }


def _write_local_summary(summary: dict, results_dir: str) -> None:
    import csv
    import json

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_min = int(summary["n_min"])
    n_max = int(summary["n_max"])
    counts = {int(k): int(v) for k, v in summary["counts_in_range"].items()}

    csv_path = out / f"counts_{n_min}_to_{n_max}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["number", "count"])
        for n in range(n_min, n_max + 1):
            writer.writerow([n, counts.get(n, 0)])
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[local] wrote {csv_path} and {out / 'summary.json'}")


def _log_summary(
    *,
    summary: dict,
    experiment_name: str,
    run_name: str,
    dataset_repo: str,
    text_key: str,
    results_subdir: str,
    n_workers_requested: int,
    recovered: bool,
) -> None:
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

    counts_in_range = {
        int(k): int(v) for k, v in summary["counts_in_range"].items()
    }
    n_min = int(summary["n_min"])
    n_max = int(summary["n_max"])
    pad = max(4, len(str(n_max)))
    artifact_name = f"counts_{n_min}_to_{n_max}.csv"

    setup_tracking(None, experiment_name)
    tags = {
        "stage": "number_frequency",
        "backend": "modal",
        "dataset": dataset_repo,
        "split": "train",
        "counter_mode": "digit_boundary_multi_range_worker",
        "results_subdir": results_subdir,
        "n_min": str(n_min),
        "n_max": str(n_max),
    }

    with start_run(run_name=run_name, tags=tags):
        log_params(
            {
                "dataset_repo": dataset_repo,
                "text_key": text_key,
                "n_workers_requested": n_workers_requested,
                "n_workers_effective": summary["n_workers"],
                "n_shards_total": summary["n_files"],
                "results_subdir": results_subdir,
                "recovered_from_volume": recovered,
                "count_range": f"{n_min},{n_max}",
                "n_min": n_min,
                "n_max": n_max,
            }
        )
        log_metric("docs_seen", summary["docs_seen"])
        log_metric("total_matches", summary["total_matches"])
        if summary["docs_seen"] > 0:
            log_metric(
                "avg_matches_per_doc",
                summary["total_matches"] / summary["docs_seen"],
            )
        log_metric("n_integers_observed_in_range", len(counts_in_range))

        batch: dict[str, float] = {}
        for n in range(n_min, n_max + 1):
            batch[f"count_N_{n:0{pad}d}"] = float(counts_in_range.get(n, 0))
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

        for i, worker in enumerate(summary["worker_summaries"]):
            mlflow.log_text(
                "\n".join(
                    [
                        f"filenames: {', '.join(worker['filenames'])}",
                        f"n_files: {worker['n_files']}",
                        f"docs_seen: {worker['docs_seen']}",
                        f"total_matches: {worker['total_matches']}",
                    ]
                ),
                f"worker_stats/worker_{i:03d}.txt",
            )

        active = mlflow.active_run()
        print(f"[log] MLflow run_id = {active.info.run_id}")


@app.local_entrypoint()
def main(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    text_key: str = "content",
    n_workers: int = 128,
    max_docs: int = -1,
    n_min: int = 0,
    n_max: int = 1000,
    batch_size: int = 1024,
    experiment_name: str = "numberline_freq_count_expv1",
    run_name: str = "refinedweb_full_number_count",
    results_subdir: str = "",
    results_base_dir: str = "results/refinedweb_full_modal",
    aggregate_only: bool = False,
    dry_run: bool = False,
) -> None:
    import time
    from datetime import datetime

    if n_workers < 1:
        raise SystemExit("--n-workers must be >= 1")
    if not results_subdir:
        results_subdir = (
            f"_results/refinedweb_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

    shards = _discover_parquet_shards(dataset_repo)
    if not shards:
        raise SystemExit(f"no train parquet shards discovered in {dataset_repo}")

    effective_workers = min(n_workers, len(shards))
    chunks = [shards[i::effective_workers] for i in range(effective_workers)]
    chunks = [chunk for chunk in chunks if chunk]
    max_per_worker = -1 if max_docs < 0 else max(1, max_docs // len(chunks))

    print(f"[refinedweb] dataset={dataset_repo}")
    print(f"[refinedweb] text_key={text_key}")
    print(f"[refinedweb] count_range=[{n_min},{n_max}]")
    print(f"[refinedweb] shards={len(shards)} workers={len(chunks)}")
    print(f"[refinedweb] results_subdir={results_subdir}")
    print(
        "[refinedweb] max_docs="
        + ("FULL" if max_docs < 0 else f"{max_docs:,}")
        + f" max_per_worker={max_per_worker}"
    )
    for shard in shards[:5]:
        print(f"  {shard}")
    if len(shards) > 5:
        print(f"  ... ({len(shards) - 5} more)")

    if dry_run:
        print("[refinedweb] --dry-run set; exiting before cloud work")
        return

    if aggregate_only:
        print(f"[refinedweb] aggregate-only from {results_subdir}")
        summary = aggregate_persisted.remote(results_subdir, n_min=n_min, n_max=n_max)
        _write_local_summary(summary, f"{results_base_dir}/{n_min}_to_{n_max}")
        _log_summary(
            summary=summary,
            experiment_name=experiment_name,
            run_name=run_name,
            dataset_repo=dataset_repo,
            text_key=text_key,
            results_subdir=results_subdir,
            n_workers_requested=n_workers,
            recovered=True,
        )
        return

    start = time.time()
    worker_results = list(
        count_parquet_shards.map(
            chunks,
            kwargs={
                "repo": dataset_repo,
                "text_key": text_key,
                "n_min": n_min,
                "n_max": n_max,
                "max_docs": max_per_worker,
                "batch_size": batch_size,
                "results_subdir": results_subdir,
            },
        )
    )
    elapsed = time.time() - start
    print(f"[refinedweb] workers complete in {elapsed:.1f}s; aggregating")

    summary = aggregate_persisted.remote(results_subdir, n_min=n_min, n_max=n_max)
    _write_local_summary(summary, f"{results_base_dir}/{n_min}_to_{n_max}")
    _log_summary(
        summary=summary,
        experiment_name=experiment_name,
        run_name=run_name,
        dataset_repo=dataset_repo,
        text_key=text_key,
        results_subdir=results_subdir,
        n_workers_requested=n_workers,
        recovered=False,
    )
    print(
        f"[refinedweb] done docs={summary['docs_seen']:,} "
        f"matches={summary['total_matches']:,}"
    )
