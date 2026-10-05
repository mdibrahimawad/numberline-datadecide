from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import mlflow

from src.multi_counter import filter_to_range
from utils.mlflow_utils import (
    log_metric,
    log_metrics,
    log_params,
    setup_tracking,
    start_run,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--experiment-name", default="numberline_freq_count_expv1")
    p.add_argument("--run-name", default="modal_full_sweep_200M_recovered")
    p.add_argument("--dataset-repo", default="monology/pile-uncopyrighted")
    p.add_argument("--results-subdir", default="_results/full_sweep_v1")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no *.json files under {results_dir}")

    print(f"[agg] found {len(files)} worker results under {results_dir}")
    t0 = time.time()
    worker_results: list[dict] = []
    for i, f in enumerate(files, 1):
        with open(f) as fh:
            worker_results.append(json.load(fh))
        if i % 5 == 0 or i == len(files):
            print(f"[agg]  loaded {i}/{len(files)} ({f.name})")
    print(f"[agg] load took {time.time() - t0:.1f}s")

    total_docs = sum(r["docs_seen"] for r in worker_results)
    total_matches = sum(r["total_matches"] for r in worker_results)
    print(f"[agg] docs_seen={total_docs:,} total_matches={total_matches:,}")

    t0 = time.time()
    combined: Counter[str] = Counter()
    for r in worker_results:
        combined.update(r["counts"])
    print(
        f"[agg] combined Counter: {len(combined):,} distinct integer strings "
        f"(took {time.time() - t0:.1f}s)"
    )

    filtered = filter_to_range(combined, 0, 1000)
    print(f"[agg] in [0, 1000]: {len(filtered)} distinct integers")

    setup_tracking(None, args.experiment_name)

    tags = {
        "stage": "number_frequency",
        "backend": "modal",
        "dataset": args.dataset_repo,
        "split": "train",
        "counter_mode": "digit_boundary_multi",
        "results_subdir": args.results_subdir,
        "aggregation": "local_from_volume_dump",
    }

    with start_run(run_name=args.run_name, tags=tags):
        log_params(
            {
                "n_workers_effective": len(worker_results),
                "n_shards_total": sum(r["n_files"] for r in worker_results),
                "dataset_repo": args.dataset_repo,
                "results_subdir": args.results_subdir,
                "recovered_from_volume": True,
            }
        )

        log_metric("docs_seen", total_docs)
        log_metric("total_matches", total_matches)
        if total_docs > 0:
            log_metric("avg_matches_per_doc", total_matches / total_docs)
        log_metric("n_integers_observed_in_range", len(filtered))
        log_metric("n_distinct_integers_total", len(combined))

        t0 = time.time()
        batch: dict[str, float] = {}
        for n, c in sorted(filtered.items()):
            batch[f"count_N_{n:04d}"] = float(c)
            if len(batch) >= 500:
                log_metrics(batch)
                batch = {}
        if batch:
            log_metrics(batch)
        print(f"[agg] logged {len(filtered)} per-integer metrics in {time.time() - t0:.1f}s")

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

        print(f"[agg] done. MLflow run logged under experiment '{args.experiment_name}'.")


if __name__ == "__main__":
    sys.exit(main())