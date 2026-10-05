from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow

from src.multi_counter import MultiProgressUpdate, filter_to_range, run_multi_count
from utils.dataset import load_pile_stream
from utils.mlflow_utils import (
    log_metric,
    log_metrics,
    log_params,
    setup_tracking,
    start_run,
)


NUMBER_MIN = 0
NUMBER_MAX = 1000


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a text corpus, count every integer in [0, 1000] in a "
            "single regex pass, and log the full frequency table to MLflow."
        )
    )
    parser.add_argument("--max-docs", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--experiment-name", type=str, default="numberline_freq_count_expv1")
    parser.add_argument("--run-name", type=str, default="local_multi_count")
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--dataset-revision", type=str, default=None)
    parser.add_argument("--dataset-data-dir", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--text-key", type=str, default="text",
                        help="field containing document text in each dataset row")

    args = parser.parse_args(argv)
    if args.max_docs <= 0:
        parser.error(f"--max-docs must be positive, got {args.max_docs}")
    if args.log_every <= 0:
        parser.error(f"--log-every must be positive, got {args.log_every}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    setup_tracking(args.tracking_uri, args.experiment_name)

    tags = {
        "stage": "number_frequency",
        "backend": "local",
        "dataset": args.dataset_name,
        "split": args.dataset_split,
        "counter_mode": "digit_boundary_multi",
    }

    print(
        f"[cli] streaming {args.dataset_name} [{args.dataset_split}], "
        f"cap = {args.max_docs:,} docs"
    )

    with start_run(run_name=args.run_name, tags=tags):
        log_params(
            {
                "max_docs": args.max_docs,
                "log_every": args.log_every,
                "dataset_name": args.dataset_name,
                "dataset_revision": args.dataset_revision,
                "dataset_data_dir": args.dataset_data_dir,
                "dataset_split": args.dataset_split,
                "text_key": args.text_key,
            }
        )

        def _on_progress(update: MultiProgressUpdate) -> None:
            log_metrics(
                {
                    "docs_seen": update.docs_seen,
                    "total_matches": update.total_matches,
                },
                step=update.docs_seen,
            )
            print(
                f"[progress] docs_seen={update.docs_seen:,} "
                f"total_matches={update.total_matches:,}"
            )

        examples = load_pile_stream(
            dataset_name=args.dataset_name,
            revision=args.dataset_revision,
            data_dir=args.dataset_data_dir,
            split=args.dataset_split,
        )

        start = time.time()
        result = run_multi_count(
            examples=examples,
            max_docs=args.max_docs,
            text_key=args.text_key,
            log_every=args.log_every,
            progress_callback=_on_progress,
        )
        elapsed = time.time() - start

        filtered = filter_to_range(result.counts, NUMBER_MIN, NUMBER_MAX)

        log_metric("elapsed_seconds", elapsed)
        if result.docs_seen > 0:
            log_metric("avg_matches_per_doc", result.total_matches / result.docs_seen)
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

        print("\nDONE")
        print(f"  docs_seen:          {result.docs_seen:,}")
        print(f"  total_matches:      {result.total_matches:,}")
        print(f"  integers in range:  {len(filtered)}")
        print(f"  elapsed_seconds:    {elapsed:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
