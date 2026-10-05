from __future__ import annotations

from collections import Counter
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

APP_NAME = "stack-number-frequency"
DATASET_REPO = "bigcode/the-stack-dedup"
DATASET_VERSION = "v1.2"
RESULTS_VOLUME_NAME = "stack-number-frequency-results"
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

_HF_SECRET_NAME = "numberline-hf-token"


def _file_key(path: str) -> str:
    import hashlib

    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:20]


def _language_from_path(path: str) -> str:
    parts = path.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "data" else "unknown"


def _stack_files(max_files: int = 0) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = [
        f
        for f in api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet")
    ]
    files = sorted(files)
    return files[:max_files] if max_files > 0 else files


@app.function(
    timeout=6 * 60 * 60,
    cpu=2.0,
    memory=8192,
    volumes={RESULTS_PATH: results_volume},
    max_containers=512,
    retries=1,
    secrets=[modal.Secret.from_name(_HF_SECRET_NAME)],
)
def count_parquet_file(
    filename: str,
    *,
    max_n: int = 10_000,
    batch_size: int = 1024,
    results_subdir: str = "_results/stack_v1_2",
) -> dict:
    import json
    import os
    import time

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from utils.text import DIGIT_TOKEN_PATTERN

    results_volume.reload()
    out_dir = Path(RESULTS_PATH) / results_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_file_key(filename)}.json"

    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path) as fh:
            cached = json.load(fh)
        print(f"[worker] cached {filename} -> {out_path}")
        return {
            "filename": filename,
            "language": cached.get("language", _language_from_path(filename)),
            "docs_seen": int(cached.get("docs_seen", 0)),
            "total_matches": int(cached.get("total_matches", 0)),
            "bad_rows": int(cached.get("bad_rows", 0)),
            "cached": True,
        }

    started = time.time()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        token=token,
    )

    pf = pq.ParquetFile(local_path)
    schema_names = set(pf.schema_arrow.names)
    if "content" in schema_names:
        text_col = "content"
    elif "text" in schema_names:
        text_col = "text"
    else:
        raise ValueError(f"{filename} has neither 'content' nor 'text' column: {sorted(schema_names)}")

    max_digits = len(str(max_n))
    counts: Counter[int] = Counter()
    docs_seen = 0
    total_matches = 0
    bad_rows = 0
    language = _language_from_path(filename)

    for batch in pf.iter_batches(batch_size=batch_size, columns=[text_col]):
        values = batch.column(0).to_pylist()
        for text in values:
            docs_seen += 1
            if not isinstance(text, str):
                bad_rows += 1
                continue
            toks = DIGIT_TOKEN_PATTERN.findall(text)
            total_matches += len(toks)
            for tok in toks:
                normalized = tok.lstrip("0") or "0"
                if len(normalized) > max_digits:
                    continue
                n = int(normalized)
                if n <= max_n:
                    counts[n] += 1

    payload = {
        "filename": filename,
        "language": language,
        "text_column": text_col,
        "docs_seen": int(docs_seen),
        "total_matches": int(total_matches),
        "bad_rows": int(bad_rows),
        "max_n": int(max_n),
        "elapsed_seconds": time.time() - started,
        "counts": {str(k): int(v) for k, v in sorted(counts.items()) if v},
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
    results_volume.commit()
    print(
        f"[worker] persisted {filename} -> {out_path} "
        f"docs={docs_seen:,} matches={total_matches:,}"
    )
    return {
        "filename": filename,
        "language": language,
        "docs_seen": docs_seen,
        "total_matches": total_matches,
        "bad_rows": bad_rows,
        "cached": False,
    }


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
    n_max: int = 10_000,
) -> dict:
    import json
    from collections import Counter, defaultdict

    results_volume.reload()
    results_dir = Path(RESULTS_PATH) / results_subdir
    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no worker JSONs under {results_dir}")

    counts: Counter[int] = Counter()
    docs_seen = 0
    total_matches = 0
    bad_rows = 0
    per_language: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "docs_seen": 0, "total_matches": 0, "bad_rows": 0}
    )

    for i, p in enumerate(paths, 1):
        with open(p) as fh:
            payload = json.load(fh)
        language = payload.get("language", "unknown")
        docs = int(payload.get("docs_seen", 0))
        matches = int(payload.get("total_matches", 0))
        bad = int(payload.get("bad_rows", 0))
        docs_seen += docs
        total_matches += matches
        bad_rows += bad
        per_language[language]["files"] += 1
        per_language[language]["docs_seen"] += docs
        per_language[language]["total_matches"] += matches
        per_language[language]["bad_rows"] += bad
        for k_s, v in payload.get("counts", {}).items():
            k = int(k_s)
            if n_min <= k <= n_max:
                counts[k] += int(v)
        if i % 100 == 0 or i == len(paths):
            print(f"[aggregate] merged {i}/{len(paths)} worker files")

    return {
        "n_min": n_min,
        "n_max": n_max,
        "n_workers": len(paths),
        "docs_seen": docs_seen,
        "total_matches": total_matches,
        "bad_rows": bad_rows,
        "n_integers_observed_in_range": len(counts),
        "counts_in_range": {str(k): int(v) for k, v in sorted(counts.items())},
        "per_language": dict(sorted(per_language.items())),
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


def _log_mlflow(
    summary: dict,
    *,
    experiment_name: str,
    run_name: str,
    results_subdir: str,
) -> str:
    import csv
    import io
    import json

    import mlflow

    from utils.mlflow_utils import log_metric, log_metrics, log_params, setup_tracking, start_run

    n_min = int(summary["n_min"])
    n_max = int(summary["n_max"])
    pad = max(4, len(str(n_max)))
    counts = {int(k): int(v) for k, v in summary["counts_in_range"].items()}

    setup_tracking(None, experiment_name)
    tags = {
        "stage": "number_frequency",
        "backend": "modal",
        "dataset": DATASET_REPO,
        "dataset_version": DATASET_VERSION,
        "split": "train",
        "counter_mode": "digit_boundary_multi_parquet",
        "results_subdir": results_subdir,
        "n_min": str(n_min),
        "n_max": str(n_max),
    }

    with start_run(run_name=run_name, tags=tags) as active:
        log_params(
            {
                "dataset_repo": DATASET_REPO,
                "dataset_version": DATASET_VERSION,
                "results_subdir": results_subdir,
                "n_workers": summary["n_workers"],
                "n_min": n_min,
                "n_max": n_max,
            }
        )
        log_metric("docs_seen", summary["docs_seen"])
        log_metric("total_matches", summary["total_matches"])
        log_metric("bad_rows", summary["bad_rows"])
        log_metric("n_integers_observed_in_range", summary["n_integers_observed_in_range"])
        if summary["docs_seen"] > 0:
            log_metric("avg_matches_per_doc", summary["total_matches"] / summary["docs_seen"])

        batch: dict[str, float] = {}
        for n in range(n_min, n_max + 1):
            batch[f"count_N_{n:0{pad}d}"] = float(counts.get(n, 0))
            if len(batch) >= 500:
                log_metrics(batch)
                batch = {}
        if batch:
            log_metrics(batch)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["number", "count"])
        for n in range(n_min, n_max + 1):
            writer.writerow([n, counts.get(n, 0)])
        mlflow.log_text(buf.getvalue(), f"counts_{n_min}_to_{n_max}.csv")
        mlflow.log_text(json.dumps(summary["per_language"], indent=2), "per_language.json")
        run_id = active.info.run_id

    print(f"[mlflow] logged {run_name}: {run_id}")
    return run_id


@app.local_entrypoint()
def main(
    max_n: int = 10_000,
    max_files: int = 0,
    batch_size: int = 1024,
    results_subdir: str = "",
    results_base_dir: str = "results/stack_v1_2",
    experiment_name: str = "numberline_freq_count_expv1",
    run_name_prefix: str = "stack_v1_2",
    aggregate_only: bool = False,
    dry_run: bool = False,
) -> None:
    from datetime import datetime

    if max_n < 1:
        raise SystemExit("--max-n must be >= 1")
    if batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not results_subdir:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results_subdir = f"_results/stack_v1_2_{stamp}"

    if aggregate_only:
        files = []
        print(f"[stack] repo={DATASET_REPO} aggregate_only=True max_n={max_n}")
        print(f"[stack] results_subdir={results_subdir} volume={RESULTS_VOLUME_NAME}")
    else:
        files = _stack_files(max_files=max_files)
        by_language = Counter(_language_from_path(f) for f in files)
        print(f"[stack] repo={DATASET_REPO} files={len(files)} languages={len(by_language)} max_n={max_n}")
        print(f"[stack] results_subdir={results_subdir} volume={RESULTS_VOLUME_NAME}")
        for lang, n in by_language.most_common(12):
            print(f"  {lang}: {n} files")
        if len(by_language) > 12:
            print(f"  ... ({len(by_language) - 12} more languages)")

    if dry_run:
        print("[stack] --dry-run set; exiting before Modal work")
        return

    if not aggregate_only:
        results = list(
            count_parquet_file.map(
                files,
                kwargs={
                    "max_n": max_n,
                    "batch_size": batch_size,
                    "results_subdir": results_subdir,
                },
                order_outputs=False,
                return_exceptions=True,
            )
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        print(f"[stack] count map returned {len(results)} results, errors={len(errors)}")
        if errors:
            for e in errors[:10]:
                print(f"[stack] ERROR: {type(e).__name__}: {e}")
            raise RuntimeError(
                f"{len(errors)} Stack parquet tasks failed; rerun with the same "
                f"--results-subdir {results_subdir!r} to reuse completed files."
            )

    for n_min, n_max in [(0, 1000), (0, max_n)]:
        summary = aggregate_persisted.remote(results_subdir, n_min=n_min, n_max=n_max)
        out_dir = f"{results_base_dir}/{n_min}_to_{n_max}"
        _write_local_summary(summary, out_dir)
        _log_mlflow(
            summary,
            experiment_name=experiment_name,
            run_name=f"{run_name_prefix}_{n_min}_to_{n_max}",
            results_subdir=results_subdir,
        )
        in_range = sum(int(v) for v in summary["counts_in_range"].values())
        print(
            f"[stack] [{n_min},{n_max}] docs={summary['docs_seen']:,} "
            f"matches={summary['total_matches']:,} in_range={in_range:,} "
            f"bad_rows={summary['bad_rows']:,}"
        )
