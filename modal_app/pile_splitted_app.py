from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "pile-original-splitted-number-frequency"
DEFAULT_DATASET_REPO = "ArmelR/the-pile-splitted"
RESULTS_VOLUME_NAME = "pile-original-splitted-number-frequency-results"
RESULTS_PATH = "/results"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub>=0.24.0",
        "pyarrow>=15.0.0",
        "requests>=2.31.0",
        "mlflow>=2.16.0",
    )
    .add_local_python_source("utils")
)

app = modal.App(APP_NAME, image=image)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def _parse_csv(value: str) -> list[str]:
    if value.strip().lower() in {"", "all"}:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _domain_from_path(path: str) -> str:
    parts = path.split("/")
    if len(parts) >= 4 and parts[0] == "data":
        return "/".join(parts[1:-2])
    return "unknown"


def _split_from_path(path: str) -> str:
    parts = path.split("/")
    if len(parts) >= 3:
        return parts[-2]
    return "unknown"


def _result_key(filename: str) -> str:
    import hashlib

    return hashlib.sha1(filename.encode()).hexdigest()[:16]


def _discover_arrow_files(
    repo_id: str,
    splits: list[str],
    domains: list[str],
    max_files: int,
) -> list[str]:
    from huggingface_hub import HfApi

    split_set = set(splits or ["train", "test"])
    domain_set = set(domains)
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    out: list[str] = []
    for filename in files:
        if not filename.startswith("data/") or not filename.endswith(".arrow"):
            continue
        if _split_from_path(filename) not in split_set:
            continue
        if domain_set and _domain_from_path(filename) not in domain_set:
            continue
        out.append(filename)
    out = sorted(out)
    if max_files > 0:
        out = out[:max_files]
    return out


@app.function(
    timeout=20 * 60,
    cpu=1.0,
    memory=1024,
    volumes={RESULTS_PATH: results_volume},
)
def list_persisted_keys(results_subdir: str) -> list[str]:
    results_volume.reload()
    results_dir = Path(RESULTS_PATH) / results_subdir
    if not results_dir.exists():
        return []
    return sorted(path.stem for path in results_dir.glob("*.json"))


@app.function(
    timeout=6 * 60 * 60,
    cpu=2.0,
    memory=4096,
    max_containers=100,
    retries=2,
    volumes={RESULTS_PATH: results_volume},
)
def count_arrow_file(
    filename: str,
    *,
    repo_id: str = DEFAULT_DATASET_REPO,
    text_key: str = "text",
    n_min: int = 0,
    n_max: int = 10000,
    results_subdir: str = "_results/pile_original_splitted",
    force: bool = False,
) -> dict:
    import json
    import time
    from collections import Counter
    from pathlib import Path

    import pyarrow.ipc as arrow_ipc
    from huggingface_hub import hf_hub_download

    from utils.text import DIGIT_TOKEN_PATTERN

    key = _result_key(filename)
    results_volume.reload()
    results_dir = Path(RESULTS_PATH) / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{key}.json"
    if out_path.exists() and not force:
        with open(out_path) as fh:
            payload = json.load(fh)
        payload["cached"] = True
        print(f"[skip] cached {filename} -> {out_path}")
        return payload

    start = time.time()
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir="/tmp/pile_splitted",
    )

    counts: Counter[int] = Counter()
    docs_seen = 0
    total_matches = 0
    domain = _domain_from_path(filename)
    split = _split_from_path(filename)

    with arrow_ipc.open_stream(local_path) as reader:
        for batch in reader:
            if text_key not in batch.column_names:
                raise KeyError(f"{filename} has no {text_key!r}; columns={batch.column_names}")
            texts = batch.column(text_key)
            for scalar in texts:
                docs_seen += 1
                if not scalar.is_valid:
                    continue
                text = scalar.as_py()
                if not isinstance(text, str):
                    continue
                toks = DIGIT_TOKEN_PATTERN.findall(text)
                total_matches += len(toks)
                for tok in toks:
                    try:
                        number = int(tok)
                    except ValueError:
                        continue
                    if n_min <= number <= n_max:
                        counts[number] += 1

    payload = {
        "filename": filename,
        "domain": domain,
        "split": split,
        "docs_seen": docs_seen,
        "total_matches": total_matches,
        "counts_in_range": {str(k): int(v) for k, v in sorted(counts.items())},
        "n_min": n_min,
        "n_max": n_max,
        "elapsed_seconds": time.time() - start,
        "cached": False,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
    results_volume.commit()
    print(
        f"[done] {filename} docs={docs_seen:,} matches={total_matches:,} "
        f"range_hits={sum(counts.values()):,} elapsed={payload['elapsed_seconds']:.1f}s"
    )
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
    n_max: int = 10000,
) -> dict:
    import json
    from collections import Counter, defaultdict

    results_volume.reload()
    results_dir = Path(RESULTS_PATH) / results_subdir
    if not results_dir.exists():
        raise FileNotFoundError(f"no results dir at {results_dir}")

    counts: Counter[int] = Counter()
    domain_docs: Counter[str] = Counter()
    domain_matches: Counter[str] = Counter()
    split_docs: Counter[str] = Counter()
    split_matches: Counter[str] = Counter()
    worker_summaries: list[dict] = []
    docs_seen = 0
    total_matches = 0
    n_files = 0
    path_by_domain: defaultdict[str, int] = defaultdict(int)

    paths = sorted(results_dir.glob("*.json"))
    for i, path in enumerate(paths, 1):
        with open(path) as fh:
            payload = json.load(fh)
        n_files += 1
        docs = int(payload["docs_seen"])
        matches = int(payload["total_matches"])
        domain = payload.get("domain", "unknown")
        split = payload.get("split", "unknown")
        docs_seen += docs
        total_matches += matches
        domain_docs[domain] += docs
        domain_matches[domain] += matches
        split_docs[split] += docs
        split_matches[split] += matches
        path_by_domain[domain] += 1
        for key, value in payload.get("counts_in_range", {}).items():
            number = int(key)
            if n_min <= number <= n_max:
                counts[number] += int(value)
        worker_summaries.append(
            {
                "filename": payload["filename"],
                "domain": domain,
                "split": split,
                "docs_seen": docs,
                "total_matches": matches,
            }
        )
        if i % 250 == 0 or i == len(paths):
            print(f"[aggregate] merged {i:,}/{len(paths):,} result files")

    return {
        "n_min": n_min,
        "n_max": n_max,
        "docs_seen": docs_seen,
        "total_matches": total_matches,
        "n_files": n_files,
        "n_integers_observed_in_range": len(counts),
        "counts_in_range": {str(k): int(v) for k, v in sorted(counts.items())},
        "domain_docs": {k: int(v) for k, v in sorted(domain_docs.items())},
        "domain_matches": {k: int(v) for k, v in sorted(domain_matches.items())},
        "domain_files": {k: int(v) for k, v in sorted(path_by_domain.items())},
        "split_docs": {k: int(v) for k, v in sorted(split_docs.items())},
        "split_matches": {k: int(v) for k, v in sorted(split_matches.items())},
        "worker_summaries": worker_summaries,
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
        for number in range(n_min, n_max + 1):
            writer.writerow([number, counts.get(number, 0)])
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[local] wrote {csv_path} and {out / 'summary.json'}")


def _log_summary(
    *,
    summary: dict,
    experiment_name: str,
    run_name: str,
    dataset_repo: str,
    splits: list[str],
    domains: list[str],
    results_subdir: str,
    n_workers_requested: int,
    recovered: bool,
) -> None:
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

    counts = {int(k): int(v) for k, v in summary["counts_in_range"].items()}
    n_min = int(summary["n_min"])
    n_max = int(summary["n_max"])
    pad = max(4, len(str(n_max)))

    setup_tracking(None, experiment_name)
    tags = {
        "stage": "number_frequency",
        "backend": "modal",
        "dataset": dataset_repo,
        "split": ",".join(splits),
        "counter_mode": "digit_boundary_multi_range_worker_arrow",
        "results_subdir": results_subdir,
        "n_min": str(n_min),
        "n_max": str(n_max),
    }
    with start_run(run_name=run_name, tags=tags):
        log_params(
            {
                "dataset_repo": dataset_repo,
                "splits": ",".join(splits),
                "domains": ",".join(domains) if domains else "all",
                "n_workers_requested": n_workers_requested,
                "n_files_completed": summary["n_files"],
                "text_key": "text",
                "results_subdir": results_subdir,
                "recovered_from_volume": recovered,
                "count_range": f"{n_min},{n_max}",
                "source_caveat": (
                    "ArmelR/the-pile-splitted is a non-dedup Pile rehost split "
                    "by meta.pile_set_name; it is not the original 30-shard "
                    "EleutherAI release layout."
                ),
            }
        )
        log_metric("docs_seen", summary["docs_seen"])
        log_metric("total_matches", summary["total_matches"])
        if summary["docs_seen"] > 0:
            log_metric("avg_matches_per_doc", summary["total_matches"] / summary["docs_seen"])
        log_metric("n_integers_observed_in_range", len(counts))
        log_metric("n_files_completed", summary["n_files"])

        for domain, value in summary.get("domain_docs", {}).items():
            log_metric(f"domain_docs_{domain}", value)
        for domain, value in summary.get("domain_matches", {}).items():
            log_metric(f"domain_matches_{domain}", value)

        batch: dict[str, float] = {}
        for number in range(n_min, n_max + 1):
            batch[f"count_N_{number:0{pad}d}"] = float(counts.get(number, 0))
            if len(batch) >= 500:
                log_metrics(batch)
                batch = {}
        if batch:
            log_metrics(batch)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["number", "count"])
        for number in range(n_min, n_max + 1):
            writer.writerow([number, counts.get(number, 0)])
        mlflow.log_text(buf.getvalue(), f"counts_{n_min}_to_{n_max}.csv")
        mlflow.log_text(json.dumps(summary.get("domain_docs", {}), indent=2), "domain_docs.json")
        mlflow.log_text(
            json.dumps(summary.get("domain_matches", {}), indent=2),
            "domain_matches.json",
        )
        active = mlflow.active_run()
        print(f"[log] MLflow run_id = {active.info.run_id}")


@app.local_entrypoint()
def main(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    splits: str = "train,test",
    domains: str = "all",
    n_workers: int = 100,
    n_min: int = 0,
    n_max: int = 10000,
    max_files: int = 0,
    experiment_name: str = "numberline_freq_count_expv1",
    run_name: str = "pile_original_splitted_full_number_count",
    results_subdir: str = "",
    results_base_dir: str = "results/pile_original_splitted_modal",
    aggregate_only: bool = False,
    recover_missing: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    import time
    from datetime import datetime

    split_list = _parse_csv(splits) or ["train", "test"]
    domain_list = _parse_csv(domains)
    if not results_subdir:
        results_subdir = (
            f"_results/pile_original_splitted_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

    files = _discover_arrow_files(
        repo_id=dataset_repo,
        splits=split_list,
        domains=domain_list,
        max_files=max_files,
    )
    if not files and not aggregate_only:
        raise SystemExit("no Arrow files discovered")

    print(f"[pile-splitted] dataset={dataset_repo}")
    print(f"[pile-splitted] splits={split_list} domains={domain_list or ['all']}")
    print(f"[pile-splitted] count_range=[{n_min},{n_max}]")
    print(f"[pile-splitted] results_subdir={results_subdir}")
    if files:
        print(f"[pile-splitted] discovered {len(files):,} Arrow files")
        for filename in files[:8]:
            print(f"  {filename}")
        if len(files) > 8:
            print(f"  ... ({len(files) - 8:,} more)")

    if dry_run:
        print("[pile-splitted] --dry-run set; exiting before cloud work")
        return

    recovered = aggregate_only
    if aggregate_only:
        print(f"[pile-splitted] aggregate-only from {results_subdir}")
    else:
        if recover_missing:
            persisted = set(list_persisted_keys.remote(results_subdir))
            before = len(files)
            files = [
                filename for filename in files
                if force or _result_key(filename) not in persisted
            ]
            print(
                f"[pile-splitted] recovery mode: completed={before - len(files):,} "
                f"missing={len(files):,}"
            )
            if not files:
                print("[pile-splitted] no missing files; aggregating existing results")
                recovered = True

        if files:
            print(
                f"[pile-splitted] counting {len(files):,} files "
                f"with up to {n_workers} Modal containers"
            )
            start = time.time()
            results = list(
                count_arrow_file.map(
                    files,
                    kwargs={
                        "repo_id": dataset_repo,
                        "n_min": n_min,
                        "n_max": n_max,
                        "results_subdir": results_subdir,
                        "force": force,
                    },
                    return_exceptions=True,
                )
            )
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                for i, error in enumerate(errors[:5], 1):
                    print(f"[pile-splitted] worker error {i}/{len(errors)}: {error!r}")
                raise RuntimeError(
                    f"{len(errors)} file worker(s) failed; successful results remain "
                    f"in Modal volume subdir {results_subdir}"
                )
            print(f"[pile-splitted] workers complete in {time.time() - start:.1f}s")

    summary = aggregate_persisted.remote(results_subdir, n_min=n_min, n_max=n_max)
    _write_local_summary(summary, f"{results_base_dir}/{n_min}_to_{n_max}")
    _log_summary(
        summary=summary,
        experiment_name=experiment_name,
        run_name=run_name,
        dataset_repo=dataset_repo,
        splits=split_list,
        domains=domain_list,
        results_subdir=results_subdir,
        n_workers_requested=n_workers,
        recovered=recovered,
    )
    print(
        f"[pile-splitted] done docs={summary['docs_seen']:,} "
        f"matches={summary['total_matches']:,} files={summary['n_files']:,}"
    )
