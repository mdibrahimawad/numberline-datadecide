from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

APP_NAME = "dolma-number-frequency"
DATASET_REPO = "allenai/dolma"
DEFAULT_DOLMA_VERSION = "v1_5-sample"
RESULTS_VOLUME_NAME = "dolma-number-frequency-results"
RESULTS_PATH = "/results"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub>=0.24.0",
        "requests>=2.32.0",
        "orjson>=3.10.0",
    )
    .add_local_python_source("utils")
)

app = modal.App(APP_NAME, image=image)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def _url_key(url: str) -> str:
    import hashlib

    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]


def _source_from_url(url: str) -> str:
    parts = url.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


@app.function(
    timeout=6 * 60 * 60,
    cpu=2.0,
    memory=4096,
    volumes={RESULTS_PATH: results_volume},
    max_containers=512,
    retries=1,
)
def count_url_task(
    url: str,
    *,
    max_n: int = 10_000,
    max_docs: int = -1,
    log_every: int = 1_000_000,
    results_subdir: str = "_results/dolma_v1_5_sample",
) -> dict:
    import gzip
    import http.client
    import io
    import json
    import time

    import orjson
    import requests
    import urllib3

    from utils.text import DIGIT_TOKEN_PATTERN

    results_volume.reload()
    out_dir = Path(RESULTS_PATH) / results_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_url_key(url)}.json"

    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path) as fh:
            cached = json.load(fh)
        print(f"[worker] cached {url} -> {out_path}")
        return {
            "url": url,
            "source": cached.get("source", _source_from_url(url)),
            "docs_seen": int(cached.get("docs_seen", 0)),
            "total_matches": int(cached.get("total_matches", 0)),
            "bad_lines": int(cached.get("bad_lines", 0)),
            "cached": True,
        }

    max_digits = len(str(max_n))
    source = _source_from_url(url)
    started = time.time()
    attempts = 4

    counts: dict[int, int] = {}
    docs_seen = 0
    total_matches = 0
    bad_lines = 0
    for attempt in range(1, attempts + 1):
        counts = {}
        docs_seen = 0
        total_matches = 0
        bad_lines = 0
        try:
            print(f"[worker] opening {url} attempt={attempt}/{attempts}")
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                # olmo-data.org serves the .json.gz file with HTTP Content-Encoding:gzip.
                # Let requests remove that transfer layer, then gzip-decode the file.
                response.raw.decode_content = True
                with gzip.GzipFile(fileobj=response.raw) as gz:
                    text_stream = io.TextIOWrapper(gz, encoding="utf-8")
                    for line in text_stream:
                        if not line:
                            continue
                        try:
                            obj = orjson.loads(line)
                        except orjson.JSONDecodeError:
                            bad_lines += 1
                            continue
                        text = obj.get("text", "")
                        if not isinstance(text, str):
                            docs_seen += 1
                            continue

                        toks = DIGIT_TOKEN_PATTERN.findall(text)
                        total_matches += len(toks)
                        for tok in toks:
                            normalized = tok.lstrip("0") or "0"
                            if len(normalized) > max_digits:
                                continue
                            k = int(normalized)
                            if k <= max_n:
                                counts[k] = counts.get(k, 0) + 1

                        docs_seen += 1
                        if max_docs > 0 and docs_seen >= max_docs:
                            break
                        if log_every > 0 and docs_seen % log_every == 0:
                            print(
                                f"[worker] {source} docs={docs_seen:,} "
                                f"matches={total_matches:,} bad_lines={bad_lines}"
                            )
            break
        except (
            requests.RequestException,
            urllib3.exceptions.HTTPError,
            http.client.IncompleteRead,
            EOFError,
            OSError,
            gzip.BadGzipFile,
            UnicodeDecodeError,
        ) as exc:
            if attempt == attempts:
                raise
            delay = min(60, 2**attempt)
            print(f"[worker] transient read failure {url}: {type(exc).__name__}: {exc}; retrying in {delay}s")
            time.sleep(delay)

    payload = {
        "url": url,
        "source": source,
        "docs_seen": docs_seen,
        "total_matches": total_matches,
        "bad_lines": bad_lines,
        "max_n": max_n,
        "elapsed_seconds": time.time() - started,
        "counts": {str(k): v for k, v in sorted(counts.items()) if v},
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
    results_volume.commit()
    print(
        f"[worker] persisted {url} -> {out_path} "
        f"docs={docs_seen:,} matches={total_matches:,}"
    )
    return {
        "url": url,
        "source": source,
        "docs_seen": docs_seen,
        "total_matches": total_matches,
        "bad_lines": bad_lines,
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
    bad_lines = 0
    per_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "docs_seen": 0, "total_matches": 0, "bad_lines": 0}
    )

    for i, p in enumerate(paths, 1):
        with open(p) as fh:
            payload = json.load(fh)
        source = payload.get("source", "unknown")
        docs = int(payload.get("docs_seen", 0))
        matches = int(payload.get("total_matches", 0))
        bad = int(payload.get("bad_lines", 0))
        docs_seen += docs
        total_matches += matches
        bad_lines += bad
        per_source[source]["files"] += 1
        per_source[source]["docs_seen"] += docs
        per_source[source]["total_matches"] += matches
        per_source[source]["bad_lines"] += bad
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
        "bad_lines": bad_lines,
        "n_integers_observed_in_range": len(counts),
        "counts_in_range": {str(k): v for k, v in sorted(counts.items())},
        "per_source": dict(sorted(per_source.items())),
    }


def _load_manifest(version: str) -> list[str]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DATASET_REPO,
        f"urls/{version}.txt",
        repo_type="dataset",
    )
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


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


def _log_mlflow(summary: dict, *, experiment_name: str, run_name: str, version: str, results_subdir: str) -> str:
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
        "dataset_config": version,
        "split": "train",
        "counter_mode": "digit_boundary_multi",
        "results_subdir": results_subdir,
        "n_min": str(n_min),
        "n_max": str(n_max),
    }

    with start_run(run_name=run_name, tags=tags) as active:
        log_params(
            {
                "dataset_repo": DATASET_REPO,
                "dataset_config": version,
                "results_subdir": results_subdir,
                "n_workers": summary["n_workers"],
                "n_min": n_min,
                "n_max": n_max,
            }
        )
        log_metric("docs_seen", summary["docs_seen"])
        log_metric("total_matches", summary["total_matches"])
        log_metric("bad_lines", summary["bad_lines"])
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
        mlflow.log_text(json.dumps(summary["per_source"], indent=2), "per_source.json")
        run_id = active.info.run_id

    print(f"[mlflow] logged {run_name}: {run_id}")
    return run_id


@app.local_entrypoint()
def main(
    version: str = DEFAULT_DOLMA_VERSION,
    max_n: int = 10_000,
    max_urls: int = -1,
    max_docs_per_file: int = -1,
    log_every: int = 1_000_000,
    results_subdir: str = "",
    results_base_dir: str = "results/dolma_v1_5_sample",
    experiment_name: str = "numberline_freq_count_expv1",
    run_name_prefix: str = "dolma_v1_5_sample",
    aggregate_only: bool = False,
    dry_run: bool = False,
) -> None:
    from datetime import datetime

    if max_n < 1:
        raise SystemExit("--max-n must be >= 1")
    if not results_subdir:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results_subdir = f"_results/dolma_{version}_{stamp}"

    urls = _load_manifest(version)
    if max_urls > 0:
        urls = urls[:max_urls]
    print(f"[dolma] version={version} urls={len(urls)} max_n={max_n}")
    print(f"[dolma] results_subdir={results_subdir} volume={RESULTS_VOLUME_NAME}")
    for url in urls[:5]:
        print(f"  {url}")
    if len(urls) > 5:
        print(f"  ... ({len(urls) - 5} more)")

    if dry_run:
        print("[dolma] --dry-run set; exiting before Modal work")
        return

    if not aggregate_only:
        results = list(
            count_url_task.map(
                urls,
                kwargs={
                    "max_n": max_n,
                    "max_docs": max_docs_per_file,
                    "log_every": log_every,
                    "results_subdir": results_subdir,
                },
                order_outputs=False,
                return_exceptions=True,
            )
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        print(f"[dolma] count map returned {len(results)} results, errors={len(errors)}")
        if errors:
            for e in errors[:10]:
                print(f"[dolma] ERROR: {type(e).__name__}: {e}")
            raise RuntimeError(
                f"{len(errors)} Dolma URL tasks failed; rerun with the same "
                f"--results-subdir {results_subdir!r} to reuse completed files."
            )

    summaries: list[tuple[int, int, dict]] = []
    for n_min, n_max in [(0, 1000), (0, max_n)]:
        summary = aggregate_persisted.remote(results_subdir, n_min=n_min, n_max=n_max)
        summaries.append((n_min, n_max, summary))
        out_dir = f"{results_base_dir}/{n_min}_to_{n_max}"
        _write_local_summary(summary, out_dir)
        _log_mlflow(
            summary,
            experiment_name=experiment_name,
            run_name=f"{run_name_prefix}_{n_min}_to_{n_max}",
            version=version,
            results_subdir=results_subdir,
        )

    for n_min, n_max, summary in summaries:
        print(
            f"[dolma] [{n_min},{n_max}] docs={summary['docs_seen']:,} "
            f"matches={summary['total_matches']:,} "
            f"in_range={sum(int(v) for v in summary['counts_in_range'].values()):,} "
            f"bad_lines={summary['bad_lines']:,}"
        )
