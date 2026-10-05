from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "redpajama-number-frequency"

DEFAULT_DATASET_REPO = "togethercomputer/RedPajama-Data-1T"
RESULTS_VOLUME_NAME = "redpajama-number-frequency-results"
RESULTS_PATH = "/results"

DEFAULT_SUBSETS = (
    "common_crawl",
    "c4",
    "github",
    "arxiv",
    "wikipedia",
    "stackexchange",
)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub>=0.24.0",
        "requests>=2.32.0",
        "zstandard>=0.22.0",
        "orjson>=3.10.0",
    )
    .add_local_python_source("utils")
)

app = modal.App(APP_NAME, image=image)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def _parse_subsets(subsets: str) -> list[str]:
    if subsets.strip().lower() in {"", "all", "default"}:
        return list(DEFAULT_SUBSETS)
    out = [s.strip() for s in subsets.split(",") if s.strip()]
    bad = sorted(set(out) - set(DEFAULT_SUBSETS))
    if bad:
        raise SystemExit(f"unknown RedPajama subset(s): {bad}")
    return out


def _download_url_lists(repo: str, subsets: list[str], max_files_per_subset: int) -> list[dict]:
    from huggingface_hub import hf_hub_download

    tasks: list[dict] = []
    for subset in subsets:
        manifest = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=f"urls/{subset}.txt",
        )
        urls = [
            line.strip()
            for line in Path(manifest).read_text().splitlines()
            if line.strip()
        ]
        if max_files_per_subset > 0:
            urls = urls[:max_files_per_subset]
        for i, url in enumerate(urls):
            tasks.append(
                {
                    "subset": subset,
                    "url": url,
                    "file_index": i,
                    "kind": "zst" if url.endswith(".zst") else "plain",
                }
            )
    return tasks


def _head_size(url: str) -> int:
    import random
    import time

    import requests

    last_error: Exception | None = None
    for attempt in range(1, 9):
        try:
            response = requests.head(url, allow_redirects=True, timeout=30)
            response.raise_for_status()
            value = response.headers.get("content-length")
            if value is None:
                raise RuntimeError(f"HEAD for {url} did not return content-length")
            return int(value)
        except Exception as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status not in {429, 500, 502, 503, 504} and attempt >= 2:
                break
            delay = min(90, 2**attempt) + random.random()
            print(
                f"[redpajama] HEAD retry {attempt}/8 for {url}: "
                f"{type(exc).__name__} status={status}; sleeping {delay:.1f}s"
            )
            time.sleep(delay)
    raise RuntimeError(f"HEAD for {url} failed after retries: {last_error}") from last_error


def _split_plain_tasks(tasks: list[dict], split_plain_bytes: int) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if split_plain_bytes <= 0:
        return tasks

    out: list[dict] = []
    plain = [task for task in tasks if task["kind"] == "plain"]
    zst = [task for task in tasks if task["kind"] == "zst"]
    out.extend(zst)

    sizes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        future_to_url = {pool.submit(_head_size, task["url"]): task["url"] for task in plain}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            sizes[url] = future.result()

    for task in plain:
        size = sizes[task["url"]]
        if size <= 0:
            continue
        start = 0
        range_idx = 0
        while start < size:
            end = min(start + split_plain_bytes - 1, size - 1)
            item = dict(task)
            item.update(
                {
                    "kind": "plain_range",
                    "size": size,
                    "range_start": start,
                    "range_end": end,
                    "range_index": range_idx,
                }
            )
            out.append(item)
            start = end + 1
            range_idx += 1
    return out


def _split_chunk_tasks_for_repair(
    chunks: list[list[dict]],
    repair_split_plain_bytes: int,
) -> list[list[dict]]:
    repair_tasks: list[dict] = []
    for chunk in chunks:
        repair_tasks.extend(_split_plain_tasks(chunk, repair_split_plain_bytes))
    return [[task] for task in repair_tasks]


def _task_label(task: dict) -> str:
    subset = task["subset"]
    url = task["url"]
    name = url.rsplit("/", 1)[-1]
    if task["kind"] == "plain_range":
        return (
            f"{subset}/{name}"
            f"#{task['range_index']}[{task['range_start']}-{task['range_end']}]"
        )
    return f"{subset}/{name}"


def _chunk_key(tasks: list[dict]) -> str:
    import hashlib

    return hashlib.sha1("\n".join(_task_label(task) for task in tasks).encode()).hexdigest()[:16]


@app.function(
    timeout=30 * 60,
    cpu=1.0,
    memory=1024,
    volumes={RESULTS_PATH: results_volume},
)
def list_persisted_keys(results_subdir: str) -> list[str]:
    from pathlib import Path

    results_volume.reload()
    results_dir = Path(RESULTS_PATH) / results_subdir
    if not results_dir.exists():
        return []
    return sorted(path.stem for path in results_dir.glob("*.json"))


@app.function(
    timeout=12 * 60 * 60,
    cpu=2.0,
    memory=4096,
    max_containers=100,
    volumes={RESULTS_PATH: results_volume},
)
def count_tasks(
    tasks: list[dict],
    *,
    max_docs: int = -1,
    log_every: int = 100_000,
    n_min: int = 0,
    n_max: int = 1000,
    results_subdir: str = "_results/redpajama_default",
) -> dict:
    import io
    import json
    import time
    from collections import Counter, defaultdict
    from pathlib import Path

    import orjson
    import requests
    import zstandard as zstd

    from utils.text import DIGIT_TOKEN_PATTERN

    counts: Counter[int] = Counter()
    subset_docs: Counter[str] = Counter()
    subset_matches: Counter[str] = Counter()
    bad_lines = 0
    docs_seen = 0
    total_matches = 0
    session = requests.Session()

    def _count_text(text: str, subset: str) -> None:
        nonlocal total_matches
        toks = DIGIT_TOKEN_PATTERN.findall(text.lower())
        n_hits = len(toks)
        total_matches += n_hits
        subset_matches[subset] += n_hits
        for tok in toks:
            try:
                n = int(tok)
            except ValueError:
                continue
            if n_min <= n <= n_max:
                counts[n] += 1

    def _process_line(line: bytes | str, subset: str) -> bool:
        nonlocal bad_lines, docs_seen
        if not line:
            return False
        try:
            obj = orjson.loads(line)
        except orjson.JSONDecodeError:
            bad_lines += 1
            return False
        docs_seen += 1
        subset_docs[subset] += 1
        text = obj.get("text", "")
        if isinstance(text, str):
            _count_text(text, subset)
        return True

    def _get(url: str, headers: dict[str, str] | None = None):
        last_error: Exception | None = None
        for attempt in range(1, 7):
            try:
                response = session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(30, 180),
                )
                response.raise_for_status()
                return response
            except Exception as exc:
                last_error = exc
                time.sleep(min(30, 2**attempt))
        raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error

    def _get_range_bytes(url: str, start: int, end: int) -> bytes:
        headers = {
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
        }
        last_error: Exception | None = None
        for attempt in range(1, 7):
            try:
                response = session.get(url, headers=headers, timeout=(30, 180))
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"server ignored Range for {url} "
                        f"(status={response.status_code})"
                    )
                return response.content
            except Exception as exc:
                last_error = exc
                time.sleep(min(30, 2**attempt))
        raise RuntimeError(
            f"failed to fetch byte range {start}-{end} from {url}: {last_error}"
        ) from last_error

    def _iter_zst(task: dict):
        with _get(task["url"], headers={"Accept-Encoding": "identity"}) as response:
            response.raw.decode_content = False
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(response.raw) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    yield line

    def _iter_plain_range(task: dict):
        start = int(task["range_start"])
        end = int(task["range_end"])
        size = int(task["size"])
        chunk_bytes = 64 * 1024 * 1024
        fetch_pos = max(0, start - 1)
        pending = b""
        pending_start = fetch_pos
        aligned_start = start == 0

        while True:
            if aligned_start and pending_start > end and not pending:
                return
            if fetch_pos >= size:
                if pending and pending_start <= end:
                    yield pending
                return

            fetch_end = min(fetch_pos + chunk_bytes - 1, size - 1)
            chunk_start = fetch_pos
            data = _get_range_bytes(task["url"], fetch_pos, fetch_end)
            if not data:
                return
            fetch_pos += len(data)

            if pending:
                data = pending + data
                data_start = pending_start
                pending = b""
            else:
                data_start = chunk_start

            if not aligned_start:
                if data[0:1] == b"\n":
                    data = data[1:]
                    data_start += 1
                    aligned_start = True
                else:
                    newline = data.find(b"\n")
                    if newline < 0:
                        pending_start = data_start + len(data)
                        continue
                    data = data[newline + 1 :]
                    data_start += newline + 1
                    aligned_start = True

            offset = 0
            while True:
                newline = data.find(b"\n", offset)
                if newline < 0:
                    break
                line_start = data_start + offset
                if line_start <= end:
                    yield data[offset : newline + 1]
                    offset = newline + 1
                else:
                    return

            pending = data[offset:]
            pending_start = data_start + offset
            if pending_start > end:
                return

    def _iter_plain_file(task: dict):
        import requests

        url = task["url"]
        offset = 0
        pending = b""
        chunk_bytes = 64 * 1024 * 1024
        last_error: Exception | None = None

        for attempt in range(1, 8):
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with _get(url, headers=headers) as response:
                    if offset and response.status_code != 206:
                        raise RuntimeError(
                            f"server ignored resume Range for {url} "
                            f"(status={response.status_code})"
                        )
                    response.raw.decode_content = False
                    for chunk in response.iter_content(chunk_size=chunk_bytes):
                        if not chunk:
                            continue
                        offset += len(chunk)
                        data = pending + chunk
                        parts = data.split(b"\n")
                        for line in parts[:-1]:
                            yield line + b"\n"
                        pending = parts[-1]
                    if pending:
                        yield pending
                    return
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                print(
                    f"[worker] stream broke for {url}; "
                    f"retry {attempt}/7 from byte {offset:,}: {exc!r}"
                )
                time.sleep(min(60, 2**attempt))

        raise RuntimeError(f"failed to stream {url} after retries: {last_error}") from last_error

    task_summaries: list[dict] = []
    cap = max_docs if max_docs >= 0 else 10**30
    for task_idx, task in enumerate(tasks, 1):
        if docs_seen >= cap:
            break
        subset = task["subset"]
        label = _task_label(task)
        before_docs = docs_seen
        before_matches = total_matches
        print(f"[worker] {task_idx}/{len(tasks)} opening {label}")

        if task["kind"] == "zst":
            iterator = _iter_zst(task)
        elif task["kind"] == "plain_range":
            iterator = _iter_plain_range(task)
        else:
            iterator = _iter_plain_file(task)

        for line in iterator:
            if docs_seen >= cap:
                break
            _process_line(line, subset)
            if log_every > 0 and docs_seen % log_every == 0:
                print(
                    f"[worker] docs={docs_seen:,} "
                    f"matches={total_matches:,} bad_lines={bad_lines:,}"
                )

        task_summaries.append(
            {
                "label": label,
                "subset": subset,
                "docs_seen": docs_seen - before_docs,
                "total_matches": total_matches - before_matches,
            }
        )
        print(
            f"[worker] progress tasks={task_idx}/{len(tasks)} "
            f"docs={docs_seen:,} matches={total_matches:,}"
        )

    payload = {
        "docs_seen": int(docs_seen),
        "total_matches": int(total_matches),
        "bad_lines": int(bad_lines),
        "counts_in_range": {str(k): int(v) for k, v in sorted(counts.items())},
        "n_min": n_min,
        "n_max": n_max,
        "n_tasks": len(tasks),
        "task_labels": [_task_label(task) for task in tasks],
        "subset_docs": {k: int(v) for k, v in sorted(subset_docs.items())},
        "subset_matches": {k: int(v) for k, v in sorted(subset_matches.items())},
        "task_summaries": task_summaries,
    }

    results_dir = Path(RESULTS_PATH) / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    key = _chunk_key(tasks)
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
    subset_docs: Counter[str] = Counter()
    subset_matches: Counter[str] = Counter()
    docs_seen = 0
    total_matches = 0
    bad_lines = 0
    n_tasks = 0
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
        bad_lines += int(payload.get("bad_lines", 0))
        n_tasks += int(payload["n_tasks"])
        counts.update(
            {
                int(k): int(v)
                for k, v in payload["counts_in_range"].items()
                if n_min <= int(k) <= n_max
            }
        )
        subset_docs.update({k: int(v) for k, v in payload.get("subset_docs", {}).items()})
        subset_matches.update(
            {k: int(v) for k, v in payload.get("subset_matches", {}).items()}
        )
        workers.append(
            {
                "task_labels": payload["task_labels"],
                "n_tasks": int(payload["n_tasks"]),
                "docs_seen": int(payload["docs_seen"]),
                "total_matches": int(payload["total_matches"]),
                "bad_lines": int(payload.get("bad_lines", 0)),
                "subset_docs": payload.get("subset_docs", {}),
                "subset_matches": payload.get("subset_matches", {}),
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
        "bad_lines": bad_lines,
        "n_tasks": n_tasks,
        "n_workers": len(paths),
        "counts_in_range": {str(k): int(v) for k, v in sorted(counts.items())},
        "subset_docs": {k: int(v) for k, v in sorted(subset_docs.items())},
        "subset_matches": {k: int(v) for k, v in sorted(subset_matches.items())},
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
    subsets: list[str],
    results_subdir: str,
    n_workers_requested: int,
    split_plain_bytes: int,
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
                "subsets": ",".join(subsets),
                "text_key": "text",
                "n_workers_requested": n_workers_requested,
                "n_workers_effective": summary["n_workers"],
                "n_tasks_total": summary["n_tasks"],
                "split_plain_bytes": split_plain_bytes,
                "results_subdir": results_subdir,
                "recovered_from_volume": recovered,
                "count_range": f"{n_min},{n_max}",
                "n_min": n_min,
                "n_max": n_max,
            }
        )
        log_metric("docs_seen", summary["docs_seen"])
        log_metric("total_matches", summary["total_matches"])
        log_metric("bad_lines", summary.get("bad_lines", 0))
        if summary["docs_seen"] > 0:
            log_metric(
                "avg_matches_per_doc",
                summary["total_matches"] / summary["docs_seen"],
            )
        log_metric("n_integers_observed_in_range", len(counts_in_range))

        for subset, value in summary.get("subset_docs", {}).items():
            log_metric(f"subset_docs_{subset}", value)
        for subset, value in summary.get("subset_matches", {}).items():
            log_metric(f"subset_matches_{subset}", value)

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
        mlflow.log_text(json.dumps(summary.get("subset_docs", {}), indent=2), "subset_docs.json")
        mlflow.log_text(
            json.dumps(summary.get("subset_matches", {}), indent=2),
            "subset_matches.json",
        )

        for i, worker in enumerate(summary["worker_summaries"]):
            mlflow.log_text(
                json.dumps(worker, indent=2),
                f"worker_stats/worker_{i:03d}.json",
            )

        active = mlflow.active_run()
        print(f"[log] MLflow run_id = {active.info.run_id}")


@app.local_entrypoint()
def main(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    subsets: str = "all",
    n_workers: int = 512,
    max_docs: int = -1,
    log_every: int = 250_000,
    n_min: int = 0,
    n_max: int = 1000,
    split_plain_mb: int = 512,
    max_files_per_subset: int = 0,
    experiment_name: str = "numberline_freq_count_expv1",
    run_name: str = "redpajama_full_number_count",
    results_subdir: str = "",
    results_base_dir: str = "results/redpajama_full_modal",
    aggregate_only: bool = False,
    recover_missing: bool = False,
    repair_missing_split: bool = False,
    repair_split_plain_mb: int = 512,
    dry_run: bool = False,
) -> None:
    import time
    from datetime import datetime

    if n_workers < 1:
        raise SystemExit("--n-workers must be >= 1")
    subset_list = _parse_subsets(subsets)
    if not results_subdir:
        results_subdir = (
            f"_results/redpajama_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

    split_plain_bytes = int(split_plain_mb) * 1024 * 1024
    if aggregate_only:
        print(f"[redpajama] dataset={dataset_repo}")
        print(f"[redpajama] subsets={subset_list}")
        print(f"[redpajama] count_range=[{n_min},{n_max}]")
        print(f"[redpajama] aggregate-only from {results_subdir}")
        summary = aggregate_persisted.remote(results_subdir, n_min=n_min, n_max=n_max)
        _write_local_summary(summary, f"{results_base_dir}/{n_min}_to_{n_max}")
        _log_summary(
            summary=summary,
            experiment_name=experiment_name,
            run_name=run_name,
            dataset_repo=dataset_repo,
            subsets=subset_list,
            results_subdir=results_subdir,
            n_workers_requested=n_workers,
            split_plain_bytes=split_plain_bytes,
            recovered=True,
        )
        return

    raw_tasks = _download_url_lists(dataset_repo, subset_list, max_files_per_subset)
    tasks = _split_plain_tasks(raw_tasks, split_plain_bytes)
    if not tasks:
        raise SystemExit("no RedPajama URL tasks discovered")

    effective_workers = min(n_workers, len(tasks))
    chunks = [tasks[i::effective_workers] for i in range(effective_workers)]
    chunks = [chunk for chunk in chunks if chunk]
    if recover_missing:
        persisted_keys = set(list_persisted_keys.remote(results_subdir))
        missing_chunks = [
            chunk for chunk in chunks if _chunk_key(chunk) not in persisted_keys
        ]
        print(f"[redpajama] recovery mode from {results_subdir}")
        print(
            f"[redpajama] persisted_chunks={len(persisted_keys)} "
            f"expected_chunks={len(chunks)} missing_chunks={len(missing_chunks)}"
        )
        for chunk in missing_chunks[:8]:
            print(f"  missing {_chunk_key(chunk)} first={_task_label(chunk[0])}")
        if len(missing_chunks) > 8:
            print(f"  ... ({len(missing_chunks) - 8} more missing chunks)")
        if not missing_chunks:
            print("[redpajama] no missing chunks; aggregating existing results")
            summary = aggregate_persisted.remote(results_subdir, n_min=n_min, n_max=n_max)
            _write_local_summary(summary, f"{results_base_dir}/{n_min}_to_{n_max}")
            _log_summary(
                summary=summary,
                experiment_name=experiment_name,
                run_name=run_name,
                dataset_repo=dataset_repo,
                subsets=subset_list,
                results_subdir=results_subdir,
                n_workers_requested=n_workers,
                split_plain_bytes=split_plain_bytes,
                recovered=True,
            )
            return
        if repair_missing_split:
            repair_split_plain_bytes = int(repair_split_plain_mb) * 1024 * 1024
            repair_chunks = _split_chunk_tasks_for_repair(
                missing_chunks,
                repair_split_plain_bytes,
            )
            print(
                f"[redpajama] repair-split enabled: "
                f"missing_chunks={len(missing_chunks)} "
                f"repair_chunks={len(repair_chunks)} "
                f"repair_split_plain_mb={repair_split_plain_mb}"
            )
            for chunk in repair_chunks[:12]:
                print(f"  repair {_chunk_key(chunk)} first={_task_label(chunk[0])}")
            if len(repair_chunks) > 12:
                print(f"  ... ({len(repair_chunks) - 12} more repair chunks)")
            if dry_run:
                print("[redpajama] --dry-run set; exiting before cloud repair work")
                return

            start = time.time()
            worker_results = list(
                count_tasks.map(
                    repair_chunks,
                    kwargs={
                        "max_docs": -1,
                        "log_every": log_every,
                        "n_min": n_min,
                        "n_max": n_max,
                        "results_subdir": results_subdir,
                    },
                    return_exceptions=True,
                )
            )
            errors = [result for result in worker_results if isinstance(result, BaseException)]
            if errors:
                for i, error in enumerate(errors[:5], 1):
                    print(f"[redpajama] repair worker error {i}/{len(errors)}: {error!r}")
                raise RuntimeError(
                    f"{len(errors)} repair worker(s) failed; successful repair shards "
                    f"remain in Modal volume subdir {results_subdir}"
                )
            elapsed = time.time() - start
            print(f"[redpajama] repair workers complete in {elapsed:.1f}s; aggregating")

            summary = aggregate_persisted.remote(results_subdir, n_min=n_min, n_max=n_max)
            _write_local_summary(summary, f"{results_base_dir}/{n_min}_to_{n_max}")
            _log_summary(
                summary=summary,
                experiment_name=experiment_name,
                run_name=run_name,
                dataset_repo=dataset_repo,
                subsets=subset_list,
                results_subdir=results_subdir,
                n_workers_requested=n_workers,
                split_plain_bytes=split_plain_bytes,
                recovered=True,
            )
            print(
                f"[redpajama] done docs={summary['docs_seen']:,} "
                f"matches={summary['total_matches']:,}"
            )
            return
        chunks = missing_chunks
    max_per_worker = -1 if max_docs < 0 else max(1, max_docs // len(chunks))

    n_zst = sum(1 for task in tasks if task["kind"] == "zst")
    n_ranges = sum(1 for task in tasks if task["kind"] == "plain_range")
    n_plain = sum(1 for task in tasks if task["kind"] == "plain")
    print(f"[redpajama] dataset={dataset_repo}")
    print(f"[redpajama] subsets={subset_list}")
    print(f"[redpajama] count_range=[{n_min},{n_max}]")
    print(
        f"[redpajama] raw_files={len(raw_tasks)} tasks={len(tasks)} "
        f"(zst={n_zst}, plain_ranges={n_ranges}, plain_files={n_plain})"
    )
    print(f"[redpajama] workers={len(chunks)} results_subdir={results_subdir}")
    print(
        "[redpajama] max_docs="
        + ("FULL" if max_docs < 0 else f"{max_docs:,}")
        + f" max_per_worker={max_per_worker}"
    )
    for task in tasks[:8]:
        print(f"  {_task_label(task)}")
    if len(tasks) > 8:
        print(f"  ... ({len(tasks) - 8} more)")

    if dry_run:
        print("[redpajama] --dry-run set; exiting before cloud work")
        return

    start = time.time()
    worker_results = list(
        count_tasks.map(
            chunks,
            kwargs={
                "max_docs": max_per_worker,
                "log_every": log_every,
                "n_min": n_min,
                "n_max": n_max,
                "results_subdir": results_subdir,
            },
            return_exceptions=True,
        )
    )
    errors = [result for result in worker_results if isinstance(result, BaseException)]
    if errors:
        for i, error in enumerate(errors[:5], 1):
            print(f"[redpajama] worker error {i}/{len(errors)}: {error!r}")
        raise RuntimeError(
            f"{len(errors)} worker(s) failed; successful worker shards remain "
            f"in Modal volume subdir {results_subdir}"
        )
    elapsed = time.time() - start
    print(f"[redpajama] workers complete in {elapsed:.1f}s; aggregating")

    summary = aggregate_persisted.remote(results_subdir, n_min=n_min, n_max=n_max)
    _write_local_summary(summary, f"{results_base_dir}/{n_min}_to_{n_max}")
    _log_summary(
        summary=summary,
        experiment_name=experiment_name,
        run_name=run_name,
        dataset_repo=dataset_repo,
        subsets=subset_list,
        results_subdir=results_subdir,
        n_workers_requested=n_workers,
        split_plain_bytes=split_plain_bytes,
        recovered=False,
    )
    print(
        f"[redpajama] done docs={summary['docs_seen']:,} "
        f"matches={summary['total_matches']:,}"
    )
