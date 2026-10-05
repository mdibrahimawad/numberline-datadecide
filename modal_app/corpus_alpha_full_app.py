from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import modal

from modal_app.corpus_alpha_app import (
    MAX_N,
    _count_text,
    _fit_alpha,
    _gitlab_lfs_url,
    _hf_headers,
    _hf_url,
    _request_with_retry,
    _tokenizer,
)


APP_NAME = "numberline-corpus-alpha-full"
RESULTS_ROOT = Path("/full_results/v1")
LOCAL_ROOT = Path("results/corpus_alpha_full")
OVERLAP = 2 << 20

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fsspec>=2025.3.0",
        "h5py>=3.12.0",
        "huggingface_hub>=0.28.0",
        "orjson>=3.10.0",
        "pyarrow>=19.0.0",
        "requests>=2.32.0",
        "sentencepiece>=0.2.0",
        "transformers>=5.5.0,<6",
        "zstandard>=0.23.0",
    )
    .add_local_python_source("modal_app")
)
app = modal.App(APP_NAME, image=image)
results = modal.Volume.from_name("numberline-corpus-alpha-full-results", create_if_missing=True)
secret_name = os.environ.get("MODAL_HF_SECRET_NAME", "numberline-hf-token").strip()
secrets = [modal.Secret.from_name(secret_name)] if secret_name else []


def _task_key(task: dict) -> str:
    identity = {key: task[key] for key in sorted(task) if key not in {"url"}}
    return hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _task_path(task: dict) -> Path:
    corpus = task["corpus"].lower().replace(" ", "_").replace("/", "_")
    return RESULTS_ROOT / corpus / "tasks" / f"{_task_key(task)}.json"


def _decode_and_count_native(tokenizer, sequences, counts: Counter[int]) -> int:
    tokens = 0
    backend = tokenizer.backend_tokenizer
    for start in range(0, len(sequences), 32):
        batch = sequences[start : start + 32]
        for text in backend.decode_batch(batch, skip_special_tokens=True):
            _count_text(text, counts)
        tokens += sum(len(ids) for ids in batch)
    return tokens


def _token_range(task: dict) -> dict:
    import orjson

    start, end, size = task["start"], task["end"], task["size"]
    request_start = max(0, start - 1)
    request_end = min(size - 1, end + OVERLAP - 1)
    response = _request_with_retry(
        "get",
        task["url"],
        headers={**_hf_headers(), "Range": f"bytes={request_start}-{request_end}"},
        timeout=300,
    )
    response.raise_for_status()
    data = response.content

    if start == 0:
        begin = 0
    elif data[start - request_start - 1 : start - request_start] == b"\n":
        begin = start - request_start
    else:
        newline = data.find(b"\n", start - request_start)
        if newline < 0:
            raise RuntimeError(f"no line boundary after byte {start} in {task['source']}")
        begin = newline + 1

    if end >= size:
        finish = len(data)
    elif data[end - request_start - 1 : end - request_start] == b"\n":
        finish = end - request_start
    else:
        newline = data.find(b"\n", end - request_start)
        if newline < 0:
            raise RuntimeError(f"line exceeds {OVERLAP} overlap in {task['source']}")
        finish = newline + 1

    counts: Counter[int] = Counter()
    tokenizer = _tokenizer(task["tokenizer"], llama=True)
    batch = []
    documents = tokens = 0
    for line in data[begin:finish].splitlines():
        try:
            token_ids = orjson.loads(line)["token_ids"]
        except (orjson.JSONDecodeError, KeyError, TypeError):
            continue
        batch.append(token_ids)
        documents += 1
        if len(batch) == 32:
            tokens += _decode_and_count_native(tokenizer, batch, counts)
            batch.clear()
    if batch:
        tokens += _decode_and_count_native(tokenizer, batch, counts)
    return _payload(task, counts, documents, tokens)


def _h5_file(task: dict) -> dict:
    import tempfile

    import h5py

    counts: Counter[int] = Counter()
    tokenizer = _tokenizer(task["tokenizer"], llama=True)
    documents = tokens = 0
    response = _request_with_retry("get", task["url"], headers=_hf_headers(), timeout=300)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".h5") as local:
        local.write(response.content)
        local.flush()
        response.close()
        with h5py.File(local.name, "r") as handle:
            data = handle["data"]
            for start in range(0, data.shape[0], 256):
                sequences = data[start : start + 256, 0, :].tolist()
                tokens += _decode_and_count_native(tokenizer, sequences, counts)
                documents += len(sequences)
    return _payload(task, counts, documents, tokens)


def _iter_parquet_texts(source, row_groups=None):
    """Yield every string `text` value of a pyarrow ParquetFile (optionally some row groups)."""
    for batch in source.iter_batches(batch_size=4096, columns=["text"], row_groups=row_groups):
        for text in batch.column(0).to_pylist():
            if isinstance(text, str):
                yield text


def _parquet_file(task: dict) -> dict:
    import tempfile

    import pyarrow.parquet as parquet

    response = _request_with_retry(
        "get", task["url"], headers=_hf_headers(), stream=True, timeout=300
    )
    response.raise_for_status()
    counts: Counter[int] = Counter()
    documents = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=".parquet") as local:
            for chunk in response.iter_content(chunk_size=8 << 20):
                if chunk:
                    local.write(chunk)
            local.flush()

            source = parquet.ParquetFile(local.name)
            for text in _iter_parquet_texts(source):
                _count_text(text, counts)
                documents += 1
    finally:
        response.close()
    return _payload(task, counts, documents, 0)


def _jsonl_text(line: bytes) -> str | None:
    """`text` field of one JSON line, or None if unparsable / not a string."""
    import orjson

    try:
        text = orjson.loads(line).get("text", "")
    except (orjson.JSONDecodeError, AttributeError):
        return None
    return text if isinstance(text, str) else None


def _text_file(task: dict) -> dict:
    import gzip
    import io

    import zstandard

    url = task["url"]
    headers = _hf_headers()
    if task.get("oid"):
        url = _gitlab_lfs_url(task["oid"], task["size"])
        headers = {}
    response = _request_with_retry("get", url, headers=headers, stream=True, timeout=300)
    response.raise_for_status()
    response.raw.decode_content = False
    if task["compression"] == "gz":
        reader = gzip.GzipFile(fileobj=response.raw)
    else:
        reader = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(response.raw))

    counts: Counter[int] = Counter()
    documents = 0
    try:
        for line in reader:
            text = _jsonl_text(line)
            if text is not None:
                _count_text(text, counts)
                documents += 1
    finally:
        reader.close()
        response.close()

    multiplier = task.get("multiplier", 1)
    if multiplier != 1:
        counts = Counter({key: value * multiplier for key, value in counts.items()})
        documents *= multiplier
    return _payload(task, counts, documents, 0)


def _payload(task: dict, counts: Counter[int], documents: int, tokens: int) -> dict:
    input_bytes = (
        task["end"] - task["start"] if task["kind"] == "token_range" else task.get("size", 0)
    )
    return {
        "task_key": _task_key(task),
        "source": task["source"],
        "documents": documents,
        "tokens": tokens,
        "input_bytes": input_bytes,
        "counts": {str(key): int(value) for key, value in counts.items()},
    }


def _count_cached(task: dict) -> dict:
    path = _task_path(task)
    results.reload()
    if path.exists():
        return {"task_key": _task_key(task), "cached": True}
    if task["kind"] == "token_range":
        payload = _token_range(task)
    elif task["kind"] == "h5":
        payload = _h5_file(task)
    elif task["kind"] == "parquet":
        payload = _parquet_file(task)
    else:
        payload = _text_file(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    results.commit()
    return {"task_key": payload["task_key"], "cached": False}


@app.function(
    timeout=12 * 60 * 60,
    cpu=4,
    memory=16384,
    max_containers=100,
    retries=2,
    volumes={"/full_results": results},
    secrets=secrets,
)
def count_task(task: dict) -> dict:
    return _count_cached(task)


@app.function(
    timeout=12 * 60 * 60,
    cpu=2,
    memory=8192,
    max_containers=8,
    retries=2,
    volumes={"/full_results": results},
    secrets=secrets,
)
def count_llmjp_task(task: dict) -> dict:
    return _count_cached(task)


@app.function(timeout=30 * 60, secrets=secrets)
def hf_manifest(repo: str, suffix: str) -> list[tuple[str, int]]:
    from huggingface_hub import HfApi, RepoFile

    return [
        (item.path, int(item.size or 0))
        for item in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_tree(
            repo, repo_type="dataset", recursive=True, expand=True
        )
        if isinstance(item, RepoFile) and item.path.endswith(suffix)
    ]


@app.function(timeout=30 * 60, secrets=secrets)
def hf_paths(repo: str, suffix: str) -> list[tuple[str, int]]:
    import requests

    response = requests.get(
        f"https://huggingface.co/api/datasets/{repo}", headers=_hf_headers(), timeout=300
    )
    response.raise_for_status()
    return [
        (entry["rfilename"], 0)
        for entry in response.json()["siblings"]
        if entry["rfilename"].endswith(suffix)
    ]


def _hf_full_tasks(
    corpus: str,
    repo: str,
    suffix: str,
    kind: str,
    tokenizer: str = "",
    chunk_bytes: int = 512 << 20,
    prefixes: tuple[str, ...] = (),
    multiplier: int = 1,
) -> list[dict]:
    files = (hf_paths if kind == "text" else hf_manifest).remote(repo, suffix)
    if prefixes:
        files = [(path, size) for path, size in files if path.startswith(prefixes)]
    tasks = []
    for path, size in files:
        common = {
            "corpus": corpus,
            "kind": kind,
            "url": _hf_url(repo, path),
            "source": path,
            "size": size,
            "tokenizer": tokenizer,
            "multiplier": multiplier,
        }
        if kind == "token_range":
            tasks.extend(
                {**common, "start": start, "end": min(start + chunk_bytes, size)}
                for start in range(0, size, chunk_bytes)
            )
        else:
            common["compression"] = "zst" if suffix.endswith("zst") else ""
            tasks.append(common)
    return tasks


LLMJP_COMPONENTS = (
    "ja/ja_cc",
    "ja/ja_warp_pdf/e0",
    "ja/ja_warp_html",
    "ja/ja_wiki",
    "ja/kaken",
    "en/en_dolma/cc-head",
    "en/en_dolma/c4",
    "en/en_dolma/reddit",
    "en/en_dolma/pes2o",
    "en/en_dolma/gutenberg",
    "en/en_dolma/wiki",
    "en/en_wiki",
    "code/code_stack",
    "zh/zh_wiki",
    "ko/ko_wiki",
)
LLMJP_TWO_EPOCHS = {
    "ja/ja_cc",
    "ja/ja_warp_pdf/e0",
    "ja/ja_warp_html",
    "ja/ja_wiki",
    "ja/kaken",
}


def _llmjp_full_tasks() -> list[dict]:
    checkout = Path("/tmp/llm-jp-corpus-v3-full")
    if not checkout.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3.git", str(checkout)],
            check=True,
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
        )
    tasks = []
    found = defaultdict(int)
    for path in checkout.rglob("*.jsonl.gz"):
        pointer = path.read_text()
        if not pointer.startswith("version https://git-lfs.github.com/spec/v1"):
            continue
        relative = path.relative_to(checkout).as_posix()
        component = next((name for name in LLMJP_COMPONENTS if relative.startswith(name + "/")), None)
        if not component:
            continue
        fields = dict(line.split(" ", 1) for line in pointer.splitlines()[1:])
        tasks.append(
            {
                "corpus": "LLM-JP Corpus v3",
                "kind": "text",
                "url": "",
                "source": relative,
                "oid": fields["oid"].split(":", 1)[1],
                "size": int(fields["size"]),
                "compression": "gz",
                "multiplier": 2 if component in LLMJP_TWO_EPOCHS else 1,
            }
        )
        found[component] += 1
    missing = [component for component in LLMJP_COMPONENTS if not found[component]]
    if missing:
        raise RuntimeError(f"LLM-JP components have no files: {missing}")
    return tasks


@app.function(
    timeout=30 * 60,
    cpu=2,
    memory=4096,
    max_containers=100,
    volumes={"/full_results": results},
)
def aggregate_part(request: dict) -> dict:
    corpus, tasks = request["corpus"], request["tasks"]
    results.reload()
    counts: Counter[int] = Counter()
    documents = tokens = input_bytes = 0
    missing = []
    for task in tasks:
        path = _task_path(task)
        if not path.exists():
            missing.append(_task_key(task))
            continue
        payload = json.loads(path.read_text())
        counts.update({int(key): int(value) for key, value in payload["counts"].items()})
        documents += int(payload["documents"])
        tokens += int(payload["tokens"])
        input_bytes += int(payload["input_bytes"])
    if missing:
        raise RuntimeError(f"{corpus}: {len(missing)}/{len(tasks)} task results missing")
    return {
        "corpus": corpus,
        "tasks": len(tasks),
        "documents": documents,
        "decoded_tokens": tokens,
        "input_bytes": input_bytes,
        "counts": {str(number): counts.get(number, 0) for number in range(MAX_N + 1)},
    }


def _aggregate_all(corpus: str, tasks: list[dict]) -> dict:
    counts: Counter[int] = Counter()
    documents = tokens = input_bytes = task_count = 0
    requests = [
        {"corpus": corpus, "tasks": tasks[start : start + 100]}
        for start in range(0, len(tasks), 100)
    ]
    for part in aggregate_part.map(requests, order_outputs=False):
        counts.update({int(key): int(value) for key, value in part["counts"].items()})
        task_count += int(part["tasks"])
        documents += int(part["documents"])
        tokens += int(part["decoded_tokens"])
        input_bytes += int(part["input_bytes"])
    alpha, r2 = _fit_alpha(counts)
    return {
        "corpus": corpus,
        "method": "full_pass",
        "tasks": task_count,
        "documents": documents,
        "decoded_tokens": tokens,
        "input_bytes": input_bytes,
        "integer_matches": sum(counts.values()),
        "alpha": alpha,
        "r2": r2,
        "counts": {str(number): counts.get(number, 0) for number in range(MAX_N + 1)},
    }


def _save(summary: dict) -> None:
    out = LOCAL_ROOT / summary["corpus"].lower().replace(" ", "_").replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    counts = summary["counts"]
    with (out / "counts_0_to_10000.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "count"])
        writer.writerows((number, counts[str(number)]) for number in range(MAX_N + 1))
    metadata = {key: value for key, value in summary.items() if key != "counts"}
    (out / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _combine_olmo(summaries: list[dict]) -> dict:
    by_name = {summary["corpus"]: summary for summary in summaries}
    required = ("OLMo-3 Stage 1", "OLMo-3 Midtraining", "OLMo-3 Long-context")
    if not all(name in by_name for name in required):
        raise RuntimeError("all three OLMo-3 stages are required for the complete training exposure")
    counts = Counter({int(key): int(value) for key, value in by_name[required[0]]["counts"].items()})
    counts.update({int(key): 2 * int(value) for key, value in by_name[required[1]]["counts"].items()})
    counts.update({int(key): int(value) for key, value in by_name[required[2]]["counts"].items()})
    alpha, r2 = _fit_alpha(counts)
    return {
        "corpus": "OLMo-3 complete training mixture",
        "method": "full_pass",
        "tasks": sum(by_name[name]["tasks"] for name in required),
        "documents": (
            by_name[required[0]]["documents"]
            + 2 * by_name[required[1]]["documents"]
            + by_name[required[2]]["documents"]
        ),
        "decoded_tokens": 0,
        "input_bytes": (
            by_name[required[0]]["input_bytes"]
            + 2 * by_name[required[1]]["input_bytes"]
            + by_name[required[2]]["input_bytes"]
        ),
        "integer_matches": sum(counts.values()),
        "alpha": alpha,
        "r2": r2,
        "counts": {str(number): counts.get(number, 0) for number in range(MAX_N + 1)},
    }


CORPORA = {
    "amber": ("AmberDatasets", "LLM360/AmberDatasets", ".jsonl", "token_range", "LLM360/Amber"),
    "crystal": ("CrystalCoderDatasets", "LLM360/CrystalCoderDatasets", ".h5", "h5", "LLM360/Crystal"),
    "k2": ("K2Datasets", "LLM360/K2Datasets", ".jsonl", "token_range", "LLM360/K2"),
    "olmo-stage1": ("OLMo-3 Stage 1", "allenai/dolma3_mix-5.5T-1125", ".jsonl.zst", "text", ""),
    "olmo-mid": ("OLMo-3 Midtraining", "allenai/dolma3_dolmino_mix-100B-1125", ".jsonl.zst", "text", ""),
    "olmo-long": ("OLMo-3 Long-context", "allenai/dolma3_longmino_mix-100B-1125", ".jsonl.zst", "text", ""),
}
SLIMPAJAMA_REUPLOAD = (
    "SlimPajama-627B Reupload",
    "gmongaras/SlimPajama-627B_Reupload",
    ".parquet",
    "parquet",
    "",
)


@app.local_entrypoint()
def full(
    corpus: str = "all",
    chunk_mb: int = 512,
    max_tasks: int = 0,
    dry_run: bool = False,
    batch_tasks: int = 1000,
):
    if batch_tasks < 1:
        raise ValueError("batch_tasks must be positive")
    known_names = list(CORPORA) + ["slimpajama", "slimpajama-reupload", "llm-jp"]
    if corpus == "all":
        names = list(CORPORA) + ["slimpajama", "llm-jp"]
    elif corpus == "olmo":
        names = ["olmo-stage1", "olmo-mid", "olmo-long"]
    else:
        names = [corpus]
    summaries = []
    crystal_tasks = None
    for name in names:
        if name == "slimpajama":
            if crystal_tasks is not None:
                tasks = [
                    task
                    for task in crystal_tasks
                    if task["source"].startswith(("phase1/SlimPajama/", "phase2/SlimPajama/"))
                ]
            else:
                tasks = _hf_full_tasks(
                    "CrystalCoderDatasets",
                    "LLM360/CrystalCoderDatasets",
                    ".h5",
                    "h5",
                    "LLM360/Crystal",
                    prefixes=("phase1/SlimPajama/", "phase2/SlimPajama/"),
                )
        elif name == "llm-jp":
            tasks = _llmjp_full_tasks()
        elif name == "slimpajama-reupload":
            corpus_name, repo, suffix, kind, tokenizer = SLIMPAJAMA_REUPLOAD
            tasks = _hf_full_tasks(corpus_name, repo, suffix, kind, tokenizer)
        else:
            if name not in CORPORA:
                raise ValueError(f"unknown corpus {name!r}; choose from {known_names}")
            corpus_name, repo, suffix, kind, tokenizer = CORPORA[name]
            tasks = _hf_full_tasks(
                corpus_name, repo, suffix, kind, tokenizer, chunk_bytes=chunk_mb << 20
            )
            if name == "crystal":
                crystal_tasks = tasks
        if max_tasks > 0:
            tasks = tasks[:max_tasks]
        print(
            f"[{name}] tasks={len(tasks):,} input="
            f"{sum((task['end'] - task['start']) if task['kind'] == 'token_range' else task.get('size', 0) for task in tasks)/1e12:.3f} TB"
        )
        if dry_run:
            continue
        if name == "slimpajama" and crystal_tasks is not None:
            summary = _aggregate_all("SlimPajama-627B", tasks)
            _save(summary)
            summaries.append(summary)
            continue
        worker = count_llmjp_task if name == "llm-jp" else count_task
        completed = 0
        errors = []
        for start in range(0, len(tasks), batch_tasks):
            batch = tasks[start : start + batch_tasks]
            for result in worker.map(
                batch,
                order_outputs=False,
                return_exceptions=True,
                wrap_returned_exceptions=False,
            ):
                if isinstance(result, BaseException):
                    errors.append(result)
                else:
                    completed += 1
                    if completed % 100 == 0 or completed == len(tasks):
                        print(f"[{name}] {completed:,}/{len(tasks):,}")
        if errors:
            raise RuntimeError(f"{name}: {len(errors)} task failures; rerun resumes cached work")
        summary = _aggregate_all(tasks[0]["corpus"], tasks)
        if max_tasks > 0:
            print(json.dumps({key: value for key, value in summary.items() if key != "counts"}, indent=2))
            continue
        _save(summary)
        summaries.append(summary)
    if corpus in {"all", "olmo"} and not dry_run and max_tasks == 0:
        olmo = _combine_olmo(summaries)
        _save(olmo)
        summaries.append(olmo)
    if summaries:
        LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
        metadata = [{key: value for key, value in summary.items() if key != "counts"} for summary in summaries]
        (LOCAL_ROOT / "corpus_alpha_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
