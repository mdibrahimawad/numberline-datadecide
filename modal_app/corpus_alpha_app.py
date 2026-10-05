from __future__ import annotations

import csv
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import modal


APP_NAME = "numberline-corpus-alpha"
MAX_N = 10_000
DEFAULT_TASKS = 64
OUT_ROOT = Path("results/corpus_alpha_extended")
CACHE_ROOT = Path("/alpha_cache/v1")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fsspec>=2025.3.0",
    "h5py>=3.12.0",
    "huggingface_hub>=0.28.0",
    "numpy>=2.0.0",
    "orjson>=3.10.0",
    "requests>=2.32.0",
    "sentencepiece>=0.2.0",
    "transformers>=5.5.0,<6",
    "zstandard>=0.23.0",
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("numberline-hf-cache", create_if_missing=True)
sample_cache = modal.Volume.from_name("numberline-corpus-alpha-cache", create_if_missing=True)
secret_name = os.environ.get("MODAL_HF_SECRET_NAME", "numberline-hf-token").strip()
secrets = [modal.Secret.from_name(secret_name)] if secret_name else []


def _tokenizer(repo: str, llama: bool = False):
    from transformers import AutoTokenizer, LlamaTokenizer

    cache = getattr(_tokenizer, "cache", {})
    key = (repo, llama)
    if key not in cache:
        cls = LlamaTokenizer if llama else AutoTokenizer
        cache[key] = cls.from_pretrained(
            repo,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
            use_fast=not llama,
        )
        _tokenizer.cache = cache
    return cache[key]


def _count_text(text: str, counts: Counter[int]) -> None:
    import re

    pattern = getattr(_count_text, "pattern", None)
    if pattern is None:
        pattern = re.compile(r"\b\d+\b")
        _count_text.pattern = pattern
    for match in pattern.findall(text):
        normalized = match.lstrip("0") or "0"
        if len(normalized) > len(str(MAX_N)):
            continue
        value = int(normalized)
        if value <= MAX_N:
            counts[value] += 1


def _decode_and_count(tokenizer, sequences, counts: Counter[int]) -> int:
    tokens = 0
    for start in range(0, len(sequences), 32):
        batch = sequences[start : start + 32]
        for text in tokenizer.batch_decode(batch, skip_special_tokens=True):
            _count_text(text, counts)
        tokens += sum(len(ids) for ids in batch)
    return tokens


def _hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _request_with_retry(method: str, url: str, **kwargs):
    import time

    import requests

    for attempt in range(8):
        try:
            response = requests.request(method, url, **kwargs)
        except requests.RequestException:
            if attempt == 7:
                raise
        else:
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == 7:
                return response
            response.close()
        time.sleep(min(60, 2**attempt) + random.random())
    raise RuntimeError("unreachable")


def _sample_token_jsonl(task: dict) -> dict:
    import orjson
    import requests

    head = _request_with_retry("head", task["url"], headers=_hf_headers(), allow_redirects=True, timeout=180)
    head.raise_for_status()
    size = int(head.headers.get("x-linked-size") or head.headers["content-length"])
    window = min(task["window"], size)
    start = int(task["fraction"] * max(size - window, 0))
    response = _request_with_retry(
        "get",
        task["url"],
        headers={**_hf_headers(), "Range": f"bytes={start}-{start + window - 1}"},
        timeout=180,
    )
    response.raise_for_status()
    lines = response.content.splitlines()
    if start:
        lines = lines[1:]
    if response.content and not response.content.endswith(b"\n"):
        lines = lines[:-1]

    sequences = []
    for line in lines:
        try:
            sequences.append(orjson.loads(line)["token_ids"])
        except (orjson.JSONDecodeError, KeyError, TypeError):
            continue

    counts: Counter[int] = Counter()
    tokenizer = _tokenizer(task["tokenizer"], llama=True)
    tokens = _decode_and_count(tokenizer, sequences, counts)
    return _result(task, counts, tokens, len(sequences))


def _sample_h5(task: dict) -> dict:
    import fsspec
    import h5py

    counts: Counter[int] = Counter()
    with fsspec.open(task["url"], "rb", headers=_hf_headers(), block_size=4 << 20) as remote:
        with h5py.File(remote, "r") as handle:
            data = handle["data"]
            rows = min(task["rows"], data.shape[0])
            start = random.Random(task["seed"]).randrange(data.shape[0] - rows + 1)
            sequences = data[start : start + rows, 0, :].tolist()

    tokenizer = _tokenizer(task["tokenizer"], llama=True)
    tokens = _decode_and_count(tokenizer, sequences, counts)
    return _result(task, counts, tokens, len(sequences))


def _gitlab_lfs_url(oid: str, size: int) -> str:
    endpoint = (
        "https://gitlab.llm-jp.nii.ac.jp/datasets/"
        "llm-jp-corpus-v3.git/info/lfs/objects/batch"
    )
    media = "application/vnd.git-lfs+json"
    payload = {
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size}],
        "ref": {"name": "refs/heads/main"},
    }
    response = _request_with_retry(
        "post", endpoint, json=payload, headers={"Accept": media}, timeout=60
    )
    response.raise_for_status()
    return response.json()["objects"][0]["actions"]["download"]["href"]


def _sample_text_stream(task: dict) -> dict:
    import gzip
    import io

    import orjson
    import requests
    import zstandard

    url = task["url"]
    if task.get("oid"):
        url = _gitlab_lfs_url(task["oid"], task["size"])
    response = _request_with_retry(
        "get", url, headers=_hf_headers(), stream=True, timeout=180
    )
    response.raise_for_status()
    response.raw.decode_content = False

    if task["compression"] == "gz":
        reader = gzip.GzipFile(fileobj=response.raw)
    else:
        reader = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(response.raw))

    tokenizer = _tokenizer(task["tokenizer"])
    counts: Counter[int] = Counter()
    tokens = docs = 0
    batch: list[str] = []
    try:
        for line in reader:
            try:
                text = orjson.loads(line).get("text", "")
            except (orjson.JSONDecodeError, AttributeError):
                continue
            if not isinstance(text, str):
                continue
            batch.append(text)
            docs += 1
            if len(batch) < 32:
                continue
            for item in batch:
                _count_text(item, counts)
            tokens += sum(len(ids) for ids in tokenizer(batch, add_special_tokens=False)["input_ids"])
            batch.clear()
            if tokens >= task["target_tokens"]:
                break
        if batch:
            for item in batch:
                _count_text(item, counts)
            tokens += sum(len(ids) for ids in tokenizer(batch, add_special_tokens=False)["input_ids"])
    finally:
        reader.close()
        response.close()
    return _result(task, counts, tokens, docs)


def _result(task: dict, counts: Counter[int], tokens: int, docs: int) -> dict:
    return {
        "corpus": task["corpus"],
        "task_id": task["task_id"],
        "source": task.get("source", ""),
        "tokens": tokens,
        "documents": docs,
        "counts": dict(counts),
    }


@app.function(
    timeout=45 * 60,
    cpu=2,
    memory=8192,
    max_containers=100,
    retries=2,
    volumes={"/root/.cache/huggingface": hf_cache, "/alpha_cache": sample_cache},
    secrets=secrets,
)
def sample_task(task: dict) -> dict:
    return _cached_sample(task)


@app.function(
    timeout=45 * 60,
    cpu=2,
    memory=8192,
    max_containers=8,
    retries=2,
    volumes={"/root/.cache/huggingface": hf_cache, "/alpha_cache": sample_cache},
    secrets=secrets,
)
def sample_llmjp_task(task: dict) -> dict:
    return _cached_sample(task)


def _cached_sample(task: dict) -> dict:
    cache_path = CACHE_ROOT / task["corpus"].replace(" ", "_").replace("/", "_") / f'{task["task_id"]:04d}.json'
    sample_cache.reload()
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    if task["kind"] == "token_jsonl":
        result = _sample_token_jsonl(task)
    elif task["kind"] == "h5":
        result = _sample_h5(task)
    else:
        result = _sample_text_stream(task)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))
    sample_cache.commit()
    return result


@app.function(secrets=secrets, timeout=10 * 60)
def hf_files(repo: str, suffix: str) -> list[tuple[str, int]]:
    import requests

    response = requests.get(
        f"https://huggingface.co/api/datasets/{repo}",
        headers=_hf_headers(),
        timeout=120,
    )
    response.raise_for_status()
    return [
        (entry["rfilename"], 0)
        for entry in response.json()["siblings"]
        if entry["rfilename"].endswith(suffix)
    ]


def _hf_url(repo: str, path: str) -> str:
    from urllib.parse import quote

    return f"https://huggingface.co/datasets/{repo}/resolve/main/{quote(path)}"


def _hf_tasks(
    corpus: str,
    repo: str,
    suffix: str,
    kind: str,
    tokenizer: str,
    n: int,
    prefixes: tuple[str, ...] = (),
):
    files = hf_files.remote(repo, suffix)
    if prefixes:
        files = [(path, size) for path, size in files if path.startswith(prefixes)]
    if not files:
        raise RuntimeError(f"no {suffix} files found in {repo}")
    rng = random.Random(20260721 + sum(map(ord, corpus)))
    tasks = []
    for task_id in range(n):
        path, size = rng.choice(files)
        url = _hf_url(repo, path)
        task = {
            "corpus": corpus,
            "task_id": task_id,
            "kind": kind,
            "url": url,
            "tokenizer": tokenizer,
            "seed": rng.randrange(2**31),
            "source": path.split("/")[0],
        }
        if kind == "token_jsonl":
            task["window"] = 12 << 20
            task["fraction"] = rng.random()
        elif kind == "h5":
            task["rows"] = 512
        else:
            task["compression"] = "zst"
            task["target_tokens"] = 1_000_000
        tasks.append(task)
    return tasks


LLMJP_COMPONENT_TOKENS = {
    "ja/ja_cc": 762.8,
    "ja/ja_warp_pdf/e0": 237.3,
    "ja/ja_warp_html": 2.7,
    "ja/ja_wiki": 2.6,
    "ja/kaken": 1.8,
    "en/en_dolma/cc-head": 608.5,
    "en/en_dolma/c4": 181.6,
    "en/en_dolma/reddit": 83.1,
    "en/en_dolma/pes2o": 62.9,
    "en/en_dolma/gutenberg": 5.5,
    "en/en_dolma/wiki": 3.9,
    "en/en_wiki": 4.7,
    "code/code_stack": 114.1,
    "zh/zh_wiki": 0.8,
    "ko/ko_wiki": 0.3,
}


def _llmjp_tasks(n: int) -> list[dict]:
    checkout = Path("/tmp/llm-jp-corpus-v3")
    if not checkout.exists():
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3.git", str(checkout)],
            check=True,
            env=env,
        )

    grouped: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for path in checkout.rglob("*.jsonl.gz"):
        text = path.read_text()
        if not text.startswith("version https://git-lfs.github.com/spec/v1"):
            continue
        fields = dict(line.split(" ", 1) for line in text.splitlines()[1:])
        relative = path.relative_to(checkout).as_posix()
        for component in LLMJP_COMPONENT_TOKENS:
            if relative.startswith(component + "/"):
                grouped[component].append((relative, fields["oid"].split(":", 1)[1], int(fields["size"])))
                break

    missing = [component for component in LLMJP_COMPONENT_TOKENS if not grouped[component]]
    if missing:
        raise RuntimeError(f"LLM-JP components have no files: {missing}")

    rng = random.Random(20260721)
    components = list(LLMJP_COMPONENT_TOKENS)
    weights = [LLMJP_COMPONENT_TOKENS[name] for name in components]
    tasks = []
    for task_id in range(n):
        component = rng.choices(components, weights=weights, k=1)[0]
        path, oid, size = rng.choice(grouped[component])
        tasks.append(
            {
                "corpus": "LLM-JP Corpus v3",
                "task_id": task_id,
                "kind": "text",
                "url": "",
                "oid": oid,
                "size": size,
                "compression": "gz",
                "tokenizer": "llm-jp/llm-jp-3-1.8b",
                "target_tokens": 1_000_000,
                "source": component,
            }
        )
    return tasks


def _fit_alpha(counts: Counter[int]) -> tuple[float, float]:
    import numpy as np

    items = [(n, c) for n, c in counts.items() if 1 <= n <= MAX_N and c > 0]
    x = np.log(np.asarray([n for n, _ in items], dtype=float))
    y = np.log(np.asarray([c for _, c in items], dtype=float))
    alpha, intercept = np.polyfit(x, y, 1)
    fitted = intercept + alpha * x
    r2 = 1.0 - float(np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))
    return float(alpha), r2


def _save_results(corpus: str, rows: list[dict]) -> dict:
    import numpy as np

    rows = sorted(rows, key=lambda row: row["task_id"])
    out = OUT_ROOT / corpus.lower().replace(" ", "_").replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    aggregate: Counter[int] = Counter()
    for row in rows:
        aggregate.update({int(k): int(v) for k, v in row["counts"].items()})
    alpha, r2 = _fit_alpha(aggregate)

    convergence = {}
    for task_count in (8, 16, 32, 64, 128, 256):
        if task_count > len(rows):
            continue
        prefix: Counter[int] = Counter()
        for row in rows[:task_count]:
            prefix.update({int(k): int(v) for k, v in row["counts"].items()})
        prefix_alpha, prefix_r2 = _fit_alpha(prefix)
        convergence[str(task_count)] = {
            "alpha": prefix_alpha,
            "r2": prefix_r2,
            "support": sum(prefix.get(n, 0) > 0 for n in range(1, MAX_N + 1)) / MAX_N,
        }

    rng = np.random.default_rng(20260721)
    boot = []
    for _ in range(500):
        sample = rng.integers(0, len(rows), len(rows))
        counts: Counter[int] = Counter()
        for index in sample:
            counts.update({int(k): int(v) for k, v in rows[int(index)]["counts"].items()})
        boot.append(_fit_alpha(counts)[0])

    with (out / "counts_0_to_10000.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "count"])
        writer.writerows((n, aggregate.get(n, 0)) for n in range(MAX_N + 1))
    with (out / "tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "source", "tokens", "documents"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)

    summary = {
        "corpus": corpus,
        "method": "sample" if corpus != "RedPajama-Data-1T" else "full_pass",
        "tasks": len(rows),
        "sampled_tokens": sum(row["tokens"] for row in rows),
        "integer_matches": sum(aggregate.values()),
        "alpha": alpha,
        "r2": r2,
        "support": sum(aggregate.get(n, 0) > 0 for n in range(1, MAX_N + 1)) / MAX_N,
        "alpha_convergence": convergence,
        "alpha_bootstrap_mean": float(np.mean(boot)),
        "alpha_bootstrap_std": float(np.std(boot, ddof=1)),
        "alpha_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _redpajama_summary() -> dict:
    source = Path("results/redpajama_full_modal/0_to_10000/counts_0_to_10000.csv")
    counts: Counter[int] = Counter()
    with source.open() as handle:
        for row in csv.DictReader(handle):
            counts[int(row["number"])] = int(row["count"])
    alpha, r2 = _fit_alpha(counts)
    return {
        "corpus": "RedPajama-Data-1T",
        "method": "full_pass",
        "tasks": None,
        "sampled_tokens": None,
        "integer_matches": sum(counts.values()),
        "alpha": alpha,
        "r2": r2,
        "alpha_bootstrap_mean": None,
        "alpha_bootstrap_std": None,
        "alpha_ci95": None,
    }


MODEL_CORPUS = {
    "OpenLLaMA-3B": "RedPajama-Data-1T",
    "OpenLLaMA-7B": "RedPajama-Data-1T",
    "OpenLLaMA-13B": "RedPajama-Data-1T",
    "LLM360/Amber": "AmberDatasets",
    "LLM360/Crystal": "CrystalCoderDatasets",
    "LLM360/K2": "K2Datasets",
    "OLMo-3-1125-32B": "Dolma 3 5.5T",
    "BTLM-3B-8K": "SlimPajama-627B",
    "LLM-JP-3-1.8B": "LLM-JP Corpus v3",
    "LLM-JP-3-3.7B": "LLM-JP Corpus v3",
    "LLM-JP-3-7.2B": "LLM-JP Corpus v3",
    "LLM-JP-3-13B": "LLM-JP Corpus v3",
    "LLM-JP-3-172B": "LLM-JP Corpus v3",
}


@app.local_entrypoint()
def main(tasks: int = DEFAULT_TASKS):
    jobs = []
    jobs += _hf_tasks("AmberDatasets", "LLM360/AmberDatasets", ".jsonl", "token_jsonl", "LLM360/Amber", tasks)
    jobs += _hf_tasks("K2Datasets", "LLM360/K2Datasets", ".jsonl", "token_jsonl", "LLM360/K2", tasks)
    jobs += _hf_tasks("CrystalCoderDatasets", "LLM360/CrystalCoderDatasets", ".h5", "h5", "LLM360/Crystal", tasks)
    jobs += _hf_tasks("Dolma 3 5.5T", "allenai/dolma3_mix-5.5T-1125", ".jsonl.zst", "text", "allenai/Olmo-3-1125-32B", tasks)
    jobs += _hf_tasks(
        "SlimPajama-627B",
        "LLM360/CrystalCoderDatasets",
        ".h5",
        "h5",
        "LLM360/Crystal",
        tasks,
        prefixes=("phase1/SlimPajama/", "phase2/SlimPajama/"),
    )
    llmjp_jobs = _llmjp_tasks(tasks)

    grouped = defaultdict(list)
    for row in sample_task.map(jobs, order_outputs=False):
        grouped[row["corpus"]].append(row)
        print(f'[{row["corpus"]}] {len(grouped[row["corpus"]])}/{tasks}')
    for row in sample_llmjp_task.map(llmjp_jobs, order_outputs=False):
        grouped[row["corpus"]].append(row)
        print(f'[{row["corpus"]}] {len(grouped[row["corpus"]])}/{tasks}')

    summaries = [_redpajama_summary()]
    summaries += [_save_results(corpus, rows) for corpus, rows in sorted(grouped.items())]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "corpus_alpha_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    by_corpus = {row["corpus"]: row for row in summaries}
    with (OUT_ROOT / "model_dataset_alpha.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "corpus", "alpha", "r2", "method", "alpha_ci95"])
        for model, corpus in MODEL_CORPUS.items():
            row = by_corpus[corpus]
            writer.writerow([model, corpus, row["alpha"], row["r2"], row["method"], row["alpha_ci95"]])
    print(json.dumps(summaries, indent=2))
