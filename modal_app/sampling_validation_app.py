"""
Part 1: does a random sample give the full-pass alpha?  (CPU only)

    modal run modal_app/sampling_validation_app.py --corpus slimpajama --dry-run
    modal run modal_app/sampling_validation_app.py --corpus slimpajama
    modal run modal_app/sampling_validation_app.py --corpus llmjp --dry-run
    modal run modal_app/sampling_validation_app.py --corpus llmjp

SlimPajama (gmongaras/SlimPajama-627B_Reupload, parquet): units are row groups,
split into contiguous row slices of <= --max-unit-mb text; schemes
uniform_files (A) and size_proportional (B), 5 replicates, nested sizes
10M..1B tokens, 500-draw bootstrap. Row groups are read with HTTP range
requests; every slice of a row group that is read is counted and cached on
the `numberline-sampling-validation` Volume, so reruns are free.

LLM-JP v3 (.jsonl.gz on gitlab, not seekable): scheme B as two-stage
sampling -- files proportional to compressed size x multiplier (ja components
count twice, as in the full pass), then random 1 MB compressed-byte units
inside each file; bootstrap over files. Each drawn file is streamed from its
start up to the last unit needed, and every unit on the way is cached.

Counting reuses _count_text (corpus_alpha_app), the parquet / jsonl readers
of corpus_alpha_full_app, and its HF / gitlab download helpers.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import modal

from modal_app.corpus_alpha_app import (
    _count_text,
    _gitlab_lfs_url,
    _hf_headers,
    _hf_url,
    _request_with_retry,
)
from modal_app.corpus_alpha_full_app import (
    SLIMPAJAMA_REUPLOAD,
    _iter_parquet_texts,
    _jsonl_text,
    _llmjp_full_tasks,
    hf_manifest,
)

APP_NAME = "numberline-sampling-validation"
CACHE = Path("/sv_cache/v1")
LOCAL_OUT = Path("results/sampling_validation")
FULL_ROOT = Path("results/corpus_alpha_full")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "huggingface_hub>=0.28.0",
        "numpy>=2.0.0",
        "orjson>=3.10.0",
        "pyarrow>=19.0.0",
        "requests>=2.32.0",
        "zstandard>=0.23.0",
        "fsspec>=2025.3.0",
        "h5py>=3.12.0",
        "sentencepiece>=0.2.0",
        "transformers>=5.5.0,<6",
    )
    .add_local_python_source("modal_app", "src", "utils")
)
app = modal.App(APP_NAME, image=image)
cache = modal.Volume.from_name("numberline-sampling-validation", create_if_missing=True)
secret_name = os.environ.get("MODAL_HF_SECRET_NAME", "numberline-hf-token").strip()
secrets = [modal.Secret.from_name(secret_name)] if secret_name else []


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #

class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (for pyarrow footers / row groups)."""

    def __init__(self, url: str, size: int, headers: dict | None = None):
        self.url, self.size, self.headers, self.pos = url, size, headers or {}, 0
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size or len(buffer) == 0:
            return 0
        end = min(self.size, self.pos + len(buffer)) - 1
        response = _request_with_retry(
            "get", self.url, headers={**self.headers, "Range": f"bytes={self.pos}-{end}"}, timeout=300
        )
        response.raise_for_status()
        data = response.content
        buffer[: len(data)] = data
        self.pos += len(data)
        self.bytes_read += len(data)
        return len(data)


def _open_parquet(repo: str, path: str, size: int):
    import pyarrow.parquet as parquet

    raw = HttpRangeFile(_hf_url(repo, path), size, _hf_headers())
    return parquet.ParquetFile(io.BufferedReader(raw, buffer_size=8 << 20)), raw


class _CountingReader(io.RawIOBase):
    """Counts compressed bytes consumed by the gzip decoder."""

    def __init__(self, raw):
        self.raw, self.count = raw, 0

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.raw.read(len(buffer))
        buffer[: len(data)] = data
        self.count += len(data)
        return len(data)


def _cached(key: str, compute):
    path = CACHE / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
    cache.reload()
    if path.exists():
        return json.loads(path.read_text())
    payload = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    cache.commit()
    return payload


# --------------------------------------------------------------------------- #
# remote functions
# --------------------------------------------------------------------------- #

def footer_row_groups(source) -> list[list[int]]:
    """[num_rows, text uncompressed bytes, text compressed bytes] per row group."""
    meta = source.metadata
    groups = []
    for rg in range(meta.num_row_groups):
        group = meta.row_group(rg)
        text = next(
            group.column(j) for j in range(group.num_columns)
            if group.column(j).path_in_schema == "text"
        )
        groups.append([group.num_rows, text.total_uncompressed_size, text.total_compressed_size])
    return groups


def count_rowgroup_slices(source, rg: int, slices: list[list[int]]) -> list[dict]:
    """Count the given row slices (sorted, disjoint) of one row group with
    _count_text; rows outside them are decoded but not counted."""
    import bisect

    slices = sorted(slices)
    bounds = [hi for _lo, hi in slices]
    counts = [Counter() for _ in slices]
    text_bytes = [0] * len(slices)
    docs = [0] * len(slices)
    for row, text in enumerate(_iter_parquet_texts(source, row_groups=[rg])):
        i = bisect.bisect_right(bounds, row)
        if i == len(slices) or row < slices[i][0]:
            continue
        _count_text(text, counts[i])
        text_bytes[i] += len(text.encode("utf-8"))
        docs[i] += 1
    return [
        {"lo": lo, "hi": hi, "text_bytes": text_bytes[i], "docs": docs[i],
         "counts": {str(k): v for k, v in counts[i].items()}}
        for i, (lo, hi) in enumerate(slices)
    ]


def count_gz_stream(raw, size: int, unit_bytes: int, parts) -> dict:
    """Decompress a .jsonl.gz byte stream; a line belongs to unit
    floor(compressed bytes consumed / unit_bytes). Only units in `parts` are
    parsed and counted; reading stops after the last of them."""
    import gzip

    parts = set(parts)
    max_part = max(parts)
    counter = _CountingReader(raw)
    reader = gzip.GzipFile(fileobj=counter)  # gzip reads 128 KB at a time
    n_units = -(-size // unit_bytes)
    units: dict[int, dict] = {}
    try:
        for line in reader:
            part = min(counter.count // unit_bytes, n_units - 1)
            if part > max_part:
                break
            if part not in parts:
                continue
            unit = units.setdefault(part, {"text_bytes": 0, "docs": 0, "counts": Counter()})
            text = _jsonl_text(line)
            if text is not None:
                _count_text(text, unit["counts"])
                unit["text_bytes"] += len(text.encode("utf-8"))
                unit["docs"] += 1
    finally:
        reader.close()
    return {
        "compressed_bytes_read": counter.count,
        "units": {
            str(part): {**u, "counts": {str(k): v for k, v in u["counts"].items()}}
            for part, u in units.items()
        },
    }


@app.function(cpu=1, memory=2048, timeout=30 * 60, max_containers=64, retries=2, secrets=secrets)
def parquet_footer(repo: str, path: str, size: int) -> dict:
    source, raw = _open_parquet(repo, path, size)
    groups = footer_row_groups(source)
    return {"path": path, "size": size, "row_groups": groups, "footer_bytes_read": raw.bytes_read}


@app.function(timeout=60 * 60, volumes={"/sv_cache": cache}, secrets=secrets)
def parquet_manifest(repo: str) -> dict:
    def compute():
        files = hf_manifest.local(repo, ".parquet")
        out = {}
        for result in parquet_footer.starmap(
            [(repo, path, size) for path, size in files], order_outputs=False
        ):
            out[result["path"]] = {"size": result["size"], "row_groups": result["row_groups"]}
        return out

    return _cached(f"parquet_manifest::{repo}", compute)


@app.function(
    cpu=2, memory=8192, timeout=4 * 60 * 60, max_containers=100, retries=2,
    volumes={"/sv_cache": cache}, secrets=secrets,
)
def count_parquet_rowgroup(task: dict) -> dict:
    """Count every slice of one row group."""

    def compute():
        source, raw = _open_parquet(task["repo"], task["path"], task["size"])
        slices = count_rowgroup_slices(source, task["rg"], task["slices"])
        return {"bytes_downloaded": raw.bytes_read, "slices": slices}

    key = f"parquet_rg::{task['repo']}::{task['path']}::{task['rg']}::{task['slices']}"
    return {"task": task, **_cached(key, compute)}


@app.function(timeout=60 * 60, volumes={"/sv_cache": cache})
def llmjp_manifest() -> list[dict]:
    return _cached("llmjp_manifest", _llmjp_full_tasks)


@app.function(
    cpu=2, memory=4096, timeout=8 * 60 * 60, max_containers=32, retries=2,
    volumes={"/sv_cache": cache},
)
def count_gz_units(task: dict, unit_bytes: int, parts: list[int]) -> dict:
    """Stream one LLM-JP file from its start through its last needed unit (see count_gz_stream)."""

    def compute():
        url = _gitlab_lfs_url(task["oid"], task["size"])
        response = _request_with_retry("get", url, stream=True, timeout=300)
        response.raise_for_status()
        response.raw.decode_content = False
        try:
            return count_gz_stream(response.raw, task["size"], unit_bytes, parts)
        finally:
            response.close()

    key = f"gz_units::{task['source']}::{task['oid']}::{unit_bytes}::{sorted(parts)}"
    return {"source": task["source"], **_cached(key, compute)}


# --------------------------------------------------------------------------- #
# local entrypoint
# --------------------------------------------------------------------------- #

def _parse_sizes(sizes: str) -> dict[str, float]:
    from src.sampling_validation import SIZE_LABELS

    return {label: SIZE_LABELS[label] for label in [s.strip() for s in sizes.split(",") if s.strip()]}


def _run_slimpajama(args: dict) -> None:
    import numpy as np

    from src import sampling_validation as sv

    repo = SLIMPAJAMA_REUPLOAD[1]
    sizes = _parse_sizes(args["sizes"] or "10M,30M,100M,300M,1B")
    schemes = [s.strip() for s in args["schemes"].split(",") if s.strip()]
    out_dir = Path(args["out_dir"]) / "slimpajama"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = parquet_manifest.remote(repo)
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    files = {p: [(g[0], g[1]) for g in m["row_groups"]] for p, m in manifest.items()}
    rg_text = np.asarray([g[1] for m in manifest.values() for g in m["row_groups"]], dtype=float)
    rg_comp = {(p, i): g[2] for p, m in manifest.items() for i, g in enumerate(m["row_groups"])}
    units = sv.parquet_units(files, args["max_unit_mb"] << 20)
    print(f"[slimpajama] files={len(files)} row_groups={len(rg_text)} units={len(units)} "
          f"corpus_text={rg_text.sum()/1e12:.2f} TB (~{rg_text.sum()/4e9:.0f}B tokens)")
    print(f"[slimpajama] row-group text MB quantiles (0,25,50,75,100%): "
          f"{np.percentile(rg_text, [0, 25, 50, 75, 100]) / 2**20}")

    plans = sv.plan_runs(units, schemes, sizes, args["replicates"], base_seed=args["seed"])
    by_rg: dict[tuple[str, int], set[tuple[int, int]]] = {}
    for draws in plans.values():
        for u, _ in draws:
            by_rg.setdefault((u.file, u.part), set()).add((u.row_start, u.row_end))
    needed = sorted(by_rg)
    download = sum(rg_comp[k] for k in needed)
    unique_units = {u.key: u.size for draws in plans.values() for u, _ in draws}
    text = sum(unique_units.values())
    print(f"[slimpajama] draws={sum(len(d) for d in plans.values())} unique units={len(unique_units)} "
          f"row groups to read={len(needed)} download~{download/1e9:.1f} GB "
          f"text to count~{text/1e9:.1f} GB (~{text/50e6/3600:.1f} core-h at 50 MB/s)")
    (out_dir / "plan.json").write_text(json.dumps(
        {f"{s}::{r}": [[u.key, w] for u, w in d] for (s, r), d in plans.items()}))
    if args["dry_run"]:
        print("[slimpajama] --dry-run: stopping before counting")
        return

    tasks = [
        {"repo": repo, "path": f, "size": manifest[f]["size"], "rg": rg,
         "slices": [list(x) for x in sorted(by_rg[(f, rg)])]}
        for f, rg in needed
    ]
    unit_counts, unit_text = {}, {}
    done = 0
    for result in count_parquet_rowgroup.map(tasks, order_outputs=False):
        t = result["task"]
        for sl in result["slices"]:
            key = sv.Unit(t["path"], t["rg"], sl["lo"], sl["hi"], 0).key
            unit_counts[key] = sv.counts_vector(sl["counts"])
            unit_text[key] = float(sl["text_bytes"])
        done += 1
        if done % 50 == 0 or done == len(tasks):
            print(f"[slimpajama] row groups counted {done}/{len(tasks)}")

    full = sv.read_counts_csv(FULL_ROOT / "slimpajama-627b_reupload" / "counts_0_to_10000.csv")
    records = sv.evaluate_runs(plans, unit_counts, unit_text, full, sizes, n_boot=args["n_boot"])
    result = sv.write_outputs(out_dir, full, records, sizes, {
        "corpus": "SlimPajama-627B Reupload", "repo": repo, "max_unit_mb": args["max_unit_mb"],
        "replicates": args["replicates"], "seed": args["seed"],
    })
    print(json.dumps({"full": result["full"], "acceptance": result["acceptance"]}, indent=2))


def _run_llmjp(args: dict) -> None:
    from src import sampling_validation as sv

    sizes = _parse_sizes(args["sizes"] or "100M,1B")
    unit_bytes = args["unit_mb"] << 20
    out_dir = Path(args["out_dir"]) / "llmjp"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = llmjp_manifest.remote()
    by_source = {t["source"]: t for t in tasks}
    files = {t["source"]: (t["size"], t["multiplier"]) for t in tasks}
    print(f"[llmjp] files={len(files)} compressed={sum(s for s, _ in files.values())/1e12:.2f} TB")

    plans = {rep: sv.two_stage_plan(files, unit_bytes, args["files_per_run"], args["seed"] + rep)
             for rep in range(args["replicates"])}
    u_max = sv.units_per_file(max(sizes.values()), args["files_per_run"], unit_bytes,
                              args["text_per_compressed"])
    needed: dict[str, set[int]] = {}
    for draws in plans.values():
        for d in draws:
            needed.setdefault(d["file"], set()).update(d["order"][:u_max])
    download = sum((max(p) + 1) * unit_bytes for p in needed.values())
    counted = sum(len(p) for p in needed.values()) * unit_bytes * args["text_per_compressed"]
    print(f"[llmjp] files_per_run={args['files_per_run']} units/file at {max(sizes, key=sizes.get)}={u_max} "
          f"files to stream={len(needed)} compressed download~{download/1e9:.0f} GB "
          f"(gunzip ~{download / 100e6 / 3600:.1f} core-h) text to count~{counted/1e9:.0f} GB "
          f"(~{counted / 50e6 / 3600:.1f} core-h at 50 MB/s)")
    (out_dir / "plan.json").write_text(json.dumps(plans))
    if args["dry_run"]:
        print("[llmjp] --dry-run: stopping before counting")
        return

    unit_counts, unit_text = {}, {}
    done = 0
    calls = [(by_source[f], unit_bytes, sorted(p)) for f, p in sorted(needed.items())]
    for result in count_gz_units.starmap(calls, order_outputs=False):
        for part, u in result["units"].items():
            key = sv.gz_unit_key(result["source"], int(part))
            unit_counts[key] = sv.counts_vector(u["counts"])
            unit_text[key] = float(u["text_bytes"])
        done += 1
        print(f"[llmjp] files streamed {done}/{len(calls)}")
    # units past the end of a file's content (e.g. trailing gzip bytes) are empty
    for draws in plans.values():
        for d in draws:
            for part in d["order"][:u_max]:
                key = sv.gz_unit_key(d["file"], part)
                unit_counts.setdefault(key, sv.counts_vector({}))
                unit_text.setdefault(key, 0.0)

    full = sv.read_counts_csv(FULL_ROOT / "llm-jp_corpus_v3" / "counts_0_to_10000.csv")
    records = sv.evaluate_two_stage(plans, unit_counts, unit_text, full, sizes, unit_bytes,
                                    args["text_per_compressed"], n_boot=args["n_boot"])
    result = sv.write_outputs(out_dir, full, records, sizes, {
        "corpus": "LLM-JP Corpus v3", "unit_mb": args["unit_mb"],
        "files_per_run": args["files_per_run"], "replicates": args["replicates"], "seed": args["seed"],
    }, accept_scheme="size_proportional_two_stage")
    print(json.dumps({"full": result["full"], "acceptance": result["acceptance"]}, indent=2))


@app.local_entrypoint()
def main(
    corpus: str = "slimpajama",
    sizes: str = "",
    replicates: int = 5,
    schemes: str = "uniform_files,size_proportional",
    n_boot: int = 500,
    max_unit_mb: int = 16,
    unit_mb: int = 1,
    files_per_run: int = 32,
    text_per_compressed: float = 3.0,
    seed: int = 0,
    out_dir: str = str(LOCAL_OUT),
    dry_run: bool = False,
) -> None:
    args = dict(locals())
    if corpus == "slimpajama":
        _run_slimpajama(args)
    elif corpus == "llmjp":
        _run_llmjp(args)
    else:
        raise SystemExit("--corpus must be slimpajama or llmjp")
