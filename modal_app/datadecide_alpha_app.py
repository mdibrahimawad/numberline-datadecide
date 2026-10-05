"""
Part 2: number-frequency alpha of the DataDecide training data, sampled the
way training saw it (see docs/datadecide_sampling.md, src/datadecide_sampling.py).

    # validate on dolma1_7 only (convergence, split-half, exact training order)
    modal run modal_app/datadecide_alpha_app.py::validate --dry-run
    modal run modal_app/datadecide_alpha_app.py::validate

    # prepared, NOT yet run: all 25 recipes
    modal run modal_app/datadecide_alpha_app.py::sweep --dry-run
    modal run modal_app/datadecide_alpha_app.py::sweep --recipes c4,falcon --windows 1000

Windows of --window-tokens (default 262,144 tokens ~ 1 MB of text) are read
with HTTP range requests from allenai/DataDecide-data-recipes (raw uint16
memmaps), split on EOS 50279, decoded with allenai/DataDecide-<recipe>-1B's
tokenizer and counted with _count_text through _decode_and_count_native.
Window counts are cached on the `numberline-datadecide-alpha` Volume.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import modal

from modal_app.corpus_alpha_app import _hf_headers, _hf_url, _request_with_retry
from modal_app.corpus_alpha_full_app import _decode_and_count_native

APP_NAME = "numberline-datadecide-alpha"
CACHE = Path("/dd_cache/v1")
LOCAL_OUT = Path("results/corpus_alpha_datadecide")
MODEL_REVISION = "step69369-seed-default"
WINDOWS_PER_CALL = 100

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub>=0.28.0",
        "numpy>=2.0.0",
        "requests>=2.32.0",
        "tokenizers>=0.20.0",
        "orjson>=3.10.0",
    )
    .add_local_python_source("modal_app", "src", "utils")
    .add_local_dir(str(Path(__file__).resolve().parent.parent / "configs"), "/root/configs")
)
app = modal.App(APP_NAME, image=image)
cache = modal.Volume.from_name("numberline-datadecide-alpha", create_if_missing=True)
secret_name = os.environ.get("MODAL_HF_SECRET_NAME", "numberline-hf-token").strip()
secrets = [modal.Secret.from_name(secret_name)] if secret_name else []


def _cache_path(key: str) -> Path:
    return CACHE / f"{hashlib.sha1(key.encode()).hexdigest()}.json"


def _cached(key: str, compute):
    path = _cache_path(key)
    cache.reload()
    if path.exists():
        return json.loads(path.read_text())
    payload = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    cache.commit()
    return payload


def _tokenizer(model_repo: str):
    """Recipe model's tokenizer.json (checkpoint revision, falling back to main)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    cached = getattr(_tokenizer, "cache", {})
    if model_repo not in cached:
        token = os.environ.get("HF_TOKEN")
        try:
            path = hf_hub_download(model_repo, "tokenizer.json", revision=MODEL_REVISION, token=token)
        except Exception:
            path = hf_hub_download(model_repo, "tokenizer.json", token=token)
        cached[model_repo] = SimpleNamespace(backend_tokenizer=Tokenizer.from_file(path))
        _tokenizer.cache = cached
    return cached[model_repo]


def _read_tokens(path: str, start: int, length: int):
    import numpy as np

    from src.datadecide_sampling import HF_DATA_REPO, TOKEN_DTYPE

    item = np.dtype(TOKEN_DTYPE).itemsize
    response = _request_with_retry(
        "get", _hf_url(HF_DATA_REPO, path),
        headers={**_hf_headers(), "Range": f"bytes={start * item}-{(start + length) * item - 1}"},
        timeout=300,
    )
    response.raise_for_status()
    return np.frombuffer(response.content, dtype=TOKEN_DTYPE)


def count_window(tokenizer, tokens, window: dict, eos: int | None = None) -> dict:
    """Counts for both edge modes of one window (pure; used by tests)."""
    from src.datadecide_sampling import EDGE_MODES, EOS_TOKEN_ID, split_documents

    eos = EOS_TOKEN_ID if eos is None else eos
    drop_mode, keep_mode = EDGE_MODES
    whole = split_documents(tokens, drop_mode, window["at_file_start"], window["at_file_end"], eos=eos)
    keep = split_documents(tokens, keep_mode, window["at_file_start"], window["at_file_end"], eos=eos)
    # keep = [left edge?] + whole + [right edge?]: decode the whole documents once
    left = 0 if not whole or (keep and keep[0] == whole[0]) else 1
    edges = keep if not whole else keep[:left] + keep[left + len(whole):]
    counts: Counter[int] = Counter()
    decoded = _decode_and_count_native(tokenizer, whole, counts) if whole else 0
    out = {"tokens": int(len(tokens)),
           drop_mode: {"docs": len(whole), "decoded_tokens": decoded,
                       "counts": {str(k): v for k, v in counts.items()}}}
    if edges:
        decoded += _decode_and_count_native(tokenizer, edges, counts)
    out[keep_mode] = {"docs": len(keep), "decoded_tokens": decoded,
                      "counts": {str(k): v for k, v in counts.items()}}
    return out


@app.function(cpu=1, memory=2048, timeout=2 * 60 * 60, max_containers=64, retries=2,
              volumes={"/dd_cache": cache}, secrets=secrets)
def count_windows(model_repo: str, windows: list[dict]) -> list[dict]:
    """Count a batch of windows; the batch is cached as ONE Volume entry
    (per-window entries made every window pay a Volume commit)."""
    ident = hashlib.sha1(json.dumps(
        [[w["path"], w["start"], w["length"]] for w in windows]).encode()).hexdigest()

    def compute():
        tok = _tokenizer(model_repo)
        return [{**w, **count_window(tok, _read_tokens(w["path"], w["start"], w["length"]), w)}
                for w in windows]

    return _cached(f"windows::{model_repo}::{ident}", compute)


@app.function(timeout=30 * 60, secrets=secrets)
def npy_sizes(paths: list[str]) -> dict[str, int]:
    """Token count per unique path from HF metadata (bytes / 2); missing paths -> -1."""
    from huggingface_hub import HfApi

    from src.datadecide_sampling import HF_DATA_REPO

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    unique = sorted(set(paths))
    sizes = {p: -1 for p in unique}
    for i in range(0, len(unique), 200):
        for info in api.get_paths_info(HF_DATA_REPO, unique[i:i + 200], repo_type="dataset"):
            if getattr(info, "size", None) is not None:
                sizes[info.path] = int(info.size) // 2
    return sizes


@app.function(cpu=4, memory=32768, timeout=2 * 60 * 60, volumes={"/dd_cache": cache})
def exact_order_windows(file_tokens: list[int], n_chunks: int, seed: int, order_seed: int) -> dict:
    """Reproduce the 1B training chunk order for data seed `order_seed`, sample
    n_chunks of the chunks used in training, and n_chunks uniformly from all chunks."""
    import numpy as np

    from src.datadecide_sampling import (
        TRAIN_INSTANCES_1B,
        chunk_offsets,
        chunks_to_windows,
        training_chunk_indices,
    )

    def compute():
        order = training_chunk_indices(file_tokens, seed=order_seed)
        rng = np.random.default_rng(seed)
        trained = np.sort(rng.choice(order, size=n_chunks, replace=False))
        total = int(chunk_offsets(file_tokens)[-1])
        uniform = np.sort(rng.choice(total, size=n_chunks, replace=False))
        return {
            "total_chunks": total,
            "train_instances": TRAIN_INSTANCES_1B,
            "first_indices": order[:5].tolist(),
            "exact_order": chunks_to_windows(trained, file_tokens),
            "uniform_chunks": chunks_to_windows(uniform, file_tokens),
        }

    key = (f"exact_order::{hashlib.sha1(json.dumps(file_tokens).encode()).hexdigest()}"
           f"::{n_chunks}::{seed}::{order_seed}")
    return _cached(key, compute)


# --------------------------------------------------------------------------- #
# exact 100B training samples (one pass over the recipe, several seeds)
# --------------------------------------------------------------------------- #

EXACT_ROOT = CACHE / "exact"
# Modal list prices (USD): CPU per physical core-second, memory per GiB-second
USD_PER_CORE_S = 0.0000131
USD_PER_GIB_S = 0.00000222
EXACT_TASK_CPU = 1.0
EXACT_TASK_MEM_GIB = 3.0


@app.function(cpu=4, memory=65536, timeout=3 * 60 * 60, volumes={"/dd_cache": cache})
def build_membership(file_tokens: list[int], seeds: list[int]) -> dict:
    """Chunk -> membership code for every seed's training run, saved on the Volume."""
    import numpy as np

    from src.datadecide_sampling import TRAIN_INSTANCES_1B, decode_code, membership_codes

    ident = hashlib.sha1(json.dumps([file_tokens, seeds, TRAIN_INSTANCES_1B]).encode()).hexdigest()
    folder = EXACT_ROOT / ident
    meta_path = folder / "meta.json"
    cache.reload()
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    codes, base = membership_codes(file_tokens, seeds)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "codes.npy", codes)
    freq_all = np.bincount(codes)
    uniq = np.flatnonzero(freq_all)
    freq = freq_all[uniq]
    per_seed_chunks = [0] * len(seeds)
    union = 0
    for code, f in zip(uniq.tolist(), freq.tolist()):
        m = decode_code(code, base, len(seeds))
        for i, k in enumerate(m):
            per_seed_chunks[i] += k * f
        union += f if any(m) else 0
    meta = {"codes_path": str(folder / "codes.npy"), "base": base, "seeds": seeds,
            "n_chunks": int(len(codes)), "union_chunks": int(union),
            "per_seed_chunks": per_seed_chunks,
            "code_chunks": {str(c): int(f) for c, f in zip(uniq.tolist(), freq.tolist())}}
    meta_path.write_text(json.dumps(meta))
    cache.commit()
    return meta


@app.function(cpu=EXACT_TASK_CPU, memory=int(EXACT_TASK_MEM_GIB * 1024), timeout=60 * 60,
              max_containers=100, retries=2, volumes={"/dd_cache": cache}, secrets=secrets)
def count_exact_task(task: dict) -> dict:
    """Count one contiguous run of training chunks of one file, grouped by membership code.
    The result stays on the Volume; only timing comes back."""
    import time

    import numpy as np

    from src.datadecide_sampling import SEQUENCE_LENGTH, count_chunks_by_code

    def compute():
        start = time.time()
        n = task["chunk_hi"] - task["chunk_lo"]
        tokens = _read_tokens(task["path"], task["chunk_lo"] * SEQUENCE_LENGTH, n * SEQUENCE_LENGTH)
        codes = np.load(task["codes_path"], mmap_mode="r")[task["global_lo"]: task["global_lo"] + n]
        tok = _tokenizer(task["model_repo"])
        groups = count_chunks_by_code(
            lambda docs, counter: _decode_and_count_native(tok, docs, counter), tokens, np.asarray(codes),
            skip_codes=(0,) if task.get("skip_unused") else (),
        )
        return {
            "tokens": int(len(tokens)),
            "seconds": time.time() - start,
            "groups": {str(c): {**g, "counts": {str(k): v for k, v in g["counts"].items()}}
                       for c, g in groups.items()},
        }

    result = _cached(task["key"], compute)
    return {"key": task["key"], "tokens": result["tokens"], "seconds": result["seconds"]}


@app.function(cpu=1, memory=4096, timeout=30 * 60, max_containers=50, volumes={"/dd_cache": cache})
def reduce_exact(keys: list[str], base: int, n_seeds: int) -> dict:
    from modal_app.corpus_alpha_app import MAX_N
    from src.datadecide_sampling import combine_codes

    cache.reload()
    total = None
    for key in keys:
        groups = {int(c): g for c, g in json.loads(_cache_path(key).read_text())["groups"].items()}
        part = combine_codes(groups, base, n_seeds, MAX_N)
        if total is None:
            total = part
        else:
            for k in ("full", "per_seed", "seed_tokens"):
                total[k] = total[k] + part[k]
            total["full_tokens"] += part["full_tokens"]
    return {"full": total["full"].tolist(), "per_seed": total["per_seed"].tolist(),
            "seed_tokens": total["seed_tokens"].tolist(), "full_tokens": int(total["full_tokens"])}


def _exact_tasks(rec: dict, file_tokens: list[int], meta: dict, task_chunks: int,
                 skip_unused: bool = False) -> list[dict]:
    from src.datadecide_sampling import SEQUENCE_LENGTH, chunk_offsets

    offsets = chunk_offsets(file_tokens)
    tasks = []
    for f, (path, t) in enumerate(zip(rec["paths"], file_tokens)):
        n = t // SEQUENCE_LENGTH
        for lo in range(0, n, task_chunks):
            hi = min(n, lo + task_chunks)
            tasks.append({
                "model_repo": rec["model_repo"], "path": path, "file_index": f,
                "chunk_lo": lo, "chunk_hi": hi, "global_lo": int(offsets[f]) + lo,
                "codes_path": meta["codes_path"], "skip_unused": skip_unused,
                "key": f"exact::{meta['codes_path']}::{f}::{path}::{lo}::{hi}::{rec['model_repo']}"
                       + ("::skip_unused" if skip_unused else ""),
            })
    return tasks


def _usd(seconds: float) -> float:
    return seconds * (EXACT_TASK_CPU * USD_PER_CORE_S + EXACT_TASK_MEM_GIB * USD_PER_GIB_S)


@app.local_entrypoint()
def exact_samples(
    recipe: str = "c4",
    seeds: str = "2,4,5,6198,14",
    task_mtokens: int = 32,
    max_tasks: int = 0,
    assumed_tokens_per_s: float = 2e6,
    skip_unused: bool = False,
    dry_run: bool = False,
    out_dir: str = str(LOCAL_OUT),
) -> None:
    _run_exact(recipe, [int(x) for x in seeds.split(",") if x.strip()], task_mtokens, max_tasks,
               assumed_tokens_per_s, skip_unused, dry_run, out_dir)


def _run_exact(recipe: str, seed_list: list[int], task_mtokens: int, max_tasks: int,
               assumed_tokens_per_s: float, skip_unused: bool, dry_run: bool, out_dir: str):
    """Reproduce the 1B training data order of `recipe` for each seed, take the
    first 69,369 x 704 chunks (~100B tokens) of each, and count numbers in all
    of them -- plus the whole recipe -- in one pass. --max-tasks N is a pilot:
    count N spread-out slices, measure throughput, project the full cost.
    --skip-unused counts only chunks some seed trained on (no full-recipe
    alpha; for large recipes where a full pass is too expensive)."""
    import random

    import numpy as np

    from src.datadecide_sampling import SEQUENCE_LENGTH, TRAIN_INSTANCES_1B, load_data_map
    from src.sampling_validation import describe

    rec = load_data_map()["recipes"][recipe]
    file_tokens, _ = _recipe_tokens(rec)
    total = sum(file_tokens)
    n_chunks = sum(t // SEQUENCE_LENGTH for t in file_tokens)
    per_seed_frac = min(1.0, TRAIN_INSTANCES_1B / n_chunks)
    union_frac = 1 - (1 - per_seed_frac) ** len(seed_list)
    est_s = total * (union_frac if skip_unused else 1.0) / assumed_tokens_per_s
    print(f"[exact:{recipe}] files={len(file_tokens)} tokens={total/1e9:.1f}B chunks={n_chunks:,} "
          f"budget/seed={TRAIN_INSTANCES_1B * SEQUENCE_LENGTH / 1e9:.1f}B "
          f"({min(1.0, TRAIN_INSTANCES_1B / n_chunks):.0%} of chunks per seed)")
    print(f"[exact:{recipe}] {'sampled chunks only (~' + format(union_frac, '.0%') + ' of the recipe)' if skip_unused else 'full pass'} "
          f"at an assumed {assumed_tokens_per_s/1e6:.1f}M tokens/s/core: "
          f"~{est_s/3600:.0f} core-hours, ~${_usd(est_s):.2f} (+ membership build ~$0.10); "
          f"download {total * 2 / 1e9:.0f} GB")
    if dry_run:
        print("[exact] --dry-run: stopping before any counting (run a pilot with --max-tasks 20 next)")
        return None

    meta = build_membership.remote(file_tokens, seed_list)
    print(f"[exact:{recipe}] membership: base={meta['base']} union={meta['union_chunks']/meta['n_chunks']:.1%} "
          f"of chunks used by at least one seed; per seed {[c * SEQUENCE_LENGTH / 1e9 for c in meta['per_seed_chunks']]} B tokens")
    tasks = _exact_tasks(rec, file_tokens, meta, max(1, task_mtokens * 1_000_000 // SEQUENCE_LENGTH),
                         skip_unused)
    out = Path(out_dir) / f"exact_100b_{recipe}"
    out.mkdir(parents=True, exist_ok=True)

    if max_tasks:
        pilot = random.Random(0).sample(tasks, min(max_tasks, len(tasks)))
        results = list(count_exact_task.map(pilot, order_outputs=False))
        tok = sum(r["tokens"] for r in results)
        sec = sum(r["seconds"] for r in results)
        rate = tok / sec
        proj = total / rate  # tokens read per second already reflects --skip-unused
        report = {"pilot_tasks": len(results), "tokens": tok, "compute_seconds": sec,
                  "tokens_per_s_per_core": rate, "projected_core_hours": proj / 3600,
                  "projected_usd": _usd(proj) * 1.15, "pilot_usd": _usd(sec)}
        (out / "pilot.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print("[exact] pilot results are cached and reused by the full run")
        return None

    done = errors = 0
    seconds = 0.0
    for r in count_exact_task.map(tasks, order_outputs=False, return_exceptions=True,
                                  wrap_returned_exceptions=False):
        if isinstance(r, BaseException):
            errors += 1
            continue
        done += 1
        seconds += r["seconds"]
        if done % 200 == 0 or done == len(tasks):
            print(f"[exact:{recipe}] {done}/{len(tasks)} slices, ~${_usd(seconds):.2f} so far")
    if errors:
        raise SystemExit(f"[exact] {errors} slices failed; rerun the same command to resume (done work is cached)")

    keys = [t["key"] for t in tasks]
    parts = list(reduce_exact.starmap(
        [(keys[i:i + 100], meta["base"], len(seed_list)) for i in range(0, len(keys), 100)]))
    full = np.sum([p["full"] for p in parts], axis=0)
    per_seed = np.sum([p["per_seed"] for p in parts], axis=0)
    seed_tokens = np.sum([p["seed_tokens"] for p in parts], axis=0)

    from src.datadecide_sampling import pairwise_overlap

    # with --skip-unused the "full" vector is only the union of the sampled chunks
    ref_name = "union_of_samples" if skip_unused else "full_recipe"
    full_desc = describe(full)
    _write_counts_csv(out / f"counts_{ref_name}.csv", full)
    seeds_out = {}
    for i, sd in enumerate(seed_list):
        d = describe(per_seed[i])
        _write_counts_csv(out / f"counts_seed_{sd}.csv", per_seed[i])
        seeds_out[str(sd)] = {**d, "tokens": float(seed_tokens[i])}
        if not skip_unused:
            seeds_out[str(sd)]["diff_ols_vs_full"] = d["alpha_ols"] - full_desc["alpha_ols"]
            seeds_out[str(sd)]["diff_mle_vs_full"] = d["alpha_mle"] - full_desc["alpha_mle"]
    spread = {}
    for fit in ("ols", "mle"):
        v = np.asarray([seeds_out[str(sd)][f"alpha_{fit}"] for sd in seed_list])
        spread[fit] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                       "max_minus_min": float(v.max() - v.min()),
                       "max_abs_diff_vs_mean": float(np.abs(v - v.mean()).max())}
        if not skip_unused:
            spread[fit]["max_abs_diff_vs_full"] = float(np.abs(v - full_desc[f"alpha_{fit}"]).max())
    overlap = pairwise_overlap(meta["code_chunks"], meta["base"], seed_list)
    summary = {"recipe": recipe, "seeds": seed_list, "skip_unused": skip_unused,
               ref_name: {**full_desc, "tokens": int(sum(p["full_tokens"] for p in parts))},
               "per_seed": seeds_out, "spread_across_seeds": spread,
               "pairwise_chunk_overlap": overlap,
               "membership": {k: meta[k] for k in ("base", "n_chunks", "union_chunks", "per_seed_chunks")},
               "compute_seconds": seconds, "compute_usd": _usd(seconds)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({ref_name: summary[ref_name], "per_seed": seeds_out, "spread": spread,
                      "pairwise_chunk_overlap": overlap, "compute_usd": summary["compute_usd"]}, indent=2))
    print(f"[exact] wrote {out}")
    return summary


# Measured on the dolma1_7 run: decoding+counting ~2.04M tokens/s/core; reading and
# skipping unused chunks ~ $0.8 per trillion tokens of recipe.
READ_USD_PER_T = 0.8
DECODE_TOKENS_PER_S = 2.04e6
ALPHA_COLUMNS_EXACT = ("recipe", "seed", "alpha_ols", "r2", "alpha_mle", "integer_matches",
                       "support", "sample_tokens", "recipe_tokens", "source")


@app.local_entrypoint()
def exact_all(
    recipes: str = "",
    seed: int = 2,
    task_mtokens: int = 256,
    dry_run: bool = False,
    out_dir: str = str(LOCAL_OUT),
) -> None:
    """Alpha of the exact 100B training stream of ONE seed (default 2, the most
    likely seed of the released seed-default 1B models) for every recipe.
    Recipes that already have that seed in exact_100b_<recipe>/summary.json
    (e.g. the 5-seed c4 and dolma1_7 runs) are reused. Results are written to
    <out_dir>/alpha_seed<seed>.csv after every recipe; rerunning resumes."""
    from src.datadecide_sampling import SEQUENCE_LENGTH, TRAIN_INSTANCES_1B, load_data_map

    data_map = load_data_map()
    names = [r.strip() for r in recipes.split(",") if r.strip()] or list(data_map["recipes"])
    unknown = [r for r in names if r not in data_map["recipes"]]
    if unknown:
        raise SystemExit(f"unknown recipes: {unknown}")
    root = Path(out_dir)
    csv_path = root / f"alpha_seed{seed}.csv"

    def row_from(slug: str, summary: dict, source: str) -> dict | None:
        s = summary.get("per_seed", {}).get(str(seed))
        if s is None:
            return None
        return {"recipe": slug, "seed": seed, "alpha_ols": s["alpha_ols"], "r2": s["r2"],
                "alpha_mle": s["alpha_mle"], "integer_matches": int(s["integer_matches"]),
                "support": s["support"], "sample_tokens": int(s["tokens"]),
                "recipe_tokens": int(summary["membership"]["n_chunks"]) * SEQUENCE_LENGTH,
                "source": source}

    rows: dict[str, dict] = {}
    for slug in data_map["recipes"]:  # every recipe with results on disk goes into the CSV
        prev = root / f"exact_100b_{slug}" / "summary.json"
        row = row_from(slug, json.loads(prev.read_text()), "reused") if prev.exists() else None
        if row:
            rows[slug] = row
    to_run = [slug for slug in names if slug not in rows]
    print(f"[exact-all] seed {seed}: {len(rows)} recipes already done ({', '.join(rows) or '-'}), "
          f"{len(to_run)} to run")
    if not to_run:
        write_csv_now = True
    else:
        write_csv_now = False

    total_usd = 0.0
    for slug in to_run:
        file_tokens, _ = _recipe_tokens(data_map["recipes"][slug])
        tokens = sum(file_tokens)
        decoded = min(tokens, TRAIN_INSTANCES_1B * SEQUENCE_LENGTH)
        usd = _usd(decoded / DECODE_TOKENS_PER_S) + tokens / 1e12 * READ_USD_PER_T + 0.1
        total_usd += usd
        print(f"[exact-all] {slug:35s} {tokens/1e9:8.1f}B tokens  ~${usd:5.2f}")
    print(f"[exact-all] estimated total for {len(to_run)} recipes: ~${total_usd:.0f} "
          f"(compute only; downloads are free)")
    def write_csv():
        root.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ALPHA_COLUMNS_EXACT)
            w.writeheader()
            for slug in data_map["recipes"]:
                if slug in rows:
                    w.writerow(rows[slug])

    write_csv()
    if dry_run or write_csv_now:
        print(f"[exact-all] wrote {csv_path} ({len(rows)} recipes)"
              + ("; --dry-run: stopping before any counting" if dry_run else ""))
        return
    failed = []
    for i, slug in enumerate(to_run, 1):
        print(f"[exact-all] ({i}/{len(to_run)}) {slug}")
        try:
            summary = _run_exact(slug, [seed], task_mtokens, 0, DECODE_TOKENS_PER_S, True, False, out_dir)
        except (Exception, SystemExit) as exc:  # keep going; a rerun resumes from the cache
            print(f"[exact-all] {slug} FAILED: {exc}")
            failed.append(slug)
            continue
        rows[slug] = row_from(slug, summary, "seed-only run")
        write_csv()
    print(f"[exact-all] wrote {csv_path} ({len(rows)}/{len(names)} recipes)")
    if failed:
        print(f"[exact-all] failed: {failed} -- rerun the same command to finish them")


# --------------------------------------------------------------------------- #
# local helpers
# --------------------------------------------------------------------------- #

def _recipe_tokens(recipe: dict) -> tuple[list[int], dict[str, int]]:
    sizes = npy_sizes.remote(recipe["paths"])
    missing = [p for p, n in sizes.items() if n < 0]
    if missing:
        raise SystemExit(f"{len(missing)} paths missing from the HF dataset, e.g. {missing[:3]}")
    return [sizes[p] for p in recipe["paths"]], sizes


def _mixture(recipe: dict, file_tokens: list[int], data_map_groups: dict[str, str]) -> dict:
    by_source: Counter = Counter()
    for p, t in zip(recipe["paths"], file_tokens):
        by_source[data_map_groups.get(p, p.split("/")[1])] += t
    total = sum(by_source.values())
    return {k: {"tokens": v, "share": v / total} for k, v in by_source.most_common()}


def _count(model_repo: str, windows: list[dict]) -> list[dict]:
    batches = [windows[i:i + WINDOWS_PER_CALL] for i in range(0, len(windows), WINDOWS_PER_CALL)]
    results = []
    for batch in count_windows.starmap([(model_repo, b) for b in batches], order_outputs=True):
        results.extend(batch)
    return results


def _vectors(results: list[dict], mode: str):
    import numpy as np

    from src.sampling_validation import counts_vector

    return np.stack([counts_vector(r[mode]["counts"]) for r in results])


def _write_counts_csv(path: Path, vec) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["number", "count"])
        w.writerows((n, int(round(vec[n]))) for n in range(len(vec)))


def _recipe_result(slug: str, recipe: dict, results: list[dict],
                   n_boot: int, out_dir: Path, file_tokens: list[int]) -> dict:
    from src.sampling_validation import bootstrap_alpha, describe

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"recipe": slug, "model_repo": recipe["model_repo"], "olmo_mix_name": recipe["olmo_mix_name"],
               "windows": len(results), "corpus_tokens": int(sum(file_tokens))}
    for mode in ("drop_partial_docs", "keep_partial"):
        mat = _vectors(results, mode)
        vec = mat.sum(axis=0)
        summary[mode] = {
            **describe(vec), **bootstrap_alpha(mat, n_boot=n_boot),
            "sampled_tokens": int(sum(r[mode]["decoded_tokens"] for r in results)),
            "documents": int(sum(r[mode]["docs"] for r in results)),
        }
        _write_counts_csv(out_dir / f"counts_0_to_10000_{mode}.csv", vec)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "windows.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("drop_partial_docs", "keep_partial")} for r in results]))
    return summary


ALPHA_COLUMNS = ("recipe", "alpha_ols", "r2", "alpha_mle", "ci95_low", "ci95_high",
                 "sampled_tokens", "integer_matches", "support")


def _summary_row(summary: dict, mode: str = "drop_partial_docs") -> dict:
    s = summary[mode]
    return {"recipe": summary["recipe"], "alpha_ols": s["alpha_ols"], "r2": s["r2"],
            "alpha_mle": s["alpha_mle"], "ci95_low": s["alpha_ols_ci95"][0],
            "ci95_high": s["alpha_ols_ci95"][1], "sampled_tokens": s["sampled_tokens"],
            "integer_matches": int(s["integer_matches"]), "support": s["support"]}


def _groups_by_path() -> dict[str, str]:
    from src.datadecide_sampling import load_data_map

    out = {}
    for r in load_data_map()["recipes"].values():
        for p in r["paths"]:
            out.setdefault(p, p.split("/")[1] + "/" + p.split("/")[2])
    return out


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def validate(
    recipe: str = "dolma1_7",
    windows: int = 1000,
    window_tokens: int = 1 << 18,
    n_boot: int = 500,
    exact_order: bool = True,
    exact_chunks: int = 1000,
    order_seeds: str = "2,6198",
    seed: int = 0,
    dry_run: bool = False,
    out_dir: str = str(LOCAL_OUT),
) -> None:

    from src.datadecide_sampling import convergence, load_data_map, plan_windows, split_half
    from src.sampling_validation import bootstrap_alpha, describe

    data_map = load_data_map()
    rec = data_map["recipes"][recipe]
    file_tokens, _ = _recipe_tokens(rec)
    total = sum(file_tokens)
    print(f"[validate:{recipe}] files={len(file_tokens)} tokens={total/1e9:.1f}B "
          f"chunks={sum(t // 2048 for t in file_tokens):,} windows={windows} x {window_tokens} tokens "
          f"(~{windows * window_tokens / 1e6:.0f}M tokens, {windows * window_tokens * 2 / 1e9:.2f} GB download)")
    out = Path(out_dir) / f"validation_{recipe}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mixture.json").write_text(json.dumps(_mixture(rec, file_tokens, _groups_by_path()), indent=2))
    if dry_run:
        print("[validate] --dry-run: stopping before counting")
        return

    plan = plan_windows(file_tokens, windows, window_tokens, seed)
    for w in plan:
        w["path"] = rec["paths"][w["file_index"]]
    results = _count(rec["model_repo"], plan)
    summary = _recipe_result(recipe, rec, results, n_boot, out, file_tokens)

    report = {"recipe": recipe, "window_tokens": window_tokens, "seed": seed, "summary": summary}
    for mode in ("drop_partial_docs", "keep_partial"):
        mat = _vectors(results, mode)
        report[f"convergence_{mode}"] = convergence(mat, [50, 100, 200, 500, 1000], n_boot, seed)
        report[f"split_half_{mode}"] = split_half(mat, n_splits=100, seed=seed, n_boot=n_boot)
    if exact_order:
        # The 1B "default seed" runs are named <mix>-1B-5xC-2; ladder.py appends
        # -<seed> only when seed != 6198, so the data seed was most likely 2.
        report["exact_order"] = {}
        for order_seed in [int(x) for x in order_seeds.split(",") if x.strip()]:
            eo = exact_order_windows.remote(file_tokens, exact_chunks, seed, order_seed)
            entry = {"total_chunks": eo["total_chunks"], "train_instances": eo["train_instances"],
                     "first_training_indices": eo["first_indices"]}
            for name in ("exact_order", "uniform_chunks"):
                wins = eo[name]
                for w in wins:
                    w["path"] = rec["paths"][w["file_index"]]
                res = _count(rec["model_repo"], wins)
                mat = _vectors(res, "keep_partial")
                entry[name] = {
                    **describe(mat.sum(axis=0)), **bootstrap_alpha(mat, n_boot=n_boot, seed=seed),
                    "chunks": len(wins),
                    "sampled_tokens": int(sum(r["keep_partial"]["decoded_tokens"] for r in res)),
                }
            report["exact_order"][f"data_seed_{order_seed}"] = entry
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    with open(out / "convergence.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["edge_mode", "windows", "alpha_ols", "ols_ci_low", "ols_ci_high",
                    "alpha_mle", "mle_ci_low", "mle_ci_high", "support", "integer_matches"])
        for mode in ("drop_partial_docs", "keep_partial"):
            for r in report[f"convergence_{mode}"]:
                w.writerow([mode, r["windows"], r["alpha_ols"], *r["alpha_ols_ci95"],
                            r["alpha_mle"], *r["alpha_mle_ci95"], r["support"], r["integer_matches"]])
    print(json.dumps({k: v for k, v in report.items() if k != "summary"}, indent=2)[:6000])
    print(f"[validate] wrote {out}")


@app.local_entrypoint()
def sweep(
    recipes: str = "",
    windows: int = 1000,
    window_tokens: int = 1 << 18,
    n_boot: int = 500,
    seed: int = 0,
    dry_run: bool = False,
    out_dir: str = str(LOCAL_OUT),
) -> None:
    from src.datadecide_sampling import load_data_map, plan_windows

    data_map = load_data_map()
    names = [r.strip() for r in recipes.split(",") if r.strip()] or list(data_map["recipes"])
    unknown = [r for r in names if r not in data_map["recipes"]]
    if unknown:
        raise SystemExit(f"unknown recipes: {unknown}")
    groups = _groups_by_path()
    root = Path(out_dir)

    def write_summary_csv() -> int:
        rows = []
        for slug in data_map["recipes"]:
            path = root / slug / "summary.json"
            if path.exists():
                rows.append(_summary_row(json.loads(path.read_text())))
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "alpha_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ALPHA_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        return len(rows)

    # One recipe at a time; a failure is logged and skipped, and alpha_summary.csv
    # is rewritten after every recipe, so an unattended run keeps whatever finished.
    # Rerunning the same command reuses every counted window batch from the Volume.
    failed = []
    for i, slug in enumerate(names, 1):
        rec = data_map["recipes"][slug]
        try:
            file_tokens, _ = _recipe_tokens(rec)
            (root / slug).mkdir(parents=True, exist_ok=True)
            (root / slug / "mixture.json").write_text(json.dumps(_mixture(rec, file_tokens, groups), indent=2))
            print(f"[sweep] ({i}/{len(names)}) {slug}: files={len(file_tokens)} "
                  f"tokens={sum(file_tokens)/1e9:.1f}B download={windows * window_tokens * 2 / 1e9:.2f} GB")
            if dry_run:
                continue
            plan = plan_windows(file_tokens, windows, window_tokens, seed)
            for w in plan:
                w["path"] = rec["paths"][w["file_index"]]
            results = _count(rec["model_repo"], plan)
            summary = _recipe_result(slug, rec, results, n_boot, root / slug, file_tokens)
        except (Exception, SystemExit) as exc:
            print(f"[sweep] {slug} FAILED: {type(exc).__name__}: {exc}")
            failed.append(slug)
            continue
        s = summary["drop_partial_docs"]
        print(f"[sweep] {slug}: alpha_ols={s['alpha_ols']:.4f} alpha_mle={s['alpha_mle']:.4f} "
              f"-> {write_summary_csv()} recipes in alpha_summary.csv")
    if dry_run:
        print("[sweep] --dry-run: recipes verified on the Hub; stopping before counting")
        return
    print(f"[sweep] wrote {root / 'alpha_summary.csv'} ({write_summary_csv()} recipes)")
    if failed:
        print(f"[sweep] failed: {failed} -- rerun the same command to retry them (finished work is reused)")
