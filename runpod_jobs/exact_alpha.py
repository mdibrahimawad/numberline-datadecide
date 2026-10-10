"""Exact 100B-token training-stream alpha on ONE machine (RunPod CPU pod), no Modal.

Same counting as `modal run modal_app/datadecide_alpha_app.py::exact_all`
(seed-only, --skip-unused): rebuild the recipe's training order for the data
seed, read every .npy file with HTTP range requests from
allenai/DataDecide-data-recipes, decode only the chunks that seed trained on,
count numbers. Outputs are byte-for-byte the same files the Modal run writes:

    results/corpus_alpha_datadecide/exact_100b_<recipe>/{summary.json,counts_seed_<seed>.csv,
                                                         counts_union_of_samples.csv}
    results/corpus_alpha_datadecide/alpha_seed<seed>.csv

Every finished slice is cached under --work-dir (put it on the pod's volume
disk, /workspace), so a crash, a stop or a spot interruption loses at most the
slices in flight: rerun the same command and it resumes.

    python -m runpod_jobs.exact_alpha --dry-run          # sizes, est. hours and USD
    python -m runpod_jobs.exact_alpha --pilot 64         # measure real speed, project the total
    python -m runpod_jobs.exact_alpha                    # every recipe still missing
    python -m runpod_jobs.exact_alpha --recipes falcon,falcon-and-cc

Several pods can split the list (--recipes on each); results merge by copying
the exact_100b_<recipe>/ folders into one results tree and running
`--dry-run` once, which rewrites alpha_seed<seed>.csv from every summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from modal_app.corpus_alpha_app import MAX_N, _hf_headers, _hf_url
from modal_app.corpus_alpha_full_app import _decode_and_count_native
from modal_app.datadecide_alpha_app import (
    ALPHA_COLUMNS_EXACT,
    DECODE_TOKENS_PER_S,
    _tokenizer,
    _write_counts_csv,
)
from src.datadecide_sampling import (
    EOS_TOKEN_ID,
    HF_DATA_REPO,
    SEQUENCE_LENGTH,
    TOKEN_DTYPE,
    TRAIN_INSTANCES_1B,
    chunk_offsets,
    combine_codes,
    count_chunks_by_code,
    load_data_map,
    membership_codes,
    pairwise_overlap,
)
from runpod_jobs.pod import stop_this_pod, terminate_this_pod
from src.sampling_validation import describe

OUT = Path("results/corpus_alpha_datadecide")


# --------------------------------------------------------------------------- #
# machine + download helpers
# --------------------------------------------------------------------------- #

def usable_vcpus() -> int:
    """vCPUs this container may use. Pods are containers with a CPU quota: the
    affinity mask / os.cpu_count() can report the whole host (e.g. 128) while
    the pod only gets its 32, so read the cgroup quota first."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if os.environ.get("RUNPOD_CPU_COUNT", "").isdigit():  # set by RunPod; trust the smallest signal
        n = min(n, max(1, int(os.environ["RUNPOD_CPU_COUNT"])))
    try:  # cgroup v2: "max 100000" or "3200000 100000"
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        try:  # cgroup v1
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                n = min(n, max(1, quota // period))
        except (OSError, ValueError):
            pass
    return n


def usable_ram_gb() -> float:
    """Memory limit of this container (cgroup), else the machine's RAM."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(path).read_text().strip()
            if v != "max" and int(v) < 1 << 50:
                return int(v) / 1e9
        except (OSError, ValueError):
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 16.0


WORKER_GB = 0.4        # measured peak of one worker with a 64 MB slice (~0.3 GB) + margin
BASE_GB = 1.5          # main process, page cache headroom


def membership_gb(n_chunks: int) -> float:
    """Peak RAM of building one recipe's training order: a uint32 index per chunk
    (shuffled in place) + a uint8 multiplicity per chunk, + 20 % margin."""
    return n_chunks * 5 * 1.2 / 1e9


def default_workers(vcpus: int, max_chunks: int) -> int:
    """3 workers per vCPU: a worker waiting on its download uses no CPU, so
    oversubscribing keeps every vCPU decoding while many downloads run; capped
    so the workers + one membership build (the largest recipe's) fit in RAM."""
    free = usable_ram_gb() - BASE_GB - membership_gb(max_chunks)
    if free < 2 * WORKER_GB:
        raise SystemExit(f"[plan] {usable_ram_gb():.0f} GB RAM is too little for the largest recipe "
                         f"(its training order alone needs {membership_gb(max_chunks):.1f} GB): "
                         "use a pod with more RAM for it, or leave it out of --recipes on this pod")
    return max(2, min(3 * vcpus, int(free / WORKER_GB)))


_RESOLVED: dict[str, tuple[str, float]] = {}
RESOLVE_TTL_S = 15 * 60


def resolve_url(path: str) -> str:
    """The CDN URL behind huggingface.co/.../resolve/main/<path>, cached for 15
    minutes. Range requests then go straight to the CDN, so Hugging Face's API
    rate limit (resolver calls per 5 minutes) is hit once per file, not once
    per slice."""
    import requests

    hit = _RESOLVED.get(path)
    if hit and time.time() - hit[1] < RESOLVE_TTL_S:
        return hit[0]
    url = _hf_url(HF_DATA_REPO, path)
    for attempt in range(10):
        try:
            r = requests.head(url, headers=_hf_headers(), allow_redirects=True, timeout=60)
            if r.status_code == 200:
                _RESOLVED[path] = (r.url, time.time())
                return r.url
            if r.status_code not in (429, 500, 502, 503, 504):
                break
            wait_s = float(r.headers.get("Retry-After", 0) or 0)
        except requests.RequestException:
            wait_s = 0
        time.sleep(max(wait_s, min(60, 2 ** attempt)) + random.random())
    return url  # fall back to the resolver URL; read_range follows its redirect


def task_url(task: dict) -> str:
    """Where a slice's file lives: a plain HTTP server when the recipe has a `base_url`
    (e.g. Paloma's files on olmo-data.org), otherwise the DataDecide HF dataset."""
    if task.get("base_url"):
        return task["base_url"].rstrip("/") + "/" + task["path"]
    return resolve_url(task["path"])


CONNECTIONS_PER_SLICE = int(os.environ.get("DD_CONNECTIONS", "8"))
MIN_PART_BYTES = 4 << 20


def _fetch(task: dict, where: dict, lo: int, hi: int) -> bytes:
    """Bytes [lo, hi] of one file with retries; `where["url"]` is shared by the parts of
    one slice so an expired CDN link is replaced once for all of them."""
    import requests

    last = None
    for attempt in range(12):
        url = where["url"]
        headers = {"Range": f"bytes={lo}-{hi}", **(_hf_headers() if "huggingface.co" in url else {})}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=300) as r:
                if r.status_code == 206:
                    data = r.content
                    if len(data) == hi - lo + 1:
                        return data
                    last = f"short read {len(data)} of {hi - lo + 1} bytes"
                elif r.status_code == 200:
                    raise RuntimeError(f"server ignored the Range header for {task['path']}")
                elif r.status_code in (401, 403, 404, 410) and "huggingface.co" not in url \
                        and not task.get("base_url"):
                    last = f"HTTP {r.status_code} (expired CDN link)"
                    where["url"] = _hf_url(HF_DATA_REPO, task["path"])  # back through the resolver
                    continue
                else:
                    last = f"HTTP {r.status_code}"
                    wait_s = float(r.headers.get("Retry-After", 0) or 0)
                    time.sleep(max(wait_s, min(60, 2 ** attempt)) + random.random())
                    continue
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(min(60, 2 ** attempt) + random.random())
    raise RuntimeError(f"giving up on {task['path']} bytes [{lo}, {hi}]: {last}")


def read_range(task: dict, start: int, length: int) -> np.ndarray:
    """`length` tokens from `start` of one .npy file, fetched over several parallel
    connections (CDNs often cap the speed of one connection). Retries with backoff
    (honours Retry-After), re-resolves an expired CDN URL, and refuses a server that
    ignores the Range header instead of downloading a whole multi-GB file."""
    item = np.dtype(TOKEN_DTYPE).itemsize
    lo, hi = start * item, (start + length) * item - 1
    where = {"url": task.get("url") or task_url(task)}
    n = max(1, min(CONNECTIONS_PER_SLICE, (hi - lo + 1) // MIN_PART_BYTES))
    if n == 1:
        return np.frombuffer(_fetch(task, where, lo, hi), dtype=TOKEN_DTYPE)
    from concurrent.futures import ThreadPoolExecutor

    cuts = [lo + (hi - lo + 1) * i // n for i in range(n)] + [hi + 1]
    buf = bytearray(hi - lo + 1)
    with ThreadPoolExecutor(n) as ex:
        parts = ex.map(lambda i: _fetch(task, where, cuts[i], cuts[i + 1] - 1), range(n))
        for i, data in enumerate(parts):
            buf[cuts[i] - lo:cuts[i + 1] - lo] = data
    return np.frombuffer(buf, dtype=TOKEN_DTYPE)  # no copy


def atomic_write(path: Path, text: str) -> None:
    """Write via a unique temp file + rename: a crash, or another pod writing the
    same file on a shared network volume, never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{random.getrandbits(32):08x}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# results off the pod: a private Hugging Face dataset repo
# --------------------------------------------------------------------------- #

def hf_results_repo(name: str) -> str:
    """<your HF user>/<name>, created private if missing. Fails fast (before any
    paid work) if HF_TOKEN cannot write."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo = name if "/" in name else f"{api.whoami()['name']}/{name}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    pod = os.environ.get("RUNPOD_POD_ID", "local")
    api.upload_file(path_or_fileobj=f"started {time.ctime()}\n".encode(),
                    path_in_repo=f"_pods/{pod}.txt", repo_id=repo, repo_type="dataset",
                    commit_message=f"pod {pod} started")
    return repo


def hf_upload_recipe(repo: str, out_root: Path, recipe: str, seed: int) -> None:
    """Upload exact_100b_<recipe>/ (summary + count CSVs, a few hundred KB). Retries:
    several pods may commit to the same repo at once."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    folder = out_root / f"exact_100b_{recipe}"
    for attempt in range(8):
        try:
            api.upload_folder(folder_path=str(folder), path_in_repo=folder.name, repo_id=repo,
                              repo_type="dataset", allow_patterns=["summary.json", "*.csv"],
                              commit_message=f"{recipe} seed {seed}")
            print(f"[{recipe}] uploaded to huggingface.co/datasets/{repo}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 -- network / concurrent-commit errors: retry
            print(f"[{recipe}] upload attempt {attempt + 1} failed: {exc}", flush=True)
            time.sleep(min(300, 15 * 2 ** attempt))
    raise RuntimeError(f"could not upload {recipe} to {repo}")


def hf_verify(repo: str, recipes: list[str], seed: int) -> list[str]:
    """Recipes whose summary.json with this seed is NOT in the repo."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = set(api.list_repo_files(repo, repo_type="dataset"))
    missing = []
    for r in recipes:
        f = f"exact_100b_{r}/summary.json"
        if f not in files or f"exact_100b_{r}/counts_seed_{seed}.csv" not in files:
            missing.append(r)
            continue
        got = json.loads(Path(hf_hub_download(repo, f, repo_type="dataset",
                                              token=os.environ.get("HF_TOKEN"))).read_text())
        if str(seed) not in got.get("per_seed", {}):
            missing.append(r)
    return missing


POD_ID = os.environ.get("RUNPOD_POD_ID") or f"local-{os.getpid()}"
CLAIM_SETTLE_S = 20


def hf_take(repo: str, recipe: str, seed: int) -> bool:
    """Work sharing between pods that run the same command: True if this pod should
    count `recipe` now. False if it is already uploaded, or another pod claimed it.
    A claim is the file _claims/<recipe>/<pod>.txt; if two pods claim at the same
    moment, the smaller pod id keeps it (both see the same files after a pause)."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    def files() -> set[str]:
        return set(api.list_repo_files(repo, repo_type="dataset"))

    try:
        have = files()
        if f"exact_100b_{recipe}/summary.json" in have and f"exact_100b_{recipe}/counts_seed_{seed}.csv" in have:
            print(f"[{recipe}] already uploaded by another pod: skipping", flush=True)
            return False
        prefix = f"_claims/{recipe}/"
        others = sorted(f[len(prefix):-4] for f in have if f.startswith(prefix) and f.endswith(".txt"))
        if any(o != POD_ID for o in others):
            print(f"[{recipe}] being counted by pod {others[0]}: skipping", flush=True)
            return False
        api.upload_file(path_or_fileobj=f"{POD_ID} {time.ctime()}\n".encode(),
                        path_in_repo=f"{prefix}{POD_ID}.txt", repo_id=repo, repo_type="dataset",
                        commit_message=f"{POD_ID} claims {recipe}")
        time.sleep(CLAIM_SETTLE_S)
        claimants = sorted(f[len(prefix):-4] for f in files() if f.startswith(prefix) and f.endswith(".txt"))
        if claimants and claimants[0] != POD_ID:
            print(f"[{recipe}] pod {claimants[0]} claimed it at the same time: leaving it to them", flush=True)
            return False
        print(f"[{recipe}] claimed by this pod ({POD_ID})", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001 -- Hub unreachable: count it anyway (worst case: twice)
        print(f"[{recipe}] could not check claims ({exc}); counting it here", flush=True)
        return True


def hf_release(repo: str, recipes: list[str], seed: int) -> None:
    """Remove this pod's claims on recipes it did not finish, so a later run takes them."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        have = set(api.list_repo_files(repo, repo_type="dataset"))
    except Exception:  # noqa: BLE001
        return
    for r in recipes:
        claim = f"_claims/{r}/{POD_ID}.txt"
        if claim in have and f"exact_100b_{r}/counts_seed_{seed}.csv" not in have:
            try:
                api.delete_file(claim, repo_id=repo, repo_type="dataset",
                                commit_message=f"{POD_ID} releases {r}")
                print(f"[{r}] claim released for a later run", flush=True)
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# per-recipe preparation (main process)
# --------------------------------------------------------------------------- #

def _http_sizes(urls: list[str]) -> dict[str, int]:
    """Content-Length of plain HTTP files (HEAD requests in parallel, with retries)."""
    import requests
    from concurrent.futures import ThreadPoolExecutor

    def head(url: str) -> tuple[str, int]:
        for attempt in range(8):
            try:
                r = requests.head(url, allow_redirects=True, timeout=60)
                if r.status_code == 200 and r.headers.get("Content-Length"):
                    return url, int(r.headers["Content-Length"])
                if r.status_code in (403, 404, 410):
                    raise SystemExit(f"{url}: HTTP {r.status_code}")
            except requests.RequestException:
                pass
            time.sleep(min(60, 2 ** attempt) + random.random())
        raise SystemExit(f"{url}: no size after retries")

    with ThreadPoolExecutor(32) as ex:
        return dict(ex.map(head, urls))


def file_sizes(paths: list[str], work: Path, base_url: str | None = None) -> list[int]:
    """Tokens per file (bytes / 2), cached in work/sizes.json: from HF metadata, or from
    HEAD requests when the recipe's files sit on a plain HTTP server (`base_url`)."""
    if base_url:
        cache_path = work / "sizes_http.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        urls = [base_url.rstrip("/") + "/" + p for p in paths]
        need = sorted(set(urls) - set(cache))
        if need:
            cache.update({u: n // 2 for u, n in _http_sizes(need).items()})
            atomic_write(cache_path, json.dumps(cache))
        return [cache[u] for u in urls]
    cache_path = work / "sizes.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    need = sorted(set(paths) - set(cache))
    if need:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        for i in range(0, len(need), 200):
            for info in api.get_paths_info(HF_DATA_REPO, need[i:i + 200], repo_type="dataset"):
                if getattr(info, "size", None) is not None:
                    cache[info.path] = int(info.size) // 2
        if cache_path.exists():  # keep entries another pod added meanwhile
            cache = {**json.loads(cache_path.read_text()), **cache}
        atomic_write(cache_path, json.dumps(cache))
    missing = [p for p in paths if p not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} paths missing from {HF_DATA_REPO}, e.g. {missing[:3]}")
    return [cache[p] for p in paths]


def membership(recipe: str, file_tokens: list[int], seed: int, work: Path,
               n_instances: int = TRAIN_INSTANCES_1B) -> tuple[Path, int]:
    """uint8 multiplicity per global chunk for `seed` (0 = not trained on), cached."""
    folder = work / recipe
    path, meta = folder / f"codes_seed{seed}.npy", folder / f"codes_seed{seed}.json"
    if not meta.exists():
        t = time.time()
        codes, base = membership_codes(file_tokens, [seed], n_instances=n_instances)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(path, codes)
        atomic_write(meta, json.dumps({"base": base, "n_chunks": int(len(codes))}))
        print(f"[{recipe}] membership built in {time.time() - t:.0f}s ({len(codes):,} chunks)", flush=True)
    return path, json.loads(meta.read_text())["base"]


def make_tasks(recipe: str, rec: dict, file_tokens: list[int], codes_path: Path,
               task_chunks: int, work: Path) -> list[dict]:
    offsets = chunk_offsets(file_tokens)
    codes = np.load(codes_path, mmap_mode="r")
    tasks = []
    for f, (path, t) in enumerate(zip(rec["paths"], file_tokens)):
        n = t // SEQUENCE_LENGTH
        for lo in range(0, n, task_chunks):
            hi = min(n, lo + task_chunks)
            g = int(offsets[f]) + lo
            if not codes[g:g + hi - lo].any():  # no trained chunk here: skip the download
                continue
            tasks.append({"recipe": recipe, "model_repo": rec["model_repo"], "path": path,
                          "base_url": rec.get("base_url"), "eos": int(rec.get("eos_token_id", EOS_TOKEN_ID)),
                          "chunk_lo": lo, "chunk_hi": hi, "global_lo": g,
                          "codes_path": str(codes_path),
                          "cache": str(work / recipe / "slices" / f"{f:05d}_{lo:09d}.json")})
    return tasks


# --------------------------------------------------------------------------- #
# one slice (worker process)
# --------------------------------------------------------------------------- #

def count_slice(task: dict) -> dict:
    cache = Path(task["cache"])
    if cache.exists():
        out = json.loads(cache.read_text())
        out["cached"] = True
        return out
    t0 = time.time()
    n = task["chunk_hi"] - task["chunk_lo"]
    tokens = read_range(task, task["chunk_lo"] * SEQUENCE_LENGTH, n * SEQUENCE_LENGTH)
    t1 = time.time()
    codes = np.asarray(np.load(task["codes_path"], mmap_mode="r")[task["global_lo"]: task["global_lo"] + n])
    tok = _tokenizer(task["model_repo"])
    groups = count_chunks_by_code(lambda docs, c: _decode_and_count_native(tok, docs, c),
                                  tokens, codes, eos=task.get("eos", EOS_TOKEN_ID), skip_codes=(0,))
    out = {"bytes": int(tokens.nbytes), "download_s": t1 - t0, "count_s": time.time() - t1,
           "groups": {str(c): {"chunks": g["chunks"], "decoded_tokens": g["decoded_tokens"],
                               "counts": {str(k): v for k, v in g["counts"].items() if k <= MAX_N}}
                      for c, g in groups.items()}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(cache)  # atomic: an interrupted write never leaves a half file
    return out


# --------------------------------------------------------------------------- #
# reduce + outputs (main process)
# --------------------------------------------------------------------------- #

def finish_recipe(recipe: str, tasks: list[dict], seed: int, base: int, n_chunks: int,
                  out_root: Path) -> dict:
    groups: dict[int, dict] = {}
    for t in tasks:
        for code, g in json.loads(Path(t["cache"]).read_text())["groups"].items():
            agg = groups.setdefault(int(code), {"counts": Counter(), "chunks": 0})
            agg["chunks"] += g["chunks"]
            agg["counts"].update({int(k): v for k, v in g["counts"].items()})
    total = combine_codes(groups, base, 1, MAX_N)
    out = out_root / f"exact_100b_{recipe}"
    out.mkdir(parents=True, exist_ok=True)
    _write_counts_csv(out / "counts_union_of_samples.csv", total["full"])
    _write_counts_csv(out / f"counts_seed_{seed}.csv", total["per_seed"][0])
    d = describe(total["per_seed"][0])
    seeds_out = {str(seed): {**d, "tokens": float(total["seed_tokens"][0])}}
    code_chunks = {str(c): g["chunks"] for c, g in groups.items()}
    summary = {"recipe": recipe, "seeds": [seed], "skip_unused": True,
               "union_of_samples": {**describe(total["full"]), "tokens": int(total["full_tokens"])},
               "per_seed": seeds_out,
               "pairwise_chunk_overlap": pairwise_overlap(code_chunks, base, [seed]),
               "membership": {"base": base, "n_chunks": n_chunks,
                              "union_chunks": int(sum(code_chunks.values())),
                              "per_seed_chunks": [int(sum(int(c) * k for c, k in code_chunks.items()))]},
               "runner": "runpod_jobs.exact_alpha"}
    atomic_write(out / "summary.json", json.dumps(summary, indent=2) + "\n")
    s = seeds_out[str(seed)]
    print(f"[{recipe}] DONE alpha_ols={s['alpha_ols']:.4f} alpha_mle={s['alpha_mle']:.5f} "
          f"numbers={s['integer_matches']:.3e} tokens={s['tokens'] / 1e9:.2f}B -> {out}", flush=True)
    return summary


def write_alpha_csv(out_root: Path, seed: int, data_map: dict) -> int:
    rows = []
    for slug in data_map["recipes"]:
        p = out_root / f"exact_100b_{slug}" / "summary.json"
        if not p.exists():
            continue
        summary = json.loads(p.read_text())
        s = summary.get("per_seed", {}).get(str(seed))
        if s is None:
            continue
        source = "seed-only run" if summary.get("skip_unused") and summary["seeds"] == [seed] else "reused"
        rows.append({"recipe": slug, "seed": seed, "alpha_ols": s["alpha_ols"], "r2": s["r2"],
                     "alpha_mle": s["alpha_mle"], "integer_matches": int(s["integer_matches"]),
                     "support": s["support"], "sample_tokens": int(s["tokens"]),
                     "recipe_tokens": int(summary["membership"]["n_chunks"]) * SEQUENCE_LENGTH,
                     "source": source})
    path = out_root / f"alpha_seed{seed}.csv"
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=ALPHA_COLUMNS_EXACT, lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
    atomic_write(path, buf.getvalue())
    return len(rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--recipes", default="", help="comma list; default: every recipe without a result")
    p.add_argument("--data-map", default="",
                   help="recipe -> files map (default configs/datadecide_data_map.json); a recipe may set "
                        "base_url (plain HTTP files), n_instances (training chunks), eos_token_id, model_repo")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--workers", type=int, default=0,
                   help="parallel slices; default 3 x vCPUs (capped by RAM) so downloads overlap decoding")
    p.add_argument("--task-mtokens", type=int, default=32, help="tokens per slice, millions (64 MB each)")
    p.add_argument("--work-dir", default="/workspace/dd_work")
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--usd-per-hour", type=float, default=0.0, help="pod price, for the running cost line")
    p.add_argument("--tokens-per-s-per-vcpu", type=float, default=DECODE_TOKENS_PER_S / 2,
                   help="decode speed assumed by --dry-run (Modal measured 2.04M per 2-vCPU core)")
    p.add_argument("--download-gb-per-s", type=float, default=0.5, help="assumed by --dry-run")
    p.add_argument("--pilot", type=int, default=0, help="count N random slices, measure, project, stop")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-stop-pod", action="store_true",
                   help="do NOT stop the RunPod pod when the full run ends (default: stop it)")
    p.add_argument("--stall-minutes", type=float, default=30,
                   help="abort (and stop the pod) if no slice finishes for this long")
    p.add_argument("--upload-hf", default="", metavar="REPO",
                   help="upload each finished recipe to this PRIVATE Hugging Face dataset "
                        "(e.g. numberline-alpha-results; HF_TOKEN needs write access)")
    p.add_argument("--delete-pod-when-done", action="store_true",
                   help="with --upload-hf: once every recipe is uploaded AND verified, delete this pod "
                        "(all billing ends). If anything is missing the pod is kept, never deleted")
    p.add_argument("--max-hours", type=float, default=0,
                   help="hard limit: after this many hours the pod deletes itself whatever happens "
                        "(finished recipes are already uploaded; unfinished ones are released)")
    p.add_argument("--retries", type=int, default=3,
                   help="rerun failed slices up to this many times before giving up")
    args = p.parse_args(argv)

    data_map = load_data_map(Path(args.data_map)) if args.data_map else load_data_map()
    out_root, work = Path(args.out_dir), Path(args.work_dir)
    vcpus = usable_vcpus()
    if args.recipes:
        names = [r.strip() for r in args.recipes.split(",") if r.strip()]
    else:
        names = [r for r in data_map["recipes"]
                 if not (out_root / f"exact_100b_{r}" / "summary.json").exists()
                 or str(args.seed) not in json.loads(
                     (out_root / f"exact_100b_{r}" / "summary.json").read_text()).get("per_seed", {})]
    unknown = [r for r in names if r not in data_map["recipes"]]
    if unknown:
        raise SystemExit(f"unknown recipes: {unknown}")

    sizes = {r: file_sizes(data_map["recipes"][r]["paths"], work, data_map["recipes"][r].get("base_url"))
             for r in names}
    chunks = {r: int(chunk_offsets(sizes[r])[-1]) for r in names}
    if args.upload_hf and not args.workers:
        # shared work: this pod only takes recipes it can build while keeping at least
        # 2 x vCPU workers; bigger ones are left to pods with more RAM (same command)
        room = usable_ram_gb() - BASE_GB - 2 * vcpus * WORKER_GB
        too_big = [r for r in names if membership_gb(chunks[r]) > room]
        if too_big and len(too_big) < len(names):
            print(f"[plan] leaving {too_big} to pods with more RAM (this one has {usable_ram_gb():.0f} GB)")
            names = [r for r in names if r not in too_big]
    max_chunks = max((chunks[r] for r in names), default=0)
    workers = args.workers or default_workers(vcpus, max_chunks)
    if args.upload_hf:  # several pods share the list: biggest first balances their finish times
        names.sort(key=lambda r: -sum(sizes[r]))
    else:               # one pod: smallest first, so most recipes are done early
        names.sort(key=lambda r: sum(sizes[r]))
    tot_h = 0.0
    print(f"[plan] {vcpus} vCPUs, {usable_ram_gb():.0f} GB RAM, {workers} workers, seed {args.seed}, "
          f"{len(names)} recipes")
    for r in names:
        tok = sum(sizes[r])
        decode_tok = min(tok, n_instances(data_map["recipes"][r]) * SEQUENCE_LENGTH)
        cpu_h = decode_tok / (args.tokens_per_s_per_vcpu * vcpus) / 3600
        net_h = tok * 2 / (args.download_gb_per_s * 1e9) / 3600
        h = max(cpu_h, net_h)
        tot_h += h
        print(f"  {r:34s} {tok / 1e9:8.1f}B tokens  download {tok * 2 / 1e12:5.2f} TB  "
              f"~{h:4.1f} h ({'download' if net_h > cpu_h else 'CPU'}-bound)")
    usd = f", ~${tot_h * args.usd_per_hour:.0f}" if args.usd_per_hour else ""
    print(f"[plan] total ~{tot_h:.1f} h on this pod{usd} (assumed {args.tokens_per_s_per_vcpu / 1e6:.2f}M "
          f"tok/s/vCPU, {args.download_gb_per_s} GB/s; run --pilot to measure the real numbers)")
    if args.dry_run:
        print(f"[plan] wrote {out_root / f'alpha_seed{args.seed}.csv'} "
              f"({write_alpha_csv(out_root, args.seed, data_map)} recipes)")
        return 0

    task_chunks = max(1, args.task_mtokens * 1_000_000 // SEQUENCE_LENGTH)
    if args.pilot:
        return _pilot(args, names, sizes, data_map, work, workers, vcpus, task_chunks)
    if args.delete_pod_when_done and not args.upload_hf:
        raise SystemExit("--delete-pod-when-done needs --upload-hf (results must be off the pod first)")
    if args.upload_hf:  # check write access now, not after hours of work
        args.upload_hf = hf_results_repo(args.upload_hf)
        print(f"[plan] results go to the private dataset huggingface.co/datasets/{args.upload_hf}")
    if args.max_hours:
        import threading

        def deadline():
            print(f"[limit] {args.max_hours} h reached: deleting the pod so it cannot bill any longer",
                  flush=True)
            if args.upload_hf:
                hf_release(args.upload_hf, list(getattr(args, "mine", [])), args.seed)
            terminate_this_pod(f"--max-hours {args.max_hours} reached")
            os._exit(3)

        timer = threading.Timer(args.max_hours * 3600, deadline)
        timer.daemon = True
        timer.start()
        print(f"[plan] hard limit: this pod deletes itself after {args.max_hours} h at the latest")
    status = "crashed"
    try:
        def done(r: str) -> bool:
            f = out_root / f"exact_100b_{r}" / "summary.json"
            return f.exists() and str(args.seed) in json.loads(f.read_text()).get("per_seed", {})

        for attempt in range(1 + max(0, args.retries)):
            todo = [r for r in names if not done(r)] if attempt else names
            if not todo:
                status = "finished"
                break
            if attempt:
                print(f"[retry] attempt {attempt + 1}: {todo} (finished slices are reused)", flush=True)
                time.sleep(60)
            try:
                if _run(args, todo, sizes, data_map, work, out_root, workers, task_chunks) == 0:
                    status = "finished"
                    break
                status = "finished with failed slices"
            except Exception as exc:  # noqa: BLE001 -- e.g. a stall or a crashed worker: retry
                status = f"crashed ({type(exc).__name__}: {exc})"
                print(f"[retry] run {status}", flush=True)
    except KeyboardInterrupt:
        status = "interrupted by you (Ctrl-C)"
        raise
    finally:
        if status != "interrupted by you (Ctrl-C)":
            _end_of_run(args, names, status)
    return 0 if status == "finished" else 1


def _end_of_run(args, names: list[str], status: str) -> None:
    """Never lose results, never bill for nothing: delete the pod only when every
    recipe is verified on Hugging Face; otherwise stop it if that is safe (a
    persistent volume), else leave it running and say why."""
    if args.upload_hf:
        names = getattr(args, "mine", names)  # what THIS pod counted (others verify their own)
        try:
            missing = hf_verify(args.upload_hf, names, args.seed)
        except Exception as exc:  # noqa: BLE001
            missing = list(names)
            print(f"[end] could not verify the uploads: {exc}", flush=True)
        if not missing:
            print(f"[end] all {len(names)} recipes verified on huggingface.co/datasets/{args.upload_hf}",
                  flush=True)
            if args.delete_pod_when_done:
                terminate_this_pod(f"run {status}, every result is safely uploaded")
                return
        else:
            print(f"[end] NOT uploaded yet: {missing} -- the pod is kept so nothing is lost", flush=True)
            if args.max_hours:
                print("[end] it deletes itself when the --max-hours limit is reached", flush=True)
                while True:  # the --max-hours timer ends this process and the pod
                    time.sleep(3600)
    if not args.no_stop_pod:
        stop_this_pod(f"run {status}")


def _pilot(args, names, sizes, data_map, work, workers, vcpus, task_chunks) -> int:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        r = names[-1]  # the largest recipe: the one that matters for the projection
        rec = data_map["recipes"][r]
        codes_path, _ = membership(r, sizes[r], args.seed, work, n_instances(rec))
        tasks = make_tasks(r, rec, sizes[r], codes_path, task_chunks, work)
        sample = random.Random(0).sample(tasks, min(args.pilot, len(tasks)))
        for t in sample:
            t["url"] = task_url(t)
        t0 = time.time()
        res = [x for x in pool.map(count_slice, sample) if not x.get("cached")]
        wall = time.time() - t0
    if not res:
        raise SystemExit("[pilot] every sampled slice was already cached; pass a larger --pilot")
    byts = sum(x["bytes"] for x in res)
    dec = sum(g["decoded_tokens"] for x in res for g in x["groups"].values())
    count_s = sum(x["count_s"] for x in res)
    per_vcpu = dec / count_s
    gbps = byts / wall / 1e9
    cpu_busy = count_s / (wall * vcpus)
    print(f"[pilot] {len(res)} slices of {r} in {wall:.0f}s with {workers} workers: "
          f"download {gbps:.2f} GB/s, decode {per_vcpu / 1e6:.2f}M tok/s per busy worker, "
          f"vCPUs busy {min(cpu_busy, 1):.0%}")
    if cpu_busy < 0.7:
        print(f"[pilot] the CPUs were idle {1 - min(cpu_busy, 1):.0%} of the time -> this pod is "
              "download-bound: more pods with fewer vCPUs each give the same speed for less money")
    print("[pilot] projection with the measured numbers:\n"
          f"  python -m runpod_jobs.exact_alpha --recipes {','.join(names)} --dry-run "
          f"--download-gb-per-s {gbps:.2f} --tokens-per-s-per-vcpu {per_vcpu:.3g}"
          + (f" --usd-per-hour {args.usd_per_hour}" if args.usd_per_hour else ""))
    return 0


def n_instances(rec: dict) -> int:
    """Training sequences (chunks) the recipe's model consumed (DataDecide 1B default)."""
    return int(rec.get("n_instances", TRAIN_INSTANCES_1B))


def _prepare(recipe: str, file_tokens: list[int], seed: int, work: str,
             n_inst: int = TRAIN_INSTANCES_1B) -> tuple[str, int]:
    """Membership of one recipe, in a helper process (see _run)."""
    path, base = membership(recipe, file_tokens, seed, Path(work), n_inst)
    return str(path), base


def _run(args, names, sizes, data_map, work, out_root, workers, task_chunks) -> int:
    start = time.time()
    stats = {"bytes": 0, "done_bytes": 0, "skipped_bytes": 0, "decoded": 0, "slices": 0,
             "last": time.time()}

    def log(remaining: int):
        el = time.time() - stats.get("t_first_submit", start)  # excludes the training-order build
        gbps = stats["bytes"] / max(el, 1e-9) / 1e9
        left = max(0, total_bytes - stats["skipped_bytes"] - stats["done_bytes"])
        eta = left / (gbps * 1e9) / 3600 if gbps > 0 else float("nan")
        spent = (time.time() - start) / 3600 * args.usd_per_hour
        cost = (f"  ${spent:.2f} so far, ~${eta * args.usd_per_hour:.2f} to go at this speed"
                if args.usd_per_hour else "")
        print(f"[run] {el / 60:6.1f} min  {stats['slices']} slices done  {gbps:.2f} GB/s  "
              f"{stats['decoded'] / max(el, 1e-9) / 1e6:.1f}M tok/s  "
              f"{stats['done_bytes'] / max(1, total_bytes - stats['skipped_bytes']):.1%} of this pod's data  "
              f"ETA {eta:.1f} h{cost}", flush=True)

    total_bytes = sum(sum(sizes[r]) * 2 for r in names)
    # the training order of every recipe is built ahead, one at a time, in a
    # separate process, so the counting workers never wait for it
    prep_pool = ProcessPoolExecutor(max_workers=1)
    prep: dict = {}

    def ensure_prep(r: str):
        if r not in prep:
            prep[r] = prep_pool.submit(_prepare, r, sizes[r], args.seed, str(work),
                                       n_instances(data_map["recipes"][r]))
        return prep[r]

    if not args.upload_hf:  # alone: build every training order ahead, in order
        for r in names:
            ensure_prep(r)
    mine: list[str] = args.__dict__.setdefault("mine", [])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending: dict = {}
        recipe_left: dict[str, int] = {}
        recipe_meta: dict[str, tuple] = {}
        failed: list[str] = []

        def drain(block: bool):
            if not pending:
                return
            done, _ = wait(list(pending), timeout=60 if block else 0, return_when=FIRST_COMPLETED)
            if not done and time.time() - stats["last"] > args.stall_minutes * 60:
                raise RuntimeError(f"no slice finished for {args.stall_minutes:.0f} min (network down? "
                                   "Hugging Face outage?) -- aborting so the pod does not bill for nothing")
            for fut in done:
                stats["last"] = time.time()
                r, nbytes = pending.pop(fut)
                stats["done_bytes"] += nbytes
                try:
                    x = fut.result()
                except Exception as exc:  # retry once later by rerunning; keep going now
                    print(f"[{r}] slice failed: {type(exc).__name__}: {exc}", flush=True)
                    if r not in failed:
                        failed.append(r)
                    recipe_left[r] -= 1
                else:
                    stats["slices"] += 1
                    if not x.get("cached"):
                        stats["bytes"] += x["bytes"]
                        stats["decoded"] += sum(g["decoded_tokens"] for g in x["groups"].values())
                    recipe_left[r] -= 1
                if recipe_left[r] == 0 and r not in failed:
                    tasks, base, n_chunks = recipe_meta[r]
                    finish_recipe(r, tasks, args.seed, base, n_chunks, out_root)
                    write_alpha_csv(out_root, args.seed, data_map)
                    if args.upload_hf:
                        try:
                            hf_upload_recipe(args.upload_hf, out_root, r, args.seed)
                            import shutil  # uploaded: free its cache (GBs) on the container disk

                            shutil.rmtree(work / r, ignore_errors=True)
                        except Exception as exc:  # noqa: BLE001 -- verified again at the end
                            print(f"[{r}] {exc}", flush=True)
                if stats["slices"] % 100 == 0:
                    log(len(pending))

        for i, r in enumerate(names):
            if args.upload_hf and not hf_take(args.upload_hf, r, args.seed):
                stats["skipped_bytes"] += sum(sizes[r]) * 2
                continue
            if r not in mine:
                mine.append(r)
            ensure_prep(r)
            if i + 1 < len(names):  # build the next candidate's order while this one counts
                ensure_prep(names[i + 1])
            rec = data_map["recipes"][r]
            while not prep[r].done():  # keep finishing slices while this recipe's order is built
                drain(block=True) if pending else prep[r].result()
            codes_path, base = prep[r].result()
            codes_path = Path(codes_path)
            tasks = make_tasks(r, rec, sizes[r], codes_path, task_chunks, work)
            n_chunks = int(chunk_offsets(sizes[r])[-1])
            recipe_meta[r] = (tasks, base, n_chunks)
            recipe_left[r] = len(tasks)
            print(f"[{r}] {len(tasks)} slices queued", flush=True)
            for t in tasks:
                while len(pending) >= workers * 4:  # bounded queue keeps memory flat
                    drain(block=True)
                t["url"] = task_url(t)  # fresh CDN link (HF) or the plain HTTP URL
                stats.setdefault("t_first_submit", time.time())
                pending[pool.submit(count_slice, t)] = (
                    r, (t["chunk_hi"] - t["chunk_lo"]) * SEQUENCE_LENGTH * 2)
            drain(block=False)
        while pending:
            drain(block=True)
        log(0)
    prep_pool.shutdown(cancel_futures=True)
    n = write_alpha_csv(out_root, args.seed, data_map)
    print(f"[done] {out_root / f'alpha_seed{args.seed}.csv'} has {n} recipes")
    if failed:
        print(f"[done] some slices failed in {failed}: rerun the same command (finished slices are cached)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
