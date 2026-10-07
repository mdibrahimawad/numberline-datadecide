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

from modal_app.corpus_alpha_app import MAX_N
from modal_app.corpus_alpha_full_app import _decode_and_count_native
from modal_app.datadecide_alpha_app import (
    ALPHA_COLUMNS_EXACT,
    DECODE_TOKENS_PER_S,
    _read_tokens,
    _tokenizer,
    _write_counts_csv,
)
from src.datadecide_sampling import (
    HF_DATA_REPO,
    SEQUENCE_LENGTH,
    TRAIN_INSTANCES_1B,
    chunk_offsets,
    combine_codes,
    count_chunks_by_code,
    load_data_map,
    membership_codes,
    pairwise_overlap,
)
from src.sampling_validation import describe

OUT = Path("results/corpus_alpha_datadecide")


# --------------------------------------------------------------------------- #
# per-recipe preparation (main process)
# --------------------------------------------------------------------------- #

def file_sizes(paths: list[str], work: Path) -> list[int]:
    """Tokens per file (bytes / 2) from HF metadata, cached in work/sizes.json."""
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
        work.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    missing = [p for p in paths if p not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} paths missing from {HF_DATA_REPO}, e.g. {missing[:3]}")
    return [cache[p] for p in paths]


def membership(recipe: str, file_tokens: list[int], seed: int, work: Path) -> tuple[Path, int]:
    """uint8 multiplicity per global chunk for `seed` (0 = not trained on), cached."""
    folder = work / recipe
    path, meta = folder / f"codes_seed{seed}.npy", folder / f"codes_seed{seed}.json"
    if not meta.exists():
        t = time.time()
        codes, base = membership_codes(file_tokens, [seed])
        folder.mkdir(parents=True, exist_ok=True)
        np.save(path, codes)
        meta.write_text(json.dumps({"base": base, "n_chunks": int(len(codes))}))
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
    tokens = _read_tokens(task["path"], task["chunk_lo"] * SEQUENCE_LENGTH, n * SEQUENCE_LENGTH)
    t1 = time.time()
    codes = np.asarray(np.load(task["codes_path"], mmap_mode="r")[task["global_lo"]: task["global_lo"] + n])
    tok = _tokenizer(task["model_repo"])
    groups = count_chunks_by_code(lambda docs, c: _decode_and_count_native(tok, docs, c),
                                  tokens, codes, skip_codes=(0,))
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
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ALPHA_COLUMNS_EXACT)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--recipes", default="", help="comma list; default: every recipe without a result")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--workers", type=int, default=0,
                   help="parallel slices; default 1.5 x vCPUs so downloads overlap decoding")
    p.add_argument("--task-mtokens", type=int, default=32, help="tokens per slice, millions (64 MB each)")
    p.add_argument("--work-dir", default="/workspace/dd_work")
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--usd-per-hour", type=float, default=0.0, help="pod price, for the running cost line")
    p.add_argument("--tokens-per-s-per-vcpu", type=float, default=DECODE_TOKENS_PER_S / 2,
                   help="decode speed assumed by --dry-run (Modal measured 2.04M per 2-vCPU core)")
    p.add_argument("--download-gb-per-s", type=float, default=0.5, help="assumed by --dry-run")
    p.add_argument("--pilot", type=int, default=0, help="count N random slices, measure, project, stop")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    data_map = load_data_map()
    out_root, work = Path(args.out_dir), Path(args.work_dir)
    vcpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    workers = args.workers or int(vcpus * 1.5)
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

    sizes = {r: file_sizes(data_map["recipes"][r]["paths"], work) for r in names}
    names.sort(key=lambda r: sum(sizes[r]))  # smallest first: most recipes done early
    tot_h = 0.0
    print(f"[plan] {vcpus} vCPUs, {workers} workers, seed {args.seed}, {len(names)} recipes")
    for r in names:
        tok = sum(sizes[r])
        decode_tok = min(tok, TRAIN_INSTANCES_1B * SEQUENCE_LENGTH)
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
    start = time.time()
    stats = {"bytes": 0, "download_s": 0.0, "count_s": 0.0, "decoded": 0, "slices": 0}

    def log(remaining: int):
        el = time.time() - start
        gbps = stats["bytes"] / max(el, 1e-9) / 1e9
        cost = f"  ${el / 3600 * args.usd_per_hour:.2f} so far" if args.usd_per_hour else ""
        print(f"[run] {el / 60:6.1f} min  {stats['slices']} slices done, {remaining} left  "
              f"{gbps:.2f} GB/s  {stats['decoded'] / max(el, 1e-9) / 1e6:.1f}M tok/s{cost}", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        if args.pilot:
            r = names[-1]  # the largest recipe: the one that matters for the projection
            rec = data_map["recipes"][r]
            codes_path, _ = membership(r, sizes[r], args.seed, work)
            tasks = make_tasks(r, rec, sizes[r], codes_path, task_chunks, work)
            sample = random.Random(0).sample(tasks, min(args.pilot, len(tasks)))
            t = time.time()
            res = [x for x in pool.map(count_slice, sample) if not x.get("cached")]
            wall = time.time() - t
            if not res:
                raise SystemExit("[pilot] every sampled slice was already cached; pass a larger --pilot")
            byts = sum(x["bytes"] for x in res)
            dec = sum(g["decoded_tokens"] for x in res for g in x["groups"].values())
            per_vcpu = dec / sum(x["count_s"] for x in res)
            gbps = byts / wall / 1e9
            print(f"[pilot] {len(res)} slices of {r} in {wall:.0f}s: download {gbps:.2f} GB/s total, "
                  f"decode {per_vcpu / 1e6:.2f}M tok/s per worker, "
                  f"{dec / wall / 1e6:.1f}M tok/s total")
            print("[pilot] rerun the dry run with these numbers:\n"
                  f"  python -m runpod_jobs.exact_alpha --dry-run --download-gb-per-s {gbps:.2f} "
                  f"--tokens-per-s-per-vcpu {per_vcpu:.3g}"
                  + (f" --usd-per-hour {args.usd_per_hour}" if args.usd_per_hour else ""))
            return 0

        pending: dict = {}
        recipe_left: dict[str, int] = {}
        recipe_meta: dict[str, tuple] = {}
        failed: list[str] = []

        def drain(block: bool):
            if not pending:
                return
            done, _ = wait(list(pending), timeout=None if block else 0, return_when=FIRST_COMPLETED)
            for fut in done:
                r = pending.pop(fut)
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
                if stats["slices"] % 100 == 0:
                    log(len(pending))

        for r in names:
            rec = data_map["recipes"][r]
            codes_path, base = membership(r, sizes[r], args.seed, work)  # overlaps the previous recipe
            tasks = make_tasks(r, rec, sizes[r], codes_path, task_chunks, work)
            n_chunks = int(chunk_offsets(sizes[r])[-1])
            recipe_meta[r] = (tasks, base, n_chunks)
            recipe_left[r] = len(tasks)
            print(f"[{r}] {len(tasks)} slices queued", flush=True)
            for t in tasks:
                while len(pending) >= workers * 4:  # bounded queue keeps memory flat
                    drain(block=True)
                pending[pool.submit(count_slice, t)] = r
            drain(block=False)
        while pending:
            drain(block=True)
        log(0)
    n = write_alpha_csv(out_root, args.seed, data_map)
    print(f"[done] {out_root / f'alpha_seed{args.seed}.csv'} has {n} recipes")
    if failed:
        print(f"[done] some slices failed in {failed}: rerun the same command (finished slices are cached)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
