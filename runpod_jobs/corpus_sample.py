"""Number counts of a random sample of a public corpus, sized to a model's training budget
(E09: the Paloma corpora whose own training files are not public). CPU pod, no Modal.

Files are taken in a seeded random order (per stratum, e.g. RedPajama's subsets, each
stratum getting its share of the budget in proportion to its estimated tokens). Every file
is streamed, decompressed and counted with the project's counting rule (`_count_text`).
Tokens are estimated per file by tokenizing every 50th document with the model's tokenizer
(+1 EOS per document, as training appends one). Files are used in random order until the
budget is reached; the last file contributes the fraction needed. No scaling of the total.

    python -m runpod_jobs.corpus_sample --corpus c4 --dry-run           # list files, sizes, plan
    python -m runpod_jobs.corpus_sample --corpus c4 --pilot 4           # measure speed on 4 files
    python -m runpod_jobs.corpus_sample --corpus c4 --usd-per-hour 0.96 \\
        --upload-hf numberline-alpha-results --delete-pod-when-done --max-hours 6

Outputs results/paloma_counts/<corpus>/{counts.csv, summary.json} (uploaded to
<HF repo>/paloma/<corpus>/). Finished files are cached, so a rerun resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from modal_app.corpus_alpha_app import MAX_N, _count_text
from runpod_jobs.exact_alpha import atomic_write, usable_vcpus
from runpod_jobs.pod import terminate_this_pod
from src.sampling_validation import describe

SOURCES = Path("configs/paloma/corpus_sources.json")
OUT = Path("results/paloma_counts")
SAMPLE_EVERY = 50          # tokenize every 50th document to estimate tokens per text byte
PILOT_FILES = 2            # files per stratum counted first to measure tokens per compressed byte
RARE_K = 4000              # = src.count_features.RARE_K (not imported: it needs pandas, absent on CPU pods)
SPLIT_BYTES = 1 << 30      # plain .jsonl files above 2 x this are cut into byte ranges of this size


# --------------------------------------------------------------------------- #
# file lists
# --------------------------------------------------------------------------- #

def hf_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _auth(url: str) -> dict:
    tok = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok and "huggingface.co" in url else {}


def _hf_sizes(repo: str, paths: list[str]) -> dict[str, int]:
    from huggingface_hub import HfApi

    api, out = HfApi(token=os.environ.get("HF_TOKEN")), {}
    for i in range(0, len(paths), 200):
        for info in api.get_paths_info(repo, paths[i:i + 200], repo_type="dataset"):
            if getattr(info, "size", None):
                out[info.path] = int(info.size)
    return out


def _head_sizes(urls: list[str]) -> dict[str, int]:
    from runpod_jobs.exact_alpha import _http_sizes

    return _http_sizes(urls)


def split_large(files: dict[str, list[dict]], piece: int | None = None) -> dict[str, list[dict]]:
    """Cut big uncompressed .jsonl files (RedPajama's wikipedia and stackexchange are one
    ~100 GB file each) into byte ranges `url#bytes=lo-hi`, so they are read in parallel and
    sampled in random pieces instead of read whole by one core. A piece holds the lines
    that START in [lo, hi), so the pieces of a file hold each line exactly once."""
    piece = piece or SPLIT_BYTES
    out = {}
    for st, fs in files.items():
        out[st] = []
        for f in fs:
            if f["format"] != "jsonl" or f["size"] <= 2 * piece:
                out[st].append(f)
                continue
            for lo in range(0, f["size"], piece):
                hi = min(f["size"], lo + piece)
                out[st].append({**f, "url": f"{f['url']}#bytes={lo}-{hi}", "size": hi - lo})
    return out


def list_files(spec: dict) -> dict[str, list[dict]]:
    """{stratum: [{"url", "size", "format"}]} for a corpus spec (big .jsonl files split)."""
    return split_large(_list_files(spec))


def _list_files(spec: dict) -> dict[str, list[dict]]:
    if "strata_urls" in spec:                            # explicit lists (tests, ad-hoc)
        return {st: [{"url": u, "size": s, "format": _fmt(spec, u)} for u, s in _head_sizes(urls).items()]
                for st, urls in spec["strata_urls"].items()}
    if "url_lists" in spec:                              # RedPajama: urls/<subset>.txt in the HF repo
        from huggingface_hub import hf_hub_download

        out = {}
        for subset in spec["url_lists"]:
            path = hf_hub_download(spec["repo"], f"urls/{subset}.txt", repo_type="dataset",
                                   token=os.environ.get("HF_TOKEN"))
            urls = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
            out[subset] = [{"url": u, "size": s, "format": _fmt(spec, u)}
                           for u, s in _head_sizes(urls).items()]
        return out
    lst = spec["list"]
    if "template" in lst:
        paths = [lst["template"].format(i=i) for i in range(lst["n"])]
    else:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        paths = sorted(f.path for f in api.list_repo_tree(spec["repo"], path_in_repo=lst["dir"],
                                                            repo_type="dataset", recursive=True)
                       if f.path.endswith(lst["suffix"]))
    sizes = _hf_sizes(spec["repo"], paths)
    missing = [p for p in paths if p not in sizes]
    if missing:
        raise SystemExit(f"{len(missing)} files not found in {spec['repo']}, e.g. {missing[:3]}")
    return {"all": [{"url": hf_url(spec["repo"], p), "size": sizes[p], "format": _fmt(spec, p)}
                    for p in paths]}


def _fmt(spec: dict, name: str) -> str:
    if spec.get("format", "auto") != "auto":
        return spec["format"]
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".zst"):
        return "jsonl.zst"
    if name.endswith(".gz"):
        return "jsonl.gz"
    return "jsonl"


# --------------------------------------------------------------------------- #
# one file (worker process)
# --------------------------------------------------------------------------- #

_TOKENIZER = {}


def _tokenizer(repo: str):
    if repo not in _TOKENIZER:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        _TOKENIZER[repo] = Tokenizer.from_file(
            hf_hub_download(repo, "tokenizer.json", token=os.environ.get("HF_TOKEN")))
    return _TOKENIZER[repo]


def _texts(url: str, fmt: str, text_key: str, tmp_dir: Path):
    """Yield the text of every document of one file, streamed (parquet goes via a temp file)."""
    import requests

    if fmt == "parquet":
        import pyarrow.parquet as pq

        tmp_dir.mkdir(parents=True, exist_ok=True)
        local = tmp_dir / (hashlib.sha1(url.encode()).hexdigest() + ".parquet")
        try:
            with requests.get(url, headers=_auth(url), stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for block in r.iter_content(8 << 20):
                        f.write(block)
            for batch in pq.ParquetFile(local).iter_batches(columns=[text_key], batch_size=4096):
                for text in batch.column(0).to_pylist():
                    if isinstance(text, str):
                        yield text
        finally:
            local.unlink(missing_ok=True)
        return
    import io

    import orjson

    url, _, frag = url.partition("#bytes=")
    lo, hi = (int(x) for x in frag.split("-")) if frag else (0, None)
    headers = _auth(url)
    if lo:   # one byte early: the line that byte ends belongs to the previous piece
        headers["Range"] = f"bytes={lo - 1}-"
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        if lo and r.status_code != 206:
            raise RuntimeError(f"{url} ignored the Range request (HTTP {r.status_code})")
        r.raw.decode_content = False
        r.raw.auto_close = False      # urllib3 would close at EOF, before BufferedReader's last read
        if fmt == "jsonl.gz":
            import gzip

            stream = io.BufferedReader(gzip.GzipFile(fileobj=r.raw), 8 << 20)
        elif fmt == "jsonl.zst":
            import zstandard

            stream = io.BufferedReader(zstandard.ZstdDecompressor(max_window_size=2 ** 31)
                                       .stream_reader(r.raw), 8 << 20)
        else:
            stream = io.BufferedReader(r.raw, 8 << 20)
        pos = max(0, lo - 1)
        for i, line in enumerate(stream):
            start, pos = pos, pos + len(line)
            if lo and i == 0:
                continue                  # ends at or after lo-1: started in the previous piece
            if hi is not None and start >= hi:
                break                     # starts in the next piece
            if not line.strip():
                continue
            try:
                text = orjson.loads(line).get(text_key)
            except orjson.JSONDecodeError:
                continue
            if isinstance(text, str):
                yield text


def count_file(task: dict) -> dict:
    """Counts, text bytes, documents and a token estimate for one file; cached on disk."""
    cache = Path(task["cache"])
    if cache.exists():
        return {**json.loads(cache.read_text()), "cached": True}
    tok = _tokenizer(task["tokenizer"])
    last = None
    for attempt in range(6):
        counts: Counter = Counter()
        docs = text_bytes = s_bytes = s_tokens = 0
        sample: list[str] = []
        t0 = time.time()
        try:
            for text in _texts(task["url"], task["format"], task["text_key"], Path(task["tmp"])):
                _count_text(text, counts)
                docs += 1
                text_bytes += len(text.encode("utf-8"))
                if docs % SAMPLE_EVERY == 0:
                    sample.append(text)
                    if len(sample) >= 64:
                        s_bytes += sum(len(x.encode("utf-8")) for x in sample)
                        s_tokens += sum(len(e.ids) for e in tok.encode_batch(sample, add_special_tokens=False))
                        sample.clear()
            if sample:
                s_bytes += sum(len(x.encode("utf-8")) for x in sample)
                s_tokens += sum(len(e.ids) for e in tok.encode_batch(sample, add_special_tokens=False))
        except Exception as exc:  # noqa: BLE001 -- network / decode error: start the file again
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(60, 2 ** attempt) + random.random())
            continue
        per_byte = s_tokens / s_bytes if s_bytes else 0.0
        out = {"url": task["url"], "stratum": task["stratum"], "compressed_bytes": task["size"],
               "docs": docs, "text_bytes": text_bytes, "sample_bytes": s_bytes, "sample_tokens": s_tokens,
               "tokens_est": text_bytes * per_byte + docs,      # + 1 EOS per document
               "seconds": time.time() - t0,
               "counts": {str(k): v for k, v in counts.items() if k <= MAX_N}}
        atomic_write(cache, json.dumps(out))
        return out
    raise RuntimeError(f"giving up on {task['url']}: {last}")


# --------------------------------------------------------------------------- #
# plan + selection (main process)
# --------------------------------------------------------------------------- #

def shuffled(files: dict[str, list[dict]], seed: int) -> dict[str, list[dict]]:
    out = {}
    for s in sorted(files):
        order = sorted(files[s], key=lambda f: f["url"])
        random.Random(f"{seed}:{s}").shuffle(order)
        out[s] = order
    return out


def stratum_targets(files: dict[str, list[dict]], done: dict[str, dict],
                    budget: float) -> tuple[dict[str, float], float]:
    """Budget per stratum in proportion to its estimated tokens (compressed bytes x the
    tokens per compressed byte measured on its counted files)."""
    est = {}
    for s, fs in files.items():
        got = [done[f["url"]] for f in fs if f["url"] in done]
        ratio = sum(g["tokens_est"] for g in got) / sum(g["compressed_bytes"] for g in got)
        est[s] = ratio * sum(f["size"] for f in fs)
    total = sum(est.values())
    return {s: budget * e / total for s, e in est.items()}, total


def select(files: dict[str, list[dict]], done: dict[str, dict], targets: dict[str, float]):
    """Per stratum, the random-order prefix of files that reaches its target; the last file
    is used with the fraction needed. Returns ({url: weight}, {stratum: tokens still
    missing}); a stratum is open while a file of its prefix is uncounted or it is short."""
    weights, short = {}, {}
    for s, fs in files.items():
        acc = 0.0
        for f in fs:
            need = targets[s] - acc
            if need <= 0:
                break
            r = done.get(f["url"])
            if r is None:
                break                 # need this file's result before the prefix is known
            w = min(1.0, need / r["tokens_est"]) if r["tokens_est"] > 0 else 1.0
            weights[f["url"]] = w
            acc += w * r["tokens_est"]
        if acc < targets[s] * (1 - 1e-9):
            short[s] = targets[s] - acc
    return weights, short


def combine(done: dict[str, dict], weights: dict[str, float]) -> tuple[np.ndarray, dict]:
    counts = np.zeros(MAX_N + 1)
    agg = Counter()
    for url, w in weights.items():
        r = done[url]
        for k, v in r["counts"].items():
            counts[int(k)] += w * v
        for key in ("docs", "text_bytes", "compressed_bytes", "tokens_est"):
            agg[key] += w * r[key]
        agg["files"] += w
    return np.rint(counts).astype(np.int64), dict(agg)


def count_summary(counts: np.ndarray) -> dict:
    """alpha (describe) plus the two E09 count predictors S and R (K = 4000) over 10..9999."""
    x = np.asarray(counts, dtype=float)[10:10000]
    return {**describe(counts), "S_K4000": float((x / (x + RARE_K)).sum()), "R_K4000": int((x < RARE_K).sum())}


def write_outputs(corpus: str, spec: dict, counts: np.ndarray, agg: dict, budget: float,
                  per_stratum: dict, out_root: Path, args) -> Path:
    out = out_root / corpus
    out.mkdir(parents=True, exist_ok=True)
    lines = ["number,count"] + [f"{n},{int(c)}" for n, c in enumerate(counts)]
    atomic_write(out / "counts.csv", "\n".join(lines) + "\n")
    summary = {"corpus": corpus, "method": "random sample sized to the training budget (runpod_jobs.corpus_sample)",
               "budget_tokens": budget, "tokens_est": agg.get("tokens_est"), "seed": args.seed,
               "tokenizer": args.tokenizer, "source": {k: spec[k] for k in spec if k != "urls"},
               "files_used": agg.get("files"), "docs": agg.get("docs"), "text_bytes": agg.get("text_bytes"),
               "compressed_bytes": agg.get("compressed_bytes"), "per_stratum": per_stratum,
               **count_summary(counts)}
    atomic_write(out / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(f"[{corpus}] DONE alpha_ols={summary['alpha_ols']:.4f} alpha_mle={summary['alpha_mle']:.5f} "
          f"numbers={summary['integer_matches']:.3e} S={summary['S_K4000']:.0f} R={summary['R_K4000']} -> {out}",
          flush=True)
    return out


def hf_upload(repo: str, folder: Path, corpus: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for attempt in range(8):
        try:
            api.upload_folder(folder_path=str(folder), path_in_repo=f"paloma/{corpus}", repo_id=repo,
                              repo_type="dataset", commit_message=f"paloma {corpus} counts")
            files = set(api.list_repo_files(repo, repo_type="dataset"))
            if {f"paloma/{corpus}/counts.csv", f"paloma/{corpus}/summary.json"} <= files:
                print(f"[{corpus}] uploaded and verified: huggingface.co/datasets/{repo}/tree/main/paloma/{corpus}",
                      flush=True)
                return
        except Exception as exc:  # noqa: BLE001
            print(f"[{corpus}] upload attempt {attempt + 1} failed: {exc}", flush=True)
        time.sleep(min(300, 15 * 2 ** attempt))
    raise RuntimeError(f"could not upload {corpus} to {repo}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sources", default=str(SOURCES))
    cfg = json.loads(Path(pre.parse_known_args(argv)[0].sources).read_text())   # defaults come from it
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", required=True, help=f"one of {sorted(cfg['corpora'])}")
    p.add_argument("--budget-tokens", type=float, default=cfg["budget_tokens"])
    p.add_argument("--seed", type=int, default=cfg["seed"])
    p.add_argument("--tokenizer", default=cfg["tokenizer"])
    p.add_argument("--sources", default=str(SOURCES), help="corpus spec file")
    p.add_argument("--workers", type=int, default=0, help="files at once; default vCPUs + 4")
    p.add_argument("--work-dir", default="/workspace/paloma_work")
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--usd-per-hour", type=float, default=0.0)
    p.add_argument("--pilot", type=int, default=0, help="count N random files, report speed, stop")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--upload-hf", default="", metavar="REPO")
    p.add_argument("--delete-pod-when-done", action="store_true")
    p.add_argument("--max-hours", type=float, default=0)
    args = p.parse_args(argv)
    spec = cfg["corpora"][args.corpus]
    work, out_root = Path(args.work_dir) / args.corpus, Path(args.out_dir)
    if args.delete_pod_when_done and not args.upload_hf:
        raise SystemExit("--delete-pod-when-done needs --upload-hf (results must be off the pod first)")

    files = shuffled(list_files(spec), args.seed)
    n_files = sum(len(v) for v in files.values())
    gb = sum(f["size"] for v in files.values() for f in v) / 1e9
    print(f"[plan] {args.corpus}: {n_files} files, {gb:,.0f} GB compressed, strata "
          + ", ".join(f"{s}={len(v)}" for s, v in files.items()) + f"; budget {args.budget_tokens / 1e9:.1f}B tokens")
    if args.dry_run:
        return 0
    vcpus = usable_vcpus()
    workers = args.workers or vcpus + 4
    if args.upload_hf:
        from runpod_jobs.exact_alpha import hf_results_repo

        args.upload_hf = hf_results_repo(args.upload_hf)
        print(f"[plan] results go to the private dataset huggingface.co/datasets/{args.upload_hf}")
    if args.max_hours:
        import threading

        def deadline():
            print(f"[limit] {args.max_hours} h reached: deleting the pod", flush=True)
            terminate_this_pod(f"--max-hours {args.max_hours} reached")
            os._exit(3)

        t = threading.Timer(args.max_hours * 3600, deadline)
        t.daemon = True
        t.start()

    start = time.time()
    done: dict[str, dict] = {}

    def task(f, s):
        return {"url": f["url"], "size": f["size"], "format": f["format"], "stratum": s,
                "text_key": spec["text_key"], "tokenizer": args.tokenizer, "tmp": str(work / "tmp"),
                "cache": str(work / "files" / (hashlib.sha1(f["url"].encode()).hexdigest() + ".json"))}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending: dict = {}

        def drain(block=True):
            got, _ = wait(list(pending), timeout=None if block else 0, return_when=FIRST_COMPLETED)
            for fut in got:
                f, _ = pending.pop(fut)
                r = fut.result()
                done[f["url"]] = r
                if not r.get("cached"):
                    mb = r["text_bytes"] / max(r["seconds"], 1e-9) / 1e6
                    el = (time.time() - start) / 3600
                    cost = f"  ${el * args.usd_per_hour:.2f} so far" if args.usd_per_hour else ""
                    print(f"[{args.corpus}] {len(done)} files  last {r['text_bytes'] / 1e9:.2f} GB text "
                          f"at {mb:.0f} MB/s, {r['tokens_est'] / 1e9:.2f}B tokens  "
                          f"({el * 60:.1f} min){cost}", flush=True)

        # 1. pilot files: the first files of every stratum (the random order is fixed by the seed)
        n_pilot = args.pilot or PILOT_FILES
        first = [(f, s) for s, fs in files.items() for f in fs[:n_pilot]]
        for f, s in first:
            pending[pool.submit(count_file, task(f, s))] = (f, s)
        while pending:
            drain()
        if args.pilot:
            new = [done[f["url"]] for f, _ in first if not done[f["url"]].get("cached")]
            if new:
                secs = max(r["seconds"] for r in new)
                tb = sum(r["text_bytes"] for r in new)
                per_c = sum(r["tokens_est"] for r in new) / sum(r["compressed_bytes"] for r in new)
                need_gb = args.budget_tokens / per_c / 1e9
                print(f"[pilot] {len(new)} files, {tb / 1e9:.1f} GB text in {secs:.0f}s wall "
                      f"({tb / secs / 1e6:.0f} MB/s with {len(new)} parallel files); "
                      f"{per_c:.2f} tokens per compressed byte -> the budget needs ~{need_gb:,.0f} GB compressed")
            return 0

        # 2. per-stratum targets from the measured tokens per compressed byte
        targets, total_est = stratum_targets(files, done, args.budget_tokens)
        if total_est < args.budget_tokens:
            print(f"[plan] the corpus has ~{total_est / 1e9:.0f}B tokens, less than the budget: "
                  f"the model saw it ~{args.budget_tokens / total_est:.2f} times; every file is counted "
                  "and repeated files contribute again (weights > 1)", flush=True)
        print("[plan] targets: " + ", ".join(f"{s} {v / 1e9:.1f}B" for s, v in targets.items()), flush=True)

        # 3. keep feeding each stratum, in its random order, until its target is covered:
        #    submit files while counted + in-flight (estimated) tokens are below 102 % of the
        #    target; a stratum still short with nothing in flight always gets its next file
        cursor = {s: n_pilot for s in files}
        while True:
            _, short = select(files, done, targets)
            if not short:
                break
            for s in short:
                fs = files[s]
                got = [done[f["url"]] for f in fs if f["url"] in done]
                ratio_s = sum(g["tokens_est"] for g in got) / max(1, sum(g["compressed_bytes"] for g in got))
                mine = [fl for fl, st in pending.values() if st == s]
                have = sum(done[f["url"]]["tokens_est"] for f in fs[:cursor[s]] if f["url"] in done)
                inflight = sum(fl["size"] for fl in mine) * ratio_s
                while cursor[s] < len(fs) and len(pending) < workers * 2 and (
                        have + inflight < targets[s] * 1.02 or not mine):
                    f = fs[cursor[s]]
                    cursor[s] += 1
                    pending[pool.submit(count_file, task(f, s))] = (f, s)
                    mine.append(f)
                    inflight += f["size"] * ratio_s
            if not pending:
                break                 # every file counted: the corpus is smaller than the budget
            drain()
        if pending:  # budget covered: stop the files still running instead of waiting for them
            for fut in list(pending):
                fut.cancel()
            for proc in list(getattr(pool, "_processes", {}).values()):
                proc.terminate()

    weights, _ = select(files, done, targets)
    if total_est < args.budget_tokens:   # repeat the whole corpus to the budget (several epochs)
        all_tok = sum(r["tokens_est"] for r in done.values())
        weights = {u: args.budget_tokens / all_tok for u in done}
    counts, agg = combine(done, weights)
    per_stratum = {s: {"target_tokens": targets[s], "files_used": sum(w for u, w in weights.items()
                                                                      if done[u]["stratum"] == s)}
                   for s in files}
    folder = write_outputs(args.corpus, spec, counts, agg, args.budget_tokens, per_stratum, out_root, args)
    if args.upload_hf:
        hf_upload(args.upload_hf, folder, args.corpus)
        if args.delete_pod_when_done:
            terminate_this_pod(f"{args.corpus} uploaded and verified")
    print(f"[done] {args.corpus} in {(time.time() - start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
