from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import pyarrow.ipc as arrow_ipc
import requests
from huggingface_hub import HfApi, hf_hub_url

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.text import DIGIT_TOKEN_PATTERN


DEFAULT_REPO = "ArmelR/the-pile-splitted"
DEFAULT_SPLITS = ("train", "test")


def _parse_csv_arg(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def discover_arrow_files(
    repo_id: str,
    splits: set[str],
    domains: set[str] | None,
) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    out: list[str] = []
    for path in files:
        if not path.startswith("data/") or not path.endswith(".arrow"):
            continue
        parts = path.split("/")
        if len(parts) < 4:
            continue
        split = parts[-2]
        domain = "/".join(parts[1:-2])
        if split not in splits:
            continue
        if domains is not None and domain not in domains:
            continue
        out.append(path)
    return sorted(out)


def _safe_local_name(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", filename)


def download_file(repo_id: str, filename: str, tmp_dir: Path, chunk_size: int) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / _safe_local_name(filename)
    part = out.with_suffix(out.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")

    mode = "ab" if existing else "wb"
    with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as resp:
        if existing and resp.status_code != 206:
            existing = 0
            mode = "wb"
            resp.close()
            with requests.get(url, stream=True, timeout=(20, 120)) as fresh:
                fresh.raise_for_status()
                with open(part, mode) as fh:
                    for chunk in fresh.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
        else:
            resp.raise_for_status()
            with open(part, mode) as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)

    part.replace(out)
    return out


def count_arrow_file(
    path: Path,
    counts: Counter[int],
    n_min: int,
    n_max: int,
    text_key: str,
    max_docs: int | None,
    docs_seen: int,
) -> tuple[int, int, bool]:
    file_docs = 0
    file_matches = 0
    with arrow_ipc.open_stream(path) as reader:
        for batch in reader:
            if text_key not in batch.column_names:
                raise KeyError(f"{path} has no column {text_key!r}; columns={batch.column_names}")
            texts = batch.column(text_key)
            for scalar in texts:
                if not scalar.is_valid:
                    docs_seen += 1
                    file_docs += 1
                    continue
                text = scalar.as_py()
                if isinstance(text, str):
                    for match in DIGIT_TOKEN_PATTERN.finditer(text):
                        token = match.group(0)
                        try:
                            number = int(token)
                        except ValueError:
                            continue
                        if n_min <= number <= n_max:
                            counts[number] += 1
                            file_matches += 1
                docs_seen += 1
                file_docs += 1
                if max_docs is not None and docs_seen >= max_docs:
                    return file_docs, file_matches, True
    return file_docs, file_matches, False


def read_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            done.add(json.loads(line)["filename"])
    return done


def write_outputs(outdir: Path, counts: Counter[int], summary: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    counts_path = outdir / f"counts_{summary['n_min']}_to_{summary['n_max']}.csv"
    with open(counts_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["number", "count"])
        for number in range(summary["n_min"], summary["n_max"] + 1):
            writer.writerow([number, counts.get(number, 0)])
    with open(outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)


def iter_limited(files: Iterable[str], max_files: int | None) -> list[str]:
    files = list(files)
    if max_files is None:
        return files
    return files[:max_files]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count integer frequencies in the non-dedup Pile rehost "
            "ArmelR/the-pile-splitted, one Arrow file at a time."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--domains", default=None, help="comma-separated domain filter")
    parser.add_argument("--outdir", default="results/pile_original_splitted/0_to_10000")
    parser.add_argument("--tmp-dir", default=".cache/pile_splitted_counter")
    parser.add_argument("--n-min", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=10000)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--chunk-size-mb", type=int, default=16)
    parser.add_argument("--keep-files", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    outdir = Path(args.outdir)
    tmp_dir = Path(args.tmp_dir)
    splits = _parse_csv_arg(args.splits) or set(DEFAULT_SPLITS)
    domains = _parse_csv_arg(args.domains)
    processed_log = outdir / "processed_files.jsonl"
    processed = set() if args.no_resume else read_processed(processed_log)
    files = iter_limited(discover_arrow_files(args.repo_id, splits, domains), args.max_files)
    counts: Counter[int] = Counter()
    docs_seen = 0
    total_matches = 0
    start = time.time()

    if not files:
        raise SystemExit("no Arrow files matched the requested repo/splits/domains")

    print(f"[pile-splitted] repo={args.repo_id}")
    print(f"[pile-splitted] files matched={len(files):,}; already processed={len(processed):,}")
    print(f"[pile-splitted] output={outdir}")

    outdir.mkdir(parents=True, exist_ok=True)
    for i, filename in enumerate(files, 1):
        if filename in processed:
            print(f"[skip] {i}/{len(files)} {filename}")
            continue

        print(f"[download] {i}/{len(files)} {filename}")
        local = download_file(
            args.repo_id,
            filename,
            tmp_dir=tmp_dir,
            chunk_size=args.chunk_size_mb * 1024 * 1024,
        )
        print(f"[count] {filename} ({local.stat().st_size / 1e9:.3f} GB)")
        file_docs, file_matches, stopped = count_arrow_file(
            local,
            counts=counts,
            n_min=args.n_min,
            n_max=args.n_max,
            text_key=args.text_key,
            max_docs=args.max_docs,
            docs_seen=docs_seen,
        )
        docs_seen += file_docs
        total_matches += file_matches

        if not args.keep_files:
            local.unlink(missing_ok=True)

        record = {
            "filename": filename,
            "docs_seen": file_docs,
            "matches_in_range": file_matches,
            "completed_at": time.time(),
        }
        if not stopped:
            with open(processed_log, "a") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")

        summary = {
            "repo_id": args.repo_id,
            "splits": sorted(splits),
            "domains": sorted(domains) if domains is not None else None,
            "n_min": args.n_min,
            "n_max": args.n_max,
            "files_matched": len(files),
            "files_completed_this_run": i,
            "docs_seen_this_run": docs_seen,
            "matches_in_range_this_run": total_matches,
            "elapsed_seconds": time.time() - start,
            "stopped_early": stopped,
        }
        write_outputs(outdir, counts, summary)
        print(
            f"[progress] docs={docs_seen:,} matches_in_range={total_matches:,} "
            f"elapsed={summary['elapsed_seconds'] / 3600:.2f}h"
        )

        if stopped:
            print("[stop] max-docs reached before completing current file")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
