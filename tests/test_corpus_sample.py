"""Offline check of runpod_jobs.corpus_sample: real files in every format (json.gz, jsonl.zst,
jsonl, parquet) served by a local HTTP server, a fake tokenizer, and the selection rules
(random-order prefix up to the budget, fractional last file, strata in proportion,
corpus smaller than the budget).

    python tests/test_corpus_sample.py
"""

import csv
import gzip
import json
import os
import sys
import tempfile
import threading
from collections import Counter
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runpod_jobs.corpus_sample as cs  # noqa: E402
from modal_app.corpus_alpha_app import _count_text  # noqa: E402


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


class _Range(_Quiet):
    """Static files with open-ended or closed single byte ranges (like S3 / Cloudflare)."""

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        rng = self.headers.get("Range")
        if not path.is_file() or not rng:
            return super().do_GET()
        data = path.read_bytes()
        lo, hi = rng.split("=")[1].split("-")
        part = data[int(lo):int(hi) + 1 if hi else None]
        self.send_response(206)
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)


class _Enc:
    def __init__(self, n):
        self.ids = list(range(n))


class _FakeTok:
    """One token per whitespace-separated word."""

    def encode_batch(self, texts, add_special_tokens=False):
        return [_Enc(len(t.split())) for t in texts]


def _docs(i: int, n: int = 200) -> list[str]:
    # identical length and shape per document, so tokens per byte is exact for the estimate
    return [f"doc {i:03d} {j:04d} year {1900 + (i * 7 + j) % 120:04d} value {(i * 31 + j) % 999:03d} end"
            for j in range(n)]


def _write(root: Path, name: str, docs: list[str], key: str = "text") -> Path:
    path = root / name
    lines = b"".join(json.dumps({key: d}).encode() + b"\n" for d in docs)
    if name.endswith(".json.gz"):
        path.write_bytes(gzip.compress(lines))
    elif name.endswith(".jsonl.zst"):
        import zstandard

        path.write_bytes(zstandard.ZstdCompressor().compress(lines))
    elif name.endswith(".parquet"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table({key: docs}), path)
    else:
        path.write_bytes(lines)
    return path


def _serve(root: Path, handler=_Quiet):
    for var in ("NO_PROXY", "no_proxy"):
        os.environ[var] = ",".join(filter(None, [os.environ.get(var), "127.0.0.1", "localhost"]))
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(handler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


def _run(tmp: Path, spec: dict, budget: float, workers: int = 3) -> tuple[dict, dict]:
    sources = tmp / "sources.json"
    sources.write_text(json.dumps({"budget_tokens": budget, "seed": 7, "tokenizer": "fake",
                                   "corpora": {"x": spec}}))
    out = tmp / "out"
    assert cs.main(["--corpus", "x", "--sources", str(sources), "--budget-tokens", str(budget),
                    "--workers", str(workers), "--work-dir", str(tmp / "work"), "--out-dir", str(out)]) == 0
    counts = {int(r["number"]): int(r["count"]) for r in csv.DictReader(open(out / "x" / "counts.csv"))}
    return counts, json.loads((out / "x" / "summary.json").read_text())


def _expected(files_docs: list[list[str]], weights: list[float]) -> dict:
    want = Counter()
    for docs, w in zip(files_docs, weights):
        c = Counter()
        for d in docs:
            _count_text(d, c)
        for k, v in c.items():
            want[k] += w * v
    return {k: round(v) for k, v in want.items() if round(v)}


def test_prefix_with_fractional_last_file_in_every_format():
    cs._tokenizer = lambda repo: _FakeTok()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "srv"
        root.mkdir()
        names = ["a.json.gz", "b.jsonl.zst", "c.jsonl", "d.parquet", "e.json.gz", "f.jsonl"]
        docs = {n: _docs(i) for i, n in enumerate(names)}
        for n in names:
            _write(root, n, docs[n])
        server, base = _serve(root)
        try:
            spec = {"strata_urls": {"all": [base + n for n in names]}, "format": "auto", "text_key": "text"}
            order = [f["url"].rsplit("/", 1)[1] for f in cs.shuffled(cs.list_files(spec), 7)["all"]]
            per_file = 200 * (len(docs[names[0]][0].split()) + 1)          # words + 1 EOS per doc
            budget = 2.5 * per_file                                         # 2 files + half of the 3rd
            counts, summary = _run(tmp, spec, budget)
        finally:
            server.shutdown()
    want = _expected([docs[n] for n in order[:3]], [1, 1, 0.5])
    assert {k: v for k, v in counts.items() if v} == want
    assert abs(summary["tokens_est"] - budget) < 1e-6 * budget
    assert abs(summary["files_used"] - 2.5) < 1e-9


def test_strata_get_budget_in_proportion_and_small_corpus_repeats():
    cs._tokenizer = lambda repo: _FakeTok()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "srv"
        root.mkdir()
        big = [_write(root, f"big{i}.jsonl", _docs(i)).name for i in range(6)]
        small = [_write(root, f"small{i}.jsonl", _docs(10 + i)).name for i in range(2)]
        server, base = _serve(root)
        try:
            spec = {"strata_urls": {"big": [base + n for n in big], "small": [base + n for n in small]},
                    "format": "auto", "text_key": "text"}
            per_file = 200 * (len(_docs(0)[0].split()) + 1)
            counts, summary = _run(tmp, spec, 4 * per_file)               # half of the 8 files' tokens
            st = summary["per_stratum"]
            assert abs(st["big"]["target_tokens"] - 3 * per_file) < 1e-6 * per_file   # 6/8 of the budget
            assert abs(st["small"]["target_tokens"] - 1 * per_file) < 1e-6 * per_file
            assert abs(st["big"]["files_used"] - 3) < 1e-9 and abs(st["small"]["files_used"] - 1) < 1e-9
            # budget larger than the corpus: every file counted, repeated to the budget
            (tmp / "again").mkdir()
            counts2, summary2 = _run(tmp / "again", spec, 12 * per_file)
        finally:
            server.shutdown()
    assert abs(summary2["tokens_est"] - 12 * per_file) < 1e-6 * per_file
    all_docs = [_docs(i) for i in range(6)] + [_docs(10 + i) for i in range(2)]
    assert {k: v for k, v in counts2.items() if v} == _expected(all_docs, [1.5] * 8)


def test_big_jsonl_split_into_byte_ranges_counts_every_line_once():
    cs._tokenizer = lambda repo: _FakeTok()
    # documents of very different lengths, then of one length L, so piece edges fall inside
    # lines and exactly on, one byte before and one byte after line starts
    ragged = [f"row {i} " + "word 7 " * (i % 37) + f"year {1800 + i}" for i in range(300)]
    even = [f"row {i:04d} year {1800 + i:04d}" for i in range(300)]
    line = len(json.dumps({"text": even[0]}).encode()) + 1
    for docs, pieces_to_try in ((ragged, (997, 1000, 4096)), (even, (3 * line - 1, 3 * line, 3 * line + 1, line))):
        _check_pieces(docs, pieces_to_try)


def _check_pieces(docs: list[str], pieces_to_try):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "srv"
        root.mkdir()
        _write(root, "big.jsonl", docs)
        server, base = _serve(root, _Range)
        try:
            size = (root / "big.jsonl").stat().st_size
            files = {"x": [{"url": base + "big.jsonl", "size": size, "format": "jsonl"}]}
            for piece in pieces_to_try:
                pieces = cs.split_large(files, piece)["x"]
                assert len(pieces) > 2 and sum(f["size"] for f in pieces) == size
                total, n_docs = Counter(), 0
                for i, f in enumerate(pieces):
                    r = cs.count_file({**f, "stratum": "x", "text_key": "text", "tokenizer": "fake",
                                       "tmp": str(tmp / "t"), "cache": str(tmp / f"c{piece}_{i}.json")})
                    total.update({int(k): v for k, v in r["counts"].items()})
                    n_docs += r["docs"]
                assert n_docs == len(docs), (piece, n_docs)
                assert dict(total) == _expected([docs], [1]), piece
            small = cs.split_large(files, size)["x"]               # not above 2 x piece: kept whole
            assert [f["url"] for f in small] == [base + "big.jsonl"]
        finally:
            server.shutdown()


def test_files_the_server_refuses_are_skipped_and_reported():
    cs._tokenizer = lambda repo: _FakeTok()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "srv"
        root.mkdir()
        names = [_write(root, f"f{i}.jsonl", _docs(i)).name for i in range(30)]
        server, base = _serve(root)
        try:
            spec = {"strata_urls": {"all": [base + n for n in names] + [base + "gone.jsonl"]},
                    "format": "auto", "text_key": "text"}
            per_file = 200 * (len(_docs(0)[0].split()) + 1)
            counts, summary = _run(tmp, spec, 2 * per_file)
            assert summary["unavailable_files"] == [base + "gone.jsonl"]
            assert abs(summary["tokens_est"] - 2 * per_file) < 1e-6 * per_file
            # too many dead files: refuse instead of sampling a biased remainder
            spec["strata_urls"]["all"] += [base + f"gone{i}.jsonl" for i in range(3)]
            try:
                cs.list_files(spec)
            except SystemExit as exc:
                assert "unavailable" in str(exc)
            else:
                raise AssertionError("4 of 34 dead files were accepted")
        finally:
            server.shutdown()


def test_rare_k_matches_the_analysis_and_needs_no_analysis_packages():
    import subprocess

    from src.count_features import RARE_K

    assert cs.RARE_K == RARE_K
    # the pod's venv has no pandas: importing the counter must not pull it in
    code = "import sys, runpod_jobs.corpus_sample; sys.exit('pandas' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent).returncode == 0


if __name__ == "__main__":
    test_prefix_with_fractional_last_file_in_every_format()
    test_strata_get_budget_in_proportion_and_small_corpus_repeats()
    test_big_jsonl_split_into_byte_ranges_counts_every_line_once()
    test_files_the_server_refuses_are_skipped_and_reported()
    test_rare_k_matches_the_analysis_and_needs_no_analysis_packages()
    print("ok")
