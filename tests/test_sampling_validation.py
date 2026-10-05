"""Offline tests for src/sampling_validation.py and modal_app/sampling_validation_app.py.

Tiny fake corpora: parquet files (several row groups) and .jsonl.gz files whose
numbers follow different power laws per file, so a file-uniform sample is
biased while the size-proportional (Hansen-Hurwitz) sample is not.

    python tests/test_sampling_validation.py      (or: pytest tests/)
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import modal_app.corpus_alpha_app as caa
import modal_app.corpus_alpha_full_app as cafa
import modal_app.sampling_validation_app as sva
from src import sampling_validation as sv

MAX_N = caa.MAX_N


def _powerlaw_docs(rng, n_docs, alpha, nums_per_doc=40):
    n = np.arange(1, MAX_N + 1)
    p = n ** alpha
    p /= p.sum()
    docs = []
    for _ in range(n_docs):
        values = rng.choice(n, size=nums_per_doc, p=p)
        docs.append("Value " + " and ".join(f"{v:03d}" if v < 10 and rng.random() < 0.2 else str(v)
                                            for v in values) + ".")
    return docs


def _full_counts(texts) -> np.ndarray:
    counts = Counter()
    for t in texts:
        caa._count_text(t, counts)
    return sv.counts_vector(counts)


# --------------------------------------------------------------------------- #
# counting / fitting
# --------------------------------------------------------------------------- #

def test_counting_code_is_shared():
    assert cafa._count_text is caa._count_text and cafa._fit_alpha is caa._fit_alpha
    assert sv._fit_alpha is caa._fit_alpha and sva._count_text is caa._count_text
    c = Counter()
    caa._count_text("007 x 10000 10001 00000 abc12 3.5 2019", c)
    assert c == Counter({7: 1, 10000: 1, 0: 1, 3: 1, 5: 1, 2019: 1})


def test_ols_matches_original_and_mle_recovers_exponent():
    rng = np.random.default_rng(0)
    n = np.arange(1, MAX_N + 1)
    p = n ** -1.3
    p /= p.sum()
    vec = np.zeros(MAX_N + 1)
    vec[1:] = rng.multinomial(20_000_000, p)
    alpha, r2 = sv.fit_alpha_ols(vec)
    assert (alpha, r2) == caa._fit_alpha(Counter({i: vec[i] for i in range(1, MAX_N + 1) if vec[i] > 0}))
    a_mle, se = sv.fit_alpha_mle(vec)
    assert abs(a_mle + 1.3) < 0.002 and se < 0.001, (a_mle, se)
    assert abs(alpha + 1.3) < 0.05, alpha
    # zeros are part of the MLE: N=0 is ignored, counts may be floats (weighted samples)
    a2, _ = sv.fit_alpha_mle(vec * 0.37)
    assert abs(a2 - a_mle) < 1e-9


def test_ground_truth_reproduces_summaries():
    gt = sv.ground_truth()
    for name, row in gt.items():
        assert row["alpha_matches_summary"], name
        assert row["integer_matches"] == row["summary_integer_matches"]


# --------------------------------------------------------------------------- #
# parquet: reader, units, schemes
# --------------------------------------------------------------------------- #

def _write_parquet_corpus(root: Path, rng):
    """2 large files (alpha -1.1, 4 row groups) + 8 small files (alpha -2.0, 1 row group)."""
    files, texts = {}, []
    specs = [("big", 2, 4, 400, -1.1), ("small", 8, 1, 50, -2.0)]
    for name, n_files, n_rg, docs_per_rg, alpha in specs:
        for i in range(n_files):
            path = root / f"{name}_{i}.parquet"
            docs = _powerlaw_docs(rng, n_rg * docs_per_rg, alpha)
            texts.extend(docs)
            table = pa.table({"text": docs, "meta": [json.dumps({"i": j}) for j in range(len(docs))]})
            pq.write_table(table, path, row_group_size=docs_per_rg)
            files[path.name] = path
    return files, texts


def test_http_range_file_reads_parquet(monkeypatch=None):
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        files, _ = _write_parquet_corpus(Path(tmp), rng)
        blob = files["big_0.parquet"].read_bytes()

        class FakeResponse:
            def __init__(self, data):
                self.content = data

            def raise_for_status(self):
                pass

        calls = []

        def fake_request(method, url, headers=None, **kw):
            lo, hi = headers["Range"].split("=")[1].split("-")
            calls.append((int(lo), int(hi)))
            return FakeResponse(blob[int(lo): int(hi) + 1])

        orig = sva._request_with_retry
        sva._request_with_retry = fake_request
        try:
            source, raw = sva._open_parquet("repo", "big_0.parquet", len(blob))
            groups = sva.footer_row_groups(source)
            assert [g[0] for g in groups] == [400] * 4 and all(g[1] > 0 for g in groups)
            sliced = sva.count_rowgroup_slices(source, 2, [[150, 400], [0, 150]])
            gap = sva.count_rowgroup_slices(source, 2, [[10, 20], [300, 310]])
        finally:
            sva._request_with_retry = orig
        local = pq.ParquetFile(files["big_0.parquet"])
        texts = list(cafa._iter_parquet_texts(local, row_groups=[2]))
        expect = _full_counts(texts)
        got = sv.counts_vector(sliced[0]["counts"]) + sv.counts_vector(sliced[1]["counts"])
        assert np.array_equal(got, expect)
        assert sliced[0]["docs"] == 150 and sliced[1]["docs"] == 250
        assert [g["docs"] for g in gap] == [10, 10]
        assert np.array_equal(sv.counts_vector(gap[0]["counts"]), _full_counts(texts[10:20]))
        assert raw.bytes_read < len(blob)  # only footer + one row group's text column


def _count_units(files, units):
    unit_counts, unit_text = {}, {}
    by_rg = {}
    for u in units:
        by_rg.setdefault((u.file, u.part), []).append([u.row_start, u.row_end])
    for (f, rg), slices in by_rg.items():
        for sl in sva.count_rowgroup_slices(pq.ParquetFile(files[f]), rg, slices):
            key = sv.Unit(f, rg, sl["lo"], sl["hi"], 0).key
            unit_counts[key] = sv.counts_vector(sl["counts"])
            unit_text[key] = float(sl["text_bytes"])
    return unit_counts, unit_text


def test_size_proportional_is_unbiased_uniform_files_is_not():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as tmp:
        files, texts = _write_parquet_corpus(Path(tmp), rng)
        full = _full_counts(texts)
        manifest = {name: sva.footer_row_groups(pq.ParquetFile(p)) for name, p in files.items()}
        units = sv.parquet_units({f: [(g[0], g[1]) for g in gs] for f, gs in manifest.items()},
                                 max_unit_bytes=40_000)
        assert len(units) > len(files) * 2  # row groups were split into slices
        sizes = {"s": 60_000, "m": 200_000}
        plans = sv.plan_runs(units, list(sv.SCHEMES), sizes, replicates=5, base_seed=0)
        unit_counts, unit_text = _count_units(files, units)
        records = sv.evaluate_runs(plans, unit_counts, unit_text, full, sizes, n_boot=100)
        table = {(r["scheme"], r["size"]): r for r in sv.summary_table(records, sizes)}

    full_alpha = sv.describe(full)["alpha_mle"]
    b = table[("size_proportional", "m")]
    a = table[("uniform_files", "m")]
    assert abs(b["alpha_mle_mean"] - full_alpha) < 0.03, (b, full_alpha)
    assert a["alpha_mle_mean"] < full_alpha - 0.1, (a, full_alpha)  # pulled toward the small files' -2.0
    assert b["mean_abs_err_mle"] < a["mean_abs_err_mle"]
    for r in records:
        assert r["sampled_tokens"] >= r["target_tokens"] or r["short"]
        assert r["alpha_ols_ci95"][0] <= r["alpha_ols_ci95"][1]
        assert 0 < r["support"] <= 1 and -1 <= r["logcount_corr"] <= 1
    # nested: a replicate's small sample is a prefix of its large sample
    assert all(r["draws"] <= next(q["draws"] for q in records if q["scheme"] == r["scheme"]
               and q["replicate"] == r["replicate"] and q["size"] == "m")
               for r in records if r["size"] == "s")
    with tempfile.TemporaryDirectory() as tmp:
        out = sv.write_outputs(Path(tmp), full, records, sizes, {"corpus": "fake"})
        assert (Path(tmp) / "alpha_vs_size.png").exists() and (Path(tmp) / "summary_table.csv").exists()
        assert set(out["acceptance"]) == {"s", "m"}


# --------------------------------------------------------------------------- #
# gz: streaming units and two-stage sampling
# --------------------------------------------------------------------------- #

def _write_gz(docs) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for d in docs:
            gz.write(json.dumps({"text": d}).encode() + b"\n")
    return buf.getvalue()


def test_gz_units_partition_the_file():
    rng = np.random.default_rng(3)
    docs = _powerlaw_docs(rng, 3000, -1.4)
    blob = _write_gz(docs)
    unit_bytes = 16_384
    n_units = -(-len(blob) // unit_bytes)
    res = sva.count_gz_stream(io.BytesIO(blob), len(blob), unit_bytes, range(n_units))
    total = sum((sv.counts_vector(u["counts"]) for u in res["units"].values()), np.zeros(MAX_N + 1))
    assert np.array_equal(total, _full_counts(docs))
    assert sum(u["docs"] for u in res["units"].values()) == len(docs)
    # an early stop reads only a prefix and returns complete leading units
    part = sva.count_gz_stream(io.BytesIO(blob), len(blob), unit_bytes, [1, 3])
    assert set(part["units"]) <= {"1", "3"} and part["compressed_bytes_read"] < len(blob)
    for k in part["units"]:
        assert part["units"][k] == res["units"][k]


def test_two_stage_with_multipliers_is_unbiased():
    rng = np.random.default_rng(4)
    unit_bytes = 8192
    spec = [("ja", 3, 1500, -1.2, 2), ("en", 6, 600, -1.9, 1)]
    files, unit_counts, unit_text, full = {}, {}, {}, np.zeros(MAX_N + 1)
    for name, n_files, n_docs, alpha, mult in spec:
        for i in range(n_files):
            docs = _powerlaw_docs(rng, n_docs, alpha)
            blob = _write_gz(docs)
            path = f"{name}/{i}.jsonl.gz"
            files[path] = (len(blob), mult)
            n_units = -(-len(blob) // unit_bytes)
            res = sva.count_gz_stream(io.BytesIO(blob), len(blob), unit_bytes, range(n_units))
            for part in range(n_units):
                u = res["units"].get(str(part), {"counts": {}, "text_bytes": 0})
                unit_counts[sv.gz_unit_key(path, part)] = sv.counts_vector(u["counts"])
                unit_text[sv.gz_unit_key(path, part)] = float(u["text_bytes"])
            full += mult * _full_counts(docs)  # the full pass counts ja twice
    sizes = {"x": 120_000}
    plans = {rep: sv.two_stage_plan(files, unit_bytes, files_per_run=12, seed=rep) for rep in range(20)}
    records = sv.evaluate_two_stage(plans, unit_counts, unit_text, full, sizes, unit_bytes,
                                    text_per_compressed=2.5, n_boot=50)
    mean = np.mean([r["alpha_mle"] for r in records])
    full_alpha = sv.describe(full)["alpha_mle"]
    assert abs(mean - full_alpha) < 0.02, (mean, full_alpha)
    assert all(r["bootstrap_unit"] == "file" for r in records)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
