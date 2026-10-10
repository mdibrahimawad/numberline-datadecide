"""Offline check of runpod_jobs.exact_alpha reading a recipe from a plain HTTP server
(the Paloma Dolma case: files on olmo-data.org, EOS = 0, its own training length), through
a real local server with HEAD + Range support. Only the tokenizer is faked.

    python tests/test_runpod_exact_alpha_http.py
"""

import csv
import json
import os
import sys
import tempfile
import threading
from collections import Counter
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runpod_jobs.exact_alpha as ea  # noqa: E402
from src.datadecide_sampling import SEQUENCE_LENGTH, membership_codes  # noqa: E402

# the real functions, captured before other test files replace them with fakes
REAL = {name: getattr(ea, name) for name in
        ("read_range", "file_sizes", "membership_codes", "load_data_map", "resolve_url")}
EOS = 0
N_INSTANCES = 120


class _RangeHandler(SimpleHTTPRequestHandler):
    """Static files with HEAD and single-range GET (enough for the reader)."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        rng = self.headers.get("Range")
        if not path.is_file() or not rng:
            return super().do_GET()
        data = path.read_bytes()
        lo, hi = (int(x) for x in rng.split("=")[1].split("-"))
        part = data[lo:hi + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)


class _Backend:
    def decode_batch(self, batch, skip_special_tokens=True):
        return [" ".join(str(int(t)) for t in ids if t != EOS) for ids in batch]


def _files(root: Path) -> dict:
    rng = np.random.default_rng(1)
    files = {}
    for i, n_chunks in enumerate((30, 55, 47, 12, 40)):
        toks = rng.integers(1, 3000, size=n_chunks * SEQUENCE_LENGTH + 77).astype(np.uint16)
        toks[rng.random(len(toks)) < 0.01] = EOS
        rel = f"preprocessed/mix/part-{i:03d}.npy"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        toks.tofile(root / rel)                    # headerless uint16, as OLMo writes them
        files[rel] = toks
    return files


def _brute_force(files: dict) -> Counter:
    sizes = [len(files[p]) for p in files]
    mult, _ = membership_codes(sizes, [6198], n_instances=N_INSTANCES)
    assert int(mult.sum()) == N_INSTANCES
    want, g = Counter(), 0
    for p in files:
        for c in range(len(files[p]) // SEQUENCE_LENGTH):
            m = int(mult[g])
            g += 1
            if m:
                for t in files[p][c * SEQUENCE_LENGTH:(c + 1) * SEQUENCE_LENGTH]:
                    if t != EOS and int(t) <= 10000:
                        want[int(t)] += m
    return want


def _run_recipes(recipes: dict, names: list[str]):
    """Serve the files over local HTTP and count `names` from `recipes` (paths filled in)."""
    for name, fn in REAL.items():
        setattr(ea, name, fn)
    for var in ("NO_PROXY", "no_proxy"):  # the local server must not go through a proxy
        os.environ[var] = ",".join(filter(None, [os.environ.get(var), "127.0.0.1", "localhost"]))
    ea._tokenizer = lambda repo: SimpleNamespace(backend_tokenizer=_Backend())
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "srv"
    files = _files(root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_RangeHandler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/"
        dmap = {"seed": 6198, "recipes": {r: {"paths": list(files), "base_url": base, "model_repo": "x/y",
                                               "n_instances": N_INSTANCES, "eos_token_id": EOS, **extra}
                                           for r, extra in recipes.items()}}
        (tmp / "map.json").write_text(json.dumps(dmap))
        out, work = tmp / "out", tmp / "work"
        args = ["--data-map", str(tmp / "map.json"), "--recipes", ",".join(names), "--seed", "6198",
                "--workers", "3", "--task-mtokens", "0", "--work-dir", str(work), "--out-dir", str(out)]
        assert ea.main(args + ["--no-stop-pod"]) == 0
    finally:
        server.shutdown()
    return files, dmap, out, work


def test_http_recipe_with_its_own_eos_and_length_matches_brute_force():
    files, _, out, _ = _run_recipes({"paloma-test": {}}, ["paloma-test"])
    summary = json.loads((out / "exact_100b_paloma-test" / "summary.json").read_text())
    got = {int(r["number"]): int(r["count"]) for r in
           csv.DictReader(open(out / "exact_100b_paloma-test" / "counts_seed_6198.csv"))}
    assert summary["per_seed"]["6198"]["tokens"] == N_INSTANCES * SEQUENCE_LENGTH
    assert {k: v for k, v in got.items() if v} == dict(_brute_force(files))


def test_parts_are_disjoint_balanced_and_cover_every_file():
    sizes = [50, 10, 40, 30, 30, 20, 5, 5, 60]
    parts = [ea.part_files({"part": [k, 3]}, sizes) for k in range(3)]
    assert set().union(*parts) == set(range(len(sizes))) and sum(map(len, parts)) == len(sizes)
    loads = [sum(sizes[f] for f in p) for p in parts]
    assert max(loads) - min(loads) <= max(sizes) // 2, loads
    assert ea.part_files({}, sizes) is None


def test_split_recipe_parts_add_up_to_the_whole_stream():
    """3 parts, each counted on its own (as 3 pods would), share one training order and
    add up to exactly the unsplit count; a missing part is refused."""
    parts = {f"paloma-test-p{k}of3": {"part": [k, 3], "membership_of": "paloma-test"} for k in range(3)}
    files, dmap, out, work = _run_recipes(parts, list(parts))
    assert sorted(p.name for p in work.iterdir() if (p / "codes_seed6198.npy").exists()) == ["paloma-test"]
    total, info = ea.merge_parts(out, dmap, "paloma-test", 6198)
    assert info["tokens"] == N_INSTANCES * SEQUENCE_LENGTH
    assert {k: int(v) for k, v in enumerate(total) if v} == dict(_brute_force(files))
    import shutil

    shutil.rmtree(out / "exact_100b_paloma-test-p1of3")
    try:
        ea.merge_parts(out, dmap, "paloma-test", 6198)
    except SystemExit as exc:
        assert "p1of3" in str(exc)
    else:
        raise AssertionError("a missing part was accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
