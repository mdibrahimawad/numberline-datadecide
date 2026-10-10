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


def test_http_recipe_with_its_own_eos_and_length_matches_brute_force():
    for name, fn in REAL.items():
        setattr(ea, name, fn)
    for var in ("NO_PROXY", "no_proxy"):  # the local server must not go through a proxy
        os.environ[var] = ",".join(filter(None, [os.environ.get(var), "127.0.0.1", "localhost"]))
    ea._tokenizer = lambda repo: SimpleNamespace(backend_tokenizer=_Backend())
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "srv"
        files = {}
        for i, n_chunks in enumerate((30, 55, 47)):
            toks = rng.integers(1, 3000, size=n_chunks * SEQUENCE_LENGTH + 77).astype(np.uint16)
            toks[rng.random(len(toks)) < 0.01] = EOS
            rel = f"preprocessed/mix/part-{i:03d}.npy"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            toks.tofile(root / rel)                    # headerless uint16, as OLMo writes them
            files[rel] = toks
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_RangeHandler, directory=str(root)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}/"
            dmap = Path(tmp) / "map.json"
            dmap.write_text(json.dumps({"recipes": {"paloma-test": {
                "paths": list(files), "base_url": base, "model_repo": "x/y",
                "n_instances": N_INSTANCES, "eos_token_id": EOS}}}))
            out, work = Path(tmp) / "out", Path(tmp) / "work"
            args = ["--data-map", str(dmap), "--recipes", "paloma-test", "--seed", "6198",
                    "--workers", "3", "--task-mtokens", "0", "--work-dir", str(work), "--out-dir", str(out)]
            assert ea.main(args + ["--no-stop-pod"]) == 0
        finally:
            server.shutdown()
        summary = json.loads((out / "exact_100b_paloma-test" / "summary.json").read_text())
        got = {int(r["number"]): int(r["count"]) for r in
               csv.DictReader(open(out / "exact_100b_paloma-test" / "counts_seed_6198.csv"))}

    sizes = [len(files[p]) for p in files]
    mult, _ = membership_codes(sizes, [6198], n_instances=N_INSTANCES)
    want, g = Counter(), 0
    for p in files:
        for c in range(len(files[p]) // SEQUENCE_LENGTH):
            m = int(mult[g])
            g += 1
            if m:
                for t in files[p][c * SEQUENCE_LENGTH:(c + 1) * SEQUENCE_LENGTH]:
                    if t != EOS and int(t) <= 10000:
                        want[int(t)] += m
    assert int(mult.sum()) == N_INSTANCES
    assert summary["per_seed"]["6198"]["tokens"] == N_INSTANCES * SEQUENCE_LENGTH
    assert {k: v for k, v in got.items() if v} == dict(want)


if __name__ == "__main__":
    test_http_recipe_with_its_own_eos_and_length_matches_brute_force()
    print("ok")
