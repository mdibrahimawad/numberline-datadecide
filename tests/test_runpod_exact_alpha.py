"""Offline check of runpod_jobs.exact_alpha on fake .npy files and a fake tokenizer.

    python tests/test_runpod_exact_alpha.py
"""

import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runpod_jobs.exact_alpha as ea  # noqa: E402
from src.datadecide_sampling import EOS_TOKEN_ID, SEQUENCE_LENGTH, membership_codes  # noqa: E402

N_INSTANCES = 150  # training budget in chunks (the real one is 48.8M)


class _Backend:
    def decode_batch(self, batch, skip_special_tokens=True):
        return [" ".join(str(int(t)) for t in ids if t != EOS_TOKEN_ID) for ids in batch]


def _fake_files(rng):
    files = {}
    for i, n_chunks in enumerate((40, 75, 61)):
        toks = rng.integers(0, 3000, size=n_chunks * SEQUENCE_LENGTH + 100).astype(np.uint16)
        toks[rng.random(len(toks)) < 0.01] = EOS_TOKEN_ID
        files[f"fake/part-{i}.npy"] = toks
    return files


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    files = _fake_files(rng)
    paths = list(files)
    data_map = {"recipes": {"fake": {"model_repo": "x/y", "paths": paths}}}
    ea.load_data_map = lambda: data_map
    ea.file_sizes = lambda ps, work: [len(files[p]) for p in ps]
    ea._read_tokens = lambda path, start, length: files[path][start:start + length]
    ea._tokenizer = lambda repo: SimpleNamespace(backend_tokenizer=_Backend())
    ea.membership_codes = lambda ft, seeds: membership_codes(ft, seeds, n_instances=N_INSTANCES)

    with tempfile.TemporaryDirectory() as tmp:
        out, work = Path(tmp) / "out", Path(tmp) / "work"
        args = ["--recipes", "fake", "--seed", "2", "--workers", "3", "--task-mtokens", "0",
                "--work-dir", str(work), "--out-dir", str(out)]
        assert ea.main(args) == 0
        summary = json.loads((out / "exact_100b_fake" / "summary.json").read_text())
        got = {int(r["number"]): int(r["count"])
               for r in csv.DictReader(open(out / "exact_100b_fake" / "counts_seed_2.csv"))}
        # rerun: everything comes from the slice cache, same answer
        assert ea.main(args) == 0
        again = json.loads((out / "exact_100b_fake" / "summary.json").read_text())
        assert again["per_seed"] == summary["per_seed"]
        rows = list(csv.DictReader(open(out / "alpha_seed2.csv")))
        assert [r["recipe"] for r in rows] == ["fake"]

    # brute force: every chunk weighted by how often seed 2's run used it
    sizes = [len(files[p]) for p in paths]
    mult, _ = membership_codes(sizes, [2], n_instances=N_INSTANCES)
    want, g = Counter(), 0
    for p in paths:
        for c in range(len(files[p]) // SEQUENCE_LENGTH):
            m = int(mult[g])
            g += 1
            if not m:
                continue
            seq = files[p][c * SEQUENCE_LENGTH:(c + 1) * SEQUENCE_LENGTH]
            for t in seq:
                if t != EOS_TOKEN_ID and int(t) <= 10000:
                    want[int(t)] += m
    assert summary["per_seed"]["2"]["tokens"] == int(mult.sum()) * SEQUENCE_LENGTH
    assert {k: v for k, v in got.items() if v} == dict(want)


if __name__ == "__main__":
    test_matches_brute_force()
    print("ok")
