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

N_INSTANCES = 150
REAL_READ_RANGE = ea.read_range  # other tests replace ea.read_range with in-memory fakes  # training budget in chunks (the real one is 48.8M)


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
    ea.file_sizes = lambda ps, work, base_url=None: [len(files[p]) for p in ps]
    ea.read_range = lambda task, start, length: files[task["path"]][start:start + length]
    ea.resolve_url = lambda path: "https://cdn.example/" + path
    ea._tokenizer = lambda repo: SimpleNamespace(backend_tokenizer=_Backend())
    ea.membership_codes = lambda ft, seeds, **kw: membership_codes(ft, seeds, n_instances=N_INSTANCES)

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


def test_usable_vcpus_and_range_reader():
    assert 1 <= ea.usable_vcpus() <= (__import__("os").cpu_count() or 1)

    class _Resp:
        def __init__(self, status, body=b"", headers=None):
            self.status_code, self.content, self.headers = status, body, headers or {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    data = np.arange(1000, dtype=np.uint16).tobytes()
    calls = []

    def fake_get(url, headers, stream, timeout):
        calls.append(url)
        lo, hi = map(int, headers["Range"].split("=")[1].split("-"))
        if len(calls) == 1:
            return _Resp(403)                       # expired CDN link -> back to the resolver
        if len(calls) == 2:
            return _Resp(429, headers={"Retry-After": "0"})  # rate limited -> retry
        return _Resp(206, data[lo:hi + 1])

    import requests
    import time as _time
    orig_get, orig_sleep = requests.get, _time.sleep
    requests.get, _time.sleep = fake_get, lambda s: None
    try:
        out = REAL_READ_RANGE({"path": "a/b.npy", "url": "https://cdn.example/x"}, 10, 5)
    finally:
        requests.get, _time.sleep = orig_get, orig_sleep
    assert out.tolist() == [10, 11, 12, 13, 14]
    assert calls[0].startswith("https://cdn.example") and "huggingface.co" in calls[1]

    requests.get = lambda url, headers, stream, timeout: _Resp(200, data)  # Range ignored
    try:
        REAL_READ_RANGE({"path": "a/b.npy", "url": "https://cdn.example/x"}, 0, 5)
        raise AssertionError("expected a refusal")
    except RuntimeError as exc:
        assert "ignored the Range" in str(exc)
    finally:
        requests.get = orig_get


def test_crash_then_resume_redoes_only_missing_slices():
    """First run: every 3rd download fails (network error / pod killed). Rerun:
    only those slices are downloaded again, and the result equals a clean run."""
    rng = np.random.default_rng(1)
    files = _fake_files(rng)
    paths = list(files)
    ea.load_data_map = lambda: {"recipes": {"fake": {"model_repo": "x/y", "paths": paths}}}
    ea.file_sizes = lambda ps, work, base_url=None: [len(files[p]) for p in ps]
    ea.resolve_url = lambda path: "https://cdn.example/" + path
    ea._tokenizer = lambda repo: SimpleNamespace(backend_tokenizer=_Backend())
    ea.membership_codes = lambda ft, seeds, **kw: membership_codes(ft, seeds, n_instances=N_INSTANCES)

    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "downloads.log"

        def reader(fail):
            def read(task, start, length):
                with open(log, "a") as fh:
                    fh.write(f"{task['path']}:{start}\n")
                if fail and (start // SEQUENCE_LENGTH) % 3 == 0:
                    raise RuntimeError("simulated network failure")
                return files[task["path"]][start:start + length]
            return read

        def run(out, work):
            return ea.main(["--recipes", "fake", "--workers", "3", "--task-mtokens", "0",
                            "--retries", "0", "--work-dir", str(work), "--out-dir", str(out)])

        ea.read_range = reader(fail=True)
        assert run(Path(tmp) / "out", Path(tmp) / "work") == 1          # some slices failed
        assert not (Path(tmp) / "out" / "exact_100b_fake" / "summary.json").exists()
        first = len(log.read_text().splitlines())
        failed = sum(1 for line in log.read_text().splitlines()
                     if (int(line.split(":")[1]) // SEQUENCE_LENGTH) % 3 == 0)
        log.unlink()
        ea.read_range = reader(fail=False)
        assert run(Path(tmp) / "out", Path(tmp) / "work") == 0          # resume
        redone = len(log.read_text().splitlines())
        assert redone == failed < first, (redone, failed, first)        # nothing downloaded twice
        resumed = json.loads((Path(tmp) / "out" / "exact_100b_fake" / "summary.json").read_text())
        assert run(Path(tmp) / "clean", Path(tmp) / "clean_work") == 0  # reference: one clean run
        clean = json.loads((Path(tmp) / "clean" / "exact_100b_fake" / "summary.json").read_text())
        assert resumed["per_seed"] == clean["per_seed"]


def test_retry_upload_verify_then_delete_pod():
    """One command: slices fail on the first attempt, the built-in retry finishes them,
    the result is uploaded, verified, and only then is the pod deleted. If the upload
    is missing, the pod is NOT deleted."""
    import shutil
    import time as _time

    import huggingface_hub

    rng = np.random.default_rng(2)
    files = _fake_files(rng)
    paths = list(files)
    ea.load_data_map = lambda: {"recipes": {"fake": {"model_repo": "x/y", "paths": paths}}}
    ea.file_sizes = lambda ps, work, base_url=None: [len(files[p]) for p in ps]
    ea.resolve_url = lambda path: "https://cdn.example/" + path
    ea._tokenizer = lambda repo: SimpleNamespace(backend_tokenizer=_Backend())
    ea.membership_codes = lambda ft, seeds, **kw: membership_codes(ft, seeds, n_instances=N_INSTANCES)

    with tempfile.TemporaryDirectory() as tmp:
        hub = Path(tmp) / "hub"
        flag = Path(tmp) / "failed_once"

        class FakeApi:
            def __init__(self, token=None):
                pass

            def whoami(self):
                return {"name": "me"}

            def create_repo(self, repo, **kw):
                (hub / repo).mkdir(parents=True, exist_ok=True)

            def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kw):
                (hub / repo_id / path_in_repo).parent.mkdir(parents=True, exist_ok=True)
                (hub / repo_id / path_in_repo).write_bytes(path_or_fileobj)

            def upload_folder(self, folder_path, path_in_repo, repo_id, **kw):
                shutil.copytree(folder_path, hub / repo_id / path_in_repo, dirs_exist_ok=True)

            def list_repo_files(self, repo, **kw):
                return [str(f.relative_to(hub / repo)) for f in (hub / repo).rglob("*") if f.is_file()]

        def fake_download(repo, f, **kw):
            return str(hub / repo / f)

        orig = (huggingface_hub.HfApi, huggingface_hub.hf_hub_download, ea.terminate_this_pod,
                ea.stop_this_pod, _time.sleep)
        deleted, stopped = [], []
        huggingface_hub.HfApi, huggingface_hub.hf_hub_download = FakeApi, fake_download
        ea.terminate_this_pod = lambda reason: deleted.append(reason)
        ea.stop_this_pod = lambda reason: stopped.append(reason)
        ea.time.sleep = lambda s: None

        def read(task, start, length):  # every 4th slice fails, but only on the first attempt
            if (start // SEQUENCE_LENGTH) % 4 == 0 and not (flag / f"{task['path']}_{start}").exists():
                flag.mkdir(exist_ok=True)
                (flag / f"{task['path'].replace('/', '_')}_{start}").touch()
                (flag / f"{task['path']}_{start}").parent.mkdir(parents=True, exist_ok=True)
                (flag / f"{task['path']}_{start}").touch()
                raise RuntimeError("simulated network failure")
            return files[task["path"]][start:start + length]

        ea.read_range = read
        try:
            args = ["--recipes", "fake", "--workers", "3", "--task-mtokens", "0", "--retries", "2",
                    "--work-dir", str(Path(tmp) / "work"), "--out-dir", str(Path(tmp) / "out"),
                    "--upload-hf", "results", "--delete-pod-when-done"]
            assert ea.main(args) == 0
            assert (hub / "me/results/exact_100b_fake/summary.json").exists()
            assert (hub / "me/results/exact_100b_fake/counts_seed_2.csv").exists()
            assert len(deleted) == 1 and not stopped, (deleted, stopped)

            # upload lost -> verification fails -> the pod is never deleted
            deleted.clear()
            FakeApi.upload_folder = lambda self, *a, **k: None
            shutil.rmtree(hub / "me/results/exact_100b_fake")
            shutil.rmtree(Path(tmp) / "out")
            ea.main(args)
            assert not deleted, deleted
        finally:
            (huggingface_hub.HfApi, huggingface_hub.hf_hub_download, ea.terminate_this_pod,
             ea.stop_this_pod, ea.time.sleep) = orig


def test_pods_share_work_through_claims():
    """Two recipes; another pod already claimed 'a'. This pod skips 'a', counts 'b',
    uploads it, verifies only its own recipe and deletes itself."""
    import shutil

    import huggingface_hub

    rng = np.random.default_rng(3)
    files_a, files_b = _fake_files(rng), _fake_files(rng)
    files = {**{f"a/{k}": v for k, v in files_a.items()}, **{f"b/{k}": v for k, v in files_b.items()}}
    dm = {"recipes": {"a": {"model_repo": "x/y", "paths": [p for p in files if p.startswith("a/")]},
                      "b": {"model_repo": "x/y", "paths": [p for p in files if p.startswith("b/")]}}}
    ea.load_data_map = lambda: dm
    ea.file_sizes = lambda ps, work, base_url=None: [len(files[p]) for p in ps]
    ea.resolve_url = lambda path: "https://cdn.example/" + path
    ea.read_range = lambda task, start, length: files[task["path"]][start:start + length]
    ea._tokenizer = lambda repo: SimpleNamespace(backend_tokenizer=_Backend())
    ea.membership_codes = lambda ft, seeds, **kw: membership_codes(ft, seeds, n_instances=N_INSTANCES)
    ea.CLAIM_SETTLE_S = 0

    with tempfile.TemporaryDirectory() as tmp:
        hub = Path(tmp) / "hub"

        class FakeApi:
            def __init__(self, token=None):
                pass

            def whoami(self):
                return {"name": "me"}

            def create_repo(self, repo, **kw):
                (hub / repo).mkdir(parents=True, exist_ok=True)

            def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kw):
                (hub / repo_id / path_in_repo).parent.mkdir(parents=True, exist_ok=True)
                (hub / repo_id / path_in_repo).write_bytes(path_or_fileobj)

            def upload_folder(self, folder_path, path_in_repo, repo_id, **kw):
                shutil.copytree(folder_path, hub / repo_id / path_in_repo, dirs_exist_ok=True)

            def list_repo_files(self, repo, **kw):
                return [str(f.relative_to(hub / repo)) for f in (hub / repo).rglob("*") if f.is_file()]

        orig = (huggingface_hub.HfApi, huggingface_hub.hf_hub_download, ea.terminate_this_pod)
        deleted = []
        huggingface_hub.HfApi = FakeApi
        huggingface_hub.hf_hub_download = lambda repo, f, **kw: str(hub / repo / f)
        ea.terminate_this_pod = lambda reason: deleted.append(reason)
        (hub / "me/results/_claims/a").mkdir(parents=True)
        (hub / "me/results/_claims/a/otherpod.txt").write_text("otherpod\n")
        try:
            out = Path(tmp) / "out"
            assert ea.main(["--recipes", "a,b", "--workers", "3", "--task-mtokens", "0",
                            "--work-dir", str(Path(tmp) / "work"), "--out-dir", str(out),
                            "--upload-hf", "results", "--delete-pod-when-done"]) == 0
            assert not (out / "exact_100b_a").exists()                       # left to the other pod
            assert (hub / "me/results/exact_100b_b/summary.json").exists()   # done and uploaded here
            assert len(deleted) == 1
        finally:
            huggingface_hub.HfApi, huggingface_hub.hf_hub_download, ea.terminate_this_pod = orig


if __name__ == "__main__":
    test_pods_share_work_through_claims()
    test_retry_upload_verify_then_delete_pod()
    test_crash_then_resume_redoes_only_missing_slices()
    test_usable_vcpus_and_range_reader()
    test_matches_brute_force()
    print("ok")
