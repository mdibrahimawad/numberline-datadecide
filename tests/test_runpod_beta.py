"""Offline check of runpod_jobs/beta.py with a fake worker and a fake Hugging Face:
full repo ids (e.g. the Paloma baselines) must map to slug-named outputs end to end.

    python tests/test_runpod_beta.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runpod_jobs.beta as rb  # noqa: E402


class _FakeProc:
    """Stands in for `python -m src.beta_fine`: writes <slug>.json/.npz and exits 0."""

    def __init__(self, cmd, stdout=None, stderr=None):
        model = cmd[cmd.index("--model") + 1]
        out = Path(cmd[cmd.index("--out-dir") + 1])
        slug = rb.recipe_slug(model)
        (out / f"{slug}.json").write_text(json.dumps({"recipe": slug, "report": {}}))
        (out / f"{slug}.npz").write_bytes(b"npz")
        self.returncode = 0

    def poll(self):
        return 0


def test_full_repo_ids_use_slug_file_names_everywhere():
    uploads, files_on_hub = [], set()

    def fake_upload(api, repo, files, prefix):
        assert all(f.exists() for f in files), files      # the uploaded files are the ones written
        uploads.append([f.name for f in files])
        files_on_hub.update(f"{prefix}/{f.name}" for f in files)
        return True

    class FakeApi:
        def list_repo_files(self, repo, repo_type):
            return sorted(files_on_hub)

    orig = (subprocess.Popen, rb._upload, rb._hf, rb._prefetch, rb._free_cache, rb.terminate_this_pod)
    deleted = []
    rb.subprocess.Popen = _FakeProc
    rb._upload = fake_upload
    rb._hf = lambda repo: (FakeApi(), "me/" + repo)
    rb._prefetch = lambda *a, **k: None
    rb._free_cache = lambda repo_id: None
    rb.terminate_this_pod = lambda reason: deleted.append(reason)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            models = "allenai/paloma-1b-baseline-c4,allenai/paloma-1b-baseline-pile"
            argv = ["--fine", "--models", models, "--revisions", "main", "--out-root", tmp,
                    "--upload-hf", "x", "--delete-pod-when-done", "--parallel", "2"]
            assert rb.main(argv) == 0
            out = Path(tmp) / "main"
            assert (out / "allenai_paloma-1b-baseline-c4.json").exists()
            assert (out / "logs" / "allenai_paloma-1b-baseline-pile.log").exists()
            assert sorted(u[0] for u in uploads[:2]) == ["allenai_paloma-1b-baseline-c4.json",
                                                         "allenai_paloma-1b-baseline-pile.json"]
            assert deleted, "pod should be deleted once both models are verified on the hub"
            # a rerun finds both outputs and has nothing left to do
            assert rb.main(["--fine", "--models", models, "--revisions", "main", "--out-root", tmp,
                            "--dry-run"]) == 0
    finally:
        (rb.subprocess.Popen, rb._upload, rb._hf, rb._prefetch, rb._free_cache, rb.terminate_this_pod) = orig


if __name__ == "__main__":
    test_full_repo_ids_use_slug_file_names_everywhere()
    print("ok")
