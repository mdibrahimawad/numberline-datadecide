"""The pre-run sanity check passes every real DataDecide β result and refuses a broken one.

    python tests/test_check_beta.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runpod_jobs.check_beta import check  # noqa: E402

RUN = Path(__file__).resolve().parent.parent / "results/beta_fine/step69369-seed-default"


def test_all_datadecide_models_pass():
    files = sorted(RUN.glob("*.json"))
    assert len(files) == 25
    assert all(check(f)[0] for f in files)


def test_a_model_without_a_number_line_fails():
    d = json.loads((RUN / "c4.json").read_text())
    for s in d["per_seed"]:                       # what random weights look like: no ordering
        for layer in s["layers"].values():
            layer["rho"] = 0.1
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "broken.json"
        p.write_text(json.dumps(d))
        assert not check(p)[0]


if __name__ == "__main__":
    test_all_datadecide_models_pass()
    test_a_model_without_a_number_line_fails()
    print("ok")
