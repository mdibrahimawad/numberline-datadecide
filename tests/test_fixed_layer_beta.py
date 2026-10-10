"""Offline checks of src.fixed_layer_beta.

    python tests/test_fixed_layer_beta.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.fixed_layer_beta as flb  # noqa: E402


def test_data_predictors_use_the_locked_definitions():
    c = np.full(10001, 1e9)
    c[10:20] = 0                       # 10 never-seen numbers
    c[[0, 10000]] = 0                  # outside 10..9999: ignored
    out = flb.data_predictors(c)
    assert out["R"] == 10
    assert abs(out["S"] - (9990 - 10)) < 0.1


def test_per_layer_table_takes_the_median_over_seeds():
    with tempfile.TemporaryDirectory() as tmp:
        layers = lambda b: {"0": {"rho": 0.1, "ev": 0.1},                       # embedding: no beta
                            "8": {"beta_cont": b, "beta_coarse": b, "rho": 0.9, "ev": 0.3}}
        payload = {"recipe": "x", "per_seed": [{"layers": layers(b)} for b in (0.5, 0.7, 0.9)]}
        (Path(tmp) / "x.json").write_text(json.dumps(payload))
        t = flb.per_layer_table(Path(tmp))
    assert t.loc["x", "beta_cont_L8"] == 0.7 and np.isnan(t.loc["x", "beta_cont_L0"])


def test_frozen_line_recovers_a_known_line():
    x = pd.Series(np.arange(20.0), index=[f"m{i}" for i in range(20)])
    y = 0.5 + 0.1 * x + np.random.default_rng(0).normal(0, 1e-3, 20)
    line = flb.frozen_line(x, y)
    assert abs(line["slope"] - 0.1) < 1e-3 and abs(line["intercept"] - 0.5) < 1e-2 and line["loo_r"] > 0.99


if __name__ == "__main__":
    test_data_predictors_use_the_locked_definitions()
    test_per_layer_table_takes_the_median_over_seeds()
    test_frozen_line_recovers_a_known_line()
    print("ok")
