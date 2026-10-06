"""Offline checks of the decade data-beta metrics (python tests/test_decade_metrics.py)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.decade_metrics import power_beta, sat_beta  # noqa: E402


def test_limits_and_scaling():
    n = np.arange(10001, dtype=float)
    counts = 1e9 * np.maximum(n, 1) ** -1.5
    assert abs(power_beta(counts, 0.0) - 10.0) < 1e-9          # equal space per number
    assert abs(sat_beta(counts, 1e-9) - 10.0) < 1e-6           # everything saturated
    assert abs(sat_beta(counts, 1e15) - power_beta(counts, 1.0)) < 1e-4  # K -> inf = infomax
    for q in (0.3, 1.0):                                         # power metric is scale-free
        assert abs(power_beta(counts, q) - power_beta(7 * counts, q)) < 1e-12
    # more numbers seen -> saturation beta rises toward 10
    assert sat_beta(counts, 1e4) < sat_beta(10 * counts, 1e4) < 10


if __name__ == "__main__":
    test_limits_and_scaling()
    print("ok")
