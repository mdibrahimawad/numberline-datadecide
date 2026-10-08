"""Offline checks of src.count_features (synthetic counts; no data files).

    python tests/test_count_features.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.count_features as cf  # noqa: E402


def _power_law(alpha: float = 1.5, total: float = 1e9) -> np.ndarray:
    n = np.arange(10001, dtype=float)
    c = np.zeros(10001)
    c[1:] = n[1:] ** -alpha
    return np.round(c / c.sum() * total)


def test_rarely_seen_count_is_the_locked_definition():
    """R = #{n in 10..9999 : c(n) < 4000}; 0..9 and 10000 never count, 4000 itself is not rare."""
    c = np.full(10001, 10_000.0)
    c[[10, 500, 9999]] = 3999          # rare
    c[[11, 600]] = 4000                # exactly K: not rare
    c[[0, 5, 10000]] = 0               # outside 10..9999: ignored
    assert cf.features(c)["F14_n_seen_lt_4000"] == 3


def test_more_data_same_shape_means_fewer_rare_numbers():
    """Same alpha, more numbers seen -> lower R (the c4 vs dolma-no-code situation)."""
    small, big = _power_law(total=1e9), _power_law(total=2e9)
    f_small, f_big = cf.features(small), cf.features(big)
    assert f_big["F14_n_seen_lt_4000"] < f_small["F14_n_seen_lt_4000"]
    assert abs(f_big["F09_slope_100_999"] - f_small["F09_slope_100_999"]) < 0.01   # shape unchanged


def test_local_slope_recovers_alpha_and_round_numbers_are_detected():
    c = _power_law(alpha=1.5, total=1e12)
    f = cf.features(c)
    assert abs(f["F09_slope_100_999"] + 1.5) < 0.01, f["F09_slope_100_999"]
    assert abs(f["F19_round100_excess"]) < 0.05                      # smooth curve: no spikes
    c[np.arange(200, 10000, 100)] *= 10
    assert cf.features(c)["F19_round100_excess"] > 2                 # log(10) ~ 2.3


if __name__ == "__main__":
    test_rarely_seen_count_is_the_locked_definition()
    test_more_data_same_shape_means_fewer_rare_numbers()
    test_local_slope_recovers_alpha_and_round_numbers_are_detected()
    print("ok")
