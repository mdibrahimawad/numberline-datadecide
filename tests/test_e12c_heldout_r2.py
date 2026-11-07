"""E12c: model ranking on synthetic data with a smooth part and a last-digit part."""
import numpy as np

from src.e12b_smooth_digits import digit_design
from src.e12c_heldout_r2 import evaluate


def test_ranking_on_synthetic():
    rng = np.random.default_rng(0); n = np.arange(100, 1000); t = (n - 100) / 900
    Q = np.linalg.qr(rng.standard_normal((60, 3)))[0]
    smooth = np.c_[np.cos(2 * t), np.sin(2 * t), t] @ Q.T
    Xd, _ = digit_design(n, [0]); digits = Xd @ (0.5 * rng.standard_normal((10, 60)))
    X = smooth + digits + 0.05 * rng.standard_normal((900, 60))
    te = rng.random(900) < 0.2
    out = evaluate(X, n, ~te, te, [0, 1], seed=0)
    assert out["R2_curve_digits"] > out["R2_curve"] + 0.1
    assert out["R2_surface2d"] > out["R2_curve"] + 0.1
    assert out["R2_shuffled"] < 0.05
