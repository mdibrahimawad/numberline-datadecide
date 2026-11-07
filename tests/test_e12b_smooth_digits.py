"""E12b: backfitting recovers a known smooth part and digit part (synthetic data only)."""
import numpy as np

from src.e12_manifold import id_mle, plateau
from src.e12b_smooth_digits import choose_h, digit_design, fit_additive, shares


def test_backfit_recovers_parts():
    rng = np.random.default_rng(0); n = np.arange(100, 1000); t = (n - 100) / 900
    Q = np.linalg.qr(rng.standard_normal((30, 4)))[0]
    f_true = np.c_[np.cos(2 * t), np.sin(2 * t), t, np.sin(4 * t)] @ Q.T
    X, _ = digit_design(n, [0, 1]); D_true = X @ (0.3 * rng.standard_normal((20, 30)))
    Z = f_true + D_true + 0.02 * rng.standard_normal(f_true.shape)
    h, frac, _ = choose_h(Z, n, [0, 1])
    assert frac >= 0.01
    f, D, _ = fit_additive(Z, n, h, [0, 1])
    rel = lambda a, b: np.linalg.norm((a - a.mean(0)) - (b - b.mean(0))) / np.linalg.norm(b - b.mean(0))
    assert rel(f, f_true) < 0.15 and rel(D, D_true) < 0.15
    s = shares(Z, f, D, Z - f - D)
    assert 0.0 < s["share_f"] < 1.0 and s["r2_f_plus_digit"] > 0.95
    assert plateau(id_mle(f))
