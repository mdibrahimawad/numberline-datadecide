"""E12 tests on synthetic shapes with known answers (no real data)."""
import numpy as np

from src.e12_manifold import id_mle, matched_smooth, plateau, principal_curve, t3_order, t7_helix, t8_topology


def _embed(Y, D=50, seed=0):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, Y.shape[1])))
    return Y @ Q.T


def test_dimension_curve_vs_sheet():
    rng = np.random.default_rng(1); t = np.sort(rng.uniform(0, 1, 900))
    curve = _embed(np.c_[np.cos(3 * t), np.sin(3 * t), t, np.sin(5 * t)]) + 1e-3 * rng.standard_normal((900, 50))
    sheet = _embed(np.c_[t, rng.uniform(0, 1, 900)]) + 1e-3 * rng.standard_normal((900, 50))
    assert plateau(id_mle(curve)) and not plateau(id_mle(sheet))
    assert 1.6 < id_mle(sheet)[20] < 2.6


def test_principal_curve_recovers_order():
    rng = np.random.default_rng(2); n = np.arange(100, 1000)
    t = (n - 100) / 900; Z = _embed(np.c_[np.cos(2.5 * t), np.sin(2.5 * t), t]) + 0.01 * rng.standard_normal((900, 50))
    out = t3_order(Z, n, seed=0)
    assert out["T3_random_rho"] > 0.95 and out["T3_block_rho"] > 0.9


def test_helix_period_found():
    rng = np.random.default_rng(3); n = np.arange(100, 1000); t = (n - 100) / 900
    Z = _embed(np.c_[t, 0.05 * np.cos(2 * np.pi * n / 10), 0.05 * np.sin(2 * np.pi * n / 10)]) + 0.01 * rng.standard_normal((900, 50))
    out = t7_helix(Z, n, seed=0, n_perm=50)
    assert out["T7_pval_p10"] < 0.05 and out["T7_share_p10"] > 0.1 and out["T7_pval_p5"] > 0.05


def test_loop_found_and_line_not():
    a = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    assert t8_topology(_embed(np.c_[np.cos(a), np.sin(a)]))["T8_long_H1"] >= 1
    assert t8_topology(_embed(np.c_[np.linspace(0, 1, 60), np.zeros(60)]))["T8_long_H1"] == 0


def test_matched_smooth_keeps_variance():
    rng = np.random.default_rng(4); n = np.arange(10, 100)
    Z = np.cumsum(rng.standard_normal((90, 5)), 0)
    G = matched_smooth(Z, n, rng)
    assert G.shape == Z.shape and np.allclose(G.var(0), Z.var(0), rtol=1e-6)
