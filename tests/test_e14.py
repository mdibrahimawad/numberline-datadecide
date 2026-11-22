"""E14 measures on exact (noise-free) curves."""
import numpy as np

from src import e14_length_log as E

n = np.arange(10, 10000).astype(float)
w = 1.0 / np.bincount(E.nlen(n))[E.nlen(n)]


def test_smooth_log():
    m = E.measures(np.log10(n), n, w, squeeze=True)
    assert abs(m["LS"]) < 1e-6 and abs(m["J"]) < 1e-6 and m["lam"] == 0.0 and 3.5 < m["R_3"] < 6


def test_staircase():
    m = E.measures(E.nlen(n) + 1e-6 * np.log10(n), n, w)
    assert m["LS"] > 0.99 and m["J"] > 0.9


def test_linear_inside():
    L = E.nlen(n); m = E.measures((L - 2) + (n / 10.0 ** (L - 1) - 1) / 9, n, w, squeeze=True)
    assert abs(m["lam"] - 1.0) < 0.06 and abs(m["R_3"] - 1) < 0.05 and m["LS"] < 0.15


def test_mixed_share():
    m = E.measures(0.375 * (E.nlen(n) - 2) + 0.625 * np.log10(n), n, w)
    assert abs(m["LS"] - 0.375) < 1e-6


def test_classify():
    assert E.classify((0.6, 0.9), (0, 1)) == "length staircase"
    assert E.classify((-0.1, 0.2), (-0.2, 0.3)) == "smooth, log inside"
    assert E.classify((-0.1, 0.2), (0.7, 1.3)) == "smooth, linear inside"
    assert E.classify((0.2, 0.6), (0, 1)) == "mixed"
    assert E.classify((-0.5, 0.9), (0, 1)) == "can't tell"
    assert E.classify((0.6, 0.9), (0, 1), rho=0.3) == "no number line"
