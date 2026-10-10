import numpy as np

from tt import topology as T
from tt.mst import mst_edges


def test_summary_length_and_empty():
    assert len(T.summary([np.empty((0, 2)), np.empty((0, 2))])) == 41


def test_h0_deaths_are_mst_edges():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 5))
    D = T.distances(X)
    h0 = np.sort(T.diagrams(D)[0][:, 1])
    e = mst_edges(X)
    assert np.allclose(h0, np.sort(D[e[:, 0], e[:, 1]]))


def test_circle_has_one_long_bar():
    th = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    X = np.c_[np.cos(th), np.sin(th)]
    h1 = T.diagrams(T.distances(X))[1]
    assert (h1[:, 1] - h1[:, 0]).max() > 1.0


def test_wasserstein2():
    a = np.array([[0.0, 2.0]])
    assert T.wasserstein2(a, a) == 0
    assert np.isclose(T.wasserstein2(a, np.empty((0, 2))), 1.0)          # to the diagonal: (2-0)/2
    b = np.array([[0.0, 2.5]])
    assert np.isclose(T.wasserstein2(a, b), 0.5)                         # L-inf ground metric
    c = np.array([[0.0, 2.0], [1.0, 1.2]])
    assert np.isclose(T.wasserstein2(a, c), 0.1)


def test_stratified_subsamples_proportions():
    rng = np.random.default_rng(0)
    strata = np.repeat([0, 1, 2], [100, 50, 50])
    S = T.stratified_subsamples(strata, np.arange(200), 4, 40, rng)
    assert S.shape == (4, 40)
    assert all(np.bincount(strata[s], minlength=3).tolist() == [20, 10, 10] for s in S)
    assert all(len(set(s)) == 40 for s in S)
