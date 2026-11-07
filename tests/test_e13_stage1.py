"""Synthetic checks for src/e13_stage1.py (run before the real Stage 1 run)."""
import numpy as np

from src import e13_stage1 as E


def _codes(n, L, kinds, scale, D=50, seed=0, noise=1.0):
    rng = np.random.default_rng(seed); dig = E.digits(n, L); Z = noise * rng.standard_normal((len(n), D))
    for p, kind in enumerate(kinds):
        lv = np.unique(dig[:, p]); T = E.planted_template(lv, kind, scale, D, rng)
        Z += T[np.searchsorted(lv, dig[:, p])]
    return Z


def test_trailing_nines_and_steps():
    assert list(E.trailing_nines(np.array([18, 19, 99, 199, 1999, 109]))) == [0, 1, 2, 2, 3, 1]
    i, j, c = E.steps_of(np.arange(100, 1000))
    assert len(i) == 899 and (c == 2).sum() == 8 and (c == 1).sum() == 81


def test_h1_recovers_planted_codes():
    n = np.arange(100, 1000)
    Z = _codes(n, 3, ["line", "circle", "simplex"], scale=4.0)
    r = E.h1(Z, n, 3, "test", B=50, n_perm=50, n_rep=4)
    t = [p["template"] for p in r["positions"]]
    assert t == ["line", "circle", "simplex"], t
    assert r["positions"][0]["O1"] > 0.9 and abs(r["positions"][2]["O1"]) < 0.4
    # debiasing: pure-noise centroids give near-zero signal
    Zn = np.random.default_rng(1).standard_normal((len(n), 50))
    st = E.h1(Zn, n, 3, "noise", B=20, n_perm=50, n_rep=2)
    assert all(abs(p["V"]) < 0.01 for p in st["positions"])


def test_h1_decay_rule():
    n = np.arange(100, 1000)
    Z = _codes(n, 3, ["line", "line", "simplex"], scale=4.0, seed=2)
    r = E.h1(Z, n, 3, "decay", B=100, n_perm=20, n_rep=5)
    assert r["decay_pass"] and r["decay_power"] >= 0.8 and r["decay_fp"] <= 0.2


def test_h2a_detects_jumps():
    n = np.arange(100, 1000); rng = np.random.default_rng(3); D = 50
    base = np.cumsum(rng.standard_normal((len(n), D)) * 0.05, 0) + rng.standard_normal((len(n), D))
    lvl = E.trailing_nines(n); J = np.zeros_like(base); cum = np.zeros(D)
    i, j, c = E.steps_of(n); m0 = np.median(np.linalg.norm(base[j[c == 0]] - base[i[c == 0]], axis=1))
    for k in range(len(n)):                      # pre-registered planted size: kappa_j = 0.5 * j * m0
        J[k] = cum
        if lvl[k] >= 1:
            u = rng.standard_normal(D); cum = cum + 0.5 * lvl[k] * m0 * u / np.linalg.norm(u)
    a = E.h2a(base + J, n, 3, "j", i, j, c, B=200, n_perm=50)
    b = E.h2a(base, n, 3, "nj", i, j, c, B=200, n_perm=50)
    assert not b["pass"] and a["e_lo"][0] > 0      # level-1 jump seen; level 2 absorbed by digits


def test_h2c_tree_vs_line():
    n = np.arange(100, 1000); rng = np.random.default_rng(4); D = 50
    lead = n // 100; pre2 = n // 10
    tree = rng.standard_normal((10, D))[lead] * 2 + rng.standard_normal((100, D))[pre2]
    noise = rng.standard_normal((len(n), D))
    line = np.outer((n - 550) / 450, rng.standard_normal(D)) * 6
    zp = lambda A, idx: E.prep(A, idx, D=A.shape[1])
    t = E.h2c(tree + noise, n, 3, "t", prep_fn=zp, B=100, light=True)
    l = E.h2c(line + noise, n, 3, "l", prep_fn=zp, B=100, light=True)
    assert t["dR2g"] > 0.1 and l["dR2g"] < 0.1, (t["dR2g"], l["dR2g"])


def test_h2a_transient_vs_cumulative():
    """Amendment 1: transient carry events are detected; cumulative jumps at level 2 are absorbed by digit effects."""
    n = np.arange(100, 1000); rng = np.random.default_rng(3); D = 50
    base = np.cumsum(rng.standard_normal((len(n), D)) * 0.05, 0) + rng.standard_normal((len(n), D))
    i, j, c = E.steps_of(n); m0 = np.median(np.linalg.norm(base[j[c == 0]] - base[i[c == 0]], axis=1))
    Jt = np.zeros_like(base)
    for lv in (1, 2):
        for k in j[c == lv]:
            u = rng.standard_normal(D); Jt[k] = 0.5 * lv * m0 * u / np.linalg.norm(u)
    a = E.h2a(base + Jt, n, 3, "t", i, j, c, B=300, n_perm=50)
    assert a["pass"], (a["e"], a["e_lo"], a["diff_lo"])
