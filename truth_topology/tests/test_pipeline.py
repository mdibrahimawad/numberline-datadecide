"""Smoke test: fake states for 4 checkpoints x 3 layers through every analysis stage, figures, report."""
import json

import numpy as np
import pandas as pd
import pytest

from tt import config as C


@pytest.fixture
def small(tmp_path, monkeypatch):
    for k, v in {"STATES": tmp_path / "states", "RESULTS": tmp_path / "results", "FIGURES": tmp_path / "fig",
                 "WORK": tmp_path, "B": 8, "M": 40, "N_NULL_T1": 2, "N_PERM_MST": 50, "N_PERM_CONC": 50,
                 "N_BOOT": 50, "N_NULL_LINEAR": 2, "NORM_REF_N": 100, "SEEDS_SUBSAMPLE": [0, 1]}.items():
        monkeypatch.setattr(C, k, v)
    rng = np.random.default_rng(0)
    n, d = 240, 32
    topics = np.repeat(["a", "b", "c"], n // 3)
    y = np.tile([0, 1], n // 2)
    it = pd.DataFrame({"item_id": range(n), "text": "x", "label": y, "topic": topics})
    (C.STATES / "P1").mkdir(parents=True)
    it.to_csv(C.STATES / "P1" / "items.csv", index=False)
    out = C.STATES / "m" / "P1"
    out.mkdir(parents=True)
    for i, step in enumerate([0, 1000, 40000, 143000]):
        H = rng.standard_normal((3, n, d))
        H[1:, :, 0] += i * (y - .5)              # truth grows with training at layers 1-2
        np.savez(out / f"step{step}.npz", H=H.astype(np.float16), loglik=rng.standard_normal(n) + i * y)
    return y, topics


def test_end_to_end(small):
    from tt import analyze as A, figures, report
    P = A.Paths("m", "P1")
    A.stage_cells(P, "m", "P1", n_jobs=1)
    A.stage_velocity(P, n_jobs=1)
    m, ref = A.stage_aggregate(P)
    A.stage_floor(P)
    assert ref in (1, 2)
    l1 = m[(m.metric == "L1_auc") & (m.layer == ref)].sort_values("step").value.to_numpy()
    assert l1[0] < 0.65 and l1[-1] > 0.9
    on = pd.read_csv(P.res / "onset.csv")
    assert set(on.metric) >= {"L1_auc", "T1_acc", "B1_loglik_auc"}
    v = pd.read_csv(P.res / "velocity.csv")
    assert len(v) == 3 * 3 * 2 * 3 * 2 * 2       # transitions x layers x seeds x kinds x hom x prep
    c = pd.read_csv(P.res / "concentration.csv")
    assert c.C.between(0, 1).all()
    assert (m.metric == "T3b_within_truth_homophily_maxz_bestlayer").sum() == 4
    vel = pd.read_csv(P.res / "velocity.csv")
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    figures.fig1(m, vel, ref, "m", C.FIGURES / "f1.png")
    figures.fig2(m, vel, "m", C.FIGURES / "f2.png")
    figures.fig3(m, ref, "m", C.FIGURES / "f3.png")
    assert (C.FIGURES / "f2.png").exists()
    # resumability: a second run does nothing and changes nothing
    A.stage_cells(P, "m", "P1", n_jobs=1)
