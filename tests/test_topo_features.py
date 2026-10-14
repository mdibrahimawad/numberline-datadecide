"""Pins the E10 definitions: the 41 barcode features and the matched frequency tiers."""
import numpy as np
import pytest

pytest.importorskip("ripser")

from src.topo_features import FEATURE_NAMES, barcode_features, diagrams


def test_41_features_and_names_fixed():
    assert len(FEATURE_NAMES) == 41
    assert FEATURE_NAMES[:3] == ["H0_count", "H0_death_mean", "H0_death_std"]
    assert "H1_entropy" in FEATURE_NAMES and "H1_ratio_mean" in FEATURE_NAMES


def test_circle_has_one_long_loop_and_scale_division():
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    pts = np.c_[np.cos(t), np.sin(t)] * 10
    f1 = barcode_features(diagrams(pts, scale=1.0))
    f10 = barcode_features(diagrams(pts, scale=10.0))
    assert f1["H1_count"] == 1 and f1["H1_life_max"] > 10       # exactly one loop, long-lived
    assert f1["H0_count"] == 59                                    # n - 1 finite H0 bars
    assert f10["H1_life_max"] == pytest.approx(f1["H1_life_max"] / 10, rel=1e-6)
    assert f10["H1_ratio_mean"] == pytest.approx(f1["H1_ratio_mean"], rel=1e-6)   # scale-free


def test_matched_tiers_balance_blocks(tmp_path, monkeypatch):
    import pandas as pd
    import src.e10_explore as e

    rng = np.random.default_rng(0)
    counts = pd.DataFrame({"number": np.arange(10001), "count": rng.integers(1, 10**6, 10001)})
    path = tmp_path / "c.csv"; counts.to_csv(path, index=False)
    monkeypatch.setattr(e, "COUNTS", str(path))
    cnt = e.load_counts()
    for control in ("block", "block_digits"):
        mt = e.matched_tiers(cnt, control)
        assert not ((mt.n >= 1500) & (mt.n <= 2099)).any() and (mt.n % 10 != 0).all()
        per = mt.groupby(["block", "tier"]).size()
        assert (per == 30).all()                                    # 30 per tier in every block
        hi, lo = mt[mt.tier == 2], mt[mt.tier == 0]
        assert hi.key.mean() > lo.key.mean()
