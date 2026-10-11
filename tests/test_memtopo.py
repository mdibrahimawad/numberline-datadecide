"""Offline checks of memtopo (E10): topology, diagram distances, statistics, the sharded token
reader + epoch-boundary search on synthetic shards, B4 statistics, and a smoke run of the
analysis on synthetic features. python -m pytest -q tests/test_memtopo.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pytest.importorskip("ripser")
from memtopo import pile, stats, textstats, topology as topo  # noqa: E402
from memtopo.config import layers_for  # noqa: E402


# ---------------------------------------------------------------------- topology
def test_summary_and_normalisation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 128))
    f = topo.summary(topo.diagrams(topo.normalise(x)))
    assert f.shape == (41,) and np.isfinite(f).all()
    named = dict(zip(topo.FEATURE_NAMES, f))
    assert named["h0_n_bars"] == 63                      # 64 points, infinite bar dropped
    assert np.allclose(f, topo.summary(topo.diagrams(topo.normalise(5 * x - 2))))
    # empty H1 gives zeros, not NaN
    line = np.c_[np.arange(64.0), np.zeros(64)]
    g = dict(zip(topo.FEATURE_NAMES, topo.summary(topo.diagrams(line))))
    assert g["h1_n_bars"] == 0 and g["h1_pers_mean"] == 0


def test_circle_has_one_long_bar():
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, 64)
    c = np.c_[np.cos(th), np.sin(th)] + 0.03 * rng.standard_normal((64, 2))
    h1 = topo.diagrams(topo.normalise(c))[1]
    p = np.sort(h1[:, 1] - h1[:, 0])[::-1]
    assert p[0] > 0.5 and (len(p) == 1 or p[0] > 5 * p[1])


def test_wasserstein():
    a = np.array([[0.0, 1.0], [0.2, 0.5]])
    assert topo.wasserstein(a, a) == 0
    # one point vs empty: distance to the diagonal
    assert np.isclose(topo.wasserstein(np.array([[0.0, 1.0]]), np.zeros((0, 2)), p=2), 1 / np.sqrt(2))
    b = a + 0.01
    assert topo.wasserstein(a, b, p=2) <= topo.wasserstein(a, b, p=1) + 1e-12
    assert np.isclose(topo.wasserstein(a, b), topo.wasserstein(b, a))


def test_geometry():
    x = np.random.default_rng(2).standard_normal((64, 16))
    g = dict(zip(topo.GEOM_NAMES, topo.geometry(x)))
    assert 1 <= g["participation_ratio"] <= 16 and 1 <= g["effective_rank"] <= 16
    assert np.isclose(g["total_var"], np.var(x, axis=0, ddof=1).sum())


def test_layers_for():
    assert layers_for(16, [3, 5, 8, 10, 13, 16]) == [3, 5, 8, 10, 13, 16]
    assert layers_for(24, [3, 5, 8, 10, 13, 16]) == [5, 8, 12, 15, 20, 24]
    assert layers_for(32, [3, 5, 8, 10, 13, 16]) == [6, 10, 16, 20, 26, 32]


# ---------------------------------------------------------------------- statistics
def test_stats():
    rng = np.random.default_rng(3)
    y = np.r_[np.ones(200), np.zeros(800)].astype(int)
    s = y + rng.standard_normal(1000)
    from sklearn.metrics import average_precision_score, roc_auc_score
    assert np.isclose(stats.auroc(y, s), roc_auc_score(y, s))
    assert np.isclose(stats.auprc(y, s), average_precision_score(y, s))
    w = np.where(y == 1, 1.0, 3.0)
    assert np.isclose(stats.auprc(y, s, w), average_precision_score(y, s, sample_weight=w))
    d, pval = stats.delong(y, s, s + 0.3 * rng.standard_normal(1000))
    assert d > 0 and 0 <= pval <= 1
    assert np.isclose(stats.cliffs_delta([2, 3], [0, 1]), 1.0)
    assert abs(stats.hedges_g(rng.standard_normal(500) + 1, rng.standard_normal(500)) - 1) < 0.2
    assert 0 <= stats.recall_at_precision(y, s, w, 0.9) <= 1


# ---------------------------------------------------------------------- pile reader
def _make_store(tmp_path, seq_len=40, batch=64, n_steps=100, boundary=60, seed=0):
    rng = np.random.default_rng(seed)
    epoch1 = rng.integers(0, 50000, size=(boundary * batch, seq_len), dtype=np.uint16)
    # epoch 2 = the same sequences again, reshuffled
    epoch2 = epoch1[rng.permutation(len(epoch1))][: (n_steps - boundary) * batch]
    flat = np.concatenate([epoch1, epoch2]).ravel().tobytes()
    cuts = [0, 123_456, 333_334, 400_002, len(flat)]          # odd-sized but even-byte shards
    shards, d = [], tmp_path / "shards"
    d.mkdir()
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        name = f"document-{i:05d}-of-{len(cuts) - 1:05d}.bin"
        (d / name).write_bytes(flat[a:b])
        shards.append(pile.Shard(name, b - a, a))
    store = pile.TokenStore(shards, seq_len, batch, 2, d, None)
    return store, np.concatenate([epoch1, epoch2])


def test_token_store_reads(tmp_path):
    store, arr = _make_store(tmp_path)
    assert pile.check_layout(store)["whole_sequences"]
    idx = np.array([0, 5, 1543, 3000, 4000, len(arr) - 1])
    assert (store.read_many(idx, 16) == arr[idx, :16]).all()
    for i in (1543, 3000, 4000):                              # includes shard-straddling ones
        assert (store.read(i) == arr[i]).all()
    assert (store.read_step(7) == arr[7 * 64:8 * 64]).all()


def test_epoch_boundary(tmp_path):
    store, _ = _make_store(tmp_path)
    ref = pile.build_reference(store, range(0, 30), k=8)
    res = pile.find_epoch_boundary(store, ref, 8, start=40, stop=100, every=7, baseline_steps=[35, 50])
    assert res["found"] and res["boundary_step"] == 60
    assert res["primary_target_T"] == 0                       # (60 // 1000) * 1000 for this tiny run


# ---------------------------------------------------------------------- B4
def test_textstats():
    assert textstats.longest_repeated_ngram(np.array([1, 2, 3, 1, 2, 3, 9])) == 3
    assert textstats.longest_repeated_ngram(np.arange(10)) == 0
    assert textstats.longest_repeated_ngram(np.array([5, 5, 5, 5])) == 3   # overlapping repeats
    lf = np.log(np.full(100, 0.01))
    r = textstats.b4_row(np.arange(64) % 50, "abc " * 30, lf, 32)
    assert len(r) == 4 and 0 < r[0] < 1


# ---------------------------------------------------------------------- analysis smoke run
def test_analysis_smoke(tmp_path):
    from memtopo import analysis
    from memtopo.config import load_config, paths

    cfg = load_config(out_root=str(tmp_path / "run"))
    cfg["analysis"].update(bootstrap=50, bal128_draws=3, lr_Cs=[0.1, 1.0], inner_cv_folds=3, probe_pca=8)
    cfg["case_control"]["seeds"] = [0]
    p = paths(cfg)
    rng = np.random.default_rng(0)
    n, sf = 3000, 10000
    idx = np.arange(n) + 1_100_000
    split = np.where(np.arange(n) < 1800, "train", "test")
    y = (rng.random(n) < 0.15).astype(int)
    lab = pd.DataFrame({"index": idx, "split": split, "mem_T": y, "mem_final": y,
                        f"mem_S{sf}": (y & (rng.random(n) < 0.3)).astype(int)})
    f = p.features("1b", sf)
    f.mkdir(parents=True)
    lab.to_parquet(f / "labels.parquet")
    loss = pd.DataFrame({"index": idx, "mem_Sf": lab[f"mem_S{sf}"]})
    for c in analysis.B1 + analysis.B2 + analysis.B3:
        loss[c] = rng.standard_normal(n) + 0.3 * y
    loss.to_parquet(f / "loss.parquet")
    pd.DataFrame({"index": idx, **{c: rng.standard_normal(n) for c in analysis.B4}}).to_parquet(f / "data.parquet")
    layers = [3, 16]

    def wide(prefix, names):
        return pd.DataFrame({"index": idx, **{f"{prefix}{nm}_L{l}": rng.standard_normal(n) + 0.5 * y
                                              for nm in names for l in layers}})
    wide("geom_", topo.GEOM_NAMES).to_parquet(f / "geom.parquet")
    wide("t1_full_", topo.FEATURE_NAMES).to_parquet(f / "topo_T1.parquet")
    wide("t0_t1_full_", topo.FEATURE_NAMES[:5]).to_parquet(f / "topo_T0.parquet")
    wide("v_", ["prepost_h1", "postsf_h1", "prepost_h0", "postsf_h0"]).to_parquet(f / "topo_T2.parquet")
    pd.DataFrame({"index": idx, "t3_h1_mean": rng.random(n), "t3_h1_max": rng.random(n)}).to_parquet(f / "topo_T3.parquet")
    np.savez(f / "probe.npz", index=idx, states=rng.standard_normal((n, 2, 16)).astype(np.float16))
    cc = lab[["index", "split"]].assign(target="mem_T", seed=0, label=y, neg_pool=0, neg_drawn=0)
    cc.to_parquet(p.data / "cc_1b.parquet")

    analysis.run_ladder(cfg, "1b", sf, rungs=["4", "6", "7", "9"], clfs=("lr",))
    lad = pd.read_csv(p.results("1b", sf) / "ladder.csv", dtype={"rung": str})
    hard7 = lad[(lad["subset"] == "hard") & (lad["rung"] == "7")].iloc[0]
    assert hard7["AUROC"] > 0.8 and abs(hard7["shuffled_AUROC"] - 0.5) < 0.15
    d = pd.read_csv(p.results("1b", sf) / "deltas.csv")
    assert set(d["comparison"]) >= {"7 vs 6"}
    assert len(analysis.residual_analysis(cfg, "1b", sf)) == 10
    eff = analysis.effects(cfg, "1b", sf)
    assert {"T1", "T2"} <= set(eff["family"])


# ---------------------------------------------------------------------- scan (tiny random model)
def _tiny_model():
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    torch.manual_seed(0)
    c = tr.GPTNeoXConfig(vocab_size=97, hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                         intermediate_size=64, max_position_embeddings=128)
    return tr.GPTNeoXForCausalLM(c).eval()


def test_teacher_forced_extractability_equals_greedy():
    torch = pytest.importorskip("torch")
    from memtopo import scan

    m = _tiny_model()
    prompts = torch.randint(0, 97, (12, 32))
    cont = scan.greedy_decode(m, prompts, 32)
    full = torch.cat([prompts, cont], 1)
    broken_at = [None] * 6 + [0, 1, 5, 17, 30, 31]
    for i, k in enumerate(broken_at):
        if k is not None:
            full[i, 32 + k] = (full[i, 32 + k] + 1) % 97
    with torch.no_grad():
        rows = scan.loss_rows(m(input_ids=full).logits, full, 32)
    mem, mlen = rows[:, 0], rows[:, 1]
    assert (mem[:6] == 1).all() and (mem[6:] == 0).all()
    assert mlen.tolist() == [32] * 6 + [0, 1, 5, 17, 30, 31]
    gd = (scan.greedy_decode(m, full[:, :32], 32) == full[:, 32:]).all(1).numpy()
    assert (gd == (mem == 1)).all()


def test_scan_writes_and_resumes(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from memtopo import scan

    m = _tiny_model()
    monkeypatch.setattr(scan, "load_model", lambda *a, **k: m)
    rng = np.random.default_rng(0)
    toks = {i: rng.integers(0, 97, 64) for i in range(10, 30)}
    kw = dict(layers=[2, 4], batch=8, workers=2, probe=True)
    scan.scan("tiny", 5, toks, list(toks), [10, 11, 12], tmp_path, **kw)
    loss = scan.read_parts(tmp_path, "loss")
    hid = scan.read_parts(tmp_path, "hidden")
    assert len(loss) == 20 and len(hid) == 6 and set(hid["layer"]) == {2, 4}
    pi, ps = scan.read_probe(tmp_path)
    assert pi.tolist() == [10, 11, 12] and ps.shape == (3, 2, 32)
    scan.scan("tiny", 5, toks, list(toks), [10, 11, 12, 13], tmp_path, **kw)     # resume: only 13 is new
    assert len(scan.read_parts(tmp_path, "hidden")) == 8 and len(scan.read_parts(tmp_path, "loss")) == 20
    dg = scan.read_parts(tmp_path, "dgm")
    assert len(np.asarray(dg.iloc[0]["h0"])) == 2 * 63
