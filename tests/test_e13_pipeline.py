"""E13 pipeline tests: prompts/positions with real tokenizers, extraction + checks + resume with a tiny random model,
causal hooks, and synthetic planted-structure recovery for the analysis panel (writes power_check_synthetic.csv)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "experiments", "E13_number_code", "src")); sys.path.insert(0, REPO)

from e13 import config as C  # noqa: E402
from e13 import prompts as P  # noqa: E402

NEXT_TEXT = {"C1": "=", "C1s7": "=", "C2": " apples", "C3": ".", "C4": " is", "C5": " or", "C6": " +", "C7": ",", "C8": " dollars"}


@pytest.fixture(scope="module")
def pythia_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")


@pytest.mark.parametrize("model_id", ["EleutherAI/pythia-1.4b", "Qwen/Qwen2.5-1.5B"])
def test_positions_real_tokenizers(model_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id); c1s7 = P.c1_generated(C.SEED_C1B)
    for ctx in P.all_contexts():
        for n in (7, 42, 137, 1987, 5371):
            p, s, e = P.build(ctx, n, c1s7); assert p[s:e] == str(n)
            pos = P.positions(tok, p, s, e)
            assert pos["ok"], (model_id, ctx, n, pos["reason"])
            toks = tok.convert_ids_to_tokens(pos["ids"])
            last = tok.convert_tokens_to_string([toks[pos["last"]]]).strip(); assert str(n).endswith(last)
            nxt = tok.convert_tokens_to_string([toks[pos["next"]]])
            if ctx in NEXT_TEXT:
                assert NEXT_TEXT[ctx].strip().startswith(nxt.strip()) or nxt.strip().startswith(NEXT_TEXT[ctx].strip()[:1]), (ctx, nxt)


def test_c1_prompts_match_old_format():
    old = P.c1_seed42(); assert len(old) == 10000 and old[137].endswith(",137=")
    gen = P.c1_generated(7, 50)
    assert all(v.endswith(f",{k}=") and v.count("=") == 5 for k, v in gen.items())


# --------------------------------------------------------------------------- tiny model: extraction, checks, resume
@pytest.fixture(scope="module")
def tiny(tmp_path_factory, pythia_tok):
    import torch
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(vocab_size=len(pythia_tok), hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                        intermediate_size=128, max_position_embeddings=256)
    model = GPTNeoXForCausalLM(cfg).eval()
    out = tmp_path_factory.mktemp("tiny")
    info = dict(model_id="tiny-neox", revision=None, commit=None, dtype="float32", device="cpu")
    return model, pythia_tok, out, (model, pythia_tok, pythia_tok, info)


def test_tiny_extraction_and_checks(tiny):
    from e13 import checks, extract
    model, tok, out, bundle = tiny; numbers = np.arange(1, 121)
    res = checks.run_checks("pythia-1.4b", model, tok, tok, str(out), numbers=numbers, force=True, log=lambda *a: None)
    assert res["a_pass"] and res["b_pass"] and res["f_pass"]
    r = extract.extract_model("pythia-1.4b", str(out), numbers=numbers, batch=32, chunk=50, model_bundle=bundle, log=lambda *a: None)
    assert r["done"]
    a = np.load(os.path.join(out, "vec_C2_num.npy")); assert a.shape == (120, 9, 64)
    meta = pd.read_csv(os.path.join(out, "meta.csv")); assert meta["ok_C2"].all() and "copy_correct_C1" in meta
    avg = np.load(os.path.join(out, "vec_avg_num_alllayers.npy")); assert avg.shape == (120, 5, 64)
    # the neutral average at the stored layer slots equals the mean of the five per-context files
    lidx = json.load(open(os.path.join(out, "config.json")))["layer_indices"]
    man = np.mean([np.load(os.path.join(out, f"vec_{c}_num.npy"))[:, 2].astype(np.float32) for c in C.NEUTRAL], 0)
    assert np.allclose(avg[:, lidx[2]].astype(np.float32), man, atol=2e-2)
    # resume: a second call finds every chunk done
    prog = json.load(open(os.path.join(out, "progress.json"))); n_before = prog["prompts"]
    extract.extract_model("pythia-1.4b", str(out), numbers=numbers, batch=32, chunk=50, model_bundle=bundle, log=lambda *a: None)
    assert json.load(open(os.path.join(out, "progress.json")))["prompts"] == n_before
    assert os.path.exists(os.path.join(out, "p7_days.npy")) and np.load(os.path.join(out, "p7_months.npy")).shape[:2] == (20, 12)


def test_tiny_causal_hooks(tiny):
    import torch
    from e13 import causal
    model, tok, out, _ = tiny
    items = causal.build_items(tok, "chunk", np.arange(10, 100), "B1", 6, np.random.default_rng(0))
    assert len(items) == 6
    base = causal.score(model, tok, items)
    d = model.config.hidden_size; layer = 2
    same = causal.score(model, tok, items, causal.make_edit("ablate", np.zeros((d, 0)), np.zeros(d)), layer)
    assert np.allclose(base, same, atol=1e-5)
    full = causal.score(model, tok, items, causal.make_edit("ablate", np.eye(d), np.zeros(d)), layer)
    assert not np.allclose(base, full)
    # patching with an item's own clean state changes nothing
    src = causal.states_at(model, tok, items, layer)
    own = causal.score(model, tok, items, causal.make_edit("patch", np.eye(d), np.zeros(d), other=src), layer)
    assert np.allclose(base, own, atol=1e-4)


# --------------------------------------------------------------------------- synthetic planted structure
def planted(n, L, curve=3.0, digits=2.0, noise=1.0, D=50, seed=0):
    from src import e13_stage1 as E1
    rng = np.random.default_rng(seed); t = (n - n.min()) / max(np.ptp(n), 1)
    Z = curve * np.c_[np.cos(2 * t), np.sin(2 * t), t] @ rng.standard_normal((3, D)) / np.sqrt(3)
    dig = E1.digits(n, L)
    for p in range(L):
        Z += digits * rng.standard_normal((10, D))[dig[:, p]] / np.sqrt(D)
    return Z + noise * rng.standard_normal((len(n), D)) / np.sqrt(D) * 3


def test_panel_recovers_planted():
    from e13 import panel
    n = np.arange(100, 1000); X = planted(n, 3)
    m = panel.m12(X, n, "S3", full=False); assert m["R2_curve"] > m["R2_shuffled"] + 0.2
    assert m["R2_curve_digits"] > m["R2_curve"] + 0.02
    p4 = panel.m4(X, n, "S3"); assert p4["probe_n_R2"] > 0.5 and p4["lin_acc_units"] > 0.5
    k = panel.p8(X, n, "S3", "t"); assert k["random"]["knee"] is not None


def test_power_check_synthetic():
    """Plant (a) a smooth curve, (b) digit-additive structure, (c) carry jumps, (d) line / circle / simplex digit codes
    (plus the P8 and H3 plants) at two noise levels; record recovery in results/power_check_synthetic.csv and require
    recovery at the low-noise level."""
    from src import e13_stage1 as E1
    from e13 import power
    n = np.arange(100, 1000); allrows = []
    for noise in (0.2, 0.7):
        Z = E1.prep(planted(n, 3, noise=noise))
        rows = power.power_m1_m2_p8_h3(Z, n, "S3", "syn", reps=5)
        h1 = E1.h1(Z, n, 3, "syn", B=50, n_perm=20, n_rep=5); h2 = E1.h2_all(Z, Z, n, 3, "syn", False, n_ab=5, n_c=3)
        rows += power.rows_from_cell(dict(h1=h1, h2=h2), reps_h1=5, reps_ab=5, reps_c=3)
        allrows += [dict(noise=noise, **r) for r in rows]
    df = pd.DataFrame(allrows); os.makedirs(os.path.join(REPO, "experiments", "E13_number_code", "results"), exist_ok=True)
    df.to_csv(os.path.join(REPO, "experiments", "E13_number_code", "results", "power_check_synthetic.csv"), index=False)
    got = df[df.noise == 0.2].set_index("test")
    # H2a and the H1 decay rule plant into the dataset's own residual (here dominated by the permuted digit code), so
    # their power is a property of the data and is only recorded; their detectors are checked directly in
    # tests/test_e13_stage1.py (test_h1_decay_rule, test_h2a_transient_vs_cumulative).
    for t in ("M1 smooth curve", "M2 digit-additive", "H3 lexicon reliability", "H2c tree vs line", "P8 knee (rank-3 planted)"):
        assert got.loc[t, "has_power"], t
    assert all(got.loc[[t for t in got.index if t.startswith("H1 template")], "recovered"] >= 0.8)


def test_p7_detector():
    from e13 import p7
    rng = np.random.default_rng(0); K, T, d = 12, 20, 40
    ang = 2 * np.pi * np.arange(K) / K; basis = rng.standard_normal((2, d))
    circ = (np.c_[np.cos(ang), np.sin(ang)] @ basis)[None] * 3 + rng.standard_normal((T, K, d)) * 0.5
    assert p7.detect(circ)["passed"]
    assert not p7.detect(rng.standard_normal((T, K, d)))["passed"]


def test_emergence_and_order_test():
    from e13 import training
    steps = np.array([0, 64, 256, 1000, 4000, 16000, 64000])
    assert training.emergence(steps, [0, 0, 1, 2, 3, 4, 4]) < training.emergence(steps, [0, 0, 0, 0, 1, 3, 4])
    rows = [dict(zip(training.PRED_ORDER, np.sort(np.random.default_rng(i).random(4)) * 1e4)) for i in range(20)]
    assert training.order_test(pd.DataFrame(rows), n_perm=500)["p"] < 0.05
