"""Offline tests for the DataDecide olmo_npy sampler (src/datadecide_sampling.py,
modal_app/datadecide_alpha_app.py) on tiny fake uint16 token files.

The tokenizer test uses OLMo's training tokenizer
(olmo_data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json, shipped with
ai2-olmo) and is skipped if it is not installed.

    python tests/test_datadecide_sampling.py      (or: pytest tests/)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import modal_app.corpus_alpha_app as caa
from modal_app.datadecide_alpha_app import count_window
from src import datadecide_sampling as ds

EOS = ds.EOS_TOKEN_ID


def test_window_coverage_is_uniform():
    lens = [3000, 500, 0, 12000, 41]
    w = 400
    plan = ds.plan_windows(lens, n_windows=40000, window_tokens=w, seed=0)
    cover = [np.zeros(n) for n in lens]
    for win in plan:
        f = win["file_index"]
        assert lens[f] > 0 and 0 < win["length"] <= w
        assert win["start"] + win["length"] <= lens[f]  # never crosses a file
        assert win["at_file_start"] == (win["start"] == 0)
        assert win["at_file_end"] == (win["start"] + win["length"] == lens[f])
        cover[f][win["start"]: win["start"] + win["length"]] += 1
    all_cover = np.concatenate(cover)
    expected = 40000 * w / sum(n + w - 1 for n in lens if n)
    # every token (file edges and tiny files included) is covered at the same rate
    assert abs(all_cover.mean() / expected - 1) < 0.02
    for c in cover:
        if len(c):
            assert abs(c.mean() / expected - 1) < 0.15, (c.mean(), expected)
    edge = np.concatenate([c[:20] for c in cover if len(c)])
    assert abs(edge.mean() / expected - 1) < 0.15


def test_split_documents_modes():
    t = np.array([5, 6, EOS, 7, 8, 9, EOS, 10, EOS, 11, 12], dtype=np.uint16)
    assert ds.split_documents(t, "drop_partial_docs", False, False) == [[7, 8, 9], [10]]
    assert ds.split_documents(t, "drop_partial_docs", True, True) == [[5, 6], [7, 8, 9], [10], [11, 12]]
    assert ds.split_documents(t, "keep_partial", False, False, edge_trim=1) == [[6], [7, 8, 9], [10], [11]]
    no_eos = np.arange(100, 130, dtype=np.uint16)
    assert ds.split_documents(no_eos, "drop_partial_docs", False, False) == []
    assert ds.split_documents(no_eos, "keep_partial", False, False, edge_trim=8) == [list(range(108, 122))]


def test_training_chunk_order_matches_olmo():
    lens = [2048 * 5 + 7, 2048 * 3, 100, 2048 * 4 + 2047]
    offsets = ds.chunk_offsets(lens)
    assert offsets.tolist() == [0, 5, 8, 8, 12]  # per-file remainders dropped (MemMapDataset)
    order = ds.training_chunk_indices(lens, n_instances=12, seed=6198)
    ref = np.arange(12, dtype=np.uint32)
    np.random.Generator(np.random.PCG64(seed=6198)).shuffle(ref)
    assert order.tolist() == ref.tolist()
    # more instances than chunks: the next epoch reshuffles with seed + 1
    longer = ds.training_chunk_indices(lens, n_instances=20, seed=6198)
    ref2 = np.arange(12, dtype=np.uint32)
    np.random.Generator(np.random.PCG64(seed=6199)).shuffle(ref2)
    assert longer.tolist() == ref.tolist() + ref2[:8].tolist()
    wins = ds.chunks_to_windows(np.array([0, 4, 5, 7, 8, 11]), lens)
    assert [(w["file_index"], w["start"]) for w in wins] == [(0, 0), (0, 8192), (1, 0), (1, 4096), (3, 0), (3, 6144)]
    assert all(w["length"] == 2048 for w in wins)


def test_convergence_and_split_half_shapes():
    rng = np.random.default_rng(0)
    n = np.arange(1, caa.MAX_N + 1)
    p = n ** -1.4 / (n ** -1.4).sum()
    mat = np.zeros((200, caa.MAX_N + 1))
    mat[:, 1:] = rng.multinomial(5000, p, size=200)
    conv = ds.convergence(mat, [50, 100, 200, 500], n_boot=50)
    assert [r["windows"] for r in conv] == [50, 100, 200]
    widths = [r["alpha_mle_ci95"][1] - r["alpha_mle_ci95"][0] for r in conv]
    assert widths[0] > widths[-1]
    sh = ds.split_half(mat, n_splits=20, seed=0, n_boot=20)
    assert sh["windows_per_half"] == 100 and sh["abs_diff_mle_mean"] < 0.02


def _olmo_tokenizer():
    try:
        import olmo_data
    except ImportError:
        return None
    from tokenizers import Tokenizer

    path = Path(olmo_data.__file__).parent / "tokenizers" / "allenai_gpt-neox-olmo-dolma-v1_5.json"
    return SimpleNamespace(backend_tokenizer=Tokenizer.from_file(str(path)))


def test_npy_window_counting_with_olmo_tokenizer():
    tok = _olmo_tokenizer()
    if tok is None:
        print("skip test_npy_window_counting_with_olmo_tokenizer: ai2-olmo not installed")
        return
    assert tok.backend_tokenizer.id_to_token(EOS) == "<|endoftext|>"
    rng = np.random.default_rng(1)
    docs = [f"Doc {i}: in {rng.integers(1800, 2030)} there were {rng.integers(0, 12000)} items, "
            f"code 007 and {rng.integers(1, 99)}.5 percent." for i in range(300)]
    stream = []
    for d in docs:
        stream.extend(tok.backend_tokenizer.encode(d).ids + [EOS])
    tokens = np.asarray(stream, dtype=np.uint16)
    raw = tokens.tobytes()  # OLMo memmap layout: headerless uint16
    assert np.array_equal(np.frombuffer(raw, dtype=ds.TOKEN_DTYPE), tokens)

    whole = {"at_file_start": True, "at_file_end": True}
    out = count_window(tok, tokens, whole)
    expected = Counter()
    for d in docs:
        caa._count_text(d, expected)
    for mode in ds.EDGE_MODES:
        assert out[mode]["docs"] == len(docs)
        assert Counter({int(k): v for k, v in out[mode]["counts"].items()}) == expected

    # a window cut in the middle of documents drops the partial ones
    lo, hi = 1000, 2600
    part = count_window(tok, tokens[lo:hi], {"at_file_start": False, "at_file_end": False})
    eos_at = np.flatnonzero(tokens == EOS)
    inside = [i for i in range(len(docs))
              if (i == 0 and lo == 0 or i > 0 and eos_at[i - 1] >= lo) and eos_at[i] < hi]
    assert part["drop_partial_docs"]["docs"] == len(inside)
    assert part["keep_partial"]["docs"] == len(inside) + 2


def test_data_map_config():
    dm = ds.load_data_map()
    models = json.loads((Path(__file__).resolve().parent.parent / "configs" / "datadecide_models.json").read_text())
    assert list(dm["recipes"]) == models["recipes"] and len(dm["recipes"]) == 25
    assert dm["eos_token_id"] == EOS and dm["token_dtype"] == "uint16" and dm["sequence_length"] == 2048
    assert dm["train_instances_1b"] == 69369 * 704
    d17 = dm["recipes"]["dolma1_7"]
    assert d17["olmo_mix_name"] == "dolma17" and d17["n_paths"] == 1033
    assert all(k == 2 and "/wiki/" in p for p, k in d17["repeated_paths"].items())
    for r in dm["recipes"].values():
        assert r["n_paths"] == len(r["paths"]) and all(p.endswith(".npy") for p in r["paths"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
