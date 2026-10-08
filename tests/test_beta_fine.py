"""Offline checks of src.beta_fine (CPU, tiny random model; no downloads).

    python tests/test_beta_fine.py
"""

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.beta_fine as bf  # noqa: E402


def _tiny_model_and_tokenizer():
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate("0123456789=,")}
    vocab["<eos>"] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<eos>"))
    tok.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=64, n_embd=32, n_layer=3,
                                       n_head=2, eos_token_id=vocab["<eos>"]))
    model.eval()
    return model, tokenizer


def test_bands_and_centres():
    assert [round(c) for c in bf.CENTRES] == [10, 22, 46, 100, 215, 464, 1000, 2154, 4642, 10000]
    spans = [(b.start, b.stop - 1) for b in map(bf.band, range(bf.N_GROUPS))]
    assert all(hi < lo2 for (_, hi), (lo2, _) in zip(spans, spans[1:])), spans  # no overlap
    assert spans[0] == (9, 11) and spans[3] == (90, 110) and spans[9] == (9000, 11000)


def test_fits_recover_known_beta():
    rng = np.random.default_rng(0)
    for true_b in (0.5, 1.0, 1.6):
        targets = np.array([bf.CENTRES[g] for g in range(bf.N_GROUPS) for _ in range(50)])
        groups = np.repeat(np.arange(bf.N_GROUPS), 50)
        u = np.log10(targets) - 1
        x = (true_b ** u if true_b != 1 else u) + rng.normal(0, 1e-3, len(u))
        states = np.column_stack([x, rng.normal(0, 1e-4, (len(x), 3))])
        m, scores, evr = bf.layer_metrics(states, targets, groups)
        assert abs(m["beta_fine"] - true_b) < 0.02, m
        assert abs(m["beta_coarse"] - true_b) < 0.02, m
        assert abs(m["beta_cont"] - true_b) < 0.02, m
        assert m["err_fine"] < 0.01 and m["err_coarse"] < 0.01 and m["rho"] > 0.99, m


def test_batched_equals_one_by_one():
    """Same-length batching must give the single-prompt hidden states and generations."""
    import torch

    model, tokenizer = _tiny_model_and_tokenizer()
    rng = random.Random(1)
    prompts = [bf.make_prompt(g, rng)[0] for g in range(bf.N_GROUPS) for _ in range(6)]
    states = bf.last_token_states(model, tokenizer, "cpu", prompts, batch=7)
    gens = bf.generate_first_integers(model, tokenizer, "cpu", prompts, max_new=5, batch=7)
    with torch.no_grad():
        for i, p in enumerate(prompts):
            x = torch.tensor([tokenizer(p)["input_ids"]])
            hs = model(input_ids=x, output_hidden_states=True).hidden_states
            one = torch.stack([h[0, -1, :] for h in hs]).to(torch.float16).numpy()
            assert np.allclose(states[:, i, :].astype(np.float32), one.astype(np.float32), atol=2e-3), i
            g = model.generate(input_ids=x, max_new_tokens=5, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(g[0, x.shape[1]:], skip_special_tokens=True).strip()
            assert gens[i][1] == text, (i, gens[i], text)


def test_full_model_run_writes_outputs():
    """run_model end to end with the tiny model; the 'model' answers correctly 70 % of the time."""
    model, tokenizer = _tiny_model_and_tokenizer()
    import src.geometry as geometry

    orig_load, orig_gen = geometry._load_model, bf.generate_first_integers
    geometry._load_model = lambda cfg, dev, tok: (model, tokenizer)
    coin = random.Random(3)

    def oracle(model_, tok_, dev, prompts, max_new, batch):
        out = []
        for p in prompts:
            n = int(p[p.rfind(",") + 1:-1])
            out.append((n, str(n)) if coin.random() < 0.7 else (n + 1, str(n + 1)))
        return out

    bf.generate_first_integers = oracle
    try:
        payload, arrays = bf.run_model("tiny", None, seeds=[5, 6], k=12, batch=16, max_new=4,
                                       max_candidates=50, device="cpu", dtype="float32", old_layer=2)
    finally:
        geometry._load_model, bf.generate_first_integers = orig_load, orig_gen
    assert set(payload["report"]) == {"selected_layer", "old_layer"}
    assert payload["report"]["old_layer"]["layer"] == 2
    for seed in ("5", "6"):
        f = payload["filter"][seed]
        assert all(v["accepted"] == 12 for v in f.values())          # every slot eventually accepted
        assert 0.1 < np.mean([v["rejection_rate"] for v in f.values()]) < 0.5
    assert arrays["seed5_pc_scores"].shape == (4, 120, 5)              # 3 blocks + embeddings
    assert sorted(set(arrays["seed6_groups"].tolist())) == list(range(10))
    with tempfile.TemporaryDirectory() as tmp:
        payload["recipe"] = "tiny"
        (Path(tmp) / "tiny.json").write_text(json.dumps(payload))
        rows = (bf.write_summary(Path(tmp))).read_text().splitlines()
        assert rows[0].startswith("recipe,layer,old_layer") and rows[1].startswith("tiny,")


def test_r2_is_blind_near_beta_one_but_relative_error_is_not():
    """Why err_* replaces R^2: a perfect fit with beta = 1 has R^2 ~ 0."""
    gaps = np.array([1.0, 1.0, 1.0]) * np.array([1.0, 1.01, 0.995])
    b, r2, err = bf.fit_geometric(gaps)
    assert abs(b - 1) < 0.01 and r2 < 0.5 and err < 0.01


if __name__ == "__main__":
    test_bands_and_centres()
    test_fits_recover_known_beta()
    test_batched_equals_one_by_one()
    test_full_model_run_writes_outputs()
    test_r2_is_blind_near_beta_one_but_relative_error_is_not()
    print("ok")
