"""Offline tests for the correct-output filter (no Hugging Face download).

A char-level tokenizer plus a fake model whose greedy completion is wrong for
targets divisible by 3 and for every target in group 3, so group 3 must be
flagged failed while groups 1, 2, 4 fill all k slots with correct prompts.

    python tests/test_filter_correct.py     (or: pytest tests/)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src.frozen_layer_evaluation import _collect_target_layer_filtered
from src.geometry import GeometryConfig, collect_hidden_states
from utils.prompts import default_interval, extract_target

CHARS = list("0123456789,=-")


def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<eos>": 0, **{c: i + 1 for i, c in enumerate(CHARS)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<eos>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tok.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", pad_token="<eos>")


def _completes_correctly(target: int) -> bool:
    return target % 3 != 0 and target not in default_interval(3)


class FakeModel(torch.nn.Module):
    """Answers `<target>,<junk>`; wrong (target + 1) when not _completes_correctly."""

    def __init__(self, tokenizer, n_layers: int = 2):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(num_hidden_layers=n_layers)
        self.generate_calls = 0

    def _target(self, input_ids) -> int:
        text = self.tokenizer.decode(input_ids[0].tolist())
        return int(extract_target(text, alphabetic=False))

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
        t = float(self._target(input_ids))
        hs = []
        for layer in range(self.config.num_hidden_layers + 1):
            h = torch.zeros(1, input_ids.shape[1], 4)
            h[0, -1] = torch.tensor([np.log10(t) * (layer + 1), t / 1e4, 1.0, layer])
            hs.append(h)
        return SimpleNamespace(hidden_states=tuple(hs))

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8, do_sample=True, pad_token_id=None):
        assert do_sample is False
        self.generate_calls += 1
        t = self._target(input_ids)
        answer = t if _completes_correctly(t) else t + 1
        new = self.tokenizer(f"{answer},12=12", add_special_tokens=False).input_ids[:max_new_tokens]
        return torch.cat([input_ids, torch.tensor([new])], dim=1)


def _cfg(**kw) -> GeometryConfig:
    base = dict(model_name="fake", groups=(1, 2, 3, 4), k=6, num_examples=3,
                context="random", device="cpu", filter_correct=True,
                filter_max_candidates=10)
    base.update(kw)
    return GeometryConfig(**base)


def test_filter_resamples_and_flags_failed_group():
    tok = _tokenizer()
    model = FakeModel(tok)
    cfg = _cfg()
    bundle = collect_hidden_states(cfg, model, tok, torch.device("cpu"), random.Random(0))
    stats = bundle.filter_stats

    for g in ("1", "2", "4"):
        assert stats[g]["accepted"] == cfg.k and not stats[g]["failed"], stats[g]
        assert stats[g]["failed_slots"] == 0
        assert stats[g]["candidates_tried"] == cfg.k + stats[g]["rejected"]
        assert len(stats[g]["example_rejections"]) == min(5, stats[g]["rejected"])
    # group 3 never completes correctly: first slot exhausts its budget
    assert stats["3"]["failed"] and stats["3"]["accepted"] == 0
    assert stats["3"]["candidates_tried"] == cfg.filter_max_candidates
    assert stats["3"]["failed_slots"] == cfg.k
    assert stats["3"]["rejection_rate"] == 1.0
    assert len(stats["3"]["example_rejections"]) == 5
    ex = stats["3"]["example_rejections"][0]
    assert ex["generated"] == ex["target"] + 1 and ex["response"].startswith(str(ex["generated"]))

    # hidden states only for accepted prompts; rejected targets never collected
    for layer in range(bundle.n_layers):
        assert sorted(bundle.states[layer]) == [1, 2, 4]
        for g in (1, 2, 4):
            assert len(bundle.states[layer][g]) == cfg.k
            assert all(_completes_correctly(int(a)) for a in bundle.answers[layer][g])
            assert all(int(a) in default_interval(g) for a in bundle.answers[layer][g])
    # resampled prompts keep the paper format and the final '=' token
    diag = bundle.tokenization_diagnostics
    assert diag["prompts_ending_equals_token"] == diag["prompt_count"] == 3 * cfg.k


def test_filter_is_deterministic_per_seed():
    tok = _tokenizer()
    a = collect_hidden_states(_cfg(), FakeModel(tok), tok, torch.device("cpu"), random.Random(7))
    b = collect_hidden_states(_cfg(), FakeModel(tok), tok, torch.device("cpu"), random.Random(7))
    assert a.answers[0] == b.answers[0]


def test_filter_off_is_unchanged():
    tok = _tokenizer()
    model = FakeModel(tok)
    bundle = collect_hidden_states(_cfg(filter_correct=False), model, tok,
                                   torch.device("cpu"), random.Random(0))
    assert model.generate_calls == 0 and bundle.filter_stats == {}
    assert all(len(bundle.states[0][g]) == 6 for g in (1, 2, 3, 4))


def test_frozen_layer_filtered_collection():
    tok = _tokenizer()
    matrix, answers, groups, stats = _collect_target_layer_filtered(
        _cfg(), 2, FakeModel(tok), tok, torch.device("cpu"), random.Random(1)
    )
    assert matrix.shape == (18, 4) and len(answers) == 18
    assert sorted(groups) == [1, 2, 4] and stats["3"]["failed"]


def test_datadecide_protocol_end_to_end(tmp_path=None):
    """Selection on one seed, frozen eval on another, JSON + summary.csv."""
    import json
    import tempfile

    import src.frozen_layer_evaluation as fle
    import src.geometry as geometry
    from src.datadecide_sweep import run_datadecide_model, write_summary_csv

    tok = _tokenizer()
    fake_load = lambda cfg, device, hf_token: (FakeModel(tok), tok)  # noqa: E731
    orig = geometry._load_model, fle._load_model
    geometry._load_model = fle._load_model = fake_load
    try:
        payload = run_datadecide_model(
            "allenai/DataDecide-fake-60M", None, k=5,
            selection_seeds=(42,), eval_seeds=(45,), device="cpu", dtype="float32",
            filter_max_candidates=10,
        )
    finally:
        geometry._load_model, fle._load_model = orig

    assert payload["final_token_is_equals"]
    layer = payload["selection"]["selected_layer"]
    assert payload["evaluation"]["target_layer"] == layer
    assert set(payload["selection"]["layers_direct"]) == set(payload["selection"]["layers_log"])
    fs = payload["filter_summary"]
    assert fs["3"]["failed_seeds"] == [42, 45]
    for g in ("1", "2", "4"):
        assert fs[g]["accepted"] == 10 and not fs[g]["failed_seeds"]

    out_dir = Path(tmp_path or tempfile.mkdtemp())
    payload["recipe"] = "fake"
    (out_dir / "fake.json").write_text(json.dumps(payload))
    rows = (write_summary_csv(out_dir)).read_text().splitlines()
    assert rows[0].split(",")[:3] == ["recipe", "selected_layer", "rho_mean"]
    assert rows[1].startswith(f"fake,{layer},") and rows[1].endswith(",3")


def test_hf_olmo_generate_path():
    """Real hf_olmo.OLMoForCausalLM (random weights) through the filter's generate call."""
    try:
        from hf_olmo import OLMoConfig, OLMoForCausalLM
    except ImportError:
        print("skip test_hf_olmo_generate_path: ai2-olmo not installed")
        return
    tok = _tokenizer()
    torch.manual_seed(0)
    olmo_cfg = OLMoConfig(d_model=32, n_heads=2, n_layers=2, mlp_ratio=2, max_sequence_length=64,
                          vocab_size=len(CHARS) + 1, embedding_size=32, pad_token_id=0,
                          eos_token_id=0, weight_tying=False, init_device="cpu")
    model = OLMoForCausalLM(olmo_cfg, init_params=True)
    if not hasattr(model, "generate"):
        raise AssertionError("hf_olmo model has no .generate(); use transformers<4.50")
    bundle = collect_hidden_states(_cfg(k=2, filter_max_candidates=3), model, tok,
                                   torch.device("cpu"), random.Random(0))
    for st in bundle.filter_stats.values():
        assert st["accepted"] == 2 or st["failed"]
        assert all(len(tok(ex["response"]).input_ids) <= 8 for ex in st["example_rejections"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
