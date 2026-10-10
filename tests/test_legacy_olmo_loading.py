"""Offline check: checkpoints in the 2023 OLMo format (model_type "olmo", e.g. the Paloma
baselines) load with their real weights through src.geometry._load_model, and a checkpoint
whose weights don't all load is refused instead of silently using random weights.

    python tests/test_legacy_olmo_loading.py
"""

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.geometry import GeometryConfig, _legacy_olmo_config, _load_model  # noqa: E402
from tests.test_filter_correct import CHARS, _tokenizer  # noqa: E402


def _save_legacy_checkpoint(out: Path, drop_key: str | None = None):
    """A tiny hf_olmo model saved the way the Paloma repos are: pytorch_model.bin and a
    config.json with model_type "olmo" and OLMo field names."""
    from hf_olmo import OLMoConfig, OLMoForCausalLM

    torch.manual_seed(0)
    cfg = OLMoConfig(d_model=32, n_heads=2, n_layers=2, mlp_ratio=2, max_sequence_length=64,
                     vocab_size=len(CHARS) + 1, embedding_size=32, pad_token_id=0, eos_token_id=0,
                     weight_tying=False, init_device="cpu")
    model = OLMoForCausalLM(cfg, init_params=True).eval()
    model.save_pretrained(out, safe_serialization=False)
    _tokenizer().save_pretrained(out)
    conf = json.loads((out / "config.json").read_text())
    conf["model_type"] = "olmo"
    conf["architectures"] = ["OlmoModelForCausalLM"]
    (out / "config.json").write_text(json.dumps(conf))
    if drop_key:
        state = torch.load(out / "pytorch_model.bin")
        del state[drop_key]
        torch.save(state, out / "pytorch_model.bin")
    return model


def _skip_without_olmo() -> bool:
    try:
        import hf_olmo  # noqa: F401
    except ImportError:
        print("skip: ai2-olmo not installed")
        return True
    return False


def test_legacy_olmo_loads_the_real_weights():
    if _skip_without_olmo():
        return
    with tempfile.TemporaryDirectory() as tmp:
        original = _save_legacy_checkpoint(Path(tmp))
        assert _legacy_olmo_config(tmp, {}) is not None
        model, tok = _load_model(GeometryConfig(model_name=tmp, dtype="float32", device="cpu"),
                                 torch.device("cpu"), None)
        ids = torch.tensor([tok("12=12,7=")["input_ids"]])
        with torch.no_grad():
            want = original(input_ids=ids).logits
            got = model(input_ids=ids).logits
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        assert torch.allclose(want, got, atol=1e-5), "legacy checkpoint did not load its own weights"
        assert len(hs) == 3                                  # embeddings + 2 blocks


def test_plain_transformers_path_would_not_load_these_weights():
    """Documents the trap the special case avoids: transformers reads model_type "olmo" as its
    native Olmo config, ignores OLMo's field names (d_model, n_layers) and would build a
    default-sized (~7B, randomly initialised) model instead of the checkpoint's."""
    if _skip_without_olmo():
        return
    from transformers import AutoConfig

    with tempfile.TemporaryDirectory() as tmp:
        _save_legacy_checkpoint(Path(tmp))
        native = AutoConfig.from_pretrained(tmp)
        assert type(native).__name__ != "OLMoConfig"
        assert native.hidden_size != 32 and native.num_hidden_layers != 2, native


def test_incomplete_checkpoint_is_refused():
    if _skip_without_olmo():
        return
    with tempfile.TemporaryDirectory() as tmp:
        _save_legacy_checkpoint(Path(tmp), drop_key="model.transformer.blocks.1.ff_out.weight")
        try:
            _load_model(GeometryConfig(model_name=tmp, dtype="float32", device="cpu"), torch.device("cpu"), None)
        except RuntimeError as exc:
            assert "did not load cleanly" in str(exc)
        else:
            raise AssertionError("a checkpoint with a missing weight was accepted")


def test_other_checkpoints_are_not_treated_as_legacy():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.json").write_text(json.dumps({"model_type": "hf_olmo", "d_model": 32}))
        assert _legacy_olmo_config(tmp, {}) is None             # DataDecide format
        (Path(tmp) / "config.json").write_text(json.dumps({"model_type": "olmo", "hidden_size": 32}))
        assert _legacy_olmo_config(tmp, {}) is None             # transformers-native OLMo
        assert _legacy_olmo_config(str(Path(tmp) / "missing"), {}) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
