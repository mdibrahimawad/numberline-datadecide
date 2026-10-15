"""Extract last-token hidden states for the 10,000 number prompts of `topology_number_line`
(`A=A,B=B,C=C,D=D,n=`, n = 1..10000, the exact saved texts) from any Hugging Face causal LM,
at the same relative depths used in E10 (8/32, 16/32, 24/32, 27/32, 32/32 of the layers).

Output (same format as `topology_number_line/data/<model>_vectors/`): layer_LL.npz with X
(10000 x width, centred, float16), mean (float32), target (1..10000); meta.json.

Made to run on a laptop (Apple MPS, or CPU). Prompts are batched with right padding and the
state of each prompt's last real token is taken, so padding never changes a result
(checked by `--check`).

  python -m src.extract_number_vectors --model Qwen/Qwen2.5-1.5B --name qwen2.5-1.5b \
      --prompts <topology_number_line>/data/openllama-3b_vectors/tokenization.json.gz \
      --out <topology_number_line>/data/qwen2.5-1.5b_vectors
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import numpy as np

FRACTIONS = [8 / 32, 16 / 32, 24 / 32, 27 / 32, 1.0]


def levels_for(n_layers: int) -> list[int]:
    """Hidden-state indices (0 = embeddings, n_layers = last) at the E10 relative depths."""
    return [int(f * n_layers + 0.5) for f in FRACTIONS]


def load_prompts(path):
    recs = json.load(gzip.open(path, "rt"))["records"]
    recs = sorted(recs, key=lambda r: r["target"])
    assert [r["target"] for r in recs] == list(range(1, 10001))
    assert all(r["prompt"].endswith(f"{r['target']}=") for r in recs)
    return [r["prompt"] for r in recs]


def run(model, tok, prompts, levels, device, batch):
    import torch
    enc = [tok(p)["input_ids"] for p in prompts]
    order = np.argsort([len(e) for e in enc], kind="stable")
    out = {L: np.zeros((len(prompts), model.config.hidden_size), np.float32) for L in levels}
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        n = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), n), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), n), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, :len(enc[i])] = torch.tensor(enc[i]); mask[r, :len(enc[i])] = 1
        with torch.no_grad():
            hs = model(input_ids=ids.to(device), attention_mask=mask.to(device), output_hidden_states=True).hidden_states
        last = torch.tensor([len(enc[i]) - 1 for i in idx])
        for L in levels:
            out[L][idx] = hs[L][torch.arange(len(idx)), last.to(device)].float().cpu().numpy()
    return out, enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face id or local path")
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke test)")
    ap.add_argument("--check", action="store_true", help="also compare 20 prompts run one by one")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(device).eval()
    n_layers = model.config.num_hidden_layers
    levels = levels_for(n_layers)
    prompts = load_prompts(a.prompts)
    if a.limit:
        prompts = prompts[: a.limit]
    print(f"{a.name}: {n_layers} layers, width {model.config.hidden_size}, levels {levels}, device {device}, "
          f"{len(prompts)} prompts", flush=True)

    t0 = time.time()
    out, enc = run(model, tok, prompts, levels, device, a.batch)
    secs = time.time() - t0
    last_pieces = [tok.convert_ids_to_tokens(e[-1]) for e in enc]
    ends_eq = float(np.mean([p.endswith("=") for p in last_pieces]))
    target_tokens = [len(tok(str(n), add_special_tokens=False)["input_ids"]) for n in (7, 42, 137, 1987, 5371, 9999)]
    print(f"done in {secs / 60:.1f} min; last token ends with '=': {ends_eq:.3f}; "
          f"tokens for 7/42/137/1987/5371/9999: {target_tokens}", flush=True)

    check = None
    if a.check:
        sub = list(range(0, len(prompts), max(1, len(prompts) // 20)))[:20]
        single, _ = run(model, tok, [prompts[i] for i in sub], levels, device, 1)
        check = max(float(np.max(np.abs(single[L] - out[L][sub])) / (np.abs(out[L][sub]).max() + 1e-12)) for L in levels)
        print(f"batched vs one-by-one, max relative difference: {check:.2e}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    report = {}
    target = np.arange(1, len(prompts) + 1)
    for L in levels:
        X = out[L]; mean = X.mean(0); X16 = (X - mean).astype(np.float16)
        assert np.isfinite(X16).all(), f"level {L}: values too large for float16"
        path = os.path.join(a.out, f"layer_{L:02d}.npz")
        np.savez_compressed(path, X=X16, mean=mean, target=target)
        report[L] = round(os.path.getsize(path) / 1e6, 1)
        print(f"level {L:2d}: saved, {report[L]} MB", flush=True)
    json.dump(dict(model=a.model, revision=getattr(model.config, "_commit_hash", None), name=a.name,
                   n_layers=n_layers, width=model.config.hidden_size, levels=levels, fractions=FRACTIONS,
                   dtype=a.dtype, device=device, minutes=round(secs / 60, 1), last_token_ends_with_eq=ends_eq,
                   batched_vs_single_max_rel_diff=check, target_token_counts_7_42_137_1987_5371_9999=target_tokens,
                   file_mb=report, prompts=os.path.basename(a.prompts), centred=True, stored="float16"),
              open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
