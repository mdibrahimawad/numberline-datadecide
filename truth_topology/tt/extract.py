"""Extract read-position hidden states (all layers) per checkpoint. GPU; resumable.

    python -m tt.extract --model pythia-1.4b-deduped --inputs P1 P2 [--steps 0 3000 143000]

Writes WORK/states/<model>/<input>/step<N>.npz with
  H         float16 (n_layers + 1, n_items, d_model); 0 = embeddings, l = output of block l
            (block outputs are taken with forward hooks, so the last one is *before* the final
            layer norm, like every other layer)
  P1: loglik      mean per-token log-probability of the statement (tokens 2..n)
  P2: p_correct, top1_correct, entropy   at the read position (the token before the attribute)
Each checkpoint is downloaded into its own cache folder and deleted after extraction; the
next checkpoint downloads in the background while the current one runs.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from . import config as C

log = logging.getLogger("extract")


def _cache(model: str, step: int):
    return C.WORK / "hf_ckpt" / model / f"step{step}"


def download(model: str, step: int):
    from huggingface_hub import list_repo_files, snapshot_download
    for attempt in range(5):
        try:
            files = list_repo_files(C.MODELS[model], revision=f"step{step}")
            weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
            return snapshot_download(C.MODELS[model], revision=f"step{step}", cache_dir=_cache(model, step),
                                     allow_patterns=["*.json", "*.txt", weights])
        except Exception as e:  # network hiccups: retry with backoff
            log.warning("download %s step%d failed (%s), retry %d", model, step, e, attempt)
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"download failed: {model} step{step}")


def load_model(path: str, dtype):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    return m.cuda().eval() if torch.cuda.is_available() else m.eval()


@torch.no_grad()
def run_batch(model, enc, read_pos):
    """Forward pass; returns hidden states (L+1, b, d) at read_pos and logits at read_pos and all positions."""
    layers = model.gpt_neox.layers
    outs = []
    hooks = [l.register_forward_hook(lambda mod, inp, out: outs.append(out[0] if isinstance(out, tuple) else out))
             for l in layers]
    try:
        res = model(**enc, output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()
    b = torch.arange(enc["input_ids"].shape[0], device=enc["input_ids"].device)
    hs = [res.hidden_states[0]] + outs
    H = torch.stack([h[b, read_pos] for h in hs]).float().cpu().numpy()
    return H, res.logits


def extract_one(model, tok, items: pd.DataFrame, input_name: str, batch: int):
    dev = next(model.parameters()).device
    n = len(items)
    H_all, extra = None, {k: np.zeros(n, np.float32) for k in
                          (["loglik"] if input_name == "P1" else ["p_correct", "top1_correct", "entropy"])}
    for s in range(0, n, batch):
        chunk = items.iloc[s:s + batch]
        enc = tok(list(chunk.text), return_tensors="pt", padding=True)  # right padding
        enc = {k: v.to(dev) for k, v in enc.items()}
        lengths = enc["attention_mask"].sum(1)
        read_pos = lengths - 1
        H, logits = run_batch(model, enc, read_pos)
        if H_all is None:
            H_all = np.zeros((H.shape[0], n, H.shape[2]), np.float16)
        H_all[:, s:s + len(chunk)] = H.astype(np.float16)
        lp = torch.log_softmax(logits.float(), -1)
        if input_name == "P1":
            tgt = enc["input_ids"][:, 1:]
            tok_lp = lp[:, :-1].gather(-1, tgt[..., None])[..., 0]
            mask = enc["attention_mask"][:, 1:].float()
            extra["loglik"][s:s + len(chunk)] = ((tok_lp * mask).sum(1) / mask.sum(1)).cpu().numpy()
        else:
            b = torch.arange(len(chunk), device=dev)
            last = lp[b, read_pos]
            tid = torch.tensor([tok(t)["input_ids"][0] for t in chunk.target], device=dev)
            extra["p_correct"][s:s + len(chunk)] = last[b, tid].exp().cpu().numpy()
            extra["top1_correct"][s:s + len(chunk)] = (last.argmax(-1) == tid).float().cpu().numpy()
            extra["entropy"][s:s + len(chunk)] = (-(last.exp() * last).sum(-1)).cpu().numpy()
    return H_all, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-1.4b-deduped", choices=list(C.MODELS))
    ap.add_argument("--inputs", nargs="+", default=["P1"])
    ap.add_argument("--steps", nargs="+", type=int, default=C.STEPS)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--keep-checkpoints", action="store_true")
    a = ap.parse_args()
    logdir = C.WORK / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(logdir / f"extract_{a.model}.log"), logging.StreamHandler()])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODELS[a.model], revision="step143000",
                                        cache_dir=C.WORK / "hf_ckpt" / a.model / "tokenizer")
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    items = {i: pd.read_csv(C.STATES / i / "items.csv") for i in a.inputs}

    def todo(step):
        return [i for i in a.inputs if not (C.STATES / a.model / i / f"step{step}.npz").exists()]

    steps = [s for s in a.steps if todo(s)]
    log.info("model %s: %d checkpoints to do: %s", a.model, len(steps), steps)
    dtype = getattr(torch, a.dtype)
    with ThreadPoolExecutor(1) as pool:
        fut = pool.submit(download, a.model, steps[0]) if steps else None
        for k, step in enumerate(steps):
            t0 = time.time()
            path = fut.result()
            fut = pool.submit(download, a.model, steps[k + 1]) if k + 1 < len(steps) else None
            model = load_model(path, dtype)
            for inp in todo(step):
                H, extra = extract_one(model, tok, items[inp], inp, a.batch)
                out = C.STATES / a.model / inp
                out.mkdir(parents=True, exist_ok=True)
                tmp = out / f"step{step}.tmp.npz"
                np.savez(tmp, H=H, step=step, model=a.model, dtype=a.dtype, **extra)
                os.replace(tmp, out / f"step{step}.npz")
                log.info("%s %s step%d: H %s, %s", a.model, inp, step, H.shape,
                         {k: round(float(v.mean()), 4) for k, v in extra.items()})
            del model
            torch.cuda.empty_cache()
            if not a.keep_checkpoints:
                shutil.rmtree(_cache(a.model, step), ignore_errors=True)
            log.info("step%d done in %.0fs", step, time.time() - t0)


if __name__ == "__main__":
    main()
