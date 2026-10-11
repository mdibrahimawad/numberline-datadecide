"""One pass of one checkpoint over a set of training sequences (GPU forward -> CPU Ripser).

For every requested sequence (first 64 tokens = 32 prompt + 32 continuation) a pass gives
  * the loss family (B0-B2 inputs): 32-extractability, greedy match length, cross-entropies,
    Min-K%, entropies -- from ONE teacher-forced forward pass;
  * for the ``hidden`` subset, per selected layer: B6 geometry, T1 summaries (normalised full
    cloud, normalised continuation cloud, raw full cloud), the normalised full-cloud
    diagrams (for T2/T3), and optionally the mean-pooled continuation state (B5 probe).

Exact 32-extractability from teacher forcing: greedy decoding reproduces the true 32 tokens
iff the argmax at every continuation position equals the true token given the true prefix
(if all earlier greedy tokens were right, the context is identical). So mem = all(argmax ==
truth) and match_len = leading run of matches. ``greedy_check`` verifies this against real
KV-cache greedy decoding on a subsample (numerics can differ between the two paths).

Outputs, under scans/<model>/step<N>/ (append-only parts, resumable):
  loss-*.parquet       index + LOSS_NAMES
  hidden-*.parquet     index, layer, geom_*, t1_full_*, t1_cont_*, t1_full_raw_*
  dgm-*.parquet        index, layer, h0 (flat list), h1 (flat list)   [normalised, full cloud]
  probe-*.npz          index, states (n, n_layers, d) float16
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import logging
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from memtopo import topology as topo

log = logging.getLogger("memtopo")

LOSS_NAMES = ["mem", "match_len", "match_frac", "ce_cont", "ce_prompt", "mink20",
              "ent_cont_mean", "ent_first", "logp_first"]


# ---------------------------------------------------------------------- model handling
def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name: str, step: int, dtype: str = "fp32"):
    import torch
    from transformers import GPTNeoXForCausalLM

    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    m = GPTNeoXForCausalLM.from_pretrained(name, revision=f"step{step}", torch_dtype=torch_dtype)
    m = m.to(device()).eval()
    return m


def free_revision(name: str, step: int) -> None:
    """Delete one checkpoint's weights from the HF cache (brief: free disk after use)."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    hashes = [r.commit_hash for repo in info.repos if repo.repo_id == name
              for r in repo.revisions if f"step{step}" in r.refs]
    if hashes:
        info.delete_revisions(*hashes).execute()


# ---------------------------------------------------------------------- loss family
def loss_rows(logits, tokens, prompt_len: int):
    """Per-sequence loss features from logits (B, T, V) and tokens (B, T) on GPU."""
    import torch

    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)            # predicts tokens[:, 1:]
    tgt = tokens[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)             # (B, T-1)
    c0 = prompt_len - 1                                               # first continuation slot
    arg = lp[:, c0:].argmax(-1)
    match = (arg == tgt[:, c0:])                                      # (B, 32)
    match_len = torch.where(match.all(1), torch.full_like(match[:, 0], match.shape[1], dtype=torch.long),
                            (~match).int().argmax(1).long())
    ent = -(lp[:, c0:].exp() * lp[:, c0:]).sum(-1)                    # (B, 32)
    k = max(1, int(round(0.2 * tok_lp.shape[1])))
    mink = tok_lp.topk(k, dim=1, largest=False).values.mean(1)
    cols = [match.all(1).float(), match_len.float(), match.float().mean(1),
            -tok_lp[:, c0:].mean(1), -tok_lp[:, :c0].mean(1), mink,
            ent.mean(1), ent[:, 0], tok_lp[:, c0]]
    return torch.stack(cols, 1).cpu().numpy()


def greedy_decode(model, prompts, n_new: int):
    """Real batched greedy decoding with a KV cache (all prompts have the same length)."""
    import torch

    with torch.no_grad():
        out = model(input_ids=prompts, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        gen = [nxt]
        for _ in range(n_new - 1):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            gen.append(nxt)
    return torch.cat(gen, 1)


# ---------------------------------------------------------------------- PH workers
def _seq_job(clouds: np.ndarray, prompt_len: int, also_raw: bool) -> list[dict]:
    return [topo.cloud_features(c, prompt_len, also_raw=also_raw) for c in clouds]


def _done_indices(out: Path, prefix: str) -> set:
    done = set()
    for f in glob.glob(str(out / f"{prefix}-*.parquet")):
        done.update(pd.read_parquet(f, columns=["index"])["index"].tolist())
    for f in glob.glob(str(out / f"{prefix}-*.npz")):
        done.update(np.load(f)["index"].tolist())
    return done


def _next_part(out: Path) -> int:
    parts = glob.glob(str(out / "loss-*.parquet")) + glob.glob(str(out / "hidden-*.parquet"))
    return 1 + max([int(Path(p).stem.split("-")[1]) for p in parts] or [-1])


def scan(name: str, step: int, tokens_by_index: dict | pd.Series, loss_idx, hidden_idx, out: Path,
         layers: list[int], prompt_len: int = 32, batch: int = 256, dtype: str = "fp32",
         probe: bool = False, also_raw: bool = True, workers: int = 30) -> None:
    """Run (or resume) one checkpoint pass. ``tokens_by_index`` maps sequence index -> 64 tokens."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    loss_todo = sorted(set(int(i) for i in loss_idx) - _done_indices(out, "loss"))
    hid_todo = sorted(set(int(i) for i in hidden_idx) - _done_indices(out, "hidden"))
    probe_todo = set(sorted(set(int(i) for i in hidden_idx) - _done_indices(out, "probe"))) if probe else set()
    todo = sorted(set(loss_todo) | set(hid_todo) | probe_todo)
    if not todo:
        log.info("%s step%d: nothing to do", name, step)
        return
    log.info("%s step%d: %d loss, %d hidden, %d probe sequences to do", name, step,
             len(loss_todo), len(hid_todo), len(probe_todo))
    hid_set, loss_set = set(hid_todo), set(loss_todo)
    ctx = mp.get_context("spawn")
    pool = cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) if hid_todo else None
    model = load_model(name, step, dtype)
    blocks = model.gpt_neox.layers
    n_layers = len(blocks)
    assert max(layers) <= n_layers, (layers, n_layers)
    captured: dict[int, torch.Tensor] = {}
    hooks = [blocks[l - 1].register_forward_hook(
        lambda mod, inp, outp, l=l: captured.__setitem__(l, outp[0] if isinstance(outp, tuple) else outp))
        for l in layers]
    d_model = model.config.hidden_size
    chunk = max(batch, min(8192, int(3e9 / (len(layers) * 64 * d_model * 4)) // batch * batch))
    part = _next_part(out)
    try:
        for c0 in range(0, len(todo), chunk):
            ids = todo[c0:c0 + chunk]
            loss_out, futures, probes = [], [], []
            for b0 in range(0, len(ids), batch):
                bid = ids[b0:b0 + batch]
                x = torch.as_tensor(np.stack([np.asarray(tokens_by_index[i], dtype=np.int64) for i in bid])).to(device())
                captured.clear()
                with torch.no_grad():
                    logits = model(input_ids=x).logits
                rows = loss_rows(logits, x, prompt_len)
                del logits
                for i, r in zip(bid, rows):
                    if i in loss_set:
                        loss_out.append([i, *r])
                need_h = [j for j, i in enumerate(bid) if i in hid_set]
                need_p = [j for j, i in enumerate(bid) if i in probe_todo]
                if need_h or need_p:
                    hs = torch.stack([captured[l].float() for l in layers], 1)    # (B, L, 64, d)
                    if need_p:
                        pooled = hs[need_p, :, prompt_len:].mean(2).half().cpu().numpy()
                        probes.append((np.array([bid[j] for j in need_p]), pooled))
                    if need_h:
                        arr = hs[need_h].cpu().numpy()
                        for j, a in zip(need_h, arr):
                            futures.append((bid[j], pool.submit(_seq_job, a, prompt_len, also_raw)))
                    del hs
            if loss_out:
                pd.DataFrame(loss_out, columns=["index", *LOSS_NAMES]).astype({"index": "int64"}) \
                    .to_parquet(out / f"loss-{part:05d}.parquet")
            if probes:
                np.savez(out / f"probe-{part:05d}.npz", index=np.concatenate([p[0] for p in probes]),
                         states=np.concatenate([p[1] for p in probes]))
            if futures:
                hrows, drows = [], []
                for i, fut in futures:
                    for l, r in zip(layers, fut.result()):
                        hrows.append({"index": i, "layer": l,
                                      **{f"geom_{n}": v for n, v in zip(topo.GEOM_NAMES, r["geom"])},
                                      **{f"t1_full_{n}": v for n, v in zip(topo.FEATURE_NAMES, r["t1_full"])},
                                      **{f"t1_cont_{n}": v for n, v in zip(topo.FEATURE_NAMES, r["t1_cont"])},
                                      **({f"t1_full_raw_{n}": v for n, v in zip(topo.FEATURE_NAMES, r["t1_full_raw"])}
                                         if also_raw else {})})
                        drows.append({"index": i, "layer": l, "h0": r["dgm_h0"].ravel().tolist(),
                                      "h1": r["dgm_h1"].ravel().tolist()})
                pd.DataFrame(hrows).to_parquet(out / f"hidden-{part:05d}.parquet")
                pd.DataFrame(drows).to_parquet(out / f"dgm-{part:05d}.parquet")
            log.info("%s step%d: part %d done (%d/%d)", name, step, part, min(c0 + chunk, len(todo)), len(todo))
            part += 1
    finally:
        for h in hooks:
            h.remove()
        if pool:
            pool.shutdown()
        del model
        torch.cuda.empty_cache()


def greedy_check(name: str, step: int, tokens: np.ndarray, prompt_len: int = 32, cont_len: int = 32,
                 batch: int = 256, dtype: str = "fp32") -> dict:
    """Agreement of teacher-forced extractability with real KV-cache greedy decoding."""
    import torch

    model = load_model(name, step, dtype)
    tf, gd = [], []
    for b0 in range(0, len(tokens), batch):
        x = torch.as_tensor(tokens[b0:b0 + batch].astype(np.int64)).to(device())
        with torch.no_grad():
            tf.append(loss_rows(model(input_ids=x).logits, x, prompt_len)[:, 0])
        gen = greedy_decode(model, x[:, :prompt_len], cont_len)
        gd.append((gen == x[:, prompt_len:prompt_len + cont_len]).all(1).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    tf, gd = np.concatenate(tf), np.concatenate(gd)
    return {"n": int(len(tf)), "agree": float((tf == gd).mean()), "tf_pos": int(tf.sum()),
            "greedy_pos": int(gd.sum()), "both_pos": int(((tf == 1) & (gd == 1)).sum())}


def read_parts(out: Path, prefix: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(out / f"{prefix}-*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    keys = ["index", "layer"] if "layer" in df.columns else ["index"]
    return df.drop_duplicates(keys, keep="last")


def read_probe(out: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(str(out / "probe-*.npz")))
    idx = np.concatenate([np.load(f)["index"] for f in files])
    st = np.concatenate([np.load(f)["states"] for f in files])
    _, first = np.unique(idx, return_index=True)
    return idx[first], st[first]


def disk_free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9
