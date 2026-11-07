"""Extraction (METHODOLOGY Part 2): contexts C1-C8 (+ second wordings), read positions P_num / P_next / P_task,
9 stored layers, the neutral-context average at all layers, static embeddings, meta.csv, config.json, P7 items.

Work is done in chunks of numbers; every chunk runs all contexts, writes its rows into the .npy memmaps and is
recorded in progress.json, so an interrupted run resumes at the next chunk."""
from __future__ import annotations


def _saved_token():
    """Token saved by `hf auth login` / `huggingface-cli login` (never printed)."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None

import json
import os
import time

import numpy as np

from . import config as C
from . import prompts as P

POS_KINDS = ("num", "next")


# --------------------------------------------------------------------------- model loading
def load_model(key, dtype="float32", device=None, revision=None, model_id=None, log=print):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = C.MODELS.get(key, {}); mid = model_id or spec["hf"]; tok_kw = {}
    token = os.environ.get("HF_TOKEN") or _saved_token() or None
    kw = dict(revision=revision, token=token)
    if spec.get("trust_remote_code"):
        kw["trust_remote_code"] = True
    if spec.get("datadecide"):
        import hf_olmo  # noqa: F401  (registers the DataDecide model type)
    tok_fast = AutoTokenizer.from_pretrained(mid, use_fast=True, **kw)
    tok_model = tok_fast if spec.get("use_fast_model", True) else AutoTokenizer.from_pretrained(mid, use_fast=False, **kw)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=getattr(torch, dtype), **kw).to(device).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    sha = None
    try:
        from huggingface_hub import HfApi
        sha = HfApi(token=token).model_info(mid, revision=revision).sha
    except Exception as e:                                   # offline / tiny test models
        log(f"  (could not resolve commit hash: {type(e).__name__})")
    return model, tok_model, tok_fast, dict(model_id=mid, revision=revision, commit=sha, dtype=dtype, device=device)


def n_layers(model):
    cfg = model.config
    for k in ("num_hidden_layers", "n_layer", "num_layers"):
        if getattr(cfg, k, None):
            return int(getattr(cfg, k))
    raise ValueError("cannot find the number of layers")


def pad_id(tok):
    if tok.pad_token_id is not None:
        return tok.pad_token_id
    return tok.eos_token_id if tok.eos_token_id is not None else 0


# --------------------------------------------------------------------------- forward pass
def forward_states(model, ids_list, gathers, layer_idx, all_layer_key=None, pad=0, want_greedy=True):
    """Right-padded batch; returns {name: [B, len(layer_idx), d]} for each gather (positions per prompt),
    {name+'_all': [B, L+1, d]} for all_layer_key, and the greedy next token after the last real token."""
    import torch
    dev = next(model.parameters()).device; B = len(ids_list); T = max(len(x) for x in ids_list)
    inp = torch.full((B, T), pad, dtype=torch.long); att = torch.zeros((B, T), dtype=torch.long)
    for i, x in enumerate(ids_list):
        inp[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
    with torch.inference_mode():
        out = model.base_model(input_ids=inp.to(dev), attention_mask=att.to(dev), output_hidden_states=True)
        hs = out.hidden_states; b = torch.arange(B, device=dev); res = {}
        for name, pos in gathers.items():
            p = torch.tensor(np.maximum(np.asarray(pos), 0), device=dev)
            res[name] = torch.stack([hs[l][b, p] for l in layer_idx], 1).float().cpu().numpy()
            if name == all_layer_key:
                res[name + "_all"] = torch.stack([h[b, p] for h in hs], 1).float().cpu().numpy()
        greedy = None
        if want_greedy:
            last = att.sum(1).to(dev) - 1
            logits = model.get_output_embeddings()(hs[-1][b, last])
            greedy = logits.argmax(-1).cpu().numpy()
    return res, greedy


def run_batched(model, items, layer_idx, pad, batch, all_layer=False, log=print):
    """items: list of dict(ids, num, next, task). Sorted by length; halves the batch on out-of-memory."""
    import torch
    order = np.argsort([len(it["ids"]) for it in items], kind="stable"); out = {}; greedy = np.zeros(len(items), int)
    s = 0
    while s < len(order):
        sel = order[s:s + batch]
        g = {k: [items[i][k] for i in sel] for k in ("num", "next", "task")}
        try:
            r, gr = forward_states(model, [items[i]["ids"] for i in sel], g, layer_idx,
                                   all_layer_key="num" if all_layer else None, pad=pad)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); batch = max(1, batch // 2); log(f"  OOM: batch -> {batch}"); continue
        for k, v in r.items():
            if k not in out:
                out[k] = np.zeros((len(items),) + v.shape[1:], np.float32)
            out[k][sel] = v
        greedy[sel] = gr; s += len(sel)
    return out, greedy, batch


# --------------------------------------------------------------------------- planning
def plan(tok_fast, tok_model, numbers, contexts, add_special_tokens=True):
    c1s7 = P.c1_generated(C.SEED_C1B); rows = {}
    for ctx in contexts:
        lst = []
        for n in numbers:
            prompt, s, e = P.build(ctx, int(n), c1s7)
            ids_m = tok_model(prompt, add_special_tokens=add_special_tokens)["input_ids"] if tok_model is not tok_fast else None
            pos = P.positions(tok_fast, prompt, s, e, ids_m, add_special_tokens)
            pos["prompt"] = prompt; pos["num"] = pos["last"]; lst.append(pos)
        rows[ctx] = lst
    return rows


def meta_table(numbers, rows, tok_fast):
    import pandas as pd
    df = pd.DataFrame({"n": numbers, "set": [next(s for s, (a, b) in C.SETS.items() if a <= n <= b) for n in numbers]})
    for ctx, lst in rows.items():
        df[f"ntok_{ctx}"] = [r["n_tokens"] for r in lst]
        df[f"ids_{ctx}"] = [" ".join(map(str, r["ids"][r["first"]:r["last"] + 1])) if r["first"] >= 0 else "" for r in lst]
        df[f"pnum_{ctx}"] = [r["last"] for r in lst]
        df[f"pnext_{ctx}"] = [r["next"] for r in lst]
        df[f"ptask_{ctx}"] = [r["task"] for r in lst]
        df[f"ok_{ctx}"] = [r["ok"] for r in lst]
        df[f"why_{ctx}"] = [r["reason"] for r in lst]
    return df


# --------------------------------------------------------------------------- main extraction
def store_paths(out_dir, contexts):
    paths = {}
    for ctx in contexts:
        kinds = ("num",) if ctx in C.VARIANTS else POS_KINDS
        for k in kinds:
            paths[(ctx, k)] = os.path.join(out_dir, f"vec_{ctx}_{k}.npy")
        if C.TASK_POS.get(ctx) == "task":
            paths[(ctx, "task")] = os.path.join(out_dir, f"vec_{ctx}_task.npy")
    return paths


def _memmap(path, shape, dtype=np.float16):
    if os.path.exists(path):
        m = np.load(path, mmap_mode="r+")
        assert m.shape == shape, (path, m.shape, shape)
        return m
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def extract_model(key, out_dir, numbers=None, contexts=None, dtype="float32", batch=256, chunk=2000,
                  revision=None, model_id=None, deadline=None, log=print, with_p7=True, model_bundle=None):
    import pandas as pd
    import torch
    import transformers
    os.makedirs(out_dir, exist_ok=True)
    numbers = np.arange(1, C.N_MAX + 1) if numbers is None else np.asarray(numbers)
    contexts = contexts or P.all_contexts()
    t0 = time.time()
    model, tok_model, tok_fast, info = model_bundle or load_model(key, dtype, revision=revision, model_id=model_id, log=log)
    L = n_layers(model); lidx = C.layer_indices(L); d = model.config.hidden_size; N = len(numbers)
    prog_path = os.path.join(out_dir, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else dict(chunks_done=[], seconds=0.0, prompts=0)
    log(f"  {key}: L={L}, d={d}, stored layers {lidx}, N={N}, contexts={len(contexts)}")
    rows = plan(tok_fast, tok_model, numbers, contexts)
    meta = meta_table(numbers, rows, tok_fast)
    meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    np.save(os.path.join(out_dir, "numbers.npy"), numbers)
    paths = store_paths(out_dir, contexts)
    mm = {k: _memmap(p, (N, len(lidx), d)) for k, p in paths.items()}
    avg_path = os.path.join(out_dir, "vec_avg_num_alllayers.npy")
    mavg = _memmap(avg_path, (N, L + 1, d))
    greedy = {c: np.full(N, -1, int) for c in contexts}
    gpath = os.path.join(out_dir, "greedy.npz")
    if os.path.exists(gpath):
        g = np.load(gpath); greedy.update({c: g[c] for c in g.files if c in greedy})
    pad = pad_id(tok_model)
    cfg = dict(model_key=key, **info, n_layers=L, hidden_size=d, layer_indices=lidx, layer_fracs=C.LAYER_FRACS,
               torch=torch.__version__, transformers=transformers.__version__, tokenizer=tok_fast.name_or_path,
               tokenizer_class=type(tok_model).__name__, contexts={c: (C.CONTEXTS.get(c) or C.VARIANTS[c]) for c in contexts},
               neutral_average=C.NEUTRAL, seeds=dict(c1=C.SEED_C1, c1b=C.SEED_C1B, dev=C.SEED_DEV),
               final_norm_note="hidden_states[L] is the HF last hidden state (final norm applied for GPT-NeoX, Llama, "
                               "Qwen2, OLMo-2); hidden_states[0] = embedding output",
               numbers=[int(numbers.min()), int(numbers.max()), int(N)], padding="right, gathered by position index")
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    chunks = [np.arange(s, min(s + chunk, N)) for s in range(0, N, chunk)]
    for ci, rr in enumerate(chunks):
        if ci in prog["chunks_done"]:
            continue
        if deadline and time.time() > deadline:
            log("  time limit reached during extraction; stopping (resumable)"); return dict(prog, done=False)
        tc = time.time(); acc = np.zeros((len(rr), L + 1, d), np.float32); nacc = np.zeros(len(rr))
        for ctx in contexts:
            items = []
            for i in rr:
                r = rows[ctx][i]
                items.append(dict(ids=r["ids"], num=r["last"], next=r["next"] if r["next"] >= 0 else r["last"], task=r["task"]))
            res, gr, batch = run_batched(model, items, lidx, pad, batch, all_layer=ctx in C.NEUTRAL, log=log)
            greedy[ctx][rr] = gr
            ok = np.array([rows[ctx][i]["ok"] for i in rr])
            for k in ("num", "next", "task"):
                if (ctx, k) in mm:
                    v = res[k].copy(); v[~ok] = np.nan; mm[(ctx, k)][rr] = v.astype(np.float16)
            if ctx in C.NEUTRAL:
                a = res["num_all"]; acc[ok] += a[ok]; nacc[ok] += 1
            prog["prompts"] += len(rr)
        avg = acc / np.maximum(nacc, 1)[:, None, None]; avg[nacc == 0] = np.nan
        mavg[rr] = avg.astype(np.float16)
        for m in list(mm.values()) + [mavg]:
            m.flush()
        np.savez(gpath, **greedy)
        prog["chunks_done"].append(ci); prog["seconds"] += time.time() - tc
        json.dump(prog, open(prog_path, "w"))
        log(f"  chunk {ci + 1}/{len(chunks)} ({len(rr)} numbers x {len(contexts)} contexts) in {time.time() - tc:.0f}s")
    # copy check (2.5c) and greedy answers
    meta["copy_correct_C1"] = [int(greedy["C1"][i] == (int(rows["C1"][i]["ids"][rows["C1"][i]["first"]])
                                if rows["C1"][i]["first"] >= 0 else -9)) for i in range(N)] if "C1" in contexts else -1
    for c in ("C5", "C6"):
        if c in contexts:
            meta[f"greedy_{c}"] = greedy[c]
    meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    static_embeddings(model, tok_fast, numbers, out_dir)
    if with_p7 and not os.path.exists(os.path.join(out_dir, "p7_months.npy")):
        extract_p7(model, tok_fast, tok_model, lidx, pad, batch, out_dir)
    prog["done"] = True; prog["wall_seconds_total"] = time.time() - t0
    json.dump(prog, open(prog_path, "w"))
    return dict(prog)


def static_embeddings(model, tok, numbers, out_dir):
    """Input-embedding and unembedding rows for numbers that are one token, with (' 347') and without ('347') a space."""
    import pandas as pd
    import torch
    E = model.get_input_embeddings().weight; U = model.get_output_embeddings().weight
    rows = []; idsE = np.full((len(numbers), 2), -1)
    for i, n in enumerate(numbers):
        for j, s in enumerate((f" {n}", f"{n}")):
            ids = tok(s, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                idsE[i, j] = ids[0]
        rows.append(dict(n=int(n), id_space=int(idsE[i, 0]), id_nospace=int(idsE[i, 1])))
    d = E.shape[1]; emb = np.full((len(numbers), 2, d), np.nan, np.float16); une = np.full_like(emb, np.nan)
    with torch.no_grad():
        for j in range(2):
            k = idsE[:, j] >= 0
            if k.any():
                t = torch.tensor(idsE[k, j], device=E.device)
                emb[k, j] = E[t].float().cpu().numpy(); une[k, j] = U[t].float().cpu().numpy()
    np.save(os.path.join(out_dir, "static_embed.npy"), emb); np.save(os.path.join(out_dir, "static_unembed.npy"), une)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "static_ids.csv"), index=False)


def extract_p7(model, tok_fast, tok_model, lidx, pad, batch, out_dir):
    """Days and months (P7): last token of the item in each of 20 templates, at the 9 stored layers."""
    for name, items, tpls in (("days", C.DAYS, C.DAY_TEMPLATES), ("months", C.MONTHS, C.MONTH_TEMPLATES)):
        its = []
        for t in tpls:
            for x in items:
                p = t.replace("{X}", x); s = t.index("{X}"); e = s + len(x)
                pos = P.positions(tok_fast, p, s, e)
                its.append(dict(ids=pos["ids"], num=pos["last"], next=max(pos["next"], pos["last"]), task=pos["task"]))
        res, _, _ = run_batched(model, its, lidx, pad, batch)
        arr = res["num"].reshape(len(tpls), len(items), len(lidx), -1)
        np.save(os.path.join(out_dir, f"p7_{name}.npy"), arr.astype(np.float16))
