"""Correctness checks before a full extraction (METHODOLOGY 2.5). `run_checks` returns a dict and raises
CheckFailed on any failure unless force=True."""
from __future__ import annotations

import json
import os

import numpy as np

from . import config as C
from . import extract as X
from . import prompts as P


class CheckFailed(RuntimeError):
    pass


def run_checks(key, model, tok_model, tok_fast, out_dir, n_batch=50, seed=0, tol=2e-3, cos_min=0.999,
               numbers=None, force=False, log=print):
    import torch
    rng = np.random.default_rng(seed); res = {}
    numbers = np.arange(1, C.N_MAX + 1) if numbers is None else np.asarray(numbers)
    L = X.n_layers(model); lidx = C.layer_indices(L); pad = X.pad_id(tok_model)
    ctxs = P.all_contexts()
    sample = [(str(rng.choice(ctxs)), int(rng.choice(numbers))) for _ in range(n_batch)]
    rows = []
    c1s7 = P.c1_generated(C.SEED_C1B)
    for ctx, n in sample:
        p, s, e = P.build(ctx, n, c1s7); pos = P.positions(tok_fast, p, s, e)
        rows.append(dict(ctx=ctx, n=n, prompt=p, ids=pos["ids"], num=pos["last"], next=max(pos["next"], pos["last"]), task=pos["task"], ok=pos["ok"]))
    # (a) batch vs single (also covers padding, e)
    items = [dict(ids=r["ids"], num=r["num"], next=r["next"], task=r["task"]) for r in rows]
    bat, gb, _ = X.run_batched(model, items, lidx, pad, batch=len(items))
    worst = 0.0
    for i, it in enumerate(items):
        one, _ = X.forward_states(model, [it["ids"]], {"num": [it["num"]], "next": [it["next"]]}, lidx, pad=pad)
        for k in ("num", "next"):
            a = bat[k][i]; b = one[k][0]
            rel = np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-12)
            worst = max(worst, float(rel.max()))
    res["a_batch_vs_single_max_rel"] = worst; res["a_pass"] = worst <= tol
    # (b) positions
    show = []
    for r in rows[:20]:
        toks = tok_fast.convert_ids_to_tokens(r["ids"])
        show.append(dict(ctx=r["ctx"], prompt=r["prompt"], P_num=toks[r["num"]], P_next=toks[r["next"]], ok=r["ok"]))
    res["b_positions"] = show
    # number tokens must cover the last character of n
    bad = 0
    for r in rows:
        toks = tok_fast.convert_ids_to_tokens(r["ids"])
        txt = tok_fast.convert_tokens_to_string([toks[r["num"]]]).strip()
        if r["ok"] and not str(r["n"]).endswith(txt[-1:] if txt else "#"):
            bad += 1
    res["b_last_char_mismatch"] = bad; res["b_pass"] = bad == 0
    # (c) copy check on C1
    nn = rng.choice(numbers, min(200, len(numbers)), replace=False); its = []; first = []
    for n in nn:
        p, s, e = P.build("C1", int(n)); pos = P.positions(tok_fast, p, s, e)
        its.append(dict(ids=pos["ids"], num=pos["last"], next=pos["next"], task=pos["task"])); first.append(pos["ids"][pos["first"]])
    _, g, _ = X.run_batched(model, its, lidx, pad, batch=64)
    res["c_copy_first_token_acc"] = float(np.mean(np.array(g) == np.array(first)))
    # (f) greedy via the output head on hidden_states[-1] equals the full model's argmax (single prompts)
    agree = []
    for it in its[:10]:
        with torch.inference_mode():
            full = model(input_ids=torch.tensor([it["ids"]], device=next(model.parameters()).device)).logits[0, -1].argmax().item()
        _, gr = X.forward_states(model, [it["ids"]], {"num": [it["num"]]}, lidx, pad=pad)
        agree.append(int(gr[0] == full))
    res["f_head_greedy_agrees"] = float(np.mean(agree)); res["f_pass"] = res["f_head_greedy_agrees"] == 1.0
    # (d) reproduce stored old C1/P_next vectors
    res.update(reproduce_old(key, model, tok_fast, lidx, pad, cos_min))
    json.dump(res, open(os.path.join(out_dir, "checks.json"), "w"), indent=2, default=str)
    fails = [k for k in ("a_pass", "b_pass", "d_pass", "f_pass") if res.get(k) is False]
    log(f"  checks: batch/single {worst:.2e}, copy acc {res['c_copy_first_token_acc']:.3f}, "
        f"old-vector cos min {res.get('d_cos_min')}, head greedy {res['f_head_greedy_agrees']:.2f}; fails: {fails or 'none'}")
    if fails and not force:
        raise CheckFailed(f"{key}: checks failed: {fails} (see {out_dir}/checks.json)")
    return res


def reproduce_old(key, model, tok_fast, lidx, pad, cos_min):
    if not os.path.exists(C.OLD_REF):
        return dict(d_pass=None, d_note="reference file missing")
    ref = np.load(C.OLD_REF); ns = ref["n"]; keys = [k for k in ref.files if k.startswith(key + "__")]
    if not keys:
        return dict(d_pass=None, d_note=f"no old vectors for {key}")
    L = max(lidx); old_layers = sorted(int(k.split("__")[1]) for k in keys)
    if max(old_layers) > L or ref[keys[0]].shape[1] != model.config.hidden_size:
        return dict(d_pass=None, d_note="model shape differs from the stored old vectors (test model?)")
    items = []
    for n in ns:
        p, s, e = P.build("C1", int(n)); pos = P.positions(tok_fast, p, s, e)
        items.append(dict(ids=pos["ids"], num=pos["last"], next=pos["next"], task=pos["task"]))
    res, _ = X.forward_states(model, [it["ids"] for it in items], {"next": [it["next"] for it in items]},
                              old_layers, pad=pad, want_greedy=False)
    cos = []
    for j, Lo in enumerate(old_layers):
        a = res["next"][:, j]; b = ref[f"{key}__{Lo}"]
        cos.append((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
    cos = np.array(cos)
    return dict(d_cos_min=float(cos.min()), d_cos_by_layer={int(l): float(c.min()) for l, c in zip(old_layers, cos)},
                d_pass=bool(cos.min() >= cos_min), d_layers=old_layers)
