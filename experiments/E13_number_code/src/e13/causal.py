"""Causal stage (METHODOLOGY Part 4): subspaces S_mag, S_dig(p), S_carry (+ random and shuffled-n controls),
interventions at the number's last token (project-out, subspace patching, steering), behaviours B1-B6 scored by
log-probability differences, the subspace x behaviour effect matrix and the pre-registered decision rule.
Layer, m and alpha are chosen on dev numbers only (G5); effects are reported on test numbers."""
from __future__ import annotations

import contextlib
import json
import os

import numpy as np

from src import e13_stage1 as E1
from src.e12c_heldout_r2 import kernel_pred

from . import config as C
from . import extract as X
from . import prompts as P
from .data import ModelData, dev_mask
from .panel import choose_h


# --------------------------------------------------------------------------- hooks
def decoder_layers(model):
    for path in ("gpt_neox.layers", "model.layers", "transformer.h", "model.decoder.layers", "transformer.blocks"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise ValueError("cannot find decoder layers")


@contextlib.contextmanager
def intervene(model, layer, positions, fn):
    """Edit the residual stream after block `layer - 1` (= hidden_states[layer]) at one position per batch row.
    fn(h [B, d] float32, rows) -> new h."""
    import torch
    blocks = decoder_layers(model); state = dict(pos=None, rows=None)

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["pos"] is None:
            return out
        b = torch.arange(h.shape[0], device=h.device); p = torch.as_tensor(state["pos"], device=h.device)
        new = fn(h[b, p].float(), state["rows"]).to(h.dtype); h = h.clone(); h[b, p] = new
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    handle = blocks[layer - 1].register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def seq_logprob(model, tok, prompts_ids, conts_ids, edit=None):
    """Sum log-probabilities of continuations given prompts (batched, right padded). edit = (layer, positions, fn)."""
    import torch
    dev = next(model.parameters()).device
    seqs = [p + c for p, c in zip(prompts_ids, conts_ids)]; T = max(map(len, seqs)); B = len(seqs)
    pad = X.pad_id(tok); inp = torch.full((B, T), pad, dtype=torch.long); att = torch.zeros((B, T), dtype=torch.long)
    for i, s in enumerate(seqs):
        inp[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    cm = intervene(model, edit[0], edit[1], edit[2]) if edit else contextlib.nullcontext()
    with cm as st, torch.inference_mode():
        if edit:
            st["pos"] = edit[1]; st["rows"] = np.arange(B)
        hs = model.base_model(input_ids=inp.to(dev), attention_mask=att.to(dev)).last_hidden_state
        head = model.get_output_embeddings(); out = np.zeros(B)
        for i, (p, c) in enumerate(zip(prompts_ids, conts_ids)):
            idx = torch.arange(len(p) - 1, len(p) + len(c) - 1, device=dev)
            lp = head(hs[i, idx]).float().log_softmax(-1)
            out[i] = lp[torch.arange(len(c)), torch.tensor(c, device=dev)].sum().item()
    return out


# --------------------------------------------------------------------------- behaviours
def _ok_number(tok, prompt, s, e, family):
    pos = P.positions(tok, prompt, s, e)
    return pos if pos["ok"] and (family != "chunk" or pos["n_tokens"] == 1) else None


def build_items(tok, family, numbers, kind, n_items, rng):
    """Items: dict(prompt_ids, pos (P_num of the edited number), cands [list of continuation ids], correct, carry, n)."""
    items = []; tries = 0; numbers = np.asarray(numbers); by_lead = {}
    for v in numbers:                                       # B1 partner: same length, different leading digit
        for ld in "123456789":
            if str(v)[0] != ld:
                by_lead.setdefault((len(str(v)), ld), []).append(int(v))
    while len(items) < n_items and tries < 50 * n_items:
        tries += 1; n = int(rng.choice(numbers)); carry = None
        if kind == "B1":
            others = by_lead.get((len(str(n)), str(n)[0]), None)
            if others is None or not len(others):
                continue
            b = int(rng.choice(others)); first = rng.random() < 0.5
            a1, a2 = (n, b) if first else (b, n); prompt = f"Which is larger, {a1} or {a2}? Answer:"
            s = len("Which is larger, ") if first else len(f"Which is larger, {a1} or ")
            cands = [f" {max(n, b)}", f" {min(n, b)}"]
        elif kind in ("B2", "B3"):
            prompt = f"The {'last' if kind == 'B2' else 'first'} digit of {n} is"; s = prompt.index(str(n))
            dgt = str(n)[-1] if kind == "B2" else str(n)[0]
            cands = [f" {dgt}"] + [f" {d}" for d in "0123456789" if d != dgt]
        elif kind == "B4":
            carry = n % 10 == 9
            if (rng.random() < 0.5) != carry:
                continue
            prompt = f"{n} + 1 ="; s = 0; cands = [f" {n + 1}", f" {n}"]
        elif kind == "B5":
            b = int(rng.integers(10, 100)); carry = (n % 10 + b % 10) >= 10
            if (rng.random() < 0.5) != carry:
                continue
            prompt = f"{n} + {b} ="; s = 0; wrong = n + b - 10 if carry else n + b + 10
            cands = [f" {n + b}", f" {wrong}"]
        elif kind == "B6":
            prompt = P.c1_seed42()[n]; s = len(prompt) - 1 - len(str(n))
            alt = n - n % 10 + (n % 10 + 1) % 10; cands = [str(n), str(alt)]
        else:
            raise ValueError(kind)
        e = s + len(str(n)); pos = _ok_number(tok, prompt, s, e, family)
        if pos is None:
            continue
        ids = pos["ids"]; cid = [tok(c, add_special_tokens=False)["input_ids"] for c in cands]
        if kind == "B1" and cid[0] == cid[1]:       # (full answers are scored, so a shared first token, e.g. a space, is fine)
            continue
        items.append(dict(n=n, prompt_ids=ids, pos=pos["last"], cands=cid, carry=carry, kind=kind))
    return items


def score(model, tok, items, edit_fn=None, layer=None, batch=64):
    """Log-prob margin per item: logP(correct) - max logP(wrong). edit_fn(item_rows, h) applied at P_num."""
    out = np.zeros(len(items))
    for s0 in range(0, len(items), batch):
        chunk = items[s0:s0 + batch]; pids, cids, owner, pos = [], [], [], []
        for k, it in enumerate(chunk):
            for c in it["cands"]:
                pids.append(it["prompt_ids"]); cids.append(c); owner.append(k); pos.append(it["pos"])
        owner = np.array(owner)
        edit = (layer, pos, (lambda h, rows: edit_fn(h, owner[rows] + s0))) if edit_fn else None
        lp = seq_logprob(model, tok, pids, cids, edit)
        for k in range(len(chunk)):
            v = lp[owner == k]; out[s0 + k] = v[0] - v[1:].max()
    return out


# --------------------------------------------------------------------------- subspaces
def orth(M):
    if M.shape[1] == 0:
        return M
    Q, R = np.linalg.qr(M); keep = np.abs(np.diag(R)) > 1e-8 * np.abs(np.diag(R)).max()
    return Q[:, keep]


def subspaces(Xraw, n, L, rng, m_dev=None):
    """Raw-space orthonormal bases (d x k). Xraw: centred-able raw vectors [N, d] of training numbers."""
    mu = Xraw.mean(0); A = Xraw - mu; D = min(50, len(n) - 1)
    Vt = np.linalg.svd(A, full_matrices=False)[2][:D]; Z = A @ Vt.T
    h = choose_h(Z, n, 0); F = kernel_pred(Z, n, n, h); Fc = F - F.mean(0)
    w, U = np.linalg.eigh(Fc.T @ Fc); w = w[::-1]; U = U[:, ::-1]; share = np.cumsum(w) / w.sum()
    m = m_dev or int(min(5, np.searchsorted(share, 0.90) + 1))
    S = dict(mag=orth(Vt.T @ U[:, :m]))
    add = E1.additive(Z, E1.digits(n, L))
    for p, Pp in enumerate(add["pos"]):
        name = "units" if p == L - 1 else ("leading" if p == 0 else f"pos{L - 1 - p}")
        Cc = Pp["C"]; S[f"dig_{name}"] = orth(Vt.T @ (Cc[1:] - Cc[0]).T)
    i, j, c = E1.steps_of(n)
    if len(c):
        res = (Z[j] - Z[i]) - (add["F"][j] - add["F"][i]); base = res[c == 0].mean(0)
        vecs = [res[c == lv].mean(0) - base for lv in E1.h2_levels(c)]
        if vecs:
            S["carry"] = orth(Vt.T @ np.array(vecs).T)
    ns = rng.permutation(n); Fs = kernel_pred(Z, ns, ns, h); Fs = Fs - Fs.mean(0)
    ws, Us = np.linalg.eigh(Fs.T @ Fs); S["shuffled_mag"] = orth(Vt.T @ Us[:, ::-1][:, :m])
    union = orth(np.hstack([v for k, v in S.items() if k != "shuffled_mag"]))
    for k in list(S):
        if k == "shuffled_mag":
            continue
        for r in range(5):
            G = rng.standard_normal((A.shape[1], S[k].shape[1])); G -= union @ (union.T @ G); S[f"random_{k}_{r}"] = orth(G)
    smooth = dict(mu=mu, Vt=Vt, Z=Z, n=n, h=h, F=F)
    return S, m, smooth


# --------------------------------------------------------------------------- interventions
def make_edit(kind, Q, mu, other=None, delta=None):
    import torch
    Qt = None

    def fn(h, rows):
        nonlocal Qt
        if Qt is None and Q is not None:
            Qt = torch.tensor(Q, dtype=torch.float32, device=h.device)
        if kind == "ablate":
            m = torch.tensor(mu, dtype=torch.float32, device=h.device)
            return h - ((h - m) @ Qt) @ Qt.T
        if kind == "patch":
            o = torch.tensor(other[rows], dtype=torch.float32, device=h.device)
            return h + ((o - h) @ Qt) @ Qt.T
        if kind == "steer":
            return h + torch.tensor(delta[rows], dtype=torch.float32, device=h.device)
        raise ValueError(kind)
    return fn


def states_at(model, tok, items, layer):
    """Clean hidden_states[layer] at P_num for each item (for patching sources)."""
    its = [dict(ids=it["prompt_ids"], num=it["pos"], next=it["pos"], task=it["pos"]) for it in items]
    res, _, _ = X.run_batched(model, its, [layer], X.pad_id(tok), 64)
    return res["num"][:, 0]


def effects(base, edited, B=1000, seed=0):
    d = base - edited; rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(len(d)), _d=d)


def run_model(key, model, tok, data_dir, out_path, n_items=500, sets=None, log=print, seed=0):
    md = ModelData(data_dir); fam = md.family; rng = np.random.default_rng(seed)
    sets = sets or (["S2", "S3", "S4"] if fam == "digit" else ["S2", "S3"])
    L_model = X.n_layers(model); slots = [sl for sl, li in enumerate(md.lidx) if li >= 1]
    nums = {s: md.set_numbers(s, ["AVG5"]) for s in sets}; dmask = {s: dev_mask(v) for s, v in nums.items()}
    beh = ["B1", "B2", "B3", "B4", "B5", "B6"]
    items = {"dev": {}, "test": {}}
    for split in ("dev", "test"):
        for b in beh:
            lst = []
            for s in sets:
                pool = nums[s][dmask[s]] if split == "dev" else nums[s][~dmask[s]]
                k = max(4, (min(100, n_items) if split == "dev" else n_items) // len(sets))
                lst += build_items(tok, fam, pool, b, k, rng)
            items[split][b] = lst
    log(f"  causal {key}: items " + ", ".join(f"{b}={len(items['test'][b])}" for b in beh))
    # ---- choose the layer on dev: S_mag ablation effect on B1 minus its random control
    best = None; sel = {}
    for sl in slots:
        li = md.lidx[sl]; Sd = {}
        for s in sets:
            dn = nums[s][dmask[s]]
            if len(dn) < 20:
                continue
            Xr = md._arr("vec_avg_num_alllayers.npy")[np.searchsorted(md.numbers, dn), li].astype(np.float64)
            Sd[s] = subspaces(Xr, dn, int(s[1]), rng)
        if not Sd:
            continue
        s0 = next(iter(Sd)); S, m, sm = Sd[s0]; dev_b1 = [it for it in items["dev"]["B1"]]
        if not dev_b1:
            continue
        base = score(model, tok, dev_b1)
        e_mag = (base - score(model, tok, dev_b1, make_edit("ablate", S["mag"], sm["mu"]), li)).mean()
        e_rnd = (base - score(model, tok, dev_b1, make_edit("ablate", S["random_mag_0"], sm["mu"]), li)).mean()
        sel[int(li)] = dict(mag=float(e_mag), random=float(e_rnd), m=int(m))
        if best is None or e_mag - e_rnd > best[0]:
            best = (e_mag - e_rnd, sl, m)
    if best is None:
        return dict(skipped="no dev items")
    sl, m = best[1], best[2]; li = md.lidx[sl]
    log(f"  causal {key}: layer {li} chosen on dev (m = {m})")
    # ---- subspaces on training (non-dev) numbers, per set; evaluate on test items of that set
    out = dict(layer=int(li), m=int(m), dev_layer_scan=sel, matrix=[])
    for s in sets:
        tn = nums[s][~dmask[s]]
        Xr = md._arr("vec_avg_num_alllayers.npy")[np.searchsorted(md.numbers, tn), li].astype(np.float64)
        S, _, sm = subspaces(Xr, tn, int(s[1]), rng, m_dev=m)
        for b in beh:
            its = [it for it in items["test"][b] if C.SETS[s][0] <= it["n"] <= C.SETS[s][1]]
            if len(its) < 10:
                continue
            base = score(model, tok, its)
            groups = {"all": np.ones(len(its), bool)}
            if b in ("B4", "B5"):
                cf = np.array([bool(it["carry"]) for it in its]); groups = {"carry": cf, "nocarry": ~cf}
            # patch source for item k: the clean state of item N-1-k (another number of the same set and behaviour)
            src = states_at(model, tok, its[::-1], li); alt = [it["n"] for it in its[::-1]]
            for name, Q in S.items():
                for kind in ("ablate", "patch"):
                    fn = make_edit(kind, Q, sm["mu"], other=src)
                    ed = score(model, tok, its, fn, li)
                    for g, mask in groups.items():
                        eff = effects(base[mask], ed[mask])
                        out["matrix"].append(dict(set=s, behaviour=b if g == "all" else f"{b}_{g}", subspace=name,
                                                  intervention=kind, dim=int(Q.shape[1]), **{k: v for k, v in eff.items() if k != "_d"},
                                                  d=eff["_d"].tolist()))
            # steering along f(n') - f(n)
            f_at = {int(v): sm["F"][k] @ sm["Vt"] for k, v in enumerate(tn)}
            for alpha in (0.5, 1.0, 2.0):
                delta = np.array([alpha * (f_at.get(int(a), 0) - f_at.get(int(it["n"]), 0)) for it, a in zip(its, alt)])
                ed = score(model, tok, its, make_edit("steer", None, None, delta=delta), li)
                for g, mask in groups.items():
                    eff = effects(base[mask], ed[mask])
                    out["matrix"].append(dict(set=s, behaviour=b if g == "all" else f"{b}_{g}", subspace=f"steer_mag_a{alpha}",
                                              intervention="steer", dim=0, **{k: v for k, v in eff.items() if k != "_d"}, d=eff["_d"].tolist()))
    out["decision"] = decide(out["matrix"])
    for r in out["matrix"]:
        r.pop("d", None)
    json.dump(out, open(out_path, "w"), default=float)
    return out


PREDICTIONS = [  # subspace, target behaviours, off-target behaviours
    ("mag", ["B1"], ["B2", "B6"]),
    ("dig_units", ["B2", "B6"], ["B1"]),
    ("carry", ["B4_carry", "B5_carry"], ["B4_nocarry", "B5_nocarry"]),
]


def decide(matrix, B=1000, seed=0):
    """Target effect >= 2 x random (CI of target - 2 random above 0) and off-target <= 1.2 x random
    (CI of off - 1.2 random below 0), random = mean over the 5 random subspaces of equal dimension."""
    rng = np.random.default_rng(seed); out = []
    rows = {(r["set"], r["behaviour"], r["subspace"], r["intervention"]): np.array(r["d"]) for r in matrix if "d" in r}
    for s in sorted({r["set"] for r in matrix}):
        for sub, tgt, off in PREDICTIONS:
            for kind in ("ablate", "patch"):
                def cmp(b, factor):
                    if (s, b, sub, kind) not in rows:
                        return None
                    d = rows[(s, b, sub, kind)]; R = [rows[(s, b, f"random_{sub}_{r}", kind)] for r in range(5) if (s, b, f"random_{sub}_{r}", kind) in rows]
                    if not R:
                        return None
                    rr = np.mean(R, 0); diff = d - factor * rr
                    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)]
                    return dict(effect=float(d.mean()), random=float(rr.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
                t = {b: cmp(b, 2.0) for b in tgt}; o = {b: cmp(b, 1.2) for b in off}
                okt = all(v is not None and v["lo"] > 0 for v in t.values()); oko = all(v is None or v["hi"] < 0 for v in o.values())
                out.append(dict(set=s, subspace=sub, intervention=kind, target=t, off_target=o, passed=bool(okt and oko)))
    return out
