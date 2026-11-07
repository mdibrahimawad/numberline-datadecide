"""Training and data stage (METHODOLOGY Part 5): Pythia-1.4B (16 checkpoints) and OLMo-2-1B (12 checkpoints),
then the 25 DataDecide recipes. Same extraction protocol (C1..C8), behaviour accuracies, light panel, emergence times
with a permutation test of the predicted order, and regressions on the recipes' numeral-frequency alpha."""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd

from . import config as C

PRED_ORDER = ["units_categories", "leading_ordinality", "carry_structure", "lexicon"]


def revisions(family):
    if family == "pythia-1.4b":
        return [f"step{s}" for s in C.PYTHIA_STEPS]
    if family == "olmo2-1b":
        from src.e11_checkpoints import olmo_revisions
        return olmo_revisions(C.MODELS["olmo2-1b"]["hf"])
    raise ValueError(family)


def datadecide_recipes():
    return pd.read_csv(C.ALPHA)["recipe"].tolist()


def register_datadecide():
    for r in datadecide_recipes():
        C.MODELS.setdefault(f"dd-{r}", dict(hf=f"allenai/DataDecide-{r}-1B", family="chunk", sub="neox", stage="D",
                                            datadecide=True))


def behaviour_acc(model, tok, family, numbers, rng, n=200):
    from .causal import build_items, score
    out = {}
    for b in ("B1", "B4", "B5", "B6"):
        its = build_items(tok, family, numbers, b, n, rng)
        if not its:
            continue
        m = score(model, tok, its) > 0
        if b in ("B4", "B5"):
            cf = np.array([bool(it["carry"]) for it in its])
            out[f"{b}_carry_acc"] = float(m[cf].mean()) if cf.any() else None
            out[f"{b}_nocarry_acc"] = float(m[~cf].mean()) if (~cf).any() else None
        else:
            out[f"{b}_acc"] = float(m.mean())
    return out


def clear_hf_cache(model_id):
    from huggingface_hub import scan_cache_dir
    try:
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_id:
                shutil.rmtree(repo.repo_path, ignore_errors=True)
    except Exception:
        pass


def run_revision(key, model_id, revision, out_dir, numbers, dtype, batch, deadline, log, workers, executor=None):
    """Extract + behaviour + light analysis for one checkpoint / recipe."""
    from . import analyse
    from .extract import extract_model, load_model
    done = os.path.join(out_dir, "revision.done")
    if os.path.exists(done):
        return json.load(open(done))
    os.makedirs(out_dir, exist_ok=True)
    bundle = load_model(key if key in C.MODELS else "pythia-1.4b", dtype, revision=revision, model_id=model_id, log=log)
    model, tok_model, tok_fast, info = bundle
    r = extract_model(key, out_dir, numbers=numbers, contexts=list(C.CONTEXTS), dtype=dtype, batch=batch,
                      revision=revision, model_id=model_id, deadline=deadline, log=log, with_p7=False, model_bundle=bundle)
    if not r.get("done"):
        return dict(done=False)
    fam = C.MODELS.get(key, {}).get("family", "chunk")
    beh = behaviour_acc(model, tok_fast, fam, numbers[(numbers >= 10) & (numbers <= 999)], np.random.default_rng(0))
    meta = pd.read_csv(os.path.join(out_dir, "meta.csv")); beh["copy_first_token_acc"] = float(meta["copy_correct_C1"].mean())
    json.dump(beh, open(os.path.join(out_dir, "behaviour.json"), "w"))
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    clear_hf_cache(model_id)
    conds = [("AVG5", "num"), ("C1", "num")]
    analyse.run_model(out_dir, workers=workers, deadline=deadline, log=log, executor=executor,
                      conditions=conds, light=True, quick=True)
    res = dict(done=True, revision=revision, behaviour=beh)
    json.dump(res, open(done, "w"))
    return res


# --------------------------------------------------------------------------- metrics over checkpoints
def summary_metrics(rev_dir, cond="AVG5_num"):
    """Per (layer slot, set): the four predicted-order metrics."""
    rows = []
    cd = os.path.join(rev_dir, "cells")
    for f in sorted(os.listdir(cd)) if os.path.isdir(cd) else []:
        if not f.startswith(f"panel__{cond}__"):
            continue
        J = json.load(open(os.path.join(cd, f))).get("result") or {}
        if "h1" not in J or not J.get("h1"):
            continue
        sl = J["layer_slot"]; s = J["set"]; pos = J["h1"]["positions"]
        h3f = os.path.join(cd, f"h3__L{sl}__{s}.json"); rel = None
        if os.path.exists(h3f):
            rel = ((json.load(open(h3f)).get("result") or {}).get("reliability") or {}).get("reliability")
        e1 = ((J.get("h2") or {}).get("h2a") or {}).get("e", [None])[0]
        lin_units = (J.get("m4") or {}).get("lin_acc_units")
        rows.append(dict(slot=sl, set=s, units_categories=lin_units, leading_ordinality=pos[0]["O1"],
                         carry_structure=(abs(e1) if e1 is not None else None), lexicon=rel,
                         ordinality_decay=J["h1"]["delta"], units_template_simplex=float(pos[-1]["template"] == "simplex")))
    return pd.DataFrame(rows)


def emergence(steps, values, frac=0.5):
    """First step (log-interpolated) where the metric reaches `frac` of its final value."""
    v = np.asarray(values, float); s = np.log1p(np.asarray(steps, float))
    if not np.isfinite(v).all() or v[-1] <= 0:
        return None
    tgt = frac * v[-1]; k = np.where(v >= tgt)[0]
    if not len(k):
        return None
    k = k[0]
    if k == 0:
        return float(steps[0])
    x = s[k - 1] + (tgt - v[k - 1]) / (v[k] - v[k - 1]) * (s[k] - s[k - 1]) if v[k] != v[k - 1] else s[k]
    return float(np.expm1(x))


def order_test(table, n_perm=10000, seed=0):
    """table: rows (cell) x columns PRED_ORDER of emergence steps. Statistic = mean Kendall tau between predicted
    and observed order over cells; null = metric labels permuted within each cell."""
    from scipy.stats import kendalltau
    T = table[PRED_ORDER].to_numpy(float); T = T[np.isfinite(T).sum(1) >= 3]
    if not len(T):
        return dict(tau=None, p=None, cells=0)
    pred = np.arange(len(PRED_ORDER)); rng = np.random.default_rng(seed)

    def stat(M):
        out = []
        for row in M:
            k = np.isfinite(row)
            if k.sum() >= 3:
                out.append(kendalltau(pred[k], row[k])[0])
        return float(np.nanmean(out))
    obs = stat(T); null = []
    for _ in range(n_perm):
        null.append(stat(np.array([rng.permutation(r) for r in T])))
    return dict(tau=obs, p=float((1 + np.sum(np.array(null) >= obs)) / (n_perm + 1)), cells=int(len(T)))


def checkpoint_report(base_dir, family, revs=None):
    revs = revs or revisions(family); steps = []; tabs = []
    for r in revs:
        d = os.path.join(base_dir, r.replace("/", "_"))
        if not os.path.exists(os.path.join(d, "revision.done")):
            continue
        t = summary_metrics(d); t["revision"] = r
        t["step"] = int(r.split("step")[1].split("-")[0]); tabs.append(t)
    if not tabs:
        return None
    T = pd.concat(tabs); em = []
    for (sl, s), g in T.groupby(["slot", "set"]):
        g = g.sort_values("step"); row = dict(slot=sl, set=s)
        for m in PRED_ORDER:
            row[m] = emergence(g.step.values, g[m].values)
        em.append(row)
    E = pd.DataFrame(em)
    return dict(table=T, emergence=E, test=order_test(E))


def datadecide_report(base_dir, B=2000, seed=0):
    alpha = pd.read_csv(C.ALPHA).set_index("recipe")["alpha_ols"]; tabs = []
    for r in alpha.index:
        d = os.path.join(base_dir, f"dd-{r}")
        if os.path.exists(os.path.join(d, "revision.done")):
            t = summary_metrics(d); t["recipe"] = r; t["alpha"] = alpha[r]; tabs.append(t)
    if not tabs:
        return None
    T = pd.concat(tabs); rng = np.random.default_rng(seed); out = []
    for (sl, s), g in T.groupby(["slot", "set"]):
        for m in PRED_ORDER + ["ordinality_decay", "units_template_simplex"]:
            y = g[m].to_numpy(float); x = g.alpha.to_numpy(float); k = np.isfinite(y)
            if k.sum() < 5:
                continue
            x, y = x[k], y[k]; b = np.polyfit(x, y, 1)[0]; bs = []
            for _ in range(B):
                i = rng.integers(0, len(x), len(x))
                if np.ptp(x[i]) > 0:
                    bs.append(np.polyfit(x[i], y[i], 1)[0])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out.append(dict(slot=sl, set=s, metric=m, slope=float(b), lo=float(lo), hi=float(hi), recipes=int(len(x)),
                            excludes_zero=bool(lo > 0 or hi < 0)))
    return dict(table=T, slopes=pd.DataFrame(out))
