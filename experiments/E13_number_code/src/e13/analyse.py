"""Per-model analysis: enumerate cells, run them in a process pool, one JSON per job (resumable).

Jobs per model:
  panel   (condition, layer slot, set, tier)  -> cells/panel__<ctx>_<pos>__L<slot>__<set>.json
  h0      (layer slot, set)                    -> cells/h0__L<slot>__<set>.json
  h3      (layer slot, set)                    -> cells/h3__L<slot>__<set>.json
  units   (layer slot, set)  P7 detector on the units digit
  expo    (condition, layer slot)              exponent probe
  p7      (model)                              days / months positive control
FULL tier = C1/P_next (old reading), C1/P_num, AVG5/P_num; every other condition is LIGHT (DEVIATIONS.md D5)."""
from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np

from src import e13_stage1 as E1

from . import config as C
from .data import ModelData, subsample_s4

FULL = [("C1", "next"), ("C1", "num"), ("AVG5", "num")]


class SkipCell(Exception):
    pass
_MD = {}


def _md(d):
    if d not in _MD:
        _MD[d] = ModelData(d)
    return _MD[d]


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def jobs_for(model_dir, sets=None, conditions=None, light=True, quick=False):
    md = _md(model_dir); sets = sets or C.MAIN_SETS; out = []
    conds = conditions or [c for c in md.conditions() if c[0] not in C.VARIANTS]
    slots = range(len(md.lidx))
    for cond in conds:
        tier = "full" if cond in FULL and not quick else "light"
        if tier == "light" and not light:
            continue
        for sl in slots:
            for s in sets:
                out.append(("panel", model_dir, cond, sl, s, tier))
            out.append(("expo", model_dir, cond, sl, None, None))
    if md.family == "chunk":
        for s in sets:
            out.append(("panel", model_dir, ("STATIC_E", "space"), 0, s, "light"))
            out.append(("panel", model_dir, ("STATIC_U", "space"), 0, s, "light"))
    for sl in slots:
        for s in sets:
            out += [("h0", model_dir, None, sl, s, None), ("h3", model_dir, None, sl, s, None), ("units", model_dir, None, sl, s, None)]
    out.append(("p7", model_dir, None, None, None, None))
    cost = {"S4": 10, "S3": 3, "S2": 1, None: 1}
    out.sort(key=lambda j: -(cost[j[4]] * (5 if j[5] == "full" else 1)))
    return out


def job_path(job):
    kind, d, cond, sl, s, tier = job; cd = os.path.join(d, "cells"); os.makedirs(cd, exist_ok=True)
    if kind == "panel":
        return os.path.join(cd, f"panel__{cond[0]}_{cond[1]}__L{sl}__{s}.json")
    if kind == "expo":
        return os.path.join(cd, f"expo__{cond[0]}_{cond[1]}__L{sl}.json")
    if kind == "p7":
        return os.path.join(cd, "p7.json")
    return os.path.join(cd, f"{kind}__L{sl}__{s}.json")


def run_job(job):
    path = job_path(job)
    if os.path.exists(path):
        return path, "cached", 0.0
    t0 = time.time(); kind, d, cond, sl, s, tier = job; md = _md(d)
    try:
        if kind == "panel":
            from . import panel, power
            n = md.set_numbers(s, [cond[0]])
            if len(n) < 30:
                res = dict(skipped=f"{len(n)} numbers")
            else:
                X = md.X(cond, sl, n)
                if not np.isfinite(X).all():
                    good = np.isfinite(X).all(1); n, X = n[good], X[good]
                if X.std(0).max() < 1e-6:                   # e.g. layer 0 at a fixed token (P_next / P_task)
                    raise SkipCell("vectors identical for every number (fixed token at the embedding layer)")
                tag = f"{md.key}|{cond[0]}_{cond[1]}|L{sl}|{s}"
                res = panel.run_cell(X, n, s, tag, tier)
                res.update(condition=list(cond), layer_slot=sl, layer=md.lidx[sl] if cond[0] not in ("STATIC_E", "STATIC_U") else None,
                           set=s, token_counts=md.token_counts(s, cond[0]) if cond[0] not in ("AVG5", "STATIC_E", "STATIC_U") else None)
                if tier == "full":
                    ns = subsample_s4(n) if s == "S4" else n; Xs = X[np.searchsorted(n, ns)]
                    res["power_rows"] = power.rows_from_cell(res) + power.power_m1_m2_p8_h3(E1.prep(Xs), ns, s, tag, reps=10)
        elif kind == "expo":
            from . import panel
            sets_n = {s_: md.set_numbers(s_, [cond[0]]) for s_ in C.SETS}
            res = panel.exponent(lambda nn: md.X(cond, sl, nn), sets_n, f"{md.key}|{cond}|{sl}")
            res = dict(result=res, condition=list(cond), layer_slot=sl)
        elif kind == "h0":
            from . import h0
            res = h0.run(md, sl, s)
        elif kind == "h3":
            from . import h3
            res = h3.run(md, sl, s)
        elif kind == "units":
            from . import p7
            res = p7.numbers_units(md, sl, s)
        elif kind == "p7":
            from . import p7
            res = p7.run(d)
        res = dict(job=[kind, cond, sl, s, tier], seconds=time.time() - t0, result=res)
    except SkipCell as e:
        res = dict(job=[kind, cond, sl, s, tier], result=dict(skipped=str(e)))
    except Exception as e:                                   # report failures, never silently drop them
        res = dict(job=[kind, cond, sl, s, tier], error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
    json.dump(clean(res), open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    return path, ("error" if "error" in res else "ok"), time.time() - t0


def run_model(model_dir, workers=4, deadline=None, log=print, executor=None, **kw):
    """Run all jobs for one model (blocking). Returns counts."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [j for j in jobs_for(model_dir, **kw) if not os.path.exists(job_path(j))]
    log(f"  analysis {os.path.basename(model_dir)}: {len(jobs)} jobs to run")
    stats = dict(ok=0, error=0, cached=0); own = executor is None
    ex = executor or ProcessPoolExecutor(max_workers=workers)
    futs = [ex.submit(run_job, j) for j in jobs]; t0 = time.time()
    try:
        for k, f in enumerate(as_completed(futs)):
            _, st, _ = f.result(); stats[st] += 1
            if (k + 1) % 50 == 0:
                log(f"    {k + 1}/{len(jobs)} jobs ({time.time() - t0:.0f}s)")
            if deadline and time.time() > deadline:
                log("  time limit reached during analysis; cancelling pending jobs (resumable)")
                for g in futs:
                    g.cancel()
                break
    finally:
        if own:
            ex.shutdown(wait=False, cancel_futures=True)
    return stats
