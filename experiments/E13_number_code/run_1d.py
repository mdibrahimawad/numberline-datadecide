#!/usr/bin/env python
"""E13-1D overnight runner (PREREG_1D.md): is there a 1-D manifold that encodes the number line?

  Stage 1  shape tests T1-T4 for every (source, model, reading, set); new prompts, old prompt (9 models), checkpoints
  Stage 2  T5 (same curve across contexts, new prompts) and T10 (forms during training, checkpoints)
  Stage 3  causal tests T7-T9 on Pythia-1.4B and Qwen2.5-1.5B (Apple MPS / CUDA / CPU)
  report   tables + verdicts (T6 across models)

Resumable: one JSON per job in <out>/jobs; rerun the same command after any stop.

  python experiments/E13_number_code/run_1d.py --new-data ~/e13_out --old-data ~/topology_number_line/data --out ~/e13_1d
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (REPO, os.path.join(HERE, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = os.pathsep.join([REPO, os.path.join(HERE, "src"), os.environ.get("PYTHONPATH", "")])
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import argparse
import glob
import json
import logging
import time
import traceback

import numpy as np

SET_NAMES = ["S2", "S3", "S4"]


# --------------------------------------------------------------------------- data sources
def token_ok(model, n):
    from e13 import config as C
    fam = C.MODELS.get(model, {}).get("family", "chunk")
    if fam != "chunk":
        return np.ones(len(n), bool)
    tc = np.array(json.load(open(os.path.join(HERE, "token_counts_in_prompt.json")))[model])
    return tc[np.asarray(n) - 1] == 1


def in_set(n, s):
    from e13.manifold1d import SETS
    lo, hi = SETS[s]; return (n >= lo) & (n <= hi)


def shape_jobs(a):
    """(job_id, kind, args) for stage 1."""
    from e13 import config as C
    jobs = []
    if a.new_data:
        for m in ("pythia-1.4b", "qwen2.5-1.5b"):
            if os.path.exists(os.path.join(a.new_data, m, "config.json")):
                for r in ("sentences", "copy_number"):
                    for s in SET_NAMES:
                        jobs.append((f"new__{m}__{r}__{s}", "new", (m, r, s)))
    if a.old_data:
        for m in C.MODELS:
            if glob.glob(os.path.join(a.old_data, f"{m}_vectors", "layer_*.npz")):
                for s in SET_NAMES:
                    jobs.append((f"old__{m}__copy_eq__{s}", "old", (m, s)))
    cost = {"S4": 3, "S3": 2, "S2": 1}
    jobs.sort(key=lambda j: -cost[j[2][-1]])
    return jobs


def load_new(a, m, r, s):
    from e13.data import ModelData
    from e13.manifold1d import subsample
    md = ModelData(os.path.join(a.new_data, m))
    if r == "sentences":
        n = subsample(md.set_numbers(s, ["AVG5"]), s); rows = np.searchsorted(md.numbers, n)
        arr = np.load(os.path.join(a.new_data, m, "vec_avg_num_alllayers.npy"), mmap_mode="r")
        layers = [(l, (lambda l=l: arr[rows, l].astype(np.float64))) for l in range(arr.shape[1])]
    else:
        n = subsample(md.set_numbers(s, ["C1"]), s)
        layers = [(md.lidx[sl], (lambda sl=sl: md.X(("C1", "num"), sl, n))) for sl in range(9)]
    return n, layers


def load_old(a, m, s):
    from e13.manifold1d import subsample
    files = sorted(glob.glob(os.path.join(a.old_data, f"{m}_vectors", "layer_*.npz")), key=lambda f: int(os.path.basename(f)[6:-4]))
    n_all = np.arange(1, 10000); n = n_all[in_set(n_all, s) & token_ok(m, n_all)]; n = subsample(n, s)
    layers = [(int(os.path.basename(f)[6:-4]), (lambda f=f: np.load(f)["X"][n - 1].astype(np.float64))) for f in files]
    return n, layers


def run_shape(args):
    jid, kind, params, a_dict, out = args
    import argparse as _ap
    from e13 import manifold1d as M
    a = _ap.Namespace(**a_dict); path = os.path.join(out, "jobs", jid + ".json")
    if os.path.exists(path):
        return jid, "cached"
    t0 = time.time()
    try:
        if kind == "new":
            m, r, s = params; n, layers = load_new(a, m, r, s)
        else:
            m, s = params; r = "copy_eq"; n, layers = load_old(a, m, s)
        if len(n) < 50:
            res = dict(skipped=f"{len(n)} numbers")
        else:
            res = M.run_job(jid, [(lab, f) for lab, f in layers], n, s, jid)
        res.update(source=kind, model=m, reading=r, seconds=time.time() - t0)
    except Exception as e:
        res = dict(error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
    json.dump(M.clean(res), open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    return jid, "error" if "error" in res else "ok"


# --------------------------------------------------------------------------- stage 2: T5 and T10
def stage_t5(a, log):
    from e13 import manifold1d as M
    from e13.data import ModelData
    for m in ("pythia-1.4b", "qwen2.5-1.5b"):
        d = os.path.join(a.new_data, m) if a.new_data else None
        if not d or not os.path.exists(os.path.join(d, "config.json")):
            continue
        md = ModelData(d); ctxs = [c for c in ["C1"] + [f"C{k}" for k in range(2, 9)] if os.path.exists(os.path.join(d, f"vec_{c}_num.npy"))]
        for s in SET_NAMES:
            path = os.path.join(a.out, "jobs", f"t5__{m}__{s}.json")
            if os.path.exists(path):
                continue
            src = os.path.join(a.out, "jobs", f"new__{m}__sentences__{s}.json")
            if not os.path.exists(src):
                continue
            J = json.load(open(src)); chosen = J.get("chosen_layer")
            if chosen is None:
                continue
            slot = int(np.argmin([abs(l - chosen) if l > 0 else 1e9 for l in md.lidx]))   # never layer 0 (token embedding only)
            n = M.subsample(md.set_numbers(s, ctxs), s)
            n = n[~M.dev_split(n)]
            try:
                res = M.t5(lambda c: md.X((c, "num"), slot, n), ctxs, n, s)
                res.update(model=m, set=s, layer=md.lidx[slot], contexts=ctxs)
            except Exception as e:
                res = dict(error=f"{type(e).__name__}: {e}")
            json.dump(M.clean(res), open(path, "w")); log(f"  T5 {m} {s}: share >= 0.90: {res.get('share_ge_090')}")


def stage_t10(a, log):
    from e13 import manifold1d as M
    root = os.path.join(a.old_data, "e11_ckpt") if a.old_data else None
    if not root or not os.path.isdir(root):
        log("  T10: no checkpoint data found"); return
    for fam in sorted(os.listdir(root)):
        path = os.path.join(a.out, "jobs", f"t10__{fam}.json")
        if os.path.exists(path) or not os.path.isdir(os.path.join(root, fam)):
            continue
        beh = {b["revision"]: b for b in json.load(open(os.path.join(root, fam, "behaviour_log.json")))}
        revs = sorted([r for r in os.listdir(os.path.join(root, fam)) if os.path.isdir(os.path.join(root, fam, r))],
                      key=lambda r: int(r.split("step")[1].split("-")[0]))
        rows = []
        for rev in revs:
            files = sorted(glob.glob(os.path.join(root, fam, rev, "layer_*.npz")), key=lambda f: int(os.path.basename(f)[6:-4]))
            for f in files:
                z = np.load(f); n_all = z["target"].astype(int); X = z["X"].astype(np.float64)
                for s in ("S2", "S4"):
                    k = in_set(n_all, s) & token_ok(fam, n_all); n = n_all[k]
                    if len(n) < 50:
                        continue
                    dv = M.dev_split(n); r_dev = M.t1_t2(X[k][dv], n[dv], s, reps=3, shuffled=False)["rho"] if dv.sum() >= 15 else None
                    r_ev = M.t1_t2(X[k][~dv], n[~dv], s, reps=3, shuffled=False)["rho"]
                    rows.append(dict(revision=rev, step=int(rev.split("step")[1].split("-")[0]), layer=int(os.path.basename(f)[6:-4]),
                                     set=s, dev_rho=r_dev, rho=r_ev, copy_acc=beh.get(rev, {}).get("copy_acc")))
            log(f"  T10 {fam} {rev} done")
        res = dict(family=fam, rows=rows, summary=t10_summary(rows))
        json.dump(M.clean(res), open(path, "w"))


def t10_summary(rows):
    import pandas as pd
    T = pd.DataFrame(rows); out = {}
    if not len(T):
        return out
    beh = T.groupby("step").copy_acc.first().sort_index(); steps = list(beh.index)
    b_step = next((st for st in steps if (beh[st] or 0) >= 0.5), None)
    for s, g in T.groupby("set"):
        last = g[g.step == g.step.max()]
        if last.dev_rho.notna().any():
            layer = int(last.loc[last.dev_rho.idxmax(), "layer"])
        else:
            layer = int(last.loc[last.rho.idxmax(), "layer"])
        traj = g[g.layer == layer].sort_values("step")
        e_step = next((int(r.step) for r in traj.itertuples() if r.rho >= 0.90), None)
        adj = None
        if e_step is not None and b_step is not None:
            adj = abs(steps.index(e_step) - steps.index(b_step)) <= 1
        out[s] = dict(layer=layer, emergence_step=e_step, behaviour_step=b_step, same_or_adjacent=adj,
                      trajectory=[(int(r.step), float(r.rho)) for r in traj.itertuples()])
    return out


# --------------------------------------------------------------------------- stage 3: causal
def stage_causal(a, log):
    import torch
    from e13 import causal1d, config as C
    from e13 import manifold1d as M
    from e13.data import ModelData
    from e13.extract import load_model
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    for m in ("pythia-1.4b", "qwen2.5-1.5b"):
        path = os.path.join(a.out, "jobs", f"causal__{m}.json")
        if os.path.exists(path) or not a.new_data or not os.path.exists(os.path.join(a.new_data, m, "config.json")):
            continue
        log(f"== causal {m} on {dev}")
        model, tok_model, tok_fast, _ = load_model(m, "float32", device=dev, log=log)
        md = ModelData(os.path.join(a.new_data, m)); arr = np.load(os.path.join(a.new_data, m, "vec_avg_num_alllayers.npy"), mmap_mode="r")
        res = dict(model=m, device=dev, sets={})
        for s in ("S2", "S3"):
            src = os.path.join(a.out, "jobs", f"new__{m}__sentences__{s}.json")
            if not os.path.exists(src):
                continue
            layer = json.load(open(src)).get("chosen_layer")
            if layer is None:
                continue
            n = md.set_numbers(s, ["AVG5"]); X = arr[np.searchsorted(md.numbers, n), int(layer)].astype(np.float64)
            try:
                res["sets"][s] = causal1d.run(model, tok_fast, C.MODELS[m]["family"], X, n, s, layer, n_items=a.causal_items, log=log)
            except Exception as e:
                res["sets"][s] = dict(error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()); log(f"  causal {m} {s} FAILED: {e}")
        json.dump(M.clean(res), open(path, "w"))
        del model
        if dev == "mps":
            torch.mps.empty_cache()


# --------------------------------------------------------------------------- report
def report(a, log):
    import pandas as pd
    from e13 import config as C
    rows = []
    for f in sorted(glob.glob(os.path.join(a.out, "jobs", "*.json"))):
        b = os.path.basename(f)
        if not b.startswith(("new__", "old__")):
            continue
        J = json.load(open(f))
        if "verdict" not in J:
            rows.append(dict(job=b[:-5], status=J.get("error") or J.get("skipped"))); continue
        L = J["layers"][str(J["chosen_layer"])]; v = J["verdict"]; p = J.get("power", {})
        rows.append(dict(job=b[:-5], source=J["source"], model=J["model"], family=C.MODELS.get(J["model"], {}).get("family"),
                         reading=J["reading"], set=J["set"], N=J["N"], chosen_layer=J["chosen_layer"], rho=L["rho"],
                         rho_shuffled=L.get("rho_shuffled"), R2_curve=L["R2_curve"], R2_knn=L["R2_knn"], R2_ridge=L["R2_ridge"],
                         tear_ratio=J["T3_full"].get("ratio"), tear_ratio_hi=J["T3_full"].get("ratio_hi"),
                         T4_blocks_passed=L["T4"]["n_pass"], T1=v["T1"], T2=v["T2"], T3=v["T3"], T4=v["T4"],
                         power_T1=p.get("T1"), power_T2=p.get("T2"), power_T4=p.get("T4"), power_T3_detects_tear=p.get("T3_detects_tear"),
                         manifold_1d=bool(v["T1"] is True and v["T2"] is True and v["T4"] is True)))
    T = pd.DataFrame(rows); os.makedirs(os.path.join(a.out, "report"), exist_ok=True)
    T.to_csv(os.path.join(a.out, "report", "shape_tests.csv"), index=False)
    lines = ["# E13-1D results (PREREG_1D.md)", ""]
    if len(T) and "manifold_1d" in T:
        V = T[T.manifold_1d.notna()]
        g = V.groupby(["source", "model", "reading"]).manifold_1d.agg(["sum", "count"]).reset_index()
        g["supported"] = g["sum"] >= 2
        lines += ["## Per model and reading: 1-D manifold (T1, T2, T4 pass with power) in >= 2 of 3 sets", g.to_string(index=False), ""]
        for src in ("old", "new"):
            S = V[V.source == src]
            if len(S):
                fam = S.groupby("family").manifold_1d.mean()
                ok = all(fam.get(f, 0) >= 0.70 for f in ("digit", "chunk"))
                lines += [f"## T6 across models ({src} prompt): share of (model, set) cells passing by family: {fam.round(2).to_dict()} -> {'PASS' if ok else 'FAIL'}", ""]
        lines += ["## All jobs", T.round(3).to_string(index=False), ""]
    for f in sorted(glob.glob(os.path.join(a.out, "jobs", "t5__*.json"))):
        J = json.load(open(f)); lines.append(f"- T5 {J.get('model')} {J.get('set')}: share of context pairs with order >= 0.90 = {J.get('share_ge_090')} -> {'PASS' if J.get('T5_pass') else 'FAIL'}")
    for f in sorted(glob.glob(os.path.join(a.out, "jobs", "t10__*.json"))):
        J = json.load(open(f))
        for s, v in J.get("summary", {}).items():
            lines.append(f"- T10 {J['family']} {s}: curve emerges at step {v['emergence_step']}, copy behaviour at step {v['behaviour_step']}, same or adjacent checkpoint: {v['same_or_adjacent']}")
    for f in sorted(glob.glob(os.path.join(a.out, "jobs", "causal__*.json"))):
        J = json.load(open(f))
        for s, v in J.get("sets", {}).items():
            if "error" in v:
                lines.append(f"- causal {J['model']} {s}: ERROR {v['error']}"); continue
            lines.append(f"- causal {J['model']} {s} (layer {v['layer']}): T7 necessary {'PASS' if v['T7']['passed'] else 'FAIL'} "
                         f"(B1 drop {v['T7']['B1_drop']:.2f} vs random {v['T7']['B1_random']:.2f}; B2 drop {v['T7']['B2_drop']:.2f} vs {v['T7']['B2_random']:.2f}); "
                         f"T8 sufficient {'PASS' if v.get('T8', {}).get('passed') else 'FAIL'} (flips {v.get('T8', {}).get('flip_curve')} vs random {v.get('T8', {}).get('flip_random')}); "
                         f"T9 behaviour {'PASS' if v['T9']['passed'] else 'FAIL'} (partial r {v['T9']['partial_spearman']:.2f})")
    open(os.path.join(a.out, "report", "RESULTS.md"), "w").write("\n".join(lines))
    log("\n".join(lines[:40]))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-data", default="", help="folder with pythia-1.4b/ and qwen2.5-1.5b/ (E13 extraction)")
    ap.add_argument("--old-data", default="", help="topology_number_line/data (old E10 vectors, e11_ckpt/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--stages", default="1,2,3,report")
    ap.add_argument("--causal-items", type=int, default=200)
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--only", default="", help="testing: run only jobs whose id contains this text")
    a = ap.parse_args()
    for k in ("new_data", "old_data", "out"):
        if getattr(a, k):
            setattr(a, k, os.path.abspath(os.path.expanduser(getattr(a, k))))
    os.makedirs(os.path.join(a.out, "jobs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(a.out, "run_1d.log")), logging.StreamHandler(sys.stdout)])
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("e13-1d").info; st = a.stages.split(","); t0 = time.time()
    log(f"E13-1D: stages {st}, new data {a.new_data or '-'}, old data {a.old_data or '-'}, workers {a.workers}")
    if "1" in st:
        jobs = [j for j in shape_jobs(a) if (not a.only or a.only in j[0]) and not os.path.exists(os.path.join(a.out, "jobs", j[0] + ".json"))]
        log(f"Stage 1: {len(jobs)} shape-test jobs")
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(run_shape, (jid, kind, params, vars(a), a.out)) for jid, kind, params in jobs]
            for k, f in enumerate(as_completed(futs)):
                try:
                    jid, stt = f.result()
                except Exception as e:
                    jid, stt = "?", f"crashed: {e}"
                log(f"  [{k + 1}/{len(futs)}] {jid}: {stt} ({(time.time() - t0) / 60:.0f} min)")
    if "2" in st:
        log("Stage 2: T5 contexts, T10 checkpoints"); stage_t5(a, log); stage_t10(a, log)
    if "3" in st:
        log("Stage 3: causal T7-T9"); stage_causal(a, log)
    if "report" in st:
        report(a, log)
    log(f"E13-1D finished after {(time.time() - t0) / 3600:.2f} h. Results: {os.path.join(a.out, 'report', 'RESULTS.md')}")


if __name__ == "__main__":
    main()
