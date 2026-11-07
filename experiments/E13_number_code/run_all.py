#!/usr/bin/env python
"""E13 master script: the WHOLE experiment in order, resumable.

  Stage A1  Pythia-1.4B, Qwen2.5-1.5B: correctness checks (2.5), extraction, upload
  Stage A2  analysis of both (panel, H0, H3, P7, P8, power) -> report -> Gate A
  Stage B   the other 7 models: checks, extraction, upload, analysis
  Stage C   causal stage (Pythia, Qwen, OLMo-2) - only if Gate A passes (or --force-causal)
  Stage D   checkpoints (Pythia 16, OLMo-2 12) and the 25 DataDecide recipes
  report    tables, verdicts, figures; upload

Every finished step writes <out>/_state/<step>.done, so a restart skips it; extraction and analysis are also
resumable inside a step. Analysis runs in background worker processes while the GPU extracts the next model.
Run inside tmux (see RUNPOD.md). The Hugging Face token is read from $HF_TOKEN and is never printed.

  python experiments/E13_number_code/run_all.py --out /workspace/e13_out --hf-repo <user>/e13-number-code \
      --time-limit-hours 10 --stop-pod
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (REPO, os.path.join(HERE, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = os.pathsep.join([REPO, os.path.join(HERE, "src")] + [os.environ.get("PYTHONPATH", "")])

import argparse
import json
import logging
import shutil
import subprocess
import time


def parse_numbers(s):
    import numpy as np
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-"); out += list(range(int(a), int(b or a) + 1))
    return np.array(sorted(set(out)))


class Runner:
    def __init__(self, a):
        self.a = a; self.t0 = time.time(); self.deadline = self.t0 + a.time_limit_hours * 3600
        os.makedirs(a.out, exist_ok=True); self.state = os.path.join(a.out, "_state"); os.makedirs(self.state, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                            handlers=[logging.FileHandler(os.path.join(a.out, "run_all.log")), logging.StreamHandler(sys.stdout)])
        self.log = logging.getLogger("e13").info
        for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        self.ex = None; self.pending = {}; self.timing = {}
        from e13 import config as C
        self.C = C; self.numbers = parse_numbers(a.numbers)

    # ------------------------------------------------------------------ helpers
    def done(self, step):
        return os.path.exists(os.path.join(self.state, step + ".done"))

    def mark(self, step, info=None):
        json.dump(dict(info or {}, at=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_h=(time.time() - self.t0) / 3600),
                  open(os.path.join(self.state, step + ".done"), "w"), default=str)

    def time_left(self):
        return self.deadline - time.time()

    def check_time(self, where):
        if self.time_left() <= 0:
            self.log(f"TIME LIMIT ({self.a.time_limit_hours} h) reached at {where}. Everything finished so far is saved; "
                     f"rerun the same command to resume.")
            self.finish(status="time_limit"); sys.exit(3)

    def executor(self):
        if self.ex is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[v] = "1"
            self.ex = ProcessPoolExecutor(max_workers=self.a.workers, mp_context=mp.get_context("spawn"))
        return self.ex

    def model_dir(self, key):
        return os.path.join(self.a.out, key)

    # ------------------------------------------------------------------ one model: checks + extraction + upload
    def extract(self, key):
        step = f"extract_{key}"
        if self.done(step):
            return
        self.check_time(step)
        from e13 import checks, extract
        d = self.model_dir(key); os.makedirs(d, exist_ok=True); t = time.time()
        self.log(f"== {key}: loading model")
        mid = self.a.llama_id if key == "llama3.2-1b" and self.a.llama_id else None
        bundle = extract.load_model(key, self.a.dtype, model_id=mid, log=self.log)
        if not self.done(f"checks_{key}"):
            res = checks.run_checks(key, *bundle[:3], d, numbers=self.numbers, force=self.a.force_checks, log=self.log)
            self.mark(f"checks_{key}", res)
        r = extract.extract_model(key, d, numbers=self.numbers, dtype=self.a.dtype, batch=self.a.batch, chunk=self.a.chunk,
                                  deadline=self.deadline, log=self.log, model_bundle=bundle)
        del bundle
        self._free()
        if not r.get("done"):
            self.check_time(step); raise RuntimeError(f"extraction of {key} did not finish")
        dt = time.time() - t; self.timing[key] = dt; self.mark(step, dict(seconds=dt, prompts=r.get("prompts")))
        self.log(f"== {key}: extraction done in {dt / 60:.1f} min ({r.get('prompts')} prompts)")

    def upload(self, key, path=None, ignore=None):
        step = f"upload_{key}"
        if self.done(step) or self.a.no_upload:
            return
        from e13.upload import upload_folder
        try:
            r = upload_folder(path or self.model_dir(key), self.a.hf_repo, key, log=self.log, ignore=ignore)
            if r.get("uploaded"):
                self.mark(step, r)
        except Exception as e:
            self.log(f"  upload of {key} FAILED ({type(e).__name__}: {e}); continuing, will retry on the next run")

    def analyse_async(self, key):
        if self.done(f"analysis_{key}") or key in self.pending:
            return
        from e13 import analyse
        jobs = [j for j in analyse.jobs_for(self.model_dir(key)) if not analyse.finished(j)]
        self.pending[key] = (time.time(), [self.executor().submit(analyse.run_job, j) for j in jobs])
        self.log(f"== {key}: {len(jobs)} analysis jobs submitted to {self.a.workers} workers")

    def wait_analysis(self, keys):
        for key in keys:
            if key not in self.pending:
                continue
            t, futs = self.pending.pop(key); n_err = 0
            for k, f in enumerate(futs):
                while True:
                    try:
                        _, st, _ = f.result(timeout=60); n_err += st == "error"; break
                    except TimeoutError:
                        self.check_time(f"analysis of {key}")
                    except Exception as e:
                        n_err += 1; self.log(f"  analysis job crashed: {e}"); break
                if (k + 1) % 200 == 0:
                    self.log(f"  {key}: {k + 1}/{len(futs)} analysis jobs done")
            dt = time.time() - t; self.timing[f"analysis_{key}"] = dt
            if n_err == 0:                       # with errors the step stays open, so the next run retries them
                self.mark(f"analysis_{key}", dict(seconds=dt, jobs=len(futs), errors=0))
            self.log(f"== {key}: analysis done in {dt / 60:.1f} min ({len(futs)} jobs, {n_err} errors)")

    def projection(self, key):
        ext = self.timing.get(key); ana = self.timing.get(f"analysis_{key}")
        n_models = len(self.C.MODELS); n_rev = 16 + 12 + 25
        msg = f"PROJECTION after {key}: extraction {ext / 60:.1f} min/model"
        tot = ext * (n_models - 1)
        if ana:
            msg += f", analysis {ana / 60:.1f} min/model (overlaps the next extraction)"; tot = max(tot, ana * (n_models - 1))
        tot += n_rev * (ext * 0.6 + (ana or ext) * 0.15)
        msg += f"; whole run ~{tot / 3600 + (time.time() - self.t0) / 3600:.1f} h in total (rough; time limit {self.a.time_limit_hours} h)"
        self.log(msg)

    def _free(self):
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def purge_vectors(self, d):
        for f in os.listdir(d):
            if f.startswith("vec_") and f.endswith(".npy"):
                os.remove(os.path.join(d, f))

    # ------------------------------------------------------------------ stages
    def stage_models(self, keys, stage):
        for i, key in enumerate(keys):
            self.extract(key); self.upload(key, ignore=["cells/*"]); self.analyse_async(key)
            if i == 0 and stage == "A" and key in self.timing:
                self.projection(key)
        self.wait_analysis(keys)
        if stage == "A" and keys and keys[0] in self.timing and f"analysis_{keys[0]}" in self.timing:
            self.projection(keys[0])
        for key in keys:
            self.upload(f"{key}_cells", path=os.path.join(self.model_dir(key), "cells"))
            if self.a.purge_main:
                self.purge_vectors(self.model_dir(key))

    def gate_a(self):
        from e13 import report
        try:
            res = report.build(self.a.out)
        except Exception as e:
            import traceback
            self.log(f"GATE A: report failed ({type(e).__name__}: {e}); causal stage not allowed\n{traceback.format_exc()}")
            res = {}
        keys = [m for m in self.C.STAGE_A if m in self.models]
        V = {(v["test"], v["condition"]): v["verdict"] for v in res.get("verdicts", [])}
        p7ok = all(res.get("p7_control_works", {}).get(m) for m in keys)
        h12 = any(V.get((t, c)) in ("supported", "one family only") for t in ("H1 decay", "H2a carry spikes", "H2b successor breaks",
                                                                               "H2c tree beats line (late)") for c in ("AVG5_num", "C1_num"))
        gate = dict(positive_control_works=p7ok, h1_or_h2_on_number_reading=h12, causal_allowed=bool(h12),
                    note="Methodology Part 4: the causal stage runs only if H1 or H2 holds on the number-token reading.")
        json.dump(gate, open(os.path.join(self.a.out, "gate_A.json"), "w"), indent=2)
        self.log(f"GATE A: {gate}")
        return gate

    def stage_c(self, gate):
        if not (gate.get("causal_allowed") or self.a.force_causal):
            self.log("Stage C skipped: Gate A failed (H1/H2 do not hold on the number reading). Use --force-causal to run anyway.")
            return
        from e13 import causal, extract
        for key in [m for m in self.C.CAUSAL_MODELS if m in self.models]:
            step = f"causal_{key}"
            if self.done(step):
                continue
            self.check_time(step)
            if not os.path.exists(os.path.join(self.model_dir(key), "vec_avg_num_alllayers.npy")):
                self.log(f"  causal {key}: no extracted vectors, skipping"); continue
            model, tok_model, tok_fast, _ = extract.load_model(key, self.a.dtype, log=self.log)
            out = os.path.join(self.model_dir(key), "causal.json")
            causal.run_model(key, model, tok_fast, self.model_dir(key), out, n_items=self.a.causal_items, log=self.log)
            del model; self._free(); self.mark(step)
            self.upload(f"{key}_causal", path=self.model_dir(key), ignore=["vec_*", "cells/*", "static_*", "p7_*"])

    def stage_d(self):
        from e13 import training
        fams = [("pythia-1.4b", self.C.MODELS["pythia-1.4b"]["hf"], None), ("olmo2-1b", self.C.MODELS["olmo2-1b"]["hf"], None)]
        if self.a.ckpt_override:
            mid, revs = self.a.ckpt_override.split(":"); fams = [("pythia-1.4b", mid, revs.split(","))]
        for fam, mid, revs in fams:
            revs = revs or training.revisions(fam)
            for rev in revs:
                step = f"ckpt_{fam}_{rev}"
                if self.done(step):
                    continue
                self.check_time(step)
                d = os.path.join(self.a.out, f"ckpt_{fam}", rev.replace("/", "_"))
                r = training.run_revision(fam, mid, rev, d, self.numbers, self.a.dtype, self.a.batch, self.deadline, self.log,
                                          self.a.workers, executor=self.executor())
                if r.get("done"):
                    self.upload(f"ckpt_{fam}_{rev}", path=d); self.purge_vectors(d); self.mark(step)
            rep = training.checkpoint_report(os.path.join(self.a.out, f"ckpt_{fam}"), fam) if not self.a.ckpt_override else None
            if rep:
                rd = os.path.join(self.a.out, "report"); os.makedirs(rd, exist_ok=True)
                rep["table"].to_csv(os.path.join(rd, f"checkpoints_{fam}.csv"), index=False)
                rep["emergence"].to_csv(os.path.join(rd, f"emergence_{fam}.csv"), index=False)
                json.dump(rep["test"], open(os.path.join(rd, f"emergence_test_{fam}.json"), "w"))
        if self.a.skip_datadecide or self.a.ckpt_override:
            return
        training.register_datadecide()
        for r in training.datadecide_recipes():
            key = f"dd-{r}"; step = f"datadecide_{r}"
            if self.done(step):
                continue
            self.check_time(step)
            d = os.path.join(self.a.out, key)
            try:
                res = training.run_revision(key, self.C.MODELS[key]["hf"], None, d, self.numbers, self.a.dtype, self.a.batch,
                                            self.deadline, self.log, self.a.workers, executor=self.executor())
            except Exception as e:
                self.log(f"  DataDecide {r} FAILED: {type(e).__name__}: {e}"); continue
            if res.get("done"):
                self.upload(key, path=d); self.purge_vectors(d); self.mark(step)
        rep = training.datadecide_report(self.a.out)
        if rep:
            rd = os.path.join(self.a.out, "report"); os.makedirs(rd, exist_ok=True)
            rep["table"].to_csv(os.path.join(rd, "datadecide_metrics.csv"), index=False)
            rep["slopes"].to_csv(os.path.join(rd, "datadecide_alpha_slopes.csv"), index=False)

    def finish(self, status="done"):
        try:
            from e13 import report
            report.build(self.a.out)
            self.upload("report", path=os.path.join(self.a.out, "report"))
        except Exception as e:
            self.log(f"report failed: {type(e).__name__}: {e}")
        if self.ex is not None:
            self.ex.shutdown(wait=False, cancel_futures=True)
        hours = (time.time() - self.t0) / 3600
        self.log(f"E13 run_all finished with status '{status}' after {hours:.2f} h")
        if self.a.stop_pod:
            pod = os.environ.get("RUNPOD_POD_ID")
            if pod and shutil.which("runpodctl"):
                self.log(f"stopping pod {pod}"); subprocess.run(["runpodctl", "stop", "pod", pod])
            else:
                self.log("--stop-pod given but RUNPOD_POD_ID / runpodctl not found; not stopping")

    def run(self):
        a = self.a; C = self.C; stages = a.stages.split(",")
        self.models = a.models.split(",") if a.models else list(C.MODELS)
        json.dump({k: v for k, v in vars(a).items()}, open(os.path.join(a.out, "run_config.json"), "w"), indent=2)
        self.log(f"E13 run_all: stages {stages}, models {self.models}, numbers {self.numbers.min()}-{self.numbers.max()} "
                 f"({len(self.numbers)}), workers {a.workers}, time limit {a.time_limit_hours} h")
        A = [m for m in C.STAGE_A if m in self.models]; B = [m for m in C.STAGE_B if m in self.models]
        if "A1" in stages or "A2" in stages:
            self.stage_models(A, "A")
        gate = self.gate_a() if ("A2" in stages and A) else json.load(open(os.path.join(a.out, "gate_A.json"))) \
            if os.path.exists(os.path.join(a.out, "gate_A.json")) else {}
        if "B" in stages:
            self.stage_models(B, "B")
        if "C" in stages:
            self.stage_c(gate)
        if "D" in stages:
            self.stage_d()
        self.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/workspace/e13_out")
    ap.add_argument("--hf-repo", default=os.environ.get("E13_HF_REPO", ""), help="PRIVATE HF dataset repo, e.g. user/e13-number-code")
    ap.add_argument("--stages", default="A1,A2,B,C,D")
    ap.add_argument("--models", default="", help="comma list (default: all 9)")
    ap.add_argument("--numbers", default="1-9999")
    ap.add_argument("--time-limit-hours", type=float, default=10.0)
    ap.add_argument("--stop-pod", action="store_true", help="stop the RunPod pod at the end or at the time limit")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--causal-items", type=int, default=500)
    ap.add_argument("--force-causal", action="store_true")
    ap.add_argument("--force-checks", action="store_true", help="continue even if a 2.5 check fails (not recommended)")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--purge-main", action="store_true", help="delete vec_*.npy of the 9 main models after upload")
    ap.add_argument("--skip-datadecide", action="store_true")
    ap.add_argument("--llama-id", default="", help="e.g. meta-llama/Llama-3.2-1B (gated); default = the E10 mirror")
    ap.add_argument("--ckpt-override", default="", help="testing: <model_id>:<rev1,rev2> replaces the checkpoint list")
    a = ap.parse_args()
    r = Runner(a)
    try:
        r.run()
    except SystemExit:
        raise
    except BaseException as e:                       # crash: save what exists and (with --stop-pod) stop the pod
        import traceback
        r.log(f"RUN CRASHED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        r.finish(status=f"crashed ({type(e).__name__})")
        sys.exit(1)


if __name__ == "__main__":
    main()
