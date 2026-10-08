"""Model beta (number-line geometry) for many DataDecide models on ONE GPU pod.

Two protocols:
  default   `python -m src.datadecide_sweep` (the 4-group paper protocol of
            modal_app/datadecide_app.py), outputs in results/datadecide_runpod/
  --fine    `python -m src.beta_fine` (10 groups, batched, per-prompt data saved),
            outputs in results/beta_fine/

--parallel models run at once on the GPU (a 1B model in bf16 needs ~3-4 GB, a
24 GB card fits 4-6), while a background thread downloads the next models'
weights so no GPU slot waits for a download. Finished models are skipped, so
rerunning resumes.

    python -m runpod_jobs.beta --fine --models c4 --parallel 1          # pilot: time one model
    python -m runpod_jobs.beta --fine --parallel 6 \\
        --upload-hf numberline-beta-results --delete-pod-when-done --max-hours 4

Unattended: --upload-hf puts each finished model in a private Hugging Face dataset
right away; --delete-pod-when-done deletes the pod once every model is uploaded
and verified; --max-hours deletes it at that limit whatever happens.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from runpod_jobs.pod import terminate_this_pod
from src.datadecide_sweep import load_model_config, repo_id_for, write_summary_csv


def _prefetch(jobs, config, ahead: int, started: dict) -> None:
    """Download model weights ahead of the GPU workers (HF cache), at most `ahead` beyond
    the models already started."""
    from huggingface_hub import snapshot_download

    for i, (m, rev, _) in enumerate(jobs):
        while i >= len(started) + ahead:
            time.sleep(2)
        try:
            snapshot_download(repo_id_for(m, config), revision=rev, token=os.environ.get("HF_TOKEN"))
            print(f"[prefetch] {m} @ {rev} downloaded", flush=True)
        except Exception as exc:  # noqa: BLE001 -- the worker will download it itself
            print(f"[prefetch] {m}: {exc}", flush=True)


def _hf(repo: str):
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo = repo if "/" in repo else f"{api.whoami()['name']}/{repo}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    return api, repo


def _upload(api, repo: str, files: list[Path], prefix: str) -> bool:
    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd(path_in_repo=f"{prefix}/{f.name}", path_or_fileobj=str(f)) for f in files if f.exists()]
    for attempt in range(6):
        try:
            api.create_commit(repo, ops, repo_type="dataset", commit_message=f"{prefix}: {files[0].stem}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[upload] attempt {attempt + 1} failed: {exc}", flush=True)
            time.sleep(min(120, 10 * 2 ** attempt))
    return False


def main(argv: list[str] | None = None) -> int:
    try:  # much faster model downloads when available (inherited by the worker processes)
        import hf_transfer  # noqa: F401

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass
    config = load_model_config()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", default="", help="comma list of recipe slugs; default all 25")
    p.add_argument("--revisions", default=config["revision"], help="comma list of HF revisions")
    p.add_argument("--fine", action="store_true", help="10-group batched protocol (src.beta_fine)")
    p.add_argument("--k", type=int, default=100, help="--fine: prompts per group per seed")
    p.add_argument("--seeds", default="45,46,47", help="--fine: prompt seeds")
    p.add_argument("--parallel", type=int, default=4, help="models at once on the GPU")
    p.add_argument("--prefetch", type=int, default=3, help="models downloaded ahead of the GPU")
    p.add_argument("--out-root", default="")
    p.add_argument("--upload-hf", default="", metavar="REPO")
    p.add_argument("--delete-pod-when-done", action="store_true")
    p.add_argument("--max-hours", type=float, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    out_root = Path(args.out_root or ("results/beta_fine" if args.fine else "results/datadecide_runpod"))

    models = [m.strip() for m in args.models.split(",") if m.strip()] or list(config["recipes"])
    revisions = [r.strip() for r in args.revisions.split(",") if r.strip()]
    jobs = [(m, rev, out_root / rev) for rev in revisions for m in models
            if not (out_root / rev / f"{m}.json").exists()]
    print(f"[beta] {'fine (10 groups)' if args.fine else 'paper (4 groups)'}: {len(jobs)} model runs to do "
          f"({len(models)} models x {len(revisions)} revisions, finished ones skipped), {args.parallel} at a time")
    if args.dry_run:
        for m, rev, _ in jobs:
            print(f"  {m} @ {rev}")
        return 0
    if args.delete_pod_when_done and not args.upload_hf:
        raise SystemExit("--delete-pod-when-done needs --upload-hf (results must be off the pod first)")
    hf = _hf(args.upload_hf) if args.upload_hf else None  # fails fast on a token without write access
    if hf:
        print(f"[beta] results go to the private dataset huggingface.co/datasets/{hf[1]}")
    if args.max_hours:
        def deadline():
            print(f"[limit] {args.max_hours} h reached: deleting the pod", flush=True)
            terminate_this_pod(f"--max-hours {args.max_hours} reached")
            os._exit(3)
        t = threading.Timer(args.max_hours * 3600, deadline)
        t.daemon = True
        t.start()

    ok = _run_all(args, jobs, revisions, config, out_root, hf)
    if hf and args.delete_pod_when_done:
        api, repo = hf
        have = set(api.list_repo_files(repo, repo_type="dataset"))
        missing = [m for m, rev, _ in jobs if f"{rev}/{m}.json" not in have]
        if not missing:
            terminate_this_pod("every model uploaded and verified")
        else:
            print(f"[end] NOT uploaded: {missing} -- the pod is kept (deleted at --max-hours)", flush=True)
            while args.max_hours:
                time.sleep(3600)
    return 0 if ok else 1


def _run_all(args, jobs, revisions, config, out_root: Path, hf) -> bool:
    running: list[tuple] = []
    failed: list[tuple] = []
    started: dict = {}
    start = time.time()
    threading.Thread(target=_prefetch, args=(jobs, config, args.prefetch, started), daemon=True).start()

    def launch(m, rev, out_dir):
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        log = open(out_dir / "logs" / f"{m}.log", "a")
        if args.fine:
            cmd = [sys.executable, "-m", "src.beta_fine", "--model", m, "--revision", rev, "--k", str(args.k),
                   "--seeds", args.seeds, "--device", "cuda", "--out-dir", str(out_dir)]
        else:
            cmd = [sys.executable, "-m", "src.datadecide_sweep", "--model", m, "--revision", rev,
                   "--device", "cuda", "--dtype", "bfloat16", "--output-dir", str(out_dir)]
        started[(m, rev)] = True
        running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), m, rev, out_dir,
                        time.time(), log))

    def reap(block: bool):
        while running:
            for item in list(running):
                proc, m, rev, out_dir, t0, log = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                log.close()
                if proc.returncode:
                    failed.append((m, rev, out_dir))
                    status = f"FAILED (see {log.name})"
                else:
                    status = "ok"
                    if hf:
                        files = [out_dir / f"{m}.json", out_dir / f"{m}.npz"]
                        status += ", uploaded" if _upload(*hf, files, rev) else ", UPLOAD FAILED"
                print(f"[beta] {m} @ {rev}: {status} in {(time.time() - t0) / 60:.1f} min "
                      f"(elapsed {(time.time() - start) / 60:.1f} min, {len(started)}/{len(jobs)} started)",
                      flush=True)
            if not block or len(running) < args.parallel:
                return
            time.sleep(2)

    for job in jobs:
        launch(*job)
        reap(block=True)
    while running:
        reap(block=False)
        time.sleep(2)
    if failed:  # one retry: transient download / CUDA errors
        print(f"[beta] retrying {len(failed)} failed runs once", flush=True)
        retry, failed[:] = list(failed), []
        for job in retry:
            launch(*job)
            reap(block=True)
        while running:
            reap(block=False)
            time.sleep(2)
    for rev in revisions:
        out_dir = out_root / rev
        if out_dir.exists():
            if args.fine:
                from src.beta_fine import write_summary

                summary = write_summary(out_dir, list(config["recipes"]))
            else:
                summary = write_summary_csv(out_dir, recipe_order=list(config["recipes"]))
            print(f"[beta] {summary}")
            if hf:
                _upload(*hf, [summary], rev)
    if failed:
        print(f"[beta] failed: {[(m, r) for m, r, _ in failed]} -- rerun the same command to retry them")
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
