"""Model beta (number-line geometry) for many DataDecide models on ONE GPU pod.

Runs `python -m src.datadecide_sweep` (the exact protocol of
modal_app/datadecide_app.py) for every (recipe, revision) pair, --parallel
models at a time on the same GPU (a 1B model in bf16 needs ~3-4 GB, so a
24 GB card fits 4-6). Finished models are skipped, so rerunning resumes.

    python -m runpod_jobs.beta --dry-run                       # what would run
    python -m runpod_jobs.beta --models c4 --parallel 1        # pilot: time one model
    python -m runpod_jobs.beta                                  # all 25 final checkpoints
    python -m runpod_jobs.beta --models falcon-and-cc-qc-10p,c4 \
        --revisions step10000-seed-default,step30000-seed-default,step69369-seed-default

Outputs: <out-root>/<revision>/<recipe>.json + summary.csv per revision,
logs in <out-root>/<revision>/logs/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.datadecide_sweep import load_model_config, write_summary_csv


def main(argv: list[str] | None = None) -> int:
    config = load_model_config()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", default="", help="comma list of recipe slugs; default all 25")
    p.add_argument("--revisions", default=config["revision"], help="comma list of HF revisions")
    p.add_argument("--parallel", type=int, default=4, help="models at once on the GPU")
    p.add_argument("--out-root", default="results/datadecide_runpod")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()] or list(config["recipes"])
    revisions = [r.strip() for r in args.revisions.split(",") if r.strip()]
    jobs = []
    for rev in revisions:
        for m in models:
            out_dir = Path(args.out_root) / rev
            if (out_dir / f"{m}.json").exists():
                continue
            jobs.append((m, rev, out_dir))
    print(f"[beta] {len(jobs)} model runs to do ({len(models)} models x {len(revisions)} revisions, "
          f"finished ones skipped), {args.parallel} at a time")
    if args.dry_run:
        for m, rev, _ in jobs:
            print(f"  {m} @ {rev}")
        return 0

    running: list[tuple] = []
    failed = []
    start = time.time()

    def reap(block: bool):
        while running:
            for item in list(running):
                proc, m, rev, t0, log = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                log.close()
                status = "ok" if proc.returncode == 0 else f"FAILED (see {log.name})"
                if proc.returncode:
                    failed.append((m, rev))
                print(f"[beta] {m} @ {rev}: {status} in {(time.time() - t0) / 60:.1f} min "
                      f"(elapsed {(time.time() - start) / 60:.1f} min)", flush=True)
            if not block or len(running) < args.parallel:
                return
            time.sleep(5)

    for m, rev, out_dir in jobs:
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        log = open(out_dir / "logs" / f"{m}.log", "w")
        cmd = [sys.executable, "-m", "src.datadecide_sweep", "--model", m, "--revision", rev,
               "--device", "cuda", "--dtype", "bfloat16", "--output-dir", str(out_dir)]
        running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), m, rev, time.time(), log))
        reap(block=True)
    while running:
        reap(block=False)
        time.sleep(5)
    for rev in revisions:
        out_dir = Path(args.out_root) / rev
        if out_dir.exists():
            print(f"[beta] {write_summary_csv(out_dir, recipe_order=list(config['recipes']))}")
    if failed:
        print(f"[beta] failed: {failed} -- rerun the same command to retry them")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
