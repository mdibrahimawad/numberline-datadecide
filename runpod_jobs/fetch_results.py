"""Download the exact-alpha results that RunPod pods uploaded (--upload-hf) into
results/corpus_alpha_datadecide/ on this machine and rebuild alpha_seed2.csv.

    python -m runpod_jobs.fetch_results                        # repo numberline-alpha-results
    python -m runpod_jobs.fetch_results --repo <user>/<name>

Needs HF_TOKEN (the same token the pods used) in the environment or `hf auth login`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from huggingface_hub import HfApi, snapshot_download

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default="numberline-alpha-results")
    p.add_argument("--out-dir", default="results/corpus_alpha_datadecide")
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    repo = args.repo if "/" in args.repo else f"{HfApi(token=token).whoami()['name']}/{args.repo}"
    out = Path(args.out_dir)
    snapshot_download(repo, repo_type="dataset", local_dir=str(out), token=token,
                      allow_patterns=["exact_100b_*/summary.json", "exact_100b_*/*.csv"])
    got = sorted(d.name for d in out.glob("exact_100b_*") if (d / "summary.json").exists())
    print(f"[fetch] {len(got)} recipe folders in {out}")

    from runpod_jobs.exact_alpha import write_alpha_csv
    from src.datadecide_sampling import load_data_map

    n = write_alpha_csv(out, args.seed, load_data_map())
    print(f"[fetch] {out / f'alpha_seed{args.seed}.csv'} now has {n} of 25 recipes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
