"""Download results that RunPod pods uploaded (--upload-hf) to this machine.

    python -m runpod_jobs.fetch_results                 # exact alpha -> results/corpus_alpha_datadecide/
    python -m runpod_jobs.fetch_results --kind beta     # fine beta   -> results/beta_fine/
    python -m runpod_jobs.fetch_results --kind paloma   # E09 counts  -> results/paloma_counts/

Needs HF_TOKEN (the same token the pods used) in the environment or `hf auth login`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from huggingface_hub import HfApi, snapshot_download

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kind", choices=("alpha", "beta", "paloma"), default="alpha")
    p.add_argument("--repo", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    name = args.repo or ("numberline-beta-results" if args.kind == "beta" else "numberline-alpha-results")
    repo = name if "/" in name else f"{HfApi(token=token).whoami()['name']}/{name}"
    if args.kind == "beta":
        out = Path(args.out_dir or "results/beta_fine")
        snapshot_download(repo, repo_type="dataset", local_dir=str(out), token=token,
                          allow_patterns=["*/*.json", "*/*.npz", "*/summary.csv"])
        got = sorted(out.glob("*/*.json"))
        print(f"[fetch] {len(got)} model results in {out}; summary: {sorted(out.glob('*/summary.csv'))}")
        return 0
    if args.kind == "paloma":
        return _fetch_paloma(repo, token, Path(args.out_dir or "results/paloma_counts"))
    out = Path(args.out_dir or "results/corpus_alpha_datadecide")
    snapshot_download(repo, repo_type="dataset", local_dir=str(out), token=token,
                      allow_patterns=["exact_100b_*/summary.json", "exact_100b_*/*.csv"])
    got = sorted(d.name for d in out.glob("exact_100b_*") if (d / "summary.json").exists())
    print(f"[fetch] {len(got)} recipe folders in {out}")

    from runpod_jobs.exact_alpha import write_alpha_csv
    from src.datadecide_sampling import load_data_map

    n = write_alpha_csv(out, args.seed, load_data_map())
    print(f"[fetch] {out / f'alpha_seed{args.seed}.csv'} now has {n} of 25 recipes")
    return 0


def _fetch_paloma(repo: str, token: str | None, out: Path) -> int:
    """E09 counts -> results/paloma_counts/<corpus>/{counts.csv,summary.json}: the five
    random-sample corpora as uploaded, Dolma by adding up its 8 exact parts."""
    import json
    import shutil

    from huggingface_hub import snapshot_download

    from runpod_jobs.corpus_sample import count_summary
    from runpod_jobs.exact_alpha import atomic_write, merge_parts
    from src.datadecide_sampling import load_data_map

    raw = out / "_hf"
    snapshot_download(repo, repo_type="dataset", local_dir=str(raw), token=token,
                      allow_patterns=["paloma/*/*", "exact_100b_paloma-dolma-*/*"])
    for d in sorted((raw / "paloma").glob("*")):
        shutil.copytree(d, out / d.name, dirs_exist_ok=True)
    dmap_path = Path("configs/paloma/paloma_dolma_exact_map.json")
    dmap = load_data_map(dmap_path)
    try:
        counts, info = merge_parts(raw, dmap, "paloma-dolma", int(dmap["seed"]))
    except SystemExit as exc:
        print(f"[fetch] dolma not complete yet: {exc}")
    else:
        d = out / "dolma"
        d.mkdir(parents=True, exist_ok=True)
        atomic_write(d / "counts.csv", "number,count\n" + "".join(f"{n},{int(c)}\n" for n, c in enumerate(counts)))
        summary = {"corpus": "dolma", "method": "exact training stream (runpod_jobs.exact_alpha, 8 parts added up)",
                   "tokens": info["tokens"], "parts": info["parts"], "seed": dmap["seed"],
                   "data_map": str(dmap_path), **count_summary(counts)}
        atomic_write(d / "summary.json", json.dumps(summary, indent=2) + "\n")
    got = sorted(d.name for d in out.iterdir() if d.name != "_hf" and (d / "summary.json").exists())
    print(f"[fetch] {len(got)} of 6 Paloma corpora in {out}: {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
