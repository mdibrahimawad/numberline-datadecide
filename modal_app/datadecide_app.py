"""
Modal sweep: number-line geometry on the 25 DataDecide 1B models (Ai2).

Paper protocol per model (see src/datadecide_sweep.py): groups 1-4, k=40,
3 in-context examples, random context, correct-output filter with resampling;
layer selected on seeds 42-44 by median sqrt(EV * |rho|), then evaluated
frozen on seeds 45-47 with both direct (+R2) and log beta fits.

DataDecide checkpoints use ai2-olmo's `hf_olmo.OLMoForCausalLM`, which only
gets `.generate()` from transformers < 4.50 (later versions dropped
GenerationMixin from PreTrainedModel), so this app pins its own image and
leaves modal_app/geometry_app.py untouched.

    modal run modal_app/datadecide_app.py --dry-run
    modal run modal_app/datadecide_app.py --models dolma1_7,c4
    modal run modal_app/datadecide_app.py            # all 25 recipes
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "numberline-datadecide"
GPU = os.environ.get("MODAL_DATADECIDE_GPU", "L4")  # or "A10G"
MAX_PARALLEL_MODELS = 5
HF_CACHE_DIR = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.49.0",
        "ai2-olmo==0.6.0",
        "accelerate>=0.33.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.11.0",
        "numpy>=1.26.0",
        "huggingface_hub>=0.24.0",
        "safetensors>=0.4.3",
        "datasets>=2.20.0",
    )
    .env({"HF_HOME": HF_CACHE_DIR})
    .add_local_python_source("src", "utils")
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("numberline-datadecide-hf-cache", create_if_missing=True)

# DataDecide repos are public; set MODAL_HF_SECRET_NAME to attach an HF_TOKEN secret.
_HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET_NAME", "").strip()
_HF_SECRETS = [modal.Secret.from_name(_HF_SECRET_NAME)] if _HF_SECRET_NAME else []


@app.function(
    gpu=GPU,
    timeout=3 * 60 * 60,
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=_HF_SECRETS,
    max_containers=MAX_PARALLEL_MODELS,
)
def run_datadecide(model_name: str, revision: str | None) -> dict:
    from src.datadecide_sweep import run_datadecide_model

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    try:
        return run_datadecide_model(
            model_name, revision, device="cuda", dtype="bfloat16", hf_token=token
        )
    finally:
        hf_cache.commit()


@app.local_entrypoint()
def main(
    models: str = "",
    dry_run: bool = False,
    output_dir: str = "results/datadecide",
) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from src.datadecide_sweep import (
        load_model_config,
        repo_id_for,
        verify_revisions,
        write_summary_csv,
    )

    config = load_model_config()
    recipes = [m.strip() for m in models.split(",") if m.strip()] or list(config["recipes"])
    unknown = [r for r in recipes if r not in config["recipes"]]
    if unknown:
        raise SystemExit(f"[datadecide] unknown recipes (see configs/datadecide_models.json): {unknown}")
    revision = config["revision"]
    repo_ids = {recipe: repo_id_for(recipe, config) for recipe in recipes}

    print(f"[datadecide] verifying {len(repo_ids)} repos at revision {revision!r} ...")
    problems = verify_revisions(list(repo_ids.values()), revision, token=os.environ.get("HF_TOKEN"))
    for repo_id in repo_ids.values():
        print(f"  {'MISSING' if repo_id in problems else 'ok     '} {repo_id}"
              + (f"  ({problems[repo_id]})" if repo_id in problems else ""))
    if problems:
        raise SystemExit(f"[datadecide] {len(problems)} repo(s) failed verification; aborting")

    print(f"[datadecide] gpu={GPU} max_parallel={MAX_PARALLEL_MODELS} output_dir={output_dir}")
    if dry_run:
        print("[datadecide] --dry-run set; exiting before any GPU work")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    futures = {recipe: run_datadecide.spawn(repo_id, revision) for recipe, repo_id in repo_ids.items()}
    failed = []
    for recipe, fut in futures.items():
        try:
            payload = fut.get()
        except Exception as exc:
            print(f"[datadecide] {recipe} FAILED: {type(exc).__name__}: {exc}")
            failed.append(recipe)
            continue
        payload["recipe"] = recipe
        (out_dir / f"{recipe}.json").write_text(json.dumps(payload, indent=2))
        ev = payload["evaluation"]["summary"]
        print(
            f"[datadecide] {recipe}: layer={payload['selection']['selected_layer']} "
            f"rho={ev['rho_mean']:.3f} EV={ev['explained_variance_mean']:.3f} "
            f"beta_direct={ev['beta_mean']:.3f} beta_log={ev['beta_log_mean']:.3f}"
        )

    summary = write_summary_csv(out_dir, recipe_order=list(config["recipes"]))
    print(f"[datadecide] wrote {summary}")
    if failed:
        print(f"[datadecide] failed models: {failed}")
