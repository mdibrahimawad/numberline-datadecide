"""Modal entrypoint for high-dimensional number-line manifold diagnostics."""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "numberline-manifold-geometry"

DEFAULT_GPU = os.environ.get("MODAL_MANIFOLD_GPU", "A100-40GB")
DEFAULT_MODELS = (
    "EleutherAI/pythia-2.8b",
    "togethercomputer/RedPajama-INCITE-Base-3B-v1",
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.44.0,<5",
        "accelerate>=0.33.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.11.0",
        "numpy>=1.26.0",
        "huggingface_hub>=0.24.0",
        "safetensors>=0.4.3",
        "sentencepiece>=0.2.0",
        "tiktoken>=0.7.0",
        "ai2-olmo==0.6.0",
        "datasets>=2.20.0",
    )
    .add_local_python_source("src", "utils")
)

app = modal.App(APP_NAME, image=image)

_HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET_NAME", "numberline-hf-token").strip()
_HF_SECRETS = [modal.Secret.from_name(_HF_SECRET_NAME)] if _HF_SECRET_NAME else []


def _gpu_function_kwargs(gpu: str, timeout: int) -> dict:
    kwargs: dict = {"gpu": gpu, "timeout": timeout}
    if _HF_SECRETS:
        kwargs["secrets"] = _HF_SECRETS
    return kwargs


@app.function(**_gpu_function_kwargs(DEFAULT_GPU, timeout=3 * 60 * 60))
def analyze_manifold_model(
    model_name: str,
    *,
    model_revision: str | None = None,
    trust_remote_code: bool = False,
    data: str = "numerics",
    groups: list[int] = [1, 2, 3, 4],
    k: int = 30,
    num_examples: int = 3,
    context: str = "random",
    runs: int = 3,
    upper_bound: int | None = None,
    seed: int = 42,
    dtype: str = "float16",
    device_map: str | None = None,
    target_min: int | None = None,
    target_max: int | None = None,
    isomap_neighbors: int = 12,
    intrinsic_components: int = 24,
    min_select_layer: int = 0,
    dense_anchors: int = 0,
    dense_min: int = 10,
    dense_max: int = 10000,
    save_visualization: bool = False,
) -> dict:
    import os

    from src.geometry import GeometryConfig
    from src.manifold_geometry import run_manifold_geometry

    cfg = GeometryConfig(
        model_name=model_name,
        model_revision=model_revision,
        trust_remote_code=trust_remote_code,
        data=data,
        groups=tuple(sorted(groups)),
        k=k,
        num_examples=num_examples,
        context=context,
        transform_dim=1,
        runs=runs,
        upper_bound=upper_bound,
        seed=seed,
        device="cuda",
        dtype=dtype,
        device_map=device_map,
        save_projections=False,
        target_min=target_min,
        target_max=target_max,
    )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def _on_run(done: int, total: int) -> None:
        print(f"[worker:{model_name}] manifold run {done}/{total}")

    rev_msg = f" revision={model_revision}" if model_revision else ""
    print(
        f"[worker] manifold start: {model_name}{rev_msg} "
        f"data={data} groups={groups} runs={runs}"
    )
    payload = run_manifold_geometry(
        cfg,
        hf_token=hf_token,
        isomap_neighbors=isomap_neighbors,
        intrinsic_components=intrinsic_components,
        min_select_layer=min_select_layer,
        dense_anchors=dense_anchors,
        dense_min=dense_min,
        dense_max=dense_max,
        save_visualization=save_visualization,
        progress_callback=_on_run,
    )
    print(
        f"[worker] manifold done: {model_name} "
        f"best_layer={payload.get('best_layer')}"
    )
    return payload


@app.local_entrypoint()
def main(
    models: str = ",".join(DEFAULT_MODELS),
    model_revisions: str = "",
    trust_remote_code: bool = False,
    data: str = "numerics",
    groups: str = "1,2,3,4",
    k: int = 30,
    num_examples: int = 3,
    context: str = "random",
    runs: int = 3,
    upper_bound: int = -1,
    seed: int = 42,
    dtype: str = "float16",
    device_map: str = "",
    target_min: int = -1,
    target_max: int = -1,
    isomap_neighbors: int = 12,
    intrinsic_components: int = 24,
    min_select_layer: int = 0,
    dense_anchors: int = 0,
    dense_min: int = 10,
    dense_max: int = 10000,
    save_visualization: bool = False,
    results_dir: str = "results/manifold_geometry/pythia_redpajama_goodfire",
    dry_run: bool = False,
) -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.manifold_geometry import write_manifold_artifacts

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        raise SystemExit("--models must contain at least one HF model id")

    revision_list = [r.strip() or None for r in model_revisions.split(",")] if model_revisions else [None] * len(model_list)
    if len(revision_list) == 1 and len(model_list) > 1:
        revision_list = revision_list * len(model_list)
    if len(revision_list) != len(model_list):
        raise SystemExit("--model-revisions must be empty, one value, or match --models")

    group_list = [int(x) for x in groups.split(",") if x.strip()]
    if not group_list:
        raise SystemExit("--groups must contain at least one integer")

    ub: int | None = None if upper_bound is None or upper_bound < 0 else int(upper_bound)
    dm: str | None = device_map or None
    tmin: int | None = None if target_min is None or target_min < 0 else int(target_min)
    tmax: int | None = None if target_max is None or target_max < 0 else int(target_max)

    print(f"[manifold-modal] gpu={DEFAULT_GPU}")
    print(f"[manifold-modal] sweep over {len(model_list)} models: {model_list}")
    if dry_run:
        print("[manifold-modal] --dry-run set; exiting before cloud work")
        return

    common_kwargs = dict(
        data=data,
        groups=group_list,
        k=k,
        num_examples=num_examples,
        context=context,
        runs=runs,
        upper_bound=ub,
        seed=seed,
        dtype=dtype,
        device_map=dm,
        trust_remote_code=trust_remote_code,
        target_min=tmin,
        target_max=tmax,
        isomap_neighbors=isomap_neighbors,
        intrinsic_components=intrinsic_components,
        min_select_layer=min_select_layer,
        dense_anchors=dense_anchors,
        dense_min=dense_min,
        dense_max=dense_max,
        save_visualization=save_visualization,
    )

    futures = [
        analyze_manifold_model.spawn(model_name=m, model_revision=r, **common_kwargs)
        for m, r in zip(model_list, revision_list)
    ]

    payloads: list[dict] = []
    for model_name, fut in zip(model_list, futures):
        print(f"[manifold-modal] waiting on {model_name} ...")
        try:
            payload = fut.get()
        except Exception as e:
            print(f"[manifold-modal] {model_name} FAILED: {e}")
            continue
        payloads.append(payload)

    if not payloads:
        raise SystemExit("[manifold-modal] no successful payloads")

    paths = write_manifold_artifacts(payloads, results_dir)
    print("[manifold-modal] wrote artifacts:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    print("[manifold-modal] all done.")
