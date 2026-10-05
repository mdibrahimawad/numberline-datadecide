"""
Modal entrypoint for the Stage 3 circuit-attribution pipeline.

For one (model, target_layer) pair, the GPU function loads the model via
TransformerLens, fits PCA on the last-token residual stream at the target
layer, and decomposes that PC1 direction into per-head and per-MLP
contributions correlated with log10(target). Results are returned as a JSON
payload, then logged to the same MLflow store as Stages 1-2.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "pile-number-circuits"

DEFAULT_GPU = os.environ.get("MODAL_CIRCUITS_GPU", "A10G")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        # Pin transformers to a TL-compatible window (rotary_pct exists,
        # TRANSFORMERS_CACHE exists). circuits_app is its own image so this
        # doesn't affect Stages 1-2.
        "transformers==4.41.2",
        "tokenizers>=0.19,<0.20",
        "accelerate==0.31.0",
        "transformer_lens==2.10.0",
        "einops>=0.7.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.11.0",
        "numpy>=1.26.0,<2.0",
        "huggingface_hub>=0.24.0",
        "safetensors>=0.4.3",
    )
    .add_local_python_source("src", "utils")
)

app = modal.App(APP_NAME, image=image)

_HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET_NAME", "").strip()
_HF_SECRETS = (
    [modal.Secret.from_name(_HF_SECRET_NAME)] if _HF_SECRET_NAME else []
)


def _gpu_function_kwargs(gpu: str, timeout: int):
    kwargs: dict = {"gpu": gpu, "timeout": timeout}
    if _HF_SECRETS:
        kwargs["secrets"] = _HF_SECRETS
    return kwargs


@app.function(**_gpu_function_kwargs(DEFAULT_GPU, timeout=2 * 60 * 60))
def attribute_model(
    model_name: str,
    target_layer: int,
    *,
    data: str = "numerics",
    groups: list[int] = [1, 2, 3, 4],
    k: int = 30,
    num_examples: int = 3,
    context: str = "random",
    upper_bound: int | None = None,
    seed: int = 42,
    dtype: str = "float16",
) -> dict:
    from src.circuits import CircuitConfig, run_attribution

    cfg = CircuitConfig(
        model_name=model_name,
        target_layer=target_layer,
        data=data,
        groups=tuple(sorted(groups)),
        k=k,
        num_examples=num_examples,
        context=context,
        upper_bound=upper_bound,
        seed=seed,
        dtype=dtype,
        device="cuda",
    )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def _on_progress(stage: str, done: int, total: int) -> None:
        print(f"[worker:{model_name}] {stage} {done}/{total}")

    print(f"[worker] attribute_model start: {model_name} target_layer={target_layer}")
    results = run_attribution(cfg, hf_token=hf_token, progress_callback=_on_progress)
    payload = results.to_dict()
    print(
        f"[worker] attribute_model done: {model_name} "
        f"top_head=(L{int(max(range(target_layer + 1), key=lambda l: max(abs(c) for c in results.head_corr_log_target[l])))},"
        f" h{int(max(range(results.n_heads), key=lambda h: max(abs(results.head_corr_log_target[l][h]) for l in range(target_layer + 1))))})"
    )
    return payload


def _log_run(payload: dict, run_name: str, dataset_repo_tag: str) -> None:
    import json as _json

    import mlflow
    import numpy as np

    from src.circuits import AttributionResults, CircuitConfig, metrics_for_mlflow
    from utils.mlflow_utils import (
        log_metric,
        log_metrics,
        log_params,
        start_run,
    )

    cfg_d = payload["config"]
    cfg = CircuitConfig(
        model_name=cfg_d["model_name"],
        target_layer=cfg_d["target_layer"],
        data=cfg_d["data"],
        groups=tuple(cfg_d["groups"]),
        k=cfg_d["k"],
        num_examples=cfg_d["num_examples"],
        context=cfg_d["context"],
        upper_bound=cfg_d["upper_bound"],
        seed=cfg_d["seed"],
        dtype=cfg_d["dtype"],
        device=cfg_d.get("device", "cuda"),
    )
    results = AttributionResults(
        config=cfg,
        n_layers_total=int(payload["n_layers_total"]),
        n_heads=int(payload["n_heads"]),
        d_model=int(payload["d_model"]),
        pc1_direction=np.asarray(payload["pc1_direction"], dtype=np.float32),
        pc1_score=np.asarray(payload["pc1_score"], dtype=np.float32),
        embed_proj=np.asarray(payload["embed_proj"], dtype=np.float32),
        head_proj=np.asarray(payload["head_proj"], dtype=np.float32),
        mlp_proj=np.asarray(payload["mlp_proj"], dtype=np.float32),
        head_corr_log_target=np.asarray(payload["head_corr_log_target"], dtype=np.float32),
        mlp_corr_log_target=np.asarray(payload["mlp_corr_log_target"], dtype=np.float32),
        targets=np.asarray(payload["targets"], dtype=float),
        log_targets=np.asarray(payload["log_targets"], dtype=float),
        pca_explained_variance=float(payload["pca_explained_variance"]),
    )

    tags = {
        "stage": "number_circuits",
        "backend": "modal",
        "dataset": dataset_repo_tag,
        "model_name": cfg.model_name,
        "data": cfg.data,
        "context": cfg.context,
        "target_layer": str(cfg.target_layer),
    }

    with start_run(run_name=run_name, tags=tags):
        log_params(
            {
                "model_name": cfg.model_name,
                "target_layer": cfg.target_layer,
                "data": cfg.data,
                "groups": ",".join(map(str, cfg.groups)),
                "k": cfg.k,
                "num_examples": cfg.num_examples,
                "context": cfg.context,
                "upper_bound": cfg.upper_bound,
                "seed": cfg.seed,
                "dtype": cfg.dtype,
            }
        )

        flat = metrics_for_mlflow(results)
        batch: dict[str, float] = {}
        for kk, vv in flat.items():
            batch[kk] = float(vv)
            if len(batch) >= 500:
                log_metrics(batch)
                batch = {}
        if batch:
            log_metrics(batch)
        log_metric("n_prompts", float(len(results.targets)))

        mlflow.log_text(_json.dumps(payload), "circuits/results.json")


@app.function(**_gpu_function_kwargs(DEFAULT_GPU, timeout=2 * 60 * 60))
def patch_model(
    model_name: str,
    target_layer: int,
    *,
    data: str = "numerics",
    groups: list[int] = [1, 2, 3, 4],
    k: int = 30,
    num_examples: int = 3,
    context: str = "random",
    upper_bound: int | None = None,
    seed: int = 42,
    dtype: str = "float16",
    n_pairs: int = 16,
) -> dict:
    from src.circuits import PatchingConfig, run_patching

    cfg = PatchingConfig(
        model_name=model_name,
        target_layer=target_layer,
        data=data,
        groups=tuple(sorted(groups)),
        k=k,
        num_examples=num_examples,
        context=context,
        upper_bound=upper_bound,
        seed=seed,
        dtype=dtype,
        device="cuda",
        n_pairs=n_pairs,
    )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def _on_progress(stage: str, done: int, total: int) -> None:
        print(f"[worker:{model_name}] {stage} {done}/{total}")

    print(f"[worker] patch_model start: {model_name} target_layer={target_layer} "
          f"n_pairs={n_pairs}")
    results = run_patching(cfg, hf_token=hf_token, progress_callback=_on_progress)
    payload = results.to_dict()
    print(f"[worker] patch_model done: {model_name}")
    return payload


def _log_patching_run(payload: dict, run_name: str, dataset_repo_tag: str) -> None:
    import json as _json

    import mlflow
    import numpy as np

    from src.circuits import (
        PatchingConfig,
        PatchingResults,
        metrics_for_mlflow_patching,
    )
    from utils.mlflow_utils import (
        log_metric,
        log_metrics,
        log_params,
        start_run,
    )

    cfg_d = payload["config"]
    cfg = PatchingConfig(
        model_name=cfg_d["model_name"],
        target_layer=cfg_d["target_layer"],
        data=cfg_d["data"],
        groups=tuple(cfg_d["groups"]),
        k=cfg_d["k"],
        num_examples=cfg_d["num_examples"],
        context=cfg_d["context"],
        upper_bound=cfg_d["upper_bound"],
        seed=cfg_d["seed"],
        dtype=cfg_d["dtype"],
        device=cfg_d.get("device", "cuda"),
        n_pairs=cfg_d.get("n_pairs", 16),
    )
    results = PatchingResults(
        config=cfg,
        n_layers_total=int(payload["n_layers_total"]),
        n_heads=int(payload["n_heads"]),
        target_layer=int(payload["target_layer"]),
        pca_explained_variance=float(payload["pca_explained_variance"]),
        donor_pc1=np.asarray(payload["donor_pc1"], dtype=np.float32),
        receiver_pc1=np.asarray(payload["receiver_pc1"], dtype=np.float32),
        donor_targets=np.asarray(payload["donor_targets"], dtype=float),
        receiver_targets=np.asarray(payload["receiver_targets"], dtype=float),
        head_effect_mean=np.asarray(payload["head_effect_mean"], dtype=np.float32),
        head_effect_std=np.asarray(payload["head_effect_std"], dtype=np.float32),
        mlp_effect_mean=np.asarray(payload["mlp_effect_mean"], dtype=np.float32),
        mlp_effect_std=np.asarray(payload["mlp_effect_std"], dtype=np.float32),
        pair_donor_groups=np.asarray(payload["pair_donor_groups"], dtype=int),
        pair_receiver_groups=np.asarray(payload["pair_receiver_groups"], dtype=int),
    )

    tags = {
        "stage": "number_circuits_patching",
        "backend": "modal",
        "dataset": dataset_repo_tag,
        "model_name": cfg.model_name,
        "data": cfg.data,
        "context": cfg.context,
        "target_layer": str(cfg.target_layer),
    }

    with start_run(run_name=run_name, tags=tags):
        log_params(
            {
                "model_name": cfg.model_name,
                "target_layer": cfg.target_layer,
                "data": cfg.data,
                "groups": ",".join(map(str, cfg.groups)),
                "k": cfg.k,
                "num_examples": cfg.num_examples,
                "context": cfg.context,
                "upper_bound": cfg.upper_bound,
                "seed": cfg.seed,
                "dtype": cfg.dtype,
                "n_pairs": cfg.n_pairs,
            }
        )

        flat = metrics_for_mlflow_patching(results)
        batch: dict[str, float] = {}
        for kk, vv in flat.items():
            batch[kk] = float(vv)
            if len(batch) >= 500:
                log_metrics(batch)
                batch = {}
        if batch:
            log_metrics(batch)

        mlflow.log_text(_json.dumps(payload), "circuits/patching.json")


@app.local_entrypoint()
def main(
    models: str = "EleutherAI/pythia-2.8b",
    target_layer: int = 9,
    data: str = "numerics",
    groups: str = "1,2,3,4",
    k: int = 30,
    num_examples: int = 3,
    context: str = "random",
    upper_bound: int = -1,
    seed: int = 42,
    dtype: str = "float16",
    experiment_name: str = "numberline_circuits_expv1",
    run_name_prefix: str = "modal_circuits",
    dataset_repo_tag: str = "monology/pile-uncopyrighted",
    dry_run: bool = False,
) -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from utils.mlflow_utils import setup_tracking

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        raise SystemExit("--models must contain at least one HF model id")

    group_list = [int(x) for x in groups.split(",") if x.strip()]
    if not group_list:
        raise SystemExit("--groups must contain at least one integer")

    ub: int | None = None if upper_bound is None or upper_bound < 0 else int(upper_bound)

    print(f"[circuits-modal] sweep over {len(model_list)} models at layer {target_layer}: "
          f"{model_list}")
    if dry_run:
        print("[circuits-modal] --dry-run set; exiting before any cloud work")
        return

    setup_tracking(None, experiment_name)

    common_kwargs = dict(
        target_layer=target_layer,
        data=data,
        groups=group_list,
        k=k,
        num_examples=num_examples,
        context=context,
        upper_bound=ub,
        seed=seed,
        dtype=dtype,
    )

    futures = [
        attribute_model.spawn(model_name=m, **common_kwargs) for m in model_list
    ]

    for model_name, fut in zip(model_list, futures):
        print(f"[circuits-modal] waiting on {model_name} ...")
        try:
            payload = fut.get()
        except Exception as e:
            print(f"[circuits-modal] {model_name} FAILED: {e}")
            continue
        run_name = (
            f"{run_name_prefix}_{model_name.replace('/', '_')}"
            f"_L{target_layer}_{data}"
        )
        _log_run(payload, run_name=run_name, dataset_repo_tag=dataset_repo_tag)
        print(f"[circuits-modal] logged MLflow run for {model_name}")

    print("[circuits-modal] all done.")


@app.local_entrypoint()
def patch(
    models: str = "EleutherAI/pythia-2.8b",
    target_layer: int = 9,
    data: str = "numerics",
    groups: str = "1,2,3,4",
    k: int = 30,
    num_examples: int = 3,
    context: str = "random",
    upper_bound: int = -1,
    seed: int = 42,
    dtype: str = "float16",
    n_pairs: int = 16,
    experiment_name: str = "numberline_circuits_expv1",
    run_name_prefix: str = "modal_patching",
    dataset_repo_tag: str = "monology/pile-uncopyrighted",
    dry_run: bool = False,
) -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from utils.mlflow_utils import setup_tracking

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        raise SystemExit("--models must contain at least one HF model id")
    group_list = [int(x) for x in groups.split(",") if x.strip()]
    ub: int | None = None if upper_bound is None or upper_bound < 0 else int(upper_bound)

    print(
        f"[patching-modal] sweep over {len(model_list)} models at layer "
        f"{target_layer} (n_pairs={n_pairs}): {model_list}"
    )
    if dry_run:
        print("[patching-modal] --dry-run set; exiting before any cloud work")
        return

    setup_tracking(None, experiment_name)

    common_kwargs = dict(
        target_layer=target_layer,
        data=data,
        groups=group_list,
        k=k,
        num_examples=num_examples,
        context=context,
        upper_bound=ub,
        seed=seed,
        dtype=dtype,
        n_pairs=n_pairs,
    )
    futures = [
        patch_model.spawn(model_name=m, **common_kwargs) for m in model_list
    ]

    for model_name, fut in zip(model_list, futures):
        print(f"[patching-modal] waiting on {model_name} ...")
        try:
            payload = fut.get()
        except Exception as e:
            print(f"[patching-modal] {model_name} FAILED: {e}")
            continue
        run_name = (
            f"{run_name_prefix}_{model_name.replace('/', '_')}"
            f"_L{target_layer}_{data}"
        )
        _log_patching_run(payload, run_name=run_name, dataset_repo_tag=dataset_repo_tag)
        print(f"[patching-modal] logged MLflow run for {model_name}")

    print("[patching-modal] all done.")
