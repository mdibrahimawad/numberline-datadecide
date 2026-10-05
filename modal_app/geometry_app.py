"""
Modal entrypoint for the number-line geometry probe.

Pairs with `src.geometry_cli` but offloads the model forward passes to a Modal
GPU container. The local entrypoint sweeps any number of HF model ids, each on
its own GPU container, and logs one MLflow run per (model, data) cell into the
same SQLite store used by Stage 1.

Public models (Pythia, GPT-2, Mistral-base) load with no auth.
Gated models (Llama-2/3) need a Modal Secret containing HF_TOKEN; opt in by
exporting MODAL_HF_SECRET_NAME=<your_secret_name> before `modal run ...`.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "numberline-geometry"

DEFAULT_GPU = os.environ.get("MODAL_GEOMETRY_GPU", "A10G")
DEFAULT_MODELS = (
    "EleutherAI/pythia-2.8b",
    "openai-community/gpt2-large",
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
hf_cache = modal.Volume.from_name("numberline-hf-cache", create_if_missing=True)

_HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET_NAME", "numberline-hf-token").strip()
_HF_SECRETS = (
    [modal.Secret.from_name(_HF_SECRET_NAME)] if _HF_SECRET_NAME else []
)


def _gpu_function_kwargs(gpu: str, timeout: int):
    kwargs: dict = {
        "gpu": gpu,
        "timeout": timeout,
        "volumes": {"/root/.cache/huggingface": hf_cache},
    }
    if _HF_SECRETS:
        kwargs["secrets"] = _HF_SECRETS
    return kwargs


@app.function(**_gpu_function_kwargs(DEFAULT_GPU, timeout=2 * 60 * 60))
def analyze_model(
    model_name: str,
    *,
    model_revision: str | None = None,
    trust_remote_code: bool = False,
    data: str = "numerics",
    groups: list[int] = [1, 2, 3, 4],
    k: int = 30,
    num_examples: int = 3,
    context: str = "random",
    transform_dim: int = 1,
    runs: int = 3,
    upper_bound: int | None = None,
    seed: int = 42,
    dtype: str = "float16",
    device_map: str | None = None,
    save_projections: bool = False,
    target_min: int | None = None,
    target_max: int | None = None,
    tokenizer_use_fast: bool = True,
    prepend_bos: bool = False,
    spacing_fit: str = "log",
    filter_correct: bool = False,
    filter_max_new_tokens: int = 8,
    filter_max_candidates: int = 100,
) -> dict:
    import os

    from src.geometry import GeometryConfig, run_geometry

    cfg = GeometryConfig(
        model_name=model_name,
        model_revision=model_revision,
        trust_remote_code=trust_remote_code,
        data=data,
        groups=tuple(sorted(groups)),
        k=k,
        num_examples=num_examples,
        context=context,
        transform_dim=transform_dim,
        runs=runs,
        upper_bound=upper_bound,
        seed=seed,
        device="cuda",
        dtype=dtype,
        device_map=device_map,
        save_projections=save_projections,
        target_min=target_min,
        target_max=target_max,
        tokenizer_use_fast=tokenizer_use_fast,
        prepend_bos=prepend_bos,
        spacing_fit=spacing_fit,
        filter_correct=filter_correct,
        filter_max_new_tokens=filter_max_new_tokens,
        filter_max_candidates=filter_max_candidates,
    )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def _on_run(done: int, total: int) -> None:
        print(f"[worker:{model_name}] run {done}/{total}")

    rev_msg = f" revision={model_revision}" if model_revision else ""
    print(f"[worker] analyze_model start: {model_name}{rev_msg} data={data} runs={runs}")
    results = run_geometry(cfg, hf_token=hf_token, progress_callback=_on_run)
    payload = results.to_dict()
    print(
        f"[worker] analyze_model done: {model_name} "
        f"best_pca_layer={payload['best_layer_pca']} "
        f"best_pls_layer={payload['best_layer_pls']}"
    )
    return payload


@app.function(**_gpu_function_kwargs(DEFAULT_GPU, timeout=2 * 60 * 60))
def heldout_analyze_model(
    model_name: str,
    *,
    k: int = 60,
    runs: int = 3,
    seed: int = 42,
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
) -> dict:
    import os

    from src.geometry import GeometryConfig
    from src.heldout_geometry import run_heldout_geometry

    cfg = GeometryConfig(
        model_name=model_name,
        groups=(1, 2, 3, 4),
        k=k,
        num_examples=3,
        context="random",
        runs=runs,
        seed=seed,
        device="cuda",
        dtype=dtype,
        device_map=device_map,
        spacing_fit="direct",
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def progress(done: int, total: int) -> None:
        print(f"[heldout:{model_name}] run {done}/{total}")

    return run_heldout_geometry(cfg, hf_token=token, progress_callback=progress)


@app.function(**_gpu_function_kwargs(DEFAULT_GPU, timeout=2 * 60 * 60))
def per_seed_analyze_model(
    model_name: str,
    *,
    model_revision: str | None = None,
    trust_remote_code: bool = False,
    k: int = 30,
    runs: int = 3,
    seed: int = 42,
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
    tokenizer_use_fast: bool = True,
    prepend_bos: bool = False,
) -> dict:
    import os

    from src.geometry import GeometryConfig
    from src.per_seed_geometry import run_per_seed_geometry

    cfg = GeometryConfig(
        model_name=model_name,
        model_revision=model_revision,
        trust_remote_code=trust_remote_code,
        groups=(1, 2, 3, 4),
        k=k,
        num_examples=3,
        context="random",
        runs=runs,
        seed=seed,
        device="cuda",
        dtype=dtype,
        device_map=device_map,
        tokenizer_use_fast=tokenizer_use_fast,
        prepend_bos=prepend_bos,
        spacing_fit="direct",
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def progress(done: int, total: int) -> None:
        print(f"[per-seed:{model_name}] run {done}/{total}")

    return run_per_seed_geometry(cfg, hf_token=token, progress_callback=progress)


@app.function(**_gpu_function_kwargs(DEFAULT_GPU, timeout=2 * 60 * 60))
def frozen_layer_analyze_model(
    model_name: str,
    target_layer: int,
    *,
    model_revision: str | None = None,
    trust_remote_code: bool = False,
    k: int = 30,
    runs: int = 3,
    seed: int = 45,
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
    tokenizer_use_fast: bool = True,
    prepend_bos: bool = False,
) -> dict:
    import os

    from src.frozen_layer_evaluation import run_frozen_layer_evaluation
    from src.geometry import GeometryConfig

    cfg = GeometryConfig(
        model_name=model_name,
        model_revision=model_revision,
        trust_remote_code=trust_remote_code,
        groups=(1, 2, 3, 4),
        k=k,
        num_examples=3,
        context="random",
        runs=runs,
        seed=seed,
        device="cuda",
        dtype=dtype,
        device_map=device_map,
        tokenizer_use_fast=tokenizer_use_fast,
        prepend_bos=prepend_bos,
        spacing_fit="direct",
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def progress(done: int, total: int) -> None:
        print(f"[frozen-layer:{model_name}:L{target_layer}] run {done}/{total}")

    return run_frozen_layer_evaluation(
        cfg,
        target_layer,
        hf_token=token,
        progress_callback=progress,
    )


def _log_one_run(payload: dict, run_name: str, dataset_repo_tag: str) -> None:
    import json as _json

    import mlflow

    from src.geometry import GeometryConfig, GeometryResults, LayerSummary, metrics_for_mlflow
    from utils.mlflow_utils import (
        log_metric,
        log_metrics,
        log_params,
        start_run,
    )

    cfg_d = payload["config"]
    cfg = GeometryConfig(
        model_name=cfg_d["model_name"],
        model_revision=cfg_d.get("model_revision"),
        trust_remote_code=cfg_d.get("trust_remote_code", False),
        data=cfg_d["data"],
        groups=tuple(cfg_d["groups"]),
        k=cfg_d["k"],
        num_examples=cfg_d["num_examples"],
        context=cfg_d["context"],
        transform_dim=cfg_d["transform_dim"],
        runs=cfg_d["runs"],
        upper_bound=cfg_d["upper_bound"],
        seed=cfg_d["seed"],
        device=cfg_d["device"],
        dtype=cfg_d["dtype"],
        device_map=cfg_d["device_map"],
        save_projections=cfg_d.get("save_projections", False),
        target_min=cfg_d.get("target_min"),
        target_max=cfg_d.get("target_max"),
        tokenizer_use_fast=cfg_d.get("tokenizer_use_fast", True),
        prepend_bos=cfg_d.get("prepend_bos", False),
        spacing_fit=cfg_d.get("spacing_fit", "log"),
        filter_correct=cfg_d.get("filter_correct", False),
        filter_max_new_tokens=cfg_d.get("filter_max_new_tokens", 8),
        filter_max_candidates=cfg_d.get("filter_max_candidates", 100),
    )

    def _hydrate(layer_map: dict) -> dict[int, LayerSummary]:
        return {int(l): LayerSummary(**v) for l, v in layer_map.items()}

    results = GeometryResults(
        config=cfg,
        pca=_hydrate(payload["pca"]),
        pls=_hydrate(payload["pls"]),
        n_layers=payload["n_layers"],
        best_layer_pca=payload["best_layer_pca"],
        best_layer_pls=payload["best_layer_pls"],
        tokenization_diagnostics=payload.get("tokenization_diagnostics"),
    )

    tags = {
        "stage": "number_geometry",
        "backend": "modal",
        "dataset": dataset_repo_tag,
        "model_name": cfg.model_name,
        "model_revision": cfg.model_revision or "",
        "data": cfg.data,
        "context": cfg.context,
    }

    with start_run(run_name=run_name, tags=tags):
        log_params(
            {
                "model_name": cfg.model_name,
                "model_revision": cfg.model_revision or "main",
                "trust_remote_code": cfg.trust_remote_code,
                "data": cfg.data,
                "groups": ",".join(map(str, cfg.groups)),
                "k": cfg.k,
                "num_examples": cfg.num_examples,
                "context": cfg.context,
                "transform_dim": cfg.transform_dim,
                "runs": cfg.runs,
                "upper_bound": cfg.upper_bound,
                "seed": cfg.seed,
                "dtype": cfg.dtype,
                "device_map": cfg.device_map or "single",
                "target_min": cfg.target_min,
                "target_max": cfg.target_max,
                "tokenizer_use_fast": cfg.tokenizer_use_fast,
                "prepend_bos": cfg.prepend_bos,
                "spacing_fit": cfg.spacing_fit,
                "filter_correct": cfg.filter_correct,
                "filter_max_new_tokens": cfg.filter_max_new_tokens,
                "filter_max_candidates": cfg.filter_max_candidates,
            }
        )
        log_metric("n_layers", float(results.n_layers))

        flat = metrics_for_mlflow(results)
        batch: dict[str, float] = {}
        for kk, vv in flat.items():
            batch[kk] = float(vv)
            if len(batch) >= 500:
                log_metrics(batch)
                batch = {}
        if batch:
            log_metrics(batch)

        mlflow.log_text(_json.dumps(payload, indent=2), "geometry/results.json")


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
    transform_dim: int = 1,
    runs: int = 3,
    upper_bound: int = -1,
    seed: int = 42,
    dtype: str = "float16",
    device_map: str = "",
    save_projections: bool = False,
    target_min: int = -1,
    target_max: int = -1,
    tokenizer_use_fast: bool = True,
    prepend_bos: bool = False,
    spacing_fit: str = "log",
    filter_correct: bool = False,
    filter_max_new_tokens: int = 8,
    filter_max_candidates: int = 100,
    experiment_name: str = "numberline_geometry_expv1",
    run_name_prefix: str = "modal_geometry",
    dataset_repo_tag: str = "numberline_numerics",
    output_dir: str = "",
    dry_run: bool = False,
) -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from utils.mlflow_utils import setup_tracking

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

    print(f"[geometry-modal] sweep over {len(model_list)} models: {model_list}")
    if any(revision_list):
        print(f"[geometry-modal] model revisions: {revision_list}")
    if dry_run:
        print("[geometry-modal] --dry-run set; exiting before any cloud work")
        return

    setup_tracking(None, experiment_name)

    common_kwargs = dict(
        data=data,
        groups=group_list,
        k=k,
        num_examples=num_examples,
        context=context,
        transform_dim=transform_dim,
        runs=runs,
        upper_bound=ub,
        seed=seed,
        dtype=dtype,
        device_map=dm,
        save_projections=save_projections,
        trust_remote_code=trust_remote_code,
        target_min=tmin,
        target_max=tmax,
        tokenizer_use_fast=tokenizer_use_fast,
        prepend_bos=prepend_bos,
        spacing_fit=spacing_fit,
        filter_correct=filter_correct,
        filter_max_new_tokens=filter_max_new_tokens,
        filter_max_candidates=filter_max_candidates,
    )

    futures = [
        analyze_model.spawn(model_name=m, model_revision=r, **common_kwargs)
        for m, r in zip(model_list, revision_list)
    ]

    for model_name, model_revision, fut in zip(model_list, revision_list, futures):
        print(f"[geometry-modal] waiting on {model_name} ...")
        try:
            payload = fut.get()
        except Exception as e:
            print(f"[geometry-modal] {model_name} FAILED: {e}")
            continue
        rev_suffix = f"_{model_revision}" if model_revision else ""
        safe_rev = rev_suffix.replace("/", "_").replace(":", "_")
        run_name = f"{run_name_prefix}_{model_name.replace('/', '_')}{safe_rev}_{data}"
        _log_one_run(payload, run_name=run_name, dataset_repo_tag=dataset_repo_tag)
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{model_name.replace('/', '_')}_results.json").write_text(
                __import__("json").dumps(payload, indent=2)
            )
        print(f"[geometry-modal] logged MLflow run for {model_name}")

    print("[geometry-modal] all done.")


@app.local_entrypoint()
def heldout(
    model: str = "allenai/Olmo-3-1125-32B",
    k: int = 60,
    runs: int = 3,
    seed: int = 42,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    output: str = "results/geometry/heldout/olmo_3_32b/results.json",
) -> None:
    import json

    payload = heldout_analyze_model.remote(
        model,
        k=k,
        runs=runs,
        seed=seed,
        dtype=dtype,
        device_map=device_map or None,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"[heldout] wrote {path}")


@app.local_entrypoint()
def per_seed(
    model: str = "allenai/Olmo-3-1125-32B",
    model_revision: str = "",
    trust_remote_code: bool = False,
    k: int = 30,
    runs: int = 3,
    seed: int = 42,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    tokenizer_use_fast: bool = True,
    prepend_bos: bool = False,
    output: str = "results/geometry/per_seed/olmo_3_32b/results.json",
) -> None:
    import json

    payload = per_seed_analyze_model.remote(
        model,
        model_revision=model_revision or None,
        trust_remote_code=trust_remote_code,
        k=k,
        runs=runs,
        seed=seed,
        dtype=dtype,
        device_map=device_map or None,
        tokenizer_use_fast=tokenizer_use_fast,
        prepend_bos=prepend_bos,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"[per-seed] wrote {path}")


@app.local_entrypoint()
def frozen_layer(
    model: str = "allenai/Olmo-3-1125-32B",
    target_layer: int = 26,
    model_revision: str = "",
    trust_remote_code: bool = False,
    k: int = 30,
    runs: int = 3,
    seed: int = 45,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    tokenizer_use_fast: bool = True,
    prepend_bos: bool = False,
    output: str = "results/geometry/robust_joint/olmo_3_32b/evaluation_seeds_45_47.json",
) -> None:
    import json

    payload = frozen_layer_analyze_model.remote(
        model,
        target_layer,
        model_revision=model_revision or None,
        trust_remote_code=trust_remote_code,
        k=k,
        runs=runs,
        seed=seed,
        dtype=dtype,
        device_map=device_map or None,
        tokenizer_use_fast=tokenizer_use_fast,
        prepend_bos=prepend_bos,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"[frozen-layer] wrote {path}")


# --------------------------------------------------------------------------- #
# Stage 2.5: regime-stratified probe
# --------------------------------------------------------------------------- #


@app.function(**_gpu_function_kwargs(DEFAULT_GPU, timeout=2 * 60 * 60))
def regime_probe_model(
    model_name: str,
    target_layer: int,
    targets_per_regime: dict[str, list[int]],
    *,
    num_examples: int = 3,
    context: str = "random",
    n_min: int = 1,
    n_max: int = 1000,
    seed: int = 42,
    dtype: str = "float16",
    device_map: str | None = None,
) -> dict:
    import os

    from src.geometry import RegimeProbeConfig, run_regime_probe

    cfg = RegimeProbeConfig(
        model_name=model_name,
        target_layer=target_layer,
        n_per_regime=max(len(v) for v in targets_per_regime.values()),
        num_examples=num_examples,
        context=context,
        n_min=n_min,
        n_max=n_max,
        seed=seed,
        dtype=dtype,
        device="cuda",
        device_map=device_map,
    )
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def _on_progress(done: int, total: int) -> None:
        print(f"[worker:{model_name}] forward {done}/{total}")

    print(f"[worker] regime_probe_model start: {model_name} target_layer={target_layer} "
          f"counts={ {k: len(v) for k, v in targets_per_regime.items()} }")
    results = run_regime_probe(
        cfg,
        targets_per_regime=targets_per_regime,
        hf_token=hf_token,
        progress_callback=_on_progress,
    )
    payload = results.to_dict()
    print(f"[worker] regime_probe_model done: {model_name}")
    return payload


def _compute_regimes_local(
    counts: dict[int, int],
    *,
    n_min: int = 1,
    n_max: int = 1000,
    gauss_mu: float | None = None,
    gauss_sigma: float | None = None,
    exp_rate: float | None = None,
) -> dict[int, int]:
    """Per-N argmin among 4 log-ratio residuals:
        0 = uniform        T / n_total
        1 = Gaussian       T * N(mu, sigma) / Z
        2 = Zipf           T * (1/rank) / Z
        3 = exponential    T * exp(-lambda * N) / Z

    `gauss_mu` / `gauss_sigma` default to (n_min+n_max)/2 and (n_max-n_min)/4.
    `exp_rate` defaults to the count-weighted MLE of the exponential rate,
    `lambda_hat = sum(counts) / sum(N * counts)`, i.e. the rate that best fits
    the empirical data under the exponential model.
    """
    import numpy as np

    if gauss_mu is None:
        gauss_mu = (n_min + n_max) / 2.0
    if gauss_sigma is None:
        gauss_sigma = (n_max - n_min) / 4.0

    xs = np.array([n for n in sorted(counts) if n_min <= n <= n_max])
    ys = np.array([counts[int(n)] for n in xs], dtype=float)
    n_total = len(xs)
    T = ys.sum()

    uniform = np.full(n_total, T / n_total)
    g = np.exp(-((xs - gauss_mu) ** 2) / (2 * gauss_sigma ** 2))
    gaussian = T * g / g.sum()
    ranks = np.argsort(np.argsort(xs)) + 1
    zipf = T * (1.0 / ranks) / np.sum(1.0 / ranks)

    if exp_rate is None:
        denom = float((xs * ys).sum())
        exp_rate = float(ys.sum() / denom) if denom > 0 else 1.0 / max(1.0, xs.mean())
    e = np.exp(-exp_rate * xs)
    exponential = T * e / e.sum()

    safe_ys = np.where(ys > 0, ys, 1e-30)
    lr = np.abs(np.stack([
        np.log(safe_ys / uniform),
        np.log(safe_ys / gaussian),
        np.log(safe_ys / zipf),
        np.log(safe_ys / exponential),
    ], axis=0))
    winners = np.argmin(lr, axis=0)
    return {int(n): int(w) for n, w in zip(xs, winners)}


def _pull_pile_counts_local(stage1_run_id: str, *, n_max: int = 1000) -> dict[int, int]:
    import os

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(
        os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{os.path.abspath('mlflow.db')}",
        )
    )
    client = MlflowClient()
    run = client.get_run(stage1_run_id)
    metrics = run.data.metrics
    pad = max(4, len(str(n_max)))
    out: dict[int, int] = {}
    for n in range(n_max + 1):
        key = f"count_N_{n:0{pad}d}"
        if key in metrics:
            out[n] = int(metrics[key])
    if not out:
        raise SystemExit(
            f"Stage 1 run {stage1_run_id} has no count_N_NNNN metrics for "
            f"n_max={n_max}"
        )
    return out


def _log_regime_run(
    payload: dict, run_name: str, dataset_repo_tag: str,
    target_layer: int, stage1_run_id: str,
) -> None:
    import json as _json

    import mlflow
    import numpy as np

    from src.geometry import (
        RegimeProbeConfig,
        RegimePerClassResults,
        RegimeProbeResults,
        metrics_for_mlflow_regime,
    )
    from utils.mlflow_utils import (
        log_metric,
        log_metrics,
        log_params,
        start_run,
    )

    cfg_d = payload["config"]
    cfg = RegimeProbeConfig(
        model_name=cfg_d["model_name"],
        target_layer=cfg_d["target_layer"],
        n_per_regime=cfg_d["n_per_regime"],
        num_examples=cfg_d["num_examples"],
        context=cfg_d["context"],
        n_min=cfg_d["n_min"],
        n_max=cfg_d["n_max"],
        seed=cfg_d["seed"],
        dtype=cfg_d["dtype"],
        device=cfg_d.get("device", "cuda"),
        device_map=cfg_d.get("device_map"),
    )
    per_regime = {
        k: RegimePerClassResults(
            targets=v["targets"],
            pc1_scores=v["pc1_scores"],
            pc1_shared_scores=v["pc1_shared_scores"],
            pca_explained_variance=v["pca_explained_variance"],
            r2_linear=v["r2_linear"],
            r2_log=v["r2_log"],
            monotonicity=v["monotonicity"],
            n_samples=v["n_samples"],
        )
        for k, v in payload["per_regime"].items()
    }
    results = RegimeProbeResults(
        config=cfg,
        per_regime=per_regime,
        shared_pca_explained_variance=payload["shared_pca_explained_variance"],
    )

    tags = {
        "stage": "number_geometry_regime",
        "backend": "modal",
        "dataset": dataset_repo_tag,
        "model_name": cfg.model_name,
        "context": cfg.context,
        "target_layer": str(cfg.target_layer),
        "stage1_run_id": stage1_run_id,
    }

    with start_run(run_name=run_name, tags=tags):
        log_params(
            {
                "model_name": cfg.model_name,
                "target_layer": cfg.target_layer,
                "n_per_regime": cfg.n_per_regime,
                "num_examples": cfg.num_examples,
                "context": cfg.context,
                "n_min": cfg.n_min,
                "n_max": cfg.n_max,
                "seed": cfg.seed,
                "dtype": cfg.dtype,
                "device_map": cfg.device_map or "single",
                "stage1_run_id": stage1_run_id,
            }
        )
        flat = metrics_for_mlflow_regime(results)
        log_metrics({k: float(v) for k, v in flat.items()})
        mlflow.log_text(_json.dumps(payload), "geometry/regime_probe.json")


REGIME_NAMES_BY_INT = {0: "uniform", 1: "gaussian", 2: "zipf", 3: "exponential"}


@app.local_entrypoint()
def regime(
    model_name: str = "EleutherAI/pythia-2.8b",
    target_layer: int = 9,
    stage1_run_id: str = "697d041ba14b4a5fb7a340978e3b7a03",
    n_per_regime: int = 60,
    num_examples: int = 3,
    context: str = "random",
    n_min: int = 1,
    n_max: int = 1000,
    gauss_mu: float = -1.0,
    gauss_sigma: float = -1.0,
    exp_rate: float = -1.0,
    seed: int = 42,
    dtype: str = "float16",
    experiment_name: str = "numberline_geometry_expv1",
    run_name_prefix: str = "modal_regime_probe",
    dataset_repo_tag: str = "monology/pile-uncopyrighted",
    dry_run: bool = False,
) -> None:
    import random as _random
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from utils.mlflow_utils import setup_tracking

    mu_arg = None if gauss_mu < 0 else float(gauss_mu)
    sigma_arg = None if gauss_sigma < 0 else float(gauss_sigma)
    exp_arg = None if exp_rate < 0 else float(exp_rate)

    print(f"[regime-modal] resolving Stage 1 run {stage1_run_id} (n_max={n_max}) ...")
    counts = _pull_pile_counts_local(stage1_run_id, n_max=n_max)
    regimes = _compute_regimes_local(
        counts, n_min=n_min, n_max=n_max,
        gauss_mu=mu_arg, gauss_sigma=sigma_arg, exp_rate=exp_arg,
    )

    by_class: dict[str, list[int]] = {
        "uniform": [], "gaussian": [], "zipf": [], "exponential": [],
    }
    for n, w in regimes.items():
        by_class[REGIME_NAMES_BY_INT[w]].append(n)
    pop = {k: len(v) for k, v in by_class.items()}
    print(f"[regime-modal] regime population in [{n_min},{n_max}]: {pop}")

    rng = _random.Random(seed)
    targets_per_regime: dict[str, list[int]] = {}
    for k, ns in by_class.items():
        if not ns:
            print(f"[regime-modal] WARNING: regime {k!r} is empty; skipping")
            continue
        rng.shuffle(ns)
        targets_per_regime[k] = ns[:n_per_regime]
    sample_sizes = {k: len(v) for k, v in targets_per_regime.items()}
    print(f"[regime-modal] sampled targets per regime: {sample_sizes}")

    if dry_run:
        print("[regime-modal] --dry-run; exiting before any cloud work")
        return

    setup_tracking(None, experiment_name)

    print(f"[regime-modal] spawning Modal worker on {model_name} layer {target_layer} ...")
    fut = regime_probe_model.spawn(
        model_name=model_name,
        target_layer=target_layer,
        targets_per_regime=targets_per_regime,
        num_examples=num_examples,
        context=context,
        n_min=n_min,
        n_max=n_max,
        seed=seed,
        dtype=dtype,
    )
    payload = fut.get()
    run_name = f"{run_name_prefix}_{model_name.replace('/', '_')}_L{target_layer}"
    _log_regime_run(
        payload,
        run_name=run_name,
        dataset_repo_tag=dataset_repo_tag,
        target_layer=target_layer,
        stage1_run_id=stage1_run_id,
    )
    print(f"[regime-modal] logged MLflow run; per-regime r2_linear / r2_log:")
    for name, blob in payload["per_regime"].items():
        print(
            f"  {name:<10}  n={blob['n_samples']:>3}  "
            f"r2_linear={blob['r2_linear']:+.3f}  r2_log={blob['r2_log']:+.3f}  "
            f"rho={blob['monotonicity']:+.3f}"
        )


def _log_synth_run(
    payload: dict, run_name: str,
    distributions: list[str], target_layer: int,
) -> None:
    """Log a synthetic-distribution probe run with its own stage tag.

    Reuses the regime payload schema since `regime_probe_model` doesn't care
    where the targets came from -- it just runs forward, fits PCA per class,
    and reports R^2 / rho. Tagging it `number_geometry_synth` keeps the
    figure pipeline able to distinguish corpus-regime vs synthetic runs.
    """
    import json as _json

    import mlflow
    import numpy as np

    from src.geometry import (
        RegimePerClassResults,
        RegimeProbeConfig,
        RegimeProbeResults,
        metrics_for_mlflow_regime,
    )
    from utils.mlflow_utils import (
        log_metrics,
        log_params,
        start_run,
    )

    cfg_d = payload["config"]
    cfg = RegimeProbeConfig(
        model_name=cfg_d["model_name"],
        target_layer=cfg_d["target_layer"],
        n_per_regime=cfg_d["n_per_regime"],
        num_examples=cfg_d["num_examples"],
        context=cfg_d["context"],
        n_min=cfg_d["n_min"],
        n_max=cfg_d["n_max"],
        seed=cfg_d["seed"],
        dtype=cfg_d["dtype"],
        device=cfg_d.get("device", "cuda"),
        device_map=cfg_d.get("device_map"),
    )
    per_regime = {
        k: RegimePerClassResults(
            targets=v["targets"],
            pc1_scores=v["pc1_scores"],
            pc1_shared_scores=v["pc1_shared_scores"],
            pca_explained_variance=v["pca_explained_variance"],
            r2_linear=v["r2_linear"],
            r2_log=v["r2_log"],
            monotonicity=v["monotonicity"],
            n_samples=v["n_samples"],
        )
        for k, v in payload["per_regime"].items()
    }
    results = RegimeProbeResults(
        config=cfg,
        per_regime=per_regime,
        shared_pca_explained_variance=payload["shared_pca_explained_variance"],
    )

    tags = {
        "stage": "number_geometry_synth",
        "backend": "modal",
        "model_name": cfg.model_name,
        "target_layer": str(cfg.target_layer),
        "distributions": ",".join(distributions),
    }

    with start_run(run_name=run_name, tags=tags):
        log_params(
            {
                "model_name": cfg.model_name,
                "target_layer": cfg.target_layer,
                "n_per_regime": cfg.n_per_regime,
                "num_examples": cfg.num_examples,
                "context": cfg.context,
                "n_min": cfg.n_min,
                "n_max": cfg.n_max,
                "seed": cfg.seed,
                "dtype": cfg.dtype,
                "distributions": ",".join(distributions),
            }
        )
        flat = metrics_for_mlflow_regime(results)
        log_metrics({k: float(v) for k, v in flat.items()})
        mlflow.log_text(_json.dumps(payload), "geometry/synth_probe.json")


@app.local_entrypoint()
def synth(
    model_name: str = "EleutherAI/pythia-2.8b",
    target_layer: int = 9,
    distributions: str = "uniform,gaussian,zipf,exponential_growth",
    n_per_distribution: int = 80,
    num_examples: int = 3,
    context: str = "random",
    n_min: int = 1,
    n_max: int = 5000,
    seed: int = 42,
    dtype: str = "float16",
    experiment_name: str = "numberline_geometry_expv1",
    run_name_prefix: str = "modal_synth_distributions",
    dry_run: bool = False,
) -> None:
    """Synthetic-distribution probe: instead of taking integers from Stage 1's
    regime classification, draw integers from synthetic priors over [n_min,
    n_max] (uniform / gaussian / zipf / exponential_decay / exponential_growth)
    and run the same per-class PCA + linear/log-fit pipeline.

    This is the *control* for the corpus-regime probe: PC1 should remain a
    magnitude axis regardless of which distribution we use to draw probe
    integers, because it is a property of the model's residual stream, not
    of the prompt sampling distribution.
    """
    import random as _random
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from utils.mlflow_utils import setup_tracking
    from utils.prompts import sample_integers

    dist_list = [d.strip() for d in distributions.split(",") if d.strip()]
    if not dist_list:
        raise SystemExit("--distributions must contain at least one name")

    rng = _random.Random(seed)
    targets_per_regime: dict[str, list[int]] = {}
    for d in dist_list:
        try:
            targets_per_regime[d] = sample_integers(
                d, n_per_distribution, lo=n_min, hi=n_max, rng=rng,
            )
        except ValueError as e:
            raise SystemExit(f"distribution {d!r} not supported: {e}") from e

    print(
        f"[synth-modal] {len(dist_list)} distributions x "
        f"{n_per_distribution} samples each over [{n_min}, {n_max}]"
    )
    for d, ts in targets_per_regime.items():
        sorted_ts = sorted(ts)
        print(
            f"  {d:<22}  median={sorted_ts[len(sorted_ts) // 2]:>5}  "
            f"min={min(ts):>5}  max={max(ts):>5}"
        )

    if dry_run:
        print("[synth-modal] --dry-run; exiting before any cloud work")
        return

    setup_tracking(None, experiment_name)

    print(f"[synth-modal] spawning Modal worker on {model_name} layer {target_layer} ...")
    fut = regime_probe_model.spawn(
        model_name=model_name,
        target_layer=target_layer,
        targets_per_regime=targets_per_regime,
        num_examples=num_examples,
        context=context,
        n_min=n_min,
        n_max=n_max,
        seed=seed,
        dtype=dtype,
    )
    payload = fut.get()
    run_name = f"{run_name_prefix}_{model_name.replace('/', '_')}_L{target_layer}"
    _log_synth_run(
        payload, run_name=run_name,
        distributions=dist_list, target_layer=target_layer,
    )
    print(f"[synth-modal] logged MLflow run; per-distribution r2_linear / r2_log:")
    for name, blob in payload["per_regime"].items():
        print(
            f"  {name:<22}  n={blob['n_samples']:>3}  "
            f"r2_linear={blob['r2_linear']:+.3f}  r2_log={blob['r2_log']:+.3f}  "
            f"rho={blob['monotonicity']:+.3f}"
        )
