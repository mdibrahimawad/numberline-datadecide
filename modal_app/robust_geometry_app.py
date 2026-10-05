"""Modal sweep for target-layer Principal Component Pursuit robustness checks."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GPU = os.environ.get("MODAL_GEOMETRY_GPU", "H100")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libopenblas-dev")
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
    )
    .add_local_python_source("src", "utils")
)

app = modal.App("numberline-robust-geometry", image=image)
secret_name = os.environ.get("MODAL_HF_SECRET_NAME", "numberline-hf-token").strip()
secrets = [modal.Secret.from_name(secret_name)] if secret_name else []
rpca_volume = modal.Volume.from_name("numberline-robust-pca", create_if_missing=True)
RPCA_CACHE = Path("/rpca")


@app.function(
    gpu=DEFAULT_GPU,
    cpu=16,
    memory=32768,
    timeout=12 * 60 * 60,
    secrets=secrets,
)
def robust_analyze_model(spec: dict, runs: int, rpca_max_iter: int) -> dict:
    from src.geometry import GeometryConfig
    from src.robust_geometry import run_robust_geometry

    cfg = GeometryConfig(
        model_name=spec["model"],
        model_revision=spec.get("revision"),
        trust_remote_code=spec.get("trust_remote_code", False),
        data="numerics",
        groups=(1, 2, 3, 4),
        k=30,
        num_examples=3,
        context="random",
        runs=runs,
        upper_bound=10000,
        seed=42,
        device="cuda",
        dtype=spec.get("dtype", "bfloat16"),
        device_map="auto",
        tokenizer_use_fast=spec.get("tokenizer_use_fast", True),
        prepend_bos=spec.get("prepend_bos", False),
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def progress(done: int, total: int) -> None:
        print(f"[{spec['model']}] robust run {done}/{total}")

    return run_robust_geometry(
        cfg,
        int(spec["target_layer"]),
        hf_token=token,
        rpca_max_iter=rpca_max_iter,
        progress_callback=progress,
    )


@app.function(
    gpu=DEFAULT_GPU,
    cpu=8,
    memory=32768,
    timeout=4 * 60 * 60,
    secrets=secrets,
    volumes={str(RPCA_CACHE): rpca_volume},
)
def collect_all_layers(spec: dict, seed: int = 42) -> dict:
    import random

    import numpy as np

    from src.geometry import (
        GeometryConfig,
        _load_model,
        _resolve_device,
        _seed_all,
        collect_hidden_states,
    )
    from src.robust_geometry import _projection_metrics

    cfg = GeometryConfig(
        model_name=spec["model"],
        model_revision=spec.get("revision"),
        trust_remote_code=spec.get("trust_remote_code", False),
        data="numerics",
        groups=(1, 2, 3, 4),
        k=30,
        num_examples=3,
        context="random",
        runs=1,
        upper_bound=10000,
        seed=seed,
        device="cuda",
        dtype=spec.get("dtype", "bfloat16"),
        device_map="auto",
        tokenizer_use_fast=spec.get("tokenizer_use_fast", True),
        prepend_bos=spec.get("prepend_bos", False),
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    _seed_all(seed)
    device = _resolve_device(cfg.device)
    model, tokenizer = _load_model(cfg, device, token)
    bundle = collect_hidden_states(cfg, model, tokenizer, device, random.Random(seed))

    job_id = f"{spec['model'].replace('/', '_')}_seed{seed}"
    job_dir = RPCA_CACHE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    ordinary: dict[str, dict] = {}

    # Layer 0 is the constant final prompt-token embedding, so its centered
    # matrix is exactly zero and Principal Component Pursuit is undefined.
    for layer in range(1, bundle.n_layers):
        features: list[np.ndarray] = []
        answers: list[float] = []
        group_ids: list[int] = []
        group_indices: dict[int, list[int]] = {}
        for group in sorted(bundle.states[layer]):
            start = len(features)
            reps = bundle.states[layer][group]
            features.extend(np.asarray(rep).reshape(-1) for rep in reps)
            answers.extend(bundle.answers[layer][group])
            group_ids.extend([int(group)] * len(reps))
            group_indices[int(group)] = list(range(start, len(features)))
        X = np.stack(features)
        y = np.asarray(answers)
        ordinary[str(layer)] = _projection_metrics(X, y, group_indices)
        path = job_dir / f"layer_{layer:03d}.npz"
        np.savez(path, X=X, answers=y, group_ids=np.asarray(group_ids))
        files.append(str(path))
    rpca_volume.commit()
    return {
        "job_id": job_id,
        "model": spec["model"],
        "seed": seed,
        "n_layers": bundle.n_layers,
        "files": files,
        "ordinary_pca": ordinary,
    }


@app.function(
    cpu=16,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={str(RPCA_CACHE): rpca_volume},
)
def robust_analyze_layer(path: str, rpca_max_iter: int = 2000) -> dict:
    import numpy as np

    from src.robust_geometry import RobustPCA, _projection_metrics

    data = np.load(path)
    X = data["X"]
    answers = data["answers"]
    group_ids = data["group_ids"]
    group_indices = {
        int(group): np.flatnonzero(group_ids == group).tolist()
        for group in np.unique(group_ids)
    }
    centered = X.astype(np.float64, copy=False) - X.mean(axis=0, dtype=np.float64)
    rpca = RobustPCA(centered)
    low_rank, sparse = rpca.fit(
        max_iter=rpca_max_iter,
        iter_print=rpca_max_iter + 1,
    )
    metrics = _projection_metrics(low_rank, answers, group_indices)
    residual = centered - low_rank - sparse
    norm = max(np.linalg.norm(centered, ord="fro"), np.finfo(float).eps)
    metrics.update(
        {
            "layer": int(Path(path).stem.split("_")[-1]),
            "iterations": rpca.n_iter,
            "converged": bool(rpca.error <= rpca.tol),
            "relative_residual": float(np.linalg.norm(residual, ord="fro") / norm),
            "sparse_energy_fraction": float(
                np.linalg.norm(sparse, ord="fro") ** 2 / norm ** 2
            ),
        }
    )
    return metrics


def _summary_row(payload: dict) -> dict:
    cfg = payload["config"]
    pca = payload["ordinary_pca"]
    robust = payload["robust_pca"]
    decomp = payload["decomposition"]
    return {
        "model": cfg["model_name"],
        "revision": cfg.get("model_revision") or "main",
        "target_layer": payload["target_layer"],
        "runs": cfg["runs"],
        "pca_rho_mean": pca["rho_mean"],
        "pca_beta_mean": pca["beta_mean"],
        "pca_ev_mean": pca["explained_variance_mean"],
        "pca_r2_beta_mean": pca["r2_beta_mean"],
        "rpca_rho_mean": robust["rho_mean"],
        "rpca_rho_std": robust["rho_std"],
        "rpca_beta_mean": robust["beta_mean"],
        "rpca_beta_std": robust["beta_std"],
        "rpca_ev_mean": robust["explained_variance_mean"],
        "rpca_ev_std": robust["explained_variance_std"],
        "rpca_r2_beta_mean": robust["r2_beta_mean"],
        "rpca_r2_beta_std": robust["r2_beta_std"],
        "rpca_converged_fraction": decomp["converged_mean"],
        "rpca_iterations_mean": decomp["iterations_mean"],
        "rpca_relative_residual_mean": decomp["relative_residual_mean"],
        "rpca_sparse_fraction_mean": decomp["sparse_fraction_mean"],
        "rpca_sparse_energy_fraction_mean": decomp["sparse_energy_fraction_mean"],
        "rpca_low_rank_rank_mean": decomp["low_rank_rank_mean"],
    }


def _log_mlflow(payload: dict, experiment_name: str) -> None:
    try:
        import mlflow
    except ModuleNotFoundError:
        print("[robust-pca] mlflow unavailable locally; keeping JSON/CSV outputs only")
        return

    from utils.mlflow_utils import log_metrics, log_params, setup_tracking, start_run

    setup_tracking(None, experiment_name)
    row = _summary_row(payload)
    tags = {
        "stage": "robust_number_geometry",
        "backend": "modal",
        "model_name": row["model"],
        "method": "dganguli_robust_pca",
    }
    run_name = f"robust_pca_{row['model'].replace('/', '_')}_L{row['target_layer']}"
    with start_run(run_name=run_name, tags=tags):
        log_params({k: v for k, v in row.items() if k in {"model", "revision", "target_layer", "runs"}})
        log_metrics({k: float(v) for k, v in row.items() if k not in {"model", "revision", "target_layer", "runs"}})
        mlflow.log_text(json.dumps(payload, indent=2), "robust_geometry/results.json")


@app.local_entrypoint()
def main(
    specs: str,
    output_dir: str = "results/geometry/robust_pca",
    runs: int = 3,
    rpca_max_iter: int = 1000,
    experiment_name: str = "numberline_geometry_expv1",
    dry_run: bool = False,
) -> None:
    spec_path = Path(specs)
    model_specs = json.loads(spec_path.read_text())
    print(f"[robust-pca] {len(model_specs)} models on {DEFAULT_GPU}")
    for spec in model_specs:
        print(f"  {spec['model']} layer={spec['target_layer']}")
    if dry_run:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    pending: list[tuple[dict, object, Path]] = []
    for spec in model_specs:
        slug = spec["model"].replace("/", "_").replace(".", "_")
        result_path = out / f"{slug}.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text())
            rows.append(_summary_row(payload))
            print(f"[robust-pca] reusing {result_path}")
            continue
        pending.append(
            (spec, robust_analyze_model.spawn(spec, runs, rpca_max_iter), result_path)
        )

    for spec, future, result_path in pending:
        try:
            payload = future.get()
        except Exception as exc:
            print(f"[robust-pca] FAILED {spec['model']}: {exc}")
            continue
        result_path.write_text(json.dumps(payload, indent=2))
        rows.append(_summary_row(payload))
        _log_mlflow(payload, experiment_name)
        print(f"[robust-pca] completed {spec['model']}")

    if rows:
        summary_path = out / "robust_pca_summary.csv"
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[robust-pca] wrote {summary_path}")


def _finish_all_layers(collected: dict, rpca_max_iter: int, output_dir: Path) -> None:
    futures = [robust_analyze_layer.spawn(path, rpca_max_iter) for path in collected["files"]]
    robust_layers: dict[str, dict] = {}
    for path, future in zip(collected["files"], futures):
        layer = int(Path(path).stem.split("_")[-1])
        try:
            result = future.get()
        except Exception as exc:
            print(f"[robust-pca] layer {layer} FAILED: {exc}")
            continue
        robust_layers[str(layer)] = result
        print(
            f"[robust-pca] layer {layer:02d}: EV={result['explained_variance']:.4f} "
            f"rho={result['rho']:.4f} beta={result['beta']:.4f} "
            f"converged={result['converged']}"
        )

    ordinary = collected["ordinary_pca"]
    ordinary_best = max(ordinary, key=lambda key: ordinary[key]["explained_variance"])
    converged = {k: v for k, v in robust_layers.items() if v["converged"]}
    robust_best = max(converged, key=lambda key: converged[key]["explained_variance"])
    payload = {
        "method": "dganguli/robust-pca Principal Component Pursuit (ADMM)",
        "model": collected["model"],
        "seed": collected["seed"],
        "n_layers": collected["n_layers"],
        "ordinary_best_layer": int(ordinary_best),
        "robust_best_layer": int(robust_best),
        "ordinary_pca": ordinary,
        "robust_pca": robust_layers,
    }
    out = output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "all_layer_results.json").write_text(json.dumps(payload, indent=2))
    with (out / "all_layer_summary.csv").open("w", newline="") as handle:
        fields = [
            "layer", "pca_ev", "pca_rho", "pca_beta", "rpca_ev", "rpca_rho",
            "rpca_beta", "rpca_r2_beta", "iterations", "converged",
            "relative_residual", "sparse_energy_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer in sorted(robust_layers, key=int):
            pca = ordinary[layer]
            robust = robust_layers[layer]
            writer.writerow(
                {
                    "layer": layer,
                    "pca_ev": pca["explained_variance"],
                    "pca_rho": pca["rho"],
                    "pca_beta": pca["beta"],
                    "rpca_ev": robust["explained_variance"],
                    "rpca_rho": robust["rho"],
                    "rpca_beta": robust["beta"],
                    "rpca_r2_beta": robust["r2_beta"],
                    "iterations": robust["iterations"],
                    "converged": robust["converged"],
                    "relative_residual": robust["relative_residual"],
                    "sparse_energy_fraction": robust["sparse_energy_fraction"],
                }
            )
    print(
        f"[robust-pca] ordinary best layer={ordinary_best}; "
        f"robust best layer={robust_best}; wrote {out}"
    )


@app.local_entrypoint()
def all_layers(
    model: str = "allenai/Olmo-3-1125-32B",
    seed: int = 42,
    rpca_max_iter: int = 2000,
    output_dir: str = "results/geometry/robust_pca/olmo3_32b_all_layers",
    trust_remote_code: bool = False,
    dtype: str = "bfloat16",
) -> None:
    spec = {
        "model": model,
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
    }
    print(f"[robust-pca] collecting every layer for {model}, seed={seed}")
    collected = collect_all_layers.remote(spec, seed)
    _finish_all_layers(collected, rpca_max_iter, Path(output_dir))

