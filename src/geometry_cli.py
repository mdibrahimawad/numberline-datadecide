from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow

from src.geometry import GeometryConfig, metrics_for_mlflow, run_geometry
from utils.mlflow_utils import (
    log_metric,
    log_metrics,
    log_params,
    setup_tracking,
    start_run,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Probe number-line geometry of a HF causal LM: collect last-token "
            "hidden states across all layers, project via PCA + PLS, and log "
            "explained variance, monotonicity (rho) and compression rate "
            "(beta) per layer to MLflow."
        )
    )
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-revision", default=None,
                   help="optional Hugging Face revision/branch/tag/commit")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="allow model/tokenizer custom code from the HF repo")
    p.add_argument("--data", choices=["numerics", "symbols", "years"], default="numerics")
    p.add_argument("--groups", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="exponents i s.t. group i is centered on 10**i")
    p.add_argument("--k", type=int, default=30, help="examples per group")
    p.add_argument("--num-examples", type=int, default=3,
                   help="in-context shots before the target slot")
    p.add_argument("--context", choices=["random", "fixed", "same"], default="random")
    p.add_argument("--transform-dim", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--upper-bound", type=int, default=None)
    p.add_argument("--target-min", type=int, default=None,
                   help="minimum target value for data modes with explicit ranges, e.g. years")
    p.add_argument("--target-max", type=int, default=None,
                   help="maximum target value for data modes with explicit ranges, e.g. years")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None,
                   help="explicit torch device (cuda, cuda:0, mps, cpu); auto if unset")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--device-map", type=str, default=None,
                   help="passed to from_pretrained (e.g. 'auto' for accelerate)")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace token (also reads $HF_TOKEN if set)")
    p.add_argument("--save-projections", action="store_true",
                   help="capture per-layer PC1 projections from the first run "
                        "(needed for the PC1-vs-target scatter figure)")
    p.add_argument(
        "--tokenizer-use-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="select the Hugging Face fast or slow tokenizer implementation",
    )
    p.add_argument(
        "--prepend-bos",
        action="store_true",
        help="tokenize without implicit special tokens and prepend exactly one BOS token",
    )
    p.add_argument(
        "--spacing-fit",
        choices=["log", "direct"],
        default="log",
        help="legacy log-gap regression or paper-matched direct geometric fit",
    )
    p.add_argument(
        "--filter-correct",
        action="store_true",
        help="numerics only: keep prompts whose greedy completion equals the target, "
             "resampling rejected targets from the same group",
    )
    p.add_argument("--filter-max-new-tokens", type=int, default=8)
    p.add_argument("--filter-max-candidates", type=int, default=100,
                   help="candidates tried per prompt slot before the group is marked failed")

    p.add_argument("--experiment-name", default="numberline_geometry_expv1")
    p.add_argument("--run-name", default="local_geometry")
    p.add_argument("--tracking-uri", default=None)

    p.add_argument("--results-dir", default="results/geometry")
    args = p.parse_args(argv)

    if args.k <= 0:
        p.error(f"--k must be positive, got {args.k}")
    if args.runs <= 0:
        p.error(f"--runs must be positive, got {args.runs}")
    if args.transform_dim <= 0:
        p.error(f"--transform-dim must be positive, got {args.transform_dim}")
    if args.filter_max_new_tokens <= 0:
        p.error(f"--filter-max-new-tokens must be positive, got {args.filter_max_new_tokens}")
    if args.filter_max_candidates <= 0:
        p.error(f"--filter-max-candidates must be positive, got {args.filter_max_candidates}")
    return args


def _slugify(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cfg = GeometryConfig(
        model_name=args.model_name,
        model_revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        data=args.data,
        groups=tuple(sorted(args.groups)),
        k=args.k,
        num_examples=args.num_examples,
        context=args.context,
        transform_dim=args.transform_dim,
        runs=args.runs,
        upper_bound=args.upper_bound,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        save_projections=args.save_projections,
        target_min=args.target_min,
        target_max=args.target_max,
        tokenizer_use_fast=args.tokenizer_use_fast,
        prepend_bos=args.prepend_bos,
        spacing_fit=args.spacing_fit,
        filter_correct=args.filter_correct,
        filter_max_new_tokens=args.filter_max_new_tokens,
        filter_max_candidates=args.filter_max_candidates,
    )

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    setup_tracking(args.tracking_uri, args.experiment_name)

    tags = {
        "stage": "number_geometry",
        "backend": "local",
        "model_name": cfg.model_name,
        "model_revision": cfg.model_revision or "",
        "data": cfg.data,
        "context": cfg.context,
    }

    print(
        f"[geometry-cli] model={cfg.model_name} data={cfg.data} "
        f"groups={list(cfg.groups)} k={cfg.k} runs={cfg.runs} "
        f"context={cfg.context}"
    )

    with start_run(run_name=args.run_name, tags=tags):
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

        def _on_run(done: int, total: int) -> None:
            print(f"[geometry-cli] run {done}/{total} done")

        start = time.time()
        results = run_geometry(cfg, hf_token=hf_token, progress_callback=_on_run)
        elapsed = time.time() - start

        log_metric("elapsed_seconds", elapsed)
        log_metric("n_layers", float(results.n_layers))

        flat = metrics_for_mlflow(results)
        batch: dict[str, float] = {}
        for k, v in flat.items():
            batch[k] = float(v)
            if len(batch) >= 500:
                log_metrics(batch)
                batch = {}
        if batch:
            log_metrics(batch)

        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_json = results_dir / f"{_slugify(cfg.model_name)}_{cfg.data}.json"
        with open(out_json, "w") as fh:
            json.dump(results.to_dict(), fh, indent=2)
        mlflow.log_artifact(str(out_json), artifact_path="geometry")
        print(f"[geometry-cli] wrote {out_json}")

        if results.best_layer_pca is not None:
            best = results.pca[results.best_layer_pca]
            print(
                f"[geometry-cli] DONE elapsed={elapsed:.1f}s "
                f"PCA: best_layer={results.best_layer_pca} "
                f"EV={best.ev_mean:.3f} rho={best.rho_mean:.3f} "
                f"beta={best.beta_mean:.3f}"
            )
        else:
            print("[geometry-cli] DONE -- PCA produced no layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
