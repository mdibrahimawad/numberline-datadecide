"""Evaluate a frozen numerical layer on prompt seeds unseen during selection."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict

import numpy as np

from src.geometry import GeometryConfig, _load_model, _resolve_device, _seed_all
from src.robust_geometry import _collect_target_layer, _projection_metrics


def run_frozen_layer_evaluation(
    cfg: GeometryConfig,
    target_layer: int,
    *,
    hf_token: str | None = None,
    progress_callback=None,
) -> dict:
    _seed_all(cfg.seed)
    device = _resolve_device(cfg.device)
    model, tokenizer = _load_model(cfg, device, hf_token)
    rows = []
    for run_index in range(max(1, cfg.runs)):
        seed = cfg.seed + run_index
        matrix, answers, group_indices = _collect_target_layer(
            cfg,
            target_layer,
            model,
            tokenizer,
            device,
            random.Random(seed),
        )
        metrics = _projection_metrics(
            matrix, answers, group_indices, spacing_fit="direct"
        )
        rows.append(
            {
                "run": run_index,
                "seed": seed,
                "target_hash": hashlib.sha256(
                    json.dumps(answers.tolist()).encode()
                ).hexdigest(),
                **metrics,
            }
        )
        if progress_callback:
            progress_callback(run_index + 1, cfg.runs)

    summary = {}
    for metric in ("explained_variance", "rho", "beta", "r2_beta"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std())
    return {
        "method": "frozen-layer evaluation on unseen prompt seeds",
        "target_layer": int(target_layer),
        "spacing_fit": "direct raw-gap least squares",
        "config": asdict(cfg),
        "runs": rows,
        "summary": summary,
    }
