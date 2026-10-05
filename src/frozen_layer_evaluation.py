"""Evaluate a frozen numerical layer on prompt seeds unseen during selection."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict

import numpy as np

from src.geometry import (
    GeometryConfig,
    _load_model,
    _resolve_device,
    _seed_all,
    collect_hidden_states,
)
from src.robust_geometry import _collect_target_layer, _projection_metrics


def _collect_target_layer_filtered(cfg, target_layer, model, tokenizer, device, rng):
    """Like _collect_target_layer, but only on prompts accepted by filter_correct."""
    bundle = collect_hidden_states(cfg, model, tokenizer, device, rng)
    if not 0 <= target_layer < bundle.n_layers:
        raise ValueError(f"target layer {target_layer} outside [0, {bundle.n_layers - 1}]")
    features: list[np.ndarray] = []
    answers: list[float] = []
    group_indices: dict[int, list[int]] = {}
    for group in sorted(bundle.states[target_layer], key=int):
        start = len(features)
        features.extend(bundle.states[target_layer][group])
        answers.extend(bundle.answers[target_layer][group])
        group_indices[int(group)] = list(range(start, len(features)))
    return np.stack(features), np.asarray(answers), group_indices, bundle.filter_stats


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
        filter_stats = None
        if cfg.filter_correct:
            matrix, answers, group_indices, filter_stats = _collect_target_layer_filtered(
                cfg, target_layer, model, tokenizer, device, random.Random(seed)
            )
        else:
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
        log_metrics = _projection_metrics(
            matrix, answers, group_indices, spacing_fit="log"
        )
        row = {
            "run": run_index,
            "seed": seed,
            "target_hash": hashlib.sha256(
                json.dumps(answers.tolist()).encode()
            ).hexdigest(),
            **metrics,
            "beta_log": log_metrics["beta"],
            "r2_beta_log": log_metrics["r2_beta"],
        }
        if filter_stats is not None:
            row["filter_stats"] = filter_stats
        rows.append(row)
        if progress_callback:
            progress_callback(run_index + 1, cfg.runs)

    summary = {}
    for metric in ("explained_variance", "rho", "beta", "r2_beta", "beta_log"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std())
    return {
        "method": "frozen-layer evaluation on unseen prompt seeds",
        "target_layer": int(target_layer),
        "spacing_fit": "direct raw-gap least squares (beta, r2_beta); log-gap regression (beta_log)",
        "config": asdict(cfg),
        "runs": rows,
        "summary": summary,
    }
