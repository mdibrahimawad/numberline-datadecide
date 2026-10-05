"""Original-style PCA layer selection with explicit per-seed winners."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict

import numpy as np

from src.geometry import (
    GeometryConfig,
    HiddenStateBundle,
    _load_model,
    _monotonicity,
    _resolve_device,
    _seed_all,
    _transform_layer,
    collect_hidden_states,
)
from src.spacing_fit import fit_spacing_direct


def score_bundle(bundle: HiddenStateBundle) -> dict:
    layers = {}
    for layer in range(bundle.n_layers):
        transformed = _transform_layer(bundle, layer, "PCA", 1)
        if transformed is None:
            continue
        pc1 = transformed.components[:, 0]
        means = np.asarray([
            pc1[transformed.group_indices[group]].mean()
            for group in sorted(transformed.group_indices)
        ])
        beta, scale, _ = fit_spacing_direct(
            np.abs(np.diff(means)), compute_r2=False
        )
        layers[str(layer)] = {
            "ev": transformed.explained_variance,
            "rho": abs(_monotonicity(transformed.answers, pc1)),
            "beta": beta,
            "beta_scale": scale,
        }
    candidates = {
        layer: metrics for layer, metrics in layers.items()
        if np.isfinite(metrics["ev"])
    }
    if not candidates:
        raise RuntimeError("PCA produced no finite layers")
    best = max(candidates, key=lambda layer: candidates[layer]["ev"])
    return {"selected_layer": int(best), "selected_metrics": layers[best], "layers": layers}


def run_per_seed_geometry(
    cfg: GeometryConfig,
    *,
    hf_token: str | None = None,
    progress_callback=None,
) -> dict:
    _seed_all(cfg.seed)
    device = _resolve_device(cfg.device)
    model, tokenizer = _load_model(cfg, device, hf_token)
    runs = []
    for run_index in range(max(1, cfg.runs)):
        seed = cfg.seed + run_index
        bundle = collect_hidden_states(
            cfg, model, tokenizer, device, random.Random(seed)
        )
        result = score_bundle(bundle)
        result.update(
            {
                "run": run_index,
                "seed": seed,
                "target_hash": hashlib.sha256(
                    json.dumps(bundle.answers[0], sort_keys=True).encode()
                ).hexdigest(),
            }
        )
        runs.append(result)
        if progress_callback:
            progress_callback(run_index + 1, cfg.runs)

    layer_ids = sorted({layer for run in runs for layer in run["layers"]}, key=int)
    aggregate = {}
    for layer in layer_ids:
        aggregate[layer] = {}
        for metric in ("ev", "rho", "beta"):
            values = np.asarray([run["layers"][layer][metric] for run in runs], dtype=float)
            values = values[np.isfinite(values)]
            aggregate[layer][f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            aggregate[layer][f"{metric}_std"] = float(values.std()) if len(values) else float("nan")

    candidates = {
        layer: metrics for layer, metrics in aggregate.items()
        if np.isfinite(metrics["ev_mean"])
    }
    best = max(candidates, key=lambda layer: candidates[layer]["ev_mean"])
    return {
        "method": "original max-mean-EV selection with explicit reproducible seeds",
        "selection": "per seed: argmax EV; aggregate: argmax mean EV",
        "spacing_fit": "direct raw-gap least squares",
        "r2_computed": False,
        "config": asdict(cfg),
        "runs": runs,
        "aggregate_selected_layer": int(best),
        "aggregate_selected_metrics": aggregate[best],
        "aggregate_layers": aggregate,
    }


def joint_reselection(payload: dict) -> dict:
    """Reselect saved full-sample runs using geometric mean of EV and |rho|."""
    selected_runs = []
    for run in payload["runs"]:
        candidates = {
            layer: metrics for layer, metrics in run["layers"].items()
            if np.isfinite(metrics["ev"]) and np.isfinite(metrics["rho"])
        }
        best = max(
            candidates,
            key=lambda layer: np.sqrt(candidates[layer]["ev"] * candidates[layer]["rho"]),
        )
        metrics = candidates[best]
        selected_runs.append(
            {
                "seed": run["seed"],
                "target_hash": run["target_hash"],
                "selected_layer": int(best),
                "joint_score": float(np.sqrt(metrics["ev"] * metrics["rho"])),
                **metrics,
            }
        )

    aggregate = payload["aggregate_layers"]
    best = max(
        aggregate,
        key=lambda layer: np.sqrt(
            aggregate[layer]["ev_mean"] * aggregate[layer]["rho_mean"]
        ) if np.isfinite(aggregate[layer]["ev_mean"]) and np.isfinite(aggregate[layer]["rho_mean"])
        else -np.inf,
    )
    counts = Counter(run["selected_layer"] for run in selected_runs)
    return {
        "method": "full-sample geometric-mean EV-rho selection",
        "selection": "per seed: argmax sqrt(EV * abs(rho)); aggregate: argmax sqrt(mean_EV * mean_abs_rho)",
        "spacing_fit": payload["spacing_fit"],
        "r2_computed": False,
        "runs": selected_runs,
        "selected_layer_counts": {str(k): v for k, v in sorted(counts.items())},
        "aggregate_selected_layer": int(best),
        "aggregate_joint_score": float(np.sqrt(aggregate[best]["ev_mean"] * aggregate[best]["rho_mean"])),
        "aggregate_selected_metrics": aggregate[best],
    }


def robust_joint_selection(payload: dict) -> dict:
    """Select one layer by median per-seed geometric-mean EV-rho score."""
    layers = set.intersection(*(set(run["layers"]) for run in payload["runs"]))
    scores = {}
    for layer in layers:
        per_seed = [
            float(np.sqrt(max(run["layers"][layer]["ev"], 0.0) * abs(run["layers"][layer]["rho"])))
            for run in payload["runs"]
        ]
        if np.all(np.isfinite(per_seed)):
            scores[layer] = {
                "q_per_seed": per_seed,
                "Q_median": float(np.median(per_seed)),
            }
    best = max(scores, key=lambda layer: scores[layer]["Q_median"])
    return {
        "method": "median per-seed geometric-mean EV-rho selection",
        "selection_seeds": [run["seed"] for run in payload["runs"]],
        "selection": "argmax_layer median_seed sqrt(EV_seed_layer * abs(rho_seed_layer))",
        "selected_layer": int(best),
        "selected_Q": scores[best]["Q_median"],
        "layer_scores": {str(layer): values for layer, values in scores.items()},
    }


if __name__ == "__main__":
    answers = {group: [group * 100 + i for i in range(8)] for group in range(1, 5)}
    states = {
        0: {group: [np.zeros(2) for _ in values] for group, values in answers.items()},
        1: {
            group: [np.array([value, value * 1e-4]) for value in values]
            for group, values in answers.items()
        },
    }
    result = score_bundle(HiddenStateBundle(n_layers=2, states=states, answers={0: answers, 1: answers}))
    assert result["selected_layer"] == 1 and "r2" not in result["selected_metrics"]
