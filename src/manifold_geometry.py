"""High-dimensional number-line geometry diagnostics.

This module extends the PCA/PLS number-line probe in :mod:`src.geometry` with
metrics that test whether the numerical trajectory is straight or curved in the
full residual-stream space before projection.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

from src.geometry import (
    GeometryConfig,
    HiddenStateBundle,
    _compression_rate,
    _load_model,
    _monotonicity,
    _resolve_device,
    _seed_all,
    _transform_layer,
    collect_hidden_states,
)


def _layer_matrix(bundle: HiddenStateBundle, layer: int) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    feats: list[np.ndarray] = []
    answers: list[float] = []
    group_indices: dict[int, list[int]] = {}

    for g in sorted(bundle.states[layer].keys(), key=int):
        reps = bundle.states[layer][g]
        idx = list(range(len(feats), len(feats) + len(reps)))
        group_indices[g] = idx
        for r, a in zip(reps, bundle.answers[layer][g]):
            feats.append(np.asarray(r, dtype=float).reshape(-1))
            answers.append(float(a))

    if not feats:
        return np.empty((0, 0)), np.empty((0,)), {}
    return np.stack(feats, axis=0), np.asarray(answers, dtype=float), group_indices


def _format_numeral_prompt(numbers: Sequence[int]) -> str:
    body = ",".join(f"{n}={n}" for n in numbers[:-1])
    return f"{body},{numbers[-1]}=" if body else f"{numbers[-1]}="


def _dense_log_bins(n_min: int, n_max: int, n_anchors: int) -> list[dict[str, float | int]]:
    if n_anchors < 3:
        raise ValueError(f"dense_anchors must be >= 3, got {n_anchors}")
    if n_min < 1 or n_max <= n_min:
        raise ValueError(f"expected 1 <= dense_min < dense_max, got {n_min}, {n_max}")

    logs = np.linspace(np.log10(n_min), np.log10(n_max), n_anchors)
    centers = np.power(10.0, logs)
    edge_logs = (logs[:-1] + logs[1:]) / 2.0
    edges = np.power(10.0, edge_logs)

    bins: list[dict[str, float | int]] = []
    prev_hi = n_min - 1
    for i, center in enumerate(centers):
        if i == 0:
            lo = n_min
        else:
            lo = prev_hi + 1
        if i == n_anchors - 1:
            hi = n_max
        else:
            hi = int(max(lo, math.floor(edges[i])))
        prev_hi = hi
        bins.append(
            {
                "group": i,
                "center": float(center),
                "log10_center": float(logs[i]),
                "lo": int(lo),
                "hi": int(hi),
            }
        )
    return bins


def collect_dense_log_hidden_states(
    cfg: GeometryConfig,
    model,
    tokenizer,
    device: torch.device,
    rng,
    *,
    dense_anchors: int,
    dense_min: int,
    dense_max: int,
) -> tuple[HiddenStateBundle, list[dict[str, float | int]]]:
    """Collect hidden states from log-spaced numeric bins over [dense_min, dense_max]."""
    bins = _dense_log_bins(dense_min, dense_max, dense_anchors)
    n_layers = int(model.config.num_hidden_layers) + 1

    states: dict[int, dict[int, list[np.ndarray]]] = {l: {} for l in range(n_layers)}
    answers: dict[int, dict[int, list[float]]] = {l: {} for l in range(n_layers)}

    model.eval()
    with torch.no_grad():
        for b in bins:
            group = int(b["group"])
            lo = int(b["lo"])
            hi = int(b["hi"])
            grp_states: dict[int, list[np.ndarray]] = {l: [] for l in range(n_layers)}
            grp_answers: list[float] = []
            for _ in range(cfg.k):
                n = int(rng.randint(lo, hi))
                if cfg.context == "random":
                    ctx_nums = [int(rng.randint(dense_min, dense_max)) for _ in range(cfg.num_examples)]
                elif cfg.context == "fixed":
                    fixed_ctx = (4, 54, 432, 9543)
                    ctx_nums = list(fixed_ctx[: cfg.num_examples])
                elif cfg.context == "same":
                    ctx_nums = [n for _ in range(cfg.num_examples)]
                else:
                    raise ValueError(f"unknown context option: {cfg.context!r}")
                prompt = _format_numeral_prompt([*ctx_nums, n])
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                inputs.pop("token_type_ids", None)
                outputs = model(**inputs, output_hidden_states=True, use_cache=False)
                for layer, hs in enumerate(outputs.hidden_states):
                    h = hs[0, -1, :].detach().to(torch.float32).cpu().numpy()
                    grp_states[layer].append(h)
                grp_answers.append(float(n))
            for layer in range(n_layers):
                states[layer][group] = grp_states[layer]
                answers[layer][group] = list(grp_answers)

    return HiddenStateBundle(n_layers=n_layers, states=states, answers=answers), bins


def _fit_beta_from_diffs(diffs: Sequence[float], *, eps: float = 1e-8) -> float:
    vals = np.asarray(diffs, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan")
    vals = np.clip(vals, eps, None)
    x = np.arange(1, vals.size + 1, dtype=float)
    y = np.log(vals)
    slope, _intercept = np.polyfit(x, y, deg=1)
    return float(np.exp(slope))


def _angle_deg(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return float("nan")
    cos = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _group_means(X: np.ndarray, group_indices: dict[int, list[int]]) -> tuple[list[int], np.ndarray]:
    groups = sorted(group_indices.keys(), key=int)
    means = np.stack([X[group_indices[g]].mean(axis=0) for g in groups], axis=0)
    return groups, means


def _safe_spearman(y: np.ndarray, z: np.ndarray) -> float:
    rho, _ = spearmanr(np.asarray(y, dtype=float), np.asarray(z, dtype=float))
    return float(abs(rho)) if not np.isnan(rho) else float("nan")


def _participation_ratio(eigs: np.ndarray) -> float:
    vals = np.asarray(eigs, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float("nan")
    return float((vals.sum() ** 2) / np.square(vals).sum())


def _nan_summary(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _summarize_runs(per_run: list[dict[int, dict[str, float]]]) -> dict[int, dict[str, dict[str, float]]]:
    layers = sorted({layer for run in per_run for layer in run})
    metrics = sorted({k for run in per_run for row in run.values() for k in row})
    out: dict[int, dict[str, dict[str, float]]] = {}
    for layer in layers:
        out[layer] = {}
        for metric in metrics:
            out[layer][metric] = _nan_summary(
                run[layer][metric] for run in per_run if layer in run and metric in run[layer]
            )
    return out


def _best_layer(
    summary: dict[int, dict[str, dict[str, float]]],
    metric: str = "pca_ev",
    *,
    min_layer: int = 0,
) -> int | None:
    candidates = [
        layer
        for layer, vals in summary.items()
        if layer >= min_layer and metric in vals and np.isfinite(vals[metric]["mean"])
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda layer: summary[layer][metric]["mean"])


def analyze_manifold_layer(
    bundle: HiddenStateBundle,
    layer: int,
    *,
    isomap_neighbors: int = 12,
    intrinsic_components: int = 24,
) -> dict[str, float]:
    X, answers, group_indices = _layer_matrix(bundle, layer)
    if X.size == 0:
        return {}

    metrics: dict[str, float] = {}

    pca_t = _transform_layer(bundle, layer, "PCA", 1)
    if pca_t is not None:
        pc1 = pca_t.components[:, 0]
        metrics["pca_ev"] = float(pca_t.explained_variance)
        metrics["pca_rho"] = abs(_monotonicity(pca_t.answers, pc1))
        metrics["pca_beta"] = _compression_rate(pca_t)
    else:
        metrics["pca_ev"] = float("nan")
        metrics["pca_rho"] = float("nan")
        metrics["pca_beta"] = float("nan")

    _groups, means = _group_means(X, group_indices)
    segments = np.diff(means, axis=0)
    seg_lengths = np.linalg.norm(segments, axis=1)
    path_length = float(seg_lengths.sum()) if seg_lengths.size else float("nan")
    chord = float(np.linalg.norm(means[-1] - means[0])) if len(means) > 1 else float("nan")
    straightness = float(chord / path_length) if path_length and path_length > 0 else float("nan")
    angles = [_angle_deg(segments[i], segments[i + 1]) for i in range(max(0, len(segments) - 1))]
    finite_angles = np.asarray([a for a in angles if np.isfinite(a)], dtype=float)

    metrics.update(
        {
            "hd_beta": _fit_beta_from_diffs(seg_lengths),
            "hd_path_length": path_length,
            "hd_chord": chord,
            "hd_straightness": straightness,
            "hd_curvature": float(1.0 - straightness) if np.isfinite(straightness) else float("nan"),
            "turn_angle_mean_deg": float(finite_angles.mean()) if finite_angles.size else float("nan"),
            "turn_angle_max_deg": float(finite_angles.max()) if finite_angles.size else float("nan"),
        }
    )
    for i, value in enumerate(seg_lengths, start=1):
        metrics[f"hd_segment_{i}"] = float(value)

    n_comp = int(min(max(1, intrinsic_components), X.shape[0] - 1, X.shape[1]))
    try:
        pca_full = PCA(n_components=n_comp)
        pca_full.fit(X)
        ev = np.asarray(pca_full.explained_variance_, dtype=float)
        evr = np.asarray(pca_full.explained_variance_ratio_, dtype=float)
        metrics["intrinsic_dim_pr"] = _participation_ratio(ev)
        metrics["pca_var_first3"] = float(evr[:3].sum()) if evr.size else float("nan")
        metrics["pca_var_first10"] = float(evr[:10].sum()) if evr.size else float("nan")
        for i in range(min(3, evr.size)):
            metrics[f"pca_var_{i + 1}"] = float(evr[i])
    except Exception:
        metrics["intrinsic_dim_pr"] = float("nan")
        metrics["pca_var_first3"] = float("nan")
        metrics["pca_var_first10"] = float("nan")

    n_neighbors = int(min(max(2, isomap_neighbors), X.shape[0] - 1))
    try:
        X_centered = X - X.mean(axis=0, keepdims=True)
        iso = Isomap(n_neighbors=n_neighbors, n_components=1)
        coord = np.asarray(iso.fit_transform(X_centered), dtype=float).reshape(-1)
        metrics["isomap_rho"] = _safe_spearman(answers, coord)
        iso_means = np.asarray([coord[group_indices[g]].mean() for g in sorted(group_indices, key=int)])
        metrics["isomap_beta"] = _fit_beta_from_diffs(np.abs(np.diff(iso_means)))
        try:
            metrics["isomap_reconstruction_error"] = float(iso.reconstruction_error())
        except Exception:
            metrics["isomap_reconstruction_error"] = float("nan")
    except Exception:
        metrics["isomap_rho"] = float("nan")
        metrics["isomap_beta"] = float("nan")
        metrics["isomap_reconstruction_error"] = float("nan")

    return metrics


def _visualization_for_layer(bundle: HiddenStateBundle, layer: int) -> dict[str, Any] | None:
    X, answers, group_indices = _layer_matrix(bundle, layer)
    if X.size == 0 or X.shape[0] < 4:
        return None
    n_comp = int(min(3, X.shape[0] - 1, X.shape[1]))
    if n_comp < 2:
        return None
    pca = PCA(n_components=n_comp)
    coords = np.asarray(pca.fit_transform(X), dtype=float)
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), mode="constant")

    groups_sorted = sorted(group_indices.keys(), key=int)
    group_ids = np.empty(len(answers), dtype=int)
    means: list[list[float]] = []
    for g in groups_sorted:
        idx = group_indices[g]
        group_ids[idx] = int(g)
        means.append(coords[idx, :3].mean(axis=0).tolist())

    return {
        "layer": int(layer),
        "coords": coords[:, :3].tolist(),
        "answers": answers.astype(float).tolist(),
        "groups": group_ids.astype(int).tolist(),
        "group_order": [int(g) for g in groups_sorted],
        "group_means": means,
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
    }


def run_manifold_geometry(
    cfg: GeometryConfig,
    *,
    hf_token: str | None = None,
    isomap_neighbors: int = 12,
    intrinsic_components: int = 24,
    min_select_layer: int = 0,
    dense_anchors: int = 0,
    dense_min: int = 10,
    dense_max: int = 10000,
    save_visualization: bool = False,
    progress_callback=None,
) -> dict[str, Any]:
    """Run high-dimensional manifold diagnostics for every layer."""
    _seed_all(cfg.seed)
    device = _resolve_device(cfg.device)
    model, tokenizer = _load_model(cfg, device, hf_token)

    per_run: list[dict[int, dict[str, float]]] = []
    visualizations_run0: dict[int, dict[str, Any]] = {}
    dense_bins: list[dict[str, float | int]] | None = None
    n_layers = 0
    for run_idx in range(max(1, cfg.runs)):
        import random

        rng = random.Random(cfg.seed + run_idx)
        if dense_anchors and dense_anchors > 0:
            bundle, dense_bins = collect_dense_log_hidden_states(
                cfg,
                model,
                tokenizer,
                device,
                rng,
                dense_anchors=dense_anchors,
                dense_min=dense_min,
                dense_max=dense_max,
            )
        else:
            bundle = collect_hidden_states(cfg, model, tokenizer, device, rng)
        n_layers = bundle.n_layers
        run_metrics: dict[int, dict[str, float]] = {}
        for layer in range(bundle.n_layers):
            run_metrics[layer] = analyze_manifold_layer(
                bundle,
                layer,
                isomap_neighbors=isomap_neighbors,
                intrinsic_components=intrinsic_components,
            )
            if run_idx == 0 and save_visualization:
                viz = _visualization_for_layer(bundle, layer)
                if viz is not None:
                    visualizations_run0[layer] = viz
        per_run.append(run_metrics)
        if progress_callback is not None:
            progress_callback(run_idx + 1, cfg.runs)

    summary = _summarize_runs(per_run)
    best = _best_layer(summary, "pca_ev", min_layer=max(0, int(min_select_layer)))
    best_visualization = visualizations_run0.get(best) if best is not None else None
    return {
        "config": asdict(cfg),
        "manifold_config": {
            "isomap_neighbors": isomap_neighbors,
            "intrinsic_components": intrinsic_components,
            "best_layer_metric": "pca_ev",
            "min_select_layer": int(min_select_layer),
            "dense_anchors": dense_anchors,
            "dense_min": dense_min,
            "dense_max": dense_max,
            "dense_bins": dense_bins,
            "save_visualization": save_visualization,
        },
        "n_layers": n_layers,
        "best_layer": best,
        "best_layer_metrics": summary.get(best, {}) if best is not None else {},
        "layers": {str(layer): metrics for layer, metrics in summary.items()},
        "visualization": best_visualization,
    }


def _model_label(model_name: str) -> str:
    label_map = {
        "EleutherAI/pythia-2.8b": "Pythia-2.8B",
        "tiiuae/falcon-rw-1b": "Falcon-RW-1B",
        "tiiuae/falcon-rw-7b": "Falcon-RW-7B",
        "togethercomputer/RedPajama-INCITE-Base-3B-v1": "RedPajama-3B",
        "togethercomputer/RedPajama-INCITE-7B-Base": "RedPajama-7B",
        "allenai/OLMo-7B": "OLMo-7B-2T",
        "allenai/OLMo-7B-Twin-2T": "OLMo-7B-Twin-2T",
        "bigcode/starcoderbase-1b": "StarCoderBase-1B",
        "bigcode/starcoderbase-3b": "StarCoderBase-3B",
        "bigcode/starcoderbase-7b": "StarCoderBase-7B",
        "gpt2-large": "GPT2-L",
        "openai-community/gpt2-large": "GPT2-L",
    }
    return label_map.get(model_name, model_name.split("/")[-1])


def _metric(payload: dict[str, Any], layer: int, name: str, stat: str = "mean") -> float:
    return float(payload["layers"].get(str(layer), {}).get(name, {}).get(stat, float("nan")))


def write_manifold_artifacts(payloads: Sequence[dict[str, Any]], outdir: str | Path) -> dict[str, str]:
    """Write JSON, CSV summaries, and compact diagnostic figures."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    combined_json = out / "manifold_geometry_payloads.json"
    combined_json.write_text(json.dumps(list(payloads), indent=2), encoding="utf-8")

    layer_csv = out / "manifold_layer_metrics.csv"
    all_metric_names = sorted(
        {
            metric
            for payload in payloads
            for layer in payload["layers"].values()
            for metric in layer.keys()
        }
    )
    with layer_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["model", "model_name", "layer"]
        for metric in all_metric_names:
            header.extend([f"{metric}_mean", f"{metric}_std"])
        writer.writerow(header)
        for payload in payloads:
            model_name = payload["config"]["model_name"]
            label = _model_label(model_name)
            for layer_s in sorted(payload["layers"], key=lambda x: int(x)):
                vals = payload["layers"][layer_s]
                row: list[Any] = [label, model_name, int(layer_s)]
                for metric in all_metric_names:
                    row.append(vals.get(metric, {}).get("mean", float("nan")))
                    row.append(vals.get(metric, {}).get("std", float("nan")))
                writer.writerow(row)

    best_csv = out / "manifold_best_layer_metrics.csv"
    best_fields = [
        "pca_ev",
        "pca_rho",
        "pca_beta",
        "hd_beta",
        "hd_straightness",
        "hd_curvature",
        "turn_angle_mean_deg",
        "turn_angle_max_deg",
        "intrinsic_dim_pr",
        "pca_var_first3",
        "isomap_rho",
        "isomap_beta",
    ]
    with best_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "model_name", "best_layer", *[f"{m}_mean" for m in best_fields]])
        for payload in payloads:
            model_name = payload["config"]["model_name"]
            label = _model_label(model_name)
            best = int(payload["best_layer"]) if payload.get("best_layer") is not None else -1
            writer.writerow([label, model_name, best, *[_metric(payload, best, m) for m in best_fields]])

    fig_paths = _write_figures(payloads, out)
    visual_paths = _write_manifold_visualizations(payloads, out)
    plotly_paths = _write_plotly_manifold_visualizations(payloads, out)
    similarity_paths = _write_shape_similarity(payloads, out)
    return {
        "combined_json": str(combined_json),
        "layer_csv": str(layer_csv),
        "best_csv": str(best_csv),
        **fig_paths,
        **visual_paths,
        **plotly_paths,
        **similarity_paths,
    }


def _write_figures(payloads: Sequence[dict[str, Any]], out: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n = len(payloads)
    fig, axes = plt.subplots(n, 3, figsize=(10.8, max(2.5, 2.5 * n)), squeeze=False)
    for row, payload in enumerate(payloads):
        label = _model_label(payload["config"]["model_name"])
        layers = np.array(sorted(int(x) for x in payload["layers"].keys()), dtype=int)
        best = payload.get("best_layer")

        ax = axes[row, 0]
        for metric, color, style, name in [
            ("pca_beta", "black", "-", "PCA"),
            ("hd_beta", "#2b6f92", "--", "HD path"),
            ("isomap_beta", "#9a5c2e", ":", "Isomap"),
        ]:
            y = np.array([_metric(payload, int(l), metric) for l in layers], dtype=float)
            ax.plot(layers, y, linestyle=style, color=color, linewidth=1.2, label=name)
        if best is not None:
            ax.axvline(int(best), color="0.45", linestyle=":", linewidth=0.8)
        ax.set_title(f"{label}: scaling")
        ax.set_xlabel("layer")
        ax.set_ylabel(r"$\beta$")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

        ax = axes[row, 1]
        y = np.array([_metric(payload, int(l), "turn_angle_mean_deg") for l in layers], dtype=float)
        ax.plot(layers, y, color="#6b4c9a", linewidth=1.2)
        if best is not None:
            ax.axvline(int(best), color="0.45", linestyle=":", linewidth=0.8)
        ax.set_title(f"{label}: curvature")
        ax.set_xlabel("layer")
        ax.set_ylabel("mean turn angle")
        ax.grid(True, alpha=0.25)

        ax = axes[row, 2]
        y = np.array([_metric(payload, int(l), "intrinsic_dim_pr") for l in layers], dtype=float)
        ax.plot(layers, y, color="#3b7f52", linewidth=1.2)
        if best is not None:
            ax.axvline(int(best), color="0.45", linestyle=":", linewidth=0.8)
        ax.set_title(f"{label}: local dimension")
        ax.set_xlabel("layer")
        ax.set_ylabel("participation ratio")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    summary_pdf = out / "manifold_layer_diagnostics.pdf"
    summary_png = out / "manifold_layer_diagnostics.png"
    fig.savefig(summary_pdf, bbox_inches="tight")
    fig.savefig(summary_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))
    labels = [_model_label(p["config"]["model_name"]) for p in payloads]
    x = np.arange(len(labels))
    bar_color = "#4c6f91"

    values = [
        ("hd_straightness", "straightness\n(chord/path)", (0, 1.05)),
        ("turn_angle_mean_deg", "turn angle\n(degrees)", None),
        ("isomap_beta", r"Isomap $\beta$", None),
    ]
    for ax, (metric, ylabel, ylim) in zip(axes, values):
        y = []
        err = []
        for payload in payloads:
            best = int(payload["best_layer"])
            y.append(_metric(payload, best, metric))
            err.append(_metric(payload, best, metric, "std"))
        ax.bar(x, y, yerr=err, color=bar_color, edgecolor="black", linewidth=0.5, capsize=3)
        ax.set_xticks(x, labels, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    best_pdf = out / "manifold_best_layer_summary.pdf"
    best_png = out / "manifold_best_layer_summary.png"
    fig.savefig(best_pdf, bbox_inches="tight")
    fig.savefig(best_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

    return {
        "layer_diagnostics_pdf": str(summary_pdf),
        "layer_diagnostics_png": str(summary_png),
        "best_layer_summary_pdf": str(best_pdf),
        "best_layer_summary_png": str(best_png),
    }


def _write_manifold_visualizations(payloads: Sequence[dict[str, Any]], out: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    with_viz = [p for p in payloads if p.get("visualization")]
    if not with_viz:
        return {}

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n = len(with_viz)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(4.1 * cols, 3.6 * rows))
    cmap = cm.get_cmap("viridis")
    all_answers = np.concatenate([
        np.asarray(p["visualization"]["answers"], dtype=float) for p in with_viz
    ])
    norm = Normalize(vmin=float(np.log10(np.maximum(1, all_answers)).min()),
                     vmax=float(np.log10(np.maximum(1, all_answers)).max()))

    axes = []
    for i, payload in enumerate(with_viz, start=1):
        viz = payload["visualization"]
        coords = np.asarray(viz["coords"], dtype=float)
        answers = np.asarray(viz["answers"], dtype=float)
        groups = np.asarray(viz["groups"], dtype=int)
        means = np.asarray(viz["group_means"], dtype=float)
        colors = cmap(norm(np.log10(np.maximum(1, answers))))
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        axes.append(ax)

        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
        z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
        x_pad = 0.10 * max(1e-9, x_max - x_min)
        y_pad = 0.10 * max(1e-9, y_max - y_min)
        z_pad = 0.16 * max(1e-9, z_max - z_min)
        z_floor = z_min - z_pad

        xx, yy = np.meshgrid(
            np.linspace(x_min - x_pad, x_max + x_pad, 2),
            np.linspace(y_min - y_pad, y_max + y_pad, 2),
        )
        zz = np.full_like(xx, z_floor)
        ax.plot_surface(xx, yy, zz, color="0.93", alpha=0.50, shade=False, linewidth=0)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            np.full(coords.shape[0], z_floor),
            s=6,
            c="0.1",
            alpha=0.045,
            depthshade=False,
        )
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=18,
            c=colors,
            alpha=0.86,
            edgecolors="none",
            depthshade=True,
        )
        ax.plot(
            means[:, 0],
            means[:, 1],
            means[:, 2],
            color="black",
            linewidth=1.4,
            alpha=0.78,
        )
        group_colors: list[Any] = []
        for g in viz["group_order"]:
            mask = groups == int(g)
            group_log = float(np.log10(np.maximum(1, answers[mask])).mean()) if np.any(mask) else norm.vmin
            group_colors.append(cmap(norm(group_log)))
        ax.scatter(
            means[:, 0],
            means[:, 1],
            means[:, 2],
            s=44,
            c=group_colors,
            edgecolors="black",
            linewidths=0.4,
            depthshade=True,
        )

        label = _model_label(payload["config"]["model_name"])
        best = payload.get("best_layer")
        straight = _metric(payload, int(best), "hd_straightness") if best is not None else float("nan")
        angle = _metric(payload, int(best), "turn_angle_mean_deg") if best is not None else float("nan")
        ax.set_title(f"{label}\nL={best}, straightness={straight:.2f}, angle={angle:.0f}°", pad=2)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_zlim(z_floor, z_max + z_pad)
        ax.set_box_aspect((1.2, 1.0, 0.72))
        ax.set_axis_off()
        ax.view_init(elev=24, azim=-58)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.50)
    cbar.set_label(r"$\log_{10}(N)$")

    pdf = out / "manifold_best_layer_3d.pdf"
    png = out / "manifold_best_layer_3d.png"
    clean_pdf = out / "manifold_best_layer_3d_clean.pdf"
    clean_png = out / "manifold_best_layer_3d_clean.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(clean_pdf, bbox_inches="tight")
    fig.savefig(clean_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "manifold_3d_pdf": str(pdf),
        "manifold_3d_png": str(png),
        "manifold_3d_clean_pdf": str(clean_pdf),
        "manifold_3d_clean_png": str(clean_png),
    }


def _safe_filename(label: str) -> str:
    keep = []
    for ch in label:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "model"


def _write_plotly_manifold_visualizations(payloads: Sequence[dict[str, Any]], out: Path) -> dict[str, str]:
    """Write interactive 3-D manifold plots with every sampled number as a point."""
    with_viz = [p for p in payloads if p.get("visualization")]
    if not with_viz:
        return {}

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return {}

    paths: dict[str, str] = {}

    all_answers = np.concatenate([
        np.asarray(p["visualization"]["answers"], dtype=float) for p in with_viz
    ])
    log_min = float(np.log10(np.maximum(1, all_answers)).min())
    log_max = float(np.log10(np.maximum(1, all_answers)).max())

    def _point_trace(payload: dict[str, Any], *, showscale: bool) -> Any:
        viz = payload["visualization"]
        coords = np.asarray(viz["coords"], dtype=float)
        answers = np.asarray(viz["answers"], dtype=float)
        groups = np.asarray(viz["groups"], dtype=int)
        return go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker={
                "size": 3.4,
                "opacity": 0.82,
                "color": np.log10(np.maximum(1, answers)),
                "colorscale": "Viridis",
                "cmin": log_min,
                "cmax": log_max,
                "colorbar": {"title": "log10(N)"} if showscale else None,
                "showscale": showscale,
            },
            text=[f"N={int(a)}<br>group={int(g)}" for a, g in zip(answers, groups)],
            hovertemplate="%{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>",
            name="numbers",
            showlegend=False,
        )

    def _mean_trace(payload: dict[str, Any]) -> Any:
        viz = payload["visualization"]
        means = np.asarray(viz["group_means"], dtype=float)
        return go.Scatter3d(
            x=means[:, 0],
            y=means[:, 1],
            z=means[:, 2],
            mode="lines+markers",
            line={"color": "black", "width": 5},
            marker={"size": 4, "color": "black"},
            hovertemplate="anchor mean<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>",
            name="anchor means",
            showlegend=False,
        )

    for payload in with_viz:
        label = _model_label(payload["config"]["model_name"])
        best = payload.get("best_layer")
        fig = go.Figure()
        fig.add_trace(_point_trace(payload, showscale=True))
        fig.add_trace(_mean_trace(payload))
        fig.update_layout(
            title=f"{label} 3D number manifold (L={best})",
            width=980,
            height=760,
            margin={"l": 0, "r": 0, "t": 52, "b": 0},
            scene={
                "xaxis_title": "PC1",
                "yaxis_title": "PC2",
                "zaxis_title": "PC3",
                "aspectmode": "data",
            },
        )
        path = out / f"manifold_3d_plotly_{_safe_filename(label)}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        paths[f"plotly_3d_{_safe_filename(label)}"] = str(path)

    n = len(with_viz)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]
    labels = [_model_label(p["config"]["model_name"]) for p in with_viz]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=[
            f"{label} (L={payload.get('best_layer')})"
            for label, payload in zip(labels, with_viz)
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )
    for i, payload in enumerate(with_viz):
        row = i // cols + 1
        col = i % cols + 1
        fig.add_trace(_point_trace(payload, showscale=(i == 0)), row=row, col=col)
        fig.add_trace(_mean_trace(payload), row=row, col=col)

    fig.update_layout(
        title="3D number manifolds across models",
        width=360 * cols,
        height=330 * rows + 80,
        margin={"l": 6, "r": 6, "t": 72, "b": 6},
    )
    for i in range(1, n + 1):
        fig.update_layout(
            **{
                f"scene{i}": {
                    "xaxis_title": "",
                    "yaxis_title": "",
                    "zaxis_title": "",
                    "aspectmode": "data",
                }
            }
        )
    combined = out / "manifold_3d_plotly_all_models.html"
    fig.write_html(combined, include_plotlyjs="cdn")
    paths["plotly_3d_all_models"] = str(combined)
    return paths


def _shape_vectors(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segs = np.diff(points, axis=0)
    seg_len = np.linalg.norm(segs, axis=1)
    angles: list[float] = []
    for i in range(max(0, len(segs) - 1)):
        angles.append(_angle_deg(segs[i], segs[i + 1]))
    dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    pairwise = dmat[np.triu_indices_from(dmat, k=1)]
    return pairwise, seg_len, np.asarray(angles, dtype=float)


def _corr_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.stats import pearsonr

    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan"), float("nan")
    r = pearsonr(x[mask], y[mask])
    return float(r.statistic), float(r.pvalue)


def _write_shape_similarity(payloads: Sequence[dict[str, Any]], out: Path) -> dict[str, str]:
    from scipy.spatial import procrustes

    usable: list[tuple[str, np.ndarray]] = []
    for payload in payloads:
        viz = payload.get("visualization")
        if not viz:
            continue
        points = np.asarray(viz["group_means"], dtype=float)
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] >= 3:
            usable.append((_model_label(payload["config"]["model_name"]), points[:, :3]))
    if len(usable) < 2:
        return {}

    csv_path = out / "manifold_shape_similarity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_a",
                "model_b",
                "procrustes_disparity",
                "procrustes_rmsd",
                "pairwise_distance_r",
                "pairwise_distance_p",
                "segment_length_r",
                "segment_length_p",
                "turn_angle_r",
                "turn_angle_p",
            ]
        )
        for i, (label_a, a) in enumerate(usable):
            for label_b, b in usable[i + 1 :]:
                _, _, disparity = procrustes(a, b)
                p_a, s_a, t_a = _shape_vectors(a)
                p_b, s_b, t_b = _shape_vectors(b)
                pair_r, pair_p = _corr_pair(p_a, p_b)
                seg_r, seg_p = _corr_pair(s_a, s_b)
                turn_r, turn_p = _corr_pair(t_a, t_b)
                writer.writerow(
                    [
                        label_a,
                        label_b,
                        float(disparity),
                        float(math.sqrt(disparity / len(a))),
                        pair_r,
                        pair_p,
                        seg_r,
                        seg_p,
                        turn_r,
                        turn_p,
                    ]
                )

    import matplotlib.pyplot as plt

    labels = [label for label, _points in usable]
    n = len(labels)
    disparity_mat = np.zeros((n, n), dtype=float)
    pair_r_mat = np.ones((n, n), dtype=float)
    for i, (_label_a, a) in enumerate(usable):
        for j, (_label_b, b) in enumerate(usable):
            if i == j:
                continue
            _, _, disparity = procrustes(a, b)
            p_a, _s_a, _t_a = _shape_vectors(a)
            p_b, _s_b, _t_b = _shape_vectors(b)
            pair_r, _ = _corr_pair(p_a, p_b)
            disparity_mat[i, j] = disparity
            pair_r_mat[i, j] = pair_r

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], disparity_mat, "Procrustes disparity", "magma_r", 0.0, np.nanmax(disparity_mat)),
        (axes[1], pair_r_mat, "Pairwise-distance correlation", "viridis", 0.0, 1.0),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(n), labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(n), labels)
        ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    heat_pdf = out / "manifold_shape_similarity_heatmap.pdf"
    heat_png = out / "manifold_shape_similarity_heatmap.png"
    fig.savefig(heat_pdf, bbox_inches="tight")
    fig.savefig(heat_png, dpi=260, bbox_inches="tight")
    plt.close(fig)

    return {
        "shape_similarity_csv": str(csv_path),
        "shape_similarity_heatmap_pdf": str(heat_pdf),
        "shape_similarity_heatmap_png": str(heat_png),
    }
