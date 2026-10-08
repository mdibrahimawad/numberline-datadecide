"""
Figure generation for the number-line geometry probe.

Mirrors the figure code that lived in `reference/llm_natural_log/Figure_4.ipynb` and
`reference/llm_natural_log/utils/visual_utils.py`, ported into the same NeurIPS-style
scaffolding used by `src/number_analysis.py`. Three figures total:

  fig_layer_profile     EV / rho / beta as a function of layer depth, for one
                        model -- the natural visualization of a single MLflow
                        run. Mirrors the per-layer trace implicit in
                        `analyze_transformed_hidden_states`.

  fig_demos_sweep       Best-layer EV / rho / beta as a function of an external
                        sweep axis (default: number of in-context
                        demonstrations), with one line per model. This is the
                        port of `Figure_4.ipynb`.

  fig_pc1_scatter       PC1 of the residual stream vs. log10(target value),
                        coloured by group, at one or more layers. Mirrors
                        `visual_utils.plot_pca_projections`.

Inputs always come from MLflow (or, for the PC1 scatter, from a JSON dump
written by `src.geometry_cli --save-projections`). No notebook dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mlflow.tracking import MlflowClient

import mlflow


# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #

W_SINGLE = 3.3
W_DOUBLE = 6.8
W_TRIPLE = 9.6

C = {
    "pca": "#0072B2",
    "pls": "#D55E00",
    "muted": "#BBBBBB",
    "rule": "#888888",
}

# Colorblind-safe Okabe & Ito 7-color palette + black, used for per-model lines.
MODEL_PALETTE = [
    "#000000", "#0072B2", "#D55E00", "#009E73",
    "#CC79A7", "#E69F00", "#56B4E9", "#F0E442",
]


def _apply_neurips_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {outdir}/{name}.pdf + .png")


# --------------------------------------------------------------------------- #
# MLflow helpers
# --------------------------------------------------------------------------- #

def _connect(tracking_uri: str | None) -> MlflowClient:
    if tracking_uri is None:
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{os.path.abspath('mlflow.db')}",
        )
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def _layer_metric(metrics: dict, prefix: str, layer: int, kind: str) -> float:
    return metrics.get(f"{prefix}_{kind}_layer_{layer:02d}", float("nan"))


def _layer_trace(metrics: dict, prefix: str, n_layers: int) -> dict[str, np.ndarray]:
    layers = np.arange(n_layers)
    out = {"layer": layers}
    for kind in ("ev_mean", "ev_std", "rho_mean", "rho_std", "beta_mean", "beta_std"):
        out[kind] = np.array([_layer_metric(metrics, prefix, l, kind) for l in layers])
    return out


def _resolve_run(client: MlflowClient, run_id: str | None,
                 experiment_name: str | None, filter_string: str | None) -> str:
    if run_id:
        return run_id
    if not experiment_name:
        raise SystemExit("provide either --run-id or --experiment-name")
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(f"experiment {experiment_name!r} not found")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=filter_string or "",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"no runs found in {experiment_name!r}")
    return runs[0].info.run_id


def _latest_model_run(client: MlflowClient, experiment_name: str,
                      model_name: str) -> str:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(f"experiment {experiment_name!r} not found")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.stage = 'number_geometry'",
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    for run in runs:
        run_model = run.data.tags.get("model_name") or run.data.params.get("model_name")
        if run_model == model_name:
            return run.info.run_id
    raise SystemExit(f"no number_geometry run found for model {model_name!r}")


# --------------------------------------------------------------------------- #
# figure 1: per-layer EV / rho / beta for one run
# --------------------------------------------------------------------------- #

def fig_layer_profile(client: MlflowClient, run_id: str, outdir: Path,
                      *, name: str = "geom_01_layer_profile") -> None:
    run = client.get_run(run_id)
    metrics = run.data.metrics
    n_layers = int(metrics.get("n_layers", 0))
    if n_layers <= 0:
        raise SystemExit(f"run {run_id} has no n_layers metric")

    pca = _layer_trace(metrics, "pca", n_layers)
    pls = _layer_trace(metrics, "pls", n_layers)
    model_name = run.data.tags.get("model_name", "?")

    fig, axes = plt.subplots(1, 3, figsize=(W_TRIPLE, 2.4), sharex=True)

    panels = [
        ("ev_mean", "ev_std", r"explained variance", False),
        ("rho_mean", "rho_std", r"$|\rho|$  (PC1 vs target)", False),
        ("beta_mean", "beta_std", r"compression rate $\beta$", True),
    ]

    for ax, (m_key, s_key, ylabel, log) in zip(axes, panels):
        for trace, color, label in [
            (pca, C["pca"], "PCA"),
            (pls, C["pls"], "PLS"),
        ]:
            y = trace[m_key]
            s = trace[s_key]
            mask = np.isfinite(y)
            if not mask.any():
                continue
            x = trace["layer"][mask]
            y = y[mask]
            s = s[mask]
            ax.plot(x, y, "-", color=color, lw=1.2, label=label)
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.18, linewidth=0)
        if m_key == "beta_mean":
            ax.axhline(1.0, color=C["rule"], lw=0.5, ls=":")
        ax.set_xlabel("layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, ls=":")
        if log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(1e-2, np.nanmin([np.nanmin(pca[m_key]), np.nanmin(pls[m_key])]) * 0.7))
    axes[0].legend(loc="best")

    fig.suptitle(f"Layer profile: {model_name}", y=1.04)
    fig.subplots_adjust(wspace=0.30)
    _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
# figure 1b: PCA layer traces + rendered best-layer table for selected models
# --------------------------------------------------------------------------- #

def _fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    if not (math.isfinite(mean) and math.isfinite(std)):
        return "--"
    return rf"${mean:.{digits}f}\pm{std:.{digits}f}$"


def _write_pca_table_tex(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Best layer & EV & $|\rho_S|$ & $\beta$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            rf"{row['model']} & {row['layer']} & {row['ev']} & "
            rf"{row['rho']} & {row['beta']} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))
    print(f"  wrote {path}")


def fig_pca_model_comparison(
    client: MlflowClient,
    experiment_name: str,
    model_names: list[str],
    outdir: Path,
    *,
    run_ids: list[str] | None = None,
    labels: list[str] | None = None,
    name: str = "geom_08_pca_model_comparison",
) -> None:
    if run_ids is not None and len(run_ids) != len(model_names):
        raise SystemExit("--comparison-run-ids must match --compare-models length")
    if labels is not None and len(labels) != len(model_names):
        raise SystemExit("--comparison-labels must match --compare-models length")

    resolved: list[tuple[str, str]] = []
    for i, model_name in enumerate(model_names):
        rid = run_ids[i] if run_ids is not None else _latest_model_run(
            client, experiment_name, model_name,
        )
        resolved.append((model_name, rid))

    fig = plt.figure(figsize=(W_TRIPLE, 4.6))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[2.25, 1.0],
        hspace=0.38, wspace=0.30,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")

    panels = [
        ("ev_mean", "ev_std", "explained variance", False),
        ("rho_mean", "rho_std", r"$|\rho_S|$  (PC1 vs target)", False),
        ("beta_mean", "beta_std", r"compression rate $\beta$", True),
    ]
    table_rows: list[dict[str, object]] = []

    for i, (model_name, run_id) in enumerate(resolved):
        run = client.get_run(run_id)
        metrics = run.data.metrics
        n_layers = int(metrics.get("n_layers", 0))
        if n_layers <= 0:
            raise SystemExit(f"run {run_id} has no n_layers metric")
        trace = _layer_trace(metrics, "pca", n_layers)
        color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        label = labels[i] if labels is not None else _pretty_model(model_name)

        for ax, (m_key, s_key, ylabel, log) in zip(axes, panels):
            y = trace[m_key]
            s = trace[s_key]
            mask = np.isfinite(y)
            if not mask.any():
                continue
            x = trace["layer"][mask]
            y = y[mask]
            s = s[mask]
            ax.plot(x, y, "-", color=color, lw=1.3, label=label)
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.14, linewidth=0)
            if log:
                ax.set_yscale("log")
            ax.set_xlabel("layer")
            ax.set_ylabel(ylabel)
            ax.grid(True, ls=":")

        best_layer_raw = metrics.get("pca_best_layer", float("nan"))
        if not math.isfinite(best_layer_raw):
            raise SystemExit(f"run {run_id} has no finite pca_best_layer metric")
        best_layer = int(best_layer_raw)
        ev = _layer_metric(metrics, "pca", best_layer, "ev_mean")
        ev_std = _layer_metric(metrics, "pca", best_layer, "ev_std")
        rho = _layer_metric(metrics, "pca", best_layer, "rho_mean")
        rho_std = _layer_metric(metrics, "pca", best_layer, "rho_std")
        beta = _layer_metric(metrics, "pca", best_layer, "beta_mean")
        beta_std = _layer_metric(metrics, "pca", best_layer, "beta_std")
        table_rows.append({
            "model": label,
            "layer": best_layer,
            "ev": _fmt_pm(ev, ev_std),
            "rho": _fmt_pm(rho, rho_std),
            "beta": _fmt_pm(beta, beta_std),
        })

    axes[2].axhline(1.0, color=C["rule"], lw=0.5, ls=":")
    axes[0].legend(loc="best")

    col_labels = ["Model", "Best PCA layer", "EV", r"$|\rho_S|$", r"$\beta$"]
    cell_text = [
        [str(r["model"]), str(r["layer"]), str(r["ev"]), str(r["rho"]), str(r["beta"])]
        for r in table_rows
    ]
    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.28)
    for (r, _c), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F2F2F2")

    fig.suptitle("Number-line PCA layer profile", y=0.99)
    _save(fig, outdir, name)
    _write_pca_table_tex(table_rows, outdir / f"{name}.tex")


# --------------------------------------------------------------------------- #
# figure 2: cross-model sweep (mirror of Figure_4.ipynb)
# --------------------------------------------------------------------------- #

def _pretty_model(name: str) -> str:
    short = name.split("/")[-1]
    return short


def fig_demos_sweep(client: MlflowClient, experiment_name: str, outdir: Path,
                    *,
                    sweep_param: str = "num_examples",
                    metric_prefix: str = "pca",
                    name: str = "geom_02_metric_vs_sweep",
                    extra_filter: str | None = None) -> None:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(f"experiment {experiment_name!r} not found")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=extra_filter or "tags.stage = 'number_geometry'",
        order_by=["attributes.start_time ASC"],
        max_results=2000,
    )
    if not runs:
        raise SystemExit(f"no number_geometry runs found in {experiment_name!r}")

    by_model: dict[str, dict[float, dict[str, float]]] = defaultdict(dict)
    for r in runs:
        params = r.data.params
        metrics = r.data.metrics
        if sweep_param not in params:
            continue
        try:
            x_val = float(params[sweep_param])
        except (TypeError, ValueError):
            continue
        model_name = r.data.tags.get("model_name") or params.get("model_name", "?")
        best = metrics.get(f"{metric_prefix}_best_layer")
        if best is None or not math.isfinite(best):
            continue
        L = int(best)
        cell = {
            "ev_mean": _layer_metric(metrics, metric_prefix, L, "ev_mean"),
            "ev_std":  _layer_metric(metrics, metric_prefix, L, "ev_std"),
            "rho_mean": _layer_metric(metrics, metric_prefix, L, "rho_mean"),
            "rho_std":  _layer_metric(metrics, metric_prefix, L, "rho_std"),
            "beta_mean": _layer_metric(metrics, metric_prefix, L, "beta_mean"),
            "beta_std":  _layer_metric(metrics, metric_prefix, L, "beta_std"),
        }
        by_model[model_name][x_val] = cell

    if not by_model:
        raise SystemExit(f"no runs in {experiment_name!r} have param {sweep_param!r}")

    panels = [
        ("ev_mean", "ev_std", "explained variance", False),
        ("rho_mean", "rho_std", r"$|\rho|$", False),
        ("beta_mean", "beta_std", r"compression rate $\beta$", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(W_TRIPLE, 2.5), sharex=True)

    for ax, (m_key, s_key, ylabel, log) in zip(axes, panels):
        for i, (model_name, by_x) in enumerate(sorted(by_model.items())):
            xs = sorted(by_x.keys())
            ys = np.array([by_x[x][m_key] for x in xs])
            es = np.array([by_x[x][s_key] for x in xs])
            color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
            ax.plot(xs, ys, "-o", color=color, lw=1.2, ms=3.5,
                    label=_pretty_model(model_name))
            mask = np.isfinite(ys) & np.isfinite(es)
            if mask.any():
                ax.fill_between(np.array(xs)[mask], (ys - es)[mask], (ys + es)[mask],
                                color=color, alpha=0.18, linewidth=0)
        if m_key == "beta_mean":
            ax.axhline(1.0, color=C["rule"], lw=0.5, ls=":")
            if log:
                ax.set_yscale("log")
        ax.set_xlabel(sweep_param.replace("_", " "))
        ax.set_ylabel(ylabel)
        ax.grid(True, ls=":")

    axes[-1].legend(loc="best", fontsize=6)
    fig.suptitle(f"{metric_prefix.upper()} best-layer geometry vs {sweep_param}", y=1.04)
    fig.subplots_adjust(wspace=0.30)
    _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
# figure 3: PC1 vs log10(target), coloured by group  (mirror of visual_utils)
# --------------------------------------------------------------------------- #

def _load_projections(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def fig_pc1_scatter(projections_json: Path, outdir: Path,
                    *,
                    method: str = "pca",
                    layers: list[int] | None = None,
                    name: str = "geom_03_pc1_scatter") -> None:
    blob = _load_projections(projections_json)
    proj_root = blob.get(f"projections_{method}")
    if not proj_root:
        raise SystemExit(
            f"{projections_json} has no projections_{method} block. Re-run "
            f"`src.geometry_cli` with `--save-projections`."
        )

    available = sorted(int(l) for l in proj_root.keys())
    if layers is None:
        best = blob.get(f"best_layer_{method}")
        if best is None or not isinstance(best, int):
            best = available[len(available) // 2]
        layers = [best]
    layers = [l for l in layers if l in available]
    if not layers:
        raise SystemExit("no requested layers exist in the projections JSON")

    n = len(layers)
    fig, axes = plt.subplots(
        1, n, figsize=(W_SINGLE * min(n, 3), 2.4), squeeze=False, sharey=True,
    )

    cmap = plt.get_cmap("viridis")
    for ax, layer in zip(axes[0], layers):
        layer_blob = proj_root[str(layer)]
        groups = sorted(int(g) for g in layer_blob["projections"].keys())
        all_x = []
        all_y = []
        all_g = []
        for g in groups:
            ys = layer_blob["answers"][str(g)]
            ps = layer_blob["projections"][str(g)]
            for y, p in zip(ys, ps):
                if y <= 0:
                    continue
                all_x.append(math.log10(float(y)))
                all_y.append(float(p[0]))
                all_g.append(g)
        if not all_x:
            continue
        gs = np.array(all_g, dtype=float)
        norm = mpl.colors.Normalize(vmin=gs.min(), vmax=gs.max())
        ax.scatter(all_x, all_y, c=gs, cmap=cmap, norm=norm,
                   s=10, alpha=0.85, edgecolors="none", rasterized=True)
        ax.set_title(rf"layer {layer}")
        ax.set_xlabel(r"$\log_{10}(\mathrm{target})$")
        ax.grid(True, ls=":")
    axes[0][0].set_ylabel(rf"PC1 ({method.upper()})")

    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=mpl.colors.Normalize(
            vmin=min(int(g) for g in proj_root[str(layers[0])]["projections"]),
            vmax=max(int(g) for g in proj_root[str(layers[0])]["projections"]),
        ),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0].tolist(), shrink=0.85, pad=0.015,
                        ticks=range(0, 5))
    cbar.set_label(r"group $i$  (target $\sim 10^i$)")

    fig.suptitle(rf"PC1 vs $\log_{{10}}(\mathrm{{target}})$ ({method.upper()})", y=1.05)
    _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
# figure 4: Stage 1 (corpus distribution) vs Stage 2 (model PC1) alignment
# --------------------------------------------------------------------------- #

CORPUS_NMAX = 1000

BOOT_DEFAULT = 2000


def _bootstrap_residual_spearman(
    n_arr: np.ndarray,
    pc1_arr: np.ndarray,
    log_count_arr: np.ndarray,
    *,
    n_boot: int = BOOT_DEFAULT,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Partial Spearman of (log_count, PC1) | log10(N), with percentile CI.

    Both PC1 and log_count are residualised by linear regression on log10(N)
    before correlation; bootstraps over (N, PC1, log_count) triples with
    replacement at the same n.
    """
    n_arr = np.asarray(n_arr, dtype=float)
    pc1_arr = np.asarray(pc1_arr, dtype=float)
    log_count_arr = np.asarray(log_count_arr, dtype=float)
    log_n = np.log10(np.maximum(n_arr, 1.0))
    rng = np.random.default_rng(seed)
    n = len(n_arr)

    def _stat(idx: np.ndarray) -> float:
        ln = log_n[idx]
        if np.std(ln) == 0:
            return float("nan")
        coef_pc = np.polyfit(ln, pc1_arr[idx], 1)
        coef_lc = np.polyfit(ln, log_count_arr[idx], 1)
        pc_res = pc1_arr[idx] - np.polyval(coef_pc, ln)
        lc_res = log_count_arr[idx] - np.polyval(coef_lc, ln)
        if np.std(pc_res) == 0 or np.std(lc_res) == 0:
            return float("nan")
        rho, _ = spearmanr(lc_res, pc_res)
        return float(rho) if not np.isnan(rho) else float("nan")

    pt = _stat(np.arange(n))
    samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = _stat(idx)
        if not np.isnan(s):
            samples.append(s)
    if len(samples) < max(50, n_boot // 20):
        return pt, float("nan"), float("nan")
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return pt, float(lo), float(hi)


def _bootstrap_r2_gap(
    targets: np.ndarray,
    pc1: np.ndarray,
    *,
    n_boot: int = BOOT_DEFAULT,
    seed: int = 0,
) -> tuple[float, float, float, float, float]:
    """Bootstrap CI on (R^2_log - R^2_linear).

    Returns (r2_linear, r2_log, gap, gap_ci_lo, gap_ci_hi).
    """
    targets = np.asarray(targets, dtype=float)
    pc1 = np.asarray(pc1, dtype=float)
    log_targets = np.log10(np.maximum(targets, 1.0))
    rng = np.random.default_rng(seed)
    n = len(targets)

    def _r2(x: np.ndarray, y: np.ndarray) -> float:
        if np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        coef = np.polyfit(x, y, 1)
        y_hat = np.polyval(coef, x)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    r2_lin_pt = _r2(targets, pc1)
    r2_log_pt = _r2(log_targets, pc1)
    gap_pt = r2_log_pt - r2_lin_pt

    gaps: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lin = _r2(targets[idx], pc1[idx])
        lg = _r2(log_targets[idx], pc1[idx])
        if not (np.isnan(lin) or np.isnan(lg)):
            gaps.append(lg - lin)
    if len(gaps) < max(50, n_boot // 20):
        return r2_lin_pt, r2_log_pt, gap_pt, float("nan"), float("nan")
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return r2_lin_pt, r2_log_pt, gap_pt, float(lo), float(hi)


def _pull_corpus_counts(client: MlflowClient, run_id: str) -> dict[int, int]:
    run = client.get_run(run_id)
    metrics = run.data.metrics
    out: dict[int, int] = {}
    for n in range(CORPUS_NMAX + 1):
        key = f"count_N_{n:04d}"
        if key in metrics:
            out[n] = int(metrics[key])
    if not out:
        raise SystemExit(
            f"Stage 1 run {run_id} has no count_N_NNNN metrics; pick a run "
            f"under experiment 'numberline_freq_count_expv1'."
        )
    return out


def _projections_per_n(layer_blob: dict) -> tuple[
    dict[int, float], dict[int, float], dict[int, int]
]:
    """Aggregate per-sample PC1 values to one (mean, std) per integer N."""
    per_n_pc1: dict[int, list[float]] = defaultdict(list)
    per_n_group: dict[int, int] = {}
    for g_str, projs in layer_blob["projections"].items():
        answers = layer_blob["answers"][g_str]
        g = int(g_str)
        for pc, ans in zip(projs, answers):
            n_val = int(round(float(ans)))
            per_n_pc1[n_val].append(float(pc[0]))
            per_n_group[n_val] = g
    means = {n: float(np.mean(v)) for n, v in per_n_pc1.items()}
    stds = {n: float(np.std(v)) for n, v in per_n_pc1.items()}
    return means, stds, per_n_group


GROUP_COLORS = {
    1: "#0072B2",
    2: "#D55E00",
    3: "#009E73",
    4: "#CC79A7",
}


def fig_stage1_alignment(
    client: MlflowClient,
    stage1_run_id: str,
    projections_json: Path,
    outdir: Path,
    *,
    method: str = "pca",
    layer: int | None = None,
    name: str = "geom_04_stage1_alignment",
) -> None:
    counts = _pull_corpus_counts(client, stage1_run_id)
    blob = _load_projections(projections_json)
    proj_root = blob.get(f"projections_{method}")
    if not proj_root:
        raise SystemExit(
            f"{projections_json} has no projections_{method} block. Re-run "
            f"`src.geometry_cli` with `--save-projections`."
        )

    available = sorted(int(l) for l in proj_root.keys())
    if layer is None:
        best = blob.get(f"best_layer_{method}")
        layer = int(best) if isinstance(best, int) else available[len(available) // 2]
    if layer not in available:
        raise SystemExit(f"layer {layer} not in projections JSON (have {available})")

    layer_blob = proj_root[str(layer)]
    pc1_mean, pc1_std, group_of = _projections_per_n(layer_blob)

    overlap = sorted(
        n for n in pc1_mean
        if 1 <= n <= CORPUS_NMAX and counts.get(n, 0) > 0
    )
    n_dropped = sum(1 for n in pc1_mean if n > CORPUS_NMAX)
    if not overlap:
        raise SystemExit(
            "no integers appear in both Stage 1 (counts in [1, 1000]) and "
            "Stage 2 (model groups). Did Stage 2 only sample group 4?"
        )

    model_name = blob.get("config", {}).get("model_name", "")
    pretty_model = _pretty_model(model_name) if model_name else "model"

    ns = np.array(overlap)
    pcs = np.array([pc1_mean[n] for n in overlap])
    psd = np.array([pc1_std[n] for n in overlap])
    grp = np.array([group_of[n] for n in overlap])
    log_count = np.array([math.log10(counts[n]) for n in overlap])
    is_round = (ns % 100 == 0) | ((ns % 10 == 0) & (ns >= 10))

    rho_full, _ = spearmanr(log_count, pcs)
    rho_resid: list[tuple[int, float, int, float, float]] = []
    for g in sorted(set(grp.tolist())):
        m = grp == g
        if m.sum() < 3:
            continue
        rho_g, lo_g, hi_g = _bootstrap_residual_spearman(
            ns[m], pcs[m], log_count[m], seed=g,
        )
        rho_resid.append((int(g), rho_g, int(m.sum()), lo_g, hi_g))

    fig = plt.figure(figsize=(W_TRIPLE, 2.7))
    gs = fig.add_gridspec(
        1, 3, wspace=0.30,
        left=0.05, right=0.985, bottom=0.20, top=0.82,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2], sharey=ax_b)
    plt.setp(ax_c.get_yticklabels(), visible=False)
    ax_c.tick_params(axis="y", which="both", left=True, labelleft=False)

    pc_lo = float(np.floor(pcs.min() / 5.0) * 5.0 - 1.0)
    pc_hi = float(np.ceil(pcs.max() / 5.0) * 5.0 + 1.0)
    ax_b.set_ylim(pc_lo, pc_hi)

    # ----- (a) Stage 1: corpus distribution
    xs = np.array(sorted(counts))
    ys = np.array([counts[n] for n in xs])
    is_h = (xs > 0) & (xs % 100 == 0)
    is_t = (xs > 0) & (xs % 10 == 0) & (xs % 100 != 0)
    is_o = ~(is_h | is_t)
    ax_a.loglog(xs[is_o], ys[is_o], ".", color=C["muted"], ms=1.5,
                alpha=0.55, rasterized=True)
    ax_a.loglog(xs[is_t], ys[is_t], "o", color="#56B4E9", ms=2.6,
                mfc="none", mew=0.5, label=r"$10\,|\,N$")
    ax_a.loglog(xs[is_h], ys[is_h], "s", color="#CC79A7", ms=2.8,
                label=r"$100\,|\,N$")
    ax_a.set_xlabel(r"$N$")
    ax_a.set_ylabel(r"count$(N)$  (Pile)")
    ax_a.set_xlim(0.9, 1100)
    ax_a.grid(True, which="major", ls=":")
    ax_a.legend(loc="lower left", fontsize=6, frameon=False,
                handletextpad=0.4, borderaxespad=0.3)

    # ----- (b) Stage 2: PC1 number-line
    group_handles: list = []
    group_labels: list[str] = []
    for g in sorted(set(grp.tolist())):
        m = grp == g
        h = ax_b.errorbar(ns[m], pcs[m], yerr=psd[m], fmt="o", ms=2.8,
                          color=GROUP_COLORS.get(g, "#000"),
                          elinewidth=0.4, capsize=0, alpha=0.9)
        group_handles.append(h)
        group_labels.append(rf"group ${g}$")
    round_handle = None
    if is_round.any():
        round_handle = ax_b.scatter(
            ns[is_round], pcs[is_round],
            s=44, facecolors="none", edgecolors="#222222",
            linewidths=0.7, zorder=3,
        )
    ax_b.set_xscale("log")
    ax_b.set_xlabel(r"$N$")
    ax_b.set_ylabel(rf"PC1 ({method.upper()})")
    ax_b.set_xlim(0.9, 1100)
    ax_b.grid(True, which="major", ls=":")
    handles = list(group_handles)
    labels = list(group_labels)
    if round_handle is not None:
        handles.append(round_handle)
        labels.append("round")
    ax_b.legend(handles, labels,
                loc="lower right", fontsize=6, frameon=False, ncol=2,
                columnspacing=0.8, handletextpad=0.3, borderaxespad=0.3)

    # ----- (c) Alignment scatter (shares PC1 axis with b)
    for g in sorted(set(grp.tolist())):
        m = grp == g
        if m.sum() < 3:
            continue
        ax_c.scatter(log_count[m], pcs[m], s=14,
                     color=GROUP_COLORS.get(g, "#000"),
                     edgecolors="none", alpha=0.85)
    if is_round.any():
        ax_c.scatter(log_count[is_round], pcs[is_round],
                     s=44, facecolors="none", edgecolors="#222222",
                     linewidths=0.7)
    ax_c.set_xlabel(r"$\log_{10}\,$count$(N)$  (Pile)")
    ax_c.grid(True, which="major", ls=":")

    def _ci_str(lo: float, hi: float) -> str:
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return ""
        marker = "*" if (lo > 0 or hi < 0) else ""
        return rf" $[{lo:+.2f},\,{hi:+.2f}]${marker}"

    rho_lines = [rf"$\rho_S{{=}}{rho_full:+.2f}$  (all)"] + [
        (rf"$\rho_S^{{\mathrm{{r}}}}{{=}}{rho_g:+.2f}$  (g${g}$, $n{{=}}{n_pts}$)"
         + _ci_str(lo_g, hi_g))
        for g, rho_g, n_pts, lo_g, hi_g in rho_resid
    ]
    ax_c.text(
        0.04, 0.04, "\n".join(rho_lines),
        transform=ax_c.transAxes, fontsize=6.5, va="bottom", ha="left",
        linespacing=1.3,
        bbox=dict(facecolor="white", edgecolor="0.6",
                  linewidth=0.4, pad=2.5, alpha=0.92),
    )

    for ax, label in zip(
        (ax_a, ax_b, ax_c),
        ("(a) Stage 1: Pile counts",
         f"(b) Stage 2: {pretty_model} layer {layer}",
         "(c) Stage 1 $\\,\\Leftrightarrow\\,$ Stage 2"),
    ):
        ax.set_title(label, loc="left", fontsize=7.5, pad=3)

    if n_dropped > 0:
        fig.text(
            0.985, 0.005,
            rf"group 4 ($N{{\sim}}10^4$, $n{{=}}{n_dropped}$) omitted from (b,c): "
            r"outside Stage 1 range",
            ha="right", va="bottom", fontsize=5.5, color="0.4",
        )

    _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
# figure 5: PC1 number-line broken down by Stage-1 corpus REGIME
#
# Each N in [1, 1000] is tagged by which Stage-1 null (uniform / Gaussian /
# Zipf) is closest to its empirical Pile count -- exactly the regime mapping
# that produced `10_regime_map.pdf` in Stage 1. We then re-plot the existing
# Stage-2 PC1 number-line, coloured by that tag, to see whether integers
# from different corpus regimes occupy different positions in Pythia's
# residual stream.
# --------------------------------------------------------------------------- #

REGIME_NAMES = {0: "uniform", 1: "Gaussian", 2: "Zipf"}
REGIME_COLORS = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}


def _compute_regimes(
    counts: dict[int, int],
    *,
    n_min: int = 1,
    n_max: int = 1000,
    gauss_mu: float = 500.0,
    gauss_sigma: float = 250.0,
) -> dict[int, int]:
    """Per-N winner among {0=uniform, 1=Gaussian, 2=Zipf} nulls (log-ratio min)."""
    xs = np.array([n for n in sorted(counts) if n_min <= n <= n_max])
    ys = np.array([counts[int(n)] for n in xs], dtype=float)
    n_total = len(xs)
    T = ys.sum()

    uniform = np.full(n_total, T / n_total)
    g = np.exp(-((xs - gauss_mu) ** 2) / (2 * gauss_sigma ** 2))
    gaussian = T * g / g.sum()
    ranks = np.argsort(np.argsort(xs)) + 1
    zipf = T * (1.0 / ranks) / np.sum(1.0 / ranks)

    safe_ys = np.where(ys > 0, ys, 1e-30)
    lr_u = np.log(safe_ys / uniform)
    lr_g = np.log(safe_ys / gaussian)
    lr_z = np.log(safe_ys / zipf)
    stacked = np.abs(np.stack([lr_u, lr_g, lr_z], axis=0))
    winners = np.argmin(stacked, axis=0)
    return {int(n): int(w) for n, w in zip(xs, winners)}


def fig_distribution_breakdown(
    client: MlflowClient,
    stage1_run_id: str,
    projections_json: Path,
    outdir: Path,
    *,
    method: str = "pca",
    layer: int | None = None,
    name: str = "geom_05_distribution_breakdown",
) -> None:
    counts = _pull_corpus_counts(client, stage1_run_id)
    regimes = _compute_regimes(counts)

    blob = _load_projections(projections_json)
    proj_root = blob.get(f"projections_{method}")
    if not proj_root:
        raise SystemExit(
            f"{projections_json} has no projections_{method} block. Re-run "
            f"`src.geometry_cli` with `--save-projections`."
        )

    available = sorted(int(l) for l in proj_root.keys())
    if layer is None:
        best = blob.get(f"best_layer_{method}")
        layer = int(best) if isinstance(best, int) else available[len(available) // 2]
    if layer not in available:
        raise SystemExit(f"layer {layer} not in projections JSON (have {available})")

    pc1_mean, _, group_of = _projections_per_n(proj_root[str(layer)])

    rows: list[dict] = []
    for n, pc in pc1_mean.items():
        if 1 <= n <= 1000 and n in regimes and counts.get(n, 0) > 0:
            rows.append({
                "n": n,
                "log_n": math.log10(n),
                "pc1": pc,
                "regime": regimes[n],
                "group": group_of[n],
                "log_count": math.log10(counts[n]),
            })
    rows.sort(key=lambda r: r["n"])
    if not rows:
        raise SystemExit(
            "no Stage-2 N values overlap Stage 1 range; cannot break down by regime"
        )

    pretty_model = _pretty_model(blob.get("config", {}).get("model_name", "")) or "model"
    n_total = len(rows)

    # ---- regime coverage breakdown of Stage-2 sample
    regime_count = {0: 0, 1: 0, 2: 0}
    for r in rows:
        regime_count[r["regime"]] += 1
    # ---- regime coverage breakdown of full Stage-1 range
    regime_pop = {0: 0, 1: 0, 2: 0}
    for r in regimes.values():
        regime_pop[r] += 1

    # ---- figure: 2 panels side-by-side
    #      (a) PC1 vs N, coloured by regime
    #      (b) Pile count vs N, regime-coloured for context (mirrors Stage 1)
    fig = plt.figure(figsize=(W_DOUBLE, 2.7))
    gs = fig.add_gridspec(
        1, 2, wspace=0.30, left=0.07, right=0.99, bottom=0.20, top=0.84,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---- (a) PC1 number-line, coloured by Stage-1 regime
    for r in [0, 1, 2]:
        pts = [d for d in rows if d["regime"] == r]
        if not pts:
            continue
        xs = np.array([d["log_n"] for d in pts])
        ys = np.array([d["pc1"] for d in pts])
        ax_a.scatter(
            xs, ys, s=22, color=REGIME_COLORS[r],
            edgecolors="none", alpha=0.85,
            label=(f"{REGIME_NAMES[r]}  (n={len(pts)}, "
                   f"corpus pop {regime_pop[r] / sum(regime_pop.values()) * 100:.0f}%)"),
        )
    ax_a.set_xlabel(r"$\log_{10} N$")
    ax_a.set_ylabel(rf"PC1 ({method.upper()})")
    ax_a.grid(True, ls=":")
    ax_a.legend(loc="lower right", fontsize=6, frameon=False,
                handletextpad=0.4, borderaxespad=0.3)
    ax_a.set_title(
        rf"(a) Stage-2 PC1 by corpus regime: {pretty_model} layer {layer}",
        loc="left", fontsize=8.5, pad=4,
    )

    # ---- (b) Stage-1 corpus distribution, regime-coloured (full N range)
    xs_all = np.array(sorted(counts))
    ys_all = np.array([counts[int(n)] for n in xs_all])
    in_range = (xs_all >= 1) & (xs_all <= 1000)
    xs_all, ys_all = xs_all[in_range], ys_all[in_range]
    reg_arr = np.array([regimes.get(int(n), -1) for n in xs_all])
    for r in [0, 1, 2]:
        m = reg_arr == r
        if not m.any():
            continue
        ax_b.loglog(xs_all[m], ys_all[m], ".", color=REGIME_COLORS[r],
                    ms=1.8, alpha=0.65, rasterized=True,
                    label=rf"{REGIME_NAMES[r]}")
    # mark Stage-2 sampled N values as ringed circles
    ringed_x = np.array([r["n"] for r in rows])
    ringed_y = np.array([counts[r["n"]] for r in rows])
    ax_b.loglog(ringed_x, ringed_y, "o", mfc="none", mec="#222", mew=0.5,
                ms=4.2, alpha=0.8, label="Stage-2 sample")
    ax_b.set_xlabel(r"$N$")
    ax_b.set_ylabel(r"count$(N)$  (Pile)")
    ax_b.grid(True, which="major", ls=":")
    ax_b.legend(loc="lower left", fontsize=6, frameon=False,
                handletextpad=0.4, borderaxespad=0.3, ncol=2)
    ax_b.set_title(
        rf"(b) Stage-1 regime map ($n_{{\mathrm{{total}}}}$ overlap = {n_total})",
        loc="left", fontsize=8.5, pad=4,
    )

    _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
# figure 6: per-regime number-line shape (linear vs logarithmic)
# --------------------------------------------------------------------------- #

def _resolve_regime_run(client: MlflowClient, run_id: str | None,
                        experiment_name: str) -> str:
    if run_id:
        return run_id
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(f"experiment {experiment_name!r} not found")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.stage = 'number_geometry_regime'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"no regime probe runs found in {experiment_name!r}")
    return runs[0].info.run_id


def _load_regime_payload(client: MlflowClient, run_id: str, scratch: Path) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    out = client.download_artifacts(run_id, "geometry/regime_probe.json", str(scratch))
    p = Path(out)
    if not p.is_file():
        for c in [scratch / "geometry" / "regime_probe.json",
                  scratch / "regime_probe.json"]:
            if c.exists():
                p = c
                break
    with open(p) as fh:
        return json.load(fh)


def fig_regime_shape(payload: dict, outdir: Path,
                     name: str = "geom_06_regime_shape") -> None:
    """For each regime, scatter PC1 vs N with linear and log fits overlaid.

    Layout: 1 row x 3 cols (uniform / Gaussian / Zipf), shared y-axis (PC1).
    Per-panel annotations: R^2_linear, R^2_log, |Spearman|, n. Below each
    panel, the residuals to the BETTER fit are drawn as a thin strip.
    """
    per_regime = payload["per_regime"]
    short = payload["config"]["model_name"].split("/")[-1]
    target_layer = payload["config"]["target_layer"]

    canonical_order = (
        "uniform", "gaussian", "zipf",
        "exponential", "exponential_decay", "exponential_growth",
    )
    name_order = [r for r in canonical_order if r in per_regime]
    leftover = [k for k in per_regime if k not in name_order]
    name_order = name_order + leftover
    if not name_order:
        raise SystemExit("no regimes in payload")

    n_panels = len(name_order)
    fig_w = max(W_TRIPLE, 3.0 * n_panels)
    fig = plt.figure(figsize=(fig_w, 2.6))
    gs = fig.add_gridspec(
        1, n_panels, wspace=0.18,
        left=0.05, right=0.99, bottom=0.20, top=0.84,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]

    for ax, regime_name in zip(axes, name_order):
        blob = per_regime[regime_name]
        targets = np.array(blob["targets"], dtype=float)
        pc1 = np.array(blob["pc1_scores"], dtype=float)
        order = np.argsort(targets)
        targets = targets[order]
        pc1 = pc1[order]
        log_targets = np.log10(np.maximum(targets, 1.0))

        ax.scatter(targets, pc1, s=10,
                   color=REGIME_COLORS_FIG6[regime_name],
                   edgecolors="none", alpha=0.85)

        # linear fit
        if len(targets) >= 3:
            coef_lin = np.polyfit(targets, pc1, 1)
            xs_lin = np.linspace(targets.min(), targets.max(), 200)
            ax.plot(xs_lin, np.polyval(coef_lin, xs_lin),
                    "-", color="0.30", lw=0.9,
                    label=rf"linear $R^2{{=}}{blob['r2_linear']:.2f}$")

        # log fit
        if len(targets) >= 3:
            coef_log = np.polyfit(log_targets, pc1, 1)
            xs_lin_log = np.linspace(targets.min(), targets.max(), 200)
            xs_log = np.log10(np.maximum(xs_lin_log, 1.0))
            ax.plot(xs_lin_log, np.polyval(coef_log, xs_log),
                    "--", color="0.30", lw=0.9,
                    label=rf"log    $R^2{{=}}{blob['r2_log']:.2f}$")

        ax.set_xlabel(r"$N$")
        ax.set_xlim(targets.min() * 0.9, targets.max() * 1.05)
        ax.grid(True, ls=":")
        ax.legend(loc="best", fontsize=6.5, frameon=False,
                  handletextpad=0.4, borderaxespad=0.3)

        # bootstrap CI on (R^2_log - R^2_linear)
        _, _, gap_pt, gap_lo, gap_hi = _bootstrap_r2_gap(
            np.array(blob["targets"], dtype=float),
            np.array(blob["pc1_scores"], dtype=float),
            seed=hash(regime_name) & 0xFFFF,
        )

        ci_excludes_zero = (
            math.isfinite(gap_lo) and math.isfinite(gap_hi)
            and (gap_lo > 0 or gap_hi < 0)
        )
        if not (np.isnan(blob["r2_linear"]) or np.isnan(blob["r2_log"])):
            sig_marker = "*" if ci_excludes_zero else ""
            verdict = (
                f"log{sig_marker}" if gap_pt > 0.02
                else (f"linear{sig_marker}" if gap_pt < -0.02 else "ambiguous")
            )
        else:
            verdict = "?"

        pretty_name = {
            "exponential_decay":  "Exp. decay",
            "exponential_growth": "Exp. growth",
            "exponential":        "Exponential",
        }.get(regime_name, regime_name.capitalize())
        ax.set_title(
            rf"({chr(97 + name_order.index(regime_name))}) {pretty_name}-class  "
            rf"($n{{=}}{blob['n_samples']}$,  best: {verdict})",
            loc="left", fontsize=7.5, pad=4,
        )

        # In-panel CI annotation (top-right corner)
        if math.isfinite(gap_lo) and math.isfinite(gap_hi):
            ax.text(
                0.97, 0.06,
                rf"$\Delta R^2{{=}}{gap_pt:+.2f}$"
                "\n"
                rf"$95\%\,\mathrm{{CI}}=[{gap_lo:+.2f},\,{gap_hi:+.2f}]$",
                transform=ax.transAxes, fontsize=6.0,
                ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="0.7",
                          linewidth=0.4, pad=2.0, alpha=0.9),
            )

    axes[0].set_ylabel(rf"PC1 (regime-specific PCA, layer {target_layer})")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    fig.suptitle(rf"Per-regime number-line shape: {short}", fontsize=8.5, y=1.00)
    _save(fig, outdir, name)


REGIME_COLORS_FIG6 = {
    "uniform":             "#0072B2",
    "gaussian":            "#D55E00",
    "zipf":                "#009E73",
    "exponential":         "#CC79A7",
    "exponential_decay":   "#CC79A7",
    "exponential_growth":  "#E69F00",
}


def _resolve_synth_run(client: MlflowClient, run_id: str | None,
                       experiment_name: str) -> str:
    if run_id:
        return run_id
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(f"experiment {experiment_name!r} not found")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.stage = 'number_geometry_synth'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"no synth-distribution runs found in {experiment_name!r}")
    return runs[0].info.run_id


def _load_synth_payload(client: MlflowClient, run_id: str, scratch: Path) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    out = client.download_artifacts(run_id, "geometry/synth_probe.json", str(scratch))
    p = Path(out)
    if not p.is_file():
        for c in [scratch / "geometry" / "synth_probe.json",
                  scratch / "synth_probe.json"]:
            if c.exists():
                p = c
                break
    with open(p) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #

def _maybe_download_projections(client: MlflowClient, run_id: str,
                                local_dir: Path) -> Path | None:
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        out = client.download_artifacts(run_id, "geometry/results.json", str(local_dir))
        candidate = Path(out)
        if candidate.is_file():
            return candidate
        # some MLflow versions return a directory; locate the JSON inside it
        for c in [
            local_dir / "geometry" / "results.json",
            local_dir / "results.json",
        ]:
            if c.exists():
                return c
    except Exception as e:
        print(f"  could not pull geometry/results.json from MLflow: {e}")
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Render figures for the number-line geometry probe. Pulls metrics "
            "from MLflow, plus an optional projections JSON for the PC1 "
            "scatter. Mirrors `reference/llm_natural_log/Figure_4.ipynb` and "
            "`reference/llm_natural_log/utils/visual_utils.py`."
        )
    )
    p.add_argument("--run-id", default=None,
                   help="MLflow run id for the per-layer profile + PC1 scatter")
    p.add_argument("--experiment-name", default="numberline_geometry_expv1",
                   help="MLflow experiment name for the cross-model sweep")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--results-dir", default="results/figs")
    p.add_argument("--projections-json", default=None,
                   help="Path to a JSON file containing projections_pca/_pls "
                        "(if absent, will be pulled from the run's MLflow "
                        "artifacts under geometry/results.json)")
    p.add_argument("--sweep-param", default="num_examples",
                   help="MLflow param name to use as the x-axis in the sweep "
                        "figure (default: num_examples, matches Figure 4)")
    p.add_argument("--scatter-layers", type=int, nargs="*", default=None,
                   help="layer indices for the PC1 scatter (default: best PCA)")
    p.add_argument("--scatter-method", choices=["pca", "pls"], default="pca")
    p.add_argument("--skip-sweep", action="store_true",
                   help="don't render the cross-model sweep figure")
    p.add_argument("--skip-scatter", action="store_true",
                   help="don't render the PC1 scatter")
    p.add_argument("--stage1-run-id", default=None,
                   help="MLflow run id for the Stage 1 corpus counts; if "
                        "absent, picks the latest run under "
                        "--stage1-experiment-name")
    p.add_argument("--stage1-experiment-name", default="numberline_freq_count_expv1",
                   help="Stage 1 experiment to look up the corpus run in")
    p.add_argument("--skip-alignment", action="store_true",
                   help="don't render the Stage 1 / Stage 2 alignment figure")
    p.add_argument("--skip-regime", action="store_true",
                   help="don't render the per-regime breakdown figure")
    p.add_argument("--skip-regime-shape", action="store_true",
                   help="don't render the regime-shape (linear vs log) figure")
    p.add_argument("--regime-run-id", default=None,
                   help="MLflow run id of a regime-probe run (latest if absent)")
    p.add_argument("--skip-synth", action="store_true",
                   help="don't render the synthetic-distribution probe figure")
    p.add_argument("--synth-run-id", default=None,
                   help="MLflow run id of a synth-distribution run (latest if absent)")
    p.add_argument("--compare-models", nargs="*", default=None,
                   help="render a PCA layer comparison/table for these model ids")
    p.add_argument("--comparison-run-ids", nargs="*", default=None,
                   help="optional run ids aligned with --compare-models")
    p.add_argument("--comparison-labels", nargs="*", default=None,
                   help="optional display labels aligned with --compare-models")
    p.add_argument("--comparison-name", default="geom_08_pca_model_comparison",
                   help="output basename for the PCA comparison/table figure")
    p.add_argument("--only-comparison", action="store_true",
                   help="render only --compare-models and skip the default figures")
    args = p.parse_args(argv)

    _apply_neurips_style()
    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = _connect(args.tracking_uri)

    if args.compare_models:
        fig_pca_model_comparison(
            client,
            args.experiment_name,
            args.compare_models,
            outdir,
            run_ids=args.comparison_run_ids,
            labels=args.comparison_labels,
            name=args.comparison_name,
        )
        if args.only_comparison:
            print("[geometry-fig] done.")
            return 0

    run_id = _resolve_run(
        client, args.run_id, args.experiment_name,
        filter_string="tags.stage = 'number_geometry'",
    )
    print(f"[geometry-fig] run_id = {run_id}")

    fig_layer_profile(client, run_id, outdir)

    needs_proj = not (args.skip_scatter and args.skip_alignment and args.skip_regime)
    proj_path = Path(args.projections_json) if args.projections_json else None
    if proj_path is None and needs_proj:
        proj_path = _maybe_download_projections(client, run_id, outdir / "_artifacts")

    if not args.skip_scatter:
        if proj_path is not None and proj_path.exists():
            fig_pc1_scatter(
                proj_path, outdir,
                method=args.scatter_method,
                layers=args.scatter_layers,
            )
        else:
            print(
                "[geometry-fig] no projections JSON available; skipping "
                "PC1 scatter. Re-run with `src.geometry_cli "
                "--save-projections` to capture them."
            )

    if not args.skip_sweep:
        try:
            fig_demos_sweep(
                client, args.experiment_name, outdir,
                sweep_param=args.sweep_param,
                metric_prefix="pca",
                name="geom_02_pca_vs_" + args.sweep_param,
            )
        except SystemExit as e:
            print(f"[geometry-fig] skipping sweep figure: {e}")

    stage1_run_id = None
    if (not args.skip_alignment) or (not args.skip_regime):
        if proj_path is not None and proj_path.exists():
            stage1_run_id = args.stage1_run_id or _resolve_run(
                client, None, args.stage1_experiment_name,
                filter_string="tags.stage = 'number_frequency'",
            )

    if not args.skip_alignment:
        if proj_path is None or not proj_path.exists():
            print(
                "[geometry-fig] no projections JSON; skipping Stage 1 "
                "alignment figure."
            )
        else:
            try:
                fig_stage1_alignment(
                    client,
                    stage1_run_id=stage1_run_id,
                    projections_json=proj_path,
                    outdir=outdir,
                    method=args.scatter_method,
                )
            except SystemExit as e:
                print(f"[geometry-fig] skipping alignment figure: {e}")

    if not args.skip_regime:
        if proj_path is None or not proj_path.exists() or stage1_run_id is None:
            print(
                "[geometry-fig] no projections JSON or Stage 1 run; "
                "skipping regime-breakdown figure."
            )
        else:
            try:
                fig_distribution_breakdown(
                    client,
                    stage1_run_id=stage1_run_id,
                    projections_json=proj_path,
                    outdir=outdir,
                    method=args.scatter_method,
                )
            except SystemExit as e:
                print(f"[geometry-fig] skipping regime breakdown: {e}")

    if not args.skip_regime_shape:
        try:
            regime_run_id = _resolve_regime_run(
                client, args.regime_run_id, args.experiment_name,
            )
            print(f"[geometry-fig] regime probe run_id = {regime_run_id}")
            payload = _load_regime_payload(
                client, regime_run_id, outdir / "_artifacts_regime",
            )
            fig_regime_shape(payload, outdir)
        except SystemExit as e:
            print(f"[geometry-fig] skipping regime-shape figure: {e}")

    if not args.skip_synth:
        try:
            synth_run_id = _resolve_synth_run(
                client, args.synth_run_id, args.experiment_name,
            )
            print(f"[geometry-fig] synth probe run_id = {synth_run_id}")
            payload = _load_synth_payload(
                client, synth_run_id, outdir / "_artifacts_synth",
            )
            fig_regime_shape(payload, outdir, name="geom_07_synth_distributions")
        except SystemExit as e:
            print(f"[geometry-fig] skipping synth-distribution figure: {e}")

    print("[geometry-fig] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
