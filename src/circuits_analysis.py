"""
Figures for Stage 3 (mech-interp circuit attribution).

Three NeurIPS-style figures, mirroring `src/geometry_analysis.py`:

  fig_head_heatmap     (layer x head) heatmap of correlation between each
                       head's PC1-projected output and log10(target). The
                       canonical "which heads are number heads" plot.

  fig_mlp_profile      Per-layer MLP correlation with log10(target). Tells you
                       at which depth the magnitude code is *written* by MLP
                       blocks vs attention.

  fig_top_heads        For the top-K heads by |corr|, scatter the head's PC1
                       projection vs log10(target) with a within-group fit --
                       confirms the correlation is monotone, not bimodal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mlflow.tracking import MlflowClient

import mlflow

W_SINGLE = 3.3
W_DOUBLE = 6.8
W_TRIPLE = 9.6


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
        "xtick.direction": "in",
        "ytick.direction": "in",
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


def _connect(tracking_uri: str | None) -> MlflowClient:
    if tracking_uri is None:
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{os.path.abspath('mlflow.db')}",
        )
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def _resolve_run(client: MlflowClient, run_id: str | None,
                 experiment_name: str) -> str:
    if run_id:
        return run_id
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(f"experiment {experiment_name!r} not found")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.stage = 'number_circuits'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"no Stage 3 runs found in {experiment_name!r}")
    return runs[0].info.run_id


def _load_payload(client: MlflowClient, run_id: str, scratch: Path) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    out = client.download_artifacts(run_id, "circuits/results.json", str(scratch))
    p = Path(out)
    if not p.is_file():
        for c in [scratch / "circuits" / "results.json", scratch / "results.json"]:
            if c.exists():
                p = c
                break
    with open(p) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# figure 1: head heatmap
# --------------------------------------------------------------------------- #

def fig_head_heatmap(payload: dict, outdir: Path,
                     name: str = "circ_01_head_heatmap") -> None:
    head_corr = np.array(payload["head_corr_log_target"])              # (L+1, n_heads)
    L = head_corr.shape[0] - 1
    n_heads = head_corr.shape[1]
    model_name = payload["config"]["model_name"]
    short = model_name.split("/")[-1]

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.6))
    vmax = max(0.2, float(np.nanmax(np.abs(head_corr))))
    im = ax.imshow(
        head_corr.T, aspect="auto", origin="lower",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("head index")
    ax.set_xticks(np.arange(L + 1))
    ax.set_xticklabels([str(l) for l in range(L + 1)], fontsize=6)
    head_ticks = np.arange(0, n_heads, max(1, n_heads // 8))
    ax.set_yticks(head_ticks)
    ax.set_yticklabels([str(h) for h in head_ticks], fontsize=6)

    # mark top-3 heads
    abs_c = np.abs(head_corr)
    flat = abs_c.flatten()
    top3 = np.argsort(-flat)[:3]
    for rank, idx in enumerate(top3):
        l, h = np.unravel_index(int(idx), head_corr.shape)
        ax.plot(l, h, marker="o", mfc="none", mec="black", mew=0.8, ms=8.5,
                zorder=3)
        ax.text(l + 0.35, h, f"#{rank + 1}", fontsize=6.5, color="black",
                va="center", zorder=4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label(r"corr$(\langle h_{\ell,i}, d_{\mathrm{PC1}}\rangle,\ \log_{10} N)$",
                   fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title(
        rf"(a) Per-head contribution to PC1 at layer {L}: {short}",
        loc="left", fontsize=8.5, pad=4,
    )
    _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
# figure 2: MLP per-layer profile
# --------------------------------------------------------------------------- #

def fig_mlp_profile(payload: dict, outdir: Path,
                    name: str = "circ_02_mlp_profile") -> None:
    mlp_corr = np.array(payload["mlp_corr_log_target"])
    head_corr = np.array(payload["head_corr_log_target"])
    L = mlp_corr.shape[0] - 1
    short = payload["config"]["model_name"].split("/")[-1]

    layers = np.arange(L + 1)
    head_max_per_layer = np.max(np.abs(head_corr), axis=1)

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.4))
    ax.plot(layers, mlp_corr, "-o", color="#D55E00", lw=1.2, ms=3.4,
            label=r"$\mathrm{MLP}_\ell\ \to\ \log_{10}N$")
    ax.plot(layers, head_max_per_layer, "--s", color="#0072B2", lw=1.0, ms=3.0,
            label=r"$\max_i |\,\mathrm{head}_{\ell,i}\to \log_{10}N\,|$")
    ax.axhline(0, color="0.6", lw=0.4)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"corr with $\log_{10} N$")
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], fontsize=6)
    ax.grid(True, ls=":")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    ax.set_title(
        rf"(b) MLP and best-head contribution by layer: {short}",
        loc="left", fontsize=8.5, pad=4,
    )
    _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
# figure 3: top-K head scatter
# --------------------------------------------------------------------------- #

def fig_top_heads(payload: dict, outdir: Path, *, k: int = 4,
                  name: str = "circ_03_top_heads") -> None:
    head_proj = np.array(payload["head_proj"])                  # (n, L+1, n_heads)
    head_corr = np.array(payload["head_corr_log_target"])
    log_targets = np.array(payload["log_targets"])
    targets = np.array(payload["targets"])
    short = payload["config"]["model_name"].split("/")[-1]

    abs_c = np.abs(head_corr)
    flat = abs_c.flatten()
    top_idx = np.argsort(-flat)[:k]

    fig, axes = plt.subplots(1, k, figsize=(W_TRIPLE, 2.2),
                             sharey=False)
    if k == 1:
        axes = [axes]

    for rank, (ax, idx) in enumerate(zip(axes, top_idx)):
        l, h = np.unravel_index(int(idx), head_corr.shape)
        x = log_targets
        y = head_proj[:, l, h]
        # color by group based on order of magnitude of target (1..4)
        groups = np.clip(np.round(np.log10(np.clip(targets, 1, None))).astype(int), 1, 4)
        colors_map = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#CC79A7"}
        for g in sorted(set(groups.tolist())):
            m = groups == g
            ax.scatter(x[m], y[m], s=10, color=colors_map.get(g, "#000"),
                       edgecolors="none", alpha=0.85, label=rf"g${g}$")
        if len(x) >= 3:
            coef = np.polyfit(x, y, 1)
            xs_fit = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs_fit, np.polyval(coef, xs_fit), "-", color="0.25",
                    lw=0.7, alpha=0.7)
        ax.set_xlabel(r"$\log_{10} N$")
        if rank == 0:
            ax.set_ylabel(r"$\langle \mathrm{head}_{\ell,i},\ d_{\mathrm{PC1}}\rangle$")
        ax.grid(True, ls=":")
        r_val = float(head_corr[l, h])
        ax.set_title(
            rf"#${rank + 1}$  L${l}$ h${h}$  $r{{=}}{r_val:+.2f}$",
            loc="left", fontsize=7.5, pad=3,
        )
        if rank == 0:
            ax.legend(loc="best", fontsize=6, frameon=False, ncol=2,
                      handletextpad=0.3, columnspacing=0.8)

    fig.suptitle(rf"(c) Top-$K$ heads driving PC1 in {short}",
                 fontsize=8.5, y=1.04)
    fig.subplots_adjust(wspace=0.30)
    _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
# figures 4-5: activation-patching (causal intervention)
# --------------------------------------------------------------------------- #

def _resolve_patching_run(client: MlflowClient, run_id: str | None,
                          experiment_name: str) -> str:
    if run_id:
        return run_id
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(f"experiment {experiment_name!r} not found")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.stage = 'number_circuits_patching'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"no patching runs found in {experiment_name!r}")
    return runs[0].info.run_id


def _load_patching_payload(client: MlflowClient, run_id: str, scratch: Path) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    out = client.download_artifacts(run_id, "circuits/patching.json", str(scratch))
    p = Path(out)
    if not p.is_file():
        for c in [scratch / "circuits" / "patching.json", scratch / "patching.json"]:
            if c.exists():
                p = c
                break
    with open(p) as fh:
        return json.load(fh)


def fig_patching_heatmap(payload: dict, outdir: Path,
                         name: str = "circ_04_patching_heatmap") -> None:
    head_eff = np.array(payload["head_effect_mean"])              # (L+1, n_heads)
    L = head_eff.shape[0] - 1
    n_heads = head_eff.shape[1]
    short = payload["config"]["model_name"].split("/")[-1]
    n_pairs = len(payload["donor_pc1"])

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.6))
    vmax = max(0.1, float(np.nanmax(np.abs(head_eff))))
    im = ax.imshow(
        head_eff.T, aspect="auto", origin="lower",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("head index")
    ax.set_xticks(np.arange(L + 1))
    ax.set_xticklabels([str(l) for l in range(L + 1)], fontsize=6)
    head_ticks = np.arange(0, n_heads, max(1, n_heads // 8))
    ax.set_yticks(head_ticks)
    ax.set_yticklabels([str(h) for h in head_ticks], fontsize=6)

    abs_e = np.abs(head_eff)
    flat = abs_e.flatten()
    top3 = np.argsort(-flat)[:3]
    for rank, idx in enumerate(top3):
        l, h = np.unravel_index(int(idx), head_eff.shape)
        ax.plot(l, h, marker="o", mfc="none", mec="black", mew=0.8, ms=8.5,
                zorder=3)
        ax.text(l + 0.35, h, f"#{rank + 1}", fontsize=6.5, color="black",
                va="center", zorder=4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label(
        r"causal effect: $(\mathrm{PC1}_{\mathrm{patched}} - \mathrm{PC1}_{\mathrm{recv}})\,/\,(\mathrm{PC1}_{\mathrm{donor}} - \mathrm{PC1}_{\mathrm{recv}})$",
        fontsize=6.5,
    )
    cbar.ax.tick_params(labelsize=6)

    ax.set_title(
        rf"(a) Per-head causal patching effect at layer {L}: "
        rf"{short}  ($n_{{\mathrm{{pairs}}}}={n_pairs}$)",
        loc="left", fontsize=8.5, pad=4,
    )
    _save(fig, outdir, name)


def fig_patching_mlp(payload: dict, outdir: Path,
                     name: str = "circ_05_patching_mlp") -> None:
    mlp_eff = np.array(payload["mlp_effect_mean"])
    mlp_std = np.array(payload["mlp_effect_std"])
    head_eff = np.array(payload["head_effect_mean"])
    L = mlp_eff.shape[0] - 1
    n_pairs = len(payload["donor_pc1"])
    short = payload["config"]["model_name"].split("/")[-1]

    layers = np.arange(L + 1)
    head_max_per_layer = np.max(np.abs(head_eff), axis=1)

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.4))
    ax.errorbar(layers, mlp_eff, yerr=mlp_std / max(1, n_pairs) ** 0.5,
                fmt="-o", color="#D55E00", lw=1.2, ms=3.4, elinewidth=0.5,
                capsize=2,
                label=r"patch $\mathrm{MLP}_\ell$  (causal effect $\to$ PC1)")
    ax.plot(layers, head_max_per_layer, "--s", color="#0072B2", lw=1.0, ms=3.0,
            label=r"$\max_i |\,\mathrm{patch\ head}_{\ell,i}\,|$")
    ax.axhline(0, color="0.6", lw=0.4)
    ax.axhline(1, color="0.4", lw=0.4, ls="--", alpha=0.7)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"normalised patching effect on $\mathrm{PC1}_L$")
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], fontsize=6)
    ax.grid(True, ls=":")
    ax.legend(loc="best", fontsize=6.5, frameon=False)
    ax.set_title(
        rf"(b) Causal patching by layer: {short}  ($n_{{\mathrm{{pairs}}}}={n_pairs}$)",
        loc="left", fontsize=8.5, pad=4,
    )
    _save(fig, outdir, name)


def fig_patching_correlation_compare(
    payload_corr: dict, payload_patch: dict, outdir: Path,
    name: str = "circ_06_corr_vs_causal",
) -> None:
    """Scatter: correlation (Stage 3a) vs causal effect (Stage 3b) for every head.

    Heads near y=x are 'honest' magnitude heads (correlated and causal); heads
    high on y but low on x are causal but not directly correlated (downstream
    consumers); heads high on x but low on y are spurious correlations.
    """
    head_corr = np.array(payload_corr["head_corr_log_target"])
    head_eff = np.array(payload_patch["head_effect_mean"])
    if head_corr.shape != head_eff.shape:
        L = min(head_corr.shape[0], head_eff.shape[0])
        head_corr = head_corr[:L]
        head_eff = head_eff[:L]
    L = head_corr.shape[0]
    n_heads = head_corr.shape[1]

    fig, ax = plt.subplots(figsize=(W_SINGLE * 1.4, W_SINGLE * 1.2))
    cmap = plt.get_cmap("viridis")
    layers = np.arange(L)
    norm = mpl.colors.Normalize(vmin=0, vmax=L - 1)
    for l in range(L):
        ax.scatter(head_corr[l], head_eff[l], s=14, color=cmap(norm(l)),
                   alpha=0.85, edgecolors="none")

    abs_eff = np.abs(head_eff)
    top = np.argsort(-abs_eff.flatten())[:5]
    for rank, idx in enumerate(top):
        l, h = np.unravel_index(int(idx), head_eff.shape)
        ax.annotate(
            rf"L{l}h{h}",
            (head_corr[l, h], head_eff[l, h]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=6.5,
        )

    lo = min(float(head_corr.min()), float(head_eff.min())) - 0.05
    hi = max(float(head_corr.max()), float(head_eff.max())) + 0.05
    ax.plot([lo, hi], [lo, hi], "-", color="0.5", lw=0.5, alpha=0.7)
    ax.axhline(0, color="0.7", lw=0.3)
    ax.axvline(0, color="0.7", lw=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"correlation $r$ (Stage 3a)")
    ax.set_ylabel(r"causal effect (Stage 3b)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("layer", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title("(c) Correlational vs causal head attribution",
                 loc="left", fontsize=8.5, pad=3)
    _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Render Stage 3 (circuit attribution) figures. Pulls per-prompt "
            "head/MLP projections from a Modal MLflow run and builds the "
            "head heatmap, MLP profile, and top-K head trace."
        )
    )
    p.add_argument("--run-id", default=None,
                   help="MLflow run id of the circuits run (latest if absent)")
    p.add_argument("--experiment-name", default="numberline_circuits_expv1")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--results-dir", default="results/figs/circuits")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--payload-json", default=None,
                   help="local path to a circuits/results.json (skips MLflow)")
    p.add_argument("--patching-run-id", default=None,
                   help="MLflow run id of a patching run (latest if absent)")
    p.add_argument("--patching-payload-json", default=None,
                   help="local path to a circuits/patching.json")
    p.add_argument("--skip-correlational", action="store_true")
    p.add_argument("--skip-patching", action="store_true")
    args = p.parse_args(argv)

    _apply_neurips_style()
    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = _connect(args.tracking_uri) if not (
        args.payload_json and args.patching_payload_json
    ) else None

    payload_corr = None
    if not args.skip_correlational:
        if args.payload_json:
            with open(args.payload_json) as fh:
                payload_corr = json.load(fh)
            print(f"[circuits-fig] using local correlational payload {args.payload_json}")
        else:
            run_id = _resolve_run(client, args.run_id, args.experiment_name)
            print(f"[circuits-fig] correlational run_id = {run_id}")
            payload_corr = _load_payload(client, run_id, outdir / "_artifacts")

        fig_head_heatmap(payload_corr, outdir)
        fig_mlp_profile(payload_corr, outdir)
        fig_top_heads(payload_corr, outdir, k=args.top_k)

    payload_patch = None
    if not args.skip_patching:
        if args.patching_payload_json:
            with open(args.patching_payload_json) as fh:
                payload_patch = json.load(fh)
            print(f"[circuits-fig] using local patching payload {args.patching_payload_json}")
        else:
            try:
                run_id_p = _resolve_patching_run(client, args.patching_run_id, args.experiment_name)
                print(f"[circuits-fig] patching run_id = {run_id_p}")
                payload_patch = _load_patching_payload(
                    client, run_id_p, outdir / "_artifacts_patch"
                )
            except SystemExit as e:
                print(f"[circuits-fig] no patching run available, skipping: {e}")
                payload_patch = None

        if payload_patch is not None:
            fig_patching_heatmap(payload_patch, outdir)
            fig_patching_mlp(payload_patch, outdir)
            if payload_corr is not None:
                fig_patching_correlation_compare(payload_corr, payload_patch, outdir)

    print("[circuits-fig] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
