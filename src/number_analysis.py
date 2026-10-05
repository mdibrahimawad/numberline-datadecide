from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from mlflow.tracking import MlflowClient

import mlflow


NUMBER_MIN = 0
NUMBER_MAX = 1000
CORPUS_LABEL = "corpus"

W_SINGLE = 3.3
W_DOUBLE = 6.8

C = {
    "empirical": "#000000",
    "uniform":   "#0072B2",
    "gaussian":  "#D55E00",
    "zipf":      "#009E73",
    "fit":       "#E69F00",
    "accent1":   "#CC79A7",
    "accent2":   "#56B4E9",
    "muted":     "#BBBBBB",
}


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


def _pull_counts(
    run_id: str,
    tracking_uri: str | None,
    n_min: int = NUMBER_MIN,
    n_max: int = NUMBER_MAX,
) -> dict[int, int]:
    if tracking_uri is None:
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{os.path.abspath('mlflow.db')}",
        )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    out: dict[int, int] = {}
    pad = max(4, len(str(n_max)))
    for n in range(n_min, n_max + 1):
        key = f"count_N_{n:0{pad}d}"
        if key in metrics:
            out[n] = int(metrics[key])
    if not out:
        raise SystemExit(f"no count_N metrics found on run {run_id} for [{n_min},{n_max}]")
    return out


def _read_counts_csv(path: Path, n_min: int, n_max: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"number", "count"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SystemExit(f"{path} must contain columns: number,count")
        for row in reader:
            n = int(row["number"])
            if n_min <= n <= n_max:
                counts[n] = int(row["count"])
    if not counts:
        raise SystemExit(f"no counts found in {path} for [{n_min},{n_max}]")
    return {n: counts.get(n, 0) for n in range(n_min, n_max + 1)}


def _null_gaussian_params(xs: np.ndarray) -> tuple[float, float]:
    lo = float(np.min(xs))
    hi = float(np.max(xs))
    return (lo + hi) / 2.0, max((hi - lo) / 4.0, 1.0)


def _inset_regions(xs: np.ndarray) -> list[tuple[int, int]]:
    lo = int(np.min(xs))
    hi = int(np.max(xs))
    span = max(hi - lo, 1)
    regions = [
        (lo + max(1, int(0.05 * span)), lo + max(2, int(0.25 * span))),
        (lo + max(2, int(0.50 * span)), hi),
    ]
    return [(a, max(a + 1, b)) for a, b in regions]


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {outdir}/{name}.pdf + .png")


def fig_linear(counts, outdir):
    xs = np.array(sorted(counts))
    ys = np.array([counts[n] for n in xs])
    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.1))
    ax.plot(xs, ys, color=C["empirical"], lw=0.8)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"count$(N)$")
    ax.set_title(f"Integer frequency in {CORPUS_LABEL}")
    ax.grid(True, which="major", ls=":")
    _save(fig, outdir, "01_linear")


def fig_semilog(counts, outdir):
    xs = np.array(sorted(counts))
    ys = np.array([counts[n] for n in xs])
    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.1))
    ax.semilogy(xs, ys, color=C["empirical"], lw=0.8)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"count$(N)$")
    ax.set_title("Integer frequency, logarithmic ordinate")
    ax.grid(True, which="both", ls=":")
    _save(fig, outdir, "02_semilogy")


def _fit_loglog(xs, ys):
    lx = np.log(xs)
    ly = np.log(ys)
    slope, intercept = np.polyfit(lx, ly, 1)
    y_hat = slope * lx + intercept
    ss_res = np.sum((ly - y_hat) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return slope, intercept, r2


def fig_loglog(counts, outdir):
    items = [(n, c) for n, c in counts.items() if n > 0 and c > 0]
    xs = np.array([n for n, _ in items])
    ys = np.array([c for _, c in items])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    slope, intercept, r2 = _fit_loglog(xs, ys)
    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.4))
    ax.loglog(xs, ys, ".", color=C["empirical"], ms=1.8, alpha=0.55,
              rasterized=True, label="empirical")
    xline = np.array([xs.min(), xs.max()])
    ax.loglog(xline, np.exp(intercept + slope * np.log(xline)),
              "--", color=C["fit"], lw=1.1,
              label=rf"power-law fit, $\alpha={-slope:.3f}$, $R^2={r2:.3f}$")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"count$(N)$")
    ax.set_title("Power-law fit in value space")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper right")
    _save(fig, outdir, "03_loglog")


def fig_zipf_rank(counts, outdir):
    vals = np.array(sorted((c for c in counts.values() if c > 0), reverse=True))
    ranks = np.arange(1, len(vals) + 1)
    slope, intercept, r2 = _fit_loglog(ranks, vals)
    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.4))
    ax.loglog(ranks, vals, ".", color=C["empirical"], ms=1.8, alpha=0.65,
              rasterized=True, label="empirical")
    xline = np.array([ranks[0], ranks[-1]])
    ax.loglog(xline, np.exp(intercept + slope * np.log(xline)),
              "--", color=C["fit"], lw=1.1,
              label=rf"Zipf fit, $s={-slope:.3f}$, $R^2={r2:.3f}$")
    ax.set_xlabel(r"rank$(N)$")
    ax.set_ylabel(r"count$(N)$")
    ax.set_title("Zipf rank--frequency")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper right")
    _save(fig, outdir, "04_zipf_rank_frequency")


def fig_top_bars(counts, outdir, k=30):
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]
    labels = [str(n) for n, _ in top]
    vals = [c for _, c in top]
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.1))
    ax.bar(range(k), vals, color=C["empirical"], width=0.78)
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("count (log)")
    ax.set_title(f"{k} most frequent integers")
    ax.grid(True, axis="y", which="both", ls=":")
    _save(fig, outdir, "05_top30_bars")


def fig_round_numbers(counts, outdir):
    xs = np.array(sorted(counts))
    ys = np.array([counts[n] for n in xs])
    is_hundred = (xs > 0) & (xs % 100 == 0)
    is_ten = (xs > 0) & (xs % 10 == 0) & (xs % 100 != 0)
    is_other = ~(is_hundred | is_ten)

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.4))
    ax.semilogy(xs[is_other], ys[is_other], ".",
                color=C["muted"], ms=2.0, alpha=0.55, rasterized=True,
                label="other")
    ax.semilogy(xs[is_ten], ys[is_ten], "o",
                color=C["accent2"], ms=4.0, mfc="none", mew=0.7,
                label=r"$\{N : 10 | N,\ 100 \nmid N\}$")
    ax.semilogy(xs[is_hundred], ys[is_hundred], "s",
                color=C["accent1"], ms=5.0, mfc=C["accent1"],
                label=r"$\{N : 100 | N\}$")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"count$(N)$")
    ax.set_title("Integer frequency by roundness")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="lower left", ncol=1)
    _save(fig, outdir, "06_round_numbers")


def fig_leading_digit(counts, outdir):
    benford = np.array([math.log10(1 + 1 / d) for d in range(1, 10)])
    emp = np.zeros(9)
    for n, c in counts.items():
        if n <= 0:
            continue
        d = int(str(n)[0])
        if 1 <= d <= 9:
            emp[d - 1] += c
    emp_p = emp / emp.sum()

    resid = emp_p - benford
    kl = float(np.sum(emp_p * np.log(emp_p / benford)))

    digits = np.arange(1, 10)
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(W_SINGLE, 2.9), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
    )

    ax.bar(digits, emp_p, width=0.72, color=C["empirical"],
           label=rf"empirical, $D_{{\mathrm{{KL}}}}={kl:.3f}$")
    ax.plot(digits, benford, "o", color=C["fit"], ms=4.5,
            mec=C["fit"], mfc="white", mew=1.0,
            label=r"Benford, $p(d)=\log_{10}(1+1/d)$")
    for d, p in zip(digits, emp_p):
        ax.text(d, p + 0.005, f"{p*100:.1f}", ha="center", va="bottom",
                fontsize=6, color=C["empirical"])
    ax.set_ylabel("proportion")
    ax.set_ylim(0, max(emp_p.max(), benford.max()) * 1.18)
    ax.set_title("Leading-digit distribution vs Benford")
    ax.grid(True, axis="y", ls=":")
    ax.legend(loc="upper right")

    axr.axhline(0, color="0.35", lw=0.5)
    colors = [C["fit"] if v < 0 else C["empirical"] for v in resid]
    axr.bar(digits, resid, width=0.72, color=colors)
    axr.set_xticks(digits)
    axr.set_xlabel("leading digit $d$")
    axr.set_ylabel(r"$p_{\mathrm{emp}}-p_{B}$")
    axr.grid(True, axis="y", ls=":")
    _save(fig, outdir, "07_leading_digit_benford")


def _kl(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.where(p > 0, p / p.sum(), 0)
    q = np.where(q > 0, q / q.sum(), 1e-300)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def fig_baseline_overlay(counts, outdir):
    xs = np.array(sorted(counts))
    ys = np.array([counts[n] for n in xs], dtype=float)
    T = ys.sum()
    N = len(xs)

    uniform = np.full(N, T / N)
    mu, sigma = _null_gaussian_params(xs)
    g = np.exp(-((xs - mu) ** 2) / (2 * sigma ** 2))
    gaussian = T * g / g.sum()
    ranks_by_x = np.argsort(np.argsort(xs)) + 1
    z = 1.0 / ranks_by_x
    zipf = T * z / z.sum()

    kl_u = _kl(ys, uniform)
    kl_g = _kl(ys, gaussian)
    kl_z = _kl(ys, zipf)

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.8), sharex=False)

    def _plot_all(ax, lw_emp=1.1, lw_base=1.1, labels=True):
        ax.plot(xs, ys, "-", color=C["empirical"], lw=lw_emp,
                label="empirical" if labels else None)
        ax.plot(xs, uniform, ":", color=C["uniform"], lw=lw_base,
                label=(rf"uniform  $D_{{\mathrm{{KL}}}}={kl_u:.3f}$" if labels else None))
        ax.plot(xs, gaussian, "--", color=C["gaussian"], lw=lw_base,
                label=(rf"Gaussian$(\mu\!=\!{mu:.0f},\sigma\!=\!{sigma:.0f})$  "
                       rf"$D_{{\mathrm{{KL}}}}={kl_g:.3f}$" if labels else None))
        ax.plot(xs, zipf, "-.", color=C["zipf"], lw=lw_base,
                label=(rf"Zipf $\propto 1/\mathrm{{rank}}(N)$  "
                       rf"$D_{{\mathrm{{KL}}}}={kl_z:.3f}$" if labels else None))

    for ax, yscale, tag in zip(axes, ["linear", "log"], ["(a) linear", "(b) log-$y$"]):
        _plot_all(ax, labels=True)
        ax.set_xlabel(r"$N$")
        ax.set_yscale(yscale)
        ax.set_title(tag, loc="left")
        ax.grid(True, which="both", ls=":")
    axes[0].set_ylabel(r"count$(N)$")
    axes[1].legend(loc="upper right", ncol=1)

    ax_lin = axes[0]

    def _mk_inset(ax_host, region, bbox, yfactor=1.0):
        lo, hi = region
        mask = (xs >= lo) & (xs <= hi)
        ymax = float(ys[mask].max()) * yfactor
        axin = inset_axes(
            ax_host, width="100%", height="100%",
            bbox_to_anchor=bbox, bbox_transform=ax_host.transAxes,
            borderpad=0,
        )
        _plot_all(axin, lw_emp=0.9, lw_base=0.9, labels=False)
        axin.set_xlim(lo, hi)
        axin.set_ylim(0, ymax)
        axin.tick_params(axis="both", which="major", labelsize=5.5,
                         length=2.0, pad=1.5)
        axin.tick_params(axis="both", which="minor", length=1.2)
        axin.set_xticks([lo, (lo + hi) // 2, hi])
        for s in axin.spines.values():
            s.set_linewidth(0.5)
        axin.grid(True, ls=":", alpha=0.35)
        mark_inset(ax_host, axin, loc1=2, loc2=4,
                   fc="none", ec="0.4", lw=0.5, ls="--")
        return axin

    left_region, right_region = _inset_regions(xs)
    _mk_inset(ax_lin, region=left_region,
              bbox=(0.30, 0.52, 0.32, 0.38), yfactor=1.05)
    _mk_inset(ax_lin, region=right_region,
              bbox=(0.65, 0.52, 0.32, 0.38), yfactor=1.10)

    fig.suptitle(
        "Empirical distribution against uniform, Gaussian, and Zipf nulls",
        y=1.02,
    )
    fig.subplots_adjust(wspace=0.28)
    _save(fig, outdir, "08_baseline_overlay")


def fig_baselines_per_panel(counts, outdir):
    xs = np.array(sorted(counts))
    ys = np.array([counts[n] for n in xs], dtype=float)
    T = ys.sum()
    N = len(xs)

    uniform = np.full(N, T / N)
    mu, sigma = _null_gaussian_params(xs)
    g = np.exp(-((xs - mu) ** 2) / (2 * sigma ** 2))
    gaussian = T * g / g.sum()
    ranks_by_x = np.argsort(np.argsort(xs)) + 1
    z = 1.0 / ranks_by_x
    zipf = T * z / z.sum()

    panels = [
        ("uniform",  uniform,  C["uniform"],  ":",  _kl(ys, uniform),
         r"uniform"),
        ("gaussian", gaussian, C["gaussian"], "--", _kl(ys, gaussian),
         rf"Gaussian$(\mu\!=\!{mu:.0f},\sigma\!=\!{sigma:.0f})$"),
        ("zipf",     zipf,     C["zipf"],     "-.", _kl(ys, zipf),
         r"Zipf $\propto 1/\mathrm{rank}(N)$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.8), sharey=True)

    inset_regions = _inset_regions(xs)
    inset_bboxes = [
        (0.30, 0.50, 0.28, 0.42),
        (0.66, 0.50, 0.28, 0.42),
    ]

    null_handles = []
    for ax, (_, base, color, style, kl, base_label) in zip(axes, panels):
        (emp_line,) = ax.plot(xs, ys, "-", color=C["empirical"], lw=1.0,
                              label=f"empirical ({CORPUS_LABEL})")
        (null_line,) = ax.plot(xs, base, style, color=color, lw=1.1,
                               label=f"null: {base_label}")
        null_handles.append(null_line)
        ax.set_xlabel(r"$N$")
        ax.set_title(rf"{base_label},  $D_{{\mathrm{{KL}}}}={kl:.3f}$", fontsize=8)
        ax.grid(True, which="both", ls=":")
        ax.set_xlim(xs.min(), xs.max())

        for (lo, hi), bbox in zip(inset_regions, inset_bboxes):
            mask = (xs >= lo) & (xs <= hi)
            ymax = float(ys[mask].max()) * 1.08
            axin = inset_axes(
                ax, width="100%", height="100%",
                bbox_to_anchor=bbox,
                bbox_transform=ax.transAxes, borderpad=0,
            )
            axin.plot(xs, ys, "-", color=C["empirical"], lw=0.7)
            axin.plot(xs, base, style, color=color, lw=0.7)
            axin.set_xlim(lo, hi)
            axin.set_ylim(0, ymax)
            axin.set_xticks([lo, hi])
            axin.tick_params(axis="both", which="major", labelsize=5.0,
                             length=1.8, pad=1.2)
            for s in axin.spines.values():
                s.set_linewidth(0.5)
            axin.grid(True, ls=":", alpha=0.35)
            mark_inset(ax, axin, loc1=2, loc2=4,
                       fc="none", ec="0.4", lw=0.4, ls="--")

    axes[0].set_ylabel(r"count$(N)$")

    handles = [emp_line] + null_handles
    labels = [f"empirical ({CORPUS_LABEL})"] + [f"null: {p[-1]}" for p in panels]
    fig.legend(handles=handles, labels=labels,
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=7)

    fig.suptitle("Empirical distribution against each null, side by side", y=1.02)
    fig.subplots_adjust(wspace=0.12, bottom=0.22)
    _save(fig, outdir, "08b_baselines_per_panel")


def fig_regime_map(counts, outdir):
    xs = np.array(sorted(counts))
    ys = np.array([counts[n] for n in xs], dtype=float)
    T = ys.sum()
    N = len(xs)

    uniform = np.full(N, T / N)
    mu, sigma = _null_gaussian_params(xs)
    g = np.exp(-((xs - mu) ** 2) / (2 * sigma ** 2))
    gaussian = T * g / g.sum()
    ranks = np.argsort(np.argsort(xs)) + 1
    z = 1.0 / ranks
    zipf = T * z / z.sum()

    safe_ys = np.where(ys > 0, ys, 1e-30)
    lr_u = np.log(safe_ys / uniform)
    lr_g = np.log(safe_ys / gaussian)
    lr_z = np.log(safe_ys / zipf)
    stacked = np.abs(np.stack([lr_u, lr_g, lr_z], axis=0))
    winner = np.argmin(stacked, axis=0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(W_DOUBLE, 3.4),
        gridspec_kw={"height_ratios": [3.4, 0.55], "hspace": 0.10},
        sharex=True,
    )

    ax1.plot(xs, np.abs(lr_u), color=C["uniform"],  ls=":",  lw=0.9, label="uniform")
    ax1.plot(xs, np.abs(lr_g), color=C["gaussian"], ls="--", lw=0.9, label="Gaussian")
    ax1.plot(xs, np.abs(lr_z), color=C["zipf"],    ls="-.", lw=1.0, label="Zipf")
    ax1.axhline(np.log(2), color="0.6", lw=0.4, ls=":")
    ax1.axhline(np.log(10), color="0.6", lw=0.4, ls=":")
    ax1.set_yscale("symlog", linthresh=0.1)
    ax1.set_ylabel(r"$|\log\,\mathrm{emp}(N)/\mathrm{null}(N)|$")
    ax1.set_title(r"Log-ratio residual to each null")
    ax1.grid(True, which="both", ls=":")
    ax1.legend(loc="upper right", ncol=3)
    ax1.set_xlim(xs.min(), xs.max())

    null_colors = [C["uniform"], C["gaussian"], C["zipf"]]
    null_names  = ["uniform", "Gaussian", "Zipf"]
    strip = np.array([mpl.colors.to_rgb(null_colors[w]) for w in winner])[None, :, :]
    ax2.imshow(strip, aspect="auto",
               extent=[xs.min() - 0.5, xs.max() + 0.5, 0, 1],
               interpolation="nearest")
    ax2.set_yticks([])
    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel(r"$\arg\min$", rotation=90, labelpad=8, fontsize=7)
    for s in ("left", "right", "top", "bottom"):
        ax2.spines[s].set_linewidth(0.5)

    fracs = [float((winner == i).sum()) / len(winner) for i in range(3)]
    fracs_text = (
        rf"uniform $\!=\!{fracs[0]*100:.0f}\%$  "
        rf"Gaussian $\!=\!{fracs[1]*100:.0f}\%$  "
        rf"Zipf $\!=\!{fracs[2]*100:.0f}\%$"
    )
    ax2.text(0.5, -1.4, fracs_text, transform=ax2.transAxes,
             fontsize=7, ha="center", va="top", color="0.2")

    _save(fig, outdir, "10_regime_map")


def fig_cumulative(counts, outdir):
    xs = np.array(sorted(counts))
    ys = np.array([counts[n] for n in xs], dtype=float)
    total = ys.sum()
    cum = np.cumsum(ys) / total

    deciles = [0.1, 0.25, 0.5, 0.75, 0.9]
    dec_x = []
    for d in deciles:
        idx = int(np.searchsorted(cum, d))
        dec_x.append(xs[min(idx, len(xs) - 1)])

    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.3))
    ax.plot(xs, cum, color=C["empirical"], lw=1.1)
    for d, x in zip(deciles, dec_x):
        ax.axvline(x, color=C["muted"], lw=0.5, ls=":")
        ax.annotate(rf"${int(d*100)}\%\!:\!N\!=\!{x}$", xy=(x, d), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6,
                    color=C["empirical"])
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("cumulative share of mass")
    ax.set_title("Cumulative distribution of integer occurrences")
    ax.set_ylim(0, 1.02)
    ax.grid(True, ls=":")
    _save(fig, outdir, "09_cumulative_share")


def main(argv=None):
    global CORPUS_LABEL

    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None)
    p.add_argument("--counts-csv", default=None)
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--n-min", type=int, default=NUMBER_MIN)
    p.add_argument("--n-max", type=int, default=NUMBER_MAX)
    p.add_argument("--corpus-label", default="corpus")
    args = p.parse_args(argv)
    if args.n_max <= args.n_min:
        p.error(f"--n-max ({args.n_max}) must be > --n-min ({args.n_min})")
    CORPUS_LABEL = args.corpus_label

    results_dir = Path(args.results_dir)
    figs_dir = results_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    _apply_neurips_style()

    if args.counts_csv:
        print(f"[number_analysis] reading counts from {args.counts_csv} ...")
        counts = _read_counts_csv(Path(args.counts_csv), args.n_min, args.n_max)
    else:
        if not args.run_id:
            p.error("one of --run-id or --counts-csv is required")
        print(f"[number_analysis] pulling counts from run {args.run_id} ...")
        counts = _pull_counts(args.run_id, args.tracking_uri, args.n_min, args.n_max)
    print(f"[number_analysis] {len(counts)} integers, total = {sum(counts.values()):,}")

    csv_path = results_dir / f"counts_{args.n_min}_to_{args.n_max}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["number", "count"])
        for n in range(args.n_min, args.n_max + 1):
            w.writerow([n, counts.get(n, 0)])
    print(f"  wrote {csv_path}")

    fig_linear(counts, figs_dir)
    fig_semilog(counts, figs_dir)
    fig_loglog(counts, figs_dir)
    fig_zipf_rank(counts, figs_dir)
    fig_top_bars(counts, figs_dir)
    fig_round_numbers(counts, figs_dir)
    fig_leading_digit(counts, figs_dir)
    fig_baseline_overlay(counts, figs_dir)
    fig_baselines_per_panel(counts, figs_dir)
    fig_regime_map(counts, figs_dir)
    fig_cumulative(counts, figs_dir)

    print("[number_analysis] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
