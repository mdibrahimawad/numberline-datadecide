from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.number_analysis import C, W_DOUBLE, _apply_neurips_style, _kl, _save


PALETTE = [
    "#000000",
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
]


def read_counts(path: Path, n_min: int, n_max: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or {"number", "count"} - set(reader.fieldnames):
            raise SystemExit(f"{path} must have number,count columns")
        for row in reader:
            n = int(row["number"])
            if n_min <= n <= n_max:
                counts[n] = int(row["count"])
    return {n: counts.get(n, 0) for n in range(n_min, n_max + 1)}


def fit_loglog(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    mask = (xs > 0) & (ys > 0)
    lx = np.log(xs[mask])
    ly = np.log(ys[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    y_hat = slope * lx + intercept
    ss_res = float(np.sum((ly - y_hat) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return float(slope), float(intercept), float(r2)


def summarize(name: str, counts: dict[int, int]) -> dict[str, object]:
    xs = np.array(sorted(counts))
    ys = np.array([counts[n] for n in xs], dtype=float)
    total = float(ys.sum())

    value_slope, _, value_r2 = fit_loglog(xs, ys)
    vals = np.array(sorted((c for c in counts.values() if c > 0), reverse=True), dtype=float)
    ranks = np.arange(1, len(vals) + 1, dtype=float)
    rank_slope, _, rank_r2 = fit_loglog(ranks, vals)

    cum = np.cumsum(ys) / total
    q = {}
    for p in (0.10, 0.25, 0.50, 0.75, 0.90):
        idx = int(np.searchsorted(cum, p))
        q[f"cdf_n_at_{int(p * 100)}"] = int(xs[min(idx, len(xs) - 1)])

    benford = np.array([math.log10(1 + 1 / d) for d in range(1, 10)])
    leading = np.zeros(9)
    for n, c in counts.items():
        if n > 0:
            leading[int(str(n)[0]) - 1] += c
    leading = leading / leading.sum()

    is_hundred = (xs > 0) & (xs % 100 == 0)
    is_ten = (xs > 0) & (xs % 10 == 0) & (xs % 100 != 0)
    is_single_digit = (xs >= 0) & (xs <= 9)
    is_year_band = (xs >= 1900) & (xs <= 2025)

    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
    return {
        "dataset": name,
        "total_in_range": int(total),
        "value_power_alpha": -value_slope,
        "value_power_r2": value_r2,
        "zipf_rank_s": -rank_slope,
        "zipf_rank_r2": rank_r2,
        "benford_kl": _kl(leading, benford),
        "single_digit_share": float(ys[is_single_digit].sum() / total),
        "ten_not_hundred_share": float(ys[is_ten].sum() / total),
        "hundred_share": float(ys[is_hundred].sum() / total),
        "year_1900_2025_share": float(ys[is_year_band].sum() / total),
        "share_n_le_100": float(ys[xs <= 100].sum() / total),
        "share_n_le_1000": float(ys[xs <= 1000].sum() / total),
        **q,
        "top20": top,
    }


def plot_semilogy(datasets: list[tuple[str, dict[int, int]]], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.6))
    for i, (name, counts) in enumerate(datasets):
        xs = np.array(sorted(counts))
        ys = np.array([counts[n] for n in xs], dtype=float)
        ax.semilogy(xs, ys, color=PALETTE[i % len(PALETTE)], lw=0.85, label=name)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"count$(N)$")
    ax.set_title("Integer frequency, 0--10000")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper right")
    _save(fig, outdir, "cmp_01_semilogy_overlay")


def plot_value_loglog(datasets: list[tuple[str, dict[int, int]]], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.7))
    for i, (name, counts) in enumerate(datasets):
        xs = np.array(sorted(counts))
        ys = np.array([counts[n] for n in xs], dtype=float)
        slope, intercept, r2 = fit_loglog(xs, ys)
        color = PALETTE[i % len(PALETTE)]
        mask = (xs > 0) & (ys > 0)
        ax.loglog(xs[mask], ys[mask], ".", ms=1.4, alpha=0.30, color=color, rasterized=True)
        xline = np.array([xs[mask].min(), xs[mask].max()])
        ax.loglog(
            xline,
            np.exp(intercept + slope * np.log(xline)),
            lw=1.1,
            color=color,
            label=rf"{name}: $\alpha={-slope:.2f}$, $R^2={r2:.2f}$",
        )
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"count$(N)$")
    ax.set_title("Value-space power-law fit")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper right")
    _save(fig, outdir, "cmp_02_value_loglog_fit")


def plot_zipf(datasets: list[tuple[str, dict[int, int]]], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.7))
    for i, (name, counts) in enumerate(datasets):
        vals = np.array(sorted((c for c in counts.values() if c > 0), reverse=True), dtype=float)
        ranks = np.arange(1, len(vals) + 1, dtype=float)
        slope, intercept, r2 = fit_loglog(ranks, vals)
        color = PALETTE[i % len(PALETTE)]
        ax.loglog(ranks, vals, ".", ms=1.5, alpha=0.35, color=color, rasterized=True)
        xline = np.array([ranks.min(), ranks.max()])
        ax.loglog(
            xline,
            np.exp(intercept + slope * np.log(xline)),
            lw=1.1,
            color=color,
            label=rf"{name}: $s={-slope:.2f}$, $R^2={r2:.2f}$",
        )
    ax.set_xlabel(r"rank$(N)$")
    ax.set_ylabel(r"count$(N)$")
    ax.set_title("Zipf rank--frequency comparison")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper right")
    _save(fig, outdir, "cmp_03_zipf_rank_frequency")


def plot_cdf(datasets: list[tuple[str, dict[int, int]]], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.6))
    for i, (name, counts) in enumerate(datasets):
        xs = np.array(sorted(counts))
        ys = np.array([counts[n] for n in xs], dtype=float)
        ax.plot(xs, np.cumsum(ys) / ys.sum(), color=PALETTE[i % len(PALETTE)], lw=1.1, label=name)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("cumulative share")
    ax.set_title("Cumulative integer-occurrence mass")
    ax.set_ylim(0, 1.01)
    ax.grid(True, ls=":")
    ax.legend(loc="lower right")
    _save(fig, outdir, "cmp_04_cumulative_share")


def plot_roundness_bars(summaries: list[dict[str, object]], outdir: Path) -> None:
    cats = [
        ("single_digit_share", "0--9"),
        ("ten_not_hundred_share", "10-multiple"),
        ("hundred_share", "100-multiple"),
        ("year_1900_2025_share", "1900--2025"),
        ("share_n_le_100", r"$N\leq100$"),
        ("share_n_le_1000", r"$N\leq1000$"),
    ]
    x = np.arange(len(cats))
    width = 0.8 / len(summaries)
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.7))
    for i, summary in enumerate(summaries):
        vals = [100.0 * float(summary[key]) for key, _ in cats]
        ax.bar(
            x - 0.4 + width / 2 + i * width,
            vals,
            width=width,
            color=PALETTE[i % len(PALETTE)],
            label=str(summary["dataset"]),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in cats])
    ax.set_ylabel("share of in-range mass (%)")
    ax.set_title("Mass concentration and roundness")
    ax.grid(True, axis="y", ls=":")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(summaries))
    fig.subplots_adjust(bottom=0.24)
    _save(fig, outdir, "cmp_05_mass_roundness_bars")


def plot_year_band(datasets: list[tuple[str, dict[int, int]]], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.5))
    for i, (name, counts) in enumerate(datasets):
        xs = np.arange(1800, 2026)
        ys = np.array([counts.get(int(n), 0) for n in xs], dtype=float)
        ax.semilogy(xs, ys, color=PALETTE[i % len(PALETTE)], lw=1.0, label=name)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"count$(N)$")
    ax.set_title("Calendar-year band frequency")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper left")
    _save(fig, outdir, "cmp_06_year_band")


def write_summary(summaries: list[dict[str, object]], outdir: Path) -> None:
    json_path = outdir / "frequency_summary_0_to_10000.json"
    with open(json_path, "w") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"  wrote {json_path}")

    fields = [
        "dataset",
        "total_in_range",
        "value_power_alpha",
        "value_power_r2",
        "zipf_rank_s",
        "zipf_rank_r2",
        "benford_kl",
        "single_digit_share",
        "ten_not_hundred_share",
        "hundred_share",
        "year_1900_2025_share",
        "share_n_le_100",
        "share_n_le_1000",
        "cdf_n_at_50",
        "cdf_n_at_90",
    ]
    csv_path = outdir / "frequency_summary_0_to_10000.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})
    print(f"  wrote {csv_path}")

    tex_path = outdir / "frequency_summary_0_to_10000.tex"
    with open(tex_path, "w") as fh:
        fh.write("\\begin{tabular}{lrrrrrr}\n")
        fh.write("\\toprule\n")
        fh.write("Dataset & $\\alpha_N$ & $R^2_N$ & $s_{rank}$ & $R^2_{rank}$ & $N_{50}$ & $N_{90}$ \\\\\n")
        fh.write("\\midrule\n")
        for summary in summaries:
            fh.write(
                f"{summary['dataset']} & "
                f"{float(summary['value_power_alpha']):.2f} & "
                f"{float(summary['value_power_r2']):.2f} & "
                f"{float(summary['zipf_rank_s']):.2f} & "
                f"{float(summary['zipf_rank_r2']):.2f} & "
                f"{int(summary['cdf_n_at_50'])} & "
                f"{int(summary['cdf_n_at_90'])} \\\\\n"
            )
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")
    print(f"  wrote {tex_path}")


def parse_dataset(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--dataset must be NAME=counts.csv")
    name, path = spec.split("=", 1)
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, type=parse_dataset)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-min", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=10000)
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    figs_dir = outdir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    _apply_neurips_style()

    datasets = [(name, read_counts(path, args.n_min, args.n_max)) for name, path in args.dataset]
    summaries = [summarize(name, counts) for name, counts in datasets]

    plot_semilogy(datasets, figs_dir)
    plot_value_loglog(datasets, figs_dir)
    plot_zipf(datasets, figs_dir)
    plot_cdf(datasets, figs_dir)
    plot_roundness_bars(summaries, figs_dir)
    plot_year_band(datasets, figs_dir)
    write_summary(summaries, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
