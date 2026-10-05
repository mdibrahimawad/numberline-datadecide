from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable


NUMBER_VALUES = tuple(range(1, 10))
NUMBER_FORMATS = {
    "lowercase": ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
    "mixedcase": ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"),
    "digits": tuple(str(x) for x in NUMBER_VALUES),
}


def _r2_score(y_true, y_pred) -> float:
    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def _minmax(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _linear_fit(x, y) -> dict:
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return {"r2": float("nan"), "slope": float("nan"), "intercept": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    return {"r2": _r2_score(y, pred), "slope": float(slope), "intercept": float(intercept)}


def _negative_exp_fit(x, y) -> dict:
    import numpy as np
    from scipy.optimize import curve_fit

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def fn(t, a, b, c):
        return a * np.exp(-b * t) + c

    if len(x) < 3:
        return {"r2": float("nan"), "a": float("nan"), "b": float("nan"), "c": float("nan")}
    y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-3)
    p0 = [y_span, 0.5, float(np.nanmin(y))]
    try:
        params, _ = curve_fit(
            fn,
            x,
            y,
            p0=p0,
            bounds=([0.0, 0.0, -math.inf], [math.inf, math.inf, math.inf]),
            maxfev=10000,
        )
        pred = fn(x, *params)
        return {
            "r2": _r2_score(y, pred),
            "a": float(params[0]),
            "b": float(params[1]),
            "c": float(params[2]),
        }
    except Exception:
        return {"r2": float("nan"), "a": float("nan"), "b": float("nan"), "c": float("nan")}


def _pearson(x, y) -> float:
    import numpy as np
    from scipy.stats import pearsonr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 2:
        return float("nan")
    return float(pearsonr(x[ok], y[ok]).statistic)


def _mds_1d(dissimilarity) -> list[float]:
    import numpy as np
    import warnings
    from sklearn.manifold import MDS

    dissimilarity = np.asarray(dissimilarity, dtype=float)
    kwargs = {
        "n_components": 1,
        "dissimilarity": "precomputed",
        "init": "random",
        "random_state": 0,
        "n_init": 8,
        "max_iter": 600,
    }
    try:
        mds = MDS(normalized_stress="auto", **kwargs)
    except TypeError:
        mds = MDS(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        coords = mds.fit_transform(dissimilarity).reshape(-1)
    if coords[-1] < coords[0]:
        coords = -coords
    return [float(x) for x in coords]


def _pairwise_rows(similarity):
    import numpy as np

    sim = np.asarray(similarity, dtype=float)
    rows = []
    values = []
    for i, x in enumerate(NUMBER_VALUES):
        for j, y in enumerate(NUMBER_VALUES):
            if i >= j:
                continue
            values.append(sim[i, j])
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "distance": abs(x - y),
                    "minimum": min(x, y),
                    "ratio": max(x, y) / min(x, y),
                    "similarity": float(sim[i, j]),
                }
            )
    norm = _minmax(values)
    for row, normalized in zip(rows, norm):
        row["similarity_norm"] = float(normalized)
    return rows


def analyze_similarity_matrix(similarity, dissimilarity) -> dict:
    import numpy as np

    pair_rows = _pairwise_rows(similarity)

    distance_x = []
    distance_y_raw = []
    for distance in range(1, 9):
        vals = [r["similarity"] for r in pair_rows if r["distance"] == distance]
        distance_x.append(distance)
        distance_y_raw.append(sum(vals) / len(vals))
    distance_y = _minmax(distance_y_raw)
    distance_fit = _linear_fit(distance_x, distance_y)

    size_x = []
    size_y = []
    for minimum in range(1, 9):
        vals = [r["similarity_norm"] for r in pair_rows if r["minimum"] == minimum]
        if vals:
            size_x.append(minimum)
            size_y.append(sum(vals) / len(vals))
    size_fit = _linear_fit(size_x, size_y)

    ratio_x = [r["ratio"] for r in pair_rows]
    ratio_y = [r["similarity_norm"] for r in pair_rows]
    ratio_fit = _negative_exp_fit(ratio_x, ratio_y)

    coords = _mds_1d(dissimilarity)
    log_values = [math.log10(x) for x in NUMBER_VALUES]
    mds_corr = _pearson(coords, log_values)

    return {
        "distance_r2": distance_fit["r2"],
        "distance_slope": distance_fit["slope"],
        "distance_intercept": distance_fit["intercept"],
        "size_r2": size_fit["r2"],
        "size_slope": size_fit["slope"],
        "size_intercept": size_fit["intercept"],
        "ratio_r2": ratio_fit["r2"],
        "ratio_a": ratio_fit["a"],
        "ratio_b": ratio_fit["b"],
        "ratio_c": ratio_fit["c"],
        "mds_log_corr": mds_corr,
        "mds_coords": coords,
        "distance_points": [
            {"distance": int(x), "similarity_norm": float(y)}
            for x, y in zip(distance_x, distance_y)
        ],
        "size_points": [
            {"minimum": int(x), "similarity_norm": float(y)}
            for x, y in zip(size_x, size_y)
        ],
        "ratio_points": [
            {"ratio": float(r["ratio"]), "similarity_norm": float(r["similarity_norm"])}
            for r in pair_rows
        ],
    }


def summarize_format_payload(payload: dict) -> dict:
    rows = payload["rows"]
    out = {
        "model_label": payload["model_label"],
        "model_name": payload["model_name"],
        "model_revision": payload.get("model_revision") or "",
        "input_format": payload["input_format"],
        "n_layers": len(rows),
    }
    for key in ("distance_r2", "size_r2", "ratio_r2", "mds_log_corr"):
        vals = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        out[f"{key}_mean"] = sum(vals) / len(vals) if vals else float("nan")
        if vals:
            best = max(rows, key=lambda row: float(row[key]) if math.isfinite(float(row[key])) else -math.inf)
            out[f"{key}_best_layer"] = best["layer"]
            out[f"{key}_best"] = best[key]
        else:
            out[f"{key}_best_layer"] = ""
            out[f"{key}_best"] = float("nan")
    return out


def load_beta_table(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            label = row.get("model_label") or row.get("model")
            beta = row.get("pca_beta_mean") or row.get("pca_beta")
            if label and beta not in {None, ""}:
                out[label] = float(beta)
    return out


def write_results(outdir: Path, payloads: list[dict]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "paper_magnitude_payloads.json", "w") as fh:
        json.dump(payloads, fh, indent=2)

    rows = [row for payload in payloads for row in payload["rows"]]
    if rows:
        keys = sorted({key for row in rows for key in row if key not in {"mds_coords", "distance_points", "size_points", "ratio_points"}})
        with open(outdir / "paper_magnitude_layer_rows.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in keys})

    format_summaries = [summarize_format_payload(payload) for payload in payloads]
    if format_summaries:
        keys = sorted({key for row in format_summaries for key in row})
        with open(outdir / "paper_magnitude_format_summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(format_summaries)

    model_summary = summarize_models(format_summaries)
    if model_summary:
        keys = sorted({key for row in model_summary for key in row})
        with open(outdir / "paper_magnitude_model_summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(model_summary)


def summarize_models(format_summaries: Iterable[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in format_summaries:
        grouped.setdefault(row["model_label"], []).append(row)

    out = []
    for label, rows in grouped.items():
        base = {
            "model_label": label,
            "model_name": rows[0]["model_name"],
            "model_revision": rows[0].get("model_revision", ""),
            "n_formats": len(rows),
        }
        for metric in ("distance_r2_mean", "size_r2_mean", "ratio_r2_mean", "mds_log_corr_mean"):
            vals = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
            base[metric.replace("_mean", "_all_formats_mean")] = (
                sum(vals) / len(vals) if vals else float("nan")
            )
        for row in rows:
            fmt = row["input_format"]
            for metric in ("distance_r2_mean", "size_r2_mean", "ratio_r2_mean", "mds_log_corr_mean"):
                base[f"{fmt}_{metric}"] = row[metric]
        out.append(base)
    return sorted(out, key=lambda row: row["model_label"])


def compute_correlations(
    outdir: Path,
    beta_csv: Path = Path("results/dataset_frequency_0_to_10000/alpha_beta_without_pile/alpha_beta_models_without_pile.csv"),
) -> dict[str, float]:
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    beta = load_beta_table(beta_csv)
    summary_path = outdir / "paper_magnitude_model_summary.csv"
    if not summary_path.exists():
        return {}
    with open(summary_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = [
        "distance_r2_all_formats_mean",
        "size_r2_all_formats_mean",
        "ratio_r2_all_formats_mean",
        "mds_log_corr_all_formats_mean",
        "digits_distance_r2_mean",
        "digits_size_r2_mean",
        "digits_ratio_r2_mean",
        "digits_mds_log_corr_mean",
    ]
    xs = np.array([beta.get(row["model_label"], float("nan")) for row in rows], dtype=float)
    out: dict[str, float] = {}
    for col in cols:
        ys = np.array([float(row.get(col, "nan")) for row in rows], dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if int(ok.sum()) >= 2:
            out[f"spearman_beta_{col}"] = float(spearmanr(xs[ok], ys[ok]).statistic)
            out[f"pearson_beta_{col}"] = float(pearsonr(xs[ok], ys[ok]).statistic)
    with open(outdir / "paper_magnitude_beta_correlations.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    return out


def plot_results(
    outdir: Path,
    beta_csv: Path = Path("results/dataset_frequency_0_to_10000/alpha_beta_without_pile/alpha_beta_models_without_pile.csv"),
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    layer_path = outdir / "paper_magnitude_layer_rows.csv"
    model_path = outdir / "paper_magnitude_model_summary.csv"
    if not layer_path.exists() or not model_path.exists():
        return

    with open(layer_path, newline="") as fh:
        layer_rows = list(csv.DictReader(fh))
    with open(model_path, newline="") as fh:
        model_rows = list(csv.DictReader(fh))

    metrics = [
        ("distance_r2", "Distance effect $R^2$"),
        ("size_r2", "Size effect $R^2$"),
        ("ratio_r2", "Ratio effect $R^2$"),
        ("mds_log_corr", "MDS-log correlation"),
    ]
    models = sorted({row["model_label"] for row in layer_rows})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), sharex=False)
    axes = axes.reshape(-1)
    for ax, (metric, title) in zip(axes, metrics):
        for model in models:
            by_layer: dict[int, list[float]] = {}
            for row in layer_rows:
                if row["model_label"] != model:
                    continue
                by_layer.setdefault(int(row["layer"]), []).append(float(row[metric]))
            xs = sorted(by_layer)
            ys = [float(np.nanmean(by_layer[x])) for x in xs]
            ax.plot(xs, ys, marker="o", markersize=2.8, linewidth=1.0, label=model)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("score")
    axes[2].set_ylabel("score")
    axes[0].legend(fontsize=6, ncols=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(outdir / "paper_magnitude_effects_by_layer.pdf")
    fig.savefig(outdir / "paper_magnitude_effects_by_layer.png", dpi=220)
    plt.close(fig)

    beta = load_beta_table(beta_csv)
    beta_metrics = [
        ("distance_r2_all_formats_mean", "Distance"),
        ("size_r2_all_formats_mean", "Size"),
        ("ratio_r2_all_formats_mean", "Ratio"),
        ("mds_log_corr_all_formats_mean", "MDS-log"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2))
    for ax, (metric, title) in zip(axes, beta_metrics):
        xs = np.array([beta.get(row["model_label"], float("nan")) for row in model_rows], dtype=float)
        ys = np.array([float(row[metric]) for row in model_rows], dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        ax.scatter(xs[ok], ys[ok], s=42, color="#2f6f8f")
        if int(ok.sum()) >= 2:
            coef = np.polyfit(xs[ok], ys[ok], 1)
            xx = np.linspace(float(xs[ok].min()), float(xs[ok].max()), 100)
            ax.plot(xx, coef[0] * xx + coef[1], color="#c7503d", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel(r"$\beta_{\mathrm{PCA}}$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("paper-style score")
    fig.tight_layout()
    fig.savefig(outdir / "paper_magnitude_beta_panels.pdf")
    fig.savefig(outdir / "paper_magnitude_beta_panels.png", dpi=220)
    plt.close(fig)

    matrix = np.array(
        [[float(row[metric]) for metric, _ in beta_metrics] for row in model_rows],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(1.0, float(np.nanmax(matrix))))
    ax.set_yticks(range(len(model_rows)), [row["model_label"] for row in model_rows], fontsize=8)
    ax.set_xticks(range(len(beta_metrics)), [title for _, title in beta_metrics], fontsize=8)
    ax.set_title("Paper-style magnitude effects")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("score")
    fig.tight_layout()
    fig.savefig(outdir / "paper_magnitude_heatmap.pdf")
    fig.savefig(outdir / "paper_magnitude_heatmap.png", dpi=220)
    plt.close(fig)


def write_latex_summary(outdir: Path) -> None:
    rows = []
    path = outdir / "paper_magnitude_model_summary.csv"
    if path.exists():
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    with open(outdir / "paper_magnitude_summary.tex", "w") as fh:
        fh.write("\\begin{table}[h!]\n")
        fh.write("    \\centering\n")
        fh.write("    \\resizebox{\\linewidth}{!}{%\n")
        fh.write("    \\begin{tabular}{lcccc}\n")
        fh.write("        \\toprule\n")
        fh.write("        Model & Distance $R^2$ & Size $R^2$ & Ratio $R^2$ & MDS-log $r$ \\\\\n")
        fh.write("        \\midrule\n")
        for row in rows:
            fh.write(
                f"        {row['model_label']} & "
                f"{float(row['distance_r2_all_formats_mean']):.2f} & "
                f"{float(row['size_r2_all_formats_mean']):.2f} & "
                f"{float(row['ratio_r2_all_formats_mean']):.2f} & "
                f"{float(row['mds_log_corr_all_formats_mean']):.2f} \\\\\n"
            )
        fh.write("        \\bottomrule\n")
        fh.write("    \\end{tabular}}\n")
        fh.write("    \\caption{Exact paper-style magnitude benchmark on numbers 1--9, averaged across lowercase number words, capitalized number words, and digits.}\n")
        fh.write("    \\label{tab:paper-magnitude-replication}\n")
        fh.write("\\end{table}\n")
