from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


DEFAULT_BASELINE_BETA = Path(
    "results/dataset_frequency_0_to_10000/alpha_beta_without_pile/"
    "alpha_beta_models_without_pile.csv"
)
DEFAULT_BASELINE_SWEEP = Path(
    "results/number_comparison/fixed_gap_wide_sg_mg_fewshot_exemplar_sweep"
)
DEFAULT_CANDIDATE_SWEEP = Path(
    "results/number_comparison/candidate_screen_wide_sg_mg_sweep"
)
DEFAULT_OUTDIR = Path(
    "results/number_comparison/candidate_screen_wide_sg_mg_analysis"
)


MODEL_REPO_TO_LABEL = {
    "facebook/opt-1.3b": "OPT-1.3B",
    "facebook/opt-2.7b": "OPT-2.7B",
    "facebook/opt-6.7b": "OPT-6.7B",
    "cerebras/Cerebras-GPT-1.3B": "Cerebras-GPT-1.3B",
    "cerebras/Cerebras-GPT-2.7B": "Cerebras-GPT-2.7B",
    "cerebras/Cerebras-GPT-6.7B": "Cerebras-GPT-6.7B",
    "bigscience/bloom-1b7": "BLOOM-1.7B",
    "bigscience/bloom-3b": "BLOOM-3B",
    "stabilityai/stablelm-base-alpha-3b": "StableLM-Base-3B",
    "stabilityai/stablelm-base-alpha-7b": "StableLM-Base-7B",
    "openlm-research/open_llama_3b_v2": "OpenLLaMA-3B-v2",
    "EleutherAI/gpt-neo-125m": "GPT-Neo-125M",
    "EleutherAI/gpt-neo-1.3B": "GPT-Neo-1.3B",
    "EleutherAI/gpt-neo-2.7B": "GPT-Neo-2.7B",
    "Qwen/Qwen1.5-0.5B": "Qwen1.5-0.5B",
    "Qwen/Qwen1.5-1.8B": "Qwen1.5-1.8B",
    "Qwen/Qwen1.5-4B": "Qwen1.5-4B",
    "Qwen/Qwen1.5-7B": "Qwen1.5-7B",
    "bigscience/bloom-7b1": "BLOOM-7B1",
}


BASELINE_LABELS = [
    "Falcon-RW-1B",
    "Falcon-RW-7B",
    "RedPajama-3B",
    "RedPajama-7B",
    "OLMo-7B-2T",
    "OLMo-7B-Twin-2T",
    "StarCoderBase-1B",
    "StarCoderBase-3B",
    "StarCoderBase-7B",
]


SHORT_LABEL = {
    "Falcon-RW-1B": "Falcon-1B",
    "Falcon-RW-7B": "Falcon-7B",
    "RedPajama-3B": "RPJ-3B",
    "RedPajama-7B": "RPJ-7B",
    "OLMo-7B-2T": "OLMo-2T",
    "OLMo-7B-Twin-2T": "OLMo-Twin",
    "StarCoderBase-1B": "SCB-1B",
    "StarCoderBase-3B": "SCB-3B",
    "StarCoderBase-7B": "SCB-7B",
    "OPT-1.3B": "OPT-1.3B",
    "OPT-2.7B": "OPT-2.7B",
    "OPT-6.7B": "OPT-6.7B",
    "Cerebras-GPT-1.3B": "CGPT-1.3B",
    "Cerebras-GPT-2.7B": "CGPT-2.7B",
    "Cerebras-GPT-6.7B": "CGPT-6.7B",
    "BLOOM-1.7B": "BLOOM-1.7B",
    "BLOOM-3B": "BLOOM-3B",
    "BLOOM-7B1": "BLOOM-7B1",
    "StableLM-Base-3B": "StableLM-3B",
    "StableLM-Base-7B": "StableLM-7B",
    "OpenLLaMA-3B-v2": "OpenLLaMA-3B",
    "GPT-Neo-125M": "Neo-125M",
    "GPT-Neo-1.3B": "Neo-1.3B",
    "GPT-Neo-2.7B": "Neo-2.7B",
    "Qwen1.5-0.5B": "Qwen-0.5B",
    "Qwen1.5-1.8B": "Qwen-1.8B",
    "Qwen1.5-4B": "Qwen-4B",
    "Qwen1.5-7B": "Qwen-7B",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_beta_csv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            label = row.get("model_label") or row.get("model")
            beta = row.get("pca_beta") or row.get("pca_beta_mean")
            if label and beta not in {None, ""}:
                out[label] = float(beta)
    return out


def selected_layer(payload: dict, method: str, selection: str) -> int:
    if selection == "payload_best":
        return int(payload[f"best_layer_{method}"])
    if selection == "max_rho":
        def rho_score(key: str) -> float:
            value = float(payload[method][key].get("rho_mean", float("-inf")))
            return value if np.isfinite(value) else float("-inf")

        return int(
            max(
                payload[method],
                key=rho_score,
            )
        )
    raise ValueError(f"unknown layer selection: {selection}")


def best_layer_summary(payload: dict, method: str, selection: str = "payload_best") -> dict[str, float | int | str]:
    layer = selected_layer(payload, method, selection)
    stats = payload[method][str(layer)]
    return {
        f"{method}_selection": selection,
        f"{method}_layer": layer,
        f"{method}_beta": float(stats["beta_mean"]),
        f"{method}_beta_std": float(stats.get("beta_std", 0.0)),
        f"{method}_rho": float(stats["rho_mean"]),
        f"{method}_rho_std": float(stats.get("rho_std", 0.0)),
        f"{method}_ev": float(stats.get("pca_explained_variance", stats.get("explained_variance", float("nan")))),
    }


def extract_geometry_beta(
    *,
    mlflow_db: Path,
    run_name_prefix: str,
    outdir: Path,
    pca_selection: str,
) -> list[dict[str, object]]:
    if not mlflow_db.exists():
        raise FileNotFoundError(f"missing MLflow DB: {mlflow_db}")
    rows: list[dict[str, object]] = []
    con = sqlite3.connect(str(mlflow_db))
    try:
        cur = con.execute(
            """
            select run_uuid, name, artifact_uri, start_time
            from runs
            where name like ?
            order by start_time asc
            """,
            (f"{run_name_prefix}%",),
        )
        run_rows = cur.fetchall()
    finally:
        con.close()

    seen: set[str] = set()
    for run_id, run_name, artifact_uri, _start_time in run_rows:
        artifact_root = artifact_uri.replace("file://", "")
        result_path = Path(artifact_root) / "geometry" / "results.json"
        if not result_path.exists():
            continue
        with open(result_path) as fh:
            payload = json.load(fh)
        model_name = payload["config"]["model_name"]
        label = MODEL_REPO_TO_LABEL.get(model_name)
        if not label or label in seen:
            continue
        seen.add(label)
        target = outdir / "geometry_results" / label / "results.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2))
        row: dict[str, object] = {
            "model_label": label,
            "model_name": model_name,
            "run_id": run_id,
            "run_name": run_name,
            "result_path": str(target),
            "n_layers": int(payload["n_layers"]),
        }
        row.update(best_layer_summary(payload, "pca", pca_selection))
        row.update(best_layer_summary(payload, "pls"))
        rows.append(row)
    rows.sort(key=lambda row: str(row["model_label"]))
    write_csv(outdir / "candidate_beta_summary.csv", rows)
    return rows


def read_condition_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for subdir in sorted(root.glob("*shot_set*")):
        rows_path = subdir / "number_comparison_rows.csv"
        if not rows_path.exists():
            continue
        with open(rows_path, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def model_accuracy_by_shot(
    roots: list[Path],
    beta_by_model: dict[str, float],
) -> list[dict[str, object]]:
    cells: dict[tuple[int, str], list[float]] = defaultdict(list)
    for root in roots:
        for row in read_condition_rows(root):
            label = row["model_label"]
            if label not in beta_by_model:
                continue
            key = (int(row["n_shots"]), label)
            cells[key].append(float(row["correct"]))

    out: list[dict[str, object]] = []
    for (shots, label), values in sorted(cells.items()):
        out.append(
            {
                "shots": shots,
                "model_label": label,
                "pca_beta": beta_by_model[label],
                "accuracy": float(np.mean(values)),
                "n_tasks": len(values),
            }
        )
    return out


def corr_for(rows: list[dict[str, object]]) -> dict[str, float]:
    xs = np.asarray([float(row["pca_beta"]) for row in rows], dtype=float)
    ys = np.asarray([float(row["accuracy"]) for row in rows], dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if int(ok.sum()) < 3:
        return {
            "spearman_rho_s": float("nan"),
            "spearman_p": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "r2": float("nan"),
            "mean_accuracy": float(np.nanmean(ys)) if ys.size else float("nan"),
            "n_models": int(ok.sum()),
        }
    sp = spearmanr(xs[ok], ys[ok])
    pr = pearsonr(xs[ok], ys[ok])
    return {
        "spearman_rho_s": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "r2": float(pr.statistic) ** 2,
        "mean_accuracy": float(np.mean(ys[ok])),
        "n_models": int(ok.sum()),
    }


def summarize_sets(
    model_rows: list[dict[str, object]],
    *,
    baseline_labels: list[str],
    candidate_labels: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_shot: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in model_rows:
        by_shot[int(row["shots"])].append(row)

    baseline_summary: list[dict[str, object]] = []
    candidate_effects: list[dict[str, object]] = []
    baseline_by_shot: dict[int, dict[str, float]] = {}
    for shots in sorted(by_shot):
        base_rows = [row for row in by_shot[shots] if row["model_label"] in baseline_labels]
        metrics = corr_for(base_rows)
        baseline_by_shot[shots] = metrics
        baseline_summary.append({"set": "baseline", "shots": shots, **metrics})

    for candidate in candidate_labels:
        deltas_r = []
        deltas_rho = []
        shot_rows: list[dict[str, object]] = []
        for shots in sorted(by_shot):
            rows = [
                row
                for row in by_shot[shots]
                if row["model_label"] in baseline_labels or row["model_label"] == candidate
            ]
            metrics = corr_for(rows)
            base = baseline_by_shot[shots]
            deltas_r.append(metrics["pearson_r"] - base["pearson_r"])
            deltas_rho.append(metrics["spearman_rho_s"] - base["spearman_rho_s"])
            shot_rows.append(
                {
                    "set": f"baseline_plus_{candidate}",
                    "candidate": candidate,
                    "shots": shots,
                    **metrics,
                    "delta_pearson_r": metrics["pearson_r"] - base["pearson_r"],
                    "delta_spearman_rho_s": metrics["spearman_rho_s"] - base["spearman_rho_s"],
                }
            )
        candidate_effects.extend(shot_rows)
        candidate_rows = [row for row in model_rows if row["model_label"] == candidate]
        candidate_effects.append(
            {
                "set": "candidate_decision",
                "candidate": candidate,
                "shots": "all",
                "candidate_beta": candidate_rows[0]["pca_beta"] if candidate_rows else float("nan"),
                "candidate_mean_accuracy": float(np.mean([float(row["accuracy"]) for row in candidate_rows])),
                "mean_delta_pearson_r": float(np.mean(deltas_r)),
                "mean_delta_spearman_rho_s": float(np.mean(deltas_rho)),
                "n_shots_pearson_improved": int(sum(delta > 0 for delta in deltas_r)),
                "n_shots_spearman_improved": int(sum(delta > 0 for delta in deltas_rho)),
                "decision": "supportive"
                if np.mean(deltas_r) > 0 and np.mean(deltas_rho) >= -0.02
                else "not_supportive",
            }
        )

    return baseline_summary, candidate_effects


def plot_screen(
    outdir: Path,
    model_rows: list[dict[str, object]],
    baseline_labels: list[str],
    candidate_labels: list[str],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 5, figsize=(15.2, 3.0), sharey=True)
    colors = {
        "baseline": "#8f8f8f",
        "candidate": "#b45b4a",
    }
    for ax, shots in zip(axes, range(5)):
        rows = [row for row in model_rows if int(row["shots"]) == shots]
        base = [row for row in rows if row["model_label"] in baseline_labels]
        cand = [row for row in rows if row["model_label"] in candidate_labels]
        for part, color, size, zorder in (
            (base, colors["baseline"], 26, 2),
            (cand, colors["candidate"], 40, 3),
        ):
            xs = [float(row["pca_beta"]) for row in part]
            ys = [float(row["accuracy"]) for row in part]
            ax.scatter(xs, ys, s=size, color=color, edgecolor="white", linewidth=0.35, zorder=zorder)
        if rows:
            xs = np.asarray([float(row["pca_beta"]) for row in rows], dtype=float)
            ys = np.asarray([float(row["accuracy"]) for row in rows], dtype=float)
            if len(rows) >= 3:
                coef = np.polyfit(xs, ys, 1)
                xx = np.linspace(float(np.nanmin(xs)), float(np.nanmax(xs)), 200)
                ax.plot(xx, coef[0] * xx + coef[1], color="#6b4b3d", linewidth=1.1, zorder=1)
            metrics = corr_for(rows)
            title = rf"{shots}-shot: $\rho_s={metrics['spearman_rho_s']:.2f}$, $r={metrics['pearson_r']:.2f}$"
            ax.set_title(title, fontsize=8.5)
        for row in cand:
            x = float(row["pca_beta"])
            y = float(row["accuracy"])
            ha = "left"
            dx = 4
            if x > 7.5:
                dx = -4
                ha = "right"
            ax.annotate(
                SHORT_LABEL.get(str(row["model_label"]), str(row["model_label"])),
                (x, y),
                xytext=(dx, 4),
                textcoords="offset points",
                fontsize=5.4,
                ha=ha,
                va="center",
            )
        ax.set_xlabel(r"PCA $\beta$")
        ax.set_ylim(0.45, 1.02)
        ax.grid(True, alpha=0.28, linewidth=0.45)
        ax.tick_params(direction="in", top=True, right=True, length=3)
        if shots == 0:
            ax.set_ylabel("accuracy")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors["baseline"], label="baseline"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors["candidate"], label="candidate"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.19, top=0.78, wspace=0.16)
    fig.savefig(outdir / "candidate_screen_beta_accuracy_by_shot.pdf", bbox_inches="tight")
    fig.savefig(outdir / "candidate_screen_beta_accuracy_by_shot.png", dpi=280, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-beta", type=Path, default=DEFAULT_BASELINE_BETA)
    parser.add_argument("--baseline-sweep", type=Path, default=DEFAULT_BASELINE_SWEEP)
    parser.add_argument(
        "--candidate-sweep",
        type=Path,
        nargs="+",
        default=[DEFAULT_CANDIDATE_SWEEP],
        help="One or more candidate comparison sweep roots to merge.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--mlflow-db", type=Path, default=Path("mlflow.db"))
    parser.add_argument("--geometry-run-prefix", default="modal_geometry_candidate_screen")
    parser.add_argument(
        "--pca-selection",
        choices=("payload_best", "max_rho"),
        default="payload_best",
        help="Layer rule for candidate PCA beta. Use max_rho to require an actual numberline layer.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    beta_rows = extract_geometry_beta(
        mlflow_db=args.mlflow_db,
        run_name_prefix=args.geometry_run_prefix,
        outdir=args.outdir,
        pca_selection=args.pca_selection,
    )
    candidate_beta = {str(row["model_label"]): float(row["pca_beta"]) for row in beta_rows}
    baseline_beta = load_beta_csv(args.baseline_beta)
    beta_by_model = {**baseline_beta, **candidate_beta}
    candidate_labels = sorted(candidate_beta)

    model_rows = model_accuracy_by_shot(
        [args.baseline_sweep, *args.candidate_sweep],
        beta_by_model,
    )
    write_csv(args.outdir / "candidate_screen_model_accuracy_by_shot.csv", model_rows)

    baseline_summary, candidate_effects = summarize_sets(
        model_rows,
        baseline_labels=BASELINE_LABELS,
        candidate_labels=candidate_labels,
    )
    write_csv(args.outdir / "candidate_screen_baseline_summary.csv", baseline_summary)
    write_csv(args.outdir / "candidate_screen_candidate_effects.csv", candidate_effects)
    plot_screen(args.outdir, model_rows, BASELINE_LABELS, candidate_labels)

    decision_rows = [row for row in candidate_effects if row["set"] == "candidate_decision"]
    print(f"wrote {args.outdir}")
    for row in sorted(decision_rows, key=lambda r: float(r["mean_delta_pearson_r"]), reverse=True):
        print(
            f"{row['candidate']}: beta={float(row['candidate_beta']):.3f} "
            f"acc={float(row['candidate_mean_accuracy']):.3f} "
            f"d_r={float(row['mean_delta_pearson_r']):+.3f} "
            f"d_rho={float(row['mean_delta_spearman_rho_s']):+.3f} "
            f"{row['decision']}"
        )


if __name__ == "__main__":
    main()
