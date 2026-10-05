from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ScaleTask:
    task_id: str
    task_type: str
    scale: str
    prompt: str
    choices: dict[str, str]
    correct: str
    true_value: float | None = None
    choice_values: dict[str, float] | None = None


def _choice_block(choices: dict[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in choices.items())


def _task(
    task_id: str,
    task_type: str,
    scale: str,
    stem: str,
    choices: dict[str, str],
    correct: str,
    true_value: float | None = None,
    choice_values: dict[str, float] | None = None,
) -> ScaleTask:
    prompt = (
        f"{stem}\n"
        f"{_choice_block(choices)}\n"
        "Answer:"
    )
    return ScaleTask(
        task_id=task_id,
        task_type=task_type,
        scale=scale,
        prompt=prompt,
        choices=choices,
        correct=correct,
        true_value=true_value,
        choice_values=choice_values,
    )


def build_scale_tasks() -> list[ScaleTask]:
    """Text-only visual-scale tasks for base LLMs.

    The tasks encode chart/axis situations in text. They are intentionally
    forced-choice so base-model behavior can be evaluated by conditional
    log-likelihood rather than brittle answer parsing.
    """
    tasks: list[ScaleTask] = []

    linear_pairs = [
        (10, 100, 2.0),
        (25, 250, 1.5),
        (40, 400, 3.0),
        (80, 800, 2.5),
        (100, 1000, 2.0),
        (250, 2500, 1.2),
        (500, 5000, 1.0),
        (1000, 10000, 0.8),
    ]
    for i, (a, b, h) in enumerate(linear_pairs, 1):
        true = h * (b / a)
        sublinear = h * math.sqrt(b / a)
        opts = {
            "A": f"{h:.1f} cm",
            "B": f"{h * 2:.1f} cm",
            "C": f"{true:.1f} cm",
            "D": f"{sublinear:.1f} cm",
        }
        values = {
            "A": h,
            "B": h * 2,
            "C": true,
            "D": sublinear,
        }
        tasks.append(
            _task(
                f"linear_bar_{i:02d}",
                "bar_height",
                "linear",
                (
                    f"On a linear bar chart, value {a:,} is drawn with height "
                    f"{h:.1f} cm. What height should value {b:,} have?"
                ),
                opts,
                "C",
                true_value=true,
                choice_values=values,
            )
        )

    log_pairs = [
        (10, 10000, 1.0, 2.0),
        (100, 10000, 2.0, 3.0),
        (1, 1000, 0.0, 1.5),
        (10, 1000, 1.0, 2.5),
        (100, 100000, 2.0, 3.5),
        (1000, 1000000, 3.0, 5.0),
    ]
    for i, (a, b, pos_a, step) in enumerate(log_pairs, 1):
        true = pos_a + (math.log10(b) - math.log10(a)) * step
        linear_like = pos_a + ((b / a) - 1) * step
        opts = {
            "A": f"{pos_a + step:.1f} cm",
            "B": f"{true:.1f} cm",
            "C": f"{linear_like:.1f} cm",
            "D": f"{pos_a:.1f} cm",
        }
        values = {
            "A": pos_a + step,
            "B": true,
            "C": linear_like,
            "D": pos_a,
        }
        tasks.append(
            _task(
                f"log_axis_pos_{i:02d}",
                "axis_position",
                "log",
                (
                    f"On a log10 chart axis, value {a:,} is at position "
                    f"{pos_a:.1f} cm, and each 10x increase moves up "
                    f"{step:.1f} cm. Where should value {b:,} be?"
                ),
                opts,
                "B",
                true_value=true,
                choice_values=values,
            )
        )

    tick_tasks = [
        ("log", "1, 10, 100, __, 10000", {"A": "250", "B": "1000", "C": "5000", "D": "101"}, "B"),
        ("log", "10, 100, __, 10000, 100000", {"A": "1000", "B": "5500", "C": "200", "D": "9999"}, "A"),
        ("log", "2, 20, 200, __, 20000", {"A": "400", "B": "2000", "C": "10000", "D": "202"}, "B"),
        ("linear", "0, 2500, 5000, __, 10000", {"A": "100", "B": "7500", "C": "1000", "D": "9000"}, "B"),
        ("linear", "100, 200, __, 400, 500", {"A": "1000", "B": "250", "C": "300", "D": "2000"}, "C"),
        ("linear", "20, 40, 60, __, 100", {"A": "80", "B": "200", "C": "600", "D": "70"}, "A"),
    ]
    for i, (scale, ticks, choices, correct) in enumerate(tick_tasks, 1):
        tasks.append(
            _task(
                f"tick_completion_{i:02d}",
                "tick_completion",
                scale,
                (
                    "A chart axis has equally spaced tick marks labeled "
                    f"{ticks}. What number should fill the blank?"
                ),
                choices,
                correct,
            )
        )

    axis_choice = [
        ("10, 100, 1000, 10000", "logarithmic"),
        ("3, 30, 300, 3000", "logarithmic"),
        ("250, 500, 750, 1000", "linear"),
        ("0, 100, 200, 300", "linear"),
        ("1, 2, 4, 8", "logarithmic"),
        ("1000, 2000, 3000, 4000", "linear"),
    ]
    for i, (values, answer) in enumerate(axis_choice, 1):
        correct = "B" if answer == "logarithmic" else "A"
        tasks.append(
            _task(
                f"axis_choice_{i:02d}",
                "axis_choice",
                answer,
                (
                    "The following values appear at equally spaced vertical "
                    f"positions on a chart axis: {values}. What axis scale "
                    "does this most directly imply?"
                ),
                {
                    "A": "linear",
                    "B": "logarithmic",
                    "C": "random",
                    "D": "reversed",
                },
                correct,
            )
        )

    distance_tasks = [
        ("log", "10 and 100", "1000 and 10000", "same"),
        ("log", "5 and 50", "500 and 5000", "same"),
        ("linear", "10 and 100", "1000 and 10000", "second"),
        ("linear", "20 and 40", "2000 and 2020", "same"),
        ("linear", "100 and 200", "1000 and 2000", "second"),
        ("log", "100 and 1000", "1000 and 10000", "same"),
    ]
    for i, (scale, pair_a, pair_b, answer) in enumerate(distance_tasks, 1):
        correct = {"first": "A", "second": "B", "same": "C"}[answer]
        tasks.append(
            _task(
                f"distance_{i:02d}",
                "pair_distance",
                scale,
                (
                    f"On a {scale} chart axis, which pair is visually farther "
                    f"apart: {pair_a}, or {pair_b}?"
                ),
                {
                    "A": f"{pair_a}",
                    "B": f"{pair_b}",
                    "C": "they are the same distance",
                    "D": "the axis scale is irrelevant",
                },
                correct,
            )
        )

    return _rebalance_correct_letters(tasks)


def _rebalance_correct_letters(tasks: list[ScaleTask]) -> list[ScaleTask]:
    """Rotate options so correct labels are balanced across A/B/C/D."""
    labels = ["A", "B", "C", "D"]
    out: list[ScaleTask] = []
    for i, task in enumerate(tasks):
        desired = labels[i % len(labels)]
        offset = labels.index(desired) - labels.index(task.correct)
        choices: dict[str, str] = {}
        choice_values: dict[str, float] | None = {} if task.choice_values is not None else None
        for old_label in labels:
            new_label = labels[(labels.index(old_label) + offset) % len(labels)]
            choices[new_label] = task.choices[old_label]
            if choice_values is not None:
                choice_values[new_label] = task.choice_values[old_label]
        ordered_choices = {label: choices[label] for label in labels}
        ordered_values = (
            {label: choice_values[label] for label in labels}
            if choice_values is not None else None
        )
        prompt = task.prompt.split("\nA. ", 1)[0].rstrip()
        out.append(
            _task(
                task.task_id,
                task.task_type,
                task.scale,
                prompt,
                ordered_choices,
                desired,
                true_value=task.true_value,
                choice_values=ordered_values,
            )
        )
    return out


def summarize_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    n = len(rows)
    correct = sum(int(r["correct_selected"]) for r in rows)
    numeric_rows = [r for r in rows if r.get("selected_log10_error") is not None]
    mean_abs_log10_error = (
        sum(abs(float(r["selected_log10_error"])) for r in numeric_rows) / len(numeric_rows)
        if numeric_rows else None
    )
    compression_rows = [
        r for r in numeric_rows
        if r["scale"] == "linear" and float(r["selected_log10_error"]) < 0
    ]
    expansion_rows = [
        r for r in numeric_rows
        if r["scale"] == "log" and float(r["selected_log10_error"]) > 0
    ]
    by_type: dict[str, dict[str, float]] = {}
    for r in rows:
        blob = by_type.setdefault(r["task_type"], {"n": 0, "correct": 0})
        blob["n"] += 1
        blob["correct"] += int(r["correct_selected"])
    for blob in by_type.values():
        blob["accuracy"] = blob["correct"] / blob["n"] if blob["n"] else float("nan")
    return {
        "n_tasks": n,
        "accuracy": correct / n if n else float("nan"),
        "n_correct": correct,
        "mean_abs_log10_error": mean_abs_log10_error,
        "linear_compression_error_rate": (
            len(compression_rows) / len([r for r in numeric_rows if r["scale"] == "linear"])
            if any(r["scale"] == "linear" for r in numeric_rows) else None
        ),
        "log_expansion_error_rate": (
            len(expansion_rows) / len([r for r in numeric_rows if r["scale"] == "log"])
            if any(r["scale"] == "log" for r in numeric_rows) else None
        ),
        "by_type": by_type,
    }


def write_results(outdir: Path, payloads: list[dict]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summaries: list[dict] = []
    for payload in payloads:
        rows.extend(payload["rows"])
        summary = dict(payload["summary"])
        summary["model_label"] = payload["model_label"]
        summary["model_name"] = payload["model_name"]
        summary["model_revision"] = payload.get("model_revision") or ""
        summaries.append(summary)

    if rows:
        with open(outdir / "motivation_scale_rows.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if summaries:
        keys = [
            "model_label",
            "model_name",
            "model_revision",
            "n_tasks",
            "n_correct",
            "accuracy",
            "mean_abs_log10_error",
            "linear_compression_error_rate",
            "log_expansion_error_rate",
        ]
        with open(outdir / "motivation_scale_summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in summaries:
                writer.writerow({k: row.get(k) for k in keys})
    with open(outdir / "motivation_scale_payloads.json", "w") as fh:
        json.dump(payloads, fh, indent=2)


def load_beta_table(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = row.get("model_label") or row.get("model")
            beta = row.get("pca_beta_mean") or row.get("pca_beta")
            if not label or beta in {None, ""}:
                continue
            out[label] = float(beta)
    return out


def plot_results(
    outdir: Path,
    beta_csv: Path = Path("results/dataset_frequency_0_to_10000/alpha_beta_without_pile/alpha_beta_models_without_pile.csv"),
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    summary_path = outdir / "motivation_scale_summary.csv"
    rows_path = outdir / "motivation_scale_rows.csv"
    if not summary_path.exists() or not rows_path.exists():
        return
    beta = load_beta_table(beta_csv)

    summaries: list[dict] = []
    with open(summary_path, newline="") as fh:
        for row in csv.DictReader(fh):
            row["accuracy"] = float(row["accuracy"])
            row["mean_abs_log10_error"] = (
                float(row["mean_abs_log10_error"])
                if row["mean_abs_log10_error"] not in {"", "None"} else float("nan")
            )
            row["pca_beta"] = beta.get(row["model_label"], float("nan"))
            summaries.append(row)

    if summaries:
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        xs = np.array([r["pca_beta"] for r in summaries], dtype=float)
        ys = np.array([r["accuracy"] for r in summaries], dtype=float)
        ax.scatter(xs, ys, s=52, color="#2f6f8f")
        for r in summaries:
            ax.annotate(
                r["model_label"].replace("StarCoderBase", "SCB"),
                (r["pca_beta"], r["accuracy"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() >= 2:
            coef = np.polyfit(xs[ok], ys[ok], 1)
            xx = np.linspace(xs[ok].min(), xs[ok].max(), 100)
            ax.plot(xx, coef[0] * xx + coef[1], color="#c7503d", linewidth=1.4)
        ax.set_xlabel(r"PCA compression factor $\beta$")
        ax.set_ylabel("Scale-task accuracy")
        ax.set_title("Text-encoded visual scale reasoning")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / "motivation_beta_vs_accuracy.pdf")
        fig.savefig(outdir / "motivation_beta_vs_accuracy.png", dpi=220)
        plt.close(fig)

    rows: list[dict] = []
    with open(rows_path, newline="") as fh:
        for row in csv.DictReader(fh):
            row["correct_selected"] = int(row["correct_selected"])
            rows.append(row)
    task_types = sorted({r["task_type"] for r in rows})
    labels = [r["model_label"] for r in summaries]
    matrix = np.zeros((len(labels), len(task_types)))
    for i, label in enumerate(labels):
        for j, task_type in enumerate(task_types):
            sub = [r for r in rows if r["model_label"] == label and r["task_type"] == task_type]
            matrix[i, j] = sum(r["correct_selected"] for r in sub) / len(sub) if sub else np.nan

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(task_types)), [t.replace("_", "\n") for t in task_types], fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title("Accuracy by text-visual task type")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("accuracy")
    fig.tight_layout()
    fig.savefig(outdir / "motivation_task_type_heatmap.pdf")
    fig.savefig(outdir / "motivation_task_type_heatmap.png", dpi=220)
    plt.close(fig)


def task_dicts() -> list[dict]:
    return [asdict(task) for task in build_scale_tasks()]
