from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BiasTask:
    task_id: str
    task_type: str
    prompt: str
    candidates: dict[str, str]
    positive_key: str
    negative_key: str
    positive_meaning: str
    metadata: dict[str, float | int | str]


def _task(
    task_id: str,
    task_type: str,
    prompt: str,
    candidates: dict[str, str],
    positive_key: str,
    negative_key: str,
    positive_meaning: str,
    **metadata: float | int | str,
) -> BiasTask:
    return BiasTask(
        task_id=task_id,
        task_type=task_type,
        prompt=prompt.rstrip(),
        candidates=candidates,
        positive_key=positive_key,
        negative_key=negative_key,
        positive_meaning=positive_meaning,
        metadata=metadata,
    )


def build_bias_tasks() -> list[BiasTask]:
    tasks: list[BiasTask] = []

    # Equal-ratio intervals. Positive score means the higher-magnitude interval
    # is treated as larger than the lower-magnitude interval.
    intervals = [
        (1, 10, 10, 100),
        (10, 100, 100, 1000),
        (100, 1000, 1000, 10000),
        (2, 20, 20, 200),
        (20, 200, 200, 2000),
        (200, 2000, 2000, 20000),
        (5, 50, 50, 500),
        (50, 500, 500, 5000),
        (500, 5000, 5000, 50000),
    ]
    for i, (la, lb, ha, hb) in enumerate(intervals, 1):
        for order in ("low_first", "high_first"):
            if order == "low_first":
                first = f"{la:,} to {lb:,}"
                second = f"{ha:,} to {hb:,}"
                positive_key = "second"
                negative_key = "first"
            else:
                first = f"{ha:,} to {hb:,}"
                second = f"{la:,} to {lb:,}"
                positive_key = "first"
                negative_key = "second"
            tasks.append(
                _task(
                    f"interval_equal_ratio_{i:02d}_{order}",
                    "interval_high_magnitude_bias",
                    (
                        "Two number intervals have the same multiplicative ratio. "
                        f"Interval A is {first}. Interval B is {second}. "
                        "On an internal number scale, the larger-feeling interval is"
                    ),
                    {
                        "first": " interval A",
                        "second": " interval B",
                        "same": " neither; they feel equal",
                    },
                    positive_key,
                    negative_key,
                    "higher-magnitude equal-ratio interval preferred",
                    low_start=la,
                    low_end=lb,
                    high_start=ha,
                    high_end=hb,
                    order=order,
                )
            )

    # Midpoint probes. Positive score means geometric midpoint is preferred over
    # arithmetic midpoint, i.e. a more logarithmic scale preference.
    midpoint_pairs = [
        (1, 100),
        (10, 1000),
        (100, 10000),
        (1, 1000),
        (10, 10000),
        (100, 100000),
        (2, 200),
        (20, 2000),
        (5, 500),
        (50, 5000),
    ]
    for i, (a, b) in enumerate(midpoint_pairs, 1):
        geom = math.sqrt(a * b)
        arith = (a + b) / 2
        geom_s = f"{geom:,.0f}" if abs(geom - round(geom)) < 1e-9 else f"{geom:,.2f}"
        arith_s = f"{arith:,.0f}" if abs(arith - round(arith)) < 1e-9 else f"{arith:,.1f}"
        tasks.append(
            _task(
                f"midpoint_log_vs_linear_{i:02d}",
                "midpoint_log_bias",
                (
                    f"A scale places {a:,} and {b:,} at its two ends. "
                    "The number that feels halfway between them is"
                ),
                {
                    "geometric": f" {geom_s}",
                    "arithmetic": f" {arith_s}",
                },
                "geometric",
                "arithmetic",
                "geometric midpoint preferred over arithmetic midpoint",
                start=a,
                end=b,
                geometric_midpoint=geom,
                arithmetic_midpoint=arith,
            )
        )

    # Tick continuation. Positive score means multiplicative continuation is
    # preferred over additive continuation.
    tick_specs = [
        ([1, 10, 100], 1000, 190),
        ([10, 100, 1000], 10000, 1900),
        ([2, 20, 200], 2000, 380),
        ([5, 50, 500], 5000, 950),
        ([100, 1000, 10000], 100000, 19000),
        ([3, 30, 300], 3000, 570),
        ([4, 40, 400], 4000, 760),
        ([8, 80, 800], 8000, 1520),
    ]
    for i, (ticks, log_next, linear_next) in enumerate(tick_specs, 1):
        tick_text = ", ".join(f"{x:,}" for x in ticks)
        tasks.append(
            _task(
                f"tick_log_continuation_{i:02d}",
                "tick_log_continuation_bias",
                (
                    "A scale has equally spaced tick labels "
                    f"{tick_text}, __. The next tick is most naturally"
                ),
                {
                    "log": f" {log_next:,}",
                    "linear": f" {linear_next:,}",
                },
                "log",
                "linear",
                "multiplicative tick continuation preferred",
                log_next=log_next,
                linear_next=linear_next,
            )
        )

    # Similarity probes. Positive score means ratio-near option beats
    # difference-near option.
    similarity = [
        (1000, 100, 1900),
        (10000, 1000, 19000),
        (500, 50, 950),
        (2000, 200, 3800),
        (3000, 300, 5700),
        (8000, 800, 15200),
        (100, 10, 190),
        (100000, 10000, 190000),
    ]
    for i, (target, ratio_near, diff_near) in enumerate(similarity, 1):
        tasks.append(
            _task(
                f"similarity_ratio_vs_difference_{i:02d}",
                "similarity_ratio_bias",
                (
                    f"The number {target:,} is more similar in magnitude to"
                ),
                {
                    "ratio": f" {ratio_near:,}",
                    "difference": f" {diff_near:,}",
                },
                "ratio",
                "difference",
                "ratio-near number preferred over difference-near number",
                target=target,
                ratio_near=ratio_near,
                difference_near=diff_near,
            )
        )

    # Balanced A/B versions of the numberline probes. These avoid scoring raw
    # number strings directly, so the model is compared on " A" vs " B" rather
    # than on tokenization/frequency of candidate numbers.
    balanced_midpoints = [
        (1, 100),
        (10, 1000),
        (100, 10000),
        (2, 200),
        (20, 2000),
        (5, 500),
        (50, 5000),
    ]
    for i, (a, b) in enumerate(balanced_midpoints, 1):
        geom = math.sqrt(a * b)
        arith = (a + b) / 2
        geom_s = f"{geom:,.0f}" if abs(geom - round(geom)) < 1e-9 else f"{geom:,.2f}"
        arith_s = f"{arith:,.0f}" if abs(arith - round(arith)) < 1e-9 else f"{arith:,.1f}"
        for order in ("log_first", "linear_first"):
            if order == "log_first":
                choices = f"A. {geom_s}\nB. {arith_s}"
                candidates = {"log": " A", "linear": " B"}
            else:
                choices = f"A. {arith_s}\nB. {geom_s}"
                candidates = {"log": " B", "linear": " A"}
            tasks.append(
                _task(
                    f"balanced_midpoint_log_vs_linear_{i:02d}_{order}",
                    "balanced_midpoint_log_bias",
                    (
                        f"A scale places {a:,} and {b:,} at its two ends. "
                        "Which option is a better intuitive halfway point in magnitude?\n"
                        f"{choices}\nAnswer:"
                    ),
                    candidates,
                    "log",
                    "linear",
                    "geometric midpoint preferred over arithmetic midpoint",
                    start=a,
                    end=b,
                    geometric_midpoint=geom,
                    arithmetic_midpoint=arith,
                    order=order,
                )
            )

    # Landmark closeness cases where linear distance and log-distance disagree.
    # The target is between the geometric and arithmetic midpoint, so a linear
    # scale picks the lower anchor while a log-like magnitude scale picks the
    # upper anchor.
    landmark_cases = [
        (10, 100, 40),
        (100, 1000, 400),
        (1000, 10000, 4000),
        (1, 100, 20),
        (10, 1000, 200),
        (100, 10000, 2000),
        (2, 200, 40),
        (20, 2000, 400),
        (5, 500, 100),
        (50, 5000, 1000),
    ]
    for i, (low, high, target) in enumerate(landmark_cases, 1):
        for order in ("log_first", "linear_first"):
            if order == "log_first":
                choices = f"A. {high:,}\nB. {low:,}"
                candidates = {"log": " A", "linear": " B"}
            else:
                choices = f"A. {low:,}\nB. {high:,}"
                candidates = {"log": " B", "linear": " A"}
            tasks.append(
                _task(
                    f"balanced_landmark_log_closeness_{i:02d}_{order}",
                    "balanced_landmark_log_closeness_bias",
                    (
                        f"Consider the rough magnitude of {target:,} between {low:,} and {high:,}. "
                        "Which landmark is it closer to in magnitude?\n"
                        f"{choices}\nAnswer:"
                    ),
                    candidates,
                    "log",
                    "linear",
                    "log-distance landmark preferred over linear-distance landmark",
                    low=low,
                    high=high,
                    target=target,
                    order=order,
                )
            )

    balanced_ticks = [
        ([1, 10, 100], 1000, 190),
        ([10, 100, 1000], 10000, 1900),
        ([2, 20, 200], 2000, 380),
        ([5, 50, 500], 5000, 950),
        ([3, 30, 300], 3000, 570),
        ([8, 80, 800], 8000, 1520),
    ]
    for i, (ticks, log_next, linear_next) in enumerate(balanced_ticks, 1):
        tick_text = ", ".join(f"{x:,}" for x in ticks)
        for order in ("log_first", "linear_first"):
            if order == "log_first":
                choices = f"A. {log_next:,}\nB. {linear_next:,}"
                candidates = {"log": " A", "linear": " B"}
            else:
                choices = f"A. {linear_next:,}\nB. {log_next:,}"
                candidates = {"log": " B", "linear": " A"}
            tasks.append(
                _task(
                    f"balanced_tick_log_continuation_{i:02d}_{order}",
                    "balanced_tick_log_continuation_bias",
                    (
                        "A scale has equally spaced tick labels "
                        f"{tick_text}, __. Which next tick best preserves the pattern?\n"
                        f"{choices}\nAnswer:"
                    ),
                    candidates,
                    "log",
                    "linear",
                    "multiplicative tick continuation preferred",
                    log_next=log_next,
                    linear_next=linear_next,
                    order=order,
                )
            )

    balanced_similarity = [
        (1000, 100, 1900),
        (10000, 1000, 19000),
        (500, 50, 950),
        (2000, 200, 3800),
        (3000, 300, 5700),
        (8000, 800, 15200),
        (100, 10, 190),
    ]
    for i, (target, ratio_near, diff_near) in enumerate(balanced_similarity, 1):
        for order in ("ratio_first", "difference_first"):
            if order == "ratio_first":
                choices = f"A. {ratio_near:,}\nB. {diff_near:,}"
                candidates = {"log": " A", "linear": " B"}
            else:
                choices = f"A. {diff_near:,}\nB. {ratio_near:,}"
                candidates = {"log": " B", "linear": " A"}
            tasks.append(
                _task(
                    f"balanced_similarity_ratio_vs_difference_{i:02d}_{order}",
                    "balanced_similarity_ratio_bias",
                    (
                        f"Which number is more similar in magnitude to {target:,}?\n"
                        f"{choices}\nAnswer:"
                    ),
                    candidates,
                    "log",
                    "linear",
                    "ratio-near number preferred over difference-near number",
                    target=target,
                    ratio_near=ratio_near,
                    difference_near=diff_near,
                    order=order,
                )
            )

    # Pair-distance judgments. Positive score means the model treats a larger
    # ratio change as the larger magnitude difference, even when the alternative
    # has a larger absolute difference.
    pair_distance_cases = [
        ((10, 20), (1_000, 1_100)),
        ((20, 40), (2_000, 2_200)),
        ((50, 100), (5_000, 5_500)),
        ((100, 200), (9_000, 9_500)),
        ((30, 60), (3_000, 3_300)),
        ((100, 150), (8_000, 8_500)),
        ((200, 300), (7_000, 7_600)),
        ((400, 800), (4_000, 4_400)),
    ]
    for i, (ratio_pair, difference_pair) in enumerate(pair_distance_cases, 1):
        ratio_abs = abs(ratio_pair[1] - ratio_pair[0])
        difference_abs = abs(difference_pair[1] - difference_pair[0])
        ratio_factor = max(ratio_pair) / min(ratio_pair)
        difference_factor = max(difference_pair) / min(difference_pair)
        for order in ("ratio_first", "difference_first"):
            if order == "ratio_first":
                choices = (
                    f"A. {ratio_pair[0]:,} and {ratio_pair[1]:,}\n"
                    f"B. {difference_pair[0]:,} and {difference_pair[1]:,}"
                )
                candidates = {"log": " A", "linear": " B"}
            else:
                choices = (
                    f"A. {difference_pair[0]:,} and {difference_pair[1]:,}\n"
                    f"B. {ratio_pair[0]:,} and {ratio_pair[1]:,}"
                )
                candidates = {"log": " B", "linear": " A"}
            tasks.append(
                _task(
                    f"balanced_pair_ratio_difference_{i:02d}_{order}",
                    "balanced_pair_ratio_difference_bias",
                    (
                        "Which pair looks more different in magnitude?\n"
                        f"{choices}\nAnswer:"
                    ),
                    candidates,
                    "log",
                    "linear",
                    "larger-ratio pair preferred over larger-absolute-difference pair",
                    ratio_pair_low=ratio_pair[0],
                    ratio_pair_high=ratio_pair[1],
                    difference_pair_low=difference_pair[0],
                    difference_pair_high=difference_pair[1],
                    ratio_abs_difference=ratio_abs,
                    linear_abs_difference=difference_abs,
                    ratio_factor=ratio_factor,
                    linear_factor=difference_factor,
                    order=order,
                )
            )

    # Change-size judgments. Positive score means a proportional change is
    # treated as more substantial than a larger absolute change.
    change_cases = [
        ((10, 20), (1_000, 1_100)),
        ((20, 40), (2_000, 2_200)),
        ((50, 100), (5_000, 5_500)),
        ((100, 200), (9_000, 9_500)),
        ((30, 60), (3_000, 3_300)),
        ((100, 150), (8_000, 8_500)),
        ((200, 300), (7_000, 7_600)),
        ((400, 800), (4_000, 4_400)),
    ]
    for i, (ratio_change, difference_change) in enumerate(change_cases, 1):
        for order in ("ratio_first", "difference_first"):
            if order == "ratio_first":
                choices = (
                    f"A. {ratio_change[0]:,} to {ratio_change[1]:,}\n"
                    f"B. {difference_change[0]:,} to {difference_change[1]:,}"
                )
                candidates = {"log": " A", "linear": " B"}
            else:
                choices = (
                    f"A. {difference_change[0]:,} to {difference_change[1]:,}\n"
                    f"B. {ratio_change[0]:,} to {ratio_change[1]:,}"
                )
                candidates = {"log": " B", "linear": " A"}
            tasks.append(
                _task(
                    f"balanced_change_ratio_difference_{i:02d}_{order}",
                    "balanced_change_ratio_difference_bias",
                    (
                        "Which change feels more substantial in magnitude?\n"
                        f"{choices}\nAnswer:"
                    ),
                    candidates,
                    "log",
                    "linear",
                    "larger-ratio change preferred over larger-absolute change",
                    ratio_start=ratio_change[0],
                    ratio_end=ratio_change[1],
                    linear_start=difference_change[0],
                    linear_end=difference_change[1],
                    ratio_factor=ratio_change[1] / ratio_change[0],
                    linear_factor=difference_change[1] / difference_change[0],
                    ratio_abs_difference=ratio_change[1] - ratio_change[0],
                    linear_abs_difference=difference_change[1] - difference_change[0],
                    order=order,
                )
            )

    # Position-estimation judgments. These are intentionally framed as an
    # intuitive magnitude line, not an exact arithmetic number line.
    position_cases = [
        (1, 10_000, 10, "about one quarter of the way from the left", "very close to the left end"),
        (1, 10_000, 100, "about halfway across", "very close to the left end"),
        (1, 10_000, 1_000, "about three quarters of the way across", "near the left side"),
        (10, 100_000, 100, "about one quarter of the way from the left", "very close to the left end"),
        (10, 100_000, 1_000, "about halfway across", "very close to the left end"),
        (10, 100_000, 10_000, "about three quarters of the way across", "near the left side"),
        (100, 10_000, 1_000, "about halfway across", "near the left side"),
    ]
    for i, (low, high, target, log_position, linear_position) in enumerate(position_cases, 1):
        for order in ("log_first", "linear_first"):
            if order == "log_first":
                choices = f"A. {log_position}\nB. {linear_position}"
                candidates = {"log": " A", "linear": " B"}
            else:
                choices = f"A. {linear_position}\nB. {log_position}"
                candidates = {"log": " B", "linear": " A"}
            tasks.append(
                _task(
                    f"balanced_position_log_vs_linear_{i:02d}_{order}",
                    "balanced_position_log_bias",
                    (
                        f"An intuitive magnitude line runs from {low:,} on the left to {high:,} on the right. "
                        f"Where should {target:,} fall?\n"
                        f"{choices}\nAnswer:"
                    ),
                    candidates,
                    "log",
                    "linear",
                    "log-position preferred over linear-position",
                    low=low,
                    high=high,
                    target=target,
                    log_position=log_position,
                    linear_position=linear_position,
                    order=order,
                )
            )

    return tasks


def summarize_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    by_type: dict[str, dict[str, float]] = {}
    for row in rows:
        blob = by_type.setdefault(
            row["task_type"],
            {"n": 0, "mean_signed_score": 0.0, "positive_rate": 0.0},
        )
        signed = float(row["signed_score"])
        blob["n"] += 1
        blob["mean_signed_score"] += signed
        blob["positive_rate"] += float(signed > 0)
    for blob in by_type.values():
        if blob["n"]:
            blob["mean_signed_score"] /= blob["n"]
            blob["positive_rate"] /= blob["n"]

    return {
        "n_tasks": len(rows),
        "mean_signed_score": (
            sum(float(row["signed_score"]) for row in rows) / len(rows)
            if rows else float("nan")
        ),
        "positive_rate": (
            sum(float(row["signed_score"]) > 0 for row in rows) / len(rows)
            if rows else float("nan")
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
        for task_type, blob in summary["by_type"].items():
            summary[f"{task_type}_mean_signed_score"] = blob["mean_signed_score"]
            summary[f"{task_type}_positive_rate"] = blob["positive_rate"]
        summaries.append(summary)

    if rows:
        keys = sorted({key for row in rows for key in row})
        with open(outdir / "motivation_bias_rows.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    if summaries:
        task_types = sorted({
            task_type
            for summary in summaries
            for task_type in summary.get("by_type", {})
        })
        keys = [
            "model_label",
            "model_name",
            "model_revision",
            "n_tasks",
            "mean_signed_score",
            "positive_rate",
        ]
        for task_type in task_types:
            keys.extend([
                f"{task_type}_mean_signed_score",
                f"{task_type}_positive_rate",
            ])
        with open(outdir / "motivation_bias_summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for summary in summaries:
                writer.writerow({key: summary.get(key) for key in keys})

    with open(outdir / "motivation_bias_payloads.json", "w") as fh:
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
            if label and beta not in {None, ""}:
                out[label] = float(beta)
    return out


def _read_summary(summary_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(summary_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


TASK_LABELS = {
    "interval_high_magnitude_bias": "interval\nhigh gap",
    "midpoint_log_bias": "midpoint\nlog",
    "tick_log_continuation_bias": "tick\nlog",
    "similarity_ratio_bias": "similarity\nratio",
    "balanced_landmark_log_closeness_bias": "balanced\ncloseness",
    "balanced_midpoint_log_bias": "balanced\nmidpoint",
    "balanced_tick_log_continuation_bias": "balanced\ntick",
    "balanced_similarity_ratio_bias": "balanced\nsimilarity",
    "balanced_pair_ratio_difference_bias": "pair\nratio gap",
    "balanced_change_ratio_difference_bias": "change\nratio gap",
    "balanced_position_log_bias": "position\nlog",
}


TASK_ORDER = [
    "interval_high_magnitude_bias",
    "midpoint_log_bias",
    "tick_log_continuation_bias",
    "similarity_ratio_bias",
    "balanced_landmark_log_closeness_bias",
    "balanced_midpoint_log_bias",
    "balanced_tick_log_continuation_bias",
    "balanced_similarity_ratio_bias",
    "balanced_pair_ratio_difference_bias",
    "balanced_change_ratio_difference_bias",
    "balanced_position_log_bias",
]


def _task_score_cols(rows: list[dict]) -> list[str]:
    available = {
        key[: -len("_mean_signed_score")]
        for row in rows
        for key in row
        if key.endswith("_mean_signed_score")
        and key != "mean_signed_score"
        and row.get(key) not in {None, ""}
    }
    ordered = [task for task in TASK_ORDER if task in available]
    ordered.extend(sorted(available - set(ordered)))
    return [f"{task}_mean_signed_score" for task in ordered]


def plot_results(
    outdir: Path,
    beta_csv: Path = Path("results/dataset_frequency_0_to_10000/alpha_beta_without_pile/alpha_beta_models_without_pile.csv"),
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    summary_path = outdir / "motivation_bias_summary.csv"
    if not summary_path.exists():
        return
    beta = load_beta_table(beta_csv)
    rows = _read_summary(summary_path)
    for row in rows:
        row["pca_beta"] = beta.get(row["model_label"], float("nan"))

    task_cols = _task_score_cols(rows)
    labels = [
        TASK_LABELS.get(col[: -len("_mean_signed_score")], col.replace("_mean_signed_score", "").replace("_", "\n"))
        for col in task_cols
    ]

    fig, axes = plt.subplots(1, len(task_cols), figsize=(2.8 * len(task_cols), 3.2), sharex=True)
    if len(task_cols) == 1:
        axes = [axes]
    for ax, col, label in zip(axes, task_cols, labels):
        xs = np.array([float(row["pca_beta"]) for row in rows], dtype=float)
        ys = np.array([float(row.get(col) or "nan") for row in rows], dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.scatter(xs[ok], ys[ok], s=42, color="#2f6f8f")
        if ok.sum() >= 2:
            coef = np.polyfit(xs[ok], ys[ok], 1)
            xx = np.linspace(xs[ok].min(), xs[ok].max(), 100)
            ax.plot(xx, coef[0] * xx + coef[1], color="#c7503d", linewidth=1.2)
        ax.set_title(label)
        ax.set_xlabel(r"$\beta_{\mathrm{PCA}}$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("signed log-likelihood bias")
    fig.tight_layout()
    fig.savefig(outdir / "motivation_bias_beta_panels.pdf")
    fig.savefig(outdir / "motivation_bias_beta_panels.png", dpi=220)
    plt.close(fig)

    model_labels = [row["model_label"] for row in rows]
    matrix = np.array([[float(row.get(col) or "nan") for col in task_cols] for row in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    vmax = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(model_labels)), model_labels, fontsize=8)
    ax.set_xticks(range(len(labels)), [x.replace("\n", " ") for x in labels], fontsize=8)
    ax.set_title("Signed scale-bias scores")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("positive direction score")
    fig.tight_layout()
    fig.savefig(outdir / "motivation_bias_heatmap.pdf")
    fig.savefig(outdir / "motivation_bias_heatmap.png", dpi=220)
    plt.close(fig)


def correlation_summary(
    outdir: Path,
    beta_csv: Path = Path("results/dataset_frequency_0_to_10000/alpha_beta_without_pile/alpha_beta_models_without_pile.csv"),
) -> dict[str, float]:
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    summary_path = outdir / "motivation_bias_summary.csv"
    if not summary_path.exists():
        return {}
    beta = load_beta_table(beta_csv)
    rows = _read_summary(summary_path)
    cols = ["mean_signed_score"] + _task_score_cols(rows)
    out: dict[str, float] = {}
    xs = np.array([beta.get(row["model_label"], float("nan")) for row in rows], dtype=float)
    for col in cols:
        ys = np.array([float(row.get(col) or "nan") for row in rows], dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() >= 2:
            out[f"spearman_beta_{col}"] = float(spearmanr(xs[ok], ys[ok]).statistic)
            out[f"pearson_beta_{col}"] = float(pearsonr(xs[ok], ys[ok]).statistic)
    with open(outdir / "motivation_bias_correlations.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    return out


def write_latex_summary(
    outdir: Path,
    beta_csv: Path = Path("results/dataset_frequency_0_to_10000/alpha_beta_without_pile/alpha_beta_models_without_pile.csv"),
) -> None:
    rows = _read_summary(outdir / "motivation_bias_summary.csv")
    beta = load_beta_table(beta_csv)
    task_cols = _task_score_cols(rows)
    # Keep the table compact by prioritizing the balanced probes.
    preferred = [
        "balanced_landmark_log_closeness_bias_mean_signed_score",
        "balanced_midpoint_log_bias_mean_signed_score",
        "balanced_tick_log_continuation_bias_mean_signed_score",
        "balanced_similarity_ratio_bias_mean_signed_score",
    ]
    table_cols = [col for col in preferred if col in task_cols] or task_cols[:4]
    path = outdir / "motivation_bias_summary.tex"
    with open(path, "w") as fh:
        fh.write("\\begin{table}[h!]\n")
        fh.write("    \\centering\n")
        fh.write("    \\resizebox{\\linewidth}{!}{%\n")
        fh.write("    \\begin{tabular}{l" + "c" * (len(table_cols) + 1) + "}\n")
        fh.write("        \\toprule\n")
        headers = [
            TASK_LABELS.get(col[: -len("_mean_signed_score")], col).replace("\n", " ")
            for col in table_cols
        ]
        fh.write("        Model & $\\beta$ & " + " & ".join(headers) + " \\\\\n")
        fh.write("        \\midrule\n")
        for row in rows:
            label = row["model_label"]
            values = " & ".join(f"{float(row.get(col) or 'nan'):+.2f}" for col in table_cols)
            fh.write(f"        {label} & {beta.get(label, float('nan')):.2f} & {values} \\\\\n")
        fh.write("        \\bottomrule\n")
        fh.write("    \\end{tabular}}\n")
        fh.write("    \\caption{Signed likelihood-bias probes for text-only scale behavior. Positive scores indicate log-like magnitude preferences over linear alternatives.}\n")
        fh.write("    \\label{tab:motivation-bias-probes}\n")
        fh.write("\\end{table}\n")


def task_dicts() -> list[dict]:
    return [asdict(task) for task in build_bias_tasks()]
