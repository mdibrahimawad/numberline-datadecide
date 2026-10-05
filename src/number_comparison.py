from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PCA_GROUPS = {
    1: (1, 39),
    2: (80, 119),
    3: (980, 1019),
    4: (9980, 10000),
}

DENSE_FIXED_GAP_WINDOWS = {
    1: (10, 30),
    2: (90, 110),
    3: (990, 1010),
    4: (9980, 10000),
}

WIDE_FIXED_GAP_WINDOWS = {
    1: (10, 99),
    2: (100, 999),
    3: (1000, 9999),
    4: (10000, 99999),
}

WIDE_FIXED_GAP_REGIME_GAPS = tuple(range(1, 11))
WIDE_FIXED_GAP_STARTS_PER_GAP = 32
WIDE_FIXED_GAP_SEED = 20260505


@dataclass(frozen=True)
class ComparisonTask:
    task_id: str
    family: str
    a_value: int
    b_value: int
    query: str
    order: str
    group: int
    gap: int
    ratio: float
    n_shots: int
    exemplar_set_id: int
    correct_key: str
    prompt: str
    candidates: dict[str, str]


def _group_for_pair(a: int, b: int) -> int:
    mid = (a + b) / 2.0
    best_group = 1
    best_dist = float("inf")
    for group, (lo, hi) in PCA_GROUPS.items():
        center = (lo + hi) / 2.0
        dist = abs(math.log10(max(mid, 1)) - math.log10(max(center, 1)))
        if dist < best_dist:
            best_group = group
            best_dist = dist
    return best_group


SHOT_EXAMPLE_SETS = {
    0: (
        (3, 8, "larger", 8),
        (17, 12, "smaller", 12),
        (104, 98, "larger", 104),
        (1007, 999, "smaller", 999),
    ),
    1: (
        (24, 9, "smaller", 9),
        (6, 14, "larger", 14),
        (230, 205, "smaller", 205),
        (2012, 2040, "larger", 2040),
    ),
    2: (
        (31, 27, "larger", 31),
        (45, 52, "smaller", 45),
        (880, 910, "larger", 910),
        (7005, 6998, "smaller", 6998),
    ),
}


def _format_comparison_block(
    first: int,
    second: int,
    query: str,
    answer_mode: str,
    *,
    answer: int | str | None = None,
) -> str:
    if answer_mode == "label":
        block = (
            "Compare the two integers.\n"
            f"A: {first}\n"
            f"B: {second}\n"
            f"Question: Which number is {query}?\n"
            "Answer with A or B.\n"
            "Answer:"
        )
        if answer is not None:
            block += f" {answer}"
        return block
    if answer_mode == "number":
        block = (
            "Compare the two integers.\n"
            f"First number: {first}\n"
            f"Second number: {second}\n"
            f"Question: Which number is {query}?\n"
            "Answer with the number only.\n"
            "Answer:"
        )
        if answer is not None:
            block += f" {answer}"
        return block
    raise ValueError(f"unknown answer_mode={answer_mode}")


def _comparison_prompt(
    first: int,
    second: int,
    query: str,
    answer_mode: str,
    n_shots: int,
    exemplar_set_id: int,
) -> str:
    if n_shots not in {0, 1, 2, 3, 4}:
        raise ValueError(f"n_shots must be one of 0, 1, 2, 3, 4; got {n_shots}")
    if exemplar_set_id not in SHOT_EXAMPLE_SETS:
        raise ValueError(
            f"unknown exemplar_set_id={exemplar_set_id}; "
            f"available={sorted(SHOT_EXAMPLE_SETS)}"
        )
    target = _format_comparison_block(first, second, query, answer_mode)
    if n_shots == 0:
        return target

    examples = []
    for ex_first, ex_second, ex_query, ex_answer_value in SHOT_EXAMPLE_SETS[exemplar_set_id][:n_shots]:
        if answer_mode == "label":
            ex_answer = "A" if ex_first == ex_answer_value else "B"
        else:
            ex_answer = ex_answer_value
        examples.append(
            _format_comparison_block(
                ex_first,
                ex_second,
                ex_query,
                answer_mode,
                answer=ex_answer,
            )
        )
    return "Examples:\n\n" + "\n\n".join(examples) + "\n\nNow solve:\n" + target


def _expand_pair(
    tasks: list[ComparisonTask],
    *,
    family: str,
    a: int,
    b: int,
    group: int | None = None,
    answer_mode: str = "label",
    n_shots: int = 0,
    exemplar_set_id: int = 0,
) -> None:
    if a == b or a < 0 or b < 0:
        return
    lo = min(a, b)
    hi = max(a, b)
    gap = hi - lo
    ratio = hi / max(lo, 1)
    group = _group_for_pair(a, b) if group is None else group
    pair_id = f"{family}_{lo}_{hi}"
    for order, first, second in (("ab", a, b), ("ba", b, a)):
        for query in ("larger", "smaller"):
            if query == "larger":
                correct_value = hi
            else:
                correct_value = lo
            correct_key = "A" if first == correct_value else "B"
            if answer_mode == "label":
                candidates = {"A": " A", "B": " B"}
            elif answer_mode == "number":
                candidates = {"A": f" {first}", "B": f" {second}"}
            else:
                raise ValueError(f"unknown answer_mode={answer_mode}")
            tasks.append(
                ComparisonTask(
                    task_id=f"{pair_id}_{order}_{query}",
                    family=family,
                    a_value=first,
                    b_value=second,
                    query=query,
                    order=order,
                    group=group,
                    gap=gap,
                    ratio=ratio,
                    n_shots=n_shots,
                    exemplar_set_id=exemplar_set_id,
                    correct_key=correct_key,
                    prompt=_comparison_prompt(
                        first,
                        second,
                        query,
                        answer_mode,
                        n_shots,
                        exemplar_set_id,
                    ),
                    candidates=candidates,
                )
            )


def _fixed_gap_pairs() -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    gaps = (1, 2, 5, 10)
    for group, (lo, hi) in PCA_GROUPS.items():
        width = hi - lo
        for gap in gaps:
            if gap > width:
                continue
            starts = {
                lo,
                lo + max(0, width // 4),
                lo + max(0, width // 2),
                hi - gap,
            }
            for start in sorted(starts):
                if start + gap <= hi:
                    pairs.append((start, start + gap, group))
    return pairs


def _dense_fixed_gap_pairs(group: int | None = None) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    groups = [group] if group is not None else sorted(DENSE_FIXED_GAP_WINDOWS)
    for group_id in groups:
        lo, hi = DENSE_FIXED_GAP_WINDOWS[group_id]
        for gap in range(1, hi - lo + 1):
            for start in range(lo, hi - gap + 1):
                pairs.append((start, start + gap, group_id))
    return pairs


def _wide_fixed_gap_sg_mg_pairs(
    group: int | None = None,
    *,
    starts_per_gap: int = WIDE_FIXED_GAP_STARTS_PER_GAP,
    seed: int = WIDE_FIXED_GAP_SEED,
) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    groups = [group] if group is not None else sorted(WIDE_FIXED_GAP_WINDOWS)
    for group_id in groups:
        lo, hi = WIDE_FIXED_GAP_WINDOWS[group_id]
        for gap in WIDE_FIXED_GAP_REGIME_GAPS:
            possible_starts = list(range(lo, hi - gap + 1))
            rng = random.Random(seed + group_id * 10_000 + gap)
            if len(possible_starts) <= starts_per_gap:
                starts = possible_starts
            else:
                starts = sorted(rng.sample(possible_starts, starts_per_gap))
            for start in starts:
                pairs.append((start, start + gap, group_id))
    return pairs


def _random_within_group_pairs(seed: int = 1234, per_group: int = 14) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    pairs: list[tuple[int, int, int]] = []
    for group, (lo, hi) in PCA_GROUPS.items():
        seen: set[tuple[int, int]] = set()
        attempts = 0
        while len(seen) < per_group and attempts < per_group * 100:
            attempts += 1
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
            if a == b:
                continue
            pair = tuple(sorted((a, b)))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append((pair[0], pair[1], group))
    return pairs


def _same_ratio_pairs() -> list[tuple[int, int, int]]:
    specs = [
        (1.10, (10, 100, 1000, 9000)),
        (1.25, (8, 80, 800, 8000)),
        (1.50, (6, 60, 600, 6000)),
        (2.00, (5, 50, 500, 5000)),
    ]
    pairs: list[tuple[int, int, int]] = []
    for ratio, starts in specs:
        for start in starts:
            end = int(round(start * ratio))
            if end <= 10000 and end != start:
                pairs.append((start, end, _group_for_pair(start, end)))
    return pairs


def _boundary_pairs() -> list[tuple[int, int, int]]:
    pairs = [
        (9, 10),
        (10, 11),
        (19, 20),
        (98, 99),
        (99, 100),
        (100, 101),
        (109, 110),
        (198, 200),
        (998, 999),
        (999, 1000),
        (1000, 1001),
        (1009, 1010),
        (1998, 2000),
        (9998, 9999),
        (9999, 10000),
        (9990, 10000),
        (9900, 10000),
    ]
    return [(a, b, _group_for_pair(a, b)) for a, b in pairs]


def _roundness_pairs() -> list[tuple[int, int, int]]:
    pairs = [
        (99, 100),
        (100, 101),
        (109, 110),
        (110, 111),
        (999, 1000),
        (1000, 1001),
        (1009, 1010),
        (1010, 1011),
        (4999, 5000),
        (5000, 5001),
        (8999, 9000),
        (9000, 9001),
        (9990, 9991),
        (9999, 10000),
    ]
    return [(a, b, _group_for_pair(a, b)) for a, b in pairs]


def build_comparison_tasks(
    families: Iterable[str] | None = None,
    *,
    answer_mode: str = "label",
    n_shots: int = 0,
    exemplar_set_id: int = 0,
) -> list[ComparisonTask]:
    family_pairs = {
        "fixed_gap": _fixed_gap_pairs(),
        "fixed_gap_dense": _dense_fixed_gap_pairs(),
        "fixed_gap_dense_g1": _dense_fixed_gap_pairs(1),
        "fixed_gap_dense_g2": _dense_fixed_gap_pairs(2),
        "fixed_gap_dense_g3": _dense_fixed_gap_pairs(3),
        "fixed_gap_dense_g4": _dense_fixed_gap_pairs(4),
        "fixed_gap_wide_sg_mg": _wide_fixed_gap_sg_mg_pairs(),
        "fixed_gap_wide_sg_mg_g1": _wide_fixed_gap_sg_mg_pairs(1),
        "fixed_gap_wide_sg_mg_g2": _wide_fixed_gap_sg_mg_pairs(2),
        "fixed_gap_wide_sg_mg_g3": _wide_fixed_gap_sg_mg_pairs(3),
        "fixed_gap_wide_sg_mg_g4": _wide_fixed_gap_sg_mg_pairs(4),
        "random_within_group": _random_within_group_pairs(),
        "same_ratio": _same_ratio_pairs(),
        "boundary": _boundary_pairs(),
        "roundness": _roundness_pairs(),
    }
    if families is None:
        selected = list(family_pairs)
    else:
        selected = list(families)
    bad = sorted(set(selected) - set(family_pairs))
    if bad:
        raise ValueError(f"unknown comparison families: {bad}")

    tasks: list[ComparisonTask] = []
    seen: set[str] = set()
    for family in selected:
        for a, b, group in family_pairs[family]:
            before = len(tasks)
            _expand_pair(
                tasks,
                family=family,
                a=a,
                b=b,
                group=group,
                answer_mode=answer_mode,
                n_shots=n_shots,
                exemplar_set_id=exemplar_set_id,
            )
            for task in tasks[before:]:
                if task.task_id in seen:
                    raise ValueError(f"duplicate task_id={task.task_id}")
                seen.add(task.task_id)
    return tasks


def task_dicts(
    families: Iterable[str] | None = None,
    *,
    answer_mode: str = "label",
    n_shots: int = 0,
    exemplar_set_id: int = 0,
) -> list[dict]:
    return [
        asdict(task)
        for task in build_comparison_tasks(
            families,
            answer_mode=answer_mode,
            n_shots=n_shots,
            exemplar_set_id=exemplar_set_id,
        )
    ]


def _mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    import numpy as np

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 2:
        return float("nan")
    slope, _ = np.polyfit(x[ok], y[ok], 1)
    return float(slope)


def summarize_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    summary = {
        "n_tasks": len(rows),
        "accuracy": _mean(row["correct"] for row in rows),
        "mean_margin": _mean(row["margin"] for row in rows),
        "selected_A_rate": _mean(row["selected_key"] == "A" for row in rows),
        "correct_A_rate": _mean(row["correct_key"] == "A" for row in rows),
    }

    families = sorted({row["family"] for row in rows})
    for family in families:
        sub = [row for row in rows if row["family"] == family]
        summary[f"{family}_accuracy"] = _mean(row["correct"] for row in sub)
        summary[f"{family}_mean_margin"] = _mean(row["margin"] for row in sub)

    fixed = [row for row in rows if str(row["family"]).startswith("fixed_gap")]
    summary["fixed_gap_accuracy"] = _mean(row["correct"] for row in fixed)
    summary["fixed_gap_mean_margin"] = _mean(row["margin"] for row in fixed)
    groups = sorted({int(row["group"]) for row in fixed})
    acc_by_group = []
    margin_by_group = []
    for group in groups:
        sub = [row for row in fixed if int(row["group"]) == group]
        acc = _mean(row["correct"] for row in sub)
        margin = _mean(row["margin"] for row in sub)
        summary[f"fixed_gap_group_{group}_accuracy"] = acc
        summary[f"fixed_gap_group_{group}_mean_margin"] = margin
        acc_by_group.append(acc)
        margin_by_group.append(margin)
    summary["fixed_gap_accuracy_slope_by_group"] = _linear_slope(groups, acc_by_group)
    summary["fixed_gap_margin_slope_by_group"] = _linear_slope(groups, margin_by_group)

    low_close = [
        row for row in fixed if int(row["group"]) == 1 and int(row["gap"]) <= 5
    ]
    high_close = [
        row for row in fixed if int(row["group"]) == 4 and int(row["gap"]) <= 5
    ]
    summary["low_group_close_accuracy"] = _mean(row["correct"] for row in low_close)
    summary["high_group_close_accuracy"] = _mean(row["correct"] for row in high_close)
    summary["high_minus_low_close_accuracy"] = (
        summary["high_group_close_accuracy"] - summary["low_group_close_accuracy"]
    )
    summary["low_group_close_margin"] = _mean(row["margin"] for row in low_close)
    summary["high_group_close_margin"] = _mean(row["margin"] for row in high_close)
    summary["high_minus_low_close_margin"] = (
        summary["high_group_close_margin"] - summary["low_group_close_margin"]
    )
    return summary


def write_results(outdir: Path, payloads: list[dict]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "number_comparison_payloads.json", "w") as fh:
        json.dump(payloads, fh, indent=2)

    rows = [row for payload in payloads for row in payload["rows"]]
    if rows:
        keys = sorted({key for row in rows for key in row})
        with open(outdir / "number_comparison_rows.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    shard_summaries = []
    for payload in payloads:
        summary = dict(payload["summary"])
        summary["model_label"] = payload["model_label"]
        summary["model_name"] = payload["model_name"]
        summary["model_revision"] = payload.get("model_revision") or ""
        summary["family_shard"] = payload["family"]
        shard_summaries.append(summary)
    if shard_summaries:
        keys = sorted({key for row in shard_summaries for key in row})
        with open(outdir / "number_comparison_shard_summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(shard_summaries)

    model_summaries = []
    for model_label in sorted({payload["model_label"] for payload in payloads}):
        model_rows = [row for row in rows if row["model_label"] == model_label]
        if not model_rows:
            continue
        first = next(payload for payload in payloads if payload["model_label"] == model_label)
        summary = summarize_rows(model_rows)
        summary["model_label"] = model_label
        summary["model_name"] = first["model_name"]
        summary["model_revision"] = first.get("model_revision") or ""
        model_summaries.append(summary)
    if model_summaries:
        keys = sorted({key for row in model_summaries for key in row})
        with open(outdir / "number_comparison_model_summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(model_summaries)


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


def compute_correlations(
    outdir: Path,
    beta_csv: Path = Path("results/dataset_frequency_0_to_10000/alpha_beta_without_pile/alpha_beta_models_without_pile.csv"),
) -> dict[str, float]:
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    summary_path = outdir / "number_comparison_model_summary.csv"
    if not summary_path.exists():
        return {}
    beta = load_beta_table(beta_csv)
    with open(summary_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = [
        "accuracy",
        "mean_margin",
        "fixed_gap_accuracy",
        "fixed_gap_mean_margin",
        "fixed_gap_accuracy_slope_by_group",
        "fixed_gap_margin_slope_by_group",
        "high_group_close_accuracy",
        "high_group_close_margin",
        "high_minus_low_close_accuracy",
        "high_minus_low_close_margin",
        "same_ratio_accuracy",
        "same_ratio_mean_margin",
    ]
    xs = np.array([beta.get(row["model_label"], float("nan")) for row in rows], dtype=float)
    out: dict[str, float] = {}
    for col in cols:
        ys = np.array([float(row.get(col, "nan")) for row in rows], dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if int(ok.sum()) >= 2:
            out[f"spearman_beta_{col}"] = float(spearmanr(xs[ok], ys[ok]).statistic)
            out[f"pearson_beta_{col}"] = float(pearsonr(xs[ok], ys[ok]).statistic)
            coef = np.polyfit(xs[ok], ys[ok], 1)
            pred = coef[0] * xs[ok] + coef[1]
            ss_tot = float(np.sum((ys[ok] - ys[ok].mean()) ** 2))
            ss_res = float(np.sum((ys[ok] - pred) ** 2))
            out[f"linear_r2_beta_{col}"] = (
                float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
            )
    with open(outdir / "number_comparison_beta_correlations.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    return out


def plot_results(
    outdir: Path,
    beta_csv: Path = Path("results/dataset_frequency_0_to_10000/alpha_beta_without_pile/alpha_beta_models_without_pile.csv"),
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    summary_path = outdir / "number_comparison_model_summary.csv"
    if not summary_path.exists():
        return
    with open(summary_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    beta = load_beta_table(beta_csv)
    for row in rows:
        row["pca_beta"] = beta.get(row["model_label"], float("nan"))

    panels = [
        ("accuracy", "Overall accuracy"),
        ("mean_margin", "Mean correct margin"),
        ("fixed_gap_margin_slope_by_group", "Close-pair margin slope"),
        ("high_minus_low_close_margin", "High-low close margin"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2))
    for ax, (metric, title) in zip(axes, panels):
        xs = np.array([float(row["pca_beta"]) for row in rows], dtype=float)
        ys = np.array([float(row[metric]) for row in rows], dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.5)
        ax.scatter(xs[ok], ys[ok], s=42, color="#2f6f8f")
        if int(ok.sum()) >= 2:
            coef = np.polyfit(xs[ok], ys[ok], 1)
            xx = np.linspace(float(xs[ok].min()), float(xs[ok].max()), 100)
            ax.plot(xx, coef[0] * xx + coef[1], color="#c7503d", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel(r"$\beta_{\mathrm{PCA}}$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("comparison score")
    fig.tight_layout()
    fig.savefig(outdir / "number_comparison_beta_panels.pdf")
    fig.savefig(outdir / "number_comparison_beta_panels.png", dpi=220)
    plt.close(fig)

    group_cols = [f"fixed_gap_group_{group}_mean_margin" for group in sorted(PCA_GROUPS)]
    labels = [f"$10^{group}$ group" for group in sorted(PCA_GROUPS)]
    model_labels = [row["model_label"] for row in rows]
    matrix = np.array([[float(row[col]) for col in group_cols] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    vmax = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(model_labels)), model_labels, fontsize=8)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_title("Fixed-gap comparison margin by magnitude group")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("correct log-likelihood margin")
    fig.tight_layout()
    fig.savefig(outdir / "number_comparison_group_heatmap.pdf")
    fig.savefig(outdir / "number_comparison_group_heatmap.png", dpi=220)
    plt.close(fig)

    xs = np.array([float(row["pca_beta"]) for row in rows], dtype=float)
    ys = np.array([float(row["fixed_gap_accuracy"]) for row in rows], dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.scatter(xs[ok], ys[ok], s=52, color="#2f6f8f", zorder=3)
    r2 = float("nan")
    if int(ok.sum()) >= 2:
        coef = np.polyfit(xs[ok], ys[ok], 1)
        xx = np.linspace(float(xs[ok].min()), float(xs[ok].max()), 100)
        yy = coef[0] * xx + coef[1]
        pred = coef[0] * xs[ok] + coef[1]
        ss_tot = float(np.sum((ys[ok] - ys[ok].mean()) ** 2))
        ss_res = float(np.sum((ys[ok] - pred) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
        ax.plot(xx, yy, color="#c7503d", linewidth=1.4)
    label_text = {
        "Falcon-RW-1B": "Falcon-1B",
        "Falcon-RW-7B": "Falcon-7B",
        "RedPajama-3B": "RPJ-3B",
        "RedPajama-7B": "RPJ-7B",
        "OLMo-7B-2T": "OLMo-2T",
        "OLMo-7B-Twin-2T": "OLMo-Twin",
        "StarCoderBase-1B": "SCB-1B",
        "StarCoderBase-3B": "SCB-3B",
        "StarCoderBase-7B": "SCB-7B",
    }
    label_offsets = {
        "Falcon-RW-1B": (6, -14),
        "Falcon-RW-7B": (6, 6),
        "RedPajama-3B": (8, -8),
        "RedPajama-7B": (8, 8),
        "OLMo-7B-2T": (8, 7),
        "OLMo-7B-Twin-2T": (8, -3),
        "StarCoderBase-1B": (8, 6),
        "StarCoderBase-3B": (-58, 7),
        "StarCoderBase-7B": (8, 7),
    }
    for row in rows:
        x = float(row["pca_beta"])
        y = float(row["fixed_gap_accuracy"])
        if np.isfinite(x) and np.isfinite(y):
            ax.annotate(
                label_text.get(row["model_label"], row["model_label"]),
                (x, y),
                xytext=label_offsets.get(row["model_label"], (5, 4)),
                textcoords="offset points",
                fontsize=8,
            )
    title = "Fixed-gap digit comparison"
    if np.isfinite(r2):
        title += rf" ($R^2={r2:.2f}$)"
    ax.set_title(title)
    ax.set_xlabel(r"PCA compression factor $\beta$")
    ax.set_ylabel("Fixed-gap comparison accuracy")
    ax.set_ylim(0.45, min(1.02, max(1.0, float(np.nanmax(ys[ok])) + 0.05 if ok.any() else 1.0)))
    if ok.any():
        ax.set_xlim(max(0.0, float(np.nanmin(xs[ok])) - 0.55), float(np.nanmax(xs[ok])) + 0.75)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "number_comparison_fixed_gap_beta_accuracy.pdf", bbox_inches="tight")
    fig.savefig(outdir / "number_comparison_fixed_gap_beta_accuracy.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_latex_summary(outdir: Path) -> None:
    path = outdir / "number_comparison_model_summary.csv"
    rows = []
    if path.exists():
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    beta = load_beta_table(
        Path("results/dataset_frequency_0_to_10000/alpha_beta_without_pile/alpha_beta_models_without_pile.csv")
    )
    rows = sorted(rows, key=lambda row: beta.get(row["model_label"], float("inf")))
    with open(outdir / "number_comparison_summary.tex", "w") as fh:
        fh.write("\\begin{table}[h!]\n")
        fh.write("    \\centering\n")
        fh.write("    \\resizebox{\\linewidth}{!}{%\n")
        fh.write("    \\begin{tabular}{lccccc}\n")
        fh.write("        \\toprule\n")
        fh.write("        Model & $\\beta$ & Acc. & Margin & Close slope & High-low close \\\\\n")
        fh.write("        \\midrule\n")
        for row in rows:
            label = row["model_label"]
            fh.write(
                f"        {label} & {beta.get(label, float('nan')):.2f} & "
                f"{float(row['accuracy']):.2f} & "
                f"{float(row['mean_margin']):+.2f} & "
                f"{float(row['fixed_gap_margin_slope_by_group']):+.2f} & "
                f"{float(row['high_minus_low_close_margin']):+.2f} \\\\\n"
            )
        fh.write("        \\bottomrule\n")
        fh.write("    \\end{tabular}}\n")
        fh.write("    \\caption{Digit-only number-comparison probe. Margin is log-likelihood of the correct A/B answer minus the incorrect answer. Close slope is the fixed-gap margin slope across PCA magnitude groups.}\n")
        fh.write("    \\label{tab:number-comparison-beta}\n")
        fh.write("\\end{table}\n")
