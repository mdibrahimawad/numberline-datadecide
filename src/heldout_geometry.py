"""Held-out layer selection for the numerical PCA probe."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from src.geometry import (
    GeometryConfig,
    HiddenStateBundle,
    _load_model,
    _resolve_device,
    _seed_all,
    collect_hidden_states,
)
from src.spacing_fit import fit_spacing_direct


def _split_by_target(
    answers: dict[int, list[float]], seed: int
) -> dict[str, dict[int, list[int]]]:
    """Split each magnitude group by target value, preventing target leakage."""
    rng = random.Random(seed)
    split = {name: {} for name in ("train", "validation", "test")}
    for group, values in sorted(answers.items()):
        targets = sorted(set(values))
        if len(targets) < 3:
            raise ValueError(f"group {group} has fewer than three distinct targets")
        rng.shuffle(targets)
        n_train = max(1, len(targets) // 2)
        n_validation = max(1, (len(targets) - n_train) // 2)
        target_sets = {
            "train": set(targets[:n_train]),
            "validation": set(targets[n_train:n_train + n_validation]),
            "test": set(targets[n_train + n_validation:]),
        }
        for name, selected in target_sets.items():
            split[name][group] = [i for i, value in enumerate(values) if value in selected]
            if not split[name][group]:
                raise ValueError(f"empty {name} split for group {group}")
    return split


def _matrix(
    bundle: HiddenStateBundle,
    layer: int,
    indices: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    rows, targets, grouped = [], [], {}
    for group in sorted(indices):
        start = len(rows)
        for index in indices[group]:
            rows.append(np.asarray(bundle.states[layer][group][index]).reshape(-1))
            targets.append(bundle.answers[layer][group][index])
        grouped[group] = list(range(start, len(rows)))
    return np.stack(rows), np.asarray(targets, dtype=float), grouped


def evaluate_split(bundle: HiddenStateBundle, split_seed: int) -> dict:
    split = _split_by_target(bundle.answers[0], split_seed)
    candidates: dict[int, dict[str, float]] = {}
    fitted: dict[int, PCA] = {}

    for layer in range(bundle.n_layers):
        train_x, _, _ = _matrix(bundle, layer, split["train"])
        validation_x, validation_y, _ = _matrix(bundle, layer, split["validation"])
        pca = PCA(n_components=1).fit(train_x)
        ev = float(pca.explained_variance_ratio_[0])
        validation_pc1 = pca.transform(validation_x)[:, 0]
        rho, _ = spearmanr(validation_y, validation_pc1)
        rho = float(abs(rho))
        if not np.isfinite(ev) or not np.isfinite(rho):
            continue
        candidates[layer] = {
            "train_ev": ev,
            "validation_rho": rho,
            "joint_score": float(np.sqrt(ev * rho)),
        }
        fitted[layer] = pca

    if not candidates:
        raise RuntimeError("no finite layer-selection candidates")
    selected = max(candidates, key=lambda layer: candidates[layer]["joint_score"])
    test_x, test_y, test_groups = _matrix(bundle, selected, split["test"])
    test_pc1 = fitted[selected].transform(test_x)[:, 0]
    test_rho, _ = spearmanr(test_y, test_pc1)
    means = np.asarray([test_pc1[test_groups[g]].mean() for g in sorted(test_groups)])
    beta, scale, _ = fit_spacing_direct(np.abs(np.diff(means)), compute_r2=False)

    return {
        "selected_layer": int(selected),
        **candidates[selected],
        "test_rho": float(abs(test_rho)),
        "test_beta": beta,
        "test_beta_scale": scale,
        "split_counts": {
            name: {str(group): len(rows) for group, rows in groups.items()}
            for name, groups in split.items()
        },
        "split_targets": {
            name: {
                str(group): sorted({bundle.answers[0][group][i] for i in rows})
                for group, rows in groups.items()
            }
            for name, groups in split.items()
        },
        "validation_layers": {str(layer): values for layer, values in candidates.items()},
    }


def run_heldout_geometry(
    cfg: GeometryConfig,
    *,
    hf_token: str | None = None,
    progress_callback=None,
) -> dict:
    _seed_all(cfg.seed)
    device = _resolve_device(cfg.device)
    model, tokenizer = _load_model(cfg, device, hf_token)
    runs = []
    for run_index in range(max(1, cfg.runs)):
        run_seed = cfg.seed + run_index
        bundle = collect_hidden_states(
            cfg, model, tokenizer, device, random.Random(run_seed)
        )
        result = evaluate_split(bundle, split_seed=10_000 + run_seed)
        result["run"] = run_index
        result["prompt_seed"] = run_seed
        runs.append(result)
        if progress_callback:
            progress_callback(run_index + 1, cfg.runs)

    selected = Counter(run["selected_layer"] for run in runs)
    summary = {"selected_layer_counts": {str(k): v for k, v in sorted(selected.items())}}
    for key in ("train_ev", "validation_rho", "joint_score", "test_rho", "test_beta"):
        values = np.asarray([run[key] for run in runs], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return {
        "method": "held-out geometric-mean PCA layer selection",
        "selection": "argmax_layer sqrt(train_EV * abs(validation_rho))",
        "spacing_fit": "direct raw-gap least squares",
        "r2_computed": False,
        "config": asdict(cfg),
        "runs": runs,
        "summary": summary,
    }


if __name__ == "__main__":
    answers = {g: [g * 100 + i for i in range(12)] for g in range(1, 5)}
    split = _split_by_target(answers, 42)
    for group in answers:
        sets = [{answers[group][i] for i in split[name][group]} for name in split]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    states = {
        0: {
            group: [np.array([value, value * 1e-4]) for value in values]
            for group, values in answers.items()
        }
    }
    result = evaluate_split(
        HiddenStateBundle(
            n_layers=1,
            states=states,
            answers={0: answers},
        ),
        split_seed=42,
    )
    assert result["selected_layer"] == 0 and result["test_rho"] > 0.99
    assert "r2" not in result
