"""Robust-PCA robustness check for the number-line geometry probe."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from src.spacing_fit import fit_spacing_direct

from src.geometry import (
    GeometryConfig,
    _build_prompts,
    _load_model,
    _resolve_device,
    _seed_all,
    _tokenize_prompt,
)
from utils.prompts import extract_target


# Vendored from https://github.com/dganguli/robust-pca (MIT License).
# Copyright (c) 2019 Deep Ganguli. The optimization steps are unchanged.
class RobustPCA:
    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu is not None:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D.flatten(), ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda is not None:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord="fro")

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum(np.abs(M) - tau, 0)

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        n_iter = 0
        err = np.inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol is not None:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        # Principal Component Pursuit by Alternating Directions (ADMM).
        while err > _tol and n_iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + self.mu_inv * Yk, self.lmbda * self.mu_inv)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.frobenius_norm(self.D - Lk - Sk)
            n_iter += 1
            if (n_iter % iter_print) == 0 or n_iter == 1 or n_iter > max_iter or err <= _tol:
                print(f"iteration: {n_iter}, error: {err}")

        self.L = Lk
        self.S = Sk
        self.n_iter = n_iter
        self.error = float(err)
        self.tol = float(_tol)
        return Lk, Sk


def _collect_target_layer(
    cfg: GeometryConfig,
    target_layer: int,
    model,
    tokenizer,
    device: torch.device,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    prompts, alphabetic = _build_prompts(cfg, rng)
    features: list[np.ndarray] = []
    answers: list[float] = []
    group_indices: dict[int, list[int]] = {}

    model.eval()
    with torch.no_grad():
        for group, group_prompts in sorted(prompts.items()):
            start = len(features)
            for prompt in group_prompts:
                inputs = _tokenize_prompt(cfg, tokenizer, prompt).to(device)
                outputs = model(**inputs, output_hidden_states=True, use_cache=False)
                if not 0 <= target_layer < len(outputs.hidden_states):
                    raise ValueError(
                        f"target layer {target_layer} outside [0, {len(outputs.hidden_states) - 1}]"
                    )
                hidden = outputs.hidden_states[target_layer][0, -1, :]
                features.append(hidden.detach().to(torch.float32).cpu().numpy())
                answers.append(extract_target(prompt, alphabetic=alphabetic))
            group_indices[int(group)] = list(range(start, len(features)))

    return np.stack(features), np.asarray(answers), group_indices


def _projection_metrics(
    matrix: np.ndarray,
    answers: np.ndarray,
    group_indices: dict[int, list[int]],
    spacing_fit: str = "direct",
) -> dict[str, float]:
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(matrix)[:, 0]
    rho, _ = spearmanr(answers, pc1)

    groups = sorted(group_indices)
    means = np.asarray([pc1[group_indices[g]].mean() for g in groups])
    gaps = np.abs(np.diff(means))
    if spacing_fit == "direct":
        beta, scale, r2_beta = fit_spacing_direct(gaps)
    elif spacing_fit == "log":
        clipped = np.clip(gaps, 1e-8, None)
        x = np.arange(1, len(clipped) + 1, dtype=float).reshape(-1, 1)
        fit = LinearRegression().fit(x, np.log(clipped))
        beta = float(np.exp(fit.coef_[0]))
        scale = float(np.exp(fit.intercept_))
        predicted = scale * beta ** x[:, 0]
        ss_res = float(np.sum((clipped - predicted) ** 2))
        ss_tot = float(np.sum((clipped - clipped.mean()) ** 2))
        r2_beta = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        raise ValueError(f"unknown spacing fit: {spacing_fit!r}")

    return {
        "rho": float(abs(rho)),
        "beta": beta,
        "explained_variance": float(pca.explained_variance_ratio_[0]),
        "r2_beta": r2_beta,
        "beta_scale": scale,
    }


def _summarize(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    return summary


def run_robust_geometry(
    cfg: GeometryConfig,
    target_layer: int,
    *,
    hf_token: str | None = None,
    rpca_max_iter: int = 1000,
    rpca_tol: float | None = None,
    progress_callback=None,
) -> dict:
    """Run paired ordinary PCA and Principal Component Pursuit at one layer."""
    _seed_all(cfg.seed)
    device = _resolve_device(cfg.device)
    model, tokenizer = _load_model(cfg, device, hf_token)

    ordinary_rows: list[dict[str, float]] = []
    robust_rows: list[dict[str, float]] = []
    decomposition_rows: list[dict[str, float]] = []

    for run_idx in range(max(1, cfg.runs)):
        rng = random.Random(cfg.seed + run_idx)
        X, answers, group_indices = _collect_target_layer(
            cfg, target_layer, model, tokenizer, device, rng
        )
        ordinary_rows.append(_projection_metrics(X, answers, group_indices))

        centered = X.astype(np.float64, copy=False) - X.mean(axis=0, dtype=np.float64)
        rpca = RobustPCA(centered)
        low_rank, sparse = rpca.fit(
            tol=rpca_tol,
            max_iter=rpca_max_iter,
            iter_print=max(rpca_max_iter + 1, 100),
        )
        robust_rows.append(_projection_metrics(low_rank, answers, group_indices))

        total_energy = float(np.linalg.norm(centered, ord="fro") ** 2)
        sparse_energy = float(np.linalg.norm(sparse, ord="fro") ** 2)
        residual = centered - low_rank - sparse
        decomposition_rows.append(
            {
                "iterations": float(rpca.n_iter),
                "converged": float(rpca.error <= rpca.tol),
                "relative_residual": float(
                    np.linalg.norm(residual, ord="fro")
                    / max(np.linalg.norm(centered, ord="fro"), np.finfo(float).eps)
                ),
                "sparse_fraction": float(np.count_nonzero(sparse) / sparse.size),
                "sparse_energy_fraction": sparse_energy / max(total_energy, np.finfo(float).eps),
                "low_rank_rank": float(np.linalg.matrix_rank(low_rank)),
            }
        )
        if progress_callback is not None:
            progress_callback(run_idx + 1, cfg.runs)

    return {
        "method": "dganguli/robust-pca Principal Component Pursuit (ADMM)",
        "source": "https://github.com/dganguli/robust-pca",
        "config": asdict(cfg),
        "target_layer": int(target_layer),
        "rpca_max_iter": int(rpca_max_iter),
        "rpca_tol": rpca_tol,
        "ordinary_pca": _summarize(ordinary_rows),
        "robust_pca": _summarize(robust_rows),
        "decomposition": _summarize(decomposition_rows),
        "runs": [
            {"ordinary_pca": p, "robust_pca": r, "decomposition": d}
            for p, r, d in zip(ordinary_rows, robust_rows, decomposition_rows)
        ],
    }


def _self_check() -> None:
    rng = np.random.default_rng(7)
    clean = rng.normal(size=(18, 2)) @ rng.normal(size=(2, 12))
    observed = clean.copy()
    observed[2, 4] += 20
    observed[11, 8] -= 15
    rpca = RobustPCA(observed)
    low_rank, sparse = rpca.fit(max_iter=1000, iter_print=1001)
    assert np.linalg.norm(observed - low_rank - sparse, ord="fro") < 1e-4
    assert abs(sparse[2, 4]) > 1 and abs(sparse[11, 8]) > 1


if __name__ == "__main__":
    _self_check()
    print("robust_geometry self-check passed")
