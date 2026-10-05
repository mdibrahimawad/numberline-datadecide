"""
Number-line geometry probe for GPT-style LMs.

For each prompt of the form `n_1=n_1,...,n_k=n_k,n=` we collect the last-token
hidden state at every transformer layer, project the per-layer activations to a
low-dimensional subspace via PCA (unsupervised) and PLS (supervised on the
target value), and score the resulting one-dimensional structure with three
metrics that each isolate one aspect of a "number-line" representation:

    explained_variance (EV)   how much of the residual stream is captured by
                              the leading direction (or PLS R^2)
    monotonicity (rho)        |Spearman(target, PC1)| -- is PC1 a rank code
                              for magnitude?
    compression_rate (beta)   geometric ratio between consecutive group means
                              along PC1; beta < 1 means each decade collapses
                              relative to the previous one (log-compression)

Pipeline mirrors Yu, Liu et al. (ICLR'25 "natural log") closely; the analyzer
is rewritten to integrate with the rest of this repo (MLflow, Modal-friendly
return types, no global HF login).
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from src.spacing_fit import fit_spacing_direct

from utils.prompts import (
    DISTRIBUTIONS,
    GROUP_RANGE_DEFAULT,
    default_interval,
    extract_target,
    generate_distribution_prompts,
    generate_numeral_prompts,
    generate_symbol_prompts,
    generate_year_prompts,
)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GeometryConfig:
    model_name: str
    model_revision: str | None = None
    trust_remote_code: bool = False
    data: str = "numerics"                              # "numerics" | "symbols" | "years"
    groups: tuple[int, ...] = GROUP_RANGE_DEFAULT
    k: int = 30
    num_examples: int = 3
    context: str = "random"                             # "random" | "fixed" | "same"
    transform_dim: int = 1
    runs: int = 3
    upper_bound: int | None = None
    seed: int = 42
    device: str | None = None
    dtype: str = "auto"                                 # "auto" | "float16" | "bfloat16" | "float32"
    device_map: str | None = None                       # e.g. "auto" for accelerate sharding
    save_projections: bool = False                      # capture per-layer PC1 from run 0 for plotting
    target_min: int | None = None
    target_max: int | None = None
    tokenizer_use_fast: bool = True
    prepend_bos: bool = False
    spacing_fit: str = "log"                             # "log" (legacy) | "direct" (paper)


# --------------------------------------------------------------------------- #
# device / seeds
# --------------------------------------------------------------------------- #

def _resolve_device(spec: str | None) -> torch.device:
    if spec:
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(spec: str, device: torch.device) -> torch.dtype:
    if spec == "float16":
        return torch.float16
    if spec == "bfloat16":
        return torch.bfloat16
    if spec == "float32":
        return torch.float32
    return torch.float16 if device.type == "cuda" else torch.float32


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# state collection
# --------------------------------------------------------------------------- #

@dataclass
class HiddenStateBundle:
    n_layers: int
    states: dict[int, dict[int, list[np.ndarray]]] = field(default_factory=dict)
    answers: dict[int, dict[int, list[float]]] = field(default_factory=dict)
    tokenization_diagnostics: dict = field(default_factory=dict)


def _tokenize_prompt(cfg: GeometryConfig, tokenizer, prompt: str):
    """Tokenize a prompt, optionally prepending exactly one explicit BOS token."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=not cfg.prepend_bos,
    )
    if cfg.prepend_bos:
        if tokenizer.bos_token_id is None:
            raise ValueError(
                f"prepend_bos=True but {type(tokenizer).__name__} has no BOS token"
            )
        input_ids = inputs["input_ids"]
        bos = torch.full(
            (input_ids.shape[0], 1),
            int(tokenizer.bos_token_id),
            dtype=input_ids.dtype,
        )
        inputs["input_ids"] = torch.cat((bos, input_ids), dim=1)
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat(
                (torch.ones((mask.shape[0], 1), dtype=mask.dtype), mask), dim=1
            )
    inputs.pop("token_type_ids", None)
    return inputs


def _build_prompts(cfg: GeometryConfig, rng: random.Random) -> tuple[dict[int, list[str]], bool]:
    if cfg.data.lower() == "numerics":
        upper = cfg.upper_bound if cfg.upper_bound is not None else 10 ** max(cfg.groups)
        prompts = generate_numeral_prompts(
            k=cfg.k,
            num_examples=cfg.num_examples,
            upper_bound=upper,
            groups=cfg.groups,
            interval_fn=default_interval,
            context=cfg.context,
            rng=rng,
        )
        return prompts, False
    if cfg.data.lower() == "symbols":
        prompts = generate_symbol_prompts(
            k=cfg.k,
            num_examples=cfg.num_examples,
            groups=cfg.groups,
            rng=rng,
        )
        return prompts, True
    if cfg.data.lower() == "years":
        prompts = generate_year_prompts(
            k=cfg.k,
            num_examples=cfg.num_examples,
            year_min=1800 if cfg.target_min is None else cfg.target_min,
            year_max=2025 if cfg.target_max is None else cfg.target_max,
            groups=cfg.groups,
            context=cfg.context,
            rng=rng,
        )
        return prompts, False
    raise ValueError(f"unknown data mode: {cfg.data!r}")


def collect_hidden_states(
    cfg: GeometryConfig,
    model,
    tokenizer,
    device: torch.device,
    rng: random.Random,
) -> HiddenStateBundle:
    prompts, alphabetic = _build_prompts(cfg, rng)

    # outputs.hidden_states is (n_layers + 1) -- index 0 is the embedding output
    # and index n_layers is the final block output. We keep all of them so the
    # per-layer plot is honest about depth.
    n_layers = int(model.config.num_hidden_layers) + 1

    states: dict[int, dict[int, list[np.ndarray]]] = {l: {} for l in range(n_layers)}
    answers: dict[int, dict[int, list[float]]] = {l: {} for l in range(n_layers)}

    standalone_equal_token_ids = tokenizer("=", add_special_tokens=False).input_ids
    prompt_count = 0
    prompts_starting_bos = 0
    prompts_with_duplicate_bos = 0
    prompts_ending_equals = 0
    target_token_lengths: dict[int, int] = {}
    numeral_token_lengths: dict[int, int] = {}
    diagnostic_samples: list[dict] = []

    model.eval()
    with torch.no_grad():
        for group, plist in prompts.items():
            grp_states: dict[int, list[np.ndarray]] = {l: [] for l in range(n_layers)}
            grp_answers: list[float] = []
            for prompt in plist:
                target = extract_target(prompt, alphabetic=alphabetic)
                inputs = _tokenize_prompt(cfg, tokenizer, prompt)
                prompt_ids = inputs["input_ids"][0].tolist()
                decoded_tokens = [
                    tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    for token_id in prompt_ids
                ]
                last_token_text = decoded_tokens[-1]
                prompt_count += 1
                starts_bos = bool(
                    prompt_ids and tokenizer.bos_token_id is not None
                    and prompt_ids[0] == tokenizer.bos_token_id
                )
                prompts_starting_bos += int(starts_bos)
                prompts_with_duplicate_bos += int(
                    starts_bos and len(prompt_ids) > 1
                    and prompt_ids[1] == tokenizer.bos_token_id
                )
                ends_equals = bool(prompt.endswith("=") and last_token_text == "=")
                prompts_ending_equals += int(ends_equals)

                if not alphabetic:
                    target_text = str(int(target))
                    comma_positions = [
                        i for i, token_text in enumerate(decoded_tokens)
                        if token_text == ","
                    ]
                    target_start = comma_positions[-1] + 1 if comma_positions else int(starts_bos)
                    target_end = len(prompt_ids) - 1 if ends_equals else len(prompt_ids)
                    target_ids_in_prompt = prompt_ids[target_start:target_end]
                    target_len = len(target_ids_in_prompt)
                    target_token_lengths[target_len] = target_token_lengths.get(target_len, 0) + 1
                    for numeral in re.findall(r"\d+", prompt):
                        token_len = len(
                            tokenizer(numeral, add_special_tokens=False).input_ids
                        )
                        numeral_token_lengths[token_len] = numeral_token_lengths.get(token_len, 0) + 1

                if len(diagnostic_samples) < 12:
                    diagnostic_samples.append(
                        {
                            "group": int(group),
                            "target": target,
                            "prompt": prompt,
                            "starts_with_bos": starts_bos,
                            "ends_with_equals_token": ends_equals,
                            "last_token_id": int(prompt_ids[-1]),
                            "last_token_text": last_token_text,
                            "target_token_ids": (
                                [int(v) for v in target_ids_in_prompt]
                                if not alphabetic else []
                            ),
                        }
                    )

                inputs = inputs.to(device)
                outputs = model(**inputs, output_hidden_states=True, use_cache=False)
                hs = outputs.hidden_states
                for l in range(n_layers):
                    h = hs[l][0, -1, :].detach().to(torch.float32).cpu().numpy()
                    grp_states[l].append(h)
                grp_answers.append(target)
            for l in range(n_layers):
                states[l][group] = grp_states[l]
                answers[l][group] = list(grp_answers)

    diagnostics = {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "configured_use_fast": bool(cfg.tokenizer_use_fast),
        "prepend_bos": bool(cfg.prepend_bos),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "standalone_equals_token_ids": [int(v) for v in standalone_equal_token_ids],
        "prompt_count": prompt_count,
        "prompts_starting_bos": prompts_starting_bos,
        "prompts_with_duplicate_bos": prompts_with_duplicate_bos,
        "prompts_ending_equals_token": prompts_ending_equals,
        "target_token_length_histogram": {
            str(k): v for k, v in sorted(target_token_lengths.items())
        },
        "standalone_numeral_token_length_histogram": {
            str(k): v for k, v in sorted(numeral_token_lengths.items())
        },
        "samples": diagnostic_samples,
    }
    return HiddenStateBundle(
        n_layers=n_layers,
        states=states,
        answers=answers,
        tokenization_diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# per-layer transform + scoring
# --------------------------------------------------------------------------- #

@dataclass
class LayerTransform:
    method: str
    components: np.ndarray
    explained_variance: float
    first_direction: np.ndarray
    group_indices: dict[int, list[int]]
    answers: list[float]


def _transform_layer(
    bundle: HiddenStateBundle,
    layer: int,
    method: str,
    n_components: int,
) -> LayerTransform | None:
    feats: list[np.ndarray] = []
    answers: list[float] = []
    group_indices: dict[int, list[int]] = {}

    for g in sorted(bundle.states[layer].keys(), key=int):
        reps = bundle.states[layer][g]
        idx = list(range(len(feats), len(feats) + len(reps)))
        group_indices[g] = idx
        for r, a in zip(reps, bundle.answers[layer][g]):
            feats.append(np.asarray(r).reshape(-1))
            answers.append(float(a))

    if not feats:
        return None

    X = np.stack(feats, axis=0)
    y = np.asarray(answers, dtype=float)

    method_u = method.upper()
    try:
        if method_u == "PLS":
            model = PLSRegression(n_components=n_components)
            model.fit(X, y)
            comps = np.asarray(model.transform(X))
            ev = float(model.score(X, y))
            first = np.asarray(model.x_weights_[:, 0]).reshape(-1)
        elif method_u == "PCA":
            model = PCA(n_components=n_components)
            comps = np.asarray(model.fit_transform(X))
            ev = float(np.asarray(model.explained_variance_ratio_).sum())
            first = np.asarray(model.components_[0]).reshape(-1)
        else:
            raise ValueError(f"unknown method: {method!r}")
    except ValueError:
        return None

    return LayerTransform(
        method=method_u,
        components=comps,
        explained_variance=ev,
        first_direction=first,
        group_indices=group_indices,
        answers=answers,
    )


def _monotonicity(answers: Sequence[float], pc1: np.ndarray) -> float:
    rho, _ = spearmanr(np.asarray(answers, dtype=float), np.asarray(pc1, dtype=float))
    return float(rho) if not np.isnan(rho) else float("nan")


def _compression_rate(t: LayerTransform, *, eps: float = 1e-8) -> float:
    """Estimate beta s.t. consecutive group-mean diffs satisfy d_{i+1} ~ beta * d_i."""
    pc1 = t.components[:, 0]
    sorted_groups = sorted(t.group_indices.keys(), key=int)
    means = np.array([pc1[t.group_indices[g]].mean() for g in sorted_groups])

    diffs = np.abs(np.diff(means))
    if len(diffs) < 2:
        return float("nan")
    diffs = np.clip(diffs, eps, None)

    x = np.arange(1, len(diffs) + 1, dtype=float).reshape(-1, 1)
    y = np.log(diffs).reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    return float(np.exp(reg.coef_[0][0]))


def _compression_rate_direct(t: LayerTransform) -> tuple[float, float, float]:
    """Paper-matched least-squares fit of d_i = scale * beta**i."""
    pc1 = t.components[:, 0]
    means = np.array([
        pc1[t.group_indices[g]].mean()
        for g in sorted(t.group_indices, key=int)
    ])
    return fit_spacing_direct(np.abs(np.diff(means)))


@dataclass
class LayerMetrics:
    explained_variance: float
    monotonicity: float
    compression_rate: float
    compression_scale: float = float("nan")
    compression_r2: float = float("nan")


def analyze_layer(t: LayerTransform, spacing_fit: str = "log") -> LayerMetrics:
    pc1 = t.components[:, 0]
    if spacing_fit == "direct":
        beta, scale, r2 = _compression_rate_direct(t)
    elif spacing_fit == "log":
        beta, scale, r2 = _compression_rate(t), float("nan"), float("nan")
    else:
        raise ValueError(f"unknown spacing fit: {spacing_fit!r}")
    return LayerMetrics(
        explained_variance=t.explained_variance,
        monotonicity=_monotonicity(t.answers, pc1),
        compression_rate=beta,
        compression_scale=scale,
        compression_r2=r2,
    )


def transform_and_score(
    bundle: HiddenStateBundle,
    method: str,
    n_components: int,
    spacing_fit: str = "log",
) -> tuple[dict[int, LayerMetrics], dict[int, LayerTransform]]:
    metrics: dict[int, LayerMetrics] = {}
    transforms: dict[int, LayerTransform] = {}
    for l in range(bundle.n_layers):
        t = _transform_layer(bundle, l, method, n_components)
        if t is None:
            continue
        metrics[l] = analyze_layer(t, spacing_fit)
        transforms[l] = t
    return metrics, transforms


@dataclass
class LayerProjections:
    """Per-layer PC1 (or PC1..PCk) projections, kept in group-structured form."""

    method: str
    projections: dict[int, list[list[float]]]    # group -> per-sample [pc1, pc2, ...]
    answers: dict[int, list[float]]              # group -> per-sample target value


def _project_to_dict(
    transforms: dict[int, LayerTransform],
    bundle: HiddenStateBundle,
) -> dict[int, LayerProjections]:
    out: dict[int, LayerProjections] = {}
    for layer, t in transforms.items():
        per_group_proj: dict[int, list[list[float]]] = {}
        per_group_ans: dict[int, list[float]] = {}
        for g, idx in t.group_indices.items():
            per_group_proj[g] = [t.components[i].tolist() for i in idx]
            per_group_ans[g] = list(bundle.answers[layer][g])
        out[layer] = LayerProjections(
            method=t.method,
            projections=per_group_proj,
            answers=per_group_ans,
        )
    return out


# --------------------------------------------------------------------------- #
# multi-run aggregation
# --------------------------------------------------------------------------- #

@dataclass
class LayerSummary:
    ev_mean: float
    ev_std: float
    rho_mean: float
    rho_std: float
    beta_mean: float
    beta_std: float
    beta_scale_mean: float = float("nan")
    beta_scale_std: float = float("nan")
    beta_r2_mean: float = float("nan")
    beta_r2_std: float = float("nan")


def _summarize(per_run: Sequence[dict[int, LayerMetrics]]) -> dict[int, LayerSummary]:
    layers = sorted({l for r in per_run for l in r})
    out: dict[int, LayerSummary] = {}
    for l in layers:
        ev = np.array([r[l].explained_variance for r in per_run if l in r])
        rho = np.array([r[l].monotonicity for r in per_run if l in r])
        beta = np.array([r[l].compression_rate for r in per_run if l in r])
        scale = np.array([r[l].compression_scale for r in per_run if l in r])
        r2 = np.array([r[l].compression_r2 for r in per_run if l in r])

        def finite_mean_std(values: np.ndarray) -> tuple[float, float]:
            values = values[np.isfinite(values)]
            if not len(values):
                return float("nan"), float("nan")
            return float(values.mean()), float(values.std())

        beta_mean, beta_std = finite_mean_std(beta)
        scale_mean, scale_std = finite_mean_std(scale)
        r2_mean, r2_std = finite_mean_std(r2)
        out[l] = LayerSummary(
            ev_mean=float(ev.mean()),
            ev_std=float(ev.std()),
            rho_mean=float(np.abs(rho).mean()),
            rho_std=float(np.abs(rho).std()),
            beta_mean=beta_mean,
            beta_std=beta_std,
            beta_scale_mean=scale_mean,
            beta_scale_std=scale_std,
            beta_r2_mean=r2_mean,
            beta_r2_std=r2_std,
        )
    return out


def _best_layer(summary: dict[int, LayerSummary], key: str = "ev_mean") -> int | None:
    """Pick the layer that maximizes `key`, ignoring NaN/inf entries.

    Layer 0 (the bare embedding) has the same last-token embedding across every
    prompt -- constant input, NaN explained variance -- so without this guard
    `max()` would silently return it.
    """
    candidates = [
        l for l in summary if np.isfinite(getattr(summary[l], key))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda l: getattr(summary[l], key))


def _best_joint_layer(summary: dict[int, LayerSummary]) -> int | None:
    """Select the layer maximizing geometric mean of EV and |rho|."""
    candidates = [
        l for l, value in summary.items()
        if np.isfinite(value.ev_mean) and np.isfinite(value.rho_mean)
        and value.ev_mean >= 0 and value.rho_mean >= 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda l: np.sqrt(summary[l].ev_mean * summary[l].rho_mean))


# --------------------------------------------------------------------------- #
# top-level entry point
# --------------------------------------------------------------------------- #

@dataclass
class GeometryResults:
    config: GeometryConfig
    pca: dict[int, LayerSummary]
    pls: dict[int, LayerSummary]
    n_layers: int
    best_layer_pca: int | None
    best_layer_pls: int | None
    projections_pca: dict[int, LayerProjections] | None = None
    projections_pls: dict[int, LayerProjections] | None = None
    tokenization_diagnostics: dict | None = None

    def to_dict(self) -> dict:
        out: dict = {
            "config": asdict(self.config),
            "n_layers": self.n_layers,
            "best_layer_pca": self.best_layer_pca,
            "best_layer_pls": self.best_layer_pls,
            "best_rho_layer_pca": _best_layer(self.pca, "rho_mean"),
            "best_rho_layer_pls": _best_layer(self.pls, "rho_mean"),
            "best_joint_layer_pca": _best_joint_layer(self.pca),
            "best_joint_layer_pls": _best_joint_layer(self.pls),
            "pca": {str(l): asdict(v) for l, v in self.pca.items()},
            "pls": {str(l): asdict(v) for l, v in self.pls.items()},
        }

        def _proj_dict(p: dict[int, LayerProjections] | None) -> dict | None:
            if p is None:
                return None
            return {
                str(l): {
                    "method": v.method,
                    "projections": {str(g): rows for g, rows in v.projections.items()},
                    "answers": {str(g): a for g, a in v.answers.items()},
                }
                for l, v in p.items()
            }

        out["projections_pca"] = _proj_dict(self.projections_pca)
        out["projections_pls"] = _proj_dict(self.projections_pls)
        out["tokenization_diagnostics"] = self.tokenization_diagnostics
        return out


def _load_model(cfg: GeometryConfig, device: torch.device, hf_token: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_kwargs: dict = {}
    if hf_token:
        hf_kwargs["token"] = hf_token
    if cfg.model_revision:
        hf_kwargs["revision"] = cfg.model_revision
    if cfg.trust_remote_code:
        hf_kwargs["trust_remote_code"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=cfg.tokenizer_use_fast,
        **hf_kwargs,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _resolve_dtype(cfg.dtype, device)
    load_kwargs: dict = {"torch_dtype": dtype, **hf_kwargs}
    if cfg.device_map is not None:
        load_kwargs["device_map"] = cfg.device_map

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if cfg.device_map is None:
        model = model.to(device)

    return model, tokenizer


def run_geometry(
    cfg: GeometryConfig,
    *,
    hf_token: str | None = None,
    progress_callback=None,
) -> GeometryResults:
    _seed_all(cfg.seed)
    device = _resolve_device(cfg.device)

    model, tokenizer = _load_model(cfg, device, hf_token)

    runs_pca: list[dict[int, LayerMetrics]] = []
    runs_pls: list[dict[int, LayerMetrics]] = []
    projections_pca: dict[int, LayerProjections] | None = None
    projections_pls: dict[int, LayerProjections] | None = None
    tokenization_diagnostics: dict | None = None
    n_layers = 0

    for run_idx in range(max(1, cfg.runs)):
        rng = random.Random(cfg.seed + run_idx)
        bundle = collect_hidden_states(cfg, model, tokenizer, device, rng)
        if run_idx == 0:
            tokenization_diagnostics = bundle.tokenization_diagnostics
        n_layers = bundle.n_layers
        pca_metrics, pca_transforms = transform_and_score(
            bundle, "PCA", cfg.transform_dim, cfg.spacing_fit
        )
        pls_metrics, pls_transforms = transform_and_score(
            bundle, "PLS", cfg.transform_dim, cfg.spacing_fit
        )
        runs_pca.append(pca_metrics)
        runs_pls.append(pls_metrics)

        if run_idx == 0 and cfg.save_projections:
            projections_pca = _project_to_dict(pca_transforms, bundle)
            projections_pls = _project_to_dict(pls_transforms, bundle)

        if progress_callback is not None:
            progress_callback(run_idx + 1, cfg.runs)

    pca = _summarize(runs_pca)
    pls = _summarize(runs_pls)

    return GeometryResults(
        config=cfg,
        pca=pca,
        pls=pls,
        n_layers=n_layers,
        best_layer_pca=_best_layer(pca),
        best_layer_pls=_best_layer(pls),
        projections_pca=projections_pca,
        projections_pls=projections_pls,
        tokenization_diagnostics=tokenization_diagnostics,
    )


# --------------------------------------------------------------------------- #
# MLflow surface
# --------------------------------------------------------------------------- #

def metrics_for_mlflow(results: GeometryResults) -> dict[str, float]:
    """Flatten per-layer summaries into a single MLflow-ready metric dict."""
    out: dict[str, float] = {"n_layers": float(results.n_layers)}

    def _add(prefix: str, summary: dict[int, LayerSummary]) -> None:
        for l, v in summary.items():
            out[f"{prefix}_ev_mean_layer_{l:02d}"] = v.ev_mean
            out[f"{prefix}_ev_std_layer_{l:02d}"] = v.ev_std
            out[f"{prefix}_rho_mean_layer_{l:02d}"] = v.rho_mean
            out[f"{prefix}_rho_std_layer_{l:02d}"] = v.rho_std
            out[f"{prefix}_beta_mean_layer_{l:02d}"] = v.beta_mean
            out[f"{prefix}_beta_std_layer_{l:02d}"] = v.beta_std
            if np.isfinite(v.beta_r2_mean):
                out[f"{prefix}_beta_r2_mean_layer_{l:02d}"] = v.beta_r2_mean
                out[f"{prefix}_beta_r2_std_layer_{l:02d}"] = v.beta_r2_std

    _add("pca", results.pca)
    _add("pls", results.pls)

    for tag, summary, best in [
        ("pca", results.pca, results.best_layer_pca),
        ("pls", results.pls, results.best_layer_pls),
    ]:
        if best is None:
            continue
        v = summary[best]
        out[f"{tag}_best_layer"] = float(best)
        out[f"{tag}_best_ev_mean"] = v.ev_mean
        out[f"{tag}_best_rho_mean"] = v.rho_mean
        out[f"{tag}_best_beta_mean"] = v.beta_mean
        if np.isfinite(v.beta_r2_mean):
            out[f"{tag}_best_beta_r2_mean"] = v.beta_r2_mean

        best_rho = _best_layer(summary, "rho_mean")
        if best_rho is not None:
            out[f"{tag}_best_rho_layer"] = float(best_rho)
        best_joint = _best_joint_layer(summary)
        if best_joint is not None:
            out[f"{tag}_best_joint_layer"] = float(best_joint)
            out[f"{tag}_best_joint_score"] = float(
                np.sqrt(summary[best_joint].ev_mean * summary[best_joint].rho_mean)
            )
    return out


# --------------------------------------------------------------------------- #
# regime-stratified probe (Stage 2.5)
#
# Test whether integers from different Stage-1 regime classes (uniform-,
# Gaussian-, Zipf-best-fit) produce qualitatively different number-lines when
# probed independently. For each regime r, sample N values of class r, run
# the model, fit PCA on JUST those samples, then ask: is the resulting PC1
# better described as linear in N (no compression), logarithmic in log10(N)
# (the canonical log-compression signature) or neither?
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RegimeProbeConfig:
    model_name: str
    target_layer: int
    n_per_regime: int = 60
    num_examples: int = 3
    context: str = "random"
    n_min: int = 1
    n_max: int = 1000
    seed: int = 42
    dtype: str = "float16"
    device: str | None = None
    device_map: str | None = None


@dataclass
class RegimePerClassResults:
    targets: list[int]
    pc1_scores: list[float]                 # projected on regime's own PC1
    pc1_shared_scores: list[float]          # projected on union-PCA PC1
    pca_explained_variance: float
    r2_linear: float                        # PC1 ~ a*N + b
    r2_log: float                           # PC1 ~ a*log10(N) + b
    monotonicity: float                     # |Spearman(N, PC1)|
    n_samples: int


@dataclass
class RegimeProbeResults:
    config: RegimeProbeConfig
    per_regime: dict[str, RegimePerClassResults]
    shared_pca_explained_variance: float

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "shared_pca_explained_variance": float(self.shared_pca_explained_variance),
            "per_regime": {
                name: {
                    "targets": v.targets,
                    "pc1_scores": v.pc1_scores,
                    "pc1_shared_scores": v.pc1_shared_scores,
                    "pca_explained_variance": float(v.pca_explained_variance),
                    "r2_linear": float(v.r2_linear),
                    "r2_log": float(v.r2_log),
                    "monotonicity": float(v.monotonicity),
                    "n_samples": int(v.n_samples),
                }
                for name, v in self.per_regime.items()
            },
        }


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    coef = np.polyfit(x, y, 1)
    y_hat = np.polyval(coef, x)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _spearman_abs(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(abs(rho)) if not np.isnan(rho) else float("nan")


def _build_regime_prompt_for_target(
    n: int, *, num_examples: int, lo: int, hi: int, context: str, rng: random.Random,
) -> str:
    fixed_ctx = (4, 54, 432, 9543)
    if context == "random":
        ctx = [rng.randint(lo, hi) for _ in range(num_examples)]
    elif context == "fixed":
        ctx = list(fixed_ctx[:num_examples])
    else:
        raise ValueError(f"unknown context option: {context!r}")
    body = ",".join(f"{x}={x}" for x in ctx)
    return f"{body},{n}=" if body else f"{n}="


def run_regime_probe(
    cfg: RegimeProbeConfig,
    *,
    targets_per_regime: dict[str, list[int]],
    hf_token: str | None = None,
    progress_callback=None,
) -> RegimeProbeResults:
    _seed_all(cfg.seed)
    device = _resolve_device(cfg.device)

    proxy_geom_cfg = GeometryConfig(
        model_name=cfg.model_name,
        seed=cfg.seed,
        dtype=cfg.dtype,
        device=cfg.device,
        device_map=cfg.device_map,
    )
    model, tokenizer = _load_model(proxy_geom_cfg, device, hf_token)
    model.eval()

    if cfg.target_layer >= model.config.num_hidden_layers + 1:
        raise ValueError(
            f"target_layer={cfg.target_layer} but model has "
            f"{model.config.num_hidden_layers + 1} hidden_states."
        )

    rng = random.Random(cfg.seed)

    flat: list[tuple[str, str, int, np.ndarray]] = []   # (regime, prompt, target, resid)
    n_total = sum(len(v) for v in targets_per_regime.values())
    print(f"[regime-probe] forward pass on {n_total} prompts ...")

    with torch.no_grad():
        i = 0
        for regime, targets in targets_per_regime.items():
            for n in targets:
                prompt = _build_regime_prompt_for_target(
                    n,
                    num_examples=cfg.num_examples,
                    lo=cfg.n_min,
                    hi=cfg.n_max,
                    context=cfg.context,
                    rng=rng,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model(**inputs, output_hidden_states=True)
                resid = (
                    outputs.hidden_states[cfg.target_layer][0, -1, :]
                    .detach().to(torch.float32).cpu().numpy()
                )
                flat.append((regime, prompt, int(n), resid))
                i += 1
                if progress_callback is not None and i % 30 == 0:
                    progress_callback(i, n_total)

    by_regime: dict[str, list[tuple[int, np.ndarray]]] = {r: [] for r in targets_per_regime}
    for regime, _prompt, n, r in flat:
        by_regime[regime].append((n, r))

    all_resid = np.stack([r for _, _, _, r in flat], axis=0)
    shared_pca = PCA(n_components=1).fit(all_resid)
    d_shared = shared_pca.components_[0]
    shared_ev = float(shared_pca.explained_variance_ratio_[0])

    per_regime: dict[str, RegimePerClassResults] = {}
    for regime, items in by_regime.items():
        if not items:
            continue
        targets = np.array([n for n, _ in items], dtype=float)
        X = np.stack([r for _, r in items], axis=0)
        try:
            pca = PCA(n_components=1).fit(X)
            d_self = pca.components_[0]
            ev_self = float(pca.explained_variance_ratio_[0])
            pc1_self = X @ d_self
        except ValueError:
            d_self = np.zeros_like(d_shared)
            ev_self = float("nan")
            pc1_self = np.zeros(len(targets))
        pc1_shared = X @ d_shared

        # orient PC1 so that high N → high PC1 (cosmetic, sign-arbitrary)
        if np.corrcoef(pc1_self, targets)[0, 1] < 0:
            pc1_self = -pc1_self
        if np.corrcoef(pc1_shared, targets)[0, 1] < 0:
            pc1_shared = -pc1_shared

        log_targets = np.log10(np.maximum(targets, 1.0))
        per_regime[regime] = RegimePerClassResults(
            targets=[int(n) for n in targets.tolist()],
            pc1_scores=[float(v) for v in pc1_self],
            pc1_shared_scores=[float(v) for v in pc1_shared],
            pca_explained_variance=ev_self,
            r2_linear=_r2(targets, pc1_self),
            r2_log=_r2(log_targets, pc1_self),
            monotonicity=_spearman_abs(targets, pc1_self),
            n_samples=int(len(targets)),
        )

    return RegimeProbeResults(
        config=cfg,
        per_regime=per_regime,
        shared_pca_explained_variance=shared_ev,
    )


def metrics_for_mlflow_regime(results: RegimeProbeResults) -> dict[str, float]:
    out: dict[str, float] = {
        "shared_pca_explained_variance": float(results.shared_pca_explained_variance),
    }
    for name, v in results.per_regime.items():
        prefix = f"regime_{name}"
        out[f"{prefix}_n"] = float(v.n_samples)
        out[f"{prefix}_pca_ev"] = float(v.pca_explained_variance)
        out[f"{prefix}_r2_linear"] = float(v.r2_linear)
        out[f"{prefix}_r2_log"] = float(v.r2_log)
        out[f"{prefix}_rho"] = float(v.monotonicity)
        # which model wins per regime: positive means log fits better than linear
        if not (np.isnan(v.r2_linear) or np.isnan(v.r2_log)):
            out[f"{prefix}_log_minus_linear"] = float(v.r2_log - v.r2_linear)
    return out
