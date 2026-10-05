"""
Stage 3 -- attention-head and MLP attribution to the PC1 number-line direction.

Given a target layer L (e.g. layer 9 for Pythia-2.8b, where Stage 2 found the
peak monotonic encoding), this module:

  1. forward-passes the same Stage 2 prompts through `transformer_lens.
     HookedTransformer`,
  2. fits PCA on the last-token residual stream at layer L to recover the
     same magnitude-aligned direction `d_PC1` Stage 2 reported,
  3. decomposes that residual using the linearity of the residual stream:
        resid_L  =  embedding  +  sum_{l<=L,h} head_{l,h}_out  +  sum_{l<=L} mlp_l_out
     and projects each component onto `d_PC1`,
  4. for every (layer, head) and every (layer, mlp), correlates that component's
     scalar PC1 contribution with `log10(target)` across prompts.

Heads / MLPs whose contribution correlates strongly with `log10(target)` are
the components that carry the magnitude code into PC1; the (layer, head) heatmap
is the standard "which heads are number heads" plot.

The pipeline is GPU-only by construction (TransformerLens loads via fp16 on
CUDA); the local CLI is mainly for sanity checks on small models.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
import torch
from sklearn.decomposition import PCA

from utils.prompts import (
    GROUP_RANGE_DEFAULT,
    default_interval,
    extract_target,
    generate_numeral_prompts,
    generate_symbol_prompts,
)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CircuitConfig:
    model_name: str
    target_layer: int                  # layer whose residual we decompose
    data: str = "numerics"             # "numerics" | "symbols"
    groups: tuple[int, ...] = GROUP_RANGE_DEFAULT
    k: int = 30
    num_examples: int = 3
    context: str = "random"
    upper_bound: int | None = None
    seed: int = 42
    dtype: str = "float16"             # "float16" | "bfloat16" | "float32"
    device: str = "cuda"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _resolve_dtype(spec: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(spec, torch.float16)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_prompts(cfg: CircuitConfig, rng: random.Random) -> list[tuple[str, float]]:
    if cfg.data.lower() == "numerics":
        upper = cfg.upper_bound if cfg.upper_bound is not None else 10 ** max(cfg.groups)
        groups = generate_numeral_prompts(
            k=cfg.k,
            num_examples=cfg.num_examples,
            upper_bound=upper,
            groups=cfg.groups,
            interval_fn=default_interval,
            context=cfg.context,
            rng=rng,
        )
        alphabetic = False
    elif cfg.data.lower() == "symbols":
        groups = generate_symbol_prompts(
            k=cfg.k,
            num_examples=cfg.num_examples,
            groups=cfg.groups,
            rng=rng,
        )
        alphabetic = True
    else:
        raise ValueError(f"unknown data mode: {cfg.data!r}")

    flat: list[tuple[str, float]] = []
    for _, plist in groups.items():
        for p in plist:
            target = extract_target(p, alphabetic=alphabetic)
            flat.append((p, target))
    return flat


# --------------------------------------------------------------------------- #
# attribution
# --------------------------------------------------------------------------- #

@dataclass
class AttributionResults:
    config: CircuitConfig
    n_layers_total: int
    n_heads: int
    d_model: int
    pc1_direction: np.ndarray          # (d_model,)
    pc1_score: np.ndarray              # (n_prompts,)
    embed_proj: np.ndarray             # (n_prompts,)
    head_proj: np.ndarray              # (n_prompts, target_layer+1, n_heads)
    mlp_proj: np.ndarray               # (n_prompts, target_layer+1)
    head_corr_log_target: np.ndarray   # (target_layer+1, n_heads)
    mlp_corr_log_target: np.ndarray    # (target_layer+1,)
    targets: np.ndarray                # (n_prompts,)
    log_targets: np.ndarray            # (n_prompts,)
    pca_explained_variance: float

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "n_layers_total": int(self.n_layers_total),
            "n_heads": int(self.n_heads),
            "d_model": int(self.d_model),
            "pca_explained_variance": float(self.pca_explained_variance),
            "pc1_direction": self.pc1_direction.tolist(),
            "pc1_score": self.pc1_score.tolist(),
            "embed_proj": self.embed_proj.tolist(),
            "head_proj": self.head_proj.tolist(),
            "mlp_proj": self.mlp_proj.tolist(),
            "head_corr_log_target": self.head_corr_log_target.tolist(),
            "mlp_corr_log_target": self.mlp_corr_log_target.tolist(),
            "targets": self.targets.tolist(),
            "log_targets": self.log_targets.tolist(),
        }


def _names_filter_pca(target_layer: int):
    target_name = f"blocks.{target_layer}.hook_resid_post"
    return lambda name: name == target_name


def _names_filter_decompose(target_layer: int):
    keep = {f"blocks.0.hook_resid_pre", f"blocks.{target_layer}.hook_resid_post"}
    for l in range(target_layer + 1):
        keep.add(f"blocks.{l}.attn.hook_z")
        keep.add(f"blocks.{l}.hook_mlp_out")
    return lambda name: name in keep


def _column_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of each column of X with y, vectorised."""
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    num = (Xc * yc[:, None]).sum(axis=0)
    den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return out


def _patch_transformers_for_tl() -> None:
    """Defensive shims for the transformer_lens 2.10 / transformers >= 4.41
    interface drift. Each shim is gated by an ACTUAL absence of the field --
    not just `hasattr` on the class -- so we don't shadow attributes that the
    pinned transformers version already provides."""
    import inspect

    import transformers

    if not hasattr(transformers, "TRANSFORMERS_CACHE"):
        try:
            from transformers.utils import TRANSFORMERS_CACHE
        except ImportError:
            TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
        transformers.TRANSFORMERS_CACHE = TRANSFORMERS_CACHE

    try:
        from transformers.models.gpt_neox.configuration_gpt_neox import (
            GPTNeoXConfig,
        )
    except ImportError:
        return
    init_sig = inspect.signature(GPTNeoXConfig.__init__)
    has_rp_param = "rotary_pct" in init_sig.parameters
    has_rp_class = "rotary_pct" in GPTNeoXConfig.__dict__
    if (not has_rp_param) and (not has_rp_class):
        def _rotary_pct(self):
            return float(
                getattr(
                    self,
                    "partial_rotary_factor",
                    getattr(self, "rotary_percentage", 1.0),
                )
            )
        GPTNeoXConfig.rotary_pct = property(_rotary_pct)


def run_attribution(
    cfg: CircuitConfig,
    *,
    hf_token: str | None = None,
    progress_callback=None,
) -> AttributionResults:
    _patch_transformers_for_tl()

    import einops
    from transformer_lens import HookedTransformer

    _seed_all(cfg.seed)

    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
    dtype = _resolve_dtype(cfg.dtype)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        dtype = torch.float32

    print(f"[circuits] loading {cfg.model_name} on {device} ({dtype}) ...")
    model = HookedTransformer.from_pretrained(
        cfg.model_name,
        device=device,
        torch_dtype=dtype,
    )
    model.eval()

    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    d_model = int(model.cfg.d_model)
    target_layer = cfg.target_layer
    if target_layer >= n_layers:
        raise ValueError(
            f"target_layer={target_layer} but model has n_layers={n_layers}"
        )
    print(f"[circuits] n_layers={n_layers} n_heads={n_heads} d_model={d_model} "
          f"target_layer={target_layer}")

    rng = random.Random(cfg.seed)
    prompts = _build_prompts(cfg, rng)
    targets = np.array([t for _, t in prompts], dtype=float)
    log_targets = np.log10(np.clip(targets, 1.0, None))
    n_prompts = len(prompts)
    print(f"[circuits] {n_prompts} prompts (data={cfg.data}, groups={cfg.groups})")

    # --- pass 1: residual stream at target layer
    pca_filter = _names_filter_pca(target_layer)
    resids = np.empty((n_prompts, d_model), dtype=np.float32)
    with torch.no_grad():
        for i, (prompt, _) in enumerate(prompts):
            tokens = model.to_tokens(prompt)
            _, cache = model.run_with_cache(tokens, names_filter=pca_filter)
            r = cache["resid_post", target_layer][0, -1].detach().to(torch.float32).cpu().numpy()
            resids[i] = r
            if progress_callback is not None and (i + 1) % 30 == 0:
                progress_callback("pca", i + 1, n_prompts)

    pca = PCA(n_components=1).fit(resids)
    d_pc1 = pca.components_[0].astype(np.float32)
    pc1_score = resids @ d_pc1
    pca_ev = float(pca.explained_variance_ratio_[0])
    print(f"[circuits] PCA at layer {target_layer} EV1 = {pca_ev:.3f}")

    d_pc1_t = torch.tensor(d_pc1, device=device, dtype=dtype)

    # --- pass 2: decompose residual into components, project onto d_pc1
    decomp_filter = _names_filter_decompose(target_layer)
    head_proj = np.empty((n_prompts, target_layer + 1, n_heads), dtype=np.float32)
    mlp_proj = np.empty((n_prompts, target_layer + 1), dtype=np.float32)
    embed_proj = np.empty(n_prompts, dtype=np.float32)

    with torch.no_grad():
        for i, (prompt, _) in enumerate(prompts):
            tokens = model.to_tokens(prompt)
            _, cache = model.run_with_cache(tokens, names_filter=decomp_filter)

            embed_vec = cache["resid_pre", 0][0, -1]
            embed_proj[i] = float((embed_vec @ d_pc1_t).item())

            for l in range(target_layer + 1):
                z = cache["z", l][0, -1]                         # (n_heads, d_head)
                W_O = model.W_O[l]                               # (n_heads, d_head, d_model)
                head_out_h = einops.einsum(
                    z, W_O,
                    "h d_head, h d_head d_model -> h d_model",
                )
                head_proj[i, l] = (head_out_h @ d_pc1_t).detach().to(torch.float32).cpu().numpy()

                mlp_vec = cache["mlp_out", l][0, -1]
                mlp_proj[i, l] = float((mlp_vec @ d_pc1_t).item())

            if progress_callback is not None and (i + 1) % 30 == 0:
                progress_callback("decompose", i + 1, n_prompts)

    head_corr = np.empty((target_layer + 1, n_heads), dtype=np.float32)
    for l in range(target_layer + 1):
        head_corr[l] = _column_corr(head_proj[:, l, :], log_targets)
    mlp_corr = _column_corr(mlp_proj, log_targets)

    return AttributionResults(
        config=cfg,
        n_layers_total=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        pc1_direction=d_pc1,
        pc1_score=pc1_score.astype(np.float32),
        embed_proj=embed_proj,
        head_proj=head_proj,
        mlp_proj=mlp_proj,
        head_corr_log_target=head_corr,
        mlp_corr_log_target=mlp_corr,
        targets=targets,
        log_targets=log_targets,
        pca_explained_variance=pca_ev,
    )


# --------------------------------------------------------------------------- #
# MLflow surface
# --------------------------------------------------------------------------- #

def metrics_for_mlflow(results: AttributionResults) -> dict[str, float]:
    out: dict[str, float] = {
        "n_layers": float(results.n_layers_total),
        "n_heads": float(results.n_heads),
        "d_model": float(results.d_model),
        "target_layer": float(results.config.target_layer),
        "pca_explained_variance": float(results.pca_explained_variance),
    }

    head_corr = results.head_corr_log_target
    mlp_corr = results.mlp_corr_log_target
    abs_head = np.abs(head_corr)

    flat_idx = int(np.argmax(abs_head))
    bl, bh = np.unravel_index(flat_idx, abs_head.shape)
    out["best_head_layer"] = float(bl)
    out["best_head_index"] = float(bh)
    out["best_head_corr"] = float(head_corr[bl, bh])

    out["best_mlp_layer"] = float(int(np.argmax(np.abs(mlp_corr))))
    out["best_mlp_corr"] = float(mlp_corr[int(np.argmax(np.abs(mlp_corr)))])

    # top-K head correlations for quick triage in MLflow
    flat = head_corr.flatten()
    order = np.argsort(-np.abs(flat))[:10]
    for rank, idx in enumerate(order):
        l, h = np.unravel_index(int(idx), head_corr.shape)
        out[f"top{rank + 1:02d}_head_layer"] = float(l)
        out[f"top{rank + 1:02d}_head_index"] = float(h)
        out[f"top{rank + 1:02d}_head_corr"] = float(head_corr[l, h])

    for l, c in enumerate(mlp_corr):
        out[f"mlp_corr_layer_{l:02d}"] = float(c)
    return out


def write_results_json(results: AttributionResults, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(results.to_dict(), fh)


# --------------------------------------------------------------------------- #
# activation patching (causal intervention)
# --------------------------------------------------------------------------- #
#
# For each (donor, receiver) pair where the donor has a large target and the
# receiver has a small target:
#
#   1. Run both prompts cleanly, save donor's per-component output at the
#      last token, and the layer-L PC1 score of each.
#   2. For each component C in {head_{l,h}, mlp_l : l <= target_layer}:
#       - re-run the receiver with the hook that replaces C's last-token
#         output by the donor's,
#       - measure layer-L PC1 score of the patched run.
#   3. Normalised patching effect:  (patched - clean_recv) / (clean_donor - clean_recv).
#      Effect = 1.0 means C alone carries the full magnitude code; 0.0 means
#      C is causally irrelevant.
#
# Heads / MLPs with high mean effect across pairs are the components the
# magnitude code causally flows through.


@dataclass(frozen=True)
class PatchingConfig:
    model_name: str
    target_layer: int
    data: str = "numerics"
    groups: tuple[int, ...] = GROUP_RANGE_DEFAULT
    k: int = 30
    num_examples: int = 3
    context: str = "random"
    upper_bound: int | None = None
    seed: int = 42
    dtype: str = "float16"
    device: str = "cuda"
    n_pairs: int = 16


@dataclass
class PatchingResults:
    config: PatchingConfig
    n_layers_total: int
    n_heads: int
    target_layer: int
    pca_explained_variance: float
    donor_pc1: np.ndarray            # (n_pairs,)
    receiver_pc1: np.ndarray         # (n_pairs,)
    donor_targets: np.ndarray
    receiver_targets: np.ndarray
    head_effect_mean: np.ndarray     # (target_layer+1, n_heads)
    head_effect_std: np.ndarray
    mlp_effect_mean: np.ndarray      # (target_layer+1,)
    mlp_effect_std: np.ndarray
    pair_donor_groups: np.ndarray    # (n_pairs,)
    pair_receiver_groups: np.ndarray

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "n_layers_total": int(self.n_layers_total),
            "n_heads": int(self.n_heads),
            "target_layer": int(self.target_layer),
            "pca_explained_variance": float(self.pca_explained_variance),
            "donor_pc1": self.donor_pc1.tolist(),
            "receiver_pc1": self.receiver_pc1.tolist(),
            "donor_targets": self.donor_targets.tolist(),
            "receiver_targets": self.receiver_targets.tolist(),
            "head_effect_mean": self.head_effect_mean.tolist(),
            "head_effect_std": self.head_effect_std.tolist(),
            "mlp_effect_mean": self.mlp_effect_mean.tolist(),
            "mlp_effect_std": self.mlp_effect_std.tolist(),
            "pair_donor_groups": self.pair_donor_groups.tolist(),
            "pair_receiver_groups": self.pair_receiver_groups.tolist(),
        }


def _build_pairs(
    flat: list[tuple[str, float, int]],
    sorted_groups: list[int],
    n_pairs: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Pair group i with group (m + 1 - i) so donor-vs-receiver target gap is maximal."""
    if len(sorted_groups) < 2:
        raise ValueError("need at least two groups to build patching pairs")
    pair_group_specs = list(
        zip(sorted_groups, list(reversed(sorted_groups)))
    )[: len(sorted_groups) // 2]

    by_group: dict[int, list[int]] = {g: [] for g in sorted_groups}
    for i, (_, _, g) in enumerate(flat):
        by_group[g].append(i)
    for g in sorted_groups:
        rng.shuffle(by_group[g])

    n_per_pg = max(1, n_pairs // len(pair_group_specs))
    pairs: list[tuple[int, int]] = []
    for g_lo, g_hi in pair_group_specs:
        lo_take = by_group[g_lo][:n_per_pg]
        hi_take = by_group[g_hi][:n_per_pg]
        for i_lo, i_hi in zip(lo_take, hi_take):
            pairs.append((i_hi, i_lo))   # (donor=large, receiver=small)
    return pairs[:n_pairs]


def run_patching(
    cfg: PatchingConfig,
    *,
    hf_token: str | None = None,
    progress_callback=None,
) -> PatchingResults:
    _patch_transformers_for_tl()

    from functools import partial

    from transformer_lens import HookedTransformer

    _seed_all(cfg.seed)
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
    dtype = _resolve_dtype(cfg.dtype)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        dtype = torch.float32

    print(f"[patch] loading {cfg.model_name} on {device} ({dtype}) ...")
    model = HookedTransformer.from_pretrained(
        cfg.model_name, device=device, torch_dtype=dtype,
    )
    model.eval()

    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    d_model = int(model.cfg.d_model)
    d_head = int(model.cfg.d_head)
    target_layer = cfg.target_layer
    if target_layer >= n_layers:
        raise ValueError(f"target_layer={target_layer} but model has n_layers={n_layers}")
    print(f"[patch] n_layers={n_layers} n_heads={n_heads} d_model={d_model} "
          f"target_layer={target_layer}")

    # build prompts ---------------------------------------------------------
    rng = random.Random(cfg.seed)
    if cfg.data.lower() == "numerics":
        upper = cfg.upper_bound if cfg.upper_bound is not None else 10 ** max(cfg.groups)
        prompts_by_g = generate_numeral_prompts(
            k=cfg.k, num_examples=cfg.num_examples, upper_bound=upper,
            groups=cfg.groups, interval_fn=default_interval, context=cfg.context, rng=rng,
        )
        alphabetic = False
    elif cfg.data.lower() == "symbols":
        prompts_by_g = generate_symbol_prompts(
            k=cfg.k, num_examples=cfg.num_examples, groups=cfg.groups, rng=rng,
        )
        alphabetic = True
    else:
        raise ValueError(f"unknown data mode: {cfg.data!r}")

    flat: list[tuple[str, float, int]] = []
    for g, plist in prompts_by_g.items():
        for p in plist:
            flat.append((p, extract_target(p, alphabetic=alphabetic), int(g)))
    n_prompts = len(flat)

    # PCA pass --------------------------------------------------------------
    resids = np.empty((n_prompts, d_model), dtype=np.float32)
    pca_filter = _names_filter_pca(target_layer)
    with torch.no_grad():
        for i, (p, _, _) in enumerate(flat):
            tokens = model.to_tokens(p)
            _, cache = model.run_with_cache(tokens, names_filter=pca_filter)
            resids[i] = (
                cache["resid_post", target_layer][0, -1].detach().to(torch.float32).cpu().numpy()
            )
    pca = PCA(n_components=1).fit(resids)
    d_pc1 = pca.components_[0].astype(np.float32)
    pc1_all = resids @ d_pc1
    pca_ev = float(pca.explained_variance_ratio_[0])
    print(f"[patch] PCA at layer {target_layer} EV1 = {pca_ev:.3f}")

    # build pairs -----------------------------------------------------------
    sorted_groups = sorted(cfg.groups)
    pairs = _build_pairs(flat, sorted_groups, cfg.n_pairs, rng)
    n_pairs = len(pairs)
    if n_pairs == 0:
        raise RuntimeError("could not build any patching pairs from these groups")
    print(f"[patch] n_pairs={n_pairs}; sample donor->recv  "
          f"{[(int(flat[d][2]), int(flat[r][2])) for d, r in pairs[:5]]}")

    # cache donor activations at last position ------------------------------
    donor_z = {
        l: torch.zeros(n_pairs, n_heads, d_head, device=device, dtype=dtype)
        for l in range(target_layer + 1)
    }
    donor_mlp = {
        l: torch.zeros(n_pairs, d_model, device=device, dtype=dtype)
        for l in range(target_layer + 1)
    }
    decomp_filter = _names_filter_decompose(target_layer)
    with torch.no_grad():
        for j, (donor_i, _) in enumerate(pairs):
            tokens = model.to_tokens(flat[donor_i][0])
            _, cache = model.run_with_cache(tokens, names_filter=decomp_filter)
            for l in range(target_layer + 1):
                donor_z[l][j] = cache["z", l][0, -1].to(dtype)
                donor_mlp[l][j] = cache["mlp_out", l][0, -1].to(dtype)

    # patching loop ---------------------------------------------------------
    d_pc1_t = torch.tensor(d_pc1, device=device, dtype=torch.float32)
    target_resid_name = f"blocks.{target_layer}.hook_resid_post"

    donor_pc1 = np.array([pc1_all[d] for d, _ in pairs], dtype=np.float32)
    recv_pc1 = np.array([pc1_all[r] for _, r in pairs], dtype=np.float32)
    donor_groups = np.array([flat[d][2] for d, _ in pairs], dtype=int)
    recv_groups = np.array([flat[r][2] for _, r in pairs], dtype=int)
    donor_targets = np.array([flat[d][1] for d, _ in pairs], dtype=float)
    recv_targets = np.array([flat[r][1] for _, r in pairs], dtype=float)
    denom = donor_pc1 - recv_pc1
    safe_denom = np.where(np.abs(denom) > 1e-6, denom, np.nan)

    def _make_z_hook(L: int, H: int, j: int):
        donor_vec = donor_z[L][j, H, :]
        def _hook(activation, hook):  # noqa: ARG001 (hook arg is TL convention)
            activation[0, -1, H, :] = donor_vec
            return activation
        return _hook

    def _make_mlp_hook(L: int, j: int):
        donor_vec = donor_mlp[L][j, :]
        def _hook(activation, hook):  # noqa: ARG001
            activation[0, -1, :] = donor_vec
            return activation
        return _hook

    head_eff = np.zeros((target_layer + 1, n_heads), dtype=np.float32)
    head_eff_std = np.zeros((target_layer + 1, n_heads), dtype=np.float32)
    mlp_eff = np.zeros(target_layer + 1, dtype=np.float32)
    mlp_eff_std = np.zeros(target_layer + 1, dtype=np.float32)

    receiver_tokens_list = [model.to_tokens(flat[r][0]) for _, r in pairs]

    total_components = (target_layer + 1) * (n_heads + 1)
    done = 0

    with torch.no_grad():
        for L in range(target_layer + 1):
            # --- head-level patches at layer L
            for H in range(n_heads):
                patched_pc1 = np.empty(n_pairs, dtype=np.float32)
                for j in range(n_pairs):
                    hook_fn = _make_z_hook(L, H, j)
                    with model.hooks(fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook_fn)]):
                        _, cache = model.run_with_cache(
                            receiver_tokens_list[j],
                            names_filter=lambda n, t=target_resid_name: n == t,
                        )
                    resid = cache[target_resid_name][0, -1].detach().to(torch.float32).cpu().numpy()
                    patched_pc1[j] = float(np.dot(resid, d_pc1))
                eff = (patched_pc1 - recv_pc1) / safe_denom
                eff_clean = eff[np.isfinite(eff)]
                head_eff[L, H] = float(np.nanmean(eff_clean)) if eff_clean.size else 0.0
                head_eff_std[L, H] = float(np.nanstd(eff_clean)) if eff_clean.size else 0.0
                done += 1
            if progress_callback is not None:
                progress_callback("head_patch", done, total_components)

            # --- MLP patch at layer L
            patched_pc1 = np.empty(n_pairs, dtype=np.float32)
            for j in range(n_pairs):
                hook_fn = _make_mlp_hook(L, j)
                with model.hooks(fwd_hooks=[(f"blocks.{L}.hook_mlp_out", hook_fn)]):
                    _, cache = model.run_with_cache(
                        receiver_tokens_list[j],
                        names_filter=lambda n, t=target_resid_name: n == t,
                    )
                resid = cache[target_resid_name][0, -1].detach().to(torch.float32).cpu().numpy()
                patched_pc1[j] = float(np.dot(resid, d_pc1))
            eff = (patched_pc1 - recv_pc1) / safe_denom
            eff_clean = eff[np.isfinite(eff)]
            mlp_eff[L] = float(np.nanmean(eff_clean)) if eff_clean.size else 0.0
            mlp_eff_std[L] = float(np.nanstd(eff_clean)) if eff_clean.size else 0.0
            done += 1
            if progress_callback is not None:
                progress_callback("mlp_patch", done, total_components)

    return PatchingResults(
        config=cfg,
        n_layers_total=n_layers,
        n_heads=n_heads,
        target_layer=target_layer,
        pca_explained_variance=pca_ev,
        donor_pc1=donor_pc1,
        receiver_pc1=recv_pc1,
        donor_targets=donor_targets,
        receiver_targets=recv_targets,
        head_effect_mean=head_eff,
        head_effect_std=head_eff_std,
        mlp_effect_mean=mlp_eff,
        mlp_effect_std=mlp_eff_std,
        pair_donor_groups=donor_groups,
        pair_receiver_groups=recv_groups,
    )


def metrics_for_mlflow_patching(results: PatchingResults) -> dict[str, float]:
    out: dict[str, float] = {
        "n_layers": float(results.n_layers_total),
        "n_heads": float(results.n_heads),
        "target_layer": float(results.target_layer),
        "pca_explained_variance": float(results.pca_explained_variance),
        "n_pairs": float(len(results.donor_pc1)),
    }

    head_e = results.head_effect_mean
    abs_h = np.abs(head_e)
    flat_idx = int(np.argmax(abs_h))
    bl, bh = np.unravel_index(flat_idx, abs_h.shape)
    out["best_head_layer"] = float(bl)
    out["best_head_index"] = float(bh)
    out["best_head_effect"] = float(head_e[bl, bh])

    mlp_e = results.mlp_effect_mean
    bml = int(np.argmax(np.abs(mlp_e)))
    out["best_mlp_layer"] = float(bml)
    out["best_mlp_effect"] = float(mlp_e[bml])

    flat = head_e.flatten()
    order = np.argsort(-np.abs(flat))[:10]
    for rank, idx in enumerate(order):
        l, h = np.unravel_index(int(idx), head_e.shape)
        out[f"top{rank + 1:02d}_head_layer"] = float(l)
        out[f"top{rank + 1:02d}_head_index"] = float(h)
        out[f"top{rank + 1:02d}_head_effect"] = float(head_e[l, h])

    for l, e in enumerate(mlp_e):
        out[f"mlp_effect_layer_{l:02d}"] = float(e)
    return out
