"""DataDecide sweep: joint-criterion layer selection + frozen-layer evaluation.

Per model:
  1. selection seeds (default 42, 43, 44): collect filtered hidden states at
     every layer, score PCA EV / |rho| / beta (direct and log fits) per layer,
     and pick the layer maximizing the median over seeds of sqrt(EV * |rho|)
     (the `_best_joint_layer` criterion, made robust as in
     `per_seed_geometry.robust_joint_selection`);
  2. evaluation seeds (default 45, 46, 47): re-score only that frozen layer on
     fresh prompts via `src.frozen_layer_evaluation`.

Heavy imports (torch, sklearn, src.geometry) are deferred so the Modal local
entrypoint can import the config / verification / CSV helpers without torch.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "datadecide_models.json"

SUMMARY_COLUMNS = (
    "recipe",
    "selected_layer",
    "rho_mean",
    "rho_std",
    "ev_mean",
    "ev_std",
    "beta_direct_mean",
    "beta_direct_std",
    "r2_direct",
    "beta_log_mean",
    "beta_log_std",
    "rejection_rate_per_group",
    "failed_groups",
)


# --------------------------------------------------------------------------- #
# config + Hugging Face verification
# --------------------------------------------------------------------------- #

def load_model_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def repo_id_for(recipe: str, config: dict) -> str:
    if "/" in recipe:  # already a full repo id
        return recipe
    return config["repo_id_template"].format(recipe=recipe)


def verify_revisions(
    repo_ids: list[str], revision: str | None, token: str | None = None
) -> dict[str, str]:
    """Return {repo_id: problem} for repos that are missing or lack `revision`."""
    from huggingface_hub import list_repo_refs

    problems: dict[str, str] = {}
    for repo_id in repo_ids:
        try:
            refs = list_repo_refs(repo_id, token=token)
        except Exception as exc:  # RepositoryNotFoundError, network errors, ...
            problems[repo_id] = f"cannot list refs: {type(exc).__name__}: {exc}"
            continue
        if revision:
            names = {ref.name for ref in refs.branches} | {ref.name for ref in refs.tags}
            if revision not in names:
                problems[repo_id] = f"revision {revision!r} not found among {len(names)} refs"
    return problems


# --------------------------------------------------------------------------- #
# layer selection
# --------------------------------------------------------------------------- #

def score_bundle_layers(bundle) -> tuple[dict, dict]:
    """Per-layer PCA metrics for one seed, with both direct and log beta fits."""
    from src.geometry import analyze_layer, transform_and_score

    direct, transforms = transform_and_score(bundle, "PCA", 1, "direct")
    log = {layer: analyze_layer(t, "log") for layer, t in transforms.items()}
    return direct, log


def select_joint_layer(per_seed_direct: list[dict]) -> dict:
    """argmax_layer median_seed sqrt(EV * |rho|), finite layers only."""
    import numpy as np

    layers = set.intersection(*(set(run) for run in per_seed_direct))
    scores: dict[int, dict] = {}
    for layer in sorted(layers):
        q = [
            float(np.sqrt(run[layer].explained_variance * abs(run[layer].monotonicity)))
            if run[layer].explained_variance >= 0 else float("nan")
            for run in per_seed_direct
        ]
        if np.all(np.isfinite(q)):
            scores[layer] = {"q_per_seed": q, "q_median": float(np.median(q))}
    if not scores:
        raise RuntimeError("no layer has finite EV and rho on every selection seed")
    best = max(scores, key=lambda layer: scores[layer]["q_median"])
    return {
        "selected_layer": int(best),
        "selected_q_median": scores[best]["q_median"],
        "layer_scores": {str(layer): v for layer, v in scores.items()},
    }


def _filter_summary(seed_stats: list[dict]) -> dict:
    """Aggregate per-group filter stats over seeds ({"seed", "groups"} entries)."""
    out: dict[str, dict] = {}
    for entry in seed_stats:
        for group, st in (entry.get("groups") or {}).items():
            agg = out.setdefault(
                group, {"candidates_tried": 0, "accepted": 0, "failed_slots": 0, "failed_seeds": []}
            )
            agg["candidates_tried"] += st["candidates_tried"]
            agg["accepted"] += st["accepted"]
            agg["failed_slots"] += st["failed_slots"]
            if st["failed"]:
                agg["failed_seeds"].append(entry["seed"])
    for agg in out.values():
        tried = agg["candidates_tried"]
        agg["rejection_rate"] = (tried - agg["accepted"]) / tried if tried else float("nan")
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def run_datadecide_model(
    model_name: str,
    revision: str | None = None,
    *,
    groups: tuple[int, ...] = (1, 2, 3, 4),
    k: int = 40,
    num_examples: int = 3,
    selection_seeds: tuple[int, ...] = (42, 43, 44),
    eval_seeds: tuple[int, ...] = (45, 46, 47),
    filter_max_new_tokens: int = 8,
    filter_max_candidates: int = 100,
    device: str | None = None,
    dtype: str = "bfloat16",
    hf_token: str | None = None,
) -> dict:
    import gc
    import random

    import torch

    from src.frozen_layer_evaluation import run_frozen_layer_evaluation
    from src.geometry import (
        GeometryConfig,
        _best_joint_layer,
        _load_model,
        _resolve_device,
        _seed_all,
        _summarize,
        collect_hidden_states,
    )

    if list(eval_seeds) != list(range(eval_seeds[0], eval_seeds[0] + len(eval_seeds))):
        raise ValueError(f"eval seeds must be consecutive, got {eval_seeds}")

    cfg = GeometryConfig(
        model_name=model_name,
        model_revision=revision,
        groups=tuple(sorted(groups)),
        k=k,
        num_examples=num_examples,
        context="random",
        runs=len(selection_seeds),
        seed=selection_seeds[0],
        device=device,
        dtype=dtype,
        spacing_fit="direct",
        filter_correct=True,
        filter_max_new_tokens=filter_max_new_tokens,
        filter_max_candidates=filter_max_candidates,
    )

    # ---- selection ------------------------------------------------------- #
    _seed_all(cfg.seed)
    torch_device = _resolve_device(cfg.device)
    model, tokenizer = _load_model(cfg, torch_device, hf_token)
    print(f"[datadecide] loaded {model_name}@{revision or 'main'} "
          f"({type(model).__name__}, {model.config.num_hidden_layers} blocks) on {torch_device}")

    per_seed_direct, per_seed_log, per_seed_out, selection_filter = [], [], [], []
    diagnostics = None
    for seed in selection_seeds:
        print(f"[datadecide] selection seed {seed}")
        bundle = collect_hidden_states(cfg, model, tokenizer, torch_device, random.Random(seed))
        if diagnostics is None:
            diagnostics = bundle.tokenization_diagnostics
        direct, log = score_bundle_layers(bundle)
        per_seed_direct.append(direct)
        per_seed_log.append(log)
        selection_filter.append({"seed": seed, "groups": bundle.filter_stats})
        per_seed_out.append(
            {
                "seed": seed,
                "layers": {
                    str(layer): {
                        "ev": m.explained_variance,
                        "rho": abs(m.monotonicity),
                        "beta_direct": m.compression_rate,
                        "r2_direct": m.compression_r2,
                        "beta_log": log[layer].compression_rate,
                    }
                    for layer, m in direct.items()
                },
            }
        )

    final_token_is_equals = bool(
        diagnostics["prompt_count"] > 0
        and diagnostics["prompts_ending_equals_token"] == diagnostics["prompt_count"]
    )
    if not final_token_is_equals:
        print(f"[datadecide] WARNING: final prompt token is not '=' for "
              f"{diagnostics['prompt_count'] - diagnostics['prompts_ending_equals_token']} prompts")

    selection = select_joint_layer(per_seed_direct)
    summary_direct = _summarize(per_seed_direct)
    summary_log = _summarize(per_seed_log)
    target_layer = selection["selected_layer"]
    print(f"[datadecide] selected layer {target_layer} (median q={selection['selected_q_median']:.4f})")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- frozen-layer evaluation ----------------------------------------- #
    eval_cfg = replace(cfg, seed=eval_seeds[0], runs=len(eval_seeds))
    evaluation = run_frozen_layer_evaluation(eval_cfg, target_layer, hf_token=hf_token)
    eval_filter = [
        {"seed": row["seed"], "groups": row.get("filter_stats", {})} for row in evaluation["runs"]
    ]

    return {
        "model_name": model_name,
        "revision": revision,
        "config": asdict(cfg),
        "selection_seeds": list(selection_seeds),
        "eval_seeds": list(eval_seeds),
        "tokenization_diagnostics": diagnostics,
        "final_token_is_equals": final_token_is_equals,
        "selection": {
            "criterion": "argmax_layer median_seed sqrt(EV * |rho|) (PCA, 1 component)",
            **selection,
            "mean_joint_layer": _best_joint_layer(summary_direct),
            "per_seed": per_seed_out,
            "layers_direct": {str(l): asdict(v) for l, v in summary_direct.items()},
            "layers_log": {str(l): asdict(v) for l, v in summary_log.items()},
            "filter_stats": selection_filter,
        },
        "evaluation": evaluation,
        "filter_summary": _filter_summary(selection_filter + eval_filter),
    }


# --------------------------------------------------------------------------- #
# summary CSV
# --------------------------------------------------------------------------- #

def summary_row(recipe: str, payload: dict) -> dict:
    s = payload["evaluation"]["summary"]
    fs = payload["filter_summary"]
    return {
        "recipe": recipe,
        "selected_layer": payload["selection"]["selected_layer"],
        "rho_mean": s["rho_mean"],
        "rho_std": s["rho_std"],
        "ev_mean": s["explained_variance_mean"],
        "ev_std": s["explained_variance_std"],
        "beta_direct_mean": s["beta_mean"],
        "beta_direct_std": s["beta_std"],
        "r2_direct": s["r2_beta_mean"],
        "beta_log_mean": s["beta_log_mean"],
        "beta_log_std": s["beta_log_std"],
        "rejection_rate_per_group": ";".join(
            f"{g}:{v['rejection_rate']:.3f}" for g, v in fs.items()
        ),
        "failed_groups": ";".join(g for g, v in fs.items() if v["failed_seeds"]),
    }


def write_summary_csv(output_dir: Path, recipe_order: list[str] | None = None) -> Path:
    """(Re)build summary.csv from every per-model JSON in `output_dir`."""
    output_dir = Path(output_dir)
    payloads = {}
    for path in sorted(output_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if "recipe" in payload and "evaluation" in payload:
            payloads[payload["recipe"]] = payload
    order = [r for r in (recipe_order or []) if r in payloads]
    order += sorted(r for r in payloads if r not in order)
    out = output_dir / "summary.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for recipe in order:
            writer.writerow(summary_row(recipe, payloads[recipe]))
    return out


def recipe_slug(model_name: str) -> str:
    """allenai/DataDecide-<recipe>-<size> -> <recipe>; other ids are slugified."""
    name = model_name.split("/")[-1]
    if name.startswith("DataDecide-"):
        return name[len("DataDecide-"):].rsplit("-", 1)[0]
    return model_name.replace("/", "_")


# --------------------------------------------------------------------------- #
# local CLI (CPU pilot / smoke runs; the full sweep is modal_app/datadecide_app.py)
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the DataDecide geometry protocol on one model locally.")
    p.add_argument("--model", required=True, help="recipe slug or full HF repo id")
    p.add_argument("--revision", default=None,
                   help="HF revision; default: the config revision for slugs, main for full ids")
    p.add_argument("--groups", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--selection-seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[45, 46, 47])
    p.add_argument("--filter-max-candidates", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--output-dir", default="results/datadecide")
    p.add_argument("--verify-only", action="store_true",
                   help="only check that the repo and revision exist on the Hub")
    args = p.parse_args(argv)

    config = load_model_config()
    repo_id = repo_id_for(args.model, config)
    revision = args.revision if args.revision is not None else (
        None if "/" in args.model else config["revision"]
    )
    problems = verify_revisions([repo_id], revision)
    if problems:
        raise SystemExit(f"[datadecide] {repo_id}: {problems[repo_id]}")
    print(f"[datadecide] verified {repo_id}@{revision or 'main'}")
    if args.verify_only:
        return 0

    payload = run_datadecide_model(
        repo_id,
        revision,
        groups=tuple(args.groups),
        k=args.k,
        selection_seeds=tuple(args.selection_seeds),
        eval_seeds=tuple(args.eval_seeds),
        filter_max_candidates=args.filter_max_candidates,
        device=args.device,
        dtype=args.dtype,
    )
    recipe = recipe_slug(repo_id)
    payload["recipe"] = recipe
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{recipe}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[datadecide] wrote {out} and {write_summary_csv(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
