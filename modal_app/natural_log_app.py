from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR_NAME = os.environ.get("NATURAL_LOG_SOURCE_DIR", "llm_natural_log")
LLM_NATURAL_LOG_DIR = REPO_ROOT / SOURCE_DIR_NAME

APP_NAME = "llm-natural-log-exact"
DEFAULT_GPU = os.environ.get("MODAL_NATURAL_LOG_GPU", "A10G")
CACHE_PATH = "/cache"
CODE_PATH = f"/root/{SOURCE_DIR_NAME}"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.44.0,<5",
        "accelerate>=0.33.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.11.0",
        "numpy>=1.26.0",
        "huggingface_hub>=0.24.0",
        "matplotlib>=3.8.0",
        "safetensors>=0.4.3",
    )
    .add_local_dir(
        LLM_NATURAL_LOG_DIR,
        remote_path=CODE_PATH,
        ignore=[".git", "__pycache__", "*.pyc", ".ipynb_checkpoints"],
    )
)

app = modal.App(APP_NAME, image=image)
cache_volume = modal.Volume.from_name("llm-natural-log-cache", create_if_missing=True)

_HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET_NAME", "numberline-hf-token").strip()
_HF_SECRETS = [modal.Secret.from_name(_HF_SECRET_NAME)] if _HF_SECRET_NAME else []


def _function_kwargs(timeout: int) -> dict:
    kwargs: dict = {
        "gpu": DEFAULT_GPU,
        "timeout": timeout,
        "memory": 32768,
        "volumes": {CACHE_PATH: cache_volume},
    }
    if _HF_SECRETS:
        kwargs["secrets"] = _HF_SECRETS
    return kwargs


@app.function(**_function_kwargs(timeout=4 * 60 * 60))
def run_exact_table1_cell(
    model_name: str = "EleutherAI/pythia-2.8b",
    *,
    data: str = "numerics",
    k: int = 30,
    num_examples: int = 3,
    context: str = "random",
    groups: list[int] = [1, 2, 3, 4],
    runs: int = 3,
    transform: str = "PCA",
    tdim: int = 1,
    device: str = "0",
    interval_mode: str = "repo",
    beta_method: str = "repo_loglinear",
    filter_correct: bool = False,
    filter_data: str = "numerics",
    filter_max_new_tokens: int = 8,
    letter_mode: str = "repo",
    source_dir_name: str = SOURCE_DIR_NAME,
) -> dict:
    import json
    import os
    import random
    import sys
    from types import SimpleNamespace

    import huggingface_hub
    import numpy as np

    os.environ.setdefault("HF_HOME", f"{CACHE_PATH}/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{CACHE_PATH}/huggingface/transformers")
    os.environ.setdefault("HF_HUB_CACHE", f"{CACHE_PATH}/huggingface/hub")

    # The released repo calls login(token="") at import time. Keep the
    # experiment code intact, but neutralize that broken empty-token login.
    huggingface_hub.login = lambda *args, **kwargs: None

    code_path = f"/root/{source_dir_name}"
    os.chdir(code_path)
    sys.path.insert(0, code_path)
    sys.modules.pop("utils", None)

    from main_lab import analyzer

    args = SimpleNamespace(
        transform=transform,
        Tdim=tdim,
        k=k,
        num_examples=num_examples,
        context=context,
        data=data,
        groups=groups,
        upper_bound=10 ** max(groups),
        save=True,
        plot=True,
        model_name=model_name,
        device=device,
        runs=runs,
        interval_mode=interval_mode,
        beta_method=beta_method,
        filter_correct=filter_correct,
        filter_data=filter_data,
        filter_max_new_tokens=filter_max_new_tokens,
        letter_mode=letter_mode,
    )

    print(
        "[natural-log] exact Table_1 cell start "
        f"model={model_name} data={data} groups={groups} k={k} "
        f"num_examples={num_examples} context={context} runs={runs}"
    )
    res = analyzer(args)
    res.groups = groups
    res.data = data
    results_pca, results_pls = res.run_multiple()

    def _clean(method_results: dict) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for layer, metrics in method_results.items():
            out[str(layer)] = {
                "EV": float(metrics["EV"]),
                "EV_std": float(metrics["EV_std"]),
                "rho": float(metrics["rho"]),
                "rho_std": float(metrics["rho_std"]),
                "beta": float(metrics["beta"]),
                "beta_std": float(metrics["beta_std"]),
            }
        return out

    def _best(method_results: dict) -> dict:
        layer, metrics = sorted(
            method_results.items(),
            key=lambda item: item[1]["EV"],
            reverse=True,
        )[0]
        return {
            "layer": int(layer),
            "EV": float(metrics["EV"]),
            "EV_std": float(metrics["EV_std"]),
            "rho": float(metrics["rho"]),
            "rho_std": float(metrics["rho_std"]),
            "beta": float(metrics["beta"]),
            "beta_std": float(metrics["beta_std"]),
        }

    payload = {
        "config": {
            "model_name": model_name,
            "data": data,
            "groups": groups,
            "k": k,
            "num_examples": num_examples,
            "context": context,
            "runs": runs,
            "transform": transform,
            "Tdim": tdim,
            "upper_bound": 10 ** max(groups),
            "code_path": CODE_PATH,
            "source": f"{source_dir_name}/main_lab.py analyzer.run_multiple",
            "source_dir": source_dir_name,
            "interval_mode": interval_mode,
            "beta_method": beta_method,
            "filter_correct": filter_correct,
            "filter_data": filter_data,
            "filter_max_new_tokens": filter_max_new_tokens,
            "letter_mode": letter_mode,
            "notes": [
                "huggingface_hub.login(token='') is monkeypatched to no-op when present",
                "generated-output correctness filtering is controlled by filter_correct/filter_data",
            ],
        },
        "best_pca": _best(results_pca),
        "best_pls": _best(results_pls),
        "pca": _clean(results_pca),
        "pls": _clean(results_pls),
    }
    print("[natural-log] exact Table_1 cell done")
    print(json.dumps({"best_pca": payload["best_pca"], "best_pls": payload["best_pls"]}, indent=2))
    return payload


def _write_outputs(payload: dict, out_dir: str) -> None:
    import csv
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "results.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    for method in ("pca", "pls"):
        path = out / f"{method}_by_layer.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["layer", "EV", "EV_std", "rho", "rho_std", "beta", "beta_std"])
            for layer, metrics in sorted(payload[method].items(), key=lambda item: int(item[0])):
                writer.writerow(
                    [
                        layer,
                        metrics["EV"],
                        metrics["EV_std"],
                        metrics["rho"],
                        metrics["rho_std"],
                        metrics["beta"],
                        metrics["beta_std"],
                    ]
                )

    tex = (
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Method & Best layer & EV & $|\\rho_S|$ & $\\beta$ \\\\\n"
        "\\midrule\n"
    )
    for method, label in (("best_pca", "PCA"), ("best_pls", "PLS")):
        row = payload[method]
        tex += (
            f"{label} & {row['layer']} & "
            f"${row['EV']:.3f}\\pm{row['EV_std']:.3f}$ & "
            f"${row['rho']:.3f}\\pm{row['rho_std']:.3f}$ & "
            f"${row['beta']:.3f}\\pm{row['beta_std']:.3f}$ \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}\n"
    (out / "summary.tex").write_text(tex)


def _plot_outputs(payload: dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    layers = [int(l) for l in sorted(payload["pca"], key=int)]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    specs = [
        ("EV", "EV_std", "Explained variance / R2"),
        ("rho", "rho_std", "|Spearman rho|"),
        ("beta", "beta_std", "beta"),
    ]
    for ax, (key, std_key, ylabel) in zip(axes, specs):
        for method, label in (("pca", "PCA"), ("pls", "PLS")):
            y = [payload[method][str(l)][key] for l in layers]
            err = [payload[method][str(l)][std_key] for l in layers]
            ax.plot(layers, y, marker="o", linewidth=1.8, label=label)
            ax.fill_between(
                layers,
                [a - b for a, b in zip(y, err)],
                [a + b for a, b in zip(y, err)],
                alpha=0.18,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if key == "beta":
            ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Layer index used by llm_natural_log")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle(f"{payload['config']['model_name']} / {payload['config']['data']} exact llm_natural_log")
    fig.tight_layout()
    fig.savefig(figs / "exact_layer_profile.png", dpi=220)
    fig.savefig(figs / "exact_layer_profile.pdf")
    plt.close(fig)


@app.local_entrypoint()
def main(
    model_name: str = "EleutherAI/pythia-2.8b",
    data: str = "numerics",
    k: int = 30,
    num_examples: int = 3,
    context: str = "random",
    groups: str = "1,2,3,4",
    runs: int = 3,
    interval_mode: str = "repo",
    beta_method: str = "repo_loglinear",
    filter_correct: bool = False,
    filter_data: str = "numerics",
    filter_max_new_tokens: int = 8,
    letter_mode: str = "repo",
    out_dir: str = "results/llm_natural_log/pythia_2_8b/numerics_exact",
) -> None:
    group_list = [int(x) for x in groups.split(",") if x.strip()]
    payload = run_exact_table1_cell.remote(
        model_name,
        data=data,
        k=k,
        num_examples=num_examples,
        context=context,
        groups=group_list,
        runs=runs,
        interval_mode=interval_mode,
        beta_method=beta_method,
        filter_correct=filter_correct,
        filter_data=filter_data,
        filter_max_new_tokens=filter_max_new_tokens,
        letter_mode=letter_mode,
        source_dir_name=SOURCE_DIR_NAME,
    )
    _write_outputs(payload, out_dir)
    _plot_outputs(payload, out_dir)
    print(f"[natural-log] wrote {out_dir}")
