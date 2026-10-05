from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "numberline-motivation-bias"
DEFAULT_GPU = os.environ.get("MODAL_MOTIVATION_BIAS_GPU", "H100")

DEFAULT_MODEL_SPECS = {
    "Llama-2-7B": {
        "name": "meta-llama/Llama-2-7b-hf",
        "revision": None,
        "trust": False,
    },
    "Pythia-2.8B": {
        "name": "EleutherAI/pythia-2.8b",
        "revision": None,
        "trust": False,
    },
    "Mistral-7B": {
        "name": "mistralai/Mistral-7B-v0.1",
        "revision": None,
        "trust": False,
    },
    "DeepSeek-Base-7B": {
        "name": "deepseek-ai/deepseek-llm-7b-base",
        "revision": None,
        "trust": False,
    },
    "Qwen1.5-7B": {
        "name": "Qwen/Qwen1.5-7B",
        "revision": None,
        "trust": False,
    },
    "Falcon-RW-1B": {"name": "tiiuae/falcon-rw-1b", "revision": None, "trust": False},
    "Falcon-RW-7B": {"name": "tiiuae/falcon-rw-7b", "revision": None, "trust": False},
    "RedPajama-3B": {
        "name": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
        "revision": None,
        "trust": False,
    },
    "RedPajama-7B": {
        "name": "togethercomputer/RedPajama-INCITE-7B-Base",
        "revision": None,
        "trust": False,
    },
    "OLMo-7B-2T": {
        "name": "allenai/OLMo-7B",
        "revision": "step452000-tokens2000B",
        "trust": True,
    },
    "OLMo-7B-Twin-2T": {
        "name": "allenai/OLMo-7B-Twin-2T",
        "revision": None,
        "trust": True,
    },
    "StarCoderBase-1B": {"name": "bigcode/starcoderbase-1b", "revision": None, "trust": False},
    "StarCoderBase-3B": {"name": "bigcode/starcoderbase-3b", "revision": None, "trust": False},
    "StarCoderBase-7B": {"name": "bigcode/starcoderbase-7b", "revision": None, "trust": False},
}


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.44.0,<5",
        "accelerate>=0.33.0",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
        "huggingface_hub>=0.24.0",
        "datasets>=2.20.0",
        "safetensors>=0.4.3",
        "ai2-olmo==0.6.0",
        "mlflow>=2.16.0",
    )
    .add_local_python_source("src", "utils")
)

app = modal.App(APP_NAME, image=image)

_HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET_NAME", "numberline-hf-token").strip()
_HF_SECRETS = [modal.Secret.from_name(_HF_SECRET_NAME)] if _HF_SECRET_NAME else []


def _gpu_kwargs(gpu: str, timeout: int) -> dict:
    kwargs: dict = {
        "gpu": gpu,
        "timeout": timeout,
        "max_containers": 100,
    }
    if _HF_SECRETS:
        kwargs["secrets"] = _HF_SECRETS
    return kwargs


@app.function(**_gpu_kwargs(DEFAULT_GPU, timeout=2 * 60 * 60))
def score_model(
    model_label: str,
    model_name: str,
    *,
    model_revision: str | None = None,
    trust_remote_code: bool = False,
    dtype: str = "float16",
    task_types: tuple[str, ...] = (),
) -> dict:
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.motivation_bias import build_bias_tasks, summarize_rows

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    print(
        f"[bias:{model_label}] loading {model_name} "
        f"revision={model_revision or 'main'} gpu={DEFAULT_GPU}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
        token=hf_token,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()

    def _candidate_logprob(prompt: str, candidate: str) -> float:
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        cand_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        input_ids = torch.cat([prompt_ids, cand_ids], dim=1)
        with torch.no_grad():
            logits = model(input_ids).logits
            log_probs = torch.log_softmax(logits, dim=-1)
        start = prompt_ids.shape[1] - 1
        total = 0.0
        for offset, token_id in enumerate(cand_ids[0]):
            total += float(log_probs[0, start + offset, int(token_id)].detach().cpu())
        return total / max(1, cand_ids.shape[1])

    rows: list[dict] = []
    tasks = build_bias_tasks()
    if task_types:
        task_type_set = set(task_types)
        tasks = [task for task in tasks if task.task_type in task_type_set]
    for i, task in enumerate(tasks, 1):
        scores = {
            key: _candidate_logprob(task.prompt, candidate)
            for key, candidate in task.candidates.items()
        }
        signed = scores[task.positive_key] - scores[task.negative_key]
        selected = max(scores, key=scores.get)
        row = {
            "model_label": model_label,
            "model_name": model_name,
            "model_revision": model_revision or "",
            "task_id": task.task_id,
            "task_type": task.task_type,
            "positive_key": task.positive_key,
            "negative_key": task.negative_key,
            "positive_meaning": task.positive_meaning,
            "signed_score": signed,
            "selected": selected,
            "positive_selected": int(selected == task.positive_key),
            "negative_selected": int(selected == task.negative_key),
        }
        for key, value in task.metadata.items():
            row[f"meta_{key}"] = value
        for key, value in scores.items():
            row[f"score_{key}"] = value
            row[f"candidate_{key}"] = task.candidates[key]
        rows.append(row)
        if i % 20 == 0 or i == len(tasks):
            print(f"[bias:{model_label}] scored {i}/{len(tasks)}")

    summary = summarize_rows(rows)
    print(
        f"[bias:{model_label}] mean_signed={summary['mean_signed_score']:+.3f} "
        f"positive_rate={summary['positive_rate']:.3f}"
    )
    return {
        "model_label": model_label,
        "model_name": model_name,
        "model_revision": model_revision,
        "trust_remote_code": trust_remote_code,
        "summary": summary,
        "rows": rows,
    }


def _parse_models(models: str) -> list[tuple[str, dict]]:
    if models.strip().lower() in {"", "all", "default"}:
        labels = list(DEFAULT_MODEL_SPECS)
    else:
        labels = [part.strip() for part in models.split(",") if part.strip()]
    bad = sorted(set(labels) - set(DEFAULT_MODEL_SPECS))
    if bad:
        raise SystemExit(f"unknown motivation model label(s): {bad}")
    return [(label, DEFAULT_MODEL_SPECS[label]) for label in labels]


def _log_payloads(
    payloads: list[dict],
    *,
    experiment_name: str,
    run_name: str,
    outdir: Path,
    gpu: str,
) -> None:
    import json

    try:
        import mlflow
    except ModuleNotFoundError:
        print("[bias] local mlflow package not found; skipping MLflow logging")
        return

    from utils.mlflow_utils import log_metric, log_params, setup_tracking, start_run

    with open(outdir / "motivation_bias_correlations.json") as fh:
        correlations = json.load(fh)

    setup_tracking(None, experiment_name)
    with start_run(run_name=run_name, tags={"stage": "motivation_bias_probe", "backend": "modal"}):
        log_params(
            {
                "n_models": len(payloads),
                "n_tasks": payloads[0]["summary"]["n_tasks"] if payloads else 0,
                "task_format": "signed_likelihood_bias",
                "gpu": gpu,
                "max_containers": 100,
                "excluded": "LoLa, Pythia/Pile baseline",
            }
        )
        for k, v in correlations.items():
            log_metric(k, float(v))
        for payload in payloads:
            prefix = payload["model_label"]
            summary = payload["summary"]
            log_metric(f"{prefix}_mean_signed_score", summary["mean_signed_score"])
            log_metric(f"{prefix}_positive_rate", summary["positive_rate"])
            for task_type, blob in summary["by_type"].items():
                log_metric(f"{prefix}_{task_type}_mean_signed_score", blob["mean_signed_score"])
                log_metric(f"{prefix}_{task_type}_positive_rate", blob["positive_rate"])
        mlflow.log_text(json.dumps(payloads, indent=2), "motivation_bias_payloads.json")
        for artifact in [
            "motivation_bias_rows.csv",
            "motivation_bias_summary.csv",
            "motivation_bias_correlations.json",
            "motivation_bias_summary.tex",
            "motivation_bias_beta_panels.pdf",
            "motivation_bias_heatmap.pdf",
        ]:
            path = outdir / artifact
            if path.exists():
                mlflow.log_artifact(str(path))
        active = mlflow.active_run()
        print(f"[bias] MLflow run_id={active.info.run_id}")


@app.local_entrypoint()
def main(
    models: str = "all",
    dtype: str = "float16",
    gpu: str = DEFAULT_GPU,
    task_types: str = "",
    experiment_name: str = "numberline_geometry_expv1",
    run_name: str = "motivation_bias_probe_without_lola_pile",
    results_dir: str = "results/motivation_bias/without_lola_pile",
    beta_csv: str = "results/dataset_frequency_0_to_10000/alpha_beta_without_pile/alpha_beta_models_without_pile.csv",
    dry_run: bool = False,
) -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.motivation_bias import (
        build_bias_tasks,
        correlation_summary,
        plot_results,
        write_latex_summary,
        write_results,
    )

    selected = _parse_models(models)
    tasks = build_bias_tasks()
    selected_task_types = tuple(part.strip() for part in task_types.split(",") if part.strip())
    if selected_task_types:
        allowed_task_types = {task.task_type for task in tasks}
        bad_task_types = sorted(set(selected_task_types) - allowed_task_types)
        if bad_task_types:
            raise SystemExit(f"unknown task type(s): {bad_task_types}")
        tasks = [task for task in tasks if task.task_type in set(selected_task_types)]
    print(f"[bias] models={ [label for label, _ in selected] }")
    if selected_task_types:
        print(f"[bias] task_types={list(selected_task_types)}")
    print(f"[bias] n_tasks={len(tasks)} gpu={gpu} max_containers=100")
    if dry_run:
        print("[bias] --dry-run set; exiting before cloud scoring")
        return

    futures = [
        score_model.spawn(
            model_label=label,
            model_name=spec["name"],
            model_revision=spec["revision"],
            trust_remote_code=spec["trust"],
            dtype=dtype,
            task_types=selected_task_types,
        )
        for label, spec in selected
    ]

    payloads: list[dict] = []
    for (label, _), fut in zip(selected, futures):
        print(f"[bias] waiting on {label} ...")
        try:
            payloads.append(fut.get())
        except Exception as exc:
            print(f"[bias] {label} FAILED: {exc}")

    outdir = Path(results_dir)
    beta_path = Path(beta_csv)
    write_results(outdir, payloads)
    plot_results(outdir, beta_path)
    correlation_summary(outdir, beta_path)
    write_latex_summary(outdir, beta_path)
    _log_payloads(
        payloads,
        experiment_name=experiment_name,
        run_name=run_name,
        outdir=outdir,
        gpu=gpu,
    )
    print(f"[bias] wrote results to {outdir}")
