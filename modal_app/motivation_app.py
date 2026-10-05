from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "numberline-motivation-scale"
DEFAULT_GPU = os.environ.get("MODAL_MOTIVATION_GPU", "A10G")

DEFAULT_MODEL_SPECS = {
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
    kwargs: dict = {"gpu": gpu, "timeout": timeout}
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
) -> dict:
    import math
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.motivation_scale import build_scale_tasks, summarize_rows

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    print(f"[motivation:{model_label}] loading {model_name} revision={model_revision or 'main'}")
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

    def _candidate_logprob(prompt: str, letter: str) -> float:
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        cand_ids = tokenizer(f" {letter}", add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
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
    tasks = build_scale_tasks()
    for i, task in enumerate(tasks, 1):
        scores = {letter: _candidate_logprob(task.prompt, letter) for letter in task.choices}
        selected = max(scores, key=scores.get)
        selected_value = None
        selected_log10_error = None
        if task.true_value is not None and task.choice_values is not None:
            selected_value = task.choice_values[selected]
            if selected_value > 0 and task.true_value > 0:
                selected_log10_error = math.log10(selected_value / task.true_value)
        row = {
            "model_label": model_label,
            "model_name": model_name,
            "model_revision": model_revision or "",
            "task_id": task.task_id,
            "task_type": task.task_type,
            "scale": task.scale,
            "correct": task.correct,
            "selected": selected,
            "correct_selected": int(selected == task.correct),
            "selected_score": scores[selected],
            "correct_score": scores[task.correct],
            "score_margin_selected_minus_correct": scores[selected] - scores[task.correct],
            "true_value": task.true_value,
            "selected_value": selected_value,
            "selected_log10_error": selected_log10_error,
        }
        for letter, score in scores.items():
            row[f"score_{letter}"] = score
            row[f"choice_{letter}"] = task.choices[letter]
        rows.append(row)
        if i % 10 == 0 or i == len(tasks):
            print(f"[motivation:{model_label}] scored {i}/{len(tasks)}")

    summary = summarize_rows(rows)
    print(
        f"[motivation:{model_label}] accuracy={summary['accuracy']:.3f} "
        f"mean_abs_log10_error={summary['mean_abs_log10_error']}"
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
) -> None:
    import json

    try:
        import mlflow
    except ModuleNotFoundError:
        print("[motivation] local mlflow package not found; skipping MLflow logging")
        return

    from utils.mlflow_utils import log_metric, log_params, setup_tracking, start_run

    setup_tracking(None, experiment_name)
    with start_run(run_name=run_name, tags={"stage": "motivation_scale_reasoning", "backend": "modal"}):
        log_params(
            {
                "n_models": len(payloads),
                "n_tasks": payloads[0]["summary"]["n_tasks"] if payloads else 0,
                "task_format": "forced_choice_text_encoded_visual_scale",
                "excluded": "LoLa, Pythia/Pile baseline",
            }
        )
        for payload in payloads:
            prefix = payload["model_label"]
            summary = payload["summary"]
            log_metric(f"{prefix}_accuracy", summary["accuracy"])
            if summary["mean_abs_log10_error"] is not None:
                log_metric(f"{prefix}_mean_abs_log10_error", summary["mean_abs_log10_error"])
            if summary["linear_compression_error_rate"] is not None:
                log_metric(
                    f"{prefix}_linear_compression_error_rate",
                    summary["linear_compression_error_rate"],
                )
            if summary["log_expansion_error_rate"] is not None:
                log_metric(
                    f"{prefix}_log_expansion_error_rate",
                    summary["log_expansion_error_rate"],
                )
        mlflow.log_text(json.dumps(payloads, indent=2), "motivation_scale_payloads.json")
        for artifact in [
            "motivation_scale_rows.csv",
            "motivation_scale_summary.csv",
            "motivation_beta_vs_accuracy.pdf",
            "motivation_task_type_heatmap.pdf",
        ]:
            path = outdir / artifact
            if path.exists():
                mlflow.log_artifact(str(path))
        active = mlflow.active_run()
        print(f"[motivation] MLflow run_id={active.info.run_id}")


@app.local_entrypoint()
def main(
    models: str = "all",
    dtype: str = "float16",
    experiment_name: str = "numberline_geometry_expv1",
    run_name: str = "motivation_scale_reasoning_without_lola_pile",
    results_dir: str = "results/motivation_scale/without_lola_pile",
    dry_run: bool = False,
) -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.motivation_scale import build_scale_tasks, plot_results, write_results

    selected = _parse_models(models)
    tasks = build_scale_tasks()
    print(f"[motivation] models={ [label for label, _ in selected] }")
    print(f"[motivation] n_tasks={len(tasks)}")
    if dry_run:
        print("[motivation] --dry-run set; exiting before cloud scoring")
        return

    futures = [
        score_model.spawn(
            model_label=label,
            model_name=spec["name"],
            model_revision=spec["revision"],
            trust_remote_code=spec["trust"],
            dtype=dtype,
        )
        for label, spec in selected
    ]

    payloads: list[dict] = []
    for (label, _), fut in zip(selected, futures):
        print(f"[motivation] waiting on {label} ...")
        try:
            payloads.append(fut.get())
        except Exception as exc:
            print(f"[motivation] {label} FAILED: {exc}")

    outdir = Path(results_dir)
    write_results(outdir, payloads)
    plot_results(outdir)
    _log_payloads(
        payloads,
        experiment_name=experiment_name,
        run_name=run_name,
        outdir=outdir,
    )
    print(f"[motivation] wrote results to {outdir}")
