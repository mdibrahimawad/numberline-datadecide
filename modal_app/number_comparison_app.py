from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "numberline-number-comparison"
DEFAULT_GPU = os.environ.get("MODAL_NUMBER_COMPARISON_GPU", "H100")

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
    "GPT2-L": {
        "name": "openai-community/gpt2-large",
        "revision": None,
        "trust": False,
    },
    "GPT-Neo-125M": {
        "name": "EleutherAI/gpt-neo-125m",
        "revision": None,
        "trust": False,
    },
    "GPT-Neo-1.3B": {
        "name": "EleutherAI/gpt-neo-1.3B",
        "revision": None,
        "trust": False,
    },
    "GPT-Neo-2.7B": {
        "name": "EleutherAI/gpt-neo-2.7B",
        "revision": None,
        "trust": False,
    },
    "GPT-J-6B": {
        "name": "EleutherAI/gpt-j-6B",
        "revision": None,
        "trust": False,
    },
    "Mistral-7B": {
        "name": "mistralai/Mistral-7B-v0.1",
        "revision": None,
        "trust": False,
    },
    "Llama-3.1-8B": {
        "name": "meta-llama/Llama-3.1-8B",
        "revision": None,
        "trust": False,
    },
    "Llama-3.2-1B-Instruct": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
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
    "OPT-1.3B": {"name": "facebook/opt-1.3b", "revision": None, "trust": False},
    "OPT-2.7B": {"name": "facebook/opt-2.7b", "revision": None, "trust": False},
    "OPT-6.7B": {"name": "facebook/opt-6.7b", "revision": None, "trust": False},
    "Cerebras-GPT-1.3B": {"name": "cerebras/Cerebras-GPT-1.3B", "revision": None, "trust": False},
    "Cerebras-GPT-2.7B": {"name": "cerebras/Cerebras-GPT-2.7B", "revision": None, "trust": False},
    "Cerebras-GPT-6.7B": {"name": "cerebras/Cerebras-GPT-6.7B", "revision": None, "trust": False},
    "BLOOM-1.7B": {"name": "bigscience/bloom-1b7", "revision": None, "trust": False},
    "BLOOM-3B": {"name": "bigscience/bloom-3b", "revision": None, "trust": False},
    "StableLM-Base-3B": {"name": "stabilityai/stablelm-base-alpha-3b", "revision": None, "trust": False},
    "StableLM-Base-7B": {"name": "stabilityai/stablelm-base-alpha-7b", "revision": None, "trust": False},
    "OpenLLaMA-3B-v2": {"name": "openlm-research/open_llama_3b_v2", "revision": None, "trust": False},
    "OpenLLaMA-3B": {"name": "openlm-research/open_llama_3b", "revision": None, "trust": False},
    "OpenLLaMA-7B": {"name": "openlm-research/open_llama_7b", "revision": None, "trust": False},
    "OpenLLaMA-13B": {"name": "openlm-research/open_llama_13b", "revision": None, "trust": False},
    "Qwen1.5-0.5B": {"name": "Qwen/Qwen1.5-0.5B", "revision": None, "trust": False},
    "Qwen1.5-1.8B": {"name": "Qwen/Qwen1.5-1.8B", "revision": None, "trust": False},
    "Qwen1.5-4B": {"name": "Qwen/Qwen1.5-4B", "revision": None, "trust": False},
    "BLOOM-7B1": {"name": "bigscience/bloom-7b1", "revision": None, "trust": False},
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
        "sentencepiece>=0.2.0",
        "tiktoken>=0.7.0",
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
def score_model_family(
    model_label: str,
    model_name: str,
    family: str,
    *,
    model_revision: str | None = None,
    trust_remote_code: bool = False,
    dtype: str = "float16",
    answer_mode: str = "number",
    n_shots: int = 0,
    exemplar_set_id: int = 0,
) -> dict:
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.number_comparison import build_comparison_tasks, summarize_rows

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    print(
        f"[compare:{model_label}:{family}] loading {model_name} "
        f"revision={model_revision or 'main'} answer_mode={answer_mode} "
        f"n_shots={n_shots} exemplar_set={exemplar_set_id} gpu={DEFAULT_GPU}"
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
    tasks = build_comparison_tasks(
        [family],
        answer_mode=answer_mode,
        n_shots=n_shots,
        exemplar_set_id=exemplar_set_id,
    )
    for i, task in enumerate(tasks, 1):
        scores = {
            key: _candidate_logprob(task.prompt, candidate)
            for key, candidate in task.candidates.items()
        }
        selected_key = max(scores, key=scores.get)
        wrong_key = "B" if task.correct_key == "A" else "A"
        row = {
            "model_label": model_label,
            "model_name": model_name,
            "model_revision": model_revision or "",
            "answer_mode": answer_mode,
            "n_shots": n_shots,
            "exemplar_set_id": exemplar_set_id,
            "family": task.family,
            "task_id": task.task_id,
            "a_value": task.a_value,
            "b_value": task.b_value,
            "query": task.query,
            "order": task.order,
            "group": task.group,
            "gap": task.gap,
            "ratio": task.ratio,
            "correct_key": task.correct_key,
            "selected_key": selected_key,
            "correct": int(selected_key == task.correct_key),
            "margin": scores[task.correct_key] - scores[wrong_key],
            "score_A": scores["A"],
            "score_B": scores["B"],
        }
        rows.append(row)
        if i % 80 == 0 or i == len(tasks):
            print(f"[compare:{model_label}:{family}] scored {i}/{len(tasks)}")

    summary = summarize_rows(rows)
    print(
        f"[compare:{model_label}:{family}] acc={summary['accuracy']:.3f} "
        f"margin={summary['mean_margin']:+.3f}"
    )
    return {
        "model_label": model_label,
        "model_name": model_name,
        "model_revision": model_revision or "",
        "trust_remote_code": trust_remote_code,
        "family": family,
        "answer_mode": answer_mode,
        "n_shots": n_shots,
        "exemplar_set_id": exemplar_set_id,
        "summary": summary,
        "rows": rows,
    }


@app.function(**_gpu_kwargs(DEFAULT_GPU, timeout=2 * 60 * 60))
def score_model_condition(
    model_label: str,
    model_name: str,
    families: list[str],
    *,
    model_revision: str | None = None,
    trust_remote_code: bool = False,
    dtype: str = "float16",
    answer_mode: str = "number",
    n_shots: int = 0,
    exemplar_set_id: int = 0,
) -> dict:
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.number_comparison import build_comparison_tasks, summarize_rows

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    family_label = "+".join(families)
    print(
        f"[compare:{model_label}:{n_shots}shot:set{exemplar_set_id}] loading {model_name} "
        f"families={family_label} revision={model_revision or 'main'} "
        f"answer_mode={answer_mode} gpu={DEFAULT_GPU}"
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
    tasks = build_comparison_tasks(
        families,
        answer_mode=answer_mode,
        n_shots=n_shots,
        exemplar_set_id=exemplar_set_id,
    )
    for i, task in enumerate(tasks, 1):
        scores = {
            key: _candidate_logprob(task.prompt, candidate)
            for key, candidate in task.candidates.items()
        }
        selected_key = max(scores, key=scores.get)
        wrong_key = "B" if task.correct_key == "A" else "A"
        rows.append(
            {
                "model_label": model_label,
                "model_name": model_name,
                "model_revision": model_revision or "",
                "answer_mode": answer_mode,
                "n_shots": n_shots,
                "exemplar_set_id": exemplar_set_id,
                "family": task.family,
                "task_id": task.task_id,
                "a_value": task.a_value,
                "b_value": task.b_value,
                "query": task.query,
                "order": task.order,
                "group": task.group,
                "gap": task.gap,
                "ratio": task.ratio,
                "correct_key": task.correct_key,
                "selected_key": selected_key,
                "correct": int(selected_key == task.correct_key),
                "margin": scores[task.correct_key] - scores[wrong_key],
                "score_A": scores["A"],
                "score_B": scores["B"],
            }
        )
        if i % 160 == 0 or i == len(tasks):
            print(
                f"[compare:{model_label}:{n_shots}shot:set{exemplar_set_id}] "
                f"scored {i}/{len(tasks)}"
            )

    summary = summarize_rows(rows)
    print(
        f"[compare:{model_label}:{n_shots}shot:set{exemplar_set_id}] "
        f"acc={summary['accuracy']:.3f} margin={summary['mean_margin']:+.3f}"
    )
    return {
        "model_label": model_label,
        "model_name": model_name,
        "model_revision": model_revision or "",
        "trust_remote_code": trust_remote_code,
        "family": family_label,
        "answer_mode": answer_mode,
        "n_shots": n_shots,
        "exemplar_set_id": exemplar_set_id,
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
        raise SystemExit(f"unknown model label(s): {bad}")
    return [(label, DEFAULT_MODEL_SPECS[label]) for label in labels]


def _parse_families(families: str) -> list[str]:
    default = ["fixed_gap", "random_within_group", "same_ratio", "boundary", "roundness"]
    available = default + [
        "fixed_gap_dense",
        "fixed_gap_dense_g1",
        "fixed_gap_dense_g2",
        "fixed_gap_dense_g3",
        "fixed_gap_dense_g4",
        "fixed_gap_wide_sg_mg",
        "fixed_gap_wide_sg_mg_g1",
        "fixed_gap_wide_sg_mg_g2",
        "fixed_gap_wide_sg_mg_g3",
        "fixed_gap_wide_sg_mg_g4",
    ]
    if families.strip().lower() in {"", "all", "default"}:
        return default
    selected = [part.strip() for part in families.split(",") if part.strip()]
    bad = sorted(set(selected) - set(available))
    if bad:
        raise SystemExit(f"unknown comparison families: {bad}")
    return selected


def _parse_shot_configs(
    shot_configs: str,
    *,
    n_shots: int,
    exemplar_set_id: int,
) -> list[tuple[int, int]]:
    if not shot_configs.strip():
        return [(n_shots, exemplar_set_id)]
    configs: list[tuple[int, int]] = []
    for part in shot_configs.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            shot_text, set_text = part.split(":", 1)
            config = (int(shot_text), int(set_text))
        else:
            config = (int(part), exemplar_set_id)
        configs.append(config)
    bad = [config for config in configs if config[0] not in {0, 1, 2, 3, 4} or config[1] not in {0, 1, 2}]
    if bad:
        raise SystemExit(f"bad shot config(s): {bad}; use shot:set with shots 0-4 and sets 0-2")
    return configs


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
        print("[compare] local mlflow package not found; skipping MLflow logging")
        return

    from utils.mlflow_utils import log_metric, log_params, setup_tracking, start_run

    correlations = {}
    corr_path = outdir / "number_comparison_beta_correlations.json"
    if corr_path.exists():
        with open(corr_path) as fh:
            correlations = json.load(fh)

    setup_tracking(None, experiment_name)
    with start_run(run_name=run_name, tags={"stage": "number_comparison_probe", "backend": "modal"}):
        log_params(
            {
                "n_payloads": len(payloads),
                "n_models": len({payload["model_label"] for payload in payloads}),
                "n_families": len({payload["family"] for payload in payloads}),
                "answer_mode": payloads[0].get("answer_mode", "unknown") if payloads else "unknown",
                "n_shots": payloads[0].get("n_shots", "unknown") if payloads else "unknown",
                "exemplar_set_id": payloads[0].get("exemplar_set_id", "unknown") if payloads else "unknown",
                "task_format": "digit_pair_likelihood",
                "gpu": gpu,
                "max_containers": 100,
                "shard": "model_x_family",
            }
        )
        for key, value in correlations.items():
            log_metric(key, float(value))
        for artifact in outdir.iterdir():
            if artifact.is_file():
                mlflow.log_artifact(str(artifact))
        active = mlflow.active_run()
        print(f"[compare] MLflow run_id={active.info.run_id}")


@app.local_entrypoint()
def main(
    models: str = "all",
    families: str = "all",
    dtype: str = "float16",
    gpu: str = DEFAULT_GPU,
    experiment_name: str = "numberline_geometry_expv1",
    run_name: str = "number_comparison_probe_target",
    results_dir: str = "results/number_comparison/target_number_answer",
    answer_mode: str = "number",
    n_shots: int = 0,
    exemplar_set_id: int = 0,
    shot_configs: str = "",
    dry_run: bool = False,
) -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.number_comparison import (
        build_comparison_tasks,
        compute_correlations,
        plot_results,
        write_latex_summary,
        write_results,
    )

    selected = _parse_models(models)
    selected_families = _parse_families(families)
    configs = _parse_shot_configs(
        shot_configs,
        n_shots=n_shots,
        exemplar_set_id=exemplar_set_id,
    )

    if shot_configs.strip():
        tasks_per_condition = len(
            build_comparison_tasks(
                selected_families,
                answer_mode=answer_mode,
                n_shots=configs[0][0],
                exemplar_set_id=configs[0][1],
            )
        )
        jobs = [(label, spec, config) for label, spec in selected for config in configs]
        print(f"[compare] models={ [label for label, _ in selected] }")
        print(
            f"[compare] families={selected_families} answer_mode={answer_mode} "
            f"shot_configs={configs} tasks_per_model_condition={tasks_per_condition} "
            f"jobs={len(jobs)} gpu={gpu} max_containers=100"
        )
        if dry_run:
            print("[compare] --dry-run set; exiting before cloud scoring")
            return

        futures = [
            score_model_condition.spawn(
                model_label=label,
                model_name=spec["name"],
                families=selected_families,
                model_revision=spec["revision"],
                trust_remote_code=spec["trust"],
                dtype=dtype,
                answer_mode=answer_mode,
                n_shots=config[0],
                exemplar_set_id=config[1],
            )
            for label, spec, config in jobs
        ]

        payloads: list[dict] = []
        for (label, _, config), fut in zip(jobs, futures):
            print(f"[compare] waiting on {label}:{config[0]}shot:set{config[1]} ...")
            try:
                payloads.append(fut.get())
            except Exception as exc:
                print(f"[compare] {label}:{config[0]}shot:set{config[1]} FAILED: {exc}")

        outdir = Path(results_dir)
        for config in configs:
            sub_payloads = [
                payload
                for payload in payloads
                if int(payload.get("n_shots", -1)) == config[0]
                and int(payload.get("exemplar_set_id", -1)) == config[1]
            ]
            subdir = outdir / f"{config[0]}shot_set{config[1]}"
            write_results(subdir, sub_payloads)
            compute_correlations(subdir)
            plot_results(subdir)
            write_latex_summary(subdir)
            print(f"[compare] wrote condition results to {subdir}")
        print(f"[compare] wrote sweep results to {outdir}")
        return

    jobs = [(label, spec, family) for label, spec in selected for family in selected_families]
    total_tasks = sum(
        len(
            build_comparison_tasks(
                [family],
                answer_mode=answer_mode,
                n_shots=n_shots,
                exemplar_set_id=exemplar_set_id,
            )
        )
        for family in selected_families
    )
    print(f"[compare] models={ [label for label, _ in selected] }")
    print(
        f"[compare] families={selected_families} answer_mode={answer_mode} n_shots={n_shots} "
        f"exemplar_set={exemplar_set_id} "
        f"tasks_per_model={total_tasks} "
        f"jobs={len(jobs)} gpu={gpu} max_containers=100"
    )
    if dry_run:
        print("[compare] --dry-run set; exiting before cloud scoring")
        return

    futures = [
        score_model_family.spawn(
            model_label=label,
            model_name=spec["name"],
            family=family,
            model_revision=spec["revision"],
            trust_remote_code=spec["trust"],
            dtype=dtype,
            answer_mode=answer_mode,
            n_shots=n_shots,
            exemplar_set_id=exemplar_set_id,
        )
        for label, spec, family in jobs
    ]

    payloads: list[dict] = []
    for (label, _, family), fut in zip(jobs, futures):
        print(f"[compare] waiting on {label}:{family} ...")
        try:
            payloads.append(fut.get())
        except Exception as exc:
            print(f"[compare] {label}:{family} FAILED: {exc}")

    outdir = Path(results_dir)
    write_results(outdir, payloads)
    compute_correlations(outdir)
    plot_results(outdir)
    write_latex_summary(outdir)
    _log_payloads(
        payloads,
        experiment_name=experiment_name,
        run_name=run_name,
        outdir=outdir,
        gpu=gpu,
    )
    print(f"[compare] wrote results to {outdir}")
