from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "numberline-paper-magnitude"
DEFAULT_GPU = os.environ.get("MODAL_PAPER_MAGNITUDE_GPU", "H100")

TARGET_MODEL_SPECS = {
    "Falcon-RW-1B": {
        "name": "tiiuae/falcon-rw-1b",
        "revision": None,
        "trust": False,
        "arch": "causal",
    },
    "Falcon-RW-7B": {
        "name": "tiiuae/falcon-rw-7b",
        "revision": None,
        "trust": False,
        "arch": "causal",
    },
    "RedPajama-3B": {
        "name": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
        "revision": None,
        "trust": False,
        "arch": "causal",
    },
    "RedPajama-7B": {
        "name": "togethercomputer/RedPajama-INCITE-7B-Base",
        "revision": None,
        "trust": False,
        "arch": "causal",
    },
    "OLMo-7B-2T": {
        "name": "allenai/OLMo-7B",
        "revision": "step452000-tokens2000B",
        "trust": True,
        "arch": "causal",
    },
    "OLMo-7B-Twin-2T": {
        "name": "allenai/OLMo-7B-Twin-2T",
        "revision": None,
        "trust": True,
        "arch": "causal",
    },
    "StarCoderBase-1B": {
        "name": "bigcode/starcoderbase-1b",
        "revision": None,
        "trust": False,
        "arch": "causal",
    },
    "StarCoderBase-3B": {
        "name": "bigcode/starcoderbase-3b",
        "revision": None,
        "trust": False,
        "arch": "causal",
    },
    "StarCoderBase-7B": {
        "name": "bigcode/starcoderbase-7b",
        "revision": None,
        "trust": False,
        "arch": "causal",
    },
}

PAPER_MODEL_SPECS = {
    "BERT-base": {"name": "bert-base-uncased", "revision": None, "trust": False, "arch": "encoder"},
    "RoBERTa-base": {"name": "roberta-base", "revision": None, "trust": False, "arch": "encoder"},
    "XLNet-base": {"name": "xlnet-base-cased", "revision": None, "trust": False, "arch": "encoder"},
    "GPT2-base": {"name": "openai-community/gpt2", "revision": None, "trust": False, "arch": "causal"},
    "T5-base": {"name": "google-t5/t5-base", "revision": None, "trust": False, "arch": "seq2seq"},
    "BART-base": {"name": "facebook/bart-base", "revision": None, "trust": False, "arch": "seq2seq"},
}

MODEL_SETS = {
    "target": TARGET_MODEL_SPECS,
    "paper": PAPER_MODEL_SPECS,
    "all": {**TARGET_MODEL_SPECS, **PAPER_MODEL_SPECS},
}


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.44.0,<5",
        "accelerate>=0.33.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.11.0",
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
    kwargs: dict = {
        "gpu": gpu,
        "timeout": timeout,
        "max_containers": 100,
    }
    if _HF_SECRETS:
        kwargs["secrets"] = _HF_SECRETS
    return kwargs


@app.function(**_gpu_kwargs(DEFAULT_GPU, timeout=2 * 60 * 60))
def score_model_format(
    model_label: str,
    model_name: str,
    input_format: str,
    *,
    model_revision: str | None = None,
    trust_remote_code: bool = False,
    arch: str = "causal",
    dtype: str = "float16",
) -> dict:
    import os

    import numpy as np
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    from src.paper_magnitude import (
        NUMBER_FORMATS,
        NUMBER_VALUES,
        _mds_1d,
        _minmax,
        _pearson,
        analyze_similarity_matrix,
    )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    print(
        f"[paper:{model_label}:{input_format}] loading {model_name} "
        f"revision={model_revision or 'main'} arch={arch} gpu={DEFAULT_GPU}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    model_kwargs = {
        "revision": model_revision,
        "trust_remote_code": trust_remote_code,
        "token": hf_token,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if arch == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to("cuda")
    elif arch == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs).to("cuda")
    elif arch == "encoder":
        model = AutoModel.from_pretrained(model_name, **model_kwargs).to("cuda")
    else:
        raise ValueError(f"unknown arch={arch}")
    model.eval()

    labels = NUMBER_FORMATS[input_format]
    reps_by_layer: dict[int, list[np.ndarray]] = {}
    tokenization = []

    def _forward(inputs):
        model_inputs = {
            key: value.to("cuda")
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask"}
        }
        with torch.no_grad():
            if arch == "seq2seq":
                encoder = model.get_encoder()
                return encoder(**model_inputs, output_hidden_states=True, return_dict=True).hidden_states
            try:
                return model(
                    **model_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                ).hidden_states
            except TypeError:
                return model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states

    for number, text in zip(NUMBER_VALUES, labels):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        hidden_states = _forward(inputs)
        attention = inputs.get("attention_mask")
        special = inputs.get("special_tokens_mask")
        if attention is None:
            content_mask = torch.ones_like(inputs["input_ids"], dtype=torch.bool)
        else:
            content_mask = attention.to(dtype=torch.bool)
        if special is not None:
            content_mask = content_mask & (special == 0)
        positions = torch.nonzero(content_mask[0], as_tuple=False).reshape(-1)
        if int(positions.numel()) == 0:
            positions = torch.nonzero(inputs["attention_mask"][0].to(dtype=torch.bool), as_tuple=False).reshape(-1)

        ids = inputs["input_ids"][0].tolist()
        tokenization.append(
            {
                "number": number,
                "text": text,
                "input_ids": ids,
                "tokens": tokenizer.convert_ids_to_tokens(ids),
                "content_positions": [int(x) for x in positions.tolist()],
            }
        )

        for layer in range(1, len(hidden_states)):
            vec = hidden_states[layer][0, positions.to("cuda"), :].mean(dim=0)
            reps_by_layer.setdefault(layer, []).append(vec.detach().float().cpu().numpy())

    rows = []
    dissimilarities = []
    for layer in sorted(reps_by_layer):
        matrix = np.stack(reps_by_layer[layer], axis=0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normalized = matrix / norms
        similarity = normalized @ normalized.T
        dissimilarity = 1.0 - similarity
        np.fill_diagonal(dissimilarity, 0.0)
        dissimilarity = (dissimilarity + dissimilarity.T) / 2.0
        dissimilarities.append(dissimilarity)

        metrics = analyze_similarity_matrix(similarity, dissimilarity)
        row = {
            "model_label": model_label,
            "model_name": model_name,
            "model_revision": model_revision or "",
            "input_format": input_format,
            "layer": layer,
        }
        row.update(metrics)
        rows.append(row)

    avg_dissimilarity = np.mean(np.stack(dissimilarities, axis=0), axis=0)
    avg_coords = _mds_1d(avg_dissimilarity)
    log_values = [float(np.log10(x)) for x in NUMBER_VALUES]
    avg_corr = _pearson(avg_coords, log_values)
    coords_norm = _minmax(avg_coords).tolist()
    log_norm = _minmax(log_values).tolist()
    residuals = [
        {
            "number": number,
            "coord_norm": float(coord),
            "log_norm": float(log_coord),
            "abs_residual": float(abs(coord - log_coord)),
        }
        for number, coord, log_coord in zip(NUMBER_VALUES, coords_norm, log_norm)
    ]

    print(
        f"[paper:{model_label}:{input_format}] layers={len(rows)} "
        f"distance={np.nanmean([r['distance_r2'] for r in rows]):.3f} "
        f"size={np.nanmean([r['size_r2'] for r in rows]):.3f} "
        f"ratio={np.nanmean([r['ratio_r2'] for r in rows]):.3f} "
        f"mds={np.nanmean([r['mds_log_corr'] for r in rows]):.3f}"
    )
    return {
        "model_label": model_label,
        "model_name": model_name,
        "model_revision": model_revision or "",
        "trust_remote_code": trust_remote_code,
        "arch": arch,
        "input_format": input_format,
        "rows": rows,
        "avg_mds_coords": avg_coords,
        "avg_mds_log_corr": avg_corr,
        "avg_mds_residuals": residuals,
        "tokenization": tokenization,
    }


def _selected_specs(model_set: str, models: str) -> list[tuple[str, dict]]:
    model_set = model_set.strip().lower()
    if model_set not in MODEL_SETS:
        raise SystemExit(f"unknown model_set={model_set}; choose one of {sorted(MODEL_SETS)}")
    specs = MODEL_SETS[model_set]
    if models.strip().lower() in {"", "all", "default"}:
        labels = list(specs)
    else:
        labels = [part.strip() for part in models.split(",") if part.strip()]
    bad = sorted(set(labels) - set(specs))
    if bad:
        raise SystemExit(f"unknown model label(s) for model_set={model_set}: {bad}")
    return [(label, specs[label]) for label in labels]


def _selected_formats(formats: str) -> list[str]:
    from src.paper_magnitude import NUMBER_FORMATS

    if formats.strip().lower() in {"", "all", "default"}:
        return list(NUMBER_FORMATS)
    selected = [part.strip().lower() for part in formats.split(",") if part.strip()]
    bad = sorted(set(selected) - set(NUMBER_FORMATS))
    if bad:
        raise SystemExit(f"unknown input format(s): {bad}")
    return selected


def _log_payloads(
    payloads: list[dict],
    *,
    experiment_name: str,
    run_name: str,
    outdir: Path,
    gpu: str,
    model_set: str,
) -> None:
    import json

    try:
        import mlflow
    except ModuleNotFoundError:
        print("[paper] local mlflow package not found; skipping MLflow logging")
        return

    from utils.mlflow_utils import log_metric, log_params, setup_tracking, start_run

    correlations = {}
    corr_path = outdir / "paper_magnitude_beta_correlations.json"
    if corr_path.exists():
        with open(corr_path) as fh:
            correlations = json.load(fh)

    setup_tracking(None, experiment_name)
    with start_run(run_name=run_name, tags={"stage": "paper_magnitude_replication", "backend": "modal"}):
        log_params(
            {
                "model_set": model_set,
                "n_payloads": len(payloads),
                "n_models": len({payload["model_label"] for payload in payloads}),
                "n_formats": len({payload["input_format"] for payload in payloads}),
                "numbers": "1-9",
                "gpu": gpu,
                "max_containers": 100,
                "shard": "model_x_input_format",
            }
        )
        for key, value in correlations.items():
            log_metric(key, float(value))
        for artifact in outdir.iterdir():
            if artifact.is_file():
                mlflow.log_artifact(str(artifact))
        active = mlflow.active_run()
        print(f"[paper] MLflow run_id={active.info.run_id}")


@app.local_entrypoint()
def main(
    model_set: str = "target",
    models: str = "all",
    formats: str = "all",
    dtype: str = "float16",
    gpu: str = DEFAULT_GPU,
    experiment_name: str = "numberline_geometry_expv1",
    run_name: str = "paper_magnitude_replication_target_1to9",
    results_dir: str = "results/paper_magnitude/target_1to9",
    dry_run: bool = False,
) -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.paper_magnitude import compute_correlations, plot_results, write_latex_summary, write_results

    selected = _selected_specs(model_set, models)
    input_formats = _selected_formats(formats)
    jobs = [
        (label, spec, input_format)
        for label, spec in selected
        for input_format in input_formats
    ]
    print(f"[paper] model_set={model_set} models={ [label for label, _ in selected] }")
    print(f"[paper] formats={input_formats} jobs={len(jobs)} gpu={gpu} max_containers=100")
    if dry_run:
        print("[paper] --dry-run set; exiting before cloud scoring")
        return

    futures = [
        score_model_format.spawn(
            model_label=label,
            model_name=spec["name"],
            input_format=input_format,
            model_revision=spec["revision"],
            trust_remote_code=spec["trust"],
            arch=spec["arch"],
            dtype=dtype,
        )
        for label, spec, input_format in jobs
    ]

    payloads: list[dict] = []
    for (label, _, input_format), fut in zip(jobs, futures):
        print(f"[paper] waiting on {label}:{input_format} ...")
        try:
            payloads.append(fut.get())
        except Exception as exc:
            print(f"[paper] {label}:{input_format} FAILED: {exc}")

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
        model_set=model_set,
    )
    print(f"[paper] wrote results to {outdir}")
