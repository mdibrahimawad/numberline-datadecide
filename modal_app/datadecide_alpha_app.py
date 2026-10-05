"""
Part 2: number-frequency alpha of the DataDecide training data, sampled the
way training saw it (see docs/datadecide_sampling.md, src/datadecide_sampling.py).

    # validate on dolma1_7 only (convergence, split-half, exact training order)
    modal run modal_app/datadecide_alpha_app.py::validate --dry-run
    modal run modal_app/datadecide_alpha_app.py::validate

    # prepared, NOT yet run: all 25 recipes
    modal run modal_app/datadecide_alpha_app.py::sweep --dry-run
    modal run modal_app/datadecide_alpha_app.py::sweep --recipes c4,falcon --windows 1000

Windows of --window-tokens (default 262,144 tokens ~ 1 MB of text) are read
with HTTP range requests from allenai/DataDecide-data-recipes (raw uint16
memmaps), split on EOS 50279, decoded with allenai/DataDecide-<recipe>-1B's
tokenizer and counted with _count_text through _decode_and_count_native.
Window counts are cached on the `numberline-datadecide-alpha` Volume.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import modal

from modal_app.corpus_alpha_app import _hf_headers, _hf_url, _request_with_retry
from modal_app.corpus_alpha_full_app import _decode_and_count_native

APP_NAME = "numberline-datadecide-alpha"
CACHE = Path("/dd_cache/v1")
LOCAL_OUT = Path("results/corpus_alpha_datadecide")
MODEL_REVISION = "step69369-seed-default"
WINDOWS_PER_CALL = 20

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub>=0.28.0",
        "numpy>=2.0.0",
        "requests>=2.32.0",
        "tokenizers>=0.20.0",
        "orjson>=3.10.0",
    )
    .add_local_python_source("modal_app", "src", "utils")
    .add_local_dir(str(Path(__file__).resolve().parent.parent / "configs"), "/root/configs")
)
app = modal.App(APP_NAME, image=image)
cache = modal.Volume.from_name("numberline-datadecide-alpha", create_if_missing=True)
secret_name = os.environ.get("MODAL_HF_SECRET_NAME", "numberline-hf-token").strip()
secrets = [modal.Secret.from_name(secret_name)] if secret_name else []


def _cached(key: str, compute):
    path = CACHE / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
    cache.reload()
    if path.exists():
        return json.loads(path.read_text())
    payload = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    cache.commit()
    return payload


def _tokenizer(model_repo: str):
    """Recipe model's tokenizer.json (checkpoint revision, falling back to main)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    cached = getattr(_tokenizer, "cache", {})
    if model_repo not in cached:
        token = os.environ.get("HF_TOKEN")
        try:
            path = hf_hub_download(model_repo, "tokenizer.json", revision=MODEL_REVISION, token=token)
        except Exception:
            path = hf_hub_download(model_repo, "tokenizer.json", token=token)
        cached[model_repo] = SimpleNamespace(backend_tokenizer=Tokenizer.from_file(path))
        _tokenizer.cache = cached
    return cached[model_repo]


def _read_tokens(path: str, start: int, length: int):
    import numpy as np

    from src.datadecide_sampling import HF_DATA_REPO, TOKEN_DTYPE

    item = np.dtype(TOKEN_DTYPE).itemsize
    response = _request_with_retry(
        "get", _hf_url(HF_DATA_REPO, path),
        headers={**_hf_headers(), "Range": f"bytes={start * item}-{(start + length) * item - 1}"},
        timeout=300,
    )
    response.raise_for_status()
    return np.frombuffer(response.content, dtype=TOKEN_DTYPE)


def count_window(tokenizer, tokens, window: dict, eos: int | None = None) -> dict:
    """Counts for both edge modes of one window (pure; used by tests)."""
    from src.datadecide_sampling import EDGE_MODES, EOS_TOKEN_ID, split_documents

    eos = EOS_TOKEN_ID if eos is None else eos
    out = {"tokens": int(len(tokens))}
    for mode in EDGE_MODES:
        docs = split_documents(tokens, mode, window["at_file_start"], window["at_file_end"], eos=eos)
        counts: Counter[int] = Counter()
        decoded = _decode_and_count_native(tokenizer, docs, counts) if docs else 0
        out[mode] = {"docs": len(docs), "decoded_tokens": decoded,
                     "counts": {str(k): v for k, v in counts.items()}}
    return out


@app.function(cpu=2, memory=4096, timeout=2 * 60 * 60, max_containers=64, retries=2,
              volumes={"/dd_cache": cache}, secrets=secrets)
def count_windows(model_repo: str, windows: list[dict]) -> list[dict]:
    results = []
    for w in windows:
        key = f"window::{w['path']}::{w['start']}::{w['length']}::{model_repo}"

        def compute(w=w):
            tokens = _read_tokens(w["path"], w["start"], w["length"])
            return count_window(_tokenizer(model_repo), tokens, w)

        results.append({**w, **_cached(key, compute)})
    return results


@app.function(timeout=30 * 60, secrets=secrets)
def npy_sizes(paths: list[str]) -> dict[str, int]:
    """Token count per unique path from HF metadata (bytes / 2); missing paths -> -1."""
    from huggingface_hub import HfApi

    from src.datadecide_sampling import HF_DATA_REPO

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    unique = sorted(set(paths))
    sizes = {p: -1 for p in unique}
    for i in range(0, len(unique), 200):
        for info in api.get_paths_info(HF_DATA_REPO, unique[i:i + 200], repo_type="dataset"):
            if getattr(info, "size", None) is not None:
                sizes[info.path] = int(info.size) // 2
    return sizes


@app.function(cpu=4, memory=32768, timeout=2 * 60 * 60, volumes={"/dd_cache": cache})
def exact_order_windows(file_tokens: list[int], n_chunks: int, seed: int, order_seed: int) -> dict:
    """Reproduce the 1B training chunk order for data seed `order_seed`, sample
    n_chunks of the chunks used in training, and n_chunks uniformly from all chunks."""
    import numpy as np

    from src.datadecide_sampling import (
        TRAIN_INSTANCES_1B,
        chunk_offsets,
        chunks_to_windows,
        training_chunk_indices,
    )

    def compute():
        order = training_chunk_indices(file_tokens, seed=order_seed)
        rng = np.random.default_rng(seed)
        trained = np.sort(rng.choice(order, size=n_chunks, replace=False))
        total = int(chunk_offsets(file_tokens)[-1])
        uniform = np.sort(rng.choice(total, size=n_chunks, replace=False))
        return {
            "total_chunks": total,
            "train_instances": TRAIN_INSTANCES_1B,
            "first_indices": order[:5].tolist(),
            "exact_order": chunks_to_windows(trained, file_tokens),
            "uniform_chunks": chunks_to_windows(uniform, file_tokens),
        }

    key = (f"exact_order::{hashlib.sha1(json.dumps(file_tokens).encode()).hexdigest()}"
           f"::{n_chunks}::{seed}::{order_seed}")
    return _cached(key, compute)


# --------------------------------------------------------------------------- #
# local helpers
# --------------------------------------------------------------------------- #

def _recipe_tokens(recipe: dict) -> tuple[list[int], dict[str, int]]:
    sizes = npy_sizes.remote(recipe["paths"])
    missing = [p for p, n in sizes.items() if n < 0]
    if missing:
        raise SystemExit(f"{len(missing)} paths missing from the HF dataset, e.g. {missing[:3]}")
    return [sizes[p] for p in recipe["paths"]], sizes


def _mixture(recipe: dict, file_tokens: list[int], data_map_groups: dict[str, str]) -> dict:
    by_source: Counter = Counter()
    for p, t in zip(recipe["paths"], file_tokens):
        by_source[data_map_groups.get(p, p.split("/")[1])] += t
    total = sum(by_source.values())
    return {k: {"tokens": v, "share": v / total} for k, v in by_source.most_common()}


def _count(model_repo: str, windows: list[dict]) -> list[dict]:
    batches = [windows[i:i + WINDOWS_PER_CALL] for i in range(0, len(windows), WINDOWS_PER_CALL)]
    results = []
    for batch in count_windows.starmap([(model_repo, b) for b in batches], order_outputs=True):
        results.extend(batch)
    return results


def _vectors(results: list[dict], mode: str):
    import numpy as np

    from src.sampling_validation import counts_vector

    return np.stack([counts_vector(r[mode]["counts"]) for r in results])


def _write_counts_csv(path: Path, vec) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["number", "count"])
        w.writerows((n, int(round(vec[n]))) for n in range(len(vec)))


def _recipe_result(slug: str, recipe: dict, results: list[dict],
                   n_boot: int, out_dir: Path, file_tokens: list[int]) -> dict:
    from src.sampling_validation import bootstrap_alpha, describe

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"recipe": slug, "model_repo": recipe["model_repo"], "olmo_mix_name": recipe["olmo_mix_name"],
               "windows": len(results), "corpus_tokens": int(sum(file_tokens))}
    for mode in ("drop_partial_docs", "keep_partial"):
        mat = _vectors(results, mode)
        vec = mat.sum(axis=0)
        summary[mode] = {
            **describe(vec), **bootstrap_alpha(mat, n_boot=n_boot),
            "sampled_tokens": int(sum(r[mode]["decoded_tokens"] for r in results)),
            "documents": int(sum(r[mode]["docs"] for r in results)),
        }
        _write_counts_csv(out_dir / f"counts_0_to_10000_{mode}.csv", vec)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "windows.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("drop_partial_docs", "keep_partial")} for r in results]))
    return summary


ALPHA_COLUMNS = ("recipe", "alpha_ols", "r2", "alpha_mle", "ci95_low", "ci95_high",
                 "sampled_tokens", "integer_matches", "support")


def _summary_row(summary: dict, mode: str = "drop_partial_docs") -> dict:
    s = summary[mode]
    return {"recipe": summary["recipe"], "alpha_ols": s["alpha_ols"], "r2": s["r2"],
            "alpha_mle": s["alpha_mle"], "ci95_low": s["alpha_ols_ci95"][0],
            "ci95_high": s["alpha_ols_ci95"][1], "sampled_tokens": s["sampled_tokens"],
            "integer_matches": int(s["integer_matches"]), "support": s["support"]}


def _groups_by_path() -> dict[str, str]:
    from src.datadecide_sampling import load_data_map

    out = {}
    for r in load_data_map()["recipes"].values():
        for p in r["paths"]:
            out.setdefault(p, p.split("/")[1] + "/" + p.split("/")[2])
    return out


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def validate(
    recipe: str = "dolma1_7",
    windows: int = 1000,
    window_tokens: int = 1 << 18,
    n_boot: int = 500,
    exact_order: bool = True,
    exact_chunks: int = 1000,
    order_seeds: str = "2,6198",
    seed: int = 0,
    dry_run: bool = False,
    out_dir: str = str(LOCAL_OUT),
) -> None:

    from src.datadecide_sampling import convergence, load_data_map, plan_windows, split_half
    from src.sampling_validation import bootstrap_alpha, describe

    data_map = load_data_map()
    rec = data_map["recipes"][recipe]
    file_tokens, _ = _recipe_tokens(rec)
    total = sum(file_tokens)
    print(f"[validate:{recipe}] files={len(file_tokens)} tokens={total/1e9:.1f}B "
          f"chunks={sum(t // 2048 for t in file_tokens):,} windows={windows} x {window_tokens} tokens "
          f"(~{windows * window_tokens / 1e6:.0f}M tokens, {windows * window_tokens * 2 / 1e9:.2f} GB download)")
    out = Path(out_dir) / f"validation_{recipe}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mixture.json").write_text(json.dumps(_mixture(rec, file_tokens, _groups_by_path()), indent=2))
    if dry_run:
        print("[validate] --dry-run: stopping before counting")
        return

    plan = plan_windows(file_tokens, windows, window_tokens, seed)
    for w in plan:
        w["path"] = rec["paths"][w["file_index"]]
    results = _count(rec["model_repo"], plan)
    summary = _recipe_result(recipe, rec, results, n_boot, out, file_tokens)

    report = {"recipe": recipe, "window_tokens": window_tokens, "seed": seed, "summary": summary}
    for mode in ("drop_partial_docs", "keep_partial"):
        mat = _vectors(results, mode)
        report[f"convergence_{mode}"] = convergence(mat, [50, 100, 200, 500, 1000], n_boot, seed)
        report[f"split_half_{mode}"] = split_half(mat, n_splits=100, seed=seed, n_boot=n_boot)
    if exact_order:
        # The 1B "default seed" runs are named <mix>-1B-5xC-2; ladder.py appends
        # -<seed> only when seed != 6198, so the data seed was most likely 2.
        report["exact_order"] = {}
        for order_seed in [int(x) for x in order_seeds.split(",") if x.strip()]:
            eo = exact_order_windows.remote(file_tokens, exact_chunks, seed, order_seed)
            entry = {"total_chunks": eo["total_chunks"], "train_instances": eo["train_instances"],
                     "first_training_indices": eo["first_indices"]}
            for name in ("exact_order", "uniform_chunks"):
                wins = eo[name]
                for w in wins:
                    w["path"] = rec["paths"][w["file_index"]]
                res = _count(rec["model_repo"], wins)
                mat = _vectors(res, "keep_partial")
                entry[name] = {
                    **describe(mat.sum(axis=0)), **bootstrap_alpha(mat, n_boot=n_boot, seed=seed),
                    "chunks": len(wins),
                    "sampled_tokens": int(sum(r["keep_partial"]["decoded_tokens"] for r in res)),
                }
            report["exact_order"][f"data_seed_{order_seed}"] = entry
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    with open(out / "convergence.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["edge_mode", "windows", "alpha_ols", "ols_ci_low", "ols_ci_high",
                    "alpha_mle", "mle_ci_low", "mle_ci_high", "support", "integer_matches"])
        for mode in ("drop_partial_docs", "keep_partial"):
            for r in report[f"convergence_{mode}"]:
                w.writerow([mode, r["windows"], r["alpha_ols"], *r["alpha_ols_ci95"],
                            r["alpha_mle"], *r["alpha_mle_ci95"], r["support"], r["integer_matches"]])
    print(json.dumps({k: v for k, v in report.items() if k != "summary"}, indent=2)[:6000])
    print(f"[validate] wrote {out}")


@app.local_entrypoint()
def sweep(
    recipes: str = "",
    windows: int = 1000,
    window_tokens: int = 1 << 18,
    n_boot: int = 500,
    seed: int = 0,
    dry_run: bool = False,
    out_dir: str = str(LOCAL_OUT),
) -> None:
    from src.datadecide_sampling import load_data_map, plan_windows

    data_map = load_data_map()
    names = [r.strip() for r in recipes.split(",") if r.strip()] or list(data_map["recipes"])
    unknown = [r for r in names if r not in data_map["recipes"]]
    if unknown:
        raise SystemExit(f"unknown recipes: {unknown}")
    groups = _groups_by_path()
    root = Path(out_dir)
    plans = {}
    for slug in names:
        rec = data_map["recipes"][slug]
        file_tokens, _ = _recipe_tokens(rec)
        (root / slug).mkdir(parents=True, exist_ok=True)
        (root / slug / "mixture.json").write_text(json.dumps(_mixture(rec, file_tokens, groups), indent=2))
        print(f"[sweep] {slug}: files={len(file_tokens)} tokens={sum(file_tokens)/1e9:.1f}B "
              f"download={windows * window_tokens * 2 / 1e9:.2f} GB")
        plan = plan_windows(file_tokens, windows, window_tokens, seed)
        for w in plan:
            w["path"] = rec["paths"][w["file_index"]]
        plans[slug] = (rec, plan, file_tokens)
    if dry_run:
        print("[sweep] --dry-run: all recipes verified on the Hub; stopping before counting")
        return
    for slug, (rec, plan, file_tokens) in plans.items():
        results = _count(rec["model_repo"], plan)
        summary = _recipe_result(slug, rec, results, n_boot, root / slug, file_tokens)
        s = summary["drop_partial_docs"]
        print(f"[sweep] {slug}: alpha_ols={s['alpha_ols']:.4f} alpha_mle={s['alpha_mle']:.4f}")
    rows = []
    for slug in data_map["recipes"]:
        path = root / slug / "summary.json"
        if path.exists():
            rows.append(_summary_row(json.loads(path.read_text())))
    with open(root / "alpha_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ALPHA_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] wrote {root / 'alpha_summary.csv'} ({len(rows)} recipes)")
