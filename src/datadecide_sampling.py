"""DataDecide training-data sampler for number-frequency alpha (Part 2).

Facts reproduced from the training code are documented, with file paths and
line numbers, in docs/datadecide_sampling.md. In short: each recipe is a list
of raw uint16 token files (OLMo memmap .npy, no header) in the HF dataset
allenai/DataDecide-data-recipes; training concatenates them in list order,
cuts each file into 2048-token chunks, shuffles the global chunk indices once
with np.random.PCG64(6198) and consumes the first 69369 x 704 chunks
(~100B tokens) for the 1B models. Documents are separated by EOS 50279.

The "olmo_npy" sampler draws fixed-size token windows with every token of the
recipe's concatenated stream (duplicated files included) equally likely to be
covered -- the training distribution, since the shuffled first 100B tokens are
a uniform random subset of chunks. Windows are split on EOS and decoded with
the recipe model's tokenizer; counting is `_count_text` via
`_decode_and_count_native` (modal_app/corpus_alpha_full_app.py).
"""

from __future__ import annotations

import argparse
import ast
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np

HF_DATA_REPO = "allenai/DataDecide-data-recipes"
TOKEN_DTYPE = np.uint16
EOS_TOKEN_ID = 50279
PAD_TOKEN_ID = 1
SEQUENCE_LENGTH = 2048
DATA_SEED = 6198
GLOBAL_BATCH_1B = 704
FINAL_STEP_1B = 69369
TRAIN_INSTANCES_1B = FINAL_STEP_1B * GLOBAL_BATCH_1B
DEFAULT_WINDOW_TOKENS = 1 << 18  # 262,144 tokens ~ 1 MB of text (0.5 MiB of uint16)
EDGE_TRIM_TOKENS = 8
EDGE_MODES = ("drop_partial_docs", "keep_partial")
MAP_PATH = Path(__file__).resolve().parent.parent / "configs" / "datadecide_data_map.json"


# --------------------------------------------------------------------------- #
# recipe -> files map (built from the OLMo DataDecide branch + DataDecide repo)
# --------------------------------------------------------------------------- #

def _recipe_display_names(datadecide_repo: Path) -> dict[str, str]:
    """`recipe_display_name` from DataDecide/release/upload_checkpoints.py, read with ast."""
    tree = ast.parse((datadecide_repo / "release" / "upload_checkpoints.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "recipe_display_name" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("recipe_display_name not found")


def build_data_map(olmo_repo: Path, datadecide_repo: Path, slugs: list[str]) -> dict:
    """Execute OLMo's named_data_mixes.py (DataDecide branch) exactly as training
    imported it -- its 25p/50p/75p subsets come from an in-place, order-dependent
    random.shuffle(seed 62540), so only executing the module reproduces them."""
    ns = runpy.run_path(str(olmo_repo / "olmo" / "data" / "named_data_mixes.py"))
    data_paths = ns["DATA_PATHS"]
    groups: dict[str, list[str]] = {}
    for table in ("DATA_SOURCES", "EXTRA_DATA_SOURCES", "DOLMA_1_6_TO_1_7_DATA_SOURCES"):
        for name, paths in ns[table].items():
            for p in paths:
                groups.setdefault(p, [])
                if name not in groups[p]:
                    groups[p].append(name)
    display = _recipe_display_names(datadecide_repo)
    mix_for_slug = {slug: mix for mix, slug in display.items()}
    mix_names = [
        json.loads(line)["name"]
        for line in (datadecide_repo / "pretraining" / "eval-for-consistent-ranking-mix-names.jsonl")
        .read_text().splitlines() if line.strip()
    ]
    recipes = {}
    for slug in slugs:
        mix = mix_for_slug[slug]
        assert mix in mix_names, (slug, mix)
        paths = list(data_paths[mix])
        counts = Counter(paths)
        sources = Counter(groups.get(p, ["?"])[0] for p in paths)
        recipes[slug] = {
            "olmo_mix_name": mix,
            "model_repo": f"allenai/DataDecide-{slug}-1B",
            "n_paths": len(paths),
            "n_unique_paths": len(counts),
            "repeated_paths": {p: k for p, k in counts.items() if k > 1},
            "files_per_source": dict(sorted(sources.items())),
            "paths": paths,
        }
    return {
        "description": "DataDecide recipe -> tokenized .npy files, in training order "
                       "(duplicates = upsampled files). See docs/datadecide_sampling.md.",
        "hf_dataset": HF_DATA_REPO,
        "hf_path_rule": "HF dataset path == S3 key under s3://ai2-llm/ (upload_to_hf.py)",
        "token_dtype": "uint16",
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "sequence_length": SEQUENCE_LENGTH,
        "data_seed": DATA_SEED,
        "global_batch_1b": GLOBAL_BATCH_1B,
        "final_step_1b": FINAL_STEP_1B,
        "train_instances_1b": TRAIN_INSTANCES_1B,
        "recipes": recipes,
    }


def load_data_map(path: Path = MAP_PATH) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# window plan: uniform coverage of the concatenated token stream
# --------------------------------------------------------------------------- #

def plan_windows(file_tokens: list[int], n_windows: int, window_tokens: int, seed: int) -> list[dict]:
    """Windows whose coverage is exactly uniform over all tokens.

    A file f is chosen with probability (len_f + W - 1) / sum, the start s is
    uniform on [-(W-1), len_f - 1] and the window is [max(0,s), min(len_f, s+W)):
    every token is covered by exactly W of the sum_f (len_f + W - 1) equally
    likely (file, start) pairs. Windows never cross files (file ends are
    document ends).
    """
    rng = np.random.default_rng(seed)
    lens = np.asarray(file_tokens, dtype=np.int64)
    weight = (lens + window_tokens - 1).astype(float)
    weight[lens == 0] = 0.0
    files = rng.choice(len(lens), size=n_windows, p=weight / weight.sum())
    out = []
    for f in files:
        s = int(rng.integers(-(window_tokens - 1), lens[f]))
        lo, hi = max(0, s), min(int(lens[f]), s + window_tokens)
        out.append({"file_index": int(f), "start": lo, "length": hi - lo,
                    "at_file_start": lo == 0, "at_file_end": hi == int(lens[f])})
    return out


def split_documents(tokens: np.ndarray, mode: str, at_file_start: bool, at_file_end: bool,
                    eos: int = EOS_TOKEN_ID, edge_trim: int = EDGE_TRIM_TOKENS) -> list[list[int]]:
    """Split a token window on EOS into documents.

    drop_partial_docs: keep only documents fully inside the window (a window
      edge at a file boundary counts as a document boundary).
    keep_partial: keep partial edge documents too, minus `edge_trim` tokens at
      a cut edge so a number split by the cut is not counted as a smaller one.
    """
    tokens = np.asarray(tokens)
    eos_at = np.flatnonzero(tokens == eos)
    bounds = [-1, *eos_at.tolist(), len(tokens)]
    docs = []
    last = len(bounds) - 2
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i] + 1, bounds[i + 1]
        first, final = i == 0, i == last
        cut_left = first and not at_file_start
        cut_right = final and not at_file_end
        if mode == "drop_partial_docs":
            if cut_left or cut_right:
                continue
        elif mode == "keep_partial":
            if cut_left:
                lo += edge_trim
            if cut_right:
                hi -= edge_trim
        else:
            raise ValueError(f"unknown edge mode {mode!r}")
        if hi > lo:
            docs.append(tokens[lo:hi].tolist())
    return docs


# --------------------------------------------------------------------------- #
# exact training order (optional)
# --------------------------------------------------------------------------- #

def chunk_offsets(file_tokens: list[int], chunk: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Global chunk index offsets per file (MemMapDataset.offsets: floor(len/chunk) each)."""
    return np.concatenate([[0], np.cumsum([t // chunk for t in file_tokens])]).astype(np.int64)


def training_chunk_indices(file_tokens: list[int], n_instances: int = TRAIN_INSTANCES_1B,
                           seed: int = DATA_SEED, chunk: int = SEQUENCE_LENGTH) -> np.ndarray:
    """First `n_instances` global chunk indices in training order
    (IterableDataset._build_global_indices; a new epoch reshuffles with seed+1)."""
    n = int(chunk_offsets(file_tokens, chunk)[-1])
    assert n < np.iinfo(np.uint32).max
    parts, need, epoch = [], n_instances, 0
    while need > 0:
        indices = np.arange(n, dtype=np.uint32)
        np.random.Generator(np.random.PCG64(seed=seed + epoch)).shuffle(indices)
        parts.append(indices[:need].copy())
        need -= len(parts[-1])
        epoch += 1
        del indices
    return np.concatenate(parts)


def chunks_to_windows(chunk_ids: np.ndarray, file_tokens: list[int],
                      chunk: int = SEQUENCE_LENGTH) -> list[dict]:
    offsets = chunk_offsets(file_tokens, chunk)
    files = np.searchsorted(offsets, chunk_ids, side="right") - 1
    out = []
    for cid, f in zip(chunk_ids.tolist(), files.tolist()):
        start = (cid - int(offsets[f])) * chunk
        out.append({"file_index": int(f), "start": start, "length": chunk,
                    "at_file_start": start == 0, "at_file_end": start + chunk == file_tokens[f]})
    return out


# --------------------------------------------------------------------------- #
# exact 100B training samples for several seeds, in one pass over the recipe
# --------------------------------------------------------------------------- #

def membership_codes(file_tokens: list[int], seeds: list[int],
                     n_instances: int = TRAIN_INSTANCES_1B,
                     chunk: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, int]:
    """One code per global chunk: code = sum_i m_i * base**i, where m_i is how
    many times seed i's training run used that chunk (0, 1, or more if the
    recipe is smaller than the token budget and training wrapped epochs)."""
    n = int(chunk_offsets(file_tokens, chunk)[-1])
    mult = []
    for seed in seeds:
        order = training_chunk_indices(file_tokens, n_instances, seed, chunk)
        m = np.zeros(n, dtype=np.uint8)
        np.add.at(m, order, 1)
        mult.append(m)
        del order
    base = int(max(int(m.max()) for m in mult)) + 1
    if base ** len(seeds) > np.iinfo(np.uint32).max:
        raise ValueError("too many seeds / epochs to encode")
    codes = np.zeros(n, dtype=np.uint32)
    for i, m in enumerate(mult):
        codes += m.astype(np.uint32) * np.uint32(base ** i)
    return codes, base


def decode_code(code: int, base: int, n_seeds: int) -> list[int]:
    return [(code // base ** i) % base for i in range(n_seeds)]


def count_chunks_by_code(decode_and_count, tokens: np.ndarray, codes: np.ndarray,
                         chunk: int = SEQUENCE_LENGTH, eos: int = EOS_TOKEN_ID,
                         batch_docs: int = 256) -> dict[int, dict]:
    """Count numbers in consecutive `chunk`-token training sequences, grouped by
    membership code. Each sequence is split on EOS (numbers never span
    documents) and counted whole, exactly the text the model saw in it.
    `decode_and_count(docs, counter) -> decoded tokens` wraps
    _decode_and_count_native with the recipe tokenizer."""
    groups: dict[int, dict] = {}
    pending: dict[int, list] = {}

    def flush(code):
        docs = pending.pop(code, [])
        if docs:
            groups[code]["decoded_tokens"] += decode_and_count(docs, groups[code]["counts"])

    for i, code in enumerate(np.asarray(codes).tolist()):
        seq = tokens[i * chunk:(i + 1) * chunk]
        g = groups.setdefault(code, {"counts": Counter(), "chunks": 0, "decoded_tokens": 0})
        g["chunks"] += 1
        docs = pending.setdefault(code, [])
        docs.extend(split_documents(seq, "keep_partial", True, True, eos=eos))
        if len(docs) >= batch_docs:
            flush(code)
    for code in list(pending):
        flush(code)
    return groups


def combine_codes(groups: dict[int, dict], base: int, n_seeds: int, max_n: int) -> dict:
    """Full-recipe vector (every whole chunk once) and one vector per seed
    (chunks weighted by how often that seed's run used them)."""
    full = np.zeros(max_n + 1)
    per_seed = np.zeros((n_seeds, max_n + 1))
    tokens = np.zeros(n_seeds)
    full_tokens = 0
    for code, g in groups.items():
        vec = np.zeros(max_n + 1)
        for k, v in g["counts"].items():
            if 0 <= int(k) <= max_n:
                vec[int(k)] += v
        full += vec
        full_tokens += g["chunks"] * SEQUENCE_LENGTH
        for i, m in enumerate(decode_code(int(code), base, n_seeds)):
            if m:
                per_seed[i] += m * vec
                tokens[i] += m * g["chunks"] * SEQUENCE_LENGTH
    return {"full": full, "full_tokens": full_tokens, "per_seed": per_seed, "seed_tokens": tokens}


# --------------------------------------------------------------------------- #
# analysis on per-window count vectors
# --------------------------------------------------------------------------- #

def convergence(window_vectors: np.ndarray, ns: list[int], n_boot: int, seed: int = 0) -> list[dict]:
    from src.sampling_validation import bootstrap_alpha, describe

    rows = []
    for n in ns:
        if n > len(window_vectors):
            continue
        mat = window_vectors[:n]
        rows.append({"windows": n, **describe(mat.sum(axis=0)),
                     **bootstrap_alpha(mat, n_boot=n_boot, seed=seed)})
    return rows


def split_half(window_vectors: np.ndarray, n_splits: int = 100, seed: int = 0,
               n_boot: int = 0) -> dict:
    """Alpha on two disjoint random halves of the windows, repeated over random splits."""
    from src.sampling_validation import bootstrap_alpha, describe

    rng = np.random.default_rng(seed)
    n = len(window_vectors)
    diffs = {"ols": [], "mle": []}
    first = None
    for split in range(n_splits):
        perm = rng.permutation(n)
        a, b = window_vectors[perm[: n // 2]], window_vectors[perm[n // 2:]]
        da, db = describe(a.sum(axis=0)), describe(b.sum(axis=0))
        for fit in ("ols", "mle"):
            diffs[fit].append(da[f"alpha_{fit}"] - db[f"alpha_{fit}"])
        if split == 0:
            first = {"half_a": da, "half_b": db}
            if n_boot:
                first["half_a"].update(bootstrap_alpha(a, n_boot=n_boot, seed=seed))
                first["half_b"].update(bootstrap_alpha(b, n_boot=n_boot, seed=seed + 1))
    out = {"n_splits": n_splits, "windows_per_half": n // 2, "first_split": first}
    for fit, values in diffs.items():
        v = np.abs(np.asarray(values))
        out[f"abs_diff_{fit}_mean"] = float(v.mean())
        out[f"abs_diff_{fit}_p95"] = float(np.percentile(v, 95))
        out[f"abs_diff_{fit}_max"] = float(v.max())
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build configs/datadecide_data_map.json")
    parser.add_argument("--olmo-repo", required=True, help="clone of allenai/OLMo at branch DataDecide")
    parser.add_argument("--datadecide-repo", required=True, help="clone of allenai/DataDecide")
    parser.add_argument("--out", default=str(MAP_PATH))
    args = parser.parse_args()
    models = json.loads((Path(__file__).resolve().parent.parent / "configs" / "datadecide_models.json").read_text())
    data_map = build_data_map(Path(args.olmo_repo), Path(args.datadecide_repo), models["recipes"])
    Path(args.out).write_text(json.dumps(data_map, indent=1) + "\n")
    for slug, r in data_map["recipes"].items():
        print(f"{slug:35s} {r['olmo_mix_name']:58s} files={r['n_paths']:5d} repeated={len(r['repeated_paths'])}")
