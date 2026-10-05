"""Does a random sample of a corpus give the same number-frequency alpha as the full pass?

Counting is NOT reimplemented here: counts come from `_count_text` (regex
\\b\\d+\\b, leading zeros stripped, 0..MAX_N) and alpha_OLS from `_fit_alpha`,
both in modal_app/corpus_alpha_app.py (corpus_alpha_full_app.py imports the
very same functions). This module adds:

* alpha_MLE: discrete power law p(N) = N^a / sum_{M=1..MAX_N} M^a fitted by
  maximum likelihood on all counts N=1..MAX_N (zeros included);
* sampling plans over "units" (parquet row groups / row slices, or compressed
  byte ranges of .jsonl.gz files) for two schemes:
    A  uniform_files      pick a file uniformly, then a random unit in it
                          (unweighted -- the naive scheme);
    B  size_proportional  pick units with probability proportional to their
                          size over all files, with replacement, and weight
                          each draw by 1/size (Hansen-Hurwitz), so every byte
                          of the corpus has the same expected weight;
  plans are nested: the 10M-token sample of a replicate is a prefix of its
  30M sample, etc.;
* a bootstrap over drawn units and the summary table / plot.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from modal_app.corpus_alpha_app import MAX_N, _fit_alpha

BYTES_PER_TOKEN = 4  # tokens are approximated as text bytes / 4
SIZE_LABELS = {"10M": 10e6, "30M": 30e6, "100M": 100e6, "300M": 300e6, "1B": 1e9}
SCHEMES = ("uniform_files", "size_proportional")

_LOG_N = np.log(np.arange(1, MAX_N + 1, dtype=float))


# --------------------------------------------------------------------------- #
# count vectors and alpha fits
# --------------------------------------------------------------------------- #

def counts_vector(counts) -> np.ndarray:
    """Counter / {str|int: count} / sequence -> float vector indexed 0..MAX_N."""
    if isinstance(counts, np.ndarray):
        return counts.astype(float)
    vec = np.zeros(MAX_N + 1, dtype=float)
    for key, value in counts.items():
        n = int(key)
        if 0 <= n <= MAX_N:
            vec[n] += float(value)
    return vec


def read_counts_csv(path: Path) -> np.ndarray:
    vec = np.zeros(MAX_N + 1, dtype=float)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            vec[int(row["number"])] = float(row["count"])
    return vec


def fit_alpha_ols(vec: np.ndarray) -> tuple[float, float]:
    """The paper's fit, via the original `_fit_alpha` (log count vs log N, nonzero N=1..MAX_N)."""
    nonzero = {n: vec[n] for n in range(1, MAX_N + 1) if vec[n] > 0}
    if len(nonzero) < 2:
        return float("nan"), float("nan")
    return _fit_alpha(Counter(nonzero))


def fit_alpha_mle(vec: np.ndarray) -> tuple[float, float]:
    """MLE of a in p(N) = N^a / sum_M M^a over N=1..MAX_N; returns (a, Fisher s.e.).

    The log-likelihood is concave in a, so Newton from a=-1 converges quickly.
    The s.e. treats every integer occurrence as i.i.d. and is far too small
    for corpus data; use the bootstrap CI instead.
    """
    c = np.asarray(vec, dtype=float)[1:]
    total = c.sum()
    if total <= 0:
        return float("nan"), float("nan")
    s = float(np.dot(c, _LOG_N)) / total  # mean log N in the data
    a = -1.0
    for _ in range(100):
        logw = a * _LOG_N
        w = np.exp(logw - logw.max())
        p = w / w.sum()
        mean = float(np.dot(p, _LOG_N))
        var = float(np.dot(p, (_LOG_N - mean) ** 2))
        step = (s - mean) / var
        a_new = float(np.clip(a + step, a - 2.0, a + 2.0))
        if abs(a_new - a) < 1e-12:
            a = a_new
            break
        a = a_new
    return a, 1.0 / math.sqrt(total * var)


def support(vec: np.ndarray) -> float:
    return float(np.mean(np.asarray(vec)[1:] > 0))


def logcount_correlation(sample: np.ndarray, full: np.ndarray) -> float:
    """Pearson r of log counts over N=1..MAX_N where both vectors are nonzero."""
    s, f = np.asarray(sample)[1:], np.asarray(full)[1:]
    mask = (s > 0) & (f > 0)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(np.log(s[mask]), np.log(f[mask]))[0, 1])


def describe(vec: np.ndarray) -> dict:
    alpha_ols, r2 = fit_alpha_ols(vec)
    alpha_mle, mle_se = fit_alpha_mle(vec)
    return {
        "alpha_ols": alpha_ols,
        "r2": r2,
        "alpha_mle": alpha_mle,
        "alpha_mle_fisher_se": mle_se,
        "integer_matches": float(np.asarray(vec).sum()),
        "support": support(vec),
    }


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #

def bootstrap_alpha(
    unit_vectors: np.ndarray, n_boot: int = 500, seed: int = 0
) -> dict:
    """Resample drawn units with replacement; percentile 95% CIs for both alphas.

    `unit_vectors` is (n_draws, MAX_N+1), already multiplied by each draw's weight.
    """
    n = unit_vectors.shape[0]
    rng = np.random.default_rng(seed)
    ols, mle = [], []
    mat = unit_vectors.astype(np.float32)
    for start in range(0, n_boot, 50):
        b = min(50, n_boot - start)
        weights = rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float32)
        for vec in weights @ mat:
            ols.append(fit_alpha_ols(vec)[0])
            mle.append(fit_alpha_mle(vec)[0])
    ols_a, mle_a = np.asarray(ols), np.asarray(mle)

    def ci(values):
        values = values[np.isfinite(values)]
        if not len(values):
            return [float("nan"), float("nan")]
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {
        "n_boot": n_boot,
        "alpha_ols_ci95": ci(ols_a),
        "alpha_mle_ci95": ci(mle_a),
        "alpha_ols_boot_sd": float(np.nanstd(ols_a)),
        "alpha_mle_boot_sd": float(np.nanstd(mle_a)),
    }


# --------------------------------------------------------------------------- #
# units and sampling plans
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Unit:
    """Smallest readable piece of a corpus.

    Row slice [row_start, row_end) of parquet row group `part` of `file`;
    `size` is its (estimated) uncompressed text bytes; `multiplier` is how
    many times the full pass counted the file.
    """

    file: str
    part: int
    row_start: int
    row_end: int
    size: int
    multiplier: int = 1

    @property
    def key(self) -> str:
        return f"{self.file}::{self.part}::{self.row_start}-{self.row_end}"


def parquet_units(files: dict[str, list[tuple[int, int]]], max_unit_bytes: int) -> list[Unit]:
    """files: path -> [(num_rows, text_bytes) per row group]. Large row groups
    are split into contiguous row slices of about `max_unit_bytes` text."""
    units: list[Unit] = []
    for path in sorted(files):
        for rg, (rows, text_bytes) in enumerate(files[path]):
            if rows <= 0:
                continue
            pieces = max(1, math.ceil(text_bytes / max_unit_bytes))
            bounds = np.linspace(0, rows, min(pieces, rows) + 1).round().astype(int)
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                if hi > lo:
                    units.append(
                        Unit(path, rg, int(lo), int(hi), int(round(text_bytes * (hi - lo) / rows)))
                    )
    return units


def draw_sequence(
    units: list[Unit],
    scheme: str,
    budget_bytes: float,
    seed: int,
    text_per_size: float = 1.0,
) -> list[tuple[Unit, float]]:
    """Draw units (with replacement) until the expected text reaches `budget_bytes`.

    Returns [(unit, weight)]. `text_per_size` converts the sampling measure to
    text bytes (1 for parquet; the expected decompression ratio for .gz).
    """
    rng = np.random.default_rng(seed)
    sizes = np.asarray([u.size for u in units], dtype=float)
    out: list[tuple[Unit, float]] = []
    if scheme == "size_proportional":
        prob = sizes * np.asarray([u.multiplier for u in units], dtype=float)
        prob /= prob.sum()
        mean_size = float(np.dot(prob, sizes))
        total = 0.0
        while total < budget_bytes:
            idx = rng.choice(len(units), size=256, p=prob)
            for i in idx:
                out.append((units[i], mean_size / sizes[i]))
                total += sizes[i] * text_per_size
                if total >= budget_bytes:
                    break
        return out
    if scheme == "uniform_files":
        by_file: dict[str, list[int]] = {}
        for i, u in enumerate(units):
            by_file.setdefault(u.file, []).append(i)
        file_names = sorted(by_file)
        total = 0.0
        while total < budget_bytes:
            members = by_file[file_names[rng.integers(len(file_names))]]
            u = units[members[rng.integers(len(members))]]
            out.append((u, float(u.multiplier)))
            total += u.size * text_per_size
        return out
    raise ValueError(f"unknown scheme {scheme!r}")


def plan_runs(
    units: list[Unit],
    schemes: list[str],
    sizes: dict[str, float],
    replicates: int,
    base_seed: int = 0,
    text_per_size: float = 1.0,
    overshoot: float = 1.0,
) -> dict[tuple[str, int], list[tuple[Unit, float]]]:
    """One nested draw sequence per (scheme, replicate), long enough for the largest size."""
    budget = max(sizes.values()) * BYTES_PER_TOKEN * overshoot
    plans = {}
    for s_i, scheme in enumerate(schemes):
        for rep in range(replicates):
            seed = base_seed + 1000 * s_i + rep
            plans[(scheme, rep)] = draw_sequence(units, scheme, budget, seed, text_per_size)
    return plans


def prefix_for_size(
    draws: list[tuple[Unit, float]], unit_text: dict[str, float], target_text_bytes: float
) -> int:
    """Number of leading draws whose actual text bytes reach the target (or all)."""
    total = 0.0
    for i, (u, _w) in enumerate(draws):
        total += unit_text[u.key]
        if total >= target_text_bytes:
            return i + 1
    return len(draws)


# --------------------------------------------------------------------------- #
# evaluation of counted runs
# --------------------------------------------------------------------------- #

def evaluate_sample(mat: np.ndarray, full: np.ndarray, full_desc: dict, n_boot: int,
                    seed: int, meta: dict) -> dict:
    """Score one sample. `mat` rows are the bootstrap units (draws or clusters),
    already weighted; the sample's count vector is their sum."""
    vec = mat.sum(axis=0)
    desc = describe(vec)
    boot = bootstrap_alpha(mat, n_boot=n_boot, seed=seed)
    lo, hi = boot["alpha_ols_ci95"]
    mlo, mhi = boot["alpha_mle_ci95"]
    return {
        **meta,
        **desc,
        **boot,
        "logcount_corr": logcount_correlation(vec, full),
        "abs_err_ols": abs(desc["alpha_ols"] - full_desc["alpha_ols"]),
        "abs_err_mle": abs(desc["alpha_mle"] - full_desc["alpha_mle"]),
        "ci_covers_full_ols": bool(lo <= full_desc["alpha_ols"] <= hi),
        "ci_covers_full_mle": bool(mlo <= full_desc["alpha_mle"] <= mhi),
    }


def evaluate_runs(
    plans: dict[tuple[str, int], list[tuple[Unit, float]]],
    unit_counts: dict[str, np.ndarray],
    unit_text: dict[str, float],
    full: np.ndarray,
    sizes: dict[str, float],
    n_boot: int = 500,
    unweighted_variant: bool = True,
) -> list[dict]:
    """One record per (scheme, replicate, size), bootstrapping over drawn units.
    For size_proportional draws an extra 'size_proportional_unweighted' record
    (same draws, weight 1) shows the size bias without the Hansen-Hurwitz weight."""
    full_desc = describe(full)
    records = []
    for (scheme, rep), draws in sorted(plans.items()):
        variants = [(scheme, True)]
        if scheme == "size_proportional" and unweighted_variant:
            variants.append(("size_proportional_unweighted", False))
        for label, size_tokens in sizes.items():
            n = prefix_for_size(draws, unit_text, size_tokens * BYTES_PER_TOKEN)
            chosen = draws[:n]
            text_bytes = sum(unit_text[u.key] for u, _ in chosen)
            for name, weighted in variants:
                mat = np.stack(
                    [unit_counts[u.key] * (w if weighted else u.multiplier) for u, w in chosen]
                )
                records.append(evaluate_sample(mat, full, full_desc, n_boot, rep, {
                    "scheme": name,
                    "replicate": rep,
                    "size": label,
                    "target_tokens": size_tokens,
                    "sampled_tokens": text_bytes / BYTES_PER_TOKEN,
                    "short": text_bytes < size_tokens * BYTES_PER_TOKEN,
                    "bootstrap_unit": "draw",
                    "draws": n,
                    "unique_units": len({u.key for u, _ in chosen}),
                    "unique_files": len({u.file for u, _ in chosen}),
                }))
    return records


# --------------------------------------------------------------------------- #
# two-stage plan for non-seekable files (.jsonl.gz)
# --------------------------------------------------------------------------- #

def two_stage_plan(
    files: dict[str, tuple[int, int]],
    unit_bytes: int,
    files_per_run: int,
    seed: int,
) -> list[dict]:
    """Stage 1: files with replacement, probability proportional to
    compressed size x multiplier. Stage 2: a random permutation of each drawn
    file's units; a sample of u units per file takes the first u of it, so the
    sample sizes of one replicate are nested.

    Hansen-Hurwitz: a draw of file f estimates the corpus total as
    m_f * (n_f / u) * sum(c) / p_f, proportional to n_f / (u * size_f) * sum(c);
    evaluate_two_stage applies that weight (about 1/unit_bytes for every file).
    """
    rng = np.random.default_rng(seed)
    names = sorted(files)
    sizes = np.asarray([files[f][0] for f in names], dtype=float)
    mult = np.asarray([files[f][1] for f in names], dtype=float)
    prob = sizes * mult / np.dot(sizes, mult)
    draws = []
    for i in rng.choice(len(names), size=files_per_run, p=prob):
        size, multiplier = files[names[i]]
        n_units = math.ceil(size / unit_bytes)
        draws.append({
            "file": names[i],
            "size": size,
            "multiplier": multiplier,
            "n_units": n_units,
            "order": rng.permutation(n_units).tolist(),
        })
    return draws


def units_per_file(size_tokens: float, files_per_run: int, unit_bytes: int,
                   text_per_compressed: float) -> int:
    text_per_unit = unit_bytes * text_per_compressed
    return max(1, math.ceil(size_tokens * BYTES_PER_TOKEN / (files_per_run * text_per_unit)))


def evaluate_two_stage(
    plans: dict[int, list[dict]],
    unit_counts: dict[str, np.ndarray],
    unit_text: dict[str, float],
    full: np.ndarray,
    sizes: dict[str, float],
    unit_bytes: int,
    text_per_compressed: float,
    n_boot: int = 500,
) -> list[dict]:
    """Bootstrap over file draws (clusters)."""
    full_desc = describe(full)
    records = []
    for rep, draws in sorted(plans.items()):
        for label, size_tokens in sizes.items():
            u = units_per_file(size_tokens, len(draws), unit_bytes, text_per_compressed)
            rows, text_bytes = [], 0.0
            for d in draws:
                keys = [gz_unit_key(d["file"], part) for part in d["order"][:u]]
                take = len(keys)
                weight = d["n_units"] / (take * d["size"]) * unit_bytes
                rows.append(weight * np.sum([unit_counts[k] for k in keys], axis=0))
                text_bytes += sum(unit_text[k] for k in keys)
            records.append(evaluate_sample(np.stack(rows), full, full_desc, n_boot, rep, {
                "scheme": "size_proportional_two_stage",
                "replicate": rep,
                "size": label,
                "target_tokens": size_tokens,
                "sampled_tokens": text_bytes / BYTES_PER_TOKEN,
                "short": text_bytes < 0.8 * size_tokens * BYTES_PER_TOKEN,
                "bootstrap_unit": "file",
                "draws": len(draws),
                "units_per_file": u,
                "unique_files": len({d["file"] for d in draws}),
            }))
    return records


def gz_unit_key(path: str, part: int) -> str:
    return f"{path}::{part}"


def summary_table(records: list[dict], sizes: dict[str, float]) -> list[dict]:
    rows = []
    for scheme in sorted({r["scheme"] for r in records}):
        for label in sizes:
            group = [r for r in records if r["scheme"] == scheme and r["size"] == label]
            if not group:
                continue
            row = {"scheme": scheme, "size": label, "replicates": len(group),
                   "mean_sampled_tokens": float(np.mean([r["sampled_tokens"] for r in group]))}
            for fit in ("ols", "mle"):
                vals = np.asarray([r[f"alpha_{fit}"] for r in group])
                row[f"alpha_{fit}_mean"] = float(np.mean(vals))
                row[f"alpha_{fit}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
                row[f"mean_abs_err_{fit}"] = float(np.mean([r[f"abs_err_{fit}"] for r in group]))
                row[f"ci_coverage_{fit}"] = float(np.mean([r[f"ci_covers_full_{fit}"] for r in group]))
            row["mean_support"] = float(np.mean([r["support"] for r in group]))
            row["mean_logcount_corr"] = float(np.mean([r["logcount_corr"] for r in group]))
            rows.append(row)
    return rows


def acceptance(table: list[dict], scheme: str = "size_proportional", tol: float = 0.01,
               coverage: float = 0.6) -> dict:
    """Scheme B at <= 1B tokens: |mean alpha - full| <= tol (per replicate mean abs err) and CI coverage."""
    out = {}
    for row in table:
        if row["scheme"] != scheme:
            continue
        out[row["size"]] = {
            fit: bool(row[f"mean_abs_err_{fit}"] <= tol and row[f"ci_coverage_{fit}"] >= coverage)
            for fit in ("ols", "mle")
        }
    return out


def write_outputs(out_dir: Path, full: np.ndarray, records: list[dict], sizes: dict[str, float],
                  meta: dict, accept_scheme: str = "size_proportional") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = summary_table(records, sizes)
    full_desc = describe(full)
    (out_dir / "runs.json").write_text(json.dumps(records, indent=2))
    with open(out_dir / "runs.csv", "w", newline="") as fh:
        keys = [k for k in records[0] if not isinstance(records[0][k], list)]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    with open(out_dir / "summary_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    result = {**meta, "full": full_desc, "table": table, "acceptance_scheme": accept_scheme,
              "acceptance": acceptance(table, scheme=accept_scheme)}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    plot_alpha_vs_size(out_dir / "alpha_vs_size.png", records, full_desc, sizes, meta.get("corpus", ""))
    return result


def plot_alpha_vs_size(path: Path, records, full_desc, sizes, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    colors = {"uniform_files": "#c0392b", "size_proportional": "#2471a3",
              "size_proportional_unweighted": "#7f8c8d"}
    for ax, fit in zip(axes, ("ols", "mle")):
        for scheme in sorted({r["scheme"] for r in records}):
            xs, means, sds = [], [], []
            for label, tok in sizes.items():
                vals = [r[f"alpha_{fit}"] for r in records if r["scheme"] == scheme and r["size"] == label]
                if vals:
                    xs.append(tok)
                    means.append(np.mean(vals))
                    sds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                    ax.scatter([tok] * len(vals), vals, s=8, alpha=0.35, color=colors.get(scheme))
            ax.errorbar(xs, means, yerr=sds, marker="o", capsize=3, label=scheme, color=colors.get(scheme))
        ax.axhline(full_desc[f"alpha_{fit}"], color="black", lw=1, ls="--", label="full pass")
        ax.axhspan(full_desc[f"alpha_{fit}"] - 0.01, full_desc[f"alpha_{fit}"] + 0.01, color="0.85")
        ax.set_xscale("log")
        ax.set_xlabel("sample size (tokens ~ text bytes / 4)")
        ax.set_ylabel(f"alpha_{fit.upper()}")
        ax.set_title(f"{title}: alpha_{fit.upper()}")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Part 0: ground truth from the full-pass CSVs
# --------------------------------------------------------------------------- #

def ground_truth(root: Path = Path("results/corpus_alpha_full")) -> dict:
    out = {}
    for name in ("slimpajama-627b_reupload", "llm-jp_corpus_v3"):
        vec = read_counts_csv(root / name / "counts_0_to_10000.csv")
        summary = json.loads((root / name / "summary.json").read_text())
        desc = describe(vec)
        out[name] = {
            **desc,
            "summary_alpha": summary["alpha"],
            "summary_r2": summary["r2"],
            "alpha_matches_summary": abs(desc["alpha_ols"] - summary["alpha"]) < 1e-9,
            "summary_integer_matches": summary["integer_matches"],
            "count_0": float(vec[0]),
        }
    return out


if __name__ == "__main__":
    gt = ground_truth()
    out = Path("results/sampling_validation/ground_truth.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gt, indent=2) + "\n")
    print(json.dumps(gt, indent=2))
