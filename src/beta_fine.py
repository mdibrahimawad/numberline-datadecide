"""Fine-grained number-line compression (beta) with 10 magnitude groups, batched.

The paper protocol (src/datadecide_sweep.py) puts 4 groups at 10, 100, 1000, 10000,
so beta is fitted to only 3 gaps. Here the groups sit every third of a decade,

    centres 10**(1 + g/3), g = 0..9:  10, 22, 46, 100, 215, 464, 1000, 2154, 4642, 10000

each a band of +-10 % around its centre, giving 9 gaps. Per layer and seed we report

    beta_fine     direct fit d_j = s * b**j over the 9 gaps, per decade: b**3, with the fit's
                  relative error (RMS residual / mean gap; R^2 is meaningless near beta = 1)
    beta_coarse   the same fit on groups 10/100/1000/10000 only (the paper's 3 gaps)
    beta_cont     continuous fit over every prompt: PC1 = a + s * B**log10(n), B per decade
    ev, rho       PCA explained variance of PC1, |Spearman(n, PC1)|

Prompts are `c1=c1,c2=c2,c3=c3,n=` with 3 random context numbers in [0, 10000]
(as before) and kept only if the model's greedy continuation starts with n
(correct-output filter, resampling within the band). Everything runs in
batches of prompts with the same token length, so no padding is involved and
each prompt sees exactly the computation of a single-prompt pass, only many
at once on the GPU. The model is loaded once per run.

Outputs per model: <out>/<recipe>.json (metrics) and <out>/<recipe>.npz (per-prompt
targets, groups and the top-5 PCA scores of every layer, per seed, for re-analysis).

    python -m src.beta_fine --model c4                 # one recipe slug or HF repo id
    python -m src.beta_fine --model c4 --device cpu --k 4 --seeds 1   # smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np

N_GROUPS = 10
CENTRES = [10 ** (1 + g / 3) for g in range(N_GROUPS)]
COARSE = (0, 3, 6, 9)  # 10, 100, 1000, 10000
BAND = 0.10
UPPER = 10_000
N_PCS = 5
FIRST_INTEGER = re.compile(r"[-+]?\d+")


def band(g: int, rel: float = BAND) -> range:
    c = CENTRES[g]
    return range(math.ceil(c * (1 - rel)), math.floor(c * (1 + rel)) + 1)


def make_prompt(g: int, rng: random.Random, num_examples: int = 3) -> tuple[str, int]:
    n = rng.choice(band(g))
    ctx = [rng.randint(0, UPPER) for _ in range(num_examples)]
    return ",".join(f"{c}={c}" for c in ctx) + f",{n}=", n


# --------------------------------------------------------------------------- #
# spacing fits
# --------------------------------------------------------------------------- #

def fit_geometric(gaps: np.ndarray) -> tuple[float, float, float]:
    """(b, R^2, relative error) of d_j = s * b**j by least squares on the raw gaps.

    R^2 is uninformative near b = 1: equal gaps leave no variance to explain, so a
    perfect fit can score R^2 ~ 0. The relative error (RMS residual / mean gap) is
    the fit-quality measure that works for every b."""
    from src.spacing_fit import fit_spacing_direct

    gaps = np.abs(np.asarray(gaps, dtype=float))
    b, scale, r2 = fit_spacing_direct(gaps)
    fitted = scale * b ** np.arange(len(gaps))
    err = float(np.sqrt(np.mean((gaps - fitted) ** 2)) / gaps.mean()) if gaps.mean() > 0 else float("nan")
    return b, r2, err


def fit_continuous(u: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(B, R^2) of y = a + s * B**u (u = log10 n) over all prompts; B = 1 is a log code."""
    from scipy.optimize import minimize_scalar

    u = np.asarray(u, float)
    y = np.asarray(y, float)
    if not np.all(np.isfinite(y)) or float(np.ptp(y)) == 0.0:
        return float("nan"), float("nan")

    def design(lb):
        z = np.exp(lb * u) if abs(lb) > 1e-6 else u  # B -> 1: the log-linear limit
        return np.column_stack([np.ones_like(u), z])

    def sse(lb):
        X = design(lb)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ coef
        return float(r @ r)

    res = minimize_scalar(sse, bounds=(-6, 6), method="bounded")
    sst = float(((y - y.mean()) ** 2).sum())
    return float(np.exp(res.x)), (1 - res.fun / sst) if sst > 0 else float("nan")


def layer_metrics(states: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA

    if float(np.ptp(states.astype(np.float32), axis=0).max()) == 0.0:  # e.g. the embedding of '='
        nan = float("nan")
        return ({"ev": nan, "rho": nan, "group_means": {}},
                np.zeros((states.shape[0], N_PCS), np.float16), np.zeros(N_PCS, np.float32))
    pca = PCA(n_components=min(N_PCS, states.shape[0], states.shape[1]))
    scores = pca.fit_transform(states.astype(np.float32))
    pc1 = scores[:, 0]
    u = np.log10(targets)
    if np.corrcoef(u, pc1)[0, 1] < 0:  # orient PC1 to grow with the number
        pc1 = -pc1
        scores[:, 0] = pc1
    rho = abs(float(spearmanr(targets, pc1)[0]))
    present = [g for g in range(N_GROUPS) if np.any(groups == g)]
    means = {g: float(pc1[groups == g].mean()) for g in present}
    out = {"ev": float(pca.explained_variance_ratio_[0]), "rho": rho,
           "group_means": {str(g): means[g] for g in present}}
    if len(present) == N_GROUPS:
        b, r2, err = fit_geometric(np.diff([means[g] for g in range(N_GROUPS)]))
        out.update(beta_fine=b ** 3, r2_fine=r2, err_fine=err)
    if all(g in means for g in COARSE):
        b, r2, err = fit_geometric(np.diff([means[g] for g in COARSE]))
        out.update(beta_coarse=b, r2_coarse=r2, err_coarse=err)
    out["beta_cont"], out["r2_cont"] = fit_continuous(u - 1, pc1)
    return out, scores.astype(np.float16), pca.explained_variance_ratio_.astype(np.float32)


# --------------------------------------------------------------------------- #
# batched model passes
# --------------------------------------------------------------------------- #

def _encode(tokenizer, prompts: list[str]) -> list[list[int]]:
    return [tokenizer(p, add_special_tokens=True)["input_ids"] for p in prompts]


def _by_length(ids: list[list[int]], batch: int):
    """Index batches of prompts that have the same token length (no padding needed)."""
    buckets: dict[int, list[int]] = {}
    for i, x in enumerate(ids):
        buckets.setdefault(len(x), []).append(i)
    for idx in buckets.values():
        for s in range(0, len(idx), batch):
            yield idx[s:s + batch]


def generate_first_integers(model, tokenizer, device, prompts: list[str], max_new: int,
                            batch: int) -> list[tuple[int | None, str]]:
    import torch

    ids = _encode(tokenizer, prompts)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out: list[tuple[int | None, str]] = [(None, "")] * len(prompts)
    with torch.no_grad():
        for idx in _by_length(ids, batch):
            x = torch.tensor([ids[i] for i in idx], device=device)
            gen = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=pad)
            for row, i in enumerate(idx):
                text = tokenizer.decode(gen[row, x.shape[1]:], skip_special_tokens=True).strip()
                m = FIRST_INTEGER.search(text)
                out[i] = (int(m.group(0)) if m else None, text)
    return out


def last_token_states(model, tokenizer, device, prompts: list[str], batch: int) -> np.ndarray:
    """[n_layers + 1, n_prompts, d] float16 last-token hidden states (index 0 = embeddings)."""
    import torch

    ids = _encode(tokenizer, prompts)
    result = None
    with torch.no_grad():
        for idx in _by_length(ids, batch):
            x = torch.tensor([ids[i] for i in idx], device=device)
            hs = model(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True,
                       use_cache=False).hidden_states
            h = torch.stack([layer[:, -1, :] for layer in hs]).to(torch.float16).cpu().numpy()
            if result is None:
                result = np.zeros((h.shape[0], len(prompts), h.shape[2]), dtype=np.float16)
            result[:, idx, :] = h
    return result


def filtered_prompts(model, tokenizer, device, seed: int, k: int, max_new: int,
                     max_candidates: int, batch: int) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """k prompts per group whose greedy continuation starts with the target; rejected
    slots are resampled from the same band, all pending slots checked in one batch."""
    rng = random.Random(seed)
    slots = [{"g": g, "tries": 0} for g in range(N_GROUPS) for _ in range(k)]
    for s in slots:
        s["prompt"], s["n"] = make_prompt(s["g"], rng)
    stats = {str(g): {"tried": 0, "accepted": 0, "examples": []} for g in range(N_GROUPS)}
    pending = list(range(len(slots)))
    while pending:
        res = generate_first_integers(model, tokenizer, device, [slots[i]["prompt"] for i in pending],
                                      max_new, batch)
        still = []
        for i, (got, text) in zip(pending, res):
            s = slots[i]
            s["tries"] += 1
            st = stats[str(s["g"])]
            st["tried"] += 1
            if got == s["n"]:
                s["ok"] = True
                st["accepted"] += 1
            else:
                if len(st["examples"]) < 3:
                    st["examples"].append({"prompt": s["prompt"], "response": text})
                if s["tries"] < max_candidates:
                    s["prompt"], s["n"] = make_prompt(s["g"], rng)
                    still.append(i)
        pending = still
    keep = [s for s in slots if s.get("ok")]
    for g, st in stats.items():
        st["rejection_rate"] = 1 - st["accepted"] / st["tried"] if st["tried"] else float("nan")
        st["failed_slots"] = k - st["accepted"]
    return ([s["prompt"] for s in keep], np.array([s["n"] for s in keep], float),
            np.array([s["g"] for s in keep]), stats)


# --------------------------------------------------------------------------- #
# one model
# --------------------------------------------------------------------------- #

def _old_layer(recipe: str, old_dir: Path) -> int | None:
    p = old_dir / f"{recipe}.json"
    if p.exists():
        return int(json.loads(p.read_text())["selection"]["selected_layer"])
    return None


def _summ(rows: list[dict], key: str) -> dict:
    v = np.array([r.get(key, np.nan) for r in rows], float)
    return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else 0.0}


def run_model(repo_id: str, revision: str | None, *, seeds: list[int], k: int, batch: int,
              max_new: int, max_candidates: int, device: str | None, dtype: str,
              old_layer: int | None) -> tuple[dict, dict]:
    from src.datadecide_sweep import select_joint_layer
    from src.geometry import GeometryConfig, LayerMetrics, _load_model, _resolve_device, _seed_all

    cfg = GeometryConfig(model_name=repo_id, model_revision=revision, device=device, dtype=dtype)
    _seed_all(seeds[0])
    dev = _resolve_device(cfg.device)
    t0 = time.time()
    model, tokenizer = _load_model(cfg, dev, None)
    model.eval()
    print(f"[beta-fine] loaded {repo_id}@{revision} in {time.time() - t0:.0f}s on {dev}", flush=True)

    per_seed, arrays, filters = [], {}, {}
    for seed in seeds:
        t = time.time()
        prompts, targets, groups, fstats = filtered_prompts(model, tokenizer, dev, seed, k, max_new,
                                                            max_candidates, batch)
        states = last_token_states(model, tokenizer, dev, prompts, batch)
        layers = {}
        scores = np.zeros((states.shape[0], len(prompts), N_PCS), np.float16)
        evr = np.zeros((states.shape[0], N_PCS), np.float32)
        for layer in range(states.shape[0]):
            m, sc, ev = layer_metrics(states[layer], targets, groups)
            layers[layer] = m
            scores[layer, :, :sc.shape[1]] = sc
            evr[layer, :len(ev)] = ev
        per_seed.append({"seed": seed, "layers": {str(l): m for l, m in layers.items()}})
        filters[str(seed)] = fstats
        arrays.update({f"seed{seed}_targets": targets, f"seed{seed}_groups": groups,
                       f"seed{seed}_pc_scores": scores, f"seed{seed}_pc_evr": evr})
        print(f"[beta-fine] seed {seed}: {len(prompts)} prompts kept "
              f"({sum(s['tried'] for s in fstats.values())} tried) in {time.time() - t:.0f}s", flush=True)

    n_layers = len(per_seed[0]["layers"])
    sel = select_joint_layer([
        {l: LayerMetrics(explained_variance=r["layers"][str(l)]["ev"],
                         monotonicity=r["layers"][str(l)]["rho"], compression_rate=float("nan"))
         for l in range(n_layers)} for r in per_seed])
    report = {}
    for name, layer in (("selected_layer", sel["selected_layer"]), ("old_layer", old_layer)):
        if layer is None or layer >= n_layers:
            continue
        rows = [r["layers"][str(layer)] for r in per_seed]
        report[name] = {"layer": int(layer), **{key: _summ(rows, key) for key in (
            "beta_fine", "r2_fine", "err_fine", "beta_coarse", "r2_coarse", "err_coarse", "beta_cont",
            "r2_cont", "ev", "rho")}}
    payload = {"model_name": repo_id, "revision": revision, "seeds": seeds, "k_per_group": k,
               "centres": CENTRES, "band": BAND, "selection": sel, "report": report,
               "per_seed": per_seed, "filter": filters, "seconds": time.time() - t0}
    return payload, arrays


SUMMARY = ("recipe", "layer", "old_layer", "beta_fine", "beta_fine_std", "err_fine", "beta_coarse",
           "beta_coarse_std", "err_coarse", "beta_cont", "beta_cont_std", "r2_cont",
           "beta_fine_oldlayer", "beta_coarse_oldlayer", "beta_cont_oldlayer", "ev", "rho")


def write_summary(out_dir: Path, order: list[str] | None = None) -> Path:
    import csv

    rows = []
    for p in sorted(out_dir.glob("*.json")):
        d = json.loads(p.read_text())
        if "report" not in d:
            continue
        s, o = d["report"].get("selected_layer", {}), d["report"].get("old_layer", {})
        g = lambda rep, k, f="mean": rep.get(k, {}).get(f, "")  # noqa: E731
        rows.append({"recipe": d["recipe"], "layer": s.get("layer", ""), "old_layer": o.get("layer", ""),
                     "beta_fine": g(s, "beta_fine"), "beta_fine_std": g(s, "beta_fine", "std"),
                     "err_fine": g(s, "err_fine"), "beta_coarse": g(s, "beta_coarse"),
                     "beta_coarse_std": g(s, "beta_coarse", "std"), "err_coarse": g(s, "err_coarse"),
                     "beta_cont": g(s, "beta_cont"), "beta_cont_std": g(s, "beta_cont", "std"),
                     "r2_cont": g(s, "r2_cont"), "beta_fine_oldlayer": g(o, "beta_fine"),
                     "beta_coarse_oldlayer": g(o, "beta_coarse"), "beta_cont_oldlayer": g(o, "beta_cont"),
                     "ev": g(s, "ev"), "rho": g(s, "rho")})
    rank = {r: i for i, r in enumerate(order or [])}
    rows.sort(key=lambda r: rank.get(r["recipe"], len(rank)))
    out = out_dir / "summary.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY)
        w.writeheader()
        w.writerows(rows)
    return out


def main(argv: list[str] | None = None) -> int:
    from src.datadecide_sweep import load_model_config, recipe_slug, repo_id_for

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="recipe slug (e.g. c4) or HF repo id")
    p.add_argument("--revision", default=None)
    p.add_argument("--seeds", default="45,46,47")
    p.add_argument("--k", type=int, default=100, help="prompts per group per seed")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-candidates", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out-dir", default="results/beta_fine")
    p.add_argument("--old-dir", default="results/datadecide", help="old 4-group runs: their selected layer")
    args = p.parse_args(argv)

    config = load_model_config()
    repo_id = repo_id_for(args.model, config)
    revision = args.revision if args.revision is not None else (None if "/" in args.model else config["revision"])
    recipe = recipe_slug(repo_id)
    payload, arrays = run_model(repo_id, revision, seeds=[int(s) for s in args.seeds.split(",")], k=args.k,
                                batch=args.batch, max_new=args.max_new_tokens,
                                max_candidates=args.max_candidates, device=args.device, dtype=args.dtype,
                                old_layer=_old_layer(recipe, Path(args.old_dir)))
    payload["recipe"] = recipe
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{recipe}.npz", **arrays)
    (out / f"{recipe}.json").write_text(json.dumps(payload, indent=1))
    rep = payload["report"]["selected_layer"]
    print(f"[beta-fine] {recipe}: layer {rep['layer']}  beta_fine {rep['beta_fine']['mean']:.3f} "
          f"(fit error {rep['err_fine']['mean']:.0%})  beta_coarse {rep['beta_coarse']['mean']:.3f}  "
          f"beta_cont {rep['beta_cont']['mean']:.3f} (R2 {rep['r2_cont']['mean']:.2f})  "
          f"in {payload['seconds'] / 60:.1f} min", flush=True)
    print(f"[beta-fine] {write_summary(out, list(config['recipes']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
