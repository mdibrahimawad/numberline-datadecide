"""E11 tier 1: new extractions for the owner's laptop (Apple MPS / CPU), one command.

  python -m src.e11_tier1_laptop --out <topology_number_line>/data/e11 --all

Parts (each can be run alone with --part):
  prompts     H26, H27: 5 new prompts per number (new random demonstrations, seeds 1-5) for
              2,100 numbers (1-100 + 200 per frequency tenth), Qwen2.5-1.5B and Pythia-1.4B.
              Saves the last-token state, and for prompt seed 1 also the mean over the
              number's own tokens (pooling control).
  randinit    H11: the same 2,100 numbers, original prompt texts, through RANDOMLY INITIALISED
              copies of Qwen2.5-1.5B, SmolLM2-1.7B, Pythia-1.4B and OLMo-2-1B (architecture only,
              seed 0; only configs and tokenizers are downloaded).
  behaviour   H7: "which is larger" for 1,000 pairs of 4-digit numbers in the same block of 100,
              Qwen2.5-1.5B and Pythia-1.4B: log-probability margin of the correct answer.

Outputs per model: <out>/<part>/<model>/layer_LL.npz (X float16 centred, mean, target, prompt_seed),
meta.json; behaviour CSVs. Levels at the E10 relative depths.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time

import numpy as np

from src.extract_number_vectors import levels_for

MODELS_PROMPTS = {"qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B", "pythia-1.4b": "EleutherAI/pythia-1.4b"}
MODELS_RANDINIT = {"qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B", "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
                   "pythia-1.4b": "EleutherAI/pythia-1.4b", "olmo2-1b": "allenai/OLMo-2-0425-1B"}


SUBSET_FILE = os.path.join(os.path.dirname(__file__), "..", "experiments", "E11_topology_hypotheses", "number_subset.txt")


def number_subset(seed=26, recompute=False):
    """1-100 plus 200 ordinary 4-digit numbers from each frequency tenth (E10 stage E levels).
    The seed-26 list is stored in SUBSET_FILE so the laptop scripts need no pandas/scipy."""
    if seed == 26 and not recompute and os.path.exists(SUBSET_FILE):
        return np.loadtxt(SUBSET_FILE, dtype=int)
    import src.e10_explore as ex
    from src.e10_graded import levels
    mt = levels(ex.load_counts(), 10)
    rng = np.random.default_rng(seed)
    picks = [rng.choice(mt["n"].values[mt["level"].values == q], 200, replace=False) for q in range(10)]
    return np.concatenate([np.arange(1, 101), np.sort(np.concatenate(picks))])


def make_prompt(n, rng):
    demos = [rng.randint(10 ** (k - 1) if k > 1 else 0, 10 ** k - 1) for k in (1, 2, 3, 4)]
    return ",".join(f"{a}={a}" for a in demos) + f",{n}="


def original_prompts(path):
    recs = json.load(gzip.open(path, "rt"))["records"]
    return {r["target"]: r["prompt"] for r in recs}


def load(model_id, randinit=False, dtype="float32"):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if randinit:
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_id)).to(getattr(torch, dtype))
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype))
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return tok, model.to(dev).eval(), dev


def states(model, tok, dev, prompts, levels, batch=16, pool_last_number=False):
    """Last-token states (and optionally the mean over the target number's tokens) at `levels`."""
    import torch
    enc = [tok(p)["input_ids"] for p in prompts]
    spans = None
    if pool_last_number:   # tokens of the target number = tokens after the prefix ending in the last ','
        pre = [tok(p[: p.rfind(",") + 1])["input_ids"] for p in prompts]
        spans = [(len(a), len(e) - 1) for a, e in zip(pre, enc)]           # [start, '=' position)
    order = np.argsort([len(e) for e in enc], kind="stable")
    out = {L: np.zeros((len(prompts), model.config.hidden_size), np.float32) for L in levels}
    pooled = {L: np.zeros_like(out[L]) for L in levels} if pool_last_number else None
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]; n = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), n), pad, dtype=torch.long); mask = torch.zeros_like(ids)
        for r, i in enumerate(idx):
            ids[r, :len(enc[i])] = torch.tensor(enc[i]); mask[r, :len(enc[i])] = 1
        with torch.no_grad():
            hs = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), output_hidden_states=True).hidden_states
        for L in levels:
            H = hs[L].float().cpu().numpy()
            for r, i in enumerate(idx):
                out[L][i] = H[r, len(enc[i]) - 1]
                if pool_last_number:
                    a, b = spans[i]
                    pooled[L][i] = H[r, a:max(b, a + 1)].mean(0)
    return out, pooled


def save(folder, arrays, targets, extra):
    os.makedirs(folder, exist_ok=True)
    for L, X in arrays.items():
        mean = X.mean(0)
        np.savez_compressed(os.path.join(folder, f"layer_{L:02d}.npz"), X=(X - mean).astype(np.float16), mean=mean,
                            target=targets, **{k: v for k, v in extra.items() if isinstance(v, np.ndarray)})
    json.dump({k: v for k, v in extra.items() if not isinstance(v, np.ndarray)} | {"levels": list(arrays)},
              open(os.path.join(folder, "meta.json"), "w"), indent=2)


def part_prompts(out, nums):
    for name, mid in MODELS_PROMPTS.items():
        tok, model, dev = load(mid); levels = levels_for(model.config.num_hidden_layers)
        allX = {L: [] for L in levels}; seeds, targets = [], []
        t0 = time.time()
        for ps in range(1, 6):
            rng = random.Random(1000 + ps)
            prompts = [make_prompt(int(n), rng) for n in nums]
            X, pooled = states(model, tok, dev, prompts, levels, pool_last_number=(ps == 1))
            for L in levels:
                allX[L].append(X[L])
            seeds.append(np.full(len(nums), ps)); targets.append(nums)
            if ps == 1:
                save(os.path.join(out, "pooled", name), pooled, nums, dict(model=mid, note="mean over the target number's tokens, prompt seed 1"))
            print(f"{name}: prompt seed {ps} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        save(os.path.join(out, "prompts", name), {L: np.vstack(v) for L, v in allX.items()}, np.concatenate(targets),
             dict(model=mid, prompt_seed=np.concatenate(seeds), n_prompts=5))
        del model


def part_randinit(out, nums, prompts_path):
    orig = original_prompts(prompts_path)
    prompts = [orig[int(n)] for n in nums]
    for name, mid in MODELS_RANDINIT.items():
        tok, model, dev = load(mid, randinit=True); levels = levels_for(model.config.num_hidden_layers)
        X, _ = states(model, tok, dev, prompts, levels)
        save(os.path.join(out, "randinit", name), X, nums, dict(model=mid, randinit_seed=0))
        print(f"{name}: random-init done", flush=True)
        del model


def seq_logprobs(model, tok, dev, contexts, conts, batch=32):
    """Sum of log-probabilities of each continuation given its context (batched, right padding)."""
    import torch
    seqs = [tok(c + x)["input_ids"] for c, x in zip(contexts, conts)]
    starts = [len(tok(c)["input_ids"]) for c in contexts]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = np.zeros(len(seqs))
    for s in range(0, len(seqs), batch):
        chunk = range(s, min(s + batch, len(seqs))); n = max(len(seqs[i]) for i in chunk)
        ids = torch.full((len(chunk), n), pad, dtype=torch.long); mask = torch.zeros_like(ids)
        for r, i in enumerate(chunk):
            ids[r, :len(seqs[i])] = torch.tensor(seqs[i]); mask[r, :len(seqs[i])] = 1
        with torch.no_grad():
            lp = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits.float().log_softmax(-1).cpu().numpy()
        for r, i in enumerate(chunk):
            out[i] = sum(lp[r, j - 1, seqs[i][j]] for j in range(starts[i], len(seqs[i])))
    return out


def part_behaviour(out, seed=7, n_pairs=1000):
    import pandas as pd
    import src.e10_explore as ex
    rng = np.random.default_rng(seed)
    pool = ex.FOUR[~ex.year_zone(ex.FOUR)]
    pairs = []
    while len(pairs) < n_pairs:
        a = int(rng.choice(pool)); b = int(a // 100 * 100 + rng.integers(0, 100))
        if a != b and not (1500 <= b <= 2099):
            pairs.append((a, b))
    shots = "Which is larger: 4512 or 4587? Answer: 4587\nWhich is larger: 7310 or 7302? Answer: 7310\n"
    os.makedirs(os.path.join(out, "behaviour"), exist_ok=True)
    for name, mid in MODELS_PROMPTS.items():
        tok, model, dev = load(mid)
        ctx = [shots + f"Which is larger: {a} or {b}? Answer:" for a, b in pairs]
        big = [max(p) for p in pairs]; small = [min(p) for p in pairs]
        lp_big = seq_logprobs(model, tok, dev, ctx, [f" {x}" for x in big])
        lp_small = seq_logprobs(model, tok, dev, ctx, [f" {x}" for x in small])
        df = pd.DataFrame(dict(a=[p[0] for p in pairs], b=[p[1] for p in pairs], larger=big,
                               margin=lp_big - lp_small, correct=lp_big > lp_small))
        df.to_csv(os.path.join(out, "behaviour", f"{name}.csv"), index=False)
        print(f"{name}: behaviour accuracy {df.correct.mean():.3f}", flush=True)
        del model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", required=True, help="tokenization.json.gz with the original prompt texts")
    ap.add_argument("--part", default="prompts,randinit,behaviour")
    a = ap.parse_args()
    nums = number_subset()
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "numbers.npy"), nums)
    parts = a.part.split(",")
    if "randinit" in parts: part_randinit(a.out, nums, a.prompts)
    if "behaviour" in parts: part_behaviour(a.out)
    if "prompts" in parts: part_prompts(a.out, nums)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
