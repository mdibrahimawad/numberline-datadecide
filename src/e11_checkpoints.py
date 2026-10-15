"""E11 tier 2: number vectors and copy behaviour across training checkpoints (H21-H25).

For each checkpoint (Hugging Face revision) of a model, extract the last-token hidden state of the
2,100-number subset (`src.e11_tier1_laptop.number_subset`, original prompt texts) at the E10
relative depths, and measure behaviour on the same prompts: the teacher-forced log-probability of
the correct number after `=` (mean over prompts) and greedy first-token accuracy.

Runs on a GPU pod (RunPod) or a laptop. Each revision is deleted from the local cache after use.

  python -m src.e11_checkpoints --family pythia-1.4b --prompts <tokenization.json.gz> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np

from src.e11_tier1_laptop import number_subset, original_prompts, save, states
from src.extract_number_vectors import levels_for

FAMILIES = {
    # Pythia: 154 checkpoints; log-spaced selection incl. step 0 and the final step 143000
    "pythia-1.4b": ("EleutherAI/pythia-1.4b",
                    [f"step{s}" for s in (0, 64, 256, 512, 1000, 2000, 4000, 8000, 13000, 23000, 33000, 53000, 73000, 103000, 123000, 143000)]),
    # OLMo-2-1B: revisions are listed at run time (stage-1 checkpoints), 12 spread over training
    "olmo2-1b": ("allenai/OLMo-2-0425-1B", None),
}


def olmo_revisions(repo, k=12):
    from huggingface_hub import list_repo_refs
    refs = [b.name for b in list_repo_refs(repo).branches if b.name.startswith("stage1-step")]
    step = lambda r: int(r.split("-step")[1].split("-")[0])
    refs = sorted(refs, key=step)
    idx = np.unique(np.round(np.geomspace(1, len(refs), k)).astype(int) - 1)
    return [refs[i] for i in idx]


def behaviour(model, tok, dev, prompts, targets, n=300):
    import torch
    lp, acc = [], []
    for p, t in list(zip(prompts, targets))[:n]:
        ids = tok(p + str(t))["input_ids"]; k = len(tok(p)["input_ids"])
        with torch.no_grad():
            logits = model(torch.tensor([ids]).to(dev)).logits[0].float().log_softmax(-1).cpu().numpy()
        lp.append(float(sum(logits[i - 1, ids[i]] for i in range(k, len(ids)))))
        acc.append(int(np.argmax(logits[k - 1]) == ids[k]))
    return float(np.mean(lp)), float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, revs = FAMILIES[a.family]
    revs = revs or olmo_revisions(repo)
    nums = number_subset(); orig = original_prompts(a.prompts); prompts = [orig[int(n)] for n in nums]
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log = []
    for rev in revs:
        out = os.path.join(a.out, a.family, rev)
        if os.path.exists(os.path.join(out, "meta.json")):
            print(rev, "already done", flush=True); continue
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=getattr(torch, a.dtype)).to(dev).eval()
        levels = levels_for(model.config.num_hidden_layers)
        X, _ = states(model, tok, dev, prompts, levels, batch=32)
        lp, acc = behaviour(model, tok, dev, prompts, nums)
        save(out, X, nums, dict(model=repo, revision=rev, copy_logprob=lp, copy_first_token_acc=acc,
                                minutes=round((time.time() - t0) / 60, 2)))
        log.append(dict(revision=rev, copy_logprob=lp, copy_acc=acc))
        print(f"{rev}: done in {(time.time() - t0) / 60:.1f} min, copy logprob {lp:.2f}, first-token acc {acc:.2f}", flush=True)
        del model
        cache = os.path.expanduser("~/.cache/huggingface/hub")
        for d in os.listdir(cache) if os.path.isdir(cache) else []:
            if repo.split("/")[1].lower() in d.lower():
                shutil.rmtree(os.path.join(cache, d), ignore_errors=True)    # keep the disk small
    json.dump(log, open(os.path.join(a.out, a.family, "behaviour_log.json"), "w"), indent=2)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
