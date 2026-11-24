"""E15: does token count make the log number line? (pre-registration: experiments/E15_length_mechanism/PREREG.md)

    python -m src.e15_length_mechanism extract --model EleutherAI/pythia-1.4b --out <dir>   # CPU, ~10-20 min
    python -m src.e15_length_mechanism analyse --vectors <dir> --out results/e15

Extraction: prompt `a=a,b=b,c=c,x=`, 3 context draws per item (paired across formats of the same number),
final-token hidden state at every layer averaged over the draws; the final token's attention share on the query tokens
(mean over heads) per layer; greedy first-token copy check.
Analysis: number axis = baseline PC1 per layer, calibrated to log10 n (u in decades); tests T1-T4 at the layer chosen by
the joint score sqrt(EV1 * |Spearman(PC1, n)|) on the baseline.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

N_CTX = 3
BOOT = 2000


# --------------------------------------------------------------------------- items

def build_items() -> list[dict]:
    """Every item: set, query string x, value n. Plain baseline items are the only ones used to fit the axis."""
    rng = np.random.default_rng(0)
    base = set(range(1, 100))
    base |= set(rng.choice(np.arange(100, 1000), 300, replace=False).tolist())
    base |= set(rng.choice(np.arange(1000, 10000), 300, replace=False).tolist())
    base |= set(range(90, 110)) | set(range(990, 1010))
    items = [{"set": "base", "x": str(n), "n": n, "k": 0} for n in sorted(base)]
    for n in range(1, 10):                      # T1: 7, 07, 007, 0007
        for w in (2, 3, 4):
            items.append({"set": "pad", "x": str(n).zfill(w), "n": n, "k": w - 1})
    for n in range(10, 100):                    # T1: 42, 042, 0042
        for w in (3, 4):
            items.append({"set": "pad", "x": str(n).zfill(w), "n": n, "k": w - 2})
    rng1 = np.random.default_rng(1)
    for n in sorted(rng1.choice(np.arange(1000, 10000), 100, replace=False).tolist()):   # T3
        items.append({"set": "fmt_plain", "x": str(n), "n": n, "k": 0})
        items.append({"set": "fmt_space", "x": f"{n // 1000} {n % 1000:03d}", "n": n, "k": 0})
        items.append({"set": "fmt_underscore", "x": f"{n // 1000}_{n % 1000:03d}", "n": n, "k": 0})
    return items


def contexts(n: int) -> list[tuple[int, int, int]]:
    """Context numbers depend only on the value n, so every format of the same number gets the same demonstrations."""
    out = []
    for k in range(N_CTX):
        r = np.random.default_rng(n * 10 + k)
        out.append(tuple(int(v) for v in r.integers(0, 10001, 3)))
    return out


# --------------------------------------------------------------------------- extraction

def extract(model_id: str, out: Path, batch: int = 32) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, attn_implementation="eager").eval()
    items = build_items()

    # tokenise every prompt; locate the query tokens from character offsets
    jobs = []
    for i, it in enumerate(items):
        for c, (a, b, cc) in enumerate(contexts(it["n"])):
            prefix = f"{a}={a},{b}={b},{cc}={cc},"
            text = prefix + it["x"] + "="
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            q0, q1 = len(prefix), len(prefix) + len(it["x"])
            q = [j for j, (s, e) in enumerate(offs) if e > s and s < q1 and e > q0]
            assert tok.decode(ids[-1:]).strip().endswith("="), (model_id, text, tok.decode(ids[-1:]))
            assert q and q[-1] == len(ids) - 2, (model_id, text, q, len(ids))
            jobs.append((i, c, ids, q))

    n_layers = model.config.num_hidden_layers + 1
    d = model.config.hidden_size
    H = np.zeros((len(items), n_layers, d), np.float64)
    att = np.zeros((len(items), n_layers - 1), np.float64)
    ntok = np.zeros(len(items), int)
    copy_ok = np.zeros(len(items), float)

    by_len = defaultdict(list)
    for job in jobs:
        by_len[len(job[2])].append(job)
    t0 = time.time()
    done = 0
    with torch.inference_mode():
        for L, group in sorted(by_len.items()):
            for s in range(0, len(group), batch):
                chunk = group[s:s + batch]
                ids = torch.tensor([j[2] for j in chunk])
                res = model(input_ids=ids, output_hidden_states=True, output_attentions=True)
                hs = torch.stack(res.hidden_states)[:, :, -1, :].double().numpy()      # layers x B x d
                attn = torch.stack(res.attentions)[:, :, :, -1, :].mean(2).double().numpy()  # layers x B x T
                nxt = res.logits[:, -1, :].argmax(-1).tolist()
                for b, (i, c, tid, q) in enumerate(chunk):
                    H[i] += hs[:, b, :] / N_CTX
                    att[i] += attn[:, b, q].sum(-1) / N_CTX
                    ntok[i] = len(q)
                    first = tok.decode([nxt[b]]).strip()
                    copy_ok[i] += float(bool(first) and items[i]["x"].startswith(first)) / N_CTX
                done += len(chunk)
            print(f"{model_id}: {done}/{len(jobs)} prompts, {time.time() - t0:.0f}s", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    slug = model_id.split("/")[-1]
    np.savez_compressed(out / f"{slug}.npz", H=H.astype(np.float32), att=att, ntok=ntok, copy_ok=copy_ok,
                        items=json.dumps(items), model=model_id)
    print(f"saved {out / (slug + '.npz')} in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- analysis

def axis(Hl: np.ndarray, n: np.ndarray):
    """Baseline PCA at one layer -> (mean, unit PC1 oriented with log n, a, b, ev1, rho)."""
    from scipy.stats import spearmanr
    mu = Hl.mean(0)
    X = Hl - mu
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]
    p = X @ v
    if np.corrcoef(p, np.log10(n))[0, 1] < 0:
        v, p = -v, -p
    b, a = np.polyfit(np.log10(n), p, 1)
    ev = float(s[0] ** 2 / (s ** 2).sum())
    rho = float(spearmanr(n, p)[0])
    return mu, v, a, b, ev, rho


def boot_ci(x: np.ndarray, rng, stat=np.mean) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(BOOT)]
    return float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyse(vec_dir: Path, out: Path) -> None:
    import pandas as pd
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# E15 results: does token count make the log number line?", "",
             "Pre-registration: `experiments/E15_length_mechanism/PREREG.md`. u = position on the baseline number axis "
             "in decades (1 = one ×10 step). 95 % bootstrap CIs over items in brackets.", ""]
    summary = []
    for f in sorted(vec_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        H, att, ntok, copy_ok = z["H"].astype(np.float64), z["att"], z["ntok"], z["copy_ok"]
        items = pd.DataFrame(json.loads(str(z["items"])))
        items["ntok"], items["copy_ok"] = ntok, copy_ok
        model = str(z["model"])
        base = (items.set == "base").to_numpy()
        n_b = items.n[base].to_numpy(float)
        rng = np.random.default_rng(0)

        # layer scan and choice
        rows = []
        for l in range(1, H.shape[1]):
            mu, v, a, b, ev, rho = axis(H[base, l], n_b)
            rows.append({"layer": l, "ev1": ev, "rho": rho, "score": np.sqrt(ev * abs(rho)), "slope_b": b})
        scan = pd.DataFrame(rows)
        L = int(scan.loc[scan.score.idxmax(), "layer"])
        mu, v, a, bb, ev, rho = axis(H[base, L], n_b)
        items["u"] = ((H[:, L] - mu) @ v - a) / bb

        def u_of(mask):
            return items[mask].set_index(["n"]).u

        # T1 leading zeros
        plain = items[base].set_index("n").u
        pad = items[items.set == "pad"].copy()
        pad["shift_per_zero"] = (pad.u - pad.n.map(plain)) / pad.k
        r_pad = boot_ci(pad.shift_per_zero.to_numpy(), rng)
        v1 = "value" if r_pad[2] < 0.2 else "length" if r_pad[1] > 0.5 else "partial"
        t1 = pad.groupby("x").agg(n=("n", "first"), k=("k", "first"), ntok=("ntok", "first"), u=("u", "mean"),
                                   copy_ok=("copy_ok", "mean"))
        t1_width = pad.assign(width=pad.x.str.len(), plain_ntok=pad.n.map(items[base].set_index("n").ntok)).groupby(
            ["width", "k"]).agg(items=("x", "size"), padded_tokens=("ntok", lambda s: float(np.mean(s))),
                                plain_tokens=("plain_ntok", "mean"), shift_per_zero=("shift_per_zero", "mean"),
                                copy_ok=("copy_ok", "mean")).reset_index()

        # T3 formats
        fp = items[items.set == "fmt_plain"].set_index("n")
        t3 = []
        for fmt in ("fmt_space", "fmt_underscore"):
            ff = items[items.set == fmt].set_index("n")
            delta = (ff.u - fp.u).to_numpy()
            added = (ff.ntok - fp.ntok).to_numpy()
            m, lo, hi = boot_ci(delta, rng)
            verdict = ("value" if lo > -0.2 and hi < 0.2 else "tokens push up" if lo > 0.2
                       else "pushes down" if hi < -0.2 else "unclear")
            t3.append({"format": fmt.replace("fmt_", ""), "delta_u": m, "ci_lo": lo, "ci_hi": hi,
                       "tokens_added_mean": float(added.mean()), "copy_ok": float(ff.copy_ok.mean()),
                       "verdict": verdict})
        t3 = pd.DataFrame(t3)

        # T4 jumps at 100 and 1000
        t4 = []
        bi = items[base].set_index("n")
        for bnd in (100, 1000):
            hi_blk = bi.loc[bnd:bnd + 9]
            lo_blk = bi.loc[bnd - 10:bnd - 1]
            J = hi_blk.u.mean() - lo_blk.u.mean()
            bs = [hi_blk.u.to_numpy()[rng.integers(0, 10, 10)].mean() - lo_blk.u.to_numpy()[rng.integers(0, 10, 10)].mean()
                  for _ in range(BOOT)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            added = hi_blk.ntok.mean() - lo_blk.ntok.mean()
            jump = bool(lo > 0 and J > 0.1)
            token_added = bool(added >= 0.5)
            t4.append({"boundary": f"{bnd - 1}->{bnd}", "J": J, "ci_lo": lo, "ci_hi": hi, "jump": jump,
                       "tokens_before": lo_blk.ntok.mean(), "tokens_after": hi_blk.ntok.mean(),
                       "token_added": token_added, "matches_token_account": jump == token_added})
        t4 = pd.DataFrame(t4)
        n_match = int(t4.matches_token_account.sum())
        v4 = "follows tokens" if n_match == 2 else "does not follow tokens" if n_match == 0 else "mixed"

        # T2 attention share on the query tokens (baseline items), by token count
        t2 = []
        for l in range(att.shape[1]):
            share = att[base, l]
            from scipy.stats import spearmanr
            rr = spearmanr(ntok[base], share)[0] if len(set(ntok[base])) > 1 else np.nan
            row = {"layer": l + 1, "share_mean": float(share.mean()), "spearman_with_ntok": float(rr)}
            for k in sorted(set(ntok[base])):
                row[f"share_{k}tok"] = float(share[ntok[base] == k].mean())
            t2.append(row)
        t2 = pd.DataFrame(t2)

        slug = f.stem
        for name, df in [("layers", scan), ("T1", t1), ("T1_by_width", t1_width), ("T2_attention", t2), ("T3", t3),
                         ("T4", t4)]:
            df.to_csv(out / f"{slug}_{name}.csv", index=False)
        all_layers = []
        for l in range(1, H.shape[1]):
            m_, v_, a_, b_, _, _ = axis(H[base, l], n_b)
            ul = ((H[:, l] - m_) @ v_ - a_) / b_
            p = items.assign(ul=ul)
            pp = p[p.set == "pad"]
            rp = float(((pp.ul - pp.n.map(p[base].set_index("n").ul)) / pp.k).mean())
            fpl = p[p.set == "fmt_plain"].set_index("n").ul
            row = {"layer": l, "R_pad": rp}
            for fmt in ("fmt_space", "fmt_underscore"):
                row[f"delta_{fmt[4:]}"] = float((p[p.set == fmt].set_index("n").ul - fpl).mean())
            bl = p[base].set_index("n").ul
            for bnd in (100, 1000):
                row[f"J_{bnd}"] = float(bl.loc[bnd:bnd + 9].mean() - bl.loc[bnd - 10:bnd - 1].mean())
            all_layers.append(row)
        pd.DataFrame(all_layers).to_csv(out / f"{slug}_all_layers.csv", index=False)

        summary.append({"model": model, "layer": L, "ev1": ev, "rho": rho, "R_pad": r_pad[0], "R_pad_lo": r_pad[1],
                        "R_pad_hi": r_pad[2], "T1": v1, "T4": v4,
                        **{f"T3_{r.format}": f"{r.delta_u:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] {r.verdict}"
                           for r in t3.itertuples()}})

        lines += [f"## {model} (L* = {L}, EV1 {ev:.2f}, Spearman {rho:.3f})", "",
                  f"**T1 leading zeros:** shift per added zero R_pad = {r_pad[0]:+.2f} [{r_pad[1]:+.2f}, {r_pad[2]:+.2f}] "
                  f"decades → **{v1}**.", "", t1_width.round(3).to_markdown(index=False), "",
                  "**T3 same value, more tokens:**", "", t3.round(3).to_markdown(index=False), "",
                  f"**T4 jumps:** → **{v4}**", "", t4.round(3).to_markdown(index=False), "",
                  "**T2 attention share of the final `=` on the query tokens (baseline; mean over heads):**", "",
                  t2.round(3).to_markdown(index=False), ""]
    s = pd.DataFrame(summary)
    s.to_csv(out / "summary.csv", index=False)
    lines[4:4] = ["## Summary", "", s.round(3).to_markdown(index=False), ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--model", required=True); e.add_argument("--out", required=True)
    e.add_argument("--batch", type=int, default=32)
    a = sub.add_parser("analyse"); a.add_argument("--vectors", required=True); a.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.model, Path(args.out), args.batch)
    else:
        analyse(Path(args.vectors), Path(args.out))


if __name__ == "__main__":
    main()
