"""E14: is the log-like number line just digit length? (pre-registration: experiments/E14_length_vs_log/PREREG.md)

Three measures on a 1-D number-line coordinate y(n) (PC1, oriented to grow with log10 n):
  LS  length share: y = c_L + s_in*log10(n); at a length boundary L->L+1 the extra step d = c_{L+1}-c_L;
      LS = d / (s_in + d) = share of one decade's rise that happens as a jump at the boundary.
      0 = smooth log line, 1 = pure digit-length staircase.
  J   local jump at boundary B: bands [0.9B, B) and [B, 1.1B]; J = (mean y_up - mean y_lo)/s_all - (mean u_up - mean u_lo),
      s_all = slope of y on u = log10 n. 0 = smooth, ~1 = staircase.
  lam within-length shape: y = c_L + b*g_lam(x), x = n/10^(L-1), Box-Cox g; 0 = log inside a length, 1 = linear inside.
  R   (descriptive, full-coverage data) squeeze ratio inside a length: gap(1xx->2xx) / gap(8xx->9xx).

Datasets: D1 = E06 beta data (25 DataDecide), D2 = E06 Paloma (6), D3 = E13 vectors (Pythia, Qwen; copy '=' and sentence).
Synthetic plants (power) are in `synthetic_*`.

  python -m src.e14_length_log power   --out results/e14
  python -m src.e14_length_log e06     --out results/e14
  python -m src.e14_length_log e13     --data /path/to/e13_data --out results/e14
  python -m src.e14_length_log report  --out results/e14
"""
from __future__ import annotations

import argparse
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

LAMS = np.round(np.arange(-1.0, 2.0001, 0.05), 2)
PRIMARY_B = (100, 1000)
LS_STAIR, LS_SMOOTH, CI_WIDE = 0.5, 0.25, 0.5
LAM_LOG, LAM_LIN = 0.5, 0.5          # lam CI hi < 0.5 -> log inside; lam CI lo > 0.5 -> linear inside
MIN_RHO = 0.5


def nlen(n):
    return np.floor(np.log10(np.asarray(n, float)) + 1e-9).astype(int) + 1


def _wls(X, y, w):
    sw = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(X * sw, y * sw[:, 0], rcond=None)
    r = y - X @ beta
    return beta, float(np.sum(w * r * r))


def boxcox(x, lam):
    return np.log(x) if abs(lam) < 1e-9 else (x ** lam - 1) / lam


def measures(y, n, w=None, boundaries=PRIMARY_B, squeeze=False):
    y = np.asarray(y, float); n = np.asarray(n, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
    u = np.log10(n); L = nlen(n); Ls = np.unique(L)
    s_all = _wls(np.c_[np.ones_like(u), u], y, w)[0][1]
    D = (L[:, None] == Ls[None]).astype(float)
    beta, _ = _wls(np.c_[D, u], y, w); c = dict(zip(Ls, beta[:-1])); s_in = beta[-1]
    out = dict(s_all=float(s_all), s_in=float(s_in))
    for B in boundaries:
        l = int(round(np.log10(B))); lo, hi = (L == l), (L == l + 1)
        if lo.any() and hi.any():
            d = c[l + 1] - c[l]; tot = s_in + d
            out[f"LS_{B}"] = float(d / tot) if tot > 0 else np.nan
        bl = (n >= 0.9 * B) & (n < B); bu = (n >= B) & (n <= 1.1 * B)
        if bl.any() and bu.any() and s_all > 0:
            out[f"J_{B}"] = float((y[bu].mean() - y[bl].mean()) / s_all - (u[bu].mean() - u[bl].mean()))
    out["LS"] = float(np.nanmean([out.get(f"LS_{B}", np.nan) for B in PRIMARY_B]))
    out["J"] = float(np.nanmean([out.get(f"J_{B}", np.nan) for B in PRIMARY_B]))
    x = n / 10.0 ** (L - 1); sse = []; bs = []
    for lam in LAMS:
        b, s = _wls(np.c_[D, boxcox(x, lam)], y, w); sse.append(s); bs.append(b[-1])
    k = int(np.argmin(sse)); out["lam"] = float(LAMS[k]); out["b_inside"] = float(bs[k])
    out["W"] = float(s_in / s_all) if s_all > 0 else np.nan
    if squeeze:
        for l in (3, 4):
            m = L == l; lead = (n // 10 ** (l - 1)).astype(int)
            if m.sum() > 50:
                g = {k_: y[m & (lead == k_)].mean() for k_ in (1, 2, 8, 9)}
                out[f"R_{l}"] = float((g[2] - g[1]) / (g[9] - g[8])) if g[9] != g[8] else np.nan
    return out


def classify(ls_ci, lam_ci, rho=1.0):
    lo, hi = ls_ci
    if not np.isfinite(rho) or rho < MIN_RHO:
        return "no number line"
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo > CI_WIDE:
        return "can't tell"
    if lo >= LS_STAIR:
        return "length staircase"
    if hi <= LS_SMOOTH:
        if lam_ci[1] < LAM_LOG:
            return "smooth, log inside"
        if lam_ci[0] > LAM_LIN:
            return "smooth, linear inside"
        return "smooth, shape unclear"
    return "mixed"


def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) > 10 else (np.nan, np.nan)


def _agg(reps, keys):
    return {k: [r.get(k, np.nan) for r in reps] for k in keys}


KEYS = ("LS", "LS_100", "LS_1000", "LS_10", "LS_10000", "J", "J_100", "J_1000", "J_10", "J_10000", "lam", "W",
        "R_3", "R_4", "s_all", "s_in", "b_inside")


def summarise(point, boot, rho):
    row = {k: point.get(k, np.nan) for k in KEYS}
    for k in ("LS", "LS_100", "LS_1000", "J", "lam"):
        row[k + "_lo"], row[k + "_hi"] = ci(boot[k])
    row["rho"] = rho
    row["class"] = classify((row["LS_lo"], row["LS_hi"]), (row["lam_lo"], row["lam_hi"]), rho)
    return row


# --------------------------------------------------------------------------- PCA coordinate
def pc1(X_fit, X_all, u_all):
    Xf = X_fit.astype(np.float32); mu = Xf.mean(0)
    _, S, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
    y = (X_all.astype(np.float32) - mu) @ Vt[0]
    if np.corrcoef(y, u_all)[0, 1] < 0:
        y = -y
    ev = float(S[0] ** 2 / np.sum(S ** 2))
    return y, ev


def balanced_idx(L, rng, per=90):
    return np.concatenate([rng.choice(np.where(L == l)[0], per, replace=False) for l in np.unique(L)])


def full_coverage(X, n, rng, n_pca=10, B=0, squeeze=True):
    """D3: every number. PCA on a length-balanced subset (90 per length), measures length-weighted."""
    L = nlen(n); w = 1.0 / np.bincount(L)[L]; u = np.log10(n)
    pts = []; evs = []; rhos = []
    for _ in range(n_pca):
        bi = balanced_idx(L, rng); y, ev = pc1(X[bi], X, u)     # rho on the length-balanced subset
        pts.append(measures(y, n, w, squeeze=squeeze)); evs.append(ev); rhos.append(abs(spearmanr(y[bi], n[bi])[0]))
    point = {k: float(np.nanmedian([p.get(k, np.nan) for p in pts])) for k in KEYS}
    ev, rho = float(np.median(evs)), float(np.median(rhos))
    boot = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(np.where(L == l)[0], (L == l).sum(), replace=True) for l in np.unique(L)])
        Xb, nb = X[idx], n[idx]; Lb = L[idx]
        y, _ = pc1(Xb[balanced_idx(Lb, rng)], Xb, np.log10(nb))
        boot.append(measures(y, nb, w[idx], squeeze=squeeze))
    return point, _agg(boot, KEYS), ev, rho


def e06_cell(pcs_by_seed, B=0, rng=None):
    """D1/D2: pcs_by_seed = list of (y, n, groups) with y = stored PC1 (already oriented)."""
    pts = [measures(y, n) for y, n, _ in pcs_by_seed]
    point = {k: float(np.nanmean([p.get(k, np.nan) for p in pts])) for k in KEYS}
    rho = float(np.mean([abs(spearmanr(y, n)[0]) for y, n, _ in pcs_by_seed]))
    boot = []
    for _ in range(B):
        reps = []
        for y, n, g in pcs_by_seed:
            idx = np.concatenate([rng.choice(np.where(g == k)[0], (g == k).sum(), replace=True) for k in np.unique(g)])
            reps.append(measures(y[idx], n[idx]))
        boot.append({k: float(np.nanmean([r.get(k, np.nan) for r in reps])) for k in KEYS})
    return point, _agg(boot, KEYS), rho


# --------------------------------------------------------------------------- synthetic plants (power)
PLANTS = {"professor toy (one-hot only)": None,
          "length staircase": lambda n: nlen(n) - 2.0,
          "smooth log": lambda n: np.log10(n),
          "smooth, linear inside": lambda n: (nlen(n) - 2.0) + (n / 10.0 ** (nlen(n) - 1) - 1) / 9.0,
          "mixed (37.5% length)": lambda n: 0.375 * (nlen(n) - 2.0) + 0.625 * np.log10(n)}
EXPECT = {"professor toy (one-hot only)": ("length staircase",), "length staircase": ("length staircase",),
          "smooth log": ("smooth, log inside",), "smooth, linear inside": ("smooth, linear inside",),
          "mixed (37.5% length)": ("mixed",)}


def synthetic_states(n, plant, noise, rng, D=64, amp=3.0, slots=5):
    X = np.zeros((len(n), D))
    for i, v in enumerate(n):
        s = str(int(v)).rjust(slots)
        for p, ch in enumerate(s):
            X[i, p * 11 + (10 if ch == " " else int(ch))] = 1.0
    if PLANTS[plant] is not None:
        uvec = rng.standard_normal(D); uvec /= np.linalg.norm(uvec)
        X += amp * np.outer(PLANTS[plant](n), uvec)
    return X + noise * rng.standard_normal(X.shape)


def e06_like_numbers(rng, k=100):
    centres = [10 * 10 ** (g / 3) for g in range(10)]
    n, g = [], []
    for j, c in enumerate(centres):
        band = np.arange(int(np.ceil(c * 0.9)), int(np.floor(c * 1.1)) + 1)
        n += list(rng.choice(band, k)); g += [j] * k
    return np.array(n, float), np.array(g)


def _power_job(args):
    design, noise, plant, reps, B, seed = args
    rng = np.random.default_rng(seed); got = []
    for r in range(reps):
        if design.startswith("E06"):
            seeds = []
            for _ in range(3):
                n, g = e06_like_numbers(rng); X = synthetic_states(n, plant, noise, rng)
                y, _ = pc1(X, X, np.log10(n)); seeds.append((y, n, g))
            p, b, rho = e06_cell(seeds, B=B, rng=rng)
        else:
            n = np.arange(10, 10000).astype(float); X = synthetic_states(n, plant, noise, rng)
            p, b, _, rho = full_coverage(X, n, rng, n_pca=3, B=B // 2)
        got.append(summarise(p, b, rho))
    cls = [g["class"] for g in got]
    return dict(design=design, noise=noise, plant=plant, correct_rate=float(np.mean([c in EXPECT[plant] for c in cls])),
                LS_median=float(np.nanmedian([g["LS"] for g in got])), lam_median=float(np.nanmedian([g["lam"] for g in got])),
                J_median=float(np.nanmedian([g["J"] for g in got])), classes=json.dumps({c: cls.count(c) for c in set(cls)}))


def power(out, reps=20, noises=(0.1, 0.3, 0.6), B=100, seed=0, log=print, workers=None):
    import zlib
    from multiprocessing import Pool
    import pandas as pd
    jobs = [(d, nz, pl, reps, B, seed + zlib.crc32(f"{d}{nz}{pl}".encode()))
            for d in ("E06 (bands, 3 seeds)", "E13 (every number 10-9999)") for nz in noises for pl in PLANTS]
    with Pool(workers or min(len(jobs), os.cpu_count() or 1)) as pool:
        rows = pool.map(_power_job, jobs)
    for r in rows:
        log(f"  {r['design']:28s} noise={r['noise']:<4} {r['plant']:30s} correct={r['correct_rate']:.2f} "
            f"LS={r['LS_median']:+.2f} lam={r['lam_median']:+.2f} {r['classes']}")
    os.makedirs(out, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(out, "power_synthetic.csv"), index=False)
    return rows


# --------------------------------------------------------------------------- real data
def run_e06(src_dir, out, tag, B=500, seed=0, fixed_layer=8, log=print):
    import pandas as pd
    rng = np.random.default_rng(seed); rows = []
    for f in sorted(glob.glob(os.path.join(src_dir, "*.npz"))):
        model = os.path.basename(f)[:-4]; z = np.load(f)
        meta = json.load(open(f[:-4] + ".json")); sel = int(meta["selection"]["selected_layer"])
        seeds = sorted({k.split("_")[0] for k in z.files})
        nl = z[f"{seeds[0]}_pc_scores"].shape[0]
        for layer in range(1, nl):
            cell = [(z[f"{s}_pc_scores"][layer, :, 0].astype(float), z[f"{s}_targets"].astype(float), z[f"{s}_groups"])
                    for s in seeds]
            if np.ptp(cell[0][0]) == 0:
                continue
            primary = layer in (sel, fixed_layer)
            p, b, rho = e06_cell(cell, B=B if primary else 0, rng=rng)
            row = dict(dataset=tag, model=model, reading="copy '=' (beta protocol)", layer=layer,
                       role=("selected" if layer == sel else "") + ("+fixed8" if layer == fixed_layer else ""),
                       **summarise(p, b, rho))
            if not primary:
                row["class"] = ""
            rows.append(row)
        r = [x for x in rows if x["model"] == model and "selected" in x["role"]][0]
        log(f"  {tag} {model:42s} L{sel:<2d} LS={r['LS']:+.2f} [{r['LS_lo']:+.2f},{r['LS_hi']:+.2f}] "
            f"J={r['J']:+.2f} lam={r['lam']:+.2f} -> {r['class']}")
    pd.DataFrame(rows).to_csv(os.path.join(out, f"{tag}.csv"), index=False)
    return rows


def run_e13(data, out, models=("pythia-1.4b", "qwen2.5-1.5b"), B=200, seed=0, log=print):
    import pandas as pd
    rng = np.random.default_rng(seed); rows = []
    for m in models:
        d = os.path.join(data, m); n_all = np.load(os.path.join(d, "numbers.npy")).astype(float)
        meta = pd.read_csv(os.path.join(d, "meta.csv"), usecols=["n", "copy_correct_C1"])
        lidx = json.load(open(os.path.join(d, "config.json")))["layer_indices"]
        keep = n_all >= 10
        for reading, fn, layers in (("copy '=' (C1, beta-like)", "vec_C1_next.npy", lidx),
                                    ("sentence (AVG5, number token)", "vec_avg_num_alllayers.npy", None)):
            V = np.load(os.path.join(d, fn), mmap_mode="r")
            k = keep.copy()
            if reading.startswith("copy"):
                k &= meta.set_index("n").loc[n_all.astype(int), "copy_correct_C1"].values.astype(bool)
            n = n_all[k]; layer_ids = layers if layers is not None else list(range(V.shape[1]))
            cells = []
            for slot, layer in enumerate(layer_ids):
                if layer == 0:
                    continue
                X = np.asarray(V[k, slot], np.float32)
                if np.ptp(X, 0).max() == 0:
                    continue
                p, _, ev, rho = full_coverage(X, n, rng, n_pca=10, B=0)
                cells.append((slot, layer, p, ev, rho))
            best = max(cells, key=lambda c: np.sqrt(c[3] * c[4]))
            for slot, layer, p, ev, rho in cells:
                role = "selected" if layer == best[1] else ""
                if role:
                    X = np.asarray(V[k, slot], np.float32)
                    p, b, ev, rho = full_coverage(X, n, rng, n_pca=10, B=B)
                    row = summarise(p, b, rho)
                else:
                    row = summarise(p, {kk: [] for kk in KEYS}, rho); row["class"] = ""
                rows.append(dict(dataset="E13", model=m, reading=reading, layer=layer, role=role, ev=ev, **row))
            r = [x for x in rows if x["model"] == m and x["reading"] == reading and x["role"] == "selected"][0]
            log(f"  E13 {m:14s} {reading:32s} L{r['layer']:<2d} LS={r['LS']:+.2f} [{r['LS_lo']:+.2f},{r['LS_hi']:+.2f}] "
                f"LS100={r['LS_100']:+.2f} LS1000={r['LS_1000']:+.2f} lam={r['lam']:+.2f} R3={r['R_3']:+.2f} -> {r['class']}")
    pd.DataFrame(rows).to_csv(os.path.join(out, "E13.csv"), index=False)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["power", "e06", "e13", "report"])
    ap.add_argument("--out", default="results/e14"); ap.add_argument("--data", default="")
    ap.add_argument("--reps", type=int, default=20); ap.add_argument("--B", type=int, default=0)
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True)
    if a.stage == "power":
        power(a.out, reps=a.reps)
    elif a.stage == "e06":
        run_e06("results/beta_fine/step69369-seed-default", a.out, "E06_DataDecide", B=a.B or 500)
        run_e06("results/beta_fine/step35000-unsharded", a.out, "E06_Paloma", B=a.B or 500)
    elif a.stage == "e13":
        run_e13(a.data, a.out, B=a.B or 200)
    else:
        from src.e14_report import write_report
        write_report(a.out)


if __name__ == "__main__":
    main()
