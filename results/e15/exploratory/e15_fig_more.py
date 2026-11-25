"""Exploratory figure for E15: all padded numbers (1-, 2-, 3-digit starts), PCA view, full-space Euclidean check."""
import json
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/user/numberline-datadecide")
from src.e15_length_mechanism import axis

V = "/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/e15vec"
OUT = "/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/e15_more.png"
MODELS = [("pythia-1.4b", "Pythia-1.4B · chunks (= DataDecide)", 10),
          ("OLMo-2-0425-1B", "OLMo-2-1B · groups of 3 digits", 7),
          ("Qwen2.5-1.5B", "Qwen2.5-1.5B · single digits", 10)]
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
C = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
MK = {1: "o", 2: "s", 3: "^", 4: "D"}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK})

fig, axs = plt.subplots(2, 3, figsize=(19, 11.5), facecolor=SURF)
rows = []
for col, (slug, title, L) in enumerate(MODELS):
    z = np.load(f"{V}/{slug}.npz")
    H = z["H"].astype(float)[:, L]
    it = pd.DataFrame(json.loads(str(z["items"])))
    z3 = np.load(f"{V}/{slug}_pad3.npz")
    H3 = z3["H"].astype(float)[:, L]
    it3 = pd.DataFrame(json.loads(str(z3["items"])))
    H = np.vstack([H, H3])
    it = pd.concat([it, it3], ignore_index=True)
    it["written"] = it.x.str.len()
    it["start"] = it.n.astype(str).str.len()
    base = (it.set == "base").to_numpy()
    mu, v, a, b, ev, rho = axis(H[base], it.n[base].to_numpy(float))
    it["u"] = ((H - mu) @ v - a) / b

    # --- top: average position vs digits written
    ax = axs[0, col]
    real = it[base].groupby("written").u.agg(["mean", "std"])
    ax.plot(real.index, real["mean"], color=MUTED, lw=6, alpha=0.35, solid_capstyle="round",
            label="real numbers with that many digits", zorder=1)
    for d in real.index:
        ax.text(d + 0.06, real.loc[d, "mean"] - 0.12, f"{real.loc[d, 'mean']:.2f}", fontsize=8.5, color=MUTED)
    groups = {1: ("start 1-digit, e.g. 7 → 07 → 007 → 0007 (9 numbers)", ["base", "pad"]),
              2: ("start 2-digit, e.g. 45 → 045 → 0045 (90 numbers)", ["base", "pad"]),
              3: ("start 3-digit, e.g. 123 → 0123 (100 numbers, extra run)", ["pad3_plain", "pad3"])}
    for s, (lab, sets) in groups.items():
        g = it[(it.start == s) & it.set.isin(sets)]
        if s < 3:
            g = g[(g.set == "pad") | (g.written == s)]
        m = g.groupby("written").u.agg(["mean", "sem"])
        ax.errorbar(m.index, m["mean"], yerr=1.96 * m["sem"], color=C[s], marker=MK[s], ms=8, lw=2.2, capsize=3,
                    label=lab, zorder=3)
        for w in m.index:
            rows.append({"model": slug, "start_digits": s, "written_digits": w, "position": m.loc[w, "mean"],
                         "real_same_digits": real.loc[w, "mean"], "n_items": int((g.written == w).sum())})
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_xlabel("how many digits are written (value unchanged)")
    ax.set_ylabel("position on the number line (PC1, in decades)")
    ax.set_title(title, loc="left", fontsize=12.5)
    ax.set_ylim(0.8, 4.3)
    if col == 0:
        ax.legend(frameon=False, fontsize=8.8, loc="upper left")

    # --- bottom: PCA picture (PC1 x PC2 of real numbers; padded numbers projected)
    ax = axs[1, col]
    Xb = H[base] - mu
    _, _, vt = np.linalg.svd(Xb, full_matrices=False)
    v2 = vt[1]
    p1 = (H - mu) @ v
    p2 = (H - mu) @ v2
    for d in (1, 2, 3, 4):
        m_ = base & (it.written == d).to_numpy()
        ax.scatter(p1[m_], p2[m_], s=14, c=C[d], marker=MK[d], alpha=0.45, edgecolors="none",
                   label=f"real {d}-digit numbers")
    pads = {"2-digit padded to 4 (0045)": ((it.set == "pad") & (it.start == 2) & (it.written == 4), "o"),
            "3-digit padded to 4 (0123)": ((it.set == "pad3"), "^"),
            "1-digit padded to 4 (0007)": ((it.set == "pad") & (it.start == 1) & (it.written == 4), "*")}
    for lab, (mask, mk) in pads.items():
        mask = mask.to_numpy()
        ax.scatter(p1[mask], p2[mask], s=55 if mk != "*" else 150, facecolors="none" if mk != "*" else INK,
                   edgecolors=INK, linewidths=1.1, marker=mk, label=lab, zorder=4)
    ax.set_xlabel("PC1 (the number line)")
    ax.set_ylabel("PC2")
    ax.set_title("PCA view: colours = real numbers, black = zero-padded to 4 digits", loc="left", fontsize=11)
    if col == 0:
        ax.legend(frameon=False, fontsize=8.5, loc="best", markerscale=1.0)

    # --- full-space Euclidean check: nearest centroid of real 1/2/3/4-digit numbers
    cents = {d: H[base & (it.written == d).to_numpy()].mean(0) for d in (1, 2, 3, 4)}
    for (s, w), lab in [((1, 4), "0007-type"), ((2, 4), "0045-type"), ((3, 4), "0123-type"),
                        ((1, 3), "007-type"), ((2, 3), "045-type"), ((1, 2), "07-type")]:
        sets = ["pad3"] if s == 3 else ["pad"]
        mask = (it.set.isin(sets) & (it.start == s) & (it.written == w)).to_numpy()
        near = [min(cents, key=lambda d: np.linalg.norm(h - cents[d])) for h in H[mask]]
        share_w = float(np.mean([x == w for x in near]))
        share_s = float(np.mean([x == s for x in near]))
        rows.append({"model": slug, "check": "euclid_nearest_centroid", "item": lab, "n_items": int(mask.sum()),
                     "nearest_is_written_length": share_w, "nearest_is_true_length": share_s})

for ax in axs.flat:
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("Adding leading zeros moves a number to where real numbers with that many digits sit — "
             "in all three tokenizers (final '=' hidden state, 3 prompts averaged)", x=0.01, ha="left",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(OUT, dpi=120, facecolor=SURF)
df = pd.DataFrame(rows)
df.to_csv(OUT.replace(".png", ".csv"), index=False)
pd.set_option("display.width", 200)
print(df.round(2).to_string(index=False))
