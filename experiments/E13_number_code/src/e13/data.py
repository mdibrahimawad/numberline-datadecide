"""Access to extracted vectors: conditions (context x position, the neutral average, static rows), number sets (G1),
and the fixed dev split (G5)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config as C


def dev_mask(numbers):
    """Fixed 20 % dev set over n = 1..N_MAX (seed 13); returns a boolean array aligned with `numbers`."""
    rng = np.random.default_rng(C.SEED_DEV); full = rng.random(C.N_MAX + 1) < C.DEV_FRAC
    return full[np.asarray(numbers)]


class ModelData:
    def __init__(self, out_dir):
        self.dir = out_dir
        self.meta = pd.read_csv(os.path.join(out_dir, "meta.csv"))
        self.cfg = json.load(open(os.path.join(out_dir, "config.json")))
        self.numbers = np.load(os.path.join(out_dir, "numbers.npy"))
        self.row = {int(n): i for i, n in enumerate(self.numbers)}
        self.lidx = self.cfg["layer_indices"]; self.key = self.cfg["model_key"]
        self.family = C.MODELS.get(self.key, {}).get("family", self.cfg.get("family", "chunk"))
        self._mm = {}

    # ----------------------------------------------------------------- conditions
    def conditions(self):
        out = []
        for f in sorted(os.listdir(self.dir)):
            if f.startswith("vec_") and f.endswith(".npy") and "alllayers" not in f:
                c, k = f[4:-4].rsplit("_", 1); out.append((c, k))
        out.append(("AVG5", "num"))
        return out

    def _arr(self, name):
        if name not in self._mm:
            self._mm[name] = np.load(os.path.join(self.dir, name), mmap_mode="r")
        return self._mm[name]

    def X(self, cond, layer_slot, numbers):
        """float64 matrix [len(numbers), d] for a condition at stored-layer slot (0..8). AVG5 uses the all-layer file."""
        ctx, pos = cond; r = np.array([self.row[int(n)] for n in numbers])
        if ctx == "AVG5":
            return self._arr("vec_avg_num_alllayers.npy")[r, self.lidx[layer_slot]].astype(np.float64)
        if ctx in ("STATIC_E", "STATIC_U"):
            a = self._arr("static_embed.npy" if ctx == "STATIC_E" else "static_unembed.npy")
            return a[r, 0 if pos == "space" else 1].astype(np.float64)
        return self._arr(f"vec_{ctx}_{pos}.npy")[r, layer_slot].astype(np.float64)

    def half_average(self, contexts, layer_slot, numbers, pos="num"):
        return np.mean([self.X((c, pos), layer_slot, numbers) for c in contexts], 0)

    # ----------------------------------------------------------------- number sets (G1)
    def set_numbers(self, s, contexts):
        """Numbers of set s valid in every context listed: ok flag, and the set's modal token count in each context;
        for chunk tokenizers the count must be exactly 1."""
        lo, hi = C.SETS[s]; m = self.meta[(self.meta.n >= lo) & (self.meta.n <= hi)]
        keep = np.ones(len(m), bool)
        for c in contexts:
            if c in ("AVG5",):
                sub = C.NEUTRAL
            elif c in ("STATIC_E", "STATIC_U"):
                ids = pd.read_csv(os.path.join(self.dir, "static_ids.csv")).set_index("n")
                keep &= (ids.loc[m.n, "id_space"].values >= 0); continue
            else:
                sub = [c]
            for cc in sub:
                if f"ntok_{cc}" not in m:
                    keep &= False; continue
                nt = m[f"ntok_{cc}"].values; ok = m[f"ok_{cc}"].values.astype(bool)
                mode = pd.Series(nt[ok]).mode()
                if not len(mode):
                    keep &= False; continue
                target = 1 if self.family == "chunk" else int(mode.iloc[0])
                keep &= ok & (nt == target)
        return m.n.values[keep]

    def token_counts(self, s, ctx):
        lo, hi = C.SETS[s]; m = self.meta[(self.meta.n >= lo) & (self.meta.n <= hi)]
        return m[f"ntok_{ctx}"].value_counts().to_dict() if f"ntok_{ctx}" in m else {}


def subsample_s4(n, size=C.S4_SUBSAMPLE):
    if len(n) <= size:
        return n
    return np.sort(np.random.default_rng(0).choice(n, size, replace=False))
