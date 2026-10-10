"""Sanity check of one finished β result before the full run spends money on the rest.

    python -m runpod_jobs.check_beta results/beta_fine/<revision>/<slug>.json

Passes (exit 0) when the model has a clear number line at the middle layers (median
|Spearman(n, PC1)| >= 0.75 at layers 6-11; every DataDecide model has >= 0.86, a model
with broken weights ~0.1) and a finite,
plausible β at layer 8 (0.1-5). Anything else means the model or its weights did not load
as intended (exit 1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MIDDLE = range(6, 12)
LAYER = 8
MIN_ORDER = 0.75


def check(path: Path) -> tuple[bool, str]:
    d = json.loads(path.read_text())
    seeds = d["per_seed"]
    med = lambda l, k: float(np.nanmedian([s["layers"][str(l)].get(k, np.nan) for s in seeds]))
    rho = {l: med(l, "rho") for l in MIDDLE if str(l) in seeds[0]["layers"]}
    beta = med(LAYER, "beta_cont")
    msg = (f"{d.get('model_name')}: layer {LAYER} beta_cont {beta:.3f}; middle-layer order quality "
           + ", ".join(f"L{l} {r:.2f}" for l, r in rho.items()))
    ok = len(rho) == len(MIDDLE) and min(rho.values()) >= MIN_ORDER and np.isfinite(beta) and 0.1 <= beta <= 5
    return ok, msg


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ok, msg = check(Path(argv[0]))
    print(("[check] OK  " if ok else "[check] FAILED  ") + msg, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
