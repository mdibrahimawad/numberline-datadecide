"""Every fixed choice of the experiment, declared before any real data was seen.

Changing a value here after results exist must be recorded in results/README.md
("Changes after seeing data") with the reason.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                      # truth_topology/
WORK = Path(os.environ.get("TT_WORK", ROOT / "work"))           # big files (raw data, states); /workspace/tt on a pod
RAW = WORK / "raw"
STATES = WORK / "states"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# ---- models and checkpoints (Ravfogel et al. 2025, App. E.4) ----
MODELS = {
    "pythia-1.4b-deduped": "EleutherAI/pythia-1.4b-deduped",   # primary
    "pythia-410m-deduped": "EleutherAI/pythia-410m-deduped",   # size check
    "pythia-6.9b": "EleutherAI/pythia-6.9b",                   # replication (stage 2)
}
STEPS = [0, 512, 1000, 3000, 5000, 10000, 20000, 40000, 60000, 80000,
         100000, 110000, 120000, 130000, 143000]
EARLY_STEPS = [64, 128, 256]                                    # optional dense early window
FIRST_THIRD_STEP = 40000                                        # T2-C: velocity accumulated by this step

# ---- data (Marks & Tegmark geometry-of-truth, pinned commit) ----
GOT_COMMIT = "5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb"
GOT_URL = f"https://raw.githubusercontent.com/saprmarks/geometry-of-truth/{GOT_COMMIT}/datasets/"
P1_TOPICS = ["cities", "sp_en_trans", "larger_than"]
P1_PER_CLASS = 300
P2_N_RELATIONS = 10            # most frequent CounterFact relations
P2_N_PAIRS = 600               # matched (true-context, false-context) pairs -> 1200 sequences
P2_CONTEXT = 3                 # statements before the final one
LENGTH_AUC_MAX = 0.6           # confound gate: length alone must not predict truth above this

# ---- seeds ----
SEED_DATA = 0
SEEDS_SUBSAMPLE = [0, 1]       # two subsample seeds for T1 and T2; seed 0 is primary
SEED_PERM = 12345

# ---- linear probes ----
LR_C = 1.0
LR_MAX_ITER = 2000
CV_FOLDS = 5
N_NULL_LINEAR = 5              # shuffled-label repeats for L1/L2

# ---- topology ----
B = 64                         # subsamples per class (Malhotra et al.)
M = 160                        # points per subsample
N_NULL_T1 = 5                  # mixed-vs-mixed pseudo-class splits for T1-null
NORM_REF_N = 500               # points used for the median pairwise distance
N_PERM_MST = 1000              # within-topic label permutations for T3
N_PERM_CONC = 2000             # checkpoint-order permutations for T2-C
N_BOOT = 1000                  # bootstrap over subsamples for Hedges' g
CORR_PRUNE = 0.5               # drop summary features correlated above this (Fay et al.)
KEY_FEATURES = ["h0_death_mean", "h1_n_bars", "h1_pers_mean", "h1_entropy"]

# ---- reporting ----
REF_LAYER_LAST_K = 3           # reference layer = argmax mean L1 AUC over the last 3 checkpoints
ONSET_CONSECUTIVE = 2
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def as_dict() -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in globals().items()
            if k.isupper() and not k.startswith("_")}
