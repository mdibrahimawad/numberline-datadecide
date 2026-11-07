"""E13 configuration: models, contexts, layers, sets, seeds, paths. Methodology: ../../METHODOLOGY.md."""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
E13_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(E13_DIR, "..", ".."))
REF = os.path.join(E13_DIR, "ref")
C1_PROMPTS = os.path.join(REF, "c1_prompts_seed42.json.gz")
OLD_REF = os.path.join(REF, "old_c1_next_vectors.npz")
COUNTS = os.path.join(REPO, "results", "corpus_alpha_full", "slimpajama-627b_reupload", "counts_0_to_10000.csv")
ALPHA = os.path.join(REPO, "results", "corpus_alpha_datadecide", "alpha_summary.csv")
DD_COUNTS = os.path.join(REPO, "results", "corpus_alpha_datadecide", "{recipe}", "counts_0_to_10000_drop_partial_docs.csv")

# --------------------------------------------------------------------------- models (G2)
# family: "digit" = digit-splitting tokenizer, "chunk" = multi-digit chunks. stage A = discovery, B = replication.
MODELS = {
    "pythia-1.4b": dict(hf="EleutherAI/pythia-1.4b", family="chunk", sub="neox", stage="A"),
    "qwen2.5-1.5b": dict(hf="Qwen/Qwen2.5-1.5B", family="digit", sub="digit", stage="A"),
    "crystal": dict(hf="LLM360/Crystal", family="digit", sub="digit", stage="B", trust_remote_code=True),
    "openllama-3b": dict(hf="openlm-research/open_llama_3b", family="digit", sub="digit", stage="B", use_fast_model=False),
    "starcoderbase-3b": dict(hf="bigcode/starcoderbase-3b", family="digit", sub="digit", stage="B", gated=True),
    "smollm2-1.7b": dict(hf="HuggingFaceTB/SmolLM2-1.7B", family="digit", sub="digit", stage="B"),
    "olmo2-1b": dict(hf="allenai/OLMo-2-0425-1B", family="chunk", sub="three", stage="B"),
    "redpajama-3b": dict(hf="togethercomputer/RedPajama-INCITE-Base-3B-v1", family="chunk", sub="neox", stage="B"),
    # the old E10 vectors used the ungated unsloth mirror; meta-llama/Llama-3.2-1B (gated) can be set with --llama-id
    "llama3.2-1b": dict(hf="unsloth/Llama-3.2-1B", family="chunk", sub="three", stage="B"),
}
STAGE_A = ["pythia-1.4b", "qwen2.5-1.5b"]
STAGE_B = [m for m in MODELS if m not in STAGE_A]
CAUSAL_MODELS = ["pythia-1.4b", "qwen2.5-1.5b", "olmo2-1b"]

# checkpoints (Part 5a): Pythia steps as in E11; OLMo-2 revisions listed at run time (12, log-spaced)
PYTHIA_STEPS = [0, 64, 256, 512, 1000, 2000, 4000, 8000, 13000, 23000, 33000, 53000, 73000, 103000, 123000, 143000]

# --------------------------------------------------------------------------- contexts (2.1)
# {n} = the integer in plain digits. Order as in METHODOLOGY 2.1. "C1" uses the stored E10 prompts (seed 42);
# "C1s7" = the same format with demonstrations drawn with seed 7. "*b" = second wording of each natural context,
# used ONLY for within-context reliability (DEVIATIONS.md D3); stored at P_num only.
CONTEXTS = {
    "C1": "A=A,B=B,C=C,D=D,{n}=",
    "C1s7": "A=A,B=B,C=C,D=D,{n}=",
    "C2": "She bought {n} apples at the market.",
    "C3": "The population of the town is {n}.",
    "C4": "Room {n} is on the third floor.",
    "C5": "Which is larger, {n} or 500? Answer:",
    "C6": "{n} + 1 =",
    "C7": "3, 8, {n}, 21, 40",
    "C8": "The price is {n} dollars.",
}
VARIANTS = {
    "C2b": "He bought {n} apples at the shop.",
    "C3b": "The population of the city is {n}.",
    "C4b": "Room {n} is on the second floor.",
    "C5b": "Which is larger, {n} or 700? Answer:",
    "C6b": "{n} plus 1 =",
    "C7b": "5, 9, {n}, 33, 70",
    "C8b": "The price is {n} euros.",
}
NEUTRAL = ["C2", "C3", "C4", "C7", "C8"]          # main averaging (user instruction)
TASK = ["C1", "C5", "C6"]                          # task contexts, reported separately
NATURAL = ["C2", "C3", "C4", "C5", "C6", "C7", "C8"]   # the "7 natural contexts" of the H0 rule
TASK_POS = {"C5": "task", "C6": "task", "C5b": None, "C6b": None}   # C5/C6 also read at the final token

# --------------------------------------------------------------------------- layers (2.3)
# Fractions 0, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 0.84375, 1. 0.84375 (= 27/32) instead of 0.84 and int(f*L + 0.5) instead
# of round() so that the five old E10 depths are reproduced exactly (DEVIATIONS.md D1).
LAYER_FRACS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.84375, 1.0]
OLD_DEPTH_FRACS = [0.25, 0.5, 0.75, 0.84375, 1.0]
DEPTH_NAMES = ["0", "12.5%", "25%", "37.5%", "50%", "62.5%", "75%", "84%", "last"]


def layer_indices(n_layers: int) -> list[int]:
    return [int(f * n_layers + 0.5) for f in LAYER_FRACS]


# --------------------------------------------------------------------------- sets (G1)
SETS = {"S1": (1, 9), "S2": (10, 99), "S3": (100, 999), "S4": (1000, 9999)}
MAIN_SETS = ["S2", "S3", "S4"]
BLOCKS = {"S2": (70, 79), "S3": (700, 799), "S4": (7000, 7999)}
N_MAX = 9999
S4_SUBSAMPLE = 3000            # M1/M2/M4/H0-transfer on S4 use this fixed subsample (as E12/E12c), seed 0

# --------------------------------------------------------------------------- seeds (G9)
SEED_C1 = 42
SEED_C1B = 7
SEED_DEV = 13
DEV_FRAC = 0.20

# --------------------------------------------------------------------------- P7 positive control
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
DAY_TEMPLATES = [
    "Today is {X}.", "The meeting is on {X}.", "See you on {X}.", "The shop is closed on {X}.",
    "We usually go swimming on {X}.", "Her flight leaves on {X}.", "The report is due on {X}.",
    "I was born on a {X}.", "The market is open every {X}.", "Let us talk again on {X}.",
    "My favourite day is {X}.", "The party will be on {X}.", "He works late every {X}.",
    "The class starts on {X}.", "It rained all day on {X}.", "The game is next {X}.",
    "They arrived last {X}.", "The museum is free on {X}.", "We have pizza every {X}.", "The test is on {X}.",
]
MONTH_TEMPLATES = [
    "Her birthday is in {X}.", "The festival takes place in {X}.", "School starts in {X}.",
    "We went on holiday in {X}.", "The project ends in {X}.", "It snowed a lot in {X}.",
    "The wedding is planned for {X}.", "They moved house in {X}.", "The report was published in {X}.",
    "My favourite month is {X}.", "The conference is held every {X}.", "Prices rose sharply in {X}.",
    "The garden looks best in {X}.", "He was born in {X}.", "The race is always in {X}.",
    "The trip is booked for {X}.", "Sales were low in {X}.", "The baby is due in {X}.",
    "The election was held in {X}.", "The museum reopens in {X}.",
]

# --------------------------------------------------------------------------- output layout
DEFAULT_OUT = "/workspace/e13_out"
