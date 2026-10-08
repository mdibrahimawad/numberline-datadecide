# E00: Prior work (before the DataDecide phase)

**Status:** done before 2026-10-05; uploaded to `main` as the repo's starting point.
The code is kept and still imports cleanly, but it has not been re-run in the DataDecide
phase. Its results lived in a local MLflow store (`./mlflow.db`) and on Modal volumes,
which are **not** in this repo, except the full-pass corpus counts in
`results/corpus_alpha_full/` (used as E02's ground truth).

## What it covered

| Stage | Question | Code |
|---|---|---|
| 1. Corpus frequencies | How often does each integer appear in large corpora (The Pile, RedPajama, RefinedWeb, Dolma, The Stack, SlimPajama, LLM-JP)? | `src/cli.py`, `src/multi_counter.py`, `src/number_analysis.py`, `modal_app/app.py` and `*_app.py` counters, `modal_app/corpus_alpha_full_app.py` |
| 2. Geometry probe | Do public LLMs (Pythia, GPT-2, Llama, Mistral, OLMo) encode a compressed number line? EV, ρ, β per layer | `src/geometry.py`, `src/geometry_cli.py`, `modal_app/geometry_app.py`, layer-selection variants, `src/robust_geometry.py`, `src/manifold_geometry.py` |
| 3. Circuits | Which attention heads / MLPs write the PC1 number direction? | `src/circuits.py`, `modal_app/circuits_app.py` |
| Probes | Number comparison ("which is larger?"), motivation experiments linking geometry to behaviour | `src/number_comparison*.py`, `src/motivation_*.py`, `paper/appendix_number_comparison.tex` |
| Reference | The original paper's released code, and a copy edited to follow the paper exactly (intervals, filter, geometric β fit, letter control) | `reference/llm_natural_log/`, `reference/llm_natural_log_paperfaithful/` |

A per-file list is in `docs/code_map.md`.

## How to run (from the original README)

Setup: `pip install -r requirements.txt`.

**Stage 1**: count every integer in [0, 1000] while streaming a corpus:
```bash
python -m src.cli --max-docs 100000 \
  --experiment-name numberline_freq_count_expv1 --run-name local_multi_count \
  --dataset-name monology/pile-uncopyrighted --dataset-split train
python -m src.number_analysis --run-id <mlflow_run_id>   # CSV + figures
```
Full-Pile sweep on Modal (30 shards, ~335 GB): see `modal_app/README.md`.

**Stage 2**: geometry probe on one model:
```bash
python -m src.geometry_cli --model-name EleutherAI/pythia-2.8b \
  --groups 1 2 3 4 --k 30 --num-examples 3 --runs 3 \
  --experiment-name numberline_geometry_expv1 --run-name pythia_2_8b_numerics
# symbol control (the paper's "letters" baseline): add --data symbols
modal run modal_app/geometry_app.py --models "EleutherAI/pythia-2.8b,openai-community/gpt2-large" --runs 3
```
Gated models need a Modal secret with `HF_TOKEN` (`numberline-hf-token`).

**MLflow UI:**
`mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" --port 5555`
(avoid port 5000 on macOS).

## Conventions inherited by all later experiments
- **Counting:** `\b\d+\b` on the text, leading zeros collapse (`"01"` → 1), so
  `count(3)` excludes the 3 inside `30` or `123`. Later experiments use the 0..10000
  version, `_count_text` in `modal_app/corpus_alpha_app.py`.
- **Probe prompts:** `n_1=n_1,…,n_k=n_k,n=`, last-token hidden state, PCA (and PLS)
  to one component; metrics EV, ρ = |Spearman(target, PC1)|, β from consecutive
  group-mean gaps (`Δ_{i+1} ≈ β·Δ_i`).
