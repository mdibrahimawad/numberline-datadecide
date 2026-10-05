# research_llm_project

Two-stage pipeline:

- **Stage 1** — corpus-side: for every `N` in `[0, 1000]`, count how often
  it appears as a standalone digit token in a streaming pass over The
  Pile, and log the full frequency table to MLflow.
- **Stage 2** — model-side: for a HF causal LM, collect last-token
  hidden states across all layers on number-prediction prompts, project
  via PCA + PLS, and log per-layer **explained variance**, **monotonicity
  ρ**, and **compression rate β** (geometric ratio of consecutive
  group-mean diffs along PC1) to the same MLflow store.

## Project layout

```
.
├── requirements.txt
├── src/
│   ├── cli.py              # Stage 1: count every N in [0, 1000] over a corpus
│   ├── multi_counter.py    # single-pass \b\d+\b counter over a doc stream
│   ├── number_analysis.py  # Stage 1 figures + CSV from an MLflow run
│   ├── geometry.py         # Stage 2: PCA/PLS probe + EV/rho/beta metrics
│   └── geometry_cli.py     # Stage 2 local CLI with MLflow logging
├── utils/
│   ├── dataset.py          # HuggingFace streaming loader
│   ├── text.py             # digit-token regex
│   ├── prompts.py          # numeral / symbol prompt generators (Stage 2)
│   └── mlflow_utils.py     # MLflow tracking helpers
├── modal_app/
│   ├── app.py              # Stage 1 cloud fan-out (CPU, zstd + regex)
│   └── geometry_app.py     # Stage 2 cloud sweep (GPU, model forward passes)
└── results/
    ├── counts_0_to_1000.csv    # per-integer frequency table
    ├── figs/                   # generated figures (.pdf + .png)
    └── geometry/               # per-model JSON results from Stage 2
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Stream the default dataset (`monology/pile-uncopyrighted`) and count
every integer in `[0, 1000]` in one pass:

```bash
python -m src.cli --max-docs 100000
```

Full flag set:

```bash
python -m src.cli \
  --max-docs 100000 \
  --log-every 10000 \
  --experiment-name numberline_freq_count_expv1 \
  --run-name local_multi_count \
  --dataset-name monology/pile-uncopyrighted \
  --dataset-split train
```

Each run logs to MLflow: `count_N_0000 … count_N_1000`, `docs_seen`,
`total_matches`, `elapsed_seconds`, `avg_matches_per_doc`, and a
`counts_0_to_1000.csv` artifact.

## Figures

Point `number_analysis.py` at an MLflow run whose metrics include
`count_N_0000 … count_N_1000`:

```bash
python -m src.number_analysis --run-id <mlflow_run_id>
```

Writes `results/counts_0_to_1000.csv` plus nine `.pdf` + `.png` pairs
under `results/figs/` (linear / semi-log / log–log with power-law fit,
Zipf rank–frequency with fit, top-30 bars, roundness breakdown,
Benford, empirical-vs-null KL divergences, cumulative mass with
deciles).

## Stage 2 — Number-line geometry probe

For a HF causal LM, generate prompts of the form
`n_1=n_1,…,n_k=n_k,n=` over groups centered at `10^i` (default
`i ∈ {1,2,3,4}`), collect the last-token hidden state at every layer,
project to one component via PCA (unsupervised) and PLS (supervised on
the target value), and report three per-layer metrics:

- **EV** — explained variance ratio (PCA) or `R²` (PLS),
- **ρ** — `|Spearman(target, PC1)|`, the rank-monotonicity of the line,
- **β** — compression rate. If consecutive group means along PC1 satisfy
  `Δ_{i+1} ≈ β · Δ_i`, then `β` is fit by `exp(slope)` of
  `log Δ_i` vs. `i`. `β < 1` means each decade collapses relative to the
  previous one (log-compression).

Local run:

```bash
python -m src.geometry_cli \
  --model-name EleutherAI/pythia-2.8b \
  --groups 1 2 3 4 \
  --k 30 --num-examples 3 --runs 3 \
  --experiment-name numberline_geometry_expv1 \
  --run-name pythia_2_8b_numerics
```

Each run logs:

- params: `model_name`, `groups`, `k`, `num_examples`, `context`,
  `transform_dim`, `runs`, `seed`, `dtype`, `device_map`.
- metrics: `pca_ev_mean_layer_LL`, `pca_rho_mean_layer_LL`,
  `pca_beta_mean_layer_LL` and the matching `pls_*`, plus best-layer
  summaries (`pca_best_layer`, `pca_best_ev_mean`, …).
- artifact: `geometry/<model>_<data>.json` — full per-layer dump.

For symbol controls (the paper's "letters" baseline):

```bash
python -m src.geometry_cli --model-name <hf_id> --data symbols ...
```

Cloud sweep over multiple models on Modal GPUs:

```bash
.venv/bin/modal run modal_app/geometry_app.py \
  --models "EleutherAI/pythia-2.8b,openai-community/gpt2-large" \
  --runs 3
```

Gated models (Llama-2, Mistral, …) need a Modal Secret named
`huggingface` containing `HF_TOKEN`. Public models work without it.

## MLflow

Both stages write to the same local SQLite DB at `./mlflow.db` by
default:

```bash
.venv/bin/mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" --port 5555
```

## Counting semantics

Matches are made against `\b\d+\b` on lowercased document text, so
`count(3)` does **not** include occurrences inside `30`, `123`, etc.
Zero-padded forms (`"1"`, `"01"`, `"001"`) collapse onto the same
integer key so small-integer counts aren't split across padded aliases.
