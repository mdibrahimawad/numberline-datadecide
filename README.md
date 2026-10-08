# Number-line compression in LLMs vs the numbers in their training data

Language models place numbers on an internal "number line" that is **compressed**:
large numbers sit closer together than small ones. This project asks whether the
amount of compression (**β**) is set by the statistics of the numbers a model was
trained on. To test it cleanly, we use the **25 DataDecide 1B models** (Ai2): same
architecture, size, tokenizer and training length, with only the pretraining data
differing. For each model we measure β from its hidden states, and we count every
integer 0–10000 in **exactly the 100B tokens it was trained on**.

## Where things stand (2026-10-08)

| | |
|---|---|
| **Established** | β is real and reproducible (±0.015) and varies 0.55–1.38 across the 25 models. The exact training-stream counts are reproducible: any 100B sample has the recipe's α, and the data seed doesn't matter. |
| **α (power-law slope)** | correlates with β (r = +0.56, p = 0.003) in the predicted direction, but it is carried by the Dolma family and fails inside other families. Not the direct driver. |
| **Best predictor so far** | **R = how many integers 10–9999 the model saw fewer than 4,000 times**: ρ = −0.73 with β, holds within families, and absorbs α (with R in the model, α has p = 0.47). Interpreted as Bayesian shrinkage of rarely-seen numbers. |
| **Caveat** | Exploratory, and confounded with code/math content (Dolma). Confirmation planned: the **checkpoint test (E08)**. |

Details: [`docs/findings.md`](docs/findings.md).

## Start here

| If you want to… | Read |
|---|---|
| understand the project and its definitions (α, β, R) | [`docs/project_overview.md`](docs/project_overview.md) |
| know what has been found, what failed, and what is open | [`docs/findings.md`](docs/findings.md) |
| see every experiment (summary + full write-up) | [`experiments/README.md`](experiments/README.md) |
| check what a result rests on | [`docs/assumptions_and_caveats.md`](docs/assumptions_and_caveats.md) |
| find accounts, cloud storage, costs, or set up a new machine | [`docs/infrastructure.md`](docs/infrastructure.md) |
| continue the project from zero (new person or new AI session) | [`docs/handover.md`](docs/handover.md) |
| find which file does what | [`docs/code_map.md`](docs/code_map.md) |
| see what happened when | [`docs/timeline.md`](docs/timeline.md) |
| run jobs on RunPod (beginner guide) | [`docs/runpod.md`](docs/runpod.md) |
| understand how DataDecide fed data to its models | [`docs/datadecide_sampling.md`](docs/datadecide_sampling.md) |

## Experiments

| ID | Experiment | Status |
|---|---|---|
| [E00](experiments/E00_prior_work/) | Prior work: corpus frequencies, geometry probe on public LLMs | done |
| [E01](experiments/E01_datadecide_beta_paper_protocol/) | β of the 25 DataDecide models (paper protocol) | done |
| [E02](experiments/E02_sampling_validation/) | Does a random sample give the full-corpus α? | done |
| [E03](experiments/E03_datadecide_training_sampling/) | DataDecide training-data reproduction + cheap window α | done |
| [E04](experiments/E04_exact_100b_alpha/) | Exact α of each model's 100B training stream | done |
| [E05](experiments/E05_alpha_vs_beta/) | α vs β, confounders, decade metrics | done |
| [E06](experiments/E06_fine_beta/) | Fine β: 10 groups, reproducible, per-prompt data saved | done |
| [E07](experiments/E07_count_features_rarely_seen/) | Count features: the rarely-seen count R | done (exploratory) |
| [E08](experiments/E08_checkpoint_test/) | Checkpoint test: does β follow R during training? | planned, pre-registered |

## Repository layout

```
docs/          project-wide documentation
experiments/   one folder per experiment: README.md (summary) + DETAILS.md (full write-up)
results/       committed outputs, one subfolder per experiment (results/README.md)
configs/       DataDecide model list; recipe -> training-file map
src/           analysis code, the geometry probe, samplers (python -m src.<module>)
modal_app/     Modal cloud entrypoints (modal run modal_app/<file>.py)
runpod_jobs/   RunPod runners (exact α on CPU pods, β on a GPU pod)
utils/         small helpers used by src/
tests/         offline tests
reference/     the original paper's code (unchanged copies)
paper/         LaTeX appendix + figures from the prior-work write-up
```

## Quick start

```bash
git clone -b alpha-sampling https://github.com/mdibrahimawad/numberline-datadecide.git
cd numberline-datadecide
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q tests          # offline tests (test_datadecide_smoke needs Hugging Face access)

# re-derive the main tables from committed results (seconds, no cloud)
python -m src.join_alpha_beta      # E05: α vs β
python -m src.decade_metrics       # E05: decade metrics
python -m src.beta_fine_analysis   # E06: fine β vs α
python -m src.count_features       # E07: count features and R
```
Cloud jobs (Modal, RunPod) and their costs are described in each experiment's README and in
[`docs/infrastructure.md`](docs/infrastructure.md).

## Ground rules
- Work on branch `alpha-sampling`. Each new experiment gets an `experiments/E##_*` folder,
  and every new result updates `docs/findings.md` in the same commit.
- Lock definitions and predictions before testing a hypothesis; never cherry-pick models,
  layers or subsets.
- Never commit tokens (Hugging Face, Modal, RunPod).
