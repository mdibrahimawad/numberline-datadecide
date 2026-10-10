# Number-line compression in LLMs vs the numbers in their training data

Language models place numbers on an internal "number line" that is **compressed**:
large numbers sit closer together than small ones. This project asks whether the
amount of compression (**β**) is set by the statistics of the numbers a model was
trained on. To test it cleanly, we use the **25 DataDecide 1B models** (Ai2): same
architecture, size, tokenizer and training length, with only the pretraining data
differing. For each model we measure β from its hidden states, and we count every
integer 0–10000 in **exactly the 100B tokens it was trained on**.

## Where things stand (2026-10-10)

| | |
|---|---|
| **Established** | β is real and reproducible (±0.015) and varies 0.52–1.18 across the 25 models (10-group β_cont at fixed layer 8; 0.55–1.38 with the paper-protocol β at each model's own layer). The exact training-stream counts are reproducible. |
| **Within DataDecide** | At one common layer (8), α_OLS correlates with β: r = +0.75, p = 1.8 × 10⁻⁵ (significant at layers 6–11). R and S do about as well (r ≈ 0.6–0.7). α_MLE does not (p = 0.32). Models with the same α can still differ by up to 0.4 in β, mostly when Dolma's math/code is involved. |
| **Out of sample (E09, pre-registered)** | On the 6 Paloma baselines **no frozen formula transfers**: α is closest (MAE 0.19 vs a 0.15 bar), S overpredicts by ~0.5, **R is falsified** (predicted β ≈ 2–2.6, measured 0.8–1.0). Among the 5 Paloma models, α and β are not related at any clean layer (n = 5). |
| **Efficient coding** | Infomax (space ∝ count) predicts the average β level (0.64 vs 0.72) but none of the differences; q = ½ ranks models (r = 0.80) at the wrong scale. |
| **Open** | Is α a cause or a correlate (content confound, e.g. math)? Training-seed noise of β. Checkpoint test (E08, pre-registered). |

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
| [E09](experiments/E09_paloma_out_of_sample/) | Paloma: do formulas frozen on DataDecide predict 6 new models? | done: no formula transfers; R falsified |

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
git clone https://github.com/mdibrahimawad/numberline-datadecide.git
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
- Work on branch `main` (the only branch). Each new experiment gets an `experiments/E##_*` folder,
  and every new result updates `docs/findings.md` in the same commit.
- Lock definitions and predictions before testing a hypothesis; never cherry-pick models,
  layers or subsets.
- Never commit tokens (Hugging Face, Modal, RunPod).
