# Experiments

Each folder has a `README.md` (one-page summary: question, method, how to run, headline
result, status). Most also have a `DETAILS.md` (full write-up: design decisions, every
number, problems met). Code lives in the shared packages (`src/`, `modal_app/`,
`runpod_jobs/`); outputs live in `results/` (see `results/README.md`).

| ID | Experiment | Question | Status | Headline |
|---|---|---|---|---|
| [E00](E00_prior_work/) | Prior work | Number frequencies in corpora; number-line geometry in public LLMs | done (before this phase) | the probe and counting code everything else builds on |
| [E01](E01_datadecide_beta_paper_protocol/) | β of the 25 DataDecide models, paper protocol | How compressed is each model's number line? | done | β = 0.62–1.36; every model has a clear line (ρ ≥ 0.90) |
| [E02](E02_sampling_validation/) | Sampling validation | Does a random sample give the full-corpus α? | done (SlimPajama); LLM-JP part not run | α_MLE: yes from 10M tokens. α_OLS: biased below ~1B |
| [E03](E03_datadecide_training_sampling/) | DataDecide training data + window α | How did DataDecide sample data? A cheap α for all 25 | done | training order reproduced; window α_MLE ≈ exact (±0.001) |
| [E04](E04_exact_100b_alpha/) | Exact 100B training-stream α | α of exactly the tokens each model saw | done (25/25) | 100B sample = full recipe; the seed doesn't matter |
| [E05](E05_alpha_vs_beta/) | α vs β | Does training-data α predict model β? | done | α_OLS r = +0.56 (p = 0.003), but Dolma-driven and fails within families |
| [E06](E06_fine_beta/) | Fine β (10 groups) | A more precise, reproducible β | done | seed noise ±0.015; same-layer α link r = 0.63; layer choice fragile |
| [E07](E07_count_features_rarely_seen/) | Count features and R | What in the counts explains β beyond α? | done (exploratory) | R = #numbers seen < 4000 times: ρ = −0.73, absorbs α (paper-protocol β only; at a fixed layer α ≈ R; **R falsified out of sample in E09**) |
| [E08](E08_checkpoint_test/) | Checkpoint test | Does β follow R during training, with α fixed? | **planned; predictions pre-registered** | — |
| [E09](E09_paloma_out_of_sample/) | Paloma out-of-sample test | Do formulas frozen on DataDecide predict β of 6 new models? | **done** | no formula transfers; α closest (MAE 0.19), S overpredicts, R falsified; only 3 models valid at layer 8 |

## Dependencies between experiments

```
E00 (counting rule, probe code)
 ├─ E01  β (paper protocol) ─────────────┐
 ├─ E02  sampling validation              │
 │   └─ E03  training-data reproduction   │
 │        └─ E04  exact 100B α ───────────┼─ E05  α vs β ─┐
 └──────────────── E06  fine β ───────────┘               ├─ E07  count features, R ─ E08 checkpoint test (planned)
                                                          │
```
