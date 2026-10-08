# Infrastructure: accounts, storage, compute, costs

Where everything lives outside this git repo, and how to get back to it from a fresh
machine or account. **Never commit or paste tokens** (Hugging Face, Modal, RunPod).
Tokens go into environment variables or the provider's secret store only.

## Map

```
GitHub  mdibrahimawad/numberline-datadecide   code + small results (this repo)
  └─ branch alpha-sampling                     all current work (see "Branches")
Hugging Face (user mdibrahimawad)
  ├─ datasets/mdibrahimawad/numberline-alpha-results   (private) exact-alpha outputs from RunPod
  └─ datasets/mdibrahimawad/numberline-beta-results    (private) fine-beta outputs from RunPod
  read-only inputs: allenai/DataDecide-<recipe>-1B (models), allenai/DataDecide-data-recipes (training tokens)
Modal   (workspaces of mdibrahimawad; used until 2026-10-06)   CPU/GPU jobs in modal_app/
RunPod  (used from 2026-10-07)                                  CPU + GPU pods via runpod_jobs/
Laptop  MacBook (M4), conda base env, repo at ~/numberline-datadecide
```

## GitHub

- Repo: `https://github.com/mdibrahimawad/numberline-datadecide`.
- **Branches**:

  | Branch | Content |
  |---|---|
  | `main` | the original upload (prior work, E00) |
  | `datadecide-sweep` | E01 code (correct-output filter + DataDecide β sweep) |
  | **`alpha-sampling`** | **everything since** (built on `datadecide-sweep` + a merge of `main`). Work here. Not yet merged into `main`. |
  | `claude/intelligent-babbage-kxhr4c` | a session branch created by the Claude Code environment; unused |
- The Claude GitHub App is installed on the repo (lets a Claude session push).
- When a push is rejected because the remote has new commits: `git pull --no-rebase --no-edit origin alpha-sampling`, then push again.

## Hugging Face

- Account: `mdibrahimawad`. Create tokens at huggingface.co → Settings → Access Tokens.
  A **write** token is needed for the RunPod jobs, because they upload results.
- **Inputs (public, Ai2):**
  - models `allenai/DataDecide-<recipe>-1B`, revision `step69369-seed-default` (final
    checkpoint). Intermediate checkpoints exist as other revisions (every 2,500 steps;
    seeds `default`, plus aux seeds) and are not used yet.
  - training data `allenai/DataDecide-data-recipes` (tokenized `.npy` files, about 42 TB
    for the 11 largest recipes).
- **Outputs (private, ours):**

  | Dataset | Written by | Layout | Fetched with |
  |---|---|---|---|
  | `numberline-alpha-results` | `runpod_jobs/exact_alpha.py --upload-hf` | `exact_100b_<recipe>/{summary.json, counts_seed_2.csv, …}`, plus `_claims/`, `_pods/` (coordination between pods, safe to ignore) | `python -m runpod_jobs.fetch_results` |
  | `numberline-beta-results` | `runpod_jobs/beta.py --fine --upload-hf` | `step69369-seed-default/<recipe>.{json,npz}`, `summary.csv` | `python -m runpod_jobs.fetch_results --kind beta` |

  Both datasets have been fetched and their contents committed to this repo
  (`results/corpus_alpha_datadecide/exact_100b_*`, `results/beta_fine/`), so they are
  backups, not the only copy.

## Modal (2026-10-05 → 10-06)

- Used for: E01 (β sweep, GPU L4), E02 (sampling validation), E03 (window sweep),
  E04 (exact α for c4, dolma1_7 and 12 more recipes), and the prior work in E00.
- Setup: `pip install modal && modal setup` (browser login). Switch workspace with `modal token new`.
- **Secret** `numberline-hf-token` (key `HF_TOKEN`) must exist in **each** workspace:
  `modal secret create numberline-hf-token HF_TOKEN=<token> --force`. A placeholder value
  (`hf_xxx`) caused failures once.
- **Volumes** (persistent caches, created automatically):
  `numberline-datadecide-hf-cache` (E01 model cache), `numberline-datadecide-alpha` (E03/E04 caches),
  `numberline-sampling-validation` (E02), `numberline-hf-cache`, `numberline-corpus-alpha-cache`,
  `numberline-corpus-alpha-full-results`, and E00's `pile-uncopyrighted-shards`,
  `*-number-frequency-results`, `llm-natural-log-cache`, `numberline-robust-pca`.
  None hold anything that isn't also in this repo or reproducible.
- Lessons: a workspace is **disabled when its budget runs out** (jobs then fail with
  "workspace … is disabled"). The real bill was ~1.35–2× the measured compute (container
  overhead). Preemption messages are harmless.

## RunPod (2026-10-07 → now)

Full beginner guide: `docs/runpod.md`. Summary:
- **Exact α (E04, CPU):** 2 CPU pods (16 and 32 vCPU, compute-optimized, ~$0.03 per
  vCPU-hour). Each ran `python -m runpod_jobs.exact_alpha --recipes … --upload-hf
  numberline-alpha-results --delete-pod-when-done --max-hours 16`. The pods shared the 11
  recipes through claim files in the HF dataset, uploaded each recipe, verified it, then
  deleted themselves (pod 1 had to be stopped by hand; cause unknown).
- **Fine β (E06, GPU):** 1 × RTX A5000 (24 GB, $0.27/h), template Runpod PyTorch, 80 GB
  container disk, no volume. 25 models in 18 min with `--parallel 4`, then self-deleted.
- SSH: key `~/.ssh/id_ed25519` added under RunPod → Settings → SSH Public Keys; connect
  with the command on the pod's Connect tab (`ssh <podid>-<n>@ssh.runpod.io -i ~/.ssh/id_ed25519`).
- Safety features in the runners: upload + verify before self-delete, `--max-hours` hard
  limit, retries, stall watchdog, disk cleanup, and a pod is never deleted while its results
  are not verified on HF.
- Billing: pods bill per second; a stopped pod still bills for its volume; with a $0
  balance, pods without a network volume are terminated.

## Laptop

- Repo at `~/numberline-datadecide`, conda `base` env with the packages in `requirements.txt`.
- The laptop is where results are fetched and committed (`fetch_results`, then `git add … && git commit && git push`).

## Cost ledger (approximate; check each provider's billing page for exact numbers)

| Experiment | Provider | Approx. cost |
|---|---|---|
| E02 sampling validation (SlimPajama) | Modal | ~$1 |
| E04 exact α, c4 (5 seeds + full recipe) | Modal | $1.31 compute |
| E04 exact α, dolma1_7 (5 seeds) | Modal | $5.67 compute |
| E03 window sweep, 25 recipes | Modal | ~$3 for the first recipe (before a caching fix), then ~$0.5 each |
| E01 β sweep, 25 models | Modal (GPU L4) | a few dollars |
| E04 exact α, 12 more recipes (smallest first) | Modal, 2 workspaces | ~$45 including overhead (the user's note: "Modal took more than estimated") |
| E04 exact α, last 11 recipes (largest, ~42 TB download) | RunPod, 2 CPU pods overnight | estimated $12–16 |
| E06 fine β, 25 models | RunPod, 1 GPU | ~$0.20 |

## Starting over on a new machine

```bash
git clone -b alpha-sampling https://github.com/mdibrahimawad/numberline-datadecide.git
cd numberline-datadecide
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                      # analysis + tests
python -m pytest -q tests                            # all offline except test_datadecide_smoke (needs HF)
python -m src.beta_fine_analysis                     # re-derive the E06 tables from committed results
python -m src.count_features                         # re-derive the E07 tables
```
GPU and DataDecide code needs `transformers==4.49.0 ai2-olmo==0.6.0` (see `runpod_jobs/setup.sh gpu`).
