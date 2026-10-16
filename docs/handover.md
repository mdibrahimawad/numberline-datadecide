# Handover: how to continue this project from zero

For a new collaborator, a new machine, or a new AI-assistant session with no memory of
earlier conversations. Everything needed to continue is in this repository.

## Read in this order (≈ 30 minutes)
1. `README.md`: what the project is, the current headline, and the repo map.
2. `docs/project_overview.md`: the research question and the exact definitions of α, β, R.
3. `docs/findings.md`: what is established, what failed, and the open questions.
4. `experiments/README.md`: the experiment index, then the README of the experiment you
   will work on (and its DETAILS.md).
5. `docs/assumptions_and_caveats.md`: before claiming anything.
6. `docs/infrastructure.md`: before spending money or touching cloud accounts.

## Working rules this project follows
- **Branch:** `main`, the only branch. Pull before working:
  `git pull --no-rebase --no-edit origin main`.
- **One experiment, one folder:** a new experiment gets `experiments/E##_<name>/` with a
  `README.md` (summary: question, method, how to run, headline result, status) and, once
  it has results, a `DETAILS.md`. Code goes in `src/` (local analysis), `modal_app/` or
  `runpod_jobs/` (cloud). Outputs go to `results/<name>/`, which is listed in
  `results/README.md`.
- **Update the docs in the same commit as the result:** the experiment README,
  `docs/findings.md` (headline and evolution table), `docs/timeline.md`.
- **Tests:** offline, tiny fake data, `python -m pytest -q tests`. Every new analysis
  module gets a test that pins its definitions.
- **Lock definitions before testing a hypothesis** (pre-registration). Write the exact
  formula, parameters and predicted direction in the experiment README and commit it
  *before* running. Don't re-tune afterwards.
- **No cherry-picking:** never drop models, pick layers or choose subsets because they
  improve a correlation. Report the pre-specified analysis first; anything else is
  labelled as a robustness check.
- **Say which β:** paper-protocol (`old_direct`), 10-group at the old layer, or at the
  newly selected layer. They differ.
- **Secrets:** never commit or paste tokens. Use `export HF_TOKEN=…` in the shell, the
  Modal secret `numberline-hf-token`, or RunPod secrets.
- **Money:** dry-run first (`--dry-run`), then a pilot, then the full run with
  `--max-hours` and self-delete. Every runner prints its cost estimate. Check the pod
  list afterwards: it should be empty.
- **Commit messages:** say *why*, one logical change per commit.

## How the project owner prefers to work
- Explanations in plain, simple English, with results shown as tables directly in
  the chat or document, not only as file paths.
- Cloud steps given click by click and command by command (which button, which command,
  what the output should look like), because cloud tooling is new to them.
- Costs stated up front, with a worst case.

## Where to pick up (as of 2026-10-16)
E10 and E11 are written up: `experiments/E10_frequency_topology/README.md` (stages A–F, D2) and
`experiments/E11_topology_hypotheses/README.md` (tier 0, checkpoints, shape search). The two strongest results are
pre-registered: same tokenizer → same late-layer number topology (E10 D, D2) and the abrupt appearance of the number line
together with the copy skill during training (E11 C1b). The frequency effect is real but small and mainly vector length.
Vectors (9 models + 28 checkpoints) are in `mdibrahimawad/topology_number_line`, `data/`. Proposed next step (not
pre-registered): tokenizer → topology (first-token rule; inference-time re-tokenization), see `docs/timeline.md` → Next.

### Earlier (2026-10-14)
Done on 2026-10-14: **E10 stage A** (exploratory topology of single numbers vs frequency, Crystal
vectors from `mdibrahimawad/topology_number_line`, `data/crystal_vectors/`); results and the list of
candidates to lock are in `experiments/E10_frequency_topology/README.md`. **E10 stage B-A** (pre-registered
replication on OpenLLaMA-3B and StarCoderBase-3B) is done too: the spacing effect replicates but is
small and model-dependent in form. Then stages C–E (2026-10-15): topology-only questions, a pre-registered
convergence test on 4 more models (pass), and 10 frequency levels on 7 models with figures
(`results/e10_graded/figures/`). Next: decide whether to run checkpoints / several prompts per number,
or start writing E10 up (the README ends with the list of PH observations). The owner does not want any model training.

### Earlier (2026-10-10)
Done on 2026-10-10: fixed-layer β (`src/fixed_layer_beta.py`, α-vs-β table per layer in
`paper/tables/alpha_beta_by_layer.tex`), efficient-coding test (`src/efficient_coding.py`),
E09 Paloma test end to end (counts → frozen predictions → β → scores; result in the E09
README: no formula transfers, R falsified), `main` made the only branch.

Open next steps:
1. **E08 checkpoint test**: pre-registered in its README (review, then freeze). Note E09
   falsified R, so E08's R-based predictions should be re-read before running.
2. **Training-seed noise**: β of the aux-seed final checkpoints, to know how much β moves
   with training randomness (explains pairs like dolma1_7 vs dolma1_7-no-reddit?).
3. **Causal test of α** (proposed, not designed): train small models on one corpus whose
   numbers are rewritten to set α, everything else equal; measure β.
4. **RedPajama counts** (E09, on hold): 536 of 859 common_crawl files are refused by
   data.together.xyz; options are listed in the E09 README.
5. Commit `results/sampling_validation/slimpajama/` from the owner's laptop (E02 outputs).
