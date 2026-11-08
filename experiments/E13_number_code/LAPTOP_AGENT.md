# Instructions for Claude Code running on the owner's Mac (E13-1D tests)

You are running on the owner's MacBook (Apple M4 Max, 36 GB). Your job: run the pre-registered E13-1D test battery to
the end, watch it, and fix problems, without changing what the tests mean. Work on branch **main**.

## Context (read first)
| File | What it is |
|---|---|
| `experiments/E13_number_code/PREREG_1D.md` | The pre-registration, including Amendments 1 and 2. **Pass thresholds and test definitions are fixed.** |
| `experiments/E13_number_code/LAPTOP_1D.md` | How the run works |
| `experiments/E13_number_code/run_1d.py` | The runner |
| `experiments/E13_number_code/src/e13/manifold1d.py` | Tests T1–T5 |
| `experiments/E13_number_code/src/e13/causal1d.py` | Causal tests T7–T9 (Apple GPU, "mps") |

- **First run, earlier on this Mac:** T1, T2 and T5 passed in most cells; the Pythia causal tests T7 and T8 passed.
- **Problems found then:** T4 had no power, T3 had a bug for 100–999, and the Qwen causal test had 0 items.
- **Amendment 2** (commit 4a7c1d0) fixes these. The shape-test jobs and the Qwen causal job must be rerun.

## Environment
| What | Where |
|---|---|
| Python environment | `source ~/e13env/bin/activate` (numpy, scipy, scikit-learn, pandas, torch, transformers) |
| New-prompt vectors | `~/e13_out/pythia-1.4b`, `~/e13_out/qwen2.5-1.5b` |
| Old-prompt vectors and checkpoints | `~/topology_number_line/data` (with `e11_ckpt/`) |
| Output | `~/e13_1d` (jobs in `~/e13_1d/jobs`, log `~/e13_1d.log`, report `~/e13_1d/report/RESULTS.md`) |
| Models for the causal tests | already in the Hugging Face cache (EleutherAI/pythia-1.4b, Qwen/Qwen2.5-1.5B) |

## Steps
1. **Update the code:**
   ```bash
   cd ~/numberline-datadecide && git pull && git log --oneline -1
   ```
   The log must show 4a7c1d0 or newer.
2. **Stop any old run:** `pkill -f run_1d.py`.
3. **Keep the first run's report:**
   ```bash
   mkdir -p ~/e13_1d/first_run && cp ~/e13_1d/report/RESULTS.md ~/e13_1d/first_run/
   ```
4. **Delete only the jobs that must be redone:**
   ```bash
   rm -f ~/e13_1d/jobs/new__*.json ~/e13_1d/jobs/old__*.json ~/e13_1d/jobs/causal__qwen2.5-1.5b.json
   ```
   Keep `t5__*`, `t10__*` and `causal__pythia-1.4b.json`.
5. **Start the run:**
   ```bash
   source ~/e13env/bin/activate && cd ~/numberline-datadecide
   nohup caffeinate -dimsu python experiments/E13_number_code/run_1d.py \
     --new-data ~/e13_out --old-data ~/topology_number_line/data --out ~/e13_1d > ~/e13_1d.log 2>&1 &
   ```
6. **Watch it** every few minutes: `tail -5 ~/e13_1d.log` and `pgrep -fl run_1d.py`.
   - It is done when the log says `E13-1D finished`.
   - Qwen's causal part must report `items B1=` greater than 0.
7. **If it stops with an error:** read the traceback, fix the code, run the unit tests, then restart with the step-5
   command. Finished jobs are skipped. Unit tests:
   ```bash
   python -m pytest -q tests/test_e13_pipeline.py -k "manifold1d or causal1d"
   ```
8. **At the end:**
   - Show the owner `~/e13_1d/report/RESULTS.md` in plain, simple English, with tables.
   - Copy the report into the repo:
     ```bash
     mkdir -p experiments/E13_number_code/results/1d_laptop && cp -r ~/e13_1d/report/* experiments/E13_number_code/results/1d_laptop/
     ```
   - Commit with a clear message and push, if git push works on this Mac. If it does not, tell the owner.

## Rules (non-negotiable)
- **Do not change** pass thresholds, test definitions, seeds or the data sets.
- **Bug fixes** (crashes, wrong file paths, an empty item list) are allowed. Write each fix as a short "Amendment 3"
  note at the end of `PREREG_1D.md`: what broke, what you changed, why.
- **If a fix would change what a test measures**, stop and ask the owner first.
- **Report every result,** including failures and "no power". Never re-run until something passes, and never pick
  the best-looking result.
- **Never print, paste or commit tokens or passwords.** Never ask the owner to paste a token into the chat.
- **Explain to the owner** in plain, simple English, with tables.
