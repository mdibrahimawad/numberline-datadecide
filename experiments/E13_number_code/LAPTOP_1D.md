# Running the 1-D number-manifold tests on your Mac (overnight)

This runs every test in `PREREG_1D.md`:
- **shape tests T1–T5** on the new prompts (Pythia and Qwen) and on the old prompt (all 9 models);
- **T6** across models;
- **T10** on the training checkpoints;
- **causal tests T7–T9** on your Mac's GPU (Apple MPS).

**Expected time on an M4 Max:** about 1–3 hours, plus about 6 GB of model downloads for the causal part.

## 1. One-time setup (Terminal)
```bash
cd ~
git clone https://github.com/mdibrahimawad/numberline-datadecide.git      # or: cd ~/numberline-datadecide && git pull
ls ~/topology_number_line/data >/dev/null 2>&1 || git clone https://github.com/mdibrahimawad/topology_number_line.git
python3 -m venv ~/e13env && source ~/e13env/bin/activate
pip install numpy scipy scikit-learn pandas matplotlib torch transformers accelerate huggingface_hub
```
Make sure the E13 vectors are fully downloaded into `~/e13_out`. Running the command again only fetches missing files:
```bash
hf download mdibrahimawad/e13-number-code --repo-type dataset --local-dir ~/e13_out --include "pythia-1.4b/*" "qwen2.5-1.5b/*"
```

## 2. Keep the Mac awake
- **Lid open (simplest):** plug in the charger. The command below uses `caffeinate`, so the Mac won't sleep. The
  screen may turn off.
- **Lid closed:** plug in the charger, run `sudo pmset -a disablesleep 1`, then close the lid. In the morning run
  `sudo pmset -a disablesleep 0`. Leave the Mac on a hard surface.

## 3. Start it (the job keeps running if you close the Terminal window)
```bash
source ~/e13env/bin/activate && cd ~/numberline-datadecide && git pull
nohup caffeinate -dimsu python experiments/E13_number_code/run_1d.py \
  --new-data ~/e13_out --old-data ~/topology_number_line/data --out ~/e13_1d > ~/e13_1d.log 2>&1 &
```
- **Check that it started** (after about 1 minute): `tail -5 ~/e13_1d.log`. You should see `Stage 1: ... shape-test jobs`
  and then lines like `[1/39] ... ok`.
- **Follow it:** `tail -f ~/e13_1d.log`. Press Control-c to stop watching; the job keeps running.

## 4. In the morning
- **Results:** `open ~/e13_1d/report/RESULTS.md`. The full table is `~/e13_1d/report/shape_tests.csv`.
- **If it stopped early** (sleep, restart, crash): run the same command from step 3 again. Finished jobs are skipped.
- **Send the results to Claude:** paste `~/e13_1d/report/RESULTS.md`, or run
  `git -C ~/numberline-datadecide status` and push the folder with your usual git workflow.
