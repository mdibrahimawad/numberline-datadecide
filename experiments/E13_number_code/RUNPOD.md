# E13 on RunPod: exact steps

## Costs (check the RunPod price page; prices change)
| Item | Typical price | E13 use |
|---|---|---|
| RTX 4090 24 GB (community cloud) | about $0.35–0.70 per hour | 10 h limit → about $4–7 |
| A40 48 GB (community cloud) | about $0.35–0.45 per hour | same, with more GPU memory |
| Volume disk 150 GB | about $0.07 per GB per month while it exists | about $10 per month; delete it when done |

- `run_all.py` stops at its time limit (default 10 h). With `--stop-pod` it also stops the pod. A stopped pod still
  bills for its volume; **terminate** the pod to stop all charges.
- Hugging Face uploads and downloads are free.

## 1. Create the pod (click by click)
1. Open runpod.io and log in. Left menu: **Pods** → **+ Deploy**.
2. Choose **RTX 4090** (or **A40**), 1 GPU.
3. Template: **RunPod PyTorch 2.x** (any recent CUDA). Click **Edit Template**:
   - **Container Disk:** 50 GB.
   - **Volume Disk:** 150 GB, mount path `/workspace`.
   - **Environment Variables:** add `HF_TOKEN`, with its value taken from a RunPod secret (next step), never typed
     into a chat or a file.
4. HF token as a secret:
   - **Create it.** Left menu: **Secrets** → **Create Secret**. Name `hf_token`, paste the token from
     huggingface.co → Settings → Access Tokens. It needs **write** access (for the private upload) and must belong
     to an account that accepted the StarCoderBase licence.
   - **Use it.** In the template's environment variables, set `HF_TOKEN` = `{{ RUNPOD_SECRET_hf_token }}`.
5. Click **Deploy On-Demand**. When the pod shows *Running*: **Connect** → **Start Web Terminal** → **Connect to Web
   Terminal** (or use SSH).

## 2. Set up (in the pod terminal)
```bash
cd /workspace
git clone https://github.com/mdibrahimawad/numberline-datadecide.git
cd numberline-datadecide
pip install -r experiments/E13_number_code/requirements_pod.txt
export HF_HOME=/workspace/hf
python -c "import os; print('HF_TOKEN is set' if os.environ.get('HF_TOKEN') else 'HF_TOKEN is MISSING')"
nproc; nvidia-smi --query-gpu=name,memory.total --format=csv
```
If the repo is private, the clone asks for a username and password. Use a GitHub fine-grained token as the password,
typed only in this terminal.

## 3. Smoke test (about 5 minutes; do this first)
```bash
python experiments/E13_number_code/run_all.py --out /workspace/e13_smoke --models pythia-1.4b \
    --numbers 1-300 --stages A1 --no-upload --time-limit-hours 0.5
cat /workspace/e13_smoke/_state/checks_pythia-1.4b.done
```
- Look for `a_pass`, `b_pass` and `f_pass` = true, `d_cos_min` ≥ 0.999 (the old-vector reproduction), and the copy
  accuracy. If any check fails, the script stops with a message; do not use `--force-checks` without asking.
- Then delete the smoke folder: `rm -rf /workspace/e13_smoke`.

## 4. Full run in tmux
```bash
tmux new -s e13
cd /workspace/numberline-datadecide && export HF_HOME=/workspace/hf
python experiments/E13_number_code/run_all.py --out /workspace/e13_out \
    --hf-repo <your-hf-username>/e13-number-code --time-limit-hours 10 --stop-pod
```
- Detach with **Ctrl-b**, then **d**. The job keeps running if the browser closes.
- Re-attach: `tmux attach -t e13`. Follow the log: `tail -f /workspace/e13_out/run_all.log`.
- After the first model, the log prints a line starting with **PROJECTION**: measured minutes per model and a rough
  total. If the total is above the limit, either let it stop at the limit and resume later, or restart with a larger
  `--time-limit-hours`.
- `--workers` defaults to (CPU cores − 1). Analysis runs on the CPU while the GPU extracts the next model.
- `--batch 256` is the default. On an out-of-memory error the batch halves automatically.

## 5. If it stops (time limit, crash, pod restart)
Run **the same command again**. Every finished step is recorded in `/workspace/e13_out/_state/*.done` and skipped.
Extraction resumes at the next chunk of numbers, and analysis resumes at the next unfinished cell.

## 6. Results
- **Location.** Everything is in the private dataset `<your-hf-username>/e13-number-code`:
  - one folder per model: vectors, meta.csv, config.json, checks.json;
  - `<model>_cells`: per-cell results;
  - `report`: tables, verdicts and figures;
  - `ckpt_*` and `dd-*`.
- **Gate A.** `/workspace/e13_out/gate_A.json`. The causal stage is skipped automatically if Gate A fails.
- **When finished.** Download the report (or tell your coding Claude the dataset name), then **terminate** the pod
  and delete the volume.
