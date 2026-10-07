# Running this project on RunPod

Replaces Modal. Two jobs, two kinds of machine:

| Job | Script | Machine | Why |
|---|---|---|---|
| Exact 100B alpha of the 11 recipes still missing | `runpod_jobs/exact_alpha.py` | **CPU pod**, 32 vCPU | download + tokenizer decode; no GPU work |
| Model beta (final models, training checkpoints) | `runpod_jobs/beta.py` | **GPU pod**, one 24 GB card (RTX A5000 / L4 / 3090 / 4090) | 1B models, several at once on one card |

Checked against the official docs (github.com/runpod/docs, Oct 2026). Pod prices are not in
the docs; **the deploy page shows the real price before you click Deploy, always check it there.**

---

## 1. RunPod in 5 minutes (what the words mean)

* **Pod** = a rented Linux machine (really a Docker container on a host) that you SSH into
  and run commands on, like your laptop's terminal. Billed **per second** while it runs.
  This is what we use for everything.
* **GPU pod / CPU pod** = with or without a GPU. For a CPU pod you pick a *flavour* and a
  vCPU count; RAM follows from the flavour: `cpu3c`/`cpu5c` = compute-optimised (2 GB RAM per
  vCPU), `cpu3g`/`cpu5g` = general (4 GB/vCPU), `cpu5m` = memory (8 GB/vCPU). 5 = newer CPUs.
* **Secure Cloud vs Community Cloud** = Tier-3/4 datacentres vs peer-to-peer hosts.
  Community is cheaper, less reliable (RunPod no longer adds new community hosts, existing
  ones remain). Fine for us — our jobs resume after any interruption.
* **On-demand vs interruptible ("spot")** = on-demand cannot be displaced; interruptible is
  cheaper but RunPod can *stop* it at any time to free the machine. A stopped pod keeps its
  `/workspace` volume, so: restart it, rerun the same command, it resumes. Use it when the
  deploy page offers it.
* **Template** = the starting software image. Use **Runpod PyTorch** (official template:
  SSH and file copy are pre-configured) for GPU pods; for CPU pods take the default CPU
  template / an official Ubuntu image.
* **Storage** (three kinds):
  * *Container disk*: wiped when the pod stops. Don't keep results there.
  * *Volume disk* (mounted at `/workspace`): survives stop/restart, deleted on terminate.
    $0.10/GB/month running, **$0.20/GB/month while stopped**.
  * *Network volume*: survives everything, can be attached to different pods in the same
    datacentre, $0.07/GB/month. **Not needed** for us (it also locks you to one datacentre,
    which limits which machines you can get).
* **Stop vs Terminate**: *stop* = compute billing ends but the volume disk keeps billing
  (double rate). *Terminate* = everything deleted, all billing ends. **Copy results to your
  laptop, then terminate.**
* **Serverless** (auto-scaling API endpoints) and **Instant Clusters** (multi-node GPU
  training, 16-64 GPUs) are for other kinds of work. We don't need them.
* No fees for downloading/uploading data (ingress/egress). Billing is per second, charged
  every 5 minutes from your credit balance.
* **You need at least 1 hour of credit for a pod to deploy it**, and there is a default
  spend limit of $80/hour across all pods (plenty for us).
* **When the balance hits $0, every pod is stopped, and pods without a network volume are
  terminated — their data is gone.** Turn on **Billing → Notifications → Low balance alert**
  before your first run, and copy results off the pod as they finish.

---

## 2. One-time setup (on your Mac)

1. **Account / team.** If your professor's funding is on a RunPod **Team** (prepaid credit),
   ask them to invite your email and switch to the team in the account menu so pods bill to it.
   If it is an **Organization** (post-paid invoice, enterprise), note that joining one is
   *permanent*: it removes you from any Team and your personal resources move into it.
2. **SSH key** (lets you log in to pods):
   ```bash
   ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -C "runpod"   # press Enter 3x
   cat ~/.ssh/id_ed25519.pub                                        # copy this line
   ```
   RunPod console → **Credentials** page (console.runpod.io/user/credentials) → **SSH public keys**
   tab → paste → Save. Every pod you create gets it. *(Optional: the pod's **Web Terminal** in the
   browser works without any SSH key.)*
3. **Hugging Face token** (raises download rate limits): RunPod console → **Credentials → Secrets** →
   create secret `hf_token` with your `hf_...` token. When deploying a pod, open
   **Edit template → Environment variables** and add `HF_TOKEN` = `{{ RUNPOD_SECRET_hf_token }}`.
   (Or just `export HF_TOKEN=hf_...` inside the pod each time.)
4. **Low balance alert**: Billing → Notifications → enable, threshold e.g. $10.
5. **runpodctl on your Mac** (for copying results back; it is preinstalled on every pod):
   `bash <(curl -sL cli.runpod.io)` (or `brew install runpod/runpodctl/runpodctl`). Optional:
   `runpodctl config --apiKey <key from Settings → API Keys>` to create/stop pods from the terminal.

---

## 3. Job A: the 11 missing exact alphas (CPU pod)

### What limits speed and cost
Per recipe the job (a) **downloads the whole recipe** (2 bytes/token: dolma-size recipes are
~3 TB, together roughly 25-30 TB) and (b) **decodes only the ~100B tokens that seed 2 trained
on** (≈ 11 × 100B = 1.1T tokens, ~1M tokens/s per vCPU). Small recipes are CPU-bound, big ones
are download-bound. A 32-vCPU pod is roughly balanced at ~1 GB/s of download.

**Cost ≈ pod price × hours, and hours ≈ total work ÷ one pod's throughput.** So 3 pods
in parallel cost about the same as 1 pod for 3× as long — **parallel pods save time at no
extra cost, as long as each pod is fully busy.** The pilot (step 4) tells you whether a pod
is CPU- or download-bound so you can size it right.

### Steps
1. **Deploy**: console → **Pods → Deploy** → **CPU** → compute-optimised flavour (`cpu5c`, else
   `cpu3c`) with the **largest vCPU count offered, ideally ~32** (needs ≥ 16 GB RAM: the
   membership step for the biggest recipe uses ~10 GB) → container disk 30 GB, and **attach a
   50 GB network volume** (CPU pods have no volume disk; the container disk is erased on stop,
   the network volume survives stop, delete and a $0 balance, ~$3.50/month — delete it at the
   end) → a **US** datacentre if you can choose (Hugging Face's CDN is fastest there) →
   Deploy. Note the price per hour shown. If only small sizes exist (e.g. 8-16 vCPU), use
   several pods instead of one big one — same cost, see "Several pods" below.
2. **Connect**: pod → **Connect** tab. Two SSH commands are shown:
   * **SSH** (`ssh <id>@ssh.runpod.io -i ~/.ssh/id_ed25519`) — always works, but **cannot copy
     files** (no scp).
   * **SSH over exposed TCP** (`ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`) — needs a public
     IP; supports scp. Use this one when it is shown.
3. **Install** (inside the pod, ~2 min):
   ```bash
   cd /workspace
   command -v git || (apt-get update && apt-get install -y git)    # bare images lack git
   git clone -b alpha-sampling https://github.com/mdibrahimawad/numberline-datadecide.git
   cd numberline-datadecide && bash runpod_jobs/setup.sh cpu
   source /workspace/venv-cpu/bin/activate
   echo ${HF_TOKEN:+HF_TOKEN is set}  # set via the pod's env var {{ RUNPOD_SECRET_hf_token }};
   export HF_TOKEN=hf_...             #   if nothing printed, export it by hand
   tmux new -s alpha                 # keeps the job alive if your laptop disconnects
   ```
   (tmux: detach `Ctrl-b d`, re-attach `tmux attach -t alpha`.)
4. **Pilot (~5 min, a few cents)** — measures this pod's real download and decode speed and
   projects the whole job:
   ```bash
   R=dolma1_7-no-math-code,dolma1_7-no-reddit,dolma1_7-no-flan,dolma1_6plus,falcon,falcon-and-cc,dclm-baseline,dclm-baseline-qc-7p-fw2,dclm-baseline-25p-dolma1.7-75p,dclm-baseline-50p-dolma1.7-50p,dclm-baseline-75p-dolma1.7-25p
   python -m runpod_jobs.exact_alpha --recipes $R --dry-run --usd-per-hour <pod price>
   python -m runpod_jobs.exact_alpha --recipes $R --pilot 64 --usd-per-hour <pod price>
   # then the dry run again with the measured numbers it prints
   ```
   **Always pass `--recipes $R`**: the pod only has the c4/dolma1_7 results from git, so without
   it the script would redo the 12 recipes already finished on your laptop.
   The pilot prints download GB/s, decode speed and **how busy the vCPUs were**, then a dry-run
   command with the measured numbers — run it to get hours and dollars for this pod.
   * vCPUs busy **≥ 70 %** → CPU-bound: the pod is sized right; add pods to finish sooner.
   * vCPUs busy **< 70 %** → download-bound: you pay for idle CPUs. Deploy smaller pods
     (e.g. 16 vCPU) and more of them — each pod brings its own network connection.
   Pilot slices are cached and reused by the full run. **Paste the pilot output to Claude
   before starting the full run.**
5. **Run**:
   ```bash
   python -m runpod_jobs.exact_alpha --recipes $R --usd-per-hour <pod price> 2>&1 | tee alpha.log
   ```
   Smallest recipes run first; each finished recipe writes
   `results/corpus_alpha_datadecide/exact_100b_<recipe>/` immediately. A progress line
   every 100 slices shows GB/s, tokens/s, % of the data done, ETA and dollars so far/to go.
   * **The pod stops itself when the run ends** — finished, crashed, or stuck (no progress for
     30 min) — so it never bills compute overnight for nothing. Stopping keeps `/workspace`
     (results + cache); only the disk keeps billing (~$0.01/h for 50 GB). Start the pod again
     to copy results, then terminate. (`--no-stop-pod` turns this off; Ctrl-C never stops it.)
   * **Interrupted, crashed or stopped?** Start the pod, run the same command again: finished
     slices are cached in `/workspace/dd_work` and are never downloaded or counted twice.
   * **Sanity check on the first finished recipe**: its `alpha_mle` should be within ~0.001 of the
     cheap window estimate in your laptop's `results/corpus_alpha_datadecide/alpha_summary.csv`
     (that held for dolma1_7: 0.0003). If not, stop and tell Claude.
   **Several pods**: give each pod a part of the list, e.g. pod 1
   `--recipes dolma1_6plus,dolma1_7-no-flan,falcon`, pod 2 the dolma/DCLM mixes, and so on
   (balance by size — the dry run prints each recipe's TB).
6. **Copy results to your Mac** (do it after each few recipes too — results are small):
   * with **runpodctl** (works with either SSH method). On the pod:
     ```bash
     cd /workspace/numberline-datadecide/results/corpus_alpha_datadecide
     tar czf alpha_runpod.tgz exact_100b_* && runpodctl send alpha_runpod.tgz
     ```
     It prints a code; on your Mac, in the repo's `results/corpus_alpha_datadecide/`:
     `runpodctl receive <code> && tar xzf alpha_runpod.tgz`
   * or with **scp** (only with "SSH over exposed TCP"), on the Mac:
     ```bash
     scp -P <port> -i ~/.ssh/id_ed25519 -r \
       "root@<ip>:/workspace/numberline-datadecide/results/corpus_alpha_datadecide/exact_100b_*" \
       results/corpus_alpha_datadecide/
     ```
   Then, on the Mac: `python -m runpod_jobs.exact_alpha --recipes "" --dry-run` rebuilds
   `alpha_seed2.csv` from every summary.
   (Empty `--recipes ""` falls back to "every recipe without a result"; when all 25 are present
   it only rewrites the CSV.) Check the CSV has 25 rows, **then terminate the pod**.

---

## 4. Job B: model beta (GPU pod)

All 25 final-checkpoint betas are already done. Use this for the training-checkpoint
experiment (beta vs numbers seen during training) and the extra seeds.

1. **Deploy**: Pods → Deploy → **GPU** → Community Cloud, **spot** if offered →
   RTX A5000 (~$0.27/h), L4 (~$0.39/h) or RTX 3090/4090; template **RunPod PyTorch**;
   set the **CUDA version filter to ≥ 12.4** (setup installs torch 2.6, built for CUDA 12.4);
   container disk 20 GB, **volume disk 100 GB** (each model checkpoint is a few GB in the HF
   cache).
2. Connect as above, then:
   ```bash
   cd /workspace && git clone -b alpha-sampling https://github.com/mdibrahimawad/numberline-datadecide.git
   cd numberline-datadecide && bash runpod_jobs/setup.sh gpu
   source /workspace/venv-gpu/bin/activate && export HF_HOME=/workspace/hf_cache
   tmux new -s beta
   ```
3. **Pilot one model** and watch the GPU in a second window (`Ctrl-b c`, then `nvidia-smi -l 5`):
   ```bash
   python -m runpod_jobs.beta --models c4 --parallel 1
   ```
   Note the minutes it takes and the GPU memory / utilisation. If one model uses a few GB and
   <50 % of the GPU, raise `--parallel` (4 is a good start on a 24 GB card) until utilisation
   is ~90 % or memory is ~80 % full.
4. **Run**, e.g. 4 recipes × 8 checkpoints:
   ```bash
   python -m runpod_jobs.beta --parallel 4 \
     --models falcon-and-cc-qc-10p,falcon-and-cc-qc-tulu-10p,c4,dolma1_7 \
     --revisions step5000-seed-default,step10000-seed-default,step20000-seed-default,step30000-seed-default,step40000-seed-default,step50000-seed-default,step60000-seed-default,step69369-seed-default
   ```
   (Check the exact revision names first:
   `python -c "from huggingface_hub import list_repo_refs as l; print(sorted(b.name for b in l('allenai/DataDecide-c4-1B').branches))"`.)
   Results: `results/datadecide_runpod/<revision>/<recipe>.json` + `summary.csv`.
5. Copy `results/datadecide_runpod` to your Mac (`tar czf` + `runpodctl send`, as in job A),
   then **terminate**.

---

## 5. Rules for not wasting a dollar

1. **Pilot before every big run** (the scripts have `--pilot` / `--dry-run`); look at the
   projected hours × price before starting the full run.
2. **Terminate, don't stop**, once results are on your laptop. A stopped pod still bills
   its volume disk at double rate.
3. **Interruptible (spot) + Community Cloud** whenever offered — the scripts resume after
   interruptions (restart the stopped pod, rerun the same command).
4. **Parallel pods are free speed** for job A (same pod-hours, less wall time). For job B,
   fill one GPU with `--parallel` before renting a second.
5. **Size to the bottleneck**: download-bound → fewer vCPUs; GPU idle → more `--parallel`.
6. **Run inside tmux**, so a closed laptop lid doesn't kill a running job (it would keep
   billing with nothing running if the job died and you didn't notice).
7. **Never let the balance reach $0**: pods without a network volume are then *terminated* and
   their data is lost. Low-balance alert on, and copy results off as recipes finish (each one is
   written to disk the moment it is done).
8. Don't rent a big GPU (A100/H100) for this project: a 1B model doesn't need it, and the
   cheapest 24 GB card does the same job.
