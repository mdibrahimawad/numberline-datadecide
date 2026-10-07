# Running this project on RunPod

Replaces Modal. Two jobs, two kinds of machine:

| Job | Script | Machine | Why |
|---|---|---|---|
| Exact 100B alpha of the 11 recipes still missing | `runpod_jobs/exact_alpha.py` | **CPU pod**, 32 vCPU | download + tokenizer decode; no GPU work |
| Model beta (final models, training checkpoints) | `runpod_jobs/beta.py` | **GPU pod**, one 24 GB card (RTX A5000 / L4 / 3090 / 4090) | 1B models, several at once on one card |

Prices below are RunPod's public list prices found in Oct 2026; **the deploy page shows the real
price before you click Deploy, always check it there.**

---

## 1. RunPod in 5 minutes (what the words mean)

* **Pod** = a rented Linux machine (really a Docker container on a host) that you SSH into
  and run commands on, like your laptop's terminal. Billed **per second** while it runs.
  This is what we use for everything.
* **GPU pod / CPU pod** = with or without a GPU. CPU pods are cheaper and come in "flavours":
  `cpu3c`/`cpu5c` = compute-optimised (2 GB RAM per vCPU), `cpu3g`/`cpu5g` = general (4 GB/vCPU),
  `cpu3m`/`cpu5m` = memory (8 GB/vCPU). 5 = newer, faster CPUs.
* **Secure Cloud vs Community Cloud** = RunPod's own datacentres vs vetted third-party
  hosts. Community is cheaper (e.g. H100 PCIe $1.99 vs $2.89/h); fine for us — our jobs
  resume after any interruption.
* **On-demand vs Spot (interruptible)** = spot is 50-70 % cheaper but can be taken away
  with a **5-second** warning. Our scripts cache every finished piece on disk, so a spot
  interruption costs only the minutes in flight. Use spot whenever it is offered.
* **Template** = the starting software image. Use **RunPod PyTorch** for GPU pods and the
  default Ubuntu/CPU template for CPU pods.
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
* No fees for downloading/uploading data (ingress/egress).

---

## 2. One-time setup (on your Mac)

1. **Account / team.** If your professor's funding is on a RunPod *team*, ask them to invite
   your email, then switch to the team in the top-left account menu so pods bill to it.
2. **SSH key** (lets you log in to pods):
   ```bash
   ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -C "runpod"   # press Enter 3x
   cat ~/.ssh/id_ed25519.pub                                        # copy this line
   ```
   RunPod console → **Settings → SSH Public Keys** → paste → Save. Every pod you create gets it.
3. **Hugging Face token** (raises download rate limits): RunPod console → **Secrets** →
   create secret `hf_token` with your `hf_...` token. (Or just `export HF_TOKEN=...` inside
   the pod each time.)
4. *(optional)* **CLI** for scripting pods: `brew install runpod/runpodctl/runpodctl`, then
   `runpodctl config --apiKey <key from Settings → API Keys>`. Not required — the web
   console does everything below.

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
1. **Deploy**: console → **Pods → Deploy** → **CPU** tab → `cpu5c` (or `cpu3c`) with
   **32 vCPU** (64 GB RAM — the membership step for the biggest recipe needs ~10 GB) →
   container disk 20 GB, **volume disk 50 GB** → pick a **US** region (Hugging Face's
   CDN is fastest there) → Deploy. Note the price per hour shown.
2. **Connect**: pod → **Connect** → copy the **"SSH over exposed TCP"** command, e.g.
   `ssh root@203.0.113.7 -p 22114 -i ~/.ssh/id_ed25519`, and run it on your Mac.
3. **Install** (inside the pod, ~2 min):
   ```bash
   cd /workspace
   git clone -b alpha-sampling https://github.com/mdibrahimawad/numberline-datadecide.git
   #   private repo? use https://<github-username>:<personal-access-token>@github.com/...
   cd numberline-datadecide && bash runpod_jobs/setup.sh cpu
   source /workspace/venv-cpu/bin/activate
   export HF_TOKEN=hf_...            # or: export HF_TOKEN=$RUNPOD_SECRET_hf_token
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
   Read the projection:
   * **download-bound** (GB/s is the limit) → a pod with fewer vCPUs costs less for the same
     hours; and split the list over 2-4 pods to finish sooner.
   * **CPU-bound** → keep 32 vCPU, split over pods to finish sooner.
   Pilot slices are cached and reused by the full run.
5. **Run**:
   ```bash
   python -m runpod_jobs.exact_alpha --recipes $R --usd-per-hour <pod price> 2>&1 | tee alpha.log
   ```
   Smallest recipes run first; each finished recipe writes
   `results/corpus_alpha_datadecide/exact_100b_<recipe>/` immediately. A progress line
   every 100 slices shows GB/s, tokens/s and dollars so far. Interrupted? Run the same
   command again — it resumes from the cache in `/workspace/dd_work`.
   **Several pods**: give each pod a part of the list, e.g. pod 1
   `--recipes dolma1_6plus,dolma1_7-no-flan,falcon`, pod 2 the dolma/DCLM mixes, and so on
   (balance by size — the dry run prints each recipe's TB).
6. **Copy results to your Mac** (run on the Mac, in the repo):
   ```bash
   scp -P <port> -i ~/.ssh/id_ed25519 -r \
     "root@<ip>:/workspace/numberline-datadecide/results/corpus_alpha_datadecide/exact_100b_*" \
     results/corpus_alpha_datadecide/
   python -m runpod_jobs.exact_alpha --recipes "" --dry-run   # rebuilds alpha_seed2.csv from all summaries
   ```
   (Empty `--recipes ""` falls back to "every recipe without a result"; when all 25 are present
   it only rewrites the CSV.) Check the CSV has 25 rows, **then terminate the pod**.

---

## 4. Job B: model beta (GPU pod)

All 25 final-checkpoint betas are already done. Use this for the training-checkpoint
experiment (beta vs numbers seen during training) and the extra seeds.

1. **Deploy**: Pods → Deploy → **GPU** → Community Cloud, **spot** if offered →
   RTX A5000 (~$0.27/h), L4 (~$0.39/h) or RTX 3090/4090; template **RunPod PyTorch**;
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
5. `scp -r` the `results/datadecide_runpod` folder to your Mac, then **terminate**.

---

## 5. Rules for not wasting a dollar

1. **Pilot before every big run** (the scripts have `--pilot` / `--dry-run`); look at the
   projected hours × price before starting the full run.
2. **Terminate, don't stop**, once results are on your laptop. A stopped pod still bills
   its volume disk at double rate.
3. **Spot + Community Cloud** whenever available — the scripts resume after interruptions.
4. **Parallel pods are free speed** for job A (same pod-hours, less wall time). For job B,
   fill one GPU with `--parallel` before renting a second.
5. **Size to the bottleneck**: download-bound → fewer vCPUs; GPU idle → more `--parallel`.
6. **Run inside tmux**, so a closed laptop lid doesn't kill a running job (it would keep
   billing with nothing running if the job died and you didn't notice).
7. **Keep an eye on the balance** in the console's Billing page; pods stop when credit runs out
   (volume disks are kept for a while but are deleted eventually — copy results early; each
   finished recipe is written immediately).
8. Don't rent a big GPU (A100/H100) for this project: a 1B model doesn't need it, and the
   cheapest 24 GB card does the same job.
