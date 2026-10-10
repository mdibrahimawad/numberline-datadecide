#!/usr/bin/env bash
# E09 step 3: β of the 6 Paloma baselines on ONE GPU pod, unattended, in one command:
#   setup -> pilot (c4) -> sanity check -> the other 5 -> upload -> the pod deletes itself.
#
#   export HF_TOKEN=...            # a Hugging Face WRITE token
#   bash runpod_jobs/paloma_beta.sh
#
# Any GPU with >= 20 GB that PyTorch 2.6 supports (A4500, A5000, 3090, 4090, L4, A6000, A40,
# L40S); not RTX 50xx / Blackwell. 3 models at once below 40 GB, all 6 at once above.
# If the pilot model fails the check, its log goes to <HF repo>/paloma_beta_failed/ and the
# pod deletes itself without running the rest. Results come back to the Mac with
#   python -m runpod_jobs.fetch_results --kind beta
set -uo pipefail
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?export HF_TOKEN=... (a Hugging Face WRITE token) first}"
REV=step35000-unsharded
REPO=numberline-beta-results
MODELS=allenai/paloma-1b-baseline-c4,allenai/paloma-1b-baseline-mc4,allenai/paloma-1b-baseline-pile,allenai/paloma-1b-baseline-falcon-refinedweb,allenai/paloma-1b-baseline-dolma,allenai/paloma-1b-baseline-redpajama
OUT=results/beta_fine/$REV
mkdir -p logs

/workspace/venv-gpu/bin/python -c "import torch, hf_olmo; assert torch.cuda.is_available()" 2>/dev/null \
  || bash runpod_jobs/setup.sh gpu || { echo "[paloma-beta] setup failed: paste the lines above to Claude"; exit 1; }
# shellcheck disable=SC1091
source /workspace/venv-gpu/bin/activate
export HF_HOME=/workspace/hf_cache

delete_pod() { python -c "from runpod_jobs.pod import terminate_this_pod; terminate_this_pod('$1')"; }

fail() {  # keep the evidence, then stop billing
  echo "[paloma-beta] FAILED: $1"
  python - "$1" <<'EOF'
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = f"{api.whoami()['name']}/numberline-beta-results"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
files = [p for p in Path("results/beta_fine/step35000-unsharded").rglob("*") if p.is_file() and p.suffix in (".log", ".json")]
files += [p for p in Path("logs").glob("*.log")]
for p in files:
    api.upload_file(path_or_fileobj=str(p), path_in_repo=f"paloma_beta_failed/{p.name}", repo_id=repo, repo_type="dataset")
print(f"[paloma-beta] {len(files)} log files uploaded to {repo}/paloma_beta_failed/ ({sys.argv[1]})")
EOF
  delete_pod "paloma beta failed: $1"
  exit 1
}

MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
PAR=$([ "${MEM:-0}" -ge 40000 ] && echo 6 || echo 3)
echo "[paloma-beta] GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), ${MEM} MiB -> $PAR models at once"

# 1. pilot: one model, checked before anything else runs (at most 45 min)
echo "[paloma-beta] pilot: allenai/paloma-1b-baseline-c4"
timeout 45m python -m runpod_jobs.beta --fine --models allenai/paloma-1b-baseline-c4 --revisions $REV \
  --parallel 1 --upload-hf $REPO 2>&1 | tee logs/paloma_beta_pilot.log
[ -f "$OUT/allenai_paloma-1b-baseline-c4.json" ] || fail "the pilot produced no result"
python -m runpod_jobs.check_beta "$OUT/allenai_paloma-1b-baseline-c4.json" || fail "the pilot result looks wrong"

# 2. the other 5 (c4 is done and skipped); uploads each, then the pod deletes itself
python -m runpod_jobs.beta --fine --models $MODELS --revisions $REV --parallel "$PAR" \
  --upload-hf $REPO --delete-pod-when-done --max-hours 2 2>&1 | tee logs/paloma_beta.log
