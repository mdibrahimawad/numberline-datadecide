#!/usr/bin/env bash
# E09 number counts for one Paloma training corpus, on a CPU pod. One pod per job; several
# pods may run "dolma" (its 8 parts are shared out through claims on Hugging Face).
#
#   bash runpod_jobs/paloma_counts.sh <job> [check|run] [max hours]
#
#   job:   c4 | mc4 | pile | falcon-refinedweb | redpajama   (random sample, runpod_jobs.corpus_sample)
#          dolma                                             (exact training stream, runpod_jobs.exact_alpha)
#   check: list the files, count a few, print speed and stop (a few cents)
#   run:   the full count (default); uploads to the private HF dataset numberline-alpha-results,
#          then deletes this pod. It deletes itself after [max hours] (default 3) whatever happens.
#
# Needs HF_TOKEN (a write token) in the environment. Results come back to the Mac with
#   python -m runpod_jobs.fetch_results --kind paloma
set -euo pipefail
JOB="${1:?usage: paloma_counts.sh <c4|mc4|pile|falcon-refinedweb|redpajama|dolma> [check|run] [max hours]}"
MODE="${2:-run}"
HOURS="${3:-3}"
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?export HF_TOKEN=... (a Hugging Face WRITE token) first}"
[ -x /workspace/venv-cpu/bin/python ] && /workspace/venv-cpu/bin/python -c "import zstandard, pyarrow" 2>/dev/null \
  || bash runpod_jobs/setup.sh cpu
# shellcheck disable=SC1091
source /workspace/venv-cpu/bin/activate
export HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p logs

if [ "$JOB" = dolma ]; then
  PARTS=$(python -c "import json; print(','.join(json.load(open('configs/paloma/paloma_dolma_exact_map.json'))['recipes']))")
  CMD=(python -m runpod_jobs.exact_alpha --data-map configs/paloma/paloma_dolma_exact_map.json
       --recipes "$PARTS" --seed 6198 --out-dir results/paloma_counts --work-dir /workspace/paloma_work)
  if [ "$MODE" = check ]; then
    "${CMD[@]}" --dry-run
    "${CMD[@]}" --pilot 32
    exit 0
  fi
  "${CMD[@]}" --upload-hf numberline-alpha-results --delete-pod-when-done --max-hours "$HOURS" 2>&1 | tee "logs/$JOB.log"
else
  CMD=(python -m runpod_jobs.corpus_sample --corpus "$JOB")
  if [ "$MODE" = check ]; then
    "${CMD[@]}" --dry-run
    "${CMD[@]}" --pilot 2
    exit 0
  fi
  "${CMD[@]}" --upload-hf numberline-alpha-results --delete-pod-when-done --max-hours "$HOURS" 2>&1 | tee "logs/$JOB.log"
fi
