#!/usr/bin/env bash
# E09 number counts for the Paloma training corpora, on a CPU pod.
#
#   bash runpod_jobs/paloma_counts.sh <job> [check|run] [max hours]
#
#   job:   all                                               every corpus below, one after another, on this pod
#          c4,mc4,dolma                                      a comma list: those, in that order, on this pod
#          c4 | mc4 | pile | falcon-refinedweb | redpajama   (random sample, runpod_jobs.corpus_sample)
#          dolma                                             (exact training stream, runpod_jobs.exact_alpha;
#                                                             several pods may run it: its 8 parts are shared
#                                                             out through claims on Hugging Face)
#   check: list the files, count a few, print speed and stop (a few cents)
#   run:   the full count (default); each result is uploaded to the private HF dataset
#          numberline-alpha-results; at the end the pod deletes itself (only if everything
#          succeeded). It deletes itself after [max hours] whatever happens
#          (default 3 for one job, 10 for several).
#
# Needs HF_TOKEN (a write token) in the environment. Results come back to the Mac with
#   python -m runpod_jobs.fetch_results --kind paloma
set -uo pipefail
JOB="${1:?usage: paloma_counts.sh <all|c4|mc4|pile|falcon-refinedweb|redpajama|dolma> [check|run] [max hours]}"
MODE="${2:-run}"
MULTI=$([ "$JOB" = all ] || [[ "$JOB" == *,* ]] && echo 1 || echo "")
HOURS="${3:-$([ -n "$MULTI" ] && echo 10 || echo 3)}"
CORPORA=(c4 mc4 pile falcon-refinedweb redpajama)
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?export HF_TOKEN=... (a Hugging Face WRITE token) first}"
[ -x /workspace/venv-cpu/bin/python ] && /workspace/venv-cpu/bin/python -c "import zstandard, pyarrow" 2>/dev/null \
  || bash runpod_jobs/setup.sh cpu || exit 1
# shellcheck disable=SC1091
source /workspace/venv-cpu/bin/activate
export HF_HOME=/workspace/hf_cache
mkdir -p logs
DMAP=configs/paloma/paloma_dolma_exact_map.json
# one job's own "delete the pod" is used only when it is the only job; `all` deletes at the end
SOLO=$([ -n "$MULTI" ] && echo "" || echo "--delete-pod-when-done")

one() {  # one job in MODE; returns its exit code
  local job=$1
  if [ "$job" = dolma ]; then
    local parts
    parts=$(python -c "import json; print(','.join(json.load(open('$DMAP'))['recipes']))")
    local cmd=(python -m runpod_jobs.exact_alpha --data-map "$DMAP" --recipes "$parts" --seed 6198
               --out-dir results/paloma_counts --work-dir /workspace/paloma_work)
    if [ "$MODE" = check ]; then
      "${cmd[@]}" --dry-run && "${cmd[@]}" --pilot 32
    else
      "${cmd[@]}" --upload-hf numberline-alpha-results --no-stop-pod $SOLO --max-hours "$HOURS"
    fi
  else
    local cmd=(python -m runpod_jobs.corpus_sample --corpus "$job")
    if [ "$MODE" = check ]; then
      "${cmd[@]}" --dry-run && "${cmd[@]}" --pilot 2
    else
      "${cmd[@]}" --upload-hf numberline-alpha-results $SOLO --max-hours "$HOURS"
    fi
  fi
}

if [ "$JOB" = all ]; then
  JOBS=("${CORPORA[@]}" dolma)
else
  IFS=, read -r -a JOBS <<< "$JOB"
fi
declare -A RESULT
for j in "${JOBS[@]}"; do
  echo "===== $j ($MODE) ====="
  one "$j" 2>&1 | tee "logs/$j.$MODE.log"
  rc=${PIPESTATUS[0]}
  RESULT[$j]=$([ "$rc" = 0 ] && echo OK || echo FAILED)
done
echo "===== summary ($MODE) ====="
failed=0
for j in "${JOBS[@]}"; do
  echo "  $j: ${RESULT[$j]}"
  [ "${RESULT[$j]}" = OK ] || failed=1
done
if [ -n "$MULTI" ] && [ "$MODE" = run ]; then
  if [ $failed = 0 ]; then
    python -c "from runpod_jobs.pod import terminate_this_pod; terminate_this_pod('all Paloma counts uploaded')"
  else
    echo "  some jobs failed: the pod is kept (it still deletes itself after $HOURS h). Paste the summary to Claude."
  fi
fi
exit $failed
