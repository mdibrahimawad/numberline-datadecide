#!/usr/bin/env bash
# The phases of the brief (section 10), with their gates. Every step is resumable: rerun the same
# command after a pod restart. Run inside tmux:  tmux new -s tt
#   bash scripts/run_pipeline.sh all            (everything, stops at a failed gate)
#   bash scripts/run_pipeline.sh data|validate|pilot|full|p2|report
set -euo pipefail
cd "$(dirname "$0")/.."
export TT_WORK=${TT_WORK:-/workspace/tt} HF_HOME=${HF_HOME:-/workspace/hf}
PRIMARY=pythia-1.4b-deduped
SMALL=pythia-410m-deduped
phase=${1:?phase}
case $phase in
  data)      # P1 + P2, token-length confound gate with the Pythia tokenizer
    python -m tt.data build --tokenizer EleutherAI/$PRIMARY --inputs P1 P2 ;;
  validate)  # synthetic tests 1-6 -> results/validation, figures/fig0_validation.png
    python -m tt.validate --d 2048 ;;
  pilot)     # 410m, P1, steps 0/3000/143000: step 0 at chance, L1 high in middle layers, timing
    python -m tt.extract --model $SMALL --inputs P1 --steps 0 3000 143000
    python -m tt.analyze --model $SMALL --input P1 --stages cells aggregate floor
    python -m tt.report ;;
  full)      # 1.4b then 410m, P1, all 15 checkpoints
    for M in $PRIMARY $SMALL; do
      python -m tt.extract --model $M --inputs P1
      python -m tt.analyze --model $M --input P1 --stages cells velocity aggregate floor
      python -m tt.figures --model $M --input P1
    done ;;
  p2)        # replication on 1.4b; add 6.9b with: MODEL=pythia-6.9b DTYPE=bfloat16 (or float32 on 80 GB)
    M=${MODEL:-$PRIMARY}
    python -m tt.extract --model $M --inputs P2 --dtype ${DTYPE:-float32}
    python -m tt.analyze --model $M --input P2 --stages cells velocity aggregate floor
    python -m tt.figures --model $M --input P2 ;;
  all)       # everything in order; stops at the first failed gate
    bash "$0" data
    python - <<'PY'
import json, glob, sys
bad = [f for f in glob.glob("results/confounds_*.json") if not json.load(open(f))["passes"]]
sys.exit(f"GATE FAILED (length confound): {bad}" if bad else 0)
PY
    bash "$0" validate
    python -c "import json,sys; r=json.load(open('results/validation/validation.json')); sys.exit(0 if r['all_pass'] else 'GATE FAILED: synthetic validation, see results/validation/validation.json')"
    bash "$0" pilot
    python - <<'PY'
import pandas as pd, sys
on = pd.read_csv("results/pythia-410m-deduped/P1/onset.csv")
truth = ["L1_auc", "L2_auc", "L1_within_auc", "L3_auc", "T1_acc", "T3b_within_truth_homophily",
         "T3c_cross_truth_homophily", "B1_loglik_auc"]   # T3a is topic structure: expected at step 0
bad = on[(on.sig_at_step0 == True) & on.metric.isin(truth)]
if len(bad):
    sys.exit("GATE FAILED: significant at step 0 (untrained model) in the pilot:\n" + bad.to_string())
print("pilot gate passed")
PY
    bash "$0" full
    bash "$0" p2
    bash "$0" report
    echo "ALL DONE: results/ and figures/" ;;
  report)
    python -m tt.report
    python -m tt.novelty || echo "novelty check failed (network?)" ;;
  *) echo "unknown phase $phase"; exit 1 ;;
esac
