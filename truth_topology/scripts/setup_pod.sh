#!/usr/bin/env bash
# One-time setup on a RunPod GPU pod (Runpod PyTorch template). Everything big lives on /workspace.
#   bash truth_topology/scripts/setup_pod.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export TT_WORK=${TT_WORK:-/workspace/tt}
mkdir -p "$TT_WORK"
if [ ! -d /workspace/venv-tt ]; then
  python3 -m venv /workspace/venv-tt --system-site-packages   # reuse the template's CUDA torch
fi
source /workspace/venv-tt/bin/activate
pip install -q -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
cat <<MSG
Add to your shell (or ~/.bashrc):
  source /workspace/venv-tt/bin/activate
  export TT_WORK=$TT_WORK HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
MSG
