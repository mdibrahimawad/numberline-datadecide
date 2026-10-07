#!/usr/bin/env bash
# One-time setup inside a fresh RunPod pod. Usage (from /workspace):
#   git clone -b alpha-sampling https://github.com/mdibrahimawad/numberline-datadecide.git
#   cd numberline-datadecide && bash runpod_jobs/setup.sh cpu     # CPU pod: alpha counting
#   cd numberline-datadecide && bash runpod_jobs/setup.sh gpu     # GPU pod: model beta
# Packages go into /workspace/venv-<mode> (the volume disk), so a stopped and
# restarted pod does not reinstall.
set -euo pipefail
MODE="${1:?usage: setup.sh cpu|gpu}"
VENV="/workspace/venv-$MODE"
export HF_HOME=/workspace/hf_cache

if ! command -v tmux >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq tmux htop) >/dev/null 2>&1 || echo "[setup] could not install tmux/htop (not needed)"
fi
python3 -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip

if [ "$MODE" = cpu ]; then
  # modal is only imported for shared helper functions; no Modal account is used
  pip install -q "numpy>=2.0" "requests>=2.32" "tokenizers>=0.20" "orjson>=3.10" \
                 "huggingface_hub>=0.28" "scipy>=1.11" modal
  python -m runpod_jobs.exact_alpha --help >/dev/null
else
  # same pins as modal_app/datadecide_app.py (ai2-olmo needs transformers<4.50)
  pip install -q "torch==2.6.0" "transformers==4.49.0" "ai2-olmo==0.6.0" "accelerate>=0.33" \
                 "scikit-learn>=1.4" "scipy>=1.11" "numpy>=1.26,<2.3" "huggingface_hub>=0.24" \
                 "safetensors>=0.4.3" "datasets>=2.20"
  python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('[setup] GPU:', torch.cuda.get_device_name(0))"
fi

grep -q "HF_HOME" ~/.bashrc 2>/dev/null || {
  echo "export HF_HOME=/workspace/hf_cache" >> ~/.bashrc
  echo "source $VENV/bin/activate" >> ~/.bashrc
}
echo "[setup] done. Next: export HF_TOKEN=hf_... (or set it as a RunPod secret), then see docs/runpod.md"
