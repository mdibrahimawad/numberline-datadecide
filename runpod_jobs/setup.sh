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

# bare Ubuntu images lack some of these; RunPod's official images have most of them
need=""
for pkg in tmux htop git; do command -v "$pkg" >/dev/null || need="$need $pkg"; done
python3 -c "import ensurepip" 2>/dev/null || need="$need python3-venv python3-pip"
echo "[setup] step 1/3: system tools (git, tmux)..."
if [ -n "$need" ]; then
  echo "[setup] apt-get install$need (1-3 min)"
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $need) >/dev/null \
    || echo "[setup] apt-get failed for:$need (tmux/htop are optional)"
fi
# a Python >= 3.10 (Ubuntu 20.04 images ship 3.8): use one that exists, else let uv fetch 3.11
echo "[setup] step 2/3: Python >= 3.10..."
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys, venv; assert sys.version_info >= (3, 10)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -n "$PY" ] && "$PY" -m venv --system-site-packages "$VENV" 2>/dev/null; then
  echo "[setup] using $($PY --version)"
else
  echo "[setup] no usable Python >= 3.10 here: installing Python 3.11 with uv"
  rm -rf "$VENV"
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if ! command -v uv >/dev/null; then
    command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl) >/dev/null || true
    curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1 \
      || python3 -m pip install -q --user uv \
      || (apt-get install -y -qq python3-pip >/dev/null && python3 -m pip install -q --user uv)
  fi
  UV_PYTHON_INSTALL_DIR=/workspace/uv-python uv venv --seed --python 3.11 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
echo "[setup] step 3/3: Python packages (3-10 min on a small pod; progress below)"
pip install -q --upgrade pip

if [ "$MODE" = cpu ]; then
  # modal is only imported for shared helper functions; no Modal account is used
  pip install --progress-bar on "numpy>=2.0" "requests>=2.32" "tokenizers>=0.20" "orjson>=3.10" \
                 "huggingface_hub>=0.28" "scipy>=1.11" modal
  python -m runpod_jobs.exact_alpha --help >/dev/null
else
  # same pins as modal_app/datadecide_app.py (ai2-olmo needs transformers<4.50)
  pip install -q "torch==2.6.0" "transformers==4.49.0" "ai2-olmo==0.6.0" "accelerate>=0.33" \
                 "scikit-learn>=1.4" "scipy>=1.11" "numpy>=1.26,<2.3" "huggingface_hub>=0.24" \
                 "safetensors>=0.4.3" "datasets>=2.20" hf_transfer
  python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('[setup] GPU:', torch.cuda.get_device_name(0))"
fi

grep -q "HF_HOME" ~/.bashrc 2>/dev/null || {
  echo "export HF_HOME=/workspace/hf_cache" >> ~/.bashrc
  echo "source $VENV/bin/activate" >> ~/.bashrc
}
echo "[setup] done. Next: export HF_TOKEN=hf_... (or set it as a RunPod secret), then see docs/runpod.md"
