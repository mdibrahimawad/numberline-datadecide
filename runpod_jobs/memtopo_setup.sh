#!/usr/bin/env bash
# E10 (memtopo) setup on a RunPod GPU pod (one A100/H100 80 GB, >= 32 vCPU, >= 300 GB volume).
# Usage, from /workspace:
#   git clone -b <branch> https://github.com/mdibrahimawad/numberline-datadecide.git
#   cd numberline-datadecide && bash runpod_jobs/memtopo_setup.sh
# Separate from setup.sh: E10 uses Pythia (GPT-NeoX in recent transformers), not ai2-olmo pins.
set -euo pipefail
VENV=/workspace/venv-memtopo
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME" /workspace/refs

PY=""
for c in python3.11 python3.12 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ] || ! "$PY" -m venv --system-site-packages "$VENV" 2>/dev/null; then
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_PYTHON_INSTALL_DIR=/workspace/uv-python uv venv --seed --python 3.11 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch transformers accelerate datasets huggingface_hub hf_transfer numpy scipy scikit-learn \
               pandas pyarrow matplotlib ripser persim joblib zstandard lightgbm shap pyyaml tabulate requests pytest
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('[memtopo] GPU:', torch.cuda.get_device_name(0))"
python -m pytest -q tests/test_memtopo.py
python -m memtopo.cli refs || echo "[memtopo] some reference repos could not be cloned (recorded in gates.json)"

grep -q "venv-memtopo" ~/.bashrc 2>/dev/null || {
  echo "export HF_HOME=/workspace/hf" >> ~/.bashrc
  echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> ~/.bashrc
  echo "source $VENV/bin/activate" >> ~/.bashrc
}
echo "[memtopo] done. Next: export HF_TOKEN=hf_...; see experiments/E10_memorization_topology/README.md"
