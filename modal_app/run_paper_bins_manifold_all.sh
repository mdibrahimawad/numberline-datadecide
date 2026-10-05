#!/usr/bin/env bash
set -euo pipefail

# Re-run the manifold diagnostic on the exact four magnitude bins used in the
# paper PCA experiment. Layer 0 is excluded because it is the embedding output.

COMMON_ARGS=(
  --trust-remote-code
  --groups "1,2,3,4"
  --k 30
  --num-examples 3
  --runs 3
  --dtype float16
  --min-select-layer 1
  --save-visualization
)

.venv/bin/modal run modal_app/manifold_geometry_app.py \
  --models "EleutherAI/pythia-2.8b,tiiuae/falcon-rw-1b,tiiuae/falcon-rw-7b,togethercomputer/RedPajama-INCITE-Base-3B-v1,togethercomputer/RedPajama-INCITE-7B-Base" \
  "${COMMON_ARGS[@]}" \
  --results-dir results/manifold_geometry/paper_bins_batch_general_best_nonzero

.venv/bin/modal run modal_app/manifold_geometry_app.py \
  --models "allenai/OLMo-7B,allenai/OLMo-7B-Twin-2T,gpt2-large" \
  --model-revisions "step452000-tokens2000B,," \
  "${COMMON_ARGS[@]}" \
  --results-dir results/manifold_geometry/paper_bins_batch_olmo_gpt2l_best_nonzero

.venv/bin/modal run modal_app/manifold_geometry_app.py \
  --models "bigcode/starcoderbase-1b,bigcode/starcoderbase-3b,bigcode/starcoderbase-7b" \
  "${COMMON_ARGS[@]}" \
  --results-dir results/manifold_geometry/paper_bins_batch_starcoder_best_nonzero

.venv/bin/python - <<'PY'
from pathlib import Path
import json
from src.manifold_geometry import write_manifold_artifacts

base = Path("results/manifold_geometry")
payloads = []
for name in [
    "paper_bins_batch_general_best_nonzero",
    "paper_bins_batch_olmo_gpt2l_best_nonzero",
    "paper_bins_batch_starcoder_best_nonzero",
]:
    payloads.extend(json.loads((base / name / "manifold_geometry_payloads.json").read_text()))

out = base / "paper_bins_all_with_gpt2l_best_nonzero"
paths = write_manifold_artifacts(payloads, out)
print(f"merged {len(payloads)} payloads into {out}")
for key, path in paths.items():
    print(f"{key}: {path}")
PY
