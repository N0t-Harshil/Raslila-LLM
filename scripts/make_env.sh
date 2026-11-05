#!/usr/bin/env bash
# Fresh, fast Python environment for the scaling study.
#
#   bash scripts/make_env.sh                  # creates ~/raslila-fast
#   bash scripts/make_env.sh /path/to/env     # somewhere else
#   source ~/raslila-fast/bin/activate        # then, in EVERY new terminal / tmux window
#
# What it does:
#   - picks the newest PyTorch build your NVIDIA driver supports
#     (torch 2.7.1 + CUDA 12.6 if the driver allows, else 2.6.0/12.4, else 2.5.1/12.1)
#   - installs only what the study needs (no quantum / legacy extras)
#   - keeps torch.compile's kernel cache inside the env, so restarts don't recompile
#   - checks: GPU visible, bf16, flash attention, torch.compile works (needs gcc)
set -euo pipefail
ENV="${1:-$HOME/raslila-fast}"
cd "$(dirname "$0")/.."

# Use a plain python, not one from another venv.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in /opt/conda/bin/python3 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
  done
fi
echo "base python: $PY ($("$PY" --version 2>&1))"

CUDA_DRV="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$' || true)"
if [ -z "$CUDA_DRV" ]; then echo "nvidia-smi not found or no GPU: cannot pick a CUDA build"; exit 1; fi
ge() { awk -v a="$1" -v b="$2" 'BEGIN{split(a,x,".");split(b,y,".");exit !((x[1]>y[1])||(x[1]==y[1]&&x[2]>=y[2]))}'; }
if ge "$CUDA_DRV" 12.6; then TORCH=2.7.1; CU=cu126
elif ge "$CUDA_DRV" 12.4; then TORCH=2.6.0; CU=cu124
else TORCH=2.5.1; CU=cu121; fi
echo "driver supports CUDA $CUDA_DRV -> torch $TORCH ($CU)"

if [ -d "$ENV" ]; then echo "$ENV exists: reusing it (delete it first for a clean install)"; fi
"$PY" -m venv "$ENV"
# shellcheck disable=SC1091
source "$ENV/bin/activate"
python -m pip install -q --upgrade pip wheel
python -m pip install "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$CU"
python -m pip install numpy pyarrow datasets tokenizers "huggingface_hub>=0.34" pyyaml matplotlib scipy

# Persist compiled kernels between runs (a resumed run then starts in seconds).
if ! grep -q TORCHINDUCTOR_CACHE_DIR "$ENV/bin/activate"; then
  cat >> "$ENV/bin/activate" <<EOF2
export TORCHINDUCTOR_CACHE_DIR="$ENV/inductor-cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export PYTHONUNBUFFERED=1
EOF2
fi
source "$ENV/bin/activate"

echo
echo "== checks =="
python - <<'EOF2'
import torch, numpy, pyarrow, datasets, tokenizers, yaml
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| numpy", numpy.__version__,
      "| pyarrow", pyarrow.__version__, "| datasets", datasets.__version__, "| tokenizers", tokenizers.__version__)
assert torch.cuda.is_available(), "no GPU visible to torch"
n = torch.cuda.device_count()
print("GPUs:", n, "x", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
import torch.nn.functional as F
q = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.bfloat16)
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    F.scaled_dot_product_attention(q, q, q, is_causal=True)
print("flash attention: OK")
try:
    f = torch.compile(lambda x: torch.nn.functional.silu(x) * x + 1)
    f(torch.randn(1024, device="cuda")); torch.cuda.synchronize()
    print("torch.compile: OK")
except Exception as e:
    print("torch.compile: FAILED (%s: %s)" % (type(e).__name__, str(e).splitlines()[0][:200]))
    print("  usually a missing C compiler: conda install -y -c conda-forge gxx_linux-64  (or apt-get install build-essential)")
EOF2
echo
echo "Done. In every new terminal / tmux window run:"
echo "    source $ENV/bin/activate"
