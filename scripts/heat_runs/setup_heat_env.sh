#!/usr/bin/env bash
# Stage code + environment for the heat-equation experiments on a Linux GPU machine or cluster
# login node:  BASE=/path/to/heat_repro [PYBOOT=python3] bash setup_heat_env.sh
# Everything lives under $BASE; nothing is written to ~/.local or ~/.cache. Then run
# run_heat_gpu.sh / heat_repro.sbatch (paper reproduction) or run_embed_gpu.sh (embed comparison).
# COMMIT picks the ginjax version that is checked out. The default, 93879da, has a heat code path
# identical to the paper snapshot d7b20d8 and is what the reproduction was run with; it predates
# scripts/heat_embed.py, so for the embed comparison set COMMIT to a commit that contains it.
set -euo pipefail

BASE=${BASE:-$HOME/heat_repro}
COMMIT=${COMMIT:-93879da}

mkdir -p "$BASE"
cd "$BASE"
export UV_CACHE_DIR="$BASE/.uvcache"
export UV_PYTHON_INSTALL_DIR="$BASE/.pythons"
export UV_PYTHON_INSTALL_BIN=0

# uv, contained in its own venv
[ -x uvtool/bin/uv ] || { "${PYBOOT:-/usr/bin/python3}" -m venv uvtool && uvtool/bin/pip install -q uv; }
UV="$BASE/uvtool/bin/uv"

# code
[ -d ginjax ] || git clone -q https://github.com/WilsonGregory/ginjax.git
git -C ginjax fetch -q origin
git -C ginjax checkout -q "$COMMIT"
echo "ginjax at $(git -C ginjax rev-parse --short HEAD)"

# Pinned deps from uv.lock, minus torch's CUDA-13 stack (torch is only used for DataLoader, and
# its cu13 libs would sit next to jax's cu12 libs). Then CPU torch at the pinned version and the
# CUDA-12 jax plugin at the pinned jax version.
(cd ginjax && "$UV" export --frozen --no-hashes --no-emit-project --no-dev --no-header) \
  | grep -v -E '^(torch|triton|nvidia-|cuda-)' | grep -v -E '^\s*#' > requirements.lock.txt
TORCH_VER=$(cd ginjax && "$UV" export --frozen --no-hashes --no-dev | grep -E '^torch==' | cut -d= -f3 | cut -d' ' -f1)
JAX_VER=$(grep -E '^jax==' requirements.lock.txt | cut -d= -f3 | cut -d' ' -f1)
echo "pins: jax==$JAX_VER torch==$TORCH_VER (cpu)"

[ -x venv/bin/python ] || "$UV" venv -q --python 3.11 --python-preference only-managed venv
PY="$BASE/venv/bin/python"
"$UV" pip install -q --python "$PY" -r requirements.lock.txt
"$UV" pip install -q --python "$PY" "torch==$TORCH_VER" --index-url https://download.pytorch.org/whl/cpu
"$UV" pip install -q --python "$PY" "jax[cuda12]==$JAX_VER"
"$UV" pip install -q --python "$PY" --no-deps -e ginjax

"$PY" - <<'EOF'
import platform, jax, jaxlib, equinox, optax, torch, ginjax
from importlib.metadata import version
print("python", platform.python_version(), "| jax", jax.__version__, "| jaxlib", jaxlib.__version__,
      "| cuda12 plugin", version("jax-cuda12-plugin"), "| equinox", equinox.__version__,
      "| optax", optax.__version__, "| torch", torch.__version__)
EOF
echo "setup OK: $BASE"
