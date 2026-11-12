#!/usr/bin/env bash
# Launch scripts/heat_embed.py on one GPU of a shared workstation. Launch detached:
#   GPU=0 LOW_D=2 TRIALS=0,1,2 OUT=$HOME/heat_repro/embed_2to3 \
#     setsid nohup bash run_embed_gpu.sh > LOG 2>&1 < /dev/null &
# Optional: DATA (default $OUT/data), METHODS (default baseline,nepo,embedrand), EXTRA (more args,
# e.g. "--lr-map lr_map.json" from scripts/heat_select_lr.py).
# For 1D->2D next to a run_heat_gpu.sh run, copy that run's data/ dir and use METHODS=embedrand.
# A 3D process peaks at ~25 GB of GPU memory: run at most one per 32 GB card.
set -euo pipefail
BASE=${BASE:-$HOME/heat_repro}
PY=$BASE/venv/bin/python
: "${GPU:?}" "${LOW_D:?}" "${TRIALS:?}" "${OUT:?}"
DATA=${DATA:-$OUT/data}
METHODS=${METHODS:-baseline,nepo,embedrand}

export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # shared machine: don't grab 75% of the GPU up front
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"
echo "$$" > "$OUT/pid_trials${TRIALS//,/-}"
echo "host $(hostname)  gpu $GPU  low_D $LOW_D  trials $TRIALS  methods $METHODS  start $(date)"
"$PY" -c "import jax; d=jax.devices(); print('JAX devices:', d); assert d[0].platform=='gpu'"
cd "$BASE/ginjax"
start=$(date +%s)
"$PY" -m scripts.heat_embed --low-D "$LOW_D" --data "$DATA" --methods "$METHODS" \
  --trials "$TRIALS" --model-dir "$OUT/models" --results-dir "$OUT/results" ${EXTRA:-}
echo "elapsed $(( $(date +%s) - start )) s   end $(date)"
