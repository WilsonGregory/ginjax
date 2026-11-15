#!/usr/bin/env bash
# Full heat-equation reproduction (Tables 1, 4, 5, 6 / Figs 2a, 3, 4a) of arXiv:2609.38916 on a
# shared workstation without a scheduler. Launch detached (from $BASE, after setup_heat_env.sh):
#   GPU=1 setsid nohup bash run_heat_gpu.sh > run.log 2>&1 < /dev/null &
# Paper settings: 1D pre-train n=128; 2D fine-tune n in {0,1,4,8,32,128}; 128 val / 128 test;
# N=64 torus, t=1, k=1; UNet depth 48; 50 epochs; batch 8; 5 trials; Nepo/Copy/Zeros vs Baseline.
set -euo pipefail

BASE=${BASE:-$HOME/heat_repro}
OUT=${OUT:-$BASE/run_$(date +%Y%m%d_%H%M%S)}  # fresh dir: script silently reuses cached files by name
PY=$BASE/venv/bin/python

# shared machine: one GPU, and don't let XLA grab 75% of its memory up front
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "$OUT"/data/{train,val,test} "$OUT"/models "$OUT"/results "$OUT"/images
echo "$$" > "$OUT/pid"
echo "host $(hostname)  gpu $CUDA_VISIBLE_DEVICES  out $OUT  start $(date)"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,driver_version,memory.used,memory.total --format=csv
"$PY" -c "import jax; d=jax.devices(); print('JAX devices:', d); assert d[0].platform=='gpu', 'no GPU visible to JAX'"

cd "$BASE/ginjax"
echo "ginjax $(git rev-parse --short HEAD)"
start=$(date +%s)
"$PY" -m scripts.heat_equation \
  --data "$OUT/data" \
  --n-train 128 --n-val 128 --n-test 128 \
  --n-tune-range 0,1,4,8,32,128 \
  --train-D-range 1 --test-D-range 2 \
  -e 50 -b 8 -t 5 \
  --rescale-list nepo,copy,zeros \
  --seed 0 \
  --model-dir "$OUT/models" \
  --results-dir "$OUT/results" \
  --images-dir "$OUT/images/heat_" \
  -v 1
echo "elapsed $(( $(date +%s) - start )) s   end $(date)"
echo "results: $OUT/results/results_D2.npy"
