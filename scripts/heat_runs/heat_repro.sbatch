#!/bin/bash
# Full heat-equation reproduction (Tables 1, 4, 5, 6 / Figs 2a, 3, 4a) of arXiv:2609.38916.
# Paper settings: 1D pre-train n=128; 2D fine-tune n in {0,1,4,8,32,128}; 128 val / 128 test;
# N=64 torus, t=1, k=1; UNet depth 48; 50 epochs; batch 8; 5 trials; Nepo/Copy/Zeros vs Baseline.
# Usage: BASE=/path/to/heat_repro sbatch [--account=...] heat_repro.sbatch
# Set the partition (and any site-specific options, e.g. --comment, --exclude) for your cluster.
#SBATCH --job-name=heat-repro
#SBATCH --partition=l40s,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
#SBATCH --output=heat_repro_%j.out
set -euo pipefail

BASE=${BASE:-$HOME/heat_repro}
OUT=$BASE/run_${SLURM_JOB_ID}   # fresh dir per job: the script silently reuses cached files by name
PY=$BASE/venv/bin/python
mkdir -p "$OUT"/data/{train,val,test} "$OUT"/models "$OUT"/results "$OUT"/images

echo "node $(hostname)  job $SLURM_JOB_ID  out $OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
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
echo "elapsed $(( $(date +%s) - start )) s"
echo "results: $OUT/results/results_D2.npy"
