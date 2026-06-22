#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
for DS in sports clothing; do
  for S in 2027 2028; do
    echo ">>> $DS GUME seed=$S $(date +%H:%M)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/train_gume_bai.py \
      --dataset $DS --variant baseline --dump_scores --seed $S --tag bprdump 2>&1 \
      | grep -E "test R@20|dumped|Error|Traceback" || true
  done
done
echo "ALL_SEEDS_DONE $(date +%H:%M)"
