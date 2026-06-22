#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
T=results/bai/scores/baby_teacher.npy
until grep -q "DISTILL SWEEP DONE" results/bai/logs_distill.log 2>/dev/null; do sleep 10; done
echo "small-w distill retry"
for W in 0.1 0.3 0.03; do
  echo ">>> DISTILL2 w=$W temp=2.0"
  $PY scripts/train_gume_bai.py --dataset baby --variant baseline \
     --distill_teacher $T --distill_weight $W --distill_temp 2.0 \
     --dump_scores --seed 2024 --tag "distill2_w${W}" 2>&1 | grep -E "test R@20|dumped|Error" || true
done
echo "=== DISTILL2 DONE ==="
