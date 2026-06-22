#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
T=results/bai/scores/baby_teacher.npy
echo "waiting for teacher matrix..."
until [ -f "$T" ]; do sleep 8; done
echo "teacher ready; distillation sweep (single GUME student, 1x cost)"
for W in 1.0 3.0 10.0; do
 for TMP in 2.0; do
  echo ">>> DISTILL w=$W temp=$TMP"
  $PY scripts/train_gume_bai.py --dataset baby --variant baseline \
     --distill_teacher $T --distill_weight $W --distill_temp $TMP \
     --dump_scores --seed 2024 --tag "distill_w${W}_t${TMP}" 2>&1 | grep -E "test R@20|dumped|Error|Traceback" || true
 done
done
echo "=== DISTILL SWEEP DONE ==="
