#!/usr/bin/env bash
# 5-seed run of an objective config. Usage: run_obj_multiseed.sh "<extra args>" <tag>
set -u
cd /workspace/MechInterp
PY=python
ARGS="$1"; TAG="$2"
for S in 2024 2025 2026 2027 2028; do
  echo ">>> $TAG seed=$S"
  $PY scripts/train_gume_bai.py --dataset baby --variant baseline $ARGS \
     --seed $S --tag "$TAG" 2>&1 | grep -E "test R@20|Error|Traceback" || true
done
echo "=== $TAG 5seed DONE ==="
