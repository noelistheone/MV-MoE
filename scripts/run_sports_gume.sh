#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
for S in 2024 2025 2026; do
  echo ">>> SPORTS GUME seed=$S"
  $PY scripts/train_gume_bai.py --dataset sports --variant baseline --dump_scores --seed $S --tag bprdump 2>&1 | grep -E "test R@20|dumped|Error|Traceback" || true
done
echo "=== SPORTS GUME DONE ==="
