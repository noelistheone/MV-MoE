#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
SEED=2024
# temperature sweep at debias=1.0
for T in 0.1 0.2 0.3; do
  echo ">>> SSM temp=$T debias=1.0"
  $PY scripts/train_gume_bai.py --dataset baby --variant baseline --bai_loss ssm \
     --ssm_temp $T --ssm_debias 1.0 --seed $SEED --tag "ssm_t${T}_d1.0" 2>&1 | grep -E "test R@20|Error|Traceback" || true
done
# debias sweep at temp=0.15
for D in 0.0 0.5; do
  echo ">>> SSM temp=0.15 debias=$D"
  $PY scripts/train_gume_bai.py --dataset baby --variant baseline --bai_loss ssm \
     --ssm_temp 0.15 --ssm_debias $D --seed $SEED --tag "ssm_t0.15_d${D}" 2>&1 | grep -E "test R@20|Error|Traceback" || true
done
echo "=== SSM SWEEP DONE ==="
