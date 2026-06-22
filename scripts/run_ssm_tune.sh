#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
S=2024
# AUX: keep GUME BPR (0.104) + add in-batch softmax sharpening (dot-product geom, debias=1)
for B in 0.2 0.5 1.0; do
  echo ">>> AUX beta=$B temp=1.0 debias=1.0"
  $PY scripts/train_gume_bai.py --dataset baby --variant baseline --ssm_aux $B \
     --ssm_temp 1.0 --ssm_debias 1.0 --seed $S --tag "aux_b${B}_t1_d1" 2>&1 | grep -E "test R@20|Error|Traceback" || true
done
# pure SSM-D temperature tuning (no_norm, debias=1)
for T in 0.5 2.0; do
  echo ">>> SSM-D nonorm debias=1 temp=$T"
  $PY scripts/train_gume_bai.py --dataset baby --variant baseline --bai_loss ssm --ssm_no_norm \
     --ssm_temp $T --ssm_debias 1.0 --seed $S --tag "ssm_nonorm_d1_t${T}" 2>&1 | grep -E "test R@20|Error|Traceback" || true
done
echo "=== SSM TUNE DONE ==="
