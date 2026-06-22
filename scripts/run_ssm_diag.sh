#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
S=2024
# A: cosine norm, NO debias  -> isolate debias effect
echo ">>> A norm debias=0 temp=0.15"
$PY scripts/train_gume_bai.py --dataset baby --variant baseline --bai_loss ssm \
   --ssm_temp 0.15 --ssm_debias 0.0 --seed $S --tag "ssm_norm_d0" 2>&1 | grep -E "test R@20|Error|Traceback" || true
# C: raw dot-product (no norm, keeps popularity), NO debias
echo ">>> C nonorm debias=0 temp=1.0"
$PY scripts/train_gume_bai.py --dataset baby --variant baseline --bai_loss ssm --ssm_no_norm \
   --ssm_temp 1.0 --ssm_debias 0.0 --seed $S --tag "ssm_nonorm_d0_t1" 2>&1 | grep -E "test R@20|Error|Traceback" || true
# D: raw dot-product, logQ debias=1.0 (correct sampling correction)
echo ">>> D nonorm debias=1.0 temp=1.0"
$PY scripts/train_gume_bai.py --dataset baby --variant baseline --bai_loss ssm --ssm_no_norm \
   --ssm_temp 1.0 --ssm_debias 1.0 --seed $S --tag "ssm_nonorm_d1_t1" 2>&1 | grep -E "test R@20|Error|Traceback" || true
echo "=== SSM DIAG DONE ==="
