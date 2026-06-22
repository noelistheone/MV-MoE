#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
echo "waiting for tune sweep to free GPU..."
until grep -q "SSM TUNE DONE" results/bai/logs_ssm_tune.log 2>/dev/null; do sleep 10; done
echo "GPU free; dumping 3 objective-diverse experts (seed 2024)"
$PY scripts/train_gume_bai.py --dataset baby --variant baseline --dump_scores --seed 2024 --tag bprdump 2>&1 | grep -E "test R@20|dumped|Error" || true
$PY scripts/train_gume_bai.py --dataset baby --variant baseline --bai_loss ssm --ssm_no_norm --ssm_temp 1.0 --ssm_debias 1.0 --dump_scores --seed 2024 --tag ssmddump 2>&1 | grep -E "test R@20|dumped|Error" || true
$PY scripts/train_gume_bai.py --dataset baby --variant baseline --ssm_aux 0.2 --ssm_temp 1.0 --ssm_debias 1.0 --dump_scores --seed 2024 --tag auxdump 2>&1 | grep -E "test R@20|dumped|Error" || true
echo "=== SANITY: evaluator vs GUME json (expect R@20~0.1027) ==="
$PY scripts/ensemble_eval.py --scores baby_bprdump_s2024.npy 2>&1 | grep -vE "Warning|warn"
echo "=== ENSEMBLE (BPR + SSM-D + aux), weights tuned on valid ==="
$PY scripts/ensemble_eval.py --search baby_bprdump_s2024.npy baby_ssmddump_s2024.npy baby_auxdump_s2024.npy 2>&1 | grep -vE "Warning|warn"
echo "=== ENSEMBLE TEST DONE ==="
