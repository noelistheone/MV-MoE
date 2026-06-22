#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
D="--dataset baby --variant baseline --dump_scores"
# architecture-diverse experts (seed 2024)
$PY scripts/train_gume_bai.py $D --n_layers 1 --seed 2024 --tag arch_L1 2>&1 | grep -E "test R@20|dumped|Error" || true
$PY scripts/train_gume_bai.py $D --n_layers 3 --seed 2024 --tag arch_L3 2>&1 | grep -E "test R@20|dumped|Error" || true
$PY scripts/train_gume_bai.py $D --knn_k 20 --seed 2024 --tag arch_k20 2>&1 | grep -E "test R@20|dumped|Error" || true
$PY scripts/train_gume_bai.py $D --knn_k 40 --seed 2024 --tag arch_k40 2>&1 | grep -E "test R@20|dumped|Error" || true
# seed-diverse BPR experts (bagging)
$PY scripts/train_gume_bai.py $D --seed 2025 --tag bprdump 2>&1 | grep -E "test R@20|dumped|Error" || true
$PY scripts/train_gume_bai.py $D --seed 2026 --tag bprdump 2>&1 | grep -E "test R@20|dumped|Error" || true
echo "=== DIVERSE DUMP DONE ==="
