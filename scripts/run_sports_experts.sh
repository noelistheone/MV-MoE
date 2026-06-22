#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
EE="$PY scripts/ensemble_eval.py --dataset sports"
until grep -q "SPORTS GUME DONE" results/bai/logs_sports_gume.log 2>/dev/null; do sleep 15; done
echo "=== build sports experts ==="
$PY scripts/userknn_scores.py sports 100 2>&1 | grep -E "saved|Error"
$PY scripts/build_ba_i2i.py sports cnn learned 20 2>&1 | grep -E "pairs|saved|Error"
$PY scripts/build_ba_i2i.py sports cnn raw 20 2>&1 | grep -E "saved|Error"
G3="sports_bprdump_s2024.npy sports_bprdump_s2025.npy sports_bprdump_s2026.npy"
echo "=== sports GUME single (s2024) ==="
$EE --scores sports_bprdump_s2024.npy 2>&1 | grep "TEST"
echo "=== sports: GUME(3) bag ==="
$EE --scores $G3 2>&1 | grep "TEST"
echo "=== sports: GUME(3)+userKNN ==="
$EE --search $G3 sports_userknn.npy 2>&1 | grep -E "best weights|TEST"
echo "=== sports: GUME(3)+userKNN+BA-I2I(learned image)  [DOES IMAGE ADD ON SPORTS?] ==="
$EE --search $G3 sports_userknn.npy sports_bai2i_cnn_learned_k20.npy 2>&1 | grep -E "best weights|TEST"
echo "=== sports: GUME(3)+userKNN+BA-I2I(RAW image)  [blind-spot ablation] ==="
$EE --search $G3 sports_userknn.npy sports_bai2i_cnn_raw_k20.npy 2>&1 | grep -E "best weights|TEST"
echo "=== SPORTS EXPERTS DONE ==="
