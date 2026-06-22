#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
EE="$PY scripts/ensemble_eval.py"
echo "=== building BA-I2I kernels ==="
$PY scripts/build_ba_i2i.py multi learned 20 2>&1 | grep -E "pairs|saved|Error"
$PY scripts/build_ba_i2i.py multi raw 20 2>&1 | grep -E "saved|Error"
$PY scripts/build_ba_i2i.py cnn learned 20 2>&1 | grep -E "saved|Error"
$PY scripts/build_ba_i2i.py text learned 20 2>&1 | grep -E "saved|Error"
echo "=== STANDALONE (TEST R@20) — learned vs raw (blind-spot test) ==="
for f in bai2i_multi_learned_k20 bai2i_multi_raw_k20 bai2i_cnn_learned_k20 bai2i_text_learned_k20; do
  $EE --scores baby_${f}.npy 2>&1 | grep "TEST" | sed "s/^/  ${f}: /"
done
G5="baby_bprdump_s2024.npy baby_bprdump_s2025.npy baby_bprdump_s2026.npy baby_bprdump_s2027.npy baby_bprdump_s2028.npy"
echo "=== FUSION A: GUME(5) + userKNN  [current best, no image] ==="
$EE --search $G5 baby_userknn.npy 2>&1 | grep -E "best weights|VALID Recall|TEST"
echo "=== FUSION B: GUME(5) + userKNN + BA-I2I(learned image)  [does image add?] ==="
$EE --search $G5 baby_userknn.npy baby_bai2i_multi_learned_k20.npy 2>&1 | grep -E "best weights|VALID Recall|TEST"
echo "=== FUSION C: GUME(5) + userKNN + BA-I2I(RAW image)  [does the blind-spot add?] ==="
$EE --search $G5 baby_userknn.npy baby_bai2i_multi_raw_k20.npy 2>&1 | grep -E "best weights|VALID Recall|TEST"
echo "GUME 5-seed: R@20 0.10368 | GUME+userKNN ~0.11127"
echo "=== BA-I2I DONE ==="
