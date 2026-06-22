#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
EE="$PY scripts/ensemble_eval.py --dataset baby"
echo "=== build SigLIP image/text/joint item-kNN experts ==="
for F in sig_img sig_text sig_joint; do
  $PY scripts/build_ba_i2i.py baby $F learned 20 2>&1 | grep -E "saved|Error|pairs" | tail -1
done
G5="baby_bprdump_s2024.npy baby_bprdump_s2025.npy baby_bprdump_s2026.npy baby_bprdump_s2027.npy baby_bprdump_s2028.npy"
echo "=== standalone (TEST R@20) ==="
for F in sig_img sig_text sig_joint; do
  $EE --scores baby_bai2i_${F}_learned_k20.npy 2>&1 | grep "TEST" | sed "s/^/  ${F}: /"
done
echo "=== ref A: GUME(5)+userKNN (no content) ==="; $EE --search $G5 baby_userknn.npy 2>&1 | grep TEST
echo "=== +sig_IMAGE  ==="; $EE --search $G5 baby_userknn.npy baby_bai2i_sig_img_learned_k20.npy 2>&1 | grep -E "best weights|TEST"
echo "=== +sig_TEXT   ==="; $EE --search $G5 baby_userknn.npy baby_bai2i_sig_text_learned_k20.npy 2>&1 | grep -E "best weights|TEST"
echo "=== +sig_JOINT  ==="; $EE --search $G5 baby_userknn.npy baby_bai2i_sig_joint_learned_k20.npy 2>&1 | grep -E "best weights|TEST"
echo "=== VLM RANK DONE ==="
