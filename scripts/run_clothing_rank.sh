#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python
EE="$PY scripts/ensemble_eval.py --dataset clothing"
G="clothing_bprdump_s2024.npy clothing_bprdump_s2025.npy clothing_bprdump_s2026.npy"
echo ">>> clothing GUME(3)+userKNN [ref]"; $EE --search $G clothing_userknn.npy 2>&1 | grep -E "best weights|TEST|Error|OutOfMemory"
for F in sig_img sig_text sig_joint; do
  echo ">>> clothing +$F"; $EE --search $G clothing_userknn.npy clothing_bai2i_${F}_learned_k20.npy 2>&1 | grep -E "best weights|TEST|Error|OutOfMemory"
done
echo "### CLOTHING RANK DONE ###"
