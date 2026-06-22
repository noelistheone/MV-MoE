#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
EE="$PY scripts/ensemble_eval.py"
echo "=== build image-diffusion expert ==="
$PY scripts/build_image_diffusion.py multi 20 2>&1 | grep -E "pairs|saved|Error|Traceback"
echo "=== imgdiff STANDALONE ==="
$EE --scores baby_imgdiff_multi.npy 2>&1 | grep "TEST"
G5="baby_bprdump_s2024.npy baby_bprdump_s2025.npy baby_bprdump_s2026.npy baby_bprdump_s2027.npy baby_bprdump_s2028.npy"
echo "=== GUME(5)+userKNN+imgDiffusion (does image-diffusion add?) ==="
$EE --search $G5 baby_userknn.npy baby_imgdiff_multi.npy 2>&1 | grep -E "best weights|VALID Recall|TEST"
echo "ref A (no image): R@20 0.11127 R@5 0.04516 N@20 0.04913 R@50 0.18256"
echo "=== IMGDIFF DONE ==="
