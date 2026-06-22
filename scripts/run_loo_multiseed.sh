#!/usr/bin/env bash
# 5-seed run of the LOO user-image-profile channel at a chosen config.
# Usage: run_loo_multiseed.sh <cs> <tag> [img_npy]
set -u
cd /workspace/MechInterp
PY=python
CS="${1:-1.0}"; TAG="${2:-loo_cnn}"; NPY="${3:-}"
NPYARG=""; [ -n "$NPY" ] && NPYARG="--bai_img_npy $NPY"
for SEED in 2024 2025 2026 2027 2028; do
  echo ">>> LOO 5seed cs=${CS} tag=${TAG} seed=${SEED} npy=${NPY:-CNN}"
  $PY scripts/train_gume_bai.py --dataset baby --variant bai --bai_channel \
      --bai_user_img_mode loo --bai_chan_scale ${CS} --bai_loo_weight 1.0 \
      $NPYARG --seed ${SEED} --tag "${TAG}_cs${CS}" 2>&1 \
      | grep -E "test R@20|Error|Traceback" || true
done
echo "=== ${TAG} 5seed done ==="
