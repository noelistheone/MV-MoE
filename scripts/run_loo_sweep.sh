#!/usr/bin/env bash
# LOO user-image-profile channel: validation sweep over channel scale (cs), 1 seed.
# Leak-free: pos item excluded from its own profile at train time (LOO); full profile at eval.
set -u
cd /workspace/MechInterp
PY=python
SEED=2024
echo "=== baseline (reference) already at results/bai/baby_baseline_s${SEED}.json ==="
for CS in 0.5 1.0 2.0 4.0; do
  echo ">>> LOO cs=${CS} feat=CNN seed=${SEED}"
  $PY scripts/train_gume_bai.py --dataset baby --variant bai --bai_channel \
      --bai_user_img_mode loo --bai_chan_scale ${CS} --bai_loo_weight 1.0 \
      --seed ${SEED} --tag "loo_cnn_cs${CS}" 2>&1 | grep -E "test R@20|Error|Traceback" || true
done
echo "=== sweep done ==="
