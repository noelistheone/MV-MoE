#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
# wait for decoupled confirm runs to release the GPU
echo "waiting for decoupled confirm runs to finish..."
until grep -q "ALL CONFIRM RUNS DONE" results/bai/logs_loo_confirm.log 2>/dev/null; do sleep 15; done
echo "GPU free; starting co-trained runs"
# cs=0 sanity: co-trained channel with zero scale MUST reproduce baseline (~0.103)
echo ">>> COTRAIN sanity cs=0.0 (expect ~baseline)"
$PY scripts/train_gume_bai.py --dataset baby --variant bai --bai_channel \
    --bai_user_img_mode loo --bai_cotrain --bai_chan_scale 0.0 --seed 2024 --tag "cotrain_cnn_cs0.0" \
    2>&1 | grep -E "test R@20|Error|Traceback" || true
# co-trained scale sweep (1 seed); base co-adapts so it can use larger scale
for CS in 1.0 2.0 4.0; do
  echo ">>> COTRAIN cs=${CS} feat=CNN seed=2024"
  $PY scripts/train_gume_bai.py --dataset baby --variant bai --bai_channel \
      --bai_user_img_mode loo --bai_cotrain --bai_chan_scale ${CS} --seed 2024 --tag "cotrain_cnn_cs${CS}" \
      2>&1 | grep -E "test R@20|Error|Traceback" || true
done
echo "=== COTRAIN SWEEP DONE ==="
