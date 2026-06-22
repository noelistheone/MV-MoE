#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
D="--dataset baby --variant baseline --dump_scores"
# user-kNN decorrelated expert (different family, not item-item/EASE)
echo ">>> user-kNN"
$PY scripts/userknn_scores.py 100 2>&1 | grep -E "saved|Error" || true
# complete the 5-seed BPR pool
$PY scripts/train_gume_bai.py $D --seed 2027 --tag bprdump 2>&1 | grep -E "test R@20|dumped|Error" || true
$PY scripts/train_gume_bai.py $D --seed 2028 --tag bprdump 2>&1 | grep -E "test R@20|dumped|Error" || true
# content-emphasized expert: GUME + co-trained LOO image-profile channel (content-tilted -> diverse)
$PY scripts/train_gume_bai.py $D --variant bai --bai_channel --bai_user_img_mode loo --bai_cotrain \
   --bai_chan_scale 1.0 --seed 2024 --tag content_loo 2>&1 | grep -E "test R@20|dumped|Error" || true
echo "=== POOL EXPAND DONE ==="
