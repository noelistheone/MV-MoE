#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
bash scripts/run_loo_multiseed.sh 0.5 loo_cnn
bash scripts/run_loo_multiseed.sh 0.5 loo_dinov2 /workspace/MechInterp/data/baby_dinov2_base.npy
echo "=== ALL CONFIRM RUNS DONE ==="
