#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
EE="$PY scripts/ensemble_eval.py"
echo "waiting for pool expand..."
until grep -q "POOL EXPAND DONE" results/bai/logs_pool_expand.log 2>/dev/null; do sleep 10; done
GUME="GUME 5-seed: R@5 0.04125 N@5 0.02748 R@10 0.06718 N@10 0.03597 R@20 0.10368 N@20 0.04534 R@50 0.17116 N@50 0.05903 (MDE R@20=0.0035)"
echo "=== A. 5-seed BPR bagging (uniform) ==="
$EE --scores baby_bprdump_s2024.npy baby_bprdump_s2025.npy baby_bprdump_s2026.npy baby_bprdump_s2027.npy baby_bprdump_s2028.npy 2>&1 | grep "TEST"
echo "=== B. GUME-variants ensemble (5 seeds + arch + obj), val-tuned ==="
$EE --search baby_bprdump_s2024.npy baby_bprdump_s2025.npy baby_bprdump_s2026.npy baby_bprdump_s2027.npy baby_bprdump_s2028.npy \
  baby_ssmddump_s2024.npy baby_auxdump_s2024.npy baby_arch_L1_s2024.npy baby_arch_L3_s2024.npy baby_arch_k40_s2024.npy 2>&1 | grep -E "best weights|VALID|TEST"
echo "=== C. + user-kNN + content_loo (decorrelated experts), val-tuned ==="
$EE --search baby_bprdump_s2024.npy baby_bprdump_s2025.npy baby_bprdump_s2026.npy baby_bprdump_s2027.npy baby_bprdump_s2028.npy \
  baby_ssmddump_s2024.npy baby_auxdump_s2024.npy baby_arch_L1_s2024.npy baby_arch_L3_s2024.npy baby_arch_k40_s2024.npy \
  baby_userknn.npy baby_content_loo_s2024.npy 2>&1 | grep -E "best weights|VALID|TEST"
echo "$GUME"
echo "=== FINAL ENSEMBLE DONE ==="
