#!/usr/bin/env bash
set -u
cd /workspace/MechInterp
PY=python
echo "waiting for diverse dump to finish..."
until grep -q "DIVERSE DUMP DONE" results/bai/logs_diverse_dump.log 2>/dev/null; do sleep 10; done
echo "=== individual experts (TEST R@20) ==="
for f in bprdump_s2024 ssmddump_s2024 auxdump_s2024 arch_L1_s2024 arch_L3_s2024 arch_k20_s2024 arch_k40_s2024 bprdump_s2025 bprdump_s2026; do
  [ -f results/bai/scores/baby_${f}.npy ] && $PY scripts/ensemble_eval.py --scores baby_${f}.npy 2>&1 | grep "TEST" | sed "s/^/  ${f}: /"
done
echo "=== FULL ENSEMBLE (greedy weight search on valid) ==="
$PY scripts/ensemble_eval.py --search \
  baby_bprdump_s2024.npy baby_ssmddump_s2024.npy baby_auxdump_s2024.npy \
  baby_arch_L1_s2024.npy baby_arch_L3_s2024.npy baby_arch_k20_s2024.npy baby_arch_k40_s2024.npy \
  baby_bprdump_s2025.npy baby_bprdump_s2026.npy 2>&1 | grep -vE "Warning|warn"
echo "=== DIVERSE ENSEMBLE DONE ==="
