#!/usr/bin/env bash
# Final deliverable: MV-MoE = validation-tuned late fusion of complementary views (incl. image), vs GUME.
set -u
cd /workspace/MechInterp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python
for DS in baby sports clothing; do
  if [ "$DS" = "baby" ]; then SE="2024 2025 2026 2027 2028"; else SE="2024 2025 2026"; fi
  G=""; for s in $SE; do G="$G ${DS}_bprdump_s${s}.npy"; done
  echo "=== $DS: GUME single (ref) ==="
  $PY scripts/ensemble_eval.py --dataset $DS --scores ${DS}_bprdump_s2024.npy 2>&1 | grep "TEST"
  echo "=== $DS: MV-MoE [GUME seeds + userKNN + image-i2i + text-i2i], val-tuned ==="
  $PY scripts/ensemble_eval.py --dataset $DS --search $G ${DS}_userknn.npy \
      ${DS}_bai2i_sig_img_learned_k20.npy ${DS}_bai2i_sig_text_learned_k20.npy 2>&1 \
      | grep -E "best weights|VALID Recall|TEST"
done
echo "### FINAL MODEL DONE ###"
