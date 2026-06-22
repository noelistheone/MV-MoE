#!/usr/bin/env bash
# Cross-dataset replication of the pair-vs-ranking image flip (Sports + Clothing).
set -u
cd /workspace/MechInterp
PY=python
SIG=google/siglip2-base-patch16-224

echo "### PHASE 1: SigLIP image+text encode (sports, clothing) ###"
for DS in sports clothing; do
  echo ">>> SigLIP image $DS"
  $PY scripts/encode_dinov2.py $SIG $DS 2>&1 | grep -E "saved|missing|Error|Traceback" | tail -2
  echo ">>> SigLIP text $DS"
  $PY scripts/encode_siglip_text.py $DS 2>&1 | grep -E "saved|items|Error" | tail -2
done

echo "### PHASE 2: Clothing GUME (3 seeds) + userKNN ###"
for S in 2024 2025 2026; do
  echo ">>> CLOTHING GUME seed=$S"
  $PY scripts/train_gume_bai.py --dataset clothing --variant baseline --dump_scores --seed $S --tag bprdump 2>&1 | grep -E "test R@20|dumped|Error|Traceback" || true
done
echo ">>> clothing user-kNN"
$PY scripts/userknn_scores.py clothing 100 2>&1 | grep -E "saved|Error"

echo "### PHASE 3: PAIR-LEVEL co-purchase decomposition ###"
for DS in sports clothing; do
  echo ">>> PAIR $DS"
  $PY scripts/vlm_decompose.py $DS 2>&1 | grep -E "items,|held-out|Δ|wrote|Error|Traceback"
done

echo "### PHASE 4: RANKING-LEVEL decomposition ###"
for DS in sports clothing; do
  echo ">>> build sig experts $DS"
  for F in sig_img sig_text sig_joint; do
    $PY scripts/build_ba_i2i.py $DS $F learned 20 2>&1 | grep -E "saved|Error" | tail -1
  done
  EE="$PY scripts/ensemble_eval.py --dataset $DS"
  G="${DS}_bprdump_s2024.npy ${DS}_bprdump_s2025.npy ${DS}_bprdump_s2026.npy"
  echo ">>> $DS GUME single"; $EE --scores ${DS}_bprdump_s2024.npy 2>&1 | grep TEST
  echo ">>> $DS GUME(3)+userKNN [ref]"; $EE --search $G ${DS}_userknn.npy 2>&1 | grep TEST
  for F in sig_img sig_text sig_joint; do
    echo ">>> $DS +$F"; $EE --search $G ${DS}_userknn.npy ${DS}_bai2i_${F}_learned_k20.npy 2>&1 | grep -E "best weights|TEST"
  done
done
echo "### XDATASET DONE ###"
