#!/bin/bash
# Finish MicroLens MV-MoE: run the FIXED user-kNN (gated+retry), wait for the text i2i view (produced
# by the other runner), then finalize. Authoritative finalize path.
set -u
MECH=/workspace/MechInterp
PYM=python
LOG=$MECH/results/bai/_microlens_finish.log
SC=$MECH/results/bai/scores
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$MECH"
echo "=== finish START $(date) ===" > "$LOG"

# user-kNN (fixed), gated on >=6GB free, up to 6 attempts
for att in 1 2 3 4 5 6; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free:-0}" -ge 6000 ]; then
    echo "userknn attempt $att (free=${free}MiB) $(date +%H:%M:%S)" >> "$LOG"
    if $PYM scripts/gen_microlens_views.py userknn >> "$LOG" 2>&1; then echo "userknn OK" >> "$LOG"; break; fi
    echo "userknn attempt $att FAILED" >> "$LOG"; sleep 45
  else
    echo "  waiting for GPU (free=${free}MiB)" >> "$LOG"; sleep 60
  fi
done

# wait for all 4 views to be present
until [ -f "$SC/microlens_userknn.npy" ] && [ -f "$SC/microlens_bai2i_cnn_learned_k20.npy" ] && [ -f "$SC/microlens_bai2i_text_learned_k20.npy" ]; do
  echo "  waiting for views $(date +%H:%M:%S)" >> "$LOG"; sleep 30
done
echo "=== all 4 views present; finalize $(date +%H:%M:%S) ===" >> "$LOG"
$PYM scripts/finalize_microlens.py >> "$LOG" 2>&1 && echo "FINALIZE OK" >> "$LOG" || echo "FINALIZE FAILED" >> "$LOG"
echo "=== MICROLENS FINISH DONE $(date) ===" >> "$LOG"
