#!/bin/bash
# Generate MicroLens MV-MoE views (frugal) + finalize, GATED on free GPU memory so it coexists with
# the concurrent baseline job + sahil's job. Waits for a >=7GB window, retries each step on OOM.
set -u
MECH=/workspace/MechInterp
PYM=python
LOG=$MECH/results/bai/_microlens_mvmoe.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$MECH"
echo "=== MV-MoE view+finalize runner START $(date) ===" > "$LOG"

wait_gpu() {  # block until >= $1 MiB free
  local need=$1 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -ge "$need" ]; then
      echo "  [gate] free ${free}MiB >= ${need} -> GO $(date +%H:%M:%S)" >> "$LOG"; return 0
    fi
    sleep 60
  done
}

run_step() {  # name  command...   (retry up to 3x with the gate)
  local name=$1; shift
  for att in 1 2 3; do
    wait_gpu 7000
    echo "=== $name attempt $att $(date +%H:%M:%S) ===" >> "$LOG"
    if "$@" >> "$LOG" 2>&1; then echo "  $name OK" >> "$LOG"; return 0; fi
    echo "  $name attempt $att FAILED; retrying after 45s" >> "$LOG"; sleep 45
  done
  echo "  $name GAVE UP after 3 attempts" >> "$LOG"; return 1
}

run_step userknn $PYM scripts/gen_microlens_views.py userknn
run_step img     $PYM scripts/gen_microlens_views.py img
run_step txt     $PYM scripts/gen_microlens_views.py txt
run_step finalize $PYM scripts/finalize_microlens.py

echo "=== MICROLENS MVMOE DONE $(date) ===" >> "$LOG"
