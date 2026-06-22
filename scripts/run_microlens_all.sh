#!/bin/bash
# MicroLens (non-Amazon) generalization column for the study.
# Trains the MV-MoE pipeline + all 13 baselines ON MicroLens, writing EVERYTHING into this repo.
# Reads MicroLens data from Recsys (read-only); never writes to Recsys; does NOT use any reserved external
# results. Order: MV-MoE pipeline FIRST (secures the headline + surfaces issues early),
# then baselines. Resilient: continues past a failed step and logs timestamps.
set -u
RECSYS=/workspace/Recsys
MECH=/workspace/MechInterp
PYR=python
PYM=python
CK=$MECH/ckpts_microlens
LG=$MECH/logs_microlens
LOG=$MECH/results/bai/_microlens_run.log
mkdir -p "$CK" "$LG" "$MECH/results/bai/scores"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================ MICROLENS RUN START $(date) ================" > "$LOG"

# ---- 1) 5 GUME(-BAI baseline) seeds + score dumps for MV-MoE bagging ----
cd "$MECH"
for s in 2024 2025 2026 2027 2028; do
  echo "=== [gume-seed] $s  $(date +%H:%M:%S) ===" >> "$LOG"
  $PYM scripts/train_gume_bai.py --dataset microlens --variant baseline --seed "$s" \
       --dump_scores --tag bprdump >> "$LOG" 2>&1 \
       && echo "    seed $s OK" >> "$LOG" || echo "    seed $s FAILED (continuing)" >> "$LOG"
done

# ---- 2) user-kNN + behavior-aligned i2i views (native 1024-d features) ----
echo "=== [userknn] $(date +%H:%M:%S) ===" >> "$LOG"
$PYM scripts/userknn_scores.py microlens >> "$LOG" 2>&1 || echo "    userknn FAILED" >> "$LOG"
echo "=== [i2i cnn] $(date +%H:%M:%S) ===" >> "$LOG"
$PYM scripts/build_ba_i2i.py microlens cnn learned 20 >> "$LOG" 2>&1 || echo "    i2i cnn FAILED" >> "$LOG"
echo "=== [i2i text] $(date +%H:%M:%S) ===" >> "$LOG"
$PYM scripts/build_ba_i2i.py microlens text learned 20 >> "$LOG" 2>&1 || echo "    i2i text FAILED" >> "$LOG"

# ---- 3) MV-MoE finalization (bag + +/-3SE band + significance) ----
echo "=== [finalize MV-MoE] $(date +%H:%M:%S) ===" >> "$LOG"
$PYM scripts/finalize_microlens.py >> "$LOG" 2>&1 || echo "    finalize FAILED" >> "$LOG"

# ---- 4) 13 baselines via the Recsys harness, outputs -> MechInterp ----
cd "$RECSYS"
for m in lightgcn vbpr mmgcn lattice bm3 mgcn mentor freedom lgmrec damrs smore cohesion gume; do
  echo "=== [baseline] $m  $(date +%H:%M:%S) ===" >> "$LOG"
  $PYR -m src.main --model "$m" --dataset microlens \
       --override ckpt_dir="$CK" log_dir="$LG" >> "$LOG" 2>&1 \
       && echo "    $m OK" >> "$LOG" || echo "    $m FAILED (continuing)" >> "$LOG"
done

# ---- 5) collect baseline metrics ----
cd "$MECH"
echo "=== [collect baselines] $(date +%H:%M:%S) ===" >> "$LOG"
$PYM scripts/collect_microlens_baselines.py >> "$LOG" 2>&1 || echo "    collect FAILED" >> "$LOG"

echo "================ MICROLENS RUN DONE $(date) ================" >> "$LOG"
