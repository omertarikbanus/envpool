#!/usr/bin/env bash
# v2 "ours" arm: continue the v1 add-semantics policy for 10M more steps
# against a much wider disturbance envelope, with the footstep capture-point
# clamp raised to match the MPC.
#
# Two changes from v1, both user-directed:
#   1. MdlFootstepPlanner kPRelMax 0.15 -> 0.30 (C++; needs the rebuild below).
#   2. external_force_max [1,1,0] -> [5,5,5] m/s (envpool_train_v2.toml).
#
# Warm start from v1 stage 3's final weights rather than from scratch: the
# gait is already there and what we want is adaptation to the wider pushes.
#
# LEARNING RATE: 1e-5, not the 1e-6 v1 stage 3 ended on. Both inputs to the
# policy changed -- a ~5x wider disturbance and a different footstep prior --
# so this is adaptation to a new task, not polish of the old one. 1e-6 would
# barely move the weights. 1e-5 is the rate v1 stage 1 used from scratch
# without instability.
set -euo pipefail

EP=/home/tarik/quadruped_ws/envpool
CFG=/app/quadcontrol/config/robots/sim/envpool_train_v2.toml
SEED_FROM=$EP/data/v1_ours_add_stage3
OUT=./data/v2_ours_wide
HOSTOUT=$EP/data/v2_ours_wide
SO=/usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so
LOG=$EP/data/v2_ours_wide_console.log

log() { echo "=== $* === $(date -Is)" | tee -a "$LOG"; }

# --- 0. do not swap the .so under the running n=100 evaluation -----------
log "waiting for the v1 evaluation ladders to finish"
while pgrep -f run_ours_add_evals.sh >/dev/null 2>&1; do sleep 60; done
log "evaluation driver has exited"

# --- 1. rebuild: kPRelMax lives in C++ -----------------------------------
OLD=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
log "so before: $OLD"
docker exec -w /app/envpool envpool-dev bash -lc 'make run' \
  > "$EP/data/v2_make_run.log" 2>&1 || true
grep -q "Successfully installed envpool" "$EP/data/v2_make_run.log" \
  || { log "BUILD FAILED -- see data/v2_make_run.log"; exit 1; }
NEW=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
log "so after:  $NEW"
[ "$OLD" != "$NEW" ] || { log "BINARY UNCHANGED -- the kPRelMax patch did not reach the build"; exit 1; }

# --- 2. warm start from v1 stage 3 ---------------------------------------
log "seeding $OUT from v1_ours_add_stage3"
mkdir -p "$HOSTOUT"
docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $OUT && \
  cp ./data/v1_ours_add_stage3/quadruped_ppo_model.zip $OUT/quadruped_ppo_model.zip && \
  cp ./data/v1_ours_add_stage3/quadruped_ppo_model_vecnormalize.pkl \
     $OUT/quadruped_ppo_model_vecnormalize.pkl"

# --- 3. 10M steps ---------------------------------------------------------
log "v2 ours-wide: 10000000 steps @ 1e-5 -> $OUT"
docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
  --sim-config-path $CFG --adaptive-lr 0 --target-kl 0.01 \
  --checkpoint-freq 1000000 --randomize-init \
  --tb-log-dir $OUT/tb \
  --model-save-path $OUT/quadruped_ppo_model \
  --total-timesteps 10000000 \
  --learning-rate 1e-5 \
  --continue-training" >> "$LOG" 2>&1 || { log "TRAINING FAILED"; exit 1; }

log "v2 ours-wide COMPLETE"
echo "final model: envpool/data/v2_ours_wide/quadruped_ppo_model.zip"
