#!/usr/bin/env bash
# v1 "ours" arm, additive-push lineage -- from-scratch, THREE-stage fixed-LR run.
#
# Same recipe as run_v1_ours.sh (no adaptive LR, target_kl 0.01, from scratch),
# with two deliberate changes:
#
#   1. ADD push semantics. envpool_train_v1.toml sets
#      external_push_semantics = "add", so the velocity kick superposes on the
#      base velocity instead of overwriting it. Every model before this
#      lineage trained against the overwrite; those are archived under
#      data/v1_ours_set_20260912*.
#
#   2. --randomize-init. Per-episode initial-CONDITION randomisation in
#      QuadrupedWBCEnv::Reset(): base x/y, attitude and joint angles are
#      perturbed around the configured nominal. It perturbs STATE only and
#      touches no command -- the config's own per-episode command resampling
#      (policy_random_vx 0..1) is left exactly as it was.
#
#      This composes with, and does not duplicate, the spawn-height noise the
#      simulator already applies (initial_height_noise = 0.03, inherited from
#      Gamma5, MdlSimDriver::resetSimulation). That covers z; this covers x/y,
#      attitude and joints.
#
# Three stages, matching the gamma5 lineage: 20M @ 1e-5, then 10M @ 1e-6, then
# 10M @ 1e-6 (40M cumulative). Each stage writes to its OWN directory, for the
# reason run_gamma5.sh documents: model.learn() defaults to
# reset_num_timesteps=True, so a later stage's checkpoint counter restarts at 0
# and would overwrite the earlier stage's at identical step numbers.
#
# Checkpoints every 1M so intermediate points can be evaluated after the fact
# -- stage 3 of the previous lineage bought a mean -0.2 survival points over
# stage 2, and that is only visible with checkpoints to compare.
set -euo pipefail

CFG=/app/quadcontrol/config/robots/sim/envpool_train_v1.toml
S1=./data/v1_ours_add
S2=./data/v1_ours_add_stage2
S3=./data/v1_ours_add_stage3
COMMON="--sim-config-path $CFG --adaptive-lr 0 --target-kl 0.01 \
  --checkpoint-freq 1000000 --randomize-init"

log() { echo "=== $* === $(date -Is)"; }

# Seed a later stage from the previous stage's final weights.
seed_stage() {
  local from=$1 to=$2
  log "Seeding $to from $from"
  docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $to && \
    cp $from/quadruped_ppo_model.zip $to/quadruped_ppo_model.zip && \
    cp $from/quadruped_ppo_model_vecnormalize.pkl \
       $to/quadruped_ppo_model_vecnormalize.pkl"
}

run_stage() {
  local dir=$1 steps=$2 lr=$3 mode=$4
  log "v1 ours-add: $steps steps @ $lr -> $dir"
  docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
    $COMMON \
    --tb-log-dir $dir/tb \
    --model-save-path $dir/quadruped_ppo_model \
    --total-timesteps $steps \
    --learning-rate $lr \
    $mode" 2>&1 || { echo "STAGE FAILED: $dir"; exit 1; }
  log "stage complete: $dir"
}

run_stage "$S1" 20000000 1e-5 --force-new
seed_stage "$S1" "$S2"
run_stage "$S2" 10000000 1e-6 --continue-training
seed_stage "$S2" "$S3"
run_stage "$S3" 10000000 1e-6 --continue-training

log "v1 ours-add COMPLETE"
echo "final model: envpool/data/v1_ours_add_stage3/quadruped_ppo_model.zip"
