#!/usr/bin/env bash
# v1 "ours" arm (learned GRF + WBIC), additive-push lineage, SEED 1 --
# from-scratch three-stage fixed-LR run: the full Gamma5 recipe through stage 3.
#
# RECIPE. Gamma5's lineage is three stages, not the two run_gamma5.sh encodes;
# stage 3 was launched by hand afterwards and its manifest
# (data/gamma5_stage3/quadruped_ppo_model_manifest.json) records 10M @ 1e-6
# continuing stage 2. Reproduced here in one driver:
#
#   stage 1: 20M steps @ fixed 1e-5, from scratch
#   stage 2: 10M steps @ fixed 1e-6, continuing stage 1
#   stage 3: 10M steps @ fixed 1e-6, continuing stage 2   (40M cumulative)
#
# KL-adaptive controller disabled (--adaptive-lr 0), so target_kl stays at
# create_ppo_model's 0.01 rather than the 4x backstop earlier Gamma runs baked
# into their weights. 256 envs, checkpoints every 2M (Gamma5's own setting).
#
# Each stage writes to its OWN directory. model.learn() runs with SB3's default
# reset_num_timesteps=True, so a later stage's checkpoint counter restarts at 0
# and its checkpoints would overwrite the previous stage's at identical step
# numbers. This is the same pattern run_gamma5.sh and run_v1_ours.sh use.
#
# SEEDS. --seed and --force-seed are both passed. external_force_seed defaults
# to 1 in envpool.toml, so --seed alone would leave every seed training against
# an identical disturbance stream and understate the spread. --force-seed sets
# external_force_seed AND policy_random_seed (examples/train.py:521).
#
# PUSH SEMANTICS. envpool_train_v1.toml sets external_push_semantics = "add":
# the velocity kick superposes on the base velocity instead of overwriting it.
# Every ours model before this lineage trained against the overwrite and is
# archived under data/v1_ours_set_20260912*.
#
# --randomize-init. Per-episode initial-CONDITION randomisation in
# QuadrupedWBCEnv::Reset(): base x/y, attitude and joint angles perturbed
# around the configured nominal. STATE only; it touches no command, and since
# envpool@bc6d3e1 it is a separate knob from the commanded-speed jitter
# (--init-speed-jitter, left off here -- the config already resamples the
# command every episode over a far wider range). It composes with the spawn
# height noise the simulator already applies (initial_height_noise = 0.03).
set -euo pipefail

SEED=1
CFG=/app/quadcontrol/config/robots/sim/envpool_train_v1.toml
S1=./data/v1_ours_add_s${SEED}
S2=./data/v1_ours_add_s${SEED}_stage2
S3=./data/v1_ours_add_s${SEED}_stage3
COMMON="--sim-config-path $CFG --adaptive-lr 0 --target-kl 0.01 \
  --num-envs 256 --checkpoint-freq 2000000 --randomize-init \
  --seed $SEED --force-seed $SEED"

SO=/usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so

log() { echo "=== $* === $(date -Is)"; }

# The three stages must share one binary: a rebuild landing mid-run would make
# the stages incomparable, and the stage-2/3 weights would continue a policy
# trained against different physics. Pin the hash at the start and re-check it
# before every stage.
SO_HASH=$(docker exec envpool-dev bash -lc "sha256sum $SO | cut -d' ' -f1")
log "simulator .so sha256: $SO_HASH"

check_so() {
  local now
  now=$(docker exec envpool-dev bash -lc "sha256sum $SO | cut -d' ' -f1")
  if [ "$now" != "$SO_HASH" ]; then
    echo "ABORT: simulator .so changed mid-run ($SO_HASH -> $now)" >&2
    exit 1
  fi
}

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
  check_so
  log "ours-add seed $SEED: $steps steps @ $lr -> $dir"
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

check_so
log "v1 ours-add seed $SEED COMPLETE"
echo "final model: envpool/data/v1_ours_add_s${SEED}_stage3/quadruped_ppo_model.zip"
