#!/usr/bin/env bash
# Gamma5 -- from-scratch, two-stage fixed-LR run. Launched on the HOST inside a
# tmux session; each stage is a `docker exec` into envpool-dev.
#
# Recipe (requested directly):
#   no adaptive LR (--adaptive-lr 0), target_kl 0.01 (alpha's value)
#   stage 1: 20M steps @ fixed 1e-5, from scratch
#   stage 2: 10M steps @ fixed 1e-6, continuing stage 1's weights
#
# Stage 2 writes to its OWN directory: PeriodicCheckpointCallback names
# checkpoints by num_timesteps, and model.learn() runs with SB3's default
# reset_num_timesteps=True, so stage 2's counter restarts at 0 and its
# checkpoints would overwrite stage 1's at identical step numbers. Copying the
# final weights into a fresh dir and resuming there is the same pattern Gamma4
# used to continue Gamma3 without touching Gamma3's files.
set -euo pipefail

CFG=/app/quadcontrol/config/robots/sim/envpool_train_Gamma5.toml
S1=./data/gamma5
S2=./data/gamma5_stage2
COMMON="--sim-config-path $CFG --adaptive-lr 0 --target-kl 0.01 --checkpoint-freq 2000000"

echo "=== Gamma5 stage 1: 20M @ 1e-5, from scratch === $(date -Is)"
docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
  $COMMON \
  --tb-log-dir $S1/tb \
  --model-save-path $S1/quadruped_ppo_model \
  --total-timesteps 20000000 \
  --learning-rate 1e-5 \
  --force-new" 2>&1 || { echo 'STAGE 1 FAILED'; exit 1; }

echo "=== Gamma5 stage 1 complete === $(date -Is)"

echo "=== Seeding stage 2 from stage 1's final weights ==="
docker exec envpool-dev bash -lc "cd /app/envpool && \
  cp $S1/quadruped_ppo_model.zip $S2/quadruped_ppo_model.zip && \
  cp $S1/quadruped_ppo_model_vecnormalize.pkl $S2/quadruped_ppo_model_vecnormalize.pkl"

echo "=== Gamma5 stage 2: 10M @ 1e-6 === $(date -Is)"
docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
  $COMMON \
  --tb-log-dir $S2/tb \
  --model-save-path $S2/quadruped_ppo_model \
  --total-timesteps 10000000 \
  --learning-rate 1e-6 \
  --continue-training" || { echo 'STAGE 2 FAILED'; exit 1; }

echo "=== Gamma5 COMPLETE === $(date -Is)"
echo "final model: envpool/data/gamma5_stage2/quadruped_ppo_model.zip"
