#!/usr/bin/env bash
# v1 "ours" arm -- from-scratch, two-stage fixed-LR run, mirroring run_gamma5.sh.
#
# Same recipe as Gamma5 (no adaptive LR, target_kl 0.01, 20M @ 1e-5 then
# 10M @ 1e-6, from scratch), but against the v1 environment:
#   - disturbance is a velocity kick (+-1.0 m/s), not a 100 N force pulse
#   - ground friction 1.0, contact_mu 0.45
#   - state_source = "noisy_truth" at Rudin's noise bands
#
# Stage 2 writes to its OWN directory, for the reason run_gamma5.sh documents:
# model.learn() defaults to reset_num_timesteps=True, so stage 2's checkpoint
# counter restarts at 0 and would overwrite stage 1's at identical step numbers.
set -euo pipefail

CFG=/app/quadcontrol/config/robots/sim/envpool_train_v1.toml
S1=./data/v1_ours
S2=./data/v1_ours_stage2
COMMON="--sim-config-path $CFG --adaptive-lr 0 --target-kl 0.01 --checkpoint-freq 2000000"

echo "=== v1 ours stage 1: 20M @ 1e-5, from scratch === $(date -Is)"
docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
  $COMMON \
  --tb-log-dir $S1/tb \
  --model-save-path $S1/quadruped_ppo_model \
  --total-timesteps 20000000 \
  --learning-rate 1e-5 \
  --force-new" 2>&1 || { echo 'STAGE 1 FAILED'; exit 1; }

echo "=== v1 ours stage 1 complete === $(date -Is)"

echo "=== Seeding stage 2 from stage 1's final weights ==="
docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $S2 && \
  cp $S1/quadruped_ppo_model.zip $S2/quadruped_ppo_model.zip && \
  cp $S1/quadruped_ppo_model_vecnormalize.pkl $S2/quadruped_ppo_model_vecnormalize.pkl"

echo "=== v1 ours stage 2: 10M @ 1e-6 === $(date -Is)"
docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
  $COMMON \
  --tb-log-dir $S2/tb \
  --model-save-path $S2/quadruped_ppo_model \
  --total-timesteps 10000000 \
  --learning-rate 1e-6 \
  --continue-training" || { echo 'STAGE 2 FAILED'; exit 1; }

echo "=== v1 ours COMPLETE === $(date -Is)"
echo "final model: envpool/data/v1_ours_stage2/quadruped_ppo_model.zip"
