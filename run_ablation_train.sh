#!/usr/bin/env bash
# Proper channel ablation: retrain from scratch with one action channel
# forced to its neutral value on EVERY training step (--action-ablation),
# not just at eval time (the existing ABLATION_RESULTS.md study only masked
# the channel post-hoc, on an already-trained unablated policy).
#
# Recipe copied verbatim from run_ramp123.sh: config envpool_train_ramp123,
# 85M steps, 256 envs, lr 1e-5, ent_coef 0.05, seed 0. The ONLY new argument
# is --action-ablation. No training parameter is changed by this campaign.
#
# NO GIT. Nothing here commits, pushes, or stages anything.
set -euo pipefail

EP=/home/tarik/quadruped_ws/envpool
QC=/home/tarik/quadruped_ws/quadcontrol
CFGDIR=$QC/config/robots/sim
SO=/usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so
CONFIG=envpool_train_ramp123
SEED=0
STEPS=85000000
NUM_ENVS=256

CHANNEL="$1"   # one of: no_velocity no_footsteps fixed_height fixed_gait zero_grf
TAG="ablation_train_${CHANNEL}_s${SEED}"
OUT=./data/${TAG}
HOST_OUT=$EP/data/${TAG}
LOG=$EP/data/run_${TAG}.log
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

PINNED="431313637a168865c3577692a8aeb365d744dfa7a8d3c734910df403a2e3a71f"
now_so=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
[ "$now_so" = "$PINNED" ] || { log "ABORT: simulator hash $now_so != pinned $PINNED"; exit 1; }

if [ -e "$HOST_OUT" ]; then log "SKIP training $CHANNEL: $OUT exists"; exit 0; fi

log "=== TRAIN ablation channel=$CHANNEL seed=$SEED: $STEPS steps @ 1e-5, ent_coef 0.05, config $CONFIG ==="
docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $OUT && \
  python3 -u examples/train.py \
    --sim-config-path /app/quadcontrol/config/robots/sim/$CONFIG.toml \
    --seed $SEED --num-envs $NUM_ENVS --randomize-init \
    --action-ablation $CHANNEL \
    --adaptive-lr 0 --target-kl 0.01 --checkpoint-freq 1000000 \
    --std-max 0.30 --std-min 0.05 \
    --ent-coef 0.05 --learning-rate 1e-5 \
    --total-timesteps $STEPS --force-new \
    --tb-log-dir $OUT/tb --model-save-path $OUT/quadruped_ppo_model" \
  >> "$EP/data/${TAG}_console.log" 2>&1 \
  || { log "CHANNEL $CHANNEL TRAINING FAILED"; exit 1; }

[ -f "$HOST_OUT/quadruped_ppo_model.zip" ] || { log "ABORT: $CHANNEL produced no model"; exit 1; }
log "=== TRAIN ablation channel=$CHANNEL complete ==="

now_so=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
[ "$now_so" = "$PINNED" ] || { log "ABORT: simulator changed mid-training ($now_so)"; exit 1; }

RD=20260916_${TAG}_n100
log "=== LADDER $CHANNEL -> $RD ==="
( cd "$QC/evaluations" && python3 -u run_ours_ladder.py \
    --model /app/envpool/data/${TAG}/quadruped_ppo_model.zip \
    --action-ablation "$CHANNEL" \
    --run-dir "$RD" --runs 100 > "$QC/evaluations/results/${RD}.log" 2>&1 ) \
  || { log "LADDER $CHANNEL FAILED"; exit 1; }
log "=== LADDER $CHANNEL complete -> results/$RD ==="

( cd "$QC/evaluations" && python3 ladder_score.py "$RD" ) 2>&1 | tee -a "$LOG"
log "=== CHANNEL $CHANNEL DONE end-to-end ==="
