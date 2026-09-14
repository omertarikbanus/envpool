#!/usr/bin/env bash
# v8: 1e-6 consolidation of v5_c1, ceilings 3.0 uniform.
#
# EVERY PARAMETER BELOW IS THE USER'S, EXCEPT total steps and checkpoint
# frequency, which are carried unchanged from run_v1_ours_add.sh's 1e-6 stages.
# Nothing here is chosen on my own judgment. See CLAUDE.md rule 1.
#
#   learning rate : 1e-6                          <- user
#   ceilings      : [3,3,3,3,3,3] m/s cardinal    <- user
#   kPRelMax      : 0.30 (already in the binary)  <- user
#   parent        : data/v5_c1, ladder 56.67      <- best model to date
#   steps         : 10M   <- v1's own 1e-6 stage length, NOT user-specified
#   ckpt freq     : 1M    <- unchanged default,   NOT user-specified
set -u
EP=/home/tarik/quadruped_ws/envpool
EVALS=/home/tarik/quadruped_ws/quadcontrol/evaluations
LOG=$EP/data/v8_lr1e6.log
OUT=./data/v8_lr1e6
RD=20260914_v8_lr1e6_n30
RB="0.5,1.0,1.5,2.0,2.5,3.0,3.5"; RE="4.0,4.5,5.0"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "=== v8: 10M @ 1e-6, ceilings 3.0 uniform, from v5_c1 (56.67) ==="
docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $OUT && \
  cp ./data/v5_c1/quadruped_ppo_model.zip $OUT/quadruped_ppo_model.zip && \
  cp ./data/v5_c1/quadruped_ppo_model_vecnormalize.pkl \
     $OUT/quadruped_ppo_model_vecnormalize.pkl" >>"$LOG" 2>&1 \
  || { log "seed FAILED"; exit 1; }

docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
  --sim-config-path /app/quadcontrol/config/robots/sim/envpool_train_v3_card_f.toml \
  --adaptive-lr 0 --target-kl 0.01 --checkpoint-freq 1000000 --randomize-init \
  --tb-log-dir $OUT/tb --model-save-path $OUT/quadruped_ppo_model \
  --total-timesteps 10000000 --learning-rate 1e-6 --continue-training" \
  >> "$EP/data/v8_lr1e6_console.log" 2>&1 || { log "TRAINING FAILED"; exit 1; }
log "=== training complete ==="

for r in "$RB" "$RE"; do
  x=""; [ "$r" = "$RE" ] && x="--no-noforce"
  ( cd "$EVALS" && python3 run.py sweep --controllers ours --disturbance velocity \
      --push-semantics add --force-frame body_latched_at_onset --seed 42 \
      --runs 30 --survival-only --skip-existing $x --run-dir "$RD" \
      --rl-model-ours "/app/envpool/${OUT#./}/quadruped_ppo_model.zip" \
      --dirs fwd,back,left,right,up,down --impulses "$r" ) >>"$LOG" 2>&1
done
( cd "$EVALS" && python3 ladder_score.py "$RD" ) 2>&1 | tee -a "$LOG"
log "=== v8 COMPLETE ==="
