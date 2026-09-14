#!/usr/bin/env bash
# v10: a pure extension of v9. Every parameter is v9's; only the step count
# differs, and both are the user's. See CLAUDE.md rule 1.
#
#   ent_coef   0.05   <- user, unchanged from v9
#   steps      40M    <- user
#   ckpt freq  5M     <- user
#   lr         1e-5   <- v9's, unchanged
#   config     card_f (ceilings 3.0 x6) <- v9's, unchanged
#   parent     data/v9_explore, ladder 59.94
#
# v9's action std rose 0.0925 -> 0.1648 over its 9M and was still climbing at
# +0.0092 per M steps when it ended, so exploration had not equilibrated. Over
# 40M that slope projects past the 0.30 --std-max clamp at roughly 15M, after
# which the clamp rather than ent_coef sets the exploration level. Checkpoints
# every 5M so the mid-run models can be laddered after the fact if the final
# one is not the best.
#
# One continuous run rather than stage splits: each restart costs a full env
# rebuild, which is the whole reason for the single 40M call.
set -u
EP=/home/tarik/quadruped_ws/envpool
EVALS=/home/tarik/quadruped_ws/quadcontrol/evaluations
LOG=$EP/data/v10_v9ext.log
OUT=./data/v10_v9ext
RD=20260914_v10_v9ext_n30
RB="0.5,1.0,1.5,2.0,2.5,3.0,3.5"; RE="4.0,4.5,5.0"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "=== v10: 40M @ lr 1e-5, ent_coef 0.05, ceilings 3.0, from v9_explore (59.94) ==="
docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $OUT && \
  cp ./data/v9_explore/quadruped_ppo_model.zip $OUT/quadruped_ppo_model.zip && \
  cp ./data/v9_explore/quadruped_ppo_model_vecnormalize.pkl \
     $OUT/quadruped_ppo_model_vecnormalize.pkl" >>"$LOG" 2>&1 \
  || { log "seed FAILED"; exit 1; }

docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
  --sim-config-path /app/quadcontrol/config/robots/sim/envpool_train_v3_card_f.toml \
  --adaptive-lr 0 --target-kl 0.01 --checkpoint-freq 5000000 --randomize-init \
  --ent-coef 0.05 --std-max 0.30 \
  --tb-log-dir $OUT/tb --model-save-path $OUT/quadruped_ppo_model \
  --total-timesteps 40000000 --learning-rate 1e-5 --continue-training" \
  >> "$EP/data/v10_v9ext_console.log" 2>&1 || { log "TRAINING FAILED"; exit 1; }
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
log "=== v10 COMPLETE ==="
