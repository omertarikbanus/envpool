#!/usr/bin/env bash
# v9: 9M more steps on v8 with raised exploration.
#
# USER SPECIFIED: ent_coef 0.05 (was 0.01), lr 1e-5, parent v8, ~30 min.
# NOT user specified: steps 9,000,000 (= 30 min at the measured 4890 fps) and
# checkpoint freq 1M (unchanged). Config, ceilings, clamp all inherited from v8
# untouched. See CLAUDE.md rule 1.
set -u
EP=/home/tarik/quadruped_ws/envpool
EVALS=/home/tarik/quadruped_ws/quadcontrol/evaluations
LOG=$EP/data/v9_explore.log
OUT=./data/v9_explore
RD=20260914_v9_explore_n30
RB="0.5,1.0,1.5,2.0,2.5,3.0,3.5"; RE="4.0,4.5,5.0"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "=== v9: 9M @ lr 1e-5, ent_coef 0.05, ceilings 3.0, from v8_lr1e6 (57.06) ==="
docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $OUT && \
  cp ./data/v8_lr1e6/quadruped_ppo_model.zip $OUT/quadruped_ppo_model.zip && \
  cp ./data/v8_lr1e6/quadruped_ppo_model_vecnormalize.pkl \
     $OUT/quadruped_ppo_model_vecnormalize.pkl" >>"$LOG" 2>&1 \
  || { log "seed FAILED"; exit 1; }

docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
  --sim-config-path /app/quadcontrol/config/robots/sim/envpool_train_v3_card_f.toml \
  --adaptive-lr 0 --target-kl 0.01 --checkpoint-freq 1000000 --randomize-init \
  --ent-coef 0.05 --std-max 0.30 \
  --tb-log-dir $OUT/tb --model-save-path $OUT/quadruped_ppo_model \
  --total-timesteps 9000000 --learning-rate 1e-5 --continue-training" \
  >> "$EP/data/v9_explore_console.log" 2>&1 || { log "TRAINING FAILED"; exit 1; }
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
log "=== v9 COMPLETE ==="
