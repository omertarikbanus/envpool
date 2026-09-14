#!/usr/bin/env bash
# v4: FROM SCRATCH with kPRelMax 0.30 and the cardinal disturbance sampler.
#
# This is the "improve the from-scratch training" run. The v3 run now finishing
# is a straight v1 replication with only the clamp changed -- a control. This
# one changes the thing the evidence says actually matters: the training
# disturbance does not resemble the evaluation disturbance.
#
#   v1/v3 box sampler : all three axes drawn together, one shared ceiling
#                       [1,1,0] -- z never trained, and one value serving six
#                       directions whose measured cliffs span 1.95 to 4.50 m/s.
#   cardinal sampler  : ONE axis per pulse, six independent ceilings set to
#                       ~1.3x each direction's own measured cliff, 75% of draws
#                       in the upper half of that ceiling, 20% of pulses
#                       skipped entirely.
#
# Two phases because a cold policy has to find the gait before it can recover
# from anything: phase a runs half ceilings, phase b the full ones. This is the
# same lesson the flat [5,5,5] attempt taught -- exposure has to sit AT the
# cliff, not past it.
set -u

EP=/home/tarik/quadruped_ws/envpool
EVALS=/home/tarik/quadruped_ws/quadcontrol/evaluations
LOG=$EP/data/v4_cardinal.log
COMMON="--adaptive-lr 0 --target-kl 0.01 --checkpoint-freq 1000000 --randomize-init"
RB="0.5,1.0,1.5,2.0,2.5,3.0,3.5"; RE="4.0,4.5,5.0"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

run_stage() { # dir steps lr cfgtag mode
  local dir=$1 steps=$2 lr=$3 tag=$4 mode=$5
  log "=== $steps steps @ $lr, cardinal phase $tag -> $dir ==="
  docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
    --sim-config-path /app/quadcontrol/config/robots/sim/envpool_train_v3_card_${tag}.toml \
    $COMMON --tb-log-dir $dir/tb --model-save-path $dir/quadruped_ppo_model \
    --total-timesteps $steps --learning-rate $lr $mode" \
    >> "$EP/data/$(basename "$dir")_console.log" 2>&1 \
    || { log "STAGE FAILED: $dir"; exit 1; }
  log "=== stage complete: $dir ==="
}

eval_stage() { # containerdir rundir
  log "evaluating $1 -> results/$2"
  for r in "$RB" "$RE"; do
    local x=""; [ "$r" = "$RE" ] && x="--no-noforce"
    ( cd "$EVALS" && python3 run.py sweep --controllers ours --disturbance velocity \
        --push-semantics add --force-frame body_latched_at_onset --seed 42 \
        --runs 30 --survival-only --skip-existing $x --run-dir "$2" \
        --rl-model-ours "/app/envpool/$1/quadruped_ppo_model.zip" \
        --dirs fwd,back,left,right,up,down --impulses "$r" ) >>"$LOG" 2>&1
  done
  ( cd "$EVALS" && python3 ladder_score.py "$2" ) 2>&1 | tee -a "$LOG"
}

# --- wait for v3's stage-1 ladder, then stand v3 down --------------------
# v3 stage 1 is the from-scratch clamp control and is worth finishing; its
# stages 2 and 3 are not, at 1e-6 they polish reward and not survival (v1's
# stage 3 bought +29 reward and zero robustness).
log "waiting for the v3 stage-1 ladder to finish"
for _ in $(seq 1 200); do
  grep -q '^SCORE' "$EP/data/v3_scratch_p30.log" 2>/dev/null && break
  sleep 30
done
log "v3 stage-1 result: $(grep '^SCORE' "$EP/data/v3_scratch_p30.log" | tail -1)"
pkill -f '[r]un_v3_scratch' 2>/dev/null; sleep 2
docker exec envpool-dev pkill -f '[e]xamples/train.py' 2>/dev/null; sleep 5
log "v3 stood down after its stage-1 ladder (all data kept)"

A=./data/v4_card_a; B=./data/v4_card_b

run_stage "$A" 8000000 1e-5 a --force-new
eval_stage "data/v4_card_a" 20260914_v4_card_a_n30

log "seeding $B from $A"
docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $B && \
  cp $A/quadruped_ppo_model.zip $B/quadruped_ppo_model.zip && \
  cp $A/quadruped_ppo_model_vecnormalize.pkl $B/quadruped_ppo_model_vecnormalize.pkl"

run_stage "$B" 14000000 1e-5 b --continue-training
eval_stage "data/v4_card_b" 20260914_v4_card_b_n30

log "=== v4 cardinal COMPLETE ==="
