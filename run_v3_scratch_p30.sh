#!/usr/bin/env bash
# kPRelMax 0.30, trained FROM SCRATCH -- an exact replication of the v1
# ours-add recipe with the footstep clamp as the only difference.
#
# WHY FROM SCRATCH: cycle 1 of the v2 campaign warm-started v1's final weights
# and retrained 5M steps at v1's own [1,1,0] disturbance. Training was healthy
# (fall 0.074, reward 732.7 against v1's 0.050/723.7) but the ladder came back
# slightly BEHIND v1, and forward -- the direction raising the clamp was meant
# to fix -- got worse, not better. A policy that learned against a 0.15 prior
# may simply not reorganise around a 0.30 one in 5M steps. From scratch removes
# that confound.
#
# The recipe is run_v1_ours_add.sh's, unchanged: 20M @ 1e-5 from scratch, then
# 10M @ 1e-6, then 10M @ 1e-6, each stage in its OWN directory because
# model.learn() defaults to reset_num_timesteps=True. Config is
# envpool_train_v2_w1.toml, which is v1's [1,1,0] ceiling -- so v1 and this run
# differ in exactly one constant.
#
# Each stage is ladder-evaluated at n=30 on the fixed rung set when it lands,
# so the intermediate points are measured rather than assumed.
set -u

EP=/home/tarik/quadruped_ws/envpool
EVALS=/home/tarik/quadruped_ws/quadcontrol/evaluations
CFG=/app/quadcontrol/config/robots/sim/envpool_train_v2_w1.toml
LOG=$EP/data/v3_scratch_p30.log
COMMON="--sim-config-path $CFG --adaptive-lr 0 --target-kl 0.01 \
  --checkpoint-freq 1000000 --randomize-init"
RUNGS_BASE="0.5,1.0,1.5,2.0,2.5,3.0,3.5"
RUNGS_EXT="4.0,4.5,5.0"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# --- 0. let the v2 campaign's cycle-1 ladder finish, then stand it down ----
# Cycle 1's score is the warm-start control this run is compared against;
# killing it mid-ladder would throw that away for nothing.
log "waiting for the v2 campaign's cycle-1 ladder to finish"
for _ in $(seq 1 60); do
  grep -q 'cycle 1 SCORE=' "$EP/data/v2_campaign.log" && break
  sleep 30
done
if grep -q 'cycle 1 SCORE=' "$EP/data/v2_campaign.log"; then
  log "cycle 1 scored: $(grep 'cycle 1 SCORE=' "$EP/data/v2_campaign.log" | tail -1)"
else
  log "WARNING: cycle-1 score never appeared; standing the campaign down anyway"
fi
pkill -f '[r]un_v2_campaign' 2>/dev/null
sleep 2
docker exec envpool-dev pkill -f '[e]xamples/train.py' 2>/dev/null
sleep 5
log "v2 campaign stood down (its data is kept)"

run_stage() {  # dir steps lr mode
  local dir=$1 steps=$2 lr=$3 mode=$4
  log "=== $steps steps @ $lr -> $dir ==="
  docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
    $COMMON --tb-log-dir $dir/tb --model-save-path $dir/quadruped_ppo_model \
    --total-timesteps $steps --learning-rate $lr $mode" \
    >> "$EP/data/$(basename "$dir")_console.log" 2>&1 \
    || { log "STAGE FAILED: $dir"; exit 1; }
  log "=== stage complete: $dir ==="
}

seed_stage() {  # from to
  log "seeding $2 from $1"
  docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $2 && \
    cp $1/quadruped_ppo_model.zip $2/quadruped_ppo_model.zip && \
    cp $1/quadruped_ppo_model_vecnormalize.pkl $2/quadruped_ppo_model_vecnormalize.pkl"
}

eval_stage() {  # containerdir rundir
  log "evaluating $1 -> results/$2"
  for rungs in "$RUNGS_BASE" "$RUNGS_EXT"; do
    local extra=""
    [ "$rungs" = "$RUNGS_EXT" ] && extra="--no-noforce"
    ( cd "$EVALS" && python3 run.py sweep --controllers ours --disturbance velocity \
        --push-semantics add --force-frame body_latched_at_onset --seed 42 \
        --runs 30 --survival-only --skip-existing $extra --run-dir "$2" \
        --rl-model-ours "/app/envpool/$1/quadruped_ppo_model.zip" \
        --dirs fwd,back,left,right,up,down --impulses "$rungs" ) >>"$LOG" 2>&1
  done
  ( cd "$EVALS" && python3 ladder_score.py "$2" ) 2>&1 | tee -a "$LOG"
}

S1=./data/v3_p30_s1; S2=./data/v3_p30_s2; S3=./data/v3_p30_s3

run_stage "$S1" 20000000 1e-5 --force-new
eval_stage "data/v3_p30_s1" 20260914_v3_p30_s1_n30

seed_stage "$S1" "$S2"
run_stage "$S2" 10000000 1e-6 --continue-training
eval_stage "data/v3_p30_s2" 20260914_v3_p30_s2_n30

seed_stage "$S2" "$S3"
run_stage "$S3" 10000000 1e-6 --continue-training
eval_stage "data/v3_p30_s3" 20260914_v3_p30_s3_n30

log "=== v3 from-scratch kPRelMax 0.30 COMPLETE ==="
