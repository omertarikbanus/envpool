#!/usr/bin/env bash
# Recreate v9_explore from scratch: cold start to final model, one script.
#
# The original v4/v5 launchers CANNOT do this. run_v4_cardinal.sh waits on a v3
# ladder and pkills it -- on a clean machine it blocks forever. And
# run_v5_continue.sh is an eight-cycle adaptive loop whose stopping rule depends
# on measured ladder scores; v5_c1 is only its first cycle. This script is the
# five stages that actually produced v9, with the session-specific scaffolding
# removed and nothing adaptive left in it.
#
# Total 55.3M steps, about 3h10m of training at ~4890 fps, plus the build and a
# ~20 min ladder.
#
# PREREQUISITES
#   * the envpool-dev container running (image envpool:may), with this repo at
#     /app/envpool and quadcontrol at /app/quadcontrol
#   * quadcontrol at a commit where MdlFootstepPlanner.cc has kPRelMax = 0.30f
#     (introduced in dfdd18d). This is compiled into the simulator, so a tree
#     with 0.15 produces a different policy and the numbers below will not hold.
#
# DETERMINISM. Every stage runs --seed 0, but PPO over 256 asynchronous envs is
# not bit-reproducible: thread interleaving changes the batch order. Expect the
# final ladder to land within the +/-2.3 point measurement interval of 59.94,
# not exactly on it.
set -euo pipefail

EP=/home/tarik/quadruped_ws/envpool
CFGDIR=/app/quadcontrol/config/robots/sim
SO=/usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so
LOG=$EP/data/reproduce_v9.log
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# Values shared by every stage. ent_coef is left at create_ppo_model's 0.01
# (examples/common/utils.py:89) for stages 1-4; ONLY stage 5 overrides it.
COMMON="--adaptive-lr 0 --target-kl 0.01 --checkpoint-freq 1000000 \
  --randomize-init --seed 0 --num-envs 256 --std-max 0.30 --std-min 0.05"

stage() {  # name steps lr cfg mode [extra...]
  local out=./data/$1 steps=$2 lr=$3 cfg=$4 mode=$5; shift 5
  log "=== $1: $steps steps @ $lr, config $cfg ${*:-} ==="
  docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $out && \
    python3 -u examples/train.py \
      --sim-config-path $CFGDIR/$cfg.toml $COMMON $* \
      --tb-log-dir $out/tb --model-save-path $out/quadruped_ppo_model \
      --total-timesteps $steps --learning-rate $lr $mode" \
    >> "$EP/data/$1_console.log" 2>&1 || { log "STAGE FAILED: $1"; exit 1; }
  log "=== $1 complete ==="
}

seed_from() {  # from to
  log "seeding $2 from $1"
  docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p ./data/$2 && \
    cp ./data/$1/quadruped_ppo_model.zip ./data/$2/quadruped_ppo_model.zip && \
    cp ./data/$1/quadruped_ppo_model_vecnormalize.pkl \
       ./data/$2/quadruped_ppo_model_vecnormalize.pkl"
}

# --- 0. build -------------------------------------------------------------
log "building (make run; its env_step.py smoke aborts on a known WBC crash)"
docker exec -w /app/envpool envpool-dev bash -lc 'make run' \
  > "$EP/data/reproduce_v9_make.log" 2>&1 || true
grep -q "Successfully installed envpool" "$EP/data/reproduce_v9_make.log" \
  || { log "BUILD FAILED -- see data/reproduce_v9_make.log"; exit 1; }
log "simulator sha: $(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)"
log "  reference:   a1238a65562aef1f391ab8af34fbeebf8d4c56abae9475ca293d7c73e3f2f6be"

# --- the five stages ------------------------------------------------------
# 1. cold start, cardinal sampler at half ceilings so the gait can form first
stage r9_s1_card_a  8000000 1e-5 envpool_train_v3_card_a --force-new

# 2. full 1.3x-cliff ceilings
seed_from r9_s1_card_a r9_s2_card_b
stage r9_s2_card_b 14000000 1e-5 envpool_train_v3_card_b --continue-training

# 3. uniform 2.0 m/s ceilings  (= v5_c1)
seed_from r9_s2_card_b r9_s3_card_c
stage r9_s3_card_c 14000000 1e-5 envpool_train_v3_card_c --continue-training

# 4. uniform 3.0 m/s ceilings at 1e-6  (= v8_lr1e6)
seed_from r9_s3_card_c r9_s4_card_f
stage r9_s4_card_f 10000000 1e-6 envpool_train_v3_card_f --continue-training

# 5. THE STAGE THAT MATTERS: same config, same lr as stage 3, ent_coef 0.01 -> 0.05
seed_from r9_s4_card_f r9_s5_v9
stage r9_s5_v9      9000000 1e-5 envpool_train_v3_card_f --continue-training --ent-coef 0.05

log "=== training complete; final model: data/r9_s5_v9/quadruped_ppo_model.zip ==="

# --- ladder ---------------------------------------------------------------
EVALS=/home/tarik/quadruped_ws/quadcontrol/evaluations
RD=reproduce_v9_n30
for r in "0.5,1.0,1.5,2.0,2.5,3.0,3.5" "4.0,4.5,5.0"; do
  x=""; [ "$r" = "4.0,4.5,5.0" ] && x="--no-noforce"
  ( cd "$EVALS" && python3 run.py sweep --controllers ours --disturbance velocity \
      --push-semantics add --force-frame body_latched_at_onset --seed 42 \
      --runs 30 --survival-only --skip-existing $x --run-dir "$RD" \
      --rl-model-ours /app/envpool/data/r9_s5_v9/quadruped_ppo_model.zip \
      --dirs fwd,back,left,right,up,down --impulses "$r" ) >>"$LOG" 2>&1
done
( cd "$EVALS" && python3 ladder_score.py "$RD" ) 2>&1 | tee -a "$LOG"
log "=== reference: v9 scored 59.94 on this grid (95% CI +/-2.3) ==="
