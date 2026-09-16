#!/usr/bin/env bash
# rudin arm + OUR cardinal velocity-kick curriculum, seed 1 pilot.
#
# WHAT THIS IS: the fourth column of the comparison -- the Rudin recipe trained
# on the disturbance distribution our arm trains on, so the paper can separate
# "the WBC structure helps" from "our policy trained on the evaluated pushes".
#
# LINEAGE (user-directed 2026-09-16): continue from the FINISHED height-
# corrected policy, data/rudin_add_height/v1_seed1/model_500.pt (base z =
# 0.304 m), adding the force curriculum while the height term KEEPS DEFENDING
# the 0.30 m posture (--base-height-scale -30 --base-height-target 0.30, the
# weight and target the height arm itself was trained at).
#
# WHY THE TERM STAYS ON. A first launch at 02:53 ran with the term off and was
# killed at 02:56, during env construction, before any checkpoint -- the user's
# call, and the right one. With nothing pricing posture the 0.30 m stance is
# held only by warm-start inertia: the exact arm's attractor is the 0.131 m
# crouch, and a 3 m/s kick curriculum independently rewards being low, so the
# arm would have drifted back toward the crouch the height column exists to
# remove. Ride height must be held FIXED while the disturbance changes, or the
# fourth column confounds curriculum with posture and answers nothing.
#
# The penalty is safe here for the same reason it was safe in rudin_add_height:
# this is a warm start, not a cold one. A converged policy earns ~1.16 per step
# and 30*(z-0.30)^2 is ~0 at the target, so only_positive_rewards never clips
# the learning signal -- the failure that broke the from-scratch height arm.
#
# The script still MEASURES base height after training, now as a check that the
# posture HELD rather than a check on how far it fell.
#
# Curriculum: config/robots/sim/envpool_train_rudin_card_f.toml -- cardinal,
# six 3.0 m/s ceilings, add semantics, 20% no-force, 0.2 s on / 1.4 s off,
# onset on the 0.4 s fallback phase clock (this arm has no gait scheduler).
# legged_gym's own push_robots is disabled with --no-push so the two push
# trains do not superpose.
#
# Pre-flight already done by hand: the curriculum fires for this arm on all six
# directions, max requested 2.964 m/s, delivered |dvx| up to 2.870 m/s, and the
# warm-start policy walks at 0.293 m under it.
#
# NO GIT. Nothing here commits, pushes or stages anything.
set -euo pipefail

EP=/home/tarik/quadruped_ws/envpool
EVALS=/home/tarik/quadruped_ws/quadcontrol/evaluations
SO=/usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so
RUN=data/rudin_cardf/v1_seed1
PARENT=/app/envpool/data/rudin_add_height/v1_seed1/model_500.pt
CFG=/app/quadcontrol/config/robots/sim/envpool_train_rudin_card_f.toml
ITERS=500
RD=20260916_rudin_cardf_seed1_n30
LOG=$EP/data/run_rudin_cardf_seed1.log

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# --- preflight ------------------------------------------------------------
if pgrep -f "[t]rain_unitree.py" > /dev/null || \
   docker exec envpool-dev pgrep -f "[t]rain_unitree.py" > /dev/null 2>&1; then
  log "ABORT: a train_unitree.py is already running"; exit 1
fi
[ -f "$EP/data/rudin_add_height/v1_seed1/model_500.pt" ] || {
  log "ABORT: warm-start checkpoint missing"; exit 1; }

# Pin the binary: a rebuild mid-campaign splits training and evaluation across
# two simulators, the failure the RL_ONLY_BASELINE_STATUS post-mortem names.
PINNED=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
log "simulator pinned at $PINNED"
check_so() {
  local now; now=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
  [ "$now" = "$PINNED" ] || { log "ABORT: simulator changed mid-campaign ($now)"; exit 1; }
}

# --- 1. train ------------------------------------------------------------
log "=== TRAIN rudin+card_f seed 1: $ITERS iters, 4096 envs, height -30 @ 0.30, from rudin_add_height seed 1 ==="
docker exec envpool-dev bash -lc \
  "cd /app/envpool && mkdir -p $RUN && python3 examples/train_unitree.py \
     --run-dir $RUN --seed 1 --max-iterations $ITERS \
     --sim-config-path $CFG --resume-from $PARENT --no-push \
     --base-height-scale -30 --base-height-target 0.30 \
     > $RUN/console.log 2>&1" \
  || { log "TRAINING FAILED (see $RUN/console.log)"; exit 1; }
[ -f "$EP/$RUN/model_$ITERS.pt" ] || { log "ABORT: no model_$ITERS.pt"; exit 1; }
log "=== training complete ==="

# --- 2. did it keep the height? ------------------------------------------
check_so
log "=== checking the 0.30 m posture held, no-kick, n=30 ==="
( cd "$EVALS" && python3 run.py sweep --controllers rudin --disturbance velocity \
    --push-semantics add --force-frame body_latched_at_onset --seed 42 \
    --runs 30 --survival-only --run-dir "${RD}_height" \
    --rl-model-rudin "/app/envpool/$RUN/model_$ITERS.pt" \
    --dirs fwd --impulses 0.0 ) >>"$LOG" 2>&1 \
  || log "height probe failed (non-fatal, ladder still runs)"

# --- 3. ladder -----------------------------------------------------------
check_so
log "=== LADDER -> $RD (30 runs/cell) ==="
( cd "$EVALS" && python3 -u run_one_ladder.py \
    --model "/app/envpool/$RUN/model_$ITERS.pt" \
    --run-dir "$RD" --runs 30 ) >>"$LOG" 2>&1 \
  || { log "LADDER FAILED"; exit 1; }

check_so
log "=== COMPLETE: $RUN -> $RD ==="
