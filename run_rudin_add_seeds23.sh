#!/usr/bin/env bash
# rudin arm, ADD semantics -- seeds 2 and 3: train both, then evaluate both in parallel.
#
# Why this exists: seeds 2 and 3 were trained and evaluated overnight on
# 2026-09-12/13 (data/rudin_unitree/v1_seed{2,3}, finishing 02:36 and 05:31,
# evaluated 06:15 and 06:56). Both halves are SET semantics -- seed 2's
# cell_meta.json has no push_semantics key at all, predating 2c27114 -- so both
# halves are stale under the additive push (quadcontrol 80754b9, envpool
# 315cf4f). Seed 1 was retrained into data/rudin_add/v1_seed1 for exactly this
# reason on 2026-09-13; seeds 2 and 3 never were.
#
# TRAINING is strictly sequential and not negotiable: each run needs ~140 GB of
# the machine's 251 GB, so two cannot coexist.
#
# EVALUATION runs both seeds CONCURRENTLY, per the user. This is safe, checked
# rather than assumed: run.py:159 returns _dispatch() without taking the
# SimulatorLock when --survival-only is set and no WBIC controller is involved,
# which is exactly this arm's configuration; and the per-cell force stub is
# passed to the worker in memory as --stub-b64 rather than through the shared
# fixed-name temp file whose removal run.py's own comment names as the thing
# that made skipping the lock safe. So nothing serialises the two ladders and
# nothing is shared between them but CPU.
#
# NO GIT. Nothing here commits, pushes, or stages anything.
set -euo pipefail

EP=/home/tarik/quadruped_ws/envpool
QC=/home/tarik/quadruped_ws/quadcontrol
EVALS=$QC/evaluations
SO=/usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so

log() { echo "[$(date -Is)] $*"; }

# --- preflight ------------------------------------------------------------
if pgrep -f "[t]rain_unitree.py" > /dev/null || \
   docker exec envpool-dev pgrep -f "[t]rain_unitree.py" > /dev/null 2>&1; then
  log "ABORT: a train_unitree.py is already running"; exit 1
fi

[ -f "$EP/data/rudin_add/v1_seed1/model_1500.pt" ] || {
  log "ABORT: data/rudin_add/v1_seed1/model_1500.pt missing -- seed 1 is not the add arm"; exit 1; }

# Pin the simulator binary for the whole campaign. It CANNOT change halfway: a
# rebuild mid-campaign splits the seeds across two binaries, the exact failure
# the RL_ONLY_BASELINE_STATUS post-mortem blames for the first port. This is NOT
# seed 1's hash (61a4547); the .so was rebuilt at 20:49 on 09-13 for the ours
# lineage. The only post-seed-1 change to this arm's env is a753901, whose
# pd_command_vx_jitter defaults to 0.0 and acts only under pd_fixed_command,
# which train_unitree.py never sets -- so training behaviour is unchanged. The
# other commits touch quadruped_wbc.h, a different env.
PINNED=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
log "simulator pinned at $PINNED"

check_so() {
  local now
  now=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
  [ "$now" = "$PINNED" ] || { log "ABORT: simulator changed mid-campaign ($now)"; exit 1; }
}

# --- 1. train seeds 2 and 3, one at a time --------------------------------
for S in 2 3; do
  RUN=data/rudin_add/v1_seed$S
  HOST_RUN=$EP/$RUN

  if [ -e "$HOST_RUN" ]; then
    log "SKIP training seed $S: $RUN already exists"
  else
    check_so
    log "=== TRAIN seed $S (add pushes, 1500 iterations, 4096 envs) ==="
    docker exec envpool-dev bash -lc \
      "cd /app/envpool && mkdir -p $RUN && python3 examples/train_unitree.py \
         --run-dir $RUN --seed $S > $RUN/console.log 2>&1" \
      || { log "SEED $S TRAINING FAILED; stopping"; exit 1; }
    log "=== TRAIN seed $S complete ==="
  fi

  [ -f "$HOST_RUN/model_1500.pt" ] || { log "ABORT: seed $S has no model_1500.pt"; exit 1; }
done

check_so
log "=== BOTH SEEDS TRAINED -- launching the two ladders in parallel ==="

# --- 2. evaluate both seeds concurrently ----------------------------------
# Straight to n=100. Seed 1 ran a nested n=10 stage first as an early read; the
# stages write separate run dirs and per-episode seeds are seed+index, so
# skipping it changes nothing in the n=100 data.
declare -A PIDS
for S in 2 3; do
  OUT=20260914_rudin_add_seed${S}_n100
  LOG=$EVALS/results/${OUT}.log
  log "=== EVAL seed $S -> $OUT (100 runs/cell), log ${OUT}.log ==="
  ( cd "$EVALS" && python3 -u run_one_ladder.py \
      --model /app/envpool/data/rudin_add/v1_seed$S/model_1500.pt \
      --run-dir "$OUT" --runs 100 > "$LOG" 2>&1 ) &
  PIDS[$S]=$!
done

FAILED=0
for S in 2 3; do
  if wait "${PIDS[$S]}"; then
    log "=== EVAL seed $S complete ==="
  else
    log "SEED $S EVALUATION FAILED (see results/20260914_rudin_add_seed${S}_n100.log)"
    FAILED=1
  fi
done
[ "$FAILED" -eq 0 ] || { log "ABORT: at least one ladder failed"; exit 1; }

check_so
log "=== SEEDS 2 AND 3 COMPLETE (trained + evaluated, add semantics) ==="
log "seed 1: data/rudin_add/v1_seed1 -> 20260913_rudin_add_seed1_n100"
log "seed 2: data/rudin_add/v1_seed2 -> 20260914_rudin_add_seed2_n100"
log "seed 3: data/rudin_add/v1_seed3 -> 20260914_rudin_add_seed3_n100"
