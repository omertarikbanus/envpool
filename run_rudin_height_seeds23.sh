#!/usr/bin/env bash
# rudin HEIGHT-CORRECTED arm, add semantics -- seeds 2 and 3.
#
# Per the user (2026-09-14): the exact arm is NOT part of the comparison, so
# only the height-corrected policies are evaluated. The exact seeds remain
# necessary as the warm-start SOURCE and are trained, but never laddered.
#
# Why warm start rather than from scratch: a height penalty from scratch is
# broken by only_positive_rewards. The exact recipe earns only ~+0.07 per step
# in its first iterations and the height term costs enough to flip that sum
# negative, so the total clips to zero and no gradient ever exists (measured
# 2026-09-13: scale -30 from scratch gave mean reward 0.00, value loss 0.0000,
# 13-step episodes, and a checkpoint lying on the floor at z = 0.077 m). A
# converged policy earns ~1.16 per step, so the same penalty stays positive.
# Matching seed 1's recipe exactly: 500 iterations, scale -30, target 0.30.
#
# Bonus of warm starting each height seed from its OWN exact seed: the initial
# policy is identical, so ride height is the only changed variable.
#
# TRAINING is sequential (~140 GB of 251 GB each). EVALUATION runs both seeds
# concurrently -- safe because run.py:159 skips the SimulatorLock for
# --survival-only without a WBIC controller, and the force stub travels in
# memory as --stub-b64 rather than through a shared temp file.
#
# NO GIT. Nothing here commits, pushes, or stages anything.
set -euo pipefail

EP=/home/tarik/quadruped_ws/envpool
EVALS=/home/tarik/quadruped_ws/quadcontrol/evaluations
SO=/usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so

log() { echo "[$(date -Is)] $*"; }

PINNED=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
log "simulator pinned at $PINNED"
check_so() {
  local now; now=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
  [ "$now" = "$PINNED" ] || { log "ABORT: simulator changed mid-campaign ($now)"; exit 1; }
}

# --- 0. wait for the in-flight exact seed 3 -------------------------------
# Its driver was stopped deliberately; the training itself was left running as
# an orphaned docker exec, so poll the process, not a log line.
if docker exec envpool-dev pgrep -f "[t]rain_unitree.py --run-dir data/rudin_add/v1_seed3" > /dev/null 2>&1; then
  log "=== waiting for exact seed 3 training to finish ==="
  while docker exec envpool-dev pgrep -f "[t]rain_unitree.py --run-dir data/rudin_add/v1_seed3" > /dev/null 2>&1; do
    sleep 60
  done
  log "=== exact seed 3 process gone ==="
fi

for S in 2 3; do
  [ -f "$EP/data/rudin_add/v1_seed$S/model_1500.pt" ] || {
    log "ABORT: exact seed $S has no model_1500.pt -- cannot warm start"; exit 1; }
done
log "both exact seeds present; they are warm-start sources only and will NOT be laddered"

# --- 1. height warm starts, one at a time --------------------------------
for S in 2 3; do
  RUN=data/rudin_add_height/v1_seed$S
  HOST_RUN=$EP/$RUN
  if [ -e "$HOST_RUN" ]; then
    log "SKIP height seed $S: $RUN already exists"
  else
    check_so
    log "=== HEIGHT WARM START seed $S (500 iters, scale -30, target 0.30) ==="
    docker exec envpool-dev bash -lc \
      "cd /app/envpool && mkdir -p $RUN && python3 examples/train_unitree.py \
         --run-dir $RUN --seed $S --max-iterations 500 \
         --resume-from /app/envpool/data/rudin_add/v1_seed$S/model_1500.pt \
         --base-height-scale -30 --base-height-target 0.30 \
         > $RUN/console.log 2>&1" \
      || { log "HEIGHT SEED $S FAILED; stopping"; exit 1; }
    log "=== HEIGHT WARM START seed $S complete ==="
  fi
  [ -f "$HOST_RUN/model_500.pt" ] || { log "ABORT: height seed $S has no model_500.pt"; exit 1; }
done

check_so
log "=== BOTH HEIGHT SEEDS TRAINED -- launching the two ladders in parallel ==="

# --- 2. evaluate both height seeds concurrently ---------------------------
declare -A PIDS
for S in 2 3; do
  OUT=20260914_rudinaddheight_seed${S}_n100
  log "=== EVAL height seed $S -> $OUT (100 runs/cell) ==="
  ( cd "$EVALS" && python3 -u run_one_ladder.py \
      --model /app/envpool/data/rudin_add_height/v1_seed$S/model_500.pt \
      --run-dir "$OUT" --runs 100 > "$EVALS/results/${OUT}.log" 2>&1 ) &
  PIDS[$S]=$!
done

FAILED=0
for S in 2 3; do
  if wait "${PIDS[$S]}"; then log "=== EVAL height seed $S complete ==="
  else log "HEIGHT SEED $S EVALUATION FAILED (see results/20260914_rudinaddheight_seed${S}_n100.log)"; FAILED=1; fi
done
[ "$FAILED" -eq 0 ] || { log "ABORT: at least one ladder failed"; exit 1; }

# --- 3. report the achieved ride height -----------------------------------
# The gate that matters for this arm: did the penalty actually raise the base?
# mean_z of the no-kick control cell is the measurement; seed 1's warm starts
# landed 0.304 / 0.300 / 0.288 m against the 0.30 m target from a 0.131 crouch.
for S in 2 3; do
  F=$EVALS/results/20260914_rudinaddheight_seed${S}_n100/rl_rudin/noforce/summary.csv
  if [ -f "$F" ]; then
    python3 - "$F" "$S" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
z=[float(r["mean_z"]) for r in rows if r.get("mean_z") not in (None,"","nan")]
print(f"  height seed {sys.argv[2]}: base z = {sum(z)/len(z):.3f} m over {len(z)} no-kick episodes (target 0.30)")
PY
  fi
done

check_so
log "=== HEIGHT SEEDS 2 AND 3 COMPLETE ==="
log "seed 1: data/rudin_add_height/v1_seed1 -> 20260913_rudinaddheight_seed1_n100"
log "seed 2: data/rudin_add_height/v1_seed2 -> 20260914_rudinaddheight_seed2_n100"
log "seed 3: data/rudin_add_height/v1_seed3 -> 20260914_rudinaddheight_seed3_n100"
