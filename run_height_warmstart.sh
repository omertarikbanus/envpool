#!/usr/bin/env bash
# Height-enforced arm, take two: warm start instead of from scratch.
#
# From-scratch failed and the reason is structural, not a bad weight. In its
# first iterations the exact recipe earns about +0.07 per step; the height term
# costs only -0.0031, but that is enough to flip the per-step total negative,
# and only_positive_rewards then clips the WHOLE reward to zero. Observed at
# iteration 500: reward 0.00, value-function loss 0.0000, action std drifting
# up from 1.0, episodes 13 steps, policy measured lying on the floor at
# z = 0.077 m. No gradient ever existed.
#
# A converged policy earns ~1.16 per step before dt. The height penalty at the
# measured crouch is 30 * (0.131 - 0.30)^2 = 0.857, which leaves the total
# positive, so the term can act without erasing the signal. Each height seed
# therefore continues from the matching exact seed.
#
# It also makes the comparison cleaner: same initial policy, one variable
# changed, so any difference is attributable to ride height.
set -euo pipefail

EP=/home/tarik/quadruped_ws/envpool
QC=/home/tarik/quadruped_ws/quadcontrol
SCALE=-30
TARGET=0.30
ITERS=500
GATE_MIN_Z=0.20
SRC=data/rudin_unitree/v1_seed        # exact arm, per seed
RUNS=data/rudin_height_ws/v1_seed     # warm-started height arm

log() { echo "[$(date -Is)] $*"; }

measure() {  # $1 = container model path, $2 = run-dir tag -> "z vx"
  cd $QC/evaluations
  python3 run.py sweep --controllers rudin --rl-model-rudin "$1" \
    --disturbance velocity --push-semantics set \
    --force-frame body_latched_at_onset --seed 42 --dirs fwd \
    --impulses 0.5 --runs 8 --survival-only --run-dir "$2" --skip-existing \
    > /dev/null 2>&1 || true
  python3 -c "
import csv,statistics
rows=list(csv.DictReader(open('results/$2/rl_rudin/noforce/summary.csv')))
z=[float(r['mean_z']) for r in rows if r['mean_z'] not in ('','nan')]
v=[float(r['mean_vx']) for r in rows if r['mean_vx'] not in ('','nan')]
print(f'{statistics.median(z):.3f} {statistics.median(v):.3f}')" 2>/dev/null || echo "nan nan"
}

train_seed() {
  local S=$1 RUN=$RUNS$1
  log "=== height-ws seed $S: warm start from ${SRC}$S/model_1500.pt, $ITERS iters, scale $SCALE target $TARGET ==="
  docker exec envpool-dev bash -lc \
    "cd /app/envpool && mkdir -p $RUN && python3 examples/train_unitree.py \
       --run-dir $RUN --seed $S --max-iterations $ITERS \
       --resume-from /app/envpool/${SRC}$S/model_1500.pt \
       --base-height-scale $SCALE --base-height-target $TARGET \
       > $RUN/console.log 2>&1"
  log "=== height-ws seed $S complete ==="
}

for S in 1 2 3; do
  if [ ! -f "$EP/${SRC}$S/model_1500.pt" ]; then
    log "MISSING source checkpoint for seed $S; skipping"; continue
  fi
  train_seed $S
  LAST=$(ls $EP/$RUNS$S/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  if [ -z "${LAST:-}" ]; then log "seed $S produced no checkpoint; stopping"; exit 1; fi
  R=$(measure "/app/envpool/$RUNS$S/model_$LAST.pt" "height_ws_seed${S}_gate")
  log "seed $S measured: mean_z / mean_vx = $R   (exact arm: 0.131 / 0.781; target $TARGET)"
  if [ "$S" = "1" ]; then
    OK=$(python3 -c "
r='$R'.split()[0]
print('yes' if r!='nan' and float(r) > $GATE_MIN_Z else 'no')")
    if [ "$OK" != "yes" ]; then
      log "GATE FAILED: seed 1 did not clear $GATE_MIN_Z m. Stopping before seeds 2-3."
      log "The weight or the iteration budget needs revisiting, not another seed."
      exit 1
    fi
    log "gate passed; continuing with seeds 2 and 3"
  fi
done
log "=== HEIGHT ARM (warm start) COMPLETE, 3 seeds ==="
