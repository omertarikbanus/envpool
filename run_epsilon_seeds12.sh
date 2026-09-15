#!/usr/bin/env bash
# Epsilon seeds 1 and 2 -- TRAINING ONLY, sequential.
#
# Deliberately not run_ramp123.sh: that script ladders each seed at the end with
# run_ours_ladder.py, which uses evaluation seed 42 and an adaptive climb. Both
# are wrong for the final collection. docs/ICRA_PAPER_REFERENCE.md requires the
# FIXED 0.5-5.0 grid and evaluation seed identifiers 100-115, and calls the
# independent per-direction climb "acceptable only for developmental range
# finding". Evaluation is therefore launched separately once the seed allocation
# is settled.
#
# Everything else matches seed 0 exactly: same config, same 85M steps, same
# lr 1e-5 and ent_coef 0.05 held fixed, same 256 envs, same std clamp.
# Sequential because each run needs ~20 GB and the CPU is shared with whatever
# ladder is running.
#
# NO GIT.
set -euo pipefail
EP=/home/tarik/quadruped_ws/envpool
SO=/usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so
LOG=$EP/data/run_epsilon_seeds12.log
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

PINNED=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
log "simulator pinned at $PINNED"
[ "$PINNED" = "431313637a168865c3577692a8aeb365d744dfa7a8d3c734910df403a2e3a71f" ] \
  || { log "ABORT: simulator is not the kPRelMax 0.30 build seed 0 used"; exit 1; }

for S in 1 2; do
  OUT=./data/epsilon_s${S}
  HOST=$EP/data/epsilon_s${S}
  if [ -e "$HOST" ]; then log "SKIP seed $S: $OUT exists"; continue; fi
  now=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
  [ "$now" = "$PINNED" ] || { log "ABORT: simulator changed mid-campaign ($now)"; exit 1; }
  log "=== TRAIN epsilon seed $S: 85000000 steps @ 1e-5, ent_coef 0.05 ==="
  docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $OUT && \
    python3 -u examples/train.py \
      --sim-config-path /app/quadcontrol/config/robots/sim/envpool_train_ramp123.toml \
      --seed $S --num-envs 256 --randomize-init \
      --adaptive-lr 0 --target-kl 0.01 --checkpoint-freq 1000000 \
      --std-max 0.30 --std-min 0.05 \
      --ent-coef 0.05 --learning-rate 1e-5 \
      --total-timesteps 85000000 --force-new \
      --tb-log-dir $OUT/tb --model-save-path $OUT/quadruped_ppo_model" \
    >> "$EP/data/epsilon_s${S}_console.log" 2>&1 \
    || { log "SEED $S TRAINING FAILED"; exit 1; }
  [ -f "$HOST/quadruped_ppo_model.zip" ] || { log "ABORT: seed $S produced no final model"; exit 1; }
  log "=== TRAIN epsilon seed $S complete ==="
done
log "=== EPSILON SEEDS 1 AND 2 TRAINED (evaluation is separate) ==="
