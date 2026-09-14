#!/usr/bin/env bash
# Current proposed-controller training recipe: train from scratch in one
# uninterrupted run per seed, with an optional matched-envelope diagnostic.
#
# WHAT THIS REPLACES. v9 came from a five-stage chain (card_a -> card_b ->
# card_c -> card_f@1e-6 -> card_f@1e-5+ent0.05) whose ceilings, learning rate
# and entropy coefficient all moved between stages. Three things make that hard
# to defend in a paper, and this script removes all three:
#
#   1. The stage sequence records a search, not a design. Here the whole
#      disturbance schedule is one line -- ceiling 1.0 -> 2.0 -> 3.0 m/s at 15%
#      and 45% of training -- implemented inside the env, so there is one run
#      per seed rather than five.
#   2. Changes were confounded. Stage 4 moved ceiling AND learning rate; stage 5
#      moved learning rate AND ent_coef. Here lr is 1e-5 throughout and
#      ent_coef is 0.05 throughout, so nothing co-varies with the ramp.
#   3. n = 1 seed. v9 is --seed 0 and nothing else, which cannot be compared
#      against the Rudin arm's three seeds; its fwd row alone has a 20.8-point
#      seed spread. This trains SEEDS independent seeds so final reporting can
#      preserve seed-level variability.
#
# The 1e-6 stage is dropped outright: it ended at approx_kl 0.0013 against a
# 0.01 budget, so it barely moved the weights and is unattributable either way.
#
# ent_coef is fixed at 0.05 from step 0. `--control` and ENT_COEF=0.01 remain
# optional diagnostics; neither is a required paper ablation.
#
# Trainings cannot overlap because of their memory footprint. The post-training
# ladders can run in parallel, but they are developmental range finders only;
# final paper collection must use the same frozen grid and evaluation-seed
# schedule for every controller.
#
# NO GIT. Nothing here commits, pushes, or stages anything.
set -euo pipefail

EP=/home/tarik/quadruped_ws/envpool
QC=/home/tarik/quadruped_ws/quadcontrol
EVALS=$QC/evaluations
CFGDIR=$QC/config/robots/sim
SO=/usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so

SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-85000000}"
NUM_ENVS="${NUM_ENVS:-256}"
ENT_COEF="${ENT_COEF:-0.05}"
LR="${LR:-1e-5}"
RUNS="${RUNS:-100}"
CONFIG="envpool_train_ramp123"
TAG="${TAG:-epsilon}"
if [ "${1:-}" = "--control" ]; then CONFIG="envpool_train_ramp111"; TAG="epsilon_matched"; fi

LOG=$EP/data/run_${TAG}.log
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# --- preflight ------------------------------------------------------------
if docker exec envpool-dev pgrep -f "[t]rain" > /dev/null 2>&1; then
  log "ABORT: something is already training"; exit 1
fi

# The ramp clock is per-environment, because the driver cannot know how many
# envs the trainer built. Compute it here rather than trusting the config's
# placeholder, which is only correct for the 54M/256 pairing.
PER_ENV=$(( STEPS / NUM_ENVS ))
log "ramp clock: $STEPS steps / $NUM_ENVS envs = $PER_ENV steps per env"
sed -i "s/^external_force_ramp_total_steps = .*/external_force_ramp_total_steps = $PER_ENV/" \
  "$CFGDIR/$CONFIG.toml"

# The low stages are fixed in ABSOLUTE steps (user, 2026-09-14): 8.1M at 1.0 m/s
# and 16.2M at 2.0, with every additional step going to the final 3.0 m/s
# ceiling. Fractions are therefore derived from STEPS rather than hard-coded --
# a hard-coded 0.0953/0.2859 is correct only at 85M, and changing STEPS without
# recomputing them would silently stretch or crush the ramp-in.
if [ "$CONFIG" = "envpool_train_ramp123" ]; then
  FRACS=$(python3 -c "
s=$STEPS
print('[%.8f, %.8f]' % (8_100_000/s, 24_300_000/s))")
  sed -i "s/^external_force_ramp_fractions = .*/external_force_ramp_fractions = $FRACS/" \
    "$CFGDIR/$CONFIG.toml"
  log "ramp stages: 1.0 m/s to 8.1M, 2.0 m/s to 24.3M, 3.0 m/s to ${STEPS}"
fi
grep -n "external_force_ramp" "$CFGDIR/$CONFIG.toml" | tee -a "$LOG"

# kPRelMax is compiled in, so the binary identifies the recipe as much as the
# config does. Pin it for the whole campaign: a rebuild between seeds would
# split them across two simulators, the failure the RL_ONLY_BASELINE_STATUS
# post-mortem blames for the first port.
PINNED=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
log "simulator pinned at $PINNED"
log "  docs' v4-v10 reference (kPRelMax 0.30): a1238a65562aef1f391ab8af34fbeebf8d4c56abae9475ca293d7c73e3f2f6be"
check_so() {
  local now; now=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
  [ "$now" = "$PINNED" ] || { log "ABORT: simulator changed mid-campaign ($now)"; exit 1; }
}

# --- 1. train each seed, one at a time ------------------------------------
for S in $SEEDS; do
  OUT=./data/${TAG}_s${S}
  HOST_OUT=$EP/data/${TAG}_s${S}
  if [ -e "$HOST_OUT" ]; then log "SKIP training seed $S: $OUT exists"; continue; fi
  check_so
  log "=== TRAIN $TAG seed $S: $STEPS steps @ $LR, ent_coef $ENT_COEF, config $CONFIG ==="
  docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $OUT && \
    python3 -u examples/train.py \
      --sim-config-path /app/quadcontrol/config/robots/sim/$CONFIG.toml \
      --seed $S --num-envs $NUM_ENVS --randomize-init \
      --adaptive-lr 0 --target-kl 0.01 --checkpoint-freq 1000000 \
      --std-max 0.30 --std-min 0.05 \
      --ent-coef $ENT_COEF --learning-rate $LR \
      --total-timesteps $STEPS --force-new \
      --tb-log-dir $OUT/tb --model-save-path $OUT/quadruped_ppo_model" \
    >> "$EP/data/${TAG}_s${S}_console.log" 2>&1 \
    || { log "SEED $S TRAINING FAILED"; exit 1; }
  [ -f "$HOST_OUT/quadruped_ppo_model.zip" ] || { log "ABORT: seed $S produced no model"; exit 1; }
  log "=== TRAIN $TAG seed $S complete ==="
done

check_so
log "=== ALL SEEDS TRAINED -- developmental range finding at n=$RUNS ==="

# --- 2. developmental range finding for every seed -------------------------
# Safe together: --survival-only with a non-WBIC controller skips the
# SimulatorLock (run.py:159) and the per-cell force stub travels in memory as
# --stub-b64, not through a shared temp file.
#
# run_ours_ladder.py extends directions independently and uses evaluation seed
# 42. Its results can select a common final ladder, but are not the final
# controlled comparison described in ICRA_PAPER_REFERENCE.md.
declare -A PIDS
for S in $SEEDS; do
  RD=20260914_${TAG}_s${S}_n${RUNS}
  log "=== LADDER seed $S -> $RD ==="
  ( cd "$EVALS" && python3 -u run_ours_ladder.py \
      --model /app/envpool/data/${TAG}_s${S}/quadruped_ppo_model.zip \
      --run-dir "$RD" --runs "$RUNS" > "$EVALS/results/${RD}.log" 2>&1 ) &
  PIDS[$S]=$!
done
FAIL=0
for S in $SEEDS; do
  if wait "${PIDS[$S]}"; then log "=== LADDER seed $S complete ==="
  else log "LADDER seed $S FAILED"; FAIL=1; fi
done

# --- 3. score ------------------------------------------------------------
for S in $SEEDS; do
  ( cd "$EVALS" && python3 ladder_score.py "20260914_${TAG}_s${S}_n${RUNS}" ) 2>&1 | tee -a "$LOG"
done
check_so
[ "$FAIL" -eq 0 ] || { log "ABORT: at least one ladder failed"; exit 1; }
log "=== $TAG COMPLETE: seeds $SEEDS, $STEPS steps, ent_coef $ENT_COEF ==="
log "reference: v9 scored 59.94 (n=30, +/-2.3) on the same grid, but on the"
log "docs' a1238a65 binary -- compare only against a v9 ladder run on $PINNED"
