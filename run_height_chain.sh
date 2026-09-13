#!/usr/bin/env bash
# After the rudin evaluation finishes: save the sweep, add the base-height
# reward term, rebuild, then train the height-enforced arm at three seeds.
#
# Ordering is the point. The rebuild swaps the .so, so it must not happen while
# any seed trains or any evaluation runs -- a campaign split across two binaries
# is what RL_ONLY_BASELINE_STATUS blames for the first port's failure.
#
# Weight: -30 at target 0.30 m. Reasoning, per step and before the dt factor:
# the converged total is ~1.16, and the height term is scale * (z - 0.30)^2.
# At the measured crouch z = 0.131 the squared error is 0.0285, so -30 costs
# 0.855 -- strong pressure that still leaves the total positive. Past about -40
# the total goes negative, only_positive_rewards clips it to zero, and the
# gradient vanishes exactly where it is needed. The term self-extinguishes
# quadratically as the robot rises.
#
# Seed 1 is its own probe: if it does not lift the robot, seeds 2 and 3 are
# never launched.
set -euo pipefail

WS=/home/tarik/quadruped_ws
EP=$WS/envpool
QC=$WS/quadcontrol
SCALE=-30
TARGET=0.30
GATE_MIN_Z=0.20          # seed 1 must clear this or the chain stops
RUNS=data/rudin_height

log() { echo "[$(date -Is)] $*"; }

# --- 1. wait for the evaluation campaign ---------------------------------
log "waiting for the rudin evaluation to finish"
while pgrep -f "[r]un_rudin_eval.py" > /dev/null; do sleep 120; done
while pgrep -f "[r]un\.py sweep" > /dev/null; do sleep 60; done
log "evaluation done"

# Independent guard. The rebuild below swaps the .so, so nothing may be
# training when it runs -- not even if the evaluation driver died early and
# left a seed running.
while pgrep -f "[t]rain_unitree.py" > /dev/null; do
  log "a training run is still active; holding the rebuild"
  sleep 120
done

# --- 2. save the current state -------------------------------------------
log "committing the seed sweep"
cd $EP
git add data/rudin_unitree/ apply_height_term.sh run_height_chain.sh || true
if ! git diff --cached --quiet; then
  git commit -q -F - <<'MSG'
Record the rudin three-seed sweep

Three seeds at 1500 iterations against the v1 plant, all on one .so
(73f18e2d...). Final models, run manifests, console logs and TensorBoard
events. Intermediate checkpoints stay ignored; model_1500.pt is un-ignored
per .gitignore:192.

Also adds the scripts that produced and will extend the campaign.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017U8fcYP6AqUmqcVsmmGnar
MSG
  git push -q origin main && log "envpool artifacts pushed"
else
  log "envpool: nothing to commit"
fi
cd $QC
git add -A || true
if ! git diff --cached --quiet; then
  git commit -q -m "Record evaluation-side changes from the rudin campaign

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017U8fcYP6AqUmqcVsmmGnar"
  git push -q origin main && log "quadcontrol pushed"
else
  log "quadcontrol: nothing to commit"
fi

# --- 3. apply the height term and rebuild --------------------------------
cd $EP
OLD_SO=$(docker exec envpool-dev sha256sum \
  /usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so | cut -d' ' -f1)
log "so before: $OLD_SO"
log "applying the base-height term"
./apply_height_term.sh

log "rebuilding (make run; its env_step.py smoke aborts on a known WBC crash)"
docker exec -w /app/envpool envpool-dev bash -lc 'make run' > $EP/data/height_make_run.log 2>&1 || true
if ! grep -q "Successfully installed envpool" $EP/data/height_make_run.log; then
  log "BUILD FAILED: no wheel installed. See data/height_make_run.log"
  exit 1
fi
NEW_SO=$(docker exec envpool-dev sha256sum \
  /usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so | cut -d' ' -f1)
log "so after: $NEW_SO"
if [ "$OLD_SO" = "$NEW_SO" ]; then
  log "BUILD DID NOT CHANGE THE BINARY -- the patch did not reach the build"
  exit 1
fi

# --- 4. prove the exact arm is unchanged ---------------------------------
log "recipe check against the new binary"
docker exec -w /app/envpool envpool-dev bash -lc \
  'python3 examples/tests/unitree_recipe_check.py --num-envs 64' \
  > $EP/data/height_recipe_check.log 2>&1
if grep -q "^FAIL" $EP/data/height_recipe_check.log; then
  log "RECIPE CHECK FAILED against the new binary; stopping"
  grep "^FAIL" $EP/data/height_recipe_check.log
  exit 1
fi
log "measuring the exact arm's height on the new binary (expect ~0.131 m)"
cd $QC/evaluations
python3 run.py sweep --controllers rudin \
  --rl-model-rudin /app/envpool/data/rudin_unitree/v1_seed1/model_1500.pt \
  --disturbance velocity --push-semantics set --force-frame body_latched_at_onset \
  --seed 42 --dirs fwd --impulses 0.5 --runs 8 --survival-only \
  --run-dir height_equivalence_check --skip-existing > /dev/null 2>&1 || true
Z=$(python3 -c "
import csv,statistics
rows=list(csv.DictReader(open('results/height_equivalence_check/rl_rudin/noforce/summary.csv')))
z=[float(r['mean_z']) for r in rows if r['mean_z'] not in ('','nan')]
print(f'{statistics.median(z):.3f}')" 2>/dev/null || echo "nan")
log "exact arm on the new binary: mean_z = $Z (baseline 0.131)"

# --- 5. commit the patch --------------------------------------------------
cd $EP
git add -A envpool/mujoco/gym/quadruped_pd.h examples/common/rsl_vec_env.py \
        examples/train_unitree.py
git commit -q -F - <<MSG
Add an opt-in base-height reward term to QuadrupedPD

legged_gym ships _reward_base_height and GO2RoughCfg declares
base_height_target = 0.25, but LeggedRobotCfg weights the term at -0., so the
recipe never enforces the height it names. The trained arm walks at 0.131 m
against WBIC's 0.306 m, and a lower centre of mass is harder to topple, which
confounds the push comparison.

The term is opt-in: pd_base_height_scale defaults to 0.0, so the exact-recipe
arm is unchanged and its three seeds stay reproducible. The height-enforced
arm passes --base-height-scale $SCALE --base-height-target $TARGET.

Exact arm re-measured on this binary: mean_z = $Z against a 0.131 baseline.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017U8fcYP6AqUmqcVsmmGnar
MSG
git push -q origin main && log "height term pushed"

# --- 6. train the height-enforced arm ------------------------------------
train_seed() {
  local S=$1 RUN=$RUNS/v1_seed$1
  log "=== height arm, seed $S (scale $SCALE, target $TARGET) ==="
  docker exec envpool-dev bash -lc \
    "cd /app/envpool && mkdir -p $RUN && python3 examples/train_unitree.py \
       --run-dir $RUN --seed $S \
       --base-height-scale $SCALE --base-height-target $TARGET \
       > $RUN/console.log 2>&1"
  log "=== height arm seed $S complete ==="
}

train_seed 1
if [ ! -f "$EP/$RUNS/v1_seed1/model_1500.pt" ]; then
  log "SEED 1 DID NOT COMPLETE; not launching seeds 2-3"
  exit 1
fi

# The gate: seed 1 is the weight probe. If the term did not lift the robot,
# two more seeds would only reproduce the same failure.
cd $QC/evaluations
python3 run.py sweep --controllers rudin \
  --rl-model-rudin /app/envpool/$RUNS/v1_seed1/model_1500.pt \
  --disturbance velocity --push-semantics set --force-frame body_latched_at_onset \
  --seed 42 --dirs fwd --impulses 0.5 --runs 8 --survival-only \
  --run-dir height_gate_seed1 --skip-existing > /dev/null 2>&1 || true
GZ=$(python3 -c "
import csv,statistics
rows=list(csv.DictReader(open('results/height_gate_seed1/rl_rudin/noforce/summary.csv')))
z=[float(r['mean_z']) for r in rows if r['mean_z'] not in ('','nan')]
v=[float(r['mean_vx']) for r in rows if r['mean_vx'] not in ('','nan')]
print(f'{statistics.median(z):.3f} {statistics.median(v):.3f}')" 2>/dev/null || echo "nan nan")
log "GATE: height-arm seed 1 mean_z / mean_vx = $GZ  (was 0.131 / 0.781; need z > $GATE_MIN_Z)"
OK=$(python3 -c "
z='$GZ'.split()[0]
print('yes' if z != 'nan' and float(z) > $GATE_MIN_Z else 'no')")
if [ "$OK" != "yes" ]; then
  log "GATE FAILED: scale $SCALE did not lift the robot above $GATE_MIN_Z m."
  log "Stopping. Re-run with a different --base-height-scale rather than"
  log "spending two more seeds on the same weight."
  exit 1
fi
log "gate passed; continuing with seeds 2 and 3"

train_seed 2
train_seed 3
log "=== HEIGHT ARM COMPLETE (3 seeds) ==="
