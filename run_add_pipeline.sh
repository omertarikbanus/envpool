#!/usr/bin/env bash
# One seed, end to end, with add semantics everywhere.
#
# The push is now a delta-v in BOTH training and evaluation. That is a
# deliberate deviation from unitree_rl_gym: legged_gym's _push_robots assigns
# to root_states[:, 7:9] ("Emulates an impulse by setting a randomized base
# velocity"), it does not accumulate. Every policy trained before this used the
# overwrite, so seed 1 is retrained from scratch rather than reused.
#
# Scope, per the user: ONE seed. The previous three-seed set-semantics campaign
# is not re-evaluated; it stands as its own measurement.
set -euo pipefail

EP=/home/tarik/quadruped_ws/envpool
QC=/home/tarik/quadruped_ws/quadcontrol
EXACT=data/rudin_add/v1_seed1
HEIGHT=data/rudin_add_height/v1_seed1

log() { echo "[$(date -Is)] $*"; }

# --- 1. rebuild with the additive push -----------------------------------
OLD=$(docker exec envpool-dev sha256sum /usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so | cut -d' ' -f1)
log "so before: $OLD"
log "rebuilding (make run; its env_step.py smoke aborts on a known WBC crash)"
docker exec -w /app/envpool envpool-dev bash -lc 'make run' > $EP/data/add_make_run.log 2>&1 || true
grep -q "Successfully installed envpool" $EP/data/add_make_run.log || { log "BUILD FAILED -- see data/add_make_run.log"; exit 1; }
NEW=$(docker exec envpool-dev sha256sum /usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so | cut -d' ' -f1)
log "so after: $NEW"
[ "$OLD" != "$NEW" ] || { log "BINARY UNCHANGED -- the patch did not reach the build"; exit 1; }

log "recipe check"
docker exec -w /app/envpool envpool-dev bash -lc \
  'python3 examples/tests/unitree_recipe_check.py --num-envs 64' > $EP/data/add_recipe_check.log 2>&1
grep -q "^FAIL" $EP/data/add_recipe_check.log && { log "RECIPE CHECK FAILED"; grep "^FAIL" $EP/data/add_recipe_check.log; exit 1; }
log "recipe check passed"

# --- 2. exact arm, seed 1, trained with additive pushes -------------------
log "=== exact arm seed 1 (add pushes), 1500 iterations ==="
docker exec envpool-dev bash -lc \
  "cd /app/envpool && mkdir -p $EXACT && python3 examples/train_unitree.py \
     --run-dir $EXACT --seed 1 > $EXACT/console.log 2>&1"
[ -f "$EP/$EXACT/model_1500.pt" ] || { log "exact seed 1 did not finish"; exit 1; }
log "=== exact arm seed 1 complete ==="

# --- 3. height arm, warm started from it ----------------------------------
log "=== height arm seed 1 (warm start, scale -30, target 0.30), 500 iterations ==="
docker exec envpool-dev bash -lc \
  "cd /app/envpool && mkdir -p $HEIGHT && python3 examples/train_unitree.py \
     --run-dir $HEIGHT --seed 1 --max-iterations 500 \
     --resume-from /app/envpool/$EXACT/model_1500.pt \
     --base-height-scale -30 --base-height-target 0.30 \
     > $HEIGHT/console.log 2>&1"
[ -f "$EP/$HEIGHT/model_500.pt" ] || { log "height seed 1 did not finish"; exit 1; }
log "=== height arm seed 1 complete ==="

# --- 4. save the trained policies ----------------------------------------
# No evaluation in this pipeline, by instruction. Ladders can be run later
# against these checkpoints; nothing here depends on them.
cd $EP
git add -A envpool/mujoco/gym/quadruped_pd.h examples/train_unitree.py \
        examples/common/rsl_vec_env.py run_add_pipeline.sh \
        data/rudin_height_ws data/rudin_add data/rudin_add_height 2>/dev/null || true
if ! git diff --cached --quiet; then
  git commit -q -F - <<MSG
Retrain seed 1 with additive pushes; record the height arm

The push is now a delta-v in training as well as evaluation. This is a
deliberate deviation from unitree_rl_gym, NOT a fidelity fix: legged_gym's
_push_robots assigns to root_states[:, 7:9] -- "Emulates an impulse by setting
a randomized base velocity" -- and does not accumulate. Every earlier policy
trained against the overwrite, so seed 1 is retrained from scratch here.

Also records the three warm-started height seeds trained under the previous
semantics, which measured 0.304, 0.300 and 0.288 m against a 0.30 m target
from a 0.131 m starting crouch.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
  git push -q origin main && log "envpool policies pushed"
fi
cd $QC
git add -A src include config evaluations docs 2>/dev/null || true
if ! git diff --cached --quiet; then
  git commit -q -F - <<MSG
Make the simulator push additive in training as well as evaluation

setBaseLinearVelocityXY overwrote qvel[0..1]; it is now
addBaseLinearVelocityXY and accumulates, renamed through SimHW and
RLPipelineRuntime so no stale caller can link. The semantics default and the
bad-value fallback are now "add", and envpool_train_v1.toml follows.

Deliberate deviation from legged_gym, which assigns rather than accumulates.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
  git push -q origin main && log "quadcontrol source pushed"
fi
log "=== ADD PIPELINE COMPLETE (1 seed trained and saved, no evaluation) ==="
