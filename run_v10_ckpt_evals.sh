#!/usr/bin/env bash
# Ladder every 5M checkpoint of the v10 run, after its training finishes.
#
# USER DIRECTED: evaluate the checkpoints, not only the final model. v9's
# action std was still climbing when it stopped and over 40M projects past the
# 0.30 --std-max clamp at roughly 15M; if the best model is mid-run rather than
# at 40M, only the checkpoint ladders will show it.
#
# Sequential, after training rather than alongside it: a 30-env eval running
# next to the 256-env trainer would slow the 40M run. Results are deterministic
# either way, so the only cost of waiting is wall clock.
#
# Same protocol as every other ladder in the campaign -- n=30, the fixed
# 60-cell grid, seed 42, add semantics, body_latched_at_onset, survival-only.
# core/_rl_eval.py:513 derives each checkpoint's VecNormalize path from its own
# model filename, so every checkpoint is evaluated with its own statistics.
set -u
EP=/home/tarik/quadruped_ws/envpool
EVALS=/home/tarik/quadruped_ws/quadcontrol/evaluations
DIR=$EP/data/v10_v9ext
LOG=$EP/data/v10_ckpt_evals.log
RB="0.5,1.0,1.5,2.0,2.5,3.0,3.5"; RE="4.0,4.5,5.0"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "waiting for v10 training to finish"
while ! grep -q 'training complete' "$EP/data/v10_v9ext.log" 2>/dev/null; do
  pgrep -f '[r]un_v10_v9ext' >/dev/null || { log "v10 driver gone before training completed -- stopping"; exit 1; }
  sleep 120
done
log "v10 training complete; waiting for its own final-model ladder"
while pgrep -f '[r]un_v10_v9ext' >/dev/null 2>&1; do sleep 60; done
log "v10 driver finished"

for ck in $(ls "$DIR"/quadruped_ppo_model_ckpt_*.zip 2>/dev/null \
            | grep -v vecnormalize | sort -t_ -k5 -n); do
  steps=$(basename "$ck" .zip | sed 's/.*_ckpt_//')
  RD=20260914_v10_ckpt${steps}_n30
  [ -f "${ck%.zip}_vecnormalize.pkl" ] || { log "SKIP $steps -- no vecnormalize"; continue; }
  log "--- laddering checkpoint $steps -> results/$RD ---"
  for r in "$RB" "$RE"; do
    x=""; [ "$r" = "$RE" ] && x="--no-noforce"
    ( cd "$EVALS" && python3 run.py sweep --controllers ours --disturbance velocity \
        --push-semantics add --force-frame body_latched_at_onset --seed 42 \
        --runs 30 --survival-only --skip-existing $x --run-dir "$RD" \
        --rl-model-ours "/app/envpool/data/v10_v9ext/$(basename "$ck")" \
        --dirs fwd,back,left,right,up,down --impulses "$r" ) >>"$LOG" 2>&1
  done
  s=$( cd "$EVALS" && python3 ladder_score.py "$RD" | tee -a "$LOG" | awk '/^SCORE/{print $3}' )
  log "checkpoint $steps SCORE=$s"
done

log "=== all checkpoint ladders complete ==="
log "summary (v9 parent = 59.94, v8 = 57.06, v1 = 54.93):"
grep -E 'checkpoint [0-9]+ SCORE=' "$LOG" | sed 's/^/  /' | tee -a "$LOG"
