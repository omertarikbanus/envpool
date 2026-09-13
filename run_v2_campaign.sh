#!/usr/bin/env bash
# v2 "ours" campaign: alternate 5M training increments with a fixed-rung n=30
# ladder, widening the disturbance only as fast as the policy can absorb it.
#
# WHY A CURRICULUM AND NOT [5,5,5] FLAT: a single jump from v1's [1,1,0] to
# [5,5,5] collapsed the policy -- is_fall_mean 1.000 for 1.7M steps, reward
# flat at ~137 against v1's 724 (data/v2_555_collapsed). All three axes draw
# independently per pulse, so [5,5,5] is a ~5 m/s typical COMBINED kick ~12
# times an episode, against a policy that died at 5.0 m/s from one kick. Every
# pulse was fatal, so nothing was learnable. The width now advances only when
# the policy is still healthy, and steps BACK when it is not.
#
# Cycle 1 is deliberately at v1's own [1,1,0]: the only thing that differs from
# v1 there is kPRelMax 0.15 -> 0.30, so cycle 1's ladder isolates the footstep
# clamp change from the disturbance change.
#
# Every cycle trains into its OWN directory. model.learn() defaults to
# reset_num_timesteps=True, so a shared directory would have later cycles
# overwrite earlier checkpoints at identical step numbers. NOTHING is deleted.
set -u

EP=/home/tarik/quadruped_ws/envpool
QC=/home/tarik/quadruped_ws/quadcontrol
EVALS=$QC/evaluations
LOG=$EP/data/v2_campaign.log
STATE=$EP/data/v2_campaign_state.tsv
BUDGET_H=6
DEADLINE=$(( $(date +%s) + BUDGET_H*3600 ))
STEPS=5000000
LR=1e-5
RUNGS_BASE="0.5,1.0,1.5,2.0,2.5,3.0,3.5"
RUNGS_EXT="4.0,4.5,5.0"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

WIDTH=1          # index into the w1..w6 configs
MAXW=6
BEST=-1
STALL=0
PREV=data/v1_ours_add_stage3     # cycle 1 warm-starts from the v1 final model

[ -f "$STATE" ] || printf 'cycle\twidth\tsteps\ttrain_fall\ttrain_rew\tscore\tbest\n' > "$STATE"
log "=== v2 campaign start; budget ${BUDGET_H}h; baseline v1 n=100 score 54.93 ==="

for c in $(seq 1 24); do
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then log "budget exhausted"; break; fi

  CC=$(printf 'c%02d' "$c")
  OUT=data/v2_$CC
  CFG=/app/quadcontrol/config/robots/sim/envpool_train_v2_w${WIDTH}.toml
  RD=20260914_v2_${CC}_n30

  log "--- cycle $c: width w$WIDTH, $STEPS steps @ $LR, $PREV -> $OUT ---"
  docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p ./$OUT && \
    cp ./$PREV/quadruped_ppo_model.zip ./$OUT/quadruped_ppo_model.zip && \
    cp ./$PREV/quadruped_ppo_model_vecnormalize.pkl \
       ./$OUT/quadruped_ppo_model_vecnormalize.pkl" >>"$LOG" 2>&1 \
    || { log "seed FAILED"; break; }

  docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
    --sim-config-path $CFG --adaptive-lr 0 --target-kl 0.01 \
    --checkpoint-freq 1000000 --randomize-init \
    --tb-log-dir ./$OUT/tb --model-save-path ./$OUT/quadruped_ppo_model \
    --total-timesteps $STEPS --learning-rate $LR --continue-training" \
    >> "$EP/data/v2_${CC}_console.log" 2>&1 \
    || { log "cycle $c TRAINING FAILED -- see data/v2_${CC}_console.log"; break; }

  read -r TFALL TREW < <(python3 - "$EP/$OUT/tb/progress.csv" <<'PY'
import csv, sys, statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r.get('time/total_timesteps')]
tail=rows[-max(1,len(rows)//4):]
f=lambda k:[float(r[k]) for r in tail if r.get(k)]
print(f"{st.mean(f('rollout/is_fall_mean')):.4f} {st.mean(f('rollout/ep_rew_mean')):.1f}")
PY
) || { TFALL=nan; TREW=nan; }
  log "cycle $c training tail: fall=$TFALL rew=$TREW"

  log "cycle $c evaluating -> results/$RD"
  ( cd "$EVALS" && python3 run.py sweep --controllers ours --disturbance velocity \
      --push-semantics add --force-frame body_latched_at_onset --seed 42 \
      --runs 30 --survival-only --skip-existing --run-dir "$RD" \
      --rl-model-ours "/app/envpool/$OUT/quadruped_ppo_model.zip" \
      --dirs fwd,back,left,right,up,down --impulses "$RUNGS_BASE" ) >>"$LOG" 2>&1
  ( cd "$EVALS" && python3 run.py sweep --controllers ours --disturbance velocity \
      --push-semantics add --force-frame body_latched_at_onset --seed 42 \
      --runs 30 --survival-only --skip-existing --no-noforce --run-dir "$RD" \
      --rl-model-ours "/app/envpool/$OUT/quadruped_ppo_model.zip" \
      --dirs fwd,back,left,right,up,down --impulses "$RUNGS_EXT" ) >>"$LOG" 2>&1

  SCORE=$( cd "$EVALS" && python3 ladder_score.py "$RD" 2>>"$LOG" | tee -a "$LOG" \
           | awk '/^SCORE/{print $3}' )
  [ -n "${SCORE:-}" ] || SCORE=0
  log "cycle $c SCORE=$SCORE (best so far $BEST)"

  IMPROVED=$(python3 -c "print(1 if float('$SCORE') > float('$BEST') else 0)")
  if [ "$IMPROVED" = "1" ]; then BEST=$SCORE; STALL=0; else STALL=$((STALL+1)); fi
  printf '%d\tw%d\t%d\t%s\t%s\t%s\t%s\n' "$c" "$WIDTH" "$STEPS" "$TFALL" "$TREW" "$SCORE" "$BEST" >> "$STATE"

  # Width control: advance only while healthy, step back when drowning.
  UNHEALTHY=$(python3 -c "print(1 if float('$TFALL') > 0.5 else 0)" 2>/dev/null || echo 0)
  HEALTHY=$(python3 -c "print(1 if float('$TFALL') < 0.25 else 0)" 2>/dev/null || echo 0)
  if [ "$UNHEALTHY" = "1" ] && [ "$WIDTH" -gt 1 ]; then
    WIDTH=$((WIDTH-1)); log "training fall $TFALL > 0.5 -- stepping width DOWN to w$WIDTH"
  elif [ "$HEALTHY" = "1" ] && [ "$WIDTH" -lt "$MAXW" ]; then
    WIDTH=$((WIDTH+1)); log "training fall $TFALL < 0.25 -- widening to w$WIDTH"
  else
    log "holding width w$WIDTH"
  fi

  if [ "$STALL" -ge 3 ]; then log "no ladder improvement for 3 cycles -- stopping"; break; fi
  PREV=$OUT
done

log "=== v2 campaign finished; state table: $STATE ==="
cat "$STATE" | tee -a "$LOG"
