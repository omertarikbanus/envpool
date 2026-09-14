#!/usr/bin/env bash
# v5: continue the v4b cardinal lineage. kPRelMax stays at 0.30 (user
# directed), so NO rebuild -- the binary in the container is already correct.
#
# v4b reached 54.28 against v1's 54.93 while carrying the ~7-point handicap the
# 0.30 clamp costs, and beat v1 outright on back, left and down. It is the best
# starting point available at this constant.
#
# Adaptive: each cycle trains 14M at 1e-5 and ladders the result. If a cycle
# improves the ladder it continues straight; if it does NOT, the next cycle
# switches to the sparse-pulse config rather than spending another 14M on a
# setting that has stopped paying. Phase b's training curve was flat for its
# whole 14M yet its ladder still rose 2 points, so the ladder -- not the
# training metric -- is what decides here.
#
# Every cycle writes to its own directory. Nothing is deleted.
set -u

EP=/home/tarik/quadruped_ws/envpool
EVALS=/home/tarik/quadruped_ws/quadcontrol/evaluations
LOG=$EP/data/v7_15x.log
STATE=$EP/data/v7_state.tsv
COMMON="--adaptive-lr 0 --target-kl 0.01 --checkpoint-freq 1000000 --randomize-init"
RB="0.5,1.0,1.5,2.0,2.5,3.0,3.5"; RE="4.0,4.5,5.0"
STEPS=14000000
DEADLINE=$(( $(date +%s) + 6*3600 ))

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
[ -f "$STATE" ] || printf 'cycle\tcfg\tsteps\ttrain_fall\ttrain_rew\tscore\tbest\n' > "$STATE"

PREV=./data/v5_c1
BEST=56.67          # v5_c1, the best model so far
CFG=e               # 1.5x each direction's measured cliff (user directed)
STALL=0

log "waiting for the v6 cycle-1 ladder before starting (same parent, clean A/B)"
for _ in $(seq 1 200); do
  grep -q 'cycle 1 SCORE=' "$EP/data/v6_restore_z.log" 2>/dev/null && break
  sleep 30
done
log "v6 cycle 1: $(grep 'cycle 1 SCORE=' "$EP/data/v6_restore_z.log" | tail -1)"
pkill -f '[r]un_v6_restore' 2>/dev/null; sleep 2
docker exec envpool-dev pkill -f '[e]xamples/train' 2>/dev/null; sleep 5
log "v6 stood down after its cycle-1 ladder (data kept)"
log "=== v7 start; 1.5x-cliff ceilings [4.0 6.3 3.5 3.5 3.8 6.8] from v5_c1 (56.67) ==="

for c in $(seq 1 8); do
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "budget exhausted"; break; }
  CC=$(printf 'e%d' "$c"); OUT=./data/v7_$CC; RD=20260914_v7_${CC}_n30

  log "--- cycle $c: cfg=$CFG, $STEPS steps @ 1e-5, $PREV -> $OUT ---"
  docker exec envpool-dev bash -lc "cd /app/envpool && mkdir -p $OUT && \
    cp $PREV/quadruped_ppo_model.zip $OUT/quadruped_ppo_model.zip && \
    cp $PREV/quadruped_ppo_model_vecnormalize.pkl $OUT/quadruped_ppo_model_vecnormalize.pkl" \
    >>"$LOG" 2>&1 || { log "seed FAILED"; break; }

  docker exec envpool-dev bash -lc "cd /app/envpool && python3 -u examples/train.py \
    --sim-config-path /app/quadcontrol/config/robots/sim/envpool_train_v3_card_${CFG}.toml \
    $COMMON --tb-log-dir $OUT/tb --model-save-path $OUT/quadruped_ppo_model \
    --total-timesteps $STEPS --learning-rate 1e-5 --continue-training" \
    >> "$EP/data/v7_${CC}_console.log" 2>&1 \
    || { log "cycle $c TRAINING FAILED"; break; }

  read -r TF TR < <(python3 - "$EP/${OUT#./}/tb/progress.csv" <<'PY'
import csv, sys, statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r.get('time/total_timesteps')]
t=rows[-max(1,len(rows)//4):]
f=lambda k:[float(r[k]) for r in t if r.get(k)]
print(f"{st.mean(f('rollout/is_fall_mean')):.4f} {st.mean(f('rollout/ep_rew_mean')):.1f}")
PY
) || { TF=nan; TR=nan; }
  log "cycle $c training tail: fall=$TF rew=$TR"

  for r in "$RB" "$RE"; do
    x=""; [ "$r" = "$RE" ] && x="--no-noforce"
    ( cd "$EVALS" && python3 run.py sweep --controllers ours --disturbance velocity \
        --push-semantics add --force-frame body_latched_at_onset --seed 42 \
        --runs 30 --survival-only --skip-existing $x --run-dir "$RD" \
        --rl-model-ours "/app/envpool/${OUT#./}/quadruped_ppo_model.zip" \
        --dirs fwd,back,left,right,up,down --impulses "$r" ) >>"$LOG" 2>&1
  done
  SCORE=$( cd "$EVALS" && python3 ladder_score.py "$RD" | tee -a "$LOG" | awk '/^SCORE/{print $3}' )
  [ -n "${SCORE:-}" ] || SCORE=0
  log "cycle $c SCORE=$SCORE  (best $BEST, v1 54.93)"
  printf '%d\t%s\t%d\t%s\t%s\t%s\t%s\n' "$c" "$CFG" "$STEPS" "$TF" "$TR" "$SCORE" "$BEST" >> "$STATE"

  if [ "$(python3 -c "print(1 if float('$SCORE')>float('$BEST') else 0)")" = "1" ]; then
    log "improved $BEST -> $SCORE; continuing on cfg=$CFG"
    BEST=$SCORE; STALL=0; PREV=$OUT
  else
    STALL=$((STALL+1))
    if [ "$CFG" = "e" ]; then
      CFG=es
      log "no improvement on dense pulses; switching to sparse-pulse cfg=bs from the same parent"
    else
      log "no improvement on cfg=$CFG either (stall $STALL)"
    fi
    [ "$STALL" -ge 3 ] && { log "three cycles without improvement -- stopping"; break; }
  fi
done

log "=== v5 finished; best $BEST ==="; cat "$STATE" | tee -a "$LOG"
