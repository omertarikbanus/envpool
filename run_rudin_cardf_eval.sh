#!/usr/bin/env bash
# Auto-start the evaluation for data/rudin_cardf/v1_seed1 the moment training
# ends. No confirmation gate, per the user (2026-09-16).
#
# Scope, per the user: ONE seed, ONE evaluation set, --survival-only.
# run_one_ladder.py is survival-only by construction (core COMMON list), and
# the separate no-kick height probe the original driver ran was DROPPED: every
# cell's summary.csv already carries mean_z, so the ladder's own noforce cell
# measures the posture. One eval set, not two.
#
# This script replaces the tail of run_rudin_cardf_seed1.sh, whose wrapper was
# detached at 03:10 so the scope change did not require killing training. The
# container-side trainer is a child of the container's init, not of that
# wrapper, so it ran on untouched.
#
# NO GIT. Nothing here commits, pushes or stages anything.
set -uo pipefail

EP=/home/tarik/quadruped_ws/envpool
EVALS=/home/tarik/quadruped_ws/quadcontrol/evaluations
SO=/usr/local/lib/python3.10/dist-packages/envpool/mujoco/mujoco_gym_envpool.so
RUN=data/rudin_cardf/v1_seed1
ITERS=500
RD=20260916_rudin_cardf_seed1_n30
LOG=$EP/data/run_rudin_cardf_seed1.log

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

PINNED=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
log "eval watcher up; simulator pinned at $PINNED"

# --- 1. wait for training to finish --------------------------------------
while docker exec envpool-dev pgrep -f "train_unitree.py --run-dir $RUN" >/dev/null 2>&1; do
  sleep 60
done
log "training process gone"

if [ ! -f "$EP/$RUN/model_$ITERS.pt" ]; then
  log "ABORT: $RUN/model_$ITERS.pt missing -- training did not finish cleanly"
  tail -20 "$EP/$RUN/console.log" | tee -a "$LOG"
  exit 1
fi
log "model_$ITERS.pt present"

NOW=$(docker exec envpool-dev sha256sum $SO | cut -d' ' -f1)
[ "$NOW" = "$PINNED" ] || { log "ABORT: simulator changed mid-campaign ($NOW)"; exit 1; }

# --- 2. the one ladder ----------------------------------------------------
log "=== LADDER -> $RD (30 runs/cell, survival-only, 6 directions) ==="
( cd "$EVALS" && python3 -u run_one_ladder.py \
    --model "/app/envpool/$RUN/model_$ITERS.pt" \
    --run-dir "$RD" --runs 30 ) >>"$LOG" 2>&1
LADDER_RC=$?
log "ladder exited rc=$LADDER_RC"

# --- 3. save the results IMMEDIATELY -------------------------------------
# Runs even if the ladder exited nonzero: a partial ladder is still data, and
# the user asked for results to be saved as soon as they exist.
log "=== scoring ==="
( cd "$EVALS" && python3 ladder_score.py "$RD" ) >>"$LOG" 2>&1

python3 - "$EVALS/results/$RD" "$EP/data/rudin_cardf/RESULTS.md" "$RD" <<'PY' >>"$LOG" 2>&1
import csv, statistics, sys
from pathlib import Path

res, out, rd = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cells = sorted((res / "rl_rudin").glob("*"))
rows = []
for c in cells:
    f = c / "summary.csv"
    if not f.is_file():
        continue
    with f.open() as fh:
        recs = list(csv.DictReader(fh))
    if not recs:
        continue
    surv = [int(r["survived"]) for r in recs]
    z = [float(r["mean_z"]) for r in recs
         if r.get("mean_z") not in (None, "", "nan")]
    rows.append((c.name, len(recs), 100.0 * sum(surv) / len(surv),
                 statistics.mean(z) if z else float("nan")))

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as fh:
    fh.write(f"# rudin + our curriculum, seed 1 -- `{rd}`\n\n")
    fh.write("Rudin height-corrected seed 1 continued 500 iterations under the\n"
             "cardinal velocity-kick curriculum (six 3.0 m/s ceilings, add\n"
             "semantics), height term held at -30 / 0.30 m throughout.\n\n")
    fh.write("Parent: `data/rudin_add_height/v1_seed1/model_500.pt` (base z 0.304 m).\n\n")
    fh.write("| cell | n | survival % | mean base z (m) |\n|---|---|---|---|\n")
    for name, n, s, z in rows:
        fh.write(f"| {name} | {n} | {s:.0f} | {z:.3f} |\n")
    zs = [z for *_ , z in rows if z == z]
    if zs:
        fh.write(f"\n**Posture check:** mean base z across all cells "
                 f"{statistics.mean(zs):.3f} m against the 0.30 m target and the "
                 f"parent's 0.304 m. The exact arm's crouch is 0.131 m.\n")
    fh.write(f"\nCells: {len(rows)}. Full data: `evaluations/results/{rd}/`.\n")
print(f"wrote {out}")
PY

log "=== RESULTS SAVED: data/rudin_cardf/RESULTS.md ==="
sed -n 1,40p "$EP/data/rudin_cardf/RESULTS.md" | tee -a "$LOG"
log "=== COMPLETE ==="
