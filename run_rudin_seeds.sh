#!/usr/bin/env bash
# rudin arm, v1 plant -- seeds 2 and 3, run back to back after seed 1 finishes.
#
# Three seeds total (seed 1 launched by hand on 2026-09-12 ~20:48). One run at
# a time: each needs ~140 GB of the machine's 251 GB, so they cannot overlap.
#
# Everything except --seed and --run-dir is the recipe default, as for seed 1:
# 4096 envs, 1500 iterations, push_robots on, envpool_train_rudin.toml. The
# seed drives both the torch init and the env's friction buckets.
set -euo pipefail

RUNS=data/rudin_unitree
SEED1=$RUNS/v1_seed1
HOST_RUNS=/home/tarik/quadruped_ws/envpool/$RUNS

# 1. Wait for seed 1. Poll its process rather than its log: a crashed run stops
#    writing too, and we must not launch a second run on top of a live one.
echo "=== waiting for seed 1 === $(date -Is)"
while docker exec envpool-dev pgrep -f "run-dir $SEED1" > /dev/null 2>&1; do
  sleep 60
done
echo "=== seed 1 process gone === $(date -Is)"

# 2. Refuse to continue if seed 1 did not reach 1500. A partial seed 1 means
#    something is wrong with the setup, and two more runs would repeat it.
if [ ! -f "$HOST_RUNS/v1_seed1/model_1500.pt" ]; then
  echo "SEED 1 DID NOT COMPLETE: no model_1500.pt. Not launching seeds 2-3."
  exit 1
fi
echo "seed 1 complete: model_1500.pt present"

# 3. Seeds 2 and 3, sequentially.
for S in 2 3; do
  RUN=$RUNS/v1_seed$S
  if [ -e "$HOST_RUNS/v1_seed$S" ]; then
    echo "SKIP seed $S: $RUN already exists"
    continue
  fi
  echo "=== rudin v1 seed $S === $(date -Is)"
  docker exec envpool-dev bash -lc \
    "cd /app/envpool && mkdir -p $RUN && python3 examples/train_unitree.py \
       --run-dir $RUN --seed $S > $RUN/console.log 2>&1" \
    || { echo "SEED $S FAILED ($(date -Is)); stopping"; exit 1; }
  echo "=== rudin v1 seed $S complete === $(date -Is)"
done

echo "=== rudin v1 seeds 1-3 COMPLETE === $(date -Is)"
