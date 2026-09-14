# ARCHIVE ONLY — do not use for current results

**Arm:** rudin (end-to-end RL, joint-PD), height-corrected variant
**Archived:** 2026-09-14   **Trained:** 2026-09-13
**Stage:** from-scratch attempt at `base_height_scale -30`, abandoned

## Why this is archived

**It never learned anything.** This was the first attempt at the
height-corrected arm: train from scratch with the height penalty on. At
iteration 500 it sat at mean reward 0.00, value-function loss 0.0000, action
std drifting UP from 1.0 to 1.21, and 13-step episodes; the checkpoint-250
policy measured base z = 0.077 m and vx = -0.003 m/s — lying on the floor.

**Cause: `only_positive_rewards`.** The exact recipe earns only about +0.07
per step in its first iterations. The height term costs -0.0031 — an order of
magnitude less than `lin_vel_z` or `dof_acc` — but that is still enough to
flip the per-step sum negative, and `only_positive_rewards` clips the whole
reward to zero, so no gradient ever exists. **No smaller weight is reliably
safe:** any penalty can flip that razor-thin early sum.

The general lesson, and the reason this is kept rather than deleted:
**sizing a reward weight against the CONVERGED reward level is the trap.**
-30 was correct for a converged policy (which earns ~1.16 per step) and fatal
from scratch.

## What replaced it

The warm start: `train_unitree.py --resume-from <ckpt>` loads a converged
exact policy and trains 500 iterations with the penalty on, so the total stays
positive throughout. Under `add` semantics that is
`data/rudin_add_height/v1_seed{1,2,3}`, the arm the comparison uses.

## Provenance

- Simulator .so at archival: `81f6ffcd15e0e9793bf0793078520d42e2774765b269ef79a43a073975018b67`
- Only final checkpoints are committed; intermediate 50-iteration checkpoints
  were left on disk.
