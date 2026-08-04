# Vertical push recovery — findings and next run

Session of 2026-08-03/04. Started from "why does the force-trained RL policy do
worse than MPC+WBIC on vertical pushes", ended with a one-line config change
that improves the *existing* models in both axes without retraining.

Evaluation throughout: a single 0.2 s pulse, impulse J in N·s, force = J/0.2 s.
Survival = body height stays in [0.20, 0.75] m for 5.5 s after onset.

---

## 1. The headline result

`floating_base_weight` on the RL path was hardcoded `setFloatingBaseWeight(1000)`
(`MdlRLLocomotionState.cc`) against `_W_rf = 1`. Lowering it to 10 improves push
recovery with **model X unchanged — no retraining**:

| direction | W=1000 (native) | W=10 |
|---|---|---|
| down | J50 (28, 36] | **(36, 44]** |
| left @20 N·s | 8/16 | **6/6** |
| left @24 N·s | 6/16 | **4/6** |

Now config-bound as `[rl_command_source] floating_base_weight`, default 1000 so
existing runs reproduce (commit d5c798a).

**`force_z_max` must stay 250.** Setting it to 150 — to centre the neutral action
on the 147 N body weight — drops ride height 0.398 → 0.325 m and collapses J50 to
(16, 22], worse than doing nothing.

Caveat: n=6 per cell, and survival was non-monotonic (67% at 28 N·s vs 83% at 36),
so the brackets are sound but the point estimates are not. Use 16 runs/cell and
`evaluations/j50.py` for publishable numbers.

## 2. Why the weight matters

The WBIC QP solves for a correction on top of the commanded foot force
(`WBIC.cpp:289`, `_Fr[i] = z[i + _dim_floating] + _Fr_des[i]`), subject to the
floating-base dynamics. `_W_floating` prices relaxing base acceleration against
`_W_rf = 1` on deviating from the command. Measured slope from the policy's
vertical force command to the realised force:

    W=1000 -> 0.02      W=100 -> 0.19      W=10 -> 0.76

At 0.02 the action head receives no gradient, which is why it sat at its
initialisation through 16 M steps and a 10x learning-rate increase, and why X and
Y are indistinguishable in the vertical axis (+6.6 vs +6.3 N under a 180 N push).

Two effects come together at lower weight: the policy gains authority, *and* the
robot rides higher because its ~250 N command now partly wins against the body
task's height target (clamped [0.28, 0.42] in `MdlRLLocomotionState`).

| config | ride height | 36 N·s |
|---|---|---|
| W=1000 fz250 | 0.2988 | 0/3 |
| W=100 fz250 | 0.3417 | 0/3 |
| W=10 fz250 | 0.3979 | 3/3 |
| W=1 fz250 | 0.4005 | 1/3 |
| W=10 fz150 | 0.3251 | 0/6 |

**For the paper:** part of the gain is ride-height trim, not better rejection —
the robot stands ~10 cm taller against a fixed 0.20 m floor, and WBIC could be
raised too. Decide whether to hold ride height constant across controllers or to
report the trim explicitly as part of the method.

## 3. What did not work

All of these were trained and measured at `W=10 + fz150` — the bad pairing — so
they were fighting a 40% survival handicap. Their J50 all sat at (16, 22]. They
are not clean negatives, but none showed promise:

- **Higher vertical training force** ([40,40,150]): no effect on the force
  channel; there was no gradient to use it.
- **Learning rate** 1e-5 -> 1e-4: `approx_kl` 0.003 -> 0.013, so the actor does
  move, but the vertical response stayed ~5% of WBIC's.
- **`ent_coef` 0.05 -> 0.005**: made discovery ~4x slower. The entropy was doing
  useful exploration; cutting it was a mistake.
- **Impulse-structured training** (0.2 s pulse, 2 s rest, commit 00cd5bd): taught
  *compliance* — the commanded force moved negative under a push. Continuous
  forcing was the only structure where it trended positive.
- **Adaptive curriculum** (commit 634c323): works correctly — probed against a
  5 M checkpoint the scale converges to 0.74 (|Fz| <= 111 N), and the implied
  ~67 N failure threshold agrees with the independent evaluation. It measures
  competence accurately; the policy simply gains little.

## 4. Other defects found

- **`base_linear_accel` is inert.** It uses gravity-inclusive specific force
  (~9.81 m/s^2), squares it, and exponentiates the negative: `exp(-100) ~ 0`
  always. 0.025 of reward weight producing no gradient. Left unchanged
  deliberately — fixing it would invalidate comparison with X and Y.
- **Observation gaps**, now closed (46 -> 56, commit 00cd5bd): body height (the
  MPC uses it; the reward penalises `|z - 0.35|`), IMU specific force `aBody`
  (available on hardware, and the only signal showing a push without integration
  lag), and foot positions for legs 2 and 3, which were computed then discarded.
- **The RL path bypasses `MdlConvexMPC` entirely.** The policy replaces the
  MPC's force solution rather than correcting it, so it must learn from scratch
  what a model-based QP computes analytically each tick. This is the remaining
  architectural difference and it is untested.

## 5. Gotchas

- **Never put a script under `envpool/`.** Python puts the script's directory on
  `sys.path[0]`, which shadows the installed wheel with the source tree — you
  silently get the old 46-dim observation. Already noted in `Notes.MD`.
- **Models are observation-dimension locked.** X and Y are 46-dim and cannot load
  against the 56-dim build. `data/wheel_obs46_backup/` holds the old `.so` with
  restore instructions; swap it in to evaluate them, and restore afterwards or
  every later measurement is silently wrong.
- **`--continue-training` resets the learning rate** to `FIXED_LEARNING_RATE`
  (`common/utils.py`) regardless of the loaded model. Use `--learning-rate`.
- **Reward and episode length are not comparable across runs** with different
  disturbance regimes; an easier environment raises both. Worse, `ep_len_mean` is
  dominated by episodes that never drew a strong push. Use the fixed-pulse
  survival curve.

## 6. Next run

X's recipe, three deviations, all evidence-backed:

| | X | next run |
|---|---|---|
| `floating_base_weight` | 1000 | **10** |
| `external_force_max` | [40,40,20] | **[40,40,40]** |
| observation | 46 | **56** |
| `force_z_max` | 250 | 250 |
| force structure | continuous | continuous |
| lr / `ent_coef` | 1e-5 / 0.05 | 1e-5 / 0.05 |
| curriculum | — | off |

20 M steps, 256 envs, from scratch, checkpoints every 1 M (~1.5 h).

Expect ride height ~0.398 m and `approx_kl` ~0.003 (the guard will not engage, as
with X). The open question is whether the vertical force channel trains now that
it has authority — under X's identical rate it never did, but it could not.

Then evaluate the best checkpoint against X at `W=10 + fz250`, **down and left**,
16 runs/cell, and fit with `evaluations/j50.py`. Lateral matters most: X's
advantage over WBIC there (21.8 vs 16.9 N·s) is the headline, and a vertical gain
that costs lateral is not a win.

## 7. Artifacts

Runs under `data/`, each with per-1 M-step checkpoints:

    260803_Z_forcez150_fbw1000_ABANDONED   150 N force, W=1000  (16 M)
    260803_Z_fbw10_fz150                   W=10, fz150          (17 M)
    260803_Z_obs56_fbw10_ent0005           obs56, ent 0.005     (20 M)
    260803_Z_obs56_fbw1                    W=1                  ( 6 M)
    260803_Z_Xhparams_fbw1                 W=1, fz250, lr 1e-5  ( 6 M, could not walk)
    260803_Z_impulse_fbw10                 impulse train        (13 M)
    260804_Z_curriculum                    curriculum           (20 M)
    wheel_obs46_backup                     46-dim .so for X/Y

Tools in `quadcontrol/evaluations/`: `force_response.py` (ramp latency,
feedforward delta, height loss), `ddes_curve.py` / `ddes_sweep.sh` (trace-averaged
response across checkpoints), `probe_curriculum_scale.py`, `snapshot_checkpoints.py`.
`QC_RL_BASE_CONFIG` swaps the RL base config without editing the shared file.
