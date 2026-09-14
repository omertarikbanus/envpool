This file belongs to the envpool project. Under this we are creating a
custom gym that depends on the quadcontrol project (`../quadcontrol`).

The code builds and runs only inside the `envpool-dev` Docker container.
Never start that container yourself -- ask the user to run `make docker-run`
if `docker ps --filter name=envpool-dev` comes back empty.

The command `make run` builds, installs, and runs the python package
(`docker exec envpool-dev bash -lc 'cd /app/envpool && make run'`).

We work on `./envpool/mujoco/gym/quadruped_wbc.h`, a custom gym environment
dependent on the quadcontrol module. (It was called `humanoid.h` until the
2026-09 rename; nothing in it relates to Gym's Humanoid.) It backs two task
ids that differ only in observation width -- `QuadrupedWBC-v0` (54, the Gamma
line) and `QuadrupedWBC-v1` (61, the Delta line's privileged critic tail).
Alongside it, `./envpool/mujoco/gym/quadruped_pd.h` backs `QuadrupedPD-v1`
(48 observations, 12 actions, policy dt 0.02 s), reproducing the Unitree
Go2 joint-PD recipe for the Rudin RL baseline arm.

For `QuadrupedWBC-*`, `kActionDim`, `kObservationDim`, and
`LocomotionReward::kNumTerms` are pinned against
quadcontrol's `include/supervisor/RLPipelineRuntime.hh` and
`include/modules/MdlRLCommandSource.hh` by
`quadcontrol/evaluations/tests/test_parameter_parity.py` -- changing one
without the others fails that test rather than silently truncating actions
or observations. `test_parameter_parity.py` likewise pins `PDConstants`
for `QuadrupedPD-v1`.

## Current ICRA campaigns

The authoritative paper-facing description is
`../quadcontrol/docs/ICRA_PAPER_REFERENCE.md`.

- **Proposed controller:** the current recipe is internally named Epsilon.
  `run_ramp123.sh` trains seeds 0, 1, and 2 serially for 85M interactions each
  with `envpool_train_ramp123.toml`. The cardinal velocity-change ceiling is
  1.0 m/s through 8.1M steps, 2.0 m/s through 24.3M, then 3.0 m/s through 85M.
  Learning rate is fixed at `1e-5`, `ent_coef` at `0.05`, and outputs are
  `data/epsilon_s{0,1,2}/`.
- **Provisional proposed-controller results:** `data/v9_explore/`. Its
  five-stage lineage is development history, not the paper recipe. v10 is a
  diagnostic extension of v9.
- **Official Rudin baseline:**
  `data/rudin_add_height/v1_seed{1,2,3}/model_500.pt`. Each is a 500-iteration
  height-adjustment warm start from its matching 1500-iteration
  `data/rudin_add` seed. The warm start is part of the official recipe; the
  exact-recipe seeds alone are not the measured paper arm.

Paper evaluations use an identical fixed ladder of instantaneous additive
base-velocity changes and report m/s. Do not combine these with legacy
force-pulse results reported in N.s.

## Training a new line

Historical lines baked argparse defaults into `examples/train.py`. Current
paper campaigns use committed launchers that pass and record their non-config
parameters. Do not edit global defaults to launch Epsilon.

Launch pattern (real tmux session, not a throwaway one, so it survives like
the user ran it themselves; never use `/tmp` for the launch script):

```bash
tmux new-session -d -s <line>_train \
  "docker exec -it envpool-dev bash -lc 'cd /app/envpool && QUADCONTROL_PARITY_DUMP=1 python3 examples/train.py 2>&1 | tee data/<line>/train.log'"
```

For the Rudin end-to-end PD arm (`QuadrupedPD-v1`), training uses
`examples/train_unitree.py` backed by vendored `rsl_rl v1.0.2`:

```bash
RUN=data/rudin_unitree/<run_id>
docker exec -it envpool-dev bash -lc "mkdir -p $RUN && python3 examples/train_unitree.py --run-dir $RUN > $RUN/console.log 2>&1"
```

`QUADCONTROL_PARITY_DUMP=1` prints the effective shared parameters
(`max_torque`, `contact_mu`, `body_height`, `floating_base_weight`,
Kp/Kd, `control_mode`) once per run -- confirm these against
`config/default/control_params.toml` before trusting a training run's
results; this is the only way to see the *effective* value after config
resolution (e.g. an `[rl_command_source] floating_base_weight` override).

`PeriodicCheckpointCallback` writes one file per checkpoint
(`<model-save-path>_ckpt_<steps>.zip` + matching `_vecnormalize.pkl`) --
checkpoint history is not recoverable if this regresses to overwriting one
file, so don't revert it.

## Continuing a line (warm start)

`create_or_load_model` uses `model_save_path` as BOTH the load source and
the save destination -- there's no separate "warm start from" path. To
continue line A into a new line B without risking A's files:

```bash
mkdir -p data/B
cp data/A/quadruped_ppo_model.zip data/B/quadruped_ppo_model.zip
cp data/A/quadruped_ppo_model_vecnormalize.pkl data/B/quadruped_ppo_model_vecnormalize.pkl
```

then point B's `--model-save-path`/defaults at `data/B/...` with
`--continue-training` (default it to `True` for that line -- the run is
non-interactive, docker exec with no TTY, and `create_or_load_model` falls
back to a `y/n` prompt with nothing able to answer it if neither
`--continue-training` nor `--force-new` is set).

## KL-adaptive learning rate (`AdaptiveLRCallback` in `train.py`)

RSL-RL-style: compare `approx_kl` to `desired_kl` each rollout and scale the
rate by `STEP` toward it. Two failure modes already hit, both one-way
ratchets when `desired_kl` sits outside what the system can actually
produce at the tested rates:

- `desired_kl` too high -> only the increase branch ever fires -> LR
  climbs to `LR_MAX` and sticks there for the rest of training (Gamma1 run
  1, `desired_kl=0.01` against a system whose `approx_kl` never exceeded
  0.007).
- `LR_MAX` too low for the task -> the value function can starve
  (`explained_variance` stuck ~0.07-0.4 all of Gamma3's run at a flat,
  non-adaptive 1e-5).

This callback belongs to the historical Gamma experiments. Epsilon disables
adaptive LR and holds `1e-5` for the entire run. See
`quadcontrol/docs/archive/gamma-line-record.md` for the older lines.

## Training artifacts are committed, not gitignored

`data/<line>/` artifacts are committed selectively for reproducibility. The
official Rudin arm retains each final source model, final height-adjusted
model, manifest, console log, TensorBoard event, and source provenance;
intermediate 50-iteration checkpoints are deliberately omitted. Each selected
proposed-controller checkpoint must retain its matching `_vecnormalize.pkl`,
manifest, logs, config/revision provenance, and evaluation cells. Do not add a
blanket `data/*` ignore rule (retired models live in `data/_archive/`).
