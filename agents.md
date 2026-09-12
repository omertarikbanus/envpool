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
Its `kActionDim`,
`kObservationDim`, and `LocomotionReward::kNumTerms` are pinned against
quadcontrol's `include/supervisor/RLPipelineRuntime.hh` and
`include/modules/MdlRLCommandSource.hh` by
`quadcontrol/evaluations/tests/test_parameter_parity.py` -- changing one
without the others fails that test rather than silently truncating actions
or observations.

## Training a new line

`examples/train.py`'s argparse defaults are baked in per line (sim config
path, model/tb save paths, LR settings) so training starts with a bare
`python3 examples/train.py` -- no flags. Each new line (Gamma2, Gamma3, ...)
means editing those defaults, not passing overrides at the command line.

Launch pattern (real tmux session, not a throwaway one, so it survives like
the user ran it themselves; never use `/tmp` for the launch script):

```bash
tmux new-session -d -s <line>_train \
  "docker exec -it envpool-dev bash -lc 'cd /app/envpool && QUADCONTROL_PARITY_DUMP=1 python3 examples/train.py 2>&1 | tee data/<line>/train.log'"
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

Gamma1 run 2's setting (`desired_kl=0.01`, `LR_MAX=5e-05`) is the one
combination measured stable across a full 40M-step run and is the default
to restore unless there's a specific reason to change it (see
`quadcontrol/docs/archive/gamma-line-record.md` for what each line actually
used and why; note the arms are now `kim`/`rudin`/`ours`).

## Training artifacts are committed, not gitignored

`data/<line>/` (checkpoints, `_vecnormalize.pkl`, tensorboard events,
`train.log`) is tracked in git, matching the existing `data/beta*` lines --
this is a deliberate project convention for reproducibility, not an
oversight. Don't add a blanket `data/*` gitignore rule.
