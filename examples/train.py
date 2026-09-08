#!/usr/bin/env python3
"""
Train a quadrupedal controller using PPO with EnvPool.

This script uses the refactored common modules for better code organization
and reusability between training and evaluation scripts.
"""

import argparse
import logging
import json
import hashlib
import tempfile
from pathlib import Path
import numpy as np
from datetime import datetime

import torch as th
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_schedule_fn

# Import refactored common modules
from common import (
    setup_environment,
    create_policy_kwargs,
    setup_logging,
    create_or_load_model,
    save_model_and_stats,
    setup_vecnormalize,
    warm_start_environment,
    find_vecnormalize_wrapper
)

# Force PyTorch to use one thread (for speed)
th.set_num_threads(1)

MONITOR_INFO_KEYWORDS = ("is_timeout", "is_fall", "force_direction", "force_onset", "force_requested_impulse")


class EpisodeInfoLoggingCallback(BaseCallback):
    """Aggregate VecMonitor episode info fields and log rollout means."""

    def __init__(self, info_keys):
        super().__init__()
        self.info_keys = tuple(info_keys)
        self._values = {key: [] for key in self.info_keys}
        self._saturation = []
        self._directional = {i: [] for i in range(-1, 6)}
        self._timeouts = 0
        self._timeouts_without_onset = 0

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            self._saturation.append(float(np.mean(np.abs(actions) >= 1.0)))
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if not isinstance(episode, dict):
                continue
            direction = int(episode.get("force_direction", -1))
            if direction in self._directional:
                self._directional[direction].append(float(episode.get("is_timeout", 0)))
            if episode.get("is_timeout"):
                self._timeouts += 1
                if float(episode.get("force_onset", -1)) < 0:
                    self._timeouts_without_onset += 1
            for key in self.info_keys:
                value = episode.get(key)
                if value is None:
                    continue
                self._values[key].append(float(value))
        return True

    # A timeout with no latched onset means that episode was never pushed. The
    # policy can cause this on its own -- phase_delta_min is 0, so a low action
    # on the last channel nearly freezes the gait and the onset cycle never
    # arrives -- so a few are expected. A large fraction instead means the
    # disturbance pipeline is broken for most envs, which is how the EnvPool
    # thread-count defect showed up (169 of 256 envs never pushed).
    UNPUSHED_TIMEOUT_LIMIT = 0.10
    UNPUSHED_MIN_SAMPLE = 20

    def _on_rollout_end(self) -> None:
        if self._timeouts:
            fraction = self._timeouts_without_onset / self._timeouts
            self.logger.record("rollout/timeouts", self._timeouts)
            self.logger.record("rollout/timeout_without_onset_fraction", float(fraction))
            if (self._timeouts >= self.UNPUSHED_MIN_SAMPLE
                    and fraction > self.UNPUSHED_TIMEOUT_LIMIT):
                raise RuntimeError(
                    f"{self._timeouts_without_onset}/{self._timeouts} timeouts never "
                    "latched a force onset; those episodes trained with no disturbance. "
                    "Check EnvPool num_threads (must equal num_envs) and gait phase advancement.")
            self._timeouts = self._timeouts_without_onset = 0
        if self._saturation:
            self.logger.record("rollout/action_saturation_fraction", float(np.mean(self._saturation)))
            self._saturation.clear()
        std = self.model.policy.log_std.detach().exp().cpu().numpy()
        self.logger.record("train/std_min", float(std.min()))
        self.logger.record("train/std_max", float(std.max()))
        for i, value in enumerate(std.flat):
            self.logger.record(f"train/action_std_{i:02d}", float(value))
        names = ["noforce", "fwd", "back", "left", "right", "up", "down"]
        for direction, values in self._directional.items():
            if values:
                name = names[direction + 1]
                self.logger.record(f"rollout/recovery_{name}", float(np.mean(values)))
                self.logger.record(f"rollout/episodes_{name}", len(values))
                values.clear()
        for key, values in self._values.items():
            if values:
                self.logger.record(f"rollout/{key}_mean", float(np.mean(values)))
        self._values = {key: [] for key in self.info_keys}


class LogStdClampCallback(BaseCallback):
    """Project log_std after every optimizer step, including the final update."""

    def __init__(self, std_min: float, std_max: float):
        super().__init__()
        if not 0 < std_min <= std_max or not np.isfinite(std_max):
            raise ValueError("std bounds require 0 < std_min <= std_max < infinity")
        self._hook = None
        self.log_min = float(np.log(std_min))
        self.log_max = float(np.log(std_max))
        self._warned = False

    def _clamp(self) -> None:
        log_std = getattr(self.model.policy, "log_std", None)
        if log_std is None:
            if not self._warned:
                logging.warning("policy has no log_std; std clamp inactive")
                self._warned = True
            return
        with th.no_grad():
            if not th.isfinite(log_std).all():
                raise FloatingPointError("non-finite policy log_std")
            before = float(log_std.max())
            log_std.clamp_(min=self.log_min, max=self.log_max)
            if before > self.log_max and not self._warned:
                logging.warning(
                    "action std hit the ceiling (%.4f > %.4f) and was clamped; "
                    "entropy pressure exceeds what the task reward can offset "
                    "-- consider lowering ent_coef.",
                    float(np.exp(before)), float(np.exp(self.log_max)))
                self._warned = True

    def _on_training_start(self) -> None:
        self._clamp()
        self._hook = self.model.policy.optimizer.register_step_post_hook(
            lambda optimizer, args, kwargs: self._clamp())

    def _on_training_end(self) -> None:
        self._clamp()
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _on_step(self) -> bool:
        return True


class PeriodicCheckpointCallback(BaseCallback):
    """Save model + VecNormalize every `save_freq` timesteps, keeping history.

    Two jobs. Crash recovery: training otherwise only writes on a clean exit or
    KeyboardInterrupt, so an OOM kill (SIGKILL) loses the whole run; resume from
    the highest-numbered file. Checkpoint SELECTION: `ep_rew_mean` is velocity +
    orientation + height + smoothness and prices push recovery not at all, so
    whichever policy happens to be in the .zip when the clock runs out is chosen
    by the clock, not by the metric that gets reported. Beta1 -> Beta3 shows the
    drift is real -- healthy reward curves throughout while undisturbed mean_vx
    fell 0.756 -> 0.656 -> 0.638 and ride height fell to 0.274 m.

    Files are therefore `<path>_ckpt_<steps>`, not one overwritten `<path>_ckpt`:
    history cannot be recovered after the run, so it is kept by default and the
    selection sweep (worst-axis J50 per checkpoint) stays available as a
    post-hoc decision. Disk grows linearly with the run -- ~1 file per
    save_freq steps.
    """

    def __init__(self, save_freq, model_save_path):
        super().__init__()
        self.save_freq = int(save_freq)
        self.ckpt_prefix = f"{model_save_path}_ckpt"
        self._next_save = self.save_freq

    def _on_step(self) -> bool:
        if self.save_freq <= 0 or self.num_timesteps < self._next_save:
            return True
        self._next_save = self.num_timesteps + self.save_freq
        save_model_and_stats(
            self.model, f"{self.ckpt_prefix}_{self.num_timesteps}",
            find_vecnormalize_wrapper(self.model.get_env()),
        )
        logging.info("Checkpoint written at %d timesteps", self.num_timesteps)
        return True



class AdaptiveLRCallback(BaseCallback):
    """KL-adaptive learning rate -- the standard for legged-locomotion PPO.

    Rudin et al., "Learning to Walk in Minutes" (CoRL 2021), and every RSL-RL /
    legged_gym descendant: after each update, compare the measured policy KL to
    a desired KL and scale the learning rate by 1.5 toward it. The rate becomes
    a controlled variable instead of a guess, which matters here because the
    right value is not constant across a run -- early updates can take large
    steps, late ones cannot.

    Gamma1 run 1 (data/gamma1_run1_lr_too_high) ran this with desired_kl=0.01
    and LR_MAX=1e-4: measured approx_kl never exceeded 0.007 at any tested
    rate, an unreachable target that only ever pushes the increase branch, so
    it ratcheted past the point that stayed stable (5.06e-05) up to 7.59e-05
    and reward dipped. Run 2 fixed it by capping LR_MAX at 5e-05 -- the level
    that was already stable -- and is what Gamma4 restores (see the LR_MAX
    comment). Gamma2/3 tried the opposite direction (LR falling from a 1e-5
    ceiling); see git history for that variant's calibration.

    It must not run against SB3's target_kl at the same threshold. target_kl
    aborts the remaining epochs once a minibatch exceeds 1.5x it, so the LOGGED
    approx_kl -- a mean over the minibatches that actually ran -- can never reach
    this controller's decrease threshold at 2x desired. The decrease branch would
    then never fire while the increase branch keeps firing on the truncated
    (hence small) mean: a one-way ratchet to LR_MAX, discarding more of each
    rollout as it climbs. --adaptive-lr therefore moves target_kl out to
    4x desired_kl, where it stops interfering and becomes a genuine backstop for
    pathology rather than the primary regulator.

    Shrinking the step is the better response to an over-large update than
    discarding the rest of the batch: early stopping wastes the data and leaves
    the cause untouched, so the next rollout hits the same wall. The real trust
    region is clip_range (0.1) either way; target_kl was only ever a second line.
    """

    # Bounds are safety rails, not the operating range; 1.5 is RSL-RL's step.
    # Gamma4 restores Gamma1 run 2's config (desired_kl=0.01, LR_MAX=5e-05):
    # the one combination already measured stable across a full 40M-step run
    # -- LR climbed from 1e-5 to the 5e-05 ceiling in the first ~3.6M steps and
    # HELD there (approx_kl ~0.006, ep_rew_mean rising, no dip) for the rest of
    # training. Run 1's failure was climbing PAST this ceiling to 7.59e-05,
    # not the ceiling itself, so restoring exactly 5e-05 -- not Gamma2/3's
    # tighter 1e-5 -- is deliberate: this continuation trains a harder task
    # (up to 100 N.s vs Gamma3's 40-70 N.s ceiling) and needs more update
    # budget than Gamma3's flat 1e-5 gave the value function, whose
    # explained_variance never cleared ~0.4 across Gamma3's whole run.
    LR_MIN = 1e-7
    LR_MAX = 5e-05
    STEP = 1.5

    # Multiplicative deadband around desired_kl: act only when measured KL is
    # outside [desired_kl / TOL, desired_kl * TOL].
    #
    # TOL = 2.0 is the RSL-RL default and is what made desired_kl unusable in
    # the 0.003-0.005 range here: this environment produces approx_kl ~0.005 at
    # LR 5e-05, so a setpoint anywhere near the natural KL sits INSIDE the band
    # and the controller never fires in either direction. Gamma2 only descended
    # because desired_kl=0.001 put 2*desired_kl (0.002) below the natural KL.
    # A smaller TOL turns desired_kl into a real setpoint the controller tracks.
    # Pair a tight TOL with a smaller STEP or the coarse 1.5x jumps hunt around
    # the equilibrium instead of settling on it.
    TOL = 2.0

    def __init__(self, desired_kl, tol=None, step=None):
        super().__init__()
        self.desired_kl = float(desired_kl)
        self.tol = float(tol) if tol else self.TOL
        self.step = float(step) if step else self.STEP

    def _on_rollout_start(self) -> None:
        # train() runs at the end of each iteration and records approx_kl; the
        # logger is dumped after the NEXT rollout, so the value is still here.
        kl = self.model.logger.name_to_value.get("train/approx_kl")
        if kl is None:
            return
        lr = self.model.policy.optimizer.param_groups[0]["lr"]
        if kl > self.tol * self.desired_kl:
            new_lr = max(self.LR_MIN, lr / self.step)
        elif kl < self.desired_kl / self.tol:
            new_lr = min(self.LR_MAX, lr * self.step)
        else:
            new_lr = lr
        # PPO.train() re-reads lr_schedule every update, so setting the
        # optimizer alone would be overwritten on the next iteration.
        self.model.learning_rate = new_lr
        self.model.lr_schedule = get_schedule_fn(new_lr)
        for param_group in self.model.policy.optimizer.param_groups:
            param_group["lr"] = new_lr
        self.logger.record("train/adaptive_lr", new_lr)

    def _on_step(self) -> bool:
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train a quadrupedal controller using EnvPool and PPO.")
    parser.add_argument("--env-name", type=str, default="Humanoid-v4", help="EnvPool environment ID")
    parser.add_argument("--sim-config-path", type=str, default="/app/quadcontrol/config/robots/sim/envpool_train_Gamma4.toml", help="Path to quadcontrol simulation TOML used by Humanoid-v4")
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--num-threads", type=int, default=0, metavar="N",
                        help="EnvPool worker threads; 0 (default) means one per env. "
                             "Fewer threads than envs silently stalls the gait "
                             "scheduler in every env past the thread count, so those "
                             "envs never latch a force onset and never get pushed.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=40_000_000, help="Total training timesteps")
    parser.add_argument("--warm-start-steps", type=int, default=0, help="Warm start steps to run before optimisation")
    parser.add_argument("--tb-log-dir", type=str, default="./data/gamma4/tb", help="TensorBoard log directory")
    parser.add_argument("--model-save-path", type=str, default="./data/gamma4/quadruped_ppo_model", help="Model save path")
    # Gamma4 is a continuation of Gamma3, so this defaults True: the run is
    # launched non-interactively (docker exec, no TTY), and create_or_load_model
    # falls back to an interactive y/n prompt when neither this nor --force-new
    # is set -- which would hang forever with nothing to answer it. The actual
    # weights being resumed are a COPY of Gamma3's final model placed at
    # ./data/gamma4/quadruped_ppo_model.zip before launch, not Gamma3's own
    # file, so checkpointing here cannot touch the Gamma3 model still being
    # evaluated.
    parser.add_argument("--continue-training", action="store_true", default=True, help="Continue training from existing model if available")
    parser.add_argument("--force-new", action="store_true", help="Force start new training even if model exists")
    parser.add_argument("--use-vecnormalize", dest="use_vecnormalize", action="store_true", help="Enable VecNormalize wrapper (normalize observations and rewards)")
    parser.add_argument("--no-vecnormalize", dest="use_vecnormalize", action="store_false", help="Disable VecNormalize wrapper")
    parser.add_argument("--checkpoint-freq", type=int, default=2_000_000, help="Save a checkpoint every N timesteps as <model-save-path>_ckpt_<steps> (0 disables). Kept for the whole run: checkpoint history cannot be recovered afterwards.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override common.utils.FIXED_LEARNING_RATE for this run")
    parser.add_argument("--adaptive-lr", type=float, default=0.01, metavar="DESIRED_KL",
                        help="KL-adaptive learning rate (RSL-RL style) at the given desired KL, "
                             "bounded to [AdaptiveLRCallback.LR_MIN, LR_MAX]. Moves PPO's target_kl "
                             "out to 4x desired so it backstops instead of competing; "
                             "--learning-rate then sets the starting point rather than a fixed "
                             "value. 0 disables it.")
    parser.add_argument("--std-max", type=float, default=0.30, metavar="STD",
                        help="Hard ceiling on the policy action std, re-applied every "
                             "iteration (default 0.30). This is the guard that makes a "
                             "larger --ent-coef safe: without it nothing bounds log_std "
                             "during training (the clamp in create_or_load_model runs "
                             "once, on the resume path only), and data/gamma5_scratch "
                             "reached std 29365 over 120M steps while approx_kl stayed "
                             "~0.005 throughout. 0 disables.")
    parser.add_argument("--std-min", type=float, default=0.05, metavar="STD",
                        help="Floor on the policy action std (default 0.05, the "
                             "log_std_init value). Stops entropy collapse.")
    parser.add_argument("--kl-tol", type=float, default=None, metavar="TOL",
                        help="Multiplicative deadband for --adaptive-lr: act only when "
                             "measured KL leaves [desired_kl/TOL, desired_kl*TOL]. "
                             "Default 2.0 (RSL-RL). This env produces approx_kl ~0.005 "
                             "at LR 5e-05, so a desired_kl of 0.003-0.005 sits INSIDE "
                             "the default band and the controller never fires -- use "
                             "TOL ~1.25 to make desired_kl a setpoint that is actually "
                             "tracked.")
    parser.add_argument("--kl-step", type=float, default=None, metavar="STEP",
                        help="Multiplicative LR step for --adaptive-lr (default 1.5). "
                             "Use a smaller value (~1.1) with a tight --kl-tol, or the "
                             "coarse jumps hunt around the equilibrium.")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="Set PPO's own target_kl for this run. Only meaningful with "
                             "--adaptive-lr 0: when the adaptive controller is on it owns "
                             "target_kl (4x desired_kl) and this is ignored. Needed because "
                             "target_kl is SAVED INTO THE MODEL ZIP -- a run resumed from "
                             "adaptive-LR weights inherits that 4x backstop (e.g. Gamma4's "
                             "0.04) rather than create_ppo_model's 0.01, and no other flag "
                             "restores it.")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Post-training evaluation episodes; 0 skips (use directional validation).")
    parser.add_argument("--force-impulses", default=None, help="Six ceilings in N.s: fwd,back,left,right,up,down (cardinal sampler).")
    parser.add_argument("--force-seed", type=int, default=None, help="Explicit simulator force and command RNG seed.")
    parser.add_argument("--force-max", default=None, metavar="X,Y,Z",
                        help="Per-axis force ceiling in N for the box sampler "
                             "(external_force_max). At a 0.2 s pulse, 100 N = 20 N.s. "
                             "This is the Gamma5-style disturbance knob: all three axes "
                             "are drawn simultaneously, so it is not comparable to "
                             "--force-impulses, which draws one signed direction.")
    parser.add_argument("--force-off-duration", type=float, default=None, metavar="S",
                        help="Seconds between pushes. Small values (~0.8) give a train of "
                             "impulses, each redrawing direction and magnitude, for dense "
                             "disturbance-rejection training. A value past the episode "
                             "length gives exactly one push per episode, matching the "
                             "single-impulse evaluation protocol.")
    parser.add_argument("--max-episode-steps", type=int, default=1000, help="Training horizon at 100 Hz.")
    parser.add_argument("--ent-coef", type=float, default=None, help="Override PPO entropy coefficient for this run")
    parser.add_argument("--recovery-reward", action="store_true",
                        help="Track the fixed user vx command and remove the slow-gait reward incentive.")
    parser.add_argument("--recovery-target-vx", type=float, default=0.8,
                        help="Fixed user vx for --recovery-reward; must match the training command.")
    parser.add_argument("--freeze-normalization", action="store_true",
                        help="Keep the loaded observation and reward statistics fixed during continuation.")
    parser.add_argument("--gae-lambda", type=float, default=None,
                        help="Override GAE lambda in PPO and its rollout buffer.")
    parser.set_defaults(use_vecnormalize=True)
    return parser.parse_args()





def main():
    # Parse command-line arguments
    args = parse_args()
    env = None
    config_stub = None

    # Setup logging
    logger = setup_logging(args.tb_log_dir)
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Experiment: quadruped_ppo_experiment")
    logging.info(f"Using EnvPool for environment {args.env_name} with {args.num_envs} envs. Seed: {args.seed}")
    print(f"Using GPU: {th.cuda.is_available()}")
    
    np.random.seed(args.seed)

    try:
        env_config = {}
        if args.env_name.startswith("Humanoid"):
            config_path = Path(args.sim_config_path)
            if (args.force_impulses is not None or args.force_seed is not None
                    or args.force_off_duration is not None or args.force_max is not None):
                lines = [f"%include {config_path.name}", "", "[simulation]"]
                if args.force_max is not None:
                    ceiling = [float(x) for x in args.force_max.split(",")]
                    if len(ceiling) != 3 or not all(np.isfinite(x) and x >= 0 for x in ceiling):
                        raise ValueError("--force-max requires three finite nonnegative values")
                    lines += [f"external_force_max = {ceiling}"]
                if args.force_off_duration is not None:
                    if not np.isfinite(args.force_off_duration) or args.force_off_duration <= 0:
                        raise ValueError("--force-off-duration must be finite and positive")
                    lines += [f"external_force_off_duration = {args.force_off_duration}"]
                if args.force_impulses is not None:
                    ceilings = [float(x) for x in args.force_impulses.split(",")]
                    if len(ceilings) != 6 or not all(np.isfinite(x) and x >= 0 for x in ceilings):
                        raise ValueError("--force-impulses requires six finite nonnegative values")
                    lines += ["external_force_cardinal = true", f"external_force_impulses = {ceilings}"]
                if args.force_seed is not None:
                    if not 0 <= args.force_seed <= 0xFFFFFFFF - args.num_envs:
                        raise ValueError("--force-seed must fit uint32 including per-env offsets")
                    lines += [f"external_force_seed = {args.force_seed}", "", "[rl_command_source]", f"policy_random_seed = {args.force_seed}"]
                with tempfile.NamedTemporaryFile(mode="w", dir=config_path.parent, prefix="_train_force_", suffix=".toml", delete=False) as f:
                    f.write("\n".join(lines) + "\n")
                    config_stub = Path(f.name)
                config_path = config_stub
            env_config["sim_config_path"] = str(config_path)
            env_config["max_episode_steps"] = args.max_episode_steps
            # One worker thread per env. With the EnvPool default
            # (min(num_envs, cpu_count)) only the first num_threads envs get a
            # gait phase that advances; the rest walk in place, never reach the
            # onset cycle, and so train with no disturbance at all. Measured at
            # 256 envs on 56 cores: 169/256 envs never saw a push, and pinning
            # threads to envs cost ~4% throughput.
            threads = args.num_threads if args.num_threads > 0 else args.num_envs
            if threads < args.num_envs:
                logging.warning(
                    "--num-threads %d < --num-envs %d: %d envs will never be "
                    "pushed. Use 0 to pin one thread per env.",
                    threads, args.num_envs, args.num_envs - threads)
            env_config["num_threads"] = threads
            logging.info("EnvPool worker threads: %d for %d envs", threads, args.num_envs)
            sources = {}
            def collect_config(path):
                path = path.resolve()
                if str(path) in sources:
                    return
                sources[str(path)] = path.read_text()
                for line in sources[str(path)].splitlines():
                    if line.strip().startswith("%include "):
                        collect_config(path.parent / line.strip().split()[1])
            collect_config(config_path)
            import envpool
            library = Path(envpool.__file__).parent / "mujoco/mujoco_gym_envpool.so"
            manifest = {"started": datetime.now().isoformat(), "args": vars(args), "config_sources": sources,
                        "envpool_module": envpool.__file__,
                        "simulator_library": str(library),
                        "simulator_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
            Path(args.model_save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.model_save_path + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        # Create EnvPool environment using our utility function
        env = setup_environment(
            env_name=args.env_name,
            num_envs=args.num_envs,
            seed=args.seed,
            env_config=env_config,
        )
        
        if args.recovery_reward:
            from common.recovery_reward import RecoveryReward
            env = RecoveryReward(env, target_vx=args.recovery_target_vx)

        # Monitor raw rewards before normalization; PPO still trains on normalized rewards.
        env = VecMonitor(env, info_keywords=MONITOR_INFO_KEYWORDS)
        env, vecnormalize_wrapper = setup_vecnormalize(env, args.use_vecnormalize)

        # Create policy kwargs using our utility function
        policy_kwargs = create_policy_kwargs()

        model, env = create_or_load_model(
            model_save_path=args.model_save_path,
            env=env,
            policy_kwargs=policy_kwargs,
            use_vecnormalize=args.use_vecnormalize,
            force_new=args.force_new,
            continue_training=args.continue_training,
            seed=args.seed,
        )

        # Applied after create_or_load_model, which pins the rate to
        # FIXED_LEARNING_RATE (1e-5) on both fresh and resumed models. At that
        # rate approx_kl settles near 0.003 against target_kl 0.01, so PPO's own
        # guard never binds and the policy barely leaves its initialisation.
        # target_kl remains the brake when this is raised.
        if args.learning_rate is not None:
            logging.info("Overriding learning rate: %g", args.learning_rate)
            model.learning_rate = args.learning_rate
            model.lr_schedule = get_schedule_fn(args.learning_rate)
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = args.learning_rate

        # At the default 0.05 the entropy bonus outran the policy loss: the
        # action std climbed 0.050 -> 0.275 against the log_std ceiling of 0.30
        # over one 17 M-step run while reward fell, which is divergence rather
        # than exploration.
        if args.ent_coef is not None:
            logging.info("Overriding ent_coef: %g", args.ent_coef)
            model.ent_coef = args.ent_coef

        if args.use_vecnormalize:
            vecnormalize_wrapper = find_vecnormalize_wrapper(env)
        else:
            vecnormalize_wrapper = None

        if args.freeze_normalization:
            if vecnormalize_wrapper is None:
                raise ValueError("--freeze-normalization requires VecNormalize")
            vecnormalize_wrapper.training = False
            logging.info("Observation and reward normalization statistics frozen")
        if args.gae_lambda is not None:
            if not 0 <= args.gae_lambda <= 1:
                raise ValueError("--gae-lambda must be in [0, 1]")
            model.gae_lambda = args.gae_lambda
            model.rollout_buffer.gae_lambda = args.gae_lambda
            logging.info("GAE lambda: %g", model.gae_lambda)

        if args.warm_start_steps > 0 and getattr(model, "num_timesteps", 0) == 0:
            logging.info("Executing warm start for %d steps", args.warm_start_steps)
            warm_start_environment(env, args.warm_start_steps)
            logging.info("Warm start complete; proceeding to training.")

        model.set_logger(logger)

        logging.info("Starting training...")
        interrupted = False
        callbacks = [EpisodeInfoLoggingCallback(MONITOR_INFO_KEYWORDS)]
        if args.std_max and args.std_max > 0:
            callbacks.append(LogStdClampCallback(args.std_min, args.std_max))
            logging.info("Action std clamped to [%g, %g] after every optimizer step",
                         args.std_min, args.std_max)
        else:
            logging.warning("Action std clamp DISABLED (--std-max 0); nothing "
                            "bounds log_std during training.")
        if args.checkpoint_freq > 0:
            callbacks.append(
                PeriodicCheckpointCallback(args.checkpoint_freq, args.model_save_path))
        if args.adaptive_lr and args.adaptive_lr > 0:
            # 4x desired: the controller regulates inside [0.5x, 2x], so the
            # early stop at 1.5 * 4x = 6x desired cannot mask the decrease
            # branch and fires only on pathology.
            model.target_kl = 4.0 * args.adaptive_lr
            _tol = args.kl_tol or AdaptiveLRCallback.TOL
            _step = args.kl_step or AdaptiveLRCallback.STEP
            logging.info(
                "KL-adaptive learning rate: desired_kl=%g, deadband [%g, %g] "
                "(tol=%g), step=%g, bounds [%g, %g], target_kl backstop %g",
                args.adaptive_lr, args.adaptive_lr / _tol,
                args.adaptive_lr * _tol, _tol, _step,
                AdaptiveLRCallback.LR_MIN, AdaptiveLRCallback.LR_MAX,
                model.target_kl)
            callbacks.append(AdaptiveLRCallback(args.adaptive_lr,
                                               tol=args.kl_tol,
                                               step=args.kl_step))
            if args.target_kl is not None:
                logging.warning(
                    "--target-kl %g ignored: --adaptive-lr owns target_kl.",
                    args.target_kl)
        elif args.target_kl is not None:
            logging.info("Setting target_kl: %g (was %s)",
                         args.target_kl, model.target_kl)
            model.target_kl = args.target_kl
        episode_info_callback = CallbackList(callbacks)
        try:
            model.learn(total_timesteps=args.total_timesteps, callback=episode_info_callback)
        except KeyboardInterrupt:
            interrupted = True
            logging.info("Training interrupted by user. Saving model...")

        for callback in callbacks:
            if isinstance(callback, LogStdClampCallback):
                callback._on_training_end()
        save_model_and_stats(model, args.model_save_path, vecnormalize_wrapper)
        logging.info(f"Model saved at: {args.model_save_path}.zip")
        if interrupted:
            return

        logging.info("Training complete.")

        if args.eval_episodes <= 0:
            return

        # Evaluate the model on the EnvPool environment.
        # For evaluation, we need to turn off VecNormalize training mode
        if args.use_vecnormalize and vecnormalize_wrapper is not None:
            vecnormalize_wrapper.training = False
            vecnormalize_wrapper.norm_reward = False  # Don't normalize rewards during evaluation
        
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.eval_episodes)
        print(f"EnvPool Evaluation - {args.env_name}")
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    finally:
        if env is not None:
            env.close()
        if config_stub is not None:
            config_stub.unlink(missing_ok=True)


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print(f"Function {main.__name__} took {elapsed_time:.2f} seconds to run.")
    
