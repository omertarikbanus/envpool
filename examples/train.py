#!/usr/bin/env python3
"""
Train a quadrupedal controller using PPO with EnvPool.

This script uses the refactored common modules for better code organization
and reusability between training and evaluation scripts.
"""

import argparse
import logging
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

MONITOR_INFO_KEYWORDS = ()


class EpisodeInfoLoggingCallback(BaseCallback):
    """Aggregate VecMonitor episode info fields and log rollout means."""

    def __init__(self, info_keys):
        super().__init__()
        self.info_keys = tuple(info_keys)
        self._values = {key: [] for key in self.info_keys}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if not isinstance(episode, dict):
                continue
            for key in self.info_keys:
                value = episode.get(key)
                if value is None:
                    continue
                self._values[key].append(float(value))
        return True

    def _on_rollout_end(self) -> None:
        for key, values in self._values.items():
            if values:
                self.logger.record(f"rollout/{key}_mean", float(np.mean(values)))
        self._values = {key: [] for key in self.info_keys}


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

    def __init__(self, desired_kl):
        super().__init__()
        self.desired_kl = float(desired_kl)

    def _on_rollout_start(self) -> None:
        # train() runs at the end of each iteration and records approx_kl; the
        # logger is dumped after the NEXT rollout, so the value is still here.
        kl = self.model.logger.name_to_value.get("train/approx_kl")
        if kl is None:
            return
        lr = self.model.policy.optimizer.param_groups[0]["lr"]
        if kl > 2.0 * self.desired_kl:
            new_lr = max(self.LR_MIN, lr / self.STEP)
        elif kl < 0.5 * self.desired_kl:
            new_lr = min(self.LR_MAX, lr * self.STEP)
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
    parser.add_argument("--ent-coef", type=float, default=None, help="Override PPO entropy coefficient for this run")
    parser.set_defaults(use_vecnormalize=True)
    return parser.parse_args()





def main():
    # Parse command-line arguments
    args = parse_args()
    env = None

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
            env_config["sim_config_path"] = args.sim_config_path

        # Create EnvPool environment using our utility function
        env = setup_environment(
            env_name=args.env_name,
            num_envs=args.num_envs,
            seed=args.seed,
            env_config=env_config,
        )
        
        # Apply VecNormalize if requested (BEFORE VecMonitor)
        env, vecnormalize_wrapper = setup_vecnormalize(env, args.use_vecnormalize)
        
        env = VecMonitor(env)  # Monitor for tracking episode stats

        # Create policy kwargs using our utility function
        policy_kwargs = create_policy_kwargs()

        model, env = create_or_load_model(
            model_save_path=args.model_save_path,
            env=env,
            policy_kwargs=policy_kwargs,
            use_vecnormalize=args.use_vecnormalize,
            force_new=args.force_new,
            continue_training=args.continue_training
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

        if args.warm_start_steps > 0 and getattr(model, "num_timesteps", 0) == 0:
            logging.info("Executing warm start for %d steps", args.warm_start_steps)
            warm_start_environment(env, args.warm_start_steps)
            logging.info("Warm start complete; proceeding to training.")

        model.set_logger(logger)

        logging.info("Starting training...")
        interrupted = False
        callbacks = [EpisodeInfoLoggingCallback(MONITOR_INFO_KEYWORDS)]
        if args.checkpoint_freq > 0:
            callbacks.append(
                PeriodicCheckpointCallback(args.checkpoint_freq, args.model_save_path))
        if args.adaptive_lr and args.adaptive_lr > 0:
            # 4x desired: the controller regulates inside [0.5x, 2x], so the
            # early stop at 1.5 * 4x = 6x desired cannot mask the decrease
            # branch and fires only on pathology.
            model.target_kl = 4.0 * args.adaptive_lr
            logging.info(
                "KL-adaptive learning rate: desired_kl=%g, bounds [%g, %g], "
                "target_kl backstop moved to %g",
                args.adaptive_lr, AdaptiveLRCallback.LR_MIN,
                AdaptiveLRCallback.LR_MAX, model.target_kl)
            callbacks.append(AdaptiveLRCallback(args.adaptive_lr))
        episode_info_callback = CallbackList(callbacks)
        try:
            model.learn(total_timesteps=args.total_timesteps, callback=episode_info_callback)
        except KeyboardInterrupt:
            interrupted = True
            logging.info("Training interrupted by user. Saving model...")

        save_model_and_stats(model, args.model_save_path, vecnormalize_wrapper)
        logging.info(f"Model saved at: {args.model_save_path}.zip")
        if interrupted:
            return

        logging.info("Training complete.")

        # Evaluate the model on the EnvPool environment.
        # For evaluation, we need to turn off VecNormalize training mode
        if args.use_vecnormalize and vecnormalize_wrapper is not None:
            vecnormalize_wrapper.training = False
            vecnormalize_wrapper.norm_reward = False  # Don't normalize rewards during evaluation
        
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
        print(f"EnvPool Evaluation - {args.env_name}")
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print(f"Function {main.__name__} took {elapsed_time:.2f} seconds to run.")
    
