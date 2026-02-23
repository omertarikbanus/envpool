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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

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





def parse_args():
    parser = argparse.ArgumentParser(description="Train a quadrupedal controller using EnvPool and PPO.")
    parser.add_argument("--env-name", type=str, default="Humanoid-v4", help="EnvPool environment ID")
    parser.add_argument("--sim-config-path", type=str, default="/app/quadcontrol/config/robots/sim/envpool.toml", help="Path to quadcontrol simulation TOML used by Humanoid-v4")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=20e6, help="Total training timesteps")
    parser.add_argument("--warm-start-steps", type=int, default=0, help="Warm start steps to run before optimisation")
    parser.add_argument("--tb-log-dir", type=str, default="./logs", help="TensorBoard log directory")
    parser.add_argument("--model-save-path", type=str, default="./data/current/quadruped_ppo_model", help="Model save path")
    parser.add_argument("--continue-training", action="store_true", help="Continue training from existing model if available")
    parser.add_argument("--force-new", action="store_true", help="Force start new training even if model exists")
    parser.add_argument("--use-vecnormalize", dest="use_vecnormalize", action="store_true", help="Enable VecNormalize wrapper (normalize observations and rewards)")
    parser.add_argument("--no-vecnormalize", dest="use_vecnormalize", action="store_false", help="Disable VecNormalize wrapper")
    parser.set_defaults(use_vecnormalize=True)
    return parser.parse_args()





def main():
    # Parse command-line arguments
    args = parse_args()
    env = None

    # Setup logging
    logger = setup_logging()
    
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
        episode_info_callback = EpisodeInfoLoggingCallback(MONITOR_INFO_KEYWORDS)
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
    
