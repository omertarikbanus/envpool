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





def parse_args():
    parser = argparse.ArgumentParser(description="Train a quadrupedal controller using EnvPool and PPO.")
    parser.add_argument("--env-name", type=str, default="Humanoid-v4", help="EnvPool environment ID")
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--stack-frames", type=int, default=3, help="Observation frames to stack per environment step")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=8_000_000, help="Total training timesteps")

    parser.add_argument("--warm-start-steps", type=int, default=0, help="Warm start steps to run before optimisation")

    parser.add_argument("--tb-log-dir", type=str, default="./logs", help="TensorBoard log directory")
    parser.add_argument("--model-save-path", type=str, default="./quadruped_ppo_model", help="Model save path")
    parser.add_argument("--render-mode", type=bool, default=False, help="Render mode")
    parser.add_argument("--continue-training", action="store_true", help="Continue training from existing model if available")
    parser.add_argument("--force-new", action="store_true", help="Force start new training even if model exists")
    parser.add_argument("--use-vecnormalize", dest="use_vecnormalize", action="store_true", help="Enable VecNormalize wrapper (normalize observations and rewards)")
    parser.add_argument("--no-vecnormalize", dest="use_vecnormalize", action="store_false", help="Disable VecNormalize wrapper")
    parser.set_defaults(use_vecnormalize=True)
    return parser.parse_args()





def main():
    # Parse command-line arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging()
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Experiment: quadruped_ppo_experiment")
    logging.info(f"Using EnvPool for environment {args.env_name} with {args.num_envs} envs. Seed: {args.seed}")
    print(f"Using GPU: {th.cuda.is_available()}")
    
    np.random.seed(args.seed)
    
    # Create EnvPool environment using our utility function
    env = setup_environment(
        env_name=args.env_name,
        num_envs=args.num_envs,
        seed=args.seed,
        render_mode=args.render_mode,
        stack_frames=args.stack_frames
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
    
    # Update vecnormalize_wrapper reference if it was modified in create_or_load_model
    if args.use_vecnormalize and vecnormalize_wrapper is None:
        vecnormalize_wrapper = find_vecnormalize_wrapper(env)

    if args.warm_start_steps > 0 and getattr(model, "num_timesteps", 0) == 0:
        logging.info("Executing warm start for %d steps", args.warm_start_steps)
        warm_start_environment(env, args.warm_start_steps)
        logging.info("Warm start complete; proceeding to training.")

    model.set_logger(logger)

    logging.info("Starting training...")
    try:
        model.learn(total_timesteps=args.total_timesteps)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving model...")
        save_model_and_stats(model, args.model_save_path, vecnormalize_wrapper)
        logging.info(f"Model saved at: {args.model_save_path}.zip")
        return
    

    logging.info("Training complete.")

    # Save model and stats using our utility function
    save_model_and_stats(model, args.model_save_path, vecnormalize_wrapper)
    logging.info(f"Model saved at: {args.model_save_path}.zip")
    
    # Evaluate the model on the EnvPool environment.
    # For evaluation, we need to turn off VecNormalize training mode
    if args.use_vecnormalize and vecnormalize_wrapper is not None:
        vecnormalize_wrapper.training = False
        vecnormalize_wrapper.norm_reward = False  # Don't normalize rewards during evaluation
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"EnvPool Evaluation - {args.env_name}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print(f"Function {main.__name__} took {elapsed_time:.2f} seconds to run.")
    