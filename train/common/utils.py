#!/usr/bin/env python3
"""
Utility functions and configuration for training and evaluation.
"""

import os
import logging
import numpy as np
import torch as th
import envpool
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.logger import configure
from datetime import datetime

from .vec_adapter import VecAdapter


def setup_environment(env_name, num_envs, seed, render_mode=None):
    """Set up the environment with proper wrappers."""
    # Create EnvPool environment
    env = envpool.make(
        env_name,
        env_type="gym",
        num_envs=num_envs,
        seed=seed,
        render_mode=render_mode
    )
    
    # Set environment ID
    env.spec.id = env_name
    
    # Apply VecAdapter
    env = VecAdapter(env)
    
    return env


def create_policy_kwargs():
    """Create policy kwargs for PPO model."""
    return dict(
        activation_fn=th.nn.Tanh,
        net_arch=[dict(pi=[64, 64, 64], vf=[64, 64, 64])],
        log_std_init=-3.0,
    )


def create_ppo_model(env, policy_kwargs):
    """Create a new PPO model with specified hyperparameters."""
    return PPO(
        policy="MlpPolicy",
        env=env,
        # PPO hyper-parameters
        learning_rate=5e-5,
        clip_range=0.1,
        target_kl=0.02,
        n_steps=4096*2,
        batch_size=2048,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.3,
        ent_coef=0.005,
        vf_coef=1.0,
        tensorboard_log="runs/ppo_taskspace",
        policy_kwargs=policy_kwargs,
        verbose=1,
    )


def setup_logging(run_dir=None):
    """Set up logging configuration."""
    if run_dir is None:
        run_dir = os.path.join("runs_csv", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    os.makedirs(run_dir, exist_ok=True)
    
    logger = configure(
        run_dir,
        format_strings=("stdout", "log", "tensorboard", "csv")
    )
    
    logging.basicConfig(level=logging.INFO)
    return logger


def ask_continue_or_restart(model_path):
    """Ask user whether to continue training or start fresh."""
    while True:
        print(f"\nFound existing model at: {model_path}.zip")
        choice = input("Do you want to (c)ontinue training or start (n)ew? [c/n]: ").lower().strip()
        if choice in ['c', 'continue']:
            return True
        elif choice in ['n', 'new']:
            return False
        else:
            print("Please enter 'c' for continue or 'n' for new.")


def create_or_load_model(model_save_path, env, policy_kwargs, use_vecnormalize=True, 
                        force_new=False, continue_training=False):
    """Create a new model or load existing one based on user choice."""
    model_exists = os.path.exists(f"{model_save_path}.zip")
    vecnorm_exists = os.path.exists(f"{model_save_path}_vecnormalize.pkl")
    
    # Determine whether to load existing model
    should_continue = False
    if model_exists:
        if force_new:
            print(f"Found existing model but force_new=True. Starting fresh training.")
            should_continue = False
        elif continue_training:
            print(f"Found existing model and continue_training=True. Continuing training.")
            should_continue = True
        else:
            # Interactive mode - ask user
            should_continue = ask_continue_or_restart(model_save_path)
    
    if should_continue and model_exists:
        print(f"Loading existing model from {model_save_path}.zip")
        
        # Load VecNormalize stats if they exist and we're using VecNormalize
        if use_vecnormalize and vecnorm_exists:
            print(f"Loading VecNormalize statistics from {model_save_path}_vecnormalize.pkl")
            env = VecNormalize.load(f"{model_save_path}_vecnormalize.pkl", env)
            # Important: set training=True to continue updating statistics
            env.training = True
        
        model = PPO.load(f"{model_save_path}.zip", env=env)
        
        print("Model loaded successfully. Continuing training...")
        print(f"Model hyperparameters: {model.policy_kwargs}")
        
        # Update hyperparameters for continued training
        model.clip_range = 0.2
        model.target_kl = 0.03
        
        # Update learning rate for actor
        for g in model.policy.optimizer.param_groups:
            g['lr'] = 1e-4
        
        model.ent_coef = 2e-4
        
        # Update log_std bounds
        with th.no_grad():
            model.policy.log_std.clamp_(min=np.log(0.05), max=np.log(0.20))
    else:
        print("Creating new model...")
        model = create_ppo_model(env, policy_kwargs)
        print("New model created.")
    
    return model, env


def save_model_and_stats(model, model_save_path, vecnormalize_wrapper=None):
    """Save model and VecNormalize statistics."""
    model.save(model_save_path)
    
    if vecnormalize_wrapper is not None:
        vecnormalize_wrapper.save(f"{model_save_path}_vecnormalize.pkl")
        logging.info(f"VecNormalize statistics saved at: {model_save_path}_vecnormalize.pkl")
    
    logging.info(f"Model saved at: {model_save_path}.zip")


def setup_vecnormalize(env, use_vecnormalize=True):
    """Set up VecNormalize wrapper if requested."""
    vecnormalize_wrapper = None
    if use_vecnormalize:
        print("Using VecNormalize wrapper...")
        vecnormalize_wrapper = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
        env = vecnormalize_wrapper
    return env, vecnormalize_wrapper


def find_vecnormalize_wrapper(env):
    """Find VecNormalize wrapper in the environment stack."""
    current_env = env
    while hasattr(current_env, 'venv') and not isinstance(current_env, VecNormalize):
        current_env = current_env.venv
    if isinstance(current_env, VecNormalize):
        return current_env
    return None
