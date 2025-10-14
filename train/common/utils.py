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
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.logger import configure
from datetime import datetime

from .vec_adapter import VecAdapter


def setup_environment(env_name, num_envs, seed, render_mode=None, stack_frames=1):
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
        net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
        log_std_init=-1.0,
        ortho_init=False,
    )


def create_ppo_model(env, policy_kwargs):
    """Create a new PPO model with specified hyperparameters."""
    return PPO(
        policy="MlpPolicy",
        env=env,
        # PPO hyper-parameters
        learning_rate=3e-4,
        clip_range=0.2,
        target_kl=0.01,
        n_steps=2048,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        clip_range_vf=0.1,
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
            monitor_wrapper = env if isinstance(env, VecMonitor) else None
            inner_env = monitor_wrapper.venv if monitor_wrapper is not None else env

            # unwrap existing VecNormalize to avoid double normalisation
            if isinstance(inner_env, VecNormalize):
                base_env = inner_env.venv
            else:
                base_env = inner_env

            vecnormalize_wrapper = VecNormalize.load(f"{model_save_path}_vecnormalize.pkl", base_env)
            # continue updating normalisation statistics during training
            vecnormalize_wrapper.training = True

            if monitor_wrapper is not None:
                monitor_wrapper.venv = vecnormalize_wrapper
                env = monitor_wrapper
            else:
                env = vecnormalize_wrapper
        elif use_vecnormalize and not vecnorm_exists:
            logging.warning(
                "VecNormalize statistics file %s_vecnormalize.pkl not found; "
                "continuing without loading normalization state.",
                model_save_path,
            )
        
        model = PPO.load(f"{model_save_path}.zip", env=env)
        
        print("Model loaded successfully. Continuing training...")
        print(f"Model hyperparameters: {model.policy_kwargs}")
        
        # Update hyperparameters for continued training
        # model.clip_range = 0.15
        # model.target_kl = 0.01
        # model.n_steps = 2048
        # model.batch_size = 1024
        # model.n_epochs = 5

        # Update learning rate; also refresh PPO's lr schedule so it is not overwritten
        new_learning_rate = 0.1e-5
        model.learning_rate = new_learning_rate
        model.lr_schedule = get_schedule_fn(new_learning_rate)
        for param_group in model.policy.optimizer.param_groups:
            param_group['lr'] = new_learning_rate

        # model.ent_coef = 0.2e-4

        # Update log_std bounds
        with th.no_grad():
            model.policy.log_std.clamp_(min=np.log(0.05), max=np.log(0.30))
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



def warm_start_environment(env, num_steps, base_action=None, noise_std=0.0):
    """Run safe actions before training to stabilise normalisation stats."""
    if num_steps <= 0:
        return
    logging.info("Warm start: running %d steps before optimisation.", num_steps)
    env.reset()
    action_space = env.action_space
    dtype = getattr(action_space, "dtype", None) or np.float32
    if base_action is None:
        base_action = np.zeros(action_space.shape, dtype=dtype)
    else:
        base_action = np.asarray(base_action, dtype=dtype)
    base_action = np.clip(base_action, action_space.low, action_space.high).astype(dtype, copy=False)
    if base_action.shape != action_space.shape:
        raise ValueError("base_action shape does not match action space shape")
    num_envs = getattr(env, "num_envs", 1)
    actions = np.repeat(base_action[np.newaxis, :], num_envs, axis=0).astype(dtype, copy=False)
    for _ in range(num_steps):
        if noise_std > 0.0:
            noise = np.random.normal(loc=0.0, scale=noise_std, size=actions.shape).astype(dtype, copy=False)
            step_actions = actions + noise
            step_actions = np.clip(step_actions, action_space.low, action_space.high).astype(dtype, copy=False)
        else:
            step_actions = actions
        _, _, _, _ = env.step(step_actions)
    env.reset()

def find_vecnormalize_wrapper(env):
    """Find VecNormalize wrapper in the environment stack."""
    current_env = env
    while hasattr(current_env, 'venv') and not isinstance(current_env, VecNormalize):
        current_env = current_env.venv
    if isinstance(current_env, VecNormalize):
        return current_env
    return None
