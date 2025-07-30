#!/usr/bin/env python3
"""
Train a quadrupedal controller using PPO with EnvPool.

This script always uses EnvPool's gym interface, wraps the EnvPool
object using a VecAdapter (inspired by EnvPool's SB3 example) to be compatible with SB3,
and converts the action and observation spaces to float32 to satisfy SB3's requirements.
"""

import argparse
import logging
import os
import numpy as np
from packaging import version
from datetime import datetime

# Import the correct Box class based on the gym version
import gym
import envpool
from envpool.python.protocol import EnvPool  # For type annotations

# Import Gymnasium spaces explicitly
import gymnasium
from gymnasium.spaces import Box

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
from stable_baselines3.common.logger import configure
# Force PyTorch to use one thread (for speed)
th.set_num_threads(1)
is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")


class VecAdapter(VecEnvWrapper):
    """
    Convert an EnvPool object to a Stable-Baselines3-compatible VecEnv.
    This adapter sets the number of environments from the EnvPool spec and
    implements step_wait to handle terminal resets, attaching terminal observations.
    Also converts spaces to be SB3-compatible.
    """
    def __init__(self, venv: EnvPool):
        # Set the number of environments from EnvPool's config.
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv)
        
        # Convert the action space to Gymnasium's Box with float32 (SB3 requires this)
        self.action_space = Box(
            low=venv.action_space.low.astype(np.float32),
            high=venv.action_space.high.astype(np.float32),
            shape=venv.action_space.shape,
            dtype=np.float32,
        )
        
        # Convert the observation space to float32 as well
        # First, check if it's a Box space
        if isinstance(venv.observation_space, (gym.spaces.Box, gymnasium.spaces.Box)):
            self.observation_space = Box(
                low=venv.observation_space.low.astype(np.float32),
                high=venv.observation_space.high.astype(np.float32),
                shape=venv.observation_space.shape,
                dtype=np.float32,
            )
        else:
            # If not a Box space, keep the original (but this might cause issues)
            self.observation_space = venv.observation_space
    
    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
    
    def reset(self):
        if is_legacy_gym:
            obs = self.venv.reset()
        else:
            obs = self.venv.reset()[0]
        # Convert observations to numpy array (if not already) and ensure float32
        obs = np.asarray(obs, dtype=np.float32)
        return obs
    
    def seed(self, seed: int = None) -> None:
        # Seeding is set at EnvPool creation.
        pass
    
    def step_wait(self):
        if is_legacy_gym:
            obs, rewards, dones, info_dict = self.venv.step(self.actions)
        else:
            obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
            dones = terms + truncs
        
        # Ensure observations are float32
        obs = np.asarray(obs, dtype=np.float32)
        
        infos = []
        for i in range(self.num_envs):
            info_i = {key: info_dict[key][i] for key in info_dict.keys() if isinstance(info_dict[key], np.ndarray)}
            if dones[i]:
                info_i["terminal_observation"] = obs[i]
                if is_legacy_gym:
                    reset_obs = self.venv.reset(np.array([i]))
                else:
                    reset_obs = self.venv.reset(np.array([i]))[0]
                obs[i] = np.asarray(reset_obs, dtype=np.float32)
            infos.append(info_i)
        return obs, rewards, dones, infos


def parse_args():
    parser = argparse.ArgumentParser(description="Train a quadrupedal controller using EnvPool and PPO.")
    parser.add_argument("--env-name", type=str, default="Humanoid-v4", help="EnvPool environment ID")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=100_000_000, help="Total training timesteps")

    parser.add_argument("--tb-log-dir", type=str, default="./logs", help="TensorBoard log directory")
    parser.add_argument("--model-save-path", type=str, default="./quadruped_ppo_model", help="Model save path")
    parser.add_argument("--render-mode", type=bool, default=False, help="Render mode")
    parser.add_argument("--continue-training", action="store_true", help="Continue training from existing model if available")
    parser.add_argument("--force-new", action="store_true", help="Force start new training even if model exists")
    parser.add_argument("--use-vecnormalize", action="store_true", help="Use VecNormalize wrapper (normalize observations and rewards)")
    return parser.parse_args()


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


def create_or_load_model(args, env, policy_kwargs, use_vecnormalize=True):
    """Create a new model or load existing one based on user choice."""
    model_exists = os.path.exists(f"{args.model_save_path}.zip")
    vecnorm_exists = os.path.exists(f"{args.model_save_path}_vecnormalize.pkl")
    
    # Determine whether to load existing model
    should_continue = False
    if model_exists:
        if args.force_new:
            print(f"Found existing model but --force-new specified. Starting fresh training.")
            should_continue = False
        elif args.continue_training:
            print(f"Found existing model and --continue-training specified. Continuing training.")
            should_continue = True
        else:
            # Interactive mode - ask user
            should_continue = ask_continue_or_restart(args.model_save_path)
    
    if should_continue and model_exists:
        print(f"Loading existing model from {args.model_save_path}.zip")
        
        # Load VecNormalize stats if they exist and we're using VecNormalize
        if use_vecnormalize and vecnorm_exists:
            print(f"Loading VecNormalize statistics from {args.model_save_path}_vecnormalize.pkl")
            env = VecNormalize.load(f"{args.model_save_path}_vecnormalize.pkl", env)
            # Important: set training=True to continue updating statistics
            env.training = True
        
        model = PPO.load(f"{args.model_save_path}.zip", env=env)
        print("Model loaded successfully. Continuing training...")
    else:
        print("Creating new model...")
        model = PPO(
            policy="MlpPolicy",
            env=env,

            # ───── PPO hyper-parameters (Appendix, Table "Hyperparameters for Proximal Policy Gradient") ─────
            learning_rate=5e-4,      # "Adam stepsize" ≈ 1 × 10⁻³
            clip_range=0.2,            # tighten the trust‐region
            target_kl=0.05,            # early stop if KL > 1%
            n_steps=1024,           # 5 000 samples/iteration (match 5 000 MuJoCo steps)
            batch_size=128,        # "Minibatch size"
            n_epochs=8,              # "Number epochs"
            gamma=0.99,              # "Discount (γ)"
            gae_lambda=0.95,         # standard value; paper does not override
            max_grad_norm=0.5,      # "Max gradient norm"
            ent_coef=0.08,            # paper does not add entropy bonus
            vf_coef=1.0,             # SB3 default; paper gives no separate weight

            # ───── bookkeeping ─────
            tensorboard_log="runs/ppo_taskspace",
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
        print("New model created.")
    
    return model, env


def main():
    # Parse command-line arguments
    args = parse_args()

    # 1️⃣  Fresh run-folder so old logs stay intact
    run_dir = os.path.join("runs_csv", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # 2️⃣  Build a logger that keeps every format
    #     (stdout, log, tensorboard, csv)
    logger = configure(
        run_dir,
        format_strings=("stdout", "log", "tensorboard", "csv")  # same as SB3 default
    )

    logging.basicConfig(level=logging.INFO)
    logging.info("Experiment: quadruped_ppo_experiment")
    logging.info(f"Using EnvPool for environment {args.env_name} with {args.num_envs} envs. Seed: {args.seed}")
    print(f"Using GPU: {th.cuda.is_available()}")
    
    np.random.seed(args.seed)
    
    # Create EnvPool environment using the gym interface.
    env = envpool.make(args.env_name, env_type="gym", num_envs=args.num_envs, seed=args.seed, render_mode=args.render_mode)
    
    # Set environment ID without modifying action_space directly
    env.spec.id = args.env_name

    # Use the adapter which will handle the action_space and observation_space conversion
    env = VecAdapter(env)
    
    # Apply VecNormalize if requested (BEFORE VecMonitor)
    vecnormalize_wrapper = None
    if args.use_vecnormalize:
        print("Using VecNormalize wrapper...")
        vecnormalize_wrapper = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
        env = vecnormalize_wrapper
    
    env = VecMonitor(env)  # Monitor for tracking episode stats

    policy_kwargs = dict(
        # 1 hidden layer, 256 units, ReLU as in the paper
        activation_fn=th.nn.ReLU,
        net_arch=[dict(pi=[64], vf=[64])],
        # initialise exploration noise to exp(–2.5) ≈ 0.082
        log_std_init=-2.0,

    )

    model, env = create_or_load_model(args, env, policy_kwargs, use_vecnormalize=args.use_vecnormalize)
    
    # Update vecnormalize_wrapper reference if it was modified in create_or_load_model
    if args.use_vecnormalize and vecnormalize_wrapper is None:
        # Find the VecNormalize wrapper in the environment stack
        current_env = env
        while hasattr(current_env, 'venv') and not isinstance(current_env, VecNormalize):
            current_env = current_env.venv
        if isinstance(current_env, VecNormalize):
            vecnormalize_wrapper = current_env

    model.set_logger(logger)

    logging.info("Starting training...")
    try:
        model.learn(total_timesteps=args.total_timesteps)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving model...")
        model.save(args.model_save_path)
        # Save VecNormalize statistics if using VecNormalize
        if args.use_vecnormalize and vecnormalize_wrapper is not None:
            vecnormalize_wrapper.save(f"{args.model_save_path}_vecnormalize.pkl")
            logging.info(f"VecNormalize statistics saved at: {args.model_save_path}_vecnormalize.pkl")
        logging.info(f"Model saved at: {args.model_save_path}.zip")
        return
    

    logging.info("Training complete.")

    model.save(args.model_save_path)
    # Save VecNormalize statistics if using VecNormalize
    if args.use_vecnormalize and vecnormalize_wrapper is not None:
        vecnormalize_wrapper.save(f"{args.model_save_path}_vecnormalize.pkl")
        logging.info(f"VecNormalize statistics saved at: {args.model_save_path}_vecnormalize.pkl")
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
    