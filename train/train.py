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

# Import the correct Box class based on the gym version
import gym
import envpool
from envpool.python.protocol import EnvPool  # For type annotations

# Import Gymnasium spaces explicitly
import gymnasium
from gymnasium.spaces import Box

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first

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
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=250000, help="Total training timesteps")
    parser.add_argument("--tb-log-dir", type=str, default="./logs", help="TensorBoard log directory")
    parser.add_argument("--model-save-path", type=str, default="./quadruped_ppo_model", help="Model save path")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info("Experiment: quadruped_ppo_experiment")
    logging.info(f"Using EnvPool for environment {args.env_name} with {args.num_envs} envs. Seed: {args.seed}")

    np.random.seed(args.seed)
    
    # Create EnvPool environment using the gym interface.
    env = envpool.make(args.env_name, env_type="gym", num_envs=args.num_envs, seed=args.seed)
    
    # Set environment ID without modifying action_space directly
    env.spec.id = args.env_name
    
    # Use the adapter which will handle the action_space and observation_space conversion
    env = VecAdapter(env)
    env = VecMonitor(env)  # Monitor for tracking episode stats

    # Configure PPO model with tuned hyperparameters.
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     n_steps=2048,
    #     learning_rate=1e-3,
    #     gamma=0.9,
    #     gae_lambda=0.95,
    #     verbose=1,
    #     seed=args.seed,
    #     tensorboard_log=args.tb_log_dir,
    # )
    

    model = PPO(
    "MlpPolicy",
    env,
    n_steps=10000,
    
    learning_rate=3e-4,
    gamma=0.995,
    gae_lambda=0.97,
    clip_range=0.3,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[dict(pi=[256, 256],
                                      vf=[256, 256])]),
    verbose=1,
    seed=args.seed,
    tensorboard_log=args.tb_log_dir,
)
    logging.info("Starting training...")
    model.learn(total_timesteps=args.total_timesteps)
    logging.info("Training complete.")

    model.save(args.model_save_path)
    logging.info(f"Model saved at: {args.model_save_path}.zip")

    # Evaluate the model on the EnvPool environment.
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"EnvPool Evaluation - {args.env_name}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")


    env.close()


if __name__ == "__main__":
    main()