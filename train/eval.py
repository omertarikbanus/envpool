#!/usr/bin/env python3
"""
Evaluate trained PPO models with proper support for VecNormalize and model loading.

This script can evaluate models trained with or without VecNormalize,
automatically detects and loads normalization statistics if available,
and provides comprehensive evaluation metrics.
"""

import argparse
import os
import logging
import numpy as np
from packaging import version
import time

# Import the correct Box class based on the gym version
import gym
import envpool
from envpool.python.protocol import EnvPool

# Import Gymnasium spaces explicitly
import gymnasium
from gymnasium.spaces import Box

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

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
    parser = argparse.ArgumentParser(description="Evaluate trained PPO models with EnvPool.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model (without .zip extension)")
    parser.add_argument("--env-name", type=str, default="Humanoid-v4", help="EnvPool environment ID")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel evaluation environments")
    parser.add_argument("--n-eval-episodes", type=int, default=20, help="Number of episodes for evaluation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (no exploration)")
    parser.add_argument("--auto-detect-vecnorm", action="store_true", default=False, help="Automatically detect and load VecNormalize statistics")
    parser.add_argument("--vecnormalize-path", type=str, default=None, help="Explicit path to VecNormalize statistics file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def setup_environment(args):
    """Set up the evaluation environment with proper wrappers."""
    # Create EnvPool environment
    env = envpool.make(
        args.env_name, 
        env_type="gym", 
        num_envs=args.num_envs, 
        seed=args.seed,
        render_mode=True if args.render else None
    )
    
    # Set environment ID
    env.spec.id = args.env_name
    
    # Apply VecAdapter
    env = VecAdapter(env)
    
    return env


def load_model_and_normalization(args, env):
    """Load the model and handle VecNormalize if present."""
    model_path = args.model_path
    if not model_path.endswith('.zip'):
        model_path += '.zip'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Check for VecNormalize statistics
    vecnormalize_path = None
    if args.vecnormalize_path:
        vecnormalize_path = args.vecnormalize_path
    elif args.auto_detect_vecnorm:
        # Auto-detect VecNormalize file
        base_path = args.model_path.rstrip('.zip')
        potential_vecnorm_path = f"{base_path}_vecnormalize.pkl"
        if os.path.exists(potential_vecnorm_path):
            vecnormalize_path = potential_vecnorm_path
    
    # Apply VecNormalize if statistics file exists
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        print(f"Loading VecNormalize statistics from: {vecnormalize_path}")
        env = VecNormalize(env, training=False)  # Set training=False for evaluation
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False  # Don't normalize rewards during evaluation
        print("VecNormalize loaded and configured for evaluation.")
    else:
        if args.vecnormalize_path:
            print(f"Warning: Specified VecNormalize file not found: {args.vecnormalize_path}")
        else:
            print("No VecNormalize statistics found. Evaluating without normalization.")
    
    # Apply VecMonitor for episode statistics
    env = VecMonitor(env)
    
    # Load the PPO model
    model = PPO.load(model_path, env=env)
    print("Model loaded successfully.")
    
    return model, env


def detailed_evaluation(model, env, args):
    """Perform detailed evaluation with additional metrics."""
    print(f"\nStarting evaluation on {args.env_name}")
    print(f"Number of episodes: {args.n_eval_episodes}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Deterministic actions: {args.deterministic}")
    print("=" * 50)
    
    # Standard evaluation
    start_time = time.time()
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=args.n_eval_episodes,
        deterministic=args.deterministic,
        return_episode_rewards=False
    )
    eval_time = time.time() - start_time
    
    # Detailed evaluation with episode-by-episode results
    print("Running detailed evaluation...")
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(args.n_eval_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_length += 1
            
            if done[0] if isinstance(done, np.ndarray) else done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if args.verbose:
            print(f"Episode {episode + 1:2d}: Reward = {episode_reward:8.2f}, Length = {episode_length:4d}")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_time': eval_time
    }


def print_evaluation_results(results, args):
    """Print comprehensive evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"Environment: {args.env_name}")
    print(f"Model: {args.model_path}")
    print(f"Episodes evaluated: {args.n_eval_episodes}")
    print(f"Evaluation time: {results['eval_time']:.2f} seconds")
    print()
    
    print("REWARD STATISTICS:")
    print(f"  Mean reward: {results['mean_reward']:8.2f} ± {results['std_reward']:6.2f}")
    print(f"  Min reward:  {np.min(results['episode_rewards']):8.2f}")
    print(f"  Max reward:  {np.max(results['episode_rewards']):8.2f}")
    print(f"  Median:      {np.median(results['episode_rewards']):8.2f}")
    print()
    
    print("EPISODE LENGTH STATISTICS:")
    print(f"  Mean length: {np.mean(results['episode_lengths']):8.1f} ± {np.std(results['episode_lengths']):6.1f}")
    print(f"  Min length:  {np.min(results['episode_lengths']):8.0f}")
    print(f"  Max length:  {np.max(results['episode_lengths']):8.0f}")
    print(f"  Median:      {np.median(results['episode_lengths']):8.1f}")
    print()
    
    # Success rate (for environments where reward > 0 indicates success)
    successful_episodes = np.sum(np.array(results['episode_rewards']) > 0)
    success_rate = successful_episodes / len(results['episode_rewards']) * 100
    print(f"SUCCESS RATE (reward > 0): {success_rate:.1f}% ({successful_episodes}/{len(results['episode_rewards'])})")
    
    print("=" * 60)


def main():
    args = parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    print("PPO Model Evaluation Script")
    print("=" * 30)
    
    try:
        # Set up environment
        env = setup_environment(args)
        
        # Load model and normalization
        model, env = load_model_and_normalization(args, env)
        
        # Perform evaluation
        results = detailed_evaluation(model, env, args)
        
        # Print results
        print_evaluation_results(results, args)
        
        # Save results to file
        results_file = f"{args.model_path}_evaluation_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"Evaluation Results for {args.model_path}\n")
            f.write(f"Environment: {args.env_name}\n")
            f.write(f"Episodes: {args.n_eval_episodes}\n")
            f.write(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
            f.write(f"Episode rewards: {results['episode_rewards']}\n")
            f.write(f"Episode lengths: {results['episode_lengths']}\n")
        
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    finally:
        # Clean up
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    main()
