#!/usr/bin/env python3
"""
Evaluation utilities for trained PPO models.
"""

import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from .utils import setup_environment
from .vec_adapter import VecAdapter


def load_model_and_normalization(model_path, env, vecnormalize_path=None, auto_detect_vecnorm=False):
    """Load the model and handle VecNormalize if present."""
    if not model_path.endswith('.zip'):
        model_path += '.zip'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Check for VecNormalize statistics
    vecnormalize_file = None
    if vecnormalize_path:
        vecnormalize_file = vecnormalize_path
    elif auto_detect_vecnorm:
        # Auto-detect VecNormalize file
        base_path = model_path[:-4] if model_path.endswith('.zip') else model_path
        potential_vecnorm_path = f"{base_path}_vecnormalize.pkl"
        if os.path.exists(potential_vecnorm_path):
            vecnormalize_file = potential_vecnorm_path
        else:
            print(f"Auto-detect VecNormalize: no stats found next to model at {potential_vecnorm_path}")
    
    # Apply VecNormalize if statistics file exists
    if vecnormalize_file and os.path.exists(vecnormalize_file):
        print(f"Loading VecNormalize statistics from: {vecnormalize_file}")
        env = VecNormalize.load(vecnormalize_file, env)
        env.training = False
        env.norm_reward = False  # Don't normalize rewards during evaluation
        print("VecNormalize loaded and configured for evaluation.")
    else:
        if vecnormalize_path:
            print(f"Warning: Specified VecNormalize file not found: {vecnormalize_path}")
        else:
            print("No VecNormalize statistics found. Evaluating without normalization.")
    
    # Apply VecMonitor for episode statistics
    env = VecMonitor(env)
    
    # Load the PPO model
    model = PPO.load(model_path, env=env)
    print("Model loaded successfully.")
    
    return model, env


def detailed_evaluation(model, env, n_eval_episodes=20, deterministic=True, verbose=False):
    """Perform detailed evaluation with additional metrics."""
    print(f"\nStarting detailed evaluation")
    print(f"Number of episodes: {n_eval_episodes}")
    print(f"Deterministic actions: {deterministic}")
    print("=" * 50)
    
    # Standard evaluation
    start_time = time.time()
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        return_episode_rewards=False
    )
    eval_time = time.time() - start_time
    
    # Detailed evaluation with episode-by-episode results
    print("Running detailed evaluation...")
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_length += 1
            
            if done[0] if isinstance(done, np.ndarray) else done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if verbose:
            print(f"Episode {episode + 1:2d}: Reward = {episode_reward:8.2f}, Length = {episode_length:4d}")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_time': eval_time
    }


def print_evaluation_results(results, env_name, model_path, n_eval_episodes):
    """Print comprehensive evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"Environment: {env_name}")
    print(f"Model: {model_path}")
    print(f"Episodes evaluated: {n_eval_episodes}")
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


def save_evaluation_results(results, model_path, env_name, n_eval_episodes):
    """Save evaluation results to file."""
    results_file = f"{model_path}_evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results for {model_path}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Episodes: {n_eval_episodes}\n")
        f.write(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
        f.write(f"Episode rewards: {results['episode_rewards']}\n")
        f.write(f"Episode lengths: {results['episode_lengths']}\n")
    
    print(f"\nResults saved to: {results_file}")
    return results_file
