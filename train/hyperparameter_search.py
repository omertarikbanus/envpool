#!/usr/bin/env python3
"""
Hyperparameter search for PPO training using Optuna.

This script performs automated hyperparameter optimization for the quadrupedal controller
training using Optuna's Tree-structured Parzen Estimator (TPE) algorithm.
"""

import argparse
import logging
import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, Any
import json

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch as th

# Add the current directory to sys.path to import train module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import envpool
from train import VecAdapter, create_or_load_model
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Force PyTorch to use one thread for reproducibility
th.set_num_threads(1)


class HyperparameterSearch:
    """Hyperparameter search class using Optuna."""
    
    def __init__(self, args):
        self.args = args
        self.study_name = f"ppo_hyperopt_{args.env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study_dir = os.path.join("hyperopt_studies", self.study_name)
        os.makedirs(self.study_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.study_dir, 'hyperopt.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the trial."""
        
        # Learning rate - log uniform distribution
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        
        # Number of steps per environment per update
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
        
        # Batch size - should be <= n_steps * num_envs
        max_batch_size = min(4096, n_steps * self.args.num_envs)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024, 2048, 4096])
        if batch_size > max_batch_size:
            batch_size = max_batch_size
        
        # Number of epochs
        n_epochs = trial.suggest_int("n_epochs", 3, 20)
        
        # GAE lambda
        gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
        
        # Clip range
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
        
        # Max gradient norm
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 2.0)
        
        # Entropy coefficient
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
        
        # Value function coefficient
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
        
        # Network architecture
        net_arch_type = trial.suggest_categorical("net_arch_type", ["small", "medium", "large"])
        if net_arch_type == "small":
            net_arch = [dict(pi=[128], vf=[128])]
        elif net_arch_type == "medium":
            net_arch = [dict(pi=[256], vf=[256])]
        else:  # large
            net_arch = [dict(pi=[512, 256], vf=[512, 256])]
        
        # Log std initialization
        log_std_init = trial.suggest_float("log_std_init", -3.0, 1.0)
        
        # VecNormalize parameters
        use_vecnormalize = trial.suggest_categorical("use_vecnormalize", [True, False])
        clip_reward = trial.suggest_float("clip_reward", 1.0, 20.0) if use_vecnormalize else 10.0
        
        return {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "max_grad_norm": max_grad_norm,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "net_arch": net_arch,
            "log_std_init": log_std_init,
            "use_vecnormalize": use_vecnormalize,
            "clip_reward": clip_reward,
        }
    
    def create_env(self, use_vecnormalize: bool, clip_reward: float):
        """Create and wrap the environment."""
        # Create EnvPool environment
        env = envpool.make(
            self.args.env_name, 
            env_type="gym", 
            num_envs=self.args.num_envs, 
            seed=self.args.seed
        )
        
        # Set environment ID
        env.spec.id = self.args.env_name
        
        # Use the adapter
        env = VecAdapter(env)
        
        # Apply VecNormalize if requested
        if use_vecnormalize:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=clip_reward)
        
        # Add monitoring
        env = VecMonitor(env)
        
        return env
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        
        # Get hyperparameters for this trial
        hyperparams = self.suggest_hyperparameters(trial)
        
        self.logger.info(f"Trial {trial.number}: Testing hyperparameters: {hyperparams}")
        
        try:
            # Create environment
            env = self.create_env(
                use_vecnormalize=hyperparams["use_vecnormalize"],
                clip_reward=hyperparams["clip_reward"]
            )
            
            # Create policy kwargs
            policy_kwargs = dict(
                activation_fn=th.nn.ReLU,
                net_arch=hyperparams["net_arch"],
                log_std_init=hyperparams["log_std_init"]
            )
            
            # Create model
            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=hyperparams["learning_rate"],
                n_steps=hyperparams["n_steps"],
                batch_size=hyperparams["batch_size"],
                n_epochs=hyperparams["n_epochs"],
                gamma=0.99,  # Keep fixed
                gae_lambda=hyperparams["gae_lambda"],
                clip_range=hyperparams["clip_range"],
                max_grad_norm=hyperparams["max_grad_norm"],
                ent_coef=hyperparams["ent_coef"],
                vf_coef=hyperparams["vf_coef"],
                policy_kwargs=policy_kwargs,
                verbose=0,  # Reduce verbosity during search
            )
            
            # Setup evaluation callback for pruning
            eval_freq = max(self.args.eval_freq // self.args.num_envs, 1)
            eval_callback = EvalCallback(
                env,
                best_model_save_path=None,  # Don't save during search
                log_path=None,
                eval_freq=eval_freq,
                n_eval_episodes=self.args.n_eval_episodes,
                deterministic=True,
                verbose=0
            )
            
            # Train the model
            model.learn(
                total_timesteps=self.args.trial_timesteps,
                callback=eval_callback
            )
            
            # Evaluate the trained model
            if hyperparams["use_vecnormalize"]:
                # Turn off training mode for evaluation
                env.training = False
                env.norm_reward = False
            
            mean_reward, std_reward = evaluate_policy(
                model, 
                env, 
                n_eval_episodes=self.args.n_eval_episodes,
                deterministic=True
            )
            
            self.logger.info(f"Trial {trial.number}: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Clean up
            env.close()
            del model
            del env
            
            return mean_reward
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            return -np.inf
    
    def run_search(self):
        """Run the hyperparameter search."""
        
        # Create study
        sampler = TPESampler(seed=self.args.seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name
        )
        
        self.logger.info(f"Starting hyperparameter search with {self.args.n_trials} trials")
        self.logger.info(f"Study name: {self.study_name}")
        self.logger.info(f"Results will be saved to: {self.study_dir}")
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.args.n_trials)
        
        # Save results
        self.save_results(study)
        
        return study
    
    def save_results(self, study: optuna.Study):
        """Save the search results."""
        
        # Best trial info
        best_trial = study.best_trial
        self.logger.info(f"Best trial: {best_trial.number}")
        self.logger.info(f"Best reward: {best_trial.value:.4f}")
        self.logger.info(f"Best parameters: {best_trial.params}")
        
        # Save best parameters to JSON
        best_params_path = os.path.join(self.study_dir, "best_hyperparameters.json")
        with open(best_params_path, 'w') as f:
            json.dump({
                "trial_number": best_trial.number,
                "best_reward": best_trial.value,
                "best_parameters": best_trial.params,
                "search_config": {
                    "env_name": self.args.env_name,
                    "num_envs": self.args.num_envs,
                    "trial_timesteps": self.args.trial_timesteps,
                    "n_trials": self.args.n_trials,
                    "seed": self.args.seed
                }
            }, f, indent=2)
        
        self.logger.info(f"Best hyperparameters saved to: {best_params_path}")
        
        # Save study object
        study_path = os.path.join(self.study_dir, "study.pkl")
        optuna.pickle.dump_object(study, study_path)
        
        # Create training command with best parameters
        self.create_training_command(best_trial.params)
    
    def create_training_command(self, best_params: Dict[str, Any]):
        """Create a training command using the best parameters."""
        
        cmd_path = os.path.join(self.study_dir, "best_training_command.sh")
        
        # Convert net_arch back to string representation
        net_arch_type = best_params.get("net_arch_type", "medium")
        
        cmd = f"""#!/bin/bash
# Best hyperparameters found by Optuna search
# Trial reward: {self.study.best_value:.4f}

python3 train.py \\
    --env-name {self.args.env_name} \\
    --num-envs {self.args.num_envs} \\
    --seed {self.args.seed} \\
    --total-timesteps {self.args.full_training_timesteps} \\
    --model-save-path ./models/best_hyperopt_model"""
        
        if best_params.get("use_vecnormalize", False):
            cmd += " \\\n    --use-vecnormalize"
        
        cmd += f"""

# Best hyperparameters (modify train.py to use these):
# learning_rate: {best_params['learning_rate']}
# n_steps: {best_params['n_steps']}
# batch_size: {best_params['batch_size']}
# n_epochs: {best_params['n_epochs']}
# gae_lambda: {best_params['gae_lambda']}
# clip_range: {best_params['clip_range']}
# max_grad_norm: {best_params['max_grad_norm']}
# ent_coef: {best_params['ent_coef']}
# vf_coef: {best_params['vf_coef']}
# net_arch_type: {net_arch_type}
# log_std_init: {best_params['log_std_init']}
# clip_reward: {best_params.get('clip_reward', 10.0)}
"""
        
        with open(cmd_path, 'w') as f:
            f.write(cmd)
        
        os.chmod(cmd_path, 0o755)  # Make executable
        self.logger.info(f"Training command saved to: {cmd_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter search for PPO training.")
    
    # Environment settings
    parser.add_argument("--env-name", type=str, default="Humanoid-v4", 
                       help="EnvPool environment ID")
    parser.add_argument("--num-envs", type=int, default=16, 
                       help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    
    # Search settings
    parser.add_argument("--n-trials", type=int, default=100, 
                       help="Number of optimization trials")
    parser.add_argument("--trial-timesteps", type=int, default=1_000_000, 
                       help="Timesteps per trial (shorter for faster search)")
    parser.add_argument("--full-training-timesteps", type=int, default=10_000_000, 
                       help="Timesteps for final training with best params")
    
    # Evaluation settings
    parser.add_argument("--n-eval-episodes", type=int, default=10, 
                       help="Number of episodes for evaluation")
    parser.add_argument("--eval-freq", type=int, default=10000, 
                       help="Frequency of evaluation during training")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    
    # Create and run hyperparameter search
    search = HyperparameterSearch(args)
    study = search.run_search()
    
    print(f"\n{'='*50}")
    print("HYPERPARAMETER SEARCH COMPLETED")
    print(f"{'='*50}")
    print(f"Best reward: {study.best_value:.4f}")
    print(f"Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print(f"\nResults saved to: {search.study_dir}")
    print(f"Run the generated training script: {search.study_dir}/best_training_command.sh")


if __name__ == "__main__":
    main()
