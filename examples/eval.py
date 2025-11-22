#!/usr/bin/env python3
"""
Evaluate trained PPO models with proper support for VecNormalize and model loading.

This script uses the refactored common modules for better code organization
and reusability between training and evaluation scripts.
"""

import argparse
import logging
import os
import sys

# Import refactored common modules. Support running either as
# `python -m envpool.examples.eval` or `python envpool/examples/eval.py`.
try:
    from .common import (
        setup_environment,
        build_cmd_profile_config,
        load_model_and_normalization,
        detailed_evaluation,
        print_evaluation_results,
        save_evaluation_results,
    )
except ImportError:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    from common import (  # type: ignore  # noqa: F401
        setup_environment,
        build_cmd_profile_config,
        load_model_and_normalization,
        detailed_evaluation,
        print_evaluation_results,
        save_evaluation_results,
    )





def parse_args():
    print("Parsing command-line arguments for evaluation...")
    parser = argparse.ArgumentParser(description="Evaluate trained PPO models with EnvPool.")
    parser.add_argument("--model-path", type=str, default="data/current/quadruped_ppo_model.zip", help="Path to the saved model (without .zip extension)")
    parser.add_argument("--env-name", type=str, default="Humanoid-v4", help="EnvPool environment ID")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel evaluation environments")
    parser.add_argument("--stack-frames", type=int, default=3, help="Observation frames to stack during evaluation")
    parser.add_argument("--n-eval-episodes", type=int, default=3, help="Number of episodes for evaluation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (no exploration, default behaviour)"
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample actions from the policy distribution during evaluation"
    )
    parser.add_argument("--auto-detect-vecnorm", action="store_true", default=True, help="Automatically detect and load VecNormalize statistics based on model path")
    parser.add_argument("--no-auto-detect-vecnorm", dest="auto_detect_vecnorm", action="store_false", help="Disable automatic VecNormalize detection")
    parser.add_argument("--vecnormalize-path", type=str, default=None, help="Explicit path to VecNormalize statistics file (overrides auto-detection)")
    parser.add_argument("--cmd-profile", type=str, default="fixed", choices=["random_episode", "fixed"], help="Command sampling strategy")
    parser.add_argument("--cmd-fixed-vx", type=float, default=3, help="Fixed command in body X (m/s)")
    parser.add_argument("--cmd-fixed-vy", type=float, default=0, help="Fixed command in body Y (m/s)")
    parser.add_argument("--cmd-fixed-yaw", type=float, default=0, help="Fixed yaw-rate command (rad/s)")
    parser.add_argument("--cmd-rand-vx-min", type=float, default=None, help="Minimum random body X command (m/s)")
    parser.add_argument("--cmd-rand-vx-max", type=float, default=None, help="Maximum random body X command (m/s)")
    parser.add_argument("--cmd-rand-vy-min", type=float, default=None, help="Minimum random body Y command (m/s)")
    parser.add_argument("--cmd-rand-vy-max", type=float, default=None, help="Maximum random body Y command (m/s)")
    parser.add_argument("--cmd-rand-yaw-min", type=float, default=None, help="Minimum random yaw-rate command (rad/s)")
    parser.add_argument("--cmd-rand-yaw-max", type=float, default=None, help="Maximum random yaw-rate command (rad/s)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def setup_evaluation_environment(args):
    """Set up the evaluation environment with proper wrappers."""
    return setup_environment(
        env_name=args.env_name,
        num_envs=args.num_envs,
        seed=args.seed,
        render_mode=True if args.render else None,
        stack_frames=args.stack_frames,
        env_config=build_cmd_profile_config(args)
    )

def main():
    print("Starting PPO Model Evaluation...")
    args = parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    print("PPO Model Evaluation Script")
    print("=" * 30)
    
    try:
        # Set up environment
        env = setup_evaluation_environment(args)
        
        # Load model and normalization using our utility function
        model, env = load_model_and_normalization(
            model_path=args.model_path,
            env=env,
            vecnormalize_path=args.vecnormalize_path,
            auto_detect_vecnorm=args.auto_detect_vecnorm
        )
        
        # Perform evaluation using our utility function
        results = detailed_evaluation(
            model=model,
            env=env,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=args.deterministic,
            verbose=args.verbose
        )
        
        # Print results using our utility function
        print_evaluation_results(
            results=results,
            env_name=args.env_name,
            model_path=args.model_path,
            n_eval_episodes=args.n_eval_episodes
        )
        
        # Save results using our utility function
        results_file = save_evaluation_results(
            results=results,
            model_path=args.model_path,
            env_name=args.env_name,
            n_eval_episodes=args.n_eval_episodes
        )
        
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    finally:
        # Clean up
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    print("Running evaluation script...")
    main()
