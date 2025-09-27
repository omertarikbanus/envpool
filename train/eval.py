#!/usr/bin/env python3
"""
Evaluate trained PPO models with proper support for VecNormalize and model loading.

This script uses the refactored common modules for better code organization
and reusability between training and evaluation scripts.
"""

import argparse
import logging

# Import refactored common modules
from common import (
    setup_environment,
    load_model_and_normalization,
    detailed_evaluation,
    print_evaluation_results,
    save_evaluation_results
)





def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO models with EnvPool.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model (without .zip extension)")
    parser.add_argument("--env-name", type=str, default="Humanoid-v4", help="EnvPool environment ID")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel evaluation environments")
    parser.add_argument("--stack-frames", type=int, default=3, help="Observation frames to stack during evaluation")
    parser.add_argument("--n-eval-episodes", type=int, default=20, help="Number of episodes for evaluation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (no exploration)")
    parser.add_argument("--auto-detect-vecnorm", action="store_true", default=False, help="Automatically detect and load VecNormalize statistics")
    parser.add_argument("--vecnormalize-path", type=str, default=None, help="Explicit path to VecNormalize statistics file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def setup_evaluation_environment(args):
    """Set up the evaluation environment with proper wrappers."""
    return setup_environment(
        env_name=args.env_name,
        num_envs=args.num_envs,
        seed=args.seed,
        render_mode=True if args.render else None,
        stack_frames=args.stack_frames
    )











def main():
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
    main()
