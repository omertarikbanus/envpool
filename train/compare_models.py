#!/usr/bin/env python3
"""
Compare evaluation results across multiple trained models.
"""
import subprocess
import sys
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

def find_models(directory="."):
    """Find all .zip model files in the directory."""
    pattern = os.path.join(directory, "*.zip")
    models = glob.glob(pattern)
    return [model.replace('.zip', '') for model in models]

def run_evaluation(model_path, n_episodes=50):
    """Run evaluation for a single model and return results."""
    eval_script = os.path.join(os.path.dirname(__file__), "eval.py")
    cmd = [
        sys.executable, eval_script,
        "--model-path", model_path,
        "--n-eval-episodes", str(n_episodes),
        "--deterministic"  # Use deterministic for fair comparison
    ]
    
    # Check if VecNormalize file exists and add it to the command
    vecnorm_path = f"{model_path}_vecnormalize.pkl"
    if os.path.exists(vecnorm_path):
        cmd.extend(["--vecnormalize-path", vecnorm_path])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse the results file that was created
        results_file = f"{model_path}_evaluation_results.txt"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                content = f.read()
                # Extract mean reward (simple parsing)
                for line in content.split('\n'):
                    if line.startswith('Mean reward:'):
                        parts = line.split(':')[1].strip().split('±')
                        mean_reward = float(parts[0].strip())
                        std_reward = float(parts[1].strip()) if len(parts) > 1 else 0
                        return mean_reward, std_reward
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed for {model_path}: {e}")
        return None, None

def create_comparison_plot(results, output_file="model_comparison.png"):
    """Create a comparison plot of model performance."""
    if not results:
        print("No results to plot.")
        return
    
    models = list(results.keys())
    means = [results[model]['mean'] for model in models]
    stds = [results[model]['std'] for model in models]
    
    plt.figure(figsize=(12, 6))
    x_pos = range(len(models))
    
    bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Mean Reward')
    plt.title('Model Performance Comparison')
    plt.xticks(x_pos, [os.path.basename(model) for model in models], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + max(means) * 0.01, f'{mean:.1f}±{std:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved as: {output_file}")

def main():
    print("Model Comparison Tool")
    print("=" * 30)
    
    # Find available models
    models = find_models(".")
    if len(models) < 2:
        print("Need at least 2 models for comparison.")
        print(f"Found: {len(models)} model(s)")
        for model in models:
            print(f"  - {os.path.basename(model)}")
        return
    
    print(f"Found {len(models)} model(s) for comparison:")
    for i, model in enumerate(models, 1):
        vecnorm_exists = os.path.exists(f"{model}_vecnormalize.pkl")
        vecnorm_status = "VecNorm" if vecnorm_exists else "No VecNorm"
        print(f"  {i}. {os.path.basename(model)} [{vecnorm_status}]")
    
    print()
    choice = input("Compare (a)ll models or (s)elect specific ones? [a/s]: ").lower().strip()
    
    selected_models = []
    if choice == 'a':
        selected_models = models
    elif choice == 's':
        print("Select models to compare (enter numbers separated by spaces):")
        try:
            indices = input("Model numbers: ").split()
            selected_models = [models[int(i)-1] for i in indices if 1 <= int(i) <= len(models)]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return
    else:
        print("Invalid choice.")
        return
    
    if len(selected_models) < 2:
        print("Need at least 2 models for comparison.")
        return
    
    # Get evaluation parameters
    try:
        n_episodes = int(input(f"Number of episodes per model (default: 50): ") or "50")
    except ValueError:
        n_episodes = 50
    
    print(f"\nEvaluating {len(selected_models)} models with {n_episodes} episodes each...")
    print("This may take a while...")
    print("=" * 50)
    
    results = {}
    for i, model in enumerate(selected_models, 1):
        vecnorm_exists = os.path.exists(f"{model}_vecnormalize.pkl")
        vecnorm_status = " (with VecNormalize)" if vecnorm_exists else " (no VecNormalize)"
        print(f"Evaluating {i}/{len(selected_models)}: {os.path.basename(model)}{vecnorm_status}")
        mean_reward, std_reward = run_evaluation(model, n_episodes)
        
        if mean_reward is not None:
            results[model] = {'mean': mean_reward, 'std': std_reward}
            print(f"  Result: {mean_reward:.2f} ± {std_reward:.2f}")
        else:
            print(f"  Failed to evaluate {model}")
        print()
    
    if not results:
        print("No successful evaluations.")
        return
    
    # Print comparison table
    print("COMPARISON RESULTS")
    print("=" * 75)
    print(f"{'Model':<25} {'Mean Reward':<15} {'Std Reward':<15} {'VecNorm':<10}")
    print("-" * 75)
    
    # Sort by mean reward (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    for model, result in sorted_results:
        model_name = os.path.basename(model)[:23]  # Truncate long names
        vecnorm_exists = os.path.exists(f"{model}_vecnormalize.pkl")
        vecnorm_status = "Yes" if vecnorm_exists else "No"
        print(f"{model_name:<25} {result['mean']:>10.2f}     {result['std']:>10.2f}     {vecnorm_status:<10}")
    
    print("=" * 75)
    
    # Best model
    best_model, best_result = sorted_results[0]
    print(f"Best performing model: {os.path.basename(best_model)}")
    print(f"Performance: {best_result['mean']:.2f} ± {best_result['std']:.2f}")
    
    # Create plot
    create_plot = input("\nCreate comparison plot? (y/n): ").lower().strip() == 'y'
    if create_plot:
        create_comparison_plot(results)
    
    # Save results
    save_results = input("Save results to CSV? (y/n): ").lower().strip() == 'y'
    if save_results:
        df_data = []
        for model, result in results.items():
            vecnorm_exists = os.path.exists(f"{model}_vecnormalize.pkl")
            df_data.append({
                'model': os.path.basename(model),
                'model_path': model,
                'mean_reward': result['mean'],
                'std_reward': result['std'],
                'episodes': n_episodes,
                'uses_vecnormalize': vecnorm_exists
            })
        
        df = pd.DataFrame(df_data)
        output_file = "model_comparison_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
