#!/usr/bin/env python3
"""
Quick evaluation script with common presets for evaluating trained models.
"""
import subprocess
import sys
import os
import glob

def find_models(directory="."):
    """Find all .zip model files in the directory."""
    pattern = os.path.join(directory, "*.zip")
    models = glob.glob(pattern)
    return [model.replace('.zip', '') for model in models]

def main():
    print("Quick Model Evaluation Tool")
    print("=" * 40)
    
    # Find available models
    models = find_models(".")
    if not models:
        print("No model files (*.zip) found in current directory.")
        return
    
    print(f"Found {len(models)} model(s):")
    for i, model in enumerate(models, 1):
        vecnorm_exists = os.path.exists(f"{model}_vecnormalize.pkl")
        vecnorm_status = "✓ VecNorm" if vecnorm_exists else "✗ No VecNorm"
        print(f"  {i}. {os.path.basename(model)} [{vecnorm_status}]")
    
    print()
    print("Evaluation options:")
    print("1. Quick evaluation (20 episodes)")
    print("2. Detailed evaluation (100 episodes)")
    print("3. Custom evaluation")
    print("4. Evaluate specific model")
    print()
    
    choice = input("Select option (1-4, or 'q' to quit): ").strip()
    
    if choice == 'q':
        return
    
    # Model selection
    if choice == '4':
        model_path = input("Enter model path (without .zip): ").strip()
        if not os.path.exists(f"{model_path}.zip"):
            print(f"Model not found: {model_path}.zip")
            return
    else:
        if len(models) == 1:
            model_path = models[0]
            print(f"Using model: {os.path.basename(model_path)}")
        else:
            print("Select a model:")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {os.path.basename(model)}")
            
            try:
                model_idx = int(input("Enter model number: ")) - 1
                if 0 <= model_idx < len(models):
                    model_path = models[model_idx]
                else:
                    print("Invalid model number.")
                    return
            except ValueError:
                print("Invalid input.")
                return
    
    # Build evaluation command
    eval_script = os.path.join(os.path.dirname(__file__), "eval.py")
    cmd = [sys.executable, eval_script, "--model-path", model_path]
    
    if choice == '1':  # Quick evaluation
        cmd.extend(["--n-eval-episodes", "20"])
    elif choice == '2':  # Detailed evaluation
        cmd.extend(["--n-eval-episodes", "100", "--verbose"])
    elif choice == '3':  # Custom evaluation
        print("\nCustom evaluation options:")
        episodes = input("Number of episodes (default: 20): ").strip() or "20"
        cmd.extend(["--n-eval-episodes", episodes])
        
        if input("Use deterministic actions? (y/n): ").strip().lower() == 'y':
            cmd.append("--deterministic")
        
        if input("Render environment? (y/n): ").strip().lower() == 'y':
            cmd.append("--render")
        
        if input("Verbose output? (y/n): ").strip().lower() == 'y':
            cmd.append("--verbose")
    
    print(f"\nRunning evaluation...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 40)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")

if __name__ == "__main__":
    main()
