# Model Evaluation Tools

This directory contains comprehensive tools for evaluating trained PPO models with proper support for VecNormalize and detailed analysis.

## Overview

The evaluation suite includes:
- `eval.py` - Main evaluation script with full features
- `quick_eval.py` - Interactive quick evaluation tool
- `compare_models.py` - Compare multiple models side-by-side

## Features

### ✅ Automatic VecNormalize Detection
- Automatically detects and loads `_vecnormalize.pkl` files
- Properly configures VecNormalize for evaluation (training=False, norm_reward=False)
- Falls back gracefully if normalization files are missing

### ✅ Comprehensive Metrics
- Mean/std reward over multiple episodes
- Episode length statistics
- Success rate analysis
- Min/max/median values
- Detailed per-episode results

### ✅ Multiple Evaluation Modes
- Quick evaluation (20 episodes)
- Detailed evaluation (100+ episodes)
- Custom episode counts
- Deterministic vs stochastic actions
- Rendering support

## Usage

### 1. Basic Evaluation

```bash
# Evaluate a specific model
python eval.py --model-path ./quadruped_ppo_model

# With custom settings
python eval.py --model-path ./my_model --n-eval-episodes 100 --deterministic --verbose
```

### 2. Quick Interactive Evaluation

```bash
# Interactive menu with model selection
python quick_eval.py
```

**Features:**
- Automatically finds all `.zip` model files
- Shows VecNormalize status for each model
- Quick/detailed/custom evaluation options
- Easy model selection

### 3. Model Comparison

```bash
# Compare multiple models
python compare_models.py
```

**Features:**
- Evaluates multiple models with same settings
- Creates comparison plots
- Exports results to CSV
- Ranks models by performance

## Command Line Options (eval.py)

```bash
python eval.py --help
```

**Required:**
- `--model-path` - Path to model (without .zip extension)

**Optional:**
- `--env-name` - Environment (default: Humanoid-v4)
- `--n-eval-episodes` - Number of episodes (default: 20)
- `--num-envs` - Parallel environments (default: 1)
- `--seed` - Random seed (default: 0)
- `--deterministic` - Use deterministic actions
- `--render` - Render during evaluation
- `--verbose` - Detailed output
- `--vecnormalize-path` - Explicit VecNormalize file path
- `--auto-detect-vecnorm` - Auto-detect VecNormalize (default: True)

## Examples

### Example 1: Basic Evaluation
```bash
python eval.py --model-path ./quadruped_ppo_model
```

### Example 2: Detailed Evaluation
```bash
python eval.py --model-path ./quadruped_ppo_model \
    --n-eval-episodes 100 \
    --deterministic \
    --verbose
```

### Example 3: Evaluate Critic-Help Model
```bash
python eval.py --model-path ./quadruped_ppo_critic_help \
    --n-eval-episodes 50 \
    --deterministic
```

### Example 4: Compare Training Methods
```bash
# First, run comparison tool
python compare_models.py

# Select models to compare:
# 1. quadruped_ppo_model (standard)
# 2. quadruped_ppo_critic_help (critic-help)
# 3. quadruped_ppo_seed_0 (seed robustness)
```

## Output Files

### Evaluation Results
Each evaluation creates:
- `{model_path}_evaluation_results.txt` - Detailed text results

### Comparison Results
Model comparison creates:
- `model_comparison_results.csv` - CSV with all results
- `model_comparison.png` - Performance comparison plot

## Sample Output

```
==============================================================
EVALUATION RESULTS
==============================================================
Environment: Humanoid-v4
Model: ./quadruped_ppo_model
Episodes evaluated: 20
Evaluation time: 45.23 seconds

REWARD STATISTICS:
  Mean reward:  1234.56 ±  123.45
  Min reward:   1000.12
  Max reward:   1456.78
  Median:       1245.33

EPISODE LENGTH STATISTICS:
  Mean length:   987.5 ±   45.2
  Min length:     800
  Max length:    1000
  Median:        995.0

SUCCESS RATE (reward > 0): 100.0% (20/20)
==============================================================
```

## Troubleshooting

### VecNormalize Issues
```bash
# If VecNormalize file is missing or corrupted
python eval.py --model-path ./my_model --auto-detect-vecnorm False

# Use explicit VecNormalize path
python eval.py --model-path ./my_model --vecnormalize-path ./custom_vecnorm.pkl
```

### Environment Issues
```bash
# Different environment
python eval.py --model-path ./my_model --env-name "Ant-v4"

# Rendering issues (try without render)
python eval.py --model-path ./my_model --render False
```

### Performance Issues
```bash
# Reduce episodes for faster evaluation
python eval.py --model-path ./my_model --n-eval-episodes 5

# Use single environment
python eval.py --model-path ./my_model --num-envs 1
```

## Integration with Training Scripts

The evaluation tools work seamlessly with models trained using:
- `../train/train.py` (main training script)
- `../scripts/train_critic_help_full.py` (critic-help training)
- Any PPO model with the correct environment

### File Structure Expected
```
your_workspace/
├── quadruped_ppo_model.zip              # PPO model
├── quadruped_ppo_model_vecnormalize.pkl # VecNormalize stats (if used)
├── eval.py                              # Main evaluation script
├── quick_eval.py                        # Interactive evaluation
└── compare_models.py                    # Model comparison
```

## Best Practices

1. **Use deterministic evaluation** for consistent comparisons
2. **Run sufficient episodes** (50-100) for reliable statistics
3. **Compare with same settings** across different models
4. **Save results** for later analysis and reporting
5. **Check VecNormalize status** when comparing models trained differently
