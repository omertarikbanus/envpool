# Hyperparameter Search for PPO Training

This directory contains tools for automated hyperparameter optimization of PPO training using Optuna.

## Files

- `hyperparameter_search.py` - Main hyperparameter search script
- `visualize_hyperopt.py` - Visualization tool for search results
- `requirements_hyperopt.txt` - Additional dependencies for hyperparameter search
- `train.py` - Original training script

## Installation

Install the additional dependencies:

```bash
pip install -r requirements_hyperopt.txt
```

## Usage

### 1. Run Hyperparameter Search

Basic usage:
```bash
python3 hyperparameter_search.py --env-name Humanoid-v4 --n-trials 50
```

Full options:
```bash
python3 hyperparameter_search.py \
    --env-name Humanoid-v4 \
    --num-envs 16 \
    --seed 42 \
    --n-trials 100 \
    --trial-timesteps 1000000 \
    --full-training-timesteps 10000000 \
    --n-eval-episodes 10 \
    --eval-freq 10000
```

### 2. Monitor Progress

The search will create a directory under `hyperopt_studies/` with:
- `hyperopt.log` - Detailed logs
- `best_hyperparameters.json` - Best parameters found
- `study.pkl` - Complete Optuna study object
- `best_training_command.sh` - Ready-to-run training command

### 3. Visualize Results

After the search completes:
```bash
python3 visualize_hyperopt.py --study-dir hyperopt_studies/ppo_hyperopt_Humanoid-v4_20250713_123456
```

This creates interactive visualizations in the `visualizations/` subdirectory:
- Optimization history
- Parameter importance
- Parallel coordinate plot
- Hyperparameter distributions
- Summary HTML report

### 4. Train with Best Parameters

The search automatically generates a training script with the best parameters:
```bash
./hyperopt_studies/ppo_hyperopt_Humanoid-v4_20250713_123456/best_training_command.sh
```

Or modify your `train.py` with the parameters from `best_hyperparameters.json`.

## Hyperparameters Optimized

The search optimizes the following PPO hyperparameters:

### Core PPO Parameters
- `learning_rate` (1e-5 to 1e-3, log scale)
- `n_steps` ([1024, 2048, 4096, 8192])
- `batch_size` ([64, 128, 256, 512, 1024, 2048, 4096])
- `n_epochs` (3 to 20)
- `gae_lambda` (0.8 to 1.0)
- `clip_range` (0.1 to 0.4)
- `max_grad_norm` (0.1 to 2.0)
- `ent_coef` (0.0 to 0.01)
- `vf_coef` (0.1 to 1.0)

### Network Architecture
- `net_arch_type` (small/medium/large)
  - Small: [128] hidden units
  - Medium: [256] hidden units  
  - Large: [512, 256] hidden units
- `log_std_init` (-3.0 to 1.0)

### Environment Wrapping
- `use_vecnormalize` (True/False)
- `clip_reward` (1.0 to 20.0, if using VecNormalize)

## Search Strategy

- **Sampler**: Tree-structured Parzen Estimator (TPE)
- **Pruner**: Median pruner (stops unpromising trials early)
- **Objective**: Mean reward over evaluation episodes
- **Trial budget**: Configurable timesteps per trial (default: 1M)

## Tips

1. **Start small**: Use fewer trials (20-50) for initial exploration
2. **Short trials**: Use 1M timesteps per trial for faster search
3. **Scale up**: Once you find promising regions, run longer trials
4. **Multiple searches**: Run searches with different seeds for robustness
5. **Domain knowledge**: Adjust search spaces based on your environment

## Example Workflow

```bash
# 1. Quick exploration (30 minutes)
python3 hyperparameter_search.py --n-trials 20 --trial-timesteps 500000

# 2. Visualize results
python3 visualize_hyperopt.py --study-dir hyperopt_studies/ppo_hyperopt_*

# 3. Refined search around best region (2 hours)
python3 hyperparameter_search.py --n-trials 50 --trial-timesteps 1000000

# 4. Final training with best parameters (overnight)
./hyperopt_studies/ppo_hyperopt_*/best_training_command.sh
```

## Customization

To add/modify hyperparameters, edit the `suggest_hyperparameters()` method in `hyperparameter_search.py`.

To change the search space or add new environments, modify the search ranges and environment creation logic.
