# Quadruped PPO Training and Evaluation

This directory contains a refactored and improved system for training and evaluating PPO (Proximal Policy Optimization) models for quadruped control using EnvPool.

## 📁 Directory Structure

```
train/
├── common/                          # Shared modules and utilities
│   ├── __init__.py                 # Package initialization with exports
│   ├── vec_adapter.py              # VecAdapter for EnvPool-SB3 compatibility
│   ├── utils.py                    # Common utilities for training/evaluation
│   └── evaluation.py               # Evaluation-specific functions
├── train.py                        # Training script (refactored)
├── eval.py                         # Evaluation script (refactored)
├── train_eval_notebook.ipynb       # Interactive Jupyter notebook
└── README.md                       # This file
```

## 🚀 Features

### ✅ Modular Design
- **Common modules**: Shared code moved to `common/` directory
- **VecAdapter**: Reusable environment wrapper for EnvPool-SB3 compatibility
- **Utility functions**: Environment setup, model creation, logging, etc.
- **Evaluation tools**: Comprehensive evaluation and visualization functions

### ✅ Multiple Interfaces
- **Command-line scripts**: `train.py` and `eval.py` for automated workflows
- **Jupyter notebook**: `train_eval_notebook.ipynb` for interactive experimentation
- **Consistent API**: All interfaces use the same underlying modules

### ✅ Enhanced Features
- **Model persistence**: Automatic save/load with VecNormalize statistics
- **Evaluation metrics**: Detailed performance analysis and visualization
- **Error handling**: Graceful interruption and recovery
- **Progress monitoring**: Real-time training progress with logging

## 📊 Usage

### Training Models

#### Command Line
```bash
# Basic training
python train.py --env-name Humanoid-v4 --num-envs 256

# Training with normalization
python train.py --env-name Ant-v4 --use-vecnormalize --total-timesteps 5000000

# Continue training from existing model
python train.py --continue-training --model-save-path ./models/my_model

# Force new training
python train.py --force-new --model-save-path ./models/my_model
```

#### Jupyter Notebook
Open `train_eval_notebook.ipynb` and follow the step-by-step workflow:
1. Configure parameters
2. Setup environment
3. Train model with progress monitoring
4. Evaluate and visualize results

### Evaluating Models

#### Command Line
```bash
# Basic evaluation
python eval.py --model-path ./models/my_model --env-name Humanoid-v4

# Detailed evaluation with rendering
python eval.py --model-path ./models/my_model --render --verbose --n-eval-episodes 50

# Auto-detect VecNormalize statistics
python eval.py --model-path ./models/my_model --auto-detect-vecnorm --deterministic
```

## 🏗️ Architecture

### VecAdapter
- Converts EnvPool environments to SB3-compatible format
- Handles action/observation space conversion to float32
- Manages environment resets and terminal observations
- Supports both legacy and modern gym interfaces

### Utility Functions
- `setup_environment()`: Create and configure environments
- `create_policy_kwargs()`: Define neural network architecture
- `create_or_load_model()`: Model creation with checkpoint support
- `setup_logging()`: Comprehensive logging configuration
- `save_model_and_stats()`: Save models with normalization data

### Evaluation Tools
- `detailed_evaluation()`: Episode-by-episode performance analysis
- `print_evaluation_results()`: Comprehensive result reporting
- `save_evaluation_results()`: Export results to file
- Visualization tools for training progress and performance metrics

## 🎯 Supported Environments

The system works with any MuJoCo environment available in EnvPool:
- `Ant-v4`: Quadruped ant locomotion
- `HalfCheetah-v4`: 2D cheetah running
- `Humanoid-v4`: Humanoid standing and walking
- `Hopper-v4`: Single-leg hopping
- `Walker2d-v4`: 2D bipedal walking
- And many more...

## 📈 Configuration Options

### Training Parameters
- `ENV_NAME`: Environment to train on
- `NUM_ENVS`: Number of parallel environments
- `TOTAL_TIMESTEPS`: Training duration
- `USE_VECNORMALIZE`: Enable observation/reward normalization
- `MODEL_SAVE_PATH`: Where to save trained models

### PPO Hyperparameters
- Learning rate: 5e-5
- Clip range: 0.1
- Network architecture: [64, 64, 64] for both policy and value functions
- Activation: Tanh
- And many more (see `create_ppo_model()` in `common/utils.py`)

### Evaluation Settings
- `N_EVAL_EPISODES`: Number of evaluation episodes
- `DETERMINISTIC_EVAL`: Use deterministic or stochastic policy
- Comprehensive metrics: rewards, episode lengths, success rates

## 🔧 Requirements

```bash
# Core dependencies
pip install torch stable-baselines3 envpool gym gymnasium
pip install numpy matplotlib seaborn jupyter
pip install packaging

# Optional for visualization
pip install tensorboard plotly
```

## 🐛 Troubleshooting

### Common Issues
1. **Import errors**: Ensure `common/` is in Python path
2. **CUDA issues**: Set `th.set_num_threads(1)` for CPU training
3. **EnvPool compatibility**: Check environment names and versions
4. **Memory issues**: Reduce `NUM_ENVS` if running out of memory

### Debug Mode
Enable verbose logging for detailed debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Examples

See the Jupyter notebook `train_eval_notebook.ipynb` for comprehensive examples with:
- Step-by-step training workflow
- Interactive parameter tuning
- Real-time progress monitoring
- Comprehensive result visualization
- Model comparison and analysis

## 🤝 Contributing

To extend the system:
1. Add new utility functions to `common/utils.py`
2. Extend evaluation metrics in `common/evaluation.py`
3. Update both scripts and notebook to use new features
4. Maintain consistent API across all interfaces

## 📄 License

This code follows the same license as the parent project.
