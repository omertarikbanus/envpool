"""
Common utilities for quadruped training and evaluation with EnvPool and PPO.
"""

from .asymmetric_policy import DEFAULT_ACTOR_OBSERVATION_DIM, AsymmetricActorCriticPolicy
from .vec_adapter import VecAdapter
from .utils import (
    setup_environment,
    create_policy_kwargs,
    create_ppo_model,
    setup_logging,
    ask_continue_or_restart,
    create_or_load_model,
    warm_start_environment,
    save_model_and_stats,
    setup_vecnormalize,
    find_vecnormalize_wrapper
)
from .evaluation import (
    load_model_and_normalization,
    detailed_evaluation,
    print_evaluation_results,
    save_evaluation_results
)

__all__ = [
    'DEFAULT_ACTOR_OBSERVATION_DIM',
    'AsymmetricActorCriticPolicy',
    'VecAdapter',
    'setup_environment',
    'create_policy_kwargs',
    'create_ppo_model',
    'setup_logging',
    'ask_continue_or_restart',
    'create_or_load_model',
    'warm_start_environment',
    'save_model_and_stats',
    'setup_vecnormalize',
    'find_vecnormalize_wrapper',
    'load_model_and_normalization',
    'detailed_evaluation',
    'print_evaluation_results',
    'save_evaluation_results'
]
