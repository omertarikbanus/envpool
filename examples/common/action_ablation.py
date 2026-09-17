#!/usr/bin/env python3
"""
Training-time action-channel ablation for the QuadrupedWBC-v0 25-dim action.

The existing ablation study (quadcontrol/docs/ABLATION_RESULTS.md) only masks
a channel to its neutral value AFTER a normally-trained policy is loaded, at
evaluation time (quadcontrol/evaluations/core/_rl_eval.py::apply_action_ablation).
That measures what happens when a channel the policy learned to rely on is
suddenly taken away -- a distribution-shift question, not "does the policy
need this channel at all". A proper ablation must never let the policy see a
non-neutral value for the masked channel during training either, so the
network never has gradient signal to use it in the first place.

This wrapper applies the identical index mapping and neutral values as
_rl_eval.py::apply_action_ablation, but in step_async, before the action
reaches the simulator, on every training step. Keep the two in sync: if the
index mapping changes in one place it must change in the other.
"""

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

# Mirrors quadcontrol/evaluations/core/_rl_eval.py::ACTION_ABLATIONS, minus the
# two multi-channel/non-neutral modes ("none", "zero_action") that do not
# correspond to a single trainable channel.
TRAINING_ACTION_ABLATIONS = (
    "none", "zero_grf", "no_velocity", "no_footsteps", "fixed_height",
    "fixed_gait",
)


def apply_action_ablation(action, name):
    """Zero one channel group of a 25-dim QuadrupedWBC-v0 action in place.

    Must stay byte-identical to quadcontrol's core/_rl_eval.py::apply_action_ablation
    for the five shared modes, so a training-time ablation and the existing
    eval-time ablation mean the same thing.
    """
    if name == "none":
        return action
    out = np.asarray(action)
    if name == "zero_grf":
        # Decision 2026-09-14 (ABLATION_PLAN.md Sec.6#1): action 0, not zero
        # Newtons -- Fx/Fy = 0 N, Fz = 125 N/leg (midpoint of [0, force_z_max]).
        out[..., 3:15] = 0.0
    elif name == "no_velocity":
        out[..., 0:3] = 0.0
    elif name == "no_footsteps":
        out[..., 15:23] = 0.0
    elif name == "fixed_height":
        out[..., 23] = 0.0
    elif name == "fixed_gait":
        out[..., 24] = 0.0
    else:
        raise ValueError(f"unknown training action ablation: {name}")
    return out


class ActionAblationWrapper(VecEnvWrapper):
    """Forces one action channel group to its neutral value on every step.

    Wraps directly around VecAdapter (before VecMonitor/VecNormalize), so the
    masked action is what both the simulator and the rollout buffer see --
    the policy's raw output for the masked indices contributes to the PPO
    loss like any other output, but the environment never responds to it,
    so there is no gradient pressure to use the channel.
    """

    def __init__(self, venv, mode):
        if mode not in TRAINING_ACTION_ABLATIONS:
            raise ValueError(f"unknown training action ablation: {mode}")
        super().__init__(venv)
        self.mode = mode

    def step_async(self, actions):
        masked = apply_action_ablation(np.array(actions, copy=True), self.mode)
        self.venv.step_async(masked)

    def step_wait(self):
        return self.venv.step_wait()

    def reset(self):
        return self.venv.reset()
