"""Opt-in reward correction for fixed-command push recovery.

The plant and the 54-observation/25-action policy contract are unchanged.
The legacy C++ reward tracks the policy-adjusted vx command and rewards slowing
the gait to zero. Replace those two terms, retaining their scale and all other
terms. Apply this before VecMonitor and VecNormalize, only during training.
"""
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


def recovery_reward(reward, vx, adjusted_vx, phase_action, fall, target_vx):
    """Replace velocity target and remove the monotone slow-gait incentive."""
    reward = np.asarray(reward, dtype=np.float64)
    old_tracking = 0.25 * np.exp(-3.0 * np.abs(vx - adjusted_vx))
    new_tracking = 0.25 * np.exp(-3.0 * np.abs(vx - target_vx))
    old_phase = 0.05 * np.exp(-(np.clip(phase_action, -1, 1) + 1.0))
    # Constant equals the old term at the nominal gait action (zero).
    corrected = reward + new_tracking - old_tracking + 0.05 / np.e - old_phase
    # C++ replaces the entire locomotion reward on a physical termination.
    return np.where(fall, reward, corrected).astype(np.float32)


class RecoveryReward(VecEnvWrapper):
    def __init__(self, venv, target_vx=0.8):
        super().__init__(venv)
        if self.observation_space.shape != (54,) or self.action_space.shape != (25,):
            raise ValueError("RecoveryReward requires the Gamma 54-observation/25-action contract")
        if not np.isfinite(target_vx):
            raise ValueError("target_vx must be finite")
        self.target_vx = float(target_vx)
        self.actions = None

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.actions = np.asarray(actions).copy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # VecAdapter has already reset done environments: reward must use the
        # pre-reset command, including on timeouts (which bootstrap in PPO).
        adjusted_vx = np.array([
            info["terminal_observation"][0] if done else row[0]
            for row, done, info in zip(obs, dones, infos)
        ])
        vx = np.array([float(info["x_velocity"]) for info in infos])
        fall = np.array([bool(info["is_fall"]) for info in infos])
        corrected = recovery_reward(rewards, vx, adjusted_vx,
                                    self.actions[:, 24], fall, self.target_vx)
        return obs, corrected, dones, infos
