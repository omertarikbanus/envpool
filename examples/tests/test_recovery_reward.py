"""Reward semantics, including terminal-state handling, without a simulator."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.recovery_reward import recovery_reward, RecoveryReward


def legacy(vx, cmd, action):
    return 0.7 + 0.25 * np.exp(-3 * np.abs(vx - cmd)) + 0.05 * np.exp(-(action + 1))


def test_policy_cannot_improve_tracking_reward_by_changing_its_target():
    vx = np.array([0.3, 0.3])
    cmd = np.array([0.3, 0.8])
    phase = np.zeros(2)
    result = recovery_reward(legacy(vx, cmd, phase), vx, cmd, phase,
                             np.zeros(2, dtype=bool), 0.8)
    np.testing.assert_allclose(result[0], result[1])
    assert result[0] < legacy(0.8, 0.8, 0)


def test_gait_slowing_has_no_direct_reward_and_fall_penalty_is_unchanged():
    phase = np.array([-1., 0., 1.])
    vx = cmd = np.full(3, 0.8)
    result = recovery_reward(legacy(vx, cmd, phase), vx, cmd, phase,
                             np.zeros(3, dtype=bool), 0.8)
    np.testing.assert_allclose(result, result[0])
    np.testing.assert_array_equal(recovery_reward(np.full(3, -10), vx, cmd,
                                                 phase, np.ones(3, dtype=bool), 0.8), -10)


def test_timeout_reward_uses_terminal_command():
    class FakeEnv:
        def step_wait(self):
            obs = np.zeros((1, 54), dtype=np.float32)
            obs[0, 0] = 1.3  # Reset state's command differs from terminal state.
            terminal = np.zeros(54)
            terminal[0] = 0.8
            info = {"terminal_observation": terminal, "x_velocity": 0.8, "is_fall": False}
            return obs, np.array([legacy(0.8, 0.8, 0)]), np.array([True]), [info]
    wrapper = object.__new__(RecoveryReward)
    wrapper.venv = FakeEnv()
    wrapper.actions = np.zeros((1, 25))
    wrapper.target_vx = 0.8
    obs, reward, done, _ = wrapper.step_wait()
    np.testing.assert_allclose(reward[0], legacy(0.8, 0.8, 0))
    assert done[0] and obs[0, 0] == np.float32(1.3)
