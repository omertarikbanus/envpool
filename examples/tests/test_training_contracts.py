"""Small behavioral tests for the EnvPool/SB3 training boundary."""
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from common import vec_adapter
from common.utils import create_or_load_model, save_model_and_stats, find_vecnormalize_wrapper
from train import LogStdClampCallback


class ToyEnv(gym.Env):
    observation_space = gym.spaces.Box(-100, 100, (2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.count += 1
        return np.ones(2, dtype=np.float32), 3.0, False, self.count == 2, {}


class TrainingContracts(unittest.TestCase):
    def test_timeout_and_terminal_observation_survive_reset(self):
        class Pool:
            def step(self, actions):
                return (np.array([[10., 20.], [11., 21.]], dtype=np.float32),
                        np.ones(2), np.array([False, True]), np.array([True, False]), {})

            def reset(self, ids):
                return np.array([[30., 40.]], dtype=np.float32), {}

        adapter = types.SimpleNamespace(venv=Pool(), num_envs=2, actions=np.zeros((2, 1)))
        old = vec_adapter.is_legacy_gym
        vec_adapter.is_legacy_gym = False
        try:
            obs, _, done, info = vec_adapter.VecAdapter.step_wait(adapter)
        finally:
            vec_adapter.is_legacy_gym = old
        np.testing.assert_array_equal(done, [True, True])
        np.testing.assert_array_equal(info[0]["terminal_observation"], [10, 20])
        np.testing.assert_array_equal(info[1]["terminal_observation"], [11, 21])
        self.assertTrue(info[0]["TimeLimit.truncated"])
        self.assertFalse(info[1]["TimeLimit.truncated"])
        self.assertTrue(info[1]["is_fall"])
        obs[:] = 0
        np.testing.assert_array_equal(info[0]["terminal_observation"], [10, 20])

    def test_std_bound_holds_inside_updates_and_after_final_update(self):
        env = DummyVecEnv([ToyEnv])
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=4,
                    ent_coef=10, learning_rate=0.1, seed=7)
        callback = LogStdClampCallback(0.05, 0.06)
        observed = []
        # The callback's hook must bound std before the next evaluation.
        original = model.policy.evaluate_actions
        def evaluate(*args, **kwargs):
            observed.append(float(model.policy.log_std.exp().max()))
            return original(*args, **kwargs)
        model.policy.evaluate_actions = evaluate
        model.learn(32, callback=callback)
        self.assertTrue(observed)
        self.assertLessEqual(max(observed), 0.060001)
        self.assertLessEqual(float(model.policy.log_std.exp().max()), 0.060001)
        self.assertIsNone(callback._hook)
        env.close()

    def test_raw_monitor_and_normalization_survive_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model")
            env = VecNormalize(VecMonitor(DummyVecEnv([ToyEnv])))
            model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, seed=7)
            env.reset()
            env.step(np.zeros((1, 1)))
            _, _, _, infos = env.step(np.zeros((1, 1)))
            self.assertEqual(infos[0]["episode"]["r"], 6.0)
            save_model_and_stats(model, path, env)
            new_env = VecNormalize(VecMonitor(DummyVecEnv([ToyEnv])))
            loaded, new_env = create_or_load_model(path, new_env, {}, continue_training=True, seed=19)
            self.assertEqual(loaded.seed, 19)
            self.assertIsInstance(new_env, VecNormalize)
            self.assertIsInstance(new_env.venv, VecMonitor)
            self.assertNotIsInstance(new_env.venv.venv, VecNormalize)
            self.assertIs(find_vecnormalize_wrapper(new_env), new_env)
            new_env.reset()
            new_env.step(np.zeros((1, 1)))
            _, _, _, infos = new_env.step(np.zeros((1, 1)))
            self.assertEqual(infos[0]["episode"]["r"], 6.0)
            env.close()
            new_env.close()

    def test_invalid_std_bounds_fail_early(self):
        for lo, hi in [(0, .3), (.4, .3), (.05, float("inf"))]:
            with self.assertRaises(ValueError):
                LogStdClampCallback(lo, hi)


if __name__ == "__main__":
    unittest.main()
