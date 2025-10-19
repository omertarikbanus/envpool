#!/usr/bin/env python3
"""
VecAdapter for converting EnvPool to Stable-Baselines3-compatible VecEnv.
"""

import numpy as np
from packaging import version
import gym
import gymnasium
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnvWrapper
from envpool.python.protocol import EnvPool

# Check gym version for compatibility
is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")


class VecAdapter(VecEnvWrapper):
    """
    Convert an EnvPool object to a Stable-Baselines3-compatible VecEnv.
    This adapter sets the number of environments from the EnvPool spec and
    implements step_wait to handle terminal resets, attaching terminal observations.
    Also converts spaces to be SB3-compatible.
    """
    def __init__(self, venv: EnvPool):
        # Set the number of environments from EnvPool's config.
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv)
        
        # Convert the action space to Gymnasium's Box with float32 (SB3 requires this)
        self.action_space = Box(
            low=venv.action_space.low.astype(np.float32),
            high=venv.action_space.high.astype(np.float32),
            shape=venv.action_space.shape,
            dtype=np.float32,
        )
        
        # Convert the observation space to float32 as well
        # First, check if it's a Box space
        if isinstance(venv.observation_space, (gym.spaces.Box, gymnasium.spaces.Box)):
            self.observation_space = Box(
                low=venv.observation_space.low.astype(np.float32),
                high=venv.observation_space.high.astype(np.float32),
                shape=venv.observation_space.shape,
                dtype=np.float32,
            )
        else:
            # If not a Box space, keep the original (but this might cause issues)
            self.observation_space = venv.observation_space
    
    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
    
    def reset(self):
        if is_legacy_gym:
            obs = self.venv.reset()
        else:
            obs = self.venv.reset()[0]
        # Convert observations to numpy array (if not already) and ensure float32
        obs = np.asarray(obs, dtype=np.float32)
        return obs
    
    def seed(self, seed: int = None) -> None:
        # Seeding is set at EnvPool creation.
        pass
    
    def step_wait(self):
        if is_legacy_gym:
            obs, rewards, dones, info_dict = self.venv.step(self.actions)
        else:
            obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
            dones = terms + truncs
        
        # Ensure observations are float32
        obs = np.asarray(obs, dtype=np.float32)
        
        infos = []
        for i in range(self.num_envs):
            info_i = {key: info_dict[key][i] for key in info_dict.keys() if isinstance(info_dict[key], np.ndarray)}
            if dones[i]:
                info_i["terminal_observation"] = obs[i]
                if is_legacy_gym:
                    reset_obs = self.venv.reset(np.array([i]))
                else:
                    reset_obs = self.venv.reset(np.array([i]))[0]
                obs[i] = np.asarray(reset_obs, dtype=np.float32)
            infos.append(info_i)
        return obs, rewards, dones, infos
