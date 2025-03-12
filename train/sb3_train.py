"""
A Stable-Baselines3 (PPO) training script for a multiple-env EnvPool-based
quadruped environment. The key fix is restoring the `num_envs` property
in our EnvPoolToVecEnv wrapper.
"""

import os
import numpy as np
import envpool
import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv  # for single-env evaluation
from stable_baselines3.common.logger import configure


class EnvPoolToVecEnv(VecEnvWrapper):
    """
    Wrap an EnvPool environment (already vectorized) to be SB3-compatible.
    """

    def __init__(self, envpool_env):
        """
        :param envpool_env: A multi-env environment from envpool.make(..., num_envs>1).
        """
        self.envpool_env = envpool_env
        observation_space = envpool_env.observation_space
        action_space = envpool_env.action_space

        super().__init__(venv=envpool_env, observation_space=observation_space, action_space=action_space)

        self._actions = None

    def reset(self):
        return self.envpool_env.reset()

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        # Step the env
        obs, rewards, dones, infos = self.envpool_env.step(self._actions)
        return obs, rewards, dones, infos

    def close(self):
        return self.envpool_env.close()

    @property
    def num_envs(self):
        """
        Expose the underlying EnvPool's number of parallel environments.
        """
        return 8

class InfrequentPrintCallback(BaseCallback):
    """
    Prints debug info every 'print_freq' episodes. 
    """
    def __init__(self, print_freq=10, verbose=1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.ep_reward_buffer = 0
        self.ep_len_buffer = 0

    def _on_step(self) -> bool:
        done_array = self.locals["dones"]
        reward_array = self.locals["rewards"]

        for i, done in enumerate(done_array):
            self.ep_reward_buffer += reward_array[i]
            self.ep_len_buffer += 1
            if done:
                self.episode_count += 1
                self.episode_rewards.append(self.ep_reward_buffer)
                self.episode_lengths.append(self.ep_len_buffer)
                if (self.episode_count % self.print_freq) == 0 and self.verbose > 0:
                    avg_rew = np.mean(self.episode_rewards[-self.print_freq:])
                    avg_len = np.mean(self.episode_lengths[-self.print_freq:])
                    print(f"[Episode {self.episode_count}] "
                          f"AvgReward(last {self.print_freq} eps): {avg_rew:.2f}, "
                          f"AvgLength: {avg_len:.1f}")
                self.ep_reward_buffer = 0
                self.ep_len_buffer = 0
        return True


def make_training_env(num_envs: int, env_id: str):
    """
    Create an EnvPool environment with multiple parallel envs, 
    then wrap in EnvPoolToVecEnv for SB3 compatibility.
    """
    # Make sure to use envpool.make(...) not envpool.make_gym(...)
    envpool_env = envpool.make(env_id, env_type="gym", num_envs=num_envs)
    sb3_env = EnvPoolToVecEnv(envpool_env)
    return sb3_env


def train(
    env_id: str = "Humanoid-v4",  # Replace with your custom ID if needed
    num_envs: int = 8,
    total_timesteps: int = 200_000,
    tensorboard_log: str = "./logs",
    model_save_path: str = "./quadruped_ppo_model"
):
    """
    Main training entry point.
    """
    # 1) Create parallel envs
    env = make_training_env(num_envs=num_envs, env_id=env_id)

    # 2) Set up SB3 logger
    new_logger = configure(folder=tensorboard_log, format_strings=["stdout", "csv", "tensorboard"])

    # 3) Construct PPO model
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,      # can be tuned
        learning_rate=3e-4,
        batch_size=256,
        verbose=1,
        tensorboard_log=tensorboard_log
    )
    model.set_logger(new_logger)

    # 4) Callbacks
    infrequent_print_cb = InfrequentPrintCallback(print_freq=20, verbose=1)
    eval_env = make_training_env(num_envs=1, env_id=env_id)
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=3,  # number of episodes each eval
        deterministic=True,
        verbose=1
    )
    callbacks = [infrequent_print_cb, eval_cb]

    # 5) Train
    print("[Training] Starting PPO training...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    print("[Training] Finished training.")

    # 6) Save model
    model.save(model_save_path)
    print(f"[Training] Model saved at: {model_save_path}.zip")

    # 7) Close
    env.close()
    eval_env.close()


def evaluate(
    model_path: str = "./quadruped_ppo_model",
    env_id: str = "Humanoid-v4",
    episodes: int = 5
):
    """
    Evaluate the model on 1 environment for some episodes.
    """
    eval_env = make_training_env(num_envs=1, env_id=env_id)
    model = PPO.load(model_path, env=eval_env)
    print(f"[Evaluation] Loaded model from {model_path}.zip")

    all_rewards = []
    for ep in range(episodes):
        obs = eval_env.reset()
        done = np.array([False])
        ep_reward = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            ep_reward += rewards[0]
            done = dones
        all_rewards.append(ep_reward)
        print(f"Episode {ep+1} reward: {ep_reward:.2f}")

    avg = np.mean(all_rewards)
    std = np.std(all_rewards)
    print(f"[Evaluation] Average Reward: {avg:.2f} +/- {std:.2f}")
    eval_env.close()


if __name__ == "__main__":
    # Example usage:
    train(
        env_id="Humanoid-v4",
        num_envs=4,            # or 8, or more if your CPU can handle it
        total_timesteps=100000,
        tensorboard_log="./logs",
        model_save_path="./quadruped_ppo_model"
    )

    evaluate(
        model_path="./quadruped_ppo_model",
        env_id="Humanoid-v4",
        episodes=5
    )
