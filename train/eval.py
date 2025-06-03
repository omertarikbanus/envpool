import numpy as np
import envpool
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor


class VecAdapter(VecEnvWrapper):
    def __init__(self, venv):
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv)
        self.action_space = Box(
            low=np.asarray(venv.action_space.low, dtype=np.float32),
            high=np.asarray(venv.action_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = Box(
            low=np.asarray(venv.observation_space.low, dtype=np.float32),
            high=np.asarray(venv.observation_space.high, dtype=np.float32),
            dtype=np.float32,
        )

    def step_async(self, actions):
        self.actions = actions

    def reset(self):
        obs = self.venv.reset()[0]
        return np.asarray(obs, dtype=np.float32)

    def seed(self, seed=None):
        pass

    def step_wait(self):
        obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
        dones = terms + truncs
        obs = np.asarray(obs, dtype=np.float32)
        infos = []
        for i in range(self.num_envs):
            info_i = {k: v[i] for k, v in info_dict.items() if isinstance(v, np.ndarray)}
            if dones[i]:
                obs[i] = self.venv.reset(np.array([i]))[0][0]
            infos.append(info_i)
        return obs, rewards, dones, infos


def main():
    env = envpool.make("Humanoid-v4", env_type="gym", num_envs=1)
    env.spec.id = "Humanoid-v4"
    env = VecAdapter(env)
    env = VecMonitor(env)

    model = PPO.load("./quadruped_ppo_model.zip", env=env)  # ✅ correct loading method

    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
