import numpy as np
import envpool
import logging
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy


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


def main(args):
    # Create the EnvPool environment and wrap it.
    env = envpool.make(args.env_name, env_type="gym", num_envs=1)
    env.spec.id = args.env_name
    env = VecAdapter(env)
    env = VecMonitor(env)
    # Wrap with VecNormalize and load normalization statistics.
    env = VecNormalize(env, training=False)
    env = VecNormalize.load(args.vecnormalize_path, env)
    env.training = False
    env.norm_reward = False

    # Load the model and pass the normalized environment.
    model = PPO.load(args.model_save_path, env=env)

    logging.info("Starting training...")
    # try:
    #     model.learn(total_timesteps=args.total_timesteps)
    # except KeyboardInterrupt:
    #     logging.info("Training interrupted by user. Saving model...")
    #     model.save(args.model_save_path)
    #     logging.info(f"Model saved at: {args.model_save_path}.zip")
    #     env.close()
    #     return

    logging.info("Training complete.")
    # model.save(args.model_save_path)
    # logging.info(f"Model saved at: {args.model_save_path}.zip")

    # Evaluate the model on the normalized EnvPool environment.
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"EnvPool Evaluation - {args.env_name}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Humanoid-v4")
    parser.add_argument("--model_save_path", type=str, default="./quadruped_ppo_model.zip")
    parser.add_argument("--vecnormalize_path", type=str, default="./vecnormalize.pkl")
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    args = parser.parse_args()
    main(args)
