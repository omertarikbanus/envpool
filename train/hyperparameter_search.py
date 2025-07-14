#!/usr/bin/env python3
"""
Fast Optuna hyper-parameter search for PPO + EnvPool with TensorBoard support.

Example:
  python hyper_search.py --env-name Humanoid-v4 --num-envs 512 --n-trials 40 \
                         --jobs 8 --trial-steps 100_000
"""

import argparse, os, time, warnings, numpy as np, torch as th, optuna
from datetime import datetime
import envpool, gym, gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, VecEnvWrapper
from packaging import version

# ─────────────── identical VecAdapter (verbatim) ───────────────
is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")
class VecAdapter(VecEnvWrapper):
    def __init__(self, venv: envpool.python.protocol.EnvPool):
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv)
        from gymnasium.spaces import Box
        self.action_space = Box(low=venv.action_space.low.astype(np.float32),
                                high=venv.action_space.high.astype(np.float32),
                                shape=venv.action_space.shape, dtype=np.float32)
        if isinstance(venv.observation_space, (gym.spaces.Box, gymnasium.spaces.Box)):
            self.observation_space = Box(low=venv.observation_space.low.astype(np.float32),
                                         high=venv.observation_space.high.astype(np.float32),
                                         shape=venv.observation_space.shape, dtype=np.float32)
        else:
            self.observation_space = venv.observation_space
    def step_async(self, actions): self.actions = actions
    def reset(self):
        obs = self.venv.reset() if is_legacy_gym else self.venv.reset()[0]
        return np.asarray(obs, np.float32)
    def step_wait(self):
        if is_legacy_gym:
            obs, rew, done, info = self.venv.step(self.actions)
        else:
            obs, rew, term, trunc, info = self.venv.step(self.actions); done = term + trunc
        obs = np.asarray(obs, np.float32)
        infos = []
        for i in range(self.num_envs):
            info_i = {k: v[i] for k, v in info.items() if isinstance(v, np.ndarray)}
            if done[i]:
                info_i["terminal_observation"] = obs[i]
                reset_obs = self.venv.reset(np.array([i])) if is_legacy_gym else self.venv.reset(np.array([i]))[0]
                obs[i] = np.asarray(reset_obs, np.float32)
            infos.append(info_i)
        return obs, rew, done, infos
# ────────────────────────────────────────────────────────────────

th.set_num_threads(1)
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

def make_env(env_name, num_envs, seed):
    env = envpool.make(env_name, env_type="gym", num_envs=num_envs, seed=seed)
    env.spec.id = env_name
    return VecMonitor(VecAdapter(env))

def objective(trial, args, tb_root):
    seed = trial.number + args.seed
    use_vn   = trial.suggest_categorical("use_vn", [0, 1])  # 0=False, 1=True

    env = make_env(args.env_name, args.num_envs, seed)
    if use_vn:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=5.0)

    # ── search space ──
    hp = dict(
        learning_rate=trial.suggest_loguniform("lr", 1e-5, 3e-4),
        n_steps=trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024]),
        gamma=trial.suggest_uniform("gamma", 0.95, 0.999),
        gae_lambda=trial.suggest_uniform("gae_lam", 0.9, 0.98),
        clip_range=trial.suggest_uniform("clip", 0.1, 0.3),
        ent_coef=trial.suggest_loguniform("ent", 1e-4, 1e-2),
        vf_coef=trial.suggest_uniform("vf", 0.4, 1.0),
        max_grad_norm=0.5,  # Add gradient clipping
    )
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=[dict(pi=[256], vf=[256])], log_std_init=-5)

    # unique TensorBoard folder per trial
    tb_dir = os.path.join(tb_root, f"trial_{trial.number}")
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=tb_dir,
                policy_kwargs=policy_kwargs, seed=seed, **hp)
    try:
        model.learn(total_timesteps=args.trial_steps, progress_bar=False)
    except:
        print(f"Trial {trial.number} failed during training.")
        env.close()
        return float("inf")
    if use_vn:
        env.training = False; env.norm_reward = False
    mean_r, _ = evaluate_policy(model, env, n_eval_episodes=5, warn=False)
    env.close()
    return -mean_r  # minimise

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--env-name", type=str, default="Humanoid-v4")
    p.add_argument("--num-envs", type=int, default=160)
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--trial-steps", type=int, default=1e6)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tb-dir", type=str, default="tb_logs")
    return p.parse_args()

def main():
    args = parse()

    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_root = os.path.join(args.tb_dir, args.env_name, stamp)
    os.makedirs(tb_root, exist_ok=True)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(lambda t: objective(t, args, tb_root),
                   n_trials=args.n_trials, n_jobs=args.jobs,
                   show_progress_bar=True)

    csv_path = f"study_{stamp}.csv"
    study.trials_dataframe().to_csv(csv_path, index=False)
    print(f"\nCSV saved → {csv_path}")
    print("\n=== Best hyper-parameters ===")
    for k, v in study.best_params.items():
        print(f"{k:10s}: {v}")
    print(f"Best mean reward: {-study.best_value:.2f}")

if __name__ == "__main__":
    main()
