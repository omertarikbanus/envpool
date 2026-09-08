"""Temporary: does num_threads == num_envs give every env a live gait/onset?"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import setup_environment, load_model_and_normalization

n = 256
for threads in (int(x) for x in sys.argv[1].split(",")):
    cfg = {"sim_config_path": "/app/quadcontrol/config/robots/sim/envpool_train_Gamma6.toml",
           "max_episode_steps": 1550}
    if threads: cfg["num_threads"] = threads
    env = setup_environment("Humanoid-v4", n, 7, env_config=cfg)
    path = "/app/envpool/data/gamma5_stage2/quadruped_ppo_model"
    model, env = load_model_and_normalization(path + ".zip", env, path + "_vecnormalize.pkl")
    obs = env.reset()
    onset = np.full(n, -1.0)
    t0 = time.time()
    steps = 300
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        for i in range(n):
            onset[i] = max(onset[i], float(infos[i]["force_onset"]))
    dt = time.time() - t0
    bad = int((onset < 0).sum())
    print(f"RESULT threads={threads or 'default':>7} no_onset={bad:4d}/{n} "
          f"env_steps_per_s={n*steps/dt:8.0f}", flush=True)
    env.close()
