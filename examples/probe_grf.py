#!/usr/bin/env python3
"""
Direct env probe: hold static GRF z-references at ~150 N/foot and watch what
happens step by step. This isolates whether WBIC can keep the Go2 standing when
handed a fixed, reasonable ground-reaction-force reference (no learning, no
random policy).

Action layout (kPolicyActionDim=24):
  [0..2]   velocity residual (vx, vy, yaw)          -> we send 0
  [3..14]  per-leg [fx, fy, fz] GRF refs, 4 legs    -> fz mapped [-1,1]->[0,force_z_max]
  [15..22] per-leg footstep residuals (2 each)      -> we send 0
  [23]     phase_delta                               -> we send 0

force_z_max = 250 N (from envpool.toml). To command ~150 N/foot in z:
  action_fz = (150/250)*2 - 1 = +0.2
"""
import numpy as np
import envpool

SIM_CONFIG = "/app/quadcontrol/config/robots/sim/envpool.toml"
FORCE_Z_MAX = 250.0
TARGET_Z_N = 150.0
N_STEPS = 400

def fz_action(newtons):
    # inverse of mapRange_(a, -1, 1, 0, force_z_max)
    return (newtons / FORCE_Z_MAX) * 2.0 - 1.0

def main():
    env = envpool.make_gym("Humanoid-v4", num_envs=1, sim_config_path=SIM_CONFIG)
    adim = env.action_space.shape[0]
    print(f"action dim = {adim}, obs dim = {env.observation_space.shape}")

    a = np.zeros((1, adim), dtype=np.float32)
    fz = fz_action(TARGET_Z_N)
    for leg in range(4):
        base = 3 + leg * 3
        a[0, base + 0] = 0.0      # fx
        a[0, base + 1] = 0.0      # fy
        a[0, base + 2] = fz       # fz -> ~150 N
    print(f"commanding fz action={fz:.4f} -> ~{TARGET_Z_N:.0f} N/foot z-GRF, all else zero")

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    first_done_step = None
    for t in range(N_STEPS):
        out = env.step(a)
        if len(out) == 5:
            obs, rew, term, trunc, info = out
            done = bool(term[0]) or bool(trunc[0])
        else:
            obs, rew, done_arr, info = out
            done = bool(done_arr[0])
        # obs[0] = cmd_vx, obs[1..3] = vBody, obs[4..6]=omega, obs[7..9]=rpy
        vbody = obs[0, 1:4]
        rpy = obs[0, 7:10]
        r = float(rew[0])
        if t < 20 or t % 10 == 0 or done:
            print(f"t={t:4d} rew={r:8.3f} vBody=[{vbody[0]:+.2f},{vbody[1]:+.2f},{vbody[2]:+.2f}] "
                  f"rpy=[{rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f}] done={done}")
        if done and first_done_step is None:
            first_done_step = t
            print(f">>> FIRST DONE at step {t} (reward={r:.3f})")
            break
    if first_done_step is None:
        print(f">>> survived all {N_STEPS} steps with static 150 N GRF refs")
    env.close()

if __name__ == "__main__":
    main()
