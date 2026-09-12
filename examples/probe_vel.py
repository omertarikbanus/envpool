#!/usr/bin/env python3
"""
Probe: does the deterministic ~130-160 step topple come from the aggressive
2.1 m/s base velocity command? Run the same static-GRF hold under different
sim configs that differ ONLY in policy_base_velocity / desired_velocity, and
log z + roll each step to see where lateral divergence starts.

Usage: probe_vel.py <sim_config_path> <tag>
"""
import sys
import numpy as np
import envpool

FORCE_Z_MAX = 250.0
TARGET_Z_N = 150.0
N_STEPS = 400

def fz_action(newtons):
    return (newtons / FORCE_Z_MAX) * 2.0 - 1.0

def main():
    cfg = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else cfg
    env = envpool.make_gym("QuadrupedWBC-v1", num_envs=1, sim_config_path=cfg)
    adim = env.action_space.shape[0]

    a = np.zeros((1, adim), dtype=np.float32)
    fz = fz_action(TARGET_Z_N)
    for leg in range(4):
        a[0, 3 + leg * 3 + 2] = fz

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    print(f"=== {tag} :: static 150N GRF, zero vel-residual ===")
    for t in range(N_STEPS):
        out = env.step(a)
        if len(out) == 5:
            obs, rew, term, trunc, info = out
            done = bool(term[0]) or bool(trunc[0])
        else:
            obs, rew, done_arr, info = out
            done = bool(done_arr[0])
        cmd_vx = obs[0, 0]
        vbody = obs[0, 1:4]
        rpy = obs[0, 7:10]
        if t % 20 == 0 or done:
            print(f"t={t:4d} cmd_vx={cmd_vx:+.2f} rew={float(rew[0]):7.2f} "
                  f"vB=[{vbody[0]:+.2f},{vbody[1]:+.2f},{vbody[2]:+.2f}] "
                  f"roll={rpy[0]:+.2f} pitch={rpy[1]:+.2f} yaw={rpy[2]:+.2f}")
        if done:
            print(f">>> {tag}: DONE step {t}  roll={rpy[0]:+.2f} yaw={rpy[2]:+.2f} vB_y={vbody[1]:+.2f}")
            break
    else:
        print(f">>> {tag}: survived all {N_STEPS} steps")
    env.close()

if __name__ == "__main__":
    main()
