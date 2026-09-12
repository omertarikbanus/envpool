#!/usr/bin/env python3
"""Deterministic checks of QuadrupedPD-v1 against the unitree_rl_gym recipe.

No learning involved; each check is a property the recipe fixes. Run inside the
container after `make run`:

    python3 examples/tests/unitree_recipe_check.py [--num-envs 64]

Exits non-zero if any check fails. The zero-action stand is the smoke gate: a
policy that outputs 0 holds legged_gym's default pose, so episodes must not end
by base contact or attitude before the 20 s timeout.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import envpool

SIM_CONFIG = "/app/quadcontrol/config/robots/sim/envpool_train_rudin.toml"
# The plant's limits, from the Go2 MJCF forcerange — not the recipe's stale
# Go1 calf effort (35.55 N.m); see quadruped_pd.h, PDConstants.
TORQUE_LIMITS = np.array([23.7, 23.7, 45.43] * 4)
FAILS: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def make(n: int, **kw):
    cfg = dict(pd_init_at_random_ep_len=False)
    cfg.update(kw)
    return envpool.make("QuadrupedPD-v1", env_type="gymnasium", num_envs=n,
                        seed=3, sim_config_path=SIM_CONFIG, **cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=64)
    n = ap.parse_args().num_envs

    # 1. Clean env: timing, reset state, torque clip.
    env = make(n, pd_obs_noise=False, pd_push_robots=False)
    check(env.observation_space.shape == (48,), f"obs dim 48 (got {env.observation_space.shape})")
    check(env.action_space.shape == (12,), f"action dim 12 (got {env.action_space.shape})")
    obs, info = env.reset()
    check(np.allclose(info["body_height"], 0.42), f"reset base height 0.42 (got {np.unique(np.round(info['body_height'], 4))[:4]})")
    t0 = np.asarray(info["sim_time"]).copy()
    zero = np.zeros((n, 12))
    obs, rew, term, trunc, info = env.step(zero)
    dt = np.asarray(info["sim_time"]) - t0
    check(np.allclose(dt, 0.02, atol=1e-9), f"policy dt 0.02 s (got {np.unique(np.round(dt, 6))})")

    # Commands: |(vx,vy)| is either 0 or > 0.2, each within [-1, 1].
    cmd = obs[:, 9:11] / 2.0
    norm = np.linalg.norm(cmd, axis=1)
    check(np.all((norm == 0) | (norm > 0.2)) and np.all(np.abs(cmd) <= 1.0),
          "commands: zeroed below 0.2, within [-1, 1]")
    check(np.all(np.abs(obs[:, 11] / 0.25) <= 1.0 + 1e-9), "yaw-rate command clipped to [-1, 1]")

    # Torque clip at the Go2 MJCF forcerange: saturate every joint.
    obs, rew, term, trunc, info = env.step(np.full((n, 12), 100.0))
    torques_term = np.asarray(info["reward_terms"])[:, 4]
    expected = -(TORQUE_LIMITS ** 2).sum() * 0.0002 * 0.02
    frac = np.mean(np.isclose(torques_term, expected, rtol=1e-3))
    check(frac > 0.9, f"saturated torques clip at [23.7, 23.7, 45.43] "
                      f"(term {np.median(torques_term):.5f}, expected {expected:.5f}, "
                      f"{frac:.0%} of envs match)")
    env.close()

    # 2. Zero-action stand, recipe defaults (noise on) but no pushes, full 20 s.
    env = make(n, pd_push_robots=False)
    env.reset()
    ended = np.zeros(n, bool)
    reason_timeout = np.zeros(n, bool)
    heights = []
    for step in range(1, 1002):
        obs, rew, term, trunc, info = env.step(zero)
        done = np.logical_or(term, trunc)
        new = done & ~ended
        reason_timeout |= new & np.asarray(info["time_out"]).astype(bool)
        ended |= done
        if step > 100:
            heights.append(np.asarray(info["body_height"])[~ended])
        if ended.all():
            break
    h = np.concatenate(heights) if heights else np.array([np.nan])
    early = ended & ~reason_timeout
    print(f"     stand: {reason_timeout.sum()}/{n} timed out at 1001 steps, "
          f"{early.sum()} ended early; base height after 2 s: "
          f"mean {h.mean():.3f} m, min {h.min():.3f} m", flush=True)
    check(early.sum() <= max(1, n // 20),
          "zero-action stand survives the 20 s episode (<=5% early terminations)")
    check(reason_timeout.sum() >= n - max(1, n // 20),
          "episodes end by timeout at step 1001 (legged_gym: > max_episode_length)")
    env.close()

    # 3. Noise is present and bounded where the recipe puts it.
    a = make(n, pd_push_robots=False, pd_obs_noise=False)
    b = make(n, pd_push_robots=False, pd_obs_noise=True)
    oa, _ = a.reset()
    ob, _ = b.reset()
    d = np.abs(ob - oa)
    bounds = np.r_[[0.2] * 3, [0.05] * 3, [0.05] * 3, [0.0] * 3, [0.01] * 12,
                   [0.075] * 12, [0.0] * 12]
    check(np.all(d <= bounds + 1e-9) and d[:, 12:24].max() > 0,
          "observation noise matches _get_noise_scale_vec bounds")
    a.close()
    b.close()

    print(f"\n{len(FAILS)} check(s) failed" if FAILS else "\nall checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
