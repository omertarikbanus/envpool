"""EnvPool `QuadrupedPD-v1` behind rsl_rl v1.0.2's `VecEnv` interface.

The environment itself reproduces unitree_rl_gym's `LeggedRobot` (see
`envpool/mujoco/gym/quadruped_pd.h`). This adapter only
reshapes EnvPool's API into the tensors `OnPolicyRunner` consumes, matching
`LeggedRobot.step()`'s return contract:

    obs, privileged_obs (None), rewards, dones, extras

with `extras["time_outs"]` (legged_gym's `time_out_buf`, used by rsl_rl to
bootstrap truncated episodes) and, when any env finishes, `extras["episode"]`
holding the mean per-term episode sum divided by `episode_length_s`, exactly as
`LeggedRobot.reset_idx()` logs it.

Two things live in the env, not here, because they need per-env state:
`init_at_random_ep_len` (pass False to `learn()`; the env randomises its own
first episode length) and the reset-then-push sequence.
"""

from __future__ import annotations

import numpy as np
import torch

import envpool

TASK_ID = "QuadrupedPD-v1"
EPISODE_LENGTH_S = 20.0
MAX_EPISODE_LENGTH = 1000

# Order of info["reward_terms"], fixed by ComputeJointPDReward().
REWARD_TERMS = (
    "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy",
    "torques", "dof_acc", "feet_air_time", "collision", "action_rate",
    "dof_pos_limits",
)


class EnvPoolLeggedVecEnv:
    """rsl_rl `VecEnv` over a synchronous EnvPool batch (duck-typed)."""

    num_obs = 48
    num_privileged_obs = None
    num_actions = 12
    max_episode_length = MAX_EPISODE_LENGTH

    def __init__(self, num_envs: int, seed: int, sim_config_path: str,
                 num_threads: int = 0, **env_kwargs):
        self.num_envs = int(num_envs)
        self.device = torch.device("cpu")
        self.env = envpool.make(
            TASK_ID, env_type="gymnasium", num_envs=self.num_envs, seed=seed,
            num_threads=num_threads, sim_config_path=sim_config_path,
            **env_kwargs)
        # Only read by OnPolicyRunner when init_at_random_ep_len=True, which
        # the env handles itself; kept for interface completeness.
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.extras: dict = {}
        self._episode_sums = np.zeros((self.num_envs, len(REWARD_TERMS)))
        self._all_ids = np.arange(self.num_envs)
        obs, _ = self.env.reset()
        self.obs_buf = torch.as_tensor(np.asarray(obs), dtype=torch.float32)

    # -- rsl_rl VecEnv -----------------------------------------------------
    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset(self):
        # OnPolicyRunner.__init__ calls this once. The pool was reset at
        # construction; a second full reset would consume the env-side
        # first-episode randomisation.
        return self.obs_buf, None

    def step(self, actions: torch.Tensor):
        act = actions.detach().to("cpu", torch.float64).numpy()
        obs, rew, term, trunc, info = self.env.step(act)
        # Sync EnvPool writes row i for the i-th requested env; assert it, since
        # a silent permutation would scramble observations across robots.
        assert np.array_equal(info["env_id"], self._all_ids), "EnvPool row order"
        obs = np.array(obs, dtype=np.float32, copy=True)
        done = np.logical_or(term, trunc)
        time_out = np.asarray(info["time_out"]).astype(bool)
        self._episode_sums += np.asarray(info["reward_terms"])

        self.extras = {"time_outs": torch.as_tensor(time_out)}
        ids = np.flatnonzero(done)
        if ids.size:
            self.extras["episode"] = {
                f"rew_{name}": torch.tensor(
                    float(self._episode_sums[ids, i].mean()) / EPISODE_LENGTH_S)
                for i, name in enumerate(REWARD_TERMS)
            }
            self._episode_sums[ids] = 0.0
            # legged_gym resets inside step() and returns the reset env's new
            # observation; EnvPool needs an explicit reset of those ids.
            reset_obs, reset_info = self.env.reset(ids.astype(np.int32))
            assert np.array_equal(reset_info["env_id"], ids), "EnvPool reset order"
            obs[ids] = np.asarray(reset_obs, dtype=np.float32)

        self.obs_buf = torch.from_numpy(obs)
        return (self.obs_buf, None,
                torch.as_tensor(np.asarray(rew), dtype=torch.float32),
                torch.as_tensor(done),
                self.extras)

    def close(self):
        close = getattr(self.env, "close", None)
        if close:
            close()
