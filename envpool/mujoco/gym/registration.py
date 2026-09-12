# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mujoco gym env registration."""

from envpool.registration import register

gym_mujoco_envs = [
  ("Ant", "v3", False, 1000),
  ("Ant", "v4", True, 1000),
  ("HalfCheetah", "v3", False, 1000),
  ("HalfCheetah", "v4", True, 1000),
  ("Hopper", "v3", False, 1000),
  ("Hopper", "v4", True, 1000),
  ("HumanoidStandup", "v2", False, 1000),
  ("HumanoidStandup", "v4", True, 100000),
  ("InvertedDoublePendulum", "v2", False, 1000),
  ("InvertedDoublePendulum", "v4", True, 1000),
  ("InvertedPendulum", "v2", False, 1000),
  ("InvertedPendulum", "v4", True, 1000),
  ("Pusher", "v2", False, 100),
  ("Pusher", "v4", True, 100),
  ("Reacher", "v2", False, 50),
  ("Reacher", "v4", True, 50),
  ("Swimmer", "v3", False, 1000),
  ("Swimmer", "v4", True, 1000),
  ("Walker2d", "v3", False, 1000),
  ("Walker2d", "v4", True, 1000),
]

for task, version, post_constraint, max_episode_steps in gym_mujoco_envs:
  extra_args = {}
  if task == "Ant" and version == "v3":
    extra_args["use_contact_force"] = True
  register(
    task_id=f"{task}-{version}",
    import_path="envpool.mujoco.gym",
    spec_cls=f"Gym{task}EnvSpec",
    dm_cls=f"Gym{task}DMEnvPool",
    gym_cls=f"Gym{task}GymEnvPool",
    gymnasium_cls=f"Gym{task}GymnasiumEnvPool",
    post_constraint=post_constraint,
    max_episode_steps=max_episode_steps,
    **extra_args,
  )

# The quadruped WBIC environment. Both task ids are the same C++ env; they
# differ only in observation width, exactly as Ant-v3/v4 differ only by
# use_contact_force.
#
#   v0  54 obs, symmetric            -- the Gamma line, incl. gamma5_stage3
#   v1  61 obs, privileged tail      -- the Delta line's asymmetric critic
#
# v0 is v1 with the trailing 7 simulator-truth values withheld; see
# quadruped_wbc.h.
for version, privileged in [("v0", False), ("v1", True)]:
  register(
    task_id=f"QuadrupedWBC-{version}",
    import_path="envpool.mujoco.gym",
    spec_cls="GymQuadrupedWBCEnvSpec",
    dm_cls="GymQuadrupedWBCDMEnvPool",
    gym_cls="GymQuadrupedWBCGymEnvPool",
    gymnasium_cls="GymQuadrupedWBCGymnasiumEnvPool",
    post_constraint=True,
    max_episode_steps=1000,
    privileged_observations=privileged,
  )

# End-to-end joint-position PD task: an exact recreation of unitree_rl_gym's
# Go2 recipe (see quadruped_pd.h). v1, not v0: v0 was an
# earlier partial port at 100 Hz whose dev checkpoints (data/rudin_*) must not
# load into these semantics.
#
#   frame_skip 10 control ticks x 2 ms = 0.02 s, the recipe's policy dt
#   (sim.dt 0.005 x decimation 4). The env refuses any other product.
#   max_episode_steps 1001: legged_gym times out when episode_length_buf >
#   max_episode_length (1000). The env reports timeouts in info["time_out"].
register(
  task_id="QuadrupedPD-v1",
  import_path="envpool.mujoco.gym",
  spec_cls="GymQuadrupedPDEnvSpec",
  dm_cls="GymQuadrupedPDDMEnvPool",
  gym_cls="GymQuadrupedPDGymEnvPool",
  gymnasium_cls="GymQuadrupedPDGymnasiumEnvPool",
  post_constraint=True,
  frame_skip=10,
  max_episode_steps=1001,
)
