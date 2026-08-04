# 46-dim observation wheel (pre-obs-extension)

Models X (260728_torque45_X_forceon), Y (260728_torque45_Y_forceoff) and every
checkpoint under 260803_Z_* were trained against a 46-dim observation. The
extended observation changes RLConstants::kObservationDim, so those policies
cannot be loaded by the new build -- SB3 rejects the observation-space mismatch.

To evaluate them again, swap this .so back in (no rebuild needed):

  docker cp mujoco_gym_envpool.so \
    envpool-dev:/usr/local/lib/python3.10/dist-packages/envpool/mujoco/

and restore the current one afterwards by re-running `make run` under /app/envpool.

Corresponding source state is the git commit made immediately before the
observation change in the quadcontrol repo.
