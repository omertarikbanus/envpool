#!/usr/bin/env bash
# Add an opt-in base-height reward term to QuadrupedPD-v1.
#
# The term legged_gym ships and GO2RoughCfg parameterises (base_height_target =
# 0.25) but weights at zero: square(base_z - target). Default scale here is 0.0,
# so the exact-recipe arm is unchanged and seeds 1-3 stay reproducible; the
# height-enforced arm opts in with --base-height-scale.
#
# DO NOT RUN while a seed is training or an evaluation is in flight: rebuilding
# swaps the .so and would split a campaign across two binaries.
set -euo pipefail
cd /home/tarik/quadruped_ws/envpool
H=envpool/mujoco/gym/quadruped_pd.h

python3 - "$H" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()

def sub(old, new):
    global s
    assert s.count(old) == 1, f"anchor not unique or missing: {old[:60]!r}"
    s = s.replace(old, new)

sub("  static constexpr int kRewardTermDim = 10;",
    "  static constexpr int kRewardTermDim = 11;")

sub('        "pd_command_vy"_.Bind(0.0), "pd_command_heading"_.Bind(0.0));',
    '        "pd_command_vy"_.Bind(0.0), "pd_command_heading"_.Bind(0.0),\n'
    '        // _reward_base_height: square(base_z - target). legged_gym ships\n'
    '        // this term and GO2RoughCfg sets base_height_target = 0.25, but\n'
    '        // LeggedRobotCfg weights it at -0., so the recipe never enforces\n'
    '        // the height it declares. Scale 0.0 keeps that behaviour exactly.\n'
    '        "pd_base_height_scale"_.Bind(0.0),\n'
    '        "pd_base_height_target"_.Bind(0.25));')

sub("  mjtNum fixed_vx_{0.0}, fixed_vy_{0.0}, fixed_heading_{0.0};",
    "  mjtNum fixed_vx_{0.0}, fixed_vy_{0.0}, fixed_heading_{0.0};\n"
    "  mjtNum base_height_scale_{0.0}, base_height_target_{0.25};")

sub('    fixed_heading_ = spec.config["pd_command_heading"_];',
    '    fixed_heading_ = spec.config["pd_command_heading"_];\n'
    '    base_height_scale_ = spec.config["pd_base_height_scale"_];\n'
    '    base_height_target_ = spec.config["pd_base_height_target"_];')

sub("        dof_limits * -10.0};                                      // dof_pos_limits (GO2)",
    "        dof_limits * -10.0,                                       // dof_pos_limits (GO2)\n"
    "        std::pow(st.base_pos[2] - base_height_target_, 2) *\n"
    "            base_height_scale_};                                  // base_height (off by default)")

p.write_text(s)
print("patched", p)
PY

python3 - <<'PY'
from pathlib import Path
p = Path("examples/common/rsl_vec_env.py"); s = p.read_text()
old = '    "dof_pos_limits",\n)'
assert s.count(old) == 1
p.write_text(s.replace(old, '    "dof_pos_limits", "base_height",\n)'))
print("patched", p)
PY

python3 - <<'PY'
from pathlib import Path
p = Path("examples/train_unitree.py"); s = p.read_text()
old = '    p.add_argument("--no-push", action="store_true",'
assert s.count(old) == 1
s = s.replace(old,
    '    p.add_argument("--base-height-scale", type=float, default=0.0,\n'
    '                   help="weight of the base-height term; 0.0 is the "\n'
    '                        "source recipe, which never enforces its own "\n'
    '                        "declared base_height_target")\n'
    '    p.add_argument("--base-height-target", type=float, default=0.25,\n'
    '                   help="target base height in m for that term")\n'
    + old)
old2 = '    env_kwargs = {"pd_push_robots": not args.no_push}'
assert s.count(old2) == 1
s = s.replace(old2,
    '    env_kwargs = {"pd_push_robots": not args.no_push,\n'
    '                  "pd_base_height_scale": args.base_height_scale,\n'
    '                  "pd_base_height_target": args.base_height_target}')
p.write_text(s)
print("patched", p)
PY

echo "--- now rebuild inside the container: cd /app/envpool && make run ---"
