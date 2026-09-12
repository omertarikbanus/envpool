
# Notes 

## TODO

- Clean up codes.
- remove unused log files.
- Evaluation harness imports: `quadcontrol/evaluations/run.py` replaces the old
  `comparison/eval_rl.py`. As mandated by `AGENTS.md`, `make run` inside the
  container is the single source of truth for building and installing the
  `envpool` package wheel into site-packages.
- Windowed visualization (GLFW): container X11 pass-through is fully integrated
  into `make docker-run` via `-e DISPLAY=$(DISPLAY)` and `/tmp/.X11-unix` mounts.
  To render visually, run with `headless = false` in the sim config.


## Development Notes
- number of environments is crucial for succesfull training. I set them to 32 vs 128 and the training was stable only for 32 environments case. 

- frameskip variable also effects the training's success. Under same conditions with 32 envs, when frameskip was set to 15 the policy's learning was much worse. 

- 
## Basic Usage

Basic Install (Inside Container)

Always follow `AGENTS.md` Hard Rules: launch container via `make docker-run`
from `envpool/`, then build and install the package wheel using:

    make run

This exports `USE_BAZEL_VERSION=6.5.0`, invokes `bazel run --config=release //:setup -- bdist_wheel`,
and pip-installs the wheel. Do NOT manually edit setuptools/jaraco cache files or run piecemeal pip installs.


Historical Mac Notes (Legacy Reference)

    symlink qt. install it if not installed.
    ln -s /usr/include/aarch64-linux-gnu/qt5 /usr/include/qt
    ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
