bazel run --config=release //:setup -- bdist_wheel
pip3 install bazel-bin/setup.runfiles/envpool/dist/envpool-0.8.4-cp310-cp310-linux_x86_64.whl --force-reinstall
python3 examples/env_step.py