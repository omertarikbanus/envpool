export USE_BAZEL_VERSION=6.5.0 && \
bazel run --config=release //:setup -- bdist_wheel > bazel_build_log.log && \
pip3 uninstall envpool -y && \
pip3 install bazel-bin/setup.runfiles/envpool/dist/envpool-0.8.4-cp310-cp310-linux_x86_64.whl && \
python3 examples/env_step.py