#!/bin/bash

# Detect OS and Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
OS=$(uname)
ARCH=$(uname -m)

if [ "$OS" = "Darwin" ]; then
  PLATFORM="macosx_$(sw_vers -productVersion | cut -d. -f1)_0_x86_64"
  if [ "$ARCH" = "arm64" ]; then
    PLATFORM="macosx_$(sw_vers -productVersion | cut -d. -f1)_0_arm64"
  fi
else
  if [ "$ARCH" = "x86_64" ]; then
    PLATFORM="linux_x86_64"
  else
    PLATFORM="linux_${ARCH}"
  fi
fi

# Set Bazel version and build
export USE_BAZEL_VERSION=6.5.0
# Disable VizDoom

# Build with cleaner output
echo "Building EnvPool wheel..."


bazel run --config=release //:setup -- bdist_wheel


# Find the wheel file
WHEEL_PATH=$(find bazel-bin/setup.runfiles/envpool/dist -name "envpool-*-${PYTHON_VERSION}-${PYTHON_VERSION}-${PLATFORM}.whl" 2>/dev/null)

if [ -z "$WHEEL_PATH" ]; then
  echo "Error: Could not find built wheel file."
  echo "Available wheels:"
  find bazel-bin/setup.runfiles/envpool/dist -name "*.whl" 2>/dev/null
  exit 1
fi

# Install the wheel
echo "Installing wheel: $WHEEL_PATH"
pip3 uninstall envpool -y
pip3 install "$WHEEL_PATH" -v

# Run example
echo "Running example..."
python3 examples/env_step.py