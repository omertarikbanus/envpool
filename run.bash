#!/bin/bash

# Start timing
START_TIME=$(date +%s)

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
echo "Detected OS: $OS"
echo "Detected Architecture: $ARCH"
echo "Detected Python version: $PYTHON_VERSION"
echo "Detected Platform: $PLATFORM"

# Set Bazel version and build
export USE_BAZEL_VERSION=6.5.0
# Disable VizDoom

# Build with cleaner output
echo "Building EnvPool wheel..."


if ! bazel run --config=release //:setup -- bdist_wheel > log.txt; then
  echo "Error: Bazel build failed. Check log.txt for details."
  exit 1
fi

echo "Build completed. Check log.txt for details."

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
if ! pip3 uninstall envpool -y; then
  echo "Error: Failed to uninstall envpool"
  exit 1
fi

if ! pip3 install "$WHEEL_PATH" -v; then
  echo "Error: Failed to install wheel"
  exit 1
fi


# Calculate and print elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - $START_TIME))
echo "Total build time: $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m $((ELAPSED%60))s"


# Run example
echo "Running example..."
python3 examples/env_step.py


# Calculate and print elapsed time
END_TIME=$(date +%s)
ELAPSED=$(($END_TIME - $START_TIME))
echo "Total execution time: $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m $((ELAPSED%60))s"