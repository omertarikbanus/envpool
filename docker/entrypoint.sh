#!/usr/bin/env bash
# cp /app/envpool/docker/certs/AselsanSubCA.crt /usr/local/share/ca-certificates/
# cp /app/envpool/docker/certs/AselsanRootCA.crt /usr/local/share/ca-certificates/

# cp /host/usr/local/share/ca-certificates/AselsanSubCA.crt /usr/local/share/ca-certificates/
# cp /host/usr/local/share/ca-certificates/AselsanRootCA.crt /usr/local/share/ca-certificates/
# update-ca-certificates

echo "User mode"
exec bash
echo "export USE_BAZEL_VERSION=6.5.0" >> ~/.bashrc
echo -e '\n# Auto-Warpify\n[[ "$-" == *i* ]] && printf '\''\eP$f{"hook": "SourcedRcFileForWarp", "value": { "shell": "bash", "uname": "'$(uname)'", "tmux": false }}\x9c'\'' ' >> ~/.bashrc
ln -sf /app/legged-sim /app/envpool/envpool/mujoco/legged-sim
ln -s /usr/include/aarch64-linux-gnu/qt5 /usr/include/qt
ln -s /usr/include/eigen3/Eigen /usr/include/Eigen