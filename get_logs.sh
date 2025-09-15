#!/usr/bin/env bash

# --- Edit these ---
LOCAL_DIR="logs"
REMOTE_USER="sstrostm"
REMOTE_HOST="192.168.123.200"
REMOTE_FILE="~/fusion/envpool/logs/0_log.csv"
SSH_KEY="$HOME/.ssh/id_ed25519.pub"
# -------------------
while true; do
    scp -i "$SSH_KEY" \
      -o StrictHostKeyChecking=accept-new \
      "${REMOTE_USER}@${REMOTE_HOST}:$REMOTE_FILE" \
      "$LOCAL_DIR/"
  sleep 1
done
