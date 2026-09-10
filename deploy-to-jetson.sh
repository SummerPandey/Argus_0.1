#!/bin/bash
# Sync this repo onto a Jetson over SSH. Run this FROM a machine that can
# reach the Jetson (your laptop, not this repo's dev/CI environment) -
# there is no direct network path from here to the device.
#
# Usage:
#   ./deploy-to-jetson.sh user@jetson-host [remote-dir]
#
# remote-dir defaults to /home/nvidia/handwash-monitor, matching the path
# argus-launcher/run_handwash.sh and run_equipment.sh already assume.
set -euo pipefail

HOST="${1:?Usage: $0 user@jetson-host [remote-dir]}"
REMOTE_DIR="${2:-/home/nvidia/handwash-monitor}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Syncing $SCRIPT_DIR -> $HOST:$REMOTE_DIR ..."

# --delete keeps the device's copy from drifting from this repo over
# repeated deploys, but never touches on-device calibration/state: those
# are excluded outright, not just left alone by chance.
rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'handwash-next/config.json' \
    --exclude 'handwash-next/.deps' \
    --exclude 'handwash-next/logs' \
    --exclude 'handwash-next/.cache' \
    --exclude 'equipment-next/config.json' \
    "$SCRIPT_DIR/" "$HOST:$REMOTE_DIR/"

echo
echo "Done. On the Jetson:"
echo "  $REMOTE_DIR/argus-launcher/argus_launcher.sh      # desktop dashboard"
echo "  $REMOTE_DIR/argus-launcher/run_equipment.sh        # equipment scan directly"
