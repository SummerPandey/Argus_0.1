#!/bin/bash
set -u

PROJECT_ROOT="/home/nvidia/handwash-monitor"
PROJECT_DIR="$PROJECT_ROOT/handwash-next"
IMAGE="dustynv/nanoowl:r36.3.0"
MODE="${1:---real}"

if ! command -v jetson-containers >/dev/null 2>&1; then
    echo "jetson-containers is not installed or is not on PATH."
    exit 1
fi

if [ ! -e /dev/video0 ] && [ ! -e /dev/video1 ]; then
    echo "No camera found. Reconnect the USB camera and try again."
    exit 1
fi

ARGS=()
if [ "$MODE" = "--room" ]; then
    ARGS=(--room)
fi

exec jetson-containers run \
    --workdir /workspace/handwash-monitor/handwash-next \
    --volume "$PROJECT_ROOT:/workspace/handwash-monitor" \
    "$IMAGE" \
    python3 nanoowl_monitor.py "${ARGS[@]}"
