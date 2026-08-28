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

# Advisory only, never fatal: prints a single hint if this Jetson has a
# faster power mode available than the one it is in (TensorRT throughput
# is gated hard by power mode + clocks), and stays silent otherwise.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$SCRIPT_DIR/jetson-performance.sh" ]; then
    "$SCRIPT_DIR/jetson-performance.sh" --check || true
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
