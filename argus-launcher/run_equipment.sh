#!/bin/bash

PROJECT_DIR="/home/nvidia/handwash-monitor/equipment-next"
PROJECT_FILE="$PROJECT_DIR/nanoowl_equipment_monitor.py"
IMAGE="dustynv/nanoowl:r36.3.0"
LOCK_FILE="/tmp/argus-equipment.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Argus Equipment Scan is already running."
    echo "Close the existing camera window with Q before opening it again."
    sleep 4
    exit 0
fi

if [ ! -f "$PROJECT_FILE" ]; then
    echo "ERROR: nanoowl_equipment_monitor.py was not found."
    echo "Expected location: $PROJECT_FILE"
    read -p "Press Enter to close..."
    exit 1
fi

if ! command -v jetson-containers >/dev/null 2>&1; then
    echo "ERROR: jetson-containers was not found."
    echo "Open the project using your normal NanoOWL setup, then try again."
    read -p "Press Enter to close..."
    exit 1
fi

# ---------------------------------------------------------
# Camera checks. A busy camera is the most common failure,
# and without these it surfaces as a NanoOWL traceback.
# ---------------------------------------------------------

if [ ! -e /dev/video0 ] && [ ! -e /dev/video1 ]; then
    echo "ERROR: no camera device was found."
    echo "Neither /dev/video0 nor /dev/video1 exists."
    echo "Reconnect the USB camera, wait a few seconds, then try again."
    read -p "Press Enter to close..."
    exit 1
fi

# Another NanoOWL container holding the camera is the usual culprit.
# docker may require sudo, so treat any failure here as "cannot tell"
# and continue rather than blocking the launch.
if command -v docker >/dev/null 2>&1; then
    RUNNING="$(docker ps --filter "ancestor=$IMAGE" --format '{{.Names}}' 2>/dev/null)"

    if [ -n "$RUNNING" ]; then
        echo "WARNING: a NanoOWL container is already running:"
        echo "$RUNNING" | sed 's/^/  /'
        echo
        echo "It is probably holding the camera. Stop it with:"
        echo "$RUNNING" | sed 's/^/  sudo docker stop /'
        echo
        read -p "Press Enter to try anyway, or Ctrl+C to cancel..."
    fi
fi

# fuser only reports processes this user can see, so a quiet result
# does not prove the camera is free. It still catches host-side
# programs such as Cheese or an old browser demo.
if command -v fuser >/dev/null 2>&1; then
    BUSY="$(fuser /dev/video0 /dev/video1 2>/dev/null | tr -d ' ')"

    if [ -n "$BUSY" ]; then
        echo "WARNING: another process is using the camera."
        echo "Inspect it from this terminal with:"
        echo "  sudo fuser -v /dev/video0 /dev/video1"
        echo
        read -p "Press Enter to try anyway, or Ctrl+C to cancel..."
    fi
fi

echo "Starting Argus Equipment Scan (general tier)..."
echo "Use R to reset and Q to quit."
echo

jetson-containers run \
    --workdir /workspace/equipment-scan \
    --volume "$PROJECT_DIR:/workspace/equipment-scan" \
    "$IMAGE" \
    python3 nanoowl_equipment_monitor.py

STATUS=$?
echo

if [ "$STATUS" -ne 0 ]; then
    echo "Argus stopped with an error. The message above can be used for troubleshooting."
    echo
    echo "If the error mentions a busy or unavailable camera, run:"
    echo "  sudo fuser -v /dev/video0 /dev/video1"
    echo "  sudo docker ps"
    echo "Then stop only the process or container that owns the camera."
else
    echo "Argus Equipment Scan has stopped."
fi

read -p "Press Enter to close..."
exit "$STATUS"
