#!/bin/bash

PROJECT_DIR="/home/nvidia/handwash-monitor"
IMAGE="dustynv/nanoowl:r36.3.0"

if [ ! -f "$PROJECT_DIR/automatic_rubbing.py" ]; then
    echo "ERROR: automatic_rubbing.py was not found."
    echo "Expected location: $PROJECT_DIR/automatic_rubbing.py"
    read -p "Press Enter to close..."
    exit 1
fi

echo "Starting Argus Handwashing Monitor..."
echo "Press Q in the camera window to quit."
echo

jetson-containers run \
    --workdir /workspace/handwash-monitor \
    --volume "$PROJECT_DIR:/workspace/handwash-monitor" \
    "$IMAGE" \
    python3 automatic_rubbing.py

echo
echo "Handwashing Monitor has stopped."
read -p "Press Enter to close..."
