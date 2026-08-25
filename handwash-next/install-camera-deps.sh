#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m pip install --target "$PROJECT_DIR/.deps" -r "$PROJECT_DIR/requirements-camera.txt"
mkdir -p "$PROJECT_DIR/models"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
curl -fL "$MODEL_URL" -o "$PROJECT_DIR/models/hand_landmarker.task"
POSE_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
curl -fL "$POSE_URL" -o "$PROJECT_DIR/models/pose_landmarker_lite.task"
echo "Camera support installed. Double-click the Argus desktop icon."
