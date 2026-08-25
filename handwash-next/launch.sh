#!/bin/bash
set -u
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1
PYTHON=python3
if [ -d "$PROJECT_DIR/.deps" ]; then
    export PYTHONPATH="$PROJECT_DIR/.deps${PYTHONPATH:+:$PYTHONPATH}"
fi
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export QT_QPA_FONTDIR="/usr/share/fonts/truetype/dejavu"
mkdir -p "$MPLCONFIGDIR"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/argus.log"
mkdir -p "$LOG_DIR"

if [ $# -eq 0 ]; then
    exec "$PYTHON" -m bubbles.chooser
fi

if [ "${1:-}" = "--demo" ]; then
    exec "$PYTHON" -m bubbles.demo
fi

# jetson-containers uses sudo. Desktop/Tk launchers have no controlling
# terminal, so sudo cannot request a password unless camera modes are
# relaunched in a terminal window.
if [ ! -t 0 ]; then
    if command -v gnome-terminal >/dev/null 2>&1; then
        MODE="${1:---real}"
        exec gnome-terminal --title="Argus Hand Hygiene" -- bash -lc \
          "'$PROJECT_DIR/launch.sh' '$MODE'; STATUS=\$?; echo; if [ \$STATUS -ne 0 ]; then echo 'Argus stopped with an error.'; read -p 'Press Enter to close...'; fi; exit \$STATUS"
    fi
    if command -v x-terminal-emulator >/dev/null 2>&1; then
        MODE="${1:---real}"
        exec x-terminal-emulator -e bash -lc \
          "'$PROJECT_DIR/launch.sh' '$MODE'; STATUS=\$?; echo; if [ \$STATUS -ne 0 ]; then echo 'Argus stopped with an error.'; read -p 'Press Enter to close...'; fi; exit \$STATUS"
    fi
fi

if ! command -v jetson-containers >/dev/null 2>&1; then
    if command -v zenity >/dev/null 2>&1; then
        zenity --question --title="Argus needs NanoOWL" \
          --text="jetson-containers was not found.\n\nOpen the simulator for now?" \
          --ok-label="Open simulator" --cancel-label="Close" || exit 1
        exec "$PYTHON" -m bubbles.demo
    fi
    echo "Missing jetson-containers / NanoOWL."
    exit 1
fi
echo "[$(date --iso-8601=seconds)] Starting Argus" >> "$LOG_FILE"
"$PROJECT_DIR/run_nanoowl.sh" "${1:---real}" >> "$LOG_FILE" 2>&1
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    MESSAGE="$(tail -n 12 "$LOG_FILE")"
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --title="Argus could not start" --width=560 \
          --text="Argus could not start.\n\n$MESSAGE\n\nLog: $LOG_FILE"
    else
        echo "$MESSAGE" >&2
    fi
fi
exit "$STATUS"
