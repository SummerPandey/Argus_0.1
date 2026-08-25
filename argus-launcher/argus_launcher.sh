#!/bin/bash

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! python3 -c "import tkinter" >/dev/null 2>&1; then
    gnome-terminal -- bash -lc \
        "echo 'The Argus menu needs Python Tkinter.'; echo 'Run: sudo apt install -y python3-tk'; read -p 'Press Enter to close...'"
    exit 1
fi

exec python3 "$INSTALL_DIR/argus_menu.py"
