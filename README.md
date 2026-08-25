# Argus — Surgical Safety Intelligence

Hand-hygiene monitoring for a real sink, built for a Jetson with a USB camera.
This repo has two generations of the project — **use `handwash-next/`**; the
top-level scripts are the earlier prototype, kept for reference.

## Start here

```bash
./argus-launcher/argus_launcher.sh
```

Opens the desktop dashboard, which launches into `handwash-next/`'s mode
chooser (Real sink test, Room hand test, or the no-camera simulator). See
[`handwash-next/README.md`](handwash-next/README.md) for the full picture:
architecture, how the WHO 11-step sequence is enforced and auto-detected,
live-camera setup, and the test suite.

## Layout

- **`handwash-next/`** — the current, actively developed version. Clean
  `bubbles/` package (state machine, MediaPipe vision, threaded camera) for
  Room hand test and the simulator; `nanoowl_monitor.py` + `nanoowl_logic.py`
  for the NanoOWL-driven Real sink test. Has the test suite.
- **`argus-launcher/`** — the Tkinter desktop dashboard (`argus_menu.py`)
  that launches `handwash-next/launch.sh`. Also shows placeholder modules
  for planned features (instrument tracking, sterile-field monitoring) that
  aren't implemented yet.
- **`automatic_rubbing.py`**, **`handwash_test.py`**, **`run_handwash.sh`** —
  the original prototype: a single-file NanoOWL script and an even earlier
  camera-less state machine. Superseded by `handwash-next/nanoowl_monitor.py`
  and `handwash-next/bubbles/session.py` respectively, which have real test
  coverage and the fixes made since. Kept for reference, not for new work.

## Requirements

A Jetson with `jetson-containers` and the `dustynv/nanoowl:r36.3.0` image for
live camera modes (see `handwash-next/install-camera-deps.sh`); everything
else (the simulator, the test suite) runs with plain Python 3.

This is a prototype, not a certified clinical compliance device.
