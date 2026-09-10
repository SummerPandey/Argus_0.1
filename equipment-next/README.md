# Argus — Equipment Scan (General tier)

A NanoOWL-driven camera check for a surgical instrument tray, structured
the same way as `handwash-next/`: a detection-agnostic logic module you
can unit test anywhere, and a thin NanoOWL/camera loop that only runs on
a Jetson.

## Scope — read this before trusting a result

This is the **general** tier only, and it checks exactly two things:

1. **Bare-minimum tray present** — is every item in a configurable
   checklist visible in frame? Default checklist: scalpel, forceps,
   scissors, hemostat, needle holder, retractor. This is a reasonable
   starting point, **not** a clinically validated or facility-specific
   requirement — edit it via `config.json` (see below) to match an
   actual tray.
2. **Visual contamination screen** — does anything in an item's own
   detection box look bloodstained or dirty, by color alone?

What this is **not**: a sterility check, a count of every instrument on
the tray, instrument-model identification, or a certified clinical
compliance device. NanoOWL gives boxes and loose labels, not segmentation
or material analysis — the contamination check is a classical-CV color
heuristic (an HSV ratio for blood-red vs. brown/dirt tones inside each
detected box), not a verified stain or pathogen detector. Lighting,
reflections off polished steel, and camera angle will all affect it.
Treat a "READY" result as "nothing obviously wrong was seen," and a
flagged item as "a human should look at this," not as ground truth
either way.

## Privacy: nothing is persisted

Same principle as the rest of Argus: results are shown live on screen
and then gone. No CSV, no database, no network call. The only file this
module ever reads is an optional local `config.json` for calibration
(see below); it never writes one.

## Quick start

```bash
cd equipment-next
python3 nanoowl_equipment_monitor.py
```

Needs the NanoOWL/TensorRT container the same way
`handwash-next/nanoowl_monitor.py` does — see the top-level
`handwash-next/install-camera-deps.sh` for the Jetson setup this depends
on. Press `R` to reset the scan, `Q` to quit.

## Calibrating on-device

Every tunable — the checklist itself, detection threshold, presence
debounce timing, and the blood/dirt HSV ratios — lives in
`equipment_logic.TUNABLE_DEFAULTS` and is read once at startup from an
optional `config.json` in this directory (gitignored; copy
`config.example.json` to start). None of the HSV bands or ratio
thresholds have been calibrated against real footage — they're
starting-point guesses, same as the untested thresholds in
`handwash-next/nanoowl_logic.py` — so tune `config.json` against your
actual tray, lighting, and camera rather than editing the source.

## Layout

- `equipment_logic.py` — checklist matching, presence debounce, and the
  blood/dirt contamination heuristic, kept free of NanoOWL/camera
  dependencies so it's unit tested anywhere. Also owns `TUNABLE_DEFAULTS`
  and `load_config()`.
- `equipment_hud.py` — the on-screen checklist card (rounded panel, status
  dots, result pill, detection-box label chips), also kept free of the
  `nanoowl` import so its look can be rendered to an image and checked
  without a Jetson - see "Checking the look without a Jetson" below.
- `nanoowl_equipment_monitor.py` — camera capture and the scan loop's
  `main()` entry point, built on the two modules above (needs the
  NanoOWL/TensorRT container to import or run, so it isn't covered by the
  test suite).
- `config.example.json` — every tunable and its default; copy to
  `config.json` (gitignored) to calibrate on-device.
- `tests/` — `test_equipment_logic.py` and `test_equipment_hud.py`.

## Checking the look without a Jetson

`equipment_hud.py` only needs cv2/numpy, so its output can be rendered to
a plain image file and inspected without any camera or NanoOWL engine:

```python
import cv2, numpy as np
from equipment_logic import reset_scan, scan_passed
from equipment_hud import draw_checklist

frame = np.zeros((480, 640, 3), dtype=np.uint8)
scan = reset_scan()
draw_checklist(frame, scan, ready=scan_passed(scan))
cv2.imwrite("preview.png", frame)
```

Worth knowing if you touch the palette: `equipment_hud.PANEL_BG`/
`PANEL_BORDER` are deliberately **not** the same literal BGR tuples as
`handwash-next/nanoowl_monitor.py`'s. Rendering that file's tuples showed
they read as a murky olive-brown on screen, not the "dark teal-navy" its
own comment claims - an apparent red/blue channel swap that nothing could
catch there, since that module needs a Jetson/camera/NanoOWL engine just
to import. This module doesn't, so the same mistake is both visible and
avoidable here.

This is a prototype, not a certified clinical compliance device.
