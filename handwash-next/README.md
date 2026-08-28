# Argus — Surgical Safety Intelligence

A separate, lightweight handwashing-monitor prototype focused on smooth UI,
testable decision logic, and temporal evidence instead of single-frame guesses.

## Privacy: nothing is persisted

Argus is real-time alerts only. It does **not** write session outcomes,
detection scores, or anything else identifying a wash attempt to disk, a
database, or the network — by design, to avoid the liability of holding
records of who did or didn't wash their hands correctly. Results appear on
screen for `RESULT_DISPLAY_TIME` seconds and are then gone. The only file
either live-camera mode ever writes is `logs/argus.log`, which is a plain
startup/crash log (container failed to launch, camera not found) — it never
contains a wash outcome. The only file `nanoowl_monitor.py` ever *reads* is
an optional local `config.json` for threshold calibration (see below); it
never writes one.

## Quick start

Double-click `Argus.desktop`, or run:

```bash
./launch.sh
```

The desktop launcher opens a mode chooser:

- **Real sink test** — NanoOWL-assisted, WHO-guided 40–60 second workflow
- **Room hand test** — real camera and rubbing detection, with soap/water bypassed
- **Button simulator** — no camera required

## Live camera (optional)

```bash
./install-camera-deps.sh
./launch.sh
```

The active camera modes use the Jetson NanoOWL/TensorRT detector for hands,
forearms, and soap/foam, plus a lightweight motion region. In the real sink
test, NanoOWL also watches for running water, a towel, and a faucet, so the
whole 11-step sequence is hands-free — no keys to press. Room mode uses the
same real camera pipeline but bypasses soap and water entirely. Press `R` to
reset and `Q` to quit. Run `./launch.sh --room` for room testing or
`./launch.sh --demo` for the simulator.

In the real sink workflow, wetting is auto-confirmed once running water is
seen continuously for 3 seconds, then follow the six on-screen surface-coverage
prompts while NanoOWL observes soap/foam, contact, and motion. After 30 seconds
of active rubbing, rinsing is auto-confirmed once water reappears at the tap
and then stops again; drying is auto-confirmed once a towel is reliably seen;
and the tap-closed-with-towel step is auto-confirmed once the towel touches
the faucet. `W`/`N`/`D`/`F` still work as a **manual fallback** in case
detection misses a step on-device (bad lighting, camera angle, an
unfamiliar towel/faucet shape) — they're just not required or shown on
screen anymore for the real sink test. Room mode is practice only and
still uses the `W`/`N`/`D`/`F` keys, since it has no water/soap detection
at all. Neither mode records anything — see Privacy above.

**Calibrating detection on-device:** open-vocabulary detectors like NanoOWL
are generally much weaker at amorphous/transparent things like running water
than at solid objects like hands or a towel, and the shipped thresholds are
untested starting points, not calibrated against real footage — this
couldn't be done without a Jetson, a camera, and the NanoOWL/TensorRT
engine, none of which are available in a plain dev environment. Two things
make tuning that on your actual sink straightforward instead of a
code-editing chore:

- Press **`C`** to toggle a live calibration readout — raw detection scores
  for soap/water/towel next to their thresholds, plus the auto-detection
  state (wet timer, rinse latch, dry-frame count). It's the same
  screen-only, nothing-saved principle as everything else here; it just
  shows you the numbers instead of the pass/fail checklist.
- Copy `config.example.json` to `config.json` in this directory and edit
  whatever needs adjusting (only the keys you set are overridden; leave the
  rest out). It's read once at startup, before the camera opens, and every
  tunable — the four auto-detection thresholds, WHO timing, camera
  resolution, motion sensitivity — lives in one place:
  `nanoowl_logic.TUNABLE_DEFAULTS`. `config.json` is gitignored on purpose,
  since it's a per-camera/per-sink calibration, not source.

The guidance follows WHO's 11-step, 40–60 second soap-and-water sequence in
order — wet, soap, the six rub-technique steps, rinse, dry, tap off with the
towel — and each step is gated behind the previous one. The WHO 40–60s clock
starts at the `W` (wet) confirmation, since that's when the timed procedure
itself begins, not whenever hands first appear in frame; a session can't be
marked complete before 40 real seconds have elapsed since then, or fail for
running long before it. Room mode has no wet/soap step and never claims WHO
compliance, so its practice timer simply starts once hands are seen. The
current box detector does not verify individual hand poses or microbiological
cleanliness, so the display reports observed steps rather than claiming that
hands are sterile or that the product is WHO-certified.

In Room hand test, the display runs independently from a 15 FPS, 448-pixel
inference worker, so landmark processing cannot stall the camera window.
The Real sink test now applies the same pattern: NanoOWL inference runs on
a single dedicated worker thread (`AsyncOwlPredictor` in
`nanoowl_monitor.py`) instead of blocking the capture loop, so the camera
window and key handling stay responsive at the camera's own rate instead
of tracking TensorRT inference latency frame-for-frame. The checklist and
detection boxes lag a frame or two behind the live video when inference is
slower than capture — the same tradeoff Room mode already makes for
MediaPipe — and stale frames are dropped rather than queued, so a slow
inference cycle never builds a backlog.

This was previously left synchronous on purpose: threading NanoOWL's
`predict()` call means a worker thread that owns its CUDA/TensorRT
context, which is easy to get subtly wrong (crashes, not just slowness),
and this repo doesn't have access to the Jetson + camera + NanoOWL engine
needed to run it for real. `AsyncOwlPredictor` keeps the one rule that
matters explicit and enforced by construction — exactly one thread is ever
created, and it is the only thread that ever calls `predict()` — and the
decision logic in `nanoowl_logic.py` (the WHO timing, checkpoint
debouncing) is completely unchanged, only *when* it runs per NanoOWL
result rather than per camera frame. The threading/queue mechanics were
exercised with a stubbed-out fake predictor (confirming `predict()` is
never called from more than one thread and that frames are dropped, not
queued, under load), but actually running this against the real
NanoOWL/TensorRT engine on a Jetson has not been done here. Treat it as
unverified on real hardware until it's been run on-device — watch for
anything CUDA/TensorRT related in `logs/argus.log` or a crash instead of a
clean exit the first time you run it live, and revert to a synchronous
`predictor.predict()` call in the capture loop if so.

Separately, and with no such risk: the camera requests a fixed 640×480
resolution and a 1-frame driver buffer (`CAP_PROP_BUFFERSIZE`), so
`camera.read()` always returns the newest frame instead of one from a
growing backlog, and every per-frame cost (color conversion, PIL
conversion, NanoOWL preprocessing) scales with a sane frame size instead
of whatever high-res default the camera driver picks. Startup failures are
shown in a dialog and recorded in `logs/argus.log`; the sink test also
always releases the camera, stops the NanoOWL worker thread, and closes
its window on exit, even if NanoOWL raises mid-session.

The on-screen HUD (`draw_checklist`, `result_screen`, detection boxes) now
shares one brand palette (`PANEL_BG`/`MINT`/`CYAN`/`AMBER`/`CORAL`/... near
the top of `nanoowl_monitor.py`) matching Room hand test's and the desktop
launcher's navy/mint/cyan Argus identity, instead of the ad hoc bracket
checklist and pure red/green result screens it used before: a real ARGUS
wordmark and mode badge, status dots with checkmarks instead of `[X]`/`[ ]`
text, an actual progress bar for active-rub time, and detection boxes
color-coded by what they represent (hands/forearms = brand mint/cyan, soap
and water and towel = one consistent "evidence" amber, the faucet = a muted
background-fixture gray) instead of eight unrelated debug colors.

## Layout

- `bubbles/session.py` — deterministic, timestamp-driven state machine
- `bubbles/demo.py` — friendly zero-dependency simulator
- `bubbles/camera.py` — threaded camera and landmark-based rubbing estimator
- `nanoowl_monitor.py` — Real sink test: NanoOWL camera capture, drawing, and
  the WHO-guided workflow's `main()` entry point (needs the NanoOWL/TensorRT
  container to import or run). `AsyncOwlPredictor` in this file runs
  NanoOWL inference on its own worker thread so the capture loop and
  display aren't blocked on TensorRT latency — see the live-camera section
  above for the threading design and its unverified-on-hardware caveat.
- `nanoowl_logic.py` — the sink test's geometry, timing, checklist, and
  automatic wet/rinse/dry/faucet-close detection logic, kept free of
  NanoOWL/camera dependencies so it can be unit tested anywhere. Also owns
  `TUNABLE_DEFAULTS` and `load_config()` for on-device calibration.
- `config.example.json` — every tunable threshold and its default; copy to
  `config.json` (gitignored) to calibrate on-device
- `tests/test_session.py`, `tests/test_vision.py`, `tests/test_nanoowl_logic.py`
  — core behavioral tests

This is a prototype, not a certified clinical compliance device.
