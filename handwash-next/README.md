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
matters enforced by construction: its single worker thread runs the
engine load and text encoding itself and is then the only thread that
ever calls `predict()`, so the CUDA/TensorRT context is created and used
by one thread for its entire life (as a side effect, the camera now opens
while the engine loads instead of after). Submitted frames are copied, so
the HUD the capture loop draws in place never contaminates what NanoOWL
or the motion detector sees. And a worker failure is never silent: an
exception while loading or predicting lands on `.error` with its
traceback printed for `logs/argus.log`, the capture loop checks it every
frame, and the session ends with that message instead of freezing on a
stale result. The same error surfacing was added to Room hand test's
camera and MediaPipe workers in `bubbles/camera.py`. The decision logic
in `nanoowl_logic.py` (WHO timing, checkpoint debouncing) is completely
unchanged — only *when* it runs, per NanoOWL result rather than per
camera frame.

All of that is covered by tests that run anywhere:
`tests/test_async_owl.py` pins the worker mechanics against a fake
predictor (one caller thread ever, loading included; frames dropped, not
queued, under load; frame isolation; failures surfacing on `.error`;
prompt shutdown), `tests/test_camera_threads.py` covers the
`bubbles/camera.py` worker failure paths, and `tests/test_monitor_loop.py`
drives the real `main()` loop headlessly end to end — session arms,
R resets, Q quits, a mid-session inference failure ends the session with
a message, a load failure exits with the reason. What is *not* covered is
the only thing that can't be off-device: the real TensorRT engine. Treat
it as unverified on real hardware until it's been run on a Jetson — watch
for anything CUDA/TensorRT related in `logs/argus.log` the first time you
run it live, and revert to a synchronous `predictor.predict()` call in
the capture loop if so.

Separately, and with no such risk: both camera paths request MJPG
(`CAP_PROP_FOURCC`) before negotiating resolution, since many USB cameras
cap at 30 FPS in their uncompressed default format but reach 60 in MJPG —
raising the ceiling the now-decoupled display loop can actually hit, and
harmlessly ignored by cameras that don't support it. The camera also
requests a fixed 640×480 resolution and a 1-frame driver buffer
(`CAP_PROP_BUFFERSIZE`), so `camera.read()` always returns the newest
frame instead of one from a growing backlog, and every per-frame cost
(color conversion, PIL conversion, NanoOWL preprocessing) scales with a
sane frame size instead of whatever high-res default the camera driver
picks. On the inference side, frames are handed to NanoOWL at most
`MAX_INFERENCE_FPS` times per second (default 15, tunable in
`config.json`) and not at all while a result screen is up — the checklist
logic accumulates wall-clock time, so checking faster buys no
responsiveness and only heats the Jetson toward thermal throttling, which
is what actually lowers the sustained rate. The sink test's prompt is
also stage-aware: OWL-ViT's decode work scales with the number of
queries, and no session needs soap and a towel at the same time, so
NanoOWL watches for soap+water until the 30-second rub is done and then
swaps to water+towel+faucet for rinse/dry/tap-off (both prompt trees are
encoded once at startup; each result carries the tree it was decoded
against, so a stage switch can never mislabel an in-flight result). And
the CPU-side preprocessing (BGR→RGB/PIL) happens on the capture thread at
submit time rather than on the inference thread, so prep of the next
frame overlaps GPU inference of the current one.

None of that outweighs the board itself: a Jetson in a default power mode
with governed clocks routinely gives up 30–50% of sustained TensorRT
throughput before any code runs. `./jetson-performance.sh` reports the
current power mode, this device's own mode table, and temperatures;
`sudo ./jetson-performance.sh --max` switches to the device's MAXN-class
mode (never a hardcoded id — the right id differs per Jetson model, and
on boards with no MAXN it lists the modes for an explicit
`--set ID` instead) and locks clocks with `jetson_clocks`. The power mode
persists across reboots; the clock lock does not, so re-run it after
booting. `run_nanoowl.sh` prints a one-line hint at launch if a faster
mode is available, and stays silent when the board is already at its best
(or isn't a Jetson at all). Startup failures are
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
- `jetson-performance.sh` — report/set the Jetson's power mode and lock
  clocks for maximum sustained inference; safe no-op off-Jetson
- `tests/test_session.py`, `tests/test_vision.py`, `tests/test_nanoowl_logic.py`
  — core behavioral tests
- `tests/test_async_owl.py`, `tests/test_camera_threads.py`,
  `tests/test_monitor_loop.py` — threading mechanics, worker failure
  propagation, and a headless end-to-end drive of the sink test's main
  loop (NanoOWL stubbed via `tests/nanoowl_test_stubs.py`)
- `tests/test_jetson_performance.py` — jetson-performance.sh behavior
  against fake nvpmodel/jetson_clocks binaries

This is a prototype, not a certified clinical compliance device.
