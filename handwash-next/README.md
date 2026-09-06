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
  resolution, motion sensitivity, and the evidence-box sanity range below —
  lives in one place: `nanoowl_logic.TUNABLE_DEFAULTS`. `config.json` is
  gitignored on purpose, since it's a per-camera/per-sink calibration, not
  source.

Three structural changes make the water/soap/towel checkpoints hold up
against real on-device flicker instead of just the raw per-frame score:

- **Evidence boxes are sanity-checked before they're ever scored or
  drawn.** Amorphous/transparent evidence is NanoOWL's weak spot: instead
  of a tight box on the actual foam or water stream, it'll sometimes sweep
  in a big chunk of background, reflections, or wet countertop. A box
  outside `MIN_EVIDENCE_BOX_AREA_RATIO`/`MAX_EVIDENCE_BOX_AREA_RATIO` (0.15%
  to 35% of the frame by default) is dropped outright, so an oversized box
  can't dominate the screen or fake a checkpoint.
- **The wet/soap/dry confirmation timers are leaky buckets, not hard
  resets.** Previously, a single frame where the detector missed the
  water/foam/towel (a hand passing in front of the stream, one bad
  inference) reset the whole "seen continuously for N seconds" timer back
  to zero, throwing away everything accumulated. Now a miss only unwinds
  the timer by its own duration (`nanoowl_logic.advance_confirmation`), so
  brief flicker costs roughly its own length instead of all prior
  progress — while evidence that's genuinely absent for as long as it was
  present still fully resets, so this doesn't create a false memory of
  soap or water that's actually gone. The rinse checkpoint's "tap is now
  off" timer is deliberately excluded from this and still resets hard on
  any water blip, since that one is timing the tap actually being closed.
- **A hand at the faucet counts as water evidence in its own right.**
  Classifying a transparent running-water stream is exactly the kind of
  thing open-vocabulary detectors are weakest at; a hand reaching the
  solid faucet object is a far more reliable signal that the tap is being
  turned on, and NanoOWL is much better at localizing solid objects like a
  faucet than amorphous ones. `nanoowl_logic.hand_at_faucet()` checks hand/
  forearm proximity to the faucet box (`HAND_FAUCET_CONTACT_PADDING`, 40px
  by default) and this is OR'd into the water-evidence signal alongside
  the raw "running water" label — either one, or both, count. On screen,
  the faucet box switches from muted gray to the evidence amber and gets a
  connecting line to the nearest hand while this is active, labeled
  "FAUCET - HAND AT TAP", so it's visually obvious *why* water is being
  counted even on a frame where the stream itself isn't picked up.

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
inference worker, so landmark processing cannot stall the camera window. The
Real sink test currently runs NanoOWL inference synchronously in the capture
loop, so its frame rate still tracks TensorRT inference latency directly —
**this is the ceiling on how smooth it can feel**, and it's not fixed here.
Threading that inference call the way Room mode does would need a dedicated
worker thread owning NanoOWL's CUDA/TensorRT context, which is easy to get
subtly wrong (crashes, not just slowness) and impossible to verify without
the actual Jetson + camera + NanoOWL engine this repo doesn't have access to
in a plain dev environment — so it was deliberately left alone rather than
guessed at. What *was* tightened, safely, with no such risk: the camera now
requests a fixed 640×480 resolution and a 1-frame driver buffer
(`CAP_PROP_BUFFERSIZE`), so `camera.read()` always returns the newest frame
instead of one from a growing backlog, and every per-frame cost (color
conversion, PIL conversion, NanoOWL preprocessing) scales with a sane frame
size instead of whatever high-res default the camera driver picks. Startup
failures are shown in a dialog and recorded in `logs/argus.log`; the sink
test also always releases the camera and closes its window on exit, even if
NanoOWL raises mid-session.

The on-screen HUD (`draw_checklist`, `result_screen`, detection boxes) now
shares one brand palette (`PANEL_BG`/`MINT`/`CYAN`/`AMBER`/`CORAL`/... near
the top of `nanoowl_monitor.py`) matching Room hand test's and the desktop
launcher's navy/mint/cyan Argus identity, instead of the ad hoc bracket
checklist and pure red/green result screens it used before: a real ARGUS
wordmark and mode badge, status dots with checkmarks instead of `[X]`/`[ ]`
text, an actual progress bar for active-rub time plus a large "RUB TIMER"
countdown next to it that's readable at sink distance instead of only as
fine print, and detection boxes color-coded by what they represent
(hands/forearms = brand mint/cyan, soap and water and towel = one
consistent "evidence" amber, the faucet = a muted background-fixture gray)
instead of eight unrelated debug colors. Evidence boxes implausibly large
or small for the real thing (see Calibrating detection on-device below)
are dropped before they're ever drawn.

## Layout

- `bubbles/session.py` — deterministic, timestamp-driven state machine
- `bubbles/demo.py` — friendly zero-dependency simulator
- `bubbles/camera.py` — threaded camera and landmark-based rubbing estimator
- `nanoowl_monitor.py` — Real sink test: NanoOWL camera capture, drawing, and
  the WHO-guided workflow's `main()` entry point (needs the NanoOWL/TensorRT
  container to import or run)
- `nanoowl_logic.py` — the sink test's geometry, timing, checklist, and
  automatic wet/rinse/dry/faucet-close detection logic, kept free of
  NanoOWL/camera dependencies so it can be unit tested anywhere. Also owns
  `TUNABLE_DEFAULTS` and `load_config()` for on-device calibration.
- `config.example.json` — every tunable threshold and its default; copy to
  `config.json` (gitignored) to calibrate on-device
- `tests/test_session.py`, `tests/test_vision.py`, `tests/test_nanoowl_logic.py`
  — core behavioral tests

This is a prototype, not a certified clinical compliance device.
