# Argus handwashing rebuild

Updated: 2026-09-23. Status: requirements and training preparation; no new model has been trained.

Build a new application and task-specific distilled model for routine WHO soap-and-water handwashing. Recognize the action actually being performed, assess visible technique and both-hand coverage, accumulate valid duration, identify omitted or visibly incorrect actions, and guide the next action. Existing Argus code is a reference for user experience and failure cases, not evidence of technique recognition.

## Confirmed inputs and remaining hardware checks

- User reports a "Jetson nano (super jetson nano)." Working deployment assumption: Jetson Orin Nano Super Developer Kit. NVIDIA lists this kit with 8 GB memory; verify the physical device, installed JetPack, power mode, and free memory before selecting the student size. Do not assume the original Jetson Nano has these capabilities.
- Camera: Logitech C270 HD, advertised at 1280 x 720 / 30 fps. Actual capture rate, hand detail, lighting, and camera distance must be measured on the installation. Camera frame rate is not a promise of inference speed.
- Training and larger teacher inference: separate workstation or GPU service; available GPU and budget remain unknown. Do not configure a paid service or upload private footage without the relevant authorization.
- Scope: one routine soap-and-water workflow; equipment, surgery, alcohol handrub, and surgical scrub are outside the first release.
- Testing logs are allowed. Live camera recordings and personal information are not to be retained.

Hardware sources: [NVIDIA kit specifications](https://www.nvidia.com/en-sg/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), [Logitech C270](https://www.logitech.com/en-us/shop/p/c270-hd-webcam).

## Procedure reference

The supplied `WHO_hand_wash_guidelines.pdf` is *WHO Guidelines on Hand Hygiene in Health Care* (2009), ISBN 978 92 4 159790 6. Relevant sections were read and visually inspected:

- Part II, section 2, printed page 152 (PDF page 160): hand hygiene technique.
- Figure II.2, printed page 156 (PDF page 164): soap-and-water handwashing sequence.
- Local source: `/Users/binayakgurubacharya/Desktop/Argus/WHO_hand_wash_guidelines.pdf`.
- Public equivalent diagram: [WHO handwashing poster](https://www.who.int/docs/default-source/patient-safety/how-to-handwash-poster.pdf).
- User-supplied video: [Hand-washing Steps Using the WHO Technique](https://www.youtube.com/watch?v=IisgnbMfKvI), published by Johns Hopkins Medicine. It is a hospital demonstration of WHO technique, not a WHO-channel upload. Publisher and description were inspected; full visual annotation is pending.

Use these labels, retaining the WHO figure numbering:

| Figure step | Action label | Evidence to annotate |
| --- | --- | --- |
| 0 | `wet_hands` | Hands in visible water stream over time |
| 1 | `apply_soap` | Observed application/dispensing and receiving hand; lather as supporting evidence |
| 2 | `rub_palms` | Palm contact and rubbing motion |
| 3 | `rub_dorsum` | Palm over opposite hand back with fingers interlaced; both sides |
| 4 | `rub_interlaced_palms` | Palms together with fingers interlaced and moving |
| 5 | `rub_finger_backs` | Finger backs against opposing palm with fingers interlocked; coverage of both hands |
| 6 | `rub_thumb` | Rotational thumb rubbing, separately for each thumb |
| 7 | `rub_fingertips` | Rotational/back-and-forth fingertip rubbing in opposite palm; both hands |
| 8 | `rinse_hands` | Hands in visible water after rubbing; annotate visible action duration |
| 9 | `dry_hands` | Towel-hand interaction and drying motion |
| 10 | `close_tap_with_towel` | Towel-mediated handle interaction with supporting before/after state evidence |

Figure step 11 is the concluding state, not another action class. Also annotate `idle`, `transition`, `other_action`, and `insufficient_visibility`. A visible action can still have a technique error; use separate error labels rather than forcing every mistake into a different action class.

WHO gives 40-60 seconds for the entire procedure. The figure does not prescribe five seconds for each rubbing technique. Per-technique minimum duration and repetition requirements need a reviewed rubric; do not inherit the old 30-second total rubbing threshold as a WHO requirement. Do not automatically equate a session exceeding 60 seconds with incorrect technique.

CDC community demonstrations can supply useful object/action examples, but their five-step guidance and at-least-20-second scrubbing advice are not complete annotations of this WHO technique sequence. Missing footage in an edited demonstration is not a demonstrated skipped action. Source: [CDC handwashing guidance](https://www.cdc.gov/handwashing).

## Staged model and Python design

### 1. Objects and visibility

Start with a shared pretrained detector adapted to separate object classes: `hand`, `soap_dispenser`, `soap_bar` when relevant, `faucet`, `tap_handle`, `sink`, and `towel`. Evaluate foam as a separate appearance task using a region classifier or segmentation if bounding boxes are a poor fit. Evaluate visible water flow separately. These are distinct labels/tasks, not necessarily one neural network per object.

Add hand landmarks or other fine hand features for the later technique recognizer. Preserve hand identity across motion when possible; screen-left is not anatomical left. Mark sidedness unknown when occlusion prevents reliable assignment. Calibrate static fixture regions for the actual sink so the system need not repeatedly rediscover fixed geometry.

Train/evaluate on hands touching, interlaced fingers, soap-covered hands, wet hands, partial occlusion, towels, and background movement. A detector working on separated dry hands is insufficient.

### 2. Relations and events

Calculate normalized overlap, proximity, motion, visibility, and persistence in Python. Box overlap itself has no training target: it is geometry. A learned interaction classifier can later consume these features together with video features when reviewed event examples justify it.

| Signal | Permitted interpretation | Additional evidence for the target event |
| --- | --- | --- |
| Hand and handle boxes overlap | Candidate handle interaction | Visible flow change or a validated, sink-specific handle state before/after |
| Hand and dispenser overlap | Candidate dispensing interaction | Dispensing motion/output or reviewed application evidence; dispenser may be empty |
| Hands overlap each other | Candidate contact/occlusion | Hand appearance, pose, and motion over time to distinguish techniques |
| Hands overlap a water region | Candidate wetting/rinsing | Flow evidence, persistence, and procedure context |
| Towel overlaps hand | Candidate towel interaction | Drying motion over time; mere towel presence is not drying |
| Towel overlaps faucet | Candidate towel-mediated interaction | Handle localization, before/after state, and evidence that the towel mediates the interaction |

Maintain tap state as `unknown`, `flowing`, or `not_flowing_observed`. No visible water while the spout is blocked means unknown. Flow stopping is not by itself proof of manual handle closure, particularly on sensor-operated taps. Record handle-closure evidence separately. Determine fixture type during calibration.

Do not equate foam detection with soap application, absent foam with absent soap, or a visible towel with proof it is single-use. Use `unconfirmed` when an event cannot be established visually.

### 3. Temporal technique recognition and coaching

Train a video model on short sequences of hand crops, optionally combined with landmarks and scene/event features. It must recognize actions independently of the expected next step. The Python procedure tracker then accumulates observed coverage, issues prompts, and handles repeats, omissions, interruptions, and returns to previous steps.

Credit left/right coverage separately for asymmetric techniques. Do not advance a technique merely because its prompt was displayed for a time interval. Pause evidence accumulation during inadequate visibility; do not interpret an occluded action as a proven error. Define both the evidence threshold for a prompt and a separate, stricter threshold for completion.

### 4. Teacher and student

Initial research candidates remain provisional:

- Qwen3-VL-32B-Instruct: offline proposal of timestamps, action labels, and visible errors; compare with human annotations before scaling automatic labeling. [Model documentation](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- V-JEPA 2.1 video backbone with an Argus classification/temporal head: candidate larger technique teacher, adapted to reviewed handwashing clips. It is not already a handwashing classifier. [Official repository](https://github.com/facebookresearch/vjepa2)
- Small student: select a pretrained visual backbone and temporal head after the target device and pilot benchmark are established. Train task-specific weights using reviewed labels and, after teacher validation, teacher probabilities/features.

Teacher-assisted hard labeling and knowledge distillation are separate experiments. Record which supervision each run uses. Compare the student with and without distillation on the same held-out data. Train and evaluate the deployed student on causal windows that contain only frames available at prediction time. The Jetson runs the compact deployed pipeline; do not plan to run the large labeling teacher on it.

## Dataset preparation

Aim for about 15 distinct, usable WHO/CDC demonstrations, plus the user's Johns Hopkins reference. The number is a curation target, not a confirmed inventory or evidence of sufficient training data. Initial candidates are in `data/sources/handwashing-videos.json`; none are approved for training yet.

- Inspect each candidate for actual hand detail, full action visibility, edits, playback speed, fixture coverage, and source identity. Track reuse status and acquisition route. Source reputation does not establish dataset suitability or training permission.
- Group reposts, translations, crops, and alternate edits of the same performance together. They are not independent demonstrations.
- Use native capture timestamps for durations; do not learn duration from edited or sped-up instructional clips. Mark cuts and unknown intervals. Avoid learning technique names from on-screen text or subtitles.
- Split by original performance, participant, and recording session before extracting frames/clips. Keep derived versions and augmentations in the same split. Reserve reviewed real-camera sessions for the final deployment evaluation.
- Annotate boxes/masks for detector training; start/end times, action, side coverage, visibility, and errors for temporal training. Whole-video correct/incorrect/incomplete labels alone are not sufficient.
- Use initial positive public demonstrations for bootstrapping recognition. Incorrect, incomplete, idle, and confusing examples are required before claiming error-detection performance.
- Later doctor/student recordings should include both correct and scripted incorrect attempts in both groups. Vary performers, viewpoint, lighting, soap appearance, and sink setup while preserving the intended camera view.
- Before collecting volunteer footage, resolve separate training-data retention and access. The confirmed no-recording policy applies to live Argus operation; permission to retain volunteer training recordings remains to be clarified.

## Logging

Use a strict metadata allowlist: model/config version, random per-attempt ID without a person mapping, relative elapsed time, predicted class, confidence, credited duration/side, visibility, event status, inference latency, and error codes. No images, video, audio, names, facial embeddings, persistent person identifiers, source-file metadata identifying a volunteer, or serialized landmark trajectories. Avoid wall-clock timestamps where relative time is sufficient. Configure a short retention period for test logs; duration is still to be selected.

Live frames may occupy a bounded in-memory inference buffer, then be discarded. Disable frame dumps, recording, and microphone capture. Dataset tools and deployment logging must have separate paths and explicit behavior.

## Execution checklist

- [x] Confirm first-release scope and reported camera.
- [x] Locate and inspect the relevant WHO guideline pages.
- [x] Verify publisher of the user-provided video.
- [x] Create staged build plan and initial source inventory.
- [ ] Verify exact Jetson identity/software, available training GPU, and camera view.
- [ ] Review source footage; acquire usable originals through supported routes and record reuse status.
- [ ] Establish the reviewed labeling rubric, per-technique timing policy, and independent data splits.
- [ ] Annotate a pilot batch and measure teacher-assisted labeling errors.
- [ ] Build the new dataset loader and detector training/evaluation pipeline.
- [ ] Train/evaluate object detection, foam evidence, visible water, and hand visibility separately.
- [ ] Implement relation features and temporal event rules, with tests for misleading overlaps and missing observations.
- [ ] Train/evaluate the larger temporal technique teacher.
- [ ] Train the compact student and compare with/without distillation.
- [ ] Integrate current-action display, coverage tracking, next-step guidance, and metadata-only logs.
- [ ] Export and measure accuracy, memory, end-to-end latency, and thermally sustained performance on the Jetson/C270 setup.

Measure per-object precision/recall; per-technique precision/recall and segment timing; both-hand coverage errors; soap false positives and misses; incorrectly completed attempts; missed scripted errors; unnecessary prompts; visibility-related abstention; and end-to-end response time. Set acceptance thresholds with the reviewed pilot, not generic detector accuracy or advertised TOPS.

Training has not started: source footage still needs acquisition and review, annotations do not exist, and training compute has not been identified. There is no measured new-model accuracy to report.
