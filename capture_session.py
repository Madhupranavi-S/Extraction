"""
capture_session.py
===================
Guided, FULLY AUTOMATIC capture tool for building the distance x pose x occlusion
x subject test set.

Workflow:
  1. Mark floor distances at 0.8 / 1.2 / 1.5 / 2.0 / 2.5 m (or edit DISTANCES_M below).
  2. Run this script. For every (distance, pose) combination it:
       - shows a countdown on the live preview so the subject can get into position
       - waits for the scene to stabilize (reuses your stability-wait / multi-frame
         -average logic so capture conditions match deployment)
       - automatically fires a burst of N_SHOTS_PER_COMBO frames, back-to-back
       - moves on to the next combination -- no key press required at all
  3. Press ESC at any time to abort the whole session early.
  4. Each capture saves:
       - <image_id>.png        -> IronRed BGR palette image (what detectors consume)
       - <image_id>_y16.npy    -> raw y16 thermal frame (for ground-truth temp checks later)
     and appends one row to metadata.csv with the known distance/pose/subject
     labels (everything except the actual canthus pixel coordinates, which you fill
     in during annotation).
  5. Since capture is automatic, you'll get some bad/blurred/mid-motion frames on
     purpose -- review the folder afterward and delete whatever you don't want to
     keep in the test set.

Run:
    python capture_session.py --port /dev/ttyACM0 --subject alice --out ./captures
"""
import sys
import os
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import deque

sys.path.insert(0, 'build/python')
import guideusb2 as g
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Capture matrix
# ---------------------------------------------------------------------------
DISTANCES_M = [0.8, 1.2, 1.5, 2.0, 2.5]

# 9 poses: straight-on, two yaw levels each side, and four occlusion+turn combos.
# "occlusion" describes which side of the face/canthus is physically blocked
# (hand, hair, mask edge, etc). "turn" describes head yaw at the same time --
# these combos are deliberately the harder cases (occluded side near vs far
# from camera depending on turn direction).
POSES = [
    "straight",
    "slight_left",
    "near_left",          # sharp/near turn left
    "slight_right",
    "near_right",          # sharp/near turn right
    "left_occluded_turn_right",
    "right_occluded_turn_left",
    "left_occluded_turn_left",
    "right_occluded_turn_right",
]

N_SHOTS_PER_COMBO = 3      # automatic burst size per (distance, pose)
COUNTDOWN_SECONDS = 4      # time given to get into position before capture starts

STABILITY_WARMUP = 1.5
STABILITY_THRESH = 0.5
STABILITY_CHECKS = 5
STABILITY_TIMEOUT = 8.0
N_AVG = 10

INTER_SHOT_DELAY = 0.4     # small pause between shots within a burst

linear_cal = g.CameraLinearCal()
last_status = [None]
last_frame = [None]
abort_flag = [False]


def on_frame(frame):
    last_frame[0] = frame
    if last_status[0] is not None:
        try:
            linear_cal.fit(frame.y16, last_status[0])
        except Exception:
            pass


def make_preview(label_lines):
    """Render current live frame with overlay text lines."""
    preview = cv2.cvtColor(
        g.apply_palette(last_frame[0].y16, g.Palette.IronRed,
                         g.PaletteOptions(auto_range=True)),
        cv2.COLOR_RGB2BGR)
    y = 25
    for line in label_lines:
        cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        y += 30
    return preview


def pump_preview(label_lines, duration):
    """Show preview + process key events for `duration` seconds without blocking
    capture logic. Returns False if user aborted (ESC)."""
    t0 = time.time()
    while time.time() - t0 < duration:
        if last_frame[0] is not None and linear_cal.ready():
            cv2.imshow("Capture", make_preview(label_lines))
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            abort_flag[0] = True
            return False
        time.sleep(0.01)
    return True


def wait_stable(timeout=STABILITY_TIMEOUT):
    time.sleep(STABILITY_WARMUP)
    recent = deque(maxlen=STABILITY_CHECKS)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if last_frame[0] is not None and linear_cal.ready():
            temps = linear_cal.decode(last_frame[0].y16)
            recent.append(float(temps.max()))
            if len(recent) == STABILITY_CHECKS and (max(recent) - min(recent)) < STABILITY_THRESH:
                return True
        time.sleep(0.2)
    return False


def capture_frame():
    frozen = last_frame[0]
    if frozen is None or not linear_cal.ready():
        return None, None, None
    decoded = []
    for _ in range(N_AVG):
        try:
            decoded.append(linear_cal.decode(frozen.y16))
        except Exception:
            pass
    if not decoded:
        return None, None, None
    temps_avg = np.mean(decoded, axis=0)
    rgb = g.apply_palette(frozen.y16, g.Palette.IronRed, g.PaletteOptions(auto_range=True))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return frozen, temps_avg, bgr


def init_metadata(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow([
                "image_id", "image_path", "y16_path",
                "subject_id", "distance_m", "pose",
                "shot_index", "captured_at"
            ])


def append_metadata(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--subject", required=True, help="Subject ID label, e.g. 'alice'")
    ap.add_argument("--out", default="./captures")
    ap.add_argument("--shots", type=int, default=N_SHOTS_PER_COMBO,
                     help="Shots per (distance, pose) combo (default: %(default)s)")
    ap.add_argument("--countdown", type=int, default=COUNTDOWN_SECONDS,
                     help="Seconds to get into position before each combo (default: %(default)s)")
    args = ap.parse_args()

    shots_per_combo = args.shots
    countdown_s = args.countdown

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.csv"
    init_metadata(meta_path)

    cam = g.Camera(g.DeviceInfo(serial_port=args.port))
    cam.configure_thermography(g.ThermographyConfig(emissivity=98))
    cam.start(on_frame)
    print("Warming up 5s...")
    time.sleep(5)

    print("Calibrating...")
    for _ in range(20):
        try:
            s = cam.serial().query_temp_status(retries=2, wait_seconds=1.5)
            if s:
                last_status[0] = s
                if last_frame[0] is not None:
                    linear_cal.fit(last_frame[0].y16, s)
                break
        except Exception:
            pass
        time.sleep(1)

    if not linear_cal.ready():
        print("Calibration failed.")
        cam.stop()
        sys.exit(1)

    cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
    shot_count = 0
    total_combos = len(DISTANCES_M) * len(POSES)
    combo_idx = 0

    print(f"\nStarting AUTOMATIC capture: {len(DISTANCES_M)} distances x "
          f"{len(POSES)} poses x {shots_per_combo} shots = "
          f"{total_combos * shots_per_combo} frames total.")
    print("Press ESC at any time to abort.\n")

    for distance in DISTANCES_M:
        if abort_flag[0]:
            break
        for pose in POSES:
            combo_idx += 1
            if abort_flag[0]:
                break

            print(f"\n=== [{combo_idx}/{total_combos}] Subject:{args.subject}  "
                  f"Distance:{distance}m  Pose:{pose} ===")

            # --- countdown so the subject can get into position ---
            for remaining in range(countdown_s, 0, -1):
                if last_frame[0] is None or not linear_cal.ready():
                    time.sleep(0.1)
                    continue
                label = [
                    f"{args.subject} | {distance}m | {pose}",
                    f"GET READY - capturing in {remaining}s",
                ]
                if not pump_preview(label, 1.0):
                    break
            if abort_flag[0]:
                break

            # --- wait for stable scene ---
            print("Waiting for stable scene...")
            if last_frame[0] is not None and linear_cal.ready():
                pump_preview([f"{args.subject} | {distance}m | {pose}",
                              "Stabilizing..."], 0.1)
            wait_stable()

            # --- automatic burst, no key press ---
            for shot_idx in range(1, shots_per_combo + 1):
                if abort_flag[0]:
                    break
                label = [
                    f"{args.subject} | {distance}m | {pose}",
                    f"CAPTURING shot {shot_idx}/{shots_per_combo}",
                ]
                pump_preview(label, 0.05)

                frozen, temps_avg, bgr = capture_frame()
                if frozen is None:
                    print(f"  Shot {shot_idx}: capture failed, skipping.")
                    continue

                shot_count += 1
                image_id = f"{args.subject}_{distance}m_{pose}_{shot_idx:02d}_{shot_count:04d}"
                img_path = out_dir / f"{image_id}.png"
                y16_path = out_dir / f"{image_id}_y16.npy"
                cv2.imwrite(str(img_path), bgr)
                np.save(str(y16_path), frozen.y16)

                append_metadata(meta_path, [
                    image_id, str(img_path), str(y16_path),
                    args.subject, distance, pose,
                    shot_idx, datetime.now().isoformat()
                ])
                print(f"  Saved {image_id}")
                time.sleep(INTER_SHOT_DELAY)

    cam.stop()
    cv2.destroyAllWindows()

    if abort_flag[0]:
        print(f"\nAborted by user. Captured {shot_count} frames before stopping.")
    else:
        print(f"\nDone. Captured {shot_count} frames total. Metadata: {meta_path}")
    print("Review the output folder and delete any unwanted/blurred/mid-motion "
          "frames before annotating.")
    print("Next step: run annotate_tool.html on the images in this folder.")


if __name__ == "__main__":
    main()
