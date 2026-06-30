#!/usr/bin/env python3
"""
run_detection_sweep.py
========================
Ad-hoc "did it detect or not" sweep over a folder of images -- NO ground
truth / annotation required. Built for exactly this: you've got a folder of
images you suspect are hard cases, and you want to know which model(s)
catch them and which don't, with visual proof, without hand-labeling
anything first.

For every image in --images-dir, for every active model, this:
  - runs detect()
  - prints one line: detected BOTH sides / LEFT only / RIGHT only /
    FACE only (no canthus) / NONE, with confidences
  - if at least one canthus point was found, saves an annotated copy to
    <out>/<model_name>/annotated/
  - if NO canthus point was found but the model's stage-1 face detector
    DID find a face (only twostage_fixedzoom / twostage_adaptivezoom
    expose this -- they report `face_box` separately from the canthus
    points), saves an annotated copy to
    <out>/<model_name>/face_only_no_canthus/ instead. This isolates "the
    face detector is fine, the canthus model on the cropped face is what's
    failing" as its own visually-browsable bucket, distinct from a true
    total miss where you can't tell what went wrong. This also covers
    twostage_adaptivezoom's "too_far" case (face found but judged too
    small to trust, so stage 2 was deliberately skipped) -- those land
    here too, flagged with a "[flagged too far]" note and a `too_far`
    column in the CSV, so you can tell that subcase apart from a genuine
    stage-2 failure.
  - logs EVERY image's outcome (hit, face-only, or true miss) to one
    combined <out>/summary.csv regardless of whether anything was saved

FILENAMES ARE NOT ASSUMED TO FOLLOW ANY PATTERN. Every file with a common
image extension in --images-dir is picked up regardless of naming scheme;
the original filename stem is preserved in each output filename for
traceability back to the source image.

USAGE
-----
# Run every model in MODEL_REGISTRY against every image in the folder:
python run_detection_sweep.py --images-dir ./hard_cases --out ./sweep_out

# Just check a couple of models:
python run_detection_sweep.py --images-dir ./hard_cases --out ./sweep_out \\
    --models twostage_adaptivezoom,tfake_landmark

WEIGHTS: edit MODEL_REGISTRY below -- it's the same shape as the one in
eval_harness.py. If you've already filled that one in, copy the `weights`
values across so the two files agree.

NOTE ON tfake_landmark: with no paired <stem>_temps.npy next to an image,
this model falls back to its grayscale-from-BGR approximation (see that
detector's own module docstring) rather than the real decoded-temperature
input it expects. If a `<image_stem>_temps.npy` file happens to sit next
to an image in --images-dir, it'll be picked up and used automatically --
otherwise tfake's results on this sweep carry the same caveat flagged
earlier in the project.
"""

import argparse
import csv
import importlib
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import cv2

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# =============================================================
# MODEL REGISTRY -- keep in sync with eval_harness.py's copy
# =============================================================
MODEL_REGISTRY = {
    "singlestage_sahi_v1": {
        "module": "detector_singlestage_sahi_v1",
        "weights": "weights/detect.pt",  # <-- EDIT ME
    },
    "singlestage_sahi_v2_cascade": {
        "module": "detector_singlestage_sahi_v2_cascade",
        "weights": "weights/detect.pt",  # <-- EDIT ME
    },
    "twostage_fixedzoom": {
        "module": "detector_twostage_fixedzoom",
        "weights": {
            "face": "weights/thermal_detector.pt",  # <-- EDIT ME
            "canthus": "weights/detect.pt",          # <-- EDIT ME
        },
    },
    "twostage_adaptivezoom": {
        "module": "detector_twostage_adaptivezoom",
        "weights": {
            "face": "weights/thermal_detector.pt",  # <-- EDIT ME
            "canthus": "weights/detect.pt",          # <-- EDIT ME
        },
    },
    "tfake_landmark": {
        "module": "detector_tfake_landmark",
        "weights": None,
    },
}
ACTIVE_MODELS = list(MODEL_REGISTRY.keys())

# Drawing
COLOR_POINT = (0, 255, 255)    # cyan -- matches the convention your own live scripts used
COLOR_FACEBOX = (255, 200, 0)
COLOR_TOOFAR = (0, 140, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def find_images(images_dir: Path):
    """Every file with a recognized image extension, any naming scheme, sorted for stable output order."""
    return sorted(p for p in images_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def setup_models(detectors_dir: Path, active_models):
    sys.path.insert(0, str(detectors_dir))
    loaded = {}
    for name in active_models:
        if name not in MODEL_REGISTRY:
            print(f"WARNING: '{name}' not in MODEL_REGISTRY -- skipping")
            continue
        cfg = MODEL_REGISTRY[name]
        print(f"Loading {name} ({cfg['module']}) ...")
        try:
            mod = importlib.import_module(cfg["module"])
            model_obj = mod.load_models(cfg.get("weights"))
        except Exception as e:
            print(f"  FAILED to load {name}: {e} -- skipping this model for the run")
            continue
        loaded[name] = (mod, model_obj)
        print("  OK")
    return loaded


def call_detect(mod, model_obj, bgr, temps_c):
    """Pass temps_c through only if this detector's detect() accepts it (only tfake does)."""
    sig = inspect.signature(mod.detect)
    kwargs = {}
    if temps_c is not None and "temps_c" in sig.parameters:
        kwargs["temps_c"] = temps_c
    t0 = time.perf_counter()
    result = mod.detect(model_obj, bgr, **kwargs)
    dt = time.perf_counter() - t0
    return result, dt


def classify_outcome(result: dict) -> str:
    """
    "both" / "left_only" / "right_only": at least one canthus point found.

    "face_only": NEW. Only meaningful for the two two-stage detectors,
    which expose `face_box` in their result dict -- this means stage 1
    (the face detector) found something, but stage 2 (the canthus model,
    run on the cropped/zoomed face region) came back empty. That's a
    different, more informative failure than "nothing was found at all":
    it isolates stage 2 as the specific point of failure rather than
    leaving you guessing whether there was even a face in frame. This
    also naturally covers twostage_adaptivezoom's "too_far" case (a face
    was found but judged too small to trust, so stage 2 was deliberately
    skipped) -- still face_box-present + no canthus point, just for an
    explainable reason flagged separately via the `too_far` key.

    "none": genuinely nothing -- either no face_box key was returned at
    all (singlestage_* and tfake_landmark don't expose this concept, so
    we can't subdivide their misses this way) or a two-stage model's
    stage 1 itself found nothing.
    """
    has_left = result.get("left_pt") is not None
    has_right = result.get("right_pt") is not None
    if has_left and has_right:
        return "both"
    if has_left:
        return "left_only"
    if has_right:
        return "right_only"
    if result.get("face_box") is not None:
        return "face_only"
    return "none"


def draw_annotation(bgr, result: dict):
    """Draw whatever this detector's result dict actually contains -- works
    across all five wrapper interfaces since extra keys (face_box, too_far)
    are read defensively and simply skipped if absent."""
    img = bgr.copy()

    face_box = result.get("face_box")
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_FACEBOX, 1)

    for pt, conf, label in (
        (result.get("left_pt"), result.get("left_conf", 0.0), "L"),
        (result.get("right_pt"), result.get("right_conf", 0.0), "R"),
    ):
        if pt is None:
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img, (x, y), 8, COLOR_POINT, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x + 10, y - 8),
                    FONT, 0.5, COLOR_POINT, 1, cv2.LINE_AA)

    if result.get("too_far"):
        cv2.putText(img, "TOO FAR (flagged by model)", (15, 30),
                    FONT, 0.7, COLOR_TOOFAR, 2, cv2.LINE_AA)

    return img


def maybe_load_temps(image_path: Path):
    """Optional sibling <stem>_temps.npy -- used only by tfake_landmark, harmless if absent."""
    p = image_path.with_name(image_path.stem + "_temps.npy")
    if p.exists():
        return np.load(str(p))
    return None


def main():
    ap = argparse.ArgumentParser(description="No-ground-truth detection sweep over a folder of images")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", default="sweep_out")
    ap.add_argument("--models", default=None,
                     help="Comma-separated subset of ACTIVE_MODELS (default: all)")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)
    detectors_dir = Path(__file__).parent / "detectors"

    active = args.models.split(",") if args.models else ACTIVE_MODELS

    images = find_images(images_dir)
    if not images:
        print(f"No images found in {images_dir} (looked for {IMAGE_EXTS})")
        sys.exit(1)
    print(f"Found {len(images)} images in {images_dir}\n")

    models = setup_models(detectors_dir, active)
    if not models:
        print("No models loaded -- nothing to run.")
        sys.exit(1)
    print(f"\nActive models: {list(models.keys())}\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in models:
        (out_dir / name / "annotated").mkdir(parents=True, exist_ok=True)
        (out_dir / name / "face_only_no_canthus").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    counts = {name: {"both": 0, "left_only": 0, "right_only": 0, "face_only": 0, "none": 0,
                      "errors": 0, "unreadable": 0}
              for name in models}

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"SKIP (unreadable): {img_path.name}")
            for name in models:
                counts[name]["unreadable"] += 1
                summary_rows.append({"model": name, "image": img_path.name, "outcome": "unreadable",
                                      "left_conf": "", "right_conf": "", "too_far": "",
                                      "latency_s": "", "annotated_path": ""})
            continue

        temps_c = maybe_load_temps(img_path)

        for name, (mod, model_obj) in models.items():
            try:
                result, dt = call_detect(mod, model_obj, bgr, temps_c)
            except Exception as e:
                print(f"[{name}] {img_path.name} -> ERROR: {e}")
                counts[name]["errors"] += 1
                summary_rows.append({"model": name, "image": img_path.name, "outcome": "error",
                                      "left_conf": "", "right_conf": "", "too_far": "",
                                      "latency_s": "", "annotated_path": ""})
                continue

            outcome = classify_outcome(result)
            counts[name][outcome] += 1
            lc = result.get("left_conf", 0.0) or 0.0
            rc = result.get("right_conf", 0.0) or 0.0
            too_far = bool(result.get("too_far"))

            tag = {"both": "BOTH detected", "left_only": "LEFT only",
                   "right_only": "RIGHT only", "face_only": "FACE only (no canthus)",
                   "none": "NONE detected"}[outcome]
            too_far_note = "  [flagged too far]" if too_far else ""
            print(f"[{name:28s}] {img_path.name:40s} -> {tag:24s} "
                  f"(L conf={lc:.2f}, R conf={rc:.2f}){too_far_note}")

            annotated_path = ""
            if outcome == "face_only":
                annotated = draw_annotation(bgr, result)
                out_name = f"{img_path.stem}__face_only{img_path.suffix}"
                out_path = out_dir / name / "face_only_no_canthus" / out_name
                cv2.imwrite(str(out_path), annotated)
                annotated_path = str(out_path)
            elif outcome != "none":
                annotated = draw_annotation(bgr, result)
                out_name = f"{img_path.stem}__{outcome}{img_path.suffix}"
                out_path = out_dir / name / "annotated" / out_name
                cv2.imwrite(str(out_path), annotated)
                annotated_path = str(out_path)

            summary_rows.append({
                "model": name, "image": img_path.name, "outcome": outcome,
                "left_conf": round(lc, 4), "right_conf": round(rc, 4),
                "too_far": too_far,
                "latency_s": round(dt, 5), "annotated_path": annotated_path,
            })

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "image", "outcome", "left_conf", "right_conf", "too_far",
            "latency_s", "annotated_path"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'=' * 70}")
    print("SUMMARY  (counts out of", len(images), "images)")
    print(f"{'=' * 70}")
    for name, c in counts.items():
        print(f"{name:28s} both={c['both']:3d}  left_only={c['left_only']:3d}  "
              f"right_only={c['right_only']:3d}  face_only={c['face_only']:3d}  "
              f"none={c['none']:3d}  errors={c['errors']:3d}  unreadable={c['unreadable']:3d}")
    print(f"\nWrote {summary_path}")
    print(f"Canthus hits saved under:        {out_dir}/<model_name>/annotated/")
    print(f"Face-found-but-no-canthus saved:  {out_dir}/<model_name>/face_only_no_canthus/")


if __name__ == "__main__":
    main()
