#!/usr/bin/env python3
"""
eval_harness.py
================
Runs every ACTIVE detector wrapper in detectors/ against an annotated set
of saved frames, and writes report.csv with one row per
(model, distance_bucket_m, occlusion_condition).

EXPECTED DIRECTORY LAYOUT
--------------------------
your_project/
  eval_harness.py          <- this file
  detectors/
    detector_singlestage_sahi_v1.py
    detector_singlestage_sahi_v2_cascade.py
    detector_twostage_fixedzoom.py
    detector_twostage_adaptivezoom.py
    detector_tfake_landmark.py
  weights/
    detect.pt               <- EDIT MODEL_REGISTRY below to point at your real files
    thermal_detector.pt
  captures/
    annotations.csv
    <image_id>.png
    <image_id>_y16.npy
    <image_id>_temps.npy     <- OPTIONAL, see "ABOUT temps_c" below

EXPECTED annotations.csv SCHEMA
---------------------------------
One row per annotated frame. Column names are matched against a list of
candidates (see COL_* below) rather than one fixed name, since a "visible"
flag and an "occluded" flag, or "subject" vs "subject_id", show up
interchangeably depending on what exported the CSV:

  image_id              - bare id, NOT necessarily extension-free in your
                           source data -- this harness only strips a
                           recognized image extension if present, it does
                           NOT use pathlib's .stem (which truncates at the
                           last dot in the string -- a real problem if your
                           ids embed a decimal, e.g. "madhu_0.8m_...")
  image_path            - optional; used as a fallback file location if
                           <image_id>.png isn't found directly
  subject / subject_id  - not used in metrics, just passed through
  distance_m            - numeric distance; used as the distance_bucket_m
                           grouping key (see DISTANCE BUCKET FOLDING below
                           for the one exception)
  occlusion              - categorical condition label (e.g. "none",
                           "left_occluded_turn_left", ...); used as the
                           occlusion_condition grouping key directly
  gt_left_x, gt_left_y / left_x, left_y
                         - ground-truth left-canthus pixel coords, blank
                           if not visible/occluded
  gt_left_visible / left_occluded
                         - EITHER a "visible" flag (1 = point exists) OR
                           an "occluded" flag (1 = no point) -- whichever
                           your CSV has, both are handled. If neither
                           column is present, a side is inferred occluded
                           simply from its x/y being blank.
  gt_right_x, gt_right_y, gt_right_visible / right_x, right_y, right_occluded
                         - same, for the right canthus

If your actual exported column names differ, edit the COL_* constants
just below MATCH_PIXEL_THRESHOLD -- that's the one place they're used.

DISTANCE BUCKET FOLDING
-------------------------
Per current calibration, a captured distance of 2.5m is folded into the
1.5m bucket for grouping/reporting purposes (DISTANCE_BUCKET_OVERRIDES
below) -- i.e. rows annotated as distance_m=2.5 will appear under
distance_bucket_m="1.5" in report.csv, not "2.5". This is purely a
grouping-key remap done in normalize_distance_bucket(); the raw
distance_m value in annotations.csv is untouched. If this assumption
ever changes (e.g. once there's enough real 2.5m data to stand on its
own), just edit/clear DISTANCE_BUCKET_OVERRIDES.

ABOUT temps_c (only matters for detector_tfake_landmark.py)
--------------------------------------------------------------
detector_tfake_landmark.detect() accepts an optional temps_c array for a
faithful comparison (see that file's docstring for why). If a file named
<image_id>_temps.npy exists next to <image_id>.png in --images-dir, this
harness will load it and pass it to any detector whose detect() signature
accepts a temps_c argument. If you don't have that file, T-FAKE silently
falls back to its (flagged, less faithful) approximation -- nothing here
crashes either way.

HOW A FRAME IS SCORED
-----------------------
For each (model, frame), each side (left/right) is independently
classified by comparing the model's reported point against ground truth:

  ground truth visible + prediction within MATCH_PIXEL_THRESHOLD px -> TP
  ground truth visible + prediction missing                         -> FN
  ground truth visible + prediction present but too far             -> FN + FP
                                                                         (missed the real point AND produced a spurious one)
  ground truth OCCLUDED + prediction present                        -> FP
                                                                         (this is what occluded_side_false_positive_rate counts)
  ground truth OCCLUDED + prediction missing                        -> correct, no counters incremented

precision/recall/f1 pool left+right sides together for a (model, bucket)
group. mean_pixel_error averages only over TP cases. mean_latency_s times
one full detect() call per frame (both sides), not per side.
"""

import argparse
import csv
import importlib
import inspect
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

# =============================================================
# TUNABLES
# =============================================================
# Calibrate this against your annotator-disagreement test (README step 2's
# tip) rather than trusting the default. Overridable at the command line
# with --match-threshold so you don't have to edit source to experiment.
MATCH_PIXEL_THRESHOLD = 8.0

# Distance-bucket grouping-key remaps. Key = raw distance_m value (as a
# float), value = the bucket it should be reported under instead. Only
# affects the grouping key used for report.csv rows -- never mutates the
# underlying annotations.csv data. See "DISTANCE BUCKET FOLDING" above.
DISTANCE_BUCKET_OVERRIDES = {
    2.5: 1.5,
}

# Column names in your annotations.csv. Each is a list of candidate
# header names tried in order -- the first one actually present in your
# CSV wins. This is tolerant of small naming differences between your
# capture metadata.csv and your annotate_tool.html export (e.g. "subject"
# vs "subject_id", or a "visible" flag instead of an "occluded" one) so
# you don't have to hand-edit this every time a header changes slightly.
# If your CSV uses something not listed, just add it to the relevant list.
COL_IMAGE_ID         = ["image_id"]
COL_IMAGE_PATH       = ["image_path"]   # optional, used as a fallback to locate the file
COL_SUBJECT          = ["subject_id", "subject"]
COL_DISTANCE_M       = ["distance_m"]
COL_OCCLUSION        = ["occlusion", "occlusion_condition"]
COL_LEFT_X           = ["gt_left_x", "left_x"]
COL_LEFT_Y           = ["gt_left_y", "left_y"]
COL_LEFT_VISIBLE     = ["gt_left_visible", "gt_left_vis", "gt_left_visibility", "left_visible"]
COL_LEFT_OCCLUDED    = ["left_occluded", "gt_left_occluded"]
COL_RIGHT_X          = ["gt_right_x", "right_x"]
COL_RIGHT_Y          = ["gt_right_y", "right_y"]
COL_RIGHT_VISIBLE    = ["gt_right_visible", "gt_right_vis", "gt_right_visibility", "right_visible"]
COL_RIGHT_OCCLUDED   = ["right_occluded", "gt_right_occluded"]

# =============================================================
# MODEL REGISTRY -- EDIT THIS
# =============================================================
# weights value is passed straight through to that module's load_models().
# - single-stage detectors expect a single path (str/Path) or None to
#   auto-detect.
# - two-stage detectors expect a dict {"face": ..., "canthus": ...} or
#   None to auto-detect.
# - detector_tfake_landmark.py ignores weights entirely (the tfan package
#   resolves its own).
MODEL_REGISTRY = {
    "singlestage_sahi_v1": {
        "module": "detector_singlestage_sahi_v1",
        "weights": "weights/detect.pt",  # <-- EDIT ME
    },
    "singlestage_sahi_v2_cascade": {
        "module": "detector_singlestage_sahi_v2_cascade",
        "weights": "weights/detect.pt",  # <-- EDIT ME (same canthus weights, different slicing)
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

# Remove/comment out any entry above you haven't wired up real weights for
# yet -- the rest will still run.
ACTIVE_MODELS = list(MODEL_REGISTRY.keys())


# =============================================================
# MODEL LOADING
# =============================================================
def setup_models(detectors_dir: Path, active_models=None):
    """Import + load_models() every active detector. Returns {name: (module, model_obj)}."""
    sys.path.insert(0, str(detectors_dir))

    names = active_models if active_models is not None else ACTIVE_MODELS
    loaded = {}
    for name in names:
        if name not in MODEL_REGISTRY:
            print(f"WARNING: '{name}' not found in MODEL_REGISTRY -- skipping")
            continue
        cfg = MODEL_REGISTRY[name]
        print(f"Loading {name} ({cfg['module']}) ...")
        try:
            mod = importlib.import_module(cfg["module"])
            model_obj = mod.load_models(cfg.get("weights"))
        except Exception as e:
            print(f"  FAILED to load {name}: {e}")
            print(f"  (skipping this model for the rest of the run)")
            continue
        loaded[name] = (mod, model_obj)
        print(f"  OK")
    return loaded


# =============================================================
# GROUND TRUTH / IMAGE LOADING
# =============================================================
def get_first(row: dict, candidates):
    """Return the value of the first candidate column name actually
    present as a key in this CSV row, or None if none of them are."""
    for c in candidates:
        if c in row:
            return row[c]
    return None


def parse_bool(s) -> bool:
    if s is None:
        return False
    return str(s).strip().lower() in ("1", "true", "yes", "y", "occluded")


def parse_float_or_none(s):
    if s is None or str(s).strip() == "":
        return None
    return float(s)


def normalize_image_id(raw_id: str) -> str:
    """
    Strip a trailing image extension if present -- WITHOUT using
    pathlib's .stem, which truncates at the LAST dot in the string. Your
    image_ids legitimately contain dots (e.g. "madhu_0.8m_..." for a
    0.8m capture distance), and Path("madhu_0.8m_x").stem gives "madhu_0"
    -- silently mangling every id and making every file lookup fail.
    This only strips a recognized image extension, nothing else.
    """
    s = str(raw_id).strip()
    for ext in (".png", ".jpg", ".jpeg"):
        if s.lower().endswith(ext):
            return s[: -len(ext)]
    return s


def normalize_distance_bucket(raw_distance) -> str:
    """
    Maps a raw distance_m value to the grouping key used for
    distance_bucket_m in report.csv.

    Currently this only does one thing: fold a captured distance of
    2.5m into the 1.5m bucket (see DISTANCE_BUCKET_OVERRIDES and the
    "DISTANCE BUCKET FOLDING" note in the module docstring). Any other
    numeric value passes through unchanged (just normalized so that
    "1.5", "1.50", and 1.5 all collapse to the same string key -- without
    this, those would silently end up as different groups in report.csv).
    Non-numeric / blank values pass through completely untouched, since
    they're not something this remap can reason about.
    """
    s = str(raw_distance).strip()
    if s == "":
        return s
    try:
        val = float(s)
    except ValueError:
        return s

    val = DISTANCE_BUCKET_OVERRIDES.get(val, val)

    if val == int(val):
        return str(int(val))
    return str(val)


def get_side_gt(row: dict, x_keys, y_keys, visible_keys, occluded_keys):
    """
    Resolve one side's (point, occluded) from a GT row, tolerant of
    either a "visible" flag (1 = point exists) or an "occluded" flag
    (1 = no point) -- whichever your CSV actually has. As a final
    safety net, a side is also treated as occluded if its x/y are simply
    blank, regardless of what any flag column says, since "no
    coordinates" is the ground truth that actually matters here.
    """
    x = parse_float_or_none(get_first(row, x_keys))
    y = parse_float_or_none(get_first(row, y_keys))
    coords_present = x is not None and y is not None

    visible_raw = get_first(row, visible_keys)
    occluded_raw = get_first(row, occluded_keys)

    if visible_raw is not None and str(visible_raw).strip() != "":
        explicit_occluded = not parse_bool(visible_raw)
    elif occluded_raw is not None and str(occluded_raw).strip() != "":
        explicit_occluded = parse_bool(occluded_raw)
    else:
        explicit_occluded = False

    occluded = explicit_occluded or (not coords_present)
    pt = (x, y) if coords_present else None
    return pt, occluded


def load_ground_truth(gt_path: Path):
    with open(gt_path, newline="") as f:
        return list(csv.DictReader(f))


def load_image(images_dir: Path, image_id: str, image_path_hint: str = None):
    for ext in (".png", ".jpg", ".jpeg"):
        p = images_dir / f"{image_id}{ext}"
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img

    # Fallback: your capture metadata/annotations may carry an
    # image_path column with the actual relative path -- try that
    # directly, and also just its filename inside images_dir, in case
    # image_id and the on-disk filename ever diverge.
    if image_path_hint:
        for p in (Path(image_path_hint), images_dir / Path(image_path_hint).name):
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    return img

    raise FileNotFoundError(f"no image found for image_id='{image_id}' in {images_dir}")


def maybe_load_temps(images_dir: Path, image_id: str):
    p = images_dir / f"{image_id}_temps.npy"
    if p.exists():
        return np.load(str(p))
    return None


# =============================================================
# DETECTION + SCORING
# =============================================================
def call_detect(mod, model_obj, bgr, temps_c):
    """Call this detector's detect(), passing temps_c through only if it accepts it."""
    sig = inspect.signature(mod.detect)
    kwargs = {}
    if temps_c is not None and "temps_c" in sig.parameters:
        kwargs["temps_c"] = temps_c

    t0 = time.perf_counter()
    result = mod.detect(model_obj, bgr, **kwargs)
    dt = time.perf_counter() - t0
    return result, dt


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def classify_side(gt_pt, gt_occluded: bool, pred_pt):
    """
    Returns (outcome, pixel_error_or_None) where outcome is one of:
    "tp", "fn", "fn_fp_localization", "fp_occlusion", "tn_occlusion"
    """
    gt_present = (not gt_occluded) and (gt_pt is not None)
    pred_present = pred_pt is not None

    if gt_present and pred_present:
        d = euclidean(gt_pt, pred_pt)
        if d <= MATCH_PIXEL_THRESHOLD:
            return "tp", d
        return "fn_fp_localization", d
    if gt_present and not pred_present:
        return "fn", None
    if (not gt_present) and pred_present:
        return "fp_occlusion", None
    return "tn_occlusion", None


def new_bucket():
    return {
        "tp": 0, "fn": 0, "fp": 0,
        "occluded_total": 0, "occluded_fp": 0,
        "pixel_errors": [], "latencies": [],
    }


def safe_div(a, b):
    return (a / b) if b else None


def fmt(x, ndigits=4):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, ndigits)


# =============================================================
# MAIN
# =============================================================
def main():
    global MATCH_PIXEL_THRESHOLD

    ap = argparse.ArgumentParser(description="Quantitative canthus-detector evaluation harness")
    ap.add_argument("--gt", required=True, help="Path to annotations.csv")
    ap.add_argument("--images-dir", required=True, help="Folder containing <image_id>.png frames")
    ap.add_argument("--out", default="report.csv", help="Output report CSV path")
    ap.add_argument("--match-threshold", type=float, default=MATCH_PIXEL_THRESHOLD,
                     help="Pixel distance within which a detection counts as correct "
                          "(calibrate against your annotator-disagreement test)")
    ap.add_argument("--models", default=None,
                     help="Comma-separated subset of ACTIVE_MODELS to run "
                          "(default: everything in ACTIVE_MODELS)")
    args = ap.parse_args()

    MATCH_PIXEL_THRESHOLD = args.match_threshold

    gt_path = Path(args.gt)
    images_dir = Path(args.images_dir)
    detectors_dir = Path(__file__).parent / "detectors"

    active = args.models.split(",") if args.models else None

    gt_rows = load_ground_truth(gt_path)
    print(f"Loaded {len(gt_rows)} annotated frames from {gt_path}")

    models = setup_models(detectors_dir, active_models=active)
    if not models:
        print("No models loaded successfully -- nothing to evaluate. Exiting.")
        sys.exit(1)
    print(f"Active models: {list(models.keys())}\n")

    # stats[model_name][(distance_m, occlusion)] = bucket dict
    stats = defaultdict(lambda: defaultdict(new_bucket))

    n_frames_seen = 0
    n_frames_skipped = 0

    for i, row in enumerate(gt_rows):
        image_id = normalize_image_id(get_first(row, COL_IMAGE_ID))
        distance_m_raw = get_first(row, COL_DISTANCE_M) or ""
        distance_m = normalize_distance_bucket(distance_m_raw)
        occlusion = get_first(row, COL_OCCLUSION) or "none"
        image_path_hint = get_first(row, COL_IMAGE_PATH)

        try:
            bgr = load_image(images_dir, image_id, image_path_hint=image_path_hint)
        except FileNotFoundError as e:
            print(f"WARNING: {e} -- skipping row {i}")
            n_frames_skipped += 1
            continue

        temps_c = maybe_load_temps(images_dir, image_id)

        gt_left, gt_left_occluded = get_side_gt(
            row, COL_LEFT_X, COL_LEFT_Y, COL_LEFT_VISIBLE, COL_LEFT_OCCLUDED)
        gt_right, gt_right_occluded = get_side_gt(
            row, COL_RIGHT_X, COL_RIGHT_Y, COL_RIGHT_VISIBLE, COL_RIGHT_OCCLUDED)

        key = (distance_m, occlusion)
        n_frames_seen += 1

        for name, (mod, model_obj) in models.items():
            try:
                result, dt = call_detect(mod, model_obj, bgr, temps_c)
            except Exception as e:
                print(f"ERROR running {name} on {image_id}: {e}")
                continue

            bucket = stats[name][key]
            bucket["latencies"].append(dt)

            for pred_pt, gt_pt, gt_occluded in (
                (result.get("left_pt"), gt_left, gt_left_occluded),
                (result.get("right_pt"), gt_right, gt_right_occluded),
            ):
                outcome, dist = classify_side(gt_pt, gt_occluded, pred_pt)

                if gt_occluded:
                    bucket["occluded_total"] += 1

                if outcome == "tp":
                    bucket["tp"] += 1
                    bucket["pixel_errors"].append(dist)
                elif outcome == "fn":
                    bucket["fn"] += 1
                elif outcome == "fn_fp_localization":
                    bucket["fn"] += 1
                    bucket["fp"] += 1
                elif outcome == "fp_occlusion":
                    bucket["fp"] += 1
                    bucket["occluded_fp"] += 1
                # "tn_occlusion": correctly silent -- nothing to increment

        if (i + 1) % 25 == 0:
            print(f"... processed {i + 1}/{len(gt_rows)} frames")

    print(f"\nDone. {n_frames_seen} frames scored, {n_frames_skipped} skipped (missing image).")

    # ---- aggregate into report rows ----
    out_rows = []
    for name, buckets in stats.items():
        for (distance_m, occlusion), b in buckets.items():
            tp, fn, fp = b["tp"], b["fn"], b["fp"]
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = (safe_div(2 * precision * recall, precision + recall)
                  if (precision is not None and recall is not None and (precision + recall) > 0)
                  else None)
            mean_pixel_error = (sum(b["pixel_errors"]) / len(b["pixel_errors"])
                                 if b["pixel_errors"] else None)
            occluded_fp_rate = safe_div(b["occluded_fp"], b["occluded_total"])
            mean_latency = (sum(b["latencies"]) / len(b["latencies"])
                             if b["latencies"] else None)

            out_rows.append({
                "model": name,
                "distance_bucket_m": distance_m,
                "occlusion_condition": occlusion,
                "n_frames": len(b["latencies"]),
                "precision": fmt(precision, 3),
                "recall": fmt(recall, 3),
                "f1": fmt(f1, 3),
                "mean_pixel_error": fmt(mean_pixel_error, 2),
                "occluded_side_false_positive_rate": fmt(occluded_fp_rate, 3),
                "mean_latency_s": fmt(mean_latency, 4),
            })

    out_rows.sort(key=lambda r: (r["model"], str(r["distance_bucket_m"]), str(r["occlusion_condition"])))

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "distance_bucket_m", "occlusion_condition", "n_frames",
            "precision", "recall", "f1", "mean_pixel_error",
            "occluded_side_false_positive_rate", "mean_latency_s",
        ])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
