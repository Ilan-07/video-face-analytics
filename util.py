"""Shared helpers: logging, geometry, image quality (Fix #7 robustness)."""
import contextlib
import json
import logging
import time
from datetime import datetime, timezone

import cv2

import config

_logger = None


def get_logger() -> logging.Logger:
    """Console + file logger (Fix #7: persistent run log)."""
    global _logger
    if _logger is None:
        config.ensure_dirs()
        log = logging.getLogger("facepipe")
        log.setLevel(logging.INFO)
        log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(config.LOG_FILE)
        fh.setFormatter(fmt)
        log.addHandler(sh)
        log.addHandler(fh)
        log.propagate = False
        _logger = log
    return _logger


# ---------------------------------------------------------- stage timing (M4)
# Milestone 4 Task 4 needs the processing time of the whole system, but the
# pipeline is idempotent: on any run after the first, most stages skip and
# measure ~0s. So timings are MERGED into config.STAGE_TIMINGS_JSON -- a stage
# that skips keeps whatever it last measured, and only a real rerun overwrites
# it. The file therefore always describes a full cold build, assembled across
# however many runs it took to produce the current artifacts.

def load_timings() -> dict:
    """Stage -> timing record, or {} when nothing has been measured yet."""
    if not config.STAGE_TIMINGS_JSON.exists():
        return {}
    try:
        return json.loads(config.STAGE_TIMINGS_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        # A truncated timings file must never take down the pipeline: it is a
        # measurement, not an input. Start fresh and let this run repopulate it.
        get_logger().warning("stage_timings.json unreadable; starting fresh")
        return {}


def record_timing(stage: str, seconds: float, **meta) -> dict:
    """Merge one measured stage duration into the timings file, and return it."""
    timings = load_timings()
    timings[stage] = {
        "seconds": round(float(seconds), 3),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "network": stage in config.NETWORK_STAGES,
        **meta,
    }
    config.STAGE_TIMINGS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.STAGE_TIMINGS_JSON.write_text(json.dumps(timings, indent=2,
                                                    sort_keys=True))
    return timings[stage]


@contextlib.contextmanager
def time_stage(stage: str, **meta):
    """Time a pipeline stage and persist the duration on success.

    Written immediately rather than batched at the end so a run that crashes in
    a later stage still leaves the earlier measurements on disk. A stage that
    raises is NOT recorded: a partial duration would understate the real cost
    and quietly corrupt the performance report."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    record_timing(stage, dt, **meta)
    get_logger().info("[time] %s took %.1fs", stage, dt)


def iou(a, b) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def blur_var(bgr) -> float:
    """Variance of the Laplacian; low => blurry (Fix #5)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_ids_by_frame() -> dict:
    """Map frame_id -> sorted list of distinct real face_ids present in it (M2).

    Shared by ocr.py, caption.py and build_metadata.py so every per-frame store
    associates the same Face IDs. Mirrors analytics._load's join: faces.csv
    (frame_id, track_id) -> identities.csv (track_id -> face_id), dropping
    unassigned / "unknown" detections. pandas is imported lazily to keep this
    module cheap for callers that only need the geometry helpers."""
    import pandas as pd

    faces = pd.read_csv(config.FACES_CSV)
    ident = pd.read_csv(config.IDENTITIES_CSV)
    faces = faces.merge(ident[["track_id", "face_id"]], on="track_id", how="left")
    faces = faces[faces["face_id"].notna() & (faces["face_id"] != "unknown")]
    out: dict[int, list[str]] = {}
    for frame_id, g in faces.groupby("frame_id"):
        out[int(frame_id)] = sorted(g["face_id"].unique().tolist())
    return out


def ocr_preprocess(bgr):
    """Grayscale + upscale + Otsu threshold to sharpen text for Tesseract (M2).

    Small on-screen text reads better after a mild upscale, and a binary image
    removes background gradients that confuse OCR. Returns a single-channel
    uint8 image ready for pytesseract."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    f = config.OCR_UPSCALE
    if f and f != 1.0:
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def nose_offset(kps) -> float:
    """|horizontal nose offset / inter-eye distance|; ~0 frontal, large => profile.

    InsightFace 5-point order: [eye, eye, nose, mouth, mouth] (Fix #5)."""
    eye_cx = (kps[0][0] + kps[1][0]) / 2.0
    eye_dist = abs(kps[1][0] - kps[0][0]) + 1e-6
    return abs((kps[2][0] - eye_cx) / eye_dist)
