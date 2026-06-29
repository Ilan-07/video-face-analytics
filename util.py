"""Shared helpers: logging, geometry, image quality (Fix #7 robustness)."""
import logging

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
