"""Milestone 2a: OCR every extracted frame with Tesseract.

For each frame in frames.csv we preprocess (grayscale + upscale + Otsu, see
util.ocr_preprocess) and run Tesseract via pytesseract.image_to_data so we get
per-token confidence. Low-confidence tokens are dropped (config.OCR_MIN_CONF) so
the downstream search index isn't polluted by OCR hallucinations on textureless
frames. Output: data/ocr.csv with one row per frame.
"""
import csv

import cv2
import pandas as pd
import pytesseract
from tqdm import tqdm

import config
import util

log = util.get_logger()


def _ocr_frame(bgr) -> tuple[str, int, float]:
    """Return (joined_text, n_tokens, mean_conf) for one frame image."""
    img = util.ocr_preprocess(bgr)
    cfg = f"--psm {config.OCR_PSM}"
    data = pytesseract.image_to_data(
        img, lang=config.OCR_LANG, config=cfg,
        output_type=pytesseract.Output.DICT)

    tokens, confs = [], []
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = -1.0
        if word and conf >= config.OCR_MIN_CONF:
            tokens.append(word)
            confs.append(conf)

    text = " ".join(tokens)
    mean_conf = round(sum(confs) / len(confs), 1) if confs else 0.0
    return text, len(tokens), mean_conf


def run() -> int:
    config.ensure_dirs()
    frames = pd.read_csv(config.FRAMES_CSV)
    log.info("OCR over %d frames (lang=%s, psm=%d, min_conf=%d)",
             len(frames), config.OCR_LANG, config.OCR_PSM, config.OCR_MIN_CONF)

    n_text = 0
    with open(config.OCR_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "timestamp_sec", "text", "n_tokens", "mean_conf"])
        for row in tqdm(frames.itertuples(index=False), total=len(frames),
                        desc="ocr"):
            img = cv2.imread(str(config.FRAME_DIR / row.filename))
            if img is None:
                log.warning("could not read frame %s", row.filename)
                w.writerow([row.frame_id, f"{row.timestamp_sec:.3f}", "", 0, 0.0])
                continue
            text, n_tok, conf = _ocr_frame(img)
            if text:
                n_text += 1
            w.writerow([row.frame_id, f"{row.timestamp_sec:.3f}",
                        text, n_tok, conf])

    log.info("OCR done: %d/%d frames had text -> %s",
             n_text, len(frames), config.OCR_CSV.name)
    return n_text


if __name__ == "__main__":
    run()
