"""Milestone 2a: OCR every extracted frame with Tesseract.

For each frame in frames.csv we preprocess (grayscale + upscale + Otsu, see
util.ocr_preprocess) and run Tesseract via pytesseract.image_to_data so we get
per-token confidence. Tokens are kept only if they clear `_keep_token`
(confidence + length + has-a-letter), which removes the bulk of Tesseract's
hallucinated noise on textureless frames. Output: data/ocr.csv with one row per
frame, including the Face IDs present in that frame (Milestone 2 Task 1).
"""
import csv
import difflib
import re
import shutil

import cv2
import pandas as pd
import pytesseract
from tqdm import tqdm

import config
import util

log = util.get_logger()

_LEX_LOWER = [w.lower() for w in config.OCR_LEXICON]
_LEX_CANON = {w.lower(): w for w in config.OCR_LEXICON}


def _keep_token(word: str, conf: float) -> bool:
    """Keep a high-confidence token that is long enough and has a letter.

    Pure + side-effect free so it can be unit-tested without Tesseract."""
    return (conf >= config.OCR_MIN_CONF
            and len(word) >= config.OCR_MIN_TOKEN_LEN
            and any(c.isalpha() for c in word))


def _match_case(canon: str, original: str) -> str:
    """Render the canonical word in the original token's case style."""
    if original.isupper():
        return canon.upper()
    if original[:1].isupper():
        return canon[:1].upper() + canon[1:]
    return canon.lower()


def _correct_token(word: str) -> str:
    """Snap a noisy OCR token to its nearest domain-lexicon entry.

    Precision-first and pure (unit-tested without Tesseract): the alphabetic
    core must be long enough, not already a valid lexicon word, and not a known
    real-word collision (config.OCR_LEXICON_STOP); only then is it rewritten to
    the closest lexicon term whose difflib ratio clears OCR_LEXICON_CUTOFF.
    Leading/trailing punctuation on the original token is preserved."""
    if not config.OCR_LEXICON_ENABLE:
        return word
    core = re.sub(r"[^a-z]", "", word.lower())
    if (len(core) < config.OCR_LEXICON_MIN_LEN
            or core in _LEX_CANON or core in config.OCR_LEXICON_STOP):
        return word
    m = difflib.get_close_matches(core, _LEX_LOWER, n=1,
                                  cutoff=config.OCR_LEXICON_CUTOFF)
    if not m:
        return word
    pre = re.match(r"^[^A-Za-z]*", word).group(0)
    post = re.search(r"[^A-Za-z]*$", word).group(0)
    return pre + _match_case(_LEX_CANON[m[0]], word) + post


def correct_tokens(tokens) -> list[str]:
    """Apply lexicon correction to a list of kept OCR tokens (count preserved)."""
    return [_correct_token(t) for t in tokens]


def _filter_tokens(words, confs) -> tuple[list[str], list[float]]:
    """Filter a Tesseract token/confidence stream down to clean tokens."""
    tokens, kept = [], []
    for word, conf in zip(words, confs):
        word = (word or "").strip()
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = -1.0
        if word and _keep_token(word, conf):
            tokens.append(word)
            kept.append(conf)
    return tokens, kept


def _ocr_frame(bgr) -> tuple[str, int, float]:
    """Return (joined_text, n_tokens, mean_conf) for one frame image."""
    img = util.ocr_preprocess(bgr)
    cfg = f"--psm {config.OCR_PSM}"
    data = pytesseract.image_to_data(
        img, lang=config.OCR_LANG, config=cfg,
        output_type=pytesseract.Output.DICT)

    tokens, confs = _filter_tokens(data["text"], data["conf"])
    tokens = correct_tokens(tokens)   # snap noisy tokens to the domain lexicon
    text = " ".join(tokens)
    mean_conf = round(sum(confs) / len(confs), 1) if confs else 0.0
    return text, len(tokens), mean_conf


def _check_tesseract() -> None:
    """Fail early with a friendly message if the Tesseract binary is missing."""
    if shutil.which("tesseract"):
        return
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        raise RuntimeError(
            "Tesseract OCR binary not found on PATH. Install it first "
            "(macOS: `brew install tesseract`; Debian/Ubuntu: "
            "`apt-get install tesseract-ocr`).")


def run() -> int:
    config.ensure_dirs()
    _check_tesseract()
    frames = pd.read_csv(config.FRAMES_CSV)
    face_map = util.face_ids_by_frame()
    log.info("OCR over %d frames (lang=%s, psm=%d, min_conf=%d, min_len=%d)",
             len(frames), config.OCR_LANG, config.OCR_PSM,
             config.OCR_MIN_CONF, config.OCR_MIN_TOKEN_LEN)

    n_text = 0
    with open(config.OCR_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "timestamp_sec", "face_ids",
                    "text", "n_tokens", "mean_conf"])
        for row in tqdm(frames.itertuples(index=False), total=len(frames),
                        desc="ocr"):
            faces = "|".join(face_map.get(int(row.frame_id), []))
            img = cv2.imread(str(config.FRAME_DIR / row.filename))
            if img is None:
                log.warning("could not read frame %s", row.filename)
                w.writerow([row.frame_id, f"{row.timestamp_sec:.3f}",
                            faces, "", 0, 0.0])
                continue
            text, n_tok, conf = _ocr_frame(img)
            if text:
                n_text += 1
            w.writerow([row.frame_id, f"{row.timestamp_sec:.3f}",
                        faces, text, n_tok, conf])

    log.info("OCR done: %d/%d frames had text -> %s",
             n_text, len(frames), config.OCR_CSV.name)
    return n_text


if __name__ == "__main__":
    run()
