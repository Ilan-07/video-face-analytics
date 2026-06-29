"""Milestone 2c/Task 4: the frame metadata repository.

Joins the per-frame artifacts into one structured dataset to be consumed by
future milestones and the search UI:

    frame_id | timestamp_sec | face_ids | ocr_text | caption | caption_echoes_text

Face IDs per frame come from util.face_ids_by_frame (the same join analytics and
the OCR/caption stores use). Frames with no faces still appear (left join), so the
repository covers every frame.

`caption_echoes_text` flags frames whose caption mostly reproduces the OCR text
(title cards) rather than describing the scene -- a quality signal so downstream
search/analysis can discount low-value captions.

Outputs both data/frame_metadata.csv (face_ids pipe-joined for spreadsheets) and
data/frame_metadata.json (face_ids as a real array -- canonical source).
"""
import json
import re

import pandas as pd

import config
import util

log = util.get_logger()

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def caption_echoes_text(caption: str, ocr_text: str) -> bool:
    """True when the caption's words mostly overlap the OCR text (Jaccard >=
    threshold) -- i.e. the model read a title card instead of describing it."""
    cap, ocr = _tokens(caption), _tokens(ocr_text)
    if not cap or not ocr:
        return False
    jac = len(cap & ocr) / len(cap | ocr)
    return jac >= config.CAPTION_TEXT_ECHO_JACCARD


def run() -> int:
    config.ensure_dirs()
    frames = pd.read_csv(config.FRAMES_CSV)
    ocr = pd.read_csv(config.OCR_CSV)
    caps = pd.read_csv(config.CAPTIONS_CSV)

    face_map = util.face_ids_by_frame()

    df = frames[["frame_id", "timestamp_sec", "filename"]].copy()
    df = df.merge(ocr[["frame_id", "text"]], on="frame_id", how="left")
    df = df.merge(caps[["frame_id", "caption"]], on="frame_id", how="left")
    df = df.rename(columns={"text": "ocr_text"})
    df["ocr_text"] = df["ocr_text"].fillna("")
    df["caption"] = df["caption"].fillna("")
    df["face_ids"] = df["frame_id"].map(lambda fid: face_map.get(int(fid), []))
    df["caption_echoes_text"] = [
        caption_echoes_text(c, t)
        for c, t in zip(df["caption"], df["ocr_text"])]

    # JSON: canonical, face_ids as array.
    records = []
    for r in df.itertuples(index=False):
        records.append({
            "frame_id": int(r.frame_id),
            "timestamp_sec": round(float(r.timestamp_sec), 3),
            "filename": r.filename,
            "face_ids": r.face_ids,
            "ocr_text": r.ocr_text,
            "caption": r.caption,
            "caption_echoes_text": bool(r.caption_echoes_text),
        })
    with open(config.METADATA_JSON, "w") as f:
        json.dump(records, f, indent=2)

    # CSV: face_ids pipe-joined for spreadsheet friendliness.
    csv_df = df[["frame_id", "timestamp_sec", "filename", "face_ids",
                 "ocr_text", "caption", "caption_echoes_text"]].copy()
    csv_df["face_ids"] = csv_df["face_ids"].map(lambda ids: "|".join(ids))
    csv_df.to_csv(config.METADATA_CSV, index=False)

    n_faces = sum(1 for r in records if r["face_ids"])
    n_text = sum(1 for r in records if r["ocr_text"])
    n_echo = sum(1 for r in records if r["caption_echoes_text"])
    log.info("metadata repository: %d frames (%d with faces, %d with text, "
             "%d text-echo captions) -> %s",
             len(records), n_faces, n_text, n_echo, config.METADATA_JSON.name)
    return len(records)


if __name__ == "__main__":
    run()
