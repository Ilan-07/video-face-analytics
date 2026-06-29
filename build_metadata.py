"""Milestone 2c/Task 4: the frame metadata repository.

Joins the per-frame artifacts into one structured dataset to be consumed by
future milestones and the search UI:

    frame_id | timestamp_sec | face_ids | ocr_text | caption

Face IDs per frame come from the same join analytics._load uses: faces.csv
(frame_id, track_id) merged with identities.csv (track_id -> face_id). Frames
with no faces still appear (left join), so the repository covers every frame.

Outputs both data/frame_metadata.csv (face_ids pipe-joined for spreadsheets) and
data/frame_metadata.json (face_ids as a real array -- canonical source).
"""
import json

import pandas as pd

import config
import util

log = util.get_logger()


def _face_ids_by_frame() -> dict[int, list[str]]:
    """Map frame_id -> sorted list of distinct real face_ids in that frame."""
    faces = pd.read_csv(config.FACES_CSV)
    ident = pd.read_csv(config.IDENTITIES_CSV)
    faces = faces.merge(ident[["track_id", "face_id"]], on="track_id", how="left")
    faces = faces[faces["face_id"].notna() & (faces["face_id"] != "unknown")]
    out: dict[int, list[str]] = {}
    for frame_id, g in faces.groupby("frame_id"):
        out[int(frame_id)] = sorted(g["face_id"].unique().tolist())
    return out


def run() -> int:
    config.ensure_dirs()
    frames = pd.read_csv(config.FRAMES_CSV)
    ocr = pd.read_csv(config.OCR_CSV)
    caps = pd.read_csv(config.CAPTIONS_CSV)

    face_map = _face_ids_by_frame()

    df = frames[["frame_id", "timestamp_sec", "filename"]].copy()
    df = df.merge(ocr[["frame_id", "text"]], on="frame_id", how="left")
    df = df.merge(caps[["frame_id", "caption"]], on="frame_id", how="left")
    df = df.rename(columns={"text": "ocr_text"})
    df["ocr_text"] = df["ocr_text"].fillna("")
    df["caption"] = df["caption"].fillna("")
    df["face_ids"] = df["frame_id"].map(lambda fid: face_map.get(int(fid), []))

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
        })
    with open(config.METADATA_JSON, "w") as f:
        json.dump(records, f, indent=2)

    # CSV: face_ids pipe-joined for spreadsheet friendliness.
    csv_df = df[["frame_id", "timestamp_sec", "filename",
                 "face_ids", "ocr_text", "caption"]].copy()
    csv_df["face_ids"] = csv_df["face_ids"].map(lambda ids: "|".join(ids))
    csv_df.to_csv(config.METADATA_CSV, index=False)

    n_faces = sum(1 for r in records if r["face_ids"])
    n_text = sum(1 for r in records if r["ocr_text"])
    log.info("metadata repository: %d frames (%d with faces, %d with text) -> %s",
             len(records), n_faces, n_text, config.METADATA_JSON.name)
    return len(records)


if __name__ == "__main__":
    run()
