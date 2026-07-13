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


def _scene_fields() -> dict:
    """frame_id -> (scene_index, story_segment) from scenes.json, or {} when
    Milestone 3 has not run. `story_segment` is the chapter the frame narrates
    within -- the Underground line, e.g. "Jubilee Line" -- which is the unit the
    story is written in (one '## <label>' section per chapter, see narrate.py)."""
    if not config.SCENES_JSON.exists():
        return {}
    with open(config.SCENES_JSON) as f:
        scenes = json.load(f)
    out = {}
    for s in scenes:
        for fid in range(s["start_frame_id"], s["end_frame_id"] + 1):
            out[fid] = (s["scene_index"], s["chapter_label"])
    return out


def _event_descriptions(timestamps) -> dict:
    """frame_id -> the timeline event in force at that frame (the most recent
    event at or before its timestamp), or {} when no timeline exists. A step
    function, so every frame after the first event carries a description."""
    if not config.TIMELINE_JSON.exists():
        return {}
    with open(config.TIMELINE_JSON) as f:
        events = json.load(f).get("events", [])
    if not events:
        return {}
    events = sorted(events, key=lambda e: e["timestamp_sec"])
    out, i = {}, 0
    for fid, ts in sorted(timestamps.items(), key=lambda kv: kv[1]):
        while i + 1 < len(events) and events[i + 1]["timestamp_sec"] <= ts:
            i += 1
        out[fid] = events[i]["description"] if events[i]["timestamp_sec"] <= ts \
            else ""
    return out


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

    # Milestone 3 enrichment. Left join: absent artifacts leave the columns empty
    # rather than failing, so an M2-only checkout still builds the repository.
    # run_pipeline calls run() again after the M3 stages to fill these in.
    scene_map = _scene_fields()
    ts_map = {int(r.frame_id): float(r.timestamp_sec)
              for r in df.itertuples(index=False)}
    event_map = _event_descriptions(ts_map)
    df["scene_index"] = df["frame_id"].map(
        lambda fid: scene_map.get(int(fid), (None, ""))[0])
    df["story_segment"] = df["frame_id"].map(
        lambda fid: scene_map.get(int(fid), (None, ""))[1])
    df["event_description"] = df["frame_id"].map(
        lambda fid: event_map.get(int(fid), ""))

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
            # Milestone 3 (Task 4: metadata enhancement). pandas turns a column of
            # Nones into object dtype but a *partially* covered one into float
            # NaN, so test with isna rather than `is None`.
            "scene_index": (None if pd.isna(r.scene_index)
                            else int(r.scene_index)),
            "story_segment": r.story_segment,
            "event_description": r.event_description,
        })
    with open(config.METADATA_JSON, "w") as f:
        json.dump(records, f, indent=2)

    # CSV: face_ids pipe-joined for spreadsheet friendliness.
    csv_df = df[["frame_id", "timestamp_sec", "filename", "face_ids",
                 "ocr_text", "caption", "caption_echoes_text",
                 "scene_index", "story_segment", "event_description"]].copy()
    csv_df["face_ids"] = csv_df["face_ids"].map(lambda ids: "|".join(ids))
    csv_df.to_csv(config.METADATA_CSV, index=False)

    n_faces = sum(1 for r in records if r["face_ids"])
    n_text = sum(1 for r in records if r["ocr_text"])
    n_echo = sum(1 for r in records if r["caption_echoes_text"])
    n_scene = sum(1 for r in records if r["scene_index"] is not None)
    n_event = sum(1 for r in records if r["event_description"])
    log.info("metadata repository: %d frames (%d with faces, %d with text, "
             "%d text-echo captions, %d with scene_index, %d with event) -> %s",
             len(records), n_faces, n_text, n_echo, n_scene, n_event,
             config.METADATA_JSON.name)
    return len(records)


if __name__ == "__main__":
    run()
