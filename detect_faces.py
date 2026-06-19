"""Phase 2: detect, track (ByteTrack), quality-score, embed faces.

Fix #1: tracking now uses the maintained `trackers` ByteTrack (Kalman motion
        model + two-stage IoU association) instead of a hand-rolled IoU tracker.
        ByteTrack returns tracker_id == -1 on a track's birth frame; a one-step
        forward IoU reconciliation folds that frame back into the track so no
        appearance is fragmented or lost.
Fix #5: per-face blur/pose quality gating + gender/age.
Fix #7: per-frame error recovery + logging.
"""
import csv

import cv2
import numpy as np
import supervision as sv
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from tqdm import tqdm
from trackers import ByteTrackTracker

import config
import util
from util import iou

log = util.get_logger()
_app = None


def get_app() -> "FaceAnalysis":
    """Lazily build InsightFace (SCRFD detect + ArcFace recog + gender/age)."""
    global _app
    if _app is None:
        _app = FaceAnalysis(name=config.MODEL_PACK, providers=config.PROVIDERS)
        _app.prepare(ctx_id=config.CTX_ID, det_size=config.DET_SIZE,
                     det_thresh=config.DET_THRESH)
    return _app


def read_frames_index():
    with open(config.FRAMES_CSV) as f:
        return list(csv.DictReader(f))


def _make_tracker() -> ByteTrackTracker:
    return ByteTrackTracker(
        frame_rate=config.TRACK_FRAME_RATE,
        lost_track_buffer=config.TRACK_MAX_GAP,
        minimum_consecutive_frames=1,
        track_activation_threshold=config.TRACK_ACTIVATION,
        minimum_iou_threshold=config.TRACK_IOU,
        high_conf_det_threshold=config.TRACK_HIGH_CONF,
    )


def _reconcile(records: list[dict]) -> dict[str, str]:
    """Map crop_id -> stable track_id, folding birth frames (tid==-1) into tracks.

    A confirmed ByteTrack id -> "t{id}". A birth-frame detection (tid==-1) adopts
    the id of the IoU-matching detection in the next frame; otherwise it becomes
    its own singleton "s{n}".
    """
    by_frame: dict[int, list[dict]] = {}
    for r in records:
        by_frame.setdefault(r["frame_id"], []).append(r)

    stable: dict[str, str] = {}
    singleton = 0
    for r in records:
        if r["bt_tid"] != -1:
            stable[r["crop_id"]] = f"t{r['bt_tid']}"
    for r in records:
        if r["bt_tid"] != -1:
            continue
        best, best_iou = None, config.TRACK_LINK_IOU
        for nxt in by_frame.get(r["frame_id"] + 1, []):
            if nxt["bt_tid"] == -1:
                continue
            s = iou(r["bbox"], nxt["bbox"])
            if s >= best_iou:
                best, best_iou = nxt, s
        if best is not None:
            stable[r["crop_id"]] = stable[best["crop_id"]]
        else:
            stable[r["crop_id"]] = f"s{singleton}"
            singleton += 1
    return stable


def detect() -> int:
    config.ensure_dirs()
    app = get_app()
    rows = read_frames_index()
    tracker = _make_tracker()

    embeddings: dict[str, np.ndarray] = {}
    records: list[dict] = []      # per-detection metadata, track resolved later

    for row in tqdm(rows, desc="[detect] frames"):
        frame_id = int(row["frame_id"])
        img = cv2.imread(str(config.FRAME_DIR / row["filename"]))
        if img is None:
            log.warning("unreadable frame %s", row["filename"])
            continue
        try:
            faces = app.get(img)
        except Exception as e:  # noqa: BLE001  (Fix #7)
            log.warning("detect failed on %s: %s", row["filename"], e)
            continue

        kept = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            if (x2 - x1) < config.MIN_FACE_PX or (y2 - y1) < config.MIN_FACE_PX:
                continue
            kept.append((face, [int(x1), int(y1), int(x2), int(y2)]))

        # Run ByteTrack on this frame's detections.
        if kept:
            dets = sv.Detections(
                xyxy=np.array([b for _, b in kept], dtype=float),
                confidence=np.array([float(f.det_score) for f, _ in kept]),
                class_id=np.zeros(len(kept), dtype=int),
                data={"local": np.arange(len(kept))})
            tracked = tracker.update(dets)
            tid_by_local = {int(li): int(ti) for li, ti
                            in zip(tracked.data["local"], tracked.tracker_id)}
        else:
            tracker.update(sv.Detections.empty())
            tid_by_local = {}

        for j, (face, bbox) in enumerate(kept):
            crop_id = f"f{frame_id:06d}_{j:02d}"
            aligned = face_align.norm_crop(img, face.kps, image_size=112)
            cv2.imwrite(str(config.FACE_DIR / f"{crop_id}.jpg"), aligned)

            embeddings[crop_id] = face.normed_embedding.astype(np.float32)
            records.append({
                "crop_id": crop_id, "frame_id": frame_id,
                "timestamp_sec": row["timestamp_sec"], "bbox": bbox,
                "bt_tid": tid_by_local.get(j, -1), "face": face,
                "aligned_blur": util.blur_var(aligned),
            })

    stable = _reconcile(records)

    header = ["crop_id", "frame_id", "timestamp_sec", "track_id",
              "x1", "y1", "x2", "y2", "det_score", "blur_var",
              "nose_offset", "gender", "age", "quality_ok", "crop_file"]
    with open(config.FACES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in records:
            face = r["face"]
            det = float(face.det_score)
            bv = r["aligned_blur"]
            noff = util.nose_offset(face.kps)
            quality_ok = int(det >= config.HQ_DET_SCORE
                             and bv >= config.MIN_BLUR_VAR
                             and noff <= config.MAX_NOSE_OFFSET)
            x1, y1, x2, y2 = r["bbox"]
            w.writerow([r["crop_id"], r["frame_id"], r["timestamp_sec"],
                        stable[r["crop_id"]], x1, y1, x2, y2,
                        f"{det:.4f}", f"{bv:.1f}", f"{noff:.3f}",
                        getattr(face, "sex", "") or "",
                        int(getattr(face, "age", 0) or 0),
                        quality_ok, f"{r['crop_id']}.jpg"])

    if embeddings:
        np.savez_compressed(config.EMB_FILE, **embeddings)
    n_tracks = len(set(stable.values()))
    log.info("detected %d faces across %d frames -> %d tracks",
             len(records), len(rows), n_tracks)
    return len(records)


if __name__ == "__main__":
    detect()
