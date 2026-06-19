"""Phase 1b: extract frames at ~FPS using supervision's video utilities.

Uses sv.VideoInfo for the true source FPS and sv.get_video_frames_generator
with a computed stride, so timestamps are derived from the real frame index.
"""
import csv
from pathlib import Path

import cv2
import supervision as sv

import config
import util

log = util.get_logger()


def find_video() -> Path:
    files = sorted(config.VIDEO_DIR.glob("video.*"))
    if not files:
        raise FileNotFoundError("No video found. Run download.py first.")
    return files[0]


def extract(target_fps: int = config.FPS) -> int:
    config.ensure_dirs()
    video = find_video()

    info = sv.VideoInfo.from_video_path(str(video))
    stride = max(1, round(info.fps / target_fps))
    log.info("source %.3f fps, %dx%d -> stride %d (~%.2f fps)",
             info.fps, info.width, info.height, stride, info.fps / stride)

    frames = sv.get_video_frames_generator(str(video), stride=stride)
    with open(config.FRAMES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "filename", "timestamp_sec"])
        n = 0
        for i, frame in enumerate(frames):
            name = f"frame_{i:06d}.jpg"
            cv2.imwrite(str(config.FRAME_DIR / name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            ts = (i * stride) / info.fps
            w.writerow([i, name, f"{ts:.3f}"])
            n = i + 1

    log.info("extracted %d frames -> %s", n, config.FRAMES_CSV.name)
    return n


if __name__ == "__main__":
    extract()
