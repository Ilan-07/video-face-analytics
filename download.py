"""Phase 1a: Download the source video with yt-dlp."""
import subprocess
import sys
from pathlib import Path

import config


def download(url: str = config.VIDEO_URL) -> Path:
    config.ensure_dirs()
    out_tmpl = str(config.VIDEO_DIR / "video.%(ext)s")

    # Prefer mp4 up to 1080p: more real pixels per face -> more consistent
    # embeddings for the same person across scenes (reduces over-segmentation).
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
        "--merge-output-format", "mp4",
        "-o", out_tmpl,
        url,
    ]
    print(f"[download] {url}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "yt-dlp not available; install it: pip install yt-dlp") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"yt-dlp failed (exit {e.returncode}) for {url}. Common causes: "
            "video removed/private, age-gated, region-blocked, or yt-dlp "
            "outdated (try: pip install -U yt-dlp).") from e

    files = sorted(config.VIDEO_DIR.glob("video.*"))
    if not files:
        raise RuntimeError("Download produced no file")
    video = files[0]
    print(f"[download] saved -> {video}")
    return video


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else config.VIDEO_URL
    download(url)
