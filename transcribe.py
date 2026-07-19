"""New capability: speech transcription with faster-whisper.

Vision-only, 1-FPS analysis had a hard scope ceiling -- everything spoken in the
video was unrecoverable, invisible to both search and narration. This stage
extracts the audio track and transcribes it into timestamped segments, which
build_metadata then joins onto frames on timestamp exactly as it joins OCR,
captions and face IDs. No schema change is needed to accommodate it; speech is
just another per-frame column, so search and the narration digest pick it up for
free.

Graceful by design. If faster-whisper is not installed, ffmpeg is missing, or the
video carries no audio, the stage logs the reason, writes an empty transcript, and
returns 0 -- every downstream stage still runs, and the vision pipeline is exactly
as it was. That mirrors how the narration stages degrade without an API key.

Output: data/transcript.json
    {"model": ..., "language": ..., "status": "ok"|<why-skipped>,
     "segments": [{"start": float, "end": float, "text": str}, ...]}
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import config
import util

log = util.get_logger()


def _find_video():
    vids = sorted(config.VIDEO_DIR.glob("video.*"))
    return vids[0] if vids else None


def _write(payload: dict) -> None:
    util.write_json_atomic(config.TRANSCRIPT_JSON, payload, indent=2)


def _skip(reason: str) -> int:
    log.warning("[transcribe] skipped: %s", reason)
    _write({"model": config.WHISPER_MODEL, "language": None,
            "status": reason, "segments": []})
    return 0


def _extract_audio(video: Path, wav: Path) -> bool:
    """16 kHz mono WAV via ffmpeg -- what Whisper expects. Returns False if the
    stream has no audio or ffmpeg fails, so the caller can skip cleanly."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(video),
           "-vn", "-ac", "1", "-ar", "16000", str(wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not wav.exists() or wav.stat().st_size == 0:
        log.warning("[transcribe] ffmpeg produced no audio: %s",
                    proc.stderr.strip().splitlines()[-1:] or "unknown")
        return False
    return True


def segments_by_frame(segments: list, timestamps: dict) -> dict:
    """frame_id -> the speech spoken at that frame's timestamp.

    A frame carries the text of every transcript segment whose [start, end)
    interval contains its timestamp (usually one). Pure and unit-tested: the
    join is the part that must stay correct as the schema grows."""
    if not segments:
        return {}
    segs = sorted(segments, key=lambda s: s["start"])
    out = {}
    for fid, ts in timestamps.items():
        hits = [s["text"].strip() for s in segs
                if s["start"] <= ts < s["end"] and s["text"].strip()]
        if hits:
            out[int(fid)] = " ".join(hits)
    return out


def run() -> int:
    config.ensure_dirs()
    if not config.WHISPER_ENABLE:
        return _skip("WHISPER_ENABLE is False")
    if not shutil.which("ffmpeg"):
        return _skip("ffmpeg not on PATH (brew install ffmpeg)")
    video = _find_video()
    if video is None:
        return _skip("no video in data/video/")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return _skip("faster-whisper not installed (pip install faster-whisper)")

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        if not _extract_audio(video, wav):
            return _skip("video has no decodable audio track")

        log.info("[transcribe] %s on %s/%s", config.WHISPER_MODEL,
                 config.WHISPER_DEVICE, config.WHISPER_COMPUTE)
        model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
                             compute_type=config.WHISPER_COMPUTE)
        seg_iter, info = model.transcribe(str(wav))
        segments = [{"start": round(s.start, 3), "end": round(s.end, 3),
                     "text": s.text.strip()} for s in seg_iter]

    _write({"model": config.WHISPER_MODEL, "language": info.language,
            "status": "ok", "segments": segments})
    log.info("[transcribe] %d segments (%s) -> %s", len(segments),
             info.language, config.TRANSCRIPT_JSON.name)
    return len(segments)


if __name__ == "__main__":
    run()
