"""Optional clothing/body appearance re-ID for cross-scene linking (scaffold).

Face embeddings at 1 FPS can't link the same person across big pose/scene changes.
Clothing/body appearance is an *orthogonal* signal: in a short video a person keeps
the same outfit, regardless of face angle. For each detected face we crop the body
region below it, embed it with a person-ReID model (torchreid OSNet), and average
per track. recognize.py then fuses these (config.APPEARANCE_ENABLE) so two clusters
may merge when CLOTHING agrees even if the face match is only loose.

Scaffold status: body cropping + per-track templating + save/load are implemented
and dependency-free; the ReID embedder lazy-imports torchreid only when enabled, so
the core pipeline carries no extra dependency. Enable with:
    pip install torchreid
    # config.APPEARANCE_ENABLE = True
    .venv/bin/python appearance.py        # then re-run recognize
"""
import numpy as np
import pandas as pd

import config
import util

log = util.get_logger()
_extractor = None


def _get_extractor():
    """Lazy person-ReID feature extractor (torchreid OSNet), CPU."""
    global _extractor
    if _extractor is None:
        try:  # import path differs across torchreid versions
            try:
                from torchreid.reid.utils import FeatureExtractor
            except ImportError:
                from torchreid.utils import FeatureExtractor
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "appearance re-ID needs torchreid: pip install torchreid") from e
        _extractor = FeatureExtractor(model_name="osnet_x1_0", device="cpu")
    return _extractor


def body_box(x1, y1, x2, y2, w, h):
    """Clothing-focused torso crop BELOW the chin, clamped to a (w x h) frame.

    Pure + dependency-free so it is unit-testable. Excludes the head (that's the
    face) and is narrowed to BODY_W_SCALE x face width to suppress side-background,
    extending BODY_DOWN_SCALE face-heights below the chin to capture clothing.
    """
    fw, fh = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2.0
    bw = config.BODY_W_SCALE * fw
    bx1 = max(0, int(cx - bw / 2))
    bx2 = min(w, int(cx + bw / 2))
    by1 = max(0, int(y2))                              # start at the chin
    by2 = min(h, int(y2 + config.BODY_DOWN_SCALE * fh))
    return bx1, by1, bx2, by2


def compute() -> int:
    """Embed a body crop per detected face; save L2-normed per-track templates."""
    import cv2
    faces = pd.read_csv(config.FACES_CSV)
    frames = (pd.read_csv(config.FRAMES_CSV)
              .set_index("frame_id")["filename"].to_dict())
    ext = _get_extractor()

    by_track: dict[str, list] = {}
    for fid, g in faces.groupby("frame_id"):
        img = cv2.imread(str(config.FRAME_DIR / frames[int(fid)]))
        if img is None:
            continue
        h, w = img.shape[:2]
        for r in g.itertuples():
            bx1, by1, bx2, by2 = body_box(r.x1, r.y1, r.x2, r.y2, w, h)
            crop = img[by1:by2, bx1:bx2]
            if crop.size == 0:
                continue
            vec = ext(crop)[0].detach().cpu().numpy().astype(np.float32)
            by_track.setdefault(str(r.track_id), []).append(
                vec / (np.linalg.norm(vec) + 1e-9))

    templates = {}
    for t, vs in by_track.items():
        v = np.mean(vs, axis=0)
        templates[t] = (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)
    np.savez_compressed(config.APPEARANCE_FILE, **templates)
    log.info("appearance templates -> %s (%d tracks)",
             config.APPEARANCE_FILE.name, len(templates))
    return len(templates)


def load_templates():
    """Return {track_id: normalized appearance vector} or None if not computed."""
    if not config.APPEARANCE_FILE.exists():
        return None
    d = np.load(config.APPEARANCE_FILE)
    return {k: d[k] for k in d.keys()}


if __name__ == "__main__":
    compute()
