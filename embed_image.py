"""Milestone 2e: build a VISUAL search index over the frames with CLIP.

Where embed_text.py embeds each frame's caption+OCR *text*, this embeds the frame
*image* itself with CLIP (config.CLIP_MODEL) into a shared image-text space. A text
query is encoded by the same model at search time, so frames are retrieved by what
they LOOK like -- "a tunnel" matches a tunnel frame even if its caption never says
"tunnel". This makes search robust to caption mistakes (search.visual_search).

Output: data/embeddings/image_embeddings.npz with
    frame_ids   (int32 [N])
    embeddings  (float32 [N, D], L2-normalized)
    timestamps  (float32 [N])
    model       (str scalar)
Vectors are normalized, so cosine similarity is a plain dot product.
"""
import numpy as np
from PIL import Image
from tqdm import tqdm

import config
import util

log = util.get_logger()


def _pick_device():
    import torch
    pref = config.CLIP_DEVICE
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    """Load the CLIP model (lazy import so torch isn't pulled in until a visual
    search or index build actually needs it)."""
    from sentence_transformers import SentenceTransformer

    device = _pick_device()
    log.info("loading CLIP model %s on %s", config.CLIP_MODEL, device)
    return SentenceTransformer(config.CLIP_MODEL, device=device)


def embed_query(model, text: str) -> np.ndarray:
    """Encode a text query into the shared CLIP space (L2-normalized)."""
    return model.encode([text], normalize_embeddings=True,
                        convert_to_numpy=True).astype(np.float32)[0]


def run() -> int:
    import pandas as pd

    config.ensure_dirs()
    frames = pd.read_csv(config.FRAMES_CSV)
    model = load_model()

    frame_ids, timestamps, vecs = [], [], []
    batch, batch_meta = [], []
    bs = 32

    def flush():
        if not batch:
            return
        emb = model.encode(batch, batch_size=bs, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        vecs.append(emb.astype(np.float32))
        for fid, ts in batch_meta:
            frame_ids.append(fid)
            timestamps.append(ts)
        for im in batch:
            im.close()
        batch.clear()
        batch_meta.clear()

    log.info("CLIP-embedding %d frame images", len(frames))
    for r in tqdm(frames.itertuples(index=False), total=len(frames),
                  desc="clip"):
        try:
            img = Image.open(config.FRAME_DIR / r.filename).convert("RGB")
        except (OSError, FileNotFoundError):
            log.warning("could not read frame %s", r.filename)
            continue
        batch.append(img)
        batch_meta.append((int(r.frame_id), float(r.timestamp_sec)))
        if len(batch) >= bs:
            flush()
    flush()

    emb = np.vstack(vecs) if vecs else np.zeros((0, 512), np.float32)
    np.savez(config.IMAGE_EMB_FILE,
             frame_ids=np.array(frame_ids, dtype=np.int32),
             embeddings=emb,
             timestamps=np.array(timestamps, dtype=np.float32),
             model=np.array(config.CLIP_MODEL))
    log.info("visual index: %d x %d -> %s",
             emb.shape[0], emb.shape[1], config.IMAGE_EMB_FILE.name)
    return len(frame_ids)


if __name__ == "__main__":
    run()
