"""Milestone 2d: build a semantic text-search index over the frames.

Each frame's searchable text -- its caption joined with its OCR text -- is encoded
with a compact sentence-transformer (config.TEXT_EMBED_MODEL). Queries are encoded
the same way at search time, so frames are retrieved by MEANING: searching "train"
surfaces a caption that says "subway", which substring/fuzzy search cannot do.

Output: data/embeddings/text_embeddings.npz with
    frame_ids   (int32 [N])
    embeddings  (float32 [N, D], L2-normalized)
    timestamps  (float32 [N])
    model       (str scalar)
The vectors are normalized, so cosine similarity is a plain dot product.
"""
import numpy as np

import config
import util

log = util.get_logger()


def _pick_device():
    import torch
    pref = config.TEXT_EMBED_DEVICE
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    """Load the sentence-transformer (lazy import so torch isn't pulled in until
    semantic search is actually used)."""
    from sentence_transformers import SentenceTransformer

    device = _pick_device()
    log.info("loading text-embedding model %s on %s",
             config.TEXT_EMBED_MODEL, device)
    return SentenceTransformer(config.TEXT_EMBED_MODEL, device=device)


def frame_document(caption: str, ocr_text: str) -> str:
    """The text embedded per frame: caption + OCR text (scene meaning + literal
    on-screen text). Empty parts are dropped."""
    parts = [p.strip() for p in (caption, ocr_text) if p and str(p).strip()]
    return ". ".join(parts)


def embed_texts(model, texts: list[str]) -> np.ndarray:
    """Encode texts to L2-normalized float32 vectors."""
    return model.encode(texts, batch_size=64, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False
                        ).astype(np.float32)


def run() -> int:
    import pandas as pd

    config.ensure_dirs()
    meta = pd.read_csv(config.METADATA_CSV).fillna("")
    docs = [frame_document(c, t)
            for c, t in zip(meta["caption"], meta["ocr_text"])]

    model = load_model()
    log.info("embedding %d frame documents", len(docs))
    emb = embed_texts(model, docs)

    np.savez(config.TEXT_EMB_FILE,
             frame_ids=meta["frame_id"].to_numpy(dtype=np.int32),
             embeddings=emb,
             timestamps=meta["timestamp_sec"].to_numpy(dtype=np.float32),
             model=np.array(config.TEXT_EMBED_MODEL))
    log.info("semantic index: %d x %d -> %s",
             emb.shape[0], emb.shape[1], config.TEXT_EMB_FILE.name)
    return len(docs)


if __name__ == "__main__":
    run()
