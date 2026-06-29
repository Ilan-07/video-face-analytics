"""Milestone 2b: generate a descriptive caption for every extracted frame.

Uses Salesforce BLIP-base (config.CAPTION_MODEL) via transformers. The model is
loaded once and frames are streamed one at a time so memory stays flat on an 8GB
M2 (Apple MPS when available, else CPU). Output: data/captions.csv, one row per
frame. First run downloads the model (~990MB) and caches it under ~/.cache.

Run `python caption.py --limit 5` for a quick smoke test before the full pass.
"""
import argparse
import csv

import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
import util

log = util.get_logger()


def _pick_device():
    import torch
    pref = config.CAPTION_DEVICE
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model():
    """Load BLIP processor + model onto the chosen device. Returns (proc, model,
    device). Imported lazily so the rest of the pipeline never pays the torch
    import cost."""
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = _pick_device()
    log.info("loading caption model %s on %s", config.CAPTION_MODEL, device)
    proc = BlipProcessor.from_pretrained(config.CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(config.CAPTION_MODEL)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    return proc, model, device


def _caption_batch(proc, model, device, images) -> list[str]:
    inputs = proc(images=images, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=config.CAPTION_MAX_TOKENS)
    return [proc.decode(o, skip_special_tokens=True).strip() for o in out]


def run(limit: int | None = None) -> int:
    config.ensure_dirs()
    frames = pd.read_csv(config.FRAMES_CSV)
    if limit:
        frames = frames.head(limit)
    proc, model, device = _load_model()
    log.info("captioning %d frames (batch=%d, max_tokens=%d)",
             len(frames), config.CAPTION_BATCH, config.CAPTION_MAX_TOKENS)

    rows = list(frames.itertuples(index=False))
    bs = max(1, config.CAPTION_BATCH)
    n = 0
    with open(config.CAPTIONS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "timestamp_sec", "caption"])
        for i in tqdm(range(0, len(rows), bs), desc="caption"):
            chunk = rows[i:i + bs]
            imgs, keep = [], []
            for r in chunk:
                try:
                    imgs.append(Image.open(
                        config.FRAME_DIR / r.filename).convert("RGB"))
                    keep.append(r)
                except (OSError, FileNotFoundError):
                    log.warning("could not read frame %s", r.filename)
                    w.writerow([r.frame_id, f"{r.timestamp_sec:.3f}", ""])
            if not imgs:
                continue
            caps = _caption_batch(proc, model, device, imgs)
            for r, cap in zip(keep, caps):
                w.writerow([r.frame_id, f"{r.timestamp_sec:.3f}", cap])
                n += 1
            for im in imgs:
                im.close()

    log.info("captioning done: %d frames -> %s", n, config.CAPTIONS_CSV.name)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BLIP frame captioning (Milestone 2)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only caption the first N frames (smoke test)")
    args = ap.parse_args()
    run(limit=args.limit)
