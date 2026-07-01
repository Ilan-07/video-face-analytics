"""Milestone 2b: generate a descriptive caption for every extracted frame.

Uses Salesforce BLIP-base (config.CAPTION_MODEL) via transformers. The model is
loaded once and frames are streamed one at a time so memory stays flat on an 8GB
M2 (Apple MPS when available, else CPU). Output: data/captions.csv, one row per
frame, carrying the Face IDs and OCR text for that frame as the spec requires
(Milestone 2 Task 3).

Resumable: rows already present in captions.csv are reused, so a crashed run
continues instead of re-captioning from scratch -- and a pure metadata/schema
refresh costs no model time. Use --restart to recaption everything.

Run `python caption.py --limit 5` for a quick smoke test before the full pass.
"""
import argparse
import csv

import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
import util
from build_metadata import caption_echoes_text

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
    device). Imported lazily so callers that reuse cached captions never pay the
    torch import / model-load cost."""
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


def _caption_batch(proc, model, device, images, prompt=None) -> list[str]:
    text = (config.CAPTION_PROMPT if prompt is None else prompt) or None
    if text:
        inputs = proc(images=images, text=[text] * len(images),
                      return_tensors="pt").to(device)
    else:
        inputs = proc(images=images, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=config.CAPTION_MAX_TOKENS)
    return [proc.decode(o, skip_special_tokens=True).strip() for o in out]


def _fix_echoes(captions: dict[int, str], ocr_map: dict[int, str],
                get_model, fname: dict[int, str]) -> int:
    """Repair captions that merely echo a title card's OCR text.

    Re-captions each echoing frame with CAPTION_RECAPTION_PROMPT to bias the model
    toward describing the scene; if the result STILL echoes (a pure text card with
    no scene), falls back to the frame's clean OCR text. Mutates `captions` in
    place, returns the number of frames repaired."""
    echo_ids = [fid for fid, cap in captions.items()
                if caption_echoes_text(cap, ocr_map.get(fid, ""))]
    if not echo_ids:
        return 0
    log.info("echo-fix: re-captioning %d title-card frame(s) (prompt=%r)",
             len(echo_ids), config.CAPTION_RECAPTION_PROMPT)
    proc, model, device = get_model()
    n_recap = n_fallback = 0
    for fid in tqdm(echo_ids, desc="echo-fix"):
        try:
            img = Image.open(config.FRAME_DIR / fname[fid]).convert("RGB")
        except (OSError, FileNotFoundError, KeyError):
            continue
        recap = _caption_batch(proc, model, device, [img],
                               prompt=config.CAPTION_RECAPTION_PROMPT)[0]
        img.close()
        # BLIP echoes the conditioning prompt back at the start; drop it so the
        # stored caption reads naturally ("a black background..." not "a photo of
        # a black background...").
        prefix = config.CAPTION_RECAPTION_PROMPT.strip()
        if prefix and recap.lower().startswith(prefix.lower()):
            recap = recap[len(prefix):].strip() or recap
        ocr = ocr_map.get(fid, "")
        if caption_echoes_text(recap, ocr):
            # Pure title card: describe it AS a text card and carry the clean OCR
            # text. The wrapper keeps token-overlap below the echo threshold, so
            # this repaired caption is not re-flagged (or re-fixed) on a rerun.
            captions[fid] = f"a title card that reads: {ocr}"
            n_fallback += 1
        else:
            captions[fid] = recap          # improved scene description
            n_recap += 1
    log.info("echo-fix: %d re-captioned, %d fell back to OCR text",
             n_recap, n_fallback)
    return n_recap + n_fallback


def _existing_captions(restart: bool) -> dict[int, str]:
    """frame_id -> caption already computed (for resume), unless --restart."""
    if restart or not config.CAPTIONS_CSV.exists():
        return {}
    try:
        prev = pd.read_csv(config.CAPTIONS_CSV)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return {}
    if "caption" not in prev.columns:
        return {}
    return {int(r.frame_id): ("" if pd.isna(r.caption) else str(r.caption))
            for r in prev.itertuples(index=False)}


def _ocr_text_by_frame() -> dict[int, str]:
    if not config.OCR_CSV.exists():
        return {}
    ocr = pd.read_csv(config.OCR_CSV)
    return {int(r.frame_id): ("" if pd.isna(r.text) else str(r.text))
            for r in ocr.itertuples(index=False)}


def run(limit: int | None = None, restart: bool = False) -> int:
    config.ensure_dirs()
    frames = pd.read_csv(config.FRAMES_CSV)
    if limit:
        frames = frames.head(limit)

    cached = _existing_captions(restart)
    face_map = util.face_ids_by_frame()
    ocr_map = _ocr_text_by_frame()

    rows = list(frames.itertuples(index=False))
    todo = [r for r in rows if int(r.frame_id) not in cached]
    log.info("captioning %d frames (%d cached, %d to generate; prompt=%r)",
             len(rows), len(rows) - len(todo), len(todo), config.CAPTION_PROMPT)

    captions = dict(cached)
    fname = {int(r.frame_id): r.filename for r in rows}
    _state = {}                       # lazily-loaded (proc, model, device)

    def get_model():
        if not _state:
            proc, model, device = _load_model()
            _state.update(proc=proc, model=model, device=device)
        return _state["proc"], _state["model"], _state["device"]

    if todo:
        proc, model, device = get_model()
        bs = max(1, config.CAPTION_BATCH)
        for i in tqdm(range(0, len(todo), bs), desc="caption"):
            chunk = todo[i:i + bs]
            imgs, keep = [], []
            for r in chunk:
                try:
                    imgs.append(Image.open(
                        config.FRAME_DIR / r.filename).convert("RGB"))
                    keep.append(r)
                except (OSError, FileNotFoundError):
                    log.warning("could not read frame %s", r.filename)
                    captions[int(r.frame_id)] = ""
            if not imgs:
                continue
            for r, cap in zip(keep, _caption_batch(proc, model, device, imgs)):
                captions[int(r.frame_id)] = cap
            for im in imgs:
                im.close()

    # Repair title-card echoes (loads the model on demand if it wasn't already).
    if config.CAPTION_ECHO_FIX:
        _fix_echoes(captions, ocr_map, get_model, fname)

    n = 0
    with open(config.CAPTIONS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "timestamp_sec", "face_ids",
                    "ocr_text", "caption"])
        for r in rows:
            fid = int(r.frame_id)
            faces = "|".join(face_map.get(fid, []))
            w.writerow([fid, f"{r.timestamp_sec:.3f}", faces,
                        ocr_map.get(fid, ""), captions.get(fid, "")])
            n += 1

    log.info("captioning done: %d frames -> %s", n, config.CAPTIONS_CSV.name)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BLIP frame captioning (Milestone 2)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only caption the first N frames (smoke test)")
    ap.add_argument("--restart", action="store_true",
                    help="ignore cached captions and recaption every frame")
    args = ap.parse_args()
    run(limit=args.limit, restart=args.restart)
