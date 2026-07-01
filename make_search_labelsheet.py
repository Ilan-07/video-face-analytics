"""Milestone 2 eval: sample frames into a hand-labeling sheet for OCR & captions.

Draws a stratified sample of frames -- some with OCR text, some without, plus a
few caption-echo (title-card) frames -- and writes them to SEARCH_LABELS_CSV with
the model's predictions alongside two blank columns for a human to fill:

    true_ocr_text     what the on-screen text ACTUALLY says (blank = no text)
    caption_score     caption adequacy, 1 (wrong) .. 5 (perfect)

eval_search.py then turns those labels into OCR precision/recall and a mean
caption-adequacy score. Existing labels are preserved: re-running keeps any row
whose frame_id is already in the sheet (with its filled-in labels) and only tops
up the sample if rows are missing.
"""
import csv

import pandas as pd

import config
import util

log = util.get_logger()

COLUMNS = ["frame_id", "timestamp_sec", "filename", "pred_ocr_text",
           "pred_caption", "caption_echoes_text",
           "true_ocr_text", "caption_score", "notes"]


def _existing() -> dict[int, dict]:
    """frame_id -> already-written row (so human labels are never clobbered)."""
    if not config.SEARCH_LABELS_CSV.exists():
        return {}
    prev = pd.read_csv(config.SEARCH_LABELS_CSV, dtype=str).fillna("")
    return {int(r["frame_id"]): dict(r) for _, r in prev.iterrows()}


def _sample(meta: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Stratified sample: ~40% with OCR text, ~20% echo frames, rest without."""
    has_text = meta[meta["ocr_text"].str.len() > 0]
    echo = meta[meta["caption_echoes_text"] == True]            # noqa: E712
    no_text = meta[meta["ocr_text"].str.len() == 0]

    n_text = min(len(has_text), round(n * 0.4))
    n_echo = min(len(echo), round(n * 0.2))
    n_none = max(0, n - n_text - n_echo)

    picks = pd.concat([
        has_text.sample(n_text, random_state=seed),
        echo.sample(n_echo, random_state=seed + 1),
        no_text.sample(min(len(no_text), n_none), random_state=seed + 2),
    ]).drop_duplicates("frame_id")
    return picks.sort_values("frame_id")


def run(n: int | None = None) -> int:
    config.ensure_dirs()
    n = n or config.SEARCH_LABEL_SAMPLE
    meta = pd.read_csv(config.METADATA_CSV).fillna("")
    meta["ocr_text"] = meta["ocr_text"].astype(str).str.strip()

    existing = _existing()
    picks = _sample(meta, n)
    # keep every already-sampled frame, then top up to n with fresh picks
    frame_ids = list(existing.keys())
    for fid in picks["frame_id"].astype(int):
        if fid not in existing and len(frame_ids) < n:
            frame_ids.append(int(fid))

    by_id = {int(r.frame_id): r for r in meta.itertuples(index=False)}
    n_labeled = 0
    with open(config.SEARCH_LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for fid in sorted(frame_ids):
            if fid in existing:                      # preserve human labels
                row = {c: existing[fid].get(c, "") for c in COLUMNS}
                if row.get("true_ocr_text") or row.get("caption_score"):
                    n_labeled += 1
            else:
                r = by_id[fid]
                row = {
                    "frame_id": fid,
                    "timestamp_sec": f"{r.timestamp_sec:.3f}",
                    "filename": r.filename,
                    "pred_ocr_text": str(r.ocr_text).strip(),
                    "pred_caption": r.caption,
                    "caption_echoes_text": r.caption_echoes_text,
                    "true_ocr_text": "", "caption_score": "", "notes": "",
                }
            w.writerow(row)

    log.info("wrote %d frames to %s (%d already labeled). Fill in "
             "true_ocr_text + caption_score (1-5), then run eval_search.py.",
             len(frame_ids), config.SEARCH_LABELS_CSV.name, n_labeled)
    return len(frame_ids)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build OCR/caption labelsheet")
    ap.add_argument("--n", type=int, default=None, help="sample size")
    args = ap.parse_args()
    run(n=args.n)
