"""Milestone 2 Task 2: search the frame metadata repository.

Reusable core (used by the Streamlit app and the tests) plus a small CLI:

    python search.py "Welcome"            # search OCR text
    python search.py "city" --captions    # also search captions

Returns every frame whose text matches the query, with timestamp and frame path
so callers can display the corresponding frames.
"""
import argparse
import json

import pandas as pd

import config


def load_metadata() -> pd.DataFrame:
    """Load frame_metadata.json into a DataFrame (face_ids kept as lists)."""
    with open(config.METADATA_JSON) as f:
        records = json.load(f)
    return pd.DataFrame.from_records(records)


def fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:d}:{s:02d}"


def _snippet(text: str, query: str, width: int = 60) -> str:
    """A short window of `text` around the first match of `query`."""
    lo = text.lower().find(query.lower())
    if lo < 0:
        return text[:width]
    start = max(0, lo - width // 3)
    end = min(len(text), lo + len(query) + width // 2)
    snip = text[start:end].strip()
    return ("…" if start else "") + snip + ("…" if end < len(text) else "")


def search(query: str, df: pd.DataFrame | None = None,
           fields: tuple[str, ...] = ("ocr_text",)) -> pd.DataFrame:
    """Case-insensitive substring search over the given text fields.

    Returns a DataFrame with frame_id, timestamp_sec, mm:ss, filename,
    frame_path, face_ids, field (which field matched) and snippet -- ordered by
    timestamp. An empty query yields no rows."""
    if df is None:
        df = load_metadata()
    cols = ["frame_id", "timestamp_sec", "mmss", "filename",
            "frame_path", "face_ids", "field", "snippet"]
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(columns=cols)

    ql = q.lower()
    hits = []
    for r in df.itertuples(index=False):
        for field in fields:
            text = getattr(r, field, "") or ""
            if ql in str(text).lower():
                hits.append({
                    "frame_id": int(r.frame_id),
                    "timestamp_sec": float(r.timestamp_sec),
                    "mmss": fmt_ts(float(r.timestamp_sec)),
                    "filename": r.filename,
                    "frame_path": str(config.FRAME_DIR / r.filename),
                    "face_ids": list(r.face_ids),
                    "field": field,
                    "snippet": _snippet(str(text), q),
                })
                break   # one row per frame even if multiple fields match
    out = pd.DataFrame(hits, columns=cols)
    return out.sort_values("timestamp_sec").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Search frame OCR text / captions")
    ap.add_argument("query", help="word or phrase to search for")
    ap.add_argument("--captions", action="store_true",
                    help="also search generated captions, not just OCR text")
    args = ap.parse_args()

    fields = ("ocr_text", "caption") if args.captions else ("ocr_text",)
    res = search(args.query, fields=fields)
    if res.empty:
        print(f'No frames matched "{args.query}".')
        return
    print(f'{len(res)} frame(s) matched "{args.query}":\n')
    for r in res.itertuples(index=False):
        faces = ", ".join(r.face_ids) if r.face_ids else "-"
        print(f"  [{r.mmss:>6}]  {r.filename}  (faces: {faces})")
        print(f"           {r.field}: {r.snippet}")
        print(f"           {r.frame_path}")


if __name__ == "__main__":
    main()
