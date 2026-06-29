"""Milestone 2 Task 2: search the frame metadata repository.

Reusable core (used by the Streamlit app and the tests) plus a small CLI:

    python search.py "Welcome"              # substring search over OCR text
    python search.py "city" --captions      # also search captions
    python search.py "embankment" --fuzzy   # tolerate OCR typos
    python search.py "Line \\d+" --regex     # regex search
    python search.py "underground train" --semantic   # search by meaning
    python search.py "Welcome" --open       # open matching frames (macOS)

The lexical modes (substring/regex/fuzzy) return every frame whose text matches,
with timestamp and frame path. `--semantic` instead ranks all frames by the cosine
similarity of their caption+OCR embedding to the query embedding (built by
embed_text.py), so it retrieves by meaning -- "train" surfaces a "subway" caption.
Consecutive frames with the same matched text are collapsible into time ranges
(group_consecutive), so a title card shown for six seconds reads as one hit.
"""
import argparse
import difflib
import json
import re
import subprocess
import sys

import numpy as np
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


def _fuzzy_hit(text: str, query: str, cutoff: float) -> bool:
    """True if any whitespace token of `text` is ~similar to the query (or any
    query word), tolerating OCR typos like 'Extravaganza' -> 'extrana'."""
    words = text.lower().split()
    if not words:
        return False
    for q in query.lower().split() or [query.lower()]:
        if difflib.get_close_matches(q, words, n=1, cutoff=cutoff):
            return True
    return False


def _match(text: str, query: str, *, regex, fuzzy, fuzzy_cutoff):
    """Return (matched: bool, score: int). score = occurrence count (relevance)."""
    text = str(text or "")
    if not text:
        return False, 0
    if regex:
        hits = re.findall(query, text, flags=re.IGNORECASE)
        return (len(hits) > 0, len(hits))
    n = text.lower().count(query.lower())
    if n:
        return True, n
    if fuzzy and _fuzzy_hit(text, query, fuzzy_cutoff):
        return True, 1
    return False, 0


def search(query: str, df: pd.DataFrame | None = None,
           fields: tuple[str, ...] = ("ocr_text",), *,
           regex: bool = False, fuzzy: bool = False,
           fuzzy_cutoff: float = 0.8) -> pd.DataFrame:
    """Search the given text fields of the metadata repository.

    Case-insensitive substring by default; `regex` switches to regex search and
    `fuzzy` adds an edit-distance fallback for OCR typos. Returns frame_id,
    timestamp_sec, mm:ss, filename, frame_path, face_ids, field (which field
    matched), score (occurrence count) and snippet -- ordered by timestamp. An
    empty query yields no rows."""
    if df is None:
        df = load_metadata()
    cols = ["frame_id", "timestamp_sec", "mmss", "filename", "frame_path",
            "face_ids", "field", "score", "snippet"]
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(columns=cols)

    hits = []
    for r in df.itertuples(index=False):
        for field in fields:
            text = getattr(r, field, "") or ""
            matched, score = _match(text, q, regex=regex, fuzzy=fuzzy,
                                    fuzzy_cutoff=fuzzy_cutoff)
            if matched:
                hits.append({
                    "frame_id": int(r.frame_id),
                    "timestamp_sec": float(r.timestamp_sec),
                    "mmss": fmt_ts(float(r.timestamp_sec)),
                    "filename": r.filename,
                    "frame_path": str(config.FRAME_DIR / r.filename),
                    "face_ids": list(r.face_ids),
                    "field": field,
                    "score": int(score),
                    "snippet": _snippet(str(text), q),
                })
                break   # one row per frame even if multiple fields match
    out = pd.DataFrame(hits, columns=cols)
    return out.sort_values("timestamp_sec").reset_index(drop=True)


def group_consecutive(results: pd.DataFrame, max_gap: float | None = None):
    """Collapse consecutive same-text frames into time ranges (Milestone 2:
    cleaner 'list of timestamps'). Returns a list of dicts with start/end mm:ss,
    frame count, representative snippet/field/face_ids and the first frame path.
    `max_gap` is the max seconds between frames still considered contiguous
    (defaults to ~2 sampling intervals)."""
    if results.empty:
        return []
    gap = max_gap if max_gap is not None else 2.0 / max(config.FPS, 1) + 0.5
    groups, cur = [], None
    for r in results.itertuples(index=False):
        same = (cur is not None and r.snippet == cur["snippet"]
                and r.field == cur["field"]
                and r.timestamp_sec - cur["_last"] <= gap)
        if same:
            cur["_last"] = r.timestamp_sec
            cur["end"] = r.mmss
            cur["frames"] += 1
        else:
            if cur:
                groups.append(cur)
            cur = {"start": r.mmss, "end": r.mmss, "_last": r.timestamp_sec,
                   "frames": 1, "field": r.field, "snippet": r.snippet,
                   "face_ids": list(r.face_ids),
                   "frame_path": r.frame_path, "score": int(r.score)}
    if cur:
        groups.append(cur)
    for g in groups:
        g.pop("_last", None)
    return groups


def _topk_similar(qvec: np.ndarray, mat: np.ndarray, k: int):
    """Cosine similarity of a normalized query vector against normalized rows of
    `mat`; return (indices, scores) of the top-k, highest first. Pure -- unit
    tested with synthetic vectors, no model needed."""
    sims = mat @ qvec
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


_ST_MODEL = None


def _get_embed_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        import embed_text
        _ST_MODEL = embed_text.load_model()
    return _ST_MODEL


def semantic_search(query: str, df: pd.DataFrame | None = None,
                    top_k: int | None = None,
                    min_score: float | None = None) -> pd.DataFrame:
    """Rank frames by semantic similarity of their caption+OCR text to the query.

    Loads the embedding index built by embed_text.py and the same model used to
    build it. Returns the same columns as search(), with `score` = cosine
    similarity (0-1) and `field` = "semantic", highest first."""
    import embed_text

    top_k = config.SEMANTIC_TOP_K if top_k is None else top_k
    min_score = config.SEMANTIC_MIN_SCORE if min_score is None else min_score
    cols = ["frame_id", "timestamp_sec", "mmss", "filename", "frame_path",
            "face_ids", "field", "score", "snippet"]
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(columns=cols)
    if df is None:
        df = load_metadata()

    data = np.load(config.TEXT_EMB_FILE, allow_pickle=True)
    mat, frame_ids = data["embeddings"], data["frame_ids"]
    qvec = embed_text.embed_texts(_get_embed_model(), [q])[0]
    idx, scores = _topk_similar(qvec, mat, top_k)

    by_id = {int(r.frame_id): r for r in df.itertuples(index=False)}
    rows = []
    for i, sc in zip(idx, scores):
        if sc < min_score:
            continue
        r = by_id.get(int(frame_ids[i]))
        if r is None:
            continue
        doc = embed_text.frame_document(getattr(r, "caption", ""),
                                        getattr(r, "ocr_text", ""))
        rows.append({
            "frame_id": int(r.frame_id),
            "timestamp_sec": float(r.timestamp_sec),
            "mmss": fmt_ts(float(r.timestamp_sec)),
            "filename": r.filename,
            "frame_path": str(config.FRAME_DIR / r.filename),
            "face_ids": list(r.face_ids),
            "field": "semantic",
            "score": round(float(sc), 3),
            "snippet": (doc[:80] + "…") if len(doc) > 80 else doc,
        })
    return pd.DataFrame(rows, columns=cols)


def main():
    ap = argparse.ArgumentParser(description="Search frame OCR text / captions")
    ap.add_argument("query", help="word or phrase to search for")
    ap.add_argument("--captions", action="store_true",
                    help="also search generated captions, not just OCR text")
    ap.add_argument("--regex", action="store_true", help="treat query as a regex")
    ap.add_argument("--fuzzy", action="store_true",
                    help="fall back to fuzzy matching for OCR typos")
    ap.add_argument("--semantic", action="store_true",
                    help="rank frames by meaning using text embeddings")
    ap.add_argument("--top-k", type=int, default=None,
                    help="max results for --semantic (default config.SEMANTIC_TOP_K)")
    ap.add_argument("--no-group", action="store_true",
                    help="list every matching frame instead of time ranges")
    ap.add_argument("--open", action="store_true", dest="open_frames",
                    help="open the matching frames (one per range) in the viewer")
    args = ap.parse_args()

    if args.semantic:
        res = semantic_search(args.query, top_k=args.top_k)
        if res.empty:
            print(f'No frames semantically matched "{args.query}".')
            return
        print(f'Top {len(res)} semantic match(es) for "{args.query}":\n')
        for r in res.itertuples(index=False):
            faces = ", ".join(r.face_ids) if r.face_ids else "-"
            print(f"  [{r.mmss:>6}]  score={r.score:.3f}  {r.filename}  "
                  f"(faces: {faces})")
            print(f"           {r.snippet}")
        if args.open_frames:
            opener = {"darwin": "open", "linux": "xdg-open"}.get(
                sys.platform, "open")
            for r in res.itertuples(index=False):
                subprocess.run([opener, r.frame_path], check=False)
        return

    fields = ("ocr_text", "caption") if args.captions else ("ocr_text",)
    res = search(args.query, fields=fields, regex=args.regex, fuzzy=args.fuzzy)
    if res.empty:
        print(f'No frames matched "{args.query}".')
        return

    if args.no_group:
        print(f'{len(res)} frame(s) matched "{args.query}":\n')
        for r in res.itertuples(index=False):
            faces = ", ".join(r.face_ids) if r.face_ids else "-"
            print(f"  [{r.mmss:>6}]  {r.filename}  (faces: {faces})")
            print(f"           {r.field}: {r.snippet}")
        return

    groups = group_consecutive(res)
    print(f'{len(res)} frame(s) in {len(groups)} time range(s) matched '
          f'"{args.query}":\n')
    for g in groups:
        span = g["start"] if g["start"] == g["end"] else f"{g['start']}–{g['end']}"
        faces = ", ".join(g["face_ids"]) if g["face_ids"] else "-"
        print(f"  [{span:>11}]  ({g['frames']} frame(s); faces: {faces})")
        print(f"               {g['field']}: {g['snippet']}")

    if args.open_frames:
        opener = {"darwin": "open", "linux": "xdg-open"}.get(
            sys.platform, "open")
        for g in groups:
            subprocess.run([opener, g["frame_path"]], check=False)


if __name__ == "__main__":
    main()
