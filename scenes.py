"""Milestone 3a: cut the video into scenes and build the per-scene digest.

Why this stage exists
---------------------
Milestone 3 must narrate the video from its captions, but the captions cannot be
fed to a narrator as-is: 1415 frames yield only 342 unique caption strings, and
BLIP flickers between near-synonyms on visually identical frames ("a train is
pulling passengers" x72, "...in a subway station" x121). Collapsing consecutive
identical captions -- the Milestone 2 `search.group_consecutive` trick -- yields
904 fragments, which is noise, not structure.

So we segment on the IMAGE, not the text. `embed_image.py` already wrote an
L2-normalized CLIP vector per frame, so the adjacent-frame cosine is a single
vectorized dot product and needs no model. Two independent signals agree on this
video: the sharpest cosine drops land exactly on the Underground line title
cards. We therefore cut where

    (a) the adjacent cosine falls below SCENE_SIM_THRESH, OR
    (b) a title card starts or ends  (forced -- see SCENE_TITLE_CARD_FORCE),

then merge away sub-SCENE_MIN_SEC scenes to kill residual flicker.

Two levels come out of this, and they map onto the assignment's two fields:

    chapter_index / chapter_label  ->  "story_segment"  (12: intro + 11 lines)
    scene_index                    ->  "scene_index"    (~30, finer CLIP cuts)

Output: data/scenes.json -- one record per scene, holding the timestamps, the
representative keyframe (CLIP medoid), the face IDs present, the distinct OCR
strings and the distinct captions. This digest is ~30 records and a couple of
thousand tokens, which is what makes single-pass narration possible downstream.
"""
import json
import re
from collections import Counter

import numpy as np

import config
import util
from search import fmt_ts

log = util.get_logger()

# A title card in this video is a full-screen black frame naming an Underground
# line. Neither signal alone finds all 12 of them, so we take the union:
#   * the caption rule misses "Piccadilly Line", whose caption was rewritten by
#     Milestone 2's echo-fix pass into "a title card that reads: ...";
#   * the OCR rule misses the intro card, whose text ("London Underground
#     Extravaganza All Lines! Tuesday November") does not end in "Line".
# The trailing `Line$` anchor is load-bearing: it excludes the ~20 platform
# signage rows at 18:25-19:40 ("Victoria line northbound platform Walthamstow")
# that a naive /\bline/ would swallow, which would shatter the Victoria chapter.
_LINE_CARD = re.compile(r"[A-Z][A-Za-z& ]{2,28}\s+Line\s*$")


def is_title_card(ocr_text: str, caption: str) -> bool:
    """True when the frame is a full-screen line-name card rather than footage.

    Pure -- unit tested against the real edge cases (Piccadilly, the intro card,
    and the Victoria-line platform signage that must NOT match)."""
    ocr = (ocr_text or "").strip()
    cap = (caption or "").strip().lower()
    if _LINE_CARD.match(ocr):
        return True
    # BLIP describes every card as a black screen ("a black background with the
    # words jubilee line", "the word metropolitan line on a black background");
    # the echo-fix pass rewrote the rest as "a title card that reads: ...".
    return "black background" in cap or cap.startswith("a title card that reads")


def chapter_label(ocr_texts) -> str:
    """Human label for the chapter a title card opens ("Jubilee Line").

    Takes EVERY OCR string in the card's frames, not just the first: the
    Piccadilly card reads "picdily line - screenshote - screenshote" on some
    frames and "Piccadilly Line" on others, and picking frame 0 blind would
    silently mislabel that chapter "Introduction". Pure."""
    for text in ocr_texts:
        ocr = re.sub(r"\s+", " ", (text or "").strip())
        if _LINE_CARD.match(ocr):
            return ocr
    return "Introduction"


def cut_indices(sim: np.ndarray, is_card: list, thresh: float,
                force_cards: bool = True) -> list:
    """Frame indices that START a new scene, given adjacent cosines `sim`
    (len N-1, sim[i] = cos(frame i, frame i+1)) and a per-frame card flag.

    A cut at i means frame i opens a scene. Card runs get a cut on BOTH edges so
    a title card is its own scene rather than being absorbed into the footage
    either side of it. Pure -- no I/O, unit tested with synthetic arrays."""
    n = len(is_card)
    cuts = {0}
    for i in range(1, n):
        if sim[i - 1] < thresh:
            cuts.add(i)
        if force_cards and is_card[i] != is_card[i - 1]:
            cuts.add(i)          # card starts, or card ends
    return sorted(cuts)


def merge_short(cuts: list, timestamps, min_sec: float, protected: set) -> list:
    """Drop boundaries whose scene is shorter than `min_sec`, merging it into the
    previous scene. Boundaries in `protected` (title-card edges) always survive.

    Without protection the ~3s title cards -- the very chapter markers we cut for
    -- would all be merged away by a 4s minimum. Pure; unit tested."""
    n = len(timestamps)
    out = [cuts[0]]
    for j, c in enumerate(cuts[1:], start=1):
        end = cuts[j + 1] if j + 1 < len(cuts) else n
        dur = float(timestamps[end - 1]) - float(timestamps[c]) if end > c else 0.0
        if c in protected or dur >= min_sec:
            out.append(c)
    return out


def medoid_index(block: np.ndarray) -> int:
    """Index (within `block`) of the row closest to the block's centroid -- the
    most representative frame of a scene. Rows are already L2-normalized, so
    similarity is a dot product (same idiom as search._topk_similar). Pure."""
    centroid = block.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return 0
    return int(np.argmax(block @ (centroid / norm)))


def is_signage(text: str) -> bool:
    """True when an OCR string looks like station signage rather than noise.

    The scene digest is a prompt, so junk here becomes hallucination downstream: a
    narrator told the on-screen text reads "Son" will invent a station called Son.
    Two accept rules, tuned against the real strings on this video:

      (a) roundel caps -- >=80% uppercase with a >=4-char word:
          "EMBANKMENT", "FINCHLEY ROAD", "CASTLE" (Elephant & Castle)
      (b) multi-word and every word alphabetic and >=5 chars:
          "Charing Cross", "Metropolitan Southbound platform"

    Everything else is dropped: OCR shrapnel ("Son", "ran", "iff", "ars", "BES",
    "ill ace") and in-carriage advertising, which rule (b) would otherwise admit
    ("London Experian Credit" is three alphabetic 5+ letter words) -- hence the
    SCENE_OCR_STOPWORDS veto. Unit tested against all of these."""
    t = (text or "").strip()
    if len(t) < 4:
        return False
    tokens = t.split()
    alpha = [c for c in t if c.isalpha()]
    if not alpha:
        return False
    if any(w.strip(".,:;'\"").lower() in config.SCENE_OCR_STOPWORDS
           for w in tokens):
        return False
    if (sum(c.isupper() for c in alpha) / len(alpha) >= 0.8
            and any(len(w) >= 4 for w in tokens)):
        return True
    return len(tokens) >= 2 and all(w.isalpha() and len(w) >= 5 for w in tokens)


def _distinct(values, limit=None):
    """Distinct non-empty strings, most frequent first (stable on ties)."""
    counts = Counter(v.strip() for v in values if v and v.strip())
    ordered = [v for v, _ in counts.most_common()]
    return ordered[:limit] if limit else ordered


def run() -> int:
    config.ensure_dirs()
    if not config.IMAGE_EMB_FILE.exists():
        raise RuntimeError(
            f"CLIP index not found at {config.IMAGE_EMB_FILE}. "
            "Build it first:  python embed_image.py")

    data = np.load(config.IMAGE_EMB_FILE, allow_pickle=True)
    emb, frame_ids = data["embeddings"], data["frame_ids"]
    timestamps = data["timestamps"]

    with open(config.METADATA_JSON) as f:
        meta_by_id = {m["frame_id"]: m for m in json.load(f)}
    # The npz's frame_ids are authoritative for row->frame mapping: embed_image
    # skips unreadable frames, so row i is not necessarily frame i.
    meta = [meta_by_id[int(fid)] for fid in frame_ids]

    sim = (emb[:-1] * emb[1:]).sum(1)
    is_card = [is_title_card(m["ocr_text"], m["caption"]) for m in meta]

    cuts = cut_indices(sim, is_card, config.SCENE_SIM_THRESH,
                       config.SCENE_TITLE_CARD_FORCE)
    protected = {c for c in cuts
                 if is_card[c] or (c > 0 and is_card[c - 1])}
    cuts = merge_short(cuts, timestamps, config.SCENE_MIN_SEC, protected)

    scenes, chapter_idx, label = [], -1, "Introduction"
    for si, start in enumerate(cuts):
        end = cuts[si + 1] if si + 1 < len(cuts) else len(meta)
        block = meta[start:end]
        card = is_card[start]
        if card:
            chapter_idx += 1
            label = chapter_label(m["ocr_text"] for m in block)

        k = start + medoid_index(emb[start:end])
        faces = sorted({f for m in block for f in m["face_ids"]})
        scenes.append({
            "scene_index": si,
            "chapter_index": max(chapter_idx, 0),
            "chapter_label": label,
            "is_title_card": bool(card),
            "start_sec": round(float(timestamps[start]), 3),
            "end_sec": round(float(timestamps[end - 1]), 3),
            "start_mmss": fmt_ts(float(timestamps[start])),
            "end_mmss": fmt_ts(float(timestamps[end - 1])),
            "n_frames": end - start,
            # inclusive frame-id bounds: an integer join key for build_metadata,
            # safer than re-deriving the span from float timestamps.
            "start_frame_id": int(frame_ids[start]),
            "end_frame_id": int(frame_ids[end - 1]),
            "keyframe_frame_id": int(frame_ids[k]),
            "keyframe_file": meta[k]["filename"],
            "face_ids": faces,
            # A card's OCR *is* its label; content scenes keep only real signage.
            "ocr_texts": [label] if card else _distinct(
                (m["ocr_text"] for m in block if is_signage(m["ocr_text"])), 5),
            "representative_caption": (_distinct(m["caption"] for m in block)
                                       or [""])[0],
            "captions": _distinct((m["caption"] for m in block), 5),
        })

    with open(config.SCENES_JSON, "w") as f:
        json.dump(scenes, f, indent=2)

    n_chapters = len({s["chapter_index"] for s in scenes})
    log.info("scenes: %d scenes across %d chapters (%d title cards) -> %s",
             len(scenes), n_chapters, sum(s["is_title_card"] for s in scenes),
             config.SCENES_JSON.name)
    for s in scenes:
        if s["is_title_card"]:
            log.info("  chapter %2d  %s  %s",
                     s["chapter_index"], s["start_mmss"], s["chapter_label"])
    return len(scenes)


if __name__ == "__main__":
    run()
