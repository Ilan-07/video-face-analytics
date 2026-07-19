"""Build a human-labeling sheet for ground-truth identity annotation.

Emits three things:
  - data/ground_truth.csv   : one row per track (track_id, predicted_face_id,
                              n_faces, first_sec, review, true_id). EXISTING
                              `true_id` values are preserved across re-runs, so
                              regenerating to pick up new tracks never destroys
                              prior labelling work.
  - reports/labelsheet.html : tracks grouped by the pipeline's predicted identity,
                              with thumbnails and review flags -- label by scanning
                              groups (confirm each is one person; merge/split as
                              needed) rather than eyeballing 359 loose crops.
  - reports/labelsheet.png  : the flat crop grid (kept for a quick overview).

Fill `true_id` with a consistent label per real person (same person across scenes
=> same label). Use `x` for non-faces / unusable crops. Then run eval_labeled.py.

`--prefill` seeds `true_id` from the pipeline's own grouping so labelling becomes
verification, not from-scratch annotation, and flags the tracks the templates say
are borderline (a near neighbour in another identity, or weak cohesion within its
own) so human attention goes where it matters. The human remains the arbiter --
prefill only fills blanks and never overwrites an existing label.
"""
import argparse
import base64
import csv
import html
import math

import cv2
import numpy as np
import pandas as pd

import config
import util

log = util.get_logger()

COLS = 12
THUMB = 96
CAP = 16


def _best_crop_per_track(faces: pd.DataFrame) -> dict[str, str]:
    faces = faces.copy()
    faces["score"] = (faces["det_score"] * faces["blur_var"]
                      / (1.0 + faces["nose_offset"]))
    best = {}
    for tid, g in faces.groupby("track_id"):
        best[str(tid)] = g.loc[g["score"].idxmax(), "crop_file"]
    return best


def _load_existing_labels() -> dict[str, str]:
    """track_id -> already-entered true_id, so a re-run preserves labelling."""
    path = config.DATA / "ground_truth.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if "true_id" not in df.columns:
        return {}
    return {str(r.track_id): r.true_id.strip()
            for r in df.itertuples(index=False) if r.true_id.strip()}


def review_flag(cohesion: float, cross_nn: float,
                merge_cos: float, split_cos: float) -> str:
    """Which tracks a human should look at first, from template geometry. Pure.

    - "merge?"  a track in a DIFFERENT predicted identity sits closer than the
                clustering ceiling -- the grouping may have split one person.
    - "split?"  a track's closest SAME-identity neighbour is far -- it may have
                been merged into the wrong group (or is an isolated mis-track).
    Both can fire; neither means the grouping is locally confident."""
    flags = []
    if cross_nn >= merge_cos:
        flags.append("merge?")
    if cohesion < split_cos:
        flags.append("split?")
    return " ".join(flags)


def _review_flags(track_ids, templates, pred) -> dict[str, str]:
    """Per-track review flag from the saved templates (empty dict if absent)."""
    if templates is None or len(track_ids) < 2:
        return {}
    merge_cos = 1.0 - config.CLUSTER_LINK_DIST          # closer than the ceiling
    split_cos = merge_cos - 0.10                         # weakly attached to own group
    sim = templates @ templates.T
    np.fill_diagonal(sim, -1.0)
    pred_arr = np.array([pred.get(t, "") for t in track_ids])
    out = {}
    for i, tid in enumerate(track_ids):
        same = pred_arr == pred_arr[i]
        same[i] = False
        cohesion = float(sim[i][same].max()) if same.any() else 1.0
        cross = ~same
        cross[i] = False
        cross_nn = float(sim[i][cross].max()) if cross.any() else -1.0
        out[tid] = review_flag(cohesion, cross_nn, merge_cos, split_cos)
    return out


def _thumb_b64(crop_file) -> str:
    im = cv2.imread(str(config.FACE_DIR / crop_file)) if crop_file else None
    if im is None:
        return ""
    im = cv2.resize(im, (THUMB, THUMB))
    ok, buf = cv2.imencode(".jpg", im)
    return base64.b64encode(buf).decode() if ok else ""


def _write_html(ident, best, flags, prefilled) -> None:
    """Tracks grouped by predicted identity, newest-cast first, so a reviewer can
    confirm each group is one person and spot cross-group duplicates."""
    groups: dict[str, list] = {}
    for _, t in ident.iterrows():
        groups.setdefault(str(t["face_id"]), []).append(str(t["track_id"]))
    order = sorted(groups, key=lambda g: -len(groups[g]))

    parts = ["<!doctype html><meta charset=utf-8><title>Label sheet</title>",
             "<style>body{font:14px system-ui;margin:20px}"
             ".grp{margin:14px 0;border-top:1px solid #ddd;padding-top:8px}"
             ".t{display:inline-block;text-align:center;margin:3px;font:11px monospace}"
             ".t img{display:block;border:2px solid #ccc}"
             ".merge\\?{border-color:#e67e22}.split\\?{border-color:#c0392b}"
             ".flag{color:#c0392b}</style>",
             "<h2>Label sheet — verify each group is one person</h2>",
             "<p>Each section is the pipeline's predicted identity. Confirm every "
             "crop in it is the same person (else note a <b>split?</b>); watch for "
             "the same person appearing across sections (a <b>merge?</b>). Edit "
             "<code>true_id</code> in <code>data/ground_truth.csv</code>.</p>"]
    for g in order:
        tids = groups[g]
        parts.append(f"<div class=grp><b>{html.escape(g)}</b> "
                     f"({len(tids)} track{'s' if len(tids) != 1 else ''})<br>")
        for tid in tids:
            b64 = _thumb_b64(best.get(tid, ""))
            fl = flags.get(tid, "")
            cls = "t " + fl.replace("?", "\\?").replace(" ", " ")
            pre = prefilled.get(tid, "")
            cap = html.escape(tid)
            if fl:
                cap += f"<br><span class=flag>{html.escape(fl)}</span>"
            if pre:
                cap += f"<br>={html.escape(pre)}"
            img = (f"<img src='data:image/jpeg;base64,{b64}' width={THUMB} "
                   f"height={THUMB}>") if b64 else "(no crop)"
            parts.append(f"<span class='{cls}'>{img}{cap}</span>")
        parts.append("</div>")
    util.write_text_atomic(config.REPORT_DIR / "labelsheet.html", "".join(parts))


def _write_png(thumbs) -> None:
    n = len(thumbs)
    cols, cell = COLS, THUMB + CAP
    grid_rows = math.ceil(n / cols) if n else 1
    canvas = np.full((grid_rows * cell, cols * THUMB, 3), 255, np.uint8)
    for i, (tid, im) in enumerate(thumbs):
        r, c = divmod(i, cols)
        y, x = r * cell, c * THUMB
        if im is not None:
            canvas[y:y+THUMB, x:x+THUMB] = cv2.resize(im, (THUMB, THUMB))
        cv2.putText(canvas, tid, (x + 2, y + THUMB + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
    cv2.imwrite(str(config.REPORT_DIR / "labelsheet.png"), canvas)


def run(prefill: bool = False) -> None:
    config.ensure_dirs()
    faces = pd.read_csv(config.FACES_CSV)
    ident = pd.read_csv(config.IDENTITIES_CSV).sort_values("first_sec")
    ident = ident.reset_index(drop=True)

    best = _best_crop_per_track(faces)
    existing = _load_existing_labels()
    pred = {str(t["track_id"]): str(t["face_id"]) for _, t in ident.iterrows()}

    flags = {}
    if config.TEMPLATE_FILE.exists():
        data = np.load(config.TEMPLATE_FILE)
        tids = [str(t) for t in data["track_ids"]]
        flags = _review_flags(tids, data["templates"], pred)

    # Prefill seeds ONLY blank labels, and never a track we already labelled.
    prefilled = {}
    rows, thumbs = [], []
    for _, t in ident.iterrows():
        tid = str(t["track_id"])
        true_id = existing.get(tid, "")
        if not true_id and prefill:
            true_id = pred.get(tid, "")
            prefilled[tid] = true_id
        rows.append({"track_id": tid, "predicted_face_id": t["face_id"],
                     "n_faces": int(t["n_faces"]), "first_sec": f"{t['first_sec']:.1f}",
                     "review": flags.get(tid, ""), "true_id": true_id})
        im = cv2.imread(str(config.FACE_DIR / best[tid])) if tid in best else None
        thumbs.append((tid, im))

    with open(config.DATA / "ground_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "predicted_face_id",
                                          "n_faces", "first_sec", "review",
                                          "true_id"])
        w.writeheader()
        w.writerows(rows)

    _write_html(ident, best, flags, prefilled)
    _write_png(thumbs)
    kept = sum(1 for r in rows if r["true_id"])
    flagged = sum(1 for r in rows if r["review"])
    log.info("labelsheet: %d tracks, %d already labelled (preserved), %d flagged "
             "for review%s -> reports/labelsheet.html, data/ground_truth.csv",
             len(rows), kept, flagged, " (prefilled)" if prefill else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the identity-labeling sheet")
    ap.add_argument("--prefill", action="store_true",
                    help="seed blank true_id from the pipeline grouping (verify, "
                         "don't annotate from scratch)")
    run(prefill=ap.parse_args().prefill)
