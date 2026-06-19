"""Build a human-labeling sheet for ground-truth identity annotation.

Emits:
  - reports/labelsheet.png  : a grid of one representative crop per track,
                              captioned with the track_id, for visual review.
  - data/ground_truth.csv   : one row per track (track_id, predicted_face_id,
                              n_faces, first_sec) with a blank `true_id` column.

A human (or reviewer) fills `true_id` with a consistent label per real person
(e.g. P1, P2, ...). Use `x` for non-faces / unusable crops. Then run
eval_labeled.py to score the clustering against these labels.
"""
import csv
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


def run() -> None:
    config.ensure_dirs()
    faces = pd.read_csv(config.FACES_CSV)
    ident = pd.read_csv(config.IDENTITIES_CSV)

    best = _best_crop_per_track(faces)
    # Order tracks by first appearance for a stable, scannable sheet.
    ident = ident.sort_values("first_sec").reset_index(drop=True)

    rows = []
    thumbs = []
    for _, t in ident.iterrows():
        tid = str(t["track_id"])
        rows.append({"track_id": tid,
                     "predicted_face_id": t["face_id"],
                     "n_faces": int(t["n_faces"]),
                     "first_sec": f"{t['first_sec']:.1f}",
                     "true_id": ""})
        im = cv2.imread(str(config.FACE_DIR / best[tid])) if tid in best else None
        thumbs.append((tid, im))

    # Write the CSV template.
    with open(config.DATA / "ground_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "predicted_face_id",
                                          "n_faces", "first_sec", "true_id"])
        w.writeheader()
        w.writerows(rows)

    # Render the labeling grid.
    n = len(thumbs)
    cols, cell = COLS, THUMB + CAP
    grid_rows = math.ceil(n / cols)
    canvas = np.full((grid_rows * cell, cols * THUMB, 3), 255, np.uint8)
    for i, (tid, im) in enumerate(thumbs):
        r, c = divmod(i, cols)
        y, x = r * cell, c * THUMB
        if im is not None:
            canvas[y:y+THUMB, x:x+THUMB] = cv2.resize(im, (THUMB, THUMB))
        cv2.putText(canvas, tid, (x + 2, y + THUMB + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
    cv2.imwrite(str(config.REPORT_DIR / "labelsheet.png"), canvas)
    log.info("labelsheet -> reports/labelsheet.png (%d tracks); "
             "fill data/ground_truth.csv true_id column", n)


if __name__ == "__main__":
    run()
