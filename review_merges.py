"""Emit clothing-proposed cross-scene merge candidates for targeted human review.

Face-only clustering keeps the same person split across scene cuts. Clothing re-ID
proposes which splits are the same person -- but that can't be auto-verified
(co-occurrence is blind to cross-cut merges). So instead of silently merging, this
emits the SMALL set of cross-cut pairs clothing would merge as a review sheet:

  reports/merge_candidates.png   side-by-side reps + timestamps + distances
  reports/merge_candidates.csv   one row per pair, blank `verdict` (same/different)

A human confirms a few dozen pairs -> authoritative cross-cut labels (active
learning). Needs appearance.compute() to have run. The candidates are exactly the
pairs the appearance fusion would act on, so confirming them validates the fusion.
"""
import csv
from itertools import combinations

import cv2
import numpy as np
import pandas as pd

import appearance
import config
import util

log = util.get_logger()
MAX_CANDIDATES = 30


def run() -> int:
    app = appearance.load_templates()
    if not app:
        raise RuntimeError("no appearance templates; run appearance.compute() first")
    faces = pd.read_csv(config.FACES_CSV)
    ident = pd.read_csv(config.IDENTITIES_CSV)
    emb = np.load(config.EMB_FILE)
    face_of = dict(zip(ident["track_id"].astype(str), ident["face_id"]))

    faces = faces.copy()
    faces["face_id"] = faces["track_id"].astype(str).map(face_of)
    faces["q"] = faces["det_score"] * faces["blur_var"] / (1.0 + faces["nose_offset"])
    named = faces[faces["face_id"].notna() & (faces["face_id"] != "unknown")]

    clusters = {}
    for fid, g in named.groupby("face_id"):
        tracks = {str(t) for t in g["track_id"].unique()}
        avecs = [app[t] for t in tracks if t in app]
        if not avecs:
            continue
        av = np.mean(avecs, axis=0)
        top = g.nlargest(config.BEST_SHOT_K, "q")["crop_id"].tolist()
        clusters[fid] = {
            "app": av / (np.linalg.norm(av) + 1e-9),
            "face": np.stack([emb[c] for c in top]).astype(float),
            "frames": set(g["frame_id"]),
            "first_sec": float(g["timestamp_sec"].min()),
        }

    cands = []
    for a, b in combinations(sorted(clusters), 2):
        ca, cb = clusters[a], clusters[b]
        if ca["frames"] & cb["frames"]:          # co-occur -> definitely different
            continue
        adist = float(1.0 - ca["app"] @ cb["app"])
        if adist > config.APPEARANCE_DIST:
            continue
        fdist = float(1.0 - (ca["face"] @ cb["face"].T).min())   # complete-linkage
        if fdist > config.APPEARANCE_FACE_DIST:   # the actual fusion criterion:
            continue                              # only pairs it would truly merge
        cands.append({"face_a": a, "face_b": b,
                      "app_dist": round(adist, 3), "face_dist": round(fdist, 3),
                      "a_first_sec": round(ca["first_sec"], 1),
                      "b_first_sec": round(cb["first_sec"], 1), "verdict": ""})
    cands.sort(key=lambda c: c["app_dist"])       # most confident clothing match first
    cands = cands[:MAX_CANDIDATES]

    out_csv = config.REPORT_DIR / "merge_candidates.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["face_a", "face_b", "app_dist",
                                          "face_dist", "a_first_sec", "b_first_sec",
                                          "verdict"])
        w.writeheader()
        w.writerows(cands)

    _montage(cands)
    log.info("%d merge candidates -> reports/merge_candidates.{png,csv} "
             "(fill 'verdict' = same/different)", len(cands))
    return len(cands)


def _montage(cands: list, thumb: int = 112) -> None:
    if not cands:
        return
    rows, cap = len(cands), 16
    canvas = np.full((rows * (thumb + cap), 2 * thumb + 220, 3), 255, np.uint8)
    for i, c in enumerate(cands):
        y = i * (thumb + cap)
        for j, fid in enumerate((c["face_a"], c["face_b"])):
            p = config.REPORT_DIR / f"{fid}_rep.jpg"
            im = cv2.imread(str(p))
            if im is not None:
                canvas[y:y + thumb, j * thumb:(j + 1) * thumb] = cv2.resize(
                    im, (thumb, thumb))
        txt = (f"{c['face_a']}@{c['a_first_sec']}s  {c['face_b']}@{c['b_first_sec']}s"
               f"  app={c['app_dist']} face={c['face_dist']}")
        cv2.putText(canvas, txt, (2 * thumb + 6, y + thumb // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(str(config.REPORT_DIR / "merge_candidates.png"), canvas)


if __name__ == "__main__":
    run()
