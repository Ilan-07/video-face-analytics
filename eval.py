"""Clustering evaluation — link-distance sweep, silhouette, cohesion, contact sheet.

Lets grouping quality be measured instead of guessed. Reads the track templates
saved by recognize.py, sweeps the complete-linkage CLUSTER_LINK_DIST (the actual
pipeline grouping), and reports cluster count + silhouette + worst within-cluster
cohesion so a sensible threshold can be chosen. Also renders a contact sheet of
every identity's representative crop for fast manual review.
"""
import csv
import math

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import config
import recognize
import util

log = util.get_logger()
LINK_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]


def _worst_cohesion(templates, labels) -> float:
    """Smallest within-cluster cosine over multi-member clusters (low=incoherent)."""
    worst = 1.0
    for lab in set(labels.tolist()):
        idx = np.where(labels == lab)[0]
        if len(idx) < 2:
            continue
        s = templates[idx] @ templates[idx].T
        worst = min(worst, float(s[np.triu_indices(len(idx), 1)].min()))
    return worst


def link_sweep() -> list[dict]:
    """Sweep the complete-linkage CLUSTER_LINK_DIST (the actual pipeline grouping)
    and report cluster count, silhouette, and worst within-cluster cohesion.
    Complete linkage guarantees cohesion >= 1 - link_dist, so junk blobs (the old
    DBSCAN failure, cohesion ~0) cannot form."""
    data = np.load(config.TEMPLATE_FILE)
    templates = data["templates"]
    track_ids = [str(t) for t in data["track_ids"]]
    faces = pd.read_csv(config.FACES_CSV)
    init = np.arange(len(track_ids))
    saved = config.CLUSTER_LINK_DIST
    rows = []
    try:
        for d in LINK_GRID:
            config.CLUSTER_LINK_DIST = d
            labels = recognize._link_clusters(track_ids, templates, init, faces)
            n_clusters = len(set(labels.tolist()))
            sil = ""
            if 2 <= n_clusters <= len(templates) - 1:
                try:
                    sil = f"{silhouette_score(templates, labels, metric='cosine'):.3f}"
                except Exception:  # noqa: BLE001
                    sil = ""
            rows.append({"link_dist": d, "clusters": n_clusters, "silhouette": sil,
                         "worst_within_cos": f"{_worst_cohesion(templates, labels):.2f}"})
    finally:
        config.CLUSTER_LINK_DIST = saved

    out = config.REPORT_DIR / "eval_link_sweep.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["link_dist", "clusters",
                                          "silhouette", "worst_within_cos"])
        w.writeheader()
        w.writerows(rows)
    log.info("link-dist sweep -> %s", out.name)
    for r in rows:
        marker = "  <- current" if abs(r["link_dist"] - saved) < 1e-6 else ""
        log.info("  link_dist=%.2f clusters=%d silhouette=%s worst_cos=%s%s",
                 r["link_dist"], r["clusters"], r["silhouette"] or "-",
                 r["worst_within_cos"], marker)
    return rows


def contact_sheet(thumb: int = 112, cols: int = 6) -> None:
    reps = sorted(config.REPORT_DIR.glob("Face_*_rep.jpg"))
    if not reps:
        return
    rows = math.ceil(len(reps) / cols)
    canvas = np.full((rows * (thumb + 18), cols * thumb, 3), 255, np.uint8)
    for i, p in enumerate(reps):
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (thumb, thumb))
        r, c = divmod(i, cols)
        y, x = r * (thumb + 18), c * thumb
        canvas[y:y+thumb, x:x+thumb] = im
        cv2.putText(canvas, p.stem.replace("_rep", ""), (x + 2, y + thumb + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(str(config.REPORT_DIR / "contact_sheet.png"), canvas)
    log.info("contact sheet -> contact_sheet.png")


def run() -> None:
    link_sweep()
    contact_sheet()
    # Label-free objective checks: precision (co-occurrence) + recall (continuity).
    import eval_cooccurrence
    eval_cooccurrence.run()
    import eval_continuity
    eval_continuity.run()


if __name__ == "__main__":
    run()
