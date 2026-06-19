"""Label-free completeness check via tracking continuity (recall analog).

eval_cooccurrence gives PRECISION (faces in the same frame must be different
people -> cannot-link). This is its RECALL counterpart: two tracks bridged across
a brief gap at nearly the same location are almost certainly the SAME person
(a momentary track break) -> must-link. Measuring how often the clustering keeps
such pairs together is an objective over-segmentation / completeness signal with
no manual labels.

Scope: this catches *intra-shot* over-segmentation (track breaks within a
continuous shot). It is blind to *cross-cut* splits -- two scenes have no temporal
bridge -- which remain unmeasurable without an orthogonal signal or labels.
"""
import json

import pandas as pd

import config
import util
from util import iou

log = util.get_logger()


def _track_endpoints(faces: pd.DataFrame) -> dict:
    """track_id -> (first_frame, last_frame, first_bbox, last_bbox)."""
    out = {}
    for tid, g in faces.groupby("track_id"):
        g = g.sort_values("frame_id")
        first, last = g.iloc[0], g.iloc[-1]
        box = lambda r: [r.x1, r.y1, r.x2, r.y2]
        out[str(tid)] = (int(first.frame_id), int(last.frame_id),
                         box(first), box(last))
    return out


def run() -> dict:
    faces = pd.read_csv(config.FACES_CSV)
    ident = pd.read_csv(config.IDENTITIES_CSV)
    face_of = dict(zip(ident["track_id"].astype(str), ident["face_id"]))
    ep = _track_endpoints(faces)
    tids = list(ep)

    must_link = respected = 0
    broken = []
    for i in range(len(tids)):
        for j in range(len(tids)):
            if i == j:
                continue
            a, b = tids[i], tids[j]
            a_last_box, b_first_box = ep[a][3], ep[b][2]
            gap = ep[b][0] - ep[a][1]          # b starts after a ends
            if 1 <= gap <= config.CONTINUITY_MAX_GAP and \
                    iou(a_last_box, b_first_box) >= config.CONTINUITY_IOU:
                must_link += 1
                fa, fb = face_of.get(a), face_of.get(b)
                same = (fa == fb and fa not in (None, "unknown"))
                if same:
                    respected += 1
                else:
                    broken.append((a, b, fa, fb))

    recall = respected / must_link if must_link else 1.0
    metrics = {
        "must_link_pairs": must_link,
        "respected": respected,
        "continuity_recall": round(recall, 4),
        "broken_examples": broken[:20],
    }
    with open(config.REPORT_DIR / "eval_continuity.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("continuity check: %d must-link pairs, %d kept together, "
             "recall=%.4f", must_link, respected, recall)
    return metrics


if __name__ == "__main__":
    run()
