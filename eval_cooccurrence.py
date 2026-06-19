"""Label-free objective grouping check over the whole video.

Two faces detected in the *same frame* must be different people. Every such
co-occurring pair is therefore a known "cannot-link" constraint. If the
clustering ever gives two co-occurring faces the same identity, that is a
guaranteed false merge. This needs no manual labels and covers every frame,
giving an objective precision/homogeneity signal that scales.
"""
import json
from itertools import combinations

import pandas as pd

import config
import util

log = util.get_logger()


def run() -> dict:
    faces = pd.read_csv(config.FACES_CSV)
    ident = pd.read_csv(config.IDENTITIES_CSV)
    faces = faces.merge(ident[["track_id", "face_id"]], on="track_id", how="left")
    known = faces[faces["face_id"].notna() & (faces["face_id"] != "unknown")]

    cannot_link = violations = 0
    bad = []
    for frame_id, g in known.groupby("frame_id"):
        ids = list(g["face_id"])
        for a, b in combinations(ids, 2):
            cannot_link += 1
            if a == b:
                violations += 1
                bad.append((int(frame_id), a))

    precision = 1.0 - violations / cannot_link if cannot_link else 1.0
    metrics = {
        "cannot_link_pairs": cannot_link,
        "false_merges": violations,
        "cannot_link_precision": round(precision, 4),
        "violations": bad[:20],
    }
    with open(config.REPORT_DIR / "eval_cooccurrence.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("co-occurrence check: %d cannot-link pairs, %d false merges, "
             "precision=%.4f", cannot_link, violations, precision)
    return metrics


if __name__ == "__main__":
    run()
