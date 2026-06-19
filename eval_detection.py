"""Detection-quality audit (gap #3).

True box-level detector recall/precision needs hand-labeled face boxes, which we
don't have. As a feasible proxy this audits the detections we *did* make: score
and size distributions, quality-gate pass rate, and heuristic "likely non-face"
flags (low confidence, extreme aspect ratio). It also reports how many lone
low-confidence detections the identity backstop will drop, so the gap between
"crops detected" and "real identities" is explicit.
"""
import json

import pandas as pd

import config
import util

log = util.get_logger()


def run() -> dict:
    f = pd.read_csv(config.FACES_CSV)
    n = len(f)
    w, h = (f["x2"] - f["x1"]), (f["y2"] - f["y1"])
    aspect = (w / h).replace([float("inf")], 0)

    low_conf = f["det_score"] < config.REAL_FACE_DET
    odd_aspect = (aspect < 0.5) | (aspect > 1.8)        # faces are roughly square-ish
    likely_nonface = low_conf & odd_aspect              # low conf AND odd shape

    metrics = {
        "total_crops": int(n),
        "det_score_min": round(float(f["det_score"].min()), 3),
        "det_score_median": round(float(f["det_score"].median()), 3),
        "det_score_max": round(float(f["det_score"].max()), 3),
        "pct_below_real_face_det": round(100 * low_conf.mean(), 1),
        "pct_quality_ok": round(100 * (f["quality_ok"] == 1).mean(), 1),
        "median_face_px": int(min(w.median(), h.median())),
        "n_odd_aspect": int(odd_aspect.sum()),
        "n_likely_nonface": int(likely_nonface.sum()),
    }
    with open(config.REPORT_DIR / "eval_detection.json", "w") as fp:
        json.dump(metrics, fp, indent=2)
    for k, v in metrics.items():
        log.info("  %-24s %s", k, v)
    log.info("NOTE: true detector recall/precision needs hand-labeled boxes.")
    return metrics


if __name__ == "__main__":
    run()
