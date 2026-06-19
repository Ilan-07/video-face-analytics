"""Score the clustering against hand-labeled ground truth (data/ground_truth.csv).

Reports standard external clustering metrics (ARI, homogeneity, completeness,
V-measure) plus pairwise precision/recall/F1 over the labeled tracks. Tracks
whose `true_id` is blank or `x` (non-face / unusable) are excluded. Tracks
predicted "unknown" are each treated as their own singleton cluster so the
backstop is neither rewarded nor punished for collapsing them together.
"""
import json
from itertools import combinations

import pandas as pd
from sklearn.metrics import (adjusted_rand_score, completeness_score,
                             homogeneity_score, v_measure_score)

import config
import util

log = util.get_logger()
GT_FILE = config.DATA / "ground_truth.csv"


def _pairwise(pred: list, true: list) -> dict:
    tp = fp = fn = tn = 0
    for i, j in combinations(range(len(pred)), 2):
        sp = pred[i] == pred[j]
        st = true[i] == true[j]
        if sp and st:
            tp += 1
        elif sp and not st:
            fp += 1
        elif not sp and st:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"pairwise_precision": round(prec, 3),
            "pairwise_recall": round(rec, 3),
            "pairwise_f1": round(f1, 3),
            "pairs_tp": tp, "pairs_fp": fp, "pairs_fn": fn, "pairs_tn": tn}


def run() -> dict:
    if not GT_FILE.exists():
        raise FileNotFoundError("Run make_labelsheet.py and fill ground_truth.csv")
    gt = pd.read_csv(GT_FILE, dtype=str).fillna("")
    gt["true_id"] = gt["true_id"].str.strip()

    # Score against the LIVE clustering (identities.csv), not the prediction
    # snapshot frozen into ground_truth.csv at labelsheet time -- otherwise
    # re-running after a clustering change silently reports stale numbers.
    if config.IDENTITIES_CSV.exists():
        ident = pd.read_csv(config.IDENTITIES_CSV, dtype=str)
        gt = gt.merge(ident[["track_id", "face_id"]], on="track_id", how="left")
        gt["face_id"] = gt["face_id"].fillna("unknown")
    else:  # fall back to the frozen snapshot if the pipeline hasn't run
        log.warning("identities.csv missing; using ground_truth predicted_face_id")
        gt["face_id"] = gt["predicted_face_id"]

    labeled = gt[~gt["true_id"].isin(["", "x", "ignore"])].copy()
    if len(labeled) < 2:
        raise ValueError("Need >=2 labeled tracks in ground_truth.csv")

    # Predicted: give each "unknown" track a unique id so they don't merge.
    pred = [f"unk_{i}" if pid in ("unknown", "") else pid
            for i, pid in enumerate(labeled["face_id"].tolist())]
    true = labeled["true_id"].tolist()

    # Over-segmentation diagnostic (gap #4): how many predicted clusters each
    # true person is split across. >1 means the same person is fragmented.
    frag = labeled.groupby("true_id")["face_id"].nunique()

    metrics = {
        "labeled_tracks": int(len(labeled)),
        "true_identities": len(set(true)),
        "predicted_identities": int(len(set(pred))),
        "mean_clusters_per_person": round(float(frag.mean()), 2),
        "max_fragmentation": int(frag.max()),
        "most_split_person": f"{frag.idxmax()} ({int(frag.max())} clusters)",
        "adjusted_rand_index": round(adjusted_rand_score(true, pred), 3),
        "homogeneity": round(homogeneity_score(true, pred), 3),
        "completeness": round(completeness_score(true, pred), 3),
        "v_measure": round(v_measure_score(true, pred), 3),
        **_pairwise(pred, true),
    }

    with open(config.REPORT_DIR / "eval_labeled.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _write_md(metrics)
    for k, v in metrics.items():
        log.info("  %-22s %s", k, v)
    return metrics


def _write_md(m: dict) -> None:
    lines = [
        "# Grouping Accuracy vs. Hand-Labeled Ground Truth", "",
        f"- Labeled tracks: **{m['labeled_tracks']}**",
        f"- True identities: **{m['true_identities']}**, "
        f"predicted: **{m['predicted_identities']}**", "",
        "## External clustering metrics (0–1, higher better)",
        f"- Adjusted Rand Index: **{m['adjusted_rand_index']}**",
        f"- Homogeneity (clusters are pure): **{m['homogeneity']}**",
        f"- Completeness (a person isn't split): **{m['completeness']}**",
        f"- V-measure (harmonic mean): **{m['v_measure']}**", "",
        "## Pairwise (same-person pair detection)",
        f"- Precision: **{m['pairwise_precision']}** "
        f"(of pairs we grouped, how many truly same person)",
        f"- Recall: **{m['pairwise_recall']}** "
        f"(of truly-same-person pairs, how many we grouped)",
        f"- F1: **{m['pairwise_f1']}**",
        f"- TP={m['pairs_tp']} FP={m['pairs_fp']} "
        f"FN={m['pairs_fn']} TN={m['pairs_tn']}",
    ]
    with open(config.REPORT_DIR / "eval_labeled.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
