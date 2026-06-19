"""Phase 3: build track templates and cluster tracks into identities.

Fix #1: cluster robust per-track template embeddings instead of raw frames.
Fix #3: track-level DBSCAN (min_samples=1) so no appearance is discarded;
        templates are saved for the evaluation harness (eval.py).
"""
import csv
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

import config
import util

log = util.get_logger()


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _quality_weight(g: pd.DataFrame) -> np.ndarray:
    """Per-face quality proxy: confident + sharp + frontal weighs more.

    Same score used to pick representative crops (det_score * sharpness / pose).
    """
    w = (g["det_score"] * g["blur_var"] / (1.0 + g["nose_offset"])).to_numpy(float)
    return np.clip(w, 1e-6, None)


def _robust_template(vecs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Quality-weighted mean embedding, after dropping within-track outliers.

    `vecs` are L2-normalized ArcFace embeddings, so `vecs @ mean` is cosine
    similarity. Frames whose cosine to the provisional mean falls below
    TEMPLATE_OUTLIER_COS (mis-tracked / occluded crops) are dropped, provided
    the track is big enough to trim and at least two frames survive.
    """
    if len(vecs) == 1:
        return _normalize(vecs[0])
    w = weights / weights.sum()
    mean = _normalize((vecs * w[:, None]).sum(axis=0))
    if len(vecs) >= config.TEMPLATE_MIN_FACES_FOR_TRIM:
        keep = (vecs @ mean) >= config.TEMPLATE_OUTLIER_COS
        if keep.sum() >= 2:
            vecs, weights = vecs[keep], weights[keep]
            w = weights / weights.sum()
            mean = _normalize((vecs * w[:, None]).sum(axis=0))
    return mean


def build_track_templates(faces: pd.DataFrame, emb: dict):
    """One robust, L2-normalized embedding per track (quality faces preferred)."""
    track_ids, templates, meta = [], [], []
    for tid, g in faces.groupby("track_id"):
        hq = g[g["quality_ok"] == 1]
        use = hq if len(hq) else g  # prefer HQ faces, fall back to all
        vecs = np.stack([emb[c] for c in use["crop_id"]]).astype(float)
        weights = (_quality_weight(use) if config.TEMPLATE_USE_QUALITY_WEIGHT
                   else np.ones(len(use)))
        templates.append(_robust_template(vecs, weights).astype(np.float32))
        track_ids.append(str(tid))      # track_id is a string ("t5"/"s0")
        meta.append({
            "track_id": str(tid),
            "n_faces": int(len(g)),
            "first_frame": int(g["frame_id"].min()),
            "last_frame": int(g["frame_id"].max()),
            "first_sec": float(g["timestamp_sec"].min()),
            "last_sec": float(g["timestamp_sec"].max()),
        })
    return track_ids, np.stack(templates).astype(np.float32), meta


def _cooccurrence_cannot_link(track_ids, labels, faces) -> dict:
    """Cluster pairs that share a frame -> different people, must never merge."""
    cl_of_track = {str(t): int(l) for t, l in zip(track_ids, labels)}
    cannot: dict[int, set] = defaultdict(set)
    for _, g in faces.groupby("frame_id"):
        cls = {cl_of_track[t] for t in g["track_id"].astype(str) if t in cl_of_track}
        for a in cls:
            cannot[a] |= (cls - {a})
    return cannot


def _link_clusters(track_ids, templates, labels, faces) -> np.ndarray:
    """Merge over-segmented clusters via complete-linkage agglomeration under
    co-occurrence cannot-link constraints.

    Complete linkage (the *max* pairwise distance between two clusters' member
    tracks must be <= CLUSTER_LINK_DIST) forbids chaining, so a loose threshold
    consolidates a person's split tracks without forming a cross-scene blob.
    """
    dist = 1.0 - templates @ templates.T          # cosine distance, templates are unit-norm
    members: dict[int, list] = defaultdict(list)
    for idx, lab in enumerate(labels):
        members[int(lab)].append(idx)
    cannot = _cooccurrence_cannot_link(track_ids, labels, faces)
    cannot = {lab: set(cannot.get(lab, set())) for lab in members}

    def complete_dist(a, b):                       # max pairwise (vectorized)
        return float(dist[np.ix_(members[a], members[b])].max())

    while True:
        # Deterministic: among allowed pairs within threshold, take the smallest
        # complete-linkage distance, ties broken by (label_a, label_b).
        best, best_d = None, float("inf")
        labs = sorted(members)
        for x in range(len(labs)):
            for y in range(x + 1, len(labs)):
                a, b = labs[x], labs[y]
                if b in cannot[a]:
                    continue
                d = complete_dist(a, b)
                if d <= config.CLUSTER_LINK_DIST and d < best_d:
                    best, best_d = (a, b), d
        if best is None:
            break
        a, b = best                                # merge b into a
        members[a].extend(members.pop(b))
        for o in cannot.pop(b):
            cannot[o].discard(b)
            if o != a:
                cannot[o].add(a)
                cannot[a].add(o)
        cannot[a].discard(a)

    new = np.array(labels).copy()
    for lab, idxs in members.items():
        for i in idxs:
            new[i] = lab
    return new


def _best_shot_link(track_ids, labels, faces, emb) -> np.ndarray:
    """Merge clusters whose best-quality (frontal/sharp) faces match tightly.

    Cross-scene splits happen because a cluster's mean template is polluted by
    pose; but each cluster usually has at least one near-frontal high-quality
    crop. Comparing best-shot prototypes (top-K by quality) cluster-to-cluster
    links the same person across pose, while the tight BEST_SHOT_DIST + the
    co-occurrence cannot-link keep different people apart. Note: this lowers
    within-cluster cohesion by design (it mixes poses) -- validate visually.
    """
    t2c = {str(t): int(l) for t, l in zip(track_ids, labels)}
    f = faces.copy()
    f["cl"] = f["track_id"].astype(str).map(t2c)
    f["q"] = f["det_score"] * f["blur_var"] / (1.0 + f["nose_offset"])
    protos: dict[int, np.ndarray] = {}
    for cl, g in f.groupby("cl"):
        gg = g[g["quality_ok"] == 1]
        use = gg if len(gg) else g
        top = use.nlargest(config.BEST_SHOT_K, "q")["crop_id"].tolist()
        protos[int(cl)] = np.stack([emb[c] for c in top]).astype(float)

    cannot = _cooccurrence_cannot_link(track_ids, labels, faces)
    members: dict[int, list] = defaultdict(list)
    for i, l in enumerate(labels):
        members[int(l)].append(i)
    cannot = {l: set(cannot.get(l, set())) for l in members}

    def best_shot_dist(a, b):  # complete-link on best shots: worst pair must pass
        return float(1.0 - (protos[a] @ protos[b].T).min())  # so it can't chain

    while True:
        best, best_d = None, float("inf")
        labs = sorted(members)
        for x in range(len(labs)):
            for y in range(x + 1, len(labs)):
                a, b = labs[x], labs[y]
                if b in cannot[a]:
                    continue
                d = best_shot_dist(a, b)
                if d <= config.BEST_SHOT_DIST and d < best_d:
                    best, best_d = (a, b), d
        if best is None:
            break
        a, b = best
        members[a].extend(members.pop(b))
        protos[a] = np.vstack([protos[a], protos.pop(b)])
        for o in cannot.pop(b):
            cannot[o].discard(b)
            if o != a:
                cannot[o].add(a)
                cannot[a].add(o)
        cannot[a].discard(a)

    new = np.array(labels).copy()
    for l, idxs in members.items():
        for i in idxs:
            new[i] = l
    return new


def _appearance_link(track_ids, labels, faces, emb) -> np.ndarray:
    """Fuse clothing/body appearance (orthogonal to face pose) to consolidate the
    same person across scenes: merge two clusters when their body appearance is
    close (<= APPEARANCE_DIST) AND their best-shot faces are at least loosely
    similar (<= APPEARANCE_FACE_DIST), still blocked by co-occurrence cannot-link.
    Requires appearance.compute() to have run (config.APPEARANCE_ENABLE)."""
    import appearance
    app = appearance.load_templates()
    if not app:
        log.warning("APPEARANCE_ENABLE set but no templates (%s); "
                    "run appearance.compute() first", config.APPEARANCE_FILE.name)
        return labels

    t2c = {str(t): int(l) for t, l in zip(track_ids, labels)}
    f = faces.copy()
    f["cl"] = f["track_id"].astype(str).map(t2c)
    f["q"] = f["det_score"] * f["blur_var"] / (1.0 + f["nose_offset"])
    face_p, app_p = {}, {}
    for cl, g in f.groupby("cl"):
        gg = g[g["quality_ok"] == 1]
        use = gg if len(gg) else g
        top = use.nlargest(config.BEST_SHOT_K, "q")["crop_id"].tolist()
        face_p[int(cl)] = np.stack([emb[c] for c in top]).astype(float)
        vs = [app[str(t)] for t in g["track_id"].unique() if str(t) in app]
        if vs:
            v = np.mean(vs, axis=0)
            app_p[int(cl)] = v / (np.linalg.norm(v) + 1e-9)

    cannot = _cooccurrence_cannot_link(track_ids, labels, faces)
    members: dict[int, list] = defaultdict(list)
    for i, l in enumerate(labels):
        members[int(l)].append(i)
    cannot = {l: set(cannot.get(l, set())) for l in members}

    def face_dist(a, b):  # complete-linkage on best-shots: worst pair must pass,
        return float(1.0 - (face_p[a] @ face_p[b].T).min())  # so it can't chain

    def app_dist(a, b):
        if a not in app_p or b not in app_p:
            return 1.0
        return float(1.0 - app_p[a] @ app_p[b])

    while True:
        best, best_d = None, float("inf")
        labs = sorted(members)
        for x in range(len(labs)):
            for y in range(x + 1, len(labs)):
                a, b = labs[x], labs[y]
                if b in cannot[a]:
                    continue
                ad = app_dist(a, b)
                if (ad <= config.APPEARANCE_DIST
                        and face_dist(a, b) <= config.APPEARANCE_FACE_DIST
                        and ad < best_d):
                    best, best_d = (a, b), ad
        if best is None:
            break
        a, b = best
        members[a].extend(members.pop(b))
        face_p[a] = np.vstack([face_p[a], face_p.pop(b)])
        if a in app_p and b in app_p:
            v = app_p[a] + app_p[b]
            app_p[a] = v / (np.linalg.norm(v) + 1e-9)
        app_p.pop(b, None)
        for o in cannot.pop(b):
            cannot[o].discard(b)
            if o != a:
                cannot[o].add(a)
                cannot[a].add(o)
        cannot[a].discard(a)

    new = np.array(labels).copy()
    for l, idxs in members.items():
        for i in idxs:
            new[i] = l
    return new


def cluster() -> int:
    faces = pd.read_csv(config.FACES_CSV)
    data = np.load(config.EMB_FILE)
    emb = {k: data[k] for k in data.keys()}

    track_ids, templates, meta = build_track_templates(faces, emb)

    # Constrained complete-linkage clustering from singletons. Complete linkage
    # forbids the single-link chaining that lets DBSCAN merge orthogonal faces
    # into a junk identity (every member pair must be within CLUSTER_LINK_DIST),
    # and the co-occurrence cannot-link forbids merging faces in the same frame.
    init = np.arange(len(track_ids))
    labels = _link_clusters(track_ids, templates, init, faces)
    log.info("complete-linkage clustering: %d tracks -> %d clusters",
             len(track_ids), len(set(labels)))

    if config.BEST_SHOT_ENABLE:
        before = len(set(labels))
        labels = _best_shot_link(track_ids, labels, faces, emb)
        log.info("best-shot linking: %d -> %d clusters", before, len(set(labels)))

    if config.APPEARANCE_ENABLE:
        before = len(set(labels))
        labels = _appearance_link(track_ids, labels, faces, emb)
        log.info("appearance linking: %d -> %d clusters", before, len(set(labels)))

    # Per-cluster face count + high-quality count (for the FP backstop).
    track_to_label = {tid: lab for tid, lab in zip(track_ids, labels)}
    faces = faces.copy()
    faces["cluster"] = faces["track_id"].map(track_to_label)
    n_faces = faces.groupby("cluster").size()
    n_hq = faces[faces["quality_ok"] == 1].groupby("cluster").size()

    # A cluster is a real identity only if corroborated: enough faces, OR a
    # high-quality face, OR a confidently-detected face. Otherwise its tracks
    # are relabeled "unknown" (drops lone low-confidence non-faces / blur).
    n_real = (faces[faces["det_score"] >= config.REAL_FACE_DET]
              .groupby("cluster").size())
    valid = {lab for lab in set(labels)
             if int(n_faces.get(lab, 0)) >= config.MIN_IDENTITY_FACES
             or int(n_hq.get(lab, 0)) >= 1
             or int(n_real.get(lab, 0)) >= 1}
    dropped = len(set(labels)) - len(valid)

    # Rank valid identities by total screen presence (face count), largest first.
    ordered = sorted(valid, key=lambda l: int(n_faces.get(l, 0)), reverse=True)
    label_to_face = {l: f"Face_{i+1:02d}" for i, l in enumerate(ordered)}
    for lab in set(labels):
        label_to_face.setdefault(lab, "unknown")

    # Persist track -> identity mapping.
    with open(config.IDENTITIES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "face_id", "cluster_label", "n_faces",
                    "first_frame", "last_frame", "first_sec", "last_sec",
                    "duration_sec"])
        for tid, lab, m in zip(track_ids, labels, meta):
            # Timestamp-based duration (+ one sample interval) since the
            # effective sampling rate is approximate after striding.
            dur = (m["last_sec"] - m["first_sec"]) + 1.0 / config.FPS
            w.writerow([tid, label_to_face[lab], int(lab), m["n_faces"],
                        m["first_frame"], m["last_frame"],
                        f"{m['first_sec']:.3f}", f"{m['last_sec']:.3f}",
                        f"{dur:.3f}"])

    # Save templates for the eval harness (Fix #3).
    np.savez_compressed(config.TEMPLATE_FILE,
                        templates=templates,
                        track_ids=np.array(track_ids),
                        labels=np.array(labels))

    _save_representatives(faces, track_ids, labels, label_to_face)

    n_ident = len(ordered)
    log.info("%d tracks -> %d unique identities (link_dist=%.2f, %d clusters "
             "dropped as unverified)", len(track_ids), n_ident,
             config.CLUSTER_LINK_DIST, dropped)
    return n_ident


def _save_representatives(faces, track_ids, labels, label_to_face) -> None:
    """Best-quality crop per identity (sharp, frontal, confident)."""
    track_to_face = {tid: label_to_face[lab]
                     for tid, lab in zip(track_ids, labels)}
    faces = faces.copy()
    faces["face_id"] = faces["track_id"].map(track_to_face)
    faces["score"] = (faces["det_score"] * faces["blur_var"]
                      / (1.0 + faces["nose_offset"]))
    for face_id, g in faces.groupby("face_id"):
        if face_id == "unknown":
            continue
        best = g.loc[g["score"].idxmax()]
        img = cv2.imread(str(config.FACE_DIR / best["crop_file"]))
        if img is not None:
            cv2.imwrite(str(config.REPORT_DIR / f"{face_id}_rep.jpg"), img)


if __name__ == "__main__":
    cluster()
