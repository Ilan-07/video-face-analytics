"""Phase 4: occurrence stats, screen-time, demographics, visualizations, report.

Fix #2: appearances (distinct tracks) + screen-time seconds, not just frame counts.
Fix #5: per-identity gender/age summary.
Fix #6: timestamped montages, annotated sample frames, appearance timeline, HTML.
"""
import json
import math

import cv2
import numpy as np
import pandas as pd
import supervision as sv

import config
import util

log = util.get_logger()

PALETTE = [(66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
           (171, 71, 188), (0, 172, 193), (255, 112, 67), (109, 76, 65)]


def _load():
    frames = pd.read_csv(config.FRAMES_CSV)
    faces = pd.read_csv(config.FACES_CSV)
    ident = pd.read_csv(config.IDENTITIES_CSV)
    faces = faces.merge(ident[["track_id", "face_id"]], on="track_id", how="left")
    # Carry the real frame filename for annotation (Fix #6 / sv extraction).
    faces = faces.merge(frames[["frame_id", "filename"]], on="frame_id", how="left")
    return frames, faces, ident


def _fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:d}:{s:02d}"


def build_montage(face_id, g, cols=8, thumb=112):
    g = g.sort_values("timestamp_sec")
    files = g["crop_file"].tolist()[:cols * 4]
    times = g["timestamp_sec"].tolist()[:cols * 4]
    imgs = []
    for cf, ts in zip(files, times):
        im = cv2.imread(str(config.FACE_DIR / cf))
        if im is None:
            continue
        im = cv2.resize(im, (thumb, thumb))
        cv2.rectangle(im, (0, thumb - 16), (thumb, thumb), (0, 0, 0), -1)
        cv2.putText(im, _fmt_ts(ts), (3, thumb - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        imgs.append(im)
    if not imgs:
        return
    rows = math.ceil(len(imgs) / cols)
    canvas = np.full((rows * thumb, cols * thumb, 3), 255, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r*thumb:(r+1)*thumb, c*thumb:(c+1)*thumb] = im
    cv2.imwrite(str(config.MONTAGE_DIR / f"{face_id}.png"), canvas)


def annotate_frames(faces) -> list[str]:
    """Draw boxes + Face IDs on the busiest frames using supervision annotators."""
    known = faces.dropna(subset=["face_id"])
    if known.empty:
        return []
    counts = known.groupby("frame_id").size().sort_values(ascending=False)
    # Stable color index per identity (sv colors by class_id).
    face_to_cls = {fid: i for i, fid in enumerate(sorted(known["face_id"].unique()))}
    box_ann = sv.BoxAnnotator(thickness=2)
    lab_ann = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    out = []
    for frame_id in counts.head(config.MAX_ANNOTATED_FRAMES).index:
        g = faces[(faces["frame_id"] == frame_id) & faces["face_id"].notna()]
        fname = g["filename"].iloc[0]
        img = cv2.imread(str(config.FRAME_DIR / fname))
        if img is None:
            continue
        dets = sv.Detections(
            xyxy=g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float),
            class_id=g["face_id"].map(face_to_cls).to_numpy(dtype=int))
        labels = g["face_id"].tolist()
        img = box_ann.annotate(img, dets)
        img = lab_ann.annotate(img, dets, labels=labels)
        name = f"annot_{int(frame_id):06d}.jpg"
        cv2.imwrite(str(config.ANNOT_DIR / name), img)
        out.append(f"annotated/{name}")
    return out


def timeline(ident, stats) -> str:
    """Gantt-style appearance timeline: one lane per identity (Fix #6)."""
    order = [s["face_id"] for s in stats]
    if not order:
        return ""
    duration = max(ident["last_sec"].max(), 1.0)
    lane_h = config.TIMELINE_LANE_H
    left, top = 90, 24
    w, h = config.TIMELINE_W, top + lane_h * len(order) + 20
    canvas = np.full((h, w, 3), 255, np.uint8)
    plot_w = w - left - 20

    def x(sec):
        return left + int(plot_w * sec / duration)

    for k in range(0, int(duration) + 1, max(1, int(duration // 6))):
        cx = x(k)
        cv2.line(canvas, (cx, top), (cx, h - 16), (230, 230, 230), 1)
        cv2.putText(canvas, _fmt_ts(k), (cx - 12, h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    for i, fid in enumerate(order):
        col = PALETTE[i % len(PALETTE)]
        y0 = top + i * lane_h
        cv2.putText(canvas, fid, (6, y0 + lane_h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
        for _, t in ident[ident["face_id"] == fid].iterrows():
            cv2.rectangle(canvas, (x(t.first_sec), y0 + 4),
                          (max(x(t.last_sec), x(t.first_sec) + 2),
                           y0 + lane_h - 6), col, -1)
    path = config.REPORT_DIR / "timeline.png"
    cv2.imwrite(str(path), canvas)
    return "timeline.png"


def run() -> dict:
    frames, faces, ident = _load()
    known = faces[faces["face_id"].notna() & (faces["face_id"] != "unknown")].copy()

    stats = []
    for face_id, g in known.groupby("face_id"):
        tg = ident[ident["face_id"] == face_id]
        # Demographics from quality_ok crops only (gender/age are unreliable on
        # blurry/profile faces); abstain if the identity has no good crop.
        gq = g[g["quality_ok"] == 1]
        dg = gq if len(gq) else g[g["det_score"] >= config.REAL_FACE_DET]
        genders = [x for x in dg["gender"].tolist() if isinstance(x, str) and x]
        gender = max(set(genders), key=genders.count) if genders else "?"
        age = int(dg["age"].median()) if len(dg) else 0
        build_montage(face_id, g)
        stats.append({
            "face_id": face_id,
            "appearances": int(len(tg)),                 # distinct tracks (Fix #2)
            "frame_count": int(len(g)),                  # per-frame occurrences
            "screen_time_sec": round(float(tg["duration_sec"].sum()), 1),
            "first_seen_sec": float(g["timestamp_sec"].min()),
            "last_seen_sec": float(g["timestamp_sec"].max()),
            "gender": gender,
            "age": age,
            "representative": f"{face_id}_rep.jpg",
            "montage": f"montages/{face_id}.png",
        })
    stats.sort(key=lambda s: s["screen_time_sec"], reverse=True)

    annotated = annotate_frames(faces)
    tl = timeline(ident, stats)

    featured = [s for s in stats
                if s["frame_count"] >= config.RECURRING_MIN_FRAMES]
    summary = {
        "total_frames": int(len(frames)),
        "fps": config.FPS,
        "total_faces_detected": int(len(faces)),
        "total_tracks": int(ident["track_id"].nunique()),
        # Two-tier count: total groups overcounts (one-off background faces);
        # featured cast (>=N frames of presence) is the meaningful headline.
        "total_face_groups": int(len(stats)),
        "featured_cast": int(len(featured)),
        "total_unique_faces": int(len(stats)),     # kept for backward compat
        "most_frequent_face": stats[0] if stats else None,
        "identities": stats,
        "timeline": tl,
        "annotated_frames": annotated,
    }
    with open(config.REPORT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _write_markdown(summary)
    _write_html(summary)
    log.info("report written -> %s", (config.REPORT_DIR / 'report.html'))
    return summary


def _write_markdown(s: dict) -> None:
    L = ["# Face Analytics Summary", "",
         f"- **Total extracted frames:** {s['total_frames']} (at {s['fps']} FPS)",
         f"- **Total faces detected:** {s['total_faces_detected']}",
         f"- **Total tracks:** {s['total_tracks']}",
         f"- **Featured cast** (>= {config.RECURRING_MIN_FRAMES} frames present): "
         f"**{s['featured_cast']}**",
         f"- **Total face groups** (incl. one-off/background): "
         f"{s['total_face_groups']}"]
    mf = s["most_frequent_face"]
    if mf:
        L += ["", f"## Most Frequently Appearing Face: {mf['face_id']}",
              f"- Screen time: **{mf['screen_time_sec']}s** across "
              f"{mf['appearances']} appearance(s)",
              f"- Frames: {mf['frame_count']}; "
              f"~{mf['gender']}, age ~{mf['age']}",
              f"- ![rep]({mf['representative']})"]
    L += ["", "## Occurrence Statistics", "",
          "| Face ID | Screen time (s) | Appearances | Frames | "
          "First | Last | ~Gender | ~Age | Montage |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in s["identities"]:
        L.append(f"| {x['face_id']} | {x['screen_time_sec']} | "
                 f"{x['appearances']} | {x['frame_count']} | "
                 f"{_fmt_ts(x['first_seen_sec'])} | {_fmt_ts(x['last_seen_sec'])} | "
                 f"{x['gender']} | {x['age']} | ![m]({x['montage']}) |")
    if s["timeline"]:
        L += ["", "## Appearance Timeline", f"![timeline]({s['timeline']})"]
    with open(config.REPORT_DIR / "summary.md", "w") as f:
        f.write("\n".join(L) + "\n")


def _write_html(s: dict) -> None:
    rows = "".join(
        f"<tr><td><img src='{x['montage']}' height='64'></td>"
        f"<td><b>{x['face_id']}</b></td><td>{x['screen_time_sec']}s</td>"
        f"<td>{x['appearances']}</td><td>{x['frame_count']}</td>"
        f"<td>{_fmt_ts(x['first_seen_sec'])}–{_fmt_ts(x['last_seen_sec'])}</td>"
        f"<td>{x['gender']}</td><td>{x['age']}</td></tr>"
        for x in s["identities"])
    annots = "".join(f"<img src='{a}' width='320' style='margin:4px'>"
                     for a in s["annotated_frames"])
    mf = s["most_frequent_face"]
    mf_html = (f"<p><img src='{mf['representative']}' height='90'> "
               f"<b>{mf['face_id']}</b> — {mf['screen_time_sec']}s, "
               f"{mf['appearances']} appearance(s)</p>") if mf else ""
    tl = f"<img src='{s['timeline']}' style='max-width:100%'>" if s["timeline"] else ""
    html = f"""<!doctype html><meta charset=utf-8>
<title>Face Analytics Report</title>
<style>body{{font-family:system-ui,Arial;margin:32px;color:#222}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;
padding:6px;text-align:left}}th{{background:#f5f5f5}}img{{vertical-align:middle}}
.cards{{display:flex;flex-wrap:wrap}}</style>
<h1>Face Analytics Report</h1>
<ul><li><b>Frames:</b> {s['total_frames']} @ {s['fps']} FPS</li>
<li><b>Faces detected:</b> {s['total_faces_detected']}</li>
<li><b>Tracks:</b> {s['total_tracks']}</li>
<li><b>Unique faces:</b> {s['total_unique_faces']}</li></ul>
<h2>Most frequent face</h2>{mf_html}
<h2>Appearance timeline</h2>{tl}
<h2>Identities</h2><table><tr><th>Montage</th><th>Face</th><th>Screen time</th>
<th>Appearances</th><th>Frames</th><th>Seen</th><th>~Gender</th><th>~Age</th></tr>
{rows}</table>
<h2>Annotated sample frames</h2><div class=cards>{annots}</div>"""
    with open(config.REPORT_DIR / "report.html", "w") as f:
        f.write(html)


if __name__ == "__main__":
    run()
