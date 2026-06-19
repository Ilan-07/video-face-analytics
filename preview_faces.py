"""Live face-detector preview: play the video and draw detections in real time.

Uses the SAME InsightFace app (model pack / det_size / det_thresh) as the
pipeline via detect_faces.get_app(), so what you see here is what Phase 2 sees.

    .venv/bin/python preview_faces.py                 # live detect on the video
    .venv/bin/python preview_faces.py --every 3       # detect every 3rd frame
    .venv/bin/python preview_faces.py --start 40      # jump to 40s in
    .venv/bin/python preview_faces.py --save out.mp4  # also write annotated mp4

Keys:  q / Esc = quit   |   space = pause/resume
"""
import argparse
import time

import cv2

import config
import detect_faces


def draw(img, faces):
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        small = (x2 - x1) < config.MIN_FACE_PX or (y2 - y1) < config.MIN_FACE_PX
        color = (0, 165, 255) if small else (0, 255, 0)   # orange = below MIN_FACE_PX
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{face.det_score:.2f}"
        sex, age = getattr(face, "sex", None), getattr(face, "age", None)
        if sex is not None and age is not None:
            label += f" {sex}{int(age)}"
        cv2.putText(img, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if face.kps is not None:
            for (kx, ky) in face.kps.astype(int):
                cv2.circle(img, (kx, ky), 2, (0, 0, 255), -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(config.VIDEO_DIR / "video.mp4"))
    ap.add_argument("--every", type=int, default=1,
                    help="run detection every Nth frame (>=1); boxes persist between")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--scale", type=float, default=1.0, help="display scale factor")
    ap.add_argument("--save", default=None, help="optional output .mp4 path")
    args = ap.parse_args()

    app = detect_faces.get_app()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    if args.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    win = "face detector preview (q=quit, space=pause)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    faces, n, paused, dt = [], 0, False, 0.0
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            if n % max(1, args.every) == 0:
                t0 = time.time()
                faces = app.get(frame)
                dt = (time.time() - t0) * 1000
            n += 1
            view = draw(frame.copy(), faces)
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            cv2.putText(view, f"{ts:6.1f}s  faces={len(faces)}  det={dt:.0f}ms",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                        cv2.LINE_AA)
            if args.scale != 1.0:
                view = cv2.resize(view, None, fx=args.scale, fy=args.scale)
            if args.save:
                if writer is None:
                    h, w = view.shape[:2]
                    writer = cv2.VideoWriter(
                        args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(view)
            cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
