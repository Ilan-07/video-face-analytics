"""End-to-end orchestrator. Idempotent: skips stages whose output exists."""
import argparse

import config
import util

log = util.get_logger()


def _stale(stage: str, outputs: list, force: bool) -> bool:
    """Rerun if forced, an output is missing, or the stage's config changed."""
    stamp = config.DATA / f".{stage}.fp"
    fp = config.stage_fingerprint(stage)
    if force or not all(o.exists() for o in outputs):
        return True
    return not stamp.exists() or stamp.read_text().strip() != fp


def _stamp(stage: str) -> None:
    (config.DATA / f".{stage}.fp").write_text(config.stage_fingerprint(stage))


def main():
    ap = argparse.ArgumentParser(description="Video face analytics pipeline")
    ap.add_argument("--url", default=config.VIDEO_URL)
    ap.add_argument("--force", action="store_true",
                    help="rerun all stages even if outputs exist")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip the clustering evaluation harness")
    args = ap.parse_args()
    config.ensure_dirs()

    # 1. Download
    ran = args.force or not any(config.VIDEO_DIR.glob("video.*"))
    if ran:
        import download
        download.download(args.url)
    else:
        log.info("[skip] video already downloaded")

    # 2. Frames  (rerun if upstream ran or FPS changed)
    if ran or _stale("extract", [config.FRAMES_CSV], args.force):
        import extract_frames
        extract_frames.extract()
        _stamp("extract")
        ran = True
    else:
        log.info("[skip] frames already extracted")

    # 3. Detect + track  (rerun if upstream ran or detection config changed)
    if ran or _stale("detect", [config.FACES_CSV, config.EMB_FILE], args.force):
        import detect_faces
        detect_faces.detect()
        _stamp("detect")
        ran = True
    else:
        log.info("[skip] faces already detected")

    # 3.5 Appearance re-ID  (optional; clothing/body templates for cross-scene
    #     linking). Gated by APPEARANCE_ENABLE; feeds recognize when present.
    if config.APPEARANCE_ENABLE:
        if ran or _stale("appearance", [config.APPEARANCE_FILE], args.force):
            import appearance
            appearance.compute()
            _stamp("appearance")
            ran = True
        else:
            log.info("[skip] appearance templates already computed")

    # 4. Recognize / cluster  (rerun if upstream ran or clustering config changed)
    if ran or _stale("recognize", [config.IDENTITIES_CSV], args.force):
        import recognize
        recognize.cluster()
        _stamp("recognize")
    else:
        log.info("[skip] identities already clustered")

    # 5. Analytics
    import analytics
    summary = analytics.run()

    # 6. Evaluation (Fix #3)
    if not args.no_eval:
        import eval as evaluation
        evaluation.run()

    mf = summary["most_frequent_face"]
    log.info("=== RESULTS ===")
    log.info("Total frames:       %d", summary["total_frames"])
    log.info("Faces detected:     %d", summary["total_faces_detected"])
    log.info("Tracks:             %d", summary["total_tracks"])
    log.info("Featured cast:      %d (>= %d frames present)",
             summary["featured_cast"], config.RECURRING_MIN_FRAMES)
    log.info("Total face groups:  %d (incl. one-off/background)",
             summary["total_face_groups"])
    if mf:
        log.info("Most frequent face: %s (%ss screen time, %d appearances)",
                 mf["face_id"], mf["screen_time_sec"], mf["appearances"])


if __name__ == "__main__":
    main()
