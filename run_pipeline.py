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
        with util.time_stage("download"):
            download.download(args.url)
    else:
        log.info("[skip] video already downloaded")

    # 2. Frames  (rerun if upstream ran or FPS changed)
    if ran or _stale("extract", [config.FRAMES_CSV], args.force):
        import extract_frames
        with util.time_stage("extract"):
            extract_frames.extract()
        _stamp("extract")
        ran = True
    else:
        log.info("[skip] frames already extracted")

    # 3. Detect + track  (rerun if upstream ran or detection config changed)
    if ran or _stale("detect", [config.FACES_CSV, config.EMB_FILE], args.force):
        import detect_faces
        with util.time_stage("detect"):
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
            with util.time_stage("appearance"):
                appearance.compute()
            _stamp("appearance")
            ran = True
        else:
            log.info("[skip] appearance templates already computed")

    # 4. Recognize / cluster  (rerun if upstream ran or clustering config changed)
    if ran or _stale("recognize", [config.IDENTITIES_CSV], args.force):
        import recognize
        with util.time_stage("recognize"):
            recognize.cluster()
        _stamp("recognize")
    else:
        log.info("[skip] identities already clustered")

    # 5. Analytics
    import analytics
    with util.time_stage("analytics"):
        summary = analytics.run()

    # ---- Milestone 2: per-frame text, captions, metadata repository ----
    m2_ran = False

    # 6. OCR  (rerun if frames changed or OCR config changed)
    if ran or _stale("ocr", [config.OCR_CSV], args.force):
        import ocr
        with util.time_stage("ocr"):
            ocr.run()
        _stamp("ocr")
        m2_ran = True
    else:
        log.info("[skip] OCR already extracted")

    # 7. Captions  (rerun if frames changed or caption config changed)
    if ran or _stale("caption", [config.CAPTIONS_CSV], args.force):
        import caption
        with util.time_stage("caption"):
            caption.run()
        _stamp("caption")
        m2_ran = True
    else:
        log.info("[skip] captions already generated")

    # 7.5 Speech transcription  (new capability: audio channel). Independent of
    #     frames/OCR; runs before the metadata join so speech joins on timestamp
    #     and reaches the search index. Skips gracefully without faster-whisper,
    #     ffmpeg, or an audio track, so it never blocks the vision pipeline.
    if config.WHISPER_ENABLE and (
            ran or _stale("transcribe", [config.TRANSCRIPT_JSON], args.force)):
        import transcribe
        with util.time_stage("transcribe"):
            transcribe.run()
        _stamp("transcribe")
        m2_ran = True
    elif config.WHISPER_ENABLE:
        log.info("[skip] transcript already generated")

    # 8. Metadata repository  (always rebuilt: it is a cheap join of ocr.csv,
    #    captions.csv, transcript.json and identities, so we never risk it going
    #    stale when an upstream artifact is regenerated out-of-band).
    import build_metadata
    with util.time_stage("metadata"):
        build_metadata.run()
    _stamp("metadata")

    # 9. Semantic index  (rerun if text changed or the embed model changed)
    if ran or m2_ran or _stale("embed", [config.TEXT_EMB_FILE], args.force):
        import embed_text
        with util.time_stage("embed"):
            embed_text.run()
        _stamp("embed")
    else:
        log.info("[skip] semantic index already built")

    # 9.5 Visual (CLIP) index  (image-native search; rerun if frames or the CLIP
    #     model changed -- independent of caption/OCR text).
    img_ran = False
    if ran or _stale("embed_image", [config.IMAGE_EMB_FILE], args.force):
        import embed_image
        with util.time_stage("embed_image"):
            embed_image.run()
        _stamp("embed_image")
        img_ran = True     # scene segmentation reads this index -- cascade to M3
    else:
        log.info("[skip] visual index already built")

    # ---- Milestone 3: scenes, story, summary, event timeline ----
    m3_ran = False

    # 10. Scene segmentation  (offline; cuts on the CLIP index + title cards)
    if ran or img_ran or _stale("scenes", [config.SCENES_JSON], args.force):
        import scenes
        with util.time_stage("scenes"):
            scenes.run()
        _stamp("scenes")
        m3_ran = True
    else:
        log.info("[skip] scenes already segmented")

    # 10.5 / 11. Narration. These are the only stages that need a network LLM, so
    #     a missing OPENROUTER_API_KEY must WARN AND SKIP, never abort: Milestone
    #     1 and 2 users have no key and their pipeline must still run to the end.
    #     A populated data/llm_cache/ replays offline, so a fresh clone with the
    #     committed cache reproduces both stages with no credential at all.
    import llm
    if not (llm.have_key() or any(config.LLM_CACHE_DIR.glob("*.json"))):
        log.warning("[skip] Milestone 3 narration: OPENROUTER_API_KEY is unset "
                    "and the response cache is empty. Scene segmentation and "
                    "scene_index/story_segment metadata are unaffected.")
    else:
        # 10.5 VLM keyframe descriptions (eval reference + ablation input)
        if config.NARRATE_VLM_ENABLE:
            if m3_ran or _stale("describe", [config.SCENE_DESC_JSON], args.force):
                import describe_scenes
                try:
                    with util.time_stage("describe"):
                        describe_scenes.run()
                    _stamp("describe")
                except RuntimeError as e:
                    log.warning("[skip] scene descriptions: %s", e)
            else:
                log.info("[skip] scene keyframes already described")

        # 11. Story, summary, event timeline (all four prompt strategies)
        if m3_ran or _stale("narrate", [config.STORY_JSON, config.TIMELINE_JSON],
                            args.force):
            import narrate
            try:
                with util.time_stage("narrate"):
                    narrate.run(all_strategies=True)
                _stamp("narrate")
            except RuntimeError as e:
                log.warning("[skip] narration: %s", e)
        else:
            log.info("[skip] story and timeline already generated")

    # 12. Metadata refresh. build_metadata is a cheap join and is always rebuilt;
    #     running it again here folds the Milestone 3 scene_index, story_segment
    #     and event_description columns into the repository. (It ran at stage 8
    #     too, because embed_text reads frame_metadata.csv and must not wait on
    #     stages that depend on the CLIP index it precedes.)
    with util.time_stage("metadata"):
        build_metadata.run()
    _stamp("metadata")

    # 13. Evaluation (Fix #3)
    if not args.no_eval:
        import eval as evaluation
        with util.time_stage("eval"):
            evaluation.run()

    # 14. System evaluation (Milestone 4 Task 4). Rolls the stage timings above
    #     together with every milestone's accuracy harness into one report. It
    #     only reads artifacts, so it is cheap and always runs -- but it must come
    #     last, since it reports on the stages before it.
    import eval_system
    eval_system.run()

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
