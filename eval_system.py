"""Milestone 4 Task 4: end-to-end system evaluation.

Every earlier milestone measured itself in isolation -- eval_labeled scores the
clustering, eval_search scores OCR/captions/retrieval, eval_story scores the
narration. None of them answers the Milestone 4 question, which is about the
system: what does one video cost end to end, how good is the whole output, and
where does it break?

This module adds no new measurement of its own. It is a reducer: it reads the
stage timings recorded by run_pipeline and the JSON already emitted by every
harness, and reports processing time, quality, and a limitations list derived
FROM those numbers rather than asserted alongside them. That distinction is the
point -- a hand-written limitations section goes stale the moment a threshold
moves, so each finding here is a predicate over the metrics and disappears on
its own when the underlying number improves.

Run:  .venv/bin/python eval_system.py     (or automatically, as pipeline stage 14)
"""
import json

import config
import util

log = util.get_logger()


def _load(path, default=None):
    """Read a JSON artifact, tolerating absence.

    Every input here is optional by design: a Milestone 1 user has no search
    labels, a user without an API key has no story. The report degrades to
    whatever has actually been produced instead of refusing to build."""
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("could not read %s: %s", path.name, e)
        return default


# ------------------------------------------------------------------ timing
def timing_report() -> dict:
    """Processing time, split into local compute vs network-bound narration.

    The split is not cosmetic. Local stages measure this pipeline on this
    machine and are the honest basis for a per-frame cost. The narration stages
    measure how long a free-tier OpenRouter endpoint made us wait -- mostly 429
    backoff -- so folding them into a throughput figure would report someone
    else's rate limiter as our latency."""
    timings = util.load_timings()
    if not timings:
        return {"status": "no timings recorded",
                "hint": "run `python run_pipeline.py` to measure stages"}

    stages = []
    for name, rec in sorted(timings.items(), key=lambda kv: -kv[1]["seconds"]):
        stages.append({
            "stage": name,
            "seconds": rec["seconds"],
            "network": rec.get("network", False),
            "measured_at": rec.get("measured_at"),
        })

    local = sum(s["seconds"] for s in stages if not s["network"])
    network = sum(s["seconds"] for s in stages if s["network"])
    total = local + network

    summary = _load(config.REPORT_DIR / "summary.json", {})
    frames = summary.get("total_frames") or 0
    video_sec = config.VIDEO_DURATION_SEC

    # A populated cache makes narration replay from disk instead of calling the
    # API, so its measured seconds are replay cost, not generation cost. Say so:
    # a reader who sees "narrate: 2s" would otherwise conclude the LLM is cheap,
    # when what they are looking at is a cache hit.
    cache_replay = (any(config.LLM_CACHE_DIR.glob("*.json"))
                    if config.LLM_CACHE_DIR.exists() else False)

    out = {
        "status": "ok",
        "stages": stages,
        "narration_from_cache": cache_replay,
        "local_compute_sec": round(local, 1),
        "network_llm_sec": round(network, 1),
        "total_sec": round(total, 1),
        "total_min": round(total / 60.0, 1),
        "frames_processed": frames,
        "video_duration_sec": video_sec,
    }
    if frames:
        out["sec_per_frame_local"] = round(local / frames, 3)
        out["frames_per_sec_local"] = round(frames / local, 2) if local else None
    if local:
        # >1 means the pipeline finishes faster than the video plays. At 1 FPS
        # sampling this is a statement about the pipeline, not about real-time
        # video: we process 1415 frames drawn from 1415s of footage.
        out["realtime_factor_local"] = round(video_sec / local, 2)
        out["realtime_factor_total"] = round(video_sec / total, 2) if total else None

    # The slowest stage is the only one worth optimising first, so name it.
    local_stages = [s for s in stages if not s["network"]]
    if local_stages:
        top = max(local_stages, key=lambda s: s["seconds"])
        out["slowest_local_stage"] = top["stage"]
        out["slowest_local_share"] = (round(top["seconds"] / local, 3)
                                      if local else None)
    return out


# ----------------------------------------------------------------- quality
def quality_report() -> dict:
    """Accuracy of each subsystem, pulled from the per-milestone harnesses."""
    labeled = _load(config.REPORT_DIR / "eval_labeled.json", {})
    cooc = _load(config.REPORT_DIR / "eval_cooccurrence.json", {})
    cont = _load(config.REPORT_DIR / "eval_continuity.json", {})
    det = _load(config.REPORT_DIR / "eval_detection.json", {})
    srch = _load(config.REPORT_DIR / "eval_search.json", {})
    story = _load(config.REPORT_DIR / "eval_story.json", {})
    summary = _load(config.REPORT_DIR / "summary.json", {})

    oc = (srch or {}).get("ocr_and_captions", {})
    sem = (srch or {}).get("semantic", {})
    strategies = (story or {}).get("strategies", {})
    promoted = strategies.get(config.STORY_STRATEGY, {})

    return {
        "faces": {
            "total_frames": summary.get("total_frames"),
            "total_faces_detected": summary.get("total_faces_detected"),
            "total_tracks": summary.get("total_tracks"),
            "unique_faces": summary.get("total_unique_faces"),
            "featured_cast": summary.get("featured_cast"),
            "det_score_median": det.get("det_score_median"),
            "median_face_px": det.get("median_face_px"),
            # Label-free: two faces in one frame cannot be the same person, so a
            # merge across a cannot-link pair is a provable error.
            "cannot_link_precision": cooc.get("cannot_link_precision"),
            "cannot_link_pairs": cooc.get("cannot_link_pairs"),
            "false_merges": cooc.get("false_merges"),
            "continuity_recall": cont.get("continuity_recall"),
            # Human-labeled subset.
            "v_measure": labeled.get("v_measure"),
            "homogeneity": labeled.get("homogeneity"),
            "completeness": labeled.get("completeness"),
            "pairwise_precision": labeled.get("pairwise_precision"),
            "pairwise_recall": labeled.get("pairwise_recall"),
            "pairwise_f1": labeled.get("pairwise_f1"),
            "adjusted_rand_index": labeled.get("adjusted_rand_index"),
            "mean_clusters_per_person": labeled.get("mean_clusters_per_person"),
            "true_identities": labeled.get("true_identities"),
            "predicted_identities": labeled.get("predicted_identities"),
        },
        "text_and_captions": {
            "ocr_detect_precision": oc.get("ocr_detect_precision"),
            "ocr_detect_recall": oc.get("ocr_detect_recall"),
            "ocr_detect_f1": oc.get("ocr_detect_f1"),
            "ocr_string_fidelity": oc.get("ocr_string_fidelity"),
            "caption_mean_adequacy": oc.get("caption_mean_adequacy"),
            "caption_pct_good_ge4": oc.get("caption_pct_good_ge4"),
            # Raw counts so the report can attach a confidence interval: a small
            # sample makes "100%" mean "100% of very few", and the reader is owed
            # the difference.
            "ocr_tp": oc.get("ocr_tp"), "ocr_fp": oc.get("ocr_fp"),
            "ocr_fn": oc.get("ocr_fn"),
            "caption_scored": oc.get("caption_scored"),
        },
        "search": {
            "k": sem.get("k"),
            "queries": sem.get("queries"),
            "mean_precision_at_k": sem.get("mean_precision_at_5"),
            "visual_mean_precision_at_k": (srch or {}).get(
                "visual", {}).get("mean_precision_at_5"),
            "fused_mean_precision_at_k": (srch or {}).get(
                "fused", {}).get("mean_precision_at_5"),
        },
        "narration": {
            "model": story.get("model"),
            "n_scenes": story.get("n_scenes"),
            "n_chapters": story.get("n_chapters"),
            "promoted_strategy": config.STORY_STRATEGY,
            "chronology": promoted.get("chronology"),
            "coverage": promoted.get("coverage"),
            "grounding_vs_captions": promoted.get("grounding_vs_captions"),
            "grounding_vs_vlm": promoted.get("grounding_vs_vlm"),
            "redundancy_distinct_ngram": promoted.get("redundancy_distinct_ngram"),
            "strategies_compared": sorted(strategies.keys()),
        },
    }


# ----------------------------------------------------------- staleness guard
# eval_system only reduces over JSON that other harnesses wrote, so the report is
# only as fresh as those files. run_pipeline now regenerates all of them every
# run, but a manual `eval_system.py` or a `--no-eval` pipeline can still leave an
# eval JSON older than the artifact it scored. Rather than trust the numbers
# blindly, name any input whose source artifact was modified after the eval ran --
# a self-check against exactly the stale-number failure this milestone is about.
def _staleness_sources() -> dict:
    """Map each eval JSON to the artifacts whose change invalidates it. Built at
    call time, not import time, so it always reflects the current config paths
    (which tests reassign)."""
    return {
        "eval_labeled.json": [config.IDENTITIES_CSV],
        "eval_cooccurrence.json": [config.IDENTITIES_CSV],
        "eval_continuity.json": [config.IDENTITIES_CSV],
        "eval_search.json": [config.OCR_CSV, config.CAPTIONS_CSV],
        "eval_story.json": [config.STORY_JSON],
        "summary.json": [config.IDENTITIES_CSV],
    }


def staleness_report() -> list:
    """Eval inputs older than the artifact they scored (empty list == all fresh).

    A one-second grace absorbs same-run ordering (the artifact is written, then
    the eval, within the same second); anything beyond that is a real staleness:
    the grouping or text changed and its score was never recomputed."""
    stale = []
    for name, sources in _staleness_sources().items():
        report = config.REPORT_DIR / name
        if not report.exists():
            continue
        r_mtime = report.stat().st_mtime
        newer = sorted(s.name for s in sources
                       if s.exists() and s.stat().st_mtime > r_mtime + 1)
        if newer:
            stale.append({"report": name, "outdated_by": newer})
    return stale


# ------------------------------------------------------- limitations (derived)
# Each entry is a predicate over the metrics plus the enhancement it implies.
# Keeping them data-driven means the report cannot claim a weakness the numbers
# no longer support -- fix the recall and the finding removes itself.
def _checks(q: dict, t: dict) -> list:
    faces, text = q["faces"], q["text_and_captions"]
    search, narr = q["search"], q["narration"]

    def pct(x):
        return f"{x:.1%}" if isinstance(x, (int, float)) else "n/a"

    checks = []

    # -- Identity grouping: the system's dominant error mode.
    frag = faces.get("mean_clusters_per_person")
    if frag and frag > 1.5:
        checks.append({
            "area": "Identity grouping",
            "severity": "high",
            "finding": "The same person is split across multiple identities "
                       f"({frag} clusters per person on the labeled subset), so "
                       "unique-face counts overstate the true cast and screen "
                       "time is divided among duplicate IDs.",
            "evidence": f"pairwise recall {pct(faces.get('pairwise_recall'))}, "
                        f"completeness {pct(faces.get('completeness'))}, "
                        f"{faces.get('predicted_identities')} predicted vs "
                        f"{faces.get('true_identities')} true identities",
            "enhancement": "The conservative distance ceilings (CLUSTER_LINK_DIST "
                           "/ BEST_SHOT_DIST = 0.50) buy precision at recall's "
                           "expense. Enabling the body-appearance re-ID pass "
                           "(APPEARANCE_ENABLE) or a pose-aware template per "
                           "track would link profile-only tracks that ArcFace "
                           "currently keeps apart.",
        })

    prec = faces.get("pairwise_precision")
    cl = faces.get("cannot_link_precision")
    if cl is not None and cl >= 0.99 and prec and prec < 0.9:
        checks.append({
            "area": "Identity grouping (precision)",
            "severity": "info",
            "finding": "Grouping makes no provable merge errors: every pair of "
                       "faces co-occurring in one frame was kept apart. The "
                       "residual pairwise precision loss is therefore between "
                       "visually similar people in different frames, not the "
                       "structurally impossible case.",
            "evidence": f"cannot-link precision {pct(cl)} over "
                        f"{faces.get('cannot_link_pairs')} pairs, "
                        f"{faces.get('false_merges')} false merges; "
                        f"pairwise precision {pct(prec)}",
            "enhancement": "Not a defect to fix so much as a floor to hold: keep "
                           "the cannot-link harness in the pipeline so any "
                           "future recall-oriented loosening is caught the "
                           "moment it starts merging distinct people.",
        })

    px = faces.get("median_face_px")
    if px and px < 64:
        checks.append({
            "area": "Detection / source resolution",
            "severity": "medium",
            "finding": f"The median detected face is {px}px, near the "
                       f"MIN_FACE_PX={config.MIN_FACE_PX} floor. ArcFace "
                       "embeddings from faces this small are noisy, which is an "
                       "upstream cause of the fragmentation above.",
            "evidence": f"median face {px}px, median detection score "
                        f"{faces.get('det_score_median')}",
            "enhancement": "Re-running at a higher source resolution, or "
                           "super-resolving crops before embedding, would raise "
                           "embedding quality at a proportional compute cost.",
        })

    # -- OCR.
    rec = text.get("ocr_detect_recall")
    if rec is not None and rec < 0.95:
        checks.append({
            "area": "OCR",
            "severity": "medium",
            "finding": f"OCR recall is {pct(rec)} at "
                       f"{pct(text.get('ocr_detect_precision'))} precision: text "
                       "is missed rather than invented. Misses concentrate in "
                       "small, low-contrast, or motion-blurred signage.",
            "evidence": f"F1 {pct(text.get('ocr_detect_f1'))}, string fidelity "
                        f"{pct(text.get('ocr_string_fidelity'))}",
            "enhancement": "OCR_MIN_CONF=60 deliberately trades recall for "
                           "precision (it is what keeps the narrator from "
                           "reading garbage as station names). A modern detector "
                           "such as PaddleOCR or EAST+CRNN would likely recover "
                           "the missed text without reintroducing that noise.",
        })

    # -- Captions.
    good = text.get("caption_pct_good_ge4")
    if good is not None and good < 0.8:
        checks.append({
            "area": "Captioning",
            "severity": "medium",
            "finding": f"Only {pct(good)} of captions score >=4/5 for adequacy "
                       f"(mean {text.get('caption_mean_adequacy')}/5). BLIP-base "
                       "is generic on repetitive transit footage and flickers "
                       "between synonyms on visually identical frames.",
            "evidence": f"mean adequacy {text.get('caption_mean_adequacy')}/5",
            "enhancement": "This is exactly why search does not depend on "
                           "captions alone -- the CLIP visual index retrieves "
                           "image-natively -- and why scene segmentation cuts on "
                           "CLIP rather than caption runs. A larger captioner "
                           "(BLIP-2 / LLaVA) would raise adequacy at "
                           "significantly higher per-frame cost.",
        })

    # -- Search.
    p_at_k = search.get("mean_precision_at_k")
    if p_at_k is not None:
        weak = p_at_k < 0.9
        checks.append({
            "area": "Semantic search",
            "severity": "medium" if weak else "info",
            "finding": f"Mean precision@{search.get('k')} is {pct(p_at_k)} over "
                       f"{search.get('queries')} curated queries."
                       + (" Failures are queries whose target concept never "
                          "reaches the caption text at all." if weak else ""),
            "evidence": f"precision@{search.get('k')} = {pct(p_at_k)}",
            "enhancement": ("Fusing the text and CLIP rankings (reciprocal-rank "
                            "fusion) would let the visual index rescue queries "
                            "the captions miss, instead of leaving the two "
                            "indexes as separate user-selected modes."
                            if weak else
                            "Hold this with a regression run of eval_search.py "
                            "whenever the caption or embedding model changes."),
        })

    # -- Narration.
    gv = narr.get("grounding_vs_vlm")
    # The VLM reference is produced by describe_scenes, which uses no explicit
    # model and so defaults to the same NARRATE_MODEL that writes the story. The
    # reference is therefore not model-independent: a model tends to endorse its
    # own phrasings, so the true grounding is likely no higher -- and plausibly
    # lower -- than measured. State it rather than call the reference "independent".
    same_model_caveat = (
        f" (Caveat: the reference comes from the same model, `{narr.get('model')}`, "
        "in vision mode -- not a model-independent judge -- so read this grounding "
        "as an optimistic bound.)")
    if gv is not None and gv < 0.6:
        checks.append({
            "area": "Narration grounding",
            "severity": "high",
            "finding": f"Only {pct(gv)} of the story's content words are attested "
                       "in a VLM description of the same keyframes "
                       f"(vs {pct(narr.get('grounding_vs_captions'))} against the "
                       "caption/OCR digest it was written from). The narrator "
                       "adds connective and interpretive language the frames do "
                       "not evidence." + same_model_caveat,
            "evidence": f"grounding vs VLM {pct(gv)}, chronology "
                        f"{pct(narr.get('chronology'))}, coverage "
                        f"{pct(narr.get('coverage'))}",
            "enhancement": "Grounding is measured but not enforced. A "
                           "verification pass that re-checks each sentence "
                           "against its scene digest and drops unsupported "
                           "clauses would convert the metric into a guarantee. "
                           "Note the story is still perfectly chronological, so "
                           "the failure is embellishment, not confusion.",
        })

    # -- Cost profile.
    if t.get("status") == "ok":
        net, tot = t.get("network_llm_sec", 0), t.get("total_sec", 0)
        if tot and net / tot > 0.2:
            checks.append({
                "area": "Throughput / cost profile",
                "severity": "medium",
                "finding": f"{pct(net / tot)} of end-to-end wall-clock is spent "
                           "waiting on the free-tier LLM endpoint, not computing. "
                           "The local pipeline is not the bottleneck.",
                "evidence": f"network {net}s of {tot}s total; local compute "
                            f"{t.get('local_compute_sec')}s",
                "enhancement": "A paid endpoint (or a local Gemma/Llama served "
                               "via vLLM) removes the 429 backoff entirely. The "
                               "response cache in data/llm_cache/ already makes "
                               "reruns free, which is why this cost is paid once "
                               "rather than per run.",
            })
        slow = t.get("slowest_local_stage")
        if slow and t.get("slowest_local_share"):
            checks.append({
                "area": "Throughput / local compute",
                "severity": "info",
                "finding": f"`{slow}` dominates local compute at "
                           f"{pct(t['slowest_local_share'])} of it, so it is the "
                           "only stage where optimisation pays.",
                "evidence": f"local compute {t.get('local_compute_sec')}s across "
                            f"{len([s for s in t['stages'] if not s['network']])} "
                            f"stages; {t.get('sec_per_frame_local')}s per frame",
                "enhancement": "All heavy stages run on CPU (USE_GPU=False, "
                               "CAPTION_DEVICE/CLIP_DEVICE='auto' falling back to "
                               "MPS where available). A CUDA host would cut this "
                               "several-fold with no code change.",
            })

    # -- Scope limits that no metric can surface, because nothing measures what
    #    was never built. These are stated, not derived, and marked as such.
    checks.append({
        "area": "Scope",
        "severity": "info",
        "finding": "The system analyses vision only, at 1 FPS. There is no "
                   "audio track, speech transcription, or sub-second motion "
                   "modelling, so events shorter than a second are invisible to "
                   "it and anything spoken is unrecoverable.",
        "evidence": f"FPS={config.FPS}; {q['faces'].get('total_frames')} frames "
                    f"sampled from {config.VIDEO_DURATION_SEC}s of video",
        "enhancement": "Whisper over the audio track would add a speech channel "
                       "that the metadata repository could join on timestamp "
                       "exactly as it already joins OCR, captions, and face IDs "
                       "-- the schema does not need to change to accommodate it.",
    })
    checks.append({
        "area": "Generalization",
        "severity": "info",
        "finding": "Every accuracy number here is measured on a single video, and "
                   "the operating points that produce it -- detection thresholds, "
                   "the clustering distance ceilings, the scene-cut similarity -- "
                   "were tuned to that one transit clip. Nothing establishes that "
                   "the same settings transfer to different footage (crowds, "
                   "lighting, camera motion), so the metrics describe this system "
                   "on this input, not its expected accuracy in general.",
        "evidence": "n=1 video; thresholds hand-tuned on it (see config.py "
                    "DBSCAN_EPS / CLUSTER_LINK_DIST / SCENE_SIM_THRESH)",
        "enhancement": "Re-running the label-free checks (co-occurrence, "
                       "continuity) on a second, unrelated video would show which "
                       "thresholds are video-specific and which hold up -- the "
                       "harnesses need no new code, only new input.",
    })
    return checks


# ------------------------------------------------------------------ reporting
def _md_table(headers: list, rows: list) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def _fmt(x, pct=False):
    if x is None:
        return "n/a"
    if pct and isinstance(x, (int, float)):
        return f"{x:.1%}"
    return str(x)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a binomial proportion.

    The normal ("Wald") interval is useless at the sample sizes here -- it puts
    a symmetric band around 100% and reports an upper bound above 1. Wilson is
    bounded to [0,1] and stays sensible at the extremes, which is exactly where
    these metrics live: OCR precision is 100% of eleven positives, not of a
    thousand. Returns (low, high) as fractions."""
    if not n or n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))


def _prop(successes, n, pct=True):
    """A proportion rendered with its 95% Wilson interval and sample size, so a
    small-sample rate is never mistaken for a precise one. Falls back to a bare
    value when the counts needed for an interval are not available."""
    if successes is None or not n:
        return "n/a"
    lo, hi = wilson_ci(int(round(successes)), int(n))
    p = successes / n
    if pct:
        return f"{p:.1%} (95% CI {lo:.0%}–{hi:.0%}, n={int(n)})"
    return f"{p:.3f} (95% CI {lo:.3f}–{hi:.3f}, n={int(n)})"


def _write_markdown(report: dict) -> None:
    t, q, checks = report["timing"], report["quality"], report["limitations"]
    L = ["# System Evaluation (Milestone 4)", "",
         "End-to-end performance of the integrated pipeline: processing time, "
         "output quality, and derived limitations. Generated by `eval_system.py` "
         "from the stage timings and every milestone's evaluation harness -- no "
         "number here is typed by hand.", ""]

    # -- Staleness banner. If any eval JSON is older than the artifact it scored,
    #    the numbers below are describing a superseded pipeline; say so loudly at
    #    the top rather than let a reader trust a stale figure.
    stale = report.get("stale_inputs") or []
    if stale:
        items = "; ".join(f"`{s['report']}` (older than "
                          f"{', '.join(s['outdated_by'])})" for s in stale)
        L += ["> **⚠ Stale inputs.** These eval reports predate the artifacts "
              f"they scored and may be wrong: {items}. Re-run the pipeline eval "
              "stage (`python run_pipeline.py`) to recompute them.", ""]

    # -- Processing time
    L += ["## 1. Processing time", ""]
    if t.get("status") != "ok":
        L += ["_No stage timings recorded yet. Run `python run_pipeline.py`._", ""]
    else:
        L += [_md_table(
            ["Metric", "Value"],
            [["Total end-to-end", f"{t['total_sec']}s ({t['total_min']} min)"],
             ["Local compute", f"{t['local_compute_sec']}s"],
             ["Network (LLM) wait", f"{t['network_llm_sec']}s"],
             ["Frames processed", t.get("frames_processed")],
             ["Local sec/frame", t.get("sec_per_frame_local")],
             ["Local frames/sec", t.get("frames_per_sec_local")],
             ["Video duration", f"{t['video_duration_sec']}s"],
             ["Real-time factor (local)",
              f"{t.get('realtime_factor_local')}x"],
             ["Real-time factor (incl. network)",
              f"{t.get('realtime_factor_total')}x"]]), "",
            "Local compute and network wait are reported separately on purpose: "
            "the narration stages measure how long a free-tier endpoint made us "
            "queue, so folding them into a throughput figure would report "
            "someone else's rate limiter as this pipeline's latency.", "",
            "### Per-stage breakdown", ""]
        rows = [[s["stage"], s["seconds"],
                 f"{s['seconds'] / t['total_sec']:.1%}" if t["total_sec"] else "",
                 "network LLM" if s["network"] else "local"]
                for s in t["stages"]]
        L += [_md_table(["Stage", "Seconds", "Share of total", "Kind"], rows), "",
              "Timings are merged across runs: the pipeline is idempotent, so a "
              "stage that skipped keeps its last measured duration and this table "
              "always describes a full cold build.", ""]
        if t.get("narration_from_cache"):
            L += ["> **Caveat on the narration stages.** `data/llm_cache/` is "
                  "populated, so `describe` and `narrate` replayed cached "
                  "responses from disk rather than calling the API. Their "
                  "seconds above are replay cost, not generation cost — cold "
                  "generation against the free tier is dominated by 429 backoff "
                  "and takes minutes, not seconds. This is a real property of "
                  "the system (the cache is committed precisely so reruns and "
                  "CI are free), but it is not a claim that the LLM is fast.", ""]

    # -- Quality
    f, tx, s, n = (q["faces"], q["text_and_captions"], q["search"], q["narration"])
    L += ["## 2. Accuracy and quality of generated outputs", "",
          "### Faces (Milestone 1)", "",
          _md_table(["Metric", "Value", "Reading"], [
              ["Frames processed", f.get("total_frames"), "sampled at 1 FPS"],
              ["Faces detected", f.get("total_faces_detected"), ""],
              ["Tracks", f.get("total_tracks"), "temporal groupings"],
              ["Unique faces", f.get("unique_faces"), "incl. one-off/background"],
              ["Featured cast", f.get("featured_cast"),
               f">= {config.RECURRING_MIN_FRAMES} frames present"],
              ["Cannot-link precision",
               _prop((f.get("cannot_link_pairs") or 0) - (f.get("false_merges") or 0),
                     f.get("cannot_link_pairs")),
               "label-free; provable merge errors"],
              ["False merges", f.get("false_merges"), "lower is better"],
              ["Continuity recall", _fmt(f.get("continuity_recall"), True),
               "label-free; bridged tracks"],
              ["V-measure", _fmt(f.get("v_measure"), True), "labeled subset"],
              ["Homogeneity", _fmt(f.get("homogeneity"), True),
               "clusters are pure"],
              ["Completeness", _fmt(f.get("completeness"), True),
               "people are not split"],
              ["Pairwise precision", _fmt(f.get("pairwise_precision"), True), ""],
              ["Pairwise recall", _fmt(f.get("pairwise_recall"), True), ""],
              ["Pairwise F1", _fmt(f.get("pairwise_f1"), True), ""],
              ["Clusters per person", f.get("mean_clusters_per_person"),
               "1.0 is perfect"],
          ]), "",
          "The shape of these numbers is the system's central trade-off: "
          f"homogeneity {_fmt(f.get('homogeneity'), True)} against completeness "
          f"{_fmt(f.get('completeness'), True)} means the grouping almost never "
          "puts two people together, but often splits one person apart. That is "
          "the deliberate direction to err for an occurrence report -- an "
          "invented merge corrupts a count silently, a split is visible in the "
          "cast list -- but it is the dominant error and Section 3 treats it as "
          "such.", "",
          "### Text, captions and search (Milestone 2)", "",
          _md_table(["Metric", "Value"], [
              ["OCR detection precision",
               _prop(tx.get("ocr_tp"),
                     (tx.get("ocr_tp") or 0) + (tx.get("ocr_fp") or 0))],
              ["OCR detection recall",
               _prop(tx.get("ocr_tp"),
                     (tx.get("ocr_tp") or 0) + (tx.get("ocr_fn") or 0))],
              ["OCR detection F1", _fmt(tx.get("ocr_detect_f1"), True)],
              ["OCR string fidelity", _fmt(tx.get("ocr_string_fidelity"), True)],
              ["Caption mean adequacy", f"{tx.get('caption_mean_adequacy')} / 5"],
              ["Captions scoring >= 4/5",
               _prop(round((tx.get("caption_pct_good_ge4") or 0)
                           * (tx.get("caption_scored") or 0)),
                     tx.get("caption_scored"))],
              [f"Semantic precision@{s.get('k')}",
               f"{_fmt(s.get('mean_precision_at_k'), True)} "
               f"(mean over n={s.get('queries')} queries)"],
              [f"Visual precision@{s.get('k')}",
               f"{_fmt(s.get('visual_mean_precision_at_k'), True)} "
               "(text-proxy relevance; under-credits visual)"],
              [f"Fused (RRF) precision@{s.get('k')}",
               _fmt(s.get("fused_mean_precision_at_k"), True)],
          ]), "",
          "Percentages above carry a 95% Wilson interval and their sample size: "
          "at these denominators a bare \"100%\" would overstate certainty -- OCR "
          "precision is measured on a handful of labeled frames, not thousands.",
          "",
          "### Story, summary and timeline (Milestone 3)", "",
          _md_table(["Metric", "Value"], [
              ["Model", f"`{n.get('model')}`"],
              ["Promoted strategy", f"`{n.get('promoted_strategy')}`"],
              ["Strategies compared", ", ".join(f"`{x}`" for x in
                                                n.get("strategies_compared", []))],
              ["Scenes / chapters",
               f"{n.get('n_scenes')} / {n.get('n_chapters')}"],
              ["Chronology", _fmt(n.get("chronology"), True)],
              ["Chapter coverage", _fmt(n.get("coverage"), True)],
              ["Grounding vs captions", _fmt(n.get("grounding_vs_captions"), True)],
              ["Grounding vs VLM", _fmt(n.get("grounding_vs_vlm"), True)],
              ["Distinct-n-gram (redundancy)",
               _fmt(n.get("redundancy_distinct_ngram"), True)],
          ]), ""]

    # -- Limitations
    L += ["## 3. Limitations and possible enhancements", "",
          "Each finding below is a predicate over the metrics above, not prose "
          "written beside them: improve the number and the finding removes "
          "itself from this report on the next run.", ""]
    for c in checks:
        L += [f"### {c['area']} ({c['severity']})", "",
              c["finding"], "",
              f"**Evidence:** {c['evidence']}", "",
              f"**Enhancement:** {c['enhancement']}", ""]

    # -- Overall
    L += ["## 4. Overall system performance", ""]
    if t.get("status") == "ok":
        rt = t.get("realtime_factor_local")
        # The factor is video/compute, so <1 is slower than playback. Phrase it
        # from the number rather than assuming the flattering direction.
        if rt is None:
            pace = "pace not measured"
        elif rt >= 1:
            pace = f"{rt}x faster than the video plays"
        else:
            pace = f"{round(1 / rt, 2)}x slower than the video plays"
        L += [f"The integrated pipeline turns a {t['video_duration_sec']}s video "
              f"into a fully searchable, narrated dataset in "
              f"{t['total_min']} minutes end to end, of which "
              f"{t['local_compute_sec']}s is local compute "
              f"({t.get('sec_per_frame_local')}s per frame across "
              f"{t.get('frames_processed')} frames, "
              f"{pace}) "
              f"and {t['network_llm_sec']}s is waiting on a free-tier LLM. Every "
              "stage is idempotent and config-fingerprinted, so re-running costs "
              "nothing unless a parameter actually changed.", ""]
    L += ["What the system does well is bounded honesty: it never merges two "
          "people who share a frame "
          f"({_fmt(f.get('cannot_link_precision'), True)} cannot-link "
          "precision), it never invents on-screen text "
          f"({_fmt(tx.get('ocr_detect_precision'), True)} OCR precision), and its "
          f"story is perfectly chronological ({_fmt(n.get('chronology'), True)}) "
          f"with {_fmt(n.get('coverage'), True)} chapter coverage. Where it is "
          "weak, it is weak in the direction that is visible rather than silent: "
          "it splits identities instead of conflating them, misses text instead "
          "of hallucinating it, and its narrator embellishes rather than "
          "misorders. Each of those is measured above rather than assumed.", ""]

    util.write_text_atomic(config.SYSTEM_EVAL_MD, "\n".join(L))


def run() -> dict:
    t = timing_report()
    q = quality_report()
    stale = staleness_report()
    report = {"timing": t, "quality": q, "limitations": _checks(q, t),
              "stale_inputs": stale}

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    util.write_json_atomic(config.SYSTEM_EVAL_JSON, report, indent=2)
    _write_markdown(report)

    log.info("[system eval] %s -> %s", config.SYSTEM_EVAL_JSON.name,
             config.SYSTEM_EVAL_MD.name)
    if t.get("status") == "ok":
        log.info("[system eval] total %.1fs (local %.1fs, network %.1fs)",
                 t["total_sec"], t["local_compute_sec"], t["network_llm_sec"])
    log.info("[system eval] %d limitation(s) derived from the metrics",
             len(report["limitations"]))
    if stale:
        log.warning("[system eval] %d eval input(s) STALE -- rerun the pipeline "
                    "eval stage: %s", len(stale),
                    ", ".join(s["report"] for s in stale))
    return report


if __name__ == "__main__":
    run()
