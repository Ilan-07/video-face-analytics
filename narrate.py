"""Milestone 3c: story, summary and event timeline.

Tasks
-----
Story generation    -- a coherent, chronological narrative of the whole video,
                       built from the per-frame captions (Task 1 + 2), with four
                       prompt-engineering strategies compared side by side (Task 3).
Video summarization -- one overall summary of the key events and scenes.
Event timeline      -- (timestamp, short description) for every significant event.

Why this fits in one prompt
---------------------------
scenes.py compressed 1415 frames / 342 distinct captions into ~36 scene digests,
about 2k tokens. Gemma 4's context is 262k, so the model sees the ENTIRE video at
once. Chronological coherence is therefore structural -- there is no chunking, no
map-reduce, and no seam at which the narrative can lose the thread. That is the
single reason this milestone reads as one story rather than 36 stitched summaries.

Trust boundary
--------------
The timeline is the one output with a machine-readable contract, so it is
VALIDATED rather than trusted: timestamps must parse, lie inside the video, and
run forwards. On violation we repair (sort, drop out-of-range) and, if the result
is unusable, fall back to the scene boundaries -- which are ground truth from
scenes.py. Same defensive posture as caption.py's echo-fix fixed-point guard.
"""
import argparse
import json
import re

import config
import llm
import util
from llm import extract_json_array   # noqa: F401  (re-exported: tests import it here)
from search import fmt_ts

log = util.get_logger()


# ---- pure helpers (unit-tested without models or network) ----

def parse_mmss(text: str) -> float | None:
    """'12:23' -> 743.0 seconds. None when it is not a timestamp."""
    m = re.fullmatch(r"(\d{1,2}):([0-5]\d)", (text or "").strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def validate_timeline(events, max_sec: float):
    """Keep events whose timestamp parses and lies in [0, max_sec], then sort by
    time. Returns (clean_events, problems). Pure -- no I/O, unit tested.

    The model is not trusted to be monotonic or in-range; sorting rather than
    rejecting keeps a usable timeline while `problems` records what was wrong so
    eval_story.py can report it honestly."""
    clean, problems = [], []
    for e in events or []:
        if not isinstance(e, dict):
            problems.append(f"not an object: {e!r}")
            continue
        sec = parse_mmss(str(e.get("timestamp", "")))
        desc = str(e.get("description", "")).strip()
        if sec is None:
            problems.append(f"unparseable timestamp: {e.get('timestamp')!r}")
            continue
        if not 0 <= sec <= max_sec:
            problems.append(f"timestamp out of range: {e.get('timestamp')!r}")
            continue
        if not desc:
            problems.append(f"empty description at {e.get('timestamp')!r}")
            continue
        clean.append({"timestamp": fmt_ts(sec), "timestamp_sec": sec,
                      "description": desc})
    was_ordered = all(a["timestamp_sec"] <= b["timestamp_sec"]
                      for a, b in zip(clean, clean[1:]))
    if not was_ordered:
        problems.append("events were not in chronological order (sorted)")
        clean.sort(key=lambda e: e["timestamp_sec"])
    return clean, problems


def scene_digest(scenes, descriptions=None) -> str:
    """Render scenes.json as the compact text block every prompt is built around.

    `descriptions` (scene_index -> VLM text) swaps the BLIP caption for the vision
    model's description -- the ablation. Pure."""
    lines = []
    for s in scenes:
        if s["is_title_card"]:
            lines.append(f'[{s["start_mmss"]}-{s["end_mmss"]}] TITLE CARD: '
                         f'"{s["chapter_label"]}"')
            continue
        text = (descriptions or {}).get(s["scene_index"],
                                        s["representative_caption"])
        faces = ", ".join(s["face_ids"]) if s["face_ids"] else "none"
        parts = [f'[{s["start_mmss"]}-{s["end_mmss"]}] scene {s["scene_index"]} '
                 f'({s["chapter_label"]}, {s["n_frames"]}s)',
                 f"  visual: {text}"]
        if s["ocr_texts"]:
            parts.append(f'  on-screen text: {"; ".join(s["ocr_texts"])}')
        parts.append(f"  recurring faces present: {faces}")
        lines.append("\n".join(parts))
    return "\n".join(lines)


def split_segments(story: str, labels) -> dict:
    """Map chapter_label -> that chapter's prose, by splitting on '## <label>'
    markdown headings. Returns {} when the model ignored the heading contract.

    Text before the first matched heading is the lead-in, assigned to
    "Introduction" when that is a label: the model narrates the intro card as an
    opening paragraph rather than under its own heading, so without this the
    Introduction segment would be spuriously empty. Pure."""
    found, current, buf, lead = {}, None, [], []
    for line in (story or "").splitlines():
        head = re.fullmatch(r"#{1,3}\s*(.+?)\s*", line)
        if head and head.group(1) in labels:
            if current:
                found[current] = "\n".join(buf).strip()
            current, buf = head.group(1), []
        elif current:
            buf.append(line)
        elif line.strip() and not line.startswith("#"):
            lead.append(line)
    if current:
        found[current] = "\n".join(buf).strip()
    if "Introduction" in labels and "Introduction" not in found and lead:
        found["Introduction"] = "\n".join(lead).strip()
    return found


# ---- prompt-engineering strategies (Task 3) ----
# Four framings of the SAME digest. eval_story.py scores them on chronology,
# grounding, chapter coverage and redundancy so the comparison is measured, not
# asserted. Every strategy is given the identical heading contract so the story
# can be split back into chapters; only the *framing* differs.

_CONTRACT = (
    "Write one section per title card, in order. Begin each section with a "
    "markdown heading naming that line exactly, e.g. '## Bakerloo Line'. "
    "Mention the timestamp of at least one moment in each section as m:ss."
)

_EXAMPLES = """Here are two worked examples of the level of detail expected.

Digest excerpt:
[0:06-0:08] TITLE CARD: "Bakerloo Line"
[0:09-0:11] scene 2 (Bakerloo Line, 3s)
  visual: a sign that says ' emannment '
  on-screen text: EMBANKMENT
  recurring faces present: none

Written as:
## Bakerloo Line
The tour opens on the Bakerloo Line at 0:06. A roundel slides into view at 0:09 \
naming the station: Embankment.

Digest excerpt:
[21:41-21:43] TITLE CARD: "Waterloo City Line"
[21:46-23:35] scene 35 (Waterloo City Line, 110s)
  visual: a red and white train is parked at the station
  recurring faces present: none

Written as:
## Waterloo City Line
The final line begins at 21:41. A red and white train sits waiting at the platform \
through to the end of the video at 23:35.
"""


def _zero_shot(digest: str) -> str:
    return (
        "Below is a scene-by-scene digest of a video, in chronological order.\n\n"
        f"{digest}\n\n"
        f"Write a coherent story describing the complete video. {_CONTRACT}")


def _few_shot(digest: str) -> str:
    return (
        "Below is a scene-by-scene digest of a video, in chronological order.\n\n"
        f"{digest}\n\n{_EXAMPLES}\n"
        f"Now write the complete story for the whole video. {_CONTRACT}")


def _chain_of_thought(digest: str) -> str:
    return (
        "Below is a scene-by-scene digest of a video, in chronological order.\n\n"
        f"{digest}\n\n"
        "First, think step by step: list the title cards in order, note what "
        "happens between each pair, and identify the single through-line that "
        "connects them. Put that reasoning inside <thinking></thinking> tags.\n\n"
        "Then, after the closing tag, write the finished story. "
        f"{_CONTRACT}\n"
        "Only the text after </thinking> will be published.")


def _structured_role(digest: str) -> str:
    return (
        "You are a documentary narrator writing voice-over for an archival "
        "transit film. You are given a scene-by-scene digest of the footage, in "
        "chronological order. Each entry gives a timestamp range, what is visible, "
        "any on-screen text, and which recurring faces appear.\n\n"
        f"{digest}\n\n"
        "Write the voice-over as a continuous, chronological story.\n\n"
        "Rules:\n"
        "- Follow the timestamps strictly. Never describe a later moment before "
        "an earlier one.\n"
        "- Ground every claim in the digest. If the digest does not say a train "
        "is present, do not put one there.\n"
        "- Name stations only when the on-screen text names them.\n"
        "- Vary the language; do not repeat the same sentence shape.\n"
        f"- {_CONTRACT}\n"
        "- No preamble, no closing commentary. Begin with the first heading.")


PROMPTS = {
    "zero_shot": _zero_shot,
    "few_shot": _few_shot,
    "chain_of_thought": _chain_of_thought,
    "structured_role": _structured_role,
}

_THINK = re.compile(r".*</thinking>", re.S | re.I)


def strip_reasoning(text: str) -> str:
    """Drop a <thinking>...</thinking> preamble (chain_of_thought). Pure."""
    return _THINK.sub("", text or "", count=1).strip()


_SUMMARY_PROMPT = """Below is a scene-by-scene digest of a video, in chronological order.

{digest}

Write an overall summary of the video in 150-200 words. Highlight the key events \
and the most important scenes, and say what the video is fundamentally a record \
of. Do not list every scene. Prose only, no headings, no bullet points."""

_TIMELINE_PROMPT = """Below is a scene-by-scene digest of a video, in chronological order.

{digest}

Produce a timeline of the significant events in this video.

Return ONLY a JSON array, no prose and no code fence. Each element must be an \
object with exactly two keys:
  "timestamp"   - the time the event occurs, formatted "m:ss"
  "description" - a short description of the event, at most 12 words

Rules:
- Be granular: aim for 2 to 3 events per line, not just the title cards. A new \
title card, arriving at a named station, a train pulling in with its doors \
opening, or people moving along a platform are all events worth an entry.
- Start each line's title card with "Title card:" then the line name.
- For other events, describe what happens concretely, naming the station when \
the on-screen text gives it.
- Timestamps must be in strictly increasing order.
- Every timestamp must come from a scene above; do not invent times.
- The video ends at {end}. No timestamp may exceed it."""


# ---- generation ----

def _load_scenes():
    if not config.SCENES_JSON.exists():
        raise RuntimeError(
            f"{config.SCENES_JSON.name} not found. Run `python scenes.py` first.")
    with open(config.SCENES_JSON) as f:
        return json.load(f)


def _load_descriptions():
    if not config.SCENE_DESC_JSON.exists():
        raise RuntimeError(
            f"{config.SCENE_DESC_JSON.name} not found. "
            "Run `python describe_scenes.py` first, or use --source captions.")
    with open(config.SCENE_DESC_JSON) as f:
        return {int(r["scene_index"]): r["description"] for r in json.load(f)}


def _digest_for(scenes, source: str) -> str:
    descriptions = _load_descriptions() if source == "vlm" else None
    return scene_digest(scenes, descriptions)


def _story_path(strategy: str, source: str):
    suffix = "" if source == "captions" else f"_{source}"
    return config.REPORT_DIR / f"story_{strategy}{suffix}.md"


def generate_story(scenes, strategy: str, source: str = "captions") -> str:
    digest = _digest_for(scenes, source)
    raw = llm.generate(PROMPTS[strategy](digest))
    story = strip_reasoning(raw) if strategy == "chain_of_thought" else raw.strip()

    path = _story_path(strategy, source)
    header = (f"# Story — `{strategy}` prompt, `{source}` source\n\n"
              f"*Generated by `{config.NARRATE_MODEL}` from "
              f"{len(scenes)} scene digests. Do not edit — regenerate with "
              f"`python narrate.py --strategy {strategy} --source {source}`.*\n\n")
    path.write_text(header + story + "\n")
    log.info("story [%s/%s]: %d chars -> %s", strategy, source, len(story),
             path.name)
    return story


def generate_summary(scenes, source: str = "captions") -> str:
    digest = _digest_for(scenes, source)
    summary = llm.generate(_SUMMARY_PROMPT.format(digest=digest),
                           max_tokens=600).strip()
    # NOT summary.md: analytics.py owns that (the Milestone 1 face-analytics
    # summary). This is the video-content summary, a different artifact.
    path = config.REPORT_DIR / "video_summary.md"
    path.write_text(f"# Video summary\n\n*Generated by `{config.NARRATE_MODEL}` "
                    f"from {len(scenes)} scene digests.*\n\n{summary}\n")
    log.info("summary: %d words -> %s", len(summary.split()), path.name)
    return summary


def _fallback_timeline(scenes):
    """Ground-truth timeline straight from scenes.py, used when the model's JSON
    is unusable. Never wrong, just less expressive."""
    return [{"timestamp": s["start_mmss"], "timestamp_sec": s["start_sec"],
             "description": (f'Title card: {s["chapter_label"]}'
                             if s["is_title_card"] else
                             s["representative_caption"] or "scene change")}
            for s in scenes]


def generate_timeline(scenes, source: str = "captions") -> list:
    digest = _digest_for(scenes, source)
    raw = llm.generate(
        _TIMELINE_PROMPT.format(digest=digest,
                                end=fmt_ts(config.VIDEO_DURATION_SEC)),
        max_tokens=2000)

    events, problems = validate_timeline(extract_json_array(raw),
                                         config.VIDEO_DURATION_SEC)
    for p in problems:
        log.warning("timeline: %s", p)
    if len(events) < len(scenes) // 3:
        log.warning("timeline: only %d usable events from the model -- falling "
                    "back to scene boundaries", len(events))
        events, problems = _fallback_timeline(scenes), problems + ["used fallback"]

    with open(config.TIMELINE_JSON, "w") as f:
        json.dump({"model": config.NARRATE_MODEL, "source": source,
                   "problems": problems, "events": events}, f, indent=2)

    lines = ["# Event timeline", "",
             f"*Generated by `{config.NARRATE_MODEL}` from {len(scenes)} scene "
             f"digests. {len(events)} events.*", "",
             "| Timestamp | Event |", "|---|---|"]
    lines += [f'| {e["timestamp"]} | {e["description"]} |' for e in events]
    (config.REPORT_DIR / "timeline.md").write_text("\n".join(lines) + "\n")
    log.info("timeline: %d events (%d problems) -> %s",
             len(events), len(problems), config.TIMELINE_JSON.name)
    return events


def run(strategy: str | None = None, source: str = "captions",
        all_strategies: bool = False) -> dict:
    config.ensure_dirs()
    scenes = _load_scenes()
    labels = [s["chapter_label"] for s in scenes if s["is_title_card"]]

    wanted = config.STORY_STRATEGIES if all_strategies else [
        strategy or config.STORY_STRATEGY]
    stories = {s: generate_story(scenes, s, source) for s in wanted}

    promoted = config.STORY_STRATEGY if config.STORY_STRATEGY in stories \
        else wanted[0]
    story = stories[promoted]

    # The summary and timeline are deliverables of the assignment, which specifies
    # the captions as the input; the VLM run exists only to produce an ablation
    # story. Regenerating them from `vlm` would overwrite the canonical artifacts
    # and burn two requests of the free tier's daily allowance for nothing.
    if source != "captions":
        log.info("source=%s: story only (summary/timeline stay caption-sourced)",
                 source)
        return {"strategies": list(stories), "promoted": promoted,
                "events": 0, "segments": 0}

    summary = generate_summary(scenes, source)
    events = generate_timeline(scenes, source)

    segments = split_segments(story, set(labels))
    # The intro card carries no footage, and the structured_role prompt tells the
    # model to open at the first line heading, so a missing "Introduction" section
    # is expected, not a defect. Only a missing *line* chapter is a real gap.
    missing_lines = [lab for lab in labels
                     if lab != "Introduction" and lab not in segments]
    if missing_lines:
        log.warning("story: %d line chapter(s) had no heading parsed (%s) -- their "
                    "story.json segment text will be empty",
                    len(missing_lines), ", ".join(missing_lines))

    with open(config.STORY_JSON, "w") as f:
        json.dump({"model": config.NARRATE_MODEL, "source": source,
                   "strategy": promoted, "strategies_generated": list(stories),
                   "summary": summary, "story": story,
                   "segments": [{"chapter_label": lab, "text": segments.get(lab, "")}
                                for lab in labels],
                   "n_events": len(events)}, f, indent=2)
    log.info("story repository -> %s (strategy=%s, source=%s)",
             config.STORY_JSON.name, promoted, source)
    return {"strategies": list(stories), "promoted": promoted,
            "events": len(events), "segments": len(segments)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Story, summary and event timeline (Milestone 3)")
    ap.add_argument("--strategy", default=None,
                    choices=config.STORY_STRATEGIES + ["all"],
                    help=f"prompt strategy (default: {config.STORY_STRATEGY})")
    ap.add_argument("--source", default="captions", choices=["captions", "vlm"],
                    help="narrate from BLIP captions (assignment) or from the "
                         "VLM keyframe descriptions (ablation)")
    args = ap.parse_args()
    run(strategy=None if args.strategy == "all" else args.strategy,
        source=args.source, all_strategies=args.strategy == "all")
