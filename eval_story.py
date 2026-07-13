"""Milestone 3 eval: score the generated stories instead of asserting they're good.

Narrative quality resists automatic scoring, so we measure four things that are
objectively checkable and that map onto the assignment's actual requirements:

1. Chronology  -- the assignment demands the narrative "follows the chronological
                  sequence of events". We parse every m:ss the story cites and
                  measure the fraction of adjacent pairs that do not go backwards.
                  1.0 means the story never time-travels. This turns Task 2 from a
                  claim into a check.
2. Grounding   -- of the story's content words, what fraction is attested by the
                  source material? Scored twice, and the difference is the point:
                    * vs the CAPTION digest -- did the narrator invent things the
                      captions never said?
                    * vs the VLM keyframe descriptions -- did the narrator repeat
                      things BLIP hallucinated? This reference is independent of
                      the captions the story was generated from, so it is not
                      circular. (Requires describe_scenes.py.)
3. Coverage    -- fraction of the 12 chapters the story actually mentions. Catches
                  a fluent narrative that quietly skips half the video.
4. Redundancy  -- distinct-n-gram ratio. BLIP's captions are extremely repetitive;
                  a narrator that merely paraphrases them scores low.

Plus `caption_adequacy`: token overlap between BLIP's caption for a scene and the
VLM's description of that same keyframe. This is the headline ablation number --
it measures the ceiling the assignment's specified input imposes on any narrator.

Every section degrades gracefully to {"status": ...} when its inputs are missing,
so the harness runs on a partial pipeline. Writes reports/eval_story.{json,md}
and reports/STORY_COMPARISON.md.
"""
import json
import re

import config
import util

log = util.get_logger()

_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "there", "then", "than",
    "we", "our", "you", "your", "they", "their", "he", "she", "his", "her",
    "not", "no", "into", "out", "up", "down", "over", "under", "through",
    "video", "scene", "footage", "camera", "shot", "moment", "line",
}


# ---- pure helpers (unit-tested without models or network) -------------------

def cited_timestamps(text: str) -> list:
    """Every m:ss the story cites, in the order it cites them, as seconds."""
    return [int(m.group(1)) * 60 + int(m.group(2))
            for m in re.finditer(r"\b(\d{1,2}):([0-5]\d)\b", text or "")]


def chronology_score(seconds: list) -> float:
    """Fraction of adjacent cited timestamps that do not go backwards.

    1.0 = perfectly chronological. Fewer than two citations is vacuously 1.0 --
    reported alongside the citation count so an empty story can't look perfect."""
    if len(seconds) < 2:
        return 1.0
    ok = sum(a <= b for a, b in zip(seconds, seconds[1:]))
    return ok / (len(seconds) - 1)


def content_words(text: str, min_len: int | None = None) -> set:
    """Lowercase alphabetic words worth grounding: long enough, not stopwords."""
    min_len = config.STORY_GROUND_MIN_LEN if min_len is None else min_len
    return {w for w in re.findall(r"[a-z]+", (text or "").lower())
            if len(w) >= min_len and w not in _STOP}


def grounding(text: str, reference: str) -> float:
    """Fraction of the text's content words that appear in the reference corpus.

    A hallucination proxy: 1.0 means every substantive word the narrator used is
    attested by the source. It cannot detect a true word used in a false way, and
    it penalises legitimate synonyms -- so read it comparatively across
    strategies, not as an absolute truth score."""
    words = content_words(text)
    if not words:
        return 0.0
    ref = content_words(reference, min_len=1)
    return len(words & ref) / len(words)


def scene_coverage(text: str, labels: list) -> float:
    """Fraction of chapter labels the story mentions (case-insensitive)."""
    if not labels:
        return 0.0
    low = (text or "").lower()
    return sum(lab.lower() in low for lab in labels) / len(labels)


def distinct_ngram_ratio(text: str, n: int | None = None) -> float:
    """Distinct n-grams / total n-grams. Low = the narrator repeats itself."""
    n = config.STORY_EVAL_NGRAM if n is None else n
    words = re.findall(r"[a-z']+", (text or "").lower())
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def caption_adequacy(caption: str, description: str) -> float:
    """Share of the VLM description's content words that BLIP's caption also has.

    Low values mean the caption omitted most of what is visibly in the frame."""
    ref = content_words(description)
    if not ref:
        return 0.0
    return len(content_words(caption) & ref) / len(ref)


def timeline_in_scene_bounds(events: list, scenes: list, tol: float = 1.05) -> float:
    """Fraction of timeline events whose timestamp falls inside some scene span.
    Every second of the video is covered by a scene, so a low value means the
    model invented a timestamp that validate_timeline's range check let through.

    `tol` (default one sampling interval, ~1.001s at 1 FPS) absorbs the fact that
    an m:ss event resolves only to the second while scene bounds carry the source
    fps's sub-second offset -- e.g. event 2:18 = 138.0s vs a scene starting at
    138.138s is a match, not a miss. Pure."""
    if not events:
        return 0.0
    spans = [(s["start_sec"], s["end_sec"]) for s in scenes]
    inside = sum(any(a - tol <= e["timestamp_sec"] <= b + tol for a, b in spans)
                 for e in events)
    return inside / len(events)


# ---- scoring ---------------------------------------------------------------

def _load(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _story_text(strategy: str, source: str = "captions") -> str | None:
    suffix = "" if source == "captions" else f"_{source}"
    path = config.REPORT_DIR / f"story_{strategy}{suffix}.md"
    if not path.exists():
        return None
    # drop the generated-by header (it cites the model name and scene count,
    # which would otherwise inflate grounding and pollute the timestamp parse)
    body = path.read_text().split("\n\n", 2)
    return body[-1] if len(body) == 3 else path.read_text()


def score_story(text: str, caption_ref: str, vlm_ref: str, labels: list) -> dict:
    ts = cited_timestamps(text)
    out = {
        "words": len(text.split()),
        "timestamps_cited": len(ts),
        "chronology": round(chronology_score(ts), 3),
        "coverage": round(scene_coverage(text, labels), 3),
        "redundancy_distinct_ngram": round(distinct_ngram_ratio(text), 3),
        "grounding_vs_captions": round(grounding(text, caption_ref), 3),
    }
    if vlm_ref:
        out["grounding_vs_vlm"] = round(grounding(text, vlm_ref), 3)
    return out


def run() -> dict:
    config.ensure_dirs()
    scenes = _load(config.SCENES_JSON)
    if not scenes:
        metrics = {"status": "no scenes.json (run scenes.py)"}
        _write(metrics)
        return metrics

    labels = [s["chapter_label"] for s in scenes if s["is_title_card"]]
    caption_ref = " ".join(
        [s["representative_caption"] for s in scenes]
        + [t for s in scenes for t in s["ocr_texts"]])

    descs = _load(config.SCENE_DESC_JSON) or []
    vlm_by_scene = {int(d["scene_index"]): d["description"] for d in descs}
    vlm_ref = " ".join(vlm_by_scene.values())

    metrics = {"n_scenes": len(scenes), "n_chapters": len(labels),
               "model": config.NARRATE_MODEL, "strategies": {}}

    # 1-4: per prompt strategy, on the caption-sourced stories (the assignment path)
    for strategy in config.STORY_STRATEGIES:
        text = _story_text(strategy)
        if text:
            metrics["strategies"][strategy] = score_story(
                text, caption_ref, vlm_ref, labels)
    if not metrics["strategies"]:
        metrics["strategies"] = {"status": "no stories (run narrate.py --strategy all)"}

    # Ablation: the promoted strategy, narrated from VLM descriptions instead.
    vlm_story = _story_text(config.STORY_STRATEGY, source="vlm")
    metrics["ablation"] = (
        score_story(vlm_story, caption_ref, vlm_ref, labels) if vlm_story
        else {"status": "no VLM story (run narrate.py --source vlm)"})

    # Caption adequacy: BLIP vs the vision model, on the same keyframes.
    if vlm_by_scene:
        scores = [caption_adequacy(s["representative_caption"],
                                   vlm_by_scene[s["scene_index"]])
                  for s in scenes if s["scene_index"] in vlm_by_scene]
        metrics["caption_adequacy"] = {
            "n_scenes": len(scores),
            "mean": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "share_below_0.25": round(
                sum(s < 0.25 for s in scores) / len(scores), 3) if scores else 0.0,
        }
    else:
        metrics["caption_adequacy"] = {
            "status": "no scene_descriptions.json (run describe_scenes.py)"}

    # Timeline validity.
    tl = _load(config.TIMELINE_JSON)
    if tl:
        events = tl.get("events", [])
        seen = {e["timestamp"] for e in events}
        metrics["timeline"] = {
            "n_events": len(events),
            "in_scene_bounds": round(timeline_in_scene_bounds(events, scenes), 3),
            "chronological": chronology_score(
                [e["timestamp_sec"] for e in events]) == 1.0,
            "title_cards_covered": round(
                sum(any(abs(e["timestamp_sec"] - s["start_sec"]) <= 2
                        for e in events)
                    for s in scenes if s["is_title_card"]) / max(len(labels), 1), 3),
            "repair_problems": tl.get("problems", []),
            "duplicate_timestamps": len(events) - len(seen),
        }
    else:
        metrics["timeline"] = {"status": "no timeline.json (run narrate.py)"}

    _write(metrics)
    return metrics


# ---- reporting -------------------------------------------------------------

def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _write(metrics: dict) -> None:
    with open(config.REPORT_DIR / "eval_story.json", "w") as f:
        json.dump(metrics, f, indent=2)

    L = ["# Milestone 3 evaluation — story, summary, timeline", ""]
    if "status" in metrics:
        L += [f"*{metrics['status']}*", ""]
        (config.REPORT_DIR / "eval_story.md").write_text("\n".join(L) + "\n")
        return

    L += [f"Model: `{metrics['model']}` · {metrics['n_scenes']} scenes · "
          f"{metrics['n_chapters']} chapters", ""]

    strat = metrics.get("strategies", {})
    L += ["## Prompt-engineering comparison (Task 3)", ""]
    if "status" in strat:
        L += [f"*{strat['status']}*", ""]
    else:
        cols = ["words", "timestamps_cited", "chronology", "coverage",
                "redundancy_distinct_ngram", "grounding_vs_captions"]
        if any("grounding_vs_vlm" in v for v in strat.values()):
            cols.append("grounding_vs_vlm")
        L += ["| strategy | " + " | ".join(cols) + " |",
              "|" + "---|" * (len(cols) + 1)]
        for name, m in strat.items():
            L.append(f"| `{name}` | "
                     + " | ".join(_fmt(m.get(c, "—")) for c in cols) + " |")
        L += ["",
              "*chronology* = share of adjacent cited timestamps that don't go "
              "backwards (1.0 = never time-travels). *coverage* = share of the 12 "
              "chapters mentioned. *redundancy* = distinct-3-gram ratio (higher = "
              "less repetitive). *grounding* = share of content words attested by "
              "the source; the `_vlm` column uses the vision model's independent "
              "keyframe descriptions, so it is **not** circular with the captions "
              "the story was written from.", ""]

    ab = metrics.get("ablation", {})
    L += ["## Ablation — captions vs. vision (does the narrator or the caption "
          "limit us?)", ""]
    if "status" in ab:
        L += [f"*{ab['status']}*", ""]
    else:
        base = strat.get(config.STORY_STRATEGY, {})
        L += [f"Both rows use the `{config.STORY_STRATEGY}` prompt. Only the "
              "*input* differs.", "",
              "| source | words | chronology | coverage | grounding_vs_vlm |",
              "|---|---|---|---|---|",
              f"| BLIP captions (assignment) | {base.get('words','—')} | "
              f"{_fmt(base.get('chronology','—'))} | {_fmt(base.get('coverage','—'))} | "
              f"{_fmt(base.get('grounding_vs_vlm','—'))} |",
              f"| VLM keyframe descriptions | {ab.get('words','—')} | "
              f"{_fmt(ab.get('chronology','—'))} | {_fmt(ab.get('coverage','—'))} | "
              f"{_fmt(ab.get('grounding_vs_vlm','—'))} |", ""]

    ca = metrics.get("caption_adequacy", {})
    L += ["## Caption adequacy — the ceiling BLIP imposes", ""]
    if "status" in ca:
        L += [f"*{ca['status']}*", ""]
    else:
        L += [f"Over {ca['n_scenes']} scene keyframes, BLIP's caption captures "
              f"**{ca['mean']:.1%}** of the content words the vision model sees in "
              f"the same image. **{ca['share_below_0.25']:.1%}** of scenes fall "
              "below 25%.", "",
              "Any narrator reading only the captions is bounded by this number: "
              "it cannot describe what its input never mentioned.", ""]

    tl = metrics.get("timeline", {})
    L += ["## Event timeline validity", ""]
    if "status" in tl:
        L += [f"*{tl['status']}*", ""]
    else:
        L += [f"- **{tl['n_events']}** events",
              f"- **{tl['in_scene_bounds']:.1%}** fall inside a real scene span",
              f"- chronological: **{tl['chronological']}**",
              f"- title cards covered: **{tl['title_cards_covered']:.1%}**",
              f"- duplicate timestamps: **{tl['duplicate_timestamps']}**"]
        if tl["repair_problems"]:
            L += ["", "Repairs applied to the model's raw JSON:"]
            L += [f"  - {p}" for p in tl["repair_problems"]]
        L += [""]

    (config.REPORT_DIR / "eval_story.md").write_text("\n".join(L) + "\n")
    log.info("story eval -> reports/eval_story.json")

    _write_comparison(metrics)


def _write_comparison(metrics: dict) -> None:
    """Side-by-side prose, so a reader can see what the metrics are measuring."""
    L = ["# Story comparison — prompt strategies and caption sources", "",
         f"Generated by `{metrics['model']}` from {metrics['n_scenes']} scene "
         f"digests. Metrics in `reports/eval_story.md`.", ""]
    for strategy in config.STORY_STRATEGIES:
        text = _story_text(strategy)
        if not text:
            continue
        m = metrics.get("strategies", {}).get(strategy, {})
        L += [f"## `{strategy}`", "",
              f"> chronology {_fmt(m.get('chronology','—'))} · "
              f"coverage {_fmt(m.get('coverage','—'))} · "
              f"grounding(vlm) {_fmt(m.get('grounding_vs_vlm','—'))} · "
              f"{m.get('words','—')} words", "",
              text.strip(), ""]
    vlm = _story_text(config.STORY_STRATEGY, source="vlm")
    if vlm:
        m = metrics.get("ablation", {})
        L += [f"## `{config.STORY_STRATEGY}` — narrated from VLM descriptions "
              "(ablation)", "",
              f"> chronology {_fmt(m.get('chronology','—'))} · "
              f"coverage {_fmt(m.get('coverage','—'))} · "
              f"grounding(vlm) {_fmt(m.get('grounding_vs_vlm','—'))} · "
              f"{m.get('words','—')} words", "", vlm.strip(), ""]
    (config.REPORT_DIR / "STORY_COMPARISON.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate Milestone 3 narration")
    ap.parse_args()
    m = run()
    print(json.dumps(m, indent=2)[:1200])
