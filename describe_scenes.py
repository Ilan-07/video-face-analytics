"""Milestone 3b: re-describe each scene's keyframe with a vision model.

BLIP is the assignment's specified caption source, and it stays the input to the
canonical story. But BLIP is demonstrably the weak link: across 1415 frames it
emits only 342 distinct captions, hallucinates trains onto empty platforms, and
compresses every shot into one of a handful of templates ("a train is pulling
passengers ..."). Anything the narrator gets wrong may simply be BLIP's error,
faithfully transcribed.

So we take the 24 non-card scene keyframes chosen by scenes.py and ask Gemma 4 --
which accepts image input -- what is actually in them. The result is used two ways:

  1. as an INDEPENDENT REFERENCE for eval_story.py, which scores how much of the
     story's content is attested by something other than the captions it was
     generated from (a hallucination measure that is not circular); and
  2. as the input to an ABLATION story (`narrate.py --source vlm`), which shows
     whether the narrative bottleneck is the narrator or the captions.

Request economy
---------------
Title-card scenes are skipped -- we already know they are a black screen naming a
line. The rest are sent NARRATE_VLM_BATCH keyframes at a time in a single
multi-image message, because OpenRouter's free tier allows 50 requests/day: 24
keyframes cost 4 requests rather than 24.

Batching means the model could mis-align descriptions to images, so the reply must
name each scene_index explicitly and we verify the set that came back. Any scene
the batch missed is retried on its own, one image, unambiguous.

Output: data/scene_descriptions.json. Resumable; every response cached by llm.py.
"""
import argparse
import json

import config
import llm
import util

log = util.get_logger()

_BATCH_PROMPT = """You are shown {n} frames from a London Underground tour video. \
They are in chronological order and correspond, in order, to these scenes:

{manifest}

For EACH image, describe what is actually visible in 2-3 sentences. Be concrete. \
State the setting (platform, tunnel, train interior, escalator, concourse), \
whether a train is present and whether its doors are open, how many people are \
visible and what they are doing, and any station name or signage you can read.

Describe only what you can see. If there is no train in a frame, say so \
explicitly. Do not speculate about what happens before or after.

Return ONLY a JSON array, no prose and no code fence. One object per image, in \
the same order, each with exactly two keys:
  "scene_index"  - the integer scene number from the list above
  "description"  - your 2-3 sentence description of that image"""


def _manifest(batch) -> str:
    return "\n".join(f'  image {i + 1}: scene_index {s["scene_index"]}, '
                     f'at {s["start_mmss"]} ({s["chapter_label"]})'
                     for i, s in enumerate(batch))


def _existing(restart: bool) -> dict:
    """scene_index -> description already computed (resume), unless --restart."""
    if restart or not config.SCENE_DESC_JSON.exists():
        return {}
    try:
        with open(config.SCENE_DESC_JSON) as f:
            return {int(r["scene_index"]): r["description"] for r in json.load(f)}
    except (json.JSONDecodeError, KeyError, OSError):
        return {}


def parse_batch(reply: str, expected: set) -> dict:
    """scene_index -> description, keeping only indices we actually asked for.

    The scene_index contract is what makes batching safe: a reply that silently
    shifts descriptions by one image fails this check instead of mislabelling 24
    scenes. Pure -- unit tested."""
    items = llm.extract_json_array(reply) or []
    out = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = item.get("scene_index")
        if raw is None:
            continue
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        desc = str(item.get("description", "")).strip()
        if idx in expected and desc:
            out[idx] = desc
    return out


def _describe_batch(batch) -> dict:
    prompt = _BATCH_PROMPT.format(n=len(batch), manifest=_manifest(batch))
    images = [config.FRAME_DIR / s["keyframe_file"] for s in batch]
    reply = llm.generate(prompt, images=images, max_tokens=300 * len(batch))
    return parse_batch(reply, {s["scene_index"] for s in batch})


def _describe_one(scene) -> str:
    prompt = _BATCH_PROMPT.format(n=1, manifest=_manifest([scene]))
    reply = llm.generate(prompt, images=[config.FRAME_DIR / scene["keyframe_file"]],
                         max_tokens=400)
    got = parse_batch(reply, {scene["scene_index"]})
    return got.get(scene["scene_index"], reply.strip())


def run(limit: int | None = None, restart: bool = False) -> int:
    config.ensure_dirs()
    if not config.SCENES_JSON.exists():
        raise RuntimeError(
            f"{config.SCENES_JSON.name} not found. Run `python scenes.py` first.")

    with open(config.SCENES_JSON) as f:
        scenes = json.load(f)

    targets = [s for s in scenes if not s["is_title_card"]]
    if limit:
        targets = targets[:limit]
    descriptions = _existing(restart)
    todo = [s for s in targets if s["scene_index"] not in descriptions]

    bs = max(1, config.NARRATE_VLM_BATCH)
    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
    log.info("describing %d scene keyframes (%d cached, %d to generate "
             "in %d request(s) of <=%d images)",
             len(targets), len(targets) - len(todo), len(todo), len(batches), bs)

    for n, batch in enumerate(batches, start=1):
        want = {s["scene_index"] for s in batch}
        try:
            got = _describe_batch(batch)
        except RuntimeError as e:
            # Partial progress is written below, so a rate-limit stop is resumable.
            log.error("batch %d/%d failed: %s", n, len(batches), e)
            break
        missing = want - set(got)
        if missing:
            log.warning("batch %d/%d: model omitted scenes %s -- retrying singly",
                        n, len(batches), sorted(missing))
            for scene in (s for s in batch if s["scene_index"] in missing):
                try:
                    got[scene["scene_index"]] = _describe_one(scene)
                except RuntimeError as e:
                    log.error("scene %d failed: %s", scene["scene_index"], e)
        descriptions.update(got)
        log.info("  [%d/%d] scenes %s", n, len(batches), sorted(got))

    records = [{"scene_index": s["scene_index"],
                "keyframe_frame_id": s["keyframe_frame_id"],
                "start_mmss": s["start_mmss"],
                "chapter_label": s["chapter_label"],
                "description": descriptions[s["scene_index"]]}
               for s in targets if s["scene_index"] in descriptions]

    with open(config.SCENE_DESC_JSON, "w") as f:
        json.dump(records, f, indent=2)
    log.info("scene descriptions: %d/%d -> %s",
             len(records), len(targets), config.SCENE_DESC_JSON.name)
    return len(records)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Describe scene keyframes with a VLM (Milestone 3)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only describe the first N scenes (smoke test)")
    ap.add_argument("--restart", action="store_true",
                    help="ignore cached descriptions and redo every scene")
    args = ap.parse_args()
    run(limit=args.limit, restart=args.restart)
