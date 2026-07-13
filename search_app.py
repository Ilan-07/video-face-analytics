"""Milestone 2 Task 2 + Milestone 3: Streamlit UI over the frame metadata repository.

Run:  .venv/bin/python -m streamlit run search_app.py

Search tab -- type a word or phrase; matching frames are shown inline with their
timestamp, the face IDs present, and the matched text snippet. Consecutive frames
with the same text are summarised as time ranges.

Story & Timeline tab (Milestone 3) -- the generated summary and story, the four
prompt strategies side by side, and the event timeline rendered against each
scene's keyframe. Degrades to an instruction when narrate.py has not been run.
"""
import json

import streamlit as st

import config
import search as search_core

st.set_page_config(page_title="Video Frame Search", layout="wide")
st.title("🔎 Video Frame Search")
st.caption("Search OCR text (and optionally captions) across all extracted "
           "frames. Results show the timestamp and the matching frame.")


@st.cache_data
def _load():
    return search_core.load_metadata()


@st.cache_data
def _load_json(path_str: str):
    with open(path_str) as f:
        return json.load(f)


if not config.METADATA_JSON.exists():
    st.error(f"Metadata repository not found at {config.METADATA_JSON}. "
             "Run the pipeline first: `python run_pipeline.py`.")
    st.stop()

df = _load()
scenes = (_load_json(str(config.SCENES_JSON))
          if config.SCENES_JSON.exists() else [])
tab_search, tab_story = st.tabs(["🔎 Search", "📖 Story & Timeline"])


# --------------------------------------------------------------- Search (M2)
with tab_search:
    # Deep-linkable searches: ?q=...&mode=Semantic&captions=1&fuzzy=1&regex=1 seeds
    # the controls, so a result view can be shared or bookmarked.
    qp = st.query_params
    _modes = ["Lexical", "Semantic", "Visual"]
    _mode_default = qp.get("mode", "Lexical")
    if _mode_default not in _modes:
        _mode_default = "Lexical"

    col_q, col_opt = st.columns([3, 1])
    with col_q:
        query = st.text_input("Search", value=qp.get("q", ""),
                              placeholder='e.g. "platform"')
    with col_opt:
        mode = st.radio("Mode", _modes, index=_modes.index(_mode_default),
                        horizontal=True,
                        help="Lexical = substring/regex/fuzzy; Semantic = by "
                             "caption+OCR meaning; Visual = by image content "
                             "(CLIP, caption-free)")
        also_caps = st.checkbox("Search captions too",
                                value=qp.get("captions") == "1")
        fuzzy = st.checkbox("Fuzzy (tolerate OCR typos)",
                            value=qp.get("fuzzy") == "1")
        use_regex = st.checkbox("Regex", value=qp.get("regex") == "1")
        cols_per_row = st.slider("Columns", 1, 5, 3)

    embedding_mode = mode in ("Semantic", "Visual")
    fields = ("ocr_text", "caption") if also_caps else ("ocr_text",)

    if query and embedding_mode:
        visual = mode == "Visual"
        index_file = config.IMAGE_EMB_FILE if visual else config.TEXT_EMB_FILE
        builder = "embed_image.py" if visual else "embed_text.py"
        if not index_file.exists():
            st.error(f"{mode} index not found. Run `python run_pipeline.py` (or "
                     f"`python {builder}`) to build it.")
            st.stop()
        finder = search_core.visual_search if visual else search_core.semantic_search
        res = finder(query, df=df)
        if res.empty:
            st.warning(f'No frames {mode.lower()}-matched "{query}".')
        else:
            st.success(f'Top {len(res)} {mode.lower()} match(es) for "{query}".')
            if visual:
                st.caption("Ranked by image content (CLIP) — independent of caption "
                           "quality. The text shown is the frame's caption/OCR.")
            rows = res.to_dict("records")
            for i in range(0, len(rows), cols_per_row):
                cols = st.columns(cols_per_row)
                for col, r in zip(cols, rows[i:i + cols_per_row]):
                    with col:
                        st.image(r["frame_path"], use_container_width=True)
                        faces = ", ".join(r["face_ids"]) if r["face_ids"] else "—"
                        st.markdown(
                            f"**⏱ {r['mmss']}** · sim {r['score']:.3f}  \n"
                            f"**Faces:** {faces}  \n{r['snippet']}")
    elif query:
        res = search_core.search(query, df=df, fields=fields,
                                 regex=use_regex, fuzzy=fuzzy)
        if res.empty:
            st.warning(f'No frames matched "{query}".')
        else:
            groups = search_core.group_consecutive(res)
            st.success(f'{len(res)} frame(s) in {len(groups)} time range(s) '
                       f'matched "{query}".')
            ranges = ", ".join(
                g["start"] if g["start"] == g["end"] else f'{g["start"]}–{g["end"]}'
                for g in groups)
            st.markdown(f"**Time ranges:** {ranges}")

            rows = res.to_dict("records")
            for i in range(0, len(rows), cols_per_row):
                cols = st.columns(cols_per_row)
                for col, r in zip(cols, rows[i:i + cols_per_row]):
                    with col:
                        st.image(r["frame_path"], use_container_width=True)
                        faces = ", ".join(r["face_ids"]) if r["face_ids"] else "—"
                        st.markdown(
                            f"**⏱ {r['mmss']}** · frame {r['frame_id']}  \n"
                            f"**Faces:** {faces}  \n"
                            f"**{r['field']}:** {r['snippet']}")
    else:
        st.info("Enter a search term above to find matching frames.")


# ----------------------------------------------- Story & Timeline (M3)
# No st.stop() in this tab: it would halt the whole script and blank the Search
# tab for anyone who has only run Milestones 1-2.
with tab_story:
    if not scenes:
        st.error("No scenes yet. Run `python scenes.py` (or the full pipeline).")
    else:
        st.markdown(
            f"**{len(scenes)} scenes** across "
            f"**{len({s['chapter_index'] for s in scenes})} chapters**, cut on "
            "adjacent-frame CLIP similarity and the on-screen title cards.")

        if not config.STORY_JSON.exists():
            st.warning("No story generated yet. Set `OPENROUTER_API_KEY` and run "
                       "`python narrate.py --strategy all`.")
        else:
            story = _load_json(str(config.STORY_JSON))
            st.caption(f"Generated by `{story['model']}` · prompt strategy "
                       f"`{story['strategy']}` · source `{story['source']}`")

            st.subheader("Summary")
            st.write(story["summary"])

            st.subheader("Story")
            st.markdown(story["story"])

            others = [s for s in story.get("strategies_generated", [])
                      if s != story["strategy"]]
            if others:
                st.subheader("Other prompt strategies")
                st.caption("Scored side by side in `reports/eval_story.md`.")
                for strategy in others:
                    path = config.REPORT_DIR / f"story_{strategy}.md"
                    if path.exists():
                        with st.expander(f"`{strategy}`"):
                            st.markdown(path.read_text())

        st.subheader("Event timeline")
        if not config.TIMELINE_JSON.exists():
            st.warning("No timeline yet. Run `python narrate.py`.")
        else:
            timeline = _load_json(str(config.TIMELINE_JSON))
            events = timeline["events"]
            st.caption(f"{len(events)} events. Each thumbnail is the keyframe of "
                       "the scene the event falls in.")
            # Map each event to its scene to show a keyframe. An m:ss event
            # resolves only to the second while scene bounds carry the source
            # fps's sub-second offset (event 0:09 = 9.0s vs a scene starting at
            # 9.009s), so "contained in [start,end]" misses; take the last scene
            # that starts at or before the event instead.
            scenes_by_start = sorted(scenes, key=lambda s: s["start_sec"])
            for e in events:
                scene = None
                for s in scenes_by_start:
                    if s["start_sec"] <= e["timestamp_sec"] + 1.05:
                        scene = s
                    else:
                        break
                col_img, col_txt = st.columns([1, 4])
                with col_img:
                    if scene:
                        st.image(str(config.FRAME_DIR / scene["keyframe_file"]),
                                 use_container_width=True)
                with col_txt:
                    label = f' · {scene["chapter_label"]}' if scene else ""
                    st.markdown(
                        f"**⏱ {e['timestamp']}**{label}  \n{e['description']}")
