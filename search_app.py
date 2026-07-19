"""Milestone 4: the integrated video analysis application.

Run:  .venv/bin/python -m streamlit run search_app.py

One UI over every artifact the pipeline produces. Five tabs:

  Dashboard      -- headline counts, face occurrence stats, and the measured
                    quality/timing numbers from eval_system.py (Task 3 + 4).
  Search         -- lexical / semantic / visual retrieval over the frame
                    metadata repository, with timestamps and frames (Task 2).
  Faces          -- per-identity occurrence statistics, montages, and the
                    frames an identity appears in (Task 2 + 3).
  Captions & OCR -- browse every frame's extracted text and caption (Task 2).
  Story          -- the generated summary, story, prompt-strategy comparison,
                    and event timeline (Task 2 + 3).

Every tab degrades to an instruction rather than an exception when its artifact
is missing: Milestone 1 users have no captions, and users without an
OPENROUTER_API_KEY have no story, but the app must still run for them. This is
also why no tab calls st.stop() -- in Streamlit that halts the whole script and
would blank the other four tabs.
"""
import json

import pandas as pd
import streamlit as st

import config
import search as search_core

st.set_page_config(page_title="Video Analysis System", layout="wide")
st.title("🎬 Video Analysis System")
st.caption("Faces, on-screen text, captions, search, and generated narration "
           "over one video — every number below is produced by the pipeline.")


@st.cache_data
def _load():
    return search_core.load_metadata()


@st.cache_data
def _load_json(path_str: str):
    with open(path_str) as f:
        return json.load(f)


def _maybe_json(path):
    """Load an optional artifact; None when the stage has not been run."""
    return _load_json(str(path)) if path.exists() else None


if not config.METADATA_JSON.exists():
    st.error(f"Metadata repository not found at {config.METADATA_JSON}. "
             "Run the pipeline first: `python run_pipeline.py`.")
    st.stop()

df = _load()
scenes = _maybe_json(config.SCENES_JSON) or []
summary = _maybe_json(config.REPORT_DIR / "summary.json") or {}
system_eval = _maybe_json(config.SYSTEM_EVAL_JSON) or {}

tab_dash, tab_search, tab_faces, tab_text, tab_story = st.tabs(
    ["📊 Dashboard", "🔎 Search", "👤 Faces", "📝 Captions & OCR",
     "📖 Story & Timeline"])


# ------------------------------------------------------------- Dashboard (M4)
with tab_dash:
    if not summary:
        st.error("No analytics summary yet. Run `python run_pipeline.py`.")
    else:
        st.subheader("Pipeline output")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Frames processed", f"{summary.get('total_frames', 0):,}",
                  help=f"Sampled at {summary.get('fps', config.FPS)} FPS")
        c2.metric("Faces detected", f"{summary.get('total_faces_detected', 0):,}")
        c3.metric("Tracks", f"{summary.get('total_tracks', 0):,}",
                  help="Temporal groupings of detections (ByteTrack)")
        # The two-tier count is the honest one: total groups includes every
        # one-off background face, so the featured cast is the headline.
        c4.metric("Unique faces", f"{summary.get('total_unique_faces', 0):,}",
                  help="All identity groups, including one-off background faces")
        c5.metric("Featured cast", f"{summary.get('featured_cast', 0):,}",
                  help=f"Identities present in >= {config.RECURRING_MIN_FRAMES} "
                       "frames — the meaningful cast size")

        mf = summary.get("most_frequent_face")
        if mf:
            st.subheader("Most frequent face")
            col_img, col_txt = st.columns([1, 5])
            rep = config.REPORT_DIR / mf["representative"]
            with col_img:
                if rep.exists():
                    st.image(str(rep), width=120)
            with col_txt:
                st.markdown(
                    f"### {mf['face_id']}\n"
                    f"**{mf['screen_time_sec']}s** of screen time across "
                    f"**{mf['appearances']}** appearances "
                    f"({mf['frame_count']} frames) · first seen "
                    f"{search_core.fmt_ts(mf['first_seen_sec'])}, last seen "
                    f"{search_core.fmt_ts(mf['last_seen_sec'])} · estimated "
                    f"{mf['gender']}, ~{mf['age']}y")

        # -- Face occurrence statistics
        idents = summary.get("identities", [])
        if idents:
            st.subheader("Face occurrence statistics")
            featured = [i for i in idents
                        if i["frame_count"] >= config.RECURRING_MIN_FRAMES]
            st.caption(f"Screen time for the {len(featured)} featured "
                       f"identities (of {len(idents)} total groups). The long "
                       "tail of one-off background faces is omitted here — see "
                       "the Faces tab for all of them.")
            chart = pd.DataFrame(
                {"screen_time_sec": [i["screen_time_sec"] for i in featured]},
                index=[i["face_id"] for i in featured])
            st.bar_chart(chart, height=280)

        # -- Appearance timeline
        tl = summary.get("timeline")
        if tl and (config.REPORT_DIR / tl).exists():
            st.subheader("Appearance timeline")
            st.caption("When each featured identity is on screen across the "
                       "video.")
            st.image(str(config.REPORT_DIR / tl), use_container_width=True)

        # -- Measured system quality (Task 4)
        st.divider()
        st.subheader("Measured system performance")
        if not system_eval:
            st.info("No system evaluation yet. Run `python eval_system.py` "
                    "(or the full pipeline) to measure timing and quality.")
        else:
            t = system_eval.get("timing", {})
            q = system_eval.get("quality", {})
            if t.get("status") == "ok":
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("End-to-end", f"{t['total_min']} min")
                d2.metric("Local compute", f"{t['local_compute_sec']}s",
                          help="Excludes time spent waiting on the LLM endpoint")
                d3.metric("Per frame", f"{t.get('sec_per_frame_local')}s",
                          help="Local compute divided by frames processed")
                d4.metric("Faster than real time",
                          f"{t.get('realtime_factor_local')}×",
                          help="Video duration / local compute time")
                with st.expander("Per-stage timing breakdown"):
                    st.dataframe(pd.DataFrame(t["stages"]),
                                 use_container_width=True, hide_index=True)
            else:
                st.info("No stage timings recorded yet — run the pipeline to "
                        "measure them.")

            faces_q = q.get("faces", {})
            text_q = q.get("text_and_captions", {})
            search_q = q.get("search", {})
            narr_q = q.get("narration", {})

            def _pct(x):
                return "n/a" if x is None else f"{x:.1%}"

            st.markdown("**Output quality** — each figure comes from that "
                        "milestone's evaluation harness.")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Cannot-link precision",
                      _pct(faces_q.get("cannot_link_precision")),
                      help="Label-free: two faces in one frame can never be the "
                           "same person. 100% = no provable merge errors.")
            e2.metric("OCR F1", _pct(text_q.get("ocr_detect_f1")),
                      help=f"Precision {_pct(text_q.get('ocr_detect_precision'))} "
                           f"/ recall {_pct(text_q.get('ocr_detect_recall'))}")
            e3.metric(f"Search precision@{search_q.get('k', 5)}",
                      _pct(search_q.get("mean_precision_at_k")),
                      help=f"Over {search_q.get('queries')} curated queries")
            e4.metric("Story chronology", _pct(narr_q.get("chronology")),
                      help="Are the story's cited timestamps monotonic?")

            lims = system_eval.get("limitations", [])
            if lims:
                st.markdown("**Limitations** — derived from the metrics above, "
                            "not written beside them.")
                sev_icon = {"high": "🔴", "medium": "🟠", "info": "🔵"}
                for c in lims:
                    with st.expander(
                            f"{sev_icon.get(c['severity'], '•')} {c['area']}"):
                        st.markdown(c["finding"])
                        st.caption(f"Evidence: {c['evidence']}")
                        st.markdown(f"**Enhancement:** {c['enhancement']}")

            if config.SYSTEM_EVAL_MD.exists():
                with st.expander("Full system evaluation report"):
                    st.markdown(config.SYSTEM_EVAL_MD.read_text())


# --------------------------------------------------------------- Search (M2)
with tab_search:
    # Deep-linkable searches: ?q=...&mode=Semantic&captions=1&fuzzy=1&regex=1 seeds
    # the controls, so a result view can be shared or bookmarked.
    qp = st.query_params
    _modes = ["Lexical", "Semantic", "Visual", "Fused"]
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
                             "(CLIP, caption-free); Fused = reciprocal-rank "
                             "fusion of Semantic + Visual")
        also_caps = st.checkbox("Search captions too",
                                value=qp.get("captions") == "1")
        fuzzy = st.checkbox("Fuzzy (tolerate OCR typos)",
                            value=qp.get("fuzzy") == "1")
        use_regex = st.checkbox("Regex", value=qp.get("regex") == "1")
        cols_per_row = st.slider("Columns", 1, 5, 3)

    embedding_mode = mode in ("Semantic", "Visual", "Fused")
    fields = ("ocr_text", "caption") if also_caps else ("ocr_text",)

    if query and embedding_mode:
        visual = mode == "Visual"
        fused = mode == "Fused"
        # Fused needs both indexes; the others need one.
        needed = ([config.TEXT_EMB_FILE, config.IMAGE_EMB_FILE] if fused
                  else [config.IMAGE_EMB_FILE] if visual
                  else [config.TEXT_EMB_FILE])
        missing = [p for p in needed if not p.exists()]
        if missing:
            st.error(f"{mode} needs {', '.join(p.name for p in missing)}. Run "
                     "`python run_pipeline.py` to build the search indexes.")
        else:
            finder = (search_core.fused_search if fused
                      else search_core.visual_search if visual
                      else search_core.semantic_search)
            res = finder(query, df=df)
            if res.empty:
                st.warning(f'No frames {mode.lower()}-matched "{query}".')
            else:
                st.success(f'Top {len(res)} {mode.lower()} match(es) for '
                           f'"{query}".')
                if visual:
                    st.caption("Ranked by image content (CLIP) — independent of "
                               "caption quality. The text shown is the frame's "
                               "caption/OCR.")
                elif fused:
                    st.caption("Reciprocal-rank fusion of Semantic + Visual — "
                               "either index can surface a frame the other misses. "
                               "Score is the RRF weight, not a cosine similarity.")
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


# ---------------------------------------------------------------- Faces (M4)
with tab_faces:
    idents = summary.get("identities", [])
    if not idents:
        st.error("No identities yet. Run `python run_pipeline.py`.")
    else:
        featured_only = st.checkbox(
            f"Featured cast only (present in >= {config.RECURRING_MIN_FRAMES} "
            "frames)", value=True,
            help="Unticking shows every group, including one-off background "
                 "faces that appear in a single frame.")
        shown = [i for i in idents
                 if not featured_only
                 or i["frame_count"] >= config.RECURRING_MIN_FRAMES]

        st.caption(f"{len(shown)} identit{'y' if len(shown) == 1 else 'ies'}, "
                   "ranked by screen time.")
        st.warning(
            "**Gender and age are unvalidated estimates** from the face model, "
            "shown per-crop and reduced to a majority vote / median. They carry "
            "no accuracy measurement in this project and are known to be biased "
            "and unreliable on small, low-resolution, or non-frontal faces — most "
            "of this footage. Treat them as a rough hint, not a fact.", icon="⚠️")
        table = pd.DataFrame([{
            "face_id": i["face_id"],
            "screen_time_sec": i["screen_time_sec"],
            "appearances": i["appearances"],
            "frames": i["frame_count"],
            "first_seen": search_core.fmt_ts(i["first_seen_sec"]),
            "last_seen": search_core.fmt_ts(i["last_seen_sec"]),
            "gender": i["gender"],
            "age": i["age"],
        } for i in shown])
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Identity detail")
        pick = st.selectbox("Identity", [i["face_id"] for i in shown])
        ident = next(i for i in idents if i["face_id"] == pick)

        col_a, col_b = st.columns([1, 4])
        with col_a:
            rep = config.REPORT_DIR / ident["representative"]
            if rep.exists():
                st.image(str(rep), width=140)
        with col_b:
            st.markdown(
                f"**{ident['screen_time_sec']}s** screen time · "
                f"**{ident['appearances']}** appearances · "
                f"**{ident['frame_count']}** frames  \n"
                f"On screen {search_core.fmt_ts(ident['first_seen_sec'])} – "
                f"{search_core.fmt_ts(ident['last_seen_sec'])}  \n"
                f"Estimated {ident['gender']}, ~{ident['age']}y")
            st.caption("`appearances` counts distinct tracks (separate times "
                       "this person appears); `frames` counts sampled frames "
                       "they are present in.")

        montage = config.REPORT_DIR / ident["montage"]
        if montage.exists():
            st.markdown("**Every crop grouped into this identity** — "
                        "consolidated across scenes, lighting and pose.")
            st.image(str(montage), use_container_width=True)

        # The frames this identity appears in, straight from the repository.
        # face_ids is a list per frame, so match by membership rather than eq.
        hits = df[df["face_ids"].apply(lambda ids: pick in (ids or []))]
        if len(hits):
            st.markdown(f"**Frames containing {pick}** ({len(hits)})")
            per_row = 5
            rows = hits.to_dict("records")[:20]
            for i in range(0, len(rows), per_row):
                cols = st.columns(per_row)
                for col, r in zip(cols, rows[i:i + per_row]):
                    with col:
                        st.image(str(config.FRAME_DIR / r["filename"]),
                                 use_container_width=True)
                        st.caption(f"⏱ {search_core.fmt_ts(r['timestamp_sec'])}")
            if len(hits) > 20:
                st.caption(f"Showing the first 20 of {len(hits)} frames.")


# ------------------------------------------------------- Captions & OCR (M2)
with tab_text:
    st.subheader("Frame metadata repository")
    st.caption("Every sampled frame with its extracted on-screen text, "
               "generated caption, and the identities present — the single "
               "joined table that powers search.")

    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        only_text = st.checkbox("Only frames with OCR text", value=False)
    with f2:
        only_faces = st.checkbox("Only frames with faces", value=False)
    with f3:
        contains = st.text_input("Filter (substring in caption or OCR)",
                                 placeholder="e.g. platform")

    view = df.copy()
    if only_text:
        view = view[view["ocr_text"].fillna("").str.strip() != ""]
    if only_faces:
        view = view[view["face_ids"].apply(lambda ids: bool(ids))]
    if contains:
        c = contains.lower()
        view = view[
            view["ocr_text"].fillna("").str.lower().str.contains(c, regex=False)
            | view["caption"].fillna("").str.lower().str.contains(c, regex=False)]

    st.markdown(f"**{len(view):,}** of **{len(df):,}** frames match.")
    show = pd.DataFrame({
        "time": view["timestamp_sec"].map(search_core.fmt_ts),
        "frame_id": view["frame_id"],
        "faces": view["face_ids"].apply(lambda ids: ", ".join(ids) if ids else "—"),
        "ocr_text": view["ocr_text"].fillna(""),
        "caption": view["caption"].fillna(""),
    })
    st.dataframe(show, use_container_width=True, hide_index=True, height=380)

    if len(view):
        st.divider()
        st.subheader("Frame viewer")
        # Index by position, not frame_id: filtering leaves gaps in frame_id and
        # a slider over a sparse id range would land on filtered-out frames.
        pos = st.slider("Frame", 0, max(len(view) - 1, 0), 0,
                        help="Steps through the filtered frames above.")
        r = view.iloc[pos]
        col_img, col_meta = st.columns([2, 3])
        with col_img:
            st.image(str(config.FRAME_DIR / r["filename"]),
                     use_container_width=True)
        with col_meta:
            faces = ", ".join(r["face_ids"]) if r["face_ids"] else "—"
            st.markdown(
                f"**⏱ {search_core.fmt_ts(r['timestamp_sec'])}** · frame "
                f"{r['frame_id']}  \n**Faces:** {faces}")
            st.markdown(f"**OCR text:** {r['ocr_text'] or '—'}")
            st.markdown(f"**Caption:** {r['caption'] or '—'}")
            if pd.notna(r.get("scene_index")):
                st.caption(f"Scene {r.get('scene_index')}")
            if isinstance(r.get("event_description"), str) and r["event_description"]:
                st.info(f"Event: {r['event_description']}")


# ----------------------------------------------- Story & Timeline (M3)
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
