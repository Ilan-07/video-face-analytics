"""Milestone 2 Task 2: Streamlit UI for searching the frame metadata repository.

Run:  .venv/bin/python -m streamlit run search_app.py

Type a word or phrase; matching frames are shown inline with their timestamp,
the face IDs present, and the matched text snippet. Consecutive frames with the
same text are summarised as time ranges.
"""
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


if not config.METADATA_JSON.exists():
    st.error(f"Metadata repository not found at {config.METADATA_JSON}. "
             "Run the pipeline first: `python run_pipeline.py`.")
    st.stop()

df = _load()

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
                    help="Lexical = substring/regex/fuzzy; Semantic = by caption+OCR "
                         "meaning; Visual = by image content (CLIP, caption-free)")
    also_caps = st.checkbox("Search captions too", value=qp.get("captions") == "1")
    fuzzy = st.checkbox("Fuzzy (tolerate OCR typos)", value=qp.get("fuzzy") == "1")
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
