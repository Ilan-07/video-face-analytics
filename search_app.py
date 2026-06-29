"""Milestone 2 Task 2: Streamlit UI for searching the frame metadata repository.

Run:  .venv/bin/python -m streamlit run search_app.py

Type a word or phrase; matching frames are shown inline with their timestamp,
the face IDs present, and the matched text snippet.
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

col_q, col_opt = st.columns([3, 1])
with col_q:
    query = st.text_input("Search", placeholder='e.g. "Welcome"')
with col_opt:
    also_caps = st.checkbox("Search captions too", value=False)
    cols_per_row = st.slider("Columns", 1, 5, 3)

fields = ("ocr_text", "caption") if also_caps else ("ocr_text",)

if query:
    res = search_core.search(query, df=df, fields=fields)
    if res.empty:
        st.warning(f'No frames matched "{query}".')
    else:
        st.success(f'{len(res)} frame(s) matched "{query}".')
        ts = ", ".join(res["mmss"].tolist())
        st.markdown(f"**Timestamps:** {ts}")

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
