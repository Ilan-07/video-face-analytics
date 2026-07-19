"""End-to-end integration tests: drive the real stage entry points on a tiny
fixture and assert that each stage's output satisfies the next stage's input
schema.

The unit suite in test_pipeline.py checks pure logic; nothing there exercises
the *seams* between stages, which is exactly where a renamed column or a changed
join key breaks the pipeline silently (both report bugs fixed this session were
of that kind). These tests wire real stage functions together on a 4-frame
fixture, no models required, so a schema drift fails CI instead of production.

Model-loading stages (embed, caption, detect, whisper) are covered by the
`@pytest.mark.slow` smoke test at the bottom -- opt-in via `pytest --run-slow`.
"""
import json

import numpy as np
import pandas as pd
import pytest


def _seed_fixture(tmp_path, monkeypatch):
    """Write the minimal per-frame artifacts a real run produces, and point every
    config path at the tmp dir. Four frames: two share a person (t0/t1 -> Face_01),
    one has a different person (t2 -> Face_02), one has no face at all (frame 3).
    One detection is 'unknown' and must be dropped by the identity join."""
    import config
    data = tmp_path / "data"
    (data / "embeddings").mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    for attr, path in {
        "DATA": data, "FRAME_DIR": data / "frames",
        "FRAMES_CSV": data / "frames.csv", "FACES_CSV": data / "faces.csv",
        "IDENTITIES_CSV": data / "identities.csv", "OCR_CSV": data / "ocr.csv",
        "CAPTIONS_CSV": data / "captions.csv",
        "METADATA_CSV": data / "frame_metadata.csv",
        "METADATA_JSON": data / "frame_metadata.json",
        "TEXT_EMB_FILE": data / "embeddings" / "text_embeddings.npz",
        "IMAGE_EMB_FILE": data / "embeddings" / "image_embeddings.npz",
        "SCENES_JSON": data / "scenes.json", "TIMELINE_JSON": data / "timeline.json",
        "TRANSCRIPT_JSON": data / "transcript.json",
        "REPORT_DIR": tmp_path / "reports",
    }.items():
        monkeypatch.setattr(config, attr, path)

    pd.DataFrame({
        "frame_id": [0, 1, 2, 3],
        "filename": [f"frame_{i:06d}.jpg" for i in range(4)],
        "timestamp_sec": [0.0, 1.0, 2.0, 3.0],
    }).to_csv(config.FRAMES_CSV, index=False)

    # faces.csv: track t0,t1 -> P1 person; t2 -> P2; t3 -> an unknown track.
    pd.DataFrame({
        "crop_id": ["c0", "c1", "c2", "c3"],
        "frame_id": [0, 1, 2, 2],
        "timestamp_sec": [0.0, 1.0, 2.0, 2.0],
        "track_id": ["t0", "t1", "t2", "t3"],
        "x1": [0]*4, "y1": [0]*4, "x2": [10]*4, "y2": [10]*4,
        "det_score": [0.9]*4, "blur_var": [100.0]*4, "nose_offset": [0.0]*4,
        "gender": ["M"]*4, "age": [30]*4, "quality_ok": [True]*4,
        "crop_file": [f"c{i}.jpg" for i in range(4)],
    }).to_csv(config.FACES_CSV, index=False)

    pd.DataFrame({
        "track_id": ["t0", "t1", "t2", "t3"],
        "face_id": ["Face_01", "Face_01", "Face_02", "unknown"],
        "cluster_label": [0, 0, 1, -1], "n_faces": [1, 1, 1, 1],
        "first_frame": [0, 1, 2, 2], "last_frame": [0, 1, 2, 2],
        "first_sec": [0.0, 1.0, 2.0, 2.0], "last_sec": [0.0, 1.0, 2.0, 2.0],
        "duration_sec": [0.0]*4,
    }).to_csv(config.IDENTITIES_CSV, index=False)

    pd.DataFrame({
        "frame_id": [0, 1, 2, 3], "timestamp_sec": [0.0, 1.0, 2.0, 3.0],
        "face_ids": ["", "", "", ""],
        "text": ["VICTORIA LINE", "", "WAY OUT", ""],
        "n_tokens": [2, 0, 2, 0], "mean_conf": [90, 0, 88, 0],
    }).to_csv(config.OCR_CSV, index=False)

    pd.DataFrame({
        "frame_id": [0, 1, 2, 3], "timestamp_sec": [0.0, 1.0, 2.0, 3.0],
        "face_ids": ["", "", "", ""], "ocr_text": ["", "", "", ""],
        "caption": ["a subway platform", "a person waiting",
                    "a train arriving", "an empty tunnel"],
    }).to_csv(config.CAPTIONS_CSV, index=False)
    return config


# ------------------------------------------------ stage seam: faces -> metadata
def test_metadata_build_joins_faces_and_covers_every_frame(tmp_path, monkeypatch):
    import build_metadata
    config = _seed_fixture(tmp_path, monkeypatch)

    n = build_metadata.run()
    assert n == 4                                   # left join keeps every frame
    records = json.loads(config.METADATA_JSON.read_text())
    by_id = {r["frame_id"]: r for r in records}

    # The identity join must land the right Face IDs and drop the unknown track.
    assert by_id[0]["face_ids"] == ["Face_01"]
    assert by_id[2]["face_ids"] == ["Face_02"]      # t3=unknown dropped, only t2
    assert by_id[3]["face_ids"] == []               # no face, still present
    assert by_id[0]["ocr_text"] == "VICTORIA LINE"
    assert by_id[0]["caption"] == "a subway platform"


# ------------------------------------------ stage seam: metadata -> search load
def test_metadata_satisfies_search_load_schema(tmp_path, monkeypatch):
    import build_metadata
    import search
    _seed_fixture(tmp_path, monkeypatch)
    build_metadata.run()

    df = search.load_metadata()
    # These are the columns search.py and embed_text.py read downstream; a rename
    # in build_metadata would break retrieval, and this is the assertion that says so.
    for col in ("frame_id", "timestamp_sec", "filename", "face_ids",
                "ocr_text", "caption"):
        assert col in df.columns, f"metadata lost column {col!r} that search needs"
    assert len(df) == 4
    assert isinstance(df.iloc[0]["face_ids"], list)  # JSON keeps arrays, not "|"-joins


# ------------------------------ stage seam: metadata text -> embedded document
def test_frame_document_feeds_the_index_from_metadata_fields(tmp_path, monkeypatch):
    import build_metadata
    import embed_text
    import search
    _seed_fixture(tmp_path, monkeypatch)
    build_metadata.run()
    df = search.load_metadata()

    # frame_document is the exact string embed_text indexes; it must consume the
    # caption + ocr_text columns build_metadata emits, for the same frame.
    row = df[df["frame_id"] == 0].iloc[0]
    doc = embed_text.frame_document(row["caption"], row["ocr_text"])
    assert "subway platform" in doc and "VICTORIA" in doc.upper()


# ------------------------------- stage seam: transcript -> metadata speech column
def test_transcript_joins_into_metadata_speech(tmp_path, monkeypatch):
    import build_metadata
    config = _seed_fixture(tmp_path, monkeypatch)
    # A transcript that speaks over frames 0 (t=0) and 2 (t=2), silent elsewhere.
    config.TRANSCRIPT_JSON.write_text(json.dumps({"segments": [
        {"start": 0.0, "end": 1.0, "text": "mind the gap"},
        {"start": 2.0, "end": 3.0, "text": "this is Victoria"}]}))
    build_metadata.run()
    by_id = {r["frame_id"]: r for r in
             json.loads(config.METADATA_JSON.read_text())}
    assert by_id[0]["speech"] == "mind the gap"
    assert by_id[2]["speech"] == "this is Victoria"
    assert by_id[1]["speech"] == ""            # silence stays empty


# ---------------------------- stage seam: embeddings + metadata -> ranked rows
def test_ranking_chains_embeddings_to_metadata_rows(tmp_path, monkeypatch):
    import build_metadata
    import search
    config = _seed_fixture(tmp_path, monkeypatch)
    build_metadata.run()
    df = search.load_metadata()

    # Synthesize a text index over the same frame_ids (no model needed): the point
    # is that _rank_by_embedding can map index rows back to metadata rows and emit
    # the search-result schema the app and CLI consume.
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((4, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.savez(config.TEXT_EMB_FILE,
             frame_ids=df["frame_id"].to_numpy(np.int32), embeddings=emb)

    cols = ["frame_id", "timestamp_sec", "mmss", "filename", "frame_path",
            "face_ids", "field", "score", "snippet"]
    res = search._rank_by_embedding(config.TEXT_EMB_FILE, emb[0], df,
                                    top_k=4, min_score=-1.0, field="semantic",
                                    cols=cols)
    assert list(res.columns) == cols
    assert len(res) == 4
    assert res.iloc[0]["frame_id"] == 0            # query == row 0's own vector
    assert res.iloc[0]["score"] == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------- opt-in real-model smoke test
@pytest.mark.slow
def test_real_text_index_and_semantic_search(tmp_path, monkeypatch):
    """The genuine embed_text.run() -> semantic_search path on the fixture. Loads
    the sentence-transformer, so it is opt-in (`pytest --run-slow`)."""
    import build_metadata
    import embed_text
    import search
    config = _seed_fixture(tmp_path, monkeypatch)
    build_metadata.run()

    embed_text.run()
    assert config.TEXT_EMB_FILE.exists()
    res = search.semantic_search("underground train", df=search.load_metadata(),
                                 top_k=4, min_score=0.0)
    assert not res.empty
    assert set(res["frame_id"]) <= {0, 1, 2, 3}
