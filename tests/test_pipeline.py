"""Unit tests for the pipeline's pure-logic components.

Run with:  .venv/bin/python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

import json

import config
from util import iou, nose_offset, blur_var
from detect_faces import _reconcile
from eval_labeled import _pairwise
from recognize import _quality_weight, _robust_template, _link_clusters
from appearance import body_box
from search import (search, fmt_ts, _snippet, group_consecutive,
                    _topk_similar)
from ocr import _keep_token, _filter_tokens
from build_metadata import caption_echoes_text
from embed_text import frame_document


# ---------------------------------------------------------------- geometry
def test_iou_identical():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_disjoint():
    assert iou([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0


def test_iou_half_overlap():
    # A=[0,0,2,2] (4), B=[1,0,3,2] (4); inter=2 -> 2/(4+4-2)=1/3
    assert iou([0, 0, 2, 2], [1, 0, 3, 2]) == pytest.approx(1 / 3)


def test_iou_containment():
    # B fully inside A: inter=4, union=16 -> 0.25
    assert iou([0, 0, 4, 4], [1, 1, 3, 3]) == pytest.approx(0.25)


def test_iou_is_symmetric():
    a, b = [0, 0, 4, 3], [2, 1, 6, 5]
    assert iou(a, b) == pytest.approx(iou(b, a))


# ---------------------------------------------------------------- pose
def test_nose_offset_frontal_is_zero():
    # eyes at x=0 and x=10, nose centered at x=5
    kps = [(0, 0), (10, 0), (5, 5), (2, 9), (8, 9)]
    assert nose_offset(kps) == pytest.approx(0.0, abs=1e-6)


def test_nose_offset_profile_is_large():
    kps = [(0, 0), (10, 0), (9, 5), (2, 9), (8, 9)]  # nose shifted right
    assert nose_offset(kps) == pytest.approx(0.4, abs=1e-6)


# ---------------------------------------------------------------- blur
def test_blur_var_constant_image_is_zero():
    img = np.full((32, 32, 3), 127, np.uint8)
    assert blur_var(img) == pytest.approx(0.0)


def test_blur_var_textured_greater_than_flat():
    flat = np.full((32, 32, 3), 127, np.uint8)
    noisy = np.zeros((32, 32, 3), np.uint8)
    noisy[::2] = 255  # high-frequency stripes
    assert blur_var(noisy) > blur_var(flat)


# ---------------------------------------------------------------- reconcile
def _rec(cid, frame, bbox, bt):
    return {"crop_id": cid, "frame_id": frame, "bbox": bbox, "bt_tid": bt}


def test_reconcile_birth_frame_folds_into_track():
    # frame0 is the birth (-1); frames 1,2 are confirmed as track 5.
    box = [0, 0, 10, 10]
    recs = [_rec("a", 0, box, -1), _rec("b", 1, box, 5), _rec("c", 2, box, 5)]
    out = _reconcile(recs)
    assert out["a"] == out["b"] == out["c"] == "t5"


def test_reconcile_isolated_minus_one_is_singleton():
    recs = [_rec("x", 0, [100, 100, 110, 110], -1)]
    out = _reconcile(recs)
    assert out["x"] == "s0"


def test_reconcile_low_iou_birth_does_not_fold():
    # -1 at frame0 far from the confirmed box at frame1 -> stays singleton.
    recs = [_rec("a", 0, [0, 0, 10, 10], -1),
            _rec("b", 1, [500, 500, 510, 510], 7)]
    out = _reconcile(recs)
    assert out["b"] == "t7"
    assert out["a"].startswith("s")
    assert out["a"] != out["b"]


def test_reconcile_distinct_confirmed_tracks_stay_distinct():
    recs = [_rec("a", 0, [0, 0, 10, 10], 1),
            _rec("b", 0, [50, 50, 60, 60], 2)]
    out = _reconcile(recs)
    assert out["a"] == "t1" and out["b"] == "t2"


# ---------------------------------------------------------------- pairwise
def test_pairwise_perfect():
    m = _pairwise(["A", "A", "B", "B"], ["X", "X", "Y", "Y"])
    assert m["pairwise_precision"] == 1.0
    assert m["pairwise_recall"] == 1.0
    assert m["pairwise_f1"] == 1.0


def test_pairwise_false_merge_lowers_precision():
    # all predicted same, but truth has two people -> precision 1/3, recall 1
    m = _pairwise(["A", "A", "A"], ["X", "X", "Y"])
    assert m["pairs_tp"] == 1 and m["pairs_fp"] == 2 and m["pairs_fn"] == 0
    assert m["pairwise_precision"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["pairwise_recall"] == 1.0


def test_pairwise_false_split_lowers_recall():
    # one true person split across two predicted clusters -> recall 0
    m = _pairwise(["A", "B"], ["X", "X"])
    assert m["pairs_tp"] == 0 and m["pairs_fn"] == 1
    assert m["pairwise_recall"] == 0.0


# ---------------------------------------------------------------- config sanity
def test_config_thresholds_sane():
    assert 0.0 < config.CLUSTER_LINK_DIST < 1.0
    assert config.FPS >= 1
    assert config.MIN_IDENTITY_FACES >= 1


# ------------------------------------------------------- track-template logic
def _df(det, blur, nose):
    return pd.DataFrame({"det_score": det, "blur_var": blur, "nose_offset": nose})


def test_quality_weight_formula():
    # weight = det_score * blur_var / (1 + nose_offset)
    w = _quality_weight(_df([0.9, 0.5], [100.0, 50.0], [0.0, 1.0]))
    assert w == pytest.approx([90.0, 12.5])


def test_quality_weight_clipped_positive():
    # a zero product must be floored to a tiny positive (no zero-weight faces).
    w = _quality_weight(_df([0.0], [0.0], [0.0]))
    assert w[0] == pytest.approx(1e-6)


def test_robust_template_single_face_is_normalized():
    out = _robust_template(np.array([[3.0, 4.0, 0.0]]), np.array([1.0]))
    assert np.linalg.norm(out) == pytest.approx(1.0)
    assert out == pytest.approx([0.6, 0.8, 0.0])


def test_robust_template_is_quality_weighted():
    # two faces, no trim (< TEMPLATE_MIN_FACES_FOR_TRIM); heavier weight wins.
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = _robust_template(vecs, np.array([3.0, 1.0]))
    expected = np.array([3.0, 1.0, 0.0]) / np.linalg.norm([3.0, 1.0, 0.0])
    assert out == pytest.approx(expected)
    assert out[0] > out[1]  # first face dominates


def test_robust_template_drops_outlier_frame():
    # 4 identical inliers + 1 orthogonal outlier (>= TRIM faces). The outlier's
    # cosine to the provisional mean (0.243) is below TEMPLATE_OUTLIER_COS=0.40,
    # so it is dropped and the template collapses onto the inlier direction.
    assert config.TEMPLATE_MIN_FACES_FOR_TRIM <= 5
    assert config.TEMPLATE_OUTLIER_COS == pytest.approx(0.40)
    vecs = np.array([[1.0, 0.0, 0.0]] * 4 + [[0.0, 0.0, 1.0]])
    out = _robust_template(vecs, np.ones(5))
    assert out == pytest.approx([1.0, 0.0, 0.0])  # outlier component gone


def test_robust_template_keeps_outlier_when_track_too_small():
    # Same disagreement but only 2 faces -> below trim threshold, both kept.
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    out = _robust_template(vecs, np.ones(2))
    assert out == pytest.approx([0.5 ** 0.5, 0.0, 0.5 ** 0.5])


# ------------------------------------------------------- cluster linking
def _unit(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


def _link(tids, T, labels, frames, dist):
    faces = pd.DataFrame({"track_id": tids, "frame_id": frames})
    old = config.CLUSTER_LINK_DIST
    config.CLUSTER_LINK_DIST = dist
    try:
        return _link_clusters(tids, np.array(T), labels, faces)
    finally:
        config.CLUSTER_LINK_DIST = old


def test_link_respects_cooccurrence_cannot_link():
    # 3 identical-embedding tracks, but a & b share a frame -> different people.
    out = _link(["a", "b", "c"], [_unit([1, 0, 0])] * 3, [0, 1, 2],
                [0, 0, 1], dist=0.60)
    assert out[0] != out[1]               # co-occurring pair never merges
    assert out[2] in (out[0], out[1])     # the third joins one of them


def test_link_complete_linkage_blocks_chaining():
    # a-b close (0.2), b-c close (0.4), a-c far (1.0). Single-linkage would chain
    # a-b-c; complete-linkage must NOT pull c in (max pair a-c = 1.0 > 0.45).
    out = _link(["a", "b", "c"],
                [_unit([1, 0, 0]), _unit([0.8, 0.6, 0]), _unit([0, 1, 0])],
                [0, 1, 2], [0, 1, 2], dist=0.45)
    assert out[0] == out[1]
    assert out[2] != out[0]


# ------------------------------------------------------- appearance body box
def test_body_box_is_clothing_focused():
    # face (100,100)-(140,160) in a tall 640x1000 frame (no clamping).
    bx1, by1, bx2, by2 = body_box(100, 100, 140, 160, 640, 1000)
    fw, fh = 40, 60
    assert bx2 - bx1 == pytest.approx(config.BODY_W_SCALE * fw, abs=1)  # narrowed
    assert by1 == 160                                  # starts at the chin (y2)
    assert by2 == 160 + int(config.BODY_DOWN_SCALE * fh)  # torso below the face


def test_body_box_clamps_to_frame_edges():
    # face near the right/bottom edge must not exceed the frame.
    bx1, by1, bx2, by2 = body_box(600, 440, 640, 480, 640, 480)
    assert bx2 <= 640 and by2 <= 480
    assert bx1 >= 0 and by1 >= 0


# ------------------------------------------------------- M2: search core
def _meta_df():
    return pd.DataFrame.from_records([
        {"frame_id": 0, "timestamp_sec": 0.0, "filename": "frame_000000.jpg",
         "face_ids": ["Face_01"], "ocr_text": "Welcome to the show",
         "caption": "a man on a stage"},
        {"frame_id": 1, "timestamp_sec": 65.0, "filename": "frame_000065.jpg",
         "face_ids": [], "ocr_text": "", "caption": "a city street at night"},
        {"frame_id": 2, "timestamp_sec": 130.0, "filename": "frame_000130.jpg",
         "face_ids": ["Face_01", "Face_02"], "ocr_text": "WELCOME BACK",
         "caption": "two people talking"},
    ])


def test_search_case_insensitive_hit():
    res = search("welcome", df=_meta_df())
    assert res["frame_id"].tolist() == [0, 2]      # both OCR rows, sorted by time


def test_search_miss_returns_empty():
    assert search("nonexistent-token", df=_meta_df()).empty


def test_search_empty_query_returns_empty():
    assert search("   ", df=_meta_df()).empty


def test_search_passes_through_face_ids_and_mmss():
    res = search("welcome back", df=_meta_df())
    assert res.iloc[0]["face_ids"] == ["Face_01", "Face_02"]
    assert res.iloc[0]["mmss"] == "2:10"          # 130s -> 2:10


def test_search_ocr_only_ignores_captions_by_default():
    # "stage" only appears in a caption -> no OCR hit by default.
    assert search("stage", df=_meta_df()).empty
    res = search("stage", df=_meta_df(), fields=("ocr_text", "caption"))
    assert res["frame_id"].tolist() == [0]
    assert res.iloc[0]["field"] == "caption"


def test_fmt_ts():
    assert fmt_ts(0) == "0:00"
    assert fmt_ts(75) == "1:15"


def test_snippet_centers_on_match():
    s = _snippet("the quick brown fox jumps", "brown", width=12)
    assert "brown" in s


def test_search_regex():
    df = _meta_df()
    res = search(r"WELCOME\s+BACK", df=df, regex=True)
    assert res["frame_id"].tolist() == [2]


def test_search_fuzzy_tolerates_typo():
    df = pd.DataFrame.from_records([
        {"frame_id": 0, "timestamp_sec": 0.0, "filename": "f.jpg",
         "face_ids": [], "ocr_text": "london embankmnt station", "caption": ""},
    ])
    # exact miss (OCR dropped a letter), fuzzy hit on the misspelled token
    assert search("embankment", df=df).empty
    assert not search("embankment", df=df, fuzzy=True).empty


def test_search_score_counts_occurrences():
    df = pd.DataFrame.from_records([
        {"frame_id": 0, "timestamp_sec": 0.0, "filename": "f.jpg",
         "face_ids": [], "ocr_text": "go go go", "caption": ""},
    ])
    assert search("go", df=df).iloc[0]["score"] == 3


# ------------------------------------------------------- M2: time-range grouping
def test_group_consecutive_collapses_same_text():
    df = pd.DataFrame.from_records([
        {"frame_id": 6, "timestamp_sec": 6.0, "filename": "a.jpg",
         "face_ids": [], "ocr_text": "Bakerloo Line", "caption": ""},
        {"frame_id": 7, "timestamp_sec": 7.0, "filename": "b.jpg",
         "face_ids": [], "ocr_text": "Bakerloo Line", "caption": ""},
        {"frame_id": 8, "timestamp_sec": 8.0, "filename": "c.jpg",
         "face_ids": [], "ocr_text": "Bakerloo Line", "caption": ""},
    ])
    groups = group_consecutive(search("Bakerloo", df=df))
    assert len(groups) == 1
    assert groups[0]["start"] == "0:06" and groups[0]["end"] == "0:08"
    assert groups[0]["frames"] == 3


def test_group_consecutive_splits_on_gap():
    df = pd.DataFrame.from_records([
        {"frame_id": 0, "timestamp_sec": 0.0, "filename": "a.jpg",
         "face_ids": [], "ocr_text": "EXIT", "caption": ""},
        {"frame_id": 99, "timestamp_sec": 99.0, "filename": "b.jpg",
         "face_ids": [], "ocr_text": "EXIT", "caption": ""},
    ])
    assert len(group_consecutive(search("EXIT", df=df))) == 2


# ------------------------------------------------------- M2: OCR token filter
def test_keep_token_rejects_short_and_lowconf():
    assert _keep_token("Welcome", 95) is True
    assert _keep_token("ei", 95) is False          # too short
    assert _keep_token("Wa}", 40) is False         # below min_conf
    assert _keep_token("123", 95) is False         # no letter


def test_filter_tokens_drops_noise_keeps_text():
    words = ["Welcome", "Wa}", "ei", "London", "a"]
    confs = [95, 40, 95, 90, 99]
    toks, kept = _filter_tokens(words, confs)
    assert toks == ["Welcome", "London"]
    assert kept == [95.0, 90.0]


# ------------------------------------------------------- M2: caption text-echo
def test_caption_echoes_text_flags_title_card():
    assert caption_echoes_text("london underground all lines",
                               "London Underground All Lines") is True


def test_caption_echoes_text_false_for_scene():
    assert caption_echoes_text("a man walking down the platform",
                               "EMBANKMENT") is False


# ------------------------------------------------------- M2: semantic ranking
def test_topk_similar_ranks_by_cosine():
    # normalized rows; query aligns with row 1, then row 2, then row 0.
    mat = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])
    q = np.array([0.0, 1.0])
    idx, scores = _topk_similar(q, mat, k=3)
    assert idx[0] == 1                       # exact match first
    assert idx[-1] == 0                      # orthogonal last
    assert scores[0] == pytest.approx(1.0)
    assert list(scores) == sorted(scores, reverse=True)


def test_topk_similar_k_caps_results():
    mat = np.eye(4)
    idx, _ = _topk_similar(np.array([1.0, 0, 0, 0]), mat, k=2)
    assert len(idx) == 2 and idx[0] == 0


def test_frame_document_joins_caption_and_ocr():
    assert frame_document("a subway train", "EMBANKMENT") == \
        "a subway train. EMBANKMENT"
    assert frame_document("a subway train", "") == "a subway train"
    assert frame_document("", "") == ""


def test_semantic_index_matches_metadata():
    # If the semantic index exists, it must align with the metadata repository.
    if not config.TEXT_EMB_FILE.exists() or not config.METADATA_JSON.exists():
        pytest.skip("semantic index not generated yet")
    data = np.load(config.TEXT_EMB_FILE, allow_pickle=True)
    with open(config.METADATA_JSON) as f:
        n_meta = len(json.load(f))
    assert data["embeddings"].shape[0] == n_meta
    assert data["embeddings"].shape[0] == len(data["frame_ids"])
    # rows are L2-normalized -> norms ~ 1
    norms = np.linalg.norm(data["embeddings"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


# ------------------------------------------------------- M2: artifact schemas
# These validate the generated dataset; they skip cleanly before the pipeline
# has been run so the unit suite stays green on a fresh checkout.
def test_ocr_csv_schema():
    if not config.OCR_CSV.exists():
        pytest.skip("ocr.csv not generated yet")
    df = pd.read_csv(config.OCR_CSV)
    # Task 1: OCR text stored with frame id, timestamp AND associated face ids.
    assert set(["frame_id", "timestamp_sec", "face_ids", "text",
                "n_tokens", "mean_conf"]).issubset(df.columns)


def test_captions_csv_schema():
    if not config.CAPTIONS_CSV.exists():
        pytest.skip("captions.csv not generated yet")
    df = pd.read_csv(config.CAPTIONS_CSV)
    # Task 3: captions stored with frame id, timestamp, face ids AND ocr text.
    assert set(["frame_id", "timestamp_sec", "face_ids",
                "ocr_text", "caption"]).issubset(df.columns)


def test_metadata_repository_has_task4_fields():
    if not config.METADATA_JSON.exists():
        pytest.skip("frame_metadata.json not generated yet")
    with open(config.METADATA_JSON) as f:
        records = json.load(f)
    assert records, "metadata repository is empty"
    r = records[0]
    for field in ("frame_id", "timestamp_sec", "face_ids", "ocr_text", "caption"):
        assert field in r
    assert isinstance(r["face_ids"], list)   # Task 4: Face IDs as a list
