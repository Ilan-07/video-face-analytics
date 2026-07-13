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
from ocr import _keep_token, _filter_tokens, _correct_token, correct_tokens
from build_metadata import caption_echoes_text
from embed_text import frame_document
from eval_search import (ocr_jaccard, ocr_matches, is_relevant,
                         precision_at_k)


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


# --------------------------------------------------- M2: OCR lexicon correction
def test_correct_token_fixes_domain_typos():
    assert _correct_token("Metropolite") == "Metropolitan"
    assert _correct_token("Southbo") == "Southbound"
    assert _correct_token("tine") == "line"        # case of the token preserved
    assert _correct_token("hotders") == "holders"


def test_correct_token_preserves_case_and_punctuation():
    assert _correct_token("METROPOLITE") == "METROPOLITAN"   # all caps
    assert _correct_token("metropolite") == "metropolitan"   # all lower
    assert _correct_token("Cregit,") == "Credit,"            # trailing punct


def test_correct_token_leaves_correct_and_real_words_alone():
    assert _correct_token("Victoria") == "Victoria"   # already in lexicon
    assert _correct_token("platform") == "platform"   # already in lexicon
    assert _correct_token("Water") == "Water"         # stoplisted real word
    assert _correct_token("the") == "the"             # below min length


def test_correct_tokens_preserves_token_count():
    toks = ["Victoria", "tine", "Southbo", "platform"]
    out = correct_tokens(toks)
    assert out == ["Victoria", "line", "Southbound", "platform"]
    assert len(out) == len(toks)


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


# ------------------------------------------------- search-quality eval (M2)
def test_ocr_jaccard_identical_is_one():
    assert ocr_jaccard("Bakerloo Line", "bakerloo line") == 1.0


def test_ocr_jaccard_partial_overlap():
    # pred captured only one of two true tokens -> Jaccard 0.5
    assert ocr_jaccard("Platform", "Platform One") == 0.5


def test_ocr_jaccard_both_empty_is_one():
    assert ocr_jaccard("", "") == 1.0


def test_ocr_jaccard_ignores_punctuation_and_ampersand():
    # '&' is non-alphanumeric so 'Hammersmith & City' == 'Hammersmith City'
    assert ocr_jaccard("Hammersmith & City Line",
                       "Hammersmith City Line") == 1.0


def test_ocr_matches_threshold():
    assert ocr_matches("a b c", "a b c d", 0.5) is True     # 3/4 = 0.75
    assert ocr_matches("a", "a b c d", 0.5) is False        # 1/4 = 0.25


def test_is_relevant_word_boundary():
    assert is_relevant("a subway train", ["subway"]) is True
    # substring-but-not-word should NOT count (avoids 'trainee' matching 'train')
    assert is_relevant("a group of trainees", ["train"]) is False


def test_is_relevant_any_term():
    assert is_relevant("a woman on a platform", ["man", "woman"]) is True
    assert is_relevant("an empty tunnel", ["subway", "train"]) is False


def test_precision_at_k_counts_relevant_in_topk():
    docs = ["a subway train", "a station", "the underground", "a man", "a sign"]
    # relevant = mentions subway/underground -> docs 0 and 2 -> 2/5
    assert precision_at_k(docs, ["subway", "underground"], 5) == pytest.approx(0.4)


def test_precision_at_k_respects_k():
    docs = ["a subway", "a subway", "a man", "a man"]
    assert precision_at_k(docs, ["subway"], 2) == 1.0        # top-2 both relevant


def test_precision_at_k_empty_is_zero():
    assert precision_at_k([], ["subway"], 5) == 0.0


# ------------------------------------------------- caption echo-fix (M2)
def test_caption_echo_is_detected():
    # a bare title-card echo (caption == the OCR text) must be flagged
    assert caption_echoes_text("bakerloo line", "Bakerloo Line") is True


def test_echo_fix_fallback_is_not_reflagged():
    # the repaired fallback caption must NOT re-echo, or the fix would loop and
    # re-run the model on every pipeline pass (idempotency regression guard).
    fixed = "a title card that reads: Northern Line"
    assert caption_echoes_text(fixed, "Northern Line") is False


def test_scene_caption_is_not_flagged_as_echo():
    assert caption_echoes_text("a subway train at a platform",
                               "Bakerloo Line") is False


# ------------------------------------------------- visual (CLIP) search (M2)
def test_visual_and_semantic_share_columns():
    from search import visual_search, semantic_search
    # empty query returns the empty frame with the canonical column set
    vcols = list(visual_search("").columns)
    scols = list(semantic_search("").columns)
    assert vcols == scols
    assert "score" in vcols and "field" in vcols


def test_visual_search_ranks_descending():
    import config
    from search import visual_search
    if not config.IMAGE_EMB_FILE.exists():
        pytest.skip("CLIP image index not built yet")
    res = visual_search("a subway train", top_k=5, min_score=0.0)
    scores = res["score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert (res["field"] == "visual").all()


# ------------------------------------------------- M3: scene segmentation
def test_is_title_card_accepts_the_line_cards():
    from scenes import is_title_card
    # OCR rule: a bare line name. This is the ONLY signal that catches Piccadilly,
    # whose caption the M2 echo-fix rewrote.
    assert is_title_card("Piccadilly Line", "picdily line - screenshote") is True
    assert is_title_card("Jubilee Line", "") is True
    # caption rule: BLIP calls every card a black screen. This is the only signal
    # that catches the intro card, whose text does not end in "Line".
    assert is_title_card(
        "London Underground Extravaganza All Lines! Tuesday November",
        "a black background with the words london extrana") is True
    # ...and the echo-fix fallback wording
    assert is_title_card("", "a title card that reads: Northern Line") is True


def test_is_title_card_rejects_platform_signage():
    from scenes import is_title_card
    # The 18:25-19:40 Victoria-line platform signage must NOT read as a card, or
    # a naive /\bline/ would shatter that chapter into 20 fake chapters.
    assert is_title_card("Victoria line northbound platform Walthamstow",
                         "a subway station with a train") is False
    assert is_title_card("EMBANKMENT", "a sign that says embankment") is False
    assert is_title_card("", "a train is pulling passengers") is False


def test_chapter_label_survives_garbled_first_frame():
    from scenes import chapter_label
    # frame 0 of the Piccadilly card OCRs as junk; a later frame is clean.
    assert chapter_label(["picdily line - screenshote", "Piccadilly Line"]) \
        == "Piccadilly Line"
    assert chapter_label(["nothing useful"]) == "Introduction"


def test_is_signage_keeps_stations_and_drops_ads():
    from scenes import is_signage
    for keep in ("EMBANKMENT", "FINCHLEY ROAD", "CASTLE", "Charing Cross",
                 "Metropolitan Southbound platform"):
        assert is_signage(keep) is True, keep
    # OCR shrapnel
    for drop in ("Son", "ran", "iff", "ars", "BES", "ill ace", "Watt", "vate"):
        assert is_signage(drop) is False, drop
    # in-carriage advertising: rule (b) would admit these without the stopword veto
    for drop in ("London Experian Credit", "Experian Score", "Ticket holders only",
                 "Now FREE Find out yours:"):
        assert is_signage(drop) is False, drop


def test_cut_indices_cuts_on_cosine_and_both_card_edges():
    import numpy as np
    from scenes import cut_indices
    # frames:      0     1     2(card) 3(card) 4     5
    sim = np.array([0.99, 0.95, 0.99,   0.95,  0.99])
    is_card = [False, False, True, True, False, False]
    cuts = cut_indices(sim, is_card, thresh=0.70, force_cards=True)
    # no cosine drop below 0.70, so only the card's two edges cut (plus frame 0)
    assert cuts == [0, 2, 4]


def test_cut_indices_cuts_on_low_cosine():
    import numpy as np
    from scenes import cut_indices
    sim = np.array([0.99, 0.41, 0.99])
    cuts = cut_indices(sim, [False] * 4, thresh=0.70, force_cards=True)
    assert cuts == [0, 2]


def test_merge_short_never_drops_a_protected_boundary():
    from scenes import merge_short
    ts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    # The scene starting at frame 5 spans ts[7]-ts[5] = 2.0s, under the 3s floor.
    assert merge_short([0, 5], ts, min_sec=3.0, protected=set()) == [0]
    # ...but the same 2s beat survives when protected. Title cards run ~3 frames
    # (2.0s by this span measure), so without protection every chapter marker --
    # the very thing we force cuts for -- would be merged away.
    assert merge_short([0, 5], ts, min_sec=3.0, protected={5}) == [0, 5]
    # a boundary whose scene clears the floor survives regardless of protection
    assert merge_short([0, 2, 4], ts, min_sec=3.0, protected=set()) == [0, 4]


def test_medoid_index_picks_the_central_frame():
    import numpy as np
    from scenes import medoid_index
    block = np.array([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]], dtype=np.float32)
    block /= np.linalg.norm(block, axis=1, keepdims=True)
    # rows 0 and 1 are near-identical; the centroid sits between them, so the
    # medoid must be one of them, never the orthogonal outlier.
    assert medoid_index(block) in (0, 1)


# ------------------------------------------------- M3: llm cache seam
def test_cache_key_is_order_stable_and_content_sensitive():
    from llm import cache_key
    p = {"temperature": 0.0, "max_tokens": 10}
    assert cache_key("m", "p", ["d"], p) == cache_key("m", "p", ["d"],
                                                      {"max_tokens": 10,
                                                       "temperature": 0.0})
    assert cache_key("m", "p", ["d"], p) != cache_key("m", "p2", ["d"], p)
    assert cache_key("m", "p", ["d"], p) != cache_key("m", "p", ["d2"], p)
    assert cache_key("m", "p", ["d"], p) != cache_key("m2", "p", ["d"], p)


def test_generate_without_key_raises_runtimeerror_not_keyerror(monkeypatch):
    import llm
    # Force the no-key path deterministically, independent of ambient env/.env, so
    # the test never reaches the network and can never pollute the committed cache
    # (an earlier version of this test did exactly that). A cache miss with no key
    # must raise RuntimeError -- never a bare KeyError on os.environ.
    monkeypatch.setattr(llm, "api_key", lambda: None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        llm.generate(f"uncacheable probe {id(object())}", use_cache=False)


# ------------------------------------------------- M3: narration helpers
def test_parse_mmss():
    from narrate import parse_mmss
    assert parse_mmss("12:23") == 743
    assert parse_mmss("0:06") == 6
    assert parse_mmss("1:75") is None      # 75 seconds is not a timestamp
    assert parse_mmss("garbage") is None


def test_extract_json_array_tolerates_fences_and_preamble():
    from narrate import extract_json_array
    assert extract_json_array('Sure!\n```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert extract_json_array('[{"a": 1}]') == [{"a": 1}]
    assert extract_json_array("[[1, 2], [3]]") == [[1, 2], [3]]
    assert extract_json_array("no array at all") is None
    assert extract_json_array("[broken") is None


def test_validate_timeline_repairs_rather_than_trusts():
    from narrate import validate_timeline
    events, problems = validate_timeline([
        {"timestamp": "5:00", "description": "later"},
        {"timestamp": "1:00", "description": "earlier"},
        {"timestamp": "99:00", "description": "past the end of the video"},
        {"timestamp": "nonsense", "description": "unparseable"},
        {"timestamp": "2:00", "description": ""},
    ], max_sec=1415.5)
    assert [e["timestamp"] for e in events] == ["1:00", "5:00"]   # sorted
    assert any("out of range" in p for p in problems)
    assert any("unparseable" in p for p in problems)
    assert any("chronological order" in p for p in problems)


def test_strip_reasoning_removes_thinking_block():
    from narrate import strip_reasoning
    assert strip_reasoning("<thinking>plan</thinking>\n\n## Story\ntext") \
        == "## Story\ntext"
    assert strip_reasoning("## Story\ntext") == "## Story\ntext"


def test_split_segments_maps_headings_to_chapters():
    from narrate import split_segments
    story = "## Bakerloo Line\nOpens at 0:06.\n\n## Central Line\nThen onward."
    assert split_segments(story, {"Bakerloo Line", "Central Line"}) == {
        "Bakerloo Line": "Opens at 0:06.", "Central Line": "Then onward."}
    assert split_segments("no headings here", {"Bakerloo Line"}) == {}


def test_scene_digest_marks_title_cards_and_swaps_in_vlm_text():
    from narrate import scene_digest
    scenes = [
        {"is_title_card": True, "start_mmss": "0:06", "end_mmss": "0:08",
         "chapter_label": "Bakerloo Line"},
        {"is_title_card": False, "start_mmss": "0:09", "end_mmss": "0:11",
         "chapter_label": "Bakerloo Line", "scene_index": 1, "n_frames": 3,
         "representative_caption": "a sign", "ocr_texts": ["EMBANKMENT"],
         "face_ids": []},
    ]
    plain = scene_digest(scenes)
    assert 'TITLE CARD: "Bakerloo Line"' in plain
    assert "visual: a sign" in plain and "EMBANKMENT" in plain
    assert "recurring faces present: none" in plain
    # the ablation swaps the caption for the vision model's description
    vlm = scene_digest(scenes, {1: "An empty platform, no train present."})
    assert "visual: An empty platform, no train present." in vlm
    assert "a sign" not in vlm


# ------------------------------------------------- M3: story evaluation
def test_chronology_score_detects_time_travel():
    from eval_story import chronology_score, cited_timestamps
    assert chronology_score(cited_timestamps("at 0:06 then 2:18 then 12:23")) == 1.0
    # one backward step out of two adjacent pairs
    assert chronology_score([743, 6, 138]) == pytest.approx(0.5)
    assert chronology_score([5]) == 1.0     # vacuous, reported with the count


def test_grounding_catches_invented_content():
    from eval_story import grounding
    # "escalator" appears in no caption -> ungrounded
    assert grounding("A crowded escalator", "a train at a subway platform") == 0.0
    assert grounding("a train platform", "a train is at the platform") == 1.0


def test_distinct_ngram_ratio_penalises_repetition():
    from eval_story import distinct_ngram_ratio
    repetitive = distinct_ngram_ratio("a train a train a train a train", 3)
    varied = distinct_ngram_ratio("the quick brown fox jumps over lazy dogs", 3)
    assert repetitive < varied
    assert varied == 1.0


def test_caption_adequacy_measures_what_blip_missed():
    from eval_story import caption_adequacy
    # BLIP hallucinates a train; the VLM sees an empty platform with an escalator
    low = caption_adequacy("a train is pulling passengers",
                           "An empty platform with an escalator and no train")
    high = caption_adequacy("an empty platform with an escalator and no train",
                            "An empty platform with an escalator and no train")
    assert low < 0.5 < high


def test_scene_coverage_counts_mentioned_chapters():
    from eval_story import scene_coverage
    assert scene_coverage("We ride the Jubilee Line.",
                          ["Jubilee Line", "Victoria Line"]) == 0.5


def test_timeline_in_scene_bounds():
    from eval_story import timeline_in_scene_bounds
    scenes = [{"start_sec": 0.0, "end_sec": 5.0}, {"start_sec": 20.0, "end_sec": 25.0}]
    # 3.0 inside scene 0; 12.0 is in neither span and beyond tolerance
    assert timeline_in_scene_bounds([{"timestamp_sec": 3.0},
                                     {"timestamp_sec": 12.0}], scenes) == pytest.approx(0.5)
    # a whole-second m:ss event just before a sub-second scene start still matches
    # (2:18 = 138.0s vs a scene starting at 138.138s must not read as out of bounds)
    assert timeline_in_scene_bounds([{"timestamp_sec": 138.0}],
                                    [{"start_sec": 138.138, "end_sec": 140.140}]) == 1.0


# ------------------------------------------------- M3: artifact schemas
def test_scenes_json_schema():
    import config
    if not config.SCENES_JSON.exists():
        pytest.skip("scenes.json not generated yet")
    with open(config.SCENES_JSON) as f:
        scenes = json.load(f)
    assert scenes, "no scenes"
    required = {"scene_index", "chapter_index", "chapter_label", "is_title_card",
                "start_sec", "end_sec", "start_mmss", "end_mmss", "n_frames",
                "start_frame_id", "end_frame_id", "keyframe_frame_id",
                "keyframe_file", "face_ids", "ocr_texts",
                "representative_caption", "captions"}
    assert required.issubset(scenes[0].keys())
    # scenes tile the video: contiguous, ordered, every frame in exactly one
    assert [s["scene_index"] for s in scenes] == list(range(len(scenes)))
    for a, b in zip(scenes, scenes[1:]):
        assert a["end_frame_id"] + 1 == b["start_frame_id"]
        assert a["start_sec"] <= b["start_sec"]
    # the video is a 12-chapter tour: an intro card plus 11 Underground lines
    cards = [s for s in scenes if s["is_title_card"]]
    assert len(cards) == 12
    assert len({c["chapter_label"] for c in cards}) == 12
    assert sum(c["chapter_label"] == "Introduction" for c in cards) == 1


def test_metadata_has_milestone3_fields():
    import config
    if not config.METADATA_JSON.exists():
        pytest.skip("frame_metadata.json not generated yet")
    with open(config.METADATA_JSON) as f:
        meta = json.load(f)
    # Task 4: metadata enhancement -- story segment, event description, scene index
    for key in ("scene_index", "story_segment", "event_description"):
        assert key in meta[0], key
    if not config.SCENES_JSON.exists():
        pytest.skip("scenes.json absent: M3 columns are legitimately empty")
    assert all(r["scene_index"] is not None for r in meta)
    assert all(r["story_segment"] for r in meta)


def test_timeline_json_is_chronological_and_in_range():
    import config
    if not config.TIMELINE_JSON.exists():
        pytest.skip("timeline.json not generated yet")
    with open(config.TIMELINE_JSON) as f:
        events = json.load(f)["events"]
    secs = [e["timestamp_sec"] for e in events]
    assert secs == sorted(secs), "timeline must run forwards"
    assert all(0 <= s <= config.VIDEO_DURATION_SEC for s in secs)
    assert all(e["description"].strip() for e in events)


def test_story_json_schema():
    import config
    if not config.STORY_JSON.exists():
        pytest.skip("story.json not generated yet")
    with open(config.STORY_JSON) as f:
        story = json.load(f)
    assert {"model", "strategy", "source", "summary", "story", "segments"} \
        .issubset(story.keys())
    assert story["story"].strip() and story["summary"].strip()


# ------------------------------------------------- M3: batched VLM descriptions
def test_parse_batch_enforces_the_scene_index_contract():
    from describe_scenes import parse_batch
    reply = ('[{"scene_index": 2, "description": "An empty platform."},'
             ' {"scene_index": 3, "description": "A train with doors open."}]')
    assert parse_batch(reply, {2, 3}) == {
        2: "An empty platform.", 3: "A train with doors open."}
    # a scene we did not ask about is discarded rather than trusted -- this is what
    # stops a mis-aligned batch reply from silently mislabelling 24 keyframes
    assert parse_batch(reply, {2}) == {2: "An empty platform."}
    # empty descriptions and non-objects are dropped
    assert parse_batch('[{"scene_index": 2, "description": ""}, 7]', {2}) == {}
    # the caller sees the omission and can retry that scene on its own
    assert set(parse_batch(reply, {2, 3, 4})) == {2, 3}


def test_parse_batch_survives_fences_and_junk():
    from describe_scenes import parse_batch
    fenced = '```json\n[{"scene_index": 5, "description": "A tunnel."}]\n```'
    assert parse_batch(fenced, {5}) == {5: "A tunnel."}
    assert parse_batch("the model refused to emit json", {5}) == {}
    assert parse_batch('[{"scene_index": "not an int", "description": "x"}]', {5}) == {}
