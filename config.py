"""Shared configuration and paths for the face analytics pipeline."""
import hashlib
import json
from pathlib import Path

# Dataset
VIDEO_URL = "https://www.youtube.com/watch?v=d2g9HlwoC-s"

# Project layout
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
VIDEO_DIR = DATA / "video"
FRAME_DIR = DATA / "frames"
FACE_DIR = DATA / "faces"
EMB_DIR = DATA / "embeddings"
REPORT_DIR = ROOT / "reports"
MONTAGE_DIR = REPORT_DIR / "montages"
ANNOT_DIR = REPORT_DIR / "annotated"

# CSV / artifacts
FRAMES_CSV = DATA / "frames.csv"
FACES_CSV = DATA / "faces.csv"
IDENTITIES_CSV = DATA / "identities.csv"
EMB_FILE = EMB_DIR / "embeddings.npz"
TEMPLATE_FILE = EMB_DIR / "track_templates.npz"
LOG_FILE = REPORT_DIR / "pipeline.log"

# ---- Milestone 2: per-frame text, captions, metadata repository ----
OCR_CSV = DATA / "ocr.csv"
CAPTIONS_CSV = DATA / "captions.csv"
METADATA_CSV = DATA / "frame_metadata.csv"
METADATA_JSON = DATA / "frame_metadata.json"
TEXT_EMB_FILE = EMB_DIR / "text_embeddings.npz"

# ---- Fix #4: sampling rate ----
# 1 FPS: one sample/sec. Sparser than 2 FPS (may miss sub-second appearances)
# but fewer near-duplicate frames and faster end-to-end.
FPS = 1

# ---- Detection (SCRFD) ----
# det_size is the size frames are resized to before detection. 640 downscaled
# the 1280x720 source enough to lose high-confidence faces purely to lost
# pixels; 1024 recovers them at ~2x detect cost. Raise toward 1280 for more.
DET_SIZE = (1024, 1024)
DET_THRESH = 0.4         # minimum confidence to keep/track a detection. Lowered
                         # for coverage; clean grouping is protected downstream by
                         # track templates + quality-gated template contribution.
MIN_FACE_PX = 36         # drop faces smaller than this (width or height). 36px at
                         # 1080p ~= the old 24px at 720p (same physical face size),
                         # so we keep coverage without admitting more tiny noise

# ---- Fix #5: quality gating for template contribution ----
HQ_DET_SCORE = 0.60      # high-quality detection confidence
MIN_BLUR_VAR = 25.0      # Laplacian variance; below = too blurry
MAX_NOSE_OFFSET = 0.75   # |nose offset / eye-dist|; above = strong profile

# ---- Fix #1: tracking via supervision/trackers ByteTrack ----
# frame_rate is kept at 30 so ByteTrack's buffer math (int(fps/30*buffer))
# is not collapsed by our low sampling rate; the buffer is then expressed
# directly in processed frames.
TRACK_IOU = 0.30            # min IoU to associate a detection to a track
TRACK_MAX_GAP = 2           # processed frames a track may be missing
TRACK_ACTIVATION = 0.50     # min confidence to start a track (== DET_THRESH)
TRACK_HIGH_CONF = 0.60      # ByteTrack high/low split
TRACK_FRAME_RATE = 30       # keep at 30 (see note above)
TRACK_LINK_IOU = 0.50       # IoU to fold a track's birth frame back in

# ---- Track-level grouping ----
# Tracks are grouped by constrained complete-linkage agglomeration over cosine
# distance (see recognize._link_clusters), tuned by CLUSTER_LINK_DIST below.
# Complete linkage (vs DBSCAN single-linkage) forbids chaining marginal faces
# into incoherent junk clusters; the co-occurrence cannot-link forbids merging
# faces seen in the same frame.

# ---- Track template robustness ----
# A track's template is a quality-weighted mean of its face embeddings, after
# dropping frames that disagree with the track (mis-tracked / occluded crops
# whose cosine to the provisional mean is low). Cleaner templates -> cleaner
# clusters, without throwing away whole tracks.
TEMPLATE_USE_QUALITY_WEIGHT = True   # weight faces by det_score * sharpness / pose
TEMPLATE_MIN_FACES_FOR_TRIM = 3      # only reject outliers when track has >=3 faces
TEMPLATE_OUTLIER_COS = 0.40          # drop a face whose cosine to track mean < this

# ---- Second-stage cluster linking (raise completeness safely) ----
# DBSCAN over-segments the same person across scenes. This pass merges clusters
# with COMPLETE-linkage agglomeration -- every member pair of the two clusters
# must be within CLUSTER_LINK_DIST, so a loose threshold can't chain into a blob
# (the eps=0.55 failure) -- and is blocked by co-occurrence cannot-link: two
# clusters that ever share a frame are different people and never merge.
CLUSTER_LINK_ENABLE = True
CLUSTER_LINK_DIST = 0.50             # the clustering distance ceiling. Complete
                                     # linkage means every member pair is within
                                     # this, so clusters stay coherent (top-cluster
                                     # within-cosine ~0.73); 0.55+ starts merging
                                     # dissimilar faces.

# ---- Best-shot cross-scene linking ----
# After clustering, merge clusters whose BEST-quality (frontal/sharp) faces match
# tightly, gated by co-occurrence cannot-link. Comparing best-shot to best-shot
# (not pose-polluted means) consolidates the same person across pose/scene. This
# intentionally lowers within-cluster cohesion (mixes poses), so VISUAL review --
# not cohesion -- is the validator here.
BEST_SHOT_ENABLE = True
BEST_SHOT_K = 3          # top-quality prototype faces compared per cluster
BEST_SHOT_DIST = 0.50    # cosine-distance ceiling. Complete-linkage on best shots
                         # (all prototype pairs must pass) lets this be loose enough
                         # to consolidate a person's FRONTAL clusters across scene
                         # cuts without chaining different people.

# ---- Clothing/body appearance re-ID (optional scaffold) ----
# Orthogonal cross-scene signal: the same outfit recurs across scenes regardless
# of face pose. Requires torchreid (extra dep); off by default. When enabled,
# appearance.py embeds body crops and recognize fuses them -- allowing a looser
# face match when clothing agrees. See appearance.py.
APPEARANCE_ENABLE = False       # optional; needs requirements-appearance.txt
# Clothing-focused body crop: the torso BELOW the chin (head is already the face),
# narrowed to suppress side-background -- shared backgrounds are the main cause of
# clothing false-matches (e.g. dark seats in a transit scene).
BODY_W_SCALE = 2.4              # torso width as a multiple of face width
BODY_DOWN_SCALE = 5.0          # torso height below the chin, in face-heights
APPEARANCE_DIST = 0.25          # body-appearance cosine-distance ceiling (tight:
                                # similar outfits in a shared setting false-match)
APPEARANCE_FACE_DIST = 0.55     # looser-than-face-clustering ceiling allowed when
                                # clothing agrees; complete-linkage so all best-shot
                                # pairs must pass (no chaining into mixed clusters)
APPEARANCE_FILE = EMB_DIR / "appearance.npz"

# ---- False-positive backstop ----
# A cluster is a real identity only if corroborated: >=MIN_IDENTITY_FACES faces,
# OR a high-quality face, OR a confidently-detected face (det_score>=REAL_FACE_DET).
# This keeps real single-appearance people but drops lone low-confidence junk /
# non-faces (e.g. signage, blur blobs) from the identity count -- the det_thresh
# was lowered to 0.4 for recall, so this is the non-face backstop.
MIN_IDENTITY_FACES = 2
REAL_FACE_DET = 0.55

# ---- Two-tier identity count ----
# "Total face groups" overcounts real people (one-off background faces, profile-
# only fragments). Report a "featured cast" -- identities with real presence
# (>= this many frames of screen time) -- as the meaningful headline, with the
# brief/background tail reported separately. Presence (not track count) is used
# because track-breaks within one scene would otherwise inflate the count.
RECURRING_MIN_FRAMES = 5

# ---- Label-free completeness check (tracking continuity must-link) ----
# Co-occurrence gives precision (same-frame => different people). Its recall
# analog: two tracks bridged across a brief gap at the same location are the SAME
# person (must-link). How often clustering keeps them together = an objective
# over-segmentation / completeness signal, no labels. (Catches intra-shot splits;
# blind to cross-cut splits, which have no temporal bridge.)
CONTINUITY_MAX_GAP = 3          # max frame gap between bridged tracks
CONTINUITY_IOU = 0.5            # min bbox IoU across the gap to call it one person

# ---- Fix #7: compute backend ----
# CPU on Mac. CoreML was measured ~3-4x SLOWER for SCRFD here: its dynamic
# input shapes (det [1,3,'?','?']) defeat ANE compilation and bounce ops back
# to CPU. Only enable CUDA on a real NVIDIA box.
USE_GPU = False                       # set True to prefer CUDA
CTX_ID = 0 if USE_GPU else -1
PROVIDERS = (["CUDAExecutionProvider", "CPUExecutionProvider"]
             if USE_GPU else ["CPUExecutionProvider"])

# ---- Fix #6: reporting ----
MAX_ANNOTATED_FRAMES = 8
TIMELINE_W = 1100
TIMELINE_LANE_H = 26

# ---- Milestone 2: OCR (Tesseract via pytesseract) ----
# Tesseract is a system binary (brew install tesseract). We keep OCR torch-free
# so it stays cheap on 8GB RAM. image_to_data gives per-token confidence; we drop
# low-confidence tokens so the search index isn't polluted by OCR hallucinations
# on textureless frames.
OCR_LANG = "eng"
OCR_PSM = 3              # Tesseract page segmentation: 3 = fully automatic
OCR_MIN_CONF = 60       # keep tokens with confidence >= this (0-100). 60 (was 40)
                        # drops the bulk of Tesseract's low-confidence hallucinations
                        # on textured/no-text frames.
OCR_MIN_TOKEN_LEN = 3   # drop tokens shorter than this; combined with the
                        # has-a-letter rule this removes "ei", "a a", stray glyphs.
OCR_UPSCALE = 1.5       # upscale factor before thresholding (helps small text)

# ---- Milestone 2: image captioning (BLIP-base) ----
# Salesforce/blip-image-captioning-base (~990MB). Loaded once, frames streamed
# one at a time so RAM stays flat (~2-3GB peak) on an 8GB M2. Device "auto" picks
# Apple MPS when available, else CPU.
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
CAPTION_DEVICE = "auto"     # "auto" | "mps" | "cpu" | "cuda"
CAPTION_MAX_TOKENS = 30     # max_new_tokens per caption
CAPTION_BATCH = 1           # frames per forward pass; raise only if RAM allows
CAPTION_PROMPT = ""         # optional conditioning text (e.g. "a photo of"). Empty
                            # = unconditional captioning. Changing it re-captions.
# When a caption mostly echoes the frame's OCR text (a title card) it is a poor
# scene description; build_metadata flags such captions when their token overlap
# with the OCR text is at/above this Jaccard ratio.
CAPTION_TEXT_ECHO_JACCARD = 0.5

# ---- Milestone 2: semantic search (text embeddings) ----
# Each frame's searchable text (caption + OCR text) is embedded with a compact
# sentence-transformer so queries match by MEANING, not just substring -- e.g.
# "train" surfaces a caption that says "subway". all-MiniLM-L6-v2 is ~80MB and
# runs in seconds on the M2 (MPS/CPU). embed_text.py builds the index.
TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_DEVICE = "auto"      # "auto" | "mps" | "cpu" | "cuda"
SEMANTIC_TOP_K = 20            # default max results for a semantic query
SEMANTIC_MIN_SCORE = 0.20     # drop matches below this cosine similarity

# InsightFace model bundle (SCRFD detector + ArcFace recog + gender/age)
MODEL_PACK = "buffalo_l"


def ensure_dirs() -> None:
    for d in (VIDEO_DIR, FRAME_DIR, FACE_DIR, EMB_DIR,
              REPORT_DIR, MONTAGE_DIR, ANNOT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---- config-aware idempotency ----
# Parameters each stage's output depends on. run_pipeline stamps a fingerprint
# so a stage is rerun when its config changed, not only when its file is missing.
_STAGE_PARAMS = {
    "extract": ["FPS"],
    "appearance": ["APPEARANCE_ENABLE"],
    "detect": ["FPS", "DET_SIZE", "DET_THRESH", "MIN_FACE_PX", "MODEL_PACK",
               "PROVIDERS", "HQ_DET_SCORE", "MIN_BLUR_VAR", "MAX_NOSE_OFFSET",
               "TRACK_IOU", "TRACK_MAX_GAP", "TRACK_ACTIVATION", "TRACK_HIGH_CONF",
               "TRACK_FRAME_RATE", "TRACK_LINK_IOU"],
    "recognize": ["MIN_IDENTITY_FACES",
                  "REAL_FACE_DET", "TEMPLATE_USE_QUALITY_WEIGHT",
                  "TEMPLATE_MIN_FACES_FOR_TRIM", "TEMPLATE_OUTLIER_COS",
                  "CLUSTER_LINK_ENABLE", "CLUSTER_LINK_DIST",
                  "BEST_SHOT_ENABLE", "BEST_SHOT_K", "BEST_SHOT_DIST",
                  "APPEARANCE_ENABLE", "APPEARANCE_DIST", "APPEARANCE_FACE_DIST"],
    "ocr": ["FPS", "OCR_LANG", "OCR_PSM", "OCR_MIN_CONF", "OCR_MIN_TOKEN_LEN",
            "OCR_UPSCALE"],
    "caption": ["FPS", "CAPTION_MODEL", "CAPTION_MAX_TOKENS", "CAPTION_PROMPT"],
    "metadata": [],     # cheap join; run_pipeline rebuilds it every run (see note)
    "embed": ["TEXT_EMBED_MODEL"],
}


def stage_fingerprint(stage: str) -> str:
    g = globals()
    payload = {k: g[k] for k in _STAGE_PARAMS[stage]}
    return hashlib.md5(json.dumps(payload, sort_keys=True,
                                  default=str).encode()).hexdigest()[:12]
