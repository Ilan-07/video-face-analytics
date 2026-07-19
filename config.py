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
TRANSCRIPT_JSON = DATA / "transcript.json"
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

# ---- New capability: face super-resolution before embedding (opt-in) ----
# Identity fragmentation is driven by faces near the detection floor: ArcFace
# embeddings from ~45px crops are noisy, so one person splits across IDs. When
# enabled, faces smaller than SR_MIN_PX have their crop super-resolved (OpenCV
# dnn_superres) before the ArcFace embedding is recomputed, aiming to make small
# faces cluster with their larger appearances. OFF by default: it changes every
# grouping number and needs a model file, so enabling it reruns detection
# (fingerprinted) and should be followed by re-running the eval harnesses. Any
# SR error falls back to the normal embedding, so it can never break detection.
SR_ENABLE = False
SR_MIN_PX = 60                       # only super-resolve faces smaller than this
SR_SCALE = 4                         # dnn_superres scale (2|3|4); must match model
SR_MODEL = "fsrcnn"                  # "fsrcnn" (tiny) | "espcn" | "edsr"
SR_MODEL_PATH = ROOT / "models" / "FSRCNN_x4.pb"   # provide this file to enable

# ---- Fix #7: compute backend ----
# Detection (SCRFD) provider. Kept a knob rather than hardcoded so the backend
# can be changed without editing get_app(), but the default is deliberate:
#   cpu    - fastest on Apple Silicon for THIS model. Measured best here.
#   coreml - ~3-4x SLOWER for SCRFD: its dynamic input shapes (det [1,3,'?','?'])
#            defeat ANE compilation and bounce ops back to CPU. Do not default to
#            it on Mac; left available only for re-measuring if a future ORT fixes it.
#   cuda   - only on a real NVIDIA box.
DET_PROVIDER = "cpu"                   # "cpu" | "coreml" | "cuda"
_PROVIDER_LISTS = {
    "cpu": ["CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}
USE_GPU = DET_PROVIDER == "cuda"       # kept for callers that branch on it
CTX_ID = 0 if USE_GPU else -1
PROVIDERS = _PROVIDER_LISTS[DET_PROVIDER]

# ---- Fix #6: reporting ----
MAX_ANNOTATED_FRAMES = 8
TIMELINE_W = 1100
TIMELINE_LANE_H = 26

# ---- New capability: speech transcription (faster-whisper) ----
# Vision-only was a scope ceiling: spoken content was invisible. This stage adds
# an audio channel that build_metadata joins on timestamp exactly like OCR. Kept
# CPU/int8: ctranslate2 (faster-whisper's backend) has no MPS path, and int8 on
# CPU is the practical Apple-Silicon choice. Optional -- absent faster-whisper or
# audio, the stage skips and the pipeline runs unchanged.
WHISPER_ENABLE = True         # set False to skip transcription entirely
WHISPER_MODEL = "base.en"     # tiny.en|base.en|small.en|medium.en|large-v3
WHISPER_DEVICE = "cpu"        # ctranslate2 backend: "cpu" | "cuda" (no MPS)
WHISPER_COMPUTE = "int8"      # "int8" | "int8_float16" | "float32"
# Hallucination control, tuned against this footage. Without help, Whisper emits
# "You" over silence and loops a mis-heard phrase over minutes of train noise.
# Measured here: Silero VAD is too aggressive -- it drops the faint platform PA
# along with the noise (0 segments) -- so it is OFF by default and left as an
# option for cleaner audio. What works instead: disable context-conditioning to
# break repetition loops, then drop pure-filler segments and any text that repeats
# implausibly often (a real announcement varies; a hallucination loops verbatim).
WHISPER_VAD = False           # Silero voice-activity filter; too aggressive here
# Left ON (Whisper's default): measured, it makes hallucinations loop the SAME
# phrase verbatim, which clean_segments removes cleanly by repeat count. Turning
# it off scattered the hallucinations into varied one-offs that no filter catches
# without also dropping real speech -- worse, not better.
WHISPER_CONDITION_ON_PREV = True
WHISPER_MAX_REPEAT = 6        # drop any transcript text repeated >= this many times
WHISPER_FILLER = {"you", "thank you", "thanks for watching", "bye", "the end",
                  "thank you.", "you.", "."}   # canonical silence-hallucinations

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

# Parallelism: frames are independent, and Tesseract is CPU-bound, so OCR fans out
# across cores. Each worker pins Tesseract to one thread to avoid oversubscription.
# Purely a speed knob -- not fingerprinted, does not change output.
import os as _os
OCR_WORKERS = min(8, (_os.cpu_count() or 2))   # 1 = serial (used by tests)
# Optional recall/speed trade: skip Tesseract on frames whose edge density is below
# OCR_TEXT_EDGE_MIN (unlikely to hold signage). OFF by default -- it trades recall,
# and OCR's whole value here is not missing platform text. Opt-in and re-check
# eval_search OCR recall before trusting it.
OCR_SKIP_LOW_TEXT = False
OCR_TEXT_EDGE_MIN = 0.010     # mean Canny-edge fraction below which a frame is skipped

# ---- Milestone 2: domain lexicon correction (clean noisy OCR) ----
# Tesseract is *confidently* wrong on the stylised Underground signage in this
# video ("Metropolite", "Victoria tine", "Southbo", "hotders"). Because the
# on-screen text is a bounded domain (tube lines, directions, stations, the
# promo copy), a post-OCR pass snaps each kept token to its nearest lexicon
# entry by edit-distance ratio. Precision-first: a token is only rewritten when
# it is long enough, is not already a valid English word in OCR_LEXICON_STOP,
# and clears OCR_LEXICON_CUTOFF -- so we fix typos without mangling real words.
OCR_LEXICON_ENABLE = True
OCR_LEXICON_CUTOFF = 0.75   # min difflib ratio (0-1) to accept a correction
OCR_LEXICON_MIN_LEN = 4     # only correct tokens with >= this many letters
OCR_LEXICON = [
    # tube lines
    "Bakerloo", "Central", "Circle", "District", "Hammersmith", "Jubilee",
    "Metropolitan", "Northern", "Piccadilly", "Victoria", "Waterloo",
    "Elizabeth", "Overground",
    # directions / platform vocabulary
    "Northbound", "Southbound", "Eastbound", "Westbound", "Platform", "Line",
    "Lines", "Underground", "Station", "Exit", "Rail", "Ticket", "Holders",
    "Only", "Way", "Out",
    # promo / signage copy seen in this video
    "London", "Find", "Score", "Credit", "Experian", "Extravaganza", "Forever",
    "Free", "Now", "Massage", "Massages",
    # station names seen
    "Charing", "Cross", "Walthamstow", "Stepney", "Green", "Finchley",
    "Finsbury", "Blackhorse", "Bethnal", "City", "Road", "Embankment",
    "Tuesday", "November",
]
# real English words that fuzzily collide with the lexicon -- never rewrite these
OCR_LEXICON_STOP = {
    "core", "water", "baker", "fore", "wine", "fine", "time", "more", "over",
    "fire", "care", "bore", "tide", "note", "none", "site", "side", "case",
}

# ---- Milestone 2: image captioning (BLIP-base) ----
# Salesforce/blip-image-captioning-base (~990MB). Loaded once, frames streamed
# one at a time so RAM stays flat (~2-3GB peak) on an 8GB M2. Device "auto" picks
# Apple MPS when available, else CPU.
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
CAPTION_DEVICE = "auto"     # "auto" | "mps" | "cpu" | "cuda"
CAPTION_MAX_TOKENS = 30     # max_new_tokens per caption
CAPTION_BATCH = 8           # frames per batched forward pass. BLIP-base at this
                            # batch is markedly faster on MPS than one-at-a-time and
                            # fits in ~4-5GB; drop to 1-2 on an 8GB machine if it swaps.
CAPTION_PROMPT = ""         # optional conditioning text (e.g. "a photo of"). Empty
                            # = unconditional captioning. Changing it re-captions.
# When a caption mostly echoes the frame's OCR text (a title card) it is a poor
# scene description; build_metadata flags such captions when their token overlap
# with the OCR text is at/above this Jaccard ratio.
CAPTION_TEXT_ECHO_JACCARD = 0.5
# Echo repair: after the first pass, frames whose caption echoes the OCR text are
# re-captioned with CAPTION_RECAPTION_PROMPT (conditioning BLIP toward describing
# the scene instead of reading the text). If the re-caption STILL echoes -- a pure
# title card with no scene to describe -- we fall back to the clean OCR text, so
# the search document carries the real signal instead of a garbled transcription.
CAPTION_ECHO_FIX = True
CAPTION_RECAPTION_PROMPT = "a photo of"

# ---- Milestone 2: semantic search (text embeddings) ----
# Each frame's searchable text (caption + OCR text) is embedded with a compact
# sentence-transformer so queries match by MEANING, not just substring -- e.g.
# "train" surfaces a caption that says "subway". all-MiniLM-L6-v2 is ~80MB and
# runs in seconds on the M2 (MPS/CPU). embed_text.py builds the index.
TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_DEVICE = "auto"      # "auto" | "mps" | "cpu" | "cuda"
SEMANTIC_TOP_K = 20            # default max results for a semantic query
SEMANTIC_MIN_SCORE = 0.20     # drop matches below this cosine similarity

# ---- Milestone 2: visual (CLIP) search ----
# A second, image-native search path so retrieval is NOT hostage to caption
# quality. CLIP embeds every frame IMAGE and the query TEXT into one shared space
# (embed_image.py builds the index), so "a tunnel" can match a tunnel frame even
# when its caption never says "tunnel". clip-ViT-B-32 (~600MB) via
# sentence-transformers encodes both images and text. CLIP cosine similarities run
# lower than text-to-text, hence the smaller default floor.
CLIP_MODEL = "clip-ViT-B-32"
CLIP_DEVICE = "auto"           # "auto" | "mps" | "cpu" | "cuda"
IMAGE_EMB_FILE = EMB_DIR / "image_embeddings.npz"
VISUAL_TOP_K = 20              # default max results for a visual query
VISUAL_MIN_SCORE = 0.20        # drop matches below this CLIP cosine similarity

# ---- Reciprocal Rank Fusion (semantic + visual) ----
FUSION_RRF_K = 60             # RRF damping; 60 is the canonical Cormack et al. value
FUSION_POOL = 50             # candidate depth pulled from each ranker before fusing

# ---- Milestone 2: search-quality evaluation ----
# make_search_labelsheet.py samples frames into SEARCH_LABELS_CSV for a human to
# fill (true OCR text + a 1-5 caption-adequacy score); eval_search.py scores OCR
# precision/recall, mean caption adequacy, and semantic precision@k over the
# curated query set in SEARCH_QUERIES_JSON. See those modules for the schema.
SEARCH_LABELS_CSV = DATA / "search_labels.csv"
SEARCH_QUERIES_JSON = DATA / "search_queries.json"
SEARCH_LABEL_SAMPLE = 30      # frames sampled into the OCR/caption labelsheet
OCR_MATCH_JACCARD = 0.5       # token-Jaccard >= this counts a predicted OCR
                              # string as matching the human-labeled true text
SEARCH_EVAL_K = 5             # k for semantic precision@k

# ---- Milestone 3: scene segmentation ----
# The captions cannot segment this video: 1415 frames yield only 342 unique
# caption strings and BLIP flickers between synonyms on visually identical
# frames ("a train is pulling passengers" x72), so consecutive-run grouping
# gives 904 fragments -- noise, not scene cuts. Instead we cut on adjacent-frame
# CLIP cosine (image_embeddings.npz, already L2-normalized, so a plain dot
# product). Measured on this video: mean 0.959, p5 0.858, p1 0.491 -- the ten
# sharpest cuts land exactly on the Tube-line title cards, so the visual and
# textual signals agree independently.
SCENES_JSON = DATA / "scenes.json"
SCENE_SIM_THRESH = 0.70   # adjacent cosine below this = cut. Swept 0.60/0.70/
                          # 0.75/0.80 -> 26/36/39/40 scenes. 0.60 is too coarse:
                          # it swallows the station-name sign that opens each
                          # line (Embankment, Temple, Finsbury Park...), which is
                          # the video's actual editorial beat. 0.75+ buys only a
                          # few more scenes and starts cutting on camera shake.
SCENE_MIN_SEC = 4.0       # scenes shorter than this merge into the previous one.
                          # A safety net that currently never fires: every short
                          # scene is a protected title-card or station-sign beat.
SCENE_TITLE_CARD_FORCE = True   # always cut at a title card, whatever the cosine

# In-carriage advertising is genuinely on screen but tells a story about adverts,
# not about the Underground. scenes.is_signage() drops any OCR string containing
# one of these, so the narrator never reads "Now FREE FOREVER" as a station name.
# Distinct from OCR_LEXICON_STOP, which guards the M2 fuzzy-correction pass.
SCENE_OCR_STOPWORDS = {
    "experian", "credit", "score", "free", "forever", "find", "now",
    "massage", "massages", "ticket", "holders", "only", "awesome",
}

# ---- Milestone 3: narration (OpenRouter / Gemma 4) ----
# gemma-4-31b-it has a 262k context, so all ~30 scene digests fit in ONE prompt
# and chronological coherence is structural rather than stitched from chunks.
# Apache-2.0, and the ":free" variant costs nothing (20 RPM / 50 req-day without
# credits). It also takes image input, which describe_scenes.py uses to build an
# independent reference for eval_story.py's grounding metric.
LLM_CACHE_DIR = DATA / "llm_cache"
SCENE_DESC_JSON = DATA / "scene_descriptions.json"
STORY_JSON = DATA / "story.json"
TIMELINE_JSON = DATA / "timeline.json"

NARRATE_MODEL = "google/gemma-4-31b-it:free"
NARRATE_BASE_URL = "https://openrouter.ai/api/v1"
# Local OpenAI-compatible fallback (e.g. Ollama). When set, narration falls back
# to it if OpenRouter has no key or its free tier is exhausted by 429s, so the
# stage degrades to local generation instead of failing. Empty = no fallback.
#   e.g. NARRATE_FALLBACK_BASE_URL = "http://localhost:11434/v1"
#        NARRATE_FALLBACK_MODEL    = "llava"   (must be vision-capable for describe)
NARRATE_FALLBACK_BASE_URL = ""
NARRATE_FALLBACK_MODEL = ""
NARRATE_TEMPERATURE = 0.0   # deterministic: the prompt comparison must reproduce
NARRATE_SEED = 0
NARRATE_MAX_TOKENS = 4096   # the ":free" variant caps completions at 8192
NARRATE_TIMEOUT = 120       # seconds per request
NARRATE_MAX_RETRIES = 20    # exponential backoff on HTTP 429. Measured on the
NARRATE_RETRY_MAX_DELAY = 60  # ":free" pool: a big text request succeeds ~1 try in
                              # 3, so 20 attempts fail ~0.03% of the time. These
                              # 429s are upstream capacity ("temporarily
                              # rate-limited upstream"), NOT our 50/day quota, so
                              # retrying costs nothing. Two providers back the
                              # model: OpenInference (text only) and Google AI
                              # Studio (the only one that accepts images, and the
                              # one every free user's vision request queues for).
NARRATE_IMAGE_MAX_SIDE = 896  # downscale keyframes before base64. Gemma 4 resizes
                              # to 896px internally, so sending 1280px originals
                              # (318KB each) only inflates the payload -- 6 images
                              # went 2.6MB -> ~0.4MB, which also makes the request
                              # far likelier to survive a congested free endpoint.
NARRATE_CACHE = True        # replay data/llm_cache/*.json instead of re-calling

# Grounding-verification pass: after generation, drop prose sentences whose
# content words are not attested in the scene digest the story was written from,
# converting the measured grounding metric into an enforced floor. OFF by default
# -- it rewrites the committed stories and is a lossy edit -- so it is opt-in and
# fingerprinted, and only condemns sentences with enough words to judge fairly.
NARRATE_VERIFY = False
NARRATE_VERIFY_MIN = 0.20      # drop a sentence grounding below this fraction
NARRATE_VERIFY_MIN_WORDS = 4   # ...but only if it has >= this many content words
NARRATE_VLM_ENABLE = True   # keyframe re-captioning (ablation + eval reference)
NARRATE_VLM_BATCH = 6       # keyframes per vision request. Gemma 4 accepts many
                            # images per message, and the ":free" tier allows only
                            # 50 requests/day: batching 24 keyframes 6-at-a-time
                            # costs 4 requests instead of 24, so the whole
                            # milestone runs in 11 and leaves room to iterate.
                            # 6 keyframes ~= 2.6MB of base64 -- comfortably inside
                            # the 262k context. Larger batches risk the model
                            # losing track of which description belongs to which
                            # image, which the scene_index contract then catches.

# Prompt-engineering comparison (Task 3). Each strategy generates its own story
# into reports/story_<strategy>.md; eval_story.py scores them side by side.
STORY_STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought", "structured_role"]
STORY_STRATEGY = "structured_role"   # the one promoted to data/story.json

# ---- Milestone 3: story evaluation ----
# eval_story.py scores each strategy on chronology (are cited timestamps
# monotonic?), grounding (are content words attested in the scene digests /
# VLM descriptions?), chapter coverage, and redundancy.
STORY_EVAL_NGRAM = 3          # n for the distinct-n-gram redundancy ratio
STORY_GROUND_MIN_LEN = 4      # ignore content words shorter than this
VIDEO_DURATION_SEC = 1415.5   # upper bound for timeline timestamp validation

# ---- Milestone 4: end-to-end integration and system evaluation ----
# run_pipeline times every stage through util.time_stage and merges the result
# into STAGE_TIMINGS_JSON. The merge matters: the pipeline is idempotent, so a
# second run skips almost every stage and would otherwise erase the cold-run
# cost that is the only honest answer to "how long does this system take?".
# Each stage therefore keeps its last MEASURED duration until it actually reruns.
STAGE_TIMINGS_JSON = DATA / "stage_timings.json"
SYSTEM_EVAL_JSON = REPORT_DIR / "eval_system.json"
SYSTEM_EVAL_MD = REPORT_DIR / "eval_system.md"

# Stages that call a network LLM. Their wall-clock is dominated by OpenRouter
# free-tier queueing and 429 backoff, not by our compute, so eval_system reports
# them separately -- averaging them into a per-frame throughput number would
# describe someone else's rate limiter rather than this pipeline.
NETWORK_STAGES = {"describe", "narrate"}

# InsightFace model bundle (SCRFD detector + ArcFace recog + gender/age)
MODEL_PACK = "buffalo_l"


def ensure_dirs() -> None:
    for d in (VIDEO_DIR, FRAME_DIR, FACE_DIR, EMB_DIR,
              REPORT_DIR, MONTAGE_DIR, ANNOT_DIR, LLM_CACHE_DIR):
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
               "TRACK_FRAME_RATE", "TRACK_LINK_IOU",
               "SR_ENABLE", "SR_MIN_PX", "SR_SCALE", "SR_MODEL"],
    "recognize": ["MIN_IDENTITY_FACES",
                  "REAL_FACE_DET", "TEMPLATE_USE_QUALITY_WEIGHT",
                  "TEMPLATE_MIN_FACES_FOR_TRIM", "TEMPLATE_OUTLIER_COS",
                  "CLUSTER_LINK_ENABLE", "CLUSTER_LINK_DIST",
                  "BEST_SHOT_ENABLE", "BEST_SHOT_K", "BEST_SHOT_DIST",
                  "APPEARANCE_ENABLE", "APPEARANCE_DIST", "APPEARANCE_FACE_DIST"],
    "ocr": ["FPS", "OCR_LANG", "OCR_PSM", "OCR_MIN_CONF", "OCR_MIN_TOKEN_LEN",
            "OCR_UPSCALE", "OCR_LEXICON_ENABLE", "OCR_LEXICON_CUTOFF",
            "OCR_LEXICON_MIN_LEN", "OCR_LEXICON"],
    "transcribe": ["WHISPER_ENABLE", "WHISPER_MODEL", "WHISPER_VAD",
                   "WHISPER_CONDITION_ON_PREV"],
    "caption": ["FPS", "CAPTION_MODEL", "CAPTION_MAX_TOKENS", "CAPTION_PROMPT",
                "CAPTION_ECHO_FIX", "CAPTION_RECAPTION_PROMPT",
                "CAPTION_TEXT_ECHO_JACCARD"],
    "metadata": [],     # cheap join; run_pipeline rebuilds it every run (see note)
    "embed": ["TEXT_EMBED_MODEL"],
    "embed_image": ["FPS", "CLIP_MODEL"],
    "scenes": ["FPS", "CLIP_MODEL", "SCENE_SIM_THRESH", "SCENE_MIN_SEC",
               "SCENE_TITLE_CARD_FORCE"],
    "describe": ["NARRATE_MODEL", "NARRATE_VLM_ENABLE", "NARRATE_VLM_BATCH",
                 "SCENE_SIM_THRESH", "SCENE_MIN_SEC", "SCENE_TITLE_CARD_FORCE"],
    "narrate": ["NARRATE_MODEL", "NARRATE_TEMPERATURE", "NARRATE_SEED",
                "NARRATE_MAX_TOKENS", "STORY_STRATEGIES", "STORY_STRATEGY",
                "SCENE_SIM_THRESH", "SCENE_MIN_SEC",
                "NARRATE_VERIFY", "NARRATE_VERIFY_MIN",
                "NARRATE_VERIFY_MIN_WORDS"],
}


def stage_fingerprint(stage: str) -> str:
    g = globals()
    payload = {k: g[k] for k in _STAGE_PARAMS[stage]}
    return hashlib.md5(json.dumps(payload, sort_keys=True,
                                  default=str).encode()).hexdigest()[:12]
