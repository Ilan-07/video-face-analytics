# Technical Documentation — Video Face Analytics Pipeline

## 1. Overview
This system ingests a YouTube video and produces, end to end: extracted frames,
detected face crops, grouped unique identities, and an occurrence report that
names the most frequently appearing person. The pipeline is split into five
standalone, idempotent stages orchestrated by `run_pipeline.py`.

Dataset: `https://www.youtube.com/watch?v=d2g9HlwoC-s`

## 2. Architecture
```
download.py → extract_frames.py → detect_faces.py → recognize.py → analytics.py
   video/        frames/+csv         faces/+emb        identities      reports/
```
Each stage writes a CSV/artifact consumed by the next, so any stage can be rerun
in isolation and intermediate results are inspectable.

## 3. Frame extraction
Frames are sampled at **~2 FPS** using **supervision**'s video utilities:
`sv.VideoInfo` reads the true source FPS and `sv.get_video_frames_generator`
yields frames at a computed `stride = round(source_fps / target_fps)`. Each saved
frame's timestamp is derived from its real source index
(`timestamp = sampled_index × stride / source_fps`), which is more accurate than
assuming an exact target rate. The mapping is written to `data/frames.csv`
(`frame_id, filename, timestamp_sec`).

## 4. Face detection model
**Model:** SCRFD ("Sample and Computation Redistribution for Efficient Face
Detection"), supplied by InsightFace in the `buffalo_l` bundle.

**Why SCRFD:** It is a fast single-shot detector that returns a bounding box, a
confidence score, **and 5 facial landmarks** (eyes, nose, mouth corners). The
landmarks are essential — they drive similarity-transform **alignment** to a
canonical 112×112 crop, which is the largest single accuracy lever for the
downstream recognition model. SCRFD outperforms classic OpenCV Haar cascades and
the OpenCV DNN (ResNet-SSD) detector on small, off-angle, and crowded faces.

**Parameters:** `det_size=640×640`, `det_thresh=0.5`, and a `MIN_FACE_PX=40`
filter that drops tiny/blurry detections before they can pollute clustering.

## 5. Face recognition / embedding model
**Model:** ArcFace (ResNet-100 backbone, `glint360k` training), also from the
InsightFace `buffalo_l` bundle.

**Why ArcFace:** It is trained with an additive angular-margin loss, so each face
maps to a **512-dimensional L2-normalized embedding** in which **cosine distance
corresponds to identity similarity**. This well-characterized geometry makes the
clustering threshold easy to reason about. Bundling detection and recognition in
one package keeps the dependency surface small.

**Process:** aligned crop → ArcFace → 512-d unit vector, stored per face in
`data/embeddings/embeddings.npz` keyed by `crop_id`.

## 6. Face grouping methodology
**Algorithm:** DBSCAN with `metric="cosine"`, `eps=0.40`, `min_samples=3` over the
L2-normalized embeddings.

**Why DBSCAN:**
- The number of distinct people is **unknown a priori**, ruling out K-Means.
- It produces an explicit **noise label (−1)**, so blurry frames and false
  positives become "unknown" rather than corrupting real identities.
- Density-based grouping naturally fits the data: the same person across
  consecutive seconds yields many tightly clustered embeddings.

**Identity assignment:** clusters are ranked by size and labelled `Face_01`,
`Face_02`, … (largest first). For each identity the **medoid** (embedding nearest
the cluster centroid) is saved as the representative thumbnail.

**Calibration:** `eps` is the sensitive knob — too small fragments one person into
several IDs, too large merges different people. It is exposed in `config.py` and
can be validated against a pairwise cosine-distance histogram.

## 7. Alternative approaches considered
| Component | Alternative | Why not chosen |
|---|---|---|
| Detection | OpenCV Haar cascade | High false-positive rate; weak on profiles/small faces; no landmarks. |
| Detection | OpenCV DNN / MTCNN | Lower recall than SCRFD; MTCNN slower. |
| Embedding | dlib / `face_recognition` (128-d) | Lower accuracy; dlib hard to build on recent Python. |
| Embedding | FaceNet (512-d) | Comparable, but separate ecosystem; InsightFace bundles detect+recog. |
| Clustering | K-Means | Needs known K; cannot express "unknown". |
| Clustering | Agglomerative / Chinese Whispers | Viable fallback; kept in reserve if DBSCAN under/over-segments. |
| Frames | OpenCV VideoCapture loop | Works, but ffmpeg is faster and simpler. |

## 8. Challenges encountered & mitigations
- **Python 3.14 wheel gaps** for insightface/onnxruntime → pinned a **Python 3.12
  virtualenv** for reliable binary wheels.
- **Near-duplicate frames at 1 FPS** inflate a single person's count → counts are
  defined and reported as per-frame appearances (documented, dedup optional).
- **Clustering threshold sensitivity** → `eps`/`min_samples` exposed in config;
  histogram-based calibration.
- **Over-segmentation vs. merging** (lighting/pose changes) → tune `eps`;
  agglomerative fallback available.
- **Non-faces / motion blur / logos** → removed via `det_score` + `MIN_FACE_PX`;
  residuals fall into DBSCAN noise.
- **CPU-only inference** → `CPUExecutionProvider` with batched embedding; GPU
  optional via a different onnxruntime provider.

## 9. Deliverables
- Frame-extraction, detection, recognition/grouping, and analytics pipelines.
- Occurrence statistics and most-frequent-face analysis.
- This technical documentation, full source, and sample montage visualizations.

## 10. Expected output
- **Total extracted frames** — rows in `frames.csv` (≈ video length × FPS).
- **Total unique faces** — number of track clusters.
- **Face occurrence statistics** — `reports/summary.{json,md}` + `report.html` per identity.
- **Most frequently appearing face** — top identity by screen time, with thumbnail.
- **Sample visualizations** — montages, annotated frames, appearance timeline, contact sheet.

## 11. Improvements over the first version
The pipeline was hardened with seven changes addressing accuracy, correctness,
and engineering robustness:

1. **Temporal tracking (ByteTrack via the `trackers` library).** Detections are
   linked across frames by **ByteTrack** (Kalman motion model + two-stage
   high/low-confidence IoU association) — more robust than a hand-rolled IoU
   tracker through brief misses and crowding. ByteTrack returns `tracker_id == -1`
   on a track's birth frame; a one-step forward-IoU **reconciliation** in
   `detect_faces.py` folds that frame back into its track so no appearance is
   fragmented, and genuinely single-frame faces become their own singletons.
   `frame_rate` is fixed at 30 so ByteTrack's buffer scaling
   (`int(frame_rate/30 × lost_track_buffer)`) is not collapsed by the ~2 FPS
   sampling. Clustering then runs on **per-track template embeddings** (mean of a
   track's good-quality faces), and `min_samples=1` ensures every track receives
   an identity — together these eliminated the previous ~51% "unclustered" rate.

   *Note on libraries:* tracking, the annotators, and frame extraction come from
   **roboflow/supervision** and its companion **`trackers`** package
   (`sv.ByteTrack` itself is deprecated in supervision ≥0.28, so the maintained
   `trackers.ByteTrackTracker` is used). Face recognition (ArcFace) and clustering
   (DBSCAN) are not in supervision and remain InsightFace + scikit-learn.

2. **Meaningful occurrence metrics.** Instead of raw per-frame counts, analytics
   reports **distinct appearances** (number of tracks) and **screen-time in
   seconds** (summed track durations), plus the legacy frame count. The
   most-frequent face is ranked by screen time.

3. **Evaluation harness (`eval.py`).** A DBSCAN **`eps` sweep** with **silhouette
   scores** (cosine) makes the grouping threshold a measured choice, and a
   **contact sheet** of every identity supports fast manual review.

4. **Higher sampling rate.** Frame extraction moved to **2 FPS**, catching
   sub-second appearances and giving finer screen-time resolution.

5. **Quality gating + demographics.** Each face gets a **blur** score (Laplacian
   variance) and a **pose** score (nose offset from landmarks); only
   high-confidence, sharp, frontal faces contribute to a track's template. The
   already-loaded **gender/age** model is now surfaced per identity.

6. **Richer reporting.** Timestamped montages, **annotated sample frames**
   (boxes + Face IDs), a Gantt-style **appearance timeline**, and a single-page
   **`report.html`** alongside the JSON/Markdown summaries.

7. **Engineering robustness.** Console+file **logging** (`reports/pipeline.log`),
   **per-frame error recovery** so one bad frame can't abort a run, a
   configurable **GPU/CPU** backend, a pinned `requirements.lock.txt`, and a
   **pytest** suite (`tests/test_pipeline.py`, 17 tests) covering the pure logic:
   IoU geometry, pose/blur scoring, the ByteTrack birth-frame reconciliation, and
   the evaluation metrics.

## 12. Accuracy validation & false-positive backstop
Two additions turn grouping quality from "looks right" into a measured number.

**False-positive backstop (`recognize.py`, `MIN_IDENTITY_FACES`).** A cluster
becomes a real identity only if it is *corroborated*: at least
`MIN_IDENTITY_FACES` (=2) faces **or** at least one high-quality face
(`quality_ok`). Otherwise its tracks are relabeled **`unknown`** and excluded
from analytics. On this video the backstop dropped 19 lone low-quality clusters
(blurry/profile/non-face passersby), tightening 53 → **34** identities.

**Ground-truth evaluation (`make_labelsheet.py` + `eval_labeled.py`).**
`make_labelsheet.py` renders one representative crop per track and writes
`data/ground_truth.csv` (a `true_id` column for a human to fill, `x` =
non-face/unusable). `eval_labeled.py` then scores the predicted clustering
against those labels with standard external metrics — **Adjusted Rand Index,
homogeneity, completeness, V-measure** — plus **pairwise precision/recall/F1**.
Tracks labeled `x` are excluded; tracks predicted `unknown` are treated as
singletons so the backstop is neither rewarded nor punished.

*Result on a 48-track / 14-person labeled sample:* ARI, homogeneity,
completeness, V-measure and pairwise F1 all **1.0** (133 same-person pairs, 0
errors). **Caveat:** this sample is dominated by within-scene tracks (faces a
fraction of a second apart, where embeddings are near-identical and grouping is
easy), and deliberately excludes the ambiguous cross-shot case (the same
light-haired interviewee at 0:55 vs 1:30). So the headline 1.0 confirms
within-scene grouping is essentially perfect but **under-tests the hard case** —
the same person recurring across different scenes/lighting/pose. A larger,
cross-scene-focused labeling pass is the next step to stress that.

**Label-free objective check (`eval_cooccurrence.py`).** Two faces in the *same
frame* must be different people, so every co-occurring pair is a known
"cannot-link" constraint — an objective false-merge test over the entire video,
no labels required. Result: **43 cannot-link pairs, 0 false merges
(precision 1.0000)**. The clustering never merges two people who appear together.
This runs automatically as part of `eval.py`.

**Cross-scene stress test (findings).** Investigating the hard case surfaced that
this video has very few genuinely recurring people: only one identity (the older
man, `Face_02`) spans a large window (~110 s) and it is internally consistent.
The apparent anomaly of "9 identities active within 1039–1061 s" was shown by
co-occurrence to be a **crowd scene** (e.g. frame 2106 contains four of them at
once), correctly separated into distinct identities. A similar check on two
look-alike bald men (`Face_04`/`Face_12`, ~835 s) confirmed they co-occur and
were correctly kept apart. So on the available cross-scene evidence the
clustering shows **no false merges and no observed false splits**; the residual
risk is recurring people that never co-occur, which only broader labeling can
fully rule out.
