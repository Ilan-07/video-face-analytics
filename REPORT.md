# Video Processing & Face Analysis — Technical Report

**Video:** https://www.youtube.com/watch?v=d2g9HlwoC-s
**Goal:** extract frames, detect faces, group the same person together, and report
how often each person appears.

---

## 1. Models Used

### 1.1 Face Detection
**SCRFD-10GF** (`det_10g.onnx`), from the InsightFace **`buffalo_l`** pack, run via
ONNX Runtime. SCRFD is a single-shot, anchor-based detector (CVPR 2021) that
returns bounding boxes, 5-point landmarks, and a confidence score per face.
Configuration: input `det_size = 1024×1024`, `det_thresh = 0.4`, and a
`min_face = 36 px` post-filter. It is markedly more robust to pose, lighting, and
scale than a classic OpenCV Haar cascade.

### 1.2 Face Recognition / Embedding
**ArcFace** (`w600k_r50.onnx`, ResNet-50 trained on WebFace600K), also from
`buffalo_l`. Each detected face is aligned to a 112×112 crop using the landmarks
(`norm_crop`) and embedded into a **512-D L2-normalized vector**. Same-person faces
produce vectors with high cosine similarity; different people, low. Gender/age come
from the pack's `genderage` model (reported only on high-quality crops).

---

## 2. Face Grouping Methodology

Grouping is done at the **track level**, not per raw crop, which is the key to
clean identities:

1. **Track & template.** Faces are tracked across frames with **ByteTrack**; each
   track is reduced to one **quality-weighted, outlier-trimmed mean embedding**
   (its "template"). This averages out per-frame noise.
2. **Constrained complete-linkage clustering.** Track templates are agglomerated
   with **complete linkage** under **cosine distance** (every member pair must be
   within the threshold), plus a **co-occurrence cannot-link** constraint: two
   faces in the *same frame* are different people and can never merge. Complete
   linkage (vs. single-linkage / DBSCAN) forbids "chaining" dissimilar faces into
   one incoherent cluster.
3. **Best-shot cross-scene linking.** A second pass merges clusters whose *most
   frontal, highest-quality* faces match — consolidating a person's frontal
   appearances across scene cuts that pose-polluted means miss.
4. **Corroboration backstop.** A cluster becomes a real identity only if it has
   ≥2 faces, a high-quality face, or a confidently-detected face — dropping lone
   low-confidence non-faces (signage, blur) from the count.
5. **Identity assignment.** Each surviving cluster is named `Face_NN`, ranked by
   screen-time.

An **optional clothing/body re-ID** stage (off by default) adds an orthogonal
cross-scene signal for people whose face pose varies too much for face embeddings
alone; it is gated behind a config flag and a human review step (see §6).

---

## 3. Alternative Approaches Considered

| Approach | Why not used |
|----------|--------------|
| OpenCV Haar cascade (detection) | Weak on side/blurred/small faces. |
| MTCNN / RetinaFace (detection) | Fine, but SCRFD in `buffalo_l` is faster and bundled with the matching recognizer. |
| Dlib / `face_recognition` embeddings | Older; ArcFace generalizes better. |
| FaceNet512 embeddings | Benchmarks below modern ArcFace; would add a heavy TF dependency and require re-tuning. |
| **K-Means** clustering | Needs the number of people up front — unknown here. |
| **DBSCAN** (single-linkage) clustering | Tried first; its single-linkage *chains* marginal faces into incoherent "junk" identities. Replaced by complete-linkage. |
| Raise frame rate to link scenes | Spec mandates 1 FPS; and scene **cuts** have no intermediate frames at any rate. |

---

## 4. Challenges Encountered

- **Junk clusters from chaining.** Naïve DBSCAN merged orthogonal faces (one
  cluster had within-cluster cosine ≈ 0.04 — clearly different people). Fixed with
  complete-linkage, which *guarantees* cluster coherence.
- **Over-segmentation at 1 FPS.** The same person across scene cuts has no temporal
  bridge, so their frontal/profile appearances split into separate clusters. The
  best-shot pass recovers the frontal cases; cross-cut *profile* recurrence remains
  the residual limit.
- **Validation without labels.** We use two label-free checks: **co-occurrence
  precision** (same-frame faces must differ) and a **tracking-continuity recall**
  metric (briefly-broken tracks at the same location must regroup). Co-occurrence
  is blind to cross-cut merges, so cluster visual review remains necessary.
- **Quality vs. coverage.** Lower detection thresholds catch more faces but feed
  noise to clustering; the corroboration backstop and quality-weighted templates
  manage this trade-off.

---

## 5. Deliverables

| Deliverable | File(s) |
|-------------|---------|
| Frame Extraction Pipeline | `download.py`, `extract_frames.py` → `data/frames/`, `data/frames.csv` |
| Face Detection Pipeline | `detect_faces.py` → `data/faces/`, `data/faces.csv`, `data/embeddings/` |
| Face Recognition & Grouping Pipeline | `recognize.py` → `data/identities.csv` |
| Face Occurrence Statistics | `analytics.py` → `reports/summary.{json,md}`, `report.html` |
| Most Frequent Face Analysis | `reports/report.html`, `reports/Face_01_rep.jpg` |
| Technical Documentation | this report + `DOCUMENTATION.md` |
| Source Code | all `*.py` (orchestrated by `run_pipeline.py`); `pytest` suite (27 tests) |
| Sample visualizations | `reports/montages/Face_*.png`, `reports/contact_sheet.png` |
| Evaluation harness | `eval.py`, `eval_cooccurrence.py`, `eval_continuity.py`, `eval_labeled.py` |
| Optional cross-scene re-ID + review | `appearance.py`, `review_merges.py`, `apply_merges.py` |

**Run:** `python run_pipeline.py` (idempotent, stage-by-stage).

---

## 6. Expected Output

| Output Item | Result |
|-------------|--------|
| Total extracted frames | **1415** (at 1 FPS, 1080p source) |
| Total detected face crops | 435 |
| Total face groups | 154 |
| **Unique people — featured cast** (≥5 frames present) | **21** |
| Most frequently appearing face | **Face_01** |
| Occurrences of most frequent face | **21 frames / 9 appearances / 21.0 s**, across 3 scenes |

> **Why two numbers for "unique faces":** the raw cluster count (154) includes a
> long tail of one-off background faces; the **featured cast (21)** — people with
> real screen presence — is the meaningful answer to "how many people feature."

### 6.1 Face Occurrence Statistics (top identities)

| Face ID | Screen time | Appearances | Frames | Demographics |
|---------|-------------|-------------|--------|--------------|
| Face_01 | 21.0 s | 9 | 21 | M ~51 |
| Face_25 | 18.0 s | 11 | 18 | M ~32 |
| Face_02 | 17.0 s | 6 | 17 | F ~24 |
| Face_03 | 15.0 s | 6 | 14 | M ~47 |
| Face_05 | 13.0 s | 7 | 13 | F ~30 |
| Face_06 | 12.0 s | 4 | 12 | F ~23 |
| Face_15 | 12.0 s | 7 | 12 | F ~27 |
| Face_07 | 10.0 s | 4 | 10 | M ~46 |

### 6.2 Most Frequent Face

**Face_01** (male, ~51) is the most-present identity — 21 frames / 21.0 s of screen
time, consolidated across **three separate scenes** (≈17:02, 19:50, 21:34) by the
best-shot cross-scene linker. Its grouped-face montage shows one consistent person,
validated visually and by 0 same-frame conflicts.

![Face_01 — grouped faces across three scenes](reports/montages/Face_01.png)

### 6.3 Quality / Validation Summary

- **Co-occurrence precision: 1.0000** (no two same-frame faces share an identity).
- **Tracking-continuity recall: 0.83** (intra-shot completeness).
- **Grouping is provably coherent** (complete-linkage guarantees within-cluster
  similarity), and cross-scene merges can be human-certified via the review sheet
  (`reports/merge_candidates.{png,csv}`) — ~6 confirmations instead of labeling all
  359 tracks.

### 6.4 Sample Visualizations

- `reports/montages/Face_<NN>.png` — grouped face crops per identity, timestamped.
- `reports/contact_sheet.png` — one representative crop per identity.
- `reports/report.html` — full interactive summary (stats, montages, timeline,
  annotated sample frames).

![Identity contact sheet — one representative crop per identity](reports/contact_sheet.png)
