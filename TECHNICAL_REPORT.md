<div class="title" markdown="1">

# Recurring Face Identification and Occurrence Analysis in Video

### Technical Project Report

**Prepared By:** Ilangkumaran Yogamani

**Date:** 19 June 2026

**Source video:** `https://www.youtube.com/watch?v=d2g9HlwoC-s`

</div>

---

## Executive Summary

This project implements a computer-vision pipeline that processes a single video,
extracts frames at a fixed rate, detects faces, generates face embeddings, groups
recurring individuals, estimates the number of unique people, computes per-person
appearance statistics, and identifies the most frequently appearing person.

Detection uses **SCRFD-10GF** and embedding uses **ArcFace (ResNet-50)**, both from
the InsightFace `buffalo_l` model pack. Grouping is performed at the **track level**
using **constrained complete-linkage agglomerative clustering** over cosine distance,
with a co-occurrence "cannot-link" constraint and a best-shot cross-scene linking
pass. From **1,415 extracted frames**, the system detected **435 face crops**,
formed **359 tracks**, and produced **154 face groups**, of which **21 are a
"featured cast"** (groups with ≥5 frames of presence). The most frequently appearing
identity is **Face_01** (21 frames; 9 distinct appearances; 21.0 s).

Because no independent human-labeled ground truth is available, accuracy/precision/
recall against ground truth are **Not Measured**. Quality is instead reported via two
**label-free** objective checks implemented in the project: a co-occurrence precision
check and a tracking-continuity recall check.

---

## 1. Problem Statement

The goal is to determine **who appears in a video and how often**, without prior
knowledge of the people present or their number. Concretely, the system must:

- extract frames from the video,
- detect and crop faces,
- represent each face numerically (embedding),
- group faces of the same person together,
- estimate the number of unique individuals, and
- report appearance statistics and the most frequently appearing person.

Recurring-face identification of this kind is useful for content indexing, screen-time
measurement, and summarisation. The number of people is unknown in advance, so the
grouping method must not require a predefined cluster count. Expected outcomes are a
set of grouped identities with occurrence statistics and supporting visualisations.

---

## 2. System Architecture

```
                         Video Input (1080p)
                                |
                       Frame Extraction (1 FPS)
                                |
                    Face Detection (SCRFD-10GF)
                                |
              Landmark Alignment + Face Cropping (112x112)
                                |
                Embedding Generation (ArcFace, 512-D)
                                |
                    Multi-frame Tracking (ByteTrack)
                                |
                 Per-track Template (quality-weighted)
                                |
        Complete-linkage Clustering + co-occurrence cannot-link
                                |
                  Best-shot Cross-scene Linking
                                |
                 Corroboration Backstop (identities)
                                |
            Occurrence Statistics + Visual / HTML Outputs
```

**Figure 1: Overall system architecture.** Each stage is a separate, independently
runnable module orchestrated as a single end-to-end pipeline. An optional
clothing/body re-identification stage (disabled by default) can be inserted before
clustering as an additional cross-scene signal.

---

## 3. Methodology

### 3.1 Frame Extraction Pipeline

Frames are sampled from the downloaded video at **1 frame per second** (the project's
configured rate) using the `supervision` video utilities, with a stride derived from
the true source FPS so that timestamps are accurate. Each frame is stored and
recorded with its frame index and timestamp. The source is acquired at up to
**1080p** to preserve detail for small faces. **1,415 frames** were extracted.

### 3.2 Face Detection Pipeline

**Face Detection Model Used: SCRFD-10GF** (InsightFace `buffalo_l` model pack), run
via ONNX Runtime on CPU.

- **Selection rationale:** SCRFD is a single-shot, anchor-based detector that returns
  bounding boxes, 5-point landmarks, and confidence scores, and is markedly more
  robust to pose, lighting, and scale than a classic Haar cascade. It is bundled with
  the matching ArcFace recogniser, simplifying the stack.
- **Process / thresholds:** input size `1024×1024`; confidence threshold `0.4`;
  detections smaller than `36 px` (width or height) are discarded.
- **Cropping:** each kept face is aligned to a `112×112` crop using its landmarks
  (`norm_crop`) and saved.
- **Multiple faces:** all faces per frame are detected and processed independently.

**Advantages:** robust multi-pose detection; landmarks enable alignment.
**Limitations:** at the chosen threshold some low-confidence/marginal detections (incl.
non-faces) are admitted, which are handled downstream by a corroboration backstop;
true detection recall is **Not Measured** (no ground-truth bounding boxes).

### 3.3 Face Recognition / Embedding Pipeline

**Face Recognition / Embedding Model Used: ArcFace** (ResNet-50, trained on
WebFace600K; InsightFace `buffalo_l` model pack).

- **Preprocessing:** landmark-aligned `112×112` crops.
- **Embedding:** each crop is mapped to a **512-dimensional, L2-normalised** vector.
- **Representation / similarity:** because vectors are unit-norm, similarity is the
  cosine (dot product); same-person faces yield high similarity, different people low.
- **Matching:** comparisons are done on **per-track template** embeddings (see §3.4),
  not raw per-crop vectors, to suppress per-frame noise.

This approach suits the project because it requires no per-video training and produces
a fixed-length, comparable representation for an unknown number of people.

### 3.4 Face Grouping Methodology

1. **Tracking & templating.** Faces are linked across frames with **ByteTrack**; each
   track is reduced to one **quality-weighted, outlier-trimmed mean embedding**.
2. **Similarity.** Cosine distance between track templates.
3. **Clustering.** **Complete-linkage agglomerative** grouping (every member pair must
   be within the distance threshold, `0.50`), which prevents the "chaining" that
   merges dissimilar faces. A **co-occurrence cannot-link** constraint forbids merging
   two faces seen in the same frame (they must be different people).
4. **Cross-scene linking.** A **best-shot** pass merges clusters whose most frontal,
   highest-quality faces match (complete-linkage, threshold `0.50`), consolidating a
   person's appearances across scene cuts.
5. **Identity consolidation.** A **corroboration backstop** keeps a cluster as a real
   identity only if it has ≥2 faces, a high-quality face, or a confidently-detected
   face; otherwise its tracks are labelled `unknown`. Surviving clusters are named
   `Face_NN`, ranked by screen-time.
6. **Threshold selection.** Thresholds were chosen by an internal sweep maximising
   cluster cohesion and the label-free checks (§6); see Limitations regarding the
   absence of authoritative ground-truth tuning.

> **NOTE — Optional Clothing / Body Re-Identification (disabled by default).**
> Face embeddings sampled at 1 FPS cannot reliably link the same person across large
> pose changes or scene cuts. As an optional, *orthogonal* signal, the system can
> additionally embed each person's **clothing/body region** — the torso below the
> detected face — using a dedicated person re-identification model, and compare it
> across groups. Because an individual's outfit is consistent within a short video and
> is independent of face pose, this can consolidate cross-scene appearances that face
> matching alone misses.
>
> The stage is applied **conservatively**: a merge requires *both* clothing agreement
> *and* a plausible face match, and can never join two faces seen in the same frame.
> Rather than merging automatically, proposed cross-scene merges are surfaced for
> **human confirmation**. It is **disabled by default** because its characteristic
> failure mode — different people wearing similar outfits or sharing a background — is
> not detectable by the automatic quality checks, so enabling it is recommended only
> in combination with human review. In internal testing it correctly consolidated a
> recurring individual across three separate scenes.

---

## 4. Alternative Approaches Considered

| Approach | Potential Advantages | Potential Limitations | Reason Not Selected |
|----------|----------------------|-----------------------|---------------------|
| OpenCV Haar cascade (detection) | Fast, simple | Weak on side/blurred/small faces | Lower-quality detection than SCRFD |
| MTCNN / RetinaFace (detection) | Good accuracy | Not bundled with the chosen recogniser | SCRFD in `buffalo_l` already adequate and integrated |
| Dlib / `face_recognition` (embedding) | Easy to use | Older representation | ArcFace expected to generalise better |
| FaceNet512 (embedding) | Common alternative | Heavy extra dependency; re-tuning | Considered and rejected in favour of bundled ArcFace |
| K-Means clustering | Simple | Requires number of people in advance | Count of people is unknown |
| DBSCAN (single-linkage) clustering | No preset cluster count | Single-linkage *chains* dissimilar faces into incoherent groups | Replaced by complete-linkage after this was observed in the project |

---

## 5. Challenges Encountered

**Grouping — chaining / false merges.** An initial DBSCAN (single-linkage) approach
produced an incoherent "junk" group (very low within-group similarity), i.e. different
people merged together. *Mitigation:* switched to complete-linkage, which guarantees
all members are mutually within threshold. *Remaining:* none of this specific failure
mode after the change.

**Grouping — over-fragmentation.** At 1 FPS, the same person across **scene cuts** has
no temporal bridge, so their appearances can split into several groups. *Mitigation:*
the best-shot cross-scene pass recovers frontal recurrences. *Remaining:* profile-only
cross-scene recurrence is not resolved by face embeddings; the total group count is
therefore an over-estimate of distinct people.

**Recognition / face quality.** Many crops are small or non-frontal; gender/age and
embedding quality degrade on these. *Mitigation:* quality-weighted templates, a
minimum-size filter, and the corroboration backstop. *Remaining:* small/profile faces
remain harder to match.

**Validation.** No ground-truth labels are available. *Mitigation:* two label-free
objective checks (co-occurrence precision, tracking-continuity recall). *Remaining:*
these do not measure cross-cut grouping correctness; see §9.

**Computational.** Detection runs on CPU. Hardware acceleration via CoreML was tested
and measured **slower** for this model, so it was disabled. Processing is batch, not
real-time.

---

## 6. Results and Analysis

### 6.1 Frame Statistics

| Metric | Value |
|--------|-------|
| Source resolution | 1080p |
| Sampling rate | 1 FPS |
| Total extracted frames | 1,415 |
| Approx. sampled duration | ≈ 1,415 s (~23.6 min), inferred from frame count at 1 FPS |

### 6.2 Face Detection Statistics

| Metric | Value |
|--------|-------|
| Total face crops detected | 435 |
| Average faces per frame | ≈ 0.31 (435 / 1,415) |
| Detection threshold | 0.4 |
| Minimum face size kept | 36 px |
| Detection rate vs. ground truth | Not Measured (no ground-truth boxes) |

### 6.3 Face Occurrence Statistics

| Metric | Value |
|--------|-------|
| Total face tracks | 359 |
| Total face groups (named identities) | 154 |
| Featured cast (≥5 frames present) | 21 |
| Crops assigned to `unknown` (backstop) | 31 |
| Largest group size | 21 frames |
| Average crops per named group | ≈ 2.62 |
| Estimated unique identities | 154 (upper bound) / 21 featured — see note |

> **Note on "unique identities":** the raw group count (154) over-counts real people
> because of cross-cut over-fragmentation and a long tail of one-off/background faces.
> The **featured cast (21)** is the more meaningful estimate. The true number is
> **Not Available** (no ground-truth validation).

### 6.4 Most Frequent Face Analysis

The most frequently appearing identity is **Face_01** — **21 frames**, **9 distinct
appearances (tracks)**, **21.0 s** of screen time, estimated **male, ~51**. It was
consolidated across **three separate scenes** by the best-shot cross-scene linker and
visually corresponds to one consistent person.

Relative frequency: 21 of 435 detected crops (≈4.8%); it is the top group by both
frame count and screen time. Because there is no ground-truth labelling, this is the
most-frequent group **as grouped by the system**, not a ground-truth-verified claim.

Top identities by screen time:

| Face ID | Screen time | Appearances | Frames | Demographics (est.) |
|---------|-------------|-------------|--------|---------------------|
| Face_01 | 21.0 s | 9 | 21 | M ~51 |
| Face_25 | 18.0 s | 11 | 18 | M ~32 |
| Face_02 | 17.0 s | 6 | 17 | F ~24 |
| Face_03 | 15.0 s | 6 | 14 | M ~47 |
| Face_05 | 13.0 s | 7 | 13 | F ~30 |
| Face_06 | 12.0 s | 4 | 12 | F ~23 |
| Face_15 | 12.0 s | 7 | 12 | F ~27 |
| Face_07 | 10.0 s | 4 | 10 | M ~46 |

### 6.5 Sample Visualizations

![Featured-cast representative faces](reports/featured_cast.png)

**Figure 2: Featured-cast overview.** One representative crop per featured identity
(the 21 groups with ≥5 frames of presence), labelled with screen time.

![Most frequent face grouped samples](reports/montages/Face_01.png)

**Figure 3: Most frequently appearing face (Face_01).** Grouped crops of the
highest-appearance identity, shown with timestamps across three scenes.

![Occurrence distribution](reports/occurrence_distribution.png)

**Figure 4: Face occurrence distribution.** Frames-present per group across all 154
groups (descending); the dashed line marks the featured-cast threshold and the long
tail reflects one-off/background detections.

### 6.6 Quality / Validation Summary (label-free)

| Check | Result | Meaning |
|-------|--------|---------|
| Co-occurrence precision | 1.0000 | No two faces in the same frame share an identity |
| Tracking-continuity recall | 0.83 | Intra-shot completeness (briefly-broken tracks regrouped) |

---

## 7. Conclusion

The project delivers a complete, modular pipeline that turns a raw video into grouped
face identities with occurrence statistics. From 1,415 frames it detected 435 faces,
formed 359 tracks, and produced 154 face groups (21 featured), identifying **Face_01**
as the most frequently appearing person. The grouping is built on SCRFD detection and
ArcFace embeddings, with a complete-linkage clustering scheme and a co-occurrence
constraint that, by construction, keeps groups internally coherent and free of
same-frame merges (co-occurrence precision 1.0000; intra-shot continuity recall 0.83).

The system is suitable for content indexing and screen-time analysis where an unknown
number of people must be grouped without training. Its principal limitation is the
absence of ground-truth validation and the residual cross-cut fragmentation inherent
to face-only matching at 1 FPS — both stated honestly as measured constraints rather
than resolved claims.
