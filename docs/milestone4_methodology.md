# Milestone 4 — System Integration and Evaluation

## Overview

The previous three milestones each added a capability. This one adds none. Its
subject is the system itself: the fourteen stages that turn a URL into a
narrated, searchable dataset had grown up one milestone at a time, each with its
own evaluation harness and its own idea of what "done" meant, and nothing in the
repository could answer the three questions a reviewer would actually ask first.
What does it cost to run the whole thing? How good is the complete output, as
opposed to any one component of it? And where, specifically, does it fail?

Three deliverables follow from those questions: a single executable pipeline, a
single application over everything the pipeline produces, and a single evaluation
that speaks about the system rather than about a stage. The interesting work in
this milestone is not in adding code — most of the integration already existed —
but in making the system able to describe itself honestly, including in the
places where the honest description is unflattering.

## Integration

Integration was, in the narrow sense, already done. `run_pipeline.py` had been the
orchestrator since Milestone 1 and had absorbed each new stage as it arrived: it
runs download, extraction, detection, tracking, grouping, analytics, OCR,
captioning, the metadata join, the text and CLIP indexes, scene segmentation, the
vision descriptions, narration, and the evaluation harness, in that order, with
each stage's output being the next stage's input through files under `data/`.
Every stage fingerprints the config parameters it depends on, so it reruns when
those parameters change and skips when they have not. This is what makes the
project iterable at all: a caption-threshold experiment does not re-detect four
hundred faces.

That idempotency, however, is exactly what made the milestone's fourth task
awkward, and the awkwardness is worth stating plainly because it shaped the
design. A pipeline that skips almost everything on a second run cannot measure
its own cost by being run. Time it on a cold machine and you get the true figure
once; time it ever again and you get nearly zero, because the work was already
done. The naive instrumentation — wrap each stage in a timer, write the results
at the end of the run — therefore produces a file that is correct exactly once
and silently wrong forever after, and wrong in the flattering direction.

The resolution is that timings are merged rather than replaced. `util.time_stage`
wraps each stage, and on completion merges that single measurement into
`data/stage_timings.json`, leaving every other stage's last measured duration
intact. A stage that skips contributes nothing and loses nothing; a stage that
genuinely reruns overwrites its own entry and only its own. The file that results
describes a full cold build even though no single run ever produced it — an
assembled truth rather than a snapshot, which is the only honest option available
to a system whose whole design is not to repeat work. Two smaller decisions fall
out of the same reasoning. A stage that raises is not recorded at all, because a
partial duration understates cost while looking like a measurement. And each
duration is written the moment its stage finishes rather than batched at the end,
so a run that dies in narration still leaves the previous nine stages' timings on
disk.

## The application

The application requirement enumerates six things a user must be able to do:
search extracted text, see timestamps of matches, view the corresponding frames,
browse captions, view face occurrence statistics, and reach the generated story
and summary. The dashboard requirement enumerates seven results to display. The
two lists overlap heavily, and both are satisfied by artifacts the pipeline had
already been writing for three milestones — which is the point. Nothing in this
milestone re-derives anything. The application is a view, and if it needed to
compute something the pipeline had not already computed, that would be evidence
of a gap in the pipeline rather than a feature of the UI.

`search_app.py` grows from two tabs to five. The Dashboard opens on the headline
counts and, below them, the system's own measured performance — timings, quality
metrics, and limitations pulled straight from the evaluation report. Search is
unchanged from Milestone 2, with its three retrieval modes: lexical, semantic over
caption and OCR meaning, and visual over CLIP image vectors. Faces presents
per-identity occurrence statistics — screen time, appearance counts, first and
last seen, estimated demographics — with the montage of every crop grouped into
that identity and the frames it appears in. Captions & OCR exposes the metadata
repository directly, filterable, with a frame viewer. Story & Timeline is
Milestone 3's tab, carrying the summary, the story, the four prompt strategies
side by side, and the event timeline against each scene's keyframe.

One constraint runs through all five: no tab may assume its stage has run. A
Milestone 1 user has no captions; a user without an API key has no story. Each tab
degrades to an instruction rather than an exception, and — a detail that matters
more than it looks — no tab calls `st.stop()`, because in Streamlit that halts the
entire script and would blank the other four tabs for the very user whose missing
artifact triggered it.

The counting displayed on the dashboard deserves a note, because it is where the
application is most tempted to flatter the system. The video contains a hundred
and fifty-four identity groups, and it would be easy to print that as "unique
faces detected" and move on. But most of those groups are background passengers
seen in a single frame, and a number that large is really a statement about a
transit video, not about the system's accuracy. The dashboard therefore shows both
tiers — every group, and the featured cast of identities present in at least five
frames — and labels which is which. The headline number is the smaller, more
defensible one.

## Evaluation

The evaluation module is deliberately not a new measurement. Every milestone
already scores itself: `eval_labeled` scores the clustering against a hand-labeled
subset, `eval_cooccurrence` and `eval_continuity` apply label-free structural
checks, `eval_search` scores OCR fidelity and retrieval precision, `eval_story`
scores the four prompt strategies on chronology, coverage, grounding and
redundancy. Those harnesses are the measurement. `eval_system.py` is a reducer
over them and over the stage timings, and adding a fifteenth opinion about
accuracy would have been worse than useless — it would have been a number with no
harness behind it.

Its one real design commitment concerns the limitations section. The obvious way
to write "limitations and possible enhancements" is as prose beside the metrics,
and the obvious problem with that is that prose does not know when it has become
false. A paragraph explaining that identity grouping over-fragments will still be
sitting in the report, in a confident voice, months after somebody fixes the
fragmentation. So each limitation here is a predicate over the metrics rather
than a sentence next to them: the fragmentation finding exists only while mean
clusters-per-person exceeds 1.5, the OCR finding only while recall is under 95%,
the caption finding only while fewer than 80% of captions clear four out of five.
Improve the number and the finding removes itself from the next report. The tests
assert exactly this — that the fragmentation finding fires at 3.09 and is gone at
1.0 — because a self-deleting limitation is a claim about behaviour, not a
formatting choice.

The reducer only reduces, which means it is only as honest as the JSON it reads,
and that exposes two ways for a stale number to slip through — both of which the
report now guards against rather than trusts. The first is staleness of source:
the three label-dependent harnesses (grouping against the labeled subset, OCR and
retrieval quality, narration scoring) were built to run standalone, so nothing
regenerated them on a normal pipeline run, and the reducer would happily
summarise whatever they last wrote by hand. They are now run as part of the
pipeline's evaluation stage, and, because a manual invocation can still outrun
them, `staleness_report` compares each eval's file time against the artifact it
scored and prints a banner at the top of the report naming any input that
predates its own evidence. It earns its place immediately: the labeled-grouping
JSON is flagged the moment its ground-truth file goes missing, which is exactly
the failure a prose report would have hidden. The second is overconfidence at
small n. Several of the sharpest numbers rest on tiny hand-labeled samples — OCR
precision is a hundred percent of eleven positives, not of a thousand — so every
proportion is now printed with a 95% Wilson interval and its denominator, which
turns a bare "100%" into an honest "74–100%, n=11" without changing the point
estimate.

Two things resist that treatment and are handled explicitly rather than pretended
away. The first is scope: no metric can report the absence of something never
built. Nothing in the system measures the audio it does not process or the events
between its one-frame-per-second samples, so the vision-only, 1-FPS scope is
stated unconditionally, marked as stated rather than derived. The second is the
narration timing. Because `data/llm_cache/` is committed, `narrate` and `describe`
replay from disk instead of calling the API, and a reader who sees two seconds
against `narrate` would reasonably conclude the language model is cheap. It is
not; cold generation against the free tier is dominated by rate-limit backoff and
takes minutes. The cache is a real property of the system and the reason CI and
reruns cost nothing, but it is not evidence about the model, and the report says
so where the number appears.

The same instinct drives the split between local compute and network wait in the
timing report. Wall-clock through the narration stages measures how long a shared
free-tier endpoint made us queue. Averaging that into a per-frame throughput
figure would publish someone else's rate limiter as this pipeline's latency, so
the two are reported separately and only local compute feeds the throughput and
real-time numbers.

## What the measurements say

The full generated report lives in [`reports/eval_system.md`](../reports/eval_system.md)
and is regenerated on every run; the summary below is the reading of it.

The system's central trade-off is visible in two numbers that sit next to each
other: homogeneity is high and completeness is markedly lower. The grouping stage
almost never puts two different people into one identity — the label-free
cannot-link check, which exploits the fact that two faces in the same frame
cannot be the same person, finds zero violations across every co-occurring pair —
but it frequently splits one person across several identities. This is the
deliberate direction to err for an occurrence report. A false merge corrupts a
count silently and invisibly; a false split shows up as duplicate entries in a
cast list, where a human can see it. But it is the system's dominant error, it is
driven upstream by faces whose median size sits close to the detection floor, and
the report ranks it first rather than burying it.

The same shape repeats across the other subsystems, which is the most interesting
finding of the milestone. OCR misses text rather than inventing it. Captioning is
mediocre and known to be — which is precisely why retrieval was given a
caption-free CLIP path and scene segmentation was cut on imagery rather than
caption runs, so that two later milestones do not rest on the weakest component.
The narrator embellishes but never misorders: chronology is perfect and grounding
against an independent vision description is not. In every case the system fails
in the direction that is visible rather than silent, and in every case that is
measured above rather than asserted.

## Closing note

The honest summary of this milestone is that integration was the small part and
self-description was the large one. The pipeline already ran end to end; what it
could not do was tell you what it cost, how well it worked, or where it broke,
without a human writing those answers by hand — and a hand-written answer is one
that starts decaying the moment the code moves under it. The three pieces here —
timings that survive idempotency, an application that shows the defensible number
rather than the impressive one, and a limitations section that deletes its own
findings when they stop being true — are all attempts at the same property. The
system should be unable to flatter itself by going stale.
