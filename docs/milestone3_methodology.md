# Milestone 3 — Higher-Level Video Understanding

## Overview

The goal of this milestone is to lift the project from per-frame perception into
whole-video understanding. Where the earlier stages looked at the London
Underground tour one frame at a time — detecting faces, reading signage, and
captioning each sampled image — this stage has to speak about the twenty-three
minutes as a single, connected thing. Three deliverables define it: a coherent
story that narrates the complete video in the order events actually happen, an
overall summary that captures what the recording is fundamentally about, and an
event timeline that pins the significant moments to the second. All three are
built from the captions produced upstream, and the entire challenge of the
milestone lies in the distance between what those captions are and what a
faithful narrative needs them to be.

## Methodology

The starting point is a bluntly honest assessment of the raw material. Sampling
the video at one frame per second yields roughly fourteen hundred frames, but the
captioner behind them is far less rich than that number suggests. Across the whole
recording it produces only a few hundred genuinely distinct caption strings; on
long stretches of visually identical footage it flickers between near-synonyms of
the same sentence, describing a train pulling into a platform dozens of times in
slightly different words, and on more than one empty platform it simply invents a
train that is not there. Handing a language model fourteen hundred rows of this
would not produce a story. It would produce a transcription of noise, and any
attempt to tidy it by merging consecutive identical lines still leaves close to a
thousand fragments — a smaller pile of the same confusion.

The methodology answers this by refusing to trust the text as the unit of
structure. Instead the video is segmented on the imagery itself and only then
described. Every sampled frame already carries a normalized visual embedding from
the earlier retrieval work, and the similarity between two neighbouring frames is
therefore a single, cheap arithmetic comparison that needs no additional model to
run. Walking through the video, the system marks a boundary wherever that
frame-to-frame similarity drops below a tuned threshold, which is precisely where
the picture changes — a cut, a new platform, a different train. On top of this it
recognises the video's chapter markers: the tour is punctuated by full-screen
title slides that name each Underground line, and a boundary is always forced at
the start and end of one of these so that a title card becomes a scene in its own
right rather than bleeding into the footage on either side. Recognising those
cards reliably takes two independent signals working together, because neither is
sufficient alone. One signal reads the on-screen text and looks for a line name;
the other reads the caption and notices that the model always describes these
slides as words on a black background. The union of the two catches every card,
including the awkward ones — the introductory slide whose text names an event
rather than a line, and the Piccadilly card whose text recognition partly garbled
its own name. A final tidying pass dissolves any scene shorter than a few seconds
into its neighbour, sweeping away the residual flicker, while protecting the short
title cards from being swallowed by that same rule. What emerges is a clean
partition of the video into roughly three dozen scenes gathered under twelve
chapters, one chapter per line plus the introduction.

Each scene is then distilled into a compact digest, and this compression is the
quiet engine of the whole milestone. For every scene the system records its
timestamp span, chooses the single most representative frame as the one whose
embedding sits closest to the average of the scene, keeps only the distinct
captions rather than the repetitive flood, and retains any on-screen text that
genuinely looks like station signage. That last filter matters more than it
appears to: because the digest becomes the prompt, any garbage that survives here
turns directly into a hallucination later. A narrator told that the on-screen text
reads a stray three-letter fragment will happily invent a station by that name, so
the filter is deliberately strict, admitting confident roundel lettering and
plausible multi-word signage while discarding both optical-recognition shrapnel
and the in-carriage advertising that would otherwise masquerade as place names.
Alongside the visual content, each scene also carries the identities of the
recurring faces that appear within it, tying the milestone back to the face
analysis that came before. The result of all this is that the entire
twenty-three-minute video is described in only a couple of thousand words of
structured text.

That number is the reason the narration can work the way it does. The model
chosen for the writing has a context window large enough to hold the whole digest
many times over, so the narrator is shown the complete video in a single pass.
There is no splitting the video into chunks, no summarising the chunks and then
summarising the summaries, and consequently no seam anywhere in the process at
which the thread of the story could be dropped or the chronology scrambled. The
coherence of the final narrative is therefore structural rather than lucky: the
story reads as one continuous account because the model genuinely saw the whole
thing at once, not as three dozen stitched-together fragments.

The event timeline is treated with more suspicion than the prose, because it is
the one output that carries a machine-readable contract and will be consumed as
data rather than read as writing. Rather than trusting the model to return
well-behaved timestamps, the system validates every one of them: each must be a
parseable time, must fall inside the true duration of the video, and the sequence
as a whole must move forward. When the model strays, the output is repaired rather
than rejected — out-of-range moments are dropped and a jumbled sequence is sorted
back into order — and every repair is recorded honestly so the evaluation can
report what had to be fixed. If the model's answer degrades so far that too little
survives to be useful, the system abandons it entirely and falls back to the scene
boundaries, which are known-good ground truth from the segmentation. In practice
the model behaved well: the final timeline holds three dozen events, all of them
landing inside real scene spans, all in chronological order, with no repairs
required.

Finally, the milestone measures itself rather than asserting that its output is
good. Narrative quality is famously resistant to automatic scoring, so instead of
pretending to grade the writing directly the evaluation checks four things that
can be verified objectively and that map onto what the assignment actually asks
for. It measures chronology by extracting every moment the story cites and
checking what fraction of them move forward rather than backward in time, turning
the requirement that the narrative follow the true sequence of events from a claim
into a number. It measures grounding by asking what share of the substantive words
in the story are actually attested by the source material, as a proxy for how much
the narrator invented. It measures coverage as the fraction of the twelve chapters
the story genuinely mentions, which catches a fluent narrative that quietly skips
half the video. And it measures redundancy through the diversity of word
sequences, since the underlying captions are so repetitive that a lazy narrator
merely paraphrasing them would score poorly. All of this is reproducible offline,
because every answer the language model ever gave is preserved and replayed rather
than re-requested, so the reported figures can be regenerated by anyone without a
network connection or an account.

## Models and approaches selected

Two families of model carry this milestone, chosen so that each does only what it
is genuinely good at. The segmentation and keyframe selection lean entirely on the
visual embeddings already computed in the retrieval work. Because those embeddings
are normalized, judging how similar two frames are, or which frame best represents
a scene, reduces to elementary arithmetic that is fast, completely deterministic,
and free of any new dependency. Reusing this representation, rather than
introducing a dedicated shot-detection model, keeps the scene boundaries
explainable and exactly reproducible, and it turns out the sharpest visual changes
in the video coincide almost perfectly with the title cards, which is a
reassuring sign that the cuts are landing where a human would put them.

The writing itself — the story, the summary, and the timeline — is produced by a
large open-weights instruction model reached through a hosted service. It was
chosen for three reasons that all matter here. Its context window is vast, which
is what makes the single-pass, whole-video narration possible in the first place.
It accepts images as well as text, which is exploited separately for the analysis
described below. And it is openly licensed and available at no cost, which keeps
the whole pipeline free to run. That last point creates a genuine engineering
constraint, because the free service is congested and rate-limited, and much of
the practical work of the milestone went into being a well-behaved and resilient
client of it. Requests that fail because the service is momentarily overloaded are
retried patiently with an exponential, randomised backoff that respects the
server's own guidance on when to try again, so a temporary refusal never becomes a
lost result. Images are shrunk to the size the model works at internally before
being sent, since transmitting larger pictures would only inflate the payload and
make it more likely to be turned away. Most importantly, every response the model
returns is stored and, on any future run, replayed exactly rather than requested
again. This makes the whole milestone reproducible without an account and immune
to the model drifting between runs, and it means the numbers reported in the
evaluation are stable rather than a moving target. Determinism is reinforced by
asking the model to write with no randomness in its sampling, so that a
regeneration reproduces the stored answer instead of diverging from it.

The image-understanding capability of that same model is put to a specific and
revealing use. The perception upstream relies on a lightweight captioner that, as
already noted, is the weak link in the chain — terse, repetitive, and occasionally
fabricating. To measure just how much that weakness costs, the system takes the
representative frame of every non-title scene and asks the far more capable vision
model to describe what is genuinely visible in it: the setting, whether a train is
present and whether its doors are open, how many people appear and what they are
doing, and any legible signage. These richer descriptions are used in two ways.
They serve as an independent yardstick against which the story's grounding can be
checked without circular reasoning, since they were produced from the images
directly and owe nothing to the captions the story was written from. And they
power an alternative telling of the story built from these descriptions instead of
the captions, which isolates whether the narrative's limits come from the narrator
or from the impoverished captions it was fed. To make these image descriptions
affordable under the daily request ceiling, several keyframes are packed into each
request rather than sent one at a time, cutting the number of calls several-fold.
Batching introduces a subtle risk — the model could quietly attach the wrong
description to the wrong image — so each reply is required to name explicitly which
scene it is describing, and the set that comes back is checked against the set that
was asked for. Any scene the model skips or misaligns is quietly retried on its
own, one unambiguous image at a time.

The verdict from this comparison is stark and worth stating plainly. Measured
against what the vision model can see in the very same frames, the lightweight
captions capture only about a tenth of the visible substance, and every single
scene falls below a quarter. This is not a criticism the narrator can overcome, no
matter how it is prompted: a writer who is only ever shown the captions simply
cannot describe what those captions never mentioned. The finding reframes whatever
shortcomings remain in the story not as failures of the narrator but as the
faithful transcription of an impoverished input, and it is exactly the kind of
honest limitation the evaluation was designed to surface rather than hide.

## Prompt engineering strategy

Because the assignment specifically asks for prompt engineering to be explored
rather than assumed, four distinct ways of framing the same request were built and
compared head to head against the identical scene digest. Holding the underlying
material constant is what makes the comparison fair: every strategy receives the
same content and the same structural contract — namely that it must write one
section per line in order, open each with a heading naming that line so the story
can later be broken back into chapters, and anchor each section to at least one
real timestamp — and only the framing of the instruction differs between them.

The first and simplest is a plain, direct instruction that presents the digest and
asks for a coherent story, serving as the baseline against which the others are
judged. The second enriches that instruction with two fully worked examples, each
showing a small excerpt of the digest and the polished paragraph it should become,
so that the model can see the intended level of detail and the desired voice
rather than having to infer them. The third asks the model to reason before it
writes: it is told to first work through the material privately — listing the
chapters in order, noting what happens between each pair of them, and identifying
the single connecting thread of the video — inside a hidden scratch space, and
only then to compose the finished story, of which only the part after the private
reasoning is kept. The fourth casts the model in an explicit role, that of a
documentary narrator writing voice-over for an archival transit film, and pairs
that persona with a short list of firm rules: never describe a later moment before
an earlier one, never assert something the digest does not support, name a station
only when the on-screen text names it, and vary the shape of the sentences so the
prose does not fall into a monotonous rhythm.

Judged by the four objective measures, the strategies separate in instructive
ways. All four turned out to be perfectly chronological, never once describing a
later moment before an earlier one, which is the strongest possible confirmation
that the requirement to follow the true sequence of events is being satisfied by
the whole-video-in-one-pass design rather than by any particular wording. Where
they diverge is in grounding and in style. The example-driven framing grounds
itself most tightly in the source and cites the most individual moments, because
concrete examples pull the model toward reusing the exact language of the digest;
it is the strongest choice when maximal fidelity to the source is the priority.
The role-based framing, by contrast, produces by far the least repetitive and most
readable prose and carries the firmest discipline against inventing detail, thanks
to its explicit rules, and it is this version that was promoted to be the canonical
story of the project. Its somewhat lower word-overlap with the source is partly an
artefact of the measure itself, which cannot tell a legitimate synonym from an
invention and so penalises the very fluency that makes the writing good — which is
precisely why grounding is read as a comparative signal across strategies rather
than as an absolute verdict on truthfulness. The plain and reasoning-first
framings sit between these poles, fully covering the chapters but writing more
plainly. The decision to promote the documentary-narrator version is therefore a
deliberate and documented judgement that weights narrative quality and restraint
above raw textual overlap, and it can be reversed with a single configuration
change should a future reader prefer the more literal telling.

The same care extends to the two outputs that must be machine-readable rather than
merely well-written. Both the summary and the timeline requests carry firm
contracts on their form. The summary is asked for as continuous prose of a
controlled length that highlights the key events and says what the recording is
fundamentally a record of, without degenerating into a scene-by-scene list. The
timeline is asked for as a strict structured list of moments, each a single
timestamp paired with a short description, under explicit rules that every
timestamp must come from a real scene and that none may exceed the true end of the
video. Small models tend to wrap such output in stray commentary or formatting, so
the system reads their replies tolerantly — stripping away any preamble or
decoration to recover the underlying structure — and then subjects the result to
the validation described earlier. The combination of a firm contract in the prompt
and forgiving-but-strict handling of the reply is what allows even a modest,
free-of-charge model to yield a clean, in-range, strictly ordered timeline in the
end.

## Closing note

Taken together, the milestone turns a flood of shallow per-frame captions into a
genuine account of the video by compressing the footage into a small structured
digest on the strength of its imagery, narrating that digest in a single coherent
pass, and holding every generated artefact to a standard it can be measured
against. The most important thing it demonstrates is not any single number but a
posture: that the quality of the story is bounded by the quality of its input, and
that the honest way to build such a system is to make that boundary visible and
measured rather than to paper over it with confident prose.
