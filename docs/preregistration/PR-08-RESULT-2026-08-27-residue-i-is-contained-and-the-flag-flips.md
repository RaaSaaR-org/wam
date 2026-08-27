# PR-08 — the determination on blocker 2's residue (i), and the flag that follows from it

**Determination under `T40_RULE_V18` §3, outcome C (CONTAINED). Decided by the project owner on
2026-08-27.** The session prepared this text, re-derived every number in it from the artifacts, and
put the question with its evidence for and against; **the decision is the owner's and is recorded as
theirs.** `T40_RULE_V18` was registered blind on 2026-08-27, before its census ran, and §3's three
outcomes have not been touched since.

V18 §3 outcome C requires four things to appear **in the determination's own text**, not in a
document a reader may not open. They are §§2–5 below, in that order, and this document is not a valid
determination without them.

---

## 1. The outcome, and why it is C rather than U or E

`runs/pr08-operating-point/EPISODE_094_CENSUS.json`
(sha256 `fb5a04d64ba5eeaea955da169f2d56fae9ae4afef7cc9f9d93e3442b42a5dfff`, tracked from
commit `76aa2ea`), both decode trees, all 509 frames of `episode_000094`, adapter counters read
around each call.

**Outcome U cannot fire.** U requires zero refusals, or refusals outside the documented ~f101–f155
run. Measured: **31 refusals on each decode**, `refused_span [109, 149]`, entirely inside it.

**Outcome E cannot fire.** E requires one non-refused mask above **3×** the episode's own median
non-refused area. Re-derived by the session from the artifact's own 509 per-frame rows:

| decode | non-refused | median area | largest | ratio | 3× bar | over the bar |
|---|---:|---:|---:|---:|---:|---:|
| `pr08-apple-640x480-h264-lossless` | 478 | 5 650.5 px | 7 383 px at f82 | **1.3066×** | 16 951.5 px | **0** |
| `pr08-apple-640x480` | 478 | 5 650.5 px | 7 383 px at f82 | **1.3066×** | 16 951.5 px | **0** |

For scale, the bar was coined blind in V18 §3 off `MASK_AUDIT.json`: correct apple masks at
1.1–1.2× against the episode median, the plate masks the audit found at ~4.3×. The largest surviving
mask in this episode sits at the correct-mask end of that gap, not near the plate end.

**Therefore C.** Neither U nor E; every non-refused mask is of apple-plausible area and the refusals
fall inside the documented low-score run.

---

## 2. Required item 1 — the refusal count, the frame indices, both decodes, and their disagreement

**31 on each decode, and the disagreement is empty.**

```
refused_in_both        109 110 111 112 114 115 117 126 127 128 129 130 131 132 133 134
                       135 136 137 138 139 140 141 142 143 144 145 146 147 148 149
refused_in_first_only  []
refused_in_second_only []
jaccard                1.0
```

`refused_is_contiguous` is **false** — 113, 116 and 118–125 are kept inside the span. On both
decodes: `n_frames 509`, `n_frames_with_mask 478`, `n_frames_with_centroid 478`, `n_no_centroid 31`,
`n_no_centroid_that_are_not_refusals` **0**, `no_detection_frames []`, `empty_mask_frames []`. So
every centroid-less frame in this episode is a refusal — not a detector miss and not an empty mask
from a real box — which is the three-way separation V18 §2 said the census exists to make.

**The two decodes agree on every frame of this episode**, which measures rather than caveats limit 1
of the operating-point result.

---

## 3. Required item 2 — this is ONE episode of 402, and it bounds nothing corpus-wide

The census establishes containment **here**. It does not bound the rate anywhere else, and the
reason is structural rather than a matter of effort: **the 16-shard corpus pass recorded no per-frame
centroid-present flag**, so there is nothing corpus-wide to census against. The 92 frames residue (i)
is about are 92 of 171 625 across 402 episodes; 52 of them were attributed to this episode. What is
measured here is those 52's neighbourhood, on one episode, on one machine.

**Corpus-wide decode bit-identity is likewise NOT MEASURED.** It was measured on this one episode —
509/509 frames identical across four decoder pairings — and that result is being used to retire a
provenance caveat that governs a 171 625-frame pass. That extrapolation is named here rather than
relied on silently.

---

## 4. Required item 3 — limit 3 stands: nobody has looked

**The area test is a proxy for a wrong-object mask, not an observation of one.** A wrong-object mask
of apple-plausible area passes every test in this document and in V18. Nothing in the census is a
look at a picture; it measures what the filter did, not whether the filter was right. V18 §2 said so
before the census ran and §4 says the discharge of blocker 1 is untouched by any of this.

**And one number inside the evidence is unexplained.** `runs/pr08-geom-tol/shards/shard-7.json`
(sha256 `faddc46469e20b25b82eccb4c753c6d075e6c534f56c4bb0ef1a3fb51c727944`, tracked from `76aa2ea`)
records this episode as `n_frames_with_centroid: 473`; the census measures **478**. The gap is five
frames and it is **not closed by this determination**:

* **The decode hypothesis is refuted.** `episode_000094` decodes bit-identically through cv2 5.0.0,
  imageio/FFMPEG and pyav 18.0.0 over both corpus trees — 509/509 frames identical on all four
  pairings. Whatever the five frames are, they are not a codec.
* **The frames are not fully named.** The artifact pins `f101`, `f108`, `f124`, one of
  `{f152, f153, f154}` and one of `{f113, f116, f125}`. An exhaustive search over 68 280 feasible
  4-subsets leaves `{101,108,124,153}` and `{101,108,124,154}` **tied** with the accepted hypothesis
  on max|D| at 0.165650 px, separated on the mean by ~1e-4 px. The fifth frame is **unrecoverable
  from any artifact in this repository.**
* **The claim that the outcome is invariant under the shard's 36 is an inference, not a
  measurement.** `shard-7.json` records no per-frame mask area, so any recomputation applies this
  workstation's areas to a cluster-derived index set. It is not offered as a ground here.

**What the determination does rest on**, and it is narrower than the invariance argument: V18 §2
registered **the census** as the instrument and §3's outcomes are written against the census, not
against the shard. Supporting it, the shard's 466 displacements match this workstation's to a
maximum of 0.166 px, which pins the cluster's centroids on 471 of its 473 kept frames — and a
~31 000 px plate mask cannot share a centroid with a ~1 000 px apple mask.

A reader who holds that a determination may not be made while five frames of its evidence are
unexplained should **refuse this document** rather than read past this section. That is why it is
here and not in a footnote.

---

## 5. Required item 4 — the predicted failure DID occur, and this is a decision about a rate

Blocker 2 predicted that the `(0.10, 0.10)` retry would buy weak detections and hide a degraded mask
inside an inflated coverage number. **Measured over the whole corpus, the retry fired zero times on
all 171 625 frames.** The failure it predicted happened anyway, by the *primary* 0.15 threshold: on
**92 of 171 625 frames** the operating point changed the outcome — **0.054 %** — 52 of them in this
episode, where `MODEL_OBSERVATIONS.json` reports a confident, well-formed mask of **the plate** at
scores 0.155–0.259, ~31 000 px, plate overlap 0.985–0.992, zero IoU with the colour heuristic.

**Accepting 0.054 % is a decision about a rate. It is not a finding that nothing happened.** The
filter caught these; the determination is that being caught is enough, on one episode, measured by a
proxy, with five frames unexplained.

---

## 6. What this does and does not do

**Does.** It satisfies the **second** of `GATE_QUALIFIED`'s two preconditions — *"somebody decides,
on the record, what to do with the residue the two 2026-08-26 entries carry forward"*. With
`GATE_QUALIFICATION_BLOCKERS` empty since 2026-08-27 (the seventh entry, the propagation difference,
discharged under `T40_RULE_V17` outcome N), **both preconditions are now met and the flag flips in a
commit that does nothing else.**

**Does not.**

* **It does not discharge blocker 1.** *"Nobody has looked at a mask"* is untouched; §4 above says so
  in the determination's own text, as V18 §3 outcome C item 3 requires.
* **It does not re-open a discharged blocker.** Blocker 2 stays discharged; its residue is what was
  decided.
* **It licenses no clip and no training.** `T40_RULE_V1` §1 binds and is not lifted by this document
  or by the flag. PR-08 §8 items 3 and 4 are both still open.
* **It does not make `GEOM_TOL = 0.478579…` committable.** That number's shards were measured with
  `gate_qualified: false` baked in at measurement time and disagree with the committed contract on
  `mask_validity_reference_max_frame_fraction`. The corpus pass must be run again at HEAD.
* **It does not close the five-frame gap**, which stays on the record as unexplained.

---

## 7. What would refute this

* A look — V16's human half, or any reader — finding a wrong-object mask of apple-plausible area
  among the 478 kept frames of this episode.
* A per-frame centroid-present flag from the re-measured corpus pass showing refusals outside
  ~f101–f155 in other episodes, or a non-refused mask over 3× its episode's median anywhere.
* An explanation of the five frames that implicates the filter rather than the machine.

Any of those reopens residue (i). The flag is a claim, and a claim whose grounds are written down can
be taken back.
