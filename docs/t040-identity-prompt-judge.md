# The judge for `T40-TODO-01` — a design, not a decision

**Status:** design document. It names candidates, prices them, and specifies the calibration set
any of them must clear. **It deliberately does not choose one.** Choosing is the decision this file
exists to inform, and it belongs to whoever reads it.

**What is already built.** `scripts/build_identity_prompt_sheet.py` is the harness, split
`build-sheet` / judge / `verdict`. The two outer steps exist and are tested
(`tests/test_identity_prompt_sheet.py`, 61 tests). The middle step is empty on purpose, and this
document is about filling it.

**What has already been produced.** A gate-qualified blank sheet:

```
runs/t040-identity-prompt/            sheet_id 862a4548ae25f52ae0f3c0dc238cfb64e2f79d93cf8732059599d90e0bf87cb4
  sheet.jsonl                         40 rows, seed 40001, stratified-systematic over 402 episodes
  sheet_meta.json                     gate_qualified: true, no disqualification reasons
  frames/*.png                        40/40 extracted at fraction 0.10, 640x480, sha256 pinned
```

Every row carries a frame path, the committed prompt verbatim, and a blank `verdict`. All that is
missing is an instrument that can fill 40 fields and be believed.

---

## 1. The question the instrument has to answer

Per row, exactly one of `match` / `mismatch` / `unsure`, plus a named axis on a mismatch, against
this committed string (`configs/transfer25/styles.toml`, `[identity_style].prompt`):

> A red and yellow apple with a glossy surface and a visible stem, on a black cloth covering a flat
> surface. A white, round plate. Even, bright, top-down lighting with minimal shadows; the lighting
> highlights the objects without creating harsh contrasts. Contrast between the black background
> and the white plate and red/yellow apple. Live-action video, realistic. The white plate keeps its
> own appearance. Scene geometry, camera framing and the robot are unchanged.

The string was machine-captioned from **one** clip, `episode_000135_clip000`, and applied unchanged
to all 402 episodes. The question is whether it also describes the other 401.

Note what the prompt actually asserts, because it decides what kind of instrument can check it: of
its five clauses, **four are colour and luminance statements** (a red/yellow glossy apple; a black
cloth; a black background; even bright top-down lighting with minimal shadows) and the fifth is a
shape-and-colour statement (a white round plate). Nothing here requires reading the scene
semantically. That is the opening the fourth candidate in §5 walks through.

## 2. The failure this must be designed against

T-041's whole run came back **VOID on G0b**. Forensics over the recorded `scores.jsonl` found the
VLM judge had answered the literal string `"NO"` to all 80 items — a constant classifier, zero
abstentions. Its calibration score was `calibration_correct: 10/20`, and every one of those ten
was a negative, earned by the negatives being NO-labelled. Nothing downstream was *wrong*; it was
vacuous. `base_failures: 30`, `G0a_defect_present: true`, `b = c = 0`.

Three separable lessons, and the calibration design in §3 answers each by name:

1. **The calibration set caught it, and the score still read like partial credit.** `10/20` invites
   "halfway there". A single aggregate number over a class-imbalanced set is the wrong report;
   per-class scores are the right one, because a constant classifier's signature is a *zero in one
   class*, and an aggregate hides zeros.
2. **The rubric folded abstention into a class.** T-041's rubric said *"Answer NO if you are
   unsure"*. That single sentence is why a constant `NO` scored 10/20 rather than 0/20: it made
   "I cannot tell" and "the answer is no" the same token, so the instrument could not express, and
   the harness could not detect, that it never decided anything. Our vocabulary keeps `unsure`
   separate, and §4 makes that separation testable rather than merely available.
3. **Nothing in the design required the judge to produce a *structured* answer that a constant
   could not fake.** A one-word YES/NO is maximally fakeable. Our `mismatch` requires an axis from
   a fixed six-word vocabulary, and `check_fill` refuses a mismatch with no axis. That turns each
   negative item into a 1-in-18 guess rather than a coin flip, and §3's arithmetic lives on it.

## 3. The calibration set

### 3.1 The circularity, and how it is broken

Any calibration item needs a known correct answer. "Does the committed prompt describe episode X"
is exactly the unknown being measured, so **corpus frames cannot be used as calibration positives
on the strength of anyone's belief that they match** — that would calibrate the instrument against
the answer we are trying to obtain.

The way out is to establish ground truth **once, expensively, on a tiny seed**, and then amplify it
mechanically:

* **Seed.** `K = 8` frames, established by a person looking at them and writing down, per clause,
  what is true. One of the eight is `episode_000135` — the clip the caption was written from, the
  only frame in the corpus whose match is asserted by provenance rather than by opinion. The other
  seven are drawn from strata the 40-row sheet did **not** sample, under a recorded seed, so no
  calibration frame is also a measured frame. Eight frames is roughly five minutes of human
  attention, and it is the only human adjudication the design requires of any candidate.
* **Amplification.** Every calibration item below is derived from a seed frame by a transformation
  whose effect on each clause is known by construction. The item's label is then a fact about the
  transformation, not an opinion about the image.

If a seed frame turns out **not** to match the prompt, that is not a problem for the design — it
becomes a natural negative with a known axis, and it is also the first data point of the
measurement itself. It should be recorded as such and reported.

### 3.2 The forty items

`C40`, forty items, three labelled classes, shuffled into the same file as the forty real rows and
judged in one session under one setting — T-041's blinding structure (a blinded sheet plus a
separate `key.json` the judging step never opens), for T-041's reason: *"the code did not look" is
a weaker guarantee than "the labels were in a different file"*.

| class | n | required answer | how it is built |
|---|---|---|---|
| **positive** | 15 | `match` | 8 seed frames unmodified; 7 seed frames under **null perturbations** — re-encode through the same ffmpeg path, exposure ±3 %, frame index ±1. No clause changes. |
| **negative** | 15 | `mismatch` + the named axis | 3 items on each of the five axes (§3.3). |
| **abstention probe** | 10 | `unsure` | Seed frames with the clause-bearing region made genuinely unjudgeable (§4). |

The null-perturbation positives are not padding. They equalise "has been through an image
processing step" across the positive and negative classes, so an instrument cannot pass by
detecting tampering instead of by reading the clause — which is the most likely way a synthetic
negative set gets gamed, and it is invisible in the score if you do not build against it.

### 3.3 The negatives, by axis

Each negative falsifies exactly **one** clause and leaves the other four intact, so the required
axis is unambiguous and a right-answer-for-the-wrong-reason is visible as a wrong axis.

| axis | image-side mutation | prompt-side mutation |
|---|---|---|
| `apple` | hue-rotate the apple region to green | swap "red and yellow apple" for "a green apple" |
| `table` | hue-shift / lighten the cloth to blue or beige | swap "black cloth" for "a blue-and-white checked tablecloth" |
| `background` | brighten the upper background band to near-white | swap "black background" for "a bright white background" |
| `lighting` | crush gamma and add a hard directional shadow | swap "even, bright, top-down lighting with minimal shadows" for "dim side lighting with long hard shadows" |
| `plate` | recolour the plate dark, or replace it with a square dark tray | swap "white, round plate" for "a small dark square tray" |

**Two mutation sides, deliberately.** Image-side negatives are unnatural pixels paired with the
real prompt; prompt-side negatives are **untouched real pixels** paired with a one-clause-falsified
prompt. An instrument that passes only the image-side items is detecting artefacts; one that passes
only the prompt-side items is reading the text and not the picture. Mix them roughly half and half
within each axis and report the split, because the two halves fail for different reasons and an
aggregate would hide which.

`plate` is in the set even though it is not one of T-040's four allowed variation axes, precisely
because it is not: the prompt asserts a white round plate, every style ends by holding the plate
fixed, and a plate disagreement in the **source** would be a worse finding than a different
tablecloth. An instrument that cannot see the plate cannot report that finding.

### 3.4 The pass rule, pre-registered, four numbers and never one

Fixed before the instrument is run, and reported as four numbers plus the leakage count. **All must
hold.**

```
positives          >= 14 / 15  answered `match`
negative tokens    >= 14 / 15  answered `mismatch`
negative axes      >= 13 / 15  named the required axis
abstention probes  >=  8 / 10  answered `unsure`
abstention leakage <=  1 / 30  `unsure` answers among the 30 decidable calibration items
```

The leakage floor is the one that is easy to leave out and expensive to leave out. Without it, an
instrument that abstains on everything it is not sure about passes the first four lines by
abstaining its way past the hard items, and then abstains through the real forty — where the
`--min-coverage 0.90` floor turns it into a run that is not gate-qualified, after the wall-clock
was spent.

### 3.5 The arithmetic: why a constant, and a coin, cannot pass

**Constant `mismatch` (the T-041 shape, in our vocabulary).** `check_fill` refuses a `mismatch` row
with no axis, so a constant classifier must also emit a constant axis. Scores:

| | positives | negative token | negative axis | probes | leakage |
|---|---|---|---|---|---|
| constant `mismatch` + one constant axis | **0 / 15** | 15 / 15 | **3 / 15** | **0 / 10** | 0 |

It fails on the very first line, with a zero. Aggregate over all forty items it scores 15/40 =
37.5 % — which is *exactly the number that looked like partial credit in T-041* — and it is never
reported, because the report is five numbers and the first one is `0 / 15`. **A zero in a class is
the signature of a constant classifier, and per-class reporting is what makes that signature
visible.**

**Constant `match`.** 15/15 positives, **0/15** negative tokens, **0/10** probes. Same aggregate,
15/40, and it fails on line two. Symmetry is the point: T-041's set was catchable only in one
direction because the rubric's abstention rule made `NO` the free answer.

**Constant `unsure`.** 0/15, 0/15, 10/10, and leakage **30 / 30** against a floor of 1. Fails three
of five lines.

**A coin flip — uniform over the three tokens, and uniform over the six axes when it says
`mismatch`.** Per item, P(a positive right) = 1/3; P(a negative right, token *and* axis) =
1/3 × 1/6 = 1/18; P(a probe right) = 1/3.

```
P(>= 14 of 15 positives)        = 2.16e-06
P(>= 13 of 15 negatives, w/axis) = 4.54e-15
P(>=  8 of 10 probes)            = 3.40e-03
P(<=  1 unsure in 30 decidable)  = 8.34e-05
joint (first three lines)        = 3.34e-23
```

Roughly one pass in 10^22 attempts, and that is before the leakage floor, which a coin flip fails
on its own with probability 1 − 8.3e-5. There is no number of retries anybody would run.

**A judge that reads colour but cannot abstain** — the realistic near-miss, and the one worth
naming because it *is* useful. It passes positives, negative tokens and negative axes, and scores
**0/10** on the probes. It fails, and the right response is not to lower the floor: it is to record
that this instrument's `unsure` is not meaningful, and therefore that its `match` on an occluded
frame is not either. That is a finding about the instrument, and it is cheap to have before forty
rows rather than after.

### 3.6 What C40 does not do

It validates the instrument on items whose truth was *manufactured*: one clause falsified hard, or
nothing falsified at all. The real forty are not like that. They are natural frames where the
honest answer may be "the cloth is black but this one looks charcoal", and no manufactured item
teaches anyone how the instrument behaves there. **Passing C40 is necessary and never sufficient**,
and the artifact should say so in those words rather than let a pass be read as a validated
instrument. The second defence against that is §5's last row: two instruments, and their
disagreements reported.

## 4. Abstention

**What `unsure` means.** "This frame cannot settle it." Not a soft mismatch, not a polite `match`.
The rubric already fixes the two canonical cases: the hand occludes the apple, or the frame is too
dark to judge the cloth.

**How the probes are built.** Take a seed frame and make one clause-bearing region genuinely
unjudgeable: composite the Dex3 hand over the apple, blur the cloth region past the point where
black and charcoal separate, or crush the exposure of the plate region to noise. Genuinely — a
probe that a careful person can still answer is not a probe, it is a hard positive, and it will
make an honest instrument look broken. The seed pass in §3.1 is where that is checked: the person
who established the seed frames should confirm that they cannot answer the probes either.

**How it is recorded.** `verdict` already does all of it, and none of it needs new code:

* its own count in `verdict_counts.unsure`, never folded into either of the other two;
* its own `abstentions` list in the evidence artifact, carrying the episode and the filler's note;
* **outside** the coverage numerator — `coverage = (match + mismatch) / rows`;
* `--min-coverage 0.90` means at most 4 abstentions in 40 before the run is stamped not
  gate-qualified, with the reason recorded and exit 3.

That last rule is worth reading correctly. A run that abstains on 12 of 40 is not evidence that arm
C is fine and it is not evidence that arm C is broken. It is a record that the frames chosen could
not answer the question, and the repair is a different frame rule or a different instrument, not a
lower floor.

**Abstention as an attack.** An instrument that abstains exactly where it would have been wrong
looks perfect on the rows it answers. C40's leakage floor is the guard: on 30 items whose answer is
known to be decidable, more than one abstention means the instrument abstains when it is merely
uncomfortable, and its silence on the real forty carries no information.

## 5. Candidate instruments, priced honestly

Wall-clock only; none of these includes the ~1 day of engineering to build the C40 harness itself
(the mutation script, the blinded interleave, the key file, the per-class scorer), which every
candidate needs and which is the same work for all of them.

| candidate | wall clock for C40 + 40 rows | what it costs, honestly |
|---|---|---|
| **A person looking at forty frames** | seed 8 frames ≈ 5 min; 80 items at 20–40 s ≈ 30–50 min; **≈ 1 h**, one person | The strongest instrument available and the only one that needs no new code — `verdict` reads a hand-filled JSONL today. But calibrating a human against ground truth a human established is partly circular: for a person, C40 functions as an **attention check**, not as validation. The real instrument-quality number is **inter-rater agreement**, which needs a second person (≈ 2 h total, Cohen's kappa over the 40 rows) and is the only way to find out whether "black cloth" means the same thing to two readers. |
| **A local VLM on the RTX 5090** | ~10 min of inference for 80 items, plus a model download and a day of harness work | 32 GB is enough for a 7–13 B vision model at 640×480. **This is exactly the shape of instrument that produced T-041's VOID**, which is an argument for C40, not against the candidate. Nothing suitable is on this workstation today (`~/models` holds Wan2.2-TI2V-5B and GR00T checkpoints only), so it needs a download the current session rules forbid. |
| **A hosted VLM** | ~5–15 min for 80 items | Cheapest to stand up and the only candidate that sends 80 frames of the corpus to a third party — a decision about data, not about instruments, and not one this file can make. It also cannot live inside `build_identity_prompt_sheet.py`: `test_build_sheet_names_no_judge_anywhere` fails the build on any transport or vendor token appearing in that file, deliberately, so a hosted judge is a **separate** script writing into the same blank field. |
| **A deterministic image statistic** — no VLM at all | seconds for a **census of all 402**; ~half a day to write | See §1: four of the five clauses are colour/luminance claims. Median hue/saturation in the apple mask (the SAM2 adapter for GEOM_TOL is already staged, `scripts/estimators/apple_sam2.py`), luminance percentiles over the cloth and background bands, largest bright ellipse for the plate, and a local-contrast spread as a shadow-harshness proxy. Validate it on C40's image-side negatives — a statistic that does not separate a hue-shifted cloth from a real one is not measuring that clause. **It answers a slightly different question**: not "does this sentence describe this frame" but "is the appearance *constant* across the 402" — which is what the TODO's own `action` sentence asks, over the whole corpus rather than over a sample of 40. It cannot decide whether "black" is a fair word for the cloth; it can decide whether episode 300's cloth is the same colour as episode 000135's. |
| **Two instruments, and their disagreements** | the sum of any two above | The only configuration that produces a number about the instrument as well as about arm C. A human over the 40 sampled frames plus the deterministic census over all 402 is the strongest pairing: they fail differently, the census covers what the sample cannot, and every episode where they disagree is worth a person's eye. |

Not a candidate: **re-deriving the prompt** by captioning all 402 clips with the T-041 captioner.
The TODO offers it as the first branch of its `action`, and it is the more expensive one — it needs
the same judge to decide whether 402 machine captions agree, so it does not avoid this problem, it
adds a generation step in front of it.

## 6. Sample size: what 40 buys over 402

The sheet is drawn; this section is about how to read what comes back.

**Detection.** Sampling 40 of 402 without replacement, if `M` episodes disagree with the prompt:

| M | share of corpus | P(the sheet catches at least one) |
|---|---|---|
| 4 | 1.0 % | 34.4 % |
| 8 | 2.0 % | 57.1 % |
| 20 | 5.0 % | 88.4 % |
| 40 | 10.0 % | 98.8 % |
| 80 | 19.9 % | > 99.9 % |

So the sheet reliably detects "a substantial minority of episodes look different" and is close to a
coin flip against "a couple of odd episodes". Which of those matters is the reader's call: arm C is
broken as a control by a systematic difference across sessions, not by two unusual episodes.

**Bounding, if the sheet comes back clean.** Zero mismatches in 40 is consistent with up to **27 of
the 402** (6.7 %) disagreeing at 95 % confidence — the exact hypergeometric bound, which confirms
the script's docstring: its rule-of-three figure of 3/40 = 7.5 % is conservative, as it says.
**This sample can detect inconstancy; it cannot certify constancy.** If the decision needs
constancy, the answer is a census, not a bigger sample — and by §5's fourth row a census is cheap
for exactly the clauses most likely to drift.

**One frame per episode.** Every verdict is about one instant, at 10 % into the clip. A light
switched mid-episode, or a sleeve entering later, is outside what this sample can see. Doubling to
two frames per episode doubles the judging cost and buys within-episode coverage; it buys nothing
on the across-episode question the TODO actually asks, so it is not the first upgrade to reach for.

**Do not re-draw.** `--allow-reseed` exists and refuses to overwrite a filled sheet for a reason:
drawing a second sample after seeing the first one's verdicts is sample-shopping and is invisible
afterwards.

## 7. What closing the TODO would take

In order, and none of it is automatic:

1. Pick an instrument from §5. Write down which, and why, before it runs.
2. Build C40 per §3 and the blinded interleave. Establish the 8-frame seed by hand.
3. Run the instrument over all 80 items in one session. Record the five calibration numbers.
4. **If it fails any of the five, stop and record the failure.** That is a VOID for the instrument,
   not a finding about arm C, and it is what T-041 did correctly.
5. If it passes, `verdict --sheet runs/t040-identity-prompt/` writes
   `configs/transfer25/pr08_identity_prompt_evidence.json` plus a pasteable TOML fragment.
6. A **person** reads the counts and decides whether they close the item, and edits
   `configs/transfer25/styles.toml`. The harness emits no `status` line and no overall pass, on
   purpose: an inconstant appearance **is** the finding, and arm C would then need a per-episode
   identity prompt rather than one shared string. Re-run
   `scripts/check_style_partition.py --write-hash --emit-json` afterwards — `blocking_todos` is
   inside the partition content hash.
