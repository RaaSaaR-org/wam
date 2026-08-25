# PR-08 §6 G0c — the empty-mask problem is not a prompt problem, and the nonsense control is what proves it

**Measured 2026-08-25 on the workstation RTX 5090. 180 distinct frames, 40 episodes, 6 phrase sets,
1 080 per-frame records. `ROBOT_TEXT_PROMPT` was NOT changed and nothing under `configs/`,
`docs/preregistration/`, `robot_composite.py` or `apple_sam2.py` was touched.**

**This is a clean negative and is written as one.** No candidate phrase set raised detection on the
frames that cost the yield without raising it at least as much on frames where a human has
adjudicated that no robot is present.

---

## 1. Why the question was asked

`robot_composite.py:318-324` says of its own constant:

> *"The phrase set's ADEQUACY on this corpus is unproven and the `measure` mode below is what would
> show it, frame by frame, before the first clip."*

`measure` has now run over the whole corpus (job 190413 + merge 190460) and the phrase set fails on
**57 835 of 171 625 frames (33.70 %)**. Because a failing frame raises `CompositeError` and
`restyle_transfer25.py:720` fails the whole unit, that refuses **366 of 402 episodes (91.0 %)** —
the binding constraint on this project's yield
(`PR-08-RESULT-2026-08-25-area-distribution-and-what-actually-refuses-clips.md` §0).

And the ordering mattered: `load_area_bound` refuses any bound measured under a different prompt
(*"Re-measure and re-decide"*), so a `max_frame_fraction` decided before this question is answered
is a decision thrown away by construction.

## 2. The design, because the controls ARE the result

| set | n | what it is |
|---|---:|---|
| **T_EDGE** | 41 | empty-mask frames carrying a **moving** frame-edge mark — the owner's *"could well be Dex3 fingertips"* population |
| **T_UNIF** | 60 | uniform draw from the 237 empty-mask frames — unbiased over the population that costs the yield |
| **C_ABSENT** | 24 | **primary control**: empty-mask frames the owner adjudicated **blind** as `arm_absent`, no edge mark. A human and two non-learned instruments agree nothing robot-shaped is there |
| **C_STATIC** | 11 | edge mark present but static — cloth |
| **C_POS** | 51 | the committed prompt already succeeds — guards against losing detections or inflating area |

Only the text handed to GroundingDINO varied. Thresholds, the highest-score box rule and the V9
object filter were untouched, and the committed arm was asserted **pixel-identical to
`Sam2RobotMasker.mask`** on 8 spread frames.

## 3. Detection rate — a non-empty final robot mask

| phrase set | T_EDGE | T_UNIF | **C_ABSENT** | C_STATIC | C_POS |
|---|---|---|---|---|---|
| **P0** committed | 0/41 | 0/60 | 0/24 | 0/11 | 51/51 |
| **P1** `+ robotic finger. fingertip.` | 12/41 (29 %) | 22/60 (37 %) | **10/24 (42 %)** | 3/11 | 48/51 |
| **P2** `robot.` | 0 | 0 | 0 | 0 | 34/51 |
| **P3** `black robotic gripper. black mechanical claw. metal robot arm.` | 41/41 | 60/60 | **24/24** | 11/11 | 51/51 |
| **P4** `mechanical arm. gripper. claw.` | 1/41 | 1/60 | 0/24 | 0/11 | 49/51 |
| **P5** *nonsense* `cat. bicycle.` | 0/41 | 1/60 | 0/24 | 0/11 | **37/51** |

### 3.1 P1 is the tempting one, and it is a new defect rather than a recovery

The fingertip phrasing recovers 29 % of exactly the frames the owner suspected. It also fires on
**42 %** of frames where a human said no robot is present — *more* than on the test set (Fisher
one-sided for test > control, **p = 0.90**).

What the masks actually are settles it. They are 52–188 px; **0 of 41 touch the 26 px border band**
where a fingertip would have to be; **0 of 41 contain a single moving-dark pixel**; and **41 of 41
lie entirely within 12 px of the apple's colour region**, centroid 0.54 apple-radii from its
centre. `area_frac_median` 0.0004, `border_frac_median` 0.0.

**P1 grounds "fingertip" on the apple's stem.** The sliver's IoU against the whole apple is far
below `ROBOT_MASK_OBJECT_MAX_IOU = 0.70`, so the V9 filter keeps it. That is not a recovered robot;
it is a **new defect class G0c would accept as a valid robot mask** and then composite the source
back through.

### 3.2 P3 detects everything, including the tablecloth

100 % on every set including C_ABSENT, at area fraction median **0.93–0.95** against a corpus
non-empty median of 0.104 and p95 of 0.253. 136 of 136 empty-set detections exceed 50 % of the
frame. It walks straight into the second mode job 106 found at 0.80–0.84 — the failure the area
bound exists to catch.

### 3.3 P5 is the control that ends the argument

`cat. bicycle.` grounds on the robot in **37 of 51 frames where the robot is verifiably present**,
median `pred_recall` 0.94 against the dark-robot predicate.

**When something salient is in shot, essentially any phrase grabs it. When nothing is, essentially
no phrase does.** The phrase is not what is doing the work.

## 4. Why: the phrase moves the noise floor, it does not create discrimination

Median threshold-0 best score, T_EDGE vs C_ABSENT, one-sided Mann-Whitney:

| phrase set | T_EDGE | C_ABSENT | Δ | p |
|---|---|---|---|---|
| P0 | 0.1277 | 0.1211 | +0.0066 | 0.417 |
| P1 | 0.1382 | 0.1352 | +0.0030 | 0.286 |
| P2 | 0.1098 | 0.1127 | −0.0029 | 0.821 |
| P3 | 0.3704 | 0.4063 | −0.0360 | 0.963 |
| P4 | 0.1061 | 0.1050 | +0.0011 | 0.802 |
| P5 | 0.0673 | 0.0699 | −0.0027 | 0.762 |

Every phrase set has a floor — 0.067 for nonsense up to 0.41 for the material words — and **not one
separates the two populations.** Changing the words slides the whole floor past a fixed 0.15 cut,
which is exactly why P3 fires on everything and P2 on nothing.

**`PR-08-RESULT-2026-08-25-detector-noise-floor.md` established this for one prompt. It now
generalises along the phrase axis.**

### 4.1 The instrument is calibrated against a known answer

The P0 row reproduces that committed document **exactly** — 0.1277 / 0.1211 / +0.0066 / p 0.417
against its 0.42 — through a path written independently of `detector_readout`. The agreement is a
check on this instrument, not a second finding.

## 5. What this closes

Two instrument-side routes out of the empty-mask problem existed. **Both are now measured and both
are negative:**

- **the threshold** — `PR-08-RESULT-2026-08-25-detector-noise-floor.md` §3.2: a cut at 0.10 admits
  a box on 34 of 41 target frames *and* 22 of 24 frames with nothing in them;
- **the phrase set** — this document.

**That leaves `T40_RULE_V12`'s rule question as the only remaining route**: what G0c should *mean*
on a frame that genuinely contains no robot. Not what the masker should be tuned to; what the rule
should say. V12 is unsigned, its §3.2 precondition is unavailable
(`PR-08-RESULT-2026-08-25-v12-preconditions.md`), and the blind (a)/(b) adjudication came back
inconclusive. **It is now the critical path, not a parallel concern.**

## 6. Caveats, stated rather than buried

- **Workstation RTX 5090.** Per the noise-floor document §3.3 this machine and the cluster H200
  disagree on frames within ~0.001 of the threshold, and P1's kept scores are 0.1505–0.1786 — inside
  that band. **P1's rate is not portable.** Its direction (control ≥ test) and its masks landing on
  the apple stem are not near-threshold artefacts.
- **Small samples.** P1 T_EDGE 29 % [18–44], C_ABSENT 42 % [24–61] — overlapping intervals. What is
  established is *"no candidate shows the required separation"*, not a precise rate.
- 76 of the frames come from a stratified detect plan, not a corpus pass. **No corpus rate is
  claimed here**; the corpus rate is job 106's.
- **P4's single T_EDGE hit** (`episode_000000` f84, 860 px, 64 % inside the border band, 13 %
  overlap with the moving-dark predicate) is the only detection in 1 080 records that looks like it
  landed on a real edge object. **n = 1. Not a result**, recorded so it is not lost.
- **Side finding for whoever needs it:** 9 of 60 frames that `detect_corpus.json` (2026-08-22)
  recorded as non-empty now return `all_boxes_dropped_as_object`. That is the post-2026-08-23 V9
  filter and is consistent with job 106, which includes V9 — but **any analysis still using
  `detect_corpus.json` as the empty/non-empty split is reading a pre-V9 masker.**

---

## 7. Provenance

| | |
|---|---|
| kind | measurement report. **Registers no rule** |
| date | 2026-08-25 |
| hardware | workstation **RTX 5090** — see §6 before quoting a rate against a cluster run |
| verdict | **negative: `ROBOT_TEXT_PROMPT` does not look fixable** |
| `ROBOT_TEXT_PROMPT` | **unchanged** |
| artifacts | `runs/pr08-prompt-fixability/` — `FINDINGS.json`, `PROBE.json` (1 080 records), `SUMMARY.json`, `RAWMAX_SEPARATION.json`, `SETS.json`, `SHEET_P1_P3.png`, `SHEET_P1_HITS.png` |
| calibrated against | `PR-08-RESULT-2026-08-25-detector-noise-floor.md`, reproduced exactly on the P0 row |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched |
| generation licensed | **no** |
| training licensed | **no** |
