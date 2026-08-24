# PR-08 §4 — the robot detector's scores carry no information on this corpus, and the control is what says so

**Measured 2026-08-25 on the workstation RTX 5090, driving `scripts/estimators/apple_sam2.py`
unmodified at its committed operating point. No gate, config or blocker was touched.**

This document reports a **negative result that refutes the reading this session was about to adopt**,
and it is written that way on purpose: the finding is the control, not the hypothesis.

---

## 1. The hypothesis, and why it looked strong

`runs/pr08-blind-adjudication/` established that 41 of 52 frame-edge marks in the blind draw carry an
object that moves or appears over ±8 frames — on frames where the committed masker returned **no box
at all** (`empty_reason: no_boxes_above_threshold`, all 237 frames of the population).

So: does the detector *nearly* see them? `detector_readout` post-processes the same forward pass a
second time at threshold 0, precisely to tell *"nothing matched"* apart from *"matched weakly and got
filtered"*. Run over those 41 frames:

| statistic | value |
|---|---|
| min `raw_max` | 0.0849 |
| **median** | **0.1277** |
| max | **0.1487** |
| frames with `raw_max ≥ 0.15` | **0 of 41** |
| frames with `raw_max ≥ 0.10` | 34 of 41 |

Every one of the 41 sits **just below** `BOX_THRESHOLD = 0.15`, and the strongest is **0.0013** under
it. That is the exact signature of a threshold sitting on top of a real signal, and the obvious
reading is *"the detector does see the fingertip, and the cut removes it."*

**That reading is wrong.**

---

## 2. The control

Two control sets were drawn from the same empty-mask population and run through the same path:

- **Control A — no edge mark at all** (24 frames): tiles where the border-anomaly rule found nothing.
- **Control B — edge mark present but static** (11 frames): a mark that does *not* move, i.e. cloth.

| set | n | min | median | max | ≥ 0.10 |
|---|---:|---:|---:|---:|---:|
| **moving edge mark** (the hypothesis) | 41 | 0.0849 | **0.1277** | 0.1487 | 34 |
| control B — static edge mark | 11 | 0.1171 | **0.1325** | 0.1470 | 11 |
| control A — **no edge mark at all** | 24 | 0.0939 | **0.1211** | 0.1491 | 22 |

Median difference between the hypothesis set and the no-mark control: **+0.0066**.
Mann-Whitney U, one-sided: **p = 0.42**.

**There is no difference.** Frames with a moving object at the edge, frames with a static cloth fold,
and frames with nothing at all produce the same distribution of best-detector-score.

---

## 3. What this establishes

### 3.1 `raw_max ≈ 0.12` is the detector's NOISE FLOOR on this scene, not a weak detection

Grounding-DINO emits a best candidate of roughly 0.12–0.13 for
`"robot arm. robotic hand. robotic gripper."` on **every** frame of this corpus, whether or not
anything robot-like is present. `raw_n = 900` on every frame — 900 queries, always. The score
carries **no information** about robot presence at this end of the range.

### 3.2 Lowering the threshold would NOT recover the edge objects

This is the practically important consequence, and it runs opposite to the intuition in §1. A cut at
0.10 would admit a box on **34 of 41** hypothesis frames — and on **22 of 24** frames where the rule
found nothing at all. It would not find fingertips; it would manufacture a robot mask on almost every
empty frame indiscriminately.

**This retroactively supports the committed 0.15 operating point** and is a second, independent
reason not to tune it — the first being `T40_RULE_V1` §4 step 2's "the same segmenter" argument,
which forbids moving it for a different reason. Two arguments, same conclusion, arrived at
independently.

### 3.3 The margin above the noise floor is ~0.001, and that explains the hardware disagreement

Worst-case noise observed here is **0.1491** against a threshold of **0.15** — a margin of
**0.0009**. A detector whose noise ceiling is within one part in a thousand of its cut will flip
frames on any perturbation that moves a score by that much.

**That is a mechanism for a finding this project already had and could not explain.**
`runs/pr08-robot-mask-empty/DIAGNOSIS.json` records the same masker, same pins, same tree, same 1 603
frames disagreeing between an RTX 5090 and the cluster's H200 on the empty count by **13 of 1 603**
(0.81 %) while min/median/p95/p99 agreed to three decimals. Frames sitting within 0.001 of the
threshold are exactly the frames that would flip on a different GPU's kernels, a different cuDNN
version, or a different video decode. The disagreement is not a bug in either machine; it is a
threshold placed on a noise floor.

**Recorded as a mechanism consistent with the observation, not as a proven cause.** Establishing
causation would require re-running these same frames on an H200 and showing the flips are the
near-threshold ones. That has not been done.

### 3.4 The edge objects cannot be adjudicated by this detector, in either direction

Since the score is uninformative here, **this instrument cannot say whether those moving edge marks
are Dex3 fingertips or not.** It says only that the detector is blind to them. If they are the robot,
no threshold setting on this detector will catch them and the fix would have to be a different
mechanism entirely. If they are not, the detector's silence is correct for the wrong reason.

---

## 4. What this does NOT establish

- **Not that the edge objects are, or are not, the robot.** §3.4. The project owner's reading on
  2026-08-25 was that they *"could well be Dex3 fingertips, hard to make out, only the tips"*, and
  nothing here settles it.
- **Not a discharge of any `GATE_QUALIFICATION_BLOCKERS` entry.** Blocker 2's full-pass conjunct was
  supplied separately on 2026-08-25; blocker 1 needs a human on overlaid masks and is untouched;
  blocker 3 is per-frame-vs-propagation and is untouched.
- **Not a corpus rate.** 76 tiles from a stratified detect plan, not a corpus pass.
- **Not a determination on `T40_RULE_V12`.** V12 stays unsigned.
- **Not measured on the cluster.** RTX 5090, on the H.264-lossless tree — and §3.3 is precisely a
  warning that this machine and the H200 do not agree at this margin.

---

## 5. Why this is written as a refutation

The §1 reading is the one this session generated, and it is attractive: it would have made the 99.2 %
G0c refusal rate an instrument bug with a one-line fix. The control was run before the conclusion was
recorded, and it removed it.

Recording the refuted hypothesis alongside the control is the point. A reader who sees only
*"raw_max median 0.1277 against a 0.15 threshold"* will re-derive §1 and be wrong, and the only thing
that stops them is the control being in the same document.

---

## 6. Provenance

| | |
|---|---|
| kind | measurement report. **Registers no rule** |
| date | 2026-08-25 |
| hardware | workstation **RTX 5090** — see §3.3 before quoting any number against a cluster run |
| corpus | `pr08-apple-640x480-h264-lossless` |
| adapter | `scripts/estimators/apple_sam2.py`, **unmodified**, committed operating point |
| driven by | `scripts/diagnose_robot_mask_empty.py detect`, `--verify` passing on 6 frames against the real `Sam2RobotMasker.mask` |
| artifacts | `runs/pr08-blind-adjudication/EDGE_DETECT.json`, `DETECT_ctrlA_no_edge_mark.json`, `DETECT_ctrlB_static_edge_mark.json` |
| hypothesis | **refuted by its own control** |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched |
| generation licensed | **no** |
| training licensed | **no** |
