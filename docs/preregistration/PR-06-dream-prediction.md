# PR-06 — Does the video branch predict the future, when there is a future to predict?

Pre-registered 2026-08-05, **before any Wan sampling run with motion-selected windows or the
anchored arms**. The reference table in §3 was measured first, on CPU, and appears here rather
than in a result document because it is an input to the design, not an outcome.

Task: T-36. Code: `src/wam/evaluation/dream.py`, `scripts/dream.py`, `tests/test_dream.py`
(67 tests). Venue: the ZeroGPU Space's **dream** tab (`deploy/wan-smoke-space/app.py`).
Supersedes: `PR-05-dream-motion.md` §4 G2, which ran and was recorded VOID.

## 1. Two defects in PR-05, both mine

**D1 — the windows were the moments nothing happens.** Both window builders picked with
`np.linspace(0, n - 1, count)`. At the two-windows-per-episode setting PR-05 ran, `linspace`
returns exactly `[0, n - 1]` — the *first and last* chunk of every episode. On
GR00T-AppleToPlate those are the two moments the arm is not in frame: the apple sitting beside
the plate before the reach, and sitting on the plate after the withdrawal. `gt.png` from that run
shows it plainly — four clips, no robot anywhere, which is what prompted this pre-registration.

So PR-05 asked "does the dream move like the data" and answered it on windows where the data is
a photograph. Its **G1 = MOVES at ratio 1.046** is arithmetically correct and means far less than
it reads: *on the two moments the recorded data barely moves, the dream also barely moves.* Worse,
its §"Caveats" conclusion that "the corpus is nearly static" is partly an artifact of this
selection rather than a property of the corpus (§3).

**D2 — the discriminator had noise on both sides.** G2 compared `d(lora, base)` against
`d(base, base_seed1)` where `base` is the adapter-disabled prior, which at 9×128×160 produces
colour noise (motion 22.5, pixel std 88.3, static fraction 0.000). Two independent noise draws
differ from each other more than a clean image differs from either, so the gate measured the base
arm's variance and fired `INDISTINGUISHABLE`. Recorded VOID in `PR-05-RESULT.md`; not re-run here.
`--base-null` now gates it off by default so it cannot fire by accident.

## 2. What changed in the code, before the run

| change | where |
|---|---|
| `--window-select {linspace,motion}` on both window builders | `scripts/dream.py`, `scripts/hf_job_wan_probe.py` |
| `select_windows_by_motion` / `window_motions` | `wam.evaluation.dream` |
| `freeze_baseline` — frame 0 held, built from `recon` | `scripts/dream.py` |
| anchored arm group (`recon_a`, `freeze_a`, `lora_a`, `lora_a_seed1`) | `run_arms` |
| `freeze_gate` verdict + `FREEZE_MARGIN = 0.9` | `wam.evaluation.dream.build_report` |
| `base_seed1` made opt-in so PR-05's void verdict cannot fire | `--base-null`, `dream_pairs` |

Selecting windows by motion is **not a random sample of the corpus** and is not reported as one.
It is a deliberately chosen subpopulation — "the moments the robot is acting" — which is the
regime the video branch exists to model. It cannot bias the gates: every arm sees the same
windows and is divided by / compared against the round-trip of those same windows.

## 3. The reference, measured (2026-08-05, CPU, no Wan weights, no GPU)

`scripts/dream.py --gt-only`, 8 episodes × 2 windows of 9 frames from `datasets/gr00t-apple-full`
at the recorded 120×160. Same command, same episodes, both selectors:

| quantity | `linspace` (PR-05) | `motion` (PR-06) |
|---|---|---|
| clips surviving the clamped-window drop | 8 of 16 | **16 of 16** |
| **gt motion** | **0.165** | **4.034** |
| pixel std | 48.5 | 56.3 |
| static fraction (pairs moving < 1/255) | 0.953 | **0.008** |

A **24× difference in the thing under test**, and the arm is in frame in all 16 motion-selected
clips (checked by eye on the contact sheet: hand closed on the apple, carrying it to the plate).

Per 9-frame window over the same 8 episodes, for scale: first+last average **0.071**, all windows
average **1.107**, the peak window of each episode averages **5.183**. PR-05 sampled ~15× quieter
than the episode average. Half its windows were dropped as bit-identical before it even measured.

**This corrects PR-05's corpus claim.** The corpus is *mostly* static — most 0.3 s windows of a
13 s episode are dead time — but it is not static: while the robot acts it moves ~1.1–5.2, not
~0.17. The archived reading and PR-05's re-reading were both measuring dead time.

## 4. Arms

Two groups. They answer different questions and a number may not cross between them.

**Free** (start from noise on text + state alone; 9 frames):

| arm | what it is |
|---|---|
| `gt` | the recording, on its own 120×160 grid — context only, never a denominator |
| **`recon`** | `decode(encode(gt))` — the VAE round-trip, what every ratio divides by |
| **`lora`** | the dream: WAM's flow, text **and state** context, 32 Euler steps, seed 0 |
| `base` | adapter disabled, same weights in memory — sanity, no gate reads it |

A free sample is **not** a future prediction: nothing tells it where the apple is except the
proprioception token, so it may not be scored against the observation pixel-for-pixel.

**Anchored** (latent frame 0 pinned to the observation, so the clip *is* a prediction of what
follows it; 8 frames after `strip_anchor` removes the copied pixel frame):

| arm | what it is |
|---|---|
| **`recon_a`** | what actually happened next — the target |
| **`freeze_a`** | `recon` frame 0, held. The trivial predictor: "nothing changes" |
| **`lora_a`** | the prediction, seed 0 |
| `lora_a_seed1` | the same prediction, seed 1 — PR-05's open caveat, closed |

`freeze_a` is built from `recon`, not from `gt`, so both sides of G2 carry the same codec and
resize loss and the comparison isolates predicted motion.

## 5. Gates — fixed here, before the run

**G1 (motion, free arms).** `motion_ratio(lora, recon) >= 0.5` → `MOVES`, else `STATIC`.
Unchanged from PR-05 (`MOTION_FLOOR_RATIO`, already in git); what changed is the windows it is
evaluated on. This is a re-run of PR-05 G1 in the regime it should have been run in, and its
result **replaces** PR-05's, which is not withdrawn but is re-read as covering dead time only.

**G2 (prediction, anchored arms).** `d(lora_a, recon_a) < 0.9 · d(freeze_a, recon_a)` →
`PREDICTS`, else `NO_BETTER_THAN_FREEZING`. `FREEZE_MARGIN = 0.9` is in git before the run.

Why a margin: on this corpus freezing is a *strong* baseline, so a 1 % win is noise dressed as a
finding. Why this baseline at all: a world model that cannot beat "nothing happens" over 0.27 s
has not been shown to model anything, and unlike PR-05's G2 this is a comparison whose two
operands are both clean images of the same scene.

**Stability.** `lora_a_seed1` is scored against `recon_a` too. If the two seeds fall on opposite
sides of the threshold, G2 is recorded as **UNSTABLE** and no verdict is claimed. Fixed here so
it cannot be waived after seeing the numbers.

## 6. Reading the outcome

| G1 | G2 | reading |
|---|---|---|
| MOVES | PREDICTS | the video branch models this task's motion. The strongest result available from one checkpoint, one geometry, 16 clips — and still not a policy result |
| MOVES | NO_BETTER_THAN_FREEZING | it generates motion of the right *magnitude* without predicting the right motion. The most likely outcome given the action branch loses to repeat-last-action by 21.80 % |
| STATIC | NO_BETTER_THAN_FREEZING | it learned to stand still — the archived 2026-07-30 reading, now on windows that can actually distinguish it |
| STATIC | PREDICTS | contradictory; treat as a bug in the anchoring or the strip, not a finding |

## 7. What this cannot answer

- **Not a policy result.** The checkpoint reaches L0 and loses to repeat-last-action (−21.80 %).
  Nothing about the video branch changes that; `dream.py` never touches the action head.
- **Not training data.** PR-04's finding stands: the next corpus needs failures, randomized
  placement and an unfrozen right hand, none of which a generator fitted to 402 success-only
  episodes can invent. The honest form of that question is `screen_corpus.py` on a *generated*
  corpus, and it stays unrun.
- **Not a corpus statistic.** Every number here is on motion-selected windows. Anyone quoting
  `gt motion 4.034` as "the corpus" repeats D1 in the other direction.
- **16 clips, one checkpoint, one geometry, two seeds on one arm.** A ratio is safe because
  numerator and denominator come from the same clips; no absolute number here generalizes.
