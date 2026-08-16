# Where action labels come from

**Scope.** One question, asked twice in 2026-08 and answered from four different files each time:
a video corpus has pixels — where do the motor values come from? This doc is an index over
answers that already exist, not a new argument. Every claim below is a pointer.

**The one-line version.** Actions are *recorded*, or *carried over* from a recording, or *inferred
by a model that was itself taught on recordings*. No route manufactures labels from nothing, and
the one experiment that measured generated video as supervision lost to holding a frame still
(PR-06, 39 %).

> **A correction this doc exists partly to carry, 2026-08-15.** The standing framing in this repo
> — `backbone-eval.md` §4's probe design, and T-040's "the action decoder cannot supply them
> without circularity" — reads as *nothing infers actions from pixels, that route is closed*. That
> is right about **generated** video and about WAM's own decoder, and **too strong as a general
> claim**: Cosmos 3 ships an inverse-dynamics mode that does exactly this, and it is a legitimate
> route on **real unlabelled** footage. §3 is split accordingly — §3a the closed part, §3b the open
> one. Checked against the model card and cookbooks, not prompted by a result.

---

## 1. Recorded — the only source of real motor values

The LeRobot sources ship per-step `observation.state` / `action` alongside the mp4.
`scripts/convert_lerobot_g1.py` maps them into the canonical space
(`src/wam/interfaces/schema.py`: 15 joints + 2 grippers, `ActionMode.JOINT_DELTA`), relabelling
actions from executed states — `targets[t] = q[t+1] - q[t]`, gripper sampled *at* `t+1`. The two
channels are anchored differently by construction; one convention does not cover both
(`docs/handoff.md` §"Facts", where two mutants slipped through on exactly that).

`src/wam/data/episode.py` then writes `states.parquet` + `actions.parquet` next to the video with
sha256 in the manifest. That paired format is the dataset format. It exists and it works.

**The Cosmos corpus deliberately drops it.** `scripts/prepare_cosmos_corpus.py` emits
`train/videos/<uuid>.mp4` plus captions and nothing else, because a generator fine-tune consumes
(clip, caption) pairs — joint states play no part in it. T-041's own §"What G1 data it would need"
opens with **"Video, not actions."** That is not an oversight to be repaired; it is the right
input set for the thing it trains.

## 2. Carried over — restyle a recording, keep its labels

**T-040 / `docs/preregistration/PR-08-photoreal-augmentation.md`.** Cosmos-Transfer2.5 repaints
frames of an episode *that already has actions*; the labels come from the teleop, not the
generator. Appearance may vary, geometry may not — move the apple and the arm grasps empty air.

This is the only sanctioned route by which generated pixels enter a training corpus here, and it
is gated accordingly: G0b is a geometry-invariance check whose tolerance is *derived* (the median
per-step object-centroid displacement in the source) rather than coined, and G0c composites the
real robot back over the generated frame **unconditionally**, so the generic-manipulator defect
cannot enter and no IoU threshold has to be invented.

Status: pre-registration written 2026-08-06, generation blocked on T-39 by `PR-08` §1, and one
acceptance criterion still open (the `emai/vla-training` consumer contract). **T-39 reported
2026-08-16 — `VOID (labels)`, whose cause PR-12 (`C`) and PR-13 (`W`) traced to our evaluation
adapter rather than the corpus, so §1's stated reason is withdrawn by measurement. §1 is not
lifted here; that is the project owner's call.**

## 3a. Inferred from pixels, by *us* — closed, and priced twice

The tempting route, and the one that keeps getting re-proposed:

- **T-040 states the circularity directly:** *"A Predict-family dream has no action labels, and
  WAM's own action decoder cannot supply them without circularity: it is the negative result, not
  a labeller."*
- **T-36 / `PR-06-RESULT.md` measured it.** The anchored dream scored **16.656** from the truth
  where holding the conditioning frame scored **12.020** — 39 % *worse than standing still*.
- **The embodiment defect makes it worse than the numbers suggest.** Both probed priors render a
  generic manipulator where the G1's arm should be
  (`runs/backbone_eval/video/embodiment_grid.png`), so joint angles read off those pixels are
  angles of the wrong arm — and no pixel metric can see it, which is why PR-08's gate needed a
  non-distance embodiment check.

## 3b. Inferred from pixels, by Cosmos — open, and it needs labels to make labels

**Verified against primary sources 2026-08-15**, prompted by the fair objection that NVIDIA markets
Cosmos as a *world action model* — so why did T-041 train video only?

The marketing is accurate. Cosmos3-Super's generator lists its outputs as **text, image, video,
sound and action (JSON)**, and the family ships three action modes: **forward dynamics** (actions →
future frames), **inverse dynamics** (frames → the action trajectory that produced them), and
**policy** (observation + task prompt → action sequence). Inverse dynamics is a genuine
video-to-action labeller, and dismissing it as "actions are input-only" — as an earlier draft of
this file and of `backbone-eval.md` §4 both did — is wrong.

**Three facts bound what that buys us.**

1. **The action vocabulary is per-embodiment, and ours is not in it.** From the Super card:
   *"Input action is only supported for compatible embodiments, including general camera motion
   (9D), autonomous vehicle (9D), egocentric motion (57D), single Franka Panda arm with RobotiQ
   gripper (10D), dual Franka Panda arm with RobotiQ gripper (20D), Agibot (29D), UR (10D), Google
   robot (10D), WidowX 250 (10D), UMI (9D)."* **No G1 and no 28-dim Dex3.** An action is not
   a universal format — a 10D WidowX vector means ten specific things — so there is no G1 output to
   request. The shipped inverse-dynamics example is autonomous-vehicle only.

   > **Corrected 2026-08-15: this used to read "no humanoid, no G1, no 28-dim Dex3", and the first
   > third was wrong** — contradicted by the same sentence it sits in. **AgiBot is a humanoid, and
   > it is supported at 29D**, on Edge as well as Super. The bound that survives is "no G1, no
   > Dex3", and the difference is load-bearing: a supported **29D humanoid** is a much closer
   > neighbour to a **28D G1** than "no humanoid at all" implies. Whether that neighbourhood is
   > worth anything — a config entry versus a new action head — is
   > `subprojects/edge-wam/tasks/E-02-*.md`, and is not yet answered.

2. **Adding an embodiment costs the thing we are short of.** NVIDIA's route is action
   post-training: *"Developers can post-train Cosmos 3 on action-labeled data."* To make it emit G1
   actions you feed it G1 video **with actions already attached**. It does not create labels; it
   *amortises* labels you already have onto footage you have not labelled. That is a real and
   standard technique — and it means the ceiling is set by how much labelled G1 data exists, not by
   the model.

3. **It must not be pointed at generated video.** Labels inferred from generated frames describe
   pixels the generator invented, so a policy trained on them learns the generator's dream physics.
   §3a's 39 % is the measured version of that argument. This is why route 2 exists in the shape it
   does: restyle a *real* episode and keep its *recorded* labels, rather than generate a clip and
   infer labels for it.

**Where the recipes actually are** (`cookbooks/cosmos3/generator/action/`, listing checked
2026-08-15): twelve inference notebooks, **all Nano**, covering forward dynamics / inverse dynamics
/ policy across four backends. The only post-training recipe under `finetune/` is
**Nano-Policy-DROID**. The README notes Super also ships `action_gen=True`, so the capability is not
Nano-exclusive — but there is no Super notebook and no Super post-training recipe, and Super is a
121 GB export. **Nano is the realistic scale for this**, which is the same conclusion T-041 reached
from the cookbook filenames alone.

**The task this implies is T-042**, and its honest blocker is not the model: it is that nobody has
counted how much *unlabelled real* G1 footage we hold. With none, an inverse-dynamics post-train
labels nothing.

> **Counted 2026-08-15. The answer is zero, and T-042 is closed.**
>
> **Clips of real G1 footage we hold that have video, no actions, and no way to get them: 0.**
>
> The naive on-disk reading looked promising — 3 554 real G1 episodes / ~25.5 h across the 14
> sources, 3 163 clips in `cosmos-g1-embodiment`, and not one `actions.parquet` beside them. But
> they are unlabelled *only because two fetch scripts skipped a directory*:
> `92_fetch_g1_corpus.sbatch` passes `--include 'meta/*' --include 'videos/**'`, and
> `workstation/10_fetch_corpus.sh` narrows further to one camera. Upstream, all 14 repos publish
> the action parquets — **415 files, 647 MB total**, in the same Apache-2.0 and CC-BY-4.0
> repositories we already pulled 69 GB of video from. Verified through the HF tree API without
> downloading anything.
>
> So the premise fails. Building a labeller to recover labels that `--include 'data/**'` would
> download is not amortisation, it is reconstructing something we chose not to copy — and the
> recovered labels would be strictly worse than the recorded ones. The two outside-chance pools
> close the same way: `USC-PSI-Lab/humanoid-everyday` (8 949 eps) and `Humanoid-Everyday-G1`
> (4 064 eps) are both fully action-labelled upstream and **neither is gated** — the earlier
> "licence unresolved" worry resolves as *the parent is Apache-2.0, and the G1 subset declares no
> licence at all*, which is an account-holder question and not a labelling one. Even resolving it
> would hand us labelled data. (`$PROJ/hf_cache/.../Humanoid-Everyday-G1` holds **0 bytes** — the
> 2026-08-07 fetch 429'd and nothing was ever obtained.)
>
> The only pool that would make this task real is teleop recorded after M1/D2, and it does not
> exist yet. **Re-open T-042 the day teleop produces video faster than it produces labels** — not
> before.

### The thing the count actually found — 3 152 labelled G1 episodes we already had access to

Step 0 was asked a narrow question and returned a wider answer, so it is recorded here rather than
lost in a closed task. Of those 3 554 real G1 episodes, **3 152 are the 13 `unitreerobotics/G1_Dex3_*`
sets, every one declaring `action float32[28]` — the exact 28-dim G1 + Dex3 vocabulary** T-042 was
going to teach Cosmos from scratch. They are labelled, they are Apache-2.0, and the labels are
647 MB away.

Every recorded number in this project comes from **402 episodes of one task**
(`nvidia/GR00T-N1.7-AppleToPlate`, 43-dim). The standing explanation for fourteen negatives is
"402 success-only episodes of one task is not enough" (PR-07 §1), and PR-07 §6's **N** verdict
points explicitly at *"the **kind** of data — PR-04's collection spec"* as the next move. Thirteen
further G1 tasks with recorded actions bear directly on both, and nobody has to collect them.

**This is not a free win and must not be written up as one.** The 28-dim Dex3 vocabulary is not the
43-dim AppleToPlate one, `convert_lerobot_g1.py` targets canonical 15 joints + 2 grippers and reads
LeRobot v2.1 where these sets are v3.0, and this corpus carries **no waist column** where the
canonical space has one. A converter and its mutant tests are real work. But it is *conversion*
work on *recorded* labels, which is route 1, not route 3b. Tracked as **T-043**.

> **Correction, 2026-08-15 — the block-order trap was pointed the wrong way here.** An earlier
> version of this paragraph carried T-041's `action[0:14]` ↔ hand / `action[14:28]` ↔ arm finding
> across to these sets. **That finding is real but belongs to a different corpus**
> (`USC-PSI-Lab/Humanoid-Everyday-G1`, LeRobot v2.1, separate `arm_joints`/`leg_joints`/
> `hand_joints` fields). The `unitreerobotics/G1_Dex3_*` sets are **LeRobot v3.0 with a flat 28-dim
> state, and they are ARM-FIRST: `[0:14]` arm, `[14:28]` hand.** Measured 2026-08-15 from
> `meta/stats.json` of all 13 sets — **zero** one-sided dims in `[0:14]` against **4–10** in
> `[14:28]`, the latter railing at a clean 100.0° or 120.0° mechanical limit, which is a finger and
> not a shoulder — and independently confirmed by `vla-training/groot/modality_g1_dex3.json`, which
> declares `arms {0,14}` / `hands {14,28}` for this same convention. Both facts are true of their
> own corpus; carrying one across to the other would have produced precisely the silent
> transposition the warning exists to prevent. Detail and the open intra-block questions:
> **T-043 §1**.

## 4. The action *port* on Predict2 — an input, and only that

`docs/backbone-eval.md` §4 / T-37. This is the one people mean when they say "but Cosmos-Predict
does actions", so it is worth being exact about which direction the arrow points.

`nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned` takes actions **in**, through an
action-embedder MLP added to the DiT's timestep embeddings, and emits frames. Ours is a state
token bolted into a text-context slot and trained from scratch on 402 episodes; theirs was
pretrained with it. That is the whole of criterion S4, and the only property in the 2026-08
candidate sweep our record does not already cover.

The planned probe asks whether that pretrained conditioning is *linearly readable off the residual
stream* — arm A: port fed the true past-action chunk; arm B: port fed zeros; same weights, same
process, same windows. It is a question about feature quality for a downstream policy. **It does
not label video.**

Status, and the two things building it surfaced (`backbone-eval.md` §4a/§4b):

- **The CPU half ran; the GPU arm never did.** Gate calibration is in
  `runs/backbone_eval/action_baselines{,_ep24,_ep48}.json`.
- **The bar went up.** G2 originally read `joints > 0.456` (the state-only ridge), but arm A is
  *fed* past actions and T-34 measured lag-1 autocorrelation **0.927** — so the comparator had to
  be what a ridge gets from the probe's own inputs with no video model at all. That is
  `past_joint_proj + state`: **0.540 / 0.539 / 0.541** joints across three seeds at 48 episodes.
  The best frozen-backbone number ever recorded in this project is **0.4267** (T-38).
- **The representation the port eats is the weaker one on our data.** `past_ee` reads 0.4576 vs a
  0.4563 floor at 12 episodes and 0.3954 vs 0.5129 at 48 — it *degrades* with corpus size, the
  signature of a small-n result, and the 12-episode reading is the one that would have been
  quoted. `past_ee_pos` is worse still (0.4631 → 0.3474).
- **It is not diffusers-native**, so it needs the `cosmos-predict2` stack alongside a
  diffusers-based harness; and Bridge is a WidowX at 4 fps against our G1 at 30 fps with 1.6 mm
  mean per-frame EE displacement, so arm A is out of distribution on scale alone.

## 5. Cosmos3-Super, specifically — no port for our embodiment

T-041 checked this against the model card before the run, and PR-09 §9 repeats it as a bound on
the verdict:

> Super's card lists supported action inputs — camera 9D, AV 9D, egocentric 57D, Franka 10/20D,
> Agibot 29D, UR/Google/WidowX 10D, UMI 9D. **No humanoid, no G1, no 28-dim Dex3.** The
> action-conditioned SFT cookbooks are `..._nano.sh` only.

**The quote is left as written, and its "no humanoid" is wrong** (annotated 2026-08-15, see §3b
above): AgiBot at 29D *is* a humanoid. The bound PR-09 §9 actually needed — no G1, no Dex3 — holds,
so the verdict it qualifies is unaffected. Rules and records are versioned here, never edited in
place, which is why the correction sits beside the quote rather than inside it.

So the T-041 checkpoint is a **video** fine-tune and cannot become an action-conditioned G1 world
model by that route. Clips generated from it — including `$PROJ/runs/t041-apple-variations/` — have
no action labels and no valid way to acquire them, and this is the one place where the §3b
correction does **not** loosen anything: route 2 needs a real episode underneath to carry labels
over, a `text2video` sample has none, and §3b's third bound forbids pointing an inverse-dynamics
model at generated frames.

**One nuance §3b adds to the sentence above.** "No action port" is loose — Super *does* ship
`action_gen=True`, so the capability is present in the weights. What is absent is (a) our
embodiment in the supported list and (b) any Super post-training recipe to add it. The accurate
statement is: **Super has the machinery and not the vocabulary**, and nothing published tells us
how to teach it ours at that scale.

`docs/backbone-eval.md` §3 already recorded that Cosmos3 takes JSON action arrays in and emits
action states out, and that T-24 never touched that port — that line was right and was never
followed up. It is the one genuinely unexplored thread here, it is Nano-scale, and it is now
written down as **T-042** instead of sitting in a paragraph.

---

## Summary

| route | gives you | status |
|---|---|---|
| **1** teleop recording | real motor values | works today — `convert_lerobot_g1.py` → `episode.py` |
| **2** Transfer2.5 restyle | real labels on restyled pixels | PR-08 written, blocked on T-39 |
| **3a** our own decoder reads pixels | nothing trustworthy | closed — PR-06 lost to a frozen frame by 39 % |
| **3b** Cosmos inverse dynamics | labels on *real unlabelled* footage | **open** — needs an embodiment post-train (T-042) |
| **4** Predict2 action port | actions *in*; better features out | screen + gates done, GPU arm never run |
| **5** Super fine-tune | video only | done (T-041); machinery present, G1 vocabulary absent |

**The distinction that keeps getting lost.** Routes 1 and 2 *measure* actions. Route 3b *predicts*
them, from a model trained on measurements — so it can extend a labelled corpus over unlabelled
footage, but it can never exceed the labelling it learned, and it must never be aimed at frames a
generator invented. Nothing on this list turns pixels alone into supervision.

**Ordering, from T-041:** *"T-040 has to establish that restyled data helps at all with a frozen
generator before the expensive version of the same idea is worth pricing. If T-040 is null, this
is moot."* And T-040 is blocked on T-39 — until that reports, "the data is wrong" and "the method
is wrong" are not separable, and generating more data is a bet on the first.
