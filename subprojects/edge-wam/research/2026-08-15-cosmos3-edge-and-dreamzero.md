# Cosmos 3 Edge and DreamZero — what the primary sources actually say

**Date:** 2026-08-15 · **Sources:** NVIDIA Cosmos repo, `nvidia/Cosmos3-Edge` model card, NVIDIA
developer blog, DreamZero project page + GitHub + arXiv 2602.15922.

**Why this exists:** the proposal is to split WAM into (1) a Cosmos3-Super **data factory** and
(2) an on-robot **edge WAM** that takes an image and emits an action. Both halves turn on facts
about models neither of which this repo has used. This document is the primary-source pass, written
before any task file, so the tasks are not built on a recollection.

Marking follows `docs/discoverer.md`: **[✓]** read off a primary source (model card, repo, paper),
**[doc]** stated by NVIDIA marketing/blog or a secondary outlet, **[?]** open.

---

## 1. The headline: the edge sub-project already has a model

**`Cosmos3-Edge` is real, it is 4B, and it runs on a Jetson** [✓]. This was the single biggest
unknown in the proposal, and it resolves in favour of the plan.

| | Cosmos3-Edge | Cosmos3-Nano | Cosmos3-Super |
|---|---|---|---|
| parameters | **4B** | 16B | 64B |
| target | **Jetson AGX Orin, Jetson Thor**, RTX PRO 6000 | RTX PRO 6000, H100, B200 | H200, B200, GB200 |
| policy variant | **`Cosmos3-Edge-Policy-DROID`** | `Cosmos3-Nano-Policy-DROID` | — (`action_gen=True`, no recipe) |
| sound output | no | yes | yes |
| video-to-video transfer | **not supported** | yes | yes |

Reported edge performance: **32 actions per inference at 15 Hz real-time control on Jetson Thor**,
at a robot-control resolution of **640×360** [doc — NVIDIA blog and secondary coverage; the model
card instead shows *action chunk size 16* and *20 fps* in its example configs [✓], so the two
numbers disagree and neither is yet reproduced by us].

Licence: **OpenMDW-1.1** [✓], released 2026-07-20 [✓]. Hardware: Ampere / Hopper / Blackwell,
Linux only [✓]. Tested platforms include Jetson Thor T5000/T4000/T3000/T2000 and Jetson AGX Orin
[✓]. The deployment story NVIDIA tells is "fine-tune on a small H100 cluster or a DGX Station, then
deploy to the Jetson" [doc] — which maps onto Discoverer+ for training and a Jetson at the robot.

**Consequence for the plan:** the edge sub-project is plausibly a *post-training* job on an existing
4B checkpoint, not a from-scratch model. That is a very different scope from what "build an edge
WAM" sounded like.

## 2. The catch: NVIDIA's "WAM" is *defined* by the video backbone

The proposal describes the edge half as "a WAM without a VLA — image in, action out, no video."
The first half is exactly NVIDIA's framing. The second half is not.

NVIDIA's definition [✓ blog]: a WAM is *"a policy built on a video world model"*, as opposed to a
VLA built on a vision-language model. Their argument is that VLMs are *"optimized to produce text
about images, not to model how a scene will evolve"*. And at inference:

> "the policy can also output a video: what the robot's cameras will see if those actions are
> executed. The action and the predicted outcome come from the same model, at the same time."

So video prediction is **not** merely a training-time objective that gets stripped for deployment —
it is the backbone, and the predicted video is a *co-product* of the same forward pass.

**This is good news, not bad.** It means "image in, action out" is available as an *interface*
choice — you simply don't decode the video head — while the world modelling that motivates the
whole project stays intact, on-device, at 15 Hz. What is *not* available is "a WAM with the world
model removed": remove it and NVIDIA would call the result a VLA.

**Open, and it matters:** `Cosmos3-Edge-Policy-DROID` generates actions *"given language
instructions and visual observations"* [✓ model card]. The released policy is language-conditioned.
The base model can run image-only for image-to-video [✓], but whether the **policy** path can run
with no text (empty/constant instruction, or post-trained without it) is **[?]** and is the single
most important unknown for a "no-VLA" edge design. It should be the first thing a probe answers.

## 3. Embodiments — and a correction to our own docs

Ten embodiment types with action dimensionality [✓ model card]:

| embodiment | dims | | embodiment | dims |
|---|---|---|---|---|
| general camera motion | 9D | | Agibot | **29D** |
| autonomous vehicle | 9D | | UR | 10D |
| egocentric motion | 57D | | Google robot | 10D |
| single Franka + Robotiq | 10D | | WidowX 250 | 10D |
| dual Franka + Robotiq | 20D | | UMI | 10D |

**There is still no Unitree G1 and no 28-dim Dex3 entry** — the existing bound in
`docs/action-labels.md` §3b holds, and adding one still requires post-training on action-labelled
data.

**But our wording is wrong in one respect.** That note says *"no humanoid/G1/28-dim Dex3 in the
supported vocabulary"*. **AgiBot is a humanoid, and it is supported at 29D** [✓]. The accurate
statement is "no G1 and no Dex3", not "no humanoid" — and the difference is load-bearing, because a
supported 29D humanoid is a far closer starting point for a 28D G1 than the sentence currently
implies. `docs/action-labels.md` should be corrected.

## 4. DreamZero — the strongest evidence for the joint thesis, and the strongest caution

**DreamZero: World Action Models are Zero-shot Policies** (NVIDIA, arXiv 2602.15922) [✓]. Code
under **Apache-2.0** at `github.com/dreamzero0/dreamzero` [✓].

Abstract claims [✓]:

- A WAM on a **14B autoregressive video diffusion backbone**, jointly modelling video and action.
- **">2x improvement in generalization to new tasks and environments compared to state-of-the-art
  VLAs in real robot experiments."**
- **Real-time closed-loop control at 7 Hz**, via model and system optimizations.
- Cross-embodiment transfer from **video-only demonstrations** from other robots *or humans*:
  **>42% relative improvement** on unseen tasks from **10–20 minutes** of data.
- Few-shot embodiment adaptation with **30 minutes of play data**, retaining zero-shot
  generalization.

Released artifacts [✓]: `DreamZero-DROID` (14B) and `DreamZero-AgiBot` (~45 GB, meant as the
starting point for post-training on new embodiments); LoRA and full fine-tuning scripts; training
recipes for new embodiments (AgiBot, YAM).

**The caution is the hardware.** Inference is documented as a **multi-GPU setup, minimum 2 GPUs,
tested on GB200 and H100** [✓], at **~0.6 s per prediction on GB200 and ~3 s on H100** [✓]. The
"7 Hz real-time" is achieved by *action chunking* — `action_horizon: 24` [✓] — the robot executes a
chunk while the next is computed. So 7 Hz is a control rate, not an inference rate, and it rests on
datacenter silicon.

**That is a direct confirmation of the proposal's premise:** a 14B joint WAM needs two H100s or a
GB200 and cannot go on a robot. It is precisely why a 4B Edge variant exists.

Also note DreamZero requires **both visual and language input**; image-only operation is not
indicated [✓]. Its embodiments are DROID, AgiBot and YAM — again, no G1.

## 5. The tension this project has to face

Our own clean same-backbone ablation says the world branch **costs 108 pp** —
`t18-real-ablation-seed0` at −129.00 % versus `d1-full-gen-seed0` at −20.88 %. NVIDIA and DreamZero
say joint video+action modelling is what *beats* VLAs, by >2x on generalization.

Both can be true, and the reconciliation is the interesting part:

1. **Scale.** DreamZero is a 14B video-diffusion backbone with large-scale robot pretraining. Our
   world branch was measured on a Wan 5B LoRA over a 48-episode corpus. A world model that has not
   seen enough world is a cost, not a prior.
2. **Pretraining.** The PRD's "no foundation pretraining, adopt-and-finetune" decision is intact —
   but DreamZero's result is *mostly a pretraining result*. Adopting Cosmos3-Edge (already
   pretrained on multi-domain action data) is the version of that decision that gets the benefit.
   The one ablation NVIDIA does publish is exactly this: two DROID policies, same recipe, data and
   compute, **36.8 % from an omni checkpoint vs 28.1 % from base — +8.7 pp attributed to
   initialization, not scale** [doc].
3. **What we actually measured.** Our 108 pp is a statement about *our* world branch on *our*
   corpus. It bounds our implementation; it does not refute the architecture class. It should stay
   on the record exactly as measured, and stop being cited as evidence about WAMs in general.

**Neither claim is safe to act on until T-39 runs.** T-39's `oracle_action` arm asks whether this
corpus's own action column clears L1 under our scorer. If it does not, both sub-projects are
building on labels that cannot support a policy, and no amount of backbone quality fixes that.

## 6. What this means for the two sub-projects

**Edge WAM** — much more tractable than assumed. It is post-training `Cosmos3-Edge` (4B, OpenMDW-1.1)
on G1 data, training on Discoverer+, deploying to a Jetson. It gets world modelling for free, at
15 Hz, without paying for it at the interface. Three things must be established first: (a) can the
policy path run without language, (b) what does adding a 28-dim G1/Dex3 embodiment actually require,
(c) do the 15 Hz / 32-action figures reproduce, given the model card says 16 / 20 fps.

**Data factory** — unchanged in direction but must stay on the legitimate route: augmenting **real**
episodes so real action labels survive (T-040 / Transfer2.5), not synthesizing episodes and
inferring labels, which `docs/handoff.md` §3 closed. One new wrinkle: **Cosmos3-Edge does not
support video-to-video transfer** [✓], so restyling work belongs to Super or Nano, not Edge — the
two sub-projects need different variants, which is itself an argument for the split.

**And one genuinely new option.** DreamZero's video-only cross-embodiment transfer (+42 % from
10–20 min of human or robot video, no action labels) is a *third* route that neither sub-project
currently has a task for, and it is adjacent to T-042. Worth a task of its own rather than being
folded into either half.

---

## Open questions, ranked

1. **[?]** Can `Cosmos3-Edge-Policy` run image-only, with no language instruction? Decides whether
   "no VLA" is a configuration or a re-training job.
2. **[?]** What exactly is required to add a 28-dim G1/Dex3 embodiment — a config entry, or
   post-training with a new action head?
3. **[?]** 32 actions @ 15 Hz (blog) vs chunk 16 @ 20 fps (model card) — which is real on which
   Jetson, and at what resolution?
4. **[?]** Are Cosmos3-Edge post-training scripts actually shipped? The blog and secondary coverage
   say yes; the model card says users are directed to "the Cosmos Framework" instead.
5. **[?]** Jetson VRAM headroom for a 4B policy at 640×360 — the card gives a 16–80 GB range across
   the tier, which is not an answer.

## Sources

- <https://github.com/nvidia/cosmos>
- <https://huggingface.co/nvidia/Cosmos3-Edge> · <https://huggingface.co/blog/nvidia/cosmos3edge>
- <https://developer.nvidia.com/blog/beyond-vlas-how-world-action-models-reshape-robot-manipulation/>
- <https://dreamzero0.github.io/> · <https://github.com/dreamzero0/dreamzero> ·
  <https://arxiv.org/abs/2602.15922>
- Secondary: MarkTechPost, 2026-07-21 (Cosmos 3 Edge release coverage)
