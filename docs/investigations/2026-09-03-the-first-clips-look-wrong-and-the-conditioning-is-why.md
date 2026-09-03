# The first licensed clips look wrong, and it is the conditioning, not the model

**Opened by the project owner on 2026-09-03, looking at the clips of the released chunk:** *„die
videos sind richtig schlecht … manchmal 'glitcht' die apfelfarbe (rot zu grün). und einmal ist sogar
der teller ein tellergroßer apfel. kann es sein, das wir auf einem holzweg sind und cosmos3 das
schon nicht kann?"*

**The observation is correct, the reading is not.** The generator is Cosmos-Transfer2.5 and it ran as
pinned. What produced these pictures is **our inference configuration**, and every defect below has
a named parameter behind it that this repository leaves at a default it never chose.

**This document changes nothing.** No parameter was altered, no job resubmitted, no rule touched.
`T40_RULE_V1` §1, `PR-08-DET-2026-09-02` and the signed `PARTITION_CEILING_GPU_H = 2013.75` all stand
exactly as they were. It is a findings page, and §6 says why acting on it is not a session's call.

---

## 1. It is Cosmos-Transfer2.5, and it ran as specified

Job `192149`, log line 208:

```
=== restyle: 404 units | set=train | controls=depth:0.5,seg:0.5 |
    resolution 640x480 -> bucket '480' aspect '4,3' | backend=transfer25
```

Pin `nvidia/Cosmos-Transfer2.5-2B@ce844032…`, 35 denoising steps, guidance 3, five chunks per clip.
No placeholder backend, no fallback. **Whatever is wrong is not that the wrong thing ran.**

**And there is no earlier, better clip to compare against.** Checked: the timing run kept none, and
`runs/t040-transfer25-restyle-timing-2026-08-28/` and `runs/t040-transfer25-restyle/` contain zero
`.mp4`. Any memory of better-looking output is a memory of NVIDIA's published demos, which are 720p.

## 2. Four causes, each with its parameter

### 2.1 The source pixels never enter the diffusion — `sigma_max`

This is the one that explains most of it. In
`cosmos_transfer2/_src/transfer2/inference/inference_pipeline.py`, inside the chunk loop:

```python
if sigma_max is not None or guided_generation_mask is not None:
    x0 = self.model.encode(...)                       # the input video, encoded
    if sigma_max is not None:
        x_sigma_max = self.model.get_x_from_clean(x0, sigma_max, seed=(seed + chunk_id))
```

`scripts/restyle_transfer25.py`'s `build_sample()` does not set `sigma_max`, and
`InferenceArguments.sigma_max` defaults to `None` (`config.py:499`). So `x_sigma_max` stays `None`
and **the denoise starts from pure noise.**

`_get_data_batch_input` (`:205-262`) makes the same point from the other side. The key the model
reads as its video is **`prev_output`**, not the source:

```python
data_batch = {
    input_key: prev_output.squeeze(2),     # <- the PREVIOUS CHUNK's output
    ...
    "input_video": video,                  # <- the source, used only below
}
```

and `"input_video"` is consumed only by `get_augmentor_for_eval(..., output_keys=hint_key, ...)`,
i.e. **to derive the control maps**. **The source frames are an input to an estimator, never a
starting point.**

**Consequence, stated plainly:** depth and segmentation carry no colour and no texture. The apple's
redness is therefore nowhere in the conditioning except the text prompt. *Colour drift is structural
here, not a glitch.*

### 2.2 A new scene decision every ~3 seconds — `num_video_frames_per_chunk` / `num_conditional_frames`

`config.py:493-501`:

```python
num_conditional_frames: Literal[0, 1, 2] = 1
num_video_frames_per_chunk: pydantic.PositiveInt = 93
```

A 424-frame episode is **five chunks**, and each is anchored on **one single frame** of the previous
one. Two further sources of per-chunk variance sit in upstream and are not ours to blame on the
config, but they compound it:

```python
random.seed(seed); seed = random.randint(0, 1000000)     # :639, INSIDE the chunk loop
"fps": torch.randint(16, 32, (self.batch_size,)).cuda(), # :235, random FPS conditioning
```

The per-chunk seeds are visible in the log — `123160 → 190112 → 234754 → 345632 → 606144` for one
clip. **`num_conditional_frames=2` is the maximum upstream allows and we pass 1.**

### 2.3 We generate below the model's own default resolution — `resolution`

`InferenceArguments.resolution` defaults to `"720"`. We pass `"480"`. From
`VIDEO_RES_SIZE_INFO` at aspect `4,3`:

| bucket | pixels |
|---|---|
| `"720"` (upstream default) | **960 × 704** |
| `"480"` (ours) | 640 × 480 |

That is **44 %** of the pixel count. It shows as mushed wood grain and washed-out background.
`keep_input_resolution=True` — which we already pass — resizes the output back to the input
resolution, **so generating at 720 and delivering 640×480 is available and costs only GPU time.**

**And the control weights themselves were trained at 720p.** Read off the object after a successful
unit rather than restated from what we passed in — `_raw/episode_000116__train-01-oak-tungsten__r00/
sample_outputs.json`:

```
edge_720p_t24or1_spaced_layer4_…/checkpoints/iter_000032000
vis_720p_t24or1_spaced_layer4_…/checkpoints/iter_000036000
depth_720p_t24or1_spaced_layer4_…/checkpoints/iter_000044000
seg_720p_t24or1_spaced_layer4_…/checkpoints/iter_000043000
```

Every control adapter is a **`_720p_`** checkpoint, and we drive all four at 640×480.

**Two things follow, and the second is free.** First, this is direct evidence for the resolution
argument above: the conditioning blocks are being run below the resolution they were trained at.
Second — **`edge` and `vis` are ALREADY LOADED**, at weight 0, because upstream's multi-control
branch loads every modality deliberately (*„load ALL control modalities even if some have control
weight = 0"*). **So moving weight onto `edge` costs no extra checkpoint load and no extra VRAM.
The block is already resident and idle.**

### 2.4 The prompt promises what the conditioning cannot enforce

Every train style ends with the committed invariance clause:

> *„The white plate keeps its own appearance. Scene geometry, camera framing and the robot are
> unchanged."*

That is prose. Nothing in `depth:0.5,seg:0.5` holds it. The dominant noun in each prompt is *apple*,
so a round disc in the segmentation map is an invitation — and in
`episode_000116__train-04-slate-lowkey__r00` and `episode_000120__train-04-slate-lowkey__r00` the
model paints a **plate-sized apple**, which is the artefact the owner reported. In
`train-03-melamine-fluorescent` it relocates the scene to a different room.

**One part of the report is by design and must not be "fixed".** `train-01`'s prompt asks literally
for *„A bright green Granny Smith apple"* while the source apple is red. **Red → green is the
augmentation axis**, committed in `configs/transfer25/styles.toml` and hashed. What is *not* by
design is that the colour is unstable *within* one clip.

## 3. We are running against NVIDIA's own recipe for this exact job

The cookbook page *Multi-Control Recipes with Cosmos Transfer 2.5 → real augmentation* is our use
case by name. Its recommended weights:

| recipe | edge | seg | vis |
|---|---|---|---|
| Background Change | **1.0** | 0.4 | 0.6 |
| Lighting Change | **1.0** | — | 0.2 |
| **Color / Texture Change** | **1.0** | — | — |
| Object Change | 0.2 | 1.0 | 0.5 |

**Every recipe leads with `edge`**, the modality the vendor describes as preserving subject
structure. Recipe 3 — *Color / Texture Change*, which is literally "different apple, different table
material" — is **edge 1.0 and nothing else**.

**We pass `depth:0.5,seg:0.5` and do not use `edge` at all.** Two consequences:

1. **Nothing in our conditioning holds edges**, which is §2.4's mechanism.
2. `97_transfer25_restyle.sbatch:371` already records that *„our manifest carries no depth or
   segmentation maps, so each control block Transfer2.5 has to ESTIMATE is GPU time the measurement
   must include."* Confirmed against the manifest: an episode entry carries only `frames`, `id`,
   `video`. So **we pay estimation time for the two modalities the vendor does not recommend for
   this job, and skip the one it does** — and `edge` is derived on the fly, i.e. it is the cheap one.

**Why `depth:0.5,seg:0.5` is nonetheless not a mistake anybody made carelessly.**
`restyle_transfer25.py`'s second refusal is explicit that the conditioning is uncommitted in
`styles.toml` and must not be defaulted, *„so picking it after looking at clips is the same failure
the style partition exists to prevent."* The value came from PR-08's committed set and it is the set
the throughput was measured under. **That reasoning is intact. What is new is evidence that the set
is wrong for the task, and evidence is what §6 says is allowed to move it.**

## 4. What this is not

* **Not a wrong model family.** Cosmos 3 with `sigma_max` unset, at bucket 480, one conditional
  frame and no edge control would produce the same mush. See
  [`../cosmos3-vs-transfer25.md`](../cosmos3-vs-transfer25.md) §5.2 for why a migration is a rule
  change with its own costs, and note its §5.2.3: **Cosmos 3 ships no depth estimator and no
  segmenter**, so for our map-less manifest it is a regression on this axis, not a fix.
* **Not a fault in G0c.** The real robot's pixels are composited back over every generated frame and
  it shows: the arm is sharp and correct in all 424 frames of the clips inspected, against a
  hallucinated background. G0c is doing exactly its job.
* **Not a label problem.** The actions are the recorded teleop trajectory, carried over unchanged
  (PR-08 §2). Nothing in this document touches them.
* **Not yet a measured verdict.** Everything above is eyes on frames plus source reading. **G0b is
  the instrument that turns it into a number**, and it has not run on these clips.

## 5. The measurement that should decide this, and has not been taken

G0b's live budget is `GEOM_TOL − EST_DRIFT_P95 = 0.47857992441961017 − 0.36010037281174667 =
**0.1184795516078635 px**`. The scene displacements visible in §2.4 are orders above that, so the
expectation is a hard fail — **but an expectation is not a result**, and the honest sequence is:

1. let the released chunk finish (`chunks/s1-train-01of04/DONE`);
2. `scripts/assemble_restyled_lerobot.py` → a LeRobot root (G0a's `--restyled-dataset` needs one;
   clips alone are not that input);
3. `scripts/run_g0_gates.py` for G0a and G0b.

Then the sentence in the record is a measured margin, not "the videos look bad".

## 6. Why nothing here was changed, and what it would cost to change it

**Every lever in §2 moves the throughput, and the throughput is load-bearing.**
`PARTITION_CEILING_GPU_H = 2013.75`, signed by the project owner on 2026-09-01, is arithmetic over
`1.6896 s/frame` measured under **these** settings: `gpu_hours_per_variant = 80.55 × 25
style-instances`. Generating at bucket 720 is roughly twice the pixels. A changed `sigma_max`,
`num_conditional_frames` or control set changes it too. **A session that quietly improves the
picture also quietly invalidates the number the spend was authorised against**, and `T40_RULE_V20`
§5 exists to stop exactly that kind of drift.

Changing the control set has a second cost: `configs/transfer25/styles.toml` is committed and
hashed, and PR-08 §6's G0b is scored against the conditioning that was in force. Swapping
`depth:0.5,seg:0.5` for `edge:1.0` after seeing clips is a decision that has to be made in the open,
in a versioned document, or it is *„a gate rewritten after seeing its output"* (`docs/handoff.md`
§3).

**The cheap experiment that settles it without touching any of that** — proposed, not run:

> One clip, one episode already in the released chunk, generated a second time with
> `edge:1.0` in place of `depth:0.5,seg:0.5`, `sigma_max` set, `resolution 720`,
> `num_conditional_frames=2`. Roughly **15 GPU-minutes**. It measures no committed quantity, writes
> no `THROUGHPUT.json`, and produces no corpus clip — it produces **one picture that answers whether
> the model or the configuration is at fault.**

If it looks right, the Cosmos 3 migration is not the fix and we saved it. If it still looks like
§2's output, the migration has earned its cost. **Either way the answer is evidence rather than
argument, which is the only reason to run it.**

## 7. Provenance

| | |
|---|---|
| opened | 2026-09-03, by the project owner, on the clips of the released chunk |
| evidence | job `192149` log; `cosmos_transfer2/config.py`, `inference.py`, `_src/transfer2/inference/inference_pipeline.py`, `_src/transfer2/inference/utils.py`, `_src/predict2/datasets/utils.py` on the cluster at `third_party/cosmos-transfer2.5`; `scripts/restyle_transfer25.py`; `configs/transfer25/styles.toml`; the source manifest; NVIDIA's Transfer 2.5 multi-control cookbook |
| clips inspected | 10 of the released chunk, plus the source `episode_000116` |
| parameters changed | **none** |
| rules changed | **none** |
| generation licensed | unchanged — the released chunk, and nothing beyond it |
| training licensed | **no** |
