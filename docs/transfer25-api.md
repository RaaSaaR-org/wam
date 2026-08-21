# Cosmos-Transfer2.5, as it actually is — the source document `97`'s TODO said did not exist

Read from `github.com/nvidia-cosmos/cosmos-transfer2.5` @ `main` on **2026-08-16**. Every claim below
carries the file and line it was read from. **Nothing here has been executed** — there is no
Transfer2.5 checkout on this workstation and none on Discoverer+; this is a reading of the source,
which is exactly the thing `97_transfer25_restyle.sbatch`'s `TODO(human)` said was missing:

> *"There is no Transfer2.5 invocation anywhere in this repo to copy, and guessing the module path,
> the payload schema and the conditioning flags would be worse than saying so."*

It was right to refuse to guess. This document replaces the guess with citations, and **it also
contradicts three things `97` assumed.** Those are in §7, and they are the reason to read this
before writing against that file.

**Pin before you trust this.** `main` moves. Anything built on this should re-read against the
revision it pins, not against `main`.

## 1. The seed exists, so arm C is executable

The one fact PR-08's arm C depends on. `97` makes the seed mandatory in the work unit and says a
driver that cannot set it *"must fail loudly rather than default"*, because arm C is ten samples of
one prompt and a driver that ignores the field produces either ten identical files or ten
irreproducible ones.

| | |
|---|---|
| field | `seed: int`, on `InferenceArguments` |
| default | **2025** |
| source | `cosmos_transfer2/config.py:517-518` — `seed: int = 2025` / `"""Seed for generation randomness."""` |
| settable via | **both** JSON spec and CLI. `InferenceOverrides` excludes only `["name","edge","depth","vis","seg"]` (`config.py:595-604`), so `seed` survives into the overrides model |
| CLI spelling | `--seed <int>`, top-level (not `--overrides.seed`) — `examples/inference.py:89` applies `tyro.conf.OmitArgPrefixes` |

**Trap:** the base class `CommonInferenceArguments` also declares `seed: int = 0` (`config.py:329-330`).
`InferenceArguments` overrides it to `2025`. Read the subclass.

## 2. 640×480 is reachable, but not by asking for it

`resolution` is **not** a `WxH` string. It is a key into a fixed bucket table.

- `resolution: str = "720"` — `config.py:496-497`
- the table is `VIDEO_RES_SIZE_INFO` — `cosmos_transfer2/_src/predict2/datasets/utils.py:44-67`
- the row that matters: `"480": {"1,1": (480, 480), "4,3": (640, 480), ...}`

So **`resolution="480"` on a 4:3 input yields exactly (640, 480)**. The aspect ratio is *derived from
the input video*, not user-set (`_src/transfer2/inference/inference_pipeline.py:473-475`, used at
`:520`). AppleToPlate's `ego_view` is 640×480, which is 4:3, so it lands in that bucket natively.

Second safety net: `keep_input_resolution: bool = True` (`config.py:509-510`) resizes the model's
bucketed output back to the input's exact resolution (`inference_pipeline.py:725-727`). Both paths
land on 640×480 here, which is what PR-08 §3 fixes as the GR00T N1.7 `ego_view` contract.

## 3. The guardrail rewrites frames, and this is now verified rather than asserted

This repo's standing rule — **`--no-guardrails` is mandatory on Cosmos inference** — was written on
the grounds that the guardrail's RetinaFace post-processor rewrites frames and blurs the hand,
editing the evidence. That was a correct inference about a different Cosmos family. **For
Transfer2.5 it is now confirmed at source level:**

- `_src/imaginaire/auxiliary/guardrail/common/presets.py:38-43` — the video guardrail runner is
  constructed as `postprocessors=[RetinaFaceFilter(...)]`, i.e. the blur is a **postprocessor**, not
  a classifier.
- `.../face_blur_filter/face_blur_filter.py:156-159` —
  `frame[y1:y2, x1:x2] = blurred_face` … `blurred_frames.append(frame)`. Pixel-level, in place.
- `cosmos_transfer2/inference.py:313-333` — the returned `processed_frames` **replace**
  `output_video` before it is saved. **With guardrails on, what lands on disk is the blurred frames,
  not the model output.**

**The flag is `--disable-guardrails`, not `--no-guardrails`** (`config.py:234-235`,
`disable_guardrails: bool`, on `SetupArguments`, **CLI-only** — `examples/inference.py:55-56` says
setup arguments cannot come from the JSON). Our driver keeps this repo's vocabulary and accepts
`--no-guardrails`, translating at the boundary; see §7.

**A licence consequence, and it is a good one.** With guardrails disabled both runners become `None`
(`inference.py:90-101`), so `nvidia/Cosmos-Guardrail1` is never downloaded. That repo is gated and
`docs/setup.md:115` requires accepting the NVIDIA Open Model License Agreement — **an act belonging
to the account holder, not to an agent.** Disabling guardrails removes that requirement rather than
routing around it.

## 4. Batching: JSONL is many jobs, JSON is one

`-i` is `input_files: list[Path]` (`examples/inference.py:50-54`): *"If multiple files are provided,
run 'batch' inference. The model will be loaded once and all samples run sequentially."*

`CommonInferenceArguments._from_file` (`config.py:354-365`):

| input | meaning |
|---|---|
| `foo.json` | **exactly one** job — `data_list = [json.loads(path.read_text())]` |
| `foo.jsonl` | **one job per line** |
| a directory | **`ValueError: Unsupported file extension`** — there is no directory-listing code |

`name` must be unique across the batch (`from_files`, `config.py:411-415`).

## 5. Output layout is FLAT, and a sidecar does not mean success

Given `-o <dir>`, `_generate_sample` sets `output_path = output_dir / sample.name`
(`inference.py:187`) and writes, all in that one directory:

| file | when | source |
|---|---|---|
| `<name>.json` | **before generation runs** | `inference.py:205-208` |
| `<name>_control_<key>.mp4`, `<name>_mask_<key>.mp4` | after control extraction | `inference.py:308-316` |
| `<name>.mp4` (or `.jpg` if one frame) | after generation **and after the guardrail rewrite** | `inference.py:338`, `:293-296` |
| `<name>.txt` | the prompt | `inference.py:340-342` |
| `config.yaml` | once per run, at `output_dir` root | `inference.py:152-156` |

**No per-job subdirectories.** `97` requires `<raw>/<unit>/vision.mp4` and
`<raw>/<unit>/sample_outputs.json`; that foldering is our driver's job, not the framework's.

**The sidecar is written before the guardrail check.** So `<name>.json` existing proves the job was
*attempted*, not that it produced a video. A harvest keyed on that file counts failures as work.

## 6. `keep_going` does not do what its name says

`SetupArguments.keep_going` is documented as *"When running batch inference, keep going if an error
occurs"* (`config.py:238-239`). It is consulted **only inside `_generate_sample`, and only for
guardrail-block cases** (`inference.py:218-221`, `:232-235`, `:324-327`).

`generate()` (`inference.py:158-183`) calls `_generate_sample` in a **plain loop with no
`try`/`except`**. A non-guardrail exception — an unreadable video, an OOM — propagates and **kills
the whole batch regardless of `keep_going`.**

**Therefore per-job fault isolation is the driver's responsibility.** On a 10 050-clip run across
four chunks this is the difference between losing one clip and losing a chunk.

## 7. Three things `97_transfer25_restyle.sbatch` assumed that are not true

Recorded plainly because that file is careful, and these are the kind of error that survives by
being plausible.

1. **`_handle_sample_exception` does not exist.** `97:548-550` justifies writing
   `sample_outputs.json` *after* asserting the mp4 on the grounds that *"the framework's
   `_handle_sample_exception` writes that file too"*. **No function of that name exists anywhere in
   the repo.** The *instruction is still right* — and §5 and §6 give it two better reasons (the args
   sidecar is written pre-guardrail, and an exception kills the batch outright) — but the reason as
   written cites a behaviour that is not there.
2. **The guardrail flag is `--disable-guardrails`.** `--no-guardrails` is this repo's spelling and
   appears nowhere upstream. Our driver accepts ours and translates; nothing that shells out to
   `examples/inference.py` directly may use ours.
3. **`--resolution 640x480` is not an upstream value.** It is `resolution="480"` plus a 4:3 input
   (§2). The driver's contract keeps `640x480` because that is the number PR-08 §3 fixes, and
   asserts the bucket it maps to rather than passing the string through.

A fourth, upstream's own: `docs/inference.md:82-127`'s first JSON example carries `"output_dir"`
inside the spec and omits the required `"name"`. `InferenceArguments` sets
`model_config = ConfigDict(extra="forbid")` (`config.py:314`), so that example **cannot validate**.
`output_dir` belongs to `SetupArguments` (CLI-only, `-o`). **Trust `config.py` over the docs.**

## 8. The spec schema

**Required:** `name` (`config.py:317-318`), one of `prompt` / `prompt_path` (`:319-322`, validator
`:334-351`), `video_path` (`:481-483`), and **at least one control block** — `model_post_init` raises
*"No controls provided…"* (`:539-541`).

**Optional, load-bearing here:** `negative_prompt` (`:520`), `guidance: int` 0–7 default 3 (`:308`,
`:331-332`), `seed` (§1), `resolution` (§2), `num_steps: int = 35` (`:502-503`),
`keep_input_resolution` (`:509-510`), `max_frames`, `num_conditional_frames` (`:484-511`).

**Control blocks** — base `ControlConfig` (`config.py:428-438`): `control_path` (**`None` ⇒ the
control map is generated on the fly from the input video**), `control_weight` ∈ [0,1] default 1.0
(normalised to sum ≤ 1.0 for multicontrol), `mask_path`, `mask_prompt` (a SAM2 text prompt).
Subclasses (`:441-475`): `DepthConfig` → VideoDepthAnything, `EdgeConfig` → Canny +
`preset_edge_threshold`, `BlurConfig` → bilateral blur + `preset_blur_strength`, `SegConfig` →
GroundDino + SAM2 + `control_prompt` (defaulting to the first 128 words of the prompt, `:578-588`).

### The item this dissolves

**PR-08 §4's estimated depth and segmentation may not need precomputing at all.** `97:504-509`
declares a `SOURCE/manifest.json` carrying `depth` and `segmentation` relpaths and calls them
*"blocked on §8 item 5 (the `isaac_binding.py` annotators)"*. But **omitting `control_path` makes
Transfer2.5 generate the control map itself** from the input video, with its own depth and
segmentation models.

**This is a decision, not a fix, and it is not made here.** In-framework estimation is a *different*
estimator from the `isaac_binding.py` annotators, so `GEOM_TOL` and `EST_DRIFT_P95` — measured
against the annotators — would be characterising something the run does not use. Either the
annotators get wired and their maps passed as `control_path`, or the framework's are used and the
geometry budget is re-measured against those. **Picking the second because it is easier would
silently invalidate the numbers PR-08 §8 item 4 committed.** Owner's call; see `D9`.

## 9. Entry point and checkpoints

**Importable, so no subprocess is required** — `examples/inference.py:75` imports it the same way:

```python
from cosmos_transfer2.inference import Control2WorldInference
Control2WorldInference(args: SetupArguments, batch_hint_keys: list[str])
    .generate(samples: list[InferenceArguments], output_dir: Path) -> list[str]
```

`generate` returns output paths, **skipping guardrail-blocked samples** (`inference.py:168-170`) —
another reason the returned list is not a per-unit status. `batch_hint_keys` is normally computed by
`from_files` (`config.py:384-420`) but is just the sorted control keys across the batch (`:410-419`).

*Unverified:* whether `init_environment()` / `init_output_dir()` (`examples/inference.py:73,82,96`)
must run first. No assertion inside `__init__` requires it; calling them is the only in-repo usage
pattern, so the driver calls them.

**Checkpoints** — all four controls live in one repo, at **four different pinned revisions**
(`packages/cosmos-oss/cosmos_oss/checkpoints_transfer2.py:32-50` edge, `:64-68` depth, `:84-88`
blur, `:104-108` seg):

```
repository = "nvidia/Cosmos-Transfer2.5-2B"
revision   = "b67b64abda3801a9aceddbff2bdb86126c06db74"   # edge; the others differ
filename   = "general/edge/61f5694b-..._ema_bf16.pt"
uuid       = "61f5694b-0ad5-4ecd-8ad7-c8545627d125"
```

### `name` is not unique, and one of the collisions is trained on mock data

**This is the sharpest trap in the file and it is easy to walk into.** `checkpoints_transfer2.py`
registers the name `nvidia/Cosmos-Transfer2.5-2B/general/edge` **twice** — at `:35` and again at
`:115` — with different uuids, different revisions, different filenames and different experiments.
The same double registration exists for `general/depth`, `general/blur` and `general/seg`
(`:55`/`:175`, `:75`/`:155`, `:95`/`:135`).

It does not raise, because **`name` is not a key.** `_CHECKPOINTS` is
`dict[str, CheckpointConfig]` keyed by **`uuid` and `s3.uri`** (`_src/imaginaire/utils/checkpoint_db.py:321`,
registration loop `:336-339`); `name` is used only by `full_name` for log lines (`:287-290`).

And the two are not interchangeable. The `experiment` strings differ, and the second `general/edge`
entry's ends **`..._rectified_flow_mock_data`** (`:117`) against the first's
`..._rectified_flow_refimdrop0pt5` (`:36`).

**So a checkpoint cannot be identified by name, and `TRANSFER_MODEL_ID` as an HF repo id identifies
nothing** — the repo id is the same string for every one of them. PR-08 §6 requires the generator
checkpoint id and a pinned revision in the record; satisfying that honestly means recording **the
`uuid`**, and the revision must be **that uuid's own**, not a repo-wide one. `97`'s
`TRANSFER_MODEL_ID` / `TRANSFER_MODEL_REVISION` pair is necessary and **not sufficient**; a
`TRANSFER_CHECKPOINT_UUID` belongs beside them, and the staging job should assert that the file it
staged is that uuid's `filename`.

*Corrected 2026-08-16, hours after this document was first written:* the original §9 said only that
`TRANSFER_MODEL_REVISION` must be the control's own revision. True, and it left the impression that
naming the control picks the weights. It does not.

### The cluster problem this creates

`CheckpointConfig.download()` (`_src/imaginaire/utils/checkpoint_db.py:292-298`) shells out to
**`uvx hf>=1.3.5 download …`** (`:153-167`). That is a network fetch *and* a `uvx` package
resolution, **at inference time, on a compute node.** Discoverer+ compute nodes are not a place to
download from, and the login node forbids it. **So the checkpoints must be pre-staged by a
build/stage job and the download path must never be reached at runtime** — which is exactly the
`cosmos_transfer_env.sh` that `97:255-259` says does not exist yet. There is no Cosmos-specific
cache variable; relocation is plain `huggingface_hub` (`HF_HOME`), which is a `huggingface_hub`
behaviour and not something this repo documents.

**Staging those weights is a download at scale and is the project owner's call**, per the
sub-project rule. Nothing here initiates it.

## 10. Two controls is a different code path, and it discards your checkpoint

*Added 2026-08-22, after job 189142 measured a crash and called it throughput.*

PR-08's committed control set is `depth:0.5,seg:0.5`. **Two hint keys is not "§9 twice".** It is a
different branch of `Control2WorldInference.__init__`, and the two branches disagree about almost
everything §9 established.

```
len(batch_hint_keys) == 1   inference.py:52-62    honours args.checkpoint_path via
                                                  has_checkpoint_override; `model` picks the variant
len(batch_hint_keys)  > 1   inference.py:64-72    NEVER READS args.checkpoint_path. Builds
                                                  checkpoint_list from MODEL_CHECKPOINTS for ALL of
                                                  CONTROL_KEYS = ["edge","vis","depth","seg"], and
                                                  hardcodes experiment = "multibranch_720p_t24_…"
```

Upstream's own comment on the second branch: *"Multi-control: load ALL control modalities even if
some have control weight = 0."* Three consequences, none of them optional:

1. **`99`'s staged tree is unused on the committed control set.** Not overridden — not consulted.
   The 29 GB at `${TRANSFER_CHECKPOINT_PATH}` and the sha256s in `STAGED.json` describe bytes this
   path does not load, and the sbatch's `=== generator … (FROZEN)` log line is a claim about them.
2. **`general/blur` is required.** `ModelVariant.VIS` is backed by
   `general/blur/ba2f44f2-…_ema_bf16.pt` (`checkpoints_transfer2.py:74-88`) — the directory `99`
   deliberately skips as *"a control PR-08 does not use"*. PR-08 does not use it. Upstream loads it.
3. **Each branch is fetched at its own pinned revision, and none is `TRANSFER_MODEL_REVISION`.**
   `load_multi_branch_checkpoints` calls `download_checkpoint(...)` per entry
   (`vid2vid_model_control_vace_rectified_flow.py:664-672`), which resolves through
   `CheckpointConfig.download()` → `hf.download()`. So a cold run pulls ~22 GB **from inside
   whatever it is doing** — and in the TIMING path, from inside the measured window.

**Measured 2026-08-22 (job 189401), and it is better than the above implies.** Staging all four and
diffing sha256 against `99`'s `STAGED.json`: `edge`, `depth` and `seg` are **byte-identical** to the
files `99` staged at `ce8440327…`, despite three different revision labels. Only `vis`
(`general/blur`, `82ede02539a4b141`) is genuinely new. So the differing revisions are commit labels
over unchanged content, and `PR-08` §6's `FROZEN` claim survives **as a statement about bytes** for
the three controls we had — which is the only form of it that was ever checkable, since §9 already
showed the repo id and revision identify nothing on their own. It also answers the question `99`
left open as `"variant_selected": null`: upstream's table selects the **5.53 GB** member of each
undocumented pair, for all four controls.

`99b_stage_transfer25_multibranch.sbatch` warms exactly those four by calling the *same*
`MODEL_CHECKPOINTS[ModelKey(variant=key)]` lookup and the *same* `download_checkpoint()`, so a pass
there is evidence the run will not download, not a claim that it should not have to. It records
repo, revision, filename, uuid and sha256 per branch in `MULTIBRANCH_STAGED.json`. Resolving through
upstream's own table rather than a typed-in uuid is also what keeps §9's name-collision trap shut:
`MODEL_CHECKPOINTS` selects between the colliding registrations, and we do not.

### `model` has a default and the default is unreachable

`SetupArguments.model` is declared with a default (`config.py:305`), which reads as optional. It is
not. `validate_model` is a **`mode="before"`** validator (`:263-270`): it runs on the raw input dict,
before pydantic applies any default, and raises `ValueError("model is required")` for a key it would
have filled in one step later. Omitting it dies **after** the checkpointer loads, which is what made
job 189142 look like a 118-second episode instead of an argument error.

Which name to pass follows from the branch table above: with one control it *is* the variant
selector and must be that control; with several, upstream only reads `.distilled` off it. The driver
passes `sorted(hint_keys)[0]` — exact in the first case, inert in the second.
