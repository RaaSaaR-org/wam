# Workstation pipeline — T-041 data preparation

Everything between "nothing" and "a dataset Cosmos3-Super can train on", run on a machine with an
NVIDIA GPU and a normal shell. The cluster then does one thing: train.

```bash
git clone <this repo> && cd wam
export T041_FREEZE_LIFTED="<why the PR-07 §7 freeze is lifted>"
workstation/00_setup_env.sh      # cosmos-framework at the cluster's pinned SHAs
workstation/10_fetch_corpus.sh   # 14 sources, one camera each
workstation/20_prepare_corpus.sh # transcode to H.264, then prove it decodes
workstation/30_caption_corpus.sh # vLLM captions + video_dataset_file.jsonl
```

Rehearse on one small source first — it costs about ten minutes and has caught something every
time:

```bash
SMOKE_SOURCES=GR00T-N1.7-AppleToPlate workstation/20_prepare_corpus.sh
SMOKE=1 workstation/30_caption_corpus.sh
```

## Why this exists

T-041 has failed four times, and not once in training:

| # | Failure | Cost | What it actually was |
|---|---------|------|----------------------|
| 1 | HTTP 429 on Humanoid Everyday | 3 jobs, 26 min | Unauthenticated Hub requests |
| 2 | `FileNotFoundError: meta/episodes.jsonl` | 2 jobs | 13 of 14 sources are LeRobot **v3.0** |
| 3 | Queue stall, 6 h estimate | ~1 day | A 4 h walltime request blocking backfill |
| 4 | `0/372 videos were successfully captioned` | 1 GPU-h | The corpus is **AV1**; vLLM's OpenCV decoded nothing |

Every one is IO, format or scheduling. Each cost hours of Slurm queue to learn something a
workstation answers in seconds, and #4 completed "successfully" while producing 372 empty files.
Cluster GPU-hours are the scarce resource and transcoding video is not what they are for.

## The two things that bite

**Every source is AV1.** LeRobot encodes with `libsvtav1` by default and none of the 14 overrode
it. vLLM decodes video through OpenCV only — every backend in its registry shares one OpenCV
mixin — and the build in its venv opened each file, read the container header correctly (377
frames, 30 fps, 640×480) and then failed every `cap.grab()`. The captioner sent 372 requests, got
372 failures, and reported completion.

It is worth being exact about the cause, because the obvious reading is wrong: **`cv2` 5.0.0 on
macOS decodes the same AV1 without complaint.** Same version, different platform wheel, different
answer. So "AV1 is undecodable" is false and "our corpus was undecodable by the thing that had to
decode it" is true. That is precisely why the gate runs `scripts/verify_clip_decode.py` **with the
captioner's own interpreter** rather than reasoning about codecs: the only build whose opinion
matters is the one that will do the reading, and the script prints which one it used so a check
against the wrong environment is visible rather than silent. Transcoding to H.264 yuv420p then
removes the dependency on getting a lucky wheel at all — and the v3.0 sources have to go through
ffmpeg regardless.

**13 of 14 sources are LeRobot v3.0.** A clip is not a file there — episodes are concatenated into
a handful of mp4s and each one is a `[from_timestamp, to_timestamp)` window recorded in
`meta/episodes/*/*.parquet`. Three details in that layout produce corpora that still train:

- Cameras roll over to new files **independently**. At episode 50 of `G1_Dex3_BlockStacking`,
  `cam_left_high` is already in `file-001` while the other three cameras are still in `file-000`.
  Resolve the file per (episode, camera), never once per episode.
- Timestamps are relative to **their own file** and reset to `0.0` at each rollover.
- `to_timestamp` is **exclusive** and equals the next episode's `from_timestamp`. Cutting with
  ffmpeg's `-to` appends a frame of the neighbouring episode to every clip in the corpus; the
  extraction uses `-frames:v <length>` instead.

## The corpus is one file

`configs/cosmos3/corpus_g1_embodiment.tsv` — one row per source: repo, camera key, licence. Both
this pipeline and the cluster job read it, so they cannot disagree about what the corpus is. The
repo list and camera list used to be two bash arrays matched by position, which is a data structure
held together by nothing: delete a line from one and every source after it silently gets its
neighbour's camera.

The camera choice is deliberate. `cam_left_high` is the head-mounted view that sees the torso, arms
and Dex3 hands — the embodiment signal PR-09 §2 is buying. Wrist cameras see the object and almost
nothing of the robot.

## Requirements

- NVIDIA GPU. Preparation is CPU/IO; captioning needs ~20 GB of VRAM for Qwen3-VL-8B-FP8.
  A Blackwell card (sm_120) needs `cu128` wheels or newer — earlier ones have no kernels for it.
- `ffmpeg` + `ffprobe` with an AV1 **decoder** and an H.264 **encoder**. `ENCODER=h264_nvenc`
  uses the GPU; the `libx264` default is better per bit and fast enough at 640×480.
- ~150 GB free: roughly 35 GB of single-camera downloads plus the transcoded corpus.
- `hf` CLI, `python3` with `pyarrow` (v3.0 boundaries are parquet).
- `git-lfs`. cosmos-framework LFS-tracks `assets/**` and every media extension; without it the
  clone succeeds and the **checkout** dies half-way, leaving a repo that reports the right SHA over
  an empty working tree.
- A C compiler on PATH as `cc`. `uv sync --all-extras` reaches evdev (via lerobot → pynput), which
  ships no wheel. Nothing here uses evdev, but `--all-extras` gives no way to decline it.

None of these need root. This machine has no `sudo`, and all three of ffmpeg, git-lfs and gcc came
from conda-forge envs symlinked into `~/.local/bin` — the same trick `90_build_cosmos_env.sbatch`
uses on Discoverer+, for the same reason.

A **CUDA toolkit is deliberately not required.** The driver is enough, and step 00 patches the one
place that assumes otherwise: transformer_engine probes for a system toolkit via nvrtc and curand,
and when both are missing re-loads cudart from the pip wheels under the CUDA 13 directory name
(`nvidia/cuda_cudart`) while the cu12 wheel installs to `nvidia/cuda_runtime`. Discoverer+ has a
CUDA 12.8 module so it never reaches that branch; a driver-only workstation always does, and the
error — `cudart shared object not found`, eleven frames into a megatron import — names neither
CUDA nor the toolkit. Step 00 aliases the directory.

Set `WORK=` to choose where everything lands (default `~/wam-t041`).

## Getting it to the cluster

Still open, and it depends on the machine's upstream. Two routes:

1. **Private HF dataset.** `hf upload --repo-type dataset`, then pull it on the cluster at the
   ~75 MB/s it already gets from the Hub. The revision hash becomes the AC-04 dataset snapshot,
   which is a better provenance record than a directory that was rsynced once.
2. **Direct rsync** to `/valhalla/projects/ehpc-aif-2026pg01-905/data/`.

Either way the upload is bounded by this machine's upload speed, not the cluster's download speed.
Ship the transcoded corpus plus captions and jsonl — not the raw downloads, which the cluster can
re-fetch faster than we can send them.

## What ships

```
<corpus>/
  manifest.json          provenance: source repo, camera, window, both hashes per clip
  MANIFEST_SHA256        what the training job records
  train/videos/*.mp4     H.264, one file per episode
  train/captions/…       NVIDIA's captioner output
  train/video_dataset_file.jsonl
  train/decode_report.json   which cv2 verified these, and that all of them decoded
  val/…                  same, held out for PR-09 §5's eval prompts
```

`manifest.json` carries `src_sha256` (which upstream bytes) and `sha256` (which pixels the run
actually saw) separately. Re-encoding breaks the identity between them, and collapsing them to one
field would make a corpus rebuilt with different settings indistinguishable from the original.
