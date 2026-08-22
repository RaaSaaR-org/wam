#!/bin/bash
# Step 24 — T-040: which decoders in 98's venv can actually pull pixels out of the PR-08 corpus.
#
#   sbatch 104_probe_clip_decode.sbatch
#
# WHY. Job 189585 (103, PILOT=1) died in 7 seconds:
#
#   [av1 @ ...] Missing Sequence Header.
#   [av1 @ ...] Your platform doesn't suppport hardware accelerated AV1 decoding.
#   FATAL: .../videos/episode_000000.mp4 opened but decoded no frames — the container parses and
#          the codec does not.
#
# THIS PROJECT HAS SEEN THIS EXACT FAILURE BEFORE. Job 186357 captioned 372 clips and produced zero
# captions: nothing crashed, ffprobe was happy, the manifest was correct, the captioner ran to
# completion at 3 it/s, and vLLM's OpenCV backend read every container header correctly and then
# failed every cap.grab(). scripts/verify_clip_decode.py exists BECAUSE of that job, and its
# docstring states the lesson in one line: *a corpus is only readable by the decoder that will
# actually read it.* The PR-08 corpus was materialised as a COPY of the AV1 source
# (manifest.json: "codecs": ["av1"], "materialized": "copy"), so the trap was still armed and 103
# walked into it. That is twice.
#
# WHAT IS ACTUALLY IN QUESTION, AND WHY THE OBVIOUS FIX IS NOT OBVIOUSLY RIGHT.
# Two different consumers read these clips and THEY DO NOT SHARE A DECODER:
#
#   * scripts/measure_geom_tol.py and scripts/measure_est_drift.py decode with cv2, because the
#     segmenter takes frames.
#   * Cosmos-Transfer2.5 is handed a PATH (restyle_transfer25.py:290, "video_path") and decodes it
#     inside upstream's own loader.
#
# So "transcode the corpus to H.264 and move on" is a decision about EVIDENCE, not about plumbing.
# PR-08 §4 step 2 requires GEOM_TOL and EST_DRIFT_P95 to be measured by the same segmenter on the
# same pixel grid, and §6 subtracts them from a generator's output. If the generator reads the AV1
# original while the tolerance is measured on a re-encoded copy, then G0b compares two different
# sets of pixels and the difference is partly the transcode. A lossy re-encode moves an
# object centroid by a fraction of a pixel, which is small — and GEOM_TOL is measured in pixels,
# so "small" is exactly the scale that matters and is not a thing to wave through.
#
# THEREFORE THIS JOB DECIDES NOTHING AND MEASURES FOUR THINGS:
#   1. can cv2 in 98's venv decode the corpus at all, and what is that cv2 built against
#   2. can any OTHER decoder present in that venv decode it (pyav, imageio-ffmpeg, torchvision,
#      decord) — if one can, the measurement scripts can be given the generator's own reader
#      instead of a transcoded corpus
#   3. what does upstream ITSELF use to read a video_path — that is the decoder GEOM_TOL should
#      match, and it is a fact in the checkout rather than an opinion
#   4. whether the system ffmpeg can decode AV1, i.e. whether a transcode is even available here
#
# FREE QoS. This is CPU work on 888 MB of video: `2cpu-single-host`, no --gres. A GPU job here
# would bill GPU-hours to watch a codec fail.

#SBATCH --job-name=wam-clip-decode
#SBATCH --partition=common
#SBATCH --account=ehpc-aif-2026pg01-905
#SBATCH --qos=2cpu-single-host
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --open-mode=append
#SBATCH -o /valhalla/projects/ehpc-aif-2026pg01-905/logs/clip-decode.%j.out

set -euo pipefail
source /valhalla/projects/ehpc-aif-2026pg01-905/caches.sh

PROJ=/valhalla/projects/ehpc-aif-2026pg01-905
WAM=${PROJ}/wam
SOURCE=${SOURCE:-${PROJ}/data/pr08-apple-640x480}
OUT=${PROJ}/runs/${RUN_ID:-pr08-geom-tol}
mkdir -p "${OUT}"

source "${PROJ}/cosmos_transfer_env.sh"
source "${FRAMEWORK}/.venv/bin/activate"
echo "=== python     $(command -v python)"
echo "=== framework  ${FRAMEWORK}"

# (1) The consumer's own gate, run with the consumer's own interpreter — the one thing
# verify_clip_decode.py's docstring insists on. --limit 4 because the answer here is categorical:
# if four AV1 clips fail identically, the 398 others are not a different question.
echo
echo "=== (1) verify_clip_decode.py, cv2, 4 clips ==================================================="
set +e
python "${WAM}/scripts/verify_clip_decode.py" "${SOURCE}/videos" --limit 4 --frames 3 \
  --report "${OUT}/clip_decode_cv2.json" 2>&1 | grep -v "^\[av1 @" | tail -25
CV2_RC=${PIPESTATUS[0]}
set -e
echo "=== verify_clip_decode.py exited ${CV2_RC} (0 = every sampled clip decoded)"

# (2)-(4). One python process, because every answer is an import away and three processes would
# only make the log harder to read. Nothing here raises: a decoder that is absent, or present and
# broken, is a RESULT — the whole point is a table of what this machine can and cannot do.
SOURCE="${SOURCE}" OUT="${OUT}" FRAMEWORK="${FRAMEWORK}" CV2_RC="${CV2_RC}" python - <<'PY'
import json, os, pathlib, subprocess, sys, traceback

src = pathlib.Path(os.environ["SOURCE"])
out = pathlib.Path(os.environ["OUT"])
clip = sorted((src / "videos").glob("*.mp4"))[0]
rec = {
    "schema": "wam.pr08_clip_decode_probe/1",
    "purpose": "Which decoder in 98's venv can read the PR-08 AV1 corpus. Job 189585 (GEOM_TOL "
               "pilot) died because cv2 could not, and this project already lost job 186357 to the "
               "identical failure on a sibling corpus.",
    "clip": str(clip),
    "verify_clip_decode_cv2_exit": int(os.environ["CV2_RC"]),
}

print("\n=== (2) what this cv2 is built against ========================================================")
try:
    import cv2
    info = cv2.getBuildInformation()
    lines = [l.strip() for l in info.splitlines()
             if any(k in l for k in ("FFMPEG", "avcodec", "AV1", "dav1d", "aom", "GStreamer"))]
    rec["cv2_version"] = cv2.__version__
    rec["cv2_build_lines"] = lines
    for l in lines:
        print("   ", l)
except Exception as exc:  # noqa: BLE001
    rec["cv2_error"] = repr(exc)
    print("    cv2 unavailable:", exc)

print("\n=== (3) every other decoder in this venv, on the same clip ====================================")

def try_pyav():
    import av
    with av.open(str(clip)) as c:
        s = c.streams.video[0]
        n = sum(1 for _ in zip(range(3), c.decode(video=0)))
    return {"lib": "av", "version": av.__version__, "codec": s.codec_context.name, "frames": n}

def try_imageio():
    import imageio.v3 as iio
    frames = [f for _, f in zip(range(3), iio.imiter(str(clip)))]
    import imageio
    return {"lib": "imageio", "version": imageio.__version__,
            "frames": len(frames), "shape": list(frames[0].shape) if frames else None}

def try_torchvision():
    import torchvision
    from torchvision.io import read_video
    v, _, _ = read_video(str(clip), start_pts=0, end_pts=0.1, pts_unit="sec")
    return {"lib": "torchvision", "version": torchvision.__version__, "frames": int(v.shape[0])}

def try_decord():
    import decord
    vr = decord.VideoReader(str(clip))
    return {"lib": "decord", "version": getattr(decord, "__version__", "?"),
            "frames": len(vr), "first_shape": list(vr[0].shape)}

results = []
for fn in (try_pyav, try_imageio, try_torchvision, try_decord):
    name = fn.__name__.removeprefix("try_")
    try:
        r = fn()
        r["ok"] = bool(r.get("frames"))
        results.append(r)
        print(f"    {name:12s} OK   {r}")
    except ImportError as exc:
        results.append({"lib": name, "ok": False, "absent": True, "error": str(exc)})
        print(f"    {name:12s} ABSENT  ({exc})")
    except Exception as exc:  # noqa: BLE001
        results.append({"lib": name, "ok": False, "absent": False,
                        "error": "".join(traceback.format_exception_only(exc)).strip()})
        print(f"    {name:12s} PRESENT BUT FAILED  {exc!r}")
rec["decoders"] = results

print("\n=== (4) what upstream itself reads a video_path with ==========================================")
# A fact in the checkout, not an opinion. The measurement should decode with whatever the GENERATOR
# decodes with -- that is what makes GEOM_TOL a property of the pipeline under test rather than of
# our own import path, and it is the same argument that fixed the segmenter in 102.
pats = ["VideoCapture", "import av", "decord", "imageio", "read_video", "av.open"]
hits = {}
srcroot = pathlib.Path(os.environ["FRAMEWORK"])
for p in pats:
    try:
        r = subprocess.run(["grep", "-rIl", "--include=*.py", p, str(srcroot / "cosmos_transfer2")],
                           capture_output=True, text=True, timeout=120)
        files = [l for l in r.stdout.splitlines() if l]
    except Exception as exc:  # noqa: BLE001
        files = [f"<grep failed: {exc}>"]
    hits[p] = files[:8]
    print(f"    {p:14s} {len(files)} file(s)" + (f"  e.g. {files[0]}" if files else ""))
rec["upstream_decoder_grep"] = hits

print("\n=== (5) can the system ffmpeg decode AV1 at all ===============================================")
# Decides whether a transcode is even an option on this machine. It does not decide whether a
# transcode is the right answer -- see the header.
try:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True,
                       timeout=60)
    av1 = [l.strip() for l in r.stdout.splitlines() if "av1" in l.lower()]
    rec["ffmpeg_av1_decoders"] = av1
    print("   ", av1 or "NONE -- this ffmpeg cannot decode AV1 either")
except FileNotFoundError:
    rec["ffmpeg_av1_decoders"] = None
    print("    no ffmpeg on PATH")

path = out / "CLIP_DECODE_PROBE.json"
path.write_text(json.dumps(rec, indent=2) + "\n")
print(f"\n=== wrote {path}")

usable = [d["lib"] for d in rec["decoders"] if d.get("ok")]
print("\n=== VERDICT")
if rec["verify_clip_decode_cv2_exit"] == 0:
    print("    cv2 decodes the corpus. 189585's failure was not the codec and this probe is wrong")
    print("    about its own premise -- read the log before believing either.")
elif usable:
    print(f"    cv2 cannot decode this corpus; {', '.join(usable)} can. The measurement scripts")
    print("    decode with cv2 today (measure_geom_tol.episode_centroids_from_video). Giving them")
    print("    a reader that works costs one seam and leaves the corpus bytes untouched, which is")
    print("    the option that does not put a transcode between GEOM_TOL and the generator.")
else:
    print("    NOTHING in this venv decodes the corpus. Then the question is no longer which reader")
    print("    the measurement uses -- it is whether the generator can read it either, and 189584")
    print("    will answer that when it starts. Do not transcode before that answer arrives: a")
    print("    corpus that the GENERATOR reads fine needs no transcode, only a reader.")
PY
