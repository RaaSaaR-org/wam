# `EST_DRIFT_P95` — the Isaac install, and whether it is worth doing

PR-08 §8 item 4 is two numbers. `GEOM_TOL` is measured on the real corpus and is blocked on a
decoder and a segmenter (`docs/preregistration/PR-08-photoreal-augmentation.md` §6, T-040 notes).
`EST_DRIFT_P95` is the other half, and PR-08 §4 says it is measured **on Isaac renders**: render N
episodes with ground-truth depth and segmentation, run the same monocular estimator and the same
segmenter on the Isaac **RGB only**, and take the 95th percentile of the estimated-vs-true
object-centroid displacement in pixels.

Isaac Sim is not on Discoverer+ and cannot be put there. The only candidate machine is this
workstation's RTX 5090. This page is the runbook for putting it there — and §6 is the part that
asks whether that is the cheapest way to get the number at all, because §4 already concedes that
whatever Isaac produces is **a lower bound on the real error**, and a lower bound is the weakest
useful form of this number.

> **Nothing on this page has been executed.** Isaac Sim is not installed on this box. Every
> command below is written from NVIDIA's documentation (URLs cited inline) and from reading this
> repository's own code. The facts that *were* measured here — the card, the driver, glibc, the
> interpreters, what is in `.venv`, what is in the weight caches — are marked **measured
> 2026-08-22** where they appear. `docs/isaac.md`'s standing warning applies unchanged: every
> module path and symbol name in the Isaac half is an assumption until `scripts/preflight_isaac.py`
> runs.
>
> **§1 was written from a first fetch and has since been re-fetched independently** (2026-08-22,
> a second session, against the same five URLs plus two more). What that second read confirmed and
> what it did not is §1.5 — read it before spending the download, because two of the numbers on
> this page came back different and one of them is the Blackwell defect this whole section turns
> on. Anything on this page that a primary source could not confirm now says **UNVERIFIED** in the
> line where it appears, rather than reading as fact.

**Read first, and this is the reason §6 exists:** installing Isaac Sim does **not** by itself
produce `EST_DRIFT_P95`. It unblocks step 1 of five. §5 prices the whole chain, and the scene — not
the install — is the expensive part.

---

## 0. What this page is for, and the one thing it must not be read as

This is a *plan*, and PR-08's gates are untouched by it. Nothing here licenses generating a clip,
training a weight, or reading `C`/`W` from PR-12/PR-13 as permission to train. PR-08 §1's forbid
list stands, T-39's premise has been withdrawn rather than satisfied, and whether training starts
is the project owner's call (`CLAUDE.md`). Measuring an estimator's error budget is explicitly
licensed by PR-08 §1 — that, and only that, is what this page is about.

Likewise, §6's alternatives are **proposals for an amendment the project owner would have to
make**, not amendments. PR-08 §4 names Isaac explicitly. Rules in this repo are versioned, never
edited (`docs/handoff.md` §3; PR-08's own header). Every alternative below therefore states the
exact sentence of §4 that a `PR-08-V4` would have to replace, so the cost of accepting it is
visible before anybody starts coding against it.

---

## 1. The RTX 5090 question, answered

**Short answer: yes, and Isaac Sim 6.0.1 is the release to install — not 5.1.** Consumer Blackwell
is in NVIDIA's *recommended* hardware row for both current releases, and 6.0 is the one whose
driver floor this box already clears and whose known Blackwell defect does not sit on our code
path.

### 1.1 What this box actually is — measured 2026-08-22

```
$ nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
NVIDIA GeForce RTX 5090, 32607 MiB, 595.84, 12.0
$ ldd --version | head -1        -> ldd (Ubuntu GLIBC 2.39-0ubuntu8.8) 2.39
$ lsb_release -ds                -> Ubuntu 24.04.4 LTS
$ /home/humanoid/.local/bin/python3.12 --version -> Python 3.12.13
$ df -h /                        -> 1.5T total, 771G free   (782G when first read, 6 h earlier;
                                     this box is in use, so re-run it rather than quoting this)
```

`compute_cap 12.0` is sm_120, i.e. Blackwell, which is the thing in question.

### 1.2 What NVIDIA says supports it

| | Isaac Sim 5.1.0 | **Isaac Sim 6.0 (6.0.1.0, the one to install)** |
|---|---|---|
| minimum GPU | GeForce RTX 4080, 16 GB | GeForce RTX 4080, 16 GB |
| **recommended GPU** | **GeForce RTX 5080, 16 GB** | **GeForce RTX 5080, 16 GB** |
| ideal GPU | RTX PRO 6000 Blackwell, 48 GB | RTX PRO 6000 Blackwell, 48 GB |
| Linux driver floor | 580.65.06 | **595.58.03** |
| Python | 3.11 — **UNVERIFIED** | **3.12, exactly** (`Requires-Python: ==3.12.*`) |
| torch NVIDIA installs alongside | 2.7.0 — **UNVERIFIED** (per Isaac Lab's pip page, read 2026-08-05) | **2.11.0**, cu128 or cu130, installed **before** `isaacsim` |
| OS | Ubuntu 22.04 / 24.04 | Ubuntu 22.04 / 24.04 |
| RAM / disk | 32 GB min, 64 GB rec / 50 GB min SSD | same (50 GB min, 500 GB rec) |

*The GPU model names and the driver, RAM, disk, OS and Python rows are NVIDIA's, quoted from the
pages cited at the end of §1.4. The VRAM figures beside the model names are the parts' own published
specifications, not cells in NVIDIA's table — they are there to make the 5080-vs-5090 comparison
legible and nothing depends on them.*

Two cells in the 5.1.0 column are **UNVERIFIED** — not from any page fetched for this document,
first read or second: 5.1.0's Python 3.11 and its torch 2.7.0 come from a search summary and from
`docs/isaac.md` §0's own 2026-08-05 reading of NVIDIA's Isaac Lab pip page, and the 5.1.0
requirements page that *was* fetched twice publishes GPU and driver rows but was not read for
either. They are here only to show that 5.1 would force a *different* interpreter and a *different*
torch; **nothing below depends on either value**, because the recommendation is 6.0.1 and every
6.0.1 cell in that column is confirmed (§1.5). If some future reader needs 5.1's interpreter, that
is a fetch nobody has done.

The recommended row is the answer to "does it support Blackwell". A **GeForce RTX 5080 is consumer
Blackwell, sm_120, the same architecture and the same compute capability as the 5090**; it differs
from a 5090 in width and in VRAM (16 GB vs this card's 32 GB), and on VRAM the 5090 is the better
part. NVIDIA's requirements page also states the exclusion in the other direction — *"GPUs without
RT Cores (A100, H100) are not supported"* — which is a statement about datacentre parts, not about
GeForce Blackwell.

**UNVERIFIED, and worth being plain about: no NVIDIA page consulted — in either read (§1.5) —
names the RTX 5090 by model.** The inference "5080 recommended ⇒ 5090 fine" rests on them being the
same architecture and the 5090 being the larger part. What would verify it:
`scripts/preflight_isaac.py` exiting 0 on this box (§3). That is the whole reason that script
exists, and until it has run this row is the weakest link in §1.

**The driver already clears 6.0's floor.** 595.84 installed ≥ 595.58.03 required (measured
2026-08-22). It also clears 5.1's 580.65.06, so the driver is not what decides between the two.

### 1.3 Which CUDA and torch that forces, and what it settles in `docs/isaac.md` §0

NVIDIA's own installation page for Isaac Sim 6.0 gives the torch line **before** the Isaac line:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # CUDA 12
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130   # CUDA 13
```

Three consequences, and the first one closes an open question this repo has been carrying:

- **`docs/isaac.md` §0's open row is now answered from an NVIDIA-hosted page: the pin is torch
  2.11.0.** That page recorded the 2.11.0 figure as *"not confirmed from an NVIDIA-hosted page …
  the 6.0 figure came from a third-party doc mirror"*. It is now confirmed, at
  `docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html`, fetched
  2026-08-22 and fetched again the same day by a second session that had not seen the first read
  (§1.5). `docs/isaac.md` §0's row has been rewritten to say so. **The two-venv split therefore stands**: this repo's `uv.lock` resolves torch 2.13.0
  from PyPI, 2.11.0 ≠ 2.13.0, and one venv cannot hold both. `pyproject.toml`'s empty
  `isaac = []` extra was right for the reason it gave.
- **cu128 is the sm_120-carrying build**, which is the same index `docs/local_gpu.md` §0a already
  chose for the WAM venv on this card and for the same reason: the compute capability a wheel
  *reports* comes from the driver, so a wheel with no sm_120 cubins reports `(12, 0)` correctly and
  then dies at the first kernel launch. Take cu128 unless something forces CUDA 13.
- **`docs/isaac.md` §1 used to say "Let Isaac resolve its own torch; do not pre-install one," and
  that is now reconciled rather than merely flagged.** Its stated reason was that the version was
  unconfirmed, so pinning it by hand would be guessing where the resolver knows. **That reason has
  expired by measurement, not by preference**: NVIDIA's own installation page pins torch to 2.11.0
  and installs it *before* `isaacsim` — confirmed twice, independently, on 2026-08-22 (§1.5) — and
  pre-installing from cu128 is the only way to *choose* the CUDA build rather than accept whichever
  one the resolver picks. §2 below follows NVIDIA, and `docs/isaac.md` §0/§1 were edited in the same
  change to say the same thing and to cite this section. **Two pages in one repo giving opposite
  install orders is the failure that was closed here**; if a later reader finds them disagreeing
  again, one of them has been edited without the other and neither should be trusted until they are
  re-reconciled.

### 1.4 The one Blackwell defect found, and why it is a watch-item rather than a blocker

`isaac-sim/IsaacLab` issue **#4951**, *"[Bug Report] TiledCamera hangs on RTX 5090 (Blackwell
sm_120) with Isaac Sim 5.1.0"*: `TiledCamera` **hangs indefinitely**, inside `omni.replicator`'s
*tiled* rendering pipeline, spinning one core at 100 % with no output. Reported against driver
590.48.01. Open, unassigned, with a workaround: use the plain `Camera` instead, which the reporter
states produces identical RGB at similar performance for a single environment (`Camera` emits RGBA,
so the channel has to be sliced).

**Correction from the second fetch, and it cuts in our favour: the reporting card is an RTX 5090
*Laptop GPU*, 24 GB — not the desktop 5090 in this box.** Same architecture and same reported
compute capability, a different part with a different memory subsystem and a different power
envelope. So the report is one step further from this machine than §1's first draft implied: it is
evidence that *something* in the tiled path is unhappy on consumer Blackwell, not a measurement on
this GPU. It does not become less of a watch-item for that — a hang is a hang and nobody has run
either path on the desktop part either — but it is not a report about our hardware.

Why this matters here and why it is not a stop: **our binding does not use `TiledCamera`.**
`IsaacSimBinding._setup_cameras` (`src/wam/robot/isaac_binding.py:932`) calls
`rep.create.render_product(prim, resolution=(W, H))` and attaches `rep.AnnotatorRegistry`
annotators to it — the plain, one-product-per-camera path, which is the path the issue says works.
But it is the *same subsystem*, the report is against 5.1.0, and nobody has run either path on this
card. So: **install 6.0.1, not 5.1**, and treat preflight check K (`camera_*`) and check N
(`annotator_*`) as the checks that decide it. A hang rather than a failure is the shape to watch
for — if `preflight_isaac.py` stops printing and never returns, this is the first thing to suspect,
and the last `[PASS]`/`[FAIL]` line it printed is the diagnosis (`docs/isaac.md` §2: the report is
written only for checks that *return*).

**Sources.** Every URL below was fetched twice on 2026-08-22 — once when this page was written,
once by a second session that had not seen the first read and was asked to confirm or refute it.
The right-hand column is the second read's verdict, and it is the reason §1.5 exists.

| source | what it is cited for | second read, 2026-08-22 |
|---|---|---|
| <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html> | 6.0 GPU/driver/RAM/disk table | **confirmed** — min RTX 4080, rec RTX 5080, ideal RTX PRO 6000 Blackwell, Linux 595.58.03, 32/64 GB RAM, 50/500 GB SSD, Ubuntu 22.04/24.04, *"GPUs without RT Cores (A100, H100) are not supported"* |
| <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html> | Python 3.12, torch 2.11.0 cu128/cu130, the pip lines, GLIBC 2.35+ | **confirmed, including the install ORDER** — torch before `isaacsim`, both lines verbatim as in §2, `manylinux_2_35_x86_64` |
| <https://pypi.org/project/isaacsim/> | 6.0.1.0, released 2026-06-22, `Requires-Python: ==3.12.*` | **confirmed** — latest 6.0.1.0, 22 Jun 2026, `Python ==3.12.*` |
| <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html> | 5.1.0's driver floor 580.65.06 | **confirmed** (the floor and the GPU rows; 5.1's *Python and torch* are not on this page and stay UNVERIFIED — §1.2) |
| <https://github.com/isaac-sim/IsaacLab/issues/4951> | TiledCamera hang on sm_120 under 5.1.0 | **confirmed with one correction** — the card is an RTX 5090 **Laptop GPU**, 24 GB, not a desktop 5090 (§1.4) |
| <https://isaac-sim.github.io/IsaacLab-Arena/release/0.2.1/pages/example_workflows/static_apple/> | the upstream Apple-to-Plate scene (§4.2) | **confirmed** — Unitree G1, static, picks an apple off a shelf onto a plate; names `nvidia/Arena-G1-Static-PickNPlace-Task` |
| <https://huggingface.co/datasets/nvidia/Arena-G1-Static-PickNPlace-Task> | that dataset's size and licence (§4.2) | **251 episodes and cc-by-4.0 confirmed; the 50 Hz teleop rate is UNVERIFIED** — neither page states a rate |
| <https://huggingface.co/datasets/USC-GVL/humanoid-everyday> | that HE ships no segmentation masks (§6b) | **supports it, does not prove it** — the card describes *"multimodal sensory streams at 30 Hz (RGB, depth, LiDAR, tactile, IMU)"* and its observation schema lists no mask field. Absence of mention, not an explicit denial |

---

### 1.5 What the second read changed, and what it could not settle

Three things, and they are the ones to carry forward:

1. **Every load-bearing 6.0.1 number confirmed**, first read to second, with no drift: the release
   to install, the 595.58.03 driver floor, `Requires-Python: ==3.12.*`, the torch 2.11.0 cu128/cu130
   pin *and NVIDIA's ordering of it before the `isaacsim` line*, GLIBC 2.35+, and the recommended-GPU
   row that the whole Blackwell argument rests on. §2's command block is NVIDIA's, in NVIDIA's order.
2. **Two numbers came back different and are corrected above**: issue #4951 is a 5090 *Laptop* GPU
   (§1.4), and the Arena dataset's 50 Hz is not published on either page consulted (§4.2's row is now
   marked).
3. **What neither read could settle, because no page addresses it: whether an RTX 5090 runs Isaac
   Sim 6.0.1.** NVIDIA names the 5080 and the RTX PRO 6000 Blackwell; **no NVIDIA page consulted, in
   either read, names the RTX 5090 by model.** The inference is still an inference — same
   architecture, same compute capability, larger part — and it is still the thing
   `scripts/preflight_isaac.py` exists to convert into a fact. **UNVERIFIED, and it is the one
   claim on this page a human should treat as untested before spending an evening on the install.**

---

## 2. The install, in its own python, with numbers

**Where it goes: `~/wam-t041/.venv-isaac`, OUTSIDE the repository working tree.** This is a
change from what `docs/isaac.md` §0/§1 said until 2026-08-22, and from this page's own first draft;
both said `/home/humanoid/develop/wam/.venv-isaac`. Both were wrong, for a reason nobody had
noticed, and `docs/isaac.md` has been corrected in the same change so that no third convention
appears.

> **Why it moved, and it is not tidiness.** `.gitignore` line 5 is `.venv/`. That pattern does
> **not** match `.venv-isaac/` — verified 2026-08-22, `git check-ignore -v .venv-isaac/x` matches
> nothing and exits 1. So the instant step 1 ran under the old instruction, **~25 GB of untracked
> files would appear in `git status` of a working tree that concurrent Claude sessions commit
> into**, and this project's own standing rule for that tree is *"never `git add -A`, never assume
> HEAD is where you left it"*. One session forgetting it once puts a 25 GB Isaac tree in a commit,
> and that is not a mistake you undo cheaply. Widening `.gitignore` would also work, but it leaves
> the hazard one edit away from returning; putting the tree where `git status` cannot see it
> removes it. **If you nonetheless install into the working tree, add `.venv-isaac/` to
> `.gitignore` before running step 1, not after.**

**`~/wam-t041/` is the right home, not an arbitrary one.** It is already this project's
out-of-tree working area on this box — it holds `pr08-apple-640x480` (the PR-08 source corpus),
`hf-cache`, `t041-calibration` and `workstation_env.sh`, 54 GB as of 2026-08-22 — so the Isaac venv
lands beside the corpus it will be measured against rather than in a third place. It is on the same
filesystem as the repo (`/dev/nvme0n1p5` mounted at `/`, measured 2026-08-22), so **moving it out of
the tree costs no disk and changes no free-space arithmetic below**; it only takes the tree out of
`git status`.

It is **not** the repo's venv and nothing below may touch that one: `.venv` keeps torch 2.13.0 and
every WAM extra. Concretely — **never run `uv sync` in this checkout in either venv**
(`docs/local_gpu.md` §0a): it reconciles the environment to `uv.lock` and would replace Isaac's
2.11.0 with 2.13.0, breaking Isaac to fix nothing.

Disk, on `/` which has **771 GB free (measured 2026-08-22; 782 GB six hours earlier — re-run
`df -h /`)**:

| item | size | how well established |
|---|---|---|
| `pip install isaacsim[all,extscache]` tree | **~25 GB** | **secondary source only** (a third-party 2026 install guide). NVIDIA does not publish a figure. What they do publish is the system requirement: **50 GB SSD minimum, 500 GB recommended** |
| torch 2.11.0 + cu128 wheels | ~3–4 GB | ordinary for a CUDA torch wheel; unverified for this exact version |
| Omniverse shader/asset cache, first boot | a few GB, grows | unverified |
| the three estimator checkpoints, **if** `measure` runs here | **already on disk: 4.8 GB** (1.7 + 1.8 + 1.3) | `apple_sam2.py`'s `CHECKPOINTS` table *declares* ~700 MB + ~900 MB + ~1.3 GB; the on-disk figure is what `du -sh` reports and it is the one to budget. **All three are cached on this box, at exactly the pinned revisions** — measured 2026-08-22, `~/.cache/huggingface/hub` holds `models--facebook--sam2-hiera-large@e6a8e880…`, `models--IDEA-Research--grounding-dino-base@12bdfa31…` and `models--depth-anything--Depth-Anything-V2-Metric-Indoor-Large-hf@d2fc6a93…`. They landed 15:43–15:44 on 2026-08-22, i.e. *during* this document's writing, which is why an earlier revision of this row said the opposite. **Re-checked independently later the same day** by a second session, with `ls ~/.cache/huggingface/hub/models--*/snapshots/` and `du -sh` rather than by trusting either earlier sentence: one snapshot directory per repo, named `e6a8e8809b8f…`, `12bdfa3120f3…`, `d2fc6a93601a…` — the exact revisions pinned in `apple_sam2.py` — at 1.7 / 1.8 / 1.3 GB. Note this is `~/.cache/huggingface/hub`, **not** `~/wam-t041/hf-cache`, which is a different cache |

Budget **60 GB** and you are safe. There is no disk problem on this machine.

```bash
# 0. preconditions, all measured true here on 2026-08-22:
#    Ubuntu 24.04, glibc 2.39 (>= 2.35, which the manylinux_2_35 wheels require),
#    driver 595.84 (>= 595.58.03), python3.12 present.
/home/humanoid/.local/bin/python3.12 --version   # must print 3.12.x — 6.0.1 is 3.12-only

# 1. a SECOND venv, on 3.12 exactly. NOT .venv, and NOT inside the working tree.
export ISAAC_VENV=~/wam-t041/.venv-isaac
/home/humanoid/.local/bin/python3.12 -m venv "$ISAAC_VENV"
"$ISAAC_VENV"/bin/pip install --upgrade pip        # NVIDIA's docs ask for this first

# 2. torch FIRST, from the CUDA build that carries sm_120 kernels (§1.3). NVIDIA's own
#    install page gives this line BEFORE the isaacsim line; confirmed twice (§1.5).
"$ISAAC_VENV"/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Isaac Sim itself. This is the ~25 GB step and NVIDIA's own guides note the extension
#    cache stage runs 10-15 minutes with no progress output. It is not hung.
"$ISAAC_VENV"/bin/pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com

# 4. WHAT TORCH SURVIVED. Record it. If the resolver replaced 2.11.0, everything after this
#    is being run against a torch neither NVIDIA nor we chose, and no later check would say so.
"$ISAAC_VENV"/bin/pip show torch

# 5. WAM itself, base deps only — numpy + pydantic + pyyaml + typing-extensions, no torch.
#    `serve` is only needed for the --policy remote rollout topology, not for this measurement.
#    `-e` against the checkout is deliberate: the venv is outside the tree, the CODE is not.
"$ISAAC_VENV"/bin/pip install -e /home/humanoid/develop/wam'[serve]'
```

Every later command on this page writes `$ISAAC_VENV/bin/python`. Export it once per shell, or
substitute the path — but do not create a second copy of the venv under a second name, because then
"which torch is this" stops having one answer.

**On `isaac-python`.** `preflight_isaac.py`'s docstring and `isaac_binding.ISAAC_MISSING_MSG` both
say "Isaac Sim's own interpreter (`isaac-python`)". With the pip install above, that interpreter
**is** `$ISAAC_VENV/bin/python`; there is no separate binary. The rule those messages protect is
the one that matters: **never run the Isaac side out of `.venv`.**

**Network.** NVIDIA's requirements page states an internet connection is required for online
assets; `IsaacSimBinding` resolves its USD through `isaacsim.storage.native.get_assets_root_path()`
by default, i.e. it streams. If asset streaming is unreachable, preflight check F
(`asset_root_resolves`) is where you find out, and `--asset /abs/path/to/g1.usd` is the escape.

---

## 3. Verification — `preflight_isaac.py`, and it runs before anything else

```bash
cd /home/humanoid/develop/wam
"$ISAAC_VENV"/bin/python scripts/preflight_isaac.py \
    --ground-truth-annotators \
    --camera-hw 480 640 \
    --out runs/preflight/isaac-est-drift.json
```

Three flags, three reasons:

- **`--ground-truth-annotators` is not optional here.** It is off by default because no rollout
  needs depth or segmentation and failing the gate every rollout passes on an annotator none of
  them uses would block the box for nothing. It adds **check N**, which is the only thing that
  tests the two annotators `EST_DRIFT_P95` is measured against: `annotator_depth_attaches`,
  `annotator_segmentation_attaches`, `depth_is_hw_float`, `depth_has_finite_values`,
  `segmentation_ids_are_not_colorized`, `segmentation_carries_id_to_labels`. Without check N
  passing there is no ground truth and no budget.
- **`--camera-hw 480 640`**, not the 256×256 default, because that is the grid the measurement has
  to happen on — see §4.3, which is the trap that would otherwise waste the whole run.
- **`--out`** under `runs/` because `runs/` is gitignored: a preflight report is evidence, not a
  committed artifact.

**What a pass looks like.** Exit code **0**, and every line printed `[PASS]`. Read three things out
of the JSON even on a full pass:

- `info.dof_names` — the 43 names in PhysX's own order. This is *discovery*, not verification: the
  order is neither URDF's nor `G1_MOTOR_JOINT_NAMES`', and diffing this list after an asset upgrade
  is how a silent re-ordering gets caught.
- `info.body_joint_pattern` / `left_finger_pattern` / `right_finger_pattern` — which naming
  convention the shipped USD actually uses. A mismatch is a finding about the asset; the repair is
  appending the observed convention to `BODY_NAME_CANDIDATES` / `FINGER_NAME_CANDIDATES` in **both**
  `isaac_binding.py` and `preflight_isaac.py` (a test asserts the two copies are identical), never
  a fallback to positional indexing.
- `info.segmentation_id_to_labels` — **the one that decides §4.** On the bare `g1.usd` this is
  expected to be empty or robot-only, and an empty map there is *correct*: which semantics a stage
  carries is a property of the scene, not of the vendor API. See §4.2 — this is where the real
  blocker becomes visible.

**If it hangs rather than fails**, suspect §1.4 and kill it; the last printed line is the result.
**If a check contradicts what the binding believes, fix the binding, not the preflight** — the
preflight is the record of the assumption.

---

## 4. The measurement — and the four traps in the exact invocation

`scripts/measure_est_drift.py` is PR-08 §4 steps 1–4, split into `capture` (needs Isaac, no
estimator) and `measure` (needs the estimator, no Isaac). The split exists so the arithmetic is
testable where Isaac cannot boot, and so a re-run with a different estimator does not re-render.

### 4.1 The two commands

> **THE THREE COMMAND-LINE DEFECTS THIS BOX USED TO CARRY LANDED ON 2026-08-22**, in
> `scripts/measure_est_drift.py`, with tests. What follows is what they are now; the paragraph each
> one replaced is kept in §4.3 and §7 so the failure shape stays readable, because it is the shape
> that matters, not the flags.
>
> 1. **`--render-hw` exists and defaults to the committed contract.** It is read out of
>    `configs/transfer25/pr08_geom_tol.json` → `segmenter.pixel_grid_hw` at run time
>    (`contract_pixel_grid_hw`), never from a literal in the script — a second copy of that number
>    is exactly what would go stale the day the contract moved. A `--render-hw` that DISAGREES with
>    the contract is fatal **before Isaac boots**, naming `resolution_disagrees_with_geom_tol`,
>    because PR-08 §6 subtracts `EST_DRIFT_P95` from `GEOM_TOL` and that is arithmetic on one pixel
>    grid only. And if the contract is missing, there is no default: the run refuses and says
>    restore it from git rather than picking a number.
> 2. **`--asset` / `--scene` exist and are one knob**, spelled the two ways
>    `configs/robot/isaac_g1.yaml` (`sim.scene`) and `wam.robot.isaac_g1.IsaacG1Robot`
>    (`scene_path` is an alias for `asset`) already spell it; passing both is refused. The USD is
>    threaded into `IsaacSimBinding(asset=…)` and recorded in `capture.json` and in the measured
>    artifact's `capture` block, so a p95 taken over the bare `g1.usd` is distinguishable from one
>    taken over a real scene. **This does not create a scene — §4.2 is still the blocker.**
> 3. **`--camera` defaults to `persp`**, taken from `DEFAULT_CAMERA_PRIMS` itself, and an unknown
>    name is an argparse-time refusal against that dict instead of a `ValueError` after a full
>    Isaac boot. A stage that carries its own camera prim is named with
>    `--camera-prim ego=/World/Scene/EgoCam` and then selected with `--camera ego`.
>
> **What is still true: a capture from this path is not yet a gate-qualified `EST_DRIFT_P95`.** The
> reasons are §4.2 (no scene, so no apple to take a centroid of) and §4.7's table (the estimator's
> own gate qualification, and `GEOM_TOL` itself). Those are not command-line defects and no flag
> fixes them.

```bash
# STEP 1 — render ground truth. Isaac venv, on the box with the 5090.
cd /home/humanoid/develop/wam
"$ISAAC_VENV"/bin/python scripts/measure_est_drift.py capture \
    --out runs/pr08-est-drift/capture \
    --scene /abs/path/to/apple_to_plate.usd \
    --frames 256 \
    --steps-per-frame 50
# NOTE the ABSENCE of --render-hw, and it is the correct invocation: the grid defaults to the
# committed contract's pixel_grid_hw ([480, 640]), and passing anything else is refused before
# Isaac boots. Pass it only to state the committed grid explicitly.
# NOTE the ABSENCE of --camera: it defaults to `persp`, the viewport camera every stage has. A
# scene with its own camera needs --camera-prim ego=/World/... --camera ego.
# --scene is the stage that does not exist yet (4.2). Drop it and you capture the bare g1.usd,
# which has no apple in it, and the run measures nothing with no crash.

# STEPS 2-4 — estimate, compare, write the budget. NOT the Isaac venv (see 4.4).
# NOTE the ABSENCE of --object-class: as of 2026-08-22 it defaults to the estimator's own
# OBJECT_TEXT_PROMPT, and omitting it is the correct invocation. See 4.5.
<env-with-the-estimator>/bin/python scripts/measure_est_drift.py measure \
    --capture runs/pr08-est-drift/capture \
    --estimators estimators.apple_sam2 \
    --min-coverage 0.90 \
    --out configs/transfer25/pr08_est_drift.json

# STEP 5 — carry the number into the document the gate actually reads. NOT a text editor: this
# writes est_drift_p95_px AND est_drift_estimator_name together, and refuses either alone. The
# name is PR-08 §4 step 2's join key, and without it run_g0_gates can never establish that the two
# halves of GEOM_TOL - EST_DRIFT_P95 came from one segmenter — which costs every G0b run its gate
# qualification, for ever. It also refuses a disqualified or null measurement, a segmenter name or
# operating point that disagrees with the committed contract, and a different pixel grid.
python scripts/measure_geom_tol.py \
    --carry-est-drift configs/transfer25/pr08_est_drift.json \
    --out configs/transfer25/pr08_geom_tol.json
```

The artifact lands at **`configs/transfer25/pr08_est_drift.json`** — a *tracked* path, and that is
deliberate: PR-08 §8 item 4 requires both constants "measured **and committed**", and `runs/` is
gitignored, so an artifact written there could never be the pre-commitment the rule asks for. The
capture directory stays under `runs/`, because it is bulk evidence, not the commitment.

`--estimators estimators.apple_sam2` resolves without a `PYTHONPATH`: `measure_est_drift` inserts
`<repo>/scripts` on `sys.path` at import, and `scripts/estimators/__init__.py` exists.

### 4.2 Trap 1 — **there is no scene, and this is the real blocker**

The cheap half of this trap was `capture`'s `--camera` defaulting to `"ego"` while
`IsaacSimBinding`'s cameras default to `DEFAULT_CAMERA_PRIMS = {"persp": "/OmniverseKit_Persp"}`,
so the DEFAULT value got you `ValueError: unknown camera 'ego'; have ['persp']` after a full Isaac
boot. That half is closed (§4.1): the default is now `persp`, taken from that dict, and an unknown
name is refused at argument-parse time. The expensive half is untouched:

**With the bare `g1.usd` there is no table, no plate, no apple, and exactly one camera — the
viewport camera.** `configs/robot/isaac_g1.yaml`'s own comment says so: *"There is no Isaac
equivalent of `configs/sim/g1_scene.xml` yet — no table, no cube, no head/wrist cameras."* And
`semantic_segmentation` only returns ids for prims carrying a semantics schema, so on that stage
`idToLabels` is empty or robot-only.

Run the two commands above against the bare asset and here is what happens, exactly, in code:
`object_ids(id_to_labels, "apple")` matches nothing → every frame increments
`n_frames_without_object_label` and appends `(None, None)` → `paired_displacements` drops all of
them → `coverage == 0.0` → `est_drift_p95_px: null`, `headline_valid: false`,
`coverage_below_floor`, exit 3. **The rig runs to completion, writes an artifact, and measures
nothing.** No crash, no traceback — which is precisely the failure shape this repository keeps
naming, and the reason it is written out here rather than left to be discovered on the box.

So §4 step 1's "render N Isaac episodes" needs an **AppleToPlate Isaac scene**, and there isn't
one. What exists:

- **Upstream, and it is the right scene**: `isaac-sim/IsaacLab-Arena` ships a *Unitree G1 Static
  Apple-to-Plate Task* (docs: `isaac-sim.github.io/IsaacLab-Arena/release/0.2.1/pages/example_workflows/static_apple/`,
  fetched 2026-08-22), with a matching HF dataset `nvidia/Arena-G1-Static-PickNPlace-Task` — **251
  episodes and cc-by-4.0, both confirmed on the dataset card 2026-08-22; the "teleop at 50 Hz" this
  page first carried is UNVERIFIED**, neither the task page nor the dataset card states a rate. That is a **second, larger install** — Docker, recursive
  submodules, a pinned Isaac Lab — and it is not wired to `IsaacSimBinding` in any way.
- **On this box, and it is not the scene**: `~/models/isaaclab_arena/static_apple_tutorial/` is 12
  GB of `GN1x-Tuned-Arena-G1-Static-PickNPlace` **checkpoint** — `find` reports **zero `.usd*`
  files** (measured 2026-08-22). It is a policy, not a stage.

Whoever takes this on has three options and none is small: author a minimal USD (table + plate +
apple + a `Semantics` label on each, plus an ego-like camera prim), pull IsaacLab-Arena's task and
teach `capture` to drive it, or take one of §6's routes. **This, not the install, is what makes
Isaac the slowest path to a number.**

### 4.3 Trap 2 — the render grid, which now has a flag and a default that cannot go stale

**Closed 2026-08-22.** What follows is the trap as it was, because the shape is worth keeping: the
render succeeded, the artifact was written, and only a disqualifier said the number was unusable.

`capture` used to construct `IsaacSimBinding(ground_truth=("depth", "segmentation"))` with **no
`render_hw`**, so it took the constructor default of **(256, 256)**. The grid the other side is on
is no longer an inference: `configs/transfer25/pr08_geom_tol.json` — the committed segmenter
contract, written 2026-08-22, which commits the *method* and leaves `geom_tol_px: null` because
neither number is measured yet — records **`pixel_grid_hw: [480, 640]`**, matching the PR-08 source
corpus (`${PROJ}/data/pr08-apple-640x480`, 402 episodes / 171 625 frames, verified on the cluster
2026-08-20). §6 computes `GEOM_TOL − EST_DRIFT_P95`, and `cross_check_geom_tol` reads that
`pixel_grid_hw` as the grid to agree with and refuses the subtraction across grids: a 256×256
capture stamps `resolution_disagrees_with_geom_tol` and the run is disqualified.

**Both flags now exist** (`--render-hw`, and `--asset`/`--scene` as one knob), and the important
part is not that the flag exists but where its DEFAULT comes from: `contract_pixel_grid_hw()` reads
`segmenter.pixel_grid_hw` out of the committed contract at run time. Writing `480 640` into
`measure_est_drift.py` would have been a second copy of a pre-commitment that lives in one place on
purpose — the day the contract moved, `capture` would keep rendering the old grid and every run
would be disqualified with nothing in the record naming the stale copy. A `--render-hw` that
disagrees with the contract is refused **before Isaac boots**, and a missing contract is refused
outright rather than defaulted around. Nothing here needs an operator workaround any more; §4.2
still does.

### 4.4 Trap 3 — which interpreter runs `measure`, and where the weights are

`measure` needs `sam2`, `transformers` and `torch`. Measured on this box 2026-08-22:

- `.venv` has torch 2.13.0, transformers, cv2 — **and `sam2` 1.1.0, with its hiera configs**.
  An earlier revision of this page said `sam2` was absent; that was true when first written and
  stopped being true at **15:45 on 2026-08-22**, one minute after the checkpoints landed
  (`site-packages/sam2/__init__.py`, mtime; `sam2.build_sam.build_sam2_hf` and
  `sam2.sam2_image_predictor.SAM2ImagePredictor` both import). With the three checkpoints also in
  `~/.cache/huggingface/hub` at the pinned revisions (4.8 GB, §2), **`apple_sam2.available()`
  returns `True` in `.venv` on this box** — re-measured 2026-08-22 by running it. So neither the
  weights nor the package is a reason `measure` cannot run here any more, and **the routing
  argument below rests on AC-04 alone**. (`available()` True means the pins resolve and the imports
  work; it is not a claim that a full `measure` has been run here, and nobody has.)
- `$ISAAC_VENV` will have torch 2.11.0 and neither `sam2` nor `transformers`.
- **The Cosmos-Transfer2.5 venv on Discoverer+ has `sam2` 1.1.0 with its hiera configs, and job
  189583 staged all three checkpoints (5.0 GB) at the exact revisions
  `scripts/estimators/apple_sam2.py` pins** (T-040 notes, 2026-08-22).

So the intended shape is: **`capture` here, `measure` there.** Copy `runs/pr08-est-drift/capture/`
to the cluster and run `measure` in the Transfer2.5 venv. **One reason survives, and it is enough:**
the cluster's copies are the *staged, checksum-verified* ones recorded in
`PR08_ESTIMATORS_STAGED.json`, which is the difference between a traceable gate number and an
untraceable one (AC-04). The two reasons this page used to give beside it — the download, and
`sam2` being missing here — are **both retracted as of 2026-08-22**: the weights are cached here at
the pinned revisions and `sam2` 1.1.0 is installed here too. A `measure` run in `.venv` on this box
would therefore *work*. What it would not have is the **independent** verification: the artifact
records the requested repo + 40-hex revision for all three checkpoints either way
(`apple_sam2.stats()`), but on the cluster those revisions were fetched **and checksum-verified**
into `PR08_ESTIMATORS_STAGED.json` by job 189583, while here they are only whatever
`huggingface_hub`'s own cache bookkeeping says is under that snapshot directory. The split into two
subcommands was designed for exactly this. **If somebody does run `measure` here anyway** — as a
smoke test, which is legitimate — say so in the artifact's provenance rather than letting "it ran"
stand in for "the weights were verified".

The capture is not large: 256 frames at 640×480 is roughly `rgb` 236 MB + `depth` 315 MB (float32)
+ `seg_ids` ~315 MB uncompressed `.npy`, so **under 1 GB**. Arithmetic, not measured.

### 4.5 Trap 4 — the object is named once now, but the *scene's* label is still a third name

`--object-class` is the label looked up in the capture's `idToLabels`; the estimator's own target
is `OBJECT_TEXT_PROMPT` (from `$WAM_PR08_OBJECT_PROMPT`, default `"apple."`). These used to be two
independent knobs with nothing comparing them, so an apple mask could be scored against a plate's
ground truth and produce a large, entirely plausible p95 with no crash and no drop in coverage.

**That has been closed, as of 2026-08-22** (`resolve_object_class`): `--object-class` now
**defaults to the estimator's own prompt**, an explicitly typed value that disagrees is **fatal**
rather than disqualifying — there is no number worth writing down when every displacement is a
distance between two different objects — and an estimator that declares no prompt carries
`estimator_does_not_declare_object_prompt` into the artifact so it cannot be read as having been
checked. **So do not pass `--object-class` at all.** Passing it can only agree (no effect) or
disagree (fatal).

**What is still open is the third name: the scene's own label text.** `object_ids` matches on
`strip().lower()` equality against whatever the USD's semantics schema carries, so **case is the one
difference it forgives** — a stage labelling the fruit `Apple` matches. The trailing period is
stripped on the *prompt* side only: `_normalise_object` turns GroundingDINO's `"apple."` into
`apple` before the lookup, so the estimator's own notation is never the problem — but a **stage**
whose label text is `apple.` still does **not** match, because `object_ids` only lowercases and
strips whitespace on the label it reads. Nothing else is forgiven: if
the stage calls it `apple_01`, `Fruit` or `red apple`, nothing matches and you land in §4.2's failure — full
run, zero coverage, `est_drift_p95_px: null`. Nothing checks that one, because nothing can: it is a
property of the scene. Read `label_vocabulary_seen` out of the artifact, which is recorded for
exactly this reason.

### 4.6 What N should be

§4 says "N Isaac episodes". `capture` implements **frames**, not episodes: it loops
`binding.step()` × `--steps-per-frame`, renders, writes. Nothing drives the arm — there is no
policy and no teleop on this path — so on a static stage N frames are N nearly identical samples
and the "95th percentile" is a percentile over one viewpoint of one configuration. That is not what
§4 means and it would be a fraudulent-looking number even though every line of code did its job.

**N has to be counted in distinct configurations, not frames.** Recommendation, and it is a
recommendation rather than a derivation because no power analysis exists for this quantity:

- **≥ 200 measured frames** for a p95 to be worth quoting at all. At the 0.90 coverage floor,
  `--frames 256` leaves ~230 measured and a p95 resting on the ~12 largest values. Below ~100 the
  p95 is essentially the 5th-largest sample and moves under reseeding.
- Those frames must span **≥ 20 distinct scene states** — apple pose, arm configuration, and
  ideally lighting — because estimator error is scene- and distance-dependent and a p95 over one
  pose measures one pose. `--steps-per-frame 50` (0.1 s of sim at 500 Hz) is how you spread frames
  across a *moving* scene once something is moving it; on a static stage it changes nothing.
- Recording the count of distinct configurations alongside `n_measured` is what makes the number
  auditable. The artifact does not have a field for it today. Put it in the commit message.

### 4.7 What comes out, and why it will be exit 3 anyway

Exit codes: **0** = gate-qualified; **2** = nothing measured; **3** = measured, but must not be
subtracted from `GEOM_TOL`. The artifact is written on 3, because "we tried and this is what came
out" is a record.

**A perfect Isaac run today still exits 3**, and it is better to know which flags are lit before
buying the GPU-hours than after:

| disqualifier | why it is lit right now | what clears it |
|---|---|---|
| `estimator_not_gate_qualified` | `apple_sam2.GATE_QUALIFIED = False`. Its blocker tuple was rewritten 2026-08-22 and **is the thing to read, not this row** — see the note below | discharging every entry in `GATE_QUALIFICATION_BLOCKERS`; each names the evidence that would do it |
| `geom_tol_does_not_record_gate_qualified` → `geom_tol_is_not_gate_qualified` | `configs/transfer25/pr08_geom_tol.json` now exists (2026-08-22) but commits the **method**, not the result: `geom_tol_px` is `null` and the file carries no `gate_qualified` key. `GEOM_TOL` is still blocked on the AV1 decoder work | measuring and committing `GEOM_TOL` itself |
| `resolution_disagrees_with_geom_tol` | **no longer lit** as of 2026-08-22 (§4.3): the capture defaults to the committed `pixel_grid_hw: [480, 640]`, and a `--render-hw` that disagrees is refused before Isaac boots | — |
| `segmenter_params_disagree_with_geom_tol` | not lit if `capture`/`measure` run against the committed contract unchanged — but note it pins `box_threshold: 0.15` and a retry pair, i.e. **not** the adapter's old demo defaults. Whatever the adapter carries at run time must equal this file field for field | leaving both alone, or amending both together under review |
| `estimator_does_not_declare_segmenter_contract` | added 2026-08-22. **Not lit for `apple_sam2`**, which declares `SEGMENTER_CONTRACT` — listed so a third-party or stub estimator's silence is not mistaken for agreement | the adapter declaring prompt, thresholds, retry and box rule |
| `capture_is_not_from_isaac_sim` | any capture from `FakeIsaacBinding`, i.e. everything anyone has run so far | a real `IsaacSimBinding` capture |
| `coverage_below_floor` | §4.2 — no apple in the stage | a scene |

**The estimator's blockers moved on 2026-08-22 and one of them changes what this number even is.**
Two were discharged or narrowed by measurement — the adapter has now run end to end over the
AppleToPlate corpus, and its thresholds are no longer copied demo defaults but
Cosmos-Transfer2.5's own operating point, which is what §4 step 2 asks for.

*One caveat on that first half, because this page's value is that it does not overclaim.* The
end-to-end evidence is **cluster job 189588** — "720 frames, two passes, 480×640, coverage 1.0 on
both" — and **its artifact is not in this working tree**: `measure_geom_tol.py` and
`103_measure_geom_tol.sbatch` both cite it at `runs/pr08-geom-tol/GEOM_TOL_PILOT.json`, `runs/` is
gitignored, and the local `runs/` has no `pr08-geom-tol/` at all (checked 2026-08-22). T-040's job
log records 189583–189586 and does not mention 189588. So on this box the job is a **citation, not
an artifact** — it is corroborated only by the several uncommitted files that cite it, which is
weaker than it reads. Two consequences worth carrying: it is unaudited here, and it necessarily ran
at the **old** operating point (`box_threshold=0.35`, no retry branch), so it is evidence about a
configuration this change has since replaced. Whoever discharges blocker 1 or 2 should fetch that
artifact rather than cite the job number.

**`GATE_QUALIFICATION_BLOCKERS` in `scripts/estimators/apple_sam2.py` is the list, and it is the
only list** — this section names its entries rather than counting them, because a count in prose is
a number that goes stale the next time somebody discharges one, and an undercount here reads as
permission. Read the tuple, not this page, for the current set. As of **2026-08-22 16:53** — a
timestamp and not just a date, because this tuple moved twice in the hour this page was written —
its entries, in its own order, were:

- **(1) nobody has looked at a mask** — below.
- **(2) the operating point is unmeasured on this corpus.** 0.15/0.25 with a (0.10, 0.10) retry is
  Cosmos-Transfer2.5's own operating point, which is what §4 step 2 asks for, so the *choice* half of
  this blocker is discharged and inverted. What survives is that nothing has measured what it does on
  AppleToPlate, and the retry buys detections by accepting weak boxes — it inflates coverage, the one
  number the harness gates on, while degrading the mask.
- **(3) per-frame segmentation is not upstream's propagation** — below.

**A fourth entry was in this tuple when the paragraph above was first written and is not any more,
and it is the one an operator most needs the current answer to.** *"The committed contract can be
overwritten by the measurement it constrains"*: `measure_geom_tol.py`'s default `--out` and
`--merge` target is `configs/transfer25/pr08_geom_tol.json`, the same path as the committed
segmenter contract, so measuring `GEOM_TOL` used to write a different schema over it, the
`segmenter` block vanished, and the cross-check then reported
`geom_tol_does_not_record_segmenter_params` and refused every later run — failing closed, and closed
forever, which made the gate **unreachable** no matter how good the Isaac capture was. It moved to
`GATE_QUALIFICATION_DISCHARGED` at 16:43 on 2026-08-22, and `measure_geom_tol.py` now carries
`merge_committed_contract()` (compares the on-disk `segmenter` block field for field against the
adapter that just ran, refuses the whole run on any disagreement, else copies the contract forward
verbatim) and `refuse_default_out_without_contract()` (refuses to write the tracked path at all when
no contract is sitting in it) — both verified present in the working tree. **So this page no longer
reports the gate as unreachable.** It is written out rather than deleted because a blocker that
merely disappears between two drafts is indistinguishable from one somebody dropped, and because
`apple_sam2.py`'s own opt-in comment still reads *"FOUR conditions above are open"* over a
three-entry tuple: that comment is stale, the tuple is not, and **the tuple is the record**.

Two of the three are load-bearing for *this* page:

- **Nobody has looked at a mask.** Coverage 1.0 says a box came back on every frame, not that it was
  the *apple's* box. This adapter's failure mode is a plausible mask on the plate or the hand, which
  produces a centroid, a displacement and a p95 that all look like measurements.
- **Per-frame segmentation is not upstream's propagation.** Cosmos drives
  `SAM2VideoPredictor` and propagates one mask across a clip; this adapter re-detects every frame.
  The blocker spells out that the bias is **two-sided** — per-frame jitter inflates the tail (safe),
  while propagation's drift-and-stay-off failure is invisible to a per-frame estimator (unsafe) —
  and concludes that with both in play the number **is neither a lower nor an upper bound**.

That is worth pausing on, because it strengthens §6 rather than weakening it. `is_lower_bound: true`
is stamped unconditionally and is not a flag; it encodes §4's stated weakness about synthetic
renders, and **the stamp is right** — the *reason string* the script writes beside it is not, which
is a separate defect written out in §6(b). But the *estimator's* own blocker now says the bound
direction is unknown for a second, independent reason. **A number whose direction is unknown is worth less than one whose direction is
known to be conservative**, which is exactly the axis §6 ranks on.

---

## 5. Wall clock, honestly

Two columns, because the difference between them is the finding.

| step | if the scene existed | what it actually is today |
|---|---|---|
| download + `pip install` (~25 GB, ~3 GB torch) | 30–90 min, network-bound | same |
| first Isaac boot — shader cache compilation | 5–20 min, **unverified**; it looks like a hang and is not | same |
| `preflight_isaac.py --ground-truth-annotators` | 5–15 min the first time, minutes after | same, **plus** however long the first FAIL takes to repair — check G's naming convention is a two-file edit, check N is a vendor-API surprise nobody has seen |
| **author an AppleToPlate Isaac stage with semantics** | — | **1–5 days.** USD authoring, semantic labelling, camera placement, a way to vary the apple pose across N configurations. Or an IsaacLab-Arena install on top, which is Docker + submodules + a pinned Isaac Lab |
| ~~add `--render-hw` / `--asset` to `capture`~~ (§4.3) | — | **done 2026-08-22**, with tests. It was hours, and it was the cheap item on this table |
| render 256 frames at 640×480 with 3 AOVs | minutes. Three `render()` calls per frame on a 5090; boot dominates | same |
| `measure` — 3 model loads + one forward pass per frame | ~5 min on the cluster with weights staged; +5–10 min and ~3 GB if fetched fresh | same |
| **retire `estimator_not_gate_qualified`** (§4.7) | — | **1–3 days**: a human looking at a sample of overlaid masks spanning the corpus, plus the both-ways propagation comparison the propagation blocker asks for, plus a reviewed edit to the blocker tuple |

**Best case, everything green, scene in hand: half a day.** **Realistic from today: a week**, and
roughly one hour of that is Isaac Sim installing. The install is not the cost. The scene and
gate-qualifying the estimator are the cost — the third item, the missing `capture` flags, was paid
on 2026-08-22 — and **neither of the two that remain is made cheaper by having Isaac installed.**

---

## 6. The alternatives — and why Isaac is not the cheapest lower bound

PR-08 §4 concedes it in its own words: *"Isaac frames are not real frames, and a monocular
estimator's error on synthetic renders is not its error on RealSense footage — plausibly optimistic.
So `EST_DRIFT_P95` is a **lower bound** on the real error."* §6 then **subtracts** it from
`GEOM_TOL`. Subtracting a lower bound leaves the tolerance **too wide**, so the error always lands
in the generator's favour. A number that errs the other way is worth strictly more to G0b than a
tighter number that errs this way, and that reframes the whole comparison: we are not shopping for
the most faithful renderer, we are shopping for the cheapest defensible bound that does not err
optimistic. (§4.7's propagation blocker sharpens this further: the estimator's per-frame-vs-propagation
difference means the Isaac number's direction is not even reliably "lower" — it is *unknown*, which
is worse than either.)

**The load-bearing observation that opens the field:** `EST_DRIFT_P95` is defined purely on
**segmentation** — the p95 centroid displacement between the estimated mask and the true mask.
The depth half of §4 step 3 is **recorded, not gated** (`apple_sam2`'s docstring says so outright:
*"It is recorded, not gated"*). So a route that supplies ground-truth **masks** but no ground-truth
**depth** still produces the gated number in full, and loses only a recorded diagnostic. Three of
the four routes below exploit that.

### Ranked by time-to-a-number

| rank | route | time to a number | direction of its error | §4 amendment needed |
|---|---|---|---|---|
| **1** | **(c) composite on real frames** | **~1 day** | lower bound, for a *different* reason | steps 1–3 rewritten |
| **2** | **(a) MuJoCo** | **1–2 days** | **conservative** (harder frames ⇒ larger p95) | one word: "Isaac" → "a simulator with ground-truth segmentation" |
| **3** | **(b) an external real RGB-D + mask corpus** | **3–10 days**, licence-bound | **conservative**, and *removes* §4's stated weakness | steps 1–2 replaced; §3's own anti-HE argument must be answered |
| **4** | **Isaac, as written** | **~1 week** (§5) | lower bound, optimistic | none — it is the letter |

Read that table once more: **the option that requires no amendment is the slowest and produces the
weakest kind of bound.**

### (a) MuJoCo — the renderer this repo already has

**Verified on this box, 2026-08-22**, with no install and no network:

```
mujoco 3.10.0 importable in .venv
mujoco.Renderer methods: enable_depth_rendering, disable_depth_rendering,
                         enable_segmentation_rendering, disable_segmentation_rendering
```

So ground-truth depth **in metres** and per-pixel **geom-id segmentation** are one method call
away, on CPU, headless, in the venv that is already built. And the scene the Isaac path is missing
already exists on this side of the seam: `configs/sim/g1_scene.xml` has the 43-DoF G1, a **static
table** with a measured top at z = 0.72 m, a graspable **cube**, and two real camera prims (`head`,
`wrist_left`) — all of it inside the measured reach envelope, with a `ready` keyframe.

**Work required.** `capture_frames` takes a binding rather than building one *precisely* so the
caller owns the boot, so this is a shim, not a fork: one class implementing `ground_truth_channels`,
`step()`, `render_frame`, `render_depth`, `render_segmentation`, `get_physics_step_count`, plus a
mapping from geom ids to an `idToLabels` dict in Replicator's `{"class": "apple"}` shape. Call it
~150 lines and a test file. Nothing in `measure` changes.

**What it costs, and this is the honest objection: an orange cube is not an apple.** GroundingDINO
would be prompted `"orange cube."` and `--object-class` set to match, and the resulting number is a
budget for *finding a cube in a MuJoCo render*, transferred to *finding an apple in a RealSense
frame*. Adding an apple mesh to the MJCF is possible (MuJoCo loads OBJ/STL) but no such mesh is in
this repo — **unverified whether a suitable one is reachable offline**; that is the first thing to
check before choosing this route.

**Which way its error points, and this is why it ranks second rather than fourth.** MuJoCo's
rasterizer is markedly less photoreal than Isaac's RTX path, so a detector trained on photographs
will do **worse** on MuJoCo frames than on Isaac frames. A larger p95 makes `GEOM_TOL − EST_DRIFT_P95`
**smaller**, i.e. G0b **stricter**. It errs against the generator, which is the safe direction — the
opposite of the failure §4 warns about. It is arguably not a lower bound at all but an upper-ish
one, and saying so is a claim that needs measuring, not asserting.

**Amendment required — the cheapest of the three.** §4 step 1 reads *"Render N Isaac episodes with
ground-truth depth + segmentation"*. A `PR-08-V4` would have to generalise the renderer:
*"Render N episodes in a simulator that emits exact per-pixel object segmentation"*, plus a
sentence acknowledging that the segmented object may be a stand-in for the apple and what that
costs. §3's comparison table (*"depth + segmentation: exact"* for the sim path) survives unchanged
— MuJoCo's is exact in the same sense Isaac's is. It also collides with **nothing**: §3 rejected
the Isaac path *as the corpus*, on trajectories, and this is calibration, not corpus, so T-25's
"sim frames are NOT training data" is untouched. Additionally `measure_est_drift`'s
`capture_is_not_from_isaac_sim` disqualifier — which hard-codes `type(binding).__name__ !=
"IsaacSimBinding"` — would need widening to an allow-list, and that edit should be reviewed
carefully, because it is the check that keeps a laptop capture out of a gate.

### (b) An existing corpus that already ships ground truth

Two sub-cases, and they are not equal.

**Humanoid Everyday — cannot do it, and this should be recorded before someone tries.** T-040 and
PR-08 §3 both hold HE up as the confirmatory measurement, and for *depth* it is exactly right: same
camera (one egocentric D435), same embodiment, real measured depth, published intrinsics and
extrinsics. But `EST_DRIFT_P95` is a **segmentation** quantity — PR-08 §4 step 4 is *"the **95th
percentile of that centroid displacement**"*, and step 3's absolute depth error is **recorded, not
gated** — and HE ships **no segmentation masks**. Its own dataset card describes the streams as
*"multimodal sensory streams at 30 Hz (RGB, depth, LiDAR, tactile, IMU)"*, and its observation
schema lists depth, LiDAR, IMU, odometry and joint states with no mask field (fetched 2026-08-22).
*That is an absence of mention rather than an explicit denial, so treat it as strong evidence and
not as proof; whoever plans this route should read the repository file listing before committing to
it.* HE can settle §4 step 3's *recorded* depth error; it cannot produce the *gated* number without
somebody hand-labelling masks, at which point the annotation is the instrument. This is a correction
to the loose reading of §4's "HE would settle it" — it settles half.

> **A drifted copy of this claim is stamped into every artifact `measure` writes, and it is wrong
> twice.** `scripts/measure_est_drift.py`'s `is_lower_bound_reason` (line ~847, anchor: the string
> *"measured on Isaac renders, not RealSense footage"*; the same wording is in the module docstring
> at lines ~73-75) reads: *"The confirmatory measurement against Humanoid Everyday is blocked on
> that corpus's licence and is deliberately off the critical path."* Both halves have expired:
>
> - **The licence half is stale against PR-08 itself.** §3 was amended on 2026-08-07 to say, in its
>   own words, *"the substitution **survives OD-09**, which now permits training on HE (2026-08-07).
>   **The reason is no longer the licence**"* — the reasons it gives instead are that Isaac's depth
>   is exact where HE's is a sensor measurement, and that HE is 247 other tasks. §3 then states
>   plainly that HE *"is now available to do that; it is still not required."* So the artifact
>   asserts a blocker the pre-registration withdrew two weeks earlier.
> - **The "confirmatory measurement" half is wrong in kind, not just in date.** HE cannot confirm
>   `EST_DRIFT_P95` at all, licence or no licence, because that number is defined on segmentation
>   and HE ships none. It can confirm the *recorded* depth error, which is not the gated quantity.
>
> **The stamp `is_lower_bound: true` is not in question** — PR-08 §4's own stated weakness (synthetic
> renders are plausibly optimistic) is exactly right, and §4.7 argues it is if anything too
> generous. It is the *justification string beside it* that is wrong, and a wrong justification
> travels into the gate record beside a correct number. **This page does not own that script and has
> not edited it.** Rewriting a reason string that quotes a pre-registration verbatim is a judgement
> for whoever owns PR-08, not a constant-fix — and the replacement has to say something true about
> two different things (why Isaac is a lower bound; what HE can and cannot confirm) where the
> current string says one false thing about both.

**A real RGB-D corpus with instance masks — the strongest number available, and the largest
domain gap.** Public tabletop RGB-D datasets with per-object masks exist (OCID, GraspNet-1Billion,
YCB-Video and HOPE are the usual candidates; **unverified** whether any contains an apple-like
object, and that is the first thing to check). What such a corpus buys is the thing no simulator
can: **the frames are real camera frames**, so §4's stated weakness — *"a monocular estimator's
error on synthetic renders is not its error on RealSense footage"* — **disappears entirely**. The
number stops being a lower bound *for that reason*.

Two costs replace it. First, human-annotated masks are not exact ground truth; annotation error
enters as apparent estimator error and **inflates** the p95 — conservative, so it errs the safe way,
but it must be recorded as an inflation and not as estimator error. Second, the domain gap is the
one §3 already argued against for HE, in its own words: *"Estimator error is scene- and
distance-dependent, so HE's budget would be transferred across a domain gap anyway."* That argument
applies with more force to a corpus that is not even this embodiment. **Whoever proposes this route
has to answer §3's objection, not route around it** — the honest form of the answer is probably
"the real-frame gain outweighs the scene gap", which is a claim, and PR-08's style is that claims
of that shape get measured or get labelled.

**Amendment required.** §4 steps 1–2 replaced outright: *"obtain N frames of a comparable tabletop
scene shipping per-pixel object masks"*, plus a licence check, plus the inflation caveat, plus a
paragraph reconciling with §3's rejection of HE-as-calibrator. And §4's depth half either moves to
a real-depth corpus or is dropped. This is the largest textual amendment of the three, which is
why it ranks below MuJoCo on cost despite producing the better number.

### (c) Synthetic-but-not-simulated — composite a known matte onto the real frames

**The cheapest, and it is not close.** Take the PR-08 source corpus, which already exists at full
resolution on the cluster and has been verified against its own metadata (`${PROJ}/data/pr08-apple-640x480`,
402 episodes / 171 625 frames, 640×480). Paste an apple sprite with a **known alpha matte** at known
positions and scales over the real frames, run `apple_sam2.segment` on the composite, and compare
the estimated centroid against the matte's centroid. **You own the ground-truth mask by
construction, because you drew it.** No renderer, no simulator, no install, no new environment —
`measure`'s arithmetic is untouched, and the capture directory can be written by a fifty-line
script in the format `load_capture` already reads.

Two further advantages that are not obvious:

- **The background is the actual scene.** Same camera, same tablecloth, same lighting, same D435
  noise — every pixel except the object is exactly the distribution the generator will meet. The
  scene- and distance-dependence §3 worries about is not transferred; it is native.
- **It measures the grid `GEOM_TOL` is measured on**, automatically, because it is the same frames.
  §4.3's resolution trap cannot occur.

**Its weakness, stated as plainly as §4 states Isaac's.** A composited apple has cleaner edges and
a lighting mismatch against the plate it sits on. Both cut both ways: cleaner edges make
GroundingDINO's job *easier* (optimistic, a lower bound again — for a different reason than Isaac's),
while the lighting mismatch can make it *harder* (conservative). The net direction is **unknown
without measuring it**, and that is the honest statement to put in the artifact — not "lower bound",
not "conservative", but "direction unmeasured, here is why". A cheap partial answer: composite the
*same* apple onto frames where the real apple is also visible and check whether the estimator
prefers one; that is a day's work on top.

Depth is not available on this route at all, so §4 step 3's depth error is simply not produced —
recorded, not gated, so the gate survives.

**Amendment required.** §4 steps 1–3 rewritten: *"obtain N frames whose object mask is known by
construction"*, with the depth clause struck or deferred, and the `is_lower_bound` stamp replaced
by an honest direction-unknown statement — which means editing `measure_est_drift.py`'s
unconditional `is_lower_bound: true` and its reason string, currently quoting §4 verbatim. That is
a small code change with a large meaning, and it should be made by the same review that accepts the
amendment, not before it.

### (d) The non-option, named so it stays closed

**Assuming `EST_DRIFT_P95 = 0` is not on this list.** §4's last sentence exists to forbid exactly
that: the number *"enters G0b's tolerance as a budget rather than being assumed to be zero"*, and
§6 says generation does not start if `GEOM_TOL − EST_DRIFT_P95 ≤ 0`. Assuming zero converts a gate
that might refuse into a gate that cannot. It is written here only so that nobody has to re-derive
why it is unavailable.

### The recommendation, and whose call it is

If the project owner wants a number **this week**, (c) is the route, with (a) as the cross-check —
two independent estimates whose errors point in different directions bracket the truth far better
than one Isaac number that is known to be optimistic. If the owner wants the number PR-08 §4
literally specifies, it is Isaac, it is roughly a week, and the scene is the work.

**Either way the amendment is the owner's call, not a session's.** §4 names Isaac; `T40_RULE_V1` is
registered and is not edited; a `PR-08-V4` sits alongside it or nothing changes.

---

## 7. What on this page is verified, and what would verify the rest

**Measured here, 2026-08-22:** the card, VRAM, driver 595.84 and compute capability 12.0; glibc
2.39; Ubuntu 24.04.4; `python3.12` at 3.12.13; 771 GB free on `/` (782 GB six hours earlier — this
figure moves, re-run `df -h /`); `~/wam-t041/` on the same filesystem as the repo, 54 GB used;
`git check-ignore -v .venv-isaac/x` matching nothing (§2); `mujoco` 3.10.0 in `.venv` with the
depth and segmentation renderer methods; `sam2` **1.1.0 present** in `.venv` (installed 15:45); SAM 2,
GroundingDINO and Depth-Anything-V2 all **present** in `~/.cache/huggingface/hub` at the revisions
`apple_sam2.py` pins, 4.8 GB total (they arrived 2026-08-22 15:43–15:44); and consequently
`apple_sam2.available()` returning **`True`** in `.venv` here; `~/models/isaaclab_arena/` containing
zero `.usd*` files.

*Two of those lines said the opposite earlier the same day, because the packages and the weights
landed at 15:43–15:45 while this page was being written.* Both are corrected above, and the lesson
is worth more than either fact: **on this box a "measured" line has a timestamp for a reason —
re-run the check rather than trusting a sentence, including these.*

**Read from NVIDIA's documentation, 2026-08-22, and re-read independently the same day** (URLs,
citations and per-source verdicts in the table at the end of §1.4; what the second read changed is
§1.5): the 6.0 and 5.1.0 requirement tables, the Python 3.12 pin, the torch 2.11.0 cu128/cu130 lines
**and NVIDIA's ordering of torch before `isaacsim`**, the `isaacsim[all,extscache]==6.0.1.0` install
line, GLIBC 2.35+, and PyPI's 6.0.1.0 release date. All confirmed. Two citations were corrected by
the second read — IsaacLab #4951's card is a 5090 **Laptop** GPU (§1.4), and the Arena dataset's
50 Hz is not published (§4.2) — and both are marked where they appear.

**Read from this repository:** every flag, default and exit code in §3 and §4, and every
disqualifier in §4.7.

**Unverified, and what would settle each:**

| claim | what would verify it |
|---|---|
| the 5090 runs Isaac Sim 6.0.1 at all | `preflight_isaac.py` exiting 0 on this box |
| the ~25 GB install size | `du -sh ~/wam-t041/.venv-isaac` after step 3 |
| the annotator path is clear of the sm_120 Replicator hang (§1.4) | preflight check N returning at all |
| first-boot shader-compile time | a stopwatch on the first `preflight_isaac.py` |
| the composite route's error direction (§6c) | the same-frame comparison described there |
| an apple mesh reachable offline for the MuJoCo route (§6a) | a look at what MuJoCo Menagerie and this box already carry |
| whether any public RGB-D+mask corpus contains an apple (§6b) | reading the object lists before downloading anything |
| Isaac Sim 5.1.0's Python and torch versions (§1.2, UNVERIFIED — nothing on this page depends on them) | fetching 5.1.0's own install page, which nobody has |
| that HE ships no segmentation masks (§6b) — its dataset card lists RGB/depth/LiDAR/tactile/IMU and no masks, which is an absence of mention | the repository file listing, read before anyone plans that route |
| job 189588's 720-frame end-to-end run (§4.7) — cited by several uncommitted files, artifact not in this tree | fetching `runs/pr08-geom-tol/GEOM_TOL_PILOT.json` from the cluster |

**Four defects found while writing this. The first three were fixed on 2026-08-22** by a later
session, in `scripts/measure_est_drift.py`, with a test for each refusal; they are kept here with
their original diagnosis because the *shape* of each is worth more than the flag that closed it:

1. ~~`measure_est_drift.py capture` has no `--render-hw`~~ (the binding was built with no
   `render_hw=`, taking `isaac_binding.py`'s `(256, 256)`), so it always captured 256×256 and was
   always disqualified against the committed `pixel_grid_hw: [480, 640]`, while PR-08 §6 requires
   both sides of the subtraction on the same pixel grid (§4.1, §4.3). **FIXED**: the flag exists and
   its default is read out of the committed contract at run time; a disagreeing value is fatal
   before Isaac boots; a missing contract is fatal rather than defaulted around.
2. ~~It has no `--asset`/`--scene` either~~, so it could only ever capture the bare `g1.usd`.
   **FIXED**: one knob under two spellings, threaded into `IsaacSimBinding(asset=…)` and recorded in
   the capture header and the measured artifact. **This does not produce a scene** — §4.2 is
   unchanged and is still the real blocker.
3. ~~`--camera` defaults to `"ego"`~~, which no default Isaac stage has, so the default value raised
   after a full boot. **FIXED**: the default is `persp`, taken from `DEFAULT_CAMERA_PRIMS` itself,
   an unknown name is refused at argument-parse time, and a scene's own camera prim is declared with
   `--camera-prim NAME=/Prim/Path`.
4. `measure_est_drift.py`'s `is_lower_bound_reason` (line ~847, and the module docstring at ~73-75)
   justifies `is_lower_bound: true` with a Humanoid Everyday licence blocker that PR-08 §3 itself
   withdrew on 2026-08-07 (*"The reason is no longer the licence"*), and calls HE the confirmatory
   measurement for a quantity HE cannot produce at all. The stamp is right; the reason beside it is
   not, and it is written into every artifact. Full argument and evidence in §6(b) (§6, route b).

*Line numbers were read on 2026-08-22 against `scripts/measure_est_drift.py` at md5 `5d031394…` and
`src/wam/robot/isaac_binding.py` at `447beb43…`. Peer sessions are editing both; if a number does not
land, the anchors quoted in §4.1 are what to search for.*

## See also

- `docs/isaac.md` — the Isaac backend itself: the two-venv split (§0), the install (§1), the preflight (§2), the VRAM arithmetic (§3), the e-stop fidelity gap (§5)
- `docs/preregistration/PR-08-photoreal-augmentation.md` — §4 defines `EST_DRIFT_P95`; §6 subtracts it; §8 item 4 requires it committed
- `scripts/measure_est_drift.py` — the rig, and the authoritative record of every refusal it makes
- `scripts/measure_geom_tol.py` — the other half of §8 item 4, and the segmenter §4 step 2 requires this one to share
- `scripts/estimators/apple_sam2.py` — the estimator pair, its pinned checkpoints, and `GATE_QUALIFICATION_BLOCKERS` / `GATE_QUALIFICATION_DISCHARGED`, which are the authoritative record of what is open and what measurement closed the rest. §4.7 names them; that tuple counts them
- `.mc/tasks/todo/T-040-cosmos-transfer-photoreal-augmentation.md` — the task record and the cluster-side history
