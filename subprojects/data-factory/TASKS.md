# Data Factory — task index

`D-NN`, in [`tasks/`](tasks/). Same file shape as the edge sub-project: frontmatter, description,
acceptance, `## Notes / Report`.

**`mc` does not see these** (separate namespaces, decided 2026-08-15). This file is the index and is
maintained by hand.

## Order

| id | task | prio | status | blocked by |
|---|---|---|---|---|
| **[D-01](tasks/D-01-charter-what-the-data-factory-may-produce.md)** | Charter — what this sub-project may and may not produce | 1 | todo | — |
| [D-02](tasks/D-02-pick-the-variant-that-can-actually-restyle.md) | Pick the variant that can actually restyle (Edge cannot) | 2 | todo | D-01 |
| [D-03](tasks/D-03-video-only-cross-embodiment-transfer.md) | Video-only cross-embodiment transfer — the new route | 2 | todo | D-01 |

## The root-project tasks this sub-project works around

Not migrated — their pre-registrations, sbatch files and commit subjects all cite them where they
are. Referenced by ID:

| id | state | why it matters here |
|---|---|---|
| **T-040** | **`PR-08` + `T40_RULE_V1` in git, 9/13 acceptance closed** | **the main use case** — restyle real episodes, recorded actions carried over unchanged. Four items open, none a decision: the missing depth/segmentation conditioning, `GEOM_TOL`/`EST_DRIFT_P95`, H200 throughput + chunked sbatch, the `vla-training` consumer contract. Gated on T-39 by `PR-08` §1 — **and T-39 has now reported (2026-08-16): `VOID (labels)`, whose cause PR-12 (`C`) and PR-13 (`W`) traced to our evaluation adapter, not the corpus. §1's stated reason is therefore withdrawn by measurement, but §1 is not lifted here — that is the project owner's call.** |
| **T-041** | ran, **verdict VOID** on G0b | the Super fine-tune; its VOID needs a decision, not a rescue |
| **T-042** | **closed 2026-08-15** | step 0 counted zero unlabelled footage — and found 3 152 already-labelled 28-dim G1 episodes instead |
| **T-043** | **file written 2026-08-15**, blocked on a 647 MB fetch | convert those 3 152 `action float32[28]` episodes — recorded labels, route 1. 23.96 h, `cam_left_high` natively 640×480. **Block order measured arm-first (`[0:14]` arm, `[14:28]` hand)** — the opposite of what these docs said this morning; left/right and intra-hand order still unverified |

## The output target — fixed 2026-08-15

Everything this sub-project emits feeds **NVIDIA Isaac GR00T N1.7** (`nvidia/GR00T-N1.7-3B`).
Read off the ONNX export rather than from docs: the video input is **`ego_view`, float32
`[1, 480, 640, 3]`** — **one** camera view at **640×480**. So restyled frames are produced at
640×480, and `datasets/gr00t-apple-full/` (120×160) cannot be the source. Details and the artifact
it was measured from: [`README.md`](README.md).

## Why D-01 comes first

The sub-project's own one-line summary — "use Cosmos to make more training data" — is one careless
paraphrase away from the thing `docs/handoff.md` §3 already closed: *generated video is not training
data, and nothing infers actions from it.* The legitimate route restyles a **real** episode so the
**real** labels survive. D-01 writes that boundary down before any task can drift across it.

## Start here

New session: [`README.md`](README.md), then D-01.
