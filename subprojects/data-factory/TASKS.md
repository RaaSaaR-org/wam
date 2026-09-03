# Data Factory — task index

`D-NN`, in [`tasks/`](tasks/). Same file shape as the edge sub-project: frontmatter, description,
acceptance, `## Notes / Report`.

**`mc` does not see these** (separate namespaces, decided 2026-08-15). This file is the index and is
maintained by hand.

## Order

| id | task | prio | status | blocked by |
|---|---|---|---|---|
| **[D-01](tasks/D-01-charter-what-the-data-factory-may-produce.md)** | Charter — what this sub-project may and may not produce | 1 | todo | — |
| [D-02](tasks/D-02-pick-the-variant-that-can-actually-restyle.md) | Pick the variant that can actually restyle (Edge cannot) | 2 | **investigated, decision open** | D-01 |
| [D-03](tasks/D-03-video-only-cross-embodiment-transfer.md) | Video-only cross-embodiment transfer — the new route | 2 | todo | D-01 |

## D-02 has an investigation, and what is left is a decision

[`docs/cosmos3-vs-transfer25.md`](../../docs/cosmos3-vs-transfer25.md) (2026-08-25, primary sources
only) answers D-02's factual half: **Cosmos3-Edge 4B cannot do video-to-video** (README footnote 2),
so a restyle runs on Nano 16B or Super 64B; transfer is a *mode* of the base checkpoints, not a
separate weight; and **Cosmos 3 ships no depth estimator and no segmenter**, so DEPTH/SEG/WSM must
be supplied pre-computed — a regression against Transfer2.5 for our manifest, which carries neither.
The document deliberately **picks nothing**: its §8 costs three options and says the choice is the
owner's. D-02 therefore stays open on the decision, not on the research.

## The root-project tasks this sub-project works around

Not migrated — their pre-registrations, sbatch files and commit subjects all cite them where they
are. Referenced by ID:

| id | state | why it matters here |
|---|---|---|
| **T-040** | **GENERATING, one chunk**, since 2026-09-03 | **the main use case** — restyle real episodes, recorded actions carried over unchanged. **PR-08 §8 is 7/7 as of 2026-09-01.** Item 3 landed at `1.6896 s/frame`; item 4 at `GEOM_TOL 0.47857992441961017 px` − `EST_DRIFT_P95 0.36010037281174667 px` = a **`+0.1185 px`** G0b budget. `PARTITION_CEILING_GPU_H = 2013.75` signed 2026-09-01, and the owner released **one chunk** on 2026-09-02 (`STAGE=1 STYLE_SET=train CHUNK_INDEX=1 CHUNK_TOTAL=4`) — [`PR-08-DET-2026-09-02`](../../docs/preregistration/PR-08-DET-2026-09-02-the-first-real-chunk-released.md). The 2026-08-27 sprint page is superseded twice over and is not the plan. **Still open on this path:** G0a/G0b have not run on the produced clips, and the clips' visual quality is under investigation. Gated on T-39 by `PR-08` §1 — **T-39 first reported `VOID (labels)` (2026-08-16); under the repaired anchoring it re-reported `VERDICT N` (2026-08-17, `PR-07-V2-RESULT.md`), and `T40_RULE_V3` §5.3 makes `N` satisfy "T-39 has reported", which is what actually closed item 7. On §1 itself, and this row said the opposite until 2026-09-03: a release was proposed and **declined on 2026-08-24**, but §8 then closed 7/7 on 2026-09-01 and the owner **released one chunk on 2026-09-02**. `T40_RULE_V1` §1 is NOT lifted — `PR-08-DET-2026-09-02` §4 is explicit that the release is what §8 being closed makes exercisable, not a substitute for it, and everything beyond that one chunk still needs a separate go.** |
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
