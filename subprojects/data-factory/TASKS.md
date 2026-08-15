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
| **T-040** | written, not started | Transfer2.5 photoreal augmentation — the one route D-01 permits |
| **T-041** | ran, **verdict VOID** on G0b | the Super fine-tune; its VOID needs a decision, not a rescue |
| **T-042** | **closed 2026-08-15** | step 0 counted zero unlabelled footage — and found 3 152 already-labelled 28-dim G1 episodes instead |
| **T-043** | **referenced, no file yet** | convert those 3 152 `action float32[28]` episodes — recorded labels, route 1 |

## Why D-01 comes first

The sub-project's own one-line summary — "use Cosmos to make more training data" — is one careless
paraphrase away from the thing `docs/handoff.md` §3 already closed: *generated video is not training
data, and nothing infers actions from it.* The legitimate route restyles a **real** episode so the
**real** labels survive. D-01 writes that boundary down before any task can drift across it.

## Start here

New session: [`README.md`](README.md), then D-01.
