# Data Factory — more and better training data from the data we already have

**Start here if this is your session.**

## What this is

We have 402 success-only episodes of one task. Fourteen recorded experiments have failed to clear a
bar on that corpus, and the standing explanation has always been "not enough data". This
sub-project attacks that directly: use the **large** Cosmos models — `Cosmos3-Super` (64B) and
`Cosmos3-Nano` (16B) on Discoverer+ — to turn the episodes we have into more, and more varied,
training data for downstream policies, including [`../edge-wam/`](../edge-wam/README.md).

It is the half of the project that is *allowed* to be expensive, because none of it runs on a robot.

## The line this sub-project must not cross

`docs/handoff.md` §3 closed this and it stays closed:

> **Generated video is not training data, and nothing infers actions from it.**

The distinction that makes this sub-project legitimate is narrow and load-bearing:

| | route | labels | allowed |
|---|---|---|---|
| ✅ | **restyle a real episode** (Cosmos-Transfer2.5, T-040) | the real ones survive — the trajectory is unchanged | **yes** |
| ❌ | synthesize a new episode, infer its actions | invented | **no** |
| ⚠️ | inverse dynamics on *real unlabelled* footage (T-042) | inferred from real pixels | **open**, bounded — see `docs/action-labels.md` §3b |

"Variations of the training data" means the first row. If a proposal here starts needing the second,
it is the wrong proposal.

## State — 2026-08-15

| | |
|---|---|
| T-041 (Cosmos3-Super fine-tune on G1) | **ran, verdict VOID** on G0b — the judge could not clear its calibration set |
| T-040 (Transfer2.5 augmentation) | written, not started, pre-register first |
| T-042 (inverse dynamics labels) | **step 0 counted 2026-08-15 → zero unlabelled footage; route closed** |
| corpus | `nvidia/GR00T-N1.7-AppleToPlate`, 402 episodes, success-only |

**T-042 closed itself, and found something bigger on the way out** (`docs/action-labels.md` §3b).
The count of real G1 footage that has video, no actions and no way to get them is **0** — the clips
looked unlabelled only because two fetch scripts passed `--include 'meta/*' --include 'videos/**'`
and never pulled `data/**`. Upstream, all 14 repos publish the action parquets: **415 files,
647 MB**, Apache-2.0 / CC-BY-4.0, already-pulled repos.

And **3 152 of those episodes are the 13 `unitreerobotics/G1_Dex3_*` sets, every one declaring
`action float32[28]`** — the exact 28-dim G1 + Dex3 vocabulary. Against the 402 episodes of one task
that every number in this project rests on, that is thirteen more G1 tasks with recorded actions,
already accessible. It is *conversion* work on *recorded* labels, not generation — route 1, and
firmly inside this sub-project's charter. Tracked as **T-043**, which has no task file yet.

**It is not a free win and must not be written up as one:** 28-dim Dex3 ≠ 43-dim AppleToPlate,
`convert_lerobot_g1.py` targets canonical 15 joints + 2 grippers, and the block-order trap
(`action[0:14]` hand vs `action[14:28]` arm) makes a wrong converter produce numbers that are
finite, plausible and wrong.

Those three keep their `T-NN` IDs in `.mc/tasks/` — they were not migrated, because their
pre-registrations, sbatch files and commit history all cite them there. This sub-project owns the
work *around* them, under `D-NN`.

## One useful constraint discovered 2026-08-15

**`Cosmos3-Edge` does not support video-to-video transfer.** Restyling therefore belongs to Super or
Nano and can never move to the edge model — which is a clean argument that these really are two
sub-projects and not one with a size knob.

## Tasks

See [`TASKS.md`](TASKS.md). **D-01** is the charter that writes the table above into a rule; **D-03**
is a genuinely new route this project had no task for.

## Rules that bind here

- **Everything in the root `CLAUDE.md` still applies.**
- **Pre-register before generating anything.** T-040 says so in its own title, and PR-09's VOID is
  the reason: a gate written after seeing output is not a gate.
- **A VOID is not a weak pass.** T-041's 60 paired clips exist and are not readable until G0b is
  satisfied. Nobody has looked at the frames, deliberately.
- **Nothing gets submitted or paid for without asking first.**
