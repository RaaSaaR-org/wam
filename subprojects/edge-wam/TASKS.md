# Edge WAM — task index

`E-NN`, in [`tasks/`](tasks/). One file per task: frontmatter, description, acceptance, and a
`## Notes / Report` section that stays empty until the task runs.

**`mc` does not see these.** That is the accepted cost of separate namespaces (decided 2026-08-15) —
this file is the index, and it is maintained by hand. Root-project `T-NN` tasks still live in
`.mc/tasks/` and are still driven by `mc task next`.

## Order

The first two are cheap, need no GPU, and between them decide what this sub-project costs.

| id | task | prio | status | blocked by |
|---|---|---|---|---|
| **[E-01](tasks/E-01-can-the-edge-policy-run-without-language.md)** | Can the Edge policy run without language? | 1 | todo | — |
| **[E-02](tasks/E-02-what-a-28-dim-g1-dex3-embodiment-actually-requires.md)** | What a 28-dim G1/Dex3 embodiment actually requires | 1 | todo | — |
| [E-05](tasks/E-05-pre-register-the-edge-policy-experiment.md) | Pre-register the edge policy experiment | 1 | todo | E-01, E-02 |
| [E-03](tasks/E-03-establish-the-target-edge-hardware-and-reproduce-the-rate.md) | Establish the target hardware, reproduce the rate | 2 | todo | — |
| [E-04](tasks/E-04-stage-cosmos3-edge-and-confirm-the-post-training-path.md) | Stage Cosmos3-Edge, confirm a post-training path | 2 | todo | — |
| [E-06](tasks/E-06-post-train-cosmos3-edge-on-the-g1-corpus.md) | Post-train Cosmos3-Edge on the G1 corpus | 3 | backlog | E-01/02/04/05 + **T-39** |

## The two gates

**E-01 is the premise.** "Image in, action out, no VLA" is the reason this sub-project exists, and
the released `Cosmos3-Edge-Policy-DROID` is documented as taking language instructions. If the
policy path structurally requires text, the premise needs revisiting before anything is staged.

**T-39 is the floor.** No training run in this sub-project starts before the positive control
reports — if the corpus's own action column cannot clear L1 under our scorer, no policy trained on
it can, and a better backbone does not fix that.

## Start here

New session: [`README.md`](README.md), then
[`research/2026-08-15-cosmos3-edge-and-dreamzero.md`](research/2026-08-15-cosmos3-edge-and-dreamzero.md)
for what is verified `[✓]` versus claimed `[doc]` versus open `[?]`. Then E-01.
