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
| **[E-01](tasks/E-01-can-the-edge-policy-run-without-language.md)** | Can the Edge policy run without language? | 1 | **review** | — |
| **[E-02](tasks/E-02-what-a-28-dim-g1-dex3-embodiment-actually-requires.md)** | What a 28-dim G1/Dex3 embodiment actually requires | 1 | **done** | — |
| [E-05](tasks/E-05-pre-register-the-edge-policy-experiment.md) | Pre-register the edge policy experiment | 1 | todo | ~~E-01, E-02~~ **unblocked** |
| [E-03](tasks/E-03-establish-the-target-edge-hardware-and-reproduce-the-rate.md) | Establish the target hardware, reproduce the rate | 2 | todo | **the user** (AC-1 is a purchasing fact) |
| [E-04](tasks/E-04-stage-cosmos3-edge-and-confirm-the-post-training-path.md) | Stage Cosmos3-Edge, confirm a post-training path | 2 | todo | — |
| [E-06](tasks/E-06-post-train-cosmos3-edge-on-the-g1-corpus.md) | Post-train Cosmos3-Edge on the G1 corpus | 3 | backlog | E-01/02/04/05 + ~~T-39~~ **reported `N`** → **the owner** |

## The two gates

**E-01 is the premise — and it reported 2026-08-16.** The answer is **outcome 2**: `prompt=""` is
accepted and nothing crashes, but text is never absent from the plumbing and it demonstrably
reaches the *action* head, not just the video branch. So **"image in, action out" survives as an
interface; "no VLA" does not survive as an architecture claim** — the text tower and tokenizer stay
resident on the robot. The cheap route is a **constant** instruction (in-distribution, ~free), not
an **empty** one (off-distribution, expect ≤ the 15.4 % "Vague" column). E-05 must pre-register
which. Full verdict and evidence in the task's `## Notes / Report`.

**E-02 reported too, and moved one thing that was assumed.** A 28-dim G1/Dex3 embodiment is one new
row in a 32-row trained table — `action_dim` is a global 64, so 28 fits with no architecture change.
But measuring the checkpoints showed **the released policy variant has exactly ONE trained row
(droid); `agibotworld` is at random init there.** The supported-29D-humanoid warm start exists only
in the **base** checkpoint. Reproduce in ~30 s with
`scripts/probe_cosmos3_domain_rows.py` — no weight download.

**T-39 is the floor, and it has reported — `VERDICT N`, 2026-08-17.** The rule was that no training
run starts here before the positive control reports, on the reasoning that if the corpus's own
action column cannot clear L1 under our scorer then no policy trained on it can. **Both halves have
now been measured, and they came back in opposite directions:**

- The **corpus's own action column does clear L1** — `oracle_action` **+68.10 % L1 / +75.40 % L2,
  level L4**, once step 0 is anchored on the previous command instead of on the state (PR-12/PR-13).
  The earlier `−359.41` was our evaluation adapter, not the labels. So the floor's premise is
  withdrawn, not confirmed.
- The **policy does not.** GR00T N1.7 — NVIDIA's own recipe, at its shipped defaults, on NVIDIA's own
  tutorial corpus — scored **−239.69 %** on the holdout and **−186.73 %** on the forty episodes it
  *trained on* (job `188408`, [`PR-07-V2-RESULT.md`](../../docs/preregistration/PR-07-V2-RESULT.md)).

**What that licenses here is narrow and easy to get backwards.** PR-07 §6's N row **forbids** reading
it as refuting any WAM or Edge design — N says the instrument saturates, not which arm is wrong — so
nothing in it is evidence against Cosmos3-Edge. What it does say is that the next move is the *kind*
of data rather than another method, and **E-06 is another method on the same corpus.** See E-06's
Description for the full reading, including the one alternative hypothesis that was checked and
eliminated (T-39 froze the vision tower) and the one that is still open (the scorer and the training
objective are not the same quantity). **It licenses nothing about training, which remains the project
owner's call.**

## Start here

New session: [`README.md`](README.md), then
[`research/2026-08-15-cosmos3-edge-and-dreamzero.md`](research/2026-08-15-cosmos3-edge-and-dreamzero.md)
for what is verified `[✓]` versus claimed `[doc]` versus open `[?]`. Then E-01.
