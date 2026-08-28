# PR-08 — §8 items 5 and 6: not a silence, a disagreement between the task ledger and the rules

**PREPARED, UNSIGNED, AND NOT A DETERMINATION.** `T40_RULE_V13` §5 — *"A session may prepare the
rationale and name the edges; it may not sign this."* This document names edges. It closes nothing,
opens nothing, moves no number, and licenses no clip. `T40_RULE_V1` §1 is untouched and forbids
generation in full.

**Filename carries `DET-DRAFT`, not `DET`, deliberately.** The four determinations of 2026-08-24 are
signed; this is not one of them and must not be cited as one.

---

## 0. Why this exists

`docs/preregistration/PR-08-DET-2026-08-24-four-determinations.md` §5 records **two secondary
silences** it did not resolve, and says what they need:

> **§8 item 6 (partition committed)** was recorded **NOT CLOSED** by V3 on 2026-08-15, when
> `git ls-files configs/transfer25/` returned zero tracked files. Those files are tracked now […]
> **No later document re-adjudicates it.** That is a silence, not a closure, and it should be closed
> by a document rather than by someone noticing the files are tracked.

> **§8 item 5**'s code half is observed landed in `src/wam/robot/isaac_binding.py`, but the
> measurement it unblocks was never taken […] Whether that closes item 5 or routes around it is
> unstated.

Both are on the critical path: §8 is conjunctive, and a session working toward a first clip that
counts only items 3 and 4 as open is counting wrong.

**The first thing this look found is that "silence" is the wrong word for item 6.** Two entries in
the task ledger adjudicate it, and they disagree with the registered documents about which items are
closed. A silence is closed by writing something down. A disagreement is closed by deciding which
record governs — and that is not a session's call.

---

## 1. Item 6 — the partition. What is settled

Everything V3 asked for has happened, and the property §5 protects is intact.

**§5's requirement**, `PR-08-photoreal-augmentation.md:143-145`:

> **The style partition is committed before generation.** The style pool is split into disjoint
> `TRAIN_STYLES` and `EVAL_STYLES` — as a committed file, in git, before the first clip — so the
> evaluation domain cannot be chosen after seeing which restyles came out well.

**V3's refusal**, `PR-08-V3-seed-schedule.md:544-563`, was about tracking and nothing else:

> The partition, its rendering, both sidecars, the verifier and the sbatch are **all untracked**. A
> file that is not in git cannot be the pre-commitment §5 asks for […] `styles.toml` itself asserts
> it "Closes PR-08 §8 item 6" — **it does not, until it is tracked.**

Measured 2026-08-28 at `b802fb8`:

| | |
|---|---|
| tracked in `configs/transfer25/` | 9 paths — 5 artifacts, 4 sidecars |
| the partition commit | `11af1d0`, 2026-08-16 16:05:49 +0200, **on `origin/main`** |
| `scripts/check_style_partition.py` | run just now: **exit 0, `PASS`**, nine checks |
| clips generated, ever | **zero** |
| `blocking_todos` | 1 recorded, 0 open |

**And the eval domain has never moved.** Diffing `11af1d0:configs/transfer25/styles.toml` against
HEAD: the `TRAIN_STYLES` and `EVAL_STYLES` id sets are **identical** — 10 and 5, same ids, same
prompts, same four axis slugs; `volume` and `seed_schedule` identical. The single thing §5 exists to
prevent — choosing the evaluation domain after seeing which restyles came out well — did not happen
and cannot now.

**One trap, named because this session fell into it.** Three of the four sidecars are byte hashes of
the files beside them. `configs/transfer25/styles.toml.sha256` is **not**: it holds a *content* hash,
`sha256(canonical_json(document minus [hash] and [consumer]))`, pinned as a rule string at
`styles.toml:530` and implemented at `check_style_partition.py:365-383`. A naive `sha256sum`
comparison therefore reports `styles.toml` as a mismatch, and it is not one — the value is
`9334fd01…`, which is exactly what the job logs print as `partition content`. **Any determination
citing these hashes must say which hash it means.**

## 2. Item 6 — the two edges that are genuinely open

Neither is a fact question. Both are document questions, and neither has an adjudicating document.

### 2.1 The pre-committed file was amended in place, after commitment

| commit | date | content hash |
|---|---|---|
| `11af1d0` | 2026-08-16 | `8d8565ffcd12ad17318f10979abb44f639084e21857866cbc3b2f32f1332628b` |
| `98d402a` | 2026-08-22 | `9334fd013d5f5d8d0e3dc42c004ee02de93b0acd7f6d6f2befce40152eba2f02` |

What changed at `98d402a` was `identity_style` — two words of arm C's prompt, *"black cloth"* →
*"dark grey cloth"*, on a 402-episode colour census — and `blocking_todos` (`T40-TODO-01` closed).
**Not `TRAIN_STYLES`, not `EVAL_STYLES`, not the seeds, not `volume`.** `identity_style` is in
neither set by construction; `check_disjoint` asserts it.

**`styles.toml` does carry an in-place-amendment justification, and it does not cover this
amendment.** `styles.toml:31-40` reads *"AMENDED IN PLACE 2026-08-15, STILL V1"* and names the
pre-amendment digest `4da3875d0c76e9b2…`, recorded *"nowhere except in this file's own sidecar and
its rendering"*. That is a **third** hash and an **earlier** event: `4da3875d…` → `8d8565ff…`
happened on 2026-08-15, *before* the file was tracked at all, and it changed arm C's sizing. The
amendment this section is about — `8d8565ff…` → `9334fd01…` at `98d402a`, 2026-08-22 — happened
**six days after the pre-commitment** and has no in-file justification of any kind. Its commit
message is substantial and the evidence behind it is real; it simply is not the argument the file
makes about itself.

So the justification and the event do not line up, and `docs/handoff.md:165` is the standard they
would have to be measured against: *"**Rules are versioned, never edited in place.** A gate
rewritten after seeing its output is not a gate."*

**The edge, stated without deciding it:** does a §5 pre-commitment tolerate in-place amendment of
the parts it does not protect? The conservative reading says a pre-commitment is the file, not the
subset of it somebody later argues was load-bearing — and that the file's own defence of an earlier
amendment does not transfer to a later one. The permissive reading says §5 names exactly one
property, a disjoint eval domain fixed before generation, that `identity_style` is outside it by
construction, and that the property is provably untouched. **Both readings are available on the
text. No document picks one.**

### 2.2 The artifact's self-declared date disagrees with the history

`configs/transfer25/pr08_style_partition.json` carries `committed = '2026-08-15'`. Git dates the
commit **2026-08-16 16:05:49 +0200**. `check_structure` requires the field to be present and never
compares it to git.

This matters only because V3's own argument for why tracking is required is *"only the history
proves the order."* If item 6's closure is written to rest on the history, the artifact's own date
field disagrees with the history it rests on, by one day.

*(A related discrepancy, noted and not load-bearing: V3's header says "Registered 2026-08-15"; its
commit `51ad9e9` is dated 2026-08-16 16:05:40 — nine seconds before the partition it declares
untracked. The observation V3 recorded was true when written and was falsified by the next commit in
the same push.)*

---

## 3. Item 5 — what it was for, and what has been built

§8 item 5, `PR-08-photoreal-augmentation.md:233-235`:

> **Depth and segmentation annotators wired into `isaac_binding.py`** — `distance_to_camera` and
> `semantic_segmentation` alongside the existing `rgb`, with tests. **Blocks §4 entirely.**

§4 is the four-step procedure that produces `EST_DRIFT_P95`: render N Isaac episodes with
ground-truth depth + segmentation (step 1), run the estimator on the RGB only (step 2), record the
error distribution — *"absolute depth error, **and** object-centroid displacement in pixels"* (step
3) — and take the 95th percentile of the centroid displacement (step 4). That number enters G0b's
tolerance as a budget.

**Both routes exist, and both are tested.**

| | Isaac | MuJoCo |
|---|---|---|
| file | `src/wam/robot/isaac_binding.py` | `src/wam/robot/mujoco_binding.py` |
| the two channels | `GROUND_TRUTH_ANNOTATORS` at `:188-191` — exactly `distance_to_camera` and `semantic_segmentation` | `render_depth` `:863`, `render_segmentation` `:894` |
| tests | 68 passed, 0 skipped | 73 passed, 0 skipped |
| **what the tests run against** | **a stub. Never real Isaac Sim.** | **real MuJoCo** (18.6 s vs 0.21 s is the tell) |

The Isaac test file says so itself, `tests/test_isaac_binding.py:1-6`: *"`IsaacSimBinding` was
written against NVIDIA's documentation and **CANNOT be executed here**"*, and at `:806-811`, of its
own payload shapes: *"a guess that is never even run against a stub is two guesses."*

**Every EST_DRIFT measurement that exists was taken on MuJoCo.** `grep -rl '"ground_truth_route":
*"isaac"' runs/` returns **zero**. No Isaac capture has ever been taken.

## 4. Item 5 — the three edges

### 4.1 V5 replaced the *naming*, and said nothing about closure

`PR-08-V5-ground-truth-route.md:427`, its own provenance row:

> | replaces | **one sentence**: `T40_RULE_V1` §4 step 1's *"Render N Isaac episodes…"*, plus step
> 0's and §8 item 5's naming of `isaac_binding.py` specifically (§1) |

and `:431`: *"| changes | **no gate, no threshold, no verdict, no arm, no clip count, no style, no
seed, no ceiling, and no committed artifact** |"*. V5 also keeps the Isaac route alive: *"V5 does not
close the Isaac route: it remains available, unchanged in every flag and refusal"* (`:128-131`).

So V5 is not evidence for closure **or** against it. The DET's "unstated" is exactly right on the
text.

### 4.2 Depth is a different physical quantity on the two routes, and §4 step 3 names depth

MuJoCo returns **distance to the image plane**; `distance_to_camera` is **euclidean ray length**.
`mujoco_binding.py:58-66` states the consequence and refuses to paper over it: *"comparing the two
inflates the error by `1 / cos(angle off the optical axis)` — 1.41 at 45°. Neither is converted into
the other here."* It is stamped into every capture as `"depth_semantics":
"distance_to_image_plane (NOT distance_to_camera)"`.

V5 §2 argues this does not matter, because **§4 step 4's gated number is defined on segmentation
alone** and depth appears only in step 3 — corroborated three ways, including that `run_g0_gates.py`
reads `est_drift_p95_px` and never a depth field, which reproduces.

**The edge:** that is a narrowing of V1, argued after V1 was written. Item 5's own clause is *"Blocks
§4 entirely"*, and §4 is a four-step procedure whose step 3 says *"absolute depth error, **and**
object-centroid displacement"*. A determination that MuJoCo **closes** item 5 must accept V5's
narrowing. A determination that MuJoCo **routes around** item 5 falls out of item 5's own word
*entirely*. **Both are available on the text.**

### 4.3 The nearest precedent is signed, and is deliberately narrower

`T40_RULE_V14` — **SIGNED 2026-08-27 by the project owner** — licenses a MuJoCo capture to stand in
for an Isaac capture. Its line 35, verbatim:

> The substitution is licensed for that measurement and for no other.

So the owner has already answered "may MuJoCo stand in for Isaac?" once, **yes, for one blocker**,
with an explicit scoping sentence that does not reach §8 item 5. If item 5 is thought already
covered by that decision, V14's own sentence says it is not. **Whether to extend it is the same kind
of decision, and it is the owner's.**

---

## 5. The thing that makes this a disagreement rather than a silence

The task ledger has adjudicated both items, and it disagrees with the registered documents.

- `.mc/tasks/done/T-39-…md` status_note: *"§8 item 7 […] is CLOSED […] and **item 6 closed
  2026-08-17** when `configs/transfer25/` + `check_style_partition.py` +
  `97_transfer25_restyle.sbatch` were committed and pushed — but items 2, 3 and 4 remain OPEN and
  **item 5's measurement is untaken**, so T-040 IS STILL NOT OPEN."*
- `.mc/tasks/todo/T-040-…md:536-539` (2026-08-22): *"§8 status after today. **Item 5 is closed** —
  `distance_to_camera` and `semantic_segmentation` are wired in `src/wam/robot/isaac_binding.py`
  with tests."*
- The same T-040 file then says *"§8 items 3 and 4 are open"* at six places, the last dated
  **2026-08-27 — three days after the DET** that calls both items silences.

Note that the two ledger entries split item 5 the same way this document does: T-39's says the
**measurement** is untaken; T-040's says the **code** is closed. Item 5's text contains both halves.

**These are task records, not registered rules.** `CLAUDE.md` makes `.mc/tasks/` the operational
index over milestones, and nothing anywhere says a `status_note` can close a §8 item. But they are
what a session reads to learn where the project stands, they have been read that way for eleven
days, and the DET's §5 does not mention them.

**So the open question is not "what happened" — it is which record governs a §8 item.** That is
prior to items 5 and 6 and applies to all seven.

---

## 6. What is owed, and to whom

**To the project owner, and to nobody else:**

1. **Whether a §5 pre-commitment survives in-place amendment** of the parts it does not protect
   (§2.1). Everything needed to decide it is in §1 and §2.1; the eval domain is provably untouched.
2. **Whether `T40_RULE_V14`'s licence extends to §8 item 5**, or whether item 5 needs its own
   decision (§4.3). V14's scoping sentence says it does not extend itself.
3. **Whether item 5 is closed by the MuJoCo route or routed around by it** (§4.2) — which is
   whether V5's narrowing of §4 to its gated half is accepted as V1's meaning.
4. **Which record governs a §8 item** when the task ledger and the registered documents disagree
   (§5). This is the largest of the four and the only one that is not about items 5 and 6.

**To nobody yet, and worth stating so it is not mistaken for a blocker:** item 5's *measurement*
half is bounded by item 4, not by item 5. Every EST_DRIFT artifact on disk carries `gate_qualified:
false`, and `configs/transfer25/pr08_geom_tol.json` still holds `geom_tol_px`, `est_drift_p95_px`
and `gate_margin_px` all `null`. Closing item 5 in either direction does not produce a committed
`EST_DRIFT_P95`.

## 7. What this document does not do

* **It does not close item 5 or item 6.** It cannot: `T40_RULE_V13` §5 forbids a session signing,
  and §5 above shows the prior question is unanswered anyway.
* **It does not amend `styles.toml`, `pr08_style_partition.json`, or any rule.** In particular it
  does not touch `styles.toml:2`'s self-assertion *"Closes PR-08 §8 item 6"* — the sentence V3
  objected to in form — because editing a pre-committed file to make a determination easier is the
  defect in §2.1, committed on purpose.
* **It does not revise the DET.** `docs/handoff.md` §3: rules are versioned, never edited in place.
  Where this look disagrees with the DET's §5 — "silence" versus "disagreement" — the DET stands as
  what it recorded and this sits beside it.
* **It writes no bound and discharges no gate.** Four §8 items are unclosed on the reading that
  gives the ledger no weight: **3, 4, 5, 6**. On the ledger's own reading, two: 3 and 4. The
  difference between those two counts is decision 4 in §6.

## 8. Reproducing it

```bash
git ls-files configs/transfer25/
.venv/bin/python scripts/check_style_partition.py                    # exit 0, PASS
.venv/bin/pytest tests/test_isaac_binding.py tests/test_mujoco_binding.py -q   # 141 passed
grep -rl '"ground_truth_route": *"isaac"' runs/                      # no hits
git log --format='%h %ad' --date=short -- configs/transfer25/styles.toml
```

The content-hash-versus-byte-hash trap of §1 is why the fourth line is `check_style_partition.py`
and not `sha256sum -c`.
