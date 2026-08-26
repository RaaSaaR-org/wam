# PR-08 §6 G0c — `max_frame_fraction` is decided under `T40_RULE_V13`, and it arms one half of one gate

**Decided 2026-08-26. `T40_RULE_V13` is signed and in force; its §5 carries the determination and
`bound_rationale` carries §3.2's five required items. This licenses no clip, discharges no blocker,
and does not make G0c pass. `GATE_QUALIFIED` is still `False`.**

---

## 1. The number, and the gap it sits in

```
max_frame_fraction = 0.64091145833333329
```

| | |
|---|---:|
| largest bulk fraction **below** | 0.6015462239583333 |
| smallest tail fraction **above** | 0.6802766927083334 |
| gap width | 0.07873046875 |
| **frames strictly inside the gap** | **0** of 171 625 |

V13 §3.1 step 3 requires a rationale that can name **both** edges, because one that cannot has not
found a gap — it has chosen a number. Both are named, and the interval between them is empty over
the whole corpus.

§3.4's two prohibitions are respected and worth stating rather than implying. The **observed
maximum** is 0.9795833333333334 and is quoted here only as an edge; committing it as the bound is
forbidden precisely because it cannot fire on the frames it was measured over. And the number is
not round — it is a midpoint, carried at the precision the arithmetic produces.

## 2. What it refuses, in both units V13 asks for

| | |
|---|---:|
| frames above the bound | **1 385** of 171 625 — 0.807 % |
| as a fraction of non-empty masks | 1.217 % of 113 790 |
| **episodes containing at least one** | **175** of 402 — **43.5 %** |
| most-affected single episode | 80 frames, 5.8 % of the 1 385 |

**Both units are given because they read differently, and the difference is the finding.** `check_mask`
refuses a whole clip on a single over-bound frame (`scripts/robot_composite.py`), so under that OR
this is not a 0.8 % cost — it is a 43.5 % cost. And because no single episode carries more than
5.8 % of the tail, this is a corpus-wide phenomenon rather than one bad episode that could be
dropped.

## 3. The frames were looked at, which is the item V13 §3.2 puts hardest

48 tail frames were rendered with the committed masker's mask overlaid, both the recorded and the
recomputed fraction in every caption, and judged one at a time by a person (reviewer `human`). The
same 48 tiles were then re-read blind by a model with no access to the recorded verdicts, which
recovered the binary mask by inverting the overlay's known linear blend per pixel rather than
reading the tint.

| | recorded | blind re-read |
|---|---:|---:|
| **`arm`** — a legitimate near-camera arm | **0** | **0** |
| `table` | 5 | 2 |
| `mixed` | 43 | 46 |

The five tiles the two readers split differently are a **within-class** difference: `table` and
`mixed` both say the mask contains scene. On the question V13 §2 actually asks — *is any frame in
this tail a legitimate near-camera arm that a bound must not discard* — **the two readers agree, and
the answer is no on all 48.** The blind reader also measured that 95–98 % of the dark arm pixels lie
**inside** the mask in all 48 frames, so the arm is never the excluded region.

That agreement corroborates nothing on its own — the second reader is a correlated observer. It is
recorded because V13 §3.2 asks whether the frames were looked at and what they were, and this is
what they were.

## 4. What this does NOT establish

- **It does not make G0c pass. It arms G0c's area half.** V13 §4 is the whole of this distinction
  and it survives signing intact.
- **It is not validated against a known-bad mask, and V13 §3.2 requires that sentence while it is
  still true.** No mask deliberately grounded on the table or the scene has been constructed and
  measured. This bound separates two populations **found** in the corpus, not two populations
  **established** on purpose. V13 §3.3 option (c) remains the only route that would change that and
  it is not taken here.
- **It licenses no clip.** `T40_RULE_V1` §1 binds in full; §8 items 3 and 4 are open; `T40_RULE_V12`
  is an unsigned draft; `GATE_QUALIFIED` is `False`.
- **It says nothing about the empty-mask half**, which is what actually refuses this corpus — see §6.

## 5. Two conditions a later reader inherits

Both measured 2026-08-26 and recorded in
`PR-08-RESULT-2026-08-26-the-area-fraction-is-stable-except-in-the-band-nobody-uses.md`.

**Per-frame classification near this bound is hardware-dependent for one narrow band.** Re-rendering
the *complete* 44-frame population of `0.36 ≤ f ≤ 0.601546` on an RTX 5090 moved 37 of 44 by more
than 0.01, and one frame recorded at the gap's own lower edge landed at 0.6152 — inside the gap. The
workstation is bit-identical to itself across two runs and the estimator pins match character for
character, so this is a between-machine difference, not run-to-run noise, and it is confined to 44
frames of 171 625 (**0.026 %**). Neither end moves: 0 of 48 sampled bulk frames reach this bound and
48 of 48 tail frames stay above it.

**Following from that, the set of refused clips is a property of the machine.** `check_mask` refuses
a whole clip on one over-bound frame. This distribution was measured on the cluster and this bound
is intended to be applied by the cluster; moving generation to different hardware inherits a
question nobody has measured.

## 6. The number V13 §4 cites is superseded, and the correction goes the helpful way

V13 §4 and `T40_RULE_V12` both say the empty-mask half refuses **128 of 129** pilot clips (99.2 %).
That is a true rate over a **contiguous 129-episode block**. Computed corpus-wide from
`runs/pr08-robot-mask-area/POOLED.json`:

| | |
|---|---:|
| episodes with ≥ 1 empty-mask frame → refused | **366** of 402 — **91.0 %** |
| episodes with none → survive the empty half | **36** of 402 — 9.0 % |

**36 surviving episodes, not one.** A 36× larger pool, on the same rule. Both V13 §4 and V12 predate
this measurement and neither has been superseded by a signed document, so anyone reasoning from
either alone is reasoning from a 129-episode block. Recorded here because this document's own §4
leans on V13 §4, and citing a superseded number while relying on it would be the same defect one
layer down.

---

## 7. Provenance

| | |
|---|---|
| kind | **a decision**, taken under a rule registered before the distribution was visible |
| rule | `T40_RULE_V13`, signed 2026-08-26, §§0–4 unchanged since registration |
| decided by | the project owner, on the instruction quoted verbatim in V13 §5; prepared by a Claude Code session, which §5 permits and no more |
| distribution | `runs/pr08-robot-mask-area/POOLED.json` — 402 episodes, 171 625 frames, stride 1, `measurement_qualified: true` |
| provenance of that | `git_commit 8b710d0119b6…`, `source_manifest_sha256 a988dd60db6b…` |
| written to | `configs/transfer25/pr08_robot_mask_area.json`, `runs/pr08-robot-mask-area/pr08_robot_mask_area.MEASURED.json` |
| the look | `runs/pr08-area-tail-look/TAIL_VERDICTS.json`, `runs/pr08-mask-audit/SECOND_OPINION.json` |
| validated against a known-bad mask | **no.** §4 |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched |
| generation licensed | **no** |
| training licensed | **no** |
