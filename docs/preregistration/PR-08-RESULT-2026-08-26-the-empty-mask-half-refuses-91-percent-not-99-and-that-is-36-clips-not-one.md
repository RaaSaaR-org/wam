# PR-08 §6 G0c — the empty-mask half refuses 91.0 % of the corpus, not 99.2 %, and 36 episodes survive rather than one

**Computed 2026-08-26 from the committed corpus distribution. No pixel was measured, no rule was
edited, nothing was generated. It corrects one number that `T40_RULE_V12` and `T40_RULE_V13` §4 both
cite, and the correction is large enough to change how the decision they frame reads.**

---

## 1. The number that is quoted everywhere

`T40_RULE_V12` §1.2 and `T40_RULE_V13` §4 both state that G0c's empty-mask half refuses **128 of 129
pilot clips — 99.2 %**, one surviving clip. Both cite it faithfully; the source
(`G0C_REFUSAL.json`, `refusal_fraction 0.992248`) says exactly that.

**It is a true rate over a contiguous 129-episode block** — `episode_000000..085` plus
`episode_000201..243` — measured on a workstation RTX 5090. It is not a corpus rate, and neither
document claims it is. But both were written before a corpus rate existed, and every argument built
on them inherits the block.

## 2. The corpus rate

`runs/pr08-robot-mask-area/POOLED.json` records, for each of the 402 episodes, the per-frame area
fraction at stride 1 with `measurement_qualified: true`. An empty mask is recorded as exactly `0.0`.
`check_mask` refuses a whole clip on the first frame with an empty robot mask, so an episode
survives the empty half if and only if it contains **no** zero.

| | | |
|---|---:|---:|
| episodes with ≥ 1 empty-mask frame → **refused** | **366** | **91.0 %** |
| episodes with none → **survive** | **36** | 9.0 % |
| total | 402 | |

**36 surviving episodes rather than one. A 36× larger pool, on the same rule, on the same corpus.**

Three consistency checks, all passing: per-episode `len(area_fractions) == n_frames` for all 402;
the count of exact `0.0` entries equals each episode's recorded `empty_frames`; and the total is
57 835, which is `POOLED.json`'s own `frames_empty_mask`.

## 3. Why 33.7 % of frames does not straightforwardly give 91 % of clips

The per-frame empty rate is 57 835 / 171 625 = **33.70 %**, and mean episode length is ~427 frames.
Under an i.i.d. assumption a 427-frame OR at that rate refuses **100 %** of clips to some seventy
decimal places — it over-predicts, not under-predicts. Turned around: to keep even half the clips at
that length the per-frame rate would have to be **0.162 %**, not 33.7 %.

**What produces a number below 100 % is that empties are blockwise, not scattered.** They cluster at
the start and end of episodes — approach and retreat — with 98 of 101 diagnosed episodes ending
inside an absent run. The i.i.d. model is simply the wrong model, and the measured whole-episode
figure (§2) is the one to use because it needs no model at all.

## 4. What this does and does not change

**It does not change the rule.** G0c's empty-mask half is untouched, `check_mask` is untouched,
V12's §3 options are untouched. This is a yield number, not a mechanism.

**It changes the shape of V12's decision materially.** V12 §3.3 — change nothing, and read the
refusal as the compositing route failing on this corpus — is a very different proposition against
36 usable episodes than against one. Whether 36 episodes is enough for the intended measurement is a
question this document does not answer and does not pretend to.

**It does not resolve whether an empty robot mask is a defect or a correct answer.** The evidence on
record points to *correct answer* — zero confirmed detector failures in the blind adjudication the
project owner labelled on 2026-08-25, and every empty mask arising as `no_boxes_above_threshold`
rather than as a box SAM 2 emptied. But 27 of 40 tiles in that draw's only unbiased arm were
**undecidable**, so the honest reading is *unrefuted*, not *established*. That gap is what
`PR-08-RESULT-2026-08-25-v12-preconditions.md` already names as the thing actually blocking V12.

## 5. What this does NOT establish

- **Not that 36 episodes suffice for anything.** No downstream requirement has been checked against
  that count here.
- **Not a G0c pass.** 366 episodes still refuse. The area bound decided the same day arms the other
  half of the same gate and does not change this.
- **Not that V12 or V13 contain an error.** Both cite their source correctly. The number they cite
  was superseded a day after they were written, and neither has been superseded by a signed
  document, which is the actual defect and is a sequencing one.
- **Not a rule edit.** `docs/handoff.md` §3 forbids rewriting a rule in place; the correction lives
  here, where a reader following either citation will be sent.

---

## 6. Provenance

| | |
|---|---|
| kind | arithmetic over an existing artifact. **Measures no new pixels, registers no rule** |
| date | 2026-08-26 |
| source | `runs/pr08-robot-mask-area/POOLED.json` — 402 episodes, 171 625 frames, stride 1, `measurement_qualified: true` |
| provenance of that | `git_commit 8b710d0119b6…`, `source_manifest_sha256 a988dd60db6b…` |
| refusal rule read from | `scripts/robot_composite.py` — `check_mask`, clip-fatal on the first empty frame |
| corrects the citation in | `T40_RULE_V12` §1.2, `T40_RULE_V13` §4 |
| consistency checks | 3 of 3 passing (§2) |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched |
| generation licensed | **no** |
| training licensed | **no** |
