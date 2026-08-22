# PR-08 V4 — the T-39 premise behind §8 item 7, and the determination it needs

**Rule `T40_RULE_V4`. Drafted 2026-08-22, before any clip is generated, before any weight is
trained, and before any job is submitted. Nothing has been generated; no clip exists.**

> ## THIS DOCUMENT IS NOT IN FORCE.
> It is a **proposal for the project owner's signature**. The determination in §7 is **unsigned**,
> and until it is signed by a named person on a dated line, nothing in this document decides
> anything, licenses anything, or closes anything. No agent may treat an unsigned V4 as a rule, and
> no agent may sign it. Where this document says "registers", read "proposes to register, on
> signature."

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md) (`T40_RULE_V1`),
[`PR-08-V2-arm-c-frame-matching.md`](PR-08-V2-arm-c-frame-matching.md) (`T40_RULE_V2`) and
[`PR-08-V3-seed-schedule.md`](PR-08-V3-seed-schedule.md) (`T40_RULE_V3`). **None of the three has
been edited and none may be.** The discipline is `docs/handoff.md` §3 — *"Rules are versioned,
never edited in place. A gate rewritten after seeing its output is not a gate."* V4 is that
versioning, not a revision. `CLAUDE.md`, `PR-07-positive-control.md`, `PR-07-RESULT.md`,
`PR-07-V2-*`, `PR-12-*`, `PR-13-*` and `cluster/discoverer/97_transfer25_restyle.sbatch` are
likewise untouched by this document.

Task: [[T-040]]. Consumer: `emai/vla-training`. Generator: **Cosmos-Transfer2.5, frozen**.

---

## 0. What V4 does not change

Stated first and exhaustively, because a V-document that quietly moves a threshold is the failure
the versioning discipline exists to prevent. **V4 moves no threshold, no gate, no verdict, no arm,
no clip count and no style.** It is a document about one *premise* and one *record*.

| | unchanged |
|---|---|
| `MATERIAL_FLOOR_PP = 10.0` | still **borrowed** from `I8_RULE_V3` in `62_eval_i8_curve.sbatch`, not coined. V4 does not move it, rescale it, or make it per-arm |
| `GEOM_TOL` | still **derived** — the median per-step object-centroid displacement in the source clips, measured and committed before generation, at the step `T40_RULE_V3` §4 registers (`GEOM_STEP_FRAMES = 1`). V4 supplies no value and changes no method |
| `EST_DRIFT_P95` | still **measured** per V1 §4, still subtracted from G0b's budget, still recorded as a **lower bound** on the real error, and a G0b margin that only clears under a lower bound is still **not a pass** |
| **G0a** label integrity | unchanged — `screen_corpus --expect` against the source's M1/M2/M3 within `EXPECT_TOL` (0.02, 0.02, 0.05); a deviation is still VOID |
| **G0b** geometry invariance | unchanged — the generator is held to `GEOM_TOL − EST_DRIFT_P95`; if that is ≤ 0, generation does not start |
| **G0c** embodiment | unchanged — the real robot's pixels are unconditionally composited back over every generated frame; robot-mask IoU is recorded as a diagnostic, **never** as a gate |
| **The ladder** | unchanged — **L1** `skill_vs_repeat_pct > 0`, **L2** `ci_skill_vs_repeat_pct > 0` (`ci_` = the task-**critical** chunk subset, not a confidence interval) |
| **The P / F / N / I verdict table** (V1 §6) | unchanged in every cell, including that **P** requires *both* B − A ≥ floor *and* B − C ≥ floor, that **F** is the generator-attributable case, that **N** is B − A ≤ 0, and that `0 < B − A < MATERIAL_FLOOR_PP` is **I**, not a weak P |
| **Arms A / B / C / D** | unchanged. B is the intervention, C is the generator-fingerprint control, D is diagnostic and never the headline |
| **Arm C's size** (`T40_RULE_V2` §1–§2) | unchanged — R2, frame-matched: 1 identity style × 10 repeats × 402 episodes = 4 020 clips against arm B's 10 × 1 × 402 = 4 020. **Arm B is still not subsampled** |
| **Clip totals** | unchanged — train 4 020, identity 4 020, eval 2 010, whole partition 10 050 over 25 style-instances |
| **The seed schedule** (`T40_RULE_V3` §1) | unchanged — train `[7001..7010]`, identity identical, eval `[7011..7015]` disjoint, assignment `style-instance-index` |
| **The two-quantity GPU-h ceiling reading** (`T40_RULE_V3` §3) | unchanged. V4 supplies no ceiling value and exempts nothing from one |
| **The committed style partition** | V4 changes **no style, no id, no slug and no prompt string**, and therefore changes no partition hash. Authority remains the sidecars plus `scripts/check_style_partition.py`, never a value quoted in prose. *(Observation, not a registration: `T40_RULE_V3` §2.4's registered content hash `8d8565ff…` was superseded by commit `98d402a` on 2026-08-22, which corrected arm C's identity prompt; the verifier passes at `9334fd01…`. Re-registering that value is a matter for whichever V-document the owner attaches it to — **V4 does not register it**.)* |
| **§1's prohibition** | unchanged and still binding in full — nothing is generated, no weight is trained on generated frames, and no number from PR-08 is quoted as a result, until **every** §8 item is closed **and** T-39 has reported |
| **§7's threat to validity** | unchanged — `EVAL_STYLES` is generated, so a **P** is a claim about generalising to held-out *generated* appearance, not to real appearance, and it licenses **exactly one thing**: recording a small real shifted eval set and re-running arms A and B against it. It never licenses adding restyled data to any training corpus, and must never be reported as "augmentation works" |
| **`T40_RULE_V3` §5.3's refusal of a VOID** | unchanged. A T-39 **VOID** still closes PR-08 rather than opening it. §4 below asks the owner to confirm one adjacent question; it asks for no relaxation of this one |
| **`PR08_OVERRIDE_T39_VOID`** | unchanged, ungranted, and **not exercised**. This document **recommends against it**, registers no circumstance under which it may be used, and does not write its value anywhere |

**And, said loudly because it is the failure mode this project keeps having to name:**

- **V4 does not license training on generated frames.** V1 §1 forbids it; V4 does not touch that.
- **V4 says nothing whatever about `docs/benchmark.md`'s L4 gate.** `CLAUDE.md` records that as a
  separate open decision for the owner — the repaired oracle cell is L4 under bench spec 0.1.0 and
  below spec 0.2.0's two-sided `smoothness_ratio` floor, so the two specs disagree about it. That
  disagreement is untouched here and V4 must not be cited in it.
- **V4 makes no claim about any policy's capability**, and no claim about GR00T beyond citing a
  committed result document by name. See §5.

---

## 1. The premise this document was commissioned on, and what the sources actually say

V4 was commissioned to resolve a situation described as: **§8 item 7 is the last open item; T-39's
standing verdict is `VOID (labels)`; `97_transfer25_restyle.sbatch` refuses a VOID; therefore either
the VOID must be determined to satisfy "T-39 has reported", or the override must be typed.**

**That premise does not survive the sources, and the correction is the most useful thing in this
document.** It is stated before anything else, because a determination signed on a stale premise is
exactly the failure the pre-registration discipline exists to prevent.

### 1.1 T-39's operative verdict is **`N`**, not `VOID`, and the policy arm did run

`docs/preregistration/PR-07-V2-RESULT.md` — committed in `ad85656`, registered in advance by
`PR-07-V2-repaired-anchoring.md` (`T39_RULE_V2`, `dbd0255`) — records:

```
VERDICT N · NOTHING CLEARS THIS BAR, AND THE POLICY CANNOT EVEN FIT THE DATA.
```

Run 2026-08-17, job **188408**, one H200, elapsed **00:14:43**. Four arms, verified against the
on-disk `bench.json` files while drafting this document:

| arm | chunks | L1 `skill_vs_repeat_pct` | L2 `ci_skill_vs_repeat_pct` | level |
|---|---:|---:|---:|---|
| `oracle_state` | 1040 | **+100.00** | +100.00 | L4 |
| `oracle_action` | 1000 | **+68.10** | **+75.40** | **L4** |
| **`policy` (holdout)** | 1000 | **−239.69** | −84.36 | none — below L0 |
| `train40` (diagnostic) | 1045 | **−186.73** | −70.06 | none — below L0 |

Against `T39_RULE_V1` §5's definitions, every threshold quoted unchanged into `T39_RULE_V2`:
`oracle_state` clears the 90 % floor, so **G0a passes**; `oracle_action` clears **L1**, so **G0b
passes and the VOID condition is not met**; the policy fails L1 by far more than
`MATERIAL_FLOOR_PP` on the holdout **and** `train40` also fails L1, which is exactly V1 §5's
conjunction for **N**.

**So the policy arm ran.** The statement that it never ran — and that job **187813** died at 108 s
on a missing `GROOT_PATCH_MISTRAL` export — is true of the `T39_RULE_V1` attempt on 2026-08-16 and
**only** of it. It was superseded on 2026-08-17.

### 1.2 §8 item 7 is therefore already closed, by the route the brief called (b)

`T40_RULE_V3` §5.3 registers that *"`P`, `N`, `M` and `I` satisfy §1's 'T-39 has reported'; VOID
does not."* T-39's operative verdict is `N`. **Item 7 closed on 2026-08-17**, and
`PR-07-V2-RESULT.md` §1 says so in those words. The route that produces a real verdict was not a
proposal — **it was executed, and it cost about a quarter-hour of H200 time.**

### 1.3 What is actually open, checked rather than remembered (2026-08-22)

| item | status |
|---|---|
| 1 — the recipe (`--tune-visual`, Recipe B, lr 5e-5) | **closed**, fixed in V1 |
| **2 — the consumer contract** | **OPEN.** `docs/contracts/vla-training-consumer.md` exists and its §0 records that V1 §8 item 2's three fields describe `unitreerobotics/G1_Dex3_*` (v3.0, 28-dim), not `nvidia/GR00T-N1.7-AppleToPlate` (v2.1, 43-dim in seven groups, measured from the corpus's own `meta/info.json`). Which corpus the contract is *supposed* to name is a decision about what the deliverable is and, per `T40_RULE_V3` §5.3, **is not an agent's to make** |
| **3 — measured throughput and a derived GPU-h ceiling** | **OPEN, running.** Three jobs were lost to one AV1 decode defect (186357, 189585, 189584); the corpus was transcoded to H.264 with all 402 clips **proven** bit-exact, cleared in the generation venv by job 189605, and the timing run is job **189609**. No `THROUGHPUT.json` exists yet, so no budget line exists yet |
| **4 — `GEOM_TOL` and `EST_DRIFT_P95` measured and committed** | **OPEN.** `GEOM_TOL` is now schedulable via the shard/merge path (a merge that averaged shard medians would return a plausible wrong number, so it is under adversarial verification before it is given four GPU-hours). **`EST_DRIFT_P95` has nowhere to run** — Isaac Sim is not on the cluster, and installing it on the workstation is the owner's call |
| 5 — depth and segmentation annotators in `isaac_binding.py` | **closed** (`5ef3535`, 2026-08-21), with tests. The *measurement* it unblocks is item 4, above |
| 6 — the partition committed in git | **closed** 2026-08-17; `configs/transfer25/` is tracked and the verifier passes |
| **7 — T-39 has reported** | **CLOSED** 2026-08-17 by `VERDICT N` (§1.1) |

**The brief's "six of seven closed or running, item 7 is the open one" is inverted.** Item 7 is the
*closed* one. The open ones are **2, 3 and 4**, and two of the three turn on decisions or
installations that are the owner's, not an agent's. **§1's prohibition therefore still binds in
full**, for reasons that have nothing to do with T-39.

### 1.4 The PR-13 facts in the brief all check out

Every measurement cited to `PR-12-RESULT.md` and `PR-13-RESULT.md` verified against those files:
`commanded_to_chunk` built chunk step 0 as `command − STATE` while every other step of both arms is
a homogeneous first difference; that one element carried **143×** the error of its neighbours and
**90.10 %** of the summed per-step MSE at `d = −2` (**91.26 %** at `d = 0`); anchored on the
previous command the same column scores **+68.10 / +75.40** and reaches **L4**; the unmodified
bridge reproduced `PR-07-RESULT.md`'s **−359.41** to a drift of **+0.002 pp** in the same run; the
blast radius is the evaluation adapter and the four sweeps importing it, with **zero hits in
`src/wam/` and zero in `scripts/train_t39_baseline.py`**; `relabel_chunks` is homogeneous at every
step; and the fourteen negatives in `docs/benchmark.md` stand. `PR-13-RESULT.md` states in its own
words that `W` *"is not a verdict on T-39"*.

**All of that is correct — and it is no longer the last word.** PR-13 was the CPU re-derivation of
G0 on 2026-08-16; the cluster re-run of the whole of T-39 under `T39_RULE_V2` followed on
2026-08-17 and produced the verdict PR-13 explicitly declined to produce. `PR-07-V2-RESULT.md` §5
records that the pre-registered expectation of `+68.10` was met to `+68.1014939946058` across
**two** independent submissions (188407 and 188408), agreeing with PR-13's independent driver.

---

## 2. The three routes, against the corrected record

### (a) Determine that a `VOID` with its premise withdrawn satisfies "T-39 has reported"

**What it would have cost:** a signed determination, plus a minimal amendment to
`97_transfer25_restyle.sbatch` so that its accepted-verdict set and its stated *reason* agree with
what the project has measured. The refusal text currently justifies itself with *"PR-07 §4 defines
a VOID as the finding that no policy trained on this dataset can clear our bar and that the defect
is our own label pipeline"* — a sentence whose basis PR-13 withdrew by measurement. A gate that
refuses for a reason the project has disproved is a gate that teaches its operator to distrust it.

**Why V4 does not recommend it: it is moot.** Route (a) exists to rescue a `VOID`. T-39's operative
verdict is `N`, which the gate already accepts, so **there is no VOID on any live path** and nothing
to rescue. Amending a live gate to admit a verdict that is no longer the verdict would be widening
the gate for no gain — precisely the kind of change `docs/handoff.md` §3 exists to prevent.

The stale *reason* remains stale, on a branch that can now only be reached counterfactually. §6
records the minimal change that would fix the wording, **as a report and not as an instruction**, so
that whoever next touches that file has it. **V4 does not change the file and does not ask for it to
be changed as a condition of anything.**

### (b) Run the T-39 policy arm to obtain a genuine `P` / `N` / `M` / `I`

**Already done.** Job **188408**, 2026-08-17, verdict **`N`** (§1.1). It was the right route, it was
the only route to a real verdict, and its cost turned out to be about fifteen minutes of H200 time
against PR-07 §7's 12 GPU-h ceiling, which this run and 188407 together did not come near.

Two things to keep straight about it. First, running it *was* a training-adjacent decision gated on
the owner — but the training run it evaluates (job **187804**, 2026-08-16) predates the gate's
current wording, and the eval consumed a finished checkpoint rather than starting a new fit.
Second, **`N` is a verdict, not permission**: `CLAUDE.md`'s standing gate — whether training may
start, and against which label space — remains the owner's call, and a verdict whose content is
*"the corpus is the finding"* is a particularly poor argument for training on it. `PR-07-V2-RESULT`
§6 says exactly this.

**V4's recommendation on (b):** nothing further is required to close item 7. PR-07 §7's one
conditional second candidate (`lerobot/pi05_base`) is licensed by outcome `N` and is now technically
available — **it is a licence to spend allocation, not an instruction, and it remains the owner's
call.** V4 neither requests nor authorises it.

### (c) Use `PR08_OVERRIDE_T39_VOID`

**V4 recommends against it, unconditionally, and would recommend against it even if item 7 were
open.** Three reasons, in increasing order of weight:

1. **It is unnecessary.** The gate it bypasses is not blocking; `N` passes it.
2. **The attestation it demands is a statement the project has measured to be false.** The
   override's only accepted value asserts, in the operator's own voice, that this corpus cannot
   clear the bar. On T-39's own 40-episode holdout, correctly anchored, the corpus's own action
   column clears L1 at **+68.10 %** and L2 at **+75.40 %**, reaching **L4**. Typing that sentence
   would put a claim the project disproved into `chunk_metadata.json`, verbatim, beside every clip
   generated under it — permanently, since that is what the override is designed to do.
3. **An override is the wrong instrument for a stale reason.** When a gate's stated justification
   goes stale, the repair is a versioned document and a corrected gate, not an operator attesting to
   the stale justification in order to get past it. Using the override here would leave the false
   sentence in the record *and* leave the gate's wrong reason in the file.

**V4 registers no circumstance under which the override may be used, does not grant it, does not
exercise it, and does not write its value into any file.**

---

## 3. What each route would and would not establish — the distinction this whole discipline exists to protect

Blurring these two sentences is the failure mode. They are set out separately so that no future
reader can collapse them:

> **Established.** The *reason* PR-08 §1 gated on T-39 has been addressed on the record. G0b passes
> under the repaired instrument (`oracle_action` +68.10 L1, L4), so the VOID condition is not met,
> and T-39 issued the verdict `N` on 2026-08-17.

> **NOT established, by any route, including (b) which was actually run.** That a policy trained on
> this corpus clears the bar. **The opposite was measured.** The policy scored **−239.69 %** on the
> holdout and **−186.73 %** on forty episodes it had *trained on*. A ceiling that is real (L4) and a
> policy 307 pp below it is what `N` means: **nothing clears this bar, and the policy cannot even
> fit the data.**

Route (a), had it been needed, would have established strictly less than that: it would have
established only that the *stated reason* for the gate had been withdrawn. It would **not** have
established that any policy clears the bar, because nobody would have run one. Any wording that
lets "the gate's premise was withdrawn" drift into "the corpus is fine" or "a policy can learn
these labels" is the error this preregistration discipline exists to prevent, and it is forbidden
here in as many words.

**`+68.10` is a statement about oracles scored against oracles.** It is the corpus's own commanded
column compared with the corpus's own executed trajectory. It is not a model, not a policy, and not
a capability claim.

---

## 4. What V4 asks the owner to determine

Item 7 is closed and no determination is needed to close it. **One narrow question remains, and it
is genuinely the owner's:**

`T40_RULE_V3` §5.3 registers that *"`P`, `N`, `M` and `I` satisfy §1's 'T-39 has reported'"*. That
sentence names **PR-07 §5's** verdict set, which was written under `T39_RULE_V1`. The verdict the
project holds was issued under **`T39_RULE_V2`** — a rule that amends V1 in exactly one respect
(which quantity chunk step 0 is differenced against), quotes every threshold and every verdict
definition from V1 unchanged, and scores **1 000 chunks instead of 1 040** because a chunk anchored
at an episode's first raw index has no preceding command and is skipped rather than silently
anchored on the state row.

**The question:** does an `N` issued under `T39_RULE_V2` count as PR-08 §1's *"T-39 has reported"*,
given that `T40_RULE_V3` §5.3 was written naming the V1 verdict set?

**V4's recommendation: yes, and this is the determination in §7.** The reasons are that
`T39_RULE_V2` §1 moves no threshold and no verdict definition; that the amendment repairs the
instrument the VOID was measured through, which is the direction §1's gate was pointing; that
`T39_RULE_V2` §3 registered the set change **in advance of the run**, with its cost measured
(dropping those 40 chunks alone moves the unmodified arm from −359.41 to −344.54); and that
`PR-07-V2-RESULT.md` records the pre-registered `+68.10` expectation being met across two
independent submissions and an independent CPU driver. **If the owner determines otherwise, item 7
reverts to open and PR-08 stays shut** — which changes nothing operationally today, since items 2,
3 and 4 are open regardless.

**Two records are stale in the owner's favour to know about. V4 corrects neither, because neither
is V4's to correct:**

1. **`CLAUDE.md`'s T-39 paragraph** still says the policy arm never ran and that job 187813 died at
   108 s, and still cites PR-07 §6's VOID-row prohibition on any statement about GR00T. That was
   accurate on 2026-08-16 and was superseded on 2026-08-17 by `PR-07-V2-RESULT.md`. **No session
   may edit `CLAUDE.md` to lift the training gate**, and V4 does not; whether to refresh that
   paragraph's *factual* half is the owner's, and the training gate it carries is unaffected either
   way.
2. **`97_transfer25_restyle.sbatch`'s VOID refusal text** justifies itself with a claim PR-13
   withdrew, and its `:?` worked example — `P (oracle_action clears L1)` — describes **G0b**, not
   the verdict **P**, which V1 §5 defines on the `groot-holdout` policy arm. An operator following
   that example would attest to the wrong thing. §6 records the minimal repair; **V4 makes no edit.**

---

## 5. What V4 does not license

- **No generation.** `T40_RULE_V1` §1 binds in full: nothing is generated, no weight is trained on
  generated frames, and no number from PR-08 is quoted as a result, until **every** §8 item is
  closed **and** T-39 has reported. **Items 2, 3 and 4 are open.** Signing V4 does not open PR-08.
- **No training, on real frames or generated ones.** `CLAUDE.md`'s gate stands: whether training may
  start, and against which label space, is the project owner's call. `C`, `W` and `N` are none of
  them permission.
- **No relaxation of the VOID refusal.** A T-39 VOID still closes PR-08. V4 asks about `N`, not
  about VOID.
- **No use of `PR08_OVERRIDE_T39_VOID`**, under any circumstance, and no value written for it.
- **No claim about `docs/benchmark.md`'s L4 gate**, which stays an open decision for the owner, with
  bench specs 0.1.0 and 0.2.0 disagreeing about the repaired cell.
- **No capability claim about GR00T or any policy.** V4 measures nothing. Where it cites policy
  numbers it cites `PR-07-V2-RESULT.md` by name, as the record of a run someone else made under a
  rule registered before it, and it draws no inference beyond that document's own verdict.
- **No retro-validation of the fourteen negatives**, none of which was scored on the 1 000-chunk V2
  set.
- **No grasp claim.** The gripper channel is degenerate (peak-to-peak 0.120, 0.00 debounced
  transitions per episode), so `gripper_accuracy` was **withheld** by the scorer on every arm cited
  here.

---

## 6. Reported, not made: the minimal `97` change route (a) would have required

**Recorded for the owner and for whoever next edits that file. It is not requested, not a condition
of §7, and V4 changes nothing in `cluster/discoverer/97_transfer25_restyle.sbatch`.** Because route
(a) is moot (§2a), the only part of this with live value is the wording repair — the accepted-verdict
change should **not** be made unless the owner deliberately decides a VOID may pass, which V4 does
**not** recommend.

Three sites, in the generation branch only (the `TIMING=1` path is exempt and untouched):

- **line 424** — the `PR08_T39_REPORTED:?` message. Its sentence *"VOID IS REFUSED HERE … PR-07 §4
  defines a VOID as the finding that no policy trained on this dataset can clear our bar and that
  the defect is our own label pipeline"* would become a refusal justified on the **rule** rather
  than on the withdrawn claim: that PR-08 §1 gates on a verdict, that `T40_RULE_V3` §5.3 admits
  `P`, `N`, `M`, `I` and refuses VOID, and that T-39's operative verdict under `T39_RULE_V2` is
  `N` (2026-08-17), so a VOID attestation is citing a superseded run. Its worked example should
  also stop being `P (oracle_action clears L1)` — that describes G0b, not verdict `P` — and become
  an `N` example matching the verdict the project actually holds. *(Note for whoever writes it: the
  word of a `${VAR:?word}` expansion is parsed by bash, and an apostrophe or quote inside one breaks
  the file at parse time. Keep it free of `'` and `"`.)*
- **lines 432–437** — the verdict-token check. **Unchanged.** `VOID` must stay in the alternation so
  a VOID attestation reaches its own diagnosis instead of falling out as "no verdict token"; that
  design is correct and route (a) does not disturb it.
- **lines 439–488** — the VOID refusal block. Under route (a) *as briefed*, the accepted-verdict set
  would widen from `P | N | M | I` to admit a VOID whose premise is withdrawn, and the refusal text
  at lines 462–473 would be replaced by the same rule-based reason. **V4 recommends against the
  widening** and recommends only the text repair, leaving the branch refusing exactly as it does
  today. The override block (lines 455, 458, 478, 484–487) would be untouched in either case.

**Operationally, `97` needs no change at all to run under the verdict the project holds.** The
attestation is a verdict token plus an ISO date plus an existing non-empty artifact — for example a
string carrying `N` and `2026-08-17`, with `PR08_T39_ARTIFACT` pointing at the **`T39_RULE_V2`** run's
committed result artifact (the four arm directories under `runs/t39-baseline-seed0/` from job
188408), **not** at `_archive-187813-oracles/`, which holds the superseded V1 oracles. Verifying that
the artifact the operator names is the V2 run and not the archived V1 one is a human check; the
script hashes what it is given and cannot tell them apart.

---

## 7. Determination — **UNSIGNED**

**Proposed determination.**

> **`T40_RULE_V4` determines that `T40_RULE_V1` §8 item 7 — "T-39 has reported" — is CLOSED, on
> `VERDICT N` issued 2026-08-17 under `T39_RULE_V2` (job 188408, `PR-07-V2-RESULT.md`); that an `N`
> issued under `T39_RULE_V2` satisfies §1's "T-39 has reported" exactly as `T40_RULE_V3` §5.3
> intends, that rule having moved no threshold and no verdict definition; that a T-39 **VOID**
> continues to close PR-08 rather than open it, unchanged; and that `PR08_OVERRIDE_T39_VOID` is
> **not** granted, is not to be used, and is recommended against — its required attestation being a
> statement this project has measured to be false.**
>
> **This determination opens nothing.** `T40_RULE_V1` §1's prohibition binds in full: **§8 items 2,
> 3 and 4 are open**, so no clip may be generated, no weight may be trained on generated frames, and
> no number from PR-08 may be quoted as a result. It licenses no training run on any label space —
> that remains the project owner's separate call — and it says nothing about `docs/benchmark.md`'s
> L4 gate.

**Signature.** This document takes effect only when the line below is completed by the project
owner. **It is not signed. No agent may sign it, and no agent may act as though it were signed.**

```
Project owner: ______________________________   Date: ____________

Determination:   [ ] signed as proposed
                 [ ] signed with the amendments noted below
                 [ ] declined — item 7 reverts to OPEN and PR-08 stays shut

Amendments / notes:



```

---

## 8. Provenance

| | |
|---|---|
| rule | `T40_RULE_V4` |
| drafted | 2026-08-22, **before any clip is generated**, before any weight is trained, before any job is submitted |
| status | **UNSIGNED. Not in force.** |
| supplements | `T40_RULE_V1`, `T40_RULE_V2`, `T40_RULE_V3` — all three stand and all three are **unedited** |
| supersedes | nothing |
| corrects in the record | the standing reading that T-39's operative verdict is `VOID` and that its policy arm never ran (§1.1); the reading that §8 item 7 is the last open item (§1.3) |
| changes | **no gate, no threshold, no verdict, no arm, no clip count, no style, no seed, no ceiling** |
| leaves open | `T40_RULE_V1` §8 items **2, 3 and 4** (§1.3), and every decision `CLAUDE.md` reserves to the owner |
| generation licensed | **no** |
| training licensed | **no** |
| a T-39 VOID | **still stops PR-08** |
| `PR08_OVERRIDE_T39_VOID` | **not granted, not exercised, recommended against, and its value is written nowhere in this document** |
| sources verified while drafting | `PR-07-positive-control.md` §5–§6; `PR-07-RESULT.md`; `PR-07-V2-repaired-anchoring.md`; `PR-07-V2-RESULT.md`; `PR-12-RESULT.md`; `PR-13-RESULT.md`; `PR-08-photoreal-augmentation.md`; `PR-08-V2-…`; `PR-08-V3-…`; `cluster/discoverer/97_transfer25_restyle.sbatch` (header (a), lines 411–505); `docs/contracts/vla-training-consumer.md`; `.mc/tasks/todo/T-040-…`; and the four `bench.json` files under `runs/t39-baseline-seed0/`, read directly |
| measurements taken here | **none.** V4 computes nothing and submits nothing. Every number in it is quoted from a committed result document and cross-checked against the artifact it names |
| decided by | **nobody yet.** §7 is unsigned |
