# PR-08 — what blocks the first clip, what blocks training, and what merely destroys the yield

**Written 2026-08-25. A reading of documents and code already in the repository. Measures nothing,
registers no rule, changes nothing.**

This note exists because this session had been describing G0c's 99.2 % refusal rate as *blocking
generation*, and that is not what the documents say. The correction changes the order in which the
open items matter, so it is worth writing down.

---

## 1. Three different gates, three different moments

| | what it gates | when it runs |
|---|---|---|
| `T40_RULE_V1` **§1** | **generating a corpus at all** | before anything |
| §6 **G0a / G0b / G0c** as VOID gates | **training on generated frames** | §6: *"all three run before any training, all on CPU"* |
| G0c's **composite** | whether an individual clip survives | during generation, per clip |

The third row is the one that was being conflated with the first two.

## 2. §1's precondition is the seven §8 items, and nothing else

> *"Forbids, until every item in §8 is closed and T-39 has reported: generating a corpus, training
> any weight on generated frames, and quoting any number from this document as a result."*

So the set that gates a first clip is exactly §8 items 1–7. Current state:

| item | status |
|---|---|
| 1 recipe | closed |
| 2 consumer contract | closed — `T40_RULE_V7`, signed 2026-08-22 |
| 3 throughput + GPU-h ceiling | **OPEN** — and the number does not exist at all |
| 4 `GEOM_TOL` + `EST_DRIFT_P95` committed | **OPEN** — measured, but `gate_qualified: false`, so `configs/transfer25/pr08_geom_tol.json` still holds nulls |
| 5 annotators | closed |
| 6 partition | closed |
| 7 T-39 reported | reported; what it licenses is the owner's call |

**G0a, G0b and G0c are not on this list.** They gate training on the result, not producing it.

## 3. What the 99.2 % refusal actually is, and why it is still serious

`cluster/discoverer/97_transfer25_restyle.sbatch:843-845`:

> *"composite runs inside `scripts/restyle_transfer25.py`, **between the model writing a clip and the
> driver calling the unit a success**. A unit whose composite refused never reached `status:
> success`, and its output was renamed to `vision.uncomposited.mp4`, so the harvest cannot file it
> either."*

So the refusal happens **after the GPU work for that clip is finished.** The model generates the
clip, the composite then refuses it, and the clip is discarded.

**This is not a precondition. It is a yield problem in which the rejects are paid for in full.**

At `T40_RULE_V12`'s measured 128-of-129 refusal rate, a generation run would spend the entire GPU-h
budget and keep **0.8 %** of the output. That is not a gate refusing to let us start; it is a
pipeline that starts, costs everything, and delivers almost nothing.

## 4. Why this reorders the open work

**Formally, V12 is not on the critical path to a first clip.** Items 3 and 4 are. V12 could stay
unsigned forever and §1 would be satisfiable without it.

**Practically, resolving V12 before generating is the difference between a run that costs its budget
and yields a corpus, and one that costs its budget and yields 80 clips.** The economic argument is
strong even though the procedural one is absent, and the two should not be confused with each other
— which is exactly the confusion this note is correcting.

**Consequence for sequencing.** The two things worth spending on right now are the two that close
§8:

- **Item 3** — job 106's area distribution, then the `max_frame_fraction` decision under
  `T40_RULE_V13`, then the TIMING run. In progress.
- **Item 4** — the `GATE_QUALIFICATION_BLOCKERS`. Blocker 1's first limb is a person looking at
  overlaid apple masks; its second limb is a mask-vs-ground-truth IoU distribution, which the MuJoCo
  route can supply. Blocker 2's full-pass conjunct was supplied 2026-08-25. Blocker 3 needs the
  temporally coherent capture.

V12 is the yield question and should be resolved before spending a budget, not before closing §8.

## 5. What this note does not do

- It does **not** license generation. §8 items 3 and 4 are open and §1 is conjunctive.
- It does **not** downgrade V12. §3 is an argument for taking V12 more seriously commercially, not
  less.
- It does **not** claim G0c would refuse 99.2 % of a *generated* corpus. The 128-of-129 figure is
  measured on SOURCE clips, which is the pass that matters for the composite, but it is a pilot
  scan and `runs/pr08-g0c-refusal/G0C_REFUSAL.json` licenses nothing by its own text.
- It resolves no open question and no rule.

---

## 6. Provenance

| | |
|---|---|
| kind | reading of existing documents and code. **Registers no rule, measures nothing** |
| date | 2026-08-25 |
| sources | `PR-08` §1, §6, §8; `97_transfer25_restyle.sbatch:843-845`; `T40_RULE_V12` §1.2; `T40_RULE_V7` |
| corrects | this session's own earlier description of G0c as blocking generation |
| generation licensed | **no** |
| training licensed | **no** |
