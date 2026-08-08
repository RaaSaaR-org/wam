# PR-09 — Does a LoRA on G1 footage fix the embodiment defect?

Pre-registered 2026-08-07, **before the framework is installed on Discoverer+, before any weight is
downloaded, before any clip is captioned and before any job is submitted.** Nothing below was
written with a measured number in view, because there is not one.

Task: **T-041**. Venue: Discoverer+ (`ehpc-aif-2026pg01-905`, H200).
Rule: `T041_RULE_V1`, §6, fixed here and in `cluster/discoverer/95_eval_t041_embodiment.sbatch`.
Pipeline: `90_build_cosmos_env` → `91_stage_cosmos_weights` → `92_fetch_g1_corpus` →
`93_caption_corpus` → `94_train_t041_cosmos_super` → `95_eval_t041_embodiment`.

> **This document does not lift a freeze.** `PR-07 §7` reads *"Frozen until T-39 reports: T-32 (§2),
> any Cosmos3-Super generation, any Cosmos3-Edge work."* Every job in the chain above is submittable
> only after **T-39 reports**, or after the freeze is lifted explicitly and recorded as an open
> decision. Writing the pipeline, the corpus and this rule is not frozen; **running any of it is.**
> The scripts refuse to run without `T041_FREEZE_LIFTED` naming the reason (§8).

---

## 1. The defect this is aimed at, and the one it is not

`runs/backbone_eval/video/embodiment_grid.png` records the finding: **both video priors render a
generic manipulator where the G1's arm should be.** Neither has seen a Unitree G1 with a Dex3 hand,
so neither paints one. That is the most legible failure in the backbone evaluation and it is the one
defect a fine-tune is actually the right instrument for — the model is missing a subject, and SFT is
how a subject is added.

**It is not the same claim as "generated frames make a better VLA."** That claim is T-040's, it has
its own pre-registration (PR-08), and PR-06 already recorded that a dream is 39 % further from the
truth than a frozen frame. A P verdict here says the generator draws the right robot. It says
nothing about whether training on its output helps anything downstream, and §9 forbids reading it
that way.

**Why this before the cheaper option.** T-040 can only patch the embodiment defect *downstream*, by
compositing the real robot back over a generated frame. That works and it is the fallback. A
generator that has actually seen a G1 would not produce the defect in the first place, which is
strictly better if it can be had for ~100 GPU-h of a 5 000-hour allocation. This experiment prices
that "if".

## 2. Goal chosen, and the goal deliberately dropped

T-041 lists two goals and states they need different corpora. **This pre-registration takes
embodiment fidelity and drops visual variety.** Fixed here so the corpus cannot be reselected after
a disappointing result:

| goal | corpus | status |
|---|---|---|
| **embodiment fidelity** — stop it painting a generic manipulator | AppleToPlate (402 ep) + the 13 `unitreerobotics/G1_Dex3_*` sets | **taken** |
| visual variety — arbitrary rooms, tables, lighting | Humanoid Everyday (4 064 ep) | **dropped, see below** |

Three reasons, in order of weight:

1. **A rank-16 LoRA on ~18 h of footage learns *those* rooms, not arbitrary rooms.** T-041 makes
   this argument against itself and it is correct. "Visual variety" from a corpus of fixed scenes is
   circular; the variety would have to come from the prior we are fine-tuning away from.
2. **The gate would have to be coined.** Embodiment fidelity has a defect already recorded and a
   paired before/after that decides it with a borrowed statistic (§6). "More varied" has no
   existing bar in this repo and inventing one after seeing the samples is the failure mode the
   whole preregistration series exists to prevent.
3. **It sidesteps OD-09 entirely.** AppleToPlate is CC-BY-4.0 and the 13 Unitree sets are
   Apache-2.0 (checked 2026-08-07: all 13 `G1_Dex3_*` repos carry the Apache-2.0 tag). Humanoid
   Everyday is unlicensed, and OD-09 accepted that risk for *mining* while explicitly not covering
   distribution or sale. A LoRA trained only on CC-BY + Apache-2.0 footage has no such asterisk on
   it. **This is a consequence of the choice, not the reason for it** — reasons 1 and 2 stand
   whatever OD-09 says, and would still stand if the corpus were MIT tomorrow.

Humanoid Everyday remains the right corpus for a variety experiment. That experiment is not this
one and does not inherit this rule.

> **Amended 2026-08-07, before any job of this chain had started** — jobs 186281–186284 were all
> still `PENDING`, nothing had been downloaded, captioned or trained. This is a pre-commit draft
> edit, not an amendment to a rule that had already produced a number.
>
> **Humanoid Everyday is added to the training corpus by operator decision.** The operator was
> shown the licence position twice and ruled: this is an internal test, so no weight trained on it
> is published, redistributed or sold. That overrides **reason 3 only**, which this section
> already labelled a consequence rather than a reason.
>
> Reasons 1 and 2 are untouched, because *the goal has not changed*. This is still an embodiment
> fidelity experiment judged by the §6 gate; Humanoid Everyday enters as 4 064 more episodes of a
> real G1, not as a variety corpus, and "more varied" is still not a claim this run can make.
>
> Two consequences that are real and are accepted, not hidden:
>
> - **The viewpoint is different.** Humanoid Everyday's only camera key is `egocentric` — a head
>   view that fills the frame with the G1's own hands and never shows its body. For "are these
>   Dex3 hands" that is on-target and arguably the best footage in the corpus. For "is this a G1"
>   it contributes nothing, and it now dominates the pool by episode count.
> - **It roughly quadruples the corpus at a fixed 500 iterations** (§7), so the run sees far fewer
>   passes over any given clip. If the verdict comes back **N**, "undertrained" becomes a live
>   alternative explanation that the pre-HE corpus would not have had. §9 owns this.
>
> The §6 rule, the statistic, the N=30, and the §5 selection rule are all unchanged.

> **Amended again 2026-08-08 — the addition above is withdrawn and the corpus reverts to the
> fourteen repos this section originally pre-registered.** Still before any training: jobs
> 186348–186350 downloaded footage and nothing was captioned or trained, so this remains a
> pre-commit edit. `T041_RULE_V1` has never run.
>
> **The reason is operational, not a reversal of OD-09.** The HF Hub rate-limited
> `USC-PSI-Lab/Humanoid-Everyday-G1` with HTTP 429 on three consecutive jobs — 186348 after
> 13 m 37 s, 186349 after 7 m 59 s, 186350 after 5 m 1 s having transferred nothing at all. The
> other fourteen repos completed (70 GB, 14/14). Fetching the fifteenth needs an authenticated
> token, and the operator chose to drop it rather than put a personal credential on a shared
> EuroHPC filesystem for a corpus that was optional to begin with. **OD-09's licence reasoning is
> not retracted** — it was never the reason for this removal, and it stands as recorded should the
> corpus be wanted again.
>
> **What this restores, at no cost to the goal:** reason 3 above is back in force, so the LoRA
> carries no OD-09 asterisk; the `egocentric` viewpoint no longer dominates the pool; and the
> corpus is roughly a quarter the episode count, so the "undertrained at 500 iterations"
> alternative explanation flagged above is substantially weaker. All three were listed as accepted
> costs of the addition. Withdrawing it withdraws them.
>
> **One measurement that changes how the addition should have been read.** Humanoid Everyday is
> ~2.5 GB against the fourteen repos' 70 GB, because its clips are far shorter. The note above
> says it "roughly quadruples the corpus", which was true by *episode count* and badly misleading
> as a proxy for training signal. Recorded here rather than corrected in place, per §3 of
> `docs/handoff.md`.
>
> Nothing else moves: the goal, the §6 rule, the statistic, the N=30, and the §5 selection rule
> are all still unchanged.

## 3. The recipe is NVIDIA's, run unmodified — checked against source 2026-08-07

Verified from `NVIDIA/cosmos@main` and `NVIDIA/cosmos-framework@main`, not from a card or a blog:

| | |
|---|---|
| model | `nvidia/Cosmos3-Super`, revision **`e0262be9d8f7586bc24c069a2aed2b665bdff266`**, 134.6 GB (≈64.6 GB params bf16), not gated |
| licence | OpenMDW-1.1 (`openmdw.ai/license/1-1/`), card states commercial use permitted |
| recipe | `cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_super.sh` |
| config | `toml/sft_config/vision_sft_super.toml` — **used byte-identical on the cold start** |
| what trains | `lora_rank 16`, `lora_alpha 32`, targets `{q,k,v,o}_proj_moe_gen` (generation tower only), `optimizer.keys_to_select = ["lora_"]` |
| parallelism | FSDP, CP=2, DP auto, full activation checkpointing, `max_iter 500`, `grad_accum_iter 2`, `save_iter 100` |
| tested on | 8×H100 80 GB (README). H200 is 141 GB, so the headroom is real but **unconfirmed on our node** |

**Our code appears in exactly three places**, and nowhere inside the trainer:

1. `scripts/prepare_cosmos_corpus.py` — lays LeRobot episodes out as the clip tree the captioner
   consumes, applies the loader's own filters, writes the provenance manifest.
2. The resume patch (§4), one TOML key, diffed into the log every time it is applied.
3. `scripts/eval_t041_embodiment.py` — the paired scorer for §6.

Captioning is **also NVIDIA's**, which is the largest correction to T-041's cost model: the
framework ships `cosmos_framework.scripts.caption_from_video` (a Qwen3-VL-8B-Instruct-FP8 vLLM
server, two-phase structured-JSON → dense narrative) and `captions_to_sft_jsonl`. T-041 recorded
captioning as "its own pipeline, cost this before anything else". It is a shipped script and one
GPU. **That item is answered, not by us building it, but by having looked.**

## 4. Four things about this cluster that will break the recipe if unhandled

Recorded here rather than discovered at 03:00 on a requeue.

**a. The resume trap — the one place a silent wrong number can enter.** The shipped TOML sets
`checkpoint.keys_to_skip_loading = ["net_ema.", "lora_"]`. That is correct for the cold start: the
base checkpoint has no LoRA tensors and they must initialise fresh. On a **resume** from
`checkpoints/iter_<N>/` the same line would skip the LoRA tensors we just spent GPU-hours training
and re-initialise them — while `optim/`, `scheduler/` and the `iteration` counter in `trainer/`
*are* restored. The run would continue from iteration 300 with a fresh adapter and a stale optimiser
state, log plausible losses, and produce a checkpoint that is not what its own metadata says it is.
This is T-37's transposed-`xmat` failure mode exactly: finite, plausible, correctly-shaped, and
invisible to any assertion about shape or range.
**Handled:** `93_train_t041_cosmos_super.sbatch` uses NVIDIA's TOML byte-identical on the cold
start, and on resume generates a copy with `keys_to_skip_loading = ["net_ema."]`, printing the diff
into the job log before torchrun starts. A pass that cannot show that diff in its log is not a
valid resume.

**b. 4-hour walltime against an 8-GPU job that must not lose work.** `save_iter = 100` and a DCP
checkpoint that carries the iteration counter make the run chainable; `PreemptMode=REQUEUE` and
`OverSubscribe=FORCE:4` make chaining mandatory rather than optional. Worst case per interruption is
99 iterations.

**c. Eight GPUs exist on exactly one node.** `dgx1` has 8; `dgx2` has 7 general-purpose plus one
`gpu_biz`. An 8-GPU request can only ever land on dgx1 and queues behind every other 4/7/8-GPU job.
`NPROC=4` is supported by the recipe README (`--nproc_per_node=4`) and is the fallback, but it is
**untested at this model size** and CP=2 then leaves DP=2. If 4 GPUs are used, the artifact records
it and the run is reported as a 4-GPU run, never as "the recipe".

**d. Billing bites before GPU-hours do.** At 66.18 billing-hours per GPU-hour the rule is ≤26
threads and ≤257 GB per GPU. An 8-GPU job is therefore `--cpus-per-task=208 --mem=1800G`, which is
essentially the whole node — and Slurm bills allocated, not used.

## 5. Arms

| arm | what it is | why |
|---|---|---|
| **base** | `Cosmos3-Super` at the pinned revision, no adapter | the defect as it stands. Without it there is no paired comparison |
| **lora** | the same model + the LoRA from this run | the intervention |

Both arms generate from the **identical** held-out prompt set, identical seeds, identical sampler
settings, in the same job.

The prompt set is drawn from episodes excluded from the SFT corpus by `prepare_cosmos_corpus.py`'s
seeded split. What is committed before generation is `configs/cosmos3/t041_eval_selection.toml`:
the **selection rule** (sort the val clips by uuid, take the first 30), the corpus seed, and every
sampler setting. The prompt *text* is a structured-JSON caption produced by job `93`, so it cannot
predate the captioning — but it is a deterministic function of the committed rule and the committed
split, `make_t041_eval_prompts.py` recomputes it, and its sha256 goes into the verdict. A prompt set
assembled after seeing base-model output is not a held-out set, and this construction makes that
impossible rather than merely forbidden.

*(Amended 2026-08-07, before this document's first commit and before any measurement was taken
under it. §5's original text said "the prompt set is `t041_eval_prompts.jsonl`, committed before the
training job is submitted"; writing the pipeline made it clear that file cannot exist that early.
This is a draft edit, not an amendment to a registered rule — `T041_RULE_V1` in §6 is unchanged and
has never been in git under a different form.)*

## 6. Gate — `T041_RULE_V1`

The measurement is **paired and binary**: for each of `N = 30` held-out prompts, does the generated
clip render a **three-fingered Dex3 hand on a Unitree G1 arm**, or a generic manipulator? Scored per
prompt for both arms by the same rubric in `scripts/eval_t041_embodiment.py`, with the arm labels
**hidden from the scorer** and the pairing revealed only after all 60 clips are scored.

No threshold is coined. The statistic is the **exact McNemar test** on the discordant pairs at
**α = 0.05**, one-sided in the direction "lora fixes what base got wrong" — a borrowed standard, not
a number chosen here. `N = 30` is fixed in advance; the run does not stop early and does not extend.

**G0 · VOID (decided before the pairing is revealed, and can stop everything).**

- **G0a — the defect must be present.** `base` must fail on **at least 15 of 30**. If the base model
  already renders a G1 on most held-out prompts, there is no defect for this experiment to fix, the
  finding is against `embodiment_grid.png`'s generality, and no verdict is issued.
- **G0b — the scorer must be able to see the thing.** Ten real held-out frames and ten
  `embodiment_grid.png` negatives, shuffled in with the 60, must be scored correctly **20/20**. A
  rubric that cannot separate a real Dex3 from a recorded failure cannot adjudicate a generated one.
- **G0c — the run must be a run.** `latest_checkpoint.txt` reports iteration 500, every resume pass
  in the chain printed its `keys_to_skip_loading` diff (§4a), and the LoRA safetensors export is
  non-empty. Any missing → VOID, not a weaker pass.

**Verdicts:**

| | condition | reading |
|---|---|---|
| **P** | McNemar significant at α = 0.05 in the stated direction | ~100 GPU-h of LoRA on G1 footage fixes the embodiment defect |
| **N** | not significant, and `lora` fixed **≤ 2** of the base's failures | it does not. The defect is T-040's to patch by compositing, and Super is not the route |
| **I** | not significant, but `lora` fixed **≥ 3** | underpowered, not refuted. Record and stop — **there is no second run under this rule** (§7) |
| **VOID** | any G0 | a defect report against the rig, not a statement about Cosmos |

**Recorded regardless of verdict:** the discordant counts both ways, the pinned model revision, the
corpus manifest hash, the exact iteration reached, `NPROC`, every pass's wall time, the measured
seconds/iteration, and the total GPU-hours billed.

## 7. Cost, and the ceiling

| stage | GPUs | ceiling | note |
|---|---|---|---|
| `90` build env | 0 | — | `2cpu-single-host` QoS, no GPU hours |
| `91` stage weights + DCP convert | 1 | 4 GPU-h | 134.6 GB download, conversion is the slow half |
| `92` fetch corpus + lay out | 0 | — | `2cpu-single-host` QoS, no GPU hours |
| `93` caption corpus | 1 | 6 GPU-h | ~800 clips through Qwen3-VL-8B-FP8 |
| `94` **probe** | 8 | 8 GPU-h | `PROBE=1`, mandatory, measures seconds/iteration |
| `94` train | 8 | **96 GPU-h** | 3 passes × 4 h × 8 GPUs, enforced by `MAX_RESTARTS=2` |
| `95` eval | 8 | 8 GPU-h | 60 clips, both arms, one job |
| | | **≤ 122 GPU-h** | **2.4 %** of the 5 000-hour allocation |

**The probe is a gate, not a warm-up.** If the measured step time implies 500 iterations cannot
finish inside 96 GPU-h, the run is **not started** and the shortfall is recorded as the finding. The
ceiling is not raised to fit the recipe. Exactly one alternative is pre-registered against that
outcome and only against it: **`Cosmos3-Edge`** (2B dense, full fine-tune, fits 4 GPUs) on the
identical corpus, prompts and rule — reported as **attempt 2 of 2**, never as attempt 1. There is no
attempt 3.

## 8. What must exist before anything is submitted

1. ✅ `scripts/prepare_cosmos_corpus.py` + `tests/test_prepare_cosmos_corpus.py` (20 tests)
2. ✅ `cluster/discoverer/90…94` — the chain, with the §4a resume patch and the §7 ceiling
3. ✅ `configs/cosmos3/t041_eval_selection.toml` + `scripts/make_t041_eval_prompts.py`.
   **Amended from "commit the 30 prompts": the prompt text is a structured-JSON caption produced
   by job `93`, so it cannot exist before the corpus is captioned.** What is committed instead is
   the *selection* — rule, N, corpus seed, and every sampler setting — because the failure being
   guarded against is choosing prompts after seeing base-model output, and a deterministic
   function of the val split cannot be chosen after the fact. `make_t041_eval_prompts.py` executes
   it, refuses if the manifest's seed differs from the registered one or if any selected clip is
   in `train`, and hashes the result into the verdict.
4. ✅ `scripts/eval_t041_embodiment.py` + `tests/test_eval_t041_embodiment.py` (31 tests) +
   `95_eval_t041_embodiment.sbatch` — the §6 rubric, the G0 gates and the exact McNemar test, in
   git before generation. Three separate steps (`build-sheet` / `judge` / `verdict`) so the arm
   labels live in a file the scoring step never opens, and so a **human** can score the same sheet
   if G0b fails
5. ✅ **The PR-07 §7 freeze is lifted — `OD-10`, 2026-08-07, by the user.** Narrowly: the
   *Cosmos3-Super generation* clause only. T-32 and Cosmos3-Edge stay frozen as written, and
   **PR-07 itself is not edited** — the rule stands, and OD-10 is the decision recorded against it.
   Every job in the chain still exits FATAL unless `T041_FREEZE_LIFTED` names a reason, and that
   string is written verbatim into `run_metadata.json`, so every artifact carries the decision that
   allowed it. What OD-10 accepts: if T-39 later returns **N**, a **P** here stays true about the
   generator but loses most of its downstream value. Bounded by §7's 122 GPU-h.
6. ⬜ 8-GPU VRAM confirmed on dgx1 by the probe (item: §7)

Items 1–4 landed **2026-08-07**, after this gate and before any cluster contact. Item 6 is the only
one that cannot be closed off-cluster: it is a measurement, and the probe is how it is taken.

## 9. What this cannot answer

- **Not a downstream result.** A P says the generator draws the right robot on 30 held-out prompts.
  It is not evidence that VLA training on generated frames helps — that is T-040/PR-08, and PR-06
  already recorded the dream losing to a frozen frame by 39 %.
- **Not an action-conditioned world model.** Super's card lists no humanoid, no G1 and no 28-dim
  Dex3 among its supported action inputs, and the action-conditioned SFT cookbooks are Nano-only.
  This is a **video** fine-tune and cannot become anything else by this route.
- **Not a statement about visual variety** (§2), and not about Humanoid Everyday, which this
  experiment does not touch.
- **Not a comparison against Transfer2.5.** Different tool, different question, different
  pre-registration.
- **30 prompts, one seed set, one rubric, one task family.** The corpus is pick-and-place on a G1;
  nothing here generalises to a scene the corpus does not contain.
- **The scorer is a VLM, and that is the weakest part of this design.** G0b's 20/20 on real footage
  is a *necessary* check, not a sufficient one: separating a real Dex3 from a real parallel gripper
  is easier than adjudicating a generated frame, where the failure mode is a plausible-looking hand
  with the wrong number of fingers. Stated here rather than discovered in review. Two things bound
  it: the judge never sees which arm produced a clip (the labels are in a file the scoring step does
  not open), and `scoring_sheet.jsonl` + `items/` are a **human-rescoreable artifact** — a person
  can score the same 80 blinded clips and `--verdict` applies the identical rule to their
  `scores.jsonl`. If G0b fails, that is not a fallback, it is the required path.
