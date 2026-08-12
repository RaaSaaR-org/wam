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

> **Amended 2026-08-09 — one of the fourteen repos was a byte-for-byte copy of another, and four
> of the thirty eval prompts were byte-identical to TRAIN clips. Train 3432 → 3133; fourteen
> training sources → thirteen.** Same standing as the notes above and the three in §5: this
> **precedes every measurement.** No training step has run, no clip has been generated by either
> arm, no job in the `90…95` chain has been submitted, and `T041_RULE_V1` has never run. The rule
> is untouched — McNemar over the G0 gates says nothing about corpus composition — and **`N = 30`
> and G0a's registered `>= 15/30` are unchanged**, which is the whole point of the shape the fix
> took. The table above is left exactly as written and this note records what the corpus now is,
> per §3 of `docs/handoff.md` ("rules are versioned, never edited in place", `docs/handoff.md:100`).
>
> **What is now registered.** `~/wam-t041/cosmos-g1-embodiment`: **3133** train clips, **30** val
> clips, **13** sources contributing to train, and `MANIFEST_SHA256` =
> **`2af81b9997f0de42e3fee01600bf34c67b7cdcb86b8ac5ab1094e21dcf77c63e`**, which is what
> `sha256sum manifest.json` returns today and what `92b_register_corpus.sbatch` will check. The
> pre-dedupe stamp is recorded as `6bec507e2816…`; that manifest no longer exists, so — unlike
> every other number in this note — it is quoted rather than re-measured.
>
> **The finding, and the evidence, all of it re-measured 2026-08-09 rather than quoted.**
> `g1-dex3-graspsquare-dataset` is not a second task. It is `g1-dex3-blockstacking-dataset`:
>
> - all six `cam_left_high` source mp4s hash identically across the two raw trees (`sha256sum` over
>   `~/wam-t041/raw/G1_Dex3_{BlockStacking,GraspSquare}_Dataset/videos/…/chunk-000/*.mp4`, 6/6);
> - the LeRobot episode metadata is identical in **79 of its 80 columns** over all 301 episodes —
>   every `from_timestamp`, `to_timestamp`, `chunk_index` and `file_index` — so "the same-numbered
>   twin" is not an inference, the two repos cut the same bytes at the same boundaries;
> - the one column that differs is `tasks`: `"stack three block"` against **`"camera packaging"`**
>   — a third dataset's label, on a second dataset's name, over the first dataset's pixels;
> - prepared, that is **299 duplicate pairs** in a corpus of 3462 clips with **3163 unique
>   sha256**.
>
> The repo stays listed at `configs/cosmos3/corpus_g1_embodiment.tsv:38`, because it is a true
> statement about what was fetched.
>
> **The serious consequence is the eval, not the training weighting, and it is not buried here.**
> Double-weighting one source is a bias: it is visible in the manifest, it is arguable, and it
> moves a loss curve. This did something worse. **Four of the thirty pre-registered eval prompts —
> 13 % of the eval set — were byte-identical to clips in TRAIN:**
>
> | held-out prompt | byte-identical TRAIN clip |
> |---|---|
> | `blockstacking_ep000077` | `graspsquare_ep000077` |
> | `blockstacking_ep000126` | `graspsquare_ep000126` |
> | `graspsquare_ep000224` | `blockstacking_ep000224` |
> | `graspsquare_ep000239` | `blockstacking_ep000239` |
>
> **And the bias ran toward the registered hypothesis.** The contamination is not symmetric between
> the arms: `lora` was fit on those exact pixels, paired with a caption the same captioner produced
> from those exact pixels, and `base` never saw them. On those four prompts the adapter is being
> asked to reproduce footage it memorised, and the arm that benefits is the one §6 predicts will
> win. A contaminated holdout biased toward the prediction is the worst case of the two available:
> it is the direction least likely to be questioned when the number comes back, and the most
> expensive to have believed.
>
> **Why the existing gate passed it.** `check_prompts_are_held_out`
> (`scripts/eval_t041_embodiment.py:303-329` as it stood) and `make_t041_eval_prompts.py:91-94`
> both establish disjointness by **uuid**, and both were satisfied — the uuids really are disjoint.
> A uuid is a filename. Neither ever asked about content, and the manifest recorded a per-clip
> sha256 the whole time.
>
> **The fix deletes from TRAIN only, and that is a decision, not a convenience.**
> `scripts/dedupe_cosmos_corpus.py:120-142`, two rules in order: rule 1 removed every train clip
> whose sha256 matched any val clip's — the four above; rule 2 kept the lexicographically smallest
> uuid of each remaining duplicate pair and removed the other 295. **Val was not touched — not one
> file, not one manifest entry — so it is byte-identical to what was registered.** The alternative
> fixes all move the eval: re-splitting draws a different holdout, and dropping the four
> contaminated prompts leaves 26. Either would put `n` and G0a's `>= 15/30` (§6;
> `scripts/eval_t041_embodiment.py:41`) up for renegotiation *after* the defect was known, which is
> precisely the move this document exists to prevent — and it was avoidable, so it was avoided.
> `n = 30` and `>= 15/30` stand as registered rather than as re-derived. The tie-break is
> lexicographic rather than filesystem order because AC-04 traces a rollout back to the corpus and
> that corpus must be the same on any machine; the post-conditions (zero shared bytes between
> splits, no duplicate bytes within train, manifest/disk/jsonl agreement) are re-checked by
> `dedupe_cosmos_corpus.py:213-247` and the manifest carries a `dedupe` block naming both rules and
> all 299 removed uuids.
>
> **Fourteen sources became thirteen, stated plainly — and the fourteenth was never a distinct
> source.** All 297 `g1-dex3-graspsquare-dataset` train clips were duplicates, so all 297 went and
> **it now contributes zero train clips**; BlockStacking keeps 295 and the pixels are the same
> pixels. §2's table should be read as thirteen training sources from here on. Its **two val clips
> remain**, because removing them is a re-split by another name; they are BlockStacking footage
> under GraspSquare's filename, their captions were generated from the pixels rather than from the
> LeRobot task string, so the prompts describe what is on screen and are not wrong. The manifest's
> `sources` block still reports what the *scan* kept per source, which is a true statement about
> the scan; `counts` is what the corpus is, and `dedupe` is the explanation of the difference.
>
> **No unique content was lost.** Corpus-wide unique sha256 is **3163 before and after** — 3133 in
> train, 30 in val, zero shared. Every deleted clip had a surviving twin with the same bytes. What
> the corpus lost is 299 copies and one name.
>
> **The gate is hardened, and it cannot change this outcome — confirmed, not assumed.**
> `check_prompts_are_held_out` now additionally refuses any prompt whose clip sha256 appears
> anywhere in train (`scripts/eval_t041_embodiment.py:303-368`), and
> `make_t041_eval_prompts.py:96-117` applies the same rule at selection time, from the other end.
> Both read the sha256 the manifest already records, so neither hashes anything at eval time. Run
> against the real manifest it **passes** — 30 prompts, all in val, none byte-identical to train —
> and run against a reconstruction of the corpus as it was, it names all four pairs. The uuid check
> is kept alongside, not replaced: it is what catches a prompt set built from the wrong split,
> where a sha comparison would notice nothing. This is defence in depth against the next corpus,
> not a new gate on this one (`tests/test_eval_t041_embodiment.py:327-378`).
>
> Nothing else moves: the goal, §5's selection rule and sampler settings, `T041_RULE_V1`, the
> statistic, `N = 30`, and §7's ceiling are all unchanged. The two items §5 leaves open — G0b's
> real 640×480 calibration clips downscaled to 320×256, and §7's 8 GPU-h line for job `95` — are
> untouched by this and still open.

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

> **Amended 2026-08-12 — generation runs `--no-guardrails`. The recipe is no longer literally
> unmodified, and this section's title is the thing being amended.** Written after job `187249`
> failed and **before its replacement is submitted**; no clip has been generated by either arm, and
> `T041_RULE_V1` has still never run.
>
> **Forced, not chosen.** `187249` got past the base-arm load this time and died 71 s later, on the
> guardrail:
>
> ```
> CalledProcessError: ['uv','run',…,'hf','download','nvidia/Cosmos-Guardrail1', …]
>                     returned non-zero exit status 1
> ```
>
> The framework swallows the subprocess stderr on non-zero ranks, so job `187303` (free QoS, no GPU)
> re-ran the command to get it: **`Error: Access denied. This repository requires approval.`** The
> Hub API says `"gated":"auto"` and a file fetch returns `x-error-code: GatedRepo`. Network is not
> the problem — `huggingface.co` answers 200 from the compute node, and only the repo's README came
> down. Access needs an accepted NVIDIA Open Model Licence and an authenticated token; there is no
> HF token on this allocation, and **accepting a licence is the account holder's act, not this
> agent's**. §3's table records Cosmos3-Super itself as "not gated" and that is still true — the
> guardrail is a different repo with a different licence.
>
> **It is also the better configuration, which is why this is recorded as a decision and not only as
> a constraint.** Guardrails are four filters (`docs/inference.md:268`), and the last one is a
> RetinaFace **face-blur post-processor** that rewrites the frames on the way out —
> `_run_video_guardrail` returns the tensor it replaces (`inference/inference.py:1735`). On 320×256
> footage of a Dex3 hand, a false-positive blur lands on the one region every §6 rubric item is
> about. A filter that silently edits the evidence is worse here than no filter. Against that: the
> prompt set is 30 captions of a G1 arm moving fruit, tissues and a phone charger, so the text
> blocklist, the text classifier and the video classifier have no legitimate work to do on it.
> NVIDIA documents `--no-guardrails` as the configuration for exactly this case — a local checkpoint
> running offline (`docs/inference.md:251`).
>
> **It cannot bias the comparison.** The guardrail is a separate model applied to the prompt and to
> the finished frames; it is not in the generation graph and does not consume the sampler's RNG. The
> flag is one token on the single `torchrun` line that both arms share, so "identical across arms"
> holds by construction rather than by inspection. Registered before the run, as required — and
> **registered once**: re-running the eval under the other setting and keeping the better verdict is
> the thing §6 exists to forbid, so this setting stands whatever comes back.
>
> **How to undo it.** Accept the licence at `huggingface.co/nvidia/Cosmos-Guardrail1` and put a
> token at `${HF_HOME}/token`; the flag then comes off and the recipe is literally unmodified again.
> That is a fresh registration, not a re-run of this one.

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

> **Added 2026-08-08 — e. Two things about the *corpus* that break the recipe, and where
> preparation now runs.**
>
> Both were found by running the pipeline, not by reading about it, and both produce a corpus that
> trains rather than a job that crashes.
>
> **The sources are AV1.** All fourteen. LeRobot's default encoder is `libsvtav1` and none of them
> overrode it. NVIDIA's captioner drives a vLLM server, and vLLM decodes video through OpenCV
> only — every backend in its registry shares one OpenCV mixin. That build opened each file, read
> the container header correctly (377 frames, 30 fps, 640×480) and then failed every `cap.grab()`.
> Job 186357 sent 372 requests, received `array([], shape=(0, 480, 640, 3))` for each, logged
> `0/372 videos were successfully captioned`, wrote 372 empty files and **exited 0**. `ffprobe`
> called the corpus valid throughout, because "is this file well-formed" and "can the decoder that
> will actually read it get pixels out" are different questions with, here, different answers.
> **Handled:** `prepare_cosmos_corpus.py --mode transcode` writes H.264 yuv420p, and
> `scripts/verify_clip_decode.py` re-checks every clip *with the captioner's own interpreter* —
> verifying with any other `cv2` proves nothing, so the script prints which one it used.
>
> **Thirteen of the fourteen are LeRobot v3.0.** A clip is not a file there: episodes are
> concatenated into a handful of mp4s and each is a `[from_timestamp, to_timestamp)` window in
> `meta/episodes/*/*.parquet`. Three details each yield a plausible wrong corpus — cameras roll
> over to new files **independently** (episode 50 of BlockStacking is in `file-001` for
> `cam_left_high` and `file-000` for the other three), timestamps reset to `0.0` at each rollover,
> and `to_timestamp` is **exclusive**, so cutting with ffmpeg's `-to` appends a frame of the next
> episode to every clip. **Handled:** the reader resolves the file per (episode, camera) and cuts
> with `-frames:v <length>`; each of the three has a test.
>
> **Preparation moved to a workstation** (`workstation/`). Every T-041 failure so far — the 429,
> v3.0, the queue stall, the AV1 captioning — has been IO, format or scheduling, and each cost
> hours of Slurm queue to learn something a workstation answers in seconds. This does not change
> the experiment, the corpus, the recipe or the gate; it changes which machine does the ffmpeg. The
> cluster's remaining job is training. **This affects §7's cost accounting favourably** — the 6
> GPU-h budgeted for captioning is no longer drawn from the allocation — and the corpus is now
> defined once, in `configs/cosmos3/corpus_g1_embodiment.tsv`, read by both paths so they cannot
> disagree about what it is.

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

> **Amended 2026-08-08 — generation geometry now matches the corpus. `720` / `16,9` → `480` /
> `4,3`.** Unlike the note above, this one edits a file that *was* already in git
> (`configs/cosmos3/t041_eval_selection.toml`), so it is recorded here rather than rewritten away.
> **It precedes every measurement: no clip has been generated by either arm, and no job in the
> `90…95` chain has been submitted.** `T041_RULE_V1` in §6 is untouched — the verdict rule is a
> McNemar test over the G0 gates and never mentioned resolution.
>
> **What was wrong.** The settings came from the audiovisual cookbook's own payload example and
> were never checked against our corpus. All 14 sources record at **640×480, 4:3**, and
> `prepare_cosmos_corpus.py` preserves source resolution on purpose (§4's "source resolution is
> required"). So the registered settings trained a LoRA on 4:3 480p and then asked it to generate
> 16:9 720p.
>
> **Why it mattered, stated precisely.** Both arms always used identical settings, so this could
> **not** have produced a false **P** — §5's paired construction held throughout. What it could
> produce is an **ambiguous N**: with the adapter evaluated in a geometry and scale it never saw,
> "the LoRA does not fix the embodiment defect" and "the LoRA fixes it, but not at 16:9 720p"
> return the same verdict. §9 already forbids over-reading a P; an N that cannot be attributed is
> the same failure pointing the other way, and it is removable in advance rather than diagnosable
> afterwards.
>
> **The second reason is §6 G0b.** The rubric is calibrated on **real** held-out clips, which are
> 640×480 because that is the resolution the robot recorded. The judge must clear G0b at that scale
> or no verdict is issued at all — so generating at 640×480 puts generated and calibration clips at
> one scale instead of two, at zero cost.
>
> **Why this direction and not the other.** `resolution` is the output height and 4:3 is natively
> supported (Cosmos3-Super card: 256p/480p/720p; 16:9, 4:3, 1:1, 3:4, 9:16), so `480` + `4,3` is
> exactly 640×480 — no resampling on either side. Moving the *corpus* to 16:9 was rejected in both
> available forms: pillarboxing to 854×480 teaches the adapter to draw black bars into every
> generation, and cropping to 640×360 discards the vertical field of view that holds the torso,
> arms and Dex3 hands — the embodiment signal §2 is buying, and the reason `cam_left_high` is the
> registered camera. Generating fewer pixels is also cheaper, which loosens §7's 8 GPU-h eval line
> rather than tightening it.
>
> **One related mismatch is left open, deliberately.** `fps = 24` against a corpus that is 30 fps
> throughout. It is the same class of finding and it is *not* changed here, because the operator's
> decision was taken on resolution; changing a second registered sampler setting under cover of the
> first is the kind of quiet scope creep pre-registration exists to prevent. It is recorded in
> `TASKS.md` as open.

> **Amended 2026-08-09 — the geometry registered above was never reachable. `480` / `4,3` → `256` /
> `4,3`, and `fps` 24 → 30.** Same standing as the note it corrects: it edits a file that is already
> in git (`configs/cosmos3/t041_eval_selection.toml`), so it is recorded here rather than rewritten
> away, and it **precedes every measurement** — no clip has been generated by either arm and
> `T041_RULE_V1` has still never run. The rule itself is untouched: McNemar over the G0 gates
> mentions neither resolution nor fps.
>
> **The note above is right about why geometry matters and wrong about the mechanism, and the
> mechanism was the whole argument.** `resolution` is not an output height. It is a key into
> `VIDEO_RES_SIZE_INFO` (`cosmos_framework/data/generator/utils.py:42-74`), whose 4:3 entries are
> exactly three — `"256"` → **320×256**, `"480"` → **736×544**, `"720"` → **1104×832**.
> So `480` + `4,3` generates 736×544, not the "exactly 640×480" claimed above, and
> **there is no 640×480 bucket in the table at all**. The corpus geometry is not expressible through
> this API. `args.py:415-434` resolves the key and `95_eval_t041_embodiment.sbatch` passes it into
> the payload untouched, so 736×544 is what the eval would have generated. This is the distinction
> that makes it an amendment rather than a preference: **the registered value could not have done
> the thing it was registered to do.** "No resampling on either side" was not achieved badly; it was
> never on offer.
>
> **Why `256`, and why the choice is forced.** Once matching the corpus is impossible the only
> remaining question is which mismatch to accept, and the answer is the one the adapter was fit
> under. `vision_sft_super.py:272` pins `resolution="256"` for the SFT dataset, and the TOML cannot
> override it — `DataloaderTrainConfig` has four fields and forbids extras
> (`sft_config.py:624-665`), so the recipe we run byte-identical (§3) trains at 320×256 whatever the
> config says. Every 640×480 clip is scaled by 0.533 to 341×256 and centre-cropped
> (`sft_dataset.py:192-195, 255, 278`). **320×256 is therefore the geometry the LoRA learned in, and
> 736×544 is a scale it never saw** — which is the objection this section already raised against
> 720p 16:9, applied to the value that replaced it.
>
> **`max_sequence_length = 45056` and `resolution = "256"` are a matched pair, not two defaults.**
> At 736×544 the median 693-frame clip needs roughly 68k tokens, over the budget — and NVIDIA's
> `PackingDataLoader` drops an over-budget sample silently rather than raising. The token budget was
> sized for the 256 bucket; changing the bucket without changing it does not fail loudly, it trains
> on a quietly different corpus. That is the same class of finding as §4a and it is why the pin is
> load-bearing rather than incidental.
>
> **What this costs, stated plainly, because it removes one of the two reasons given above.** The
> G0b argument — generated clips and calibration clips at one scale instead of two — can no longer
> be satisfied by choosing a bucket, because the calibration clips are **real 640×480 footage** and
> no bucket produces that. Matching them is impossible; matching training is possible; those are not
> both available. The remaining difference has to be removed on the calibration side, by downscaling
> those clips to 320×256 before the judge sees them. **That is not done, and §6 G0b must not be run
> until it is.**
>
> **`fps` 24 → 30, decided on its own evidence.** The note above left this open on the explicit
> ground that the operator's decision was about resolution and a second sampler setting must not
> ride in under cover of the first. That objection is answered rather than ignored: this is its own
> finding with its own mechanism, and it is recorded as its own item. `vision_sft_super.py:266` sets
> `conditioning_fps = -1`, so `sft_dataset.py:297` passes each clip's **own** fps through, and the
> corpus is 30.0 fps throughout (3432/3432 clips, `decode_report.json`). That value reaches mRoPE as
> a temporal stride of `base_fps/fps` (`mrope.py:169-174`, `enable_fps_modulation=True`): 24/30 = 0.8
> per latent frame in training, against 1.0 if we generate at 24. A 25% stride mismatch in the one
> axis a video model is most sensitive to, and — like the resolution error — legal, accepted by
> `args.py:377`, and productive of entirely plausible video. That is exactly why it had to be
> checked rather than watched.
>
> **One item is left open and is not resolved here: `num_frames = 189`.** It is `4*47+1`, so
> `args.py:536` leaves it unrounded, and at 30 fps it is 6.3 s. It is also **shorter than the
> shortest clip the adapter ever saw** (249 frames; the training median is 693), so the eval would
> ask for a duration outside the training distribution in the axis this amendment has just finished
> arguing about. The 256 bucket caps `num_frames` at 400, and the largest legal `4N+1` under that
> cap is 397. **No value is registered in place of 189 by this amendment.** Recording an open
> mismatch is not the same as leaving it registered by inattention, and it must be closed — in
> writing, here — before job 95 is submitted.

> **Amended 2026-08-09 — the item left open immediately above is closed. `num_frames` 189 → 397.**
> Same standing as the three notes it follows: it edits a file already in git
> (`configs/cosmos3/t041_eval_selection.toml`), and it **precedes every measurement** — no clip has
> been generated by either arm, no job in the `90…95` chain has been submitted, and `T041_RULE_V1`
> has still never run. The rule is untouched: McNemar over the G0 gates mentions neither duration
> nor frame count. **This closes the first of the two items the note above left open. The second —
> G0b's calibration clips are real 640×480 footage and must be downscaled to 320×256 before the
> judge sees them — is NOT closed here and still blocks G0b.**
>
> **The training duration distribution, measured rather than asserted.** The note above quoted 249
> and 693 from the decode reports; both are right, and the reason they are the *training* durations
> needed checking separately. `vision_sft_super.py:271` sets `num_video_frames = -1`, which puts
> `SFTDataset` in native-chunk mode — the whole window at the window's own `temporal_interval`
> (`sft_dataset.py:215-219`) — and `captions_to_sft_jsonl.py:172-174` writes every window as
> `start_frame = 0`, `end_frame = total-1`, `temporal_interval = 1`. So the adapter sees each clip
> entire, at 30 fps, and the manifest's frame counts **are** the sequence lengths. Over the 3432
> train clips of `~/wam-t041/cosmos-g1-embodiment/manifest.json`, all 30.0 fps:
>
> | | min | p05 | p25 | median | p75 | p95 | max |
> |---|---:|---:|---:|---:|---:|---:|---:|
> | frames | 249 | 356 | 464 | 693.5 | 911 | 1372 | 1819 |
> | seconds @ 30 fps | 8.3 | 11.9 | 15.5 | 23.1 | 30.4 | 45.7 | 60.6 |
>
> The 30 held-out val clips run 329 → 1661 frames, median 759.5 (25.3 s). **No clip in either split
> is at or below 189 frames.** 189 was not near the edge of the training distribution; it was
> outside it.
>
> **The cap is not a cap, and the note above was wrong to call it one.** `MAX_NUM_FRAMES["256"] =
> 400` (`args.py:146`) is only ever compared inside a `log.warning` — "outside the recommended range
> […] Quality may be degraded" (`args.py:529-532`). Nothing clamps `num_frames`; the only rewrite is
> the 4N+1 round-up at `args.py:536-538` (and the `SMOKE` clamp at 533-534). So 397 is chosen to stay
> **inside a range NVIDIA states and we have no measurement against**, not because 401 or 501 would
> be rejected. Registering a value past a vendor's own recommended range, on no evidence, is the
> mirror image of the error this section has twice corrected.
>
> **Why 397 and not a shorter legal 4N+1.** Of the candidates, only 397 is in the *interior* of the
> training distribution: **12.97 %** of train clips are ≤ 397, against **4.22 %** at 349, **0.20 %**
> at 297, and **0.03 % — one clip out of 3432 —** at 249. "249 exactly matches the shortest training
> clip" is an argument for the boundary, and a boundary is what the open note objected to; matching
> it would register the single most extreme sample as the eval's whole duration. The prompt points
> the same way: the eval prompt **is** the structured-JSON caption (§5, `make_t041_eval_prompts.py`),
> and its `media` dict carries the source clip's duration — a median of 25.3 s. 397 frames is 13.2 s,
> the closest this API can come to the text being generated from; 189 is 6.3 s, a quarter of it.
> Longer is also simply a stronger test: an adapter that holds a Dex3 hand together for 13 s is
> better evidenced than one that holds it for 6.
>
> **The cost estimate, and it is a back-of-envelope.** Per clip the sampler carries
> `1 + (F-1)/4` latent frames at `(320/32)·(256/32) = 80` tokens each — spatial compression 16,
> `patch_spatial = 2` (`batchers.py:205, 264`): **3840 tokens at 189, 8000 at 397, a factor of 2.08.**
> The anchor is `inference_benchmarks.md` (Cosmos3-Super, t2v, H200 141 GB HBM3, 256p, 189 frames,
> the same 35 steps as the cookbook payload we copied). **Its 8-GPU column does not apply to us:**
> `parallelism_preset = "throughput"` forces `cp = cfgp = 1` (`args.py:1364-1378`) while
> `dp_shard = world_size` (`args.py:1354-1360`), and `95_eval_t041_embodiment.sbatch:137-140` hands
> `torchrun` **one payload per launch** — so the eight ranks shard parameters and duplicate the
> single sample, which is the single-GPU latency column, not the `/8` one. That column is empty for
> PyTorch at 256p; scaling the filled H200 141 GB cells (PyTorch/vLLM-Omni = 1.06–1.18 at 480p/720p)
> off vLLM-Omni's 25.61 s, and cross-checking against B200's PyTorch 14.71 s, gives **≈ 26–31 s** for
> a 189-frame clip at 320×192. Ours is 320×256, ×1.333 tokens → **≈ 40 s**; at 397 frames, ×2.08 →
> **≈ 85 s**. Over 60 clips: **≈ 40 min at 189 against ≈ 85 min at 397, a marginal cost of ~45 min
> ≈ 6 GPU-h.**
>
> **What actually threatens the 4 h wall is not this setting.** The generation loop launches a cold
> `torchrun` per clip — 60 of them, each importing torch, building an 8-rank NCCL world and loading
> a 64.6 GB DCP checkpoint. **No measurement of that startup exists in this repo** (94's probe is
> built to separate it by subtraction and has not run); at a plausible 1.5–3 min each it is
> **90–180 min, and it does not move with `num_frames`.** Central estimate for the whole job:
> ~3.6 h at 189, ~4.4 h at 397, against `--time=04:00:00`. Both are uncomfortable; the difference
> between them is not what makes them so.
>
> **Do not read the estimate as precise. What would falsify it:** if per-launch startup exceeds
> ~3 min, the job does not fit at *any* frame count and the eval must be split or restructured; if
> the benchmark's 256p rows were taken at a step count other than 35, everything scales by
> `steps/35`; if that 256p is 4:3 rather than the 16:9 the Diffusers note states, drop the 1.333; if
> `throughput` does give sample-level speed-up on a single payload — it does not, by
> `args.py:1364-1378` — every number here is ~3× too high; and if attention is not negligible at
> 8000 tokens, 397 costs more than 2.08×.
>
> **Why the walltime risk is accepted rather than bought off with a shorter clip.** The job is
> restart-safe by construction and was written that way: generation skips any clip already on disk
> (`95:139`), the judge step takes `--resume`, and the job is `--requeue` with
> `--open-mode=append`. An overrun costs a resubmission, not a verdict. Against that, the mitigations
> in order of preference, **none of them applied here**: (1) **split the two arms across two
> submissions** — zero code change, the skip-existing loop already makes the second pass a
> continuation; (2) **batch the payloads into one `torchrun`** — `scripts/inference.py:22-27` takes
> `-i` as a list and accepts globs, and the `throughput` preset with `dp_shard = 8` exists precisely
> so eight ranks hold eight *different* samples (the `align_num_steps` all-reduce at
> `inference.py:1648-1676` is there for that case), which would collapse 60 launches to 2 and buy
> real 8-way throughput; (3) `NPROC=4` (`95:69`) — under `throughput` with one payload the extra
> ranks buy parameter memory, not latency, so this should roughly halve the GPU-h at similar wall
> time, but it is untested at this model size (§4c). **(2) is the highest-leverage and is a change to
> how clips are produced; it is not made under cover of a frame-count decision, for the same reason
> the fps item was not changed under cover of the resolution one.** Lowering `num_steps` is rejected
> outright: it is a registered sampler setting held identical across arms, and trading sampler
> quality for walltime is its own decision on its own evidence.
>
> **One thing this estimate exposes that is NOT resolved here.** §7 budgets job `95` at **8 GPU-h**,
> which on 8 GPUs is one hour of walltime, against an sbatch that requests four. Every branch of the
> model above puts the eval at 25–35 GPU-h **at 189 frames as much as at 397** — the discrepancy is
> pre-existing, is not created by this amendment, and moving §7's line is not something a frame-count
> decision gets to do quietly. Recorded as open.

> **Amended 2026-08-12 — the base arm reads the HF download instead of the DCP conversion. Same
> bytes, same revision, and it is the arm that could not load at all.** Written after job `187078`
> failed and **before its replacement is submitted**; no clip has been generated by either arm, and
> `T041_RULE_V1` has still never run. §5's table is unchanged: the base arm is still
> "`Cosmos3-Super` at the pinned revision, no adapter". This records *which copy of those bytes* the
> job opens.
>
> **What happened.** `187078` died 1:42 in, on the base arm, before the first denoising step:
>
> ```
> ValueError: Could not infer experiment from checkpoint path:
>   …/checkpoints/Cosmos3-Super/model
> ```
>
> `91_stage` downloads `nvidia/Cosmos3-Super@e0262be9…` to `checkpoints/Cosmos3-Super-hf` and then
> converts it to a DCP tree at `checkpoints/Cosmos3-Super`, because job `94` trains from DCP.
> `BASE_CHECKPOINT_PATH` pointed at the conversion, and inference cannot read it. A DCP checkpoint
> leaves `config_file` at the default `.py` module (`inference/common/args.py:598-606`), and a
> module config needs a Hydra experiment name, which the framework infers by matching
> `/<experiment>/checkpoints/iter_<N>/` against the checkpoint path. A tree named for its model
> matches nothing. Naming the experiment explicitly does not help: the one the regex wants is
> `cosmos3_ga_64bm32b_v3_midtrain` — recorded in `inference/configs/model/Cosmos3-Super.yaml`, whose
> own `_metadata.args.config_file` points into NVIDIA's internal `cosmos3/_src/vfm/…` layout — and it
> is not registered in the OSS Hydra store. It would only move the failure.
>
> **The fix is one variable, and it was verified before resubmission rather than by resubmitting.**
> Job `187248`, two CPUs on the free QoS, no GPU, resolved the arguments for all three candidate
> directories and stopped short of instantiating the model — the same call that threw:
>
> | | `CheckpointType` | `config_file` | `config_file_type` | `experiment` |
> |---|---|---|---|---|
> | `Cosmos3-Super` (DCP, used by `187078`) | **raises** | — | — | — |
> | `Cosmos3-Super-hf` (now the base arm) | `hf` | its own `config.json` | `json` | not consulted |
> | the exported adapter (lora arm) | `hf` | its own `config.json` | `json` | not consulted |
>
> **This makes the arms more comparable, not less.** They now enter through one code path instead of
> two. The DCP path additionally honours `use_ema_weights` (default `True` →
> `load_ema_to_reg`, `inference/inference.py:1193`) and the HF path has no such switch, so the
> original pairing put a possible EMA-weighted base against a definitely non-EMA fine-tune. Whether
> the staged DCP actually carries `net_ema.` tensors was never established — its size says probably
> not — and with both arms on HF the question no longer arises.
>
> **The two arms are the same model plus 36 MB.** From the shard indices: base `129,230,007,264`
> bytes, lora `129,266,082,656`, both entirely BF16, difference `36,075,392` — the adapter, which the
> export left unmerged (`lora_` tensors present in the lora index, absent from the base). 64.6 **B**
> parameters, 120.4 GiB, which also corrects §3's table: it reads "≈64.6 GB params bf16" where the
> download is 134.6 GB, and the right reading is 64.6 billion parameters at two bytes each.
>
> **Not a second attempt at anything §6 forbids.** §6 forbids a second *training* run, and none has
> been started: the adapter under test is still `iter_000000500` from job `186968`. `187078`
> generated no clip, scored nothing, and produced no verdict — it never reached the model. Its cost
> was 1:42 on 8 GPUs, **0.23 GPU-h**, which is recorded against §7's ceiling with the rest.

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

> **Amended 2026-08-10 — the probe ran, it passed, and the number it produced condemned job `95`
> as written.** The gate is discharged, not renegotiated: job `186663`, 28 minutes wall on 8 GPUs
> (**3.7 GPU-h** against the 8 budgeted above), `PROBE.json` = `seconds_per_iter 21.85`,
> `load_seconds 494.8`, `passes_needed 1`, `estimated_gpu_hours 25.4` against `ceiling 96`. One
> pass, 3.03 h of iteration plus one 8.2-minute load inside a 4 h wall, at **26 %** of the training
> ceiling. Nothing in the table is raised and the `MAX_PASSES = 3` enforcement is untouched.
>
> **`load_seconds` is the finding, not `seconds_per_iter`.** 494.8 s is the cold constant — imports,
> an 8-rank NCCL world, a 64.6 GB DCP read — before the first denoising step, and it is paid per
> `torchrun`, not per clip. Job `95` launched one `torchrun` per payload: 60 × 494.8 s = **8.25 h of
> checkpoint loading alone**, inside the same 4 h wall, generation still to pay. That job could not
> have finished, and requeueing does not repair a per-pass cost dominated by a constant repeated
> sixty times — each restart banks only what the last one completed, so it would have taken ~10
> resubmissions to grind through well under an hour of real sampling. Batched to one launch per arm
> it is 2 loads, and step 2 falls from ~9.7 h to ~30 min.
>
> **This closes the item §5's third amendment left open, but not at 8 GPU-h.** That note recorded
> the eval at 25–35 GPU-h against §7's registered 8 and refused to resolve it; the honest figure now
> is **~1.4 h × 8 GPUs ≈ 11 GPU-h** — still above the line, and recorded as still above it rather
> than rounded onto it. The dominant remaining waste is not generation: the judge holds all 8 GPUs
> while vLLM serves on `--tensor-parallel-size 1`. That is a real 8-GPU-hour-class inefficiency, it
> is **not** fixed here, and it is what stands between ~11 GPU-h and the registered 8.
>
> **Revised total, measured where measurable:** `91` + `93` as budgeted, probe **3.7**, train
> **≤ 25.4**, eval **≈ 11** — comfortably inside the **122 GPU-h** ceiling, which is unchanged.

> **Amended 2026-08-12 — `seconds_per_iter` was wrong by 52 %. The gate's decision was not.**
> Recorded because the number was wrong, not because the gate was: on the corrected arithmetic the
> probe still passes, on the same pre-registered criterion, and nothing in §7 moves.
>
> The measurement was internally exact and externally false. Job `186663` timed two `torchrun`s at
> 604 s (N=5) and 1041 s (N=25); `(1041−604)/20 = 21.85`, `604 − 5×21.85 = 494.8`. Its own log
> refutes both figures: **iterations 2–25 of the 25-iteration run averaged 33.25 s** (min 27.97,
> max 38.72), and the production run later held the same rate. 21.85 s was never a per-iteration
> time, and 494.8 s was never a checkpoint read.
>
> **The error is structural, not noise.** `t = load + n·step` assumes both runs pay identical
> one-off costs. They could not: they ran back-to-back on one node against the same 64.6 GB DCP
> checkpoint, so the second read a warm page cache.
>
> | one-off cost | run 1 (N=5, cold) | run 2 (N=25, warm) |
> |---|---|---|
> | start → first iteration begins | 124.7 s | 34.0 s |
> | first iteration (compile + warmup) | 198.3 s | 144.0 s |
> | last iteration → process exit | 158 s | 65 s |
> | **total one-off** | **481.0 s** | **243.0 s** |
>
> Run 2 was **238 s cheaper in costs that have nothing to do with its 20 extra iterations**, and the
> subtraction charged every second of that saving against those 20: `−238/20 = −11.9 s/iter`, which
> is the entire gap between 33.25 and 21.85. Both wall times close to the second —
> `124.7 + 198.3 + 122.8 + 158 = 603.7 ≈ 604` and `34.0 + 144.0 + 798.0 + 65 = 1041.0 = 1041`. The
> residual was pushed into `load`, which is why 494.8 s exceeds the true 124.7 s cold setup: it
> absorbed the first-iteration compile and the teardown as well.
>
> **What the gate should have printed.** `500 × 33.25 s = 4.62 h`; cold overhead ≈ 448 s; usable
> ≈ 13 952 s per 4 h pass ⇒ **`passes_needed = 2`, `estimated_gpu_hours ≈ 64`** against
> `max_passes 3` and `ceiling 96`. **PASS.** The 2026-08-10 amendment's `25.4` is superseded by
> **≈ 64**; `passes_needed 1` by **2**.
>
> **The argument that condemned job `95` survives, at a smaller magnitude.** It needed only that a
> per-`torchrun` constant is paid 60 times, and the constant is real — 330–480 s per launch (cold
> setup + first iteration + teardown), not 494.8 s. Sixty launches is 5.5–8 h against a 4 h wall
> instead of 8.25 h. Still impossible, still one launch per arm.
>
> **Actual spend, closing the estimate:** probe 3.7 + crashed pass 2.3 + pass 2 32 + pass 3 16 =
> **54.0 GPU-h**, under the corrected 64 for training alone. Waste recorded rather than smoothed:
> pass 2 reached iteration **396** but banked iteration **300** (`save_iter = 100`), discarding 96
> iterations ≈ 55 min × 8 ≈ **7 GPU-h**. That discard is also what produced the coverage limitation
> recorded in §9.

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

> **Amended 2026-08-12 — 500 optimiser steps, but at most 300 distinct batches.** Written before
> the eval job (`187078`) started, so it cannot be read as an excuse fitted to a result: at the
> time of writing no clip has been generated and no verdict exists.
>
> On the pass 2 → pass 3 resume the framework asked `iter_000000300` for the `dataloader` key and
> **all 8 ranks reported it absent** — `Checkpoint …/iter_000000300/dataloader does not exist, skip
> loading dataloader.` Model, optimiser, scheduler and trainer state restored; sampler position did
> not. The 32 dataloader worker seeds logged in pass 3 are **set-identical** to pass 2's. A sampler
> restarted from position 0, under the same seeds, over the same corpus, draws the same order — so
> iterations **301–500 re-drew the sample order of iterations 1–200**.
>
> The run is therefore 500 gradient steps over **300 distinct packed batches**
> (`tokens_after_packing = 45056`), the first 200 of which contributed twice. What fraction of the
> 3133 train clips those 300 batches touch is **not stated**, because the log records no
> per-iteration sample identity — no clip uuids, no mp4 paths — and the number is not recoverable
> after the fact. The bound is 300 batches, not 500.
>
> **This is an inference, and its weakest link is named.** Identical seeds plus absent sampler state
> is strong evidence of a replay, not a direct observation of one. Direct verification would have
> needed per-sample logging that this recipe does not emit.
>
> **How it bears on the verdict, in the direction that matters.** Under-training biases toward
> **N**, not toward **P** — seeing less of the corpus is not a route to a false positive. So a **P**
> stands, with the effect size read as a **floor** rather than an estimate. An **N** is the reading
> that is compromised: it would be partly a statement about a run that saw less than 500 steps
> implies, and is *not* clean evidence about what Cosmos3-Super can reach on this corpus.
>
> **Not repaired.** Repair means checkpointing the dataloader and re-running, and a re-run is a
> second attempt that §6 does not permit — the same constraint that makes **I** terminal. Recorded
> as a limitation of this run, which is the only honest option left once the run has happened.
