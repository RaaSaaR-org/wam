---
id: T-39
aliases:
- T-39
title: "The positive control this project has never had"
slug: the-positive-control-this-project-has-never-had
status: todo
priority: 1
owner: ''
projects: []
customers: []
tags:
- m3
- data
- eval
- cluster
- prereg
- training
sprint: ''
depends_on: []
due_date: ''
created: 2026-08-06
updated: 2026-08-06
status_note: "Pre-registered 2026-08-06 (PR-07), drivers + 31 tests shipped. PR-07 §8 items 4, 5 and 6 are ALL CLOSED as of 2026-08-16, none of which needed the cluster. (1) The 2026-08-06 claim that they 'need SSH or a source document, no local work is left' was simply wrong — the source document was on this workstation before the gate was written. (2) third_party/isaac-gr00t is now vendored at the pinned 1a1837f + the registered PyAV patch, gitignored, with PROVENANCE.json; rebuild via scripts/build_t39_env_local.sh. (3) Item 4: ~/venvs/t39 built from upstream's own uv.lock — python 3.12.13, torch 2.9.0+cu128, flash-attn 2.8.3 VERIFIED RUNNING on sm_120, deepspeed, torchcodec decoding real corpus video. (4) Item 5: MODEL_ID = nvidia/GR00T-N1.7-3B read off the vendor tree in four places (embodiment_tags.py:204, qwen3_backbone.py:36, two test constants); 6.93 GB fetched and pinned to HF revision 2fc962b9, the same revision the July cluster run staged; config.json matches recipeA/B, confirming those are post-trains and unusable as a base. (5) Item 6: TRAINER_ENTRYPOINT = gr00t/experiment/launch_finetune.py, POLICY_ENTRYPOINT = gr00t.policy.Gr00tPolicy. (6) configs/groot/verify_new_embodiment_config.py now PASSES inside upstream's registry at the pin. Two of the three open questions resolved: the 8-GPU/batch-128 figure was recipeA's, not T-39's — 70_*.sbatch:144 pre-registers batch 32 on --num_gpus 1, matching upstream's own examples/finetune.sh, so single-GPU IS the unmodified recipe; venue is a recording requirement, not a threshold. (7) datasets/gr00t-apple-full CONVERTED locally 2026-08-16 — 402 episodes, 83 MB, legacy gripper mapping left at default so it reproduces the dataset runs/t16-lora-seed0's snapshot ref is pinned to. (8) THE PRE-REGISTERED DRY RUN (73_*.sbatch) IS GREEN LOCALLY: 362-episode subset materialised (362 parquet + 362 mp4), 40-episode holdout proven excluded, upstream stats.py wrote stats.json + relative_stats.json with left_arm/right_arm mean shape [16,7], the offline build_processor call that killed cluster job 187802 returns Qwen3VLProcessor, and run_metadata.json carries dataset_snapshot_ref sha256:6b8fe849, config_hash 3749547d, checkpoint_ref nvidia/GR00T-N1.7-3B and the ordered 362 train_episode_ids. Every non-GPU stage of T-39 is now exercised on this box. STILL OPEN, and it is the only thing left before a local run of record: whether the 5090's 32 GB holds the recipe — the probe was written and BLOCKED by the permission classifier, not skipped. BUG FOUND BY RUNNING IT: train_t39_baseline.py:292 joins vendor_root into stats_script and :316 runs it with cwd=vendor_root, so a relative --vendor-root is doubled and the run dies 'exited 2'; absolute paths work, and the sbatch already passes absolute paths, which is why it never surfaced on the cluster. Left unfixed — that file has concurrent uncommitted changes. Inventory and every citation: docs/local_gr00t_assets.md. Submission is still the user's call."
---

# The positive control this project has never had

## Description

**The positive control this project has never had**
(`docs/preregistration/PR-07-positive-control.md`, pre-registered 2026-08-06 before any weights are
downloaded or any job is submitted; `cluster/discoverer/70_train_t39_baseline.sbatch` +
`71_eval_t39_control.sbatch`, rule `T39_RULE_V1` in git first). **Fourteen recorded experiments,
fourteen negatives, and not one of them can be read.** T-15/T-24/T-26, T-18, T-16/T-29, T-30, T-36,
T-38 and PR-03 each compare a WAM variant against a *trivial* baseline; none compares anything
against a method **known to work**, so none can separate *our approach is wrong* from *nothing
clears this bar on this corpus, under this scorer*. **The corpus makes that omission specific rather
than generic:** every real number here comes from `nvidia/GR00T-N1.7-AppleToPlate`, which is
NVIDIA's own tutorial corpus for post-training GR00T N1.7 on a G1 apple-to-plate task — published to
demonstrate that a particular recipe works on exactly this data. We adopted the data and never ran
the recipe: `grep -rn 'gr00t_n1\|isaac-gr00t\|pi0\|pi05\|smolvla\|openvla' src/ scripts/ configs/
cluster/` returns **nothing** (2026-08-06). So "402 success-only episodes of one task is not enough"
— the standing explanation for all fourteen — has never been separated from "we have not run a
method that works on 402 episodes". **Held identical by sharing the artifact, not by copying a
value:** `configs/splits/i8_train_362.txt` (the file T-32's rung 362 trains on), the 40-episode
`t18_holdout_episodes.txt`, `build_eval_pairs`'s 1 040 chunks, `bench_metrics`/`e1_metrics` under
**both** bench specs, and `eval_t16.verify_split`'s disjointness proof with the committed witness.
**The trainer is NVIDIA's, vendored unmodified** — a positive control run through our
reimplementation of someone else's recipe is not a positive control, so our code appears in exactly
two places (the episode restriction and the eval adapter). **The adapter is the weakest part and
gets two vetoes that run first:** `oracle_state` pushes the holdout's future *executed states*
through the same mapping (the identity of our own label pipeline — anything but ~perfect is our
bug), and `oracle_action` pushes the dataset's native **`action` column** through it. The second is
the one worth the most: our labels are relabeled from executed *state* and GR00T predicts the
*commanded* action, so if the ground-truth action column itself cannot clear L1 under our scorer
then **no policy trained on this dataset can**, T-39 is VOID, and every number in
`docs/benchmark.md` is bounded by a label-space mismatch nobody had measured. **Gates invent
nothing:** the bar is WAM-Bench's own ladder (L1 `skill_vs_repeat_pct > 0`, L2
`ci_skill_vs_repeat_pct > 0` — `ci_` is the task-**critical** chunk subset, not a confidence
interval), and the one borrowed constant is `MATERIAL_FLOOR_PP = 10.0` from `I8_RULE_V3`, taken
rather than coined so that the choice of floor cannot be the finding. **Symmetry, because here the
NEGATIVE is the expensive conclusion** — verdict N licenses "stop trying methods on this corpus" —
so N needs the material margin *and* a second arm (`train40`, 40 episodes the policy did train on,
reusing the committed `i8_train_040.txt`; a diagnostic and an upper bound, never a headline).
Verdicts: **P** clears L1 → the fourteen negatives are statements about WAM, T-32 is descoped, adopt
and redirect; **N** ≤ −10 pp *and* fails on its own training data → the corpus/scorer is the
finding, T-32 is answered for free and the next move is PR-04's collection spec (the *kind* of
data), not another method; **M** fails holdout but clears train40 → fits and does not generalise, so
the bar is reachable and the 362/40 split is the live question; **I** anything else, including the
−10..0 band → one seed replicate and *nothing recorded*. **Cost ceiling 12 GPU-h** of 5 000 (3 × 4 h
walltime, enforced by `MAX_RESTARTS=2`), against T-32's ~109. **Exactly one alternative candidate is
pre-registered** (`lerobot/pi05_base`, only on outcome N, reported as attempt 2 of 2) so that "try
another model until one passes" is not available after the fact; there is no attempt 3. **T-32 does
not run until this reports** — it fits a scaling curve on a method no positive control has validated
here, and if the method is broken on this corpus every branch of `I8_RULE_V3` describes the scaling
of brokenness. **The implementation landed 2026-08-06, after the gate and before any cluster
contact** — `scripts/train_t39_baseline.py` (the episode restriction materialised as a symlinked
LeRobot subset **view** with filtered `meta/episodes.jsonl` and recomputed `info.json` totals,
because a trainer that trusts `info.json` over the directory listing would otherwise iterate 402
episodes over a directory holding 362; the vendored trainer is invoked as a **subprocess**, as
shipped, in its own venv), `scripts/eval_t39_baseline.py` (the adapter, the four arms, both bench
specs per arm — and the policy arm crosses into canonical units through the **same**
`commanded_to_chunk` the oracle uses, which is what makes `oracle_action` a ceiling rather than a
nearby separate number), and `tests/test_t39_baseline.py` — 31 tests, **12 mutants introduced and
killed**, including the three plausible mis-anchorings (`action[t+1] - q[t]`, `action[t] - q[t+1]`,
and the commanded column's own first difference). **Two mutants survived the first version of that
suite and are recorded rather than quietly fixed:** a gripper read one step late and the wrong hand
selected, both because the `oracle_action` test compared only `targets` — the joint delta spans `t
-> t+1` while the gripper is sampled *at* `t+1`, so the two channels are anchored differently by
construction and one convention cannot cover both. **One correction to `70_*.sbatch`, made while
implementing and recorded in PR-07 §8:** it now also passes `--wam-dataset`, because the trainer
eats the LeRobot *source* while `dataset_snapshot_ref` must be taken over the *converted* episodes —
as first written the witness could not have been verified at all. No threshold moved; the rule is
`71_*.sbatch`'s and is untouched. **Still not submittable:** PR-07 §8 items 4-6 — the
`virt_envs/t39` cluster env, a `MODEL_ID` verified from a primary source, and
`TRAINER_ENTRYPOINT`/`POLICY_ENTRYPOINT`, all three with **no defaults** for the same reason: we do
not know the vendored entrypoint path or its inference API from a primary source, and a plausible
guess would run something adjacent and record it as NVIDIA's recipe

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
