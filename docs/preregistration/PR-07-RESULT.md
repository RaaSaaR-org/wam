# PR-07 result — **VOID (labels)**: the corpus's own action column cannot clear our bar

Ran 2026-08-15/16 on Discoverer+ (`ehpc-aif-2026pg01-905`, one H200). Pre-registration:
`PR-07-positive-control.md`, rule `T39_RULE_V1` in `cluster/discoverer/71_eval_t39_control.sbatch`.
Task **T-39**. Trained checkpoint `runs/t39-baseline-seed0/checkpoints/checkpoint-10000`,
`config_hash c290ae9b06…`, `dataset_snapshot_ref sha256:598f193f…`, base
`nvidia/GR00T-N1.7-3B` @ revision `2fc962b9`. Spent **1.37 of the 12 GPU-h ceiling**.

**The verdict is decided by G0 and no policy number can change it.** `T39_RULE_V1` evaluates
`not L1_action` before any policy branch, exactly as PR-07 §5 registered it, and that branch is
taken. Per PR-07 §6 this licenses **a defect report against the label pipeline** and **forbids any
statement about GR00T**. There is none below.

## The two oracle arms — 40 holdout episodes, 1 040 chunks, both bench specs

| arm | `mse` | `skill_vs_zero_pct` | `skill_vs_repeat_pct` (**L1**) | `ci_skill_vs_repeat_pct` (**L2**) | level |
|---|---|---|---|---|---|
| `oracle_state` — future **executed states** | `0.0` | **+100.00** | **+100.00** | **+100.00** | L4, 100.0/100 |
| `oracle_action` — native **`action` column** | `4.1979e-05` | **−157.11** | **−359.41** | **−102.54** | none — below L0, 20.0/100 |

Baselines, shared by both arms: `repeat_mse 9.1377e-06`, `zero_mse 1.6328e-05`,
`repeat_ci_mse 2.1521e-05`, `zero_ci_mse 4.8971e-05`.

### G0a passed perfectly, and that is what makes G0b a measurement

`oracle_state` scores a bit-exact `mse 0.0` and 100 % on every rung. The adapter, the joint
ordering, the delta anchoring and the gripper synergy are provably correct — this arm is the
identity of our own label pipeline and PR-07 §4 pre-registered that anything but ~perfect is *our*
bug. It is not our bug. **G0b therefore cannot be dismissed as broken plumbing**, which is the only
reason it is reportable at all.

### G0b fails L1 by 359 percentage points

The dataset's own commanded `action` column — the exact target GR00T is trained to predict — is
**4.59× worse than repeating the last action** and 2.57× worse than holding still, under our
scorer, on our chunks. Since that column is the ceiling for any policy trained on it, no such
policy can clear L1 either.

PR-07 §4 wrote this outcome down in advance, unprompted by any number:

> if the ground-truth action column itself cannot clear L1 under our scorer, then **no policy
> trained on this dataset can clear our bar, and the finding is about our label pipeline, not
> about GR00T.** […] It is pre-registered here precisely because it is the outcome I would
> otherwise be tempted to treat as a bug and quietly patch.

It is not patched. It is the result.

## The mechanism, from the two rungs nobody was watching

Absolute agreement is excellent and per-step agreement is catastrophic, which is only a
contradiction until you look at where the error sits.

| | value | gate | reading |
|---|---|---|---|
| `mse` (absolute) | `4.20e-05` | — | the command sits almost exactly on the state |
| `horizon_ratio` | **0.0044** | ≤ 4 | last-step error is **1/227th** of first-step error |
| `smoothness_ratio` | **8.52** | ≤ 2 | the command is **8.5× jerkier** than the demonstration |

Our labels are relabeled from **executed state** over `t → t+1`. GR00T's target is the
**commanded** value at `t`. Under tight position-control tracking those two are numerically
adjacent — hence the tiny absolute MSE — but their *first differences* are not the same signal at
all: essentially all of the disagreement is concentrated in the chunk's first step, and the
command carries 8.5× the jerk of the trajectory that was actually executed.

**A second, independent sighting of the same split, already in the repo before this ran.**
`docs/benchmark.md`'s gripper section records that our relabeled gripper channel is degenerate —
peak-to-peak 0.120, **0.00** debounced transitions per episode — while the raw commanded
`action.left_hand.max_joint[0]` in the same snapshot carries **2.04** debounced transitions per
episode and a complete open-close cycle in **99.8 %** of episodes. The command space contains a
grasp; our label space does not. That was written down as a gripper-audit observation and is here
re-read as what it also was: the same command-versus-state mismatch, visible on the one channel
where it is unmistakable.

**Stated as measurement, not as mechanism:** the three numbers above are measured. The reading
that the command leads the executed state by roughly one control step, with the remainder being
high-frequency command jitter that the arm's own dynamics filter out, is an *interpretation*
consistent with them and is not established here. Distinguishing the two requires a delay sweep
over the anchoring convention, which is follow-up work and is not this document.

## What this bounds

Every number in `docs/benchmark.md` was produced against labels from this same relabeling of
executed state. `oracle_state` proves the relabeling is self-consistent; `oracle_action` proves it
does **not** agree with the corpus's own commanded action space. So the benchmark's numbers remain
valid statements about predicting *what the robot did*, and are **not** statements about predicting
*what it was told to do* — a distinction the ladder's thresholds were never calibrated against.
Nothing in the fourteen recorded negatives is withdrawn, and nothing in them is exonerated either:
this measures the ruler, not the arms.

**T-39's own question remains unanswered.** "Does the bar move for anyone on this corpus" cannot be
asked through a label space the corpus's own ground truth fails. PR-07 §7's conditional second
candidate (`lerobot/pi05_base`) is pre-registered **only on outcome N** and is therefore *not*
licensed by VOID — swapping models here would be testing a second policy against the same broken
ruler.

## Receipts

| job | what | result |
|---|---|---|
| 187799 | dry run, free QoS | FAILED 0:19 — tyro rejects the enum *value*, wants the *name* |
| 187802 | train | FAILED 2:39 — `huggingface_hub.model_info()` under `HF_HUB_OFFLINE=1` |
| **187804** | **train** | **COMPLETED 1:22:14** — 10 000 steps, 1.00 epoch, loss 1.212 → **0.0216** |
| 187812 | shim probe, free CPU QoS | FAILED 1:10 — see below, the probe was impossible by construction |
| 187813 | eval, 4 arms | FAILED 1:48 after both oracle arms reported — missing `GROOT_PATCH_MISTRAL` |

Training converged cleanly: 1 000 logged steps, monotone to `loss 0.0216`, no restart consumed of
`MAX_RESTARTS=2`. **The policy fit the data it was given.** That is a statement about optimisation,
not about the verdict, and it is recorded here only so nobody later reads the VOID as a failed run.

### Three defects found by running it, recorded rather than quietly patched

1. **`71_eval_t39_control.sbatch` never exported `GROOT_PATCH_MISTRAL=1`**, which
   `70_train_t39_baseline.sbatch` does. Both load the processor through
   `AutoProcessor.from_pretrained`, and transformers' `_patch_mistral_regex` calls
   `huggingface_hub.model_info()` on the model *name* while building the tokenizer — which raises
   offline. It killed 187802 at 158 s and then killed 187813 at 108 s, in both cases *after* the
   two oracle arms had already written their artifacts, which is why the G0 numbers above are
   sound and only the policy arm is missing. Fixed; **no threshold moved** — it is an environment
   variable, not a constant of `T39_RULE_V1`, and the rule block is untouched.
2. **A CPU probe of this checkpoint is structurally impossible**, not merely unimplemented. The
   config pins `attn_implementation="flash_attention_2"`, and transformers refuses to construct
   the module without CUDA — raised inside `Qwen3VLForConditionalGeneration.__init__`, before a
   line of our code runs. `74_probe_t39_policy_shim.sbatch` is now a 20-minute GPU job and its
   header no longer claims otherwise. A gate that always fails for a reason unrelated to what it
   tests is worse than no gate.
3. **`run_metadata.json` recorded `git_commit = unknown`.** `checkpoint_ref`, `config_hash` and
   `dataset_snapshot_ref` are all present, so AC-04's chain holds, but the repo revision does not
   travel with the artifact and should.

### And one that is not fixed

`70_train_t39_baseline.sbatch` passes no `--report_to` and no logging config, so the training log
carries only a tqdm bar and **the loss curve appears nowhere in stdout**. It was recovered after
the fact from `checkpoint-N/trainer_state.json`'s `log_history`. The run was not restarted for it.

## What this cannot answer

Everything PR-07 §9 already excludes, plus: the policy arm did not run, so this document contains
no measurement of GR00T N1.7 on anything, and none may be inferred from the training loss. The
gripper channel remains degenerate — peak-to-peak `0.1196`, `0.00` debounced transitions per
episode, 85.3 % majority class — so `gripper_accuracy` was **withheld by the scorer** on both arms
and nothing here can see a grasp. `scripts/audit_gripper.py` runs before any grasping claim.
