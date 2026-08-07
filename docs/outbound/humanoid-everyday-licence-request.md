# Outbound: Humanoid Everyday dataset licence request

**Status:** drafted 2026-08-06, **not sent — deferred 2026-08-07 by the user, not cancelled.**
Sending is the user's call; this is a public post under our name to a third party's repository.

**What changed.** The user decided to train on the corpus without waiting for an answer — recorded
as **OD-09** in `TASKS.md`, with what was accepted and what it does not cover. That makes this
document lower priority, **not obsolete**: OD-09's stated review trigger is distribution or sale of
anything trained on the corpus, and a licence on the repo is what would clear that trigger. Send it
before that point, not after.

**Why this exists.** `USC-PSI-Lab/Humanoid-Everyday-G1` is the corpus [[T-041]] wants for a
generator fine-tune and [[T-040]] wants for ground-truth depth, and it **states no licence in any
field, tag, card or file**. The evidence table is in
`.mc/tasks/todo/T-041-cosmos-generator-finetune-on-g1.md` §"Licence — unresolved". Unlicensed data
is not permissively licensed data, so until a `LICENSE` exists on that repo, no weight is trained
on it. This is the unblock.

**Where to post.** Both, same text — they are different audiences and either may answer first:

1. GitHub issue → `https://github.com/physical-superintelligence-lab/Humanoid-Everyday/issues`
   (issues are enabled, 0 open as of 2026-08-06)
2. HF discussion → `https://huggingface.co/datasets/USC-PSI-Lab/Humanoid-Everyday-G1/discussions`

**Tone note.** The three asks are specific and answerable. Do not ask "what is the licence" — the
authors would reasonably reply "MIT, see the README", which is the sentence that does not resolve
the conflict. Ask for the artifact.

---

## Title

`Licence for the released datasets — LICENSE file missing on Humanoid-Everyday-G1`

## Body

Hi — thank you for releasing Humanoid Everyday. We are evaluating it as a training corpus for a
G1 + Dex3 world-model project and we ran into a licensing question we cannot resolve from the
public material. We would rather ask than assume.

Three statements exist and they do not agree:

- This GitHub repository's README ends with `# License` / "This dataset is released under the MIT
  License". There is no `LICENSE` file in the repository, and the GitHub licence API returns 404.
- The Hugging Face repo `USC-PSI-Lab/humanoid-everyday` (the mixed-embodiment one) carries
  `license: apache-2.0` in its card metadata.
- The Hugging Face repo **`USC-PSI-Lab/Humanoid-Everyday-G1`** — the one we would actually use —
  carries **no licence at all**: no `license` field, no `cardData`, no licence tag, no README and
  no LICENSE file among its 8 133 files. The same is true of `Humanoid-Everyday-H1`.

The project page and the arXiv paper state no data terms either (the paper's CC BY 4.0 is arXiv's
licence on the manuscript).

Since the two permissive names sit on a *code* repo and on a *different* dataset repo, we do not
think we can carry either across to `Humanoid-Everyday-G1`. Concretely, we would be grateful if
you could:

1. **Add a `LICENSE` file — or a `license:` field in the dataset card — to
   `USC-PSI-Lab/Humanoid-Everyday-G1` itself** (and ideally `-H1`), so the corpus carries its own
   terms.
2. **Confirm which licence governs the data**, given that the GitHub README says MIT and the mixed
   repo's card says Apache-2.0.
3. **Confirm whether the Unitree teleoperation code lineage bears on the recordings.** We noticed
   that the anonymous review mirror carries an Apache-2.0 `LICENSE` under
   `Copyright [2024] [HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")]`, which reads like
   the inherited licence of the teleop *code* rather than of the recordings — but we would rather
   have that confirmed than inferred.

Happy to help with a PR to the dataset card if that is useful. Thanks again for the release — the
egocentric depth alongside RGB on a G1 is exactly the ground truth we have been missing.

---

## What each answer unblocks

| answer | consequence |
|---|---|
| a permissive licence lands on `Humanoid-Everyday-G1` | [[T-041]]'s corpus half unblocks; [[T-040]]'s depth-estimator calibration gains its real ground truth |
| Apache-2.0 or MIT confirmed but no file added | **still blocked.** Record the reply, ask once for the file, and do not train on a forum sentence |
| a restrictive or research-only licence | T-041 loses the HE half and runs on AppleToPlate alone; T-040's calibration falls back to the Isaac route (PR-08 §4) |
| no reply within ~4 weeks | treat as unlicensed. Both tasks proceed without HE, and that is recorded as a scope cut, not a silent omission |

**Note the fallback is already designed.** PR-08 calibrates the monocular depth estimator against
Isaac's rendered ground truth rather than HE's measured depth, precisely so this request is not on
the critical path. HE would *confirm* that calibration on real frames; it is not required to
produce it.
