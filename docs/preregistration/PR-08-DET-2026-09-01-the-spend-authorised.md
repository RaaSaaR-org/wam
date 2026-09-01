# PR-08 DET — the spend authorised: the ceiling signed, and one clip released against it

**SIGNED 2026-09-01 by the project owner.** This is the authorisation that
`PR-08-RESULT-2026-08-28-the-ceiling-derived-and-the-number-the-gate-actually-accepts.md` §5 said was
owed and did not take: *"A ceiling caps what a run may reserve; whether the run happens is a separate
decision and it is the project owner's."*

**It authorises a number and releases a first tranche. It does not lift `T40_RULE_V1` §1 and it does
not close §8 item 4.** Nothing on this page permits a clip on its own: §8 is conjunctive, and at
signing item 4 is open and can still refuse.

---

## 1. The determination

```
determination:  (a) PARTITION_CEILING_GPU_H = 2013.75 is AUTHORISED as a hard ceiling
                    over the whole partition -- all 25 style-instances, 10 050 clips,
                    train + eval + identity. The shares stand as derived: 805.50 /
                    402.75 / 805.50, summing within the whole.

                (b) RELEASED AGAINST IT, FOR NOW: ONE CLIP. A single generated episode,
                    video plus its action column, so the first artifact can be looked at
                    before the rest of the ceiling is drawn against.

                (c) The remainder of the ceiling is authorised as a CAP, not as a
                    release. Generating beyond the first clip is a separate go, and it
                    is the owner's.

decided by:     the project owner, 2026-09-01, by the instruction

                    "klingt gut, lass das so machen!"

                given in direct answer to "was würdest du empfehlen?", after the three
                available shapes had been put in front of them in writing: sign the
                ceiling and release one clip first; sign the ceiling and authorise the
                full run; or authorise one clip while leaving the ceiling unsigned --
                which was named as a practical blockade, because §8 is conjunctive and
                item 3 would stay open. The owner chose among named alternatives; the
                recommendation was the session's.

                The instruction that opened the sequence, verbatim:

                    "was, lass nun alles lösen, damit wir den datensatz (oder zumindest
                     mal 1 video + action)."

                PREPARED BY a Claude Code session. T40_RULE_V13 §5 permits a session to
                prepare the rationale and name the edges; it may not sign this, and the
                signature above is the owner's instruction recorded verbatim.

date:           2026-09-01
```

## 2. What the number is, and what it is not

The derivation is not restated — it is
`PR-08-RESULT-2026-08-28-the-ceiling-derived-…` §3, and it is arithmetic over committed artifacts
that a session was entitled to perform (§1 of that document: a word-boundary search of
`T40_RULE_V2` §3 for `sign`, `signature` and `owner` returns nothing).

**A ceiling is not a bill.** The sbatch's layer-5 ledger reserves each pass's worst case
`NPROC × WALL_H` *before* the pass runs, because Slurm bills the whole allocation for the whole wall.
So what the authorised figure bounds is what may be **reserved**, and the projection the gate checks
against it is 2013.73 against 2013.75.

**What the run is expected to actually draw is between 96 and 359 GPU-h** — 85.2 for generating the
17 survivors across 25 instances, plus 10.9–273.5 for the preflight over the other 385 episodes that
never reach the generator. **The ceiling therefore authorises roughly six to twenty-one times what
the run will draw**, and that slack is a property of the formula `T40_RULE_V2` §3 fixes, not a margin
anybody chose.

**The tighter alternative was considered and passed over, and not because it would be worse.** A
yield-aware ceiling in the ~100–360 range would require changing the projection formula at
`97:2225`, which `T40_RULE_V2` §3 fixes — so a new rule version — and an honest yield-aware formula
needs two rates rather than one rate scaled by a yield fraction. Against a run already bounded by
`MAX_PASSES` and by the persisted ledger, that is a larger change than the tightening buys. Recorded
so the slack is understood as a known cost rather than an oversight.

**Every figure above descends from a throughput measured on one episode, and `T40_RULE_V20` §5
requires the provenance be carried:** *measured on an episode selected for surviving G0c*, 1.6896
s/frame on 1 × H200 at 640×480, **and the direction of that selection's bias is NOT KNOWN.**

## 3. How the authorisation is exercised

`PARTITION_CEILING_GPU_H` and `CEILING_GPU_H` are required with no default at
`97_transfer25_restyle.sbatch:434-435`, deliberately — *"a default would be a budget line nobody
measured."* **That stays true and this determination does not write a default into the sbatch.** The
authorised figure is passed explicitly on each submission, and the same
`PARTITION_CEILING_GPU_H=2013.75` in all three `STYLE_SET` submissions of a `RUN_ID`, with the share
of that set as `CEILING_GPU_H`.

Turning a required-with-no-default variable into a default would change what an operator may omit.
That is not authorised here.

## 4. What this does not do

* **It does not license a clip.** `T40_RULE_V1` §1 is unchanged. At signing, §8 item 4 is open: the
  `pr08-geom-tol-v2` merge has not been run and `configs/transfer25/pr08_geom_tol.json` holds
  `geom_tol_px = null`. If the margin comes out `≤ 0`, PR-08 §6 governs and no clip is generated —
  the release in §1(b) simply goes unused.
* **It does not commit the number to `configs/`.** No tracked artifact holds a ceiling today, and
  inventing a home for it is a design decision rather than a derivation. It lives here and on the
  command line.
* **It does not claim the allocation can afford it.** The `sreport` figure readable from the login
  node is CPU hours; the GPU-h grant is not readable from there. This authorises a ceiling against
  the project's own rules, not against a verified balance.
* **It does not authorise the full run.** §1(c). The first clip is released; the rest is a cap.

## 5. Provenance

| | |
|---|---|
| determination | `PR-08-DET-2026-09-01` — the spend |
| status | **SIGNED 2026-09-01.** In force |
| decided by | the project owner, 2026-09-01, on the instruction quoted verbatim in §1; prepared by a Claude Code session, which `T40_RULE_V13` §5 permits |
| discharges | §8 item 3's remaining half — the authorisation. The derivation was discharged 2026-08-28 |
| authorises | `PARTITION_CEILING_GPU_H = 2013.75`, shares 805.50 / 402.75 / 805.50 |
| releases | **one clip** — one generated episode, video and action column |
| amends | nothing. `T40_RULE_V2` §3's formula, `97:2225`, and the required-no-default variables are all unchanged |
| generation licensed | **only the single clip of §1(b), and only once §8 closes in full** |
| training licensed | **no** |
