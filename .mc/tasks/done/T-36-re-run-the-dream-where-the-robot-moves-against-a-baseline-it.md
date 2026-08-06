---
id: T-36
aliases:
- T-36
title: "Re-run the dream where the robot moves, against a baseline it can lose to"
slug: re-run-the-dream-where-the-robot-moves-against-a-baseline-it
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m3
- data
- prereg
- training
sprint: ''
depends_on: []
due_date: ''
created: 2026-08-05
updated: 2026-08-05
---

# Re-run the dream where the robot moves, against a baseline it can lose to

## Description

Re-run the dream where the robot is actually moving, and give it a baseline it can lose to
(`docs/preregistration/PR-06-dream-prediction.md`, pre-registered 2026-08-05 before any run with the
new windows or arms; supersedes PR-05 §4 G2). *Two defects in T-35, both mine.* **D1 — the windows
were the moments nothing happens.** Both window builders picked with `np.linspace(0, n-1, count)`,
and at the 2 windows/episode setting T-35 ran that returns exactly `[first, last]` — on
GR00T-AppleToPlate the two moments the arm is out of frame (apple beside the plate before the reach,
on the plate after the withdrawal). `runs/dream/t35-zerogpu-seed0/gt.png` shows four clips with no
robot in any of them. Measured on CPU over the same 8 episodes, same command, both selectors: **gt
motion 0.165 → 4.034 (24×)**, static fraction 0.953 → 0.008, and 8 of 16 windows dropped as
bit-identical under `linspace` against 16 of 16 under `motion`. Per 9-frame window: first+last
0.071, all windows 1.107, episode peak 5.183 — T-35 sampled ~15× quieter than the episode average.
**So PR-05's G1 MOVES (ratio 1.046) is arithmetically right and reads far stronger than it is**: on
the two moments the data barely moves, the dream also barely moves. It also means "the corpus is
nearly static" was partly the selector, not the corpus — while the robot acts the corpus moves
1.1–5.2, not 0.17. **D2 — the discriminator had noise on both sides**, already recorded VOID in
`PR-05-RESULT.md`; `base_seed1` is now opt-in (`--base-null`) so that verdict cannot fire by
accident. **The fix:** `--window-select {linspace,motion}` on both builders
(`select_windows_by_motion` — a chosen subpopulation, "the moments the robot is acting", never to be
quoted as a corpus statistic; it cannot bias a gate because every arm sees the same windows), plus
an **anchored** arm group that turns a sample into a *prediction*: `recon_a` (what actually
happened), `freeze_a` (frame 0 held — the trivial predictor, built from `recon` so both sides carry
the same codec loss), `lora_a`, and `lora_a_seed1` closing PR-05's open second-seed caveat. **G2
becomes a comparison freezing can win:** `d(lora_a, recon_a) < 0.9·d(freeze_a, recon_a)`,
`FREEZE_MARGIN = 0.9` in git before the run, and UNSTABLE if the two seeds straddle the threshold.
On a corpus where 96 % of frame pairs move under one grey level, freezing is a strong baseline, not
a straw man — a world model that cannot beat "nothing happens" over 0.27 s has not been shown to
model anything. 67 tests (was 46). **✅ ran 2026-08-05 on ZeroGPU, GPU wall 56.3 s, peak VRAM 32.5 GB
(`docs/preregistration/PR-06-RESULT.md`, `runs/dream/t36-zerogpu-motion-seed0/`) — run twice on two
independently assigned workers, the two reports byte-identical apart from VRAM.** **G1 MOVES at
ratio 0.588** (floor 0.5): it clears the gate and is a far weaker number than PR-05's 1.046 on the
*same gate, same checkpoint, same geometry* — only the windows changed. Where the robot acts the
free dream generates ~59 % of the recorded motion; on the two dead moments PR-05 sampled it
generated 105 % of almost nothing. **PR-05's G1 is superseded**, re-read as "it stands still where
the data stands still", which was never the question. **G2 NO_BETTER_THAN_FREEZING, and not
narrowly: `d(lora_a, recon_a) = 16.656` against `d(freeze_a, recon_a) = 12.020`** — the prediction
is not 10 % closer to the truth than doing nothing, it is **39 % further away**, and seed 1 (17.322)
lands the same side so the verdict is stable, not UNSTABLE. **Standing still would have been a
better prediction than the world model's.** That is §6 row 2, the outcome pre-registered as most
likely: motion of roughly the right magnitude (ratio 0.830) spent in the wrong directions. **The
clips sharpen it rather than soften it — the anchor works.** In every rendered clip `lora_a` puts
the scene where `recon_a` puts it (arm entering left with the apple in its fingers, plate correct,
apple-on-plate vs apple-beside-plate correct); what it gets wrong is dynamics — the arm wobbles
instead of travelling, the plate smears, the apple jitters, at 4.50 frame-to-frame against the
truth's 4.14. So the failure is specific: **scene layout is carried, dynamics are not.** Side
observations, ungated: `recon` reproduces `gt` motion to 1 % here (4.138 vs 4.092) where on PR-05's
static windows the round-trip read 1.20× the recording, so the denominator is more trustworthy in
this regime; and `base` scores 22.683 against PR-05's 22.456, i.e. the adapter-disabled prior is the
same colour noise whichever windows it gets. **The open threat to this result, stated rather than
argued away:** anchoring by replacement is a mode the model was never trained in, so a negative
could be the technique. Two things argue otherwise (the pin reaches the free frames or layout would
not match; anchoring moves the sample toward the truth's motion level, free 2.434 → anchored 3.435
against 4.130) but neither is a gate. The clean separation is one uncomputed CPU comparison,
`d(lora[:, 1:], recon_a)` — the free sample on the same target — and it is a new pre-registration,
not a patch to this one. **This makes the training-data case worse, not better:** a generator whose
0.3 s predictions are further from the truth than a still frame is not a source of supervision.

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
