#!/usr/bin/env python3
"""Run PR-08 §6's G0a and G0b VOID gates over a restyled corpus, and carry the verdict in the
exit code.

    .venv/bin/python scripts/run_g0_gates.py --explain            # what is missing, and why
    .venv/bin/python scripts/run_g0_gates.py \\
        --source-dataset datasets/gr00t-apple-full \\
        --restyled-dataset runs/pr08-restyle/lerobot \\
        --holdout configs/splits/t18_holdout_episodes.txt \\
        --source-centroids runs/pr08-g0/centroids-source.json \\
        --restyled-centroids runs/pr08-g0/centroids-restyled.json \\
        --out runs/pr08-g0/g0_gates.json

WHY THIS FILE EXISTS. ``cluster/discoverer/97_transfer25_restyle.sbatch`` stamps every generated
clip's record with ``"gates_note": "G0a/G0b/G0c have NOT run"``. That is honest and it is useless:
it is a promise that three gates will be evaluated by something, and until now nothing could
evaluate two of them. PR-08 §6 calls them **VOID gates, not reports** — they run before any
training, on CPU, and a failure is not a finding about the corpus, it is a finding against the
generation pipeline. This is the runner. It does not generate, it does not train, and it cannot
license either: a PASS here closes two of §8's seven items and nothing else.

G0a — LABEL INTEGRITY, WHICH IS AN IDENTITY CHECK AND NOT A SCREEN
------------------------------------------------------------------
``screen_corpus.py`` (T-34) computes M1, M2 and M3 from proprioception, the clock and the gripper
channel. **A restyle changes no action**, so on the restyled corpus that screen is vacuous taken at
face value — it must return the SOURCE's numbers by construction. PR-08 §6 keeps it with its job
restated: reproduce the source's M1/M2/M3 within ``EXPECT_TOL`` (0.02 / 0.02 / 0.05, the script's
own archived tolerances). **A deviation is not a fact about the corpus. It is proof that the
generation pipeline corrupted or reordered the action labels, and the verdict is VOID.**

So this runner does not re-implement any metric. It measures the SOURCE through
``screen_corpus.main``, injects those three numbers into ``screen_corpus.ARCHIVED`` under the key
``pr08-source``, and then runs the RESTYLED corpus through ``screen_corpus.main --expect
pr08-source``. The comparison, the tolerances and the exit status are ``screen_corpus``'s own
machinery pointed at a different reference — which is exactly what §6 asks for, and it means there
is only one implementation of M1/M2/M3 in the repository. The per-metric deltas are recomputed here
ONLY to put them in the artifact (``screen_corpus`` records ``reproduced: true|false`` and not the
numbers), and the recomputed verdict is cross-checked against ``screen_corpus``'s own exit status:
a disagreement is a bug in this file and refuses rather than reporting either answer.

Both sides are scored on the SAME holdout file and the SAME seed. A screen run against a different
split is a different measurement, and an identity check between two different measurements would
report the split as a label defect.

G0b — GEOMETRY INVARIANCE, AGAINST A DERIVED BUDGET THAT MAY REFUSE TO EXIST
----------------------------------------------------------------------------
Object and plate centroids in the restyled clip must agree with the source. The tolerance is
``GEOM_TOL - EST_DRIFT_P95``, both read from the committed
``configs/transfer25/pr08_geom_tol.json`` — never coined here, never defaulted, never assumed:

* a null ``geom_tol_px`` or a null ``est_drift_p95_px`` REFUSES the gate. It does not treat the
  missing number as zero and it does not warn and continue. A drift budget assumed to be zero
  WIDENS the tolerance, which looks conservative and is backwards, and PR-08 §4 records
  ``EST_DRIFT_P95`` as a LOWER bound on the real error in the first place;
* a margin ``<= 0`` REFUSES, quoting §6: *"if that is <= 0, the estimator is not good enough and
  generation does not start"*. Improving the estimator is the move; widening the gate is not
  available from here;
* an ``est_drift_p95_px`` whose SEGMENTER THE DOCUMENT DOES NOT NAME refuses too. The two halves of
  the subtraction are measured by two scripts into two files and merged into this one document;
  ``est_drift_estimator_name`` is what survives that merge, and ``measure_geom_tol.py
  --carry-est-drift`` writes it beside the number and refuses to write either alone. Until
  2026-08-22 no producer wrote any spelling of it, so this check could only ever answer "could not
  check" — which cost every run its gate qualification and made exit 0 unreachable from a real
  artifact. A gate that cannot say yes blocks generation exactly as a wrong one would;
* the two sides must have been measured with the SAME segmenter on the SAME pixel grid, and this
  runner VERIFIES that from the two centroid records rather than trusting the caller to have done
  it. Centroids from two segmenters are two quantities, and their difference in pixels is a
  plausible number that means nothing — the identical failure ``measure_est_drift.py``'s
  ``cross_check_geom_tol()`` exists to catch on the other end of the same subtraction.

  **THE DECODER IS PART OF THAT INSTRUMENT AND IS NOW RECORDED AND COMPARED** (2026-08-23). A
  segmenter is a function of the pixels it is handed, and two decoders hand it two different sets
  of pixels for the same file; ``resolve_decoder`` probes each side's own bytes independently, so
  one command line can resolve two. **It costs the run its gate qualification and does not refuse**
  — G0b's two sides are not the same codec by construction (the PR-08 source is av1, job 189585 is
  the record of cv2 decoding ZERO frames of it; the generator's output is not av1), so a hard
  refusal would be a gate the real corpus cannot satisfy, which is the defect this file was already
  repaired for once. Whether §6 should REQUIRE one decoder or only require the pair to be recorded
  is a question for the rule, not for this runner; ``decoder_disagreements`` says where its rows
  would move if the answer is "required".

  **Exactly how far that verification reaches is stated rather than implied.** Name, version and
  pixel grid are compared on EVERY path. The PINNED OPERATING POINT below the name — the three
  checkpoint revisions, the text prompt, both threshold pairs, the box rule, the propagation mode —
  is compared whenever a side STATES it, and a side that does not state it is recorded as
  NOT_GATE_QUALIFIED rather than passed: "two runs can share a name while disagreeing about every
  number below" is the committed contract's own sentence, and a requirement met by saying nothing
  is the default-permissiveness this repository has already removed twice. On the ``--*-clips``
  path the operating point is not left to the caller at all: it is read off the estimator adapter
  ``measure_geom_tol`` actually selected (``SEGMENTER_CONTRACT``) and carried into the side record,
  so the measuring path compares the same fields the committed contract pins.

* the committed tolerance's own ``consumer_asserts`` list — the machine-readable checklist
  ``scripts/measure_geom_tol.py`` writes INTO the artifact so that producer and consumer cannot
  drift apart in prose — is READ AND ASSERTED, entry by entry. An entry this runner has no handler
  for REFUSES the gate: a checklist the producer grew and the consumer silently ignored is worse
  than no checklist. ``step_frames`` is the sharpest of them and is asserted whether or not the
  artifact declares the list, because G0b compares source frame i to restyled frame i — a ONE-frame
  step — and GEOM_TOL scales roughly linearly with the step it was measured at. A tolerance
  measured at ``--step-frames 3`` quoted here is a gate roughly three times too loose, silently.

Centroids come from ``scripts/measure_geom_tol.py`` — ``centroid_of_mask``, ``resolve_method``,
``episode_centroids_from_video``, ``distribution`` — imported, never re-typed. Two implementations
of "where is the apple" is precisely the drift this gate exists to catch, and a gate that drifts
from the tolerance it is evaluated against is worse than no gate.

**The per-clip margin distribution is recorded, not just the pass/fail.** A corpus that clears the
budget at the median and fails at the p99 is a different fact from one that clears it everywhere,
and the second fact is the one that decides whether the failure is a handful of frames or the
generator. The gate statistic itself is ``--g0b-percentile`` (default 100, i.e. every measured
frame must be inside the budget) because §6 says "must agree", not "must usually agree"; the
percentile is a flag so the looser reading is one argument away and is recorded in the artifact
either way.

WHAT THIS DOES NOT DO
---------------------
**G0c is not evaluated here, and is not claimed.** §6 solves it by construction — the real robot's
pixels are composited back over every generated frame — so there is no threshold to run. The
artifact records it as NOT EVALUATED with that reason, because a gate record that silently omits
one of three gates reads downstream as three gates having passed.

**Frames are not decoded unless a segmenter is wired.** ``--source-clips/--restyled-clips`` drive
``measure_geom_tol``'s adapter selection, which refuses loudly when no gate-qualified segmenter is
present on this machine. That is today's actual state, which is why ``--explain`` exists: for the
next while this script is mostly a statement of what does not exist yet.

**A pass here licenses nothing on its own.** PR-08 §1 forbids generation until every §8 item is
closed and T-39 has reported; §6's gates are two of those items and the sbatch's own refusals are
the others. Nothing in this file may be read as lifting any of them.

EXIT STATUS
-----------
0   PASS               — every gate that was asked for RAN and PASSED. In ``--explain`` mode: every
                         input is present AND G0b's budget forms.
2   REFUSED            — a requested gate could NOT be evaluated: an input is missing, malformed,
                         null, or the two sides disagree about the instrument. No verdict on the
                         corpus is claimed, and none may be inferred. In ``--explain`` mode: an
                         input is missing, or the budget cannot be formed from the ones that exist.
3   NOT_GATE_QUALIFIED — the gates ran, nothing failed, but this run may not stand as the gate:
                         a partial corpus, coverage below ``--min-coverage``, an ungated segmenter,
                         or a half PR-08 §6 names that was never measured (the plate centroids).
                         The artifact is written; it must not license anything. This is the same
                         meaning exit 3 already carries in ``measure_geom_tol.py`` and
                         ``measure_est_drift.py``.
4   VOID               — a gate RAN and FAILED. PR-08 §6's own word: the labels moved (G0a) or the
                         geometry moved (G0b), and the finding is against the generation pipeline.

**4 beats 2 beats 3.** A determined VOID is reported even when another gate refused, because a
reader who sees "refused, fix your inputs" and never learns a VOID was determined is the one
failure this ordering can prevent. Every code but 0 blocks generation identically, so the ordering
costs nothing operationally and buys that.

DRIVING THIS FROM A JOB SCRIPT
------------------------------
There is ONE entry point and it is ``main(argv) -> int``: it never raises for a gate outcome, it
always writes ``--out``, and the integer it returns is the table above. ``python
scripts/run_g0_gates.py ...`` returns the same code through ``SystemExit``. A caller needs nothing
else from this module — importing it to reach an internal helper means the caller is re-deciding
something this file already decided.

The invocation ``cluster/discoverer/97_transfer25_restyle.sbatch`` should make, AFTER the restyled
corpus has been assembled into canonical episodes and BEFORE anything downstream treats those
episodes as data:

    "${WAM_PY}" scripts/run_g0_gates.py \\
        --gates g0a,g0b \\
        --source-dataset   "${SOURCE_LEROBOT}" \\
        --restyled-dataset "${RESTYLED_LEROBOT}" \\
        --holdout          configs/splits/t18_holdout_episodes.txt \\
        --source-centroids   "${G0_DIR}/centroids-source.json" \\
        --restyled-centroids "${G0_DIR}/centroids-restyled.json" \\
        --geom-config      configs/transfer25/pr08_geom_tol.json \\
        --out              "${G0_DIR}/g0_gates.json"
    G0_RC=$?

WHERE THE TWO CENTROID RECORDS COME FROM, because the job has to make them first and this is the
part that is easy to hand-wave. Each is one pass of this same script's ``--*-clips`` path with
``--dump-centroids``, and the adapter takes ONE text prompt per process, so §6's "object AND plate"
is two passes per side:

    WAM_PR08_OBJECT_PROMPT="apple." "${WAM_PY}" scripts/run_g0_gates.py --gates g0b \\
        --source-clips "${SOURCE_CLIPS}" --restyled-clips "${RESTYLED_CLIPS}" \\
        --restyled-source-map "${G0_DIR}/source_of.json" \\
        --object-label object --dump-centroids "${G0_DIR}/object" --out "${G0_DIR}/g0b-object.json"

and the same again with ``WAM_PR08_OBJECT_PROMPT="plate."`` and ``--object-label plate``. **Those
two passes produce one record per side PER LABEL, and merging a side's two records into the single
record the gate run above reads is NOT implemented here** — an object-only run is a legitimate exit
3 with the plate named as the missing half, and a job that wants a 0 has to do that merge (or the
merge has to be built). Saying so is the point: a runner that quietly gated on the object alone
would be reporting half a gate as a whole one.

What each code must mean to the job:

* ``0``  — G0a and G0b both ran and passed. This is the ONLY code under which the job may remove
  its ``NOT_TRAINING_DATA`` marker, and even then only if every OTHER §8 refusal in that file is
  independently satisfied. G0c is not covered by this artifact at all (see below), so "the G0 gates
  passed" is not what a 0 here says.
* ``2``  — refused. The corpus is unjudged. The marker stays, the job reports which input was
  missing (the artifact's ``gates.*.refusal`` carries the text) and exits non-zero.
* ``3``  — ran, nothing failed, may not stand as the gate. Operationally identical to 2 for the
  marker: a run that cannot stand as the gate has not closed the §8 item. The difference is
  diagnostic and it matters — the fix for a 3 is usually one flag or one missing field, not a
  missing measurement.
* ``4``  — VOID. The finding is against the GENERATION pipeline, not against the corpus. The job
  must keep the marker, must not delete the clips (they are the evidence), and should say so
  loudly: something reordered the action labels or moved the geometry.

Any other code is this script crashing, and a crash is not a verdict.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

SCHEMA = "wam.pr08_g0_gates/1"
WRITEUP = "docs/preregistration/PR-08-photoreal-augmentation.md"
RULE = "T40_RULE_V1"
GATE = "PR-08 §6 G0"

EXIT_PASS = 0
EXIT_REFUSED = 2
EXIT_NOT_GATE_QUALIFIED = 3
EXIT_VOID = 4

#: Worst-first. The aggregate verdict is the last of these that any gate reported; see the exit
#: table in the module docstring for why VOID outranks REFUSED.
VERDICT_ORDER: tuple[str, ...] = ("PASS", "NOT_GATE_QUALIFIED", "REFUSED", "VOID")
VERDICT_EXIT: dict[str, int] = {
    "PASS": EXIT_PASS,
    "NOT_GATE_QUALIFIED": EXIT_NOT_GATE_QUALIFIED,
    "REFUSED": EXIT_REFUSED,
    "VOID": EXIT_VOID,
}

#: The committed gate inputs. PR-08 §8 item 4: "GEOM_TOL and EST_DRIFT_P95 measured and committed".
#: This file never writes it — it is produced by scripts/measure_geom_tol.py (GEOM_TOL) and
#: scripts/measure_est_drift.py (EST_DRIFT_P95), and a gate that could write its own tolerance
#: would not be a gate.
GEOM_CONFIG_DEFAULT = _REPO_ROOT / "configs" / "transfer25" / "pr08_geom_tol.json"

#: §6, verbatim. Quoted into the refusal so the message carries the rule rather than a paraphrase
#: of it, and so nobody has to go and check whether this script invented the condition.
NON_POSITIVE_MARGIN_QUOTE = (
    "if that is <= 0, the estimator is not good enough and generation does not start"
)

#: Where the budget's two halves are read from, in order. The FIRST spelling is the schema this
#: runner is written against; the others are the spellings ``scripts/measure_geom_tol.py`` already
#: writes into the very same path (``GEOM_TOL_px``) and that the sbatch already reads
#: (``GEOM_TOL_px``/``GEOM_TOL``). Accepting both is not laxity — it is the same file being written
#: by two producers, and a gate that refused the producer this repository actually has would be a
#: gate nobody could run. Which key was used is RECORDED in the artifact, because "which field did
#: the tolerance come from" is exactly the question a wrong tolerance raises later.
GEOM_TOL_KEYS: tuple[str, ...] = ("geom_tol_px", "GEOM_TOL_px", "GEOM_TOL")
EST_DRIFT_KEYS: tuple[str, ...] = ("est_drift_p95_px", "EST_DRIFT_P95")

#: BORROWED, NOT COINED. This is ``measure_geom_tol.DEFAULT_MIN_COVERAGE`` — the fraction of steps
#: that must have yielded a measurable displacement before the producer will call its own number a
#: gate. G0b measures the same quantity on the same corpus with the same segmenter, so holding it
#: to a DIFFERENT floor would mean this gate and the tolerance it is evaluated against disagree
#: about how much of the corpus has to be visible for a pixel statistic to describe it. It is not
#: imported at parse time (that would execute a heavy module to read one float, in ``--explain``
#: mode too); ``tests/test_run_g0_gates.py`` asserts the two constants are equal instead, so a
#: change on either side is caught by a named test rather than by nobody.
MIN_COVERAGE_DEFAULT = 0.90

#: The step G0b gates under, and it is not a choice: G0b compares SOURCE frame i to RESTYLED frame
#: i. That is a one-frame step by construction. GEOM_TOL is the median per-step object-centroid
#: displacement and scales roughly linearly with ``--step-frames``, so a tolerance measured at 3
#: and quoted here is a budget roughly three times too loose — a widened pre-registered gate, with
#: nothing in the artifact to show for it. ``measure_geom_tol`` writes ``step_frames`` into the
#: artifact and names this exact assertion in its ``consumer_asserts``; this is the consumer
#: honouring it.
G0B_STEP_FRAMES = 1

#: The schema of the per-side centroid record this gate compares. One per side, each carrying the
#: segmenter that produced it and the grid it was produced on — the two things §4 step 2 and §6
#: require to be equal and that nothing else in the pipeline checks.
CENTROID_SCHEMA = "wam.pr08_centroids/1"

#: PR-08 §6 G0b gates "Object and plate centroids". GEOM_TOL is derived from the OBJECT centroid
#: alone (measure_geom_tol.py records that caveat in its own artifact), so the plate is held to a
#: tolerance that is loose for it — recorded, never silently widened. A record with no plate is not
#: refused (the segmenter adapter takes one text prompt per process, so the plate is a second run)
#: but it cannot be gate-qualified either: half a gate that reports PASS is the failure mode this
#: whole document is written against.
LABEL_OBJECT = "object"
LABEL_PLATE = "plate"
LABELS_GATED: tuple[str, ...] = (LABEL_OBJECT, LABEL_PLATE)

#: The key ``screen_corpus.ARCHIVED`` is extended with at run time. It is NOT committed into that
#: file: the source's M1/M2/M3 are a property of whichever source corpus this gate was pointed at,
#: measured in this process, and freezing them into the script would turn a measured reference into
#: an archived constant that no longer tracks the corpus it claims to describe.
SOURCE_REFERENCE_KEY = "pr08-source"

#: report field -> the short name ``EXPECT_TOL`` keys on.
SCREEN_METRIC_KEYS: dict[str, str] = {
    "m1": "m1_momentum_share",
    "m2": "m2_blind_unreachable",
    "m3": "m3_transitions_per_episode",
}


class GateRefusal(RuntimeError):
    """A gate cannot be evaluated. Never a verdict on the corpus, and never fallen back from.

    Everything that reaches this is a statement about the INPUTS: a file that is not there, a
    number that is null, two sides measured with different instruments. The distinction from VOID
    is the whole point of having two exit codes — a VOID indicts the generation pipeline, a refusal
    indicts nothing at all and licenses nothing either.
    """


def load_script(name: str) -> Any:
    """Import a sibling script by path, the way ``rederive_t39_g0.py`` does.

    ``scripts/`` is not a package and these files are CLIs first. Importing by path is how every
    other driver in this repo reaches them, and it keeps "the code that runs the gate" and "the
    code that defined the metric" the same object rather than a copy that has to be kept in step.
    """
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GateRefusal(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def repo_rel(path: Path) -> str:
    """Repo-relative when it is under the repo, absolute otherwise. Never raises."""
    try:
        return str(Path(path).resolve().relative_to(_REPO_ROOT))
    except (ValueError, OSError):
        return str(path)


# ================================================================================ G0a label integrity


def assert_expect_machinery_is_live(screen: Any) -> None:
    """Refuse unless ``screen_corpus``'s ``--expect`` really is the machinery being driven.

    This runner works by ADDING a reference to ``screen_corpus.ARCHIVED`` and then calling that
    script's own ``main`` with ``--expect``. That only works while ``main`` builds its ``--expect``
    choices from ``ARCHIVED`` at call time. If ``screen_corpus`` ever freezes the choices at import
    time, or renames the dict, the injection stops reaching the CLI — and the failure mode is not a
    crash: argparse would reject the key and ``main`` would ``SystemExit(2)``, or worse, a future
    shape could silently compare against ``gr00t``'s archived triple instead of this corpus's
    source. That would report the GR00T corpus's M1/M2/M3 as the identity reference for a restyle
    of some other corpus, which is a wrong VOID or a wrong PASS with nothing in the artifact to
    show for it.

    So the coupling is asserted, in the one place that depends on it, against the source of the
    function that is about to be called. It is cheap, and it converts a silent semantic change into
    a named refusal on the day it happens.
    """
    import inspect

    archived = getattr(screen, "ARCHIVED", None)
    if not isinstance(archived, dict):
        raise GateRefusal(
            "REFUSED: scripts/screen_corpus.py no longer exposes ARCHIVED as a dict, so the "
            "source's measured M1/M2/M3 cannot be registered as an --expect reference. G0a is an "
            "identity check against the SOURCE and must not silently fall back to the archived "
            "gr00t triple."
        )
    tol = getattr(screen, "EXPECT_TOL", None)
    if not isinstance(tol, dict) or set(tol) < set(SCREEN_METRIC_KEYS):
        raise GateRefusal(
            "REFUSED: scripts/screen_corpus.py's EXPECT_TOL does not cover m1/m2/m3. PR-08 §6 "
            "names those tolerances as 'the script's own archived tolerances'; this runner does "
            "not carry a second copy of them and will not invent one."
        )
    source = inspect.getsource(screen.main)
    if "choices=sorted(ARCHIVED)" not in source or "ARCHIVED[args.expect]" not in source:
        raise GateRefusal(
            "REFUSED: scripts/screen_corpus.py's main() no longer reads --expect out of ARCHIVED "
            "at call time (looked for 'choices=sorted(ARCHIVED)' and 'ARCHIVED[args.expect]').\n"
            "        G0a drives that machinery against the SOURCE's measured values instead of "
            "the archived gr00t ones,\n"
            "        which only works while the reference is looked up in that dict. Re-point this "
            "runner at whatever\n"
            "        replaced it — do NOT re-implement the comparison here, and do NOT let it fall "
            "back to 'gr00t'."
        )


def register_source_reference(
    screen: Any, metrics: dict[str, float], key: str = SOURCE_REFERENCE_KEY
) -> str:
    """Install the SOURCE's measured triple as an ``--expect`` reference. Returns the key.

    Mutating another module's table at run time is not free of risk, so it is done here, once, in
    a named function, and never written to disk: the reference is a measurement of the corpus this
    invocation was pointed at, and a committed constant would go stale silently the first time the
    source corpus is re-derived (PR-08 §3's 640x480 re-derivation is exactly that).
    """
    assert_expect_machinery_is_live(screen)
    missing = [k for k in SCREEN_METRIC_KEYS if metrics.get(k) is None]
    if missing:
        raise GateRefusal(
            f"REFUSED: the source screen did not produce {', '.join(missing)}. There is no "
            "reference to hold the restyled corpus to, and 'no reference' is not 'no deviation'."
        )
    screen.ARCHIVED[key] = {k: float(metrics[k]) for k in SCREEN_METRIC_KEYS}
    return key


def metrics_from_screen_report(report: dict[str, Any], where: str) -> dict[str, float]:
    """The three numbers, or a refusal that says which one was absent and why that matters."""
    out: dict[str, float] = {}
    for short, field_name in SCREEN_METRIC_KEYS.items():
        value = report.get(field_name)
        if value is None:
            raise GateRefusal(
                f"REFUSED: {where} reports {field_name} = null. screen_corpus writes null for a "
                "non-finite metric (a collapsed ceiling makes M1's denominator zero or negative), "
                "so there is nothing to compare. Fix the screen before gating on it."
            )
        out[short] = float(value)
    if not report.get("ceiling_dominates", False):
        raise GateRefusal(
            f"REFUSED: {where} reports ceiling_dominates = false. screen_corpus's own G4 says M1 "
            "and M2 are VOID in that state — a blind ceiling beaten by a zero-parameter rule. An "
            "identity check between two VOID numbers is not evidence about the labels either way."
        )
    return out


def expect_deltas(
    screen: Any, source: dict[str, float], restyled: dict[str, float]
) -> list[dict[str, Any]]:
    """Per-metric |restyled - source| against ``screen_corpus.EXPECT_TOL``.

    This is a RECORD, not a second implementation of the gate: the verdict G0a reports is
    ``screen_corpus.main --expect``'s own exit status, and ``g0a_record`` refuses if these two ever
    disagree. It exists because ``screen_corpus`` writes ``expect: {reproduced: true|false}`` into
    its artifact and not the deltas, and PR-08 §6 wants the deviation recorded — "the labels moved
    by 0.003" and "the labels moved by 0.9" are the same boolean and very different findings.

    The comparison operator is ``<=``, matching ``screen_corpus.main`` exactly: a deviation of
    precisely ``EXPECT_TOL`` reproduces. Anything else here would make the two disagree at the
    boundary, which is the one place a gate is most likely to be exercised.
    """
    rows: list[dict[str, Any]] = []
    for short, field_name in SCREEN_METRIC_KEYS.items():
        tol = float(screen.EXPECT_TOL[short])
        got, ref = float(restyled[short]), float(source[short])
        delta = got - ref
        rows.append(
            {
                "metric": short,
                "field": field_name,
                "source": ref,
                "restyled": got,
                "delta": delta,
                "abs_delta": abs(delta),
                "tol": tol,
                "within_tol": bool(abs(delta) <= tol),
            }
        )
    return rows


def deltas_verdict(deltas: list[dict[str, Any]]) -> str:
    """PASS when every metric reproduces the source, VOID otherwise. There is no third reading."""
    return "PASS" if all(row["within_tol"] for row in deltas) else "VOID"


def run_screen(
    screen: Any,
    dataset: Path,
    holdout: Path | None,
    out: Path,
    seed: int,
    expect: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Call ``screen_corpus.main`` and return ``(status, its artifact)``.

    ``SystemExit`` is converted rather than propagated: ``screen_corpus`` raises it for every input
    problem (no episodes, a holdout naming episodes that are not there), and those are refusals of
    THIS gate — they must land in the artifact with the rest of the record instead of killing the
    process with an exit code that means something else in this script's table.
    """
    argv = ["--dataset", str(dataset), "--out", str(out), "--seed", str(seed)]
    if holdout is not None:
        argv += ["--holdout", str(holdout)]
    if expect is not None:
        argv += ["--expect", expect]
    try:
        status = int(screen.main(argv))
    except SystemExit as exc:  # screen_corpus's own input refusals
        raise GateRefusal(f"REFUSED: screen_corpus on {dataset}: {exc}") from exc
    if not out.is_file():
        raise GateRefusal(
            f"REFUSED: screen_corpus returned {status} but wrote no artifact at {out}. Without it "
            "there is nothing to record and nothing to compare."
        )
    return status, json.loads(out.read_text())


def g0a_record(
    screen: Any,
    source_report: dict[str, Any],
    restyled_report: dict[str, Any],
    screen_status: int,
    reference_key: str,
) -> dict[str, Any]:
    """Assemble G0a's block, with the two verdicts cross-checked against each other.

    ``screen_status`` is ``screen_corpus.main``'s own return value under ``--expect``: 0 when all
    three metrics reproduced, non-zero otherwise. ``deltas_verdict`` recomputes the same decision
    from the same tolerances. They cannot disagree unless this file has drifted from the script it
    is driving, so a disagreement is reported as a refusal naming both answers rather than as
    either of them — reporting one would mean picking, silently, which implementation of the gate
    the project is running.
    """
    source = metrics_from_screen_report(source_report, "the SOURCE screen")
    restyled = metrics_from_screen_report(restyled_report, "the RESTYLED screen")
    deltas = expect_deltas(screen, source, restyled)
    verdict = deltas_verdict(deltas)
    screen_says = "PASS" if screen_status == 0 else "VOID"
    if verdict != screen_says:
        raise GateRefusal(
            f"REFUSED: G0a's two readings disagree — screen_corpus --expect {reference_key} "
            f"returned {screen_status} ({screen_says}) while the recorded per-metric deltas say "
            f"{verdict}. One of them is wrong and this runner will not choose. Deltas: "
            + "; ".join(
                f"{r['metric']} {r['delta']:+.4f} tol +-{r['tol']}" for r in deltas
            )
        )

    episodes_source = (source_report.get("episodes") or {})
    episodes_restyled = (restyled_report.get("episodes") or {})
    notes: list[str] = []
    structural: list[str] = []
    # A restyle changes no action and drops no episode. Different episode counts on the two sides
    # means the two screens are over different corpora, and an identity check between two different
    # corpora is not the check §6 asks for -- it would report the difference in coverage as a label
    # defect, or hide a real one behind it.
    if episodes_source != episodes_restyled:
        structural.append(
            f"the two screens scored different episode counts: source {episodes_source} vs "
            f"restyled {episodes_restyled}. A restyle emits one clip per source clip; this is "
            "either a partial corpus or a corpus whose episodes moved."
        )
    if source_report.get("holdout_file") != restyled_report.get("holdout_file"):
        structural.append(
            f"the two screens used different holdout files: {source_report.get('holdout_file')!r} "
            f"vs {restyled_report.get('holdout_file')!r}. M1/M2 are measured on the holdout, so "
            "two splits are two measurements and their difference is not a label defect."
        )
    # A STRUCTURAL MISMATCH REFUSES, WHATEVER THE METRICS SAID. It used to demote only a PASS,
    # which meant that when the two corpora differed AND the metrics deviated the runner reported
    # a confident VOID -- a formal indictment of the generation pipeline -- on a comparison it had
    # just established was not comparable. The reasoning three lines above cuts both ways: if the
    # two screens are over different corpora then this is not the identity check §6 asks for, and
    # the honest report is that no verdict on the labels was reached. The measured verdict is kept
    # in the record (``measured_verdict``) so nothing is hidden; it is simply not reported AS the
    # verdict, because the number it would indict the generator with has a second explanation.
    refusal: str | None = None
    if structural:
        refusal = (
            "REFUSED: G0a's two screens are not over the same thing, so their difference is not "
            "evidence about the labels:\n"
            + "\n".join(f"        - {row}" for row in structural)
            + f"\n        The measured identity check said {verdict}; it is recorded as "
            "measured_verdict and is NOT reported as G0a's verdict. A VOID here is an indictment "
            "of the generation pipeline, and this comparison cannot support one either way."
        )
    notes.append(
        "PR-08 §6: screen_corpus is an IDENTITY check here, not a screen. A restyle changes no "
        "action, so the restyled corpus must reproduce the source's M1/M2/M3 within EXPECT_TOL; a "
        "deviation is proof the generation pipeline corrupted or reordered the action labels."
    )
    record: dict[str, Any] = {
        "gate": "G0a",
        "what": "label integrity — the restyled corpus reproduces the SOURCE's M1/M2/M3",
        "verdict": "REFUSED" if structural else verdict,
        "measured_verdict": verdict,
        "driver": "scripts/screen_corpus.py main(--expect) — no metric is re-implemented here",
        "reference_key": reference_key,
        "expect_tol": {k: float(screen.EXPECT_TOL[k]) for k in SCREEN_METRIC_KEYS},
        "screen_corpus_expect_status": screen_status,
        "source_metrics": source,
        "restyled_metrics": restyled,
        "deltas": deltas,
        "episodes_source": episodes_source,
        "episodes_restyled": episodes_restyled,
        # Kept as a distinct field from ``refusal``: these are the STRUCTURAL facts, machine
        # readable, and a reader diffing two runs wants the list rather than the paragraph.
        "structural_mismatch": structural,
        "not_gate_qualified_reasons": [],
        "notes": notes,
    }
    if refusal is not None:
        record["refusal"] = refusal
        record["consequence"] = (
            "no statement about the restyled corpus's labels is made: the two screens were not "
            "over the same corpus or the same split."
        )
    return record


def run_g0a(args: argparse.Namespace, tmp_dir: Path) -> dict[str, Any]:
    """Measure the source, register it as the reference, screen the restyled corpus against it."""
    screen = load_script("screen_corpus")
    assert_expect_machinery_is_live(screen)

    source_out = args.source_screen_out or (tmp_dir / "screen-source.json")
    if args.source_screen is not None:
        if not args.source_screen.is_file():
            raise GateRefusal(
                f"REFUSED: --source-screen {args.source_screen} does not exist. It is an already "
                "measured screen_corpus artifact for the SOURCE corpus; drop the flag to measure "
                "it now instead."
            )
        source_report = json.loads(args.source_screen.read_text())
        # A source reference measured over some other corpus, or under some other split, is the
        # quiet way to get a wrong identity check: everything downstream still reads like a gate.
        if args.source_dataset is not None and source_report.get("dataset") not in (
            None,
            str(args.source_dataset),
        ):
            raise GateRefusal(
                f"REFUSED: --source-screen was measured on {source_report.get('dataset')!r}, not "
                f"on --source-dataset {args.source_dataset}. G0a's reference must be THIS source."
            )
    else:
        if args.source_dataset is None:
            raise GateRefusal(
                "REFUSED: G0a needs --source-dataset (or --source-screen, an artifact already "
                "measured from it). Without the source's M1/M2/M3 there is no identity to check."
            )
        _status, source_report = run_screen(
            screen, args.source_dataset, args.holdout, source_out, args.seed
        )

    key = register_source_reference(
        screen, metrics_from_screen_report(source_report, "the SOURCE screen")
    )
    restyled_out = args.restyled_screen_out or (tmp_dir / "screen-restyled.json")
    status, restyled_report = run_screen(
        screen, args.restyled_dataset, args.holdout, restyled_out, args.seed, expect=key
    )
    record = g0a_record(screen, source_report, restyled_report, status, key)
    record["source_screen_artifact"] = repo_rel(
        args.source_screen if args.source_screen is not None else source_out
    )
    record["restyled_screen_artifact"] = repo_rel(restyled_out)
    record["source_dataset"] = str(args.source_dataset) if args.source_dataset else None
    record["restyled_dataset"] = str(args.restyled_dataset)
    record["holdout"] = str(args.holdout) if args.holdout else None
    record["seed"] = args.seed
    return record


# ============================================================================ G0b geometry invariance


@dataclass
class Side:
    """One side of the comparison: the centroids, and the instrument that produced them.

    ``segmenter`` and ``resolution_hw`` are not decoration. §6 compares two centroid streams in
    pixels, and that comparison is arithmetic only if both were produced by the same segmenter on
    the same grid — so both are carried WITH the numbers and checked, rather than being assumed by
    whoever ran the two measurements.
    """

    side: str
    origin: str
    segmenter: dict[str, Any]
    resolution_hw: list[int]
    clips: dict[str, dict[str, list[tuple[float, float] | None]]]
    #: restyled clip key -> the SOURCE clip key it is a restyle of. One source clip becomes many
    #: restyled clips (ten styles, ten seeds), so the pairing cannot be by equal names and is
    #: declared by the record rather than guessed from a naming convention here.
    source_of: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: The VIDEO DECODER that turned this side's bytes into the frames the segmenter saw, when the
    #: record states one. It is part of the instrument and not a detail of it: a segmenter is a
    #: function of pixels, and two decoders hand it two different sets of pixels for the same file —
    #: different colour conversion, and on a stream one of them mis-parses, different frames
    #: entirely. This gate's two sides are the corpus most exposed to that difference, because they
    #: are not the same codec: the PR-08 source is av1 (job 189585 is the record of cv2 decoding
    #: ZERO frames of it — "Missing Sequence Header") and the generator's output is not. So the two
    #: sides can legitimately resolve two decoders on one command line, which is exactly the
    #: "plausible pixel number that means nothing" this record exists to make impossible.
    decoder: dict[str, Any] | None = None

    @property
    def segmenter_name(self) -> str:
        return str(self.segmenter.get("name"))

    @property
    def segmenter_version(self) -> str | None:
        version = self.segmenter.get("version")
        return None if version is None else str(version)

    @property
    def decoder_id(self) -> str | None:
        """``"name version"``, or ``None`` when the record does not state a decoder at all."""
        if not isinstance(self.decoder, dict) or not self.decoder.get("name"):
            return None
        return f"{self.decoder['name']} {self.decoder.get('version') or '?'}"


def load_geom_config(path: Path) -> dict[str, Any]:
    """Read the committed GEOM_TOL / EST_DRIFT_P95 artifact, or refuse naming what would write it."""
    if not path.is_file():
        raise GateRefusal(
            f"REFUSED: {repo_rel(path)} does not exist, so G0b has no tolerance.\n"
            "        GEOM_TOL comes from scripts/measure_geom_tol.py (median per-step object-"
            "centroid displacement\n"
            "        in the SOURCE clips) and EST_DRIFT_P95 from scripts/measure_est_drift.py "
            "(PR-08 §4 steps 1-4,\n"
            "        which need the Isaac ground-truth capture). PR-08 §8 item 4 requires both "
            "measured AND COMMITTED\n"
            "        before a single clip is generated; this gate reads that commitment and never "
            "writes it."
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateRefusal(f"REFUSED: {repo_rel(path)} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise GateRefusal(f"REFUSED: {repo_rel(path)} is not a JSON object.")
    return doc


def committed_digest(path: Path) -> tuple[bool | None, str]:
    """Check the tolerance against its ``.sha256`` sidecar. Returns ``(verified, note)``.

    ``measure_geom_tol.py`` writes that sidecar next to every artifact it produces, and the
    generation sbatch already treats its absence as fatal — "a GEOM_TOL with no committed digest is
    a number, not a pre-commitment". This gate is softer about ABSENCE (it must stay runnable
    against a scratch tolerance during development, and the sbatch is the place that refuses one)
    and exactly as hard about DISAGREEMENT: a file edited after it was committed is not the
    committed file, and the one move that would slip a hand-written tolerance past a checked digest
    is editing the artifact and leaving the sidecar alone.
    """
    import hashlib

    side = path.parent / (path.name + ".sha256")
    if not side.is_file():
        return None, f"no {side.name} beside it — nothing proves this is the committed artifact"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    want = side.read_text(encoding="utf-8").strip().split()[0] if side.read_text().strip() else ""
    if digest != want:
        raise GateRefusal(
            f"REFUSED: {repo_rel(path)} does not match its committed .sha256 sidecar.\n"
            f"        on disk  {digest}\n"
            f"        sidecar  {want}\n"
            "        GEOM_TOL and EST_DRIFT_P95 are a pre-commitment; an artifact edited after it "
            "was committed is not\n"
            "        one. Re-measure, do not re-hash."
        )
    return True, digest


def _first_present(doc: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, Any]:
    """``(key, value)`` for the first key the document actually carries; ``(None, None)`` if none.

    Present-and-null is deliberately NOT the same as absent, and both are returned distinctly: a
    null is a producer saying "I could not measure this", an absence is a document that never made
    the claim. Both refuse, and the messages differ because the fixes differ.

    TWO SPELLINGS THAT DISAGREE ARE REFUSED, not resolved by order. ``GEOM_TOL_KEYS`` and
    ``EST_DRIFT_KEYS`` exist because one committed path has two producers writing two schemas into
    it (see their comments), and taking the first-listed key when both are present would silently
    pick a winner between them — a stale ``geom_tol_px`` quoted over a freshly measured
    ``GEOM_TOL_px``, or the reverse, with nothing in the artifact but a ``geom_tol_key`` field to
    show which. That is the whole class of failure this gate is built against, committed by the
    reader. One document stating one quantity twice, differently, is not a document either half of
    which can be quoted.
    """
    present = [(key, doc[key]) for key in keys if key in doc]
    distinct = {repr(value) for _key, value in present}
    if len(distinct) > 1:
        raise GateRefusal(
            "REFUSED: the tolerance artifact states one quantity under two spellings that "
            "disagree: "
            + ", ".join(f"{key}={value!r}" for key, value in present)
            + ".\n        These are the same number written by two producers into one committed "
            "path. Nothing here picks\n        between them: re-measure the artifact with a single "
            "producer, or delete the stale key. A gate that\n        chose by key order would quote "
            "a number nobody meant."
        )
    return present[0] if present else (None, None)


def gate_budget(doc: dict[str, Any], path: Path) -> dict[str, Any]:
    """``GEOM_TOL - EST_DRIFT_P95``, or a refusal. Never assumes, never defaults, never widens.

    Three refusals live here and each one is PR-08 §6 read literally:

    * a null ``est_drift_p95_px`` is the estimator saying it has not been characterised. §4 puts
      that measurement BEFORE generation "never after", and treating the missing budget as zero
      would hand the generator the whole tolerance — the widest possible gate, produced by the
      absence of a measurement. That is the opposite of conservative.
    * a null ``geom_tol_px`` leaves nothing to hold the generator to at all.
    * a margin ``<= 0`` is §6's own sentence, quoted in the message. It is a FINDING — record it —
      and the move is a better estimator, never a wider gate.

    ``gate_margin_px``, when the artifact carries it, is CHECKED rather than trusted: it is the
    same subtraction, and an artifact whose own arithmetic disagrees with itself is not a document
    this gate can quote either half of.
    """
    tol_key, tol = _first_present(doc, GEOM_TOL_KEYS)
    drift_key, drift = _first_present(doc, EST_DRIFT_KEYS)
    label = repo_rel(path)

    if tol_key is None:
        raise GateRefusal(
            f"REFUSED: {label} records none of {', '.join(GEOM_TOL_KEYS)}. G0b has no tolerance to "
            "hold the generator to, so nothing generated could be gated. Produce it with "
            "scripts/measure_geom_tol.py."
        )
    if tol is None:
        raise GateRefusal(
            f"REFUSED: {label} records {tol_key} = null. GEOM_TOL was not measured, and this gate "
            "does not assume one. It is the median per-step object-centroid displacement in the "
            "SOURCE clips; scripts/measure_geom_tol.py measures it."
        )
    if drift_key is None:
        raise GateRefusal(
            f"REFUSED: {label} records none of {', '.join(EST_DRIFT_KEYS)}. PR-08 §6 holds the "
            "generator to GEOM_TOL - EST_DRIFT_P95, and a budget with one term missing is not that "
            "quantity. scripts/measure_est_drift.py measures it (PR-08 §4 steps 1-4)."
        )
    if drift is None:
        blocked = doc.get("est_drift_source") or doc.get("est_drift_p95_blocked_by")
        raise GateRefusal(
            f"REFUSED: {label} records {drift_key} = null, so the estimator's error budget is "
            "unknown and G0b cannot be evaluated.\n"
            + (f"        blocked_by: {blocked}\n" if blocked else "")
            + "        This gate does NOT assume zero. §4 records EST_DRIFT_P95 as a LOWER bound "
            "on the real error, so\n"
            "        zero is not a conservative stand-in — it is the widest tolerance the gate "
            "can have, granted by the\n"
            "        absence of a measurement. §4 requires the estimator to be characterised "
            "BEFORE generation, never after."
        )

    try:
        tol_px, drift_px = float(tol), float(drift)
    except (TypeError, ValueError) as exc:
        raise GateRefusal(
            f"REFUSED: {label} carries non-numeric {tol_key}={tol!r} / {drift_key}={drift!r}."
        ) from exc
    margin = tol_px - drift_px

    stated = doc.get("gate_margin_px")
    if stated is not None and abs(float(stated) - margin) > 1e-9:
        raise GateRefusal(
            f"REFUSED: {label} states gate_margin_px = {float(stated)} but "
            f"{tol_key}({tol_px}) - {drift_key}({drift_px}) = {margin}. The artifact disagrees "
            "with its own arithmetic, so neither number can be quoted as the budget. Re-measure; "
            "do not re-type."
        )
    if margin <= 0.0:
        raise GateRefusal(
            f"REFUSED: GEOM_TOL({tol_px}) - EST_DRIFT_P95({drift_px}) = {margin:.6f} px.\n"
            f'        PR-08 §6: "{NON_POSITIVE_MARGIN_QUOTE}".\n'
            "        That is the finding — record it. Improving the estimator is the move, not "
            "widening the gate."
        )
    return {
        "geom_tol_px": tol_px,
        "geom_tol_key": tol_key,
        "geom_tol_source": doc.get("geom_tol_source") or doc.get("measured_by"),
        "est_drift_p95_px": drift_px,
        "est_drift_key": drift_key,
        "est_drift_source": doc.get("est_drift_source"),
        "gate_margin_px": margin,
        "gate_margin_stated_in_config": stated,
        "artifact": repo_rel(path),
        "spec_version": doc.get("spec_version"),
        "derivation": "GEOM_TOL - EST_DRIFT_P95, PR-08 §6 G0b. Neither term is coined here.",
    }


def config_instrument(doc: dict[str, Any], path: Path) -> dict[str, Any]:
    """The segmenter and grid the TOLERANCE was measured with, read off the committed artifact.

    Three spellings are accepted, and that is not laxity — it is one path written by two producers.
    The committed PR-08 gate schema spells the block ``segmenter`` and names it ``method_name``, on
    a ``pixel_grid_hw``; ``scripts/measure_geom_tol.py`` writes the identical facts as
    ``mask_method.name`` on a ``resolution_hw``, into the same default path. Which spelling was read
    is RECORDED, because "which field did the tolerance's instrument come from" is exactly the
    question a wrong tolerance raises later.

    An artifact that names NEITHER is refused. PR-08 §4 step 2's "the same segmenter" is
    uncheckable against a tolerance that never said which segmenter produced it, and a requirement
    passed by saying nothing is the default-permissiveness this repository has already removed
    twice (``97_transfer25_restyle.sbatch``: "saying nothing is exactly what a fabricated artifact
    does").

    The whole ``segmenter`` block is carried out as ``contract``. The committed file pins the
    detector, the segmenter, the depth model AND their revisions, the text prompt, both threshold
    pairs, the box rule and the propagation mode — "two runs can share a name while disagreeing
    about every number below", in its own words. A side record that carries the same block gets
    compared field for field; one that does not is compared on the name and the grid, which is all
    it offered.
    """
    block = doc.get("segmenter")
    origin = "segmenter"
    if not isinstance(block, dict) or not (block.get("name") or block.get("method_name")):
        block, origin = doc.get("mask_method"), "mask_method"
    if not isinstance(block, dict) or not (block.get("name") or block.get("method_name")):
        raise GateRefusal(
            f"REFUSED: {repo_rel(path)} names no segmenter (looked for segmenter.method_name, "
            "segmenter.name and mask_method.name).\n"
            "        PR-08 §4 step 2 requires the tolerance and the gate to use THE SAME "
            "segmenter, and a tolerance\n"
            "        that cannot say which one produced it makes that requirement uncheckable. An "
            "unstated claim is not a claim."
        )
    grid = (
        block.get("resolution_hw")
        or block.get("pixel_grid_hw")
        or doc.get("resolution_hw")
        or doc.get("frame_hw")
    )
    return {
        "name": str(block.get("name") or block.get("method_name")),
        "version": None if block.get("version") is None else str(block.get("version")),
        "gate_qualified": block.get("gate_qualified", doc.get("gate_qualified")),
        "resolution_hw": list(grid) if grid else None,
        "contract": {k: v for k, v in block.items() if k not in ("gate_qualified",)},
        "read_from": origin,
    }


# ------------------------------------------------- the producer's own checklist, actually checked


def _assert_present(doc: dict[str, Any], key: str, entry: str) -> Any:
    """The field a declared assertion is ABOUT must exist. Absence is not a pass."""
    if key not in doc:
        raise GateRefusal(
            f"REFUSED: the tolerance artifact declares the consumer assertion {entry!r} and then "
            f"does not carry {key!r}. The producer wrote that checklist so the two sides could not "
            "drift; an entry whose own field is missing is a checklist item that can never be "
            "satisfied, and passing it by saying nothing is the failure this list exists against."
        )
    return doc[key]


def _ca_mask_method_name(doc: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool | None, str]:
    """PR-08 §4 step 2's "the SAME segmenter", as far as THIS consumer can see it.

    The entry is written against a two-artifact world (GEOM_TOL here, EST_DRIFT_P95 in
    ``pr08_est_drift.json``) and this runner reads only one file. So it checks what that one file
    can support — its own recorded estimator name for the drift half, when it records one — and
    reports honestly when it cannot, which costs the run its gate qualification rather than being
    waved through. The name is separately compared against BOTH sides' centroids in
    ``instrument_disagreements``, which is the other half of the same requirement.
    """
    ours = ctx["instrument"].get("name")
    theirs: Any = None
    estimators = doc.get("estimators")
    if isinstance(estimators, dict):
        theirs = estimators.get("name")
    for spelling in ("est_drift_estimator_name", "est_drift_estimator"):
        if isinstance(doc.get(spelling), str):
            theirs = doc[spelling]
    if theirs is None:
        # A DOCUMENT THAT STATES THE NUMBER AND NOT THE SEGMENTER IS REFUSED, not merely
        # unqualified. Until 2026-08-22 no producer wrote any of the spellings above, so this
        # branch was the ONLY outcome against a measured tolerance and no G0b run could return 0 —
        # a gate structurally unable to say yes, which blocks generation exactly as a wrong one
        # would. `measure_geom_tol.py --carry-est-drift` now writes est_drift_estimator_name beside
        # est_drift_p95_px and refuses to write one without the other, so an artifact that carries
        # a budget and no name did not come from that path: it was assembled by hand, and the one
        # thing PR-08 §4 step 2 asks about it cannot be established. Soft-failing it would leave
        # the fixed and the unfixable cases indistinguishable in the record.
        _drift_key, drift = _first_present(doc, EST_DRIFT_KEYS)
        if drift is not None:
            raise GateRefusal(
                f"REFUSED: {ctx['label']} states est_drift_p95_px = {drift!r} and names no "
                "segmenter for it.\n"
                f"        The GEOM_TOL half was measured with {ours!r}; PR-08 §4 step 2 requires "
                "the SAME segmenter\n"
                "        for both halves because §6 subtracts them, and this document offers "
                "nothing to join on —\n"
                "        two segmenters subtract to a plausible pixel number that means nothing.\n"
                "        Carry the budget across with `scripts/measure_geom_tol.py "
                "--carry-est-drift <est_drift.json>`,\n"
                "        which writes the name with the number and refuses to write either alone."
            )
        return None, (
            "the artifact records no estimator name for the EST_DRIFT_P95 half, so this runner "
            "cannot check that the two halves of GEOM_TOL - EST_DRIFT_P95 came from one segmenter. "
            f"The GEOM_TOL half names {ours!r} and that IS compared against both sides' centroids."
        )
    if str(theirs) != str(ours):
        raise GateRefusal(
            f"REFUSED: the tolerance's two halves name two segmenters — GEOM_TOL {ours!r} vs "
            f"EST_DRIFT_P95 {theirs!r}. PR-08 §4 step 2 requires the SAME one, because §6 "
            "subtracts them; two segmenters subtract to a plausible pixel number that means "
            "nothing."
        )
    return True, f"both halves of the budget name segmenter {ours!r}"


def _ca_segmenter_block(doc: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool | None, str]:
    """The producer's second entry: the segmenter block agrees, and not only the name.

    The tolerance document carries the operating point TWICE and on purpose: the top-level
    ``segmenter`` block is the contract that was COMMITTED before the number was measured, and
    ``mask_method.params.segmenter`` is the adapter's ``SEGMENTER_CONTRACT`` as it was when the
    measurement RAN. ``measure_geom_tol`` refuses to write over a committed contract it disagrees
    with, so on the path this repository actually has they are equal — but that guard only runs at
    MEASUREMENT time, exactly like the estimator-name join above, and a consumer holding a finished
    artifact it did not watch being produced gets no benefit from it. Hence the entry, and hence
    this handler: a name is the one property of a segmenter that does not change when its behaviour
    does, and a moved revision or a moved box threshold makes GEOM_TOL a tolerance for a different
    instrument than the one G0b is about to gate with.

    Absence on either side is a refusal, not a pass. An artifact that DECLARES this assertion is a
    measured artifact from that producer, which writes both blocks; one that declares it and
    carries only one of them cannot support the comparison, and passing it by saying nothing is
    what the checklist exists against.
    """
    committed = doc.get("segmenter")
    mask_method = doc.get("mask_method")
    params = mask_method.get("params") if isinstance(mask_method, dict) else None
    ran = params.get("segmenter") if isinstance(params, dict) else None
    for block, where in ((committed, "segmenter"), (ran, "mask_method.params.segmenter")):
        if not isinstance(block, dict) or not block:
            raise GateRefusal(
                f"REFUSED: {ctx['label']} declares the consumer assertion that the committed "
                f"segmenter contract and the one that ran agree, and carries no {where} block "
                f"(found {type(block).__name__}).\n"
                "        The committed contract is what PR-08 §4 step 2's 'the same segmenter' is "
                "checked against, and\n        mask_method.params.segmenter is what actually ran. "
                "One of the two missing makes the comparison\n        impossible, and an "
                "impossible comparison is not a satisfied one."
            )
    shared = sorted(set(committed) & set(ran))
    if not shared:
        raise GateRefusal(
            f"REFUSED: {ctx['label']}'s committed segmenter contract and the contract the "
            "measurement ran under state no field in common "
            f"(committed: {sorted(committed)}; ran: {sorted(ran)}), so 'they agree' is a claim "
            "about nothing. One of the two blocks is not a segmenter contract."
        )
    rows = contract_disagreements(ran, committed, "segmenter")
    if rows:
        raise GateRefusal(
            f"REFUSED: {ctx['label']}'s committed segmenter contract and the contract the "
            "measurement ran under disagree:\n"
            + "\n".join(f"        - {row}" for row in rows)
            + "\n        GEOM_TOL was then measured with an instrument other than the one "
            "committed beside it, and the\n        gate would be quoting it against a third. Two "
            "runs can share a name while disagreeing about\n        every number below."
        )
    return True, (
        "the committed segmenter contract and mask_method.params.segmenter agree on all "
        f"{len(shared)} field(s) they both state: {', '.join(shared)}"
    )


def _ca_resolution_hw(doc: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool | None, str]:
    """The entry's own wording: assert the grid EXISTS before a clean grid check means anything."""
    grid = ctx["instrument"].get("resolution_hw")
    if not grid:
        raise GateRefusal(
            f"REFUSED: {ctx['label']} declares the consumer assertion on resolution_hw and states "
            "no pixel grid (looked for segmenter.resolution_hw, segmenter.pixel_grid_hw, "
            "resolution_hw, frame_hw).\n"
            "        GEOM_TOL - EST_DRIFT_P95 is a pixel budget, and a budget with no grid behind "
            "it is comparable with\n        anything. The producer's own note says a consumer must "
            "assert the key EXISTS, because the grid check\n        it delegates to is permissive "
            "about absence."
        )
    return True, f"the tolerance states its pixel grid {list(grid)} [height, width]"


def _ca_gate_qualified(doc: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool | None, str]:
    value = _assert_present(doc, "gate_qualified", ctx["entry"])
    if value is not True:
        raise GateRefusal(
            f"REFUSED: {ctx['label']} records gate_qualified = {value!r}. The producer's own "
            "checklist requires true before this number may be quoted as G0b's tolerance. Reasons "
            "the measurement recorded: "
            + "; ".join(doc.get("gate_disqualified_reasons") or ["(none recorded)"])
        )
    return True, "gate_qualified is true"


def _ca_partial_measurement(doc: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool | None, str]:
    value = _assert_present(doc, "partial_measurement", ctx["entry"])
    if value is not False:
        raise GateRefusal(
            f"REFUSED: {ctx['label']} records partial_measurement = {value!r}. A tolerance "
            "measured over part of the corpus is a smoke test, and a smoke test is not the "
            "pre-commitment PR-08 §8 item 4 requires."
        )
    return True, "partial_measurement is false"


def _ca_n_episodes(doc: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool | None, str]:
    got = _assert_present(doc, "n_episodes", ctx["entry"])
    found = _assert_present(doc, "n_episodes_found", ctx["entry"])
    if got != found:
        raise GateRefusal(
            f"REFUSED: {ctx['label']} measured {got} of {found} episodes it found. GEOM_TOL is the "
            "median over the corpus; a median over a subset of it is a different number, and "
            "nothing downstream can tell them apart."
        )
    return True, f"n_episodes == n_episodes_found == {got}"


def _ca_step_frames(doc: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool | None, str]:
    _assert_present(doc, "step_frames", ctx["entry"])
    return step_frames_check(doc, ctx["label"])


def _ca_coverage(doc: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool | None, str]:
    raw_coverage = _assert_present(doc, "coverage", ctx["entry"])
    raw_floor = _assert_present(doc, "min_coverage", ctx["entry"])
    try:
        coverage, floor = float(raw_coverage), float(raw_floor)
    except (TypeError, ValueError) as exc:
        # A null coverage is the producer saying it did not measure one; it is not a pass, and it
        # is not a traceback either -- an uncaught TypeError here would leave no artifact at all.
        raise GateRefusal(
            f"REFUSED: {ctx['label']} carries non-numeric coverage={raw_coverage!r} / "
            f"min_coverage={raw_floor!r}, and the producer's checklist requires them compared."
        ) from exc
    if coverage < floor:
        raise GateRefusal(
            f"REFUSED: {ctx['label']} records coverage {coverage:.3f} < min_coverage {floor}. The "
            "object was invisible for too much of the SOURCE corpus for the median displacement to "
            "describe it, and the producer's checklist says so."
        )
    return True, f"coverage {coverage:.3f} >= min_coverage {floor}"


def _ca_sha256(doc: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool | None, str]:
    """``committed_digest`` has already refused a DISAGREEING sidecar before this runs."""
    if ctx["digest_ok"] is True:
        return True, "the artifact matches its committed .sha256 sidecar"
    return None, (
        "no .sha256 sidecar beside the artifact, so it cannot be shown to be the committed one. A "
        "DISAGREEING sidecar refuses outright (committed_digest); an absent one lands here."
    )


#: ``scripts/measure_geom_tol.py`` writes ``consumer_asserts`` — a machine-readable list of what a
#: consumer must check before quoting GEOM_TOL — INTO the artifact, so producer and consumer cannot
#: drift apart in prose. This is the consumer honouring it. Each entry is dispatched on its leading
#: token, which is the field it is about; the prose after it can be reworded freely. Entries the
#: producer writes as a sentence rather than as ``<field> <op> <value>`` are matched by a
#: distinctive phrase instead — see :data:`CONSUMER_ASSERT_PHRASE_HANDLERS`.
#:
#: AN ENTRY WITH NO HANDLER REFUSES THE GATE. That is the point of the table: the next assertion
#: the producer grows is either implemented here or it stops this runner, and the one outcome not
#: available is the producer adding a requirement and the consumer ignoring it silently — which is
#: exactly the state this table was written to end.
CONSUMER_ASSERT_HANDLERS: dict[str, Any] = {
    "mask_method.name": _ca_mask_method_name,
    "resolution_hw": _ca_resolution_hw,
    "gate_qualified": _ca_gate_qualified,
    "partial_measurement": _ca_partial_measurement,
    "n_episodes": _ca_n_episodes,
    "step_frames": _ca_step_frames,
    "coverage": _ca_coverage,
    "sha256sum": _ca_sha256,
}

#: The producer does not write every entry as ``<field> <operator> <value>``. The segmenter-block
#: entry is a sentence, and its leading token is the word "the" — which must NEVER become a
#: dispatch key, because then the next entry the producer happens to start with "the" would be
#: silently handled by whatever handler got there first. That is the same silent-drift failure the
#: token table is written against, one level down. So entries whose token is unknown are matched
#: against these DISTINCTIVE phrases instead, and an entry matching none of them still refuses.
CONSUMER_ASSERT_PHRASE_HANDLERS: tuple[tuple[str, Any], ...] = (
    ("the segmenter block agrees", _ca_segmenter_block),
)


def consumer_assert_handler(entry: str) -> tuple[str, Any]:
    """``(dispatch key, handler)`` for one checklist entry; the handler is ``None`` if there is none.

    Separate from ``check_tolerance_asserts`` so that a test can ask this question of the entries
    ``scripts/measure_geom_tol.py`` ACTUALLY writes — parsed out of that file's source rather than
    copied into a fixture — and fail on the day the producer grows an entry this consumer would
    refuse. A hand-copied fixture of the producer's list is the same drift one level removed: it
    goes stale silently, and the first thing to notice is a refused gate run on the cluster.
    """
    token = entry.split()[0].strip(".,;:") if entry.split() else ""
    handler = CONSUMER_ASSERT_HANDLERS.get(token)
    if handler is None:
        for phrase, by_phrase in CONSUMER_ASSERT_PHRASE_HANDLERS:
            if phrase in entry:
                return phrase, by_phrase
    return token, handler


def step_frames_check(doc: dict[str, Any], label: str) -> tuple[bool | None, str]:
    """``step_frames`` must be the step G0b gates under, and G0b's step is not a choice.

    Checked whether or not the artifact declares the assertion, because it is the one field whose
    silent disagreement produces a WRONG BUDGET rather than a missing one: G0b compares source
    frame i to restyled frame i, GEOM_TOL scales roughly linearly with the step it was measured at,
    and a tolerance measured at 3 quoted here is a gate roughly three times too loose with nothing
    in the record to show for it. An artifact that does not state its step cannot be shown to be
    the right tolerance, which costs the run its gate qualification rather than refusing it — the
    committed pre-measurement contract legitimately has no step yet.
    """
    if "step_frames" not in doc:
        return None, (
            f"{label} does not record step_frames, so it cannot be shown to have been measured at "
            f"the {G0B_STEP_FRAMES}-frame step G0b gates under. GEOM_TOL scales roughly "
            "linearly with the step (measure_geom_tol.py's own note), so quoting a tolerance of "
            "unknown step is a gate of unknown width."
        )
    step = doc["step_frames"]
    try:
        step_int = int(step)
    except (TypeError, ValueError) as exc:
        raise GateRefusal(
            f"REFUSED: {label} records step_frames = {step!r}, which is not a frame count. G0b "
            f"gates under a {G0B_STEP_FRAMES}-frame step and cannot check an unreadable one."
        ) from exc
    if step_int != G0B_STEP_FRAMES:
        raise GateRefusal(
            f"REFUSED: {label} records step_frames = {step!r}, but G0b compares SOURCE frame i to "
            f"RESTYLED frame i — a {G0B_STEP_FRAMES}-frame step.\n"
            "        GEOM_TOL scales roughly linearly with the step it was measured at, so this "
            f"budget is about {step_int}x\n        too loose for the comparison it would be "
            "applied to. That is a silently widened pre-registered gate,\n        which is the one "
            "thing this runner exists to make impossible. Re-measure with --step-frames "
            f"{G0B_STEP_FRAMES}."
        )
    return True, f"step_frames == {G0B_STEP_FRAMES}, the step G0b compares under"


def check_tolerance_asserts(
    doc: dict[str, Any], path: Path, instrument: dict[str, Any], digest_ok: bool | None
) -> dict[str, Any]:
    """Honour every assertion the tolerance's producer declared, plus the ones it cannot skip.

    Returns the record block; raises ``GateRefusal`` for a violated assertion or an entry this
    consumer does not know how to check. Entries that are DECLARED and cannot be checked from this
    artifact alone come back as not-gate-qualified reasons: loud, in the record, and blocking — the
    one thing they are never is silent.
    """
    label = repo_rel(path)
    checked: list[dict[str, Any]] = []
    reasons: list[str] = []

    # Always, list or no list. See step_frames_check() for why this one cannot wait to be declared.
    ok, note = step_frames_check(doc, label)
    checked.append({"entry": "step_frames (always checked)", "checked": ok, "note": note})
    if ok is None:
        reasons.append(note)

    declared = doc.get("consumer_asserts")
    if declared is None:
        return {
            "declared_by_artifact": None,
            "checked": checked,
            "note": (
                f"{label} declares no consumer_asserts list. scripts/measure_geom_tol.py writes "
                "one into every artifact it produces; a tolerance without it was assembled by "
                "something else, and only the always-checked assertions above apply."
            ),
            "not_gate_qualified_reasons": reasons,
        }
    if not isinstance(declared, list) or not all(isinstance(e, str) for e in declared):
        raise GateRefusal(
            f"REFUSED: {label} carries consumer_asserts of shape {type(declared).__name__}, not a "
            "list of strings. This runner asserts every entry of that list and cannot assert a "
            "shape it does not recognise."
        )

    for entry in declared:
        token, handler = consumer_assert_handler(entry)
        if handler is None:
            raise GateRefusal(
                f"REFUSED: {label} declares a consumer assertion this runner has no handler for:\n"
                f"          {entry}\n"
                f"        dispatch token {token!r}; known: "
                + ", ".join(sorted(CONSUMER_ASSERT_HANDLERS))
                + "; phrases: "
                + ", ".join(repr(p) for p, _ in CONSUMER_ASSERT_PHRASE_HANDLERS)
                + "\n        The producer writes that list so the consumer cannot silently stop "
                "checking something. Implement\n        the handler in scripts/run_g0_gates.py, or "
                "the gate is quoting a number under a condition nobody\n        verified."
            )
        ctx = {"instrument": instrument, "digest_ok": digest_ok, "label": label, "entry": entry}
        ok, note = handler(doc, ctx)
        checked.append({"entry": entry, "token": token, "checked": ok, "note": note})
        if ok is None:
            reasons.append(
                "the tolerance declares a consumer assertion this run could not check: " + note
            )

    return {
        "declared_by_artifact": len(declared),
        "checked": checked,
        "not_gate_qualified_reasons": reasons,
    }


def contract_disagreements(ours: Any, theirs: Any, where: str) -> list[str]:
    """Field-for-field comparison of two segmenter contracts, on the fields they BOTH state.

    Fields only one side states are not disagreements: the committed contract is the longer
    document by design, and a side record that says less has not contradicted it. Fields both state
    and state differently are the failure this file exists to catch — a revision, a box threshold
    or a text prompt that moved between the tolerance and the gate makes two plausible pixel
    numbers that are not the same quantity.
    """
    out: list[str] = []
    if not isinstance(ours, dict) or not isinstance(theirs, dict):
        return out
    for key in sorted(set(ours) & set(theirs)):
        mine, yours = ours[key], theirs[key]
        if isinstance(mine, dict) and isinstance(yours, dict):
            out += contract_disagreements(mine, yours, f"{where}.{key}")
            continue
        if isinstance(mine, list) and isinstance(yours, list):
            mine, yours = list(mine), list(yours)
        if mine != yours:
            out.append(f"{where}.{key}: {mine!r} vs {yours!r}")
    return out


def _centroid_list(
    raw: Any, clip: str, label: str, side: str
) -> list[tuple[float, float] | None]:
    """Parse one clip/label centroid stream. ``null`` means "the object was not visible here"."""
    if not isinstance(raw, list):
        raise GateRefusal(
            f"REFUSED: {side} centroid record, clip {clip!r}, label {label!r}: expected a list of "
            f"[x, y] or null, got {type(raw).__name__}."
        )
    out: list[tuple[float, float] | None] = []
    for i, item in enumerate(raw):
        if item is None:
            out.append(None)
            continue
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise GateRefusal(
                f"REFUSED: {side} centroid record, clip {clip!r}, label {label!r}, frame {i}: "
                f"expected [x, y] or null, got {item!r}."
            )
        try:
            out.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError) as exc:
            raise GateRefusal(
                f"REFUSED: {side} centroid record, clip {clip!r}, label {label!r}, frame {i}: "
                f"non-numeric centroid {item!r}."
            ) from exc
    return out


def load_centroid_record(path: Path, side: str) -> Side:
    """Read one side's centroid artifact and refuse anything that cannot be gated on.

    A centroid stream with no segmenter and no grid is exactly as useful as a tolerance with none:
    it can be compared with anything and proves nothing. Both are required, present and non-null,
    before a single displacement is computed.
    """
    if not path.is_file():
        raise GateRefusal(
            f"REFUSED: --{side}-centroids {path} does not exist. Produce it with this script's "
            f"--{side}-clips path and a gate-qualified segmenter, or by writing the "
            f"{CENTROID_SCHEMA} record from wherever the masks were measured."
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateRefusal(f"REFUSED: {path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise GateRefusal(f"REFUSED: {path} is not a JSON object.")
    schema = doc.get("schema")
    if schema != CENTROID_SCHEMA:
        raise GateRefusal(
            f"REFUSED: {path} declares schema {schema!r}, not {CENTROID_SCHEMA!r}. This gate "
            "compares centroid streams field by field; a record of unknown shape cannot be one "
            "of them."
        )
    segmenter = doc.get("segmenter")
    if not isinstance(segmenter, dict) or not segmenter.get("name"):
        raise GateRefusal(
            f"REFUSED: {path} names no segmenter (segmenter.name). PR-08 §6 requires both sides "
            "to be measured with the same one, and a stream that cannot say what produced it "
            "cannot be shown to satisfy that."
        )
    grid = doc.get("resolution_hw")
    if not (isinstance(grid, (list, tuple)) and len(grid) == 2 and all(g for g in grid)):
        raise GateRefusal(
            f"REFUSED: {path} records resolution_hw = {grid!r}. Centroids are in pixels and the "
            "budget is in pixels; without the grid they were measured on, the comparison is not "
            "arithmetic. Absence is not agreement."
        )
    clips_raw = doc.get("clips")
    if not isinstance(clips_raw, dict) or not clips_raw:
        raise GateRefusal(f"REFUSED: {path} carries no clips. An empty comparison is not a pass.")

    clips: dict[str, dict[str, list[tuple[float, float] | None]]] = {}
    source_of: dict[str, str] = {}
    for clip, entry in clips_raw.items():
        if not isinstance(entry, dict):
            raise GateRefusal(
                f"REFUSED: {path} clip {clip!r} is {type(entry).__name__}, expected an object "
                "keyed by label ('object', 'plate', ...)."
            )
        parsed: dict[str, list[tuple[float, float] | None]] = {}
        for label, raw in entry.items():
            if label == "source_clip":
                continue
            parsed[label] = _centroid_list(raw, clip, label, side)
        if LABEL_OBJECT not in parsed:
            raise GateRefusal(
                f"REFUSED: {path} clip {clip!r} carries no {LABEL_OBJECT!r} centroids. G0b is "
                "about where the object is; there is nothing to gate without it."
            )
        clips[clip] = parsed
        source_of[clip] = str(entry.get("source_clip", clip))
    return Side(
        side=side,
        origin=repo_rel(path),
        segmenter=dict(segmenter),
        resolution_hw=[int(grid[0]), int(grid[1])],
        clips=clips,
        source_of=source_of,
        notes=list(doc.get("notes") or []),
        # ABSENT IS ALLOWED AND IS NOT AGREEMENT. Records written before this field existed carry
        # no decoder, and refusing them would retire every centroid record ever produced for a
        # property nobody could have stated. It costs the run its gate qualification instead, and
        # only when the OTHER side does state one — see run_g0b.
        decoder=doc["decoder"] if isinstance(doc.get("decoder"), dict) else None,
    )


def instrument_disagreements(
    source: Side, restyled: Side, tolerance_instrument: dict[str, Any]
) -> list[str]:
    """Every way the two sides (and the tolerance) were NOT measured with one instrument.

    Verified from the records, never assumed from the fact that one command line produced both.
    Two segmenters produce two quantities whose difference in pixels is a plausible number that
    means nothing, and two grids make the comparison against a pixel budget not arithmetic. This is
    the same join ``measure_est_drift.cross_check_geom_tol()`` enforces on the other end of the
    same subtraction, and it is enforced here for the same reason: nothing else in the pipeline
    does it, and a mismatch is invisible in the result.
    """
    out: list[str] = []
    if source.segmenter_name != restyled.segmenter_name:
        out.append(
            f"the two sides used different segmenters: source {source.segmenter_name!r} vs "
            f"restyled {restyled.segmenter_name!r}. Centroids from two segmenters are two "
            "quantities and their difference is not a geometry measurement."
        )
    if source.segmenter_version != restyled.segmenter_version:
        out.append(
            f"the two sides used different segmenter VERSIONS: source "
            f"{source.segmenter_version!r} vs restyled {restyled.segmenter_version!r}. The version "
            "is the pinned checkpoint set (ESTIMATOR_VERSION); different weights are a different "
            "segmenter wearing the same name."
        )
    # THE DECODER IS DELIBERATELY NOT COMPARED HERE. It is part of the instrument and it IS
    # checked — in ``decoder_disagreements``, whose rows cost the run its gate qualification rather
    # than refusing it. The reason for the asymmetry is written out there, and it is the same
    # reason a gate is not allowed to be unpassable.
    if list(source.resolution_hw) != list(restyled.resolution_hw):
        out.append(
            f"the two sides were measured on different pixel grids: source {source.resolution_hw} "
            f"vs restyled {restyled.resolution_hw} [height, width]. A displacement in pixels is "
            "not comparable across grids, and the budget is in pixels."
        )
    theirs = tolerance_instrument.get("name")
    if theirs and theirs != source.segmenter_name:
        out.append(
            f"the tolerance was measured with segmenter {theirs!r} but these centroids come from "
            f"{source.segmenter_name!r}. PR-08 §4 step 2: the SAME segmenter, because §6 subtracts "
            "the two numbers."
        )
    grid = tolerance_instrument.get("resolution_hw")
    if grid and list(grid) != list(source.resolution_hw):
        out.append(
            f"the tolerance was measured at {list(grid)} [height, width] and these centroids at "
            f"{source.resolution_hw}. GEOM_TOL - EST_DRIFT_P95 is a pixel budget on one grid."
        )
    # The pinned operating point, when either side states it. The committed contract exists because
    # "two runs can share a name while disagreeing about every number below" — a moved revision, a
    # different box threshold or a different text prompt is a different segmenter with the same name.
    contract = tolerance_instrument.get("contract")
    for side in (source, restyled):
        for row in contract_disagreements(side.segmenter.get("contract"), contract, "contract"):
            out.append(f"the {side.side} segmenter contract disagrees with the tolerance's: {row}")
    for row in contract_disagreements(
        source.segmenter.get("contract"), restyled.segmenter.get("contract"), "contract"
    ):
        out.append(f"the two sides' segmenter contracts disagree: {row}")
    return out


def decoder_disagreements(source: Side, restyled: Side) -> list[str]:
    """Everything the two records say about how their bytes became frames, and cannot join on.

    THE DECODER IS PART OF THE INSTRUMENT. Until 2026-08-23 nothing here looked at it, while the
    module docstring claimed the two sides are verified to have been measured with one. A segmenter
    is a function of the pixels it is handed: two decoders differ in colour conversion and, on a
    stream one of them mis-parses, in which frames exist at all — and G0b's arithmetic is source
    frame *i* minus restyled frame *i*. ``resolve_decoder`` probes each side's own bytes
    independently, so one command line can resolve two decoders with nothing in the record to show
    for it.

    **AND THESE ROWS DO NOT REFUSE, WHICH IS THE WHOLE DESIGN DECISION.** G0b's two sides are not
    the same codec by construction — the PR-08 source is av1 (job 189585 is the record of cv2
    decoding ZERO frames of it, *"Missing Sequence Header"*) and the generator's output is not — so
    "both sides must name one decoder" is a condition the real corpus may be unable to satisfy at
    all. A gate that cannot say yes blocks generation exactly as a wrong one would, and this file
    has already been repaired once for precisely that (``_ca_mask_method_name``, 2026-08-22). So a
    decoder difference is recorded as a loss of gate qualification — exit 3, "ran, nothing failed,
    may not stand as the gate" — which is this repository's own slot for a comparison that
    happened and cannot be shown to be one instrument.

    Whether §6 should REQUIRE one decoder across both sides, or only require that the pair be
    recorded, is a question about the rule and not about this runner. It is left to the owner and
    to a versioned document; if the answer is "required", this function's rows move into
    :func:`instrument_disagreements` and nothing else changes.

    Neither side stating a decoder produces nothing: every centroid record written before this
    field existed is silent, and silence about a property nobody could state is not a finding.
    """
    if source.decoder_id and restyled.decoder_id:
        if source.decoder_id == restyled.decoder_id:
            return []
        return [
            f"the two sides were decoded by different decoders: source {source.decoder_id!r} vs "
            f"restyled {restyled.decoder_id!r}. A segmenter is a function of the pixels it is "
            "handed, and two decoders hand it two different sets of pixels for the same file, so "
            "the displacement between them cannot be shown to be a geometry measurement. This is "
            "not refused because the two sides are not the same codec by construction (av1 source, "
            "generated output that is not av1), and a gate the real corpus cannot satisfy is not a "
            "gate."
        ]
    if bool(source.decoder_id) != bool(restyled.decoder_id):
        stated, silent = (source, restyled) if source.decoder_id else (restyled, source)
        return [
            f"the {stated.side} centroid record was decoded by {stated.decoder_id!r} and the "
            f"{silent.side} record states no decoder, so the two sides could not be shown to have "
            "been decoded by one. The --*-clips path states its decoder by construction, so a "
            "record that does not was written by something else."
        ]
    return []


def paired_displacements(
    src: list[tuple[float, float] | None],
    dst: list[tuple[float, float] | None],
    clip: str,
    label: str,
) -> tuple[np.ndarray, int]:
    """Per-frame Euclidean centroid displacement, and the frames that could not be measured.

    A frame is measurable only when BOTH sides found the object. Occlusion by the Dex3 hand and the
    apple leaving frame are real events in this corpus; folding them in as 0 px would pull every
    statistic down, which makes the gate look cleaner exactly where it saw nothing. Dropped and
    counted, as ``measure_geom_tol`` does for the same reason.

    A frame-count mismatch is NOT a droppable event and is refused: a restyle emits one frame per
    source frame, so a different length is proof that frames were dropped, duplicated or reordered
    — the pixel-side twin of the label defect G0a exists to catch, and comparing the two streams
    index by index after that would be comparing different moments in the episode.
    """
    if len(src) != len(dst):
        raise GateRefusal(
            f"REFUSED: clip {clip!r} label {label!r} has {len(src)} source frames and {len(dst)} "
            "restyled frames. A restyle emits one frame per source frame; a different count is "
            "proof that frames were dropped, duplicated or reordered, and index-by-index "
            "comparison after that compares different moments."
        )
    values: list[float] = []
    dropped = 0
    for a, b in zip(src, dst):
        if a is None or b is None:
            dropped += 1
            continue
        values.append(float(np.hypot(b[0] - a[0], b[1] - a[1])))
    out = np.asarray(values, dtype=float)
    # A NaN or an infinity is not a displacement, and it must not be allowed to become a verdict.
    # ``json.loads`` accepts the bare ``NaN`` literal, so a hand-written or corrupted centroid
    # record reaches here intact; ``np.percentile`` then returns NaN, ``NaN <= budget`` is False,
    # and the clip is reported as a VOID row -- a formal indictment of the generation pipeline
    # produced by an unreadable number. (The histogram behind it would then fail to bin, so the
    # run ends in a traceback rather than a verdict.) Refuse where the number enters instead.
    if out.size and not np.all(np.isfinite(out)):
        raise GateRefusal(
            f"REFUSED: clip {clip!r} label {label!r} yielded "
            f"{int(np.count_nonzero(~np.isfinite(out)))} non-finite displacement(s) of "
            f"{out.size}. A NaN or an infinity in a centroid stream is an unreadable record, not a "
            "geometry measurement, and a percentile taken over one compares False against the "
            "budget — which would be reported as VOID."
        )
    return out, dropped


def signed_distribution(values: np.ndarray, bin_px: float) -> dict[str, Any]:
    """``measure_geom_tol.distribution()``'s shape, for values that may be NEGATIVE.

    WHY THIS IS NOT ``geom.distribution``. That function is correct for what it was written for —
    displacements, which are non-negative — and it builds its bin edges as
    ``np.arange(0.0, top + bin_px, bin_px)``, from ZERO upward. Handed the signed margin
    ``budget - displacement`` it drops every negative value on the floor: ``np.histogram`` counts
    nothing outside its edges, so the frames that BLEW THE BUDGET — the only frames the margin
    distribution exists to characterise — vanish from ``counts`` while ``n`` still counts them. A
    reader who asks the histogram "how many frames were outside budget, and by how much" is told
    zero, in a pre-registered gate record. That is the class of wrong number that gets quoted
    downstream, and it is wrong exactly in the VOID case.

    The edges here are floored/ceiled around the ACTUAL range, so a distribution of negative
    margins is a histogram of negative margins. The producer's function is deliberately left
    untouched (it is another session's file, and it is not wrong for its own callers); the
    displacement histogram beside this one still goes through it, so the gate's displacements and
    the tolerance's displacements remain binned identically and can be read side by side.

    The self-check at the end is not defensive noise. ``counts`` failing to sum to ``n`` IS the bug
    this function was written to fix, and a gate record whose histogram silently disagrees with its
    own count is precisely what must not be written; so it refuses instead of writing one.
    """
    if values.size == 0:
        return {"n": 0}
    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    q = np.percentile(values, pcts)
    lo = float(np.floor(float(values.min()) / bin_px) * bin_px)
    hi = float(np.ceil(float(values.max()) / bin_px) * bin_px)
    if hi <= lo:  # every value identical and on a bin edge: one bin, not zero
        hi = lo + bin_px
    n_bins = max(1, int(round((hi - lo) / bin_px)))
    edges = lo + bin_px * np.arange(n_bins + 1, dtype=float)
    # Floating point: floor/ceil of a quotient can land a hair inside the data. Widen rather than
    # lose a sample, since losing samples is the whole defect being repaired.
    edges[0] = min(edges[0], float(values.min()))
    edges[-1] = max(edges[-1], float(values.max()))
    counts, edges = np.histogram(values, bins=edges)
    if int(counts.sum()) != int(values.size):
        raise GateRefusal(
            f"REFUSED: the margin histogram binned {int(counts.sum())} of {int(values.size)} "
            "values. A gate record whose histogram disagrees with its own n is a wrong number "
            "waiting to be quoted; this is a bug in signed_distribution(), not a finding."
        )
    return {
        "n": int(values.size),
        "min_px": float(values.min()),
        "max_px": float(values.max()),
        "mean_px": float(values.mean()),
        "std_px": float(values.std(ddof=0)),
        "percentiles_px": {f"p{p}": float(v) for p, v in zip(pcts, q)},
        "histogram": {
            "bin_px": float(bin_px),
            "bin_edges_px": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
            "signed": True,
        },
    }


def clip_margins(
    displacement: np.ndarray, budget: float, percentile: float
) -> dict[str, Any]:
    """The per-clip margin record: the gate statistic AND the distribution behind it.

    PR-08 §6 asks whether the centroids agree; a single boolean cannot distinguish "every frame is
    0.2 px off" from "one frame in a thousand is 40 px off and the rest are perfect", and those two
    are a healthy generator and a broken one. So the whole distribution of ``budget - displacement``
    is recorded per clip, and the pass/fail is one derived number on top of it.
    """
    if displacement.size == 0:
        return {
            "n_measured": 0,
            "gate_statistic_px": None,
            "margin_at_gate_statistic_px": None,
            "within_budget": None,
        }
    stat = float(np.percentile(displacement, percentile))
    margins = budget - displacement
    pcts = (0, 1, 5, 25, 50, 75, 95, 99, 100)
    return {
        "n_measured": int(displacement.size),
        "displacement_px": {
            "mean": float(displacement.mean()),
            "percentiles": {
                f"p{p}": float(v) for p, v in zip(pcts, np.percentile(displacement, pcts))
            },
        },
        "margin_px": {
            "min": float(margins.min()),
            "mean": float(margins.mean()),
            "max": float(margins.max()),
            "percentiles": {
                f"p{p}": float(v) for p, v in zip(pcts, np.percentile(margins, pcts))
            },
        },
        "frames_outside_budget": int(np.count_nonzero(displacement > budget)),
        "fraction_inside_budget": float(np.count_nonzero(displacement <= budget) / displacement.size),
        "gate_statistic_px": stat,
        "margin_at_gate_statistic_px": float(budget - stat),
        "within_budget": bool(stat <= budget),
    }


def adapter_segmenter_contract(method: Any) -> tuple[dict[str, Any] | None, str]:
    """The PINNED OPERATING POINT of the estimator ``measure_geom_tol`` just selected.

    ``MaskMethod.params`` carries the adapter's spec, its file and its checkpoints, but NOT the
    thresholds, the prompt, the box rule or the propagation mode — so a side measured through the
    ``--*-clips`` path used to reach ``instrument_disagreements`` with nothing to compare against
    the committed contract, and the operating-point check quietly did not happen on the ONE path
    that actually measures. The adapter exports the whole thing as ``SEGMENTER_CONTRACT`` for
    exactly this purpose ("a constant that is only in the code cannot be cross-checked by a script
    reading two JSON artifacts six months later"), and the committed tolerance's ``segmenter``
    block is a verbatim copy of it. So it is read off the module that was actually loaded.

    THIS IS DELIBERATELY NOT A HELPER IN ``measure_geom_tol``. That file is under concurrent
    rewrite by another session and its ``params`` shape is theirs to choose; reaching through the
    two keys it already publishes (``estimator_module_file``, ``estimator_spec``) and reading a
    constant off the adapter keeps this consumer's requirement in this consumer's file.

    Returns ``(contract, note)``. A ``None`` contract is not fatal here — it is recorded, and the
    absence costs the run its gate qualification in ``run_g0b``, the same as any other side that
    cannot state what produced it.
    """
    params = getattr(method, "params", None) or {}
    rel = params.get("estimator_module_file")
    spec = params.get("estimator_spec")
    if not rel:
        return None, (
            f"the mask method {getattr(method, 'name', None)!r} publishes no "
            "estimator_module_file, so this runner cannot find an adapter to read "
            "SEGMENTER_CONTRACT off (--method precomputed, or a method with no adapter behind it)"
        )
    path = _REPO_ROOT / str(rel)
    if not path.is_file():
        return None, f"{rel} does not exist, so its SEGMENTER_CONTRACT could not be read"
    try:
        module_spec = importlib.util.spec_from_file_location(f"_g0b_adapter_{path.stem}", path)
        if module_spec is None or module_spec.loader is None:
            return None, f"{rel} could not be loaded as a module"
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    except Exception as exc:  # a contract we cannot read is recorded, never guessed
        return None, f"{rel} could not be imported to read SEGMENTER_CONTRACT: {exc!r}"
    contract = getattr(module, "SEGMENTER_CONTRACT", None)
    if not isinstance(contract, dict):
        return None, (
            f"{rel} exports no SEGMENTER_CONTRACT dict, so the operating point it ran at is not "
            "stated anywhere this runner can compare against the committed contract"
        )
    return dict(contract), f"read from {rel} (adapter {spec})"


def load_source_map(path: Path, side: str) -> dict[str, str]:
    """``{restyled clip key: source clip key}``, for the many-to-one pairing a restyle produces.

    ONE SOURCE CLIP BECOMES MANY RESTYLED CLIPS — 25 style-instances over 402 source clips in
    ``97_transfer25_restyle.sbatch`` — so the restyled side cannot be paired with the source by
    equal names, and guessing a suffix convention here would be this runner inventing the pairing
    the corpus is supposed to declare. Without this file the ``--restyled-clips`` path can only
    assume identity, which is correct for a 1:1 tree and produces a wall of "names a source clip
    the source record does not carry" for a real one.
    """
    if not path.is_file():
        raise GateRefusal(
            f"REFUSED: --{side}-source-map {path} does not exist. It maps each restyled clip key "
            "to the SOURCE clip key it is a restyle of."
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateRefusal(f"REFUSED: --{side}-source-map {path} is not valid JSON: {exc}") from exc
    if isinstance(doc, dict) and isinstance(doc.get("source_of"), dict):
        doc = doc["source_of"]
    if not isinstance(doc, dict) or not doc:
        raise GateRefusal(
            f"REFUSED: --{side}-source-map {path} is not a non-empty object of "
            '{"restyled_clip_key": "source_clip_key"} (a top-level "source_of" object is also '
            "accepted, so a record written by something else can be handed over whole)."
        )
    out: dict[str, str] = {}
    for key, value in doc.items():
        if not isinstance(value, str) or not value:
            raise GateRefusal(
                f"REFUSED: --{side}-source-map {path}: clip {key!r} maps to {value!r}, not to a "
                "source clip key."
            )
        out[str(key)] = value
    return out


def side_from_clips(args: argparse.Namespace, side: str, corpus: Path) -> Side:
    """Measure one side's centroids from clips, through ``measure_geom_tol``'s own machinery.

    Nothing about segmentation, decoding or centroids is implemented here. ``resolve_method``
    selects (or refuses) the segmenter, ``resolve_decoder`` probes a decoder against these actual
    bytes (a container that parses is not a codec that decodes — job 189585), and
    ``episode_centroids_from_video`` produces the stream. Two implementations of "where is the
    apple" is exactly the drift this gate exists to catch, so there is exactly one.

    ONE LABEL PER RUN, and it is recorded rather than papered over: the shared adapter takes its
    object prompt from ``WAM_PR08_OBJECT_PROMPT`` at import time, so the plate is a second
    invocation with that variable set, not a second call here. A run that measured only the object
    cannot be gate-qualified for §6's "object AND plate", and the artifact says so.
    """
    geom = load_script("measure_geom_tol")
    ns = argparse.Namespace(
        method=args.method,
        masks=args.masks,
        min_area_px=args.min_area_px,
    )
    try:
        method = geom.resolve_method(ns)
        episodes, layout = geom.find_episodes(corpus, args.camera_key)
    except geom.MethodUnavailable as exc:
        raise GateRefusal(f"REFUSED: {side} centroids could not be measured.\n{exc}") from exc

    if args.limit:
        episodes = episodes[: args.limit]
    clips: dict[str, dict[str, list[tuple[float, float] | None]]] = {}
    grid: tuple[int, int] | None = None
    try:
        probe = next((ep.clip for ep in episodes if ep.clip is not None), None)
        if probe is None:
            raise geom.MethodUnavailable(
                f"FATAL: no clip found under {corpus}; there is nothing to segment."
            )
        decoder = geom.resolve_decoder(args.decoder, probe)
        for ep in episodes:
            cents, size, _fps = geom.episode_centroids_from_video(
                ep.clip, method, args.min_area_px, args.max_frames, decoder
            )
            if grid is None:
                grid = size
            elif grid != size:
                raise geom.MethodUnavailable(
                    f"FATAL: {ep.key} is {size[0]}x{size[1]} but the first clip was "
                    f"{grid[0]}x{grid[1]}. G0b's budget is in pixels on ONE grid."
                )
            clips[ep.key] = {args.object_label: cents}
    except geom.MethodUnavailable as exc:
        raise GateRefusal(f"REFUSED: {side} centroids could not be measured.\n{exc}") from exc
    if grid is None:
        raise GateRefusal(f"REFUSED: {side} clips under {corpus} yielded no frames.")

    contract, contract_note = adapter_segmenter_contract(method)
    segmenter: dict[str, Any] = {
        "name": method.name,
        "version": method.version,
        "gate_qualified": method.gate_qualified,
        "provenance": method.provenance or None,
        "contract_source": contract_note,
    }
    if contract is not None:
        segmenter["contract"] = contract

    # THE PAIRING IS DECLARED, NEVER GUESSED. Identity by clip name is right for a 1:1 tree (the
    # source side, and a restyled tree that kept its source's keys) and wrong for the corpus this
    # will actually be pointed at, where one source clip becomes many restyled clips. So identity
    # is what happens when nobody says otherwise, it is RECORDED as an assumption in the side's
    # notes, and --restyled-source-map is the way to state the real thing.
    mapping = getattr(args, f"{side}_source_map", None)
    notes = [
        f"measured in this process from {corpus} with --method {args.method}",
        f"one label only ({args.object_label}); PR-08 §6 gates object AND plate, and the "
        "adapter takes one text prompt per process (WAM_PR08_OBJECT_PROMPT)",
        f"segmenter contract: {contract_note}",
    ]
    if mapping is not None:
        source_of = load_source_map(mapping, side)
        unmapped = sorted(set(clips) - set(source_of))
        if unmapped:
            raise GateRefusal(
                f"REFUSED: --{side}-source-map {mapping} does not name a source clip for "
                f"{len(unmapped)} of the {len(clips)} clips measured under {corpus}: "
                + ", ".join(unmapped[:5])
                + ("..." if len(unmapped) > 5 else "")
                + ".\n        A clip with no declared source cannot be compared to anything, and "
                "this runner will not fall back to\n        pairing it by name — that is the "
                "guess the map exists to replace."
            )
        source_of = {key: source_of[key] for key in clips}
        notes.append(f"pairing declared by --{side}-source-map {repo_rel(mapping)}")
    else:
        source_of = {key: key for key in clips}
        notes.append(
            "PAIRING ASSUMED BY NAME: no --{0}-source-map was given, so each clip is taken to be "
            "a restyle of the source clip with the SAME key. That holds for a 1:1 tree and is "
            "false for a real restyled corpus (one source clip, many styles and repeats), where "
            "it surfaces as every clip naming a source the source record does not carry.".format(
                side
            )
        )

    return Side(
        side=side,
        origin=f"{corpus} ({layout}, decoded by {decoder.name} {decoder.version})",
        segmenter=segmenter,
        resolution_hw=[grid[1], grid[0]],
        clips=clips,
        source_of=source_of,
        notes=notes,
        decoder={"name": decoder.name, "version": decoder.version},
    )


def write_side_record(side: Side, path: Path) -> None:
    """Persist a measured side so the next run does not have to re-decode 402 clips to re-gate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CENTROID_SCHEMA,
        "side": side.side,
        "origin": side.origin,
        "segmenter": side.segmenter,
        # Stated as its own field and not only inside `origin`'s prose, because the consumer
        # COMPARES it: a decoder named in a sentence is a decoder nothing can join on.
        "decoder": side.decoder,
        "resolution_hw": side.resolution_hw,
        "notes": side.notes,
        "clips": {
            clip: {
                **{
                    label: [None if c is None else [c[0], c[1]] for c in stream]
                    for label, stream in labels.items()
                },
                "source_clip": side.source_of.get(clip, clip),
            }
            for clip, labels in side.clips.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_side(args: argparse.Namespace, side: str) -> Side:
    """``--<side>-centroids`` if given, else ``--<side>-clips``, else refuse naming both."""
    record = getattr(args, f"{side}_centroids")
    clips = getattr(args, f"{side}_clips")
    if record is not None and clips is not None:
        raise GateRefusal(
            f"REFUSED: --{side}-centroids and --{side}-clips both name a source of {side} "
            "centroids and only one of them produced the numbers this gate would report. Pick one."
        )
    if record is not None:
        return load_centroid_record(record, side)
    if clips is not None:
        measured = side_from_clips(args, side, clips)
        if args.dump_centroids is not None:
            write_side_record(measured, args.dump_centroids / f"centroids-{side}.json")
        return measured
    raise GateRefusal(
        f"REFUSED: G0b has no {side} centroids. Pass --{side}-centroids <{CENTROID_SCHEMA} "
        f"record> or --{side}-clips <corpus> (which needs a gate-qualified segmenter on this "
        "machine — see scripts/measure_geom_tol.py's refusal for what that costs)."
    )


def run_g0b(args: argparse.Namespace) -> dict[str, Any]:
    """Geometry invariance against ``GEOM_TOL - EST_DRIFT_P95``. Refuses far more often than it runs."""
    geom = load_script("measure_geom_tol")
    doc = load_geom_config(args.geom_config)
    budget_block = gate_budget(doc, args.geom_config)
    budget = float(budget_block["gate_margin_px"])
    instrument = config_instrument(doc, args.geom_config)

    if instrument.get("gate_qualified") is False:
        raise GateRefusal(
            f"REFUSED: {repo_rel(args.geom_config)} says gate_qualified=false, so the tolerance it "
            "carries is not usable as a gate. Reasons the measurement itself recorded: "
            + "; ".join(doc.get("gate_disqualified_reasons") or ["(none recorded)"])
        )

    digest_ok, digest_note = committed_digest(args.geom_config)
    # The producer's own checklist, before a single displacement is computed: a tolerance that
    # fails one of the conditions its own artifact says a consumer must assert is not a tolerance
    # this gate may quote, and finding that out after decoding 402 clips helps nobody.
    asserts_block = check_tolerance_asserts(doc, args.geom_config, instrument, digest_ok)

    source = resolve_side(args, "source")
    restyled = resolve_side(args, "restyled")
    disagreements = instrument_disagreements(source, restyled, instrument)
    if disagreements:
        raise GateRefusal(
            "REFUSED: the two sides of G0b were not measured with one instrument, so their "
            "difference in pixels is not a geometry measurement:\n"
            + "\n".join(f"        - {d}" for d in disagreements)
        )

    per_clip: list[dict[str, Any]] = []
    pooled: dict[str, list[np.ndarray]] = {}
    dropped_by_label: dict[str, int] = {}
    total_by_label: dict[str, int] = {}
    missing_source: list[str] = []
    void_rows: list[str] = []

    for clip in sorted(restyled.clips):
        src_key = restyled.source_of.get(clip, clip)
        if src_key not in source.clips:
            missing_source.append(f"{clip} -> {src_key}")
            continue
        src_labels = source.clips[src_key]
        dst_labels = restyled.clips[clip]
        shared = sorted(set(src_labels) & set(dst_labels))
        one_sided = sorted(set(src_labels) ^ set(dst_labels))
        if one_sided:
            raise GateRefusal(
                f"REFUSED: clip {clip!r} carries labels {sorted(dst_labels)} and its source "
                f"{src_key!r} carries {sorted(src_labels)}. {one_sided} exist on one side only, "
                "so they cannot be compared and cannot be silently dropped either."
            )
        entry: dict[str, Any] = {"clip": clip, "source_clip": src_key, "labels": {}}
        for label in shared:
            disp, dropped = paired_displacements(
                src_labels[label], dst_labels[label], clip, label
            )
            stats = clip_margins(disp, budget, args.g0b_percentile)
            entry["labels"][label] = {
                "n_frames": len(src_labels[label]),
                "n_dropped_object_not_visible": dropped,
                **stats,
            }
            pooled.setdefault(label, []).append(disp)
            dropped_by_label[label] = dropped_by_label.get(label, 0) + dropped
            total_by_label[label] = total_by_label.get(label, 0) + len(src_labels[label])
            if stats["within_budget"] is False:
                void_rows.append(
                    f"{clip}/{label}: p{args.g0b_percentile:g} displacement "
                    f"{stats['gate_statistic_px']:.3f} px > budget {budget:.3f} px "
                    f"({stats['frames_outside_budget']} of {stats['n_measured']} frames outside)"
                )
        per_clip.append(entry)

    if missing_source:
        assumed = "PAIRING ASSUMED BY NAME" in " ".join(restyled.notes)
        raise GateRefusal(
            "REFUSED: "
            + f"{len(missing_source)} of {len(restyled.clips)} restyled clip(s) name a source clip "
            "that the source record does not carry, so they cannot be compared to anything: "
            + ", ".join(missing_source[:5])
            + ("..." if len(missing_source) > 5 else "")
            + (
                "\n        The restyled side was paired BY NAME because no --restyled-source-map "
                "was given. One source clip\n        becomes many restyled clips (styles, "
                "repeats), so on a real restyled corpus that assumption fails\n        for every "
                "clip — which is what this looks like. Declare the pairing with "
                "--restyled-source-map\n        (a JSON object mapping each restyled clip key to "
                "its source clip key)."
                if assumed
                else ""
            )
        )
    if not per_clip:
        raise GateRefusal(
            "REFUSED: no restyled clip was compared against a source clip. An empty comparison is "
            "not a pass."
        )

    by_label: dict[str, Any] = {}
    n_measured_all = 0
    n_dropped_all = 0
    for label, arrays in sorted(pooled.items()):
        values = np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
        dropped = dropped_by_label.get(label, 0)
        total = int(values.size + dropped)
        n_measured_all += int(values.size)
        n_dropped_all += dropped
        margins = budget - values if values.size else values
        by_label[label] = {
            "n_frames": total_by_label.get(label, 0),
            "n_measured": int(values.size),
            "n_dropped_object_not_visible": dropped,
            "coverage": float(values.size / total) if total else 0.0,
            "worst_displacement_px": float(values.max()) if values.size else None,
            "worst_margin_px": float(margins.min()) if values.size else None,
            # NOT the gate statistic, and no longer named like it. The verdict is decided PER CLIP
            # (per_clip[*].labels[*].gate_statistic_px, via clip_margins -> void_rows); this is the
            # same percentile taken over every clip POOLED, and percentiles do not compose -- at
            # --g0b-percentile 100 the two coincide (the max of maxes is the max) and at anything
            # lower they can disagree in either direction. Two quantities under one name, one of
            # which gates and one of which does not, is a reader reconciling a VOID against a
            # number that is inside the budget with nothing in the artifact to explain it.
            "pooled_statistic_px": (
                float(np.percentile(values, args.g0b_percentile)) if values.size else None
            ),
            "pooled_statistic_note": (
                f"p{args.g0b_percentile:g} over ALL clips pooled — a summary, never the verdict. "
                "The gate is decided per clip; see per_clip[*].labels[*].gate_statistic_px."
            ),
            # The FULL distribution, from measure_geom_tol.distribution() -- the same function that
            # recorded the source corpus's own displacement distribution, so the gate's numbers and
            # the tolerance's numbers are histogrammed identically and can be read side by side.
            # Displacements are non-negative, which is what that function bins correctly.
            "displacement_distribution": geom.distribution(values, args.hist_bin_px),
            # MARGINS ARE SIGNED and a negative margin is a frame that blew the budget, so this one
            # cannot go through the same function: see signed_distribution()'s docstring.
            "margin_distribution": signed_distribution(margins, args.hist_bin_px)
            if values.size
            else {"n": 0},
        }

    coverage = float(n_measured_all / (n_measured_all + n_dropped_all)) if (
        n_measured_all + n_dropped_all
    ) else 0.0

    # Per-clip margin distribution ACROSS clips: a gate that clears at the median clip and fails at
    # the worst one is a different fact from one that clears everywhere, and only this row shows it.
    worst_per_clip = [
        entry["labels"][label]["margin_px"]["min"]
        for entry in per_clip
        for label in entry["labels"]
        if entry["labels"][label].get("margin_px")
    ]
    across_clips = (
        {
            f"p{p}": float(v)
            for p, v in zip(
                (0, 5, 25, 50, 75, 95, 100), np.percentile(worst_per_clip, (0, 5, 25, 50, 75, 95, 100))
            )
        }
        if worst_per_clip
        else {}
    )

    disqualified: list[str] = list(asserts_block["not_gate_qualified_reasons"])
    # The pinned operating point, when the TOLERANCE states one and a side does not. Absence is not
    # agreement: instrument_disagreements() can only compare fields both sides state, so a side
    # record carrying nothing under `contract` passes that comparison by saying nothing — the
    # default-permissiveness this repository has removed twice elsewhere. It is not a refusal
    # (name, version and grid were compared, which is a real check), but it is not the gate either.
    # The --*-clips path does not land here: it reads the operating point off the adapter that was
    # actually run.
    tolerance_contract = instrument.get("contract") or {}
    for side in (source, restyled):
        if tolerance_contract and not (side.segmenter.get("contract") or {}):
            disqualified.append(
                f"the {side.side} centroid record states no segmenter contract, so the tolerance's "
                f"pinned operating point ({', '.join(sorted(tolerance_contract))}) could not be "
                "compared against what produced these centroids. Name, version and pixel grid "
                "were; 'two runs can share a name while disagreeing about every number below' is "
                "the committed contract's own sentence about why that is not enough."
            )
    # The decoder, which is part of the instrument and does NOT refuse — see decoder_disagreements
    # for why a hard refusal here would make G0b unpassable on a corpus whose two sides are not the
    # same codec, which is the defect this file was already repaired for once.
    disqualified.extend(decoder_disagreements(source, restyled))
    for label in LABELS_GATED:
        if label not in by_label:
            disqualified.append(
                f"PR-08 §6 G0b gates object AND plate centroids; {label!r} was never measured, so "
                "half the gate did not run."
            )
    if coverage < args.min_coverage:
        disqualified.append(
            f"coverage {coverage:.3f} < --min-coverage {args.min_coverage}: the object was "
            "invisible on one side or the other for too much of the corpus for the rest to "
            "describe it."
        )
    for side in (source, restyled):
        if side.segmenter.get("gate_qualified") is not True:
            disqualified.append(
                f"the {side.side} centroids came from segmenter "
                f"{side.segmenter.get('name')!r} which does not declare gate_qualified=true. An "
                "unstated claim is not a claim."
            )
    # The committed schema carries no gate_qualified field (it commits the METHOD, before either
    # number is measured), so its absence cannot disqualify on its own without making G0b
    # unpassable by construction. What it does carry is where each number came from, and a
    # tolerance whose source is null is a number with no measurement behind it — which is the same
    # question asked in the field the schema actually has.
    if digest_ok is None:
        disqualified.append(
            f"{repo_rel(args.geom_config)}: {digest_note}. The sbatch refuses this outright; here "
            "it means the tolerance cannot be shown to be the committed one."
        )
    for key in ("geom_tol_source", "est_drift_source"):
        if budget_block.get(key) in (None, ""):
            disqualified.append(
                f"{repo_rel(args.geom_config)} records {key} = null, so the artifact does not say "
                "which measurement produced that half of the budget. A number with no stated "
                "source cannot be shown to be the committed one."
            )
    # STRUCTURAL, derived from the flags rather than from the numbers those flags produced —
    # measure_geom_tol.py's rule, for its reason: coverage is a fraction of what was COMPARED, so a
    # sample of three clips reports coverage 1.000 and every other field reads like a finished gate.
    # Both flags disqualify even on the --*-centroids path, where they truncate nothing: a run that
    # ASKED to be truncated is not the committed gate, and a gate record is not the place to reason
    # about which code path happened to honour the request.
    if args.limit:
        disqualified.append(
            f"--limit {args.limit} was requested. It truncates the --*-clips path to that many "
            "clips; on precomputed centroid records it truncates nothing. Either way this is a "
            "smoke test and not the gate, and coverage cannot detect it."
        )
    if args.max_frames:
        disqualified.append(
            f"--max-frames {args.max_frames} was requested. On the --*-clips path every clip was "
            "truncated, so the phases past that point — typically the transfer, which is where the "
            "geometry moves — were never seen; on precomputed records it truncates nothing and "
            "disqualifies for the same reason --limit does."
        )

    verdict = "VOID" if void_rows else ("NOT_GATE_QUALIFIED" if disqualified else "PASS")
    return {
        "gate": "G0b",
        "what": "geometry invariance — restyled object/plate centroids agree with the source",
        "verdict": verdict,
        "measured_verdict": "VOID" if void_rows else "PASS",
        "budget": budget_block,
        "tolerance_instrument": instrument,
        "tolerance_digest": {"verified": digest_ok, "note": digest_note},
        "tolerance_consumer_asserts": asserts_block,
        "criterion": {
            "statistic": (
                f"p{args.g0b_percentile:g} of the per-frame centroid displacement, TAKEN PER CLIP. "
                "A clip whose statistic exceeds the budget is a VOID row; by_label carries the "
                "same percentile pooled over all clips as pooled_statistic_px, which summarises "
                "and does not gate."
            ),
            "percentile": args.g0b_percentile,
            "rule": f"displacement <= {budget} px (GEOM_TOL - EST_DRIFT_P95)",
            "note": (
                "PR-08 §6 says the centroids must AGREE and does not name a statistic. The default "
                "here is p100 — every measured frame inside the budget — because 'agree' read as "
                "'usually agree' would be a coined threshold. The choice is a flag and is recorded "
                "either way."
            ),
        },
        "instrument_verified": {
            "source": {
                "segmenter": source.segmenter,
                "decoder": source.decoder,
                "resolution_hw": source.resolution_hw,
                "origin": source.origin,
            },
            "restyled": {
                "segmenter": restyled.segmenter,
                "decoder": restyled.decoder,
                "resolution_hw": restyled.resolution_hw,
                "origin": restyled.origin,
            },
            "checked": (
                "segmenter name, segmenter version, pixel grid — on both sides and against the "
                "committed tolerance's own instrument. Verified from the records, not assumed "
                "from one command line having produced both. Any of these disagreeing REFUSES."
            ),
            "decoder_checked": (
                "The video decoder is compared wherever both sides state it, and a difference is "
                "recorded in not_gate_qualified_reasons rather than refused: G0b's two sides are "
                "not the same codec by construction (av1 source, generated output that is not "
                "av1), so requiring one decoder could make this gate unpassable on the real "
                "corpus. The --*-clips path states its decoder by construction; a record written "
                "before 2026-08-23 states none, and two silent records produce nothing here."
            ),
            "operating_point_checked": (
                "The pinned operating point below the name (checkpoint revisions, text prompt, "
                "both threshold pairs, box rule, propagation) is compared field for field wherever "
                "a side STATES it. A side that states none is recorded in "
                "not_gate_qualified_reasons rather than passed: absence is not agreement. On the "
                "--*-clips path the operating point is read off the estimator adapter that was "
                "actually run (SEGMENTER_CONTRACT), so the measuring path states it by "
                "construction."
            ),
        },
        "n_clips_compared": len(per_clip),
        "labels_measured": sorted(by_label),
        "labels_gated_by_pr08": list(LABELS_GATED),
        "coverage": coverage,
        "min_coverage": args.min_coverage,
        "by_label": by_label,
        "worst_margin_across_clips_px": across_clips,
        "per_clip": per_clip,
        "void_rows": void_rows,
        "not_gate_qualified_reasons": disqualified,
        "notes": [
            "GEOM_TOL is derived from the OBJECT centroid alone (measure_geom_tol.py's own "
            "caveat), so the same number applied to the near-static plate is loose for it. PR-08 "
            "§6 is silent on the split and is registered, so this is recorded rather than resolved.",
            "Frames where either side found no object are DROPPED AND COUNTED, never folded in as "
            "zero displacement: a zero would pull every statistic down exactly where the gate saw "
            "nothing.",
            "A pass here closes PR-08 §8's geometry half and licenses nothing else. §1 forbids "
            "generation until every §8 item is closed and T-39 has reported.",
        ],
    }


# ==================================================================================== G0c, recorded


def g0c_record() -> dict[str, Any]:
    """G0c is not evaluated here, and saying nothing about it would read as having evaluated it.

    §6 solves G0c by construction: the real robot's pixels are unconditionally composited back over
    every generated frame using the robot segmentation mask, so there is no threshold and nothing
    to run. ``video_fidelity`` provably cannot see the generic-manipulator defect and any IoU bar
    on the robot mask would be a coined number — which is why §6 chose construction over a gate.
    That is a property of the GENERATION step, enforced where the compositing happens, not here.
    """
    return {
        "gate": "G0c",
        "verdict": "NOT_EVALUATED_HERE",
        "what": "embodiment — the real robot's pixels composited back over every generated frame",
        "why": (
            "PR-08 §6 solves G0c by construction rather than by a threshold: the composite is "
            "unconditional, so there is nothing for a gate runner to decide. It is enforced in the "
            "generation step (cluster/discoverer/97_transfer25_restyle.sbatch, "
            "scripts/restyle_transfer25.py), and robot-mask IoU is recorded there as a diagnostic "
            "on the generator, never as a gate."
        ),
        "consequence": (
            "This artifact covers G0a and G0b only. A reader who needs all three of §6's VOID "
            "gates must look at the generation record for G0c; this file must not be quoted as "
            "'the G0 gates passed'."
        ),
    }


# ============================================================================================== CLI


@dataclass(frozen=True)
class InputSpec:
    """One thing a gate needs, whether it is here, and what would produce it."""

    gate: str
    name: str
    path: Path | None
    produced_by: str

    @property
    def present(self) -> bool:
        return self.path is not None and self.path.exists()


def inventory(args: argparse.Namespace, gates: list[str]) -> list[InputSpec]:
    """Every input the requested gates will read, in the order they will be read."""
    out: list[InputSpec] = []
    if "g0a" in gates:
        out.append(
            InputSpec(
                "G0a",
                "--source-screen (already measured source screen)"
                if args.source_screen
                else "--source-dataset (the SOURCE corpus)",
                args.source_screen or args.source_dataset,
                "scripts/screen_corpus.py --dataset <source> --out <artifact>; or point "
                "--source-dataset at the canonical episode tree the restyle was derived from",
            )
        )
        out.append(
            InputSpec(
                "G0a",
                "--restyled-dataset (the generated corpus, canonical episodes)",
                args.restyled_dataset,
                "cluster/discoverer/97_transfer25_restyle.sbatch, then "
                "scripts/assemble_restyled_lerobot.py + scripts/convert_lerobot_g1.py. NO "
                "RESTYLED CORPUS EXISTS. Generated frames now DO exist — job 189926 produced 4 "
                "clips / 384 frames on 2026-08-23 — but they are the V8 hallucination probe's "
                "QUARANTINED output (every clip suffixed .mp4.quarantined), and PR-08 V8 §4 "
                "forbids anything downstream from consuming them as a corpus, so they are not "
                "this input and may not be assembled into it. PR-08 §1 forbids the licensed "
                "generation run 'until every item in §8 is closed and T-39 has reported'; T-39 "
                "reported on 2026-08-16 and §8 items 3 and 4 are open, so §1 still forbids it",
            )
        )
        if args.holdout is not None:
            out.append(
                InputSpec(
                    "G0a",
                    "--holdout (the episode split BOTH screens are scored on)",
                    args.holdout,
                    "configs/splits/t18_holdout_episodes.txt, committed",
                )
            )
    if "g0b" in gates:
        out.append(
            InputSpec(
                "G0b",
                "--geom-config: GEOM_TOL and EST_DRIFT_P95",
                args.geom_config,
                "scripts/measure_geom_tol.py writes GEOM_TOL to this path (+ .sha256); "
                "scripts/measure_est_drift.py measures EST_DRIFT_P95 (PR-08 §4 steps 1-4, which "
                "need the Isaac ground-truth capture from §4 step 0). PR-08 §8 item 4",
            )
        )
        for side in ("source", "restyled"):
            record = getattr(args, f"{side}_centroids")
            clips = getattr(args, f"{side}_clips")
            out.append(
                InputSpec(
                    "G0b",
                    f"--{side}-centroids ({CENTROID_SCHEMA})"
                    if record or not clips
                    else f"--{side}-clips (segmented in this process)",
                    record or clips,
                    f"this script's --{side}-clips path, which needs a gate-qualified segmenter on "
                    "this machine (scripts/estimators/apple_sam2.py). The checkpoints being "
                    "STAGED is no longer the blocker — job 189583 staged all three at the pinned "
                    "revisions; the adapter's GATE_QUALIFICATION_BLOCKERS tuple is, and it is the "
                    "single place that answers whether this path may produce a gate number",
                )
            )
    return out


def explain(args: argparse.Namespace, gates: list[str]) -> int:
    """Print exactly which inputs are missing and what would produce them.

    Exit 0 only when every input is present AND the budget actually forms. Those are two different
    facts — a committed tolerance file exists today and both of its numbers are null — and a dry run
    that reported "all present" over a budget that cannot be computed would be the most misleading
    thing this mode could print.

    This is the mode this script will spend most of its life in, and that is not a defect: PR-08
    §8's list has seven items, four of them are still open, and the two constants G0b needs have
    never been measured. A gate runner that could only either pass or crash would make that state
    look like a bug in the runner. It is the state of the project, and printing it precisely is
    more useful than a traceback.
    """
    specs = inventory(args, gates)
    print(f"PR-08 §6 G0 gates — dry run over {', '.join(g.upper() for g in gates)}")
    print(f"  repository: {_REPO_ROOT}")
    print()
    width = max(len(s.name) for s in specs) if specs else 0
    missing: list[InputSpec] = []
    for spec in specs:
        mark = "present" if spec.present else ("MISSING" if spec.path else "NOT GIVEN")
        print(f"  [{spec.gate}] {spec.name:<{width}}  {mark:>9}  {spec.path or '-'}")
        if not spec.present:
            missing.append(spec)

    if missing:
        print(f"\n{len(missing)} input(s) missing. What would produce each:")
        for spec in missing:
            print(f"\n  [{spec.gate}] {spec.name}")
            print(f"        path: {spec.path or '(not given on the command line)'}")
            for line in spec.produced_by.split("; "):
                print(f"        <- {line}")

    # ALWAYS probed when the file is there, present inputs or not: "the config exists" and "its two
    # numbers are non-null and subtract to something positive" are different facts, and only the
    # second decides whether G0b can run. A dry run that reported every file present while the
    # budget could not be formed would be the most misleading thing this mode could print.
    budget_blocked: str | None = None
    if "g0b" in gates and args.geom_config.is_file():
        try:
            doc = load_geom_config(args.geom_config)
            budget = gate_budget(doc, args.geom_config)
            print(
                f"\n  G0b budget: GEOM_TOL {budget['geom_tol_px']} px - EST_DRIFT_P95 "
                f"{budget['est_drift_p95_px']} px = {budget['gate_margin_px']:.3f} px"
            )
        except GateRefusal as exc:
            budget_blocked = str(exc)
            print(f"\n  G0b budget cannot be formed:\n{budget_blocked}")

    if not missing and budget_blocked is None:
        print("\nevery input the requested gates need is present; drop --explain to run them.")
        return EXIT_PASS
    print(
        "\nNothing here licenses generation. PR-08 §1 forbids it until every §8 item is closed and "
        "T-39 has reported."
    )
    return EXIT_REFUSED


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--gates",
        default="g0a,g0b",
        help="comma-separated subset of g0a,g0b (default: %(default)s). G0c is not evaluated by "
        "this script at all — §6 solves it by construction in the generation step",
    )
    ap.add_argument(
        "--explain",
        "--dry-run",
        dest="explain",
        action="store_true",
        help="list every input the requested gates need, whether it exists, and what would "
        "produce it. Exits 0 when all are present, 2 when any is missing",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "runs" / "pr08-g0" / "g0_gates.json",
        help="the verdict artifact (default: %(default)s). A run record, deliberately under "
        "gitignored runs/ — unlike GEOM_TOL this is a result, not a pre-commitment",
    )

    g0a = ap.add_argument_group("G0a — label integrity (screen_corpus identity check)")
    g0a.add_argument("--source-dataset", type=Path, help="canonical episodes of the SOURCE corpus")
    g0a.add_argument(
        "--restyled-dataset", type=Path, help="canonical episodes of the RESTYLED corpus"
    )
    g0a.add_argument(
        "--source-screen",
        type=Path,
        help="an already measured screen_corpus artifact for the source, instead of re-measuring "
        "it (the blind ceiling is the expensive part of this gate)",
    )
    g0a.add_argument("--source-screen-out", type=Path, help="keep the source screen artifact here")
    g0a.add_argument(
        "--restyled-screen-out", type=Path, help="keep the restyled screen artifact here"
    )
    g0a.add_argument(
        "--holdout",
        type=Path,
        help="episode split file; BOTH screens are scored on it. Two splits would be two "
        "measurements and their difference is not a label defect",
    )
    g0a.add_argument("--seed", type=int, default=0, help="screen_corpus seed, shared by both sides")

    g0b = ap.add_argument_group("G0b — geometry invariance")
    g0b.add_argument(
        "--geom-config",
        type=Path,
        default=GEOM_CONFIG_DEFAULT,
        help="the COMMITTED GEOM_TOL / EST_DRIFT_P95 artifact (default: %(default)s). Read, never "
        "written: a gate that could write its own tolerance would not be a gate",
    )
    g0b.add_argument("--source-centroids", type=Path, help=f"{CENTROID_SCHEMA} record, source side")
    g0b.add_argument(
        "--restyled-centroids", type=Path, help=f"{CENTROID_SCHEMA} record, restyled side"
    )
    g0b.add_argument("--source-clips", type=Path, help="segment the source clips in this process")
    g0b.add_argument(
        "--restyled-clips", type=Path, help="segment the restyled clips in this process"
    )
    g0b.add_argument(
        "--restyled-source-map",
        type=Path,
        help="JSON object mapping each RESTYLED clip key to the SOURCE clip key it is a restyle "
        "of, for the --restyled-clips path. One source clip becomes many restyled clips (25 "
        "style-instances over 402 source clips in 97_transfer25_restyle.sbatch), so the pairing "
        "cannot be by equal names. Without it the clips path assumes identity and says so in the "
        "record",
    )
    g0b.add_argument(
        "--source-source-map",
        type=Path,
        help=argparse.SUPPRESS,  # symmetry for resolve_side/side_from_clips; the source side of a
        # restyle is its own source, so declaring a map for it is meaningless. Present so the
        # getattr in side_from_clips has something to find rather than a special case per side.
    )
    g0b.add_argument(
        "--dump-centroids",
        type=Path,
        help="write each measured side's centroid record into this directory, so a re-gate does "
        "not have to re-decode the corpus",
    )
    g0b.add_argument(
        "--g0b-percentile",
        type=float,
        default=100.0,
        help="the percentile of the per-frame displacement the budget is applied to (default: "
        "%(default)s = every measured frame). §6 says 'agree' and names no statistic; the choice "
        "is recorded in the artifact",
    )
    g0b.add_argument(
        "--min-coverage",
        type=float,
        default=MIN_COVERAGE_DEFAULT,
        help="fraction of compared frames on which BOTH sides had to find the object before this "
        "run may stand as the gate (default: %(default)s). BORROWED, NOT COINED: it is "
        "measure_geom_tol.DEFAULT_MIN_COVERAGE, the floor the tolerance's own producer holds "
        "itself to on the same corpus with the same segmenter",
    )
    g0b.add_argument("--hist-bin-px", type=float, default=0.5)
    g0b.add_argument("--object-label", default=LABEL_OBJECT, help="label a --*-clips run measures")
    g0b.add_argument("--method", default="auto", help="passed to measure_geom_tol.resolve_method")
    g0b.add_argument("--masks", type=Path, help="precomputed masks, per measure_geom_tol")
    g0b.add_argument("--decoder", default="auto")
    g0b.add_argument("--camera-key", default=None)
    g0b.add_argument("--min-area-px", type=int, default=40)
    g0b.add_argument("--limit", type=int, default=0, help="compare at most N clips; forces exit 3")
    g0b.add_argument(
        "--max-frames", type=int, default=0, help="decode at most N frames per clip; forces exit 3"
    )
    return ap.parse_args(argv)


def worst_verdict(verdicts: list[str]) -> str:
    """The aggregate. Ordering — and the reason VOID outranks REFUSED — is in the exit table."""
    if not verdicts:
        return "REFUSED"
    return max(verdicts, key=lambda v: VERDICT_ORDER.index(v) if v in VERDICT_ORDER else 0)


def main(argv: list[str] | None = None) -> int:
    import tempfile

    args = parse_args(argv)
    gates = [g.strip().lower() for g in args.gates.split(",") if g.strip()]
    unknown = [g for g in gates if g not in ("g0a", "g0b")]
    if unknown or not gates:
        print(
            f"FATAL: --gates {args.gates!r} names {unknown or 'nothing'}; this runner implements "
            "g0a and g0b. G0c is solved by construction in the generation step and has no runner.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if args.explain:
        return explain(args, gates)

    records: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="pr08-g0-") as tmp:
        if "g0a" in gates:
            try:
                if args.restyled_dataset is None:
                    raise GateRefusal(
                        "REFUSED: G0a needs --restyled-dataset. It is the corpus whose action "
                        "labels are being checked for having survived the restyle."
                    )
                records["G0a"] = run_g0a(args, Path(tmp))
            except GateRefusal as exc:
                records["G0a"] = {
                    "gate": "G0a",
                    "verdict": "REFUSED",
                    "refusal": str(exc),
                    "consequence": "no statement about the restyled corpus's labels is made",
                }
        if "g0b" in gates:
            try:
                records["G0b"] = run_g0b(args)
            except GateRefusal as exc:
                records["G0b"] = {
                    "gate": "G0b",
                    "verdict": "REFUSED",
                    "refusal": str(exc),
                    "consequence": "no statement about the restyled corpus's geometry is made",
                }
    records["G0c"] = g0c_record()

    verdict = worst_verdict([r["verdict"] for k, r in records.items() if k != "G0c"])
    exit_code = VERDICT_EXIT[verdict]
    record = {
        "schema": SCHEMA,
        "writeup": WRITEUP,
        "rule": RULE,
        "gate": GATE,
        "run_by": "scripts/run_g0_gates.py",
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "gates_requested": gates,
        "verdict": verdict,
        "exit_code": exit_code,
        "exit_code_meaning": {
            "0": "PASS — every requested gate ran and passed",
            "2": "REFUSED — a gate could not be evaluated; no verdict on the corpus is claimed",
            "3": "NOT_GATE_QUALIFIED — ran, nothing failed, but this run may not stand as the gate",
            "4": "VOID — a gate ran and FAILED (PR-08 §6's own word)",
        },
        "gates": records,
        "licenses": (
            "Nothing. A PASS closes part of PR-08 §8's list; §1 forbids generation until every "
            "item is closed and T-39 has reported, and this file may not be read as lifting that."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for name in ("G0a", "G0b"):
        entry = records.get(name)
        if entry is None:
            continue
        print(f"=== {name}: {entry['verdict']}", file=sys.stderr)
        if entry.get("refusal"):
            print(entry["refusal"], file=sys.stderr)
        for reason in entry.get("not_gate_qualified_reasons", []):
            print(f"    not gate-qualified: {reason}", file=sys.stderr)
        for row in entry.get("void_rows", []):
            print(f"    VOID: {row}", file=sys.stderr)
        for row in entry.get("deltas", []):
            print(
                f"    {row['metric'].upper()}  source {row['source']:8.4f}  restyled "
                f"{row['restyled']:8.4f}  delta {row['delta']:+.4f}  tol +-{row['tol']}  "
                f"{'OK' if row['within_tol'] else 'MISS'}",
                file=sys.stderr,
            )
    print(f"=== G0c: {records['G0c']['verdict']} ({records['G0c']['why'].split(':')[0]})",
          file=sys.stderr)
    print(f"\nverdict {verdict} (exit {exit_code})\nwrote {args.out}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
