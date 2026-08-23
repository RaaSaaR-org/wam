"""T-040 / PR-08 `T40_RULE_V8` — the hallucination probe, and every refusal it is built out of.

THE REFUSALS ARE THE FEATURE HERE, so they are what is tested. `107_hallucination_probe.sbatch`
exists to answer one question and to be incapable of answering any other, and a guard that is
reworded into uselessness fails silently unless something runs it. So the shell guards are LIFTED
VERBATIM out of the sbatch and executed — the same trick `tests/test_t39_baseline.py` uses on 71's
MODEL_DIR guard — rather than being restated here where they could drift.

THE ROBOT MASKER IS FAKED and the pairing is not. A masker whose answers are known is the only way
to assert that "source empty + generated grounds something" reaches the candidate bucket and that
"source non-empty" is excluded from the headline; the code that decides which bucket a frame lands
in is the real code.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import probe_hallucination as ph  # noqa: E402
import robot_composite as rc  # noqa: E402
import restyle_transfer25 as rt  # noqa: E402

SBATCH_107 = _REPO_ROOT / "cluster" / "discoverer" / "107_hallucination_probe.sbatch"
V8_DOC = _REPO_ROOT / "docs" / "preregistration" / "PR-08-V8-hallucination-probe.md"

GUARD_OPEN = "# --- BEGIN V8 GUARDS"
GUARD_CLOSE = "# --- END V8 GUARDS ---"


# =================================================================================================
# the shell guards, run as shipped
# =================================================================================================


def _guards() -> str:
    """The guard block lifted VERBATIM from the sbatch, so the tests run the shipped code.

    Deleting or renaming the sentinels makes the extraction fail, which fails the test — the point
    of lifting it rather than restating it here.
    """
    text = SBATCH_107.read_text(encoding="utf-8")
    assert GUARD_OPEN in text and GUARD_CLOSE in text, (
        "107_hallucination_probe.sbatch no longer delimits its guard block"
    )
    start = text.index(GUARD_OPEN)
    return text[start : text.index(GUARD_CLOSE)]


def _run_guards(*, v8_doc: pathlib.Path, out: str = "/x/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS",
                env: dict[str, str] | None = None):
    prelude = f'set -uo pipefail\nV8_DOC={v8_doc}\nOUT={out}\n'
    return subprocess.run(
        ["bash", "-c", prelude + _guards()],
        capture_output=True, text=True, check=False,
        env={"PATH": "/usr/bin:/bin", **(env or {})},
    )


SIGNED = """\
Project owner: huhn.dev@gmail.com               Date: 2026-08-23

Determination:   [x] signed as proposed
                 [ ] signed with the amendments noted below
                 [ ] declined
"""

DECLINED = """\
Project owner: huhn.dev@gmail.com               Date: 2026-08-23

Determination:   [ ] signed as proposed
                 [ ] signed with the amendments noted below
                 [x] declined
"""


@pytest.fixture()
def signed_v8(tmp_path: pathlib.Path) -> pathlib.Path:
    doc = tmp_path / "PR-08-V8-hallucination-probe.md"
    doc.write_text(SIGNED, encoding="utf-8")
    return doc


def test_an_unsigned_v8_makes_the_job_refuse_by_rule_name(tmp_path):
    """The gate itself, exercised against an unsigned copy rather than against the shipped file.

    ORIGINALLY this asserted that the SHIPPED V8 is unsigned, which was true when it was written
    and stopped being true on 2026-08-23 when the owner signed it. That version had to go red the
    moment the project advanced, and a test that goes red on progress gets deleted rather than
    read -- the identity-prompt todo guards failed exactly this way earlier in the same week. What
    was worth protecting was never the document's momentary state; it is that an unsigned
    registration STOPS THE JOB. That property is permanent, so it is now exercised against a
    constructed unsigned copy and holds whatever the shipped file says.
    """
    unsigned = tmp_path / "PR-08-V8-unsigned.md"
    unsigned.write_text(
        V8_DOC.read_text().replace("[x] signed as proposed", "[ ] signed as proposed")
        .replace("Project owner: huhn.dev@gmail.com", "Project owner: ______________")
    )
    result = _run_guards(v8_doc=unsigned)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "T40_RULE_V8's determination is NOT SIGNED" in result.stdout
    assert "NO AGENT MAY SIGN IT" in result.stdout


def test_the_shipped_v8_is_signed_and_says_who_signed_it_and_how():
    """The committed state, asserted deliberately rather than as a side effect.

    Going red here means either that the signature was removed -- in which case the probe must
    refuse and the test above is the one that matters -- or that someone filled it in. The
    transcription paragraph is asserted too, because a signature line without it would read as an
    agent's signature, and V8 §8 makes that a forgery rather than a shortcut.
    """
    text = V8_DOC.read_text()
    assert "[x] signed as proposed" in text
    assert "2026-08-23" in text
    assert "A transcribed signature is the owner's decision and not an agent's" in text
    assert "has forged it" in text
    # The narrow scope survives signing: signing licenses the probe, not generation.
    assert "§8 items 3 and 4 are open" in text
    # Collapse newlines AND blockquote markers: the clause is wrapped inside a "> " quote block,
    # so plain whitespace-flattening leaves a stray ">" mid-sentence. Pinning the claim, not the
    # line wrapping or the markdown.
    flat = " ".join(
        line.lstrip().lstrip(">").strip() for line in text.splitlines()
    )
    flat = " ".join(flat.split())
    assert "none of the three licenses generation" in flat.lower()
    assert "no number from PR-08 — including any number from this probe — may be quoted" in flat
    assert "decided by | the **project owner**" in flat


def test_a_missing_registration_refuses_by_name(tmp_path):
    result = _run_guards(v8_doc=tmp_path / "nothing.md")
    assert result.returncode == 1, result.stdout
    assert "T40_RULE_V8 is not registered" in result.stdout
    assert "licenses by ENUMERATION" in result.stdout or "licenses four things" in result.stdout


def test_a_signed_registration_at_the_default_shape_passes(signed_v8):
    result = _run_guards(v8_doc=signed_v8)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_DECLINED_determination_is_not_a_signature(tmp_path):
    """A ticked `declined` box is a decision, and the decision is no. It must not read as signed."""
    doc = tmp_path / "declined.md"
    doc.write_text(DECLINED, encoding="utf-8")
    result = _run_guards(v8_doc=doc)
    assert result.returncode == 1, result.stdout
    assert "NOT SIGNED" in result.stdout


def test_a_blank_owner_line_is_not_a_signature_even_with_a_ticked_box(tmp_path):
    """The box and the name are separate claims and the job wants both."""
    doc = tmp_path / "half.md"
    doc.write_text(
        "Project owner: ______________________________     Date: ____________\n\n"
        "Determination:   [x] signed as proposed\n",
        encoding="utf-8",
    )
    result = _run_guards(v8_doc=doc)
    assert result.returncode == 1, result.stdout
    assert "name       <blank>" in result.stdout


@pytest.mark.parametrize(
    "env,needle",
    [
        ({"FRAMES": "400"}, "caps a probe clip at 121 frames"),
        ({"EPISODES": "9"}, "caps this probe at 3 episodes"),
        ({"STYLE_COUNT": "8"}, "caps this probe at 3 committed styles"),
        ({"EPISODES": "3", "STYLE_COUNT": "3"}, "caps this probe at 6 clips"),
    ],
)
def test_the_cap_cannot_be_raised_from_the_submit_line(signed_v8, env, needle):
    """Every over-cap submission EXITS 2. Refused, not clamped.

    A clamp would turn an operator asking for a corpus into a run of 726 frames that looks like it
    did what was asked, and the log line saying otherwise is the one nobody reads.
    """
    result = _run_guards(v8_doc=signed_v8, env=env)
    assert result.returncode == 2, result.stdout + result.stderr
    assert needle in result.stdout


def test_a_non_integer_count_is_refused_rather_than_arithmetic(signed_v8):
    result = _run_guards(v8_doc=signed_v8, env={"FRAMES": "96; rm -rf /"})
    assert result.returncode == 2, result.stdout
    assert "not a positive integer" in result.stdout


def test_an_output_directory_that_does_not_say_what_it_holds_is_refused(signed_v8):
    result = _run_guards(v8_doc=signed_v8, out="/valhalla/runs/t040-restyle-train")
    assert result.returncode == 2, result.stdout
    assert "NOT-A-CORPUS" in result.stdout


@pytest.mark.parametrize("variable", ["STYLE_SET", "CHUNK_INDEX", "TIMING", "PARTITION_CEILING_GPU_H",
                                      "PR08_T39_REPORTED", "PR08_OVERRIDE_T39_VOID"])
def test_97s_variables_are_refused_rather_than_ignored(signed_v8, variable):
    """An operator who set one of these meant to run the generation job, not this one.

    Ignoring them silently would let somebody believe they had passed a style set, a chunk or a
    GPU-h ceiling to a job that has none of those things — and `PR08_OVERRIDE_T39_VOID` reaching a
    generation path by accident is the specific failure T40_RULE_V4 §5 refuses.
    """
    result = _run_guards(v8_doc=signed_v8, env={variable: "1"})
    assert result.returncode == 2, result.stdout
    assert f"{variable} is set" in result.stdout
    assert "97_transfer25_restyle.sbatch" in result.stdout


def test_the_total_frame_cap_is_exactly_the_product_of_the_other_two():
    """The 726 is not an independent number and must not drift into being one.

    It cannot fire while the clip cap and the per-clip cap hold — 6 x 121 is exactly 726 — and that
    is the point: it is there so that raising ONE of the other two caps in a future edit trips this
    test instead of silently raising the total the pre-registration quotes.
    """
    text = SBATCH_107.read_text(encoding="utf-8")
    caps = {k: int(re.search(rf"^CAP_{k}=(\d+)$", text, re.M).group(1))
            for k in ("CLIPS", "FRAMES_PER_CLIP", "TOTAL_FRAMES", "EPISODES", "STYLES")}
    assert caps["CLIPS"] * caps["FRAMES_PER_CLIP"] == caps["TOTAL_FRAMES"]
    assert caps["EPISODES"] * caps["STYLES"] >= caps["CLIPS"]


def test_the_sbatch_and_the_script_carry_the_SAME_cap():
    """Two independent enforcements, so neither file can be widened alone.

    The shell refuses before Python starts and Python refuses again; a session that edited one and
    forgot the other would produce a job whose guard and whose driver disagree about the licence.
    """
    text = SBATCH_107.read_text(encoding="utf-8")
    for shell, module in (
        ("CAP_EPISODES", ph.PROBE_MAX_EPISODES),
        ("CAP_STYLES", ph.PROBE_MAX_STYLES),
        ("CAP_CLIPS", ph.PROBE_MAX_CLIPS),
        ("CAP_FRAMES_PER_CLIP", ph.PROBE_MAX_FRAMES_PER_CLIP),
        ("CAP_TOTAL_FRAMES", ph.PROBE_MAX_TOTAL_FRAMES),
    ):
        found = int(re.search(rf"^{shell}=(\d+)$", text, re.M).group(1))
        assert found == module, f"{shell}={found} but probe_hallucination says {module}"


# =================================================================================================
# the sbatch's command line
# =================================================================================================


def test_the_sbatch_invokes_the_probe_with_no_flag_it_does_not_declare():
    """The production command line, read out of the sbatch rather than remembered.

    Two assertions. It must not pass --backend, because the default is transfer25 and --backend null
    is the placeholder generator that must stay unreachable from the cluster path — the same rule
    97 lives under. And every flag it does pass has to be one the parser declares, so a rename here
    cannot leave the job running the old contract.
    """
    text = SBATCH_107.read_text(encoding="utf-8")
    calls = re.findall(r'python "\$\{PROBE\}"((?:[^\n]*\\\n)*[^\n]*)', text)
    assert len(calls) == 1, f"expected exactly one probe invocation, found {len(calls)}"
    declared = {o for a in ph.build_parser()._actions for o in a.option_strings}
    flags = set(re.findall(r"\s(--[a-z0-9]+(?:-[a-z0-9]+)*)\b", calls[0]))
    assert flags <= declared, f"the sbatch passes {sorted(flags - declared)}, undeclared"
    assert "--backend" not in flags
    assert "--no-guardrails" in flags


def test_the_sbatch_clears_the_detection_thresholds_a_submit_environment_could_smuggle_in():
    """Here a threshold decides which pixels count as a robot on BOTH sides of the comparison — so
    a value inherited from a submit environment would decide the answer. 97 and 106 clear them for
    the same reason."""
    text = SBATCH_107.read_text(encoding="utf-8")
    for variable in ("WAM_PR08_BOX_THRESHOLD", "WAM_PR08_TEXT_THRESHOLD",
                     "WAM_PR08_RETRY_BOX_THRESHOLD", "WAM_PR08_RETRY_TEXT_THRESHOLD",
                     "WAM_PR08_OBJECT_PROMPT"):
        assert re.search(rf"^unset .*\b{variable}\b", text, re.M), f"{variable} is not cleared"
    assert text.index("unset WAM_PR08_BOX_THRESHOLD") < text.index('python "${PROBE}"')


def test_the_sbatch_is_not_a_requeue_rail_and_is_not_an_array():
    """A requeued probe would re-generate frames it already generated and could walk past the cap
    across restarts, which is the one thing the cap exists to prevent. 97 has a requeue rail
    because its chunked generation must resume; this must not."""
    text = SBATCH_107.read_text(encoding="utf-8")
    assert "#SBATCH --requeue" not in text
    assert "#SBATCH --array" not in text
    assert "--qos=ehpc-aif-2026pg01-905" in text


def test_the_sbatch_audits_its_own_output_tree_for_anything_consumable():
    """The script does this too. It is repeated in the shell because the script's walk cannot run
    if the script died, and a tree left by a crash is exactly the tree somebody later finds."""
    text = SBATCH_107.read_text(encoding="utf-8")
    audit = text[text.index('CONSUMABLE=""'):]
    for name in ("'*.mp4'", "'manifest.json'", "'work.jsonl'", "'sample_outputs.json'",
                 "'*.parquet'"):
        assert name in audit, f"the output audit does not look for {name}"
    # The one exception, and it must be exact on BOTH halves in the shell too: dirname compared for
    # equality (not a prefix) and the name suffix. A prefix test would allow a nested directory.
    assert '"$(dirname "${candidate}")" == "${OUT}/probe_clips"' in audit
    assert '*.probe-source.mp4' in audit


def test_the_shell_audit_and_the_python_audit_allow_THE_SAME_one_thing():
    """Two enforcements of one rule, so neither can drift into being a different rule.

    The shell walk exists because the script's own walk cannot run if the script died; that is only
    worth having if both allow exactly the same pair.
    """
    text = SBATCH_107.read_text(encoding="utf-8")
    assert f'"${{OUT}}/{ph.PROBE_INPUT_DIR}"' in text
    assert f"*{ph.PROBE_INPUT_SUFFIX}" in text


# =================================================================================================
# the cap, in Python
# =================================================================================================


def test_enforce_caps_accepts_the_default_shape():
    shape = ph.enforce_caps(episodes=2, styles=2, frames=96)
    assert shape["total_generated_frames"] == 384
    assert shape["cap"]["raisable_from_the_submit_line"] is False


@pytest.mark.parametrize("kwargs,needle", [
    ({"episodes": 4, "styles": 1, "frames": 96}, "caps the probe at 3 episodes"),
    ({"episodes": 1, "styles": 4, "frames": 96}, "caps the probe at 3 committed styles"),
    ({"episodes": 1, "styles": 1, "frames": 122}, "caps a probe clip at 121 frames"),
    ({"episodes": 3, "styles": 3, "frames": 96}, "caps the probe at 6 clips"),
    ({"episodes": 1, "styles": 1, "frames": 12}, "below 48"),
    ({"episodes": 0, "styles": 1, "frames": 96}, "must all be at least 1"),
])
def test_enforce_caps_refuses_and_names_the_rule(kwargs, needle):
    with pytest.raises(ph.ProbeError) as exc:
        ph.enforce_caps(**kwargs)
    assert needle in str(exc.value)
    assert "T40_RULE_V8" in str(exc.value)


def test_the_output_directory_must_say_what_it_holds(tmp_path):
    with pytest.raises(ph.ProbeError, match="quarantined by construction"):
        ph.require_quarantined_out(tmp_path / "clips" / "train")
    ok = tmp_path / "pr08-hallucination-probe-QUARANTINE"
    assert ph.require_quarantined_out(ok) == ok


def test_the_probe_never_calls_the_gate_whose_premise_it_measures():
    """G0c's refusal is the thing being measured around. Running `check_mask` would refuse every
    unit before a number existed, so the module imports the masker and not the gate — asserted
    against the source, because this is a sentence that would survive its own mechanism being
    deleted."""
    source = (_REPO_ROOT / "scripts" / "probe_hallucination.py").read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in source.splitlines()
        if "check_mask" not in line or not line.lstrip().startswith(("#", "*", '"', "'"))
    )
    assert not re.search(r"check_mask\s*\(", executable), (
        "probe_hallucination.py calls check_mask; it must not"
    )


# =================================================================================================
# the measurement
# =================================================================================================


def test_the_selection_takes_the_longest_contiguous_absent_run_not_the_first():
    """A contiguous run is what removes temporal propagation as a confound, and the LONGEST one is
    what keeps a two-frame flicker at the start of an episode from being chosen over the approach.
    """
    below, above = ph.ABSENT_BELOW - 1, ph.PRESENT_ABOVE + 1
    areas = [below, below, above, above, below, below, below, below, above]
    assert ph.longest_absent_run(areas) == (4, 4)
    assert ph.longest_absent_run([above] * 5) == (0, 0)


def test_a_frame_in_the_band_is_not_absent():
    """The band is not a threshold. A frame between the modes — the arm entering or leaving shot —
    is the population the whole question turns on and it belongs in neither bucket."""
    middle = (ph.ABSENT_BELOW + ph.PRESENT_ABOVE) // 2
    assert ph.longest_absent_run([middle] * 5) == (0, 0)


def _mask(shape, on: bool):
    m = np.zeros(shape, dtype=bool)
    if on:
        m[1:3, 1:3] = True
    return m


def test_the_three_pairings_are_what_the_registration_says_they_are():
    shape = (8, 8)
    assert ph.pair_frame(_mask(shape, False), _mask(shape, False)) == ph.PAIRING_BOTH_EMPTY
    assert ph.pair_frame(_mask(shape, False), _mask(shape, True)) == ph.PAIRING_CANDIDATE
    # The masker grounds the APPLE on 41 % of robot-absent SOURCE frames. On such a frame nothing
    # about invention is provable, so it is excluded from the headline rather than counted either
    # way — with the generated mask empty AND non-empty.
    assert ph.pair_frame(_mask(shape, True), _mask(shape, True)) == ph.PAIRING_EXCLUDED
    assert ph.pair_frame(_mask(shape, True), _mask(shape, False)) == ph.PAIRING_EXCLUDED


def test_U_is_not_a_quiet_N():
    """A unit with too few paired frames measured the instrument, not the generator."""
    thin = [{"pairing": ph.PAIRING_BOTH_EMPTY}] * (ph.MIN_PAIRED_PROBE_FRAMES - 1)
    assert ph.unit_outcome(thin) == "U"
    enough = [{"pairing": ph.PAIRING_BOTH_EMPTY}] * ph.MIN_PAIRED_PROBE_FRAMES
    assert ph.unit_outcome(enough) == "N"
    assert ph.unit_outcome(enough + [{"pairing": ph.PAIRING_CANDIDATE}]) == "H"
    # Excluded frames do not pad the paired population into existence.
    padded = enough[:2] + [{"pairing": ph.PAIRING_EXCLUDED}] * 50
    assert ph.unit_outcome(padded) == "U"


def test_the_output_audit_catches_every_name_a_consumer_could_file(tmp_path):
    """assemble_restyled_lerobot.py files a clip directory by glob('*.mp4') and 97's harvest keys
    on vision.mp4. This is walked rather than asserted, because a guarantee that rests on every
    future edit remembering a rule is weaker than one that rests on looking."""
    root = tmp_path / "pr08-hallucination-probe"
    (root / "units" / "u0").mkdir(parents=True)
    (root / "units" / "u0" / "clip.mp4.quarantined").write_bytes(b"ok")
    (root / "PROBE.json").write_text("{}")
    assert ph.audit_output_tree(root) == []
    (root / "units" / "u0" / "vision.mp4").write_bytes(b"bad")
    (root / "manifest.json").write_text("{}")
    offenders = ph.audit_output_tree(root)
    assert any(o.endswith("vision.mp4") for o in offenders)
    assert any(o.endswith("manifest.json") for o in offenders)


def test_the_audit_allows_the_generators_INPUT_in_one_place_and_nowhere_else(tmp_path):
    """The single named exception, tested on both halves separately.

    The probe-source clip is the generator's input and keeps a readable extension, because upstream
    refuses an input by extension. It is not generated data and carries no generated frame, so the
    quarantine has nothing to contain there — but it still must not be mistakable for a corpus, and
    what does that work is the pair (this directory, this name). Either half alone is not enough,
    so either half alone must fail.
    """
    root = tmp_path / "pr08-hallucination-probe"
    (root / ph.PROBE_INPUT_DIR).mkdir(parents=True)
    (root / ph.PROBE_INPUT_DIR / ("ep000" + ph.PROBE_INPUT_SUFFIX)).write_bytes(b"ok")
    assert ph.audit_output_tree(root) == [], "the allowed pair must pass"

    # right name, wrong place
    (root / "units").mkdir()
    stray = root / "units" / ("ep000" + ph.PROBE_INPUT_SUFFIX)
    stray.write_bytes(b"bad")
    assert [o for o in ph.audit_output_tree(root) if o == str(stray)], (
        "a .probe-source.mp4 outside probe_clips must still be refused"
    )
    stray.unlink()

    # right place, wrong name — a generated clip hiding among the inputs
    generated = root / ph.PROBE_INPUT_DIR / "unit-00.mp4"
    generated.write_bytes(b"bad")
    assert [o for o in ph.audit_output_tree(root) if o == str(generated)], (
        "an ordinary .mp4 in probe_clips must still be refused"
    )
    generated.unlink()

    # and nesting does not inherit the allowance
    nested = root / ph.PROBE_INPUT_DIR / "sub"
    nested.mkdir()
    deep = nested / ("ep001" + ph.PROBE_INPUT_SUFFIX)
    deep.write_bytes(b"bad")
    assert [o for o in ph.audit_output_tree(root) if o == str(deep)]


# =================================================================================================
# end to end, on the placeholder backend
# =================================================================================================


class ProbeMasker:
    """Empty for the first ``source_calls`` masks, then whatever the test asked for.

    Counting calls rather than inspecting pixels, for the reason ``ShiftedMasker`` gives in
    tests/test_restyle_transfer25.py: the fixture's frames and the null backend's are both flat
    fields, so a masker that guessed from content would be testing itself. The probe masks every
    SOURCE frame of an episode before it generates anything, which is what makes the split clean.
    """

    def __init__(self, source_calls: int, invent: bool) -> None:
        self.source_calls, self.invent = source_calls, invent
        self.calls = 0
        self.preflighted = 0

    def preflight(self) -> None:
        self.preflighted += 1

    def provenance(self) -> dict:
        return {"name": "probe-test-masker", "version": "test", "prompt": rc.ROBOT_TEXT_PROMPT,
                "box_threshold": 0.15, "text_threshold": 0.25, "box_rule": "test rectangle",
                "upstream_retry_not_run": "test masker"}

    def mask(self, rgb) -> np.ndarray:
        self.calls += 1
        out = np.zeros(np.asarray(rgb).shape[:2], dtype=bool)
        if self.calls > self.source_calls and self.invent:
            out[2:10, 2:10] = True
        return out


@pytest.fixture()
def corpus(tmp_path: pathlib.Path):
    """Two episodes of flat frames — the reference predicate calls every one of them robot-absent.

    Flat rather than random: `robot_dark_mask` fires on pixels DARK relative to the frame's own
    modal luminance, near-neutral and different from the episode's temporal median, and a flat clip
    has none. That is the honest way to build a robot-free fixture — it satisfies the shipped
    predicate rather than stubbing it out.

    TWO episodes and TWO styles because V8 §5.2 defines outcome N over at least two of each, so a
    one-by-one fixture could never exercise the N path at all.
    """
    src = tmp_path / "source"
    (src / "videos").mkdir(parents=True)
    entries = []
    for index, level in enumerate((130, 118)):
        frames = np.full((60, 48, 64, 3), level, dtype=np.uint8)
        name = f"ep{index:03d}"
        rc.encode_clip(frames, src / "videos" / f"{name}.mp4", 30.0)
        entries.append({"id": name, "frames": 60, "video": f"videos/{name}.mp4"})
    (src / "manifest.json").write_text(json.dumps({
        "resolution": [640, 480], "episodes": entries,
    }), encoding="utf-8")
    styles = tmp_path / "styles.json"
    styles.write_text(json.dumps({
        "train": [
            {"id": "train-01-oak-tungsten", "repeats": 1, "seeds": [7001],
             "prompt": "warm tungsten light. the robot is unchanged."},
            {"id": "train-02-linen-overcast", "repeats": 1, "seeds": [7002],
             "prompt": "cool overcast daylight. the robot is unchanged."},
        ],
    }), encoding="utf-8")
    return {"manifest": src / "manifest.json", "styles": styles,
            "out": tmp_path / "pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS"}


def _argv(corpus, **over) -> list[str]:
    args = {
        "--manifest": str(corpus["manifest"]),
        "--styles": str(corpus["styles"]),
        "--out": str(corpus["out"]),
        "--checkpoint-path": "s3://stub/ckpt",
        "--control": "depth:0.5,seg:0.5",
        "--episodes": "2",
        "--style-count": "2",
        "--frames": "48",
    }
    args.update(over)
    flat = [x for kv in args.items() for x in kv]
    return flat + ["--no-guardrails", "--backend", "null"]


def test_a_generator_that_invents_nothing_reports_N_and_writes_a_lookable_artifact(corpus, monkeypatch):
    masker = ProbeMasker(source_calls=96, invent=False)
    monkeypatch.setattr(rc, "build_masker", lambda: masker)
    assert ph.main(_argv(corpus)) == 0

    payload = json.loads((corpus["out"] / "PROBE.json").read_text())
    assert payload["complete"] is True
    assert payload["verdict"] == "N"
    assert payload["totals"] == {"clips": 4, "generated_frames": 192, "paired_frames": 192,
                                 "candidate_frames": 0, "both_empty_frames": 192,
                                 "excluded_frames": 0}
    assert payload["coverage"]["meets_outcome_N_coverage"] is True
    # The letter is never the finding on its own, and the artifact has to say so itself.
    assert payload["human_review"]["looked_at"] is False
    assert payload["licensed"] == {
        "generation_of_a_corpus": False, "training": False,
        "satisfies_pr08_section8_item3": False, "derives_a_g0c_area_bound": False,
        "note": payload["licensed"]["note"],
    }
    assert payload["instrument"]["check_mask_called"] is False
    # every frame recorded WITH the reason it was selected
    unit = payload["units"][0]
    assert len(unit["frames"]) == 48
    assert unit["frames"][0]["source_frame_index"] == 0
    assert payload["selection"][0]["longest_absent_run"] == {"start": 0, "length": 60}
    # and a sheet a person can open
    assert (corpus["out"] / "sheets").is_dir()
    assert any((corpus["out"] / "sheets").glob("*both_empty.png"))


def test_a_generator_that_grounds_a_robot_the_source_never_had_reports_H(corpus, monkeypatch):
    masker = ProbeMasker(source_calls=96, invent=True)
    monkeypatch.setattr(rc, "build_masker", lambda: masker)
    assert ph.main(_argv(corpus)) == 0
    payload = json.loads((corpus["out"] / "PROBE.json").read_text())
    assert payload["verdict"] == "H"
    assert payload["totals"]["candidate_frames"] == 192
    # The candidate sheet is the first thing the artifact points a reader at.
    assert payload["human_review"]["read_first"], "an H with no candidate sheet is unreadable"
    assert any((corpus["out"] / "sheets").glob("*candidate_invention.png"))


def test_nothing_it_writes_can_be_filed_by_a_downstream_consumer(corpus, monkeypatch):
    """The GENERATED clips survive under a name assemble_restyled_lerobot.py's glob cannot see, and
    the only ordinary .mp4 in the tree is the generator's input, in the one directory it may be in.
    """
    monkeypatch.setattr(rc, "build_masker", lambda: ProbeMasker(source_calls=96, invent=False))
    assert ph.main(_argv(corpus)) == 0
    assert ph.audit_output_tree(corpus["out"]) == []

    quarantined = list(corpus["out"].rglob("*" + ph.QUARANTINE_SUFFIX))
    assert len(quarantined) == 4, "four GENERATED clips, all quarantined"
    ordinary = sorted(corpus["out"].rglob("*.mp4"))
    assert len(ordinary) == 2, "two probe-source INPUTS, and nothing else readable"
    for path in ordinary:
        assert path.parent == corpus["out"] / ph.PROBE_INPUT_DIR
        assert path.name.endswith(ph.PROBE_INPUT_SUFFIX)
    for path in quarantined + ordinary:
        assert rc.decode_clip(path).shape[0] == 48, "the bytes must stay readable"
    assert (corpus["out"] / "NOT_A_CORPUS").read_text().startswith("NOT A CORPUS")


def test_the_generator_is_handed_an_input_it_can_actually_OPEN(corpus, monkeypatch):
    """THE REGRESSION FROM JOB 189769, PINNED.

    That run reached `Loading input video...` and died in upstream's read_and_process_video with
    `ValueError: Invalid video extension: .quarantined`, because the probe had renamed its own
    INPUT out of the way of a glob that only ever mattered for OUTPUT. Upstream validates an input
    by extension, so this asserts the extension the generator is handed — not the file's existence,
    which the old code also satisfied. Re-applying QUARANTINE_SUFFIX to the probe-source clip fails
    here instead of costing a GPU-hour of queue time.
    """
    monkeypatch.setattr(rc, "build_masker", lambda: ProbeMasker(source_calls=96, invent=False))
    handed: list[str] = []
    original = rt._null_backend

    def watching(sample, out_dir):
        handed.append(sample["video_path"])
        return original(sample, out_dir)

    monkeypatch.setattr(rt, "_null_backend", watching)
    assert ph.main(_argv(corpus)) == 0

    assert handed, "the generator was never called"
    for path in handed:
        assert path.endswith(".mp4"), f"upstream refuses this input by extension: {path}"
        assert not path.endswith(ph.QUARANTINE_SUFFIX), (
            "the quarantine suffix is a rule about GENERATED video. Applying it to the input is "
            "what killed job 189769."
        )
        assert pathlib.Path(path).parent.name == ph.PROBE_INPUT_DIR
    # and the asymmetry is the whole point: the OUTPUT of that same unit is quarantined.
    payload = json.loads((corpus["out"] / "PROBE.json").read_text())
    assert payload["units"][0]["clip"].endswith(ph.QUARANTINE_SUFFIX)
    assert payload["selection"][0]["probe_clip"].endswith(ph.PROBE_INPUT_SUFFIX)


def test_the_artifact_records_that_the_seg_control_is_conditioned_on_the_style_prompt(corpus, monkeypatch):
    """Upstream logged: no control_prompt supplied, so it uses the first 128 words of the input
    prompt. Every committed prompt ends '...and the robot are unchanged.', so the segmentation
    control is conditioned on wording that names a robot — a second route by which the word reaches
    the generator, which a reading of this probe's result has to carry."""
    monkeypatch.setattr(rc, "build_masker", lambda: ProbeMasker(source_calls=96, invent=False))
    assert ph.main(_argv(corpus)) == 0
    note = json.loads((corpus["out"] / "PROBE.json").read_text())["instrument"]["generator"]
    assert "seg_control_prompt" in note
    assert "first 128 words" in note["seg_control_prompt"]
    assert "does not separate them" in note["seg_control_prompt"]


def test_a_partial_run_reports_U_rather_than_the_letter_its_finished_units_agree_on(corpus, monkeypatch):
    """PROBE.json is rewritten after every unit, and an unfinished file must not read as an answer.

    No Cosmos-Transfer2.5 throughput has ever been measured on this project, so the Slurm wall is a
    request rather than a derivation and a wall kill is a real outcome. "Every unit that got as far
    as running said N" and "the probe found nothing" are different claims.
    """
    monkeypatch.setattr(rc, "build_masker", lambda: ProbeMasker(source_calls=96, invent=False))
    seen: list[dict] = []
    original = rt._null_backend

    def watching(sample, out_dir):
        result = original(sample, out_dir)
        seen.append(json.loads((corpus["out"] / "PROBE.json").read_text()))
        return result

    monkeypatch.setattr(rt, "_null_backend", watching)
    assert ph.main(_argv(corpus)) == 0
    assert seen and seen[0]["complete"] is False
    assert seen[0]["verdict"] == "U"


def test_it_refuses_to_run_through_the_guardrail_that_blurs_hands(corpus, monkeypatch):
    """The RetinaFace postprocessor writes blurred pixels back into the frame and the blurred frames
    are what land on disk. A probe about whether a manipulator appears must not run through a stage
    that erases hands."""
    monkeypatch.setattr(rc, "build_masker", lambda: ProbeMasker(source_calls=96, invent=False))
    argv = [a for a in _argv(corpus) if a != "--no-guardrails"]
    assert ph.main(argv) == 1


def test_it_refuses_an_episode_whose_robot_free_run_is_too_short(tmp_path, monkeypatch):
    """Not enough contiguous robot-free frames is a refusal, and the message must not suggest
    loosening the band — which is a constant precisely so it cannot be the lever."""
    src = tmp_path / "source"
    (src / "videos").mkdir(parents=True)
    # A moving dark blob against a flat cloth: dark, near-neutral and NOT in the episode's temporal
    # median, which is the shipped predicate's three clauses. It therefore scores well above the
    # absent band on every frame, so no run of 48 robot-free frames exists. The blob is narrow and
    # travels far enough that no pixel is covered on a majority of frames — otherwise the median
    # would absorb it, which is the predicate's own documented blind spot.
    frames = np.full((60, 96, 128, 3), 130, dtype=np.uint8)
    for i in range(60):
        col = (i * 3) % 96
        frames[i, 18:78, col : col + 30] = 10
    rc.encode_clip(frames, src / "videos" / "ep000.mp4", 30.0)
    (src / "manifest.json").write_text(json.dumps({
        "resolution": [640, 480],
        "episodes": [{"id": "ep000", "frames": 60, "video": "videos/ep000.mp4"}],
    }), encoding="utf-8")
    styles = tmp_path / "styles.json"
    styles.write_text(json.dumps({"train": [
        {"id": "train-01", "prompt": "p", "repeats": 1, "seeds": [7001]}]}), encoding="utf-8")
    monkeypatch.setattr(rc, "build_masker", lambda: ProbeMasker(source_calls=0, invent=False))
    out = tmp_path / "pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS"
    assert ph.main(_argv({"manifest": src / "manifest.json", "styles": styles, "out": out})) == 1


def test_it_coins_no_seed_and_refuses_a_style_that_carries_none(tmp_path):
    """T40_RULE_V3 committed one seed per style-instance; this probe uses it and invents none."""
    styles = tmp_path / "styles.json"
    styles.write_text(json.dumps({"train": [{"id": "train-01", "prompt": "p", "repeats": 1}]}),
                      encoding="utf-8")
    with pytest.raises(ph.ProbeError, match="coins no seed"):
        ph.load_train_styles(styles, 1)


def test_the_style_set_is_a_constant_and_there_is_no_flag_that_moves_it():
    """TRAIN_STYLES is what arm B runs. EVAL_STYLES is the held-out domain and has no business being
    touched before the experiment; the identity style is arm C's question, not this one."""
    assert ph.PROBE_STYLE_SET == "train"
    declared = {o for a in ph.build_parser()._actions for o in a.option_strings}
    assert "--style-set" not in declared


def test_EVERY_generated_video_is_quarantined_not_just_the_ones_we_named(tmp_path):
    """THE REGRESSION FROM JOB 189926, PINNED.

    That run generated all four units correctly and then failed its own audit, because upstream
    writes ``<name>_control_depth.mp4`` and ``<name>_control_seg.mp4`` alongside the sample
    (cosmos_transfer2/inference.py:311) and this module only knew two filenames. Enumerating what
    the framework writes is a guess; the rule is a property of the directory. When the sweep
    returns, nothing under it ends in .mp4 — including a name nobody has seen yet.
    """
    unit = tmp_path / "episode_000000__train-01__probe"
    unit.mkdir()
    names = [
        "episode_000000__train-01__probe.mp4",
        "episode_000000__train-01__probe_control_depth.mp4",
        "episode_000000__train-01__probe_control_seg.mp4",
        "some_future_upstream_artifact.mp4",
    ]
    for name in names:
        (unit / name).write_bytes(b"generated")

    moved = ph.quarantine_every_generated_video(unit)

    assert len(moved) == len(names)
    assert not list(unit.rglob("*.mp4")), "the sweep is a property of the directory, not a list"
    for name in names:
        assert (unit / (name + ".quarantined")).is_file(), (
            f"{name} must keep its own stem: which video a byte came from is what a reader needs"
        )
    # and the audit, which is the backstop, now finds nothing to refuse
    assert ph.audit_output_tree(unit) == []


def test_the_control_maps_are_refused_by_the_audit_if_the_sweep_ever_stops_running(tmp_path):
    """The backstop must still catch what the sweep exists to prevent.

    If a future edit drops the sweep, the audit is the only thing between a control map and
    assemble_restyled_lerobot.py's glob("*.mp4"). This asserts it fails on exactly the file job
    189926 left behind, so removing the sweep cannot go unnoticed.
    """
    root = tmp_path / "pr08-hallucination-probe"
    unit = root / "units" / "episode_000000__train-01__probe"
    unit.mkdir(parents=True)
    control = unit / "episode_000000__train-01__probe_control_seg.mp4"
    control.write_bytes(b"generated depth/seg map, rendered as video")

    assert str(control) in ph.audit_output_tree(root)
