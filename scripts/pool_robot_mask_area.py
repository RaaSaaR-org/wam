#!/usr/bin/env python3
"""Rebuild the pooled robot-mask-area distribution from the shard artifacts that produced it.

WHY THIS FILE EXISTS. Slurm job 190981 asked for an H200 to time one episode and died six seconds
later because `${PROJ}/runs/pr08-robot-mask-area/POOLED.json` is not on the cluster. Chasing that
one missing file turned up the real defect underneath it:

    POOLED.json is a hand-built artifact that NO COMMITTED SCRIPT IN THIS REPOSITORY WRITES.

It carries exactly six top-level keys — `git_commit`, `source_manifest_sha256`, `prompt`,
`estimator`, `measurement_qualified`, `per_episode` — with no `schema`, no `measured` block and no
merge-condition verdicts, so it is not the `wam.robot_mask_area/1` artifact either.
`robot_composite.py measure --merge` does pool the same shards, but its output DROPS `per_episode`
entirely (it writes the five summary numbers and the bound), so re-running the merge does not
produce this file and cannot answer the question `T40_RULE_V20` asks of it. Until commit `0c2b1eb`
force-added it, the file lived under gitignored `runs/` on exactly one workstation — and four
separate consumers — the pre-GPU screen in `97_transfer25_restyle.sbatch`,
`submit_timing_episode.sh`, `render_area_tail_sheet.py` and the empty-mask scripts — read it as if
it were evidence with provenance. Force-adding it fixed where it lives. It did not give it a
producer, which is why the shards and not this file are what everything now derives from.

`docs/investigations/2026-08-27-pr08-fronts/F5-yield-empty-mask.md` §A.7 found this first, and its
conclusion is the one implemented here: a pre-flight that authorises GPU time must not validate a
weaker thing than the repository's own loader already does, and §5's caveat adds that a gitignored
file "is not the pre-commitment a rule can point at."

WHAT REPLACES IT, AND WHY THAT IS NOT A RULE CHANGE. The sixteen `wam.robot_mask_area_shard/1`
artifacts POOLED.json was pooled from are on the cluster, at
`${PROJ}/runs/pr08-robot-mask-area/shards/`, and each one carries a real schema, a `shard` coverage
block, its own `measurement_qualified`, and the RAW per-frame fractions. Measured on 2026-08-28,
before this file was written:

    the 402 per-episode records rebuilt from those sixteen shards are IDENTICAL to POOLED.json's,
    field for field, every float of every `area_fractions` list included — the lists differ only in
    their ORDER, which a set-membership criterion and a median over `n_frames` cannot see;

    and POOLED.json hashes to 631103a8a97010c4804ac039aecc7fd8425c226c750294335fad5938c35233db,
    which is the sha256 `T40_RULE_V20` §2 registered for it.

So pooling the shards does not introduce new evidence and does not move the population `T40_RULE_V20`
§2 computed. It reconstructs the registered evidence from its own inputs, on the machine that needs
it. `--assert-equivalent` is how that claim is checked rather than trusted, and
`tests/test_pool_robot_mask_area.py` runs it against the registered hash.

It is also strictly MORE than POOLED.json could offer. The shards carry `measured.stride`, so the
"stride 1" that §A.7 caught being quoted against a file it is not derivable from is derivable here;
they carry `corpus_episode_keys_sha256`, so a pool over shards that saw different corpora refuses;
and they carry the partition, so an episode sitting in a shard it does not hash to refuses.

Stdlib only, and deliberately so: the pre-GPU screen in `97_transfer25_restyle.sbatch` imports this
module, and that screen must reach its verdict without torch, without transformers and without a
working masker — the whole point of it is that the verdict is computable off the cluster.

    python scripts/pool_robot_mask_area.py runs/pr08-robot-mask-area/shards \\
        --out runs/pr08-robot-mask-area/POOLED-REBUILT.json \\
        --assert-equivalent runs/pr08-robot-mask-area/POOLED.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Iterable

SHARD_SCHEMA = "wam.robot_mask_area_shard/1"
POOLED_SCHEMA = "wam.robot_mask_area_pooled/1"

#: The sha256 `T40_RULE_V20` §2 registered for the pooled evidence its population was computed from.
#: Named here so `--assert-equivalent` can say WHICH file it reproduced, not merely that two files
#: agree. It is a value to check against and never a value anything is derived from.
V20_REGISTERED_POOLED_SHA256 = (
    "631103a8a97010c4804ac039aecc7fd8425c226c750294335fad5938c35233db"
)

#: Copied from the `shard.assignment` string the shards themselves carry, and re-derived rather than
#: trusted: a shard holding an episode that does not hash to it means two shards may hold the same
#: episode, and the pool would weight one episode twice while proving a coverage it never had.
_ASSIGNMENT = (
    "int.from_bytes(blake2b(episode_key.utf8, digest_size=8).digest(), 'big') % num_shards"
)

#: Provenance every shard must agree on before its frames may be pooled with another's. A
#: distribution is a claim ABOUT a corpus measured BY a segmenter; two shards disagreeing on either
#: are two measurements, and pooling them produces a number describing neither.
_PROVENANCE_FIELDS = (
    "prompt",
    "source_manifest_sha256",
    "corpus_episode_keys_sha256",
    "estimator",
    "stride",
)


class PoolError(Exception):
    """The shards cannot be pooled at all — not that the pool is incomplete.

    The distinction is `robot_composite.merge_shard_records`'s and is kept identical here. An
    episode weighted twice, an episode in the wrong shard, a shard that kept only its summary: there
    is no honest distribution to write in those cases, not even a disqualified one, so nothing is
    written and this is raised. A MISSING shard is different in kind — the arithmetic over what
    landed is exactly right about the frames it saw and merely is not the corpus — so that is
    written with ``measurement_qualified: false`` and the reason named, exactly as a ``--limit`` run
    is, and exactly as the loaders already refuse by name.
    """


def _shard_of(episode_key: str, num_shards: int) -> int:
    digest = hashlib.blake2b(episode_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % int(num_shards)


def _file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(record: dict) -> dict:
    """The fields two shards must agree on, gathered from wherever each one lives.

    ``stride`` sits inside ``measured`` and the rest at the top level or inside ``shard``; pulling
    them into one dict is what lets the disagreement message name the field rather than the nesting.
    """
    measured = record.get("measured") or {}
    block = record.get("shard") or {}
    return {
        "prompt": record.get("prompt"),
        "source_manifest_sha256": record.get("source_manifest_sha256"),
        "corpus_episode_keys_sha256": block.get("corpus_episode_keys_sha256"),
        "estimator": json.dumps(record.get("estimator"), sort_keys=True),
        "stride": measured.get("stride"),
    }


def _load_shards(paths: Iterable[pathlib.Path]) -> list[tuple[pathlib.Path, dict]]:
    loaded: list[tuple[pathlib.Path, dict]] = []
    for path in sorted(paths):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PoolError(f"{path} could not be read ({exc}).") from exc
        if not isinstance(record, dict):
            raise PoolError(f"{path} is not a JSON object.")
        if record.get("schema") != SHARD_SCHEMA:
            # Named explicitly rather than skipped. A directory is scanned, but a file whose schema
            # is wrong is a file somebody expected to be pooled, and silently dropping it is how a
            # pool over fifteen shards gets reported as a pool over sixteen.
            raise PoolError(
                f"{path} declares schema {record.get('schema')!r}, not {SHARD_SCHEMA!r}.\n"
                "       The merged artifact and the committed bound are NOT shards: they carry no "
                "per-frame\n"
                "       fractions, so a pool including one would be a pool over a summary."
            )
        loaded.append((path, record))
    return loaded


def pool_shard_dir(directory: pathlib.Path) -> dict:
    """Rebuild the pooled per-episode distribution from a directory of shard artifacts."""
    directory = pathlib.Path(directory)
    if not directory.is_dir():
        raise PoolError(f"{directory} is not a directory.")
    loaded = _load_shards(directory.glob("shard-*.json"))
    if not loaded:
        raise PoolError(
            f"{directory} holds no {SHARD_SCHEMA} artifact named shard-*.json.\n"
            "       The shards are produced by cluster/discoverer/106_measure_robot_mask_area.sbatch "
            "as an array;\n"
            "       without them there is no per-frame evidence and no population to select from."
        )

    # -- the partition ----------------------------------------------------------------------------
    totals = {int((rec.get("shard") or {}).get("num_shards", -1)) for _, rec in loaded}
    if len(totals) != 1 or -1 in totals:
        raise PoolError(
            f"the shards disagree about how many shards the corpus was partitioned into: "
            f"{sorted(totals)}.\n"
            "       They are then partitions of different corpora and their frames do not pool."
        )
    num_shards = totals.pop()

    by_index: dict[int, pathlib.Path] = {}
    for path, rec in loaded:
        index = int((rec.get("shard") or {}).get("index", -1))
        if index in by_index:
            raise PoolError(
                f"shard index {index} appears twice: {by_index[index]} and {path}.\n"
                "       Every episode in it would be weighted twice and the pooled distribution "
                "would be over a\n"
                "       corpus nobody can name."
            )
        by_index[index] = path

    # -- provenance -------------------------------------------------------------------------------
    reference_path, reference = loaded[0]
    want = _provenance(reference)
    for path, rec in loaded[1:]:
        got = _provenance(rec)
        for field in _PROVENANCE_FIELDS:
            if got[field] != want[field]:
                raise PoolError(
                    f"{path} and {reference_path} disagree about {field}:\n"
                    f"       {got[field]!r}\n"
                    f"       {want[field]!r}\n"
                    "       Pooling them produces a distribution describing neither measurement."
                )

    # -- the episodes -----------------------------------------------------------------------------
    per_episode: list[dict] = []
    seen: dict[str, pathlib.Path] = {}
    for path, rec in loaded:
        index = int(rec["shard"]["index"])
        entries = rec.get("per_episode")
        if not isinstance(entries, list) or not entries:
            raise PoolError(
                f"{path} carries no per_episode list.\n"
                "       A shard that kept only its own summary cannot be merged: a median and two "
                "percentiles do\n"
                "       not decompose, so such a shard can be averaged and never pooled."
            )
        for entry in entries:
            episode = str(entry.get("episode"))
            if episode in seen:
                raise PoolError(
                    f"{episode} appears in both {seen[episode]} and {path}."
                )
            seen[episode] = path
            if _shard_of(episode, num_shards) != index:
                raise PoolError(
                    f"{path} holds {episode}, which hashes to shard "
                    f"{_shard_of(episode, num_shards)} of {num_shards}, not {index}.\n"
                    f"       The partition is {_ASSIGNMENT};\n"
                    "       a shard holding an episode it was never assigned means some other shard "
                    "holds it too."
                )
            fractions = entry.get("area_fractions")
            n_frames = int(entry.get("n_frames", -1))
            if not isinstance(fractions, list) or len(fractions) != n_frames:
                raise PoolError(
                    f"{path}: {episode} carries "
                    f"{len(fractions) if isinstance(fractions, list) else 'no'} area fractions for "
                    f"{n_frames} frames.\n"
                    "       Every frame's fraction, or a maximum over them is a maximum over an "
                    "unknown subset."
                )
            per_episode.append(entry)

    # -- coverage, which is STAMPED rather than raised ---------------------------------------------
    disqualified: list[str] = []
    missing = [i for i in range(num_shards) if i not in by_index]
    if missing:
        disqualified.append(
            f"shard(s) {', '.join(str(i) for i in missing)} of {num_shards} did not land, so this "
            "pool is over part of the corpus"
        )
    for path, rec in loaded:
        if rec.get("measurement_qualified") is not True:
            reasons = rec.get("measurement_disqualified_reasons") or []
            disqualified.append(f"{path.name} is itself disqualified: {reasons!r}")

    # The whole enumeration each shard saw. All sixteen agree (checked above via
    # corpus_episode_keys_sha256), so any one of them is the corpus, and a pool that covered fewer
    # episodes than the corpus enumerates is not the corpus's even with every shard present.
    corpus_keys = [str(k) for k in (reference.get("corpus_episode_keys") or [])]
    if corpus_keys and set(seen) != set(corpus_keys):
        uncovered = sorted(set(corpus_keys) - set(seen))
        disqualified.append(
            f"{len(uncovered)} of the {len(corpus_keys)} enumerated episode(s) are in no shard: "
            + ", ".join(uncovered[:8])
            + ("..." if len(uncovered) > 8 else "")
        )

    frames = sum(int(e["n_frames"]) for e in per_episode)
    return {
        "schema": POOLED_SCHEMA,
        # A bound, a selection or a GPU submission may only rest on a pool that IS the corpus's.
        # False here is not a warning to weigh: every consumer refuses the record by name.
        "measurement_qualified": not disqualified,
        "measurement_disqualified_reasons": disqualified,
        "n_episodes": len(per_episode),
        "n_frames": frames,
        # Derivable HERE and not from POOLED.json — the omission §A.7 caught being quoted anyway.
        "stride": want["stride"],
        "prompt": reference.get("prompt"),
        "estimator": reference.get("estimator"),
        "git_commit": reference.get("git_commit"),
        "source_manifest": reference.get("source_manifest"),
        "source_manifest_sha256": reference.get("source_manifest_sha256"),
        "corpus_episode_keys_sha256": want["corpus_episode_keys_sha256"],
        "num_shards": num_shards,
        # WHAT THIS POOL IS OVER, by name and by hash, so a determination citing it cites something
        # that cannot change without the citation going stale.
        "pooled_from": [
            {
                "path": str(path),
                "index": int(rec["shard"]["index"]),
                "sha256": _file_sha256(path),
                "n_episodes": len(rec["per_episode"]),
            }
            for path, rec in sorted(loaded, key=lambda pair: int(pair[1]["shard"]["index"]))
        ],
        "per_episode": per_episode,
    }


def load_area_evidence(path: pathlib.Path | str) -> dict:
    """Read the pooled per-episode evidence from a shard DIRECTORY or from a pooled FILE.

    The one entry point every consumer uses, so that "which evidence answered this" has a single
    answer on both machines. A directory is pooled; a file is read as-is and checked only for the
    shape the consumers rely on, because a pooled file may be one this module wrote or the
    registered POOLED.json, and neither is re-derived here.
    """
    path = pathlib.Path(path)
    if path.is_dir():
        return pool_shard_dir(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PoolError(f"{path} could not be read ({exc}).") from exc
    if not isinstance(record, dict) or not isinstance(record.get("per_episode"), list):
        raise PoolError(
            f"{path} carries no per_episode list, so it cannot say anything about one episode."
        )
    return record


def equivalent(left: dict, right: dict) -> tuple[bool, str]:
    """Do two pooled records describe the same measurement, ignoring per_episode ORDER?

    Order is the one difference between POOLED.json and a rebuild from its own shards, and no
    consumer can see it: the survivor criterion is set membership and the median is over
    ``n_frames``. Everything a consumer CAN see is compared exactly, floats included.
    """
    a = {str(e.get("episode")): e for e in left.get("per_episode") or ()}
    b = {str(e.get("episode")): e for e in right.get("per_episode") or ()}
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        return False, (
            f"different episode sets: {len(a)} vs {len(b)}; "
            f"only left: {only_a[:5]}; only right: {only_b[:5]}"
        )
    for episode in sorted(a):
        for field in ("n_frames", "empty_frames", "area_fractions", "episode_index"):
            if a[episode].get(field) != b[episode].get(field):
                return False, f"{episode}: {field} differs"
    for field in ("prompt", "source_manifest_sha256", "git_commit"):
        if left.get(field) != right.get(field):
            return False, f"{field} differs: {left.get(field)!r} vs {right.get(field)!r}"
    if json.dumps(left.get("estimator"), sort_keys=True) != json.dumps(
        right.get("estimator"), sort_keys=True
    ):
        return False, "estimator differs"
    return True, f"{len(a)} episodes identical field for field, per_episode order aside"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shards", type=pathlib.Path, help="directory of shard-*.json artifacts")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="write the pooled record here")
    parser.add_argument(
        "--assert-equivalent",
        type=pathlib.Path,
        default=None,
        metavar="POOLED_JSON",
        help=(
            "check the rebuild against an existing pooled file and exit 4 if they differ. If that "
            "file hashes to the sha256 T40_RULE_V20 §2 registered, the check is reported as such."
        ),
    )
    args = parser.parse_args(argv)

    try:
        record = pool_shard_dir(args.shards)
    except PoolError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 3

    qualified = record["measurement_qualified"]
    print(
        f"pooled {record['num_shards']} shard(s): {record['n_episodes']} episodes, "
        f"{record['n_frames']} frames, stride {record['stride']}, "
        f"measurement_qualified={qualified}"
    )
    for reason in record["measurement_disqualified_reasons"]:
        print(f"  DISQUALIFIED: {reason}")

    if args.assert_equivalent is not None:
        reference_path = args.assert_equivalent
        try:
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"FATAL: {reference_path} could not be read ({exc}).", file=sys.stderr)
            return 4
        digest = _file_sha256(reference_path)
        registered = digest == V20_REGISTERED_POOLED_SHA256
        ok, why = equivalent(record, reference)
        print(f"vs {reference_path} (sha256 {digest}): {'EQUIVALENT' if ok else 'DIFFERS'} — {why}")
        if registered:
            print("     that file is the pooled evidence T40_RULE_V20 §2 registered, by sha256.")
        if not ok:
            return 4

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")

    # A pool that is not the corpus's is an exit code, not a line in a log somebody has to notice.
    return 0 if qualified else 5


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
