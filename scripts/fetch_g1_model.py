#!/usr/bin/env python3
"""Fetch the MuJoCo Menagerie ``unitree_g1`` model into ``assets/mujoco/unitree_g1`` (T-21, E2).

Usage:
  .venv/bin/python scripts/fetch_g1_model.py              # idempotent: no-op if complete
  .venv/bin/python scripts/fetch_g1_model.py --force      # re-fetch and replace
  .venv/bin/python scripts/fetch_g1_model.py --ref main   # track upstream HEAD instead of the pin
  .venv/bin/python scripts/fetch_g1_model.py --check      # verify only, fetch nothing

``configs/sim/g1_scene.xml`` ``<include>``s the vendor model verbatim, so the sim needs these
files on disk before it can load. They are FETCHED, NOT VENDORED: ~38 MB of meshes and PNGs
have no business in a source repo, so ``assets/`` is gitignored and this script is the
reproducible way to (re)create it. The fetch is a shallow (``--depth 1``), blobless
(``--filter=blob:none``), sparse (``unitree_g1`` only) clone pinned to
:data:`DEFAULT_REF` — the same commit the scene was built and measured against.

ATTRIBUTION: the model comes from MuJoCo Menagerie
(https://github.com/google-deepmind/mujoco_menagerie), which distributes ``unitree_g1``
under the BSD-3-Clause license of HangZhou YuShu TECHNOLOGY CO., LTD ("Unitree Robotics").
The license text ships inside the fetched folder as ``LICENSE`` and is checked for by
:data:`REQUIRED_FILES` — it must travel with any redistribution of these files.

Exit code 0 iff the model is present and complete afterwards.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_SUBDIR = "unitree_g1"

#: Pinned Menagerie commit ("Clean up scene <statistic> tags", #298) — the revision
#: ``configs/sim/g1_scene.xml`` was authored and verified against. Override with ``--ref``.
DEFAULT_REF = "71f066ad0be9cd271f7ed58c030243ef157af9f4"

#: Repo-relative destination; ``configs/sim/g1_scene.xml`` includes ``g1_with_hands.xml``
#: from here by relative path, so this location is part of the scene's contract.
DEFAULT_DEST = Path("assets/mujoco/unitree_g1")

#: Files the scene (and the LICENSE obligation) depend on. ``assets`` is the mesh directory.
REQUIRED_FILES: tuple[str, ...] = (
    "g1.xml",
    "g1_with_hands.xml",
    "scene.xml",
    "scene_with_hands.xml",
    "LICENSE",
)
REQUIRED_DIRS: tuple[str, ...] = ("assets",)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=_REPO_ROOT / DEFAULT_DEST)
    parser.add_argument("--ref", default=DEFAULT_REF, help="commit or branch (default: the pin)")
    parser.add_argument("--url", default=MENAGERIE_URL)
    parser.add_argument("--force", action="store_true", help="re-fetch even if complete")
    parser.add_argument("--check", action="store_true", help="verify only, never fetch")
    return parser.parse_args(argv)


def missing_paths(dest: Path) -> list[str]:
    """Names from :data:`REQUIRED_FILES`/:data:`REQUIRED_DIRS` absent (or empty) under ``dest``."""
    if not dest.is_dir():
        return [f"{dest}/ (directory)"]
    missing = [name for name in REQUIRED_FILES if not (dest / name).is_file()]
    for name in REQUIRED_DIRS:
        sub = dest / name
        if not sub.is_dir() or not any(sub.iterdir()):
            missing.append(f"{name}/")
    return missing


def _dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _sparse_fetch(url: str, ref: str, subdir: str, work: Path) -> str:
    """Shallow + blobless + sparse clone of ``subdir`` at ``ref`` into ``work``; returns the SHA."""
    _git(["init", "-q", "."], work)
    _git(["remote", "add", "origin", url], work)
    _git(["sparse-checkout", "set", "--cone", subdir], work)
    _git(["fetch", "--depth", "1", "--filter=blob:none", "origin", ref], work)
    _git(["checkout", "-q", "FETCH_HEAD"], work)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=work, check=True, capture_output=True, text=True
    )
    return sha.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dest: Path = args.dest

    missing = missing_paths(dest)
    if not missing and not args.force:
        print(f"OK model already present: {dest} ({_dir_size_mb(dest):.1f} MB) — nothing to do")
        return 0
    if args.check:
        print(f"FAILED incomplete model at {dest}: missing {missing}")
        print("       run: .venv/bin/python scripts/fetch_g1_model.py")
        return 1

    reason = "--force" if not missing else f"missing {missing}"
    print(f"fetching {MENAGERIE_SUBDIR} from {args.url} @ {args.ref} ({reason})")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Stage in a sibling temp dir so the swap is a same-filesystem rename, never a half-written
    # destination: an interrupted fetch leaves the previous model intact.
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".fetch-g1-") as tmp:
        work = Path(tmp)
        try:
            sha = _sparse_fetch(args.url, args.ref, MENAGERIE_SUBDIR, work)
        except FileNotFoundError:
            print("FAILED git not found on PATH")
            return 1
        except subprocess.CalledProcessError as exc:
            print(f"FAILED git {' '.join(exc.cmd[1:])} (exit {exc.returncode})")
            print((exc.stderr or "").strip())
            return 1

        fetched = work / MENAGERIE_SUBDIR
        incomplete = missing_paths(fetched)
        if incomplete:
            print(f"FAILED fetched tree is incomplete: missing {incomplete}")
            return 1

        staged = dest.parent / f".{dest.name}.new"
        shutil.rmtree(staged, ignore_errors=True)
        shutil.move(str(fetched), str(staged))
        previous = dest.parent / f".{dest.name}.old"
        shutil.rmtree(previous, ignore_errors=True)
        if dest.exists():
            dest.rename(previous)
        staged.rename(dest)
        shutil.rmtree(previous, ignore_errors=True)

    still_missing = missing_paths(dest)
    if still_missing:
        print(f"FAILED model incomplete after fetch: missing {still_missing}")
        return 1
    print(f"OK fetched {MENAGERIE_SUBDIR} @ {sha[:12]} -> {dest} ({_dir_size_mb(dest):.1f} MB)")
    print(f"   license: {dest / 'LICENSE'} (BSD-3-Clause, Unitree Robotics) — ships with the files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
